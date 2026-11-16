# Chapter 18 · The KV Cache Subsystem

> - **Compute tier**: ✅ any laptop (the packed formats matter on devices; nothing here needs one)
> - **Status**: draft
> - **Build step S18**: A KV cache that can append, share a prefix between sessions, shrink, defragment, and be saved to and loaded from a versioned file.
> - **Rule of this book**: The attention that reads the cache is Chapter 10 and its device kernels are Chapter 25. This chapter is storage: cells, spans, layout, persistence.

## First, the gist

Chapter 10 sized the cache and left it as an array: one region per layer, a position index into it, done. That
works until four things happen at once, and in a serving engine they always do. Two sessions want the same system
prompt. One request grows past its reservation while another sits idle. The allocator runs out of *contiguous*
space while half the arena is free. And a session has to survive a restart without becoming a different session.

So the cache stops being an array and becomes a subsystem:

- memory is handed out in **cells**, and a sequence is a **reservation plus a list of spans** over cells, so
  "position 12" is no longer "cell 12";
- a prefix can be **shared**: a second sequence reads the donor's cells in place, which means the donor can no
  longer be released while the share is live;
- **compaction** moves rows to close holes, and its contract is that addresses change and nothing else;
- the whole thing can be **written to a versioned file** that an older engine refuses rather than reinterprets.

On this model the region is 96.0 MiB at a 4,096-token context (Chapter 10's table), it grows exactly linearly
because it is allocated for the whole window up front, and the machinery this chapter adds costs 384 KiB of
bookkeeping against those 96.0 MiB.

**How this chapter got written is part of its content.** The engine's measurement tests were built to supply the
numbers below, and the first thing they did was find three bugs: a counter that reported free space that was not
writable, a compaction that could place a run on a live share, and a donor whose next prefill overwrote the rows it
had lent. All three are fixed at the book's baseline, each with a test, and the fix is an invariant the code now
states rather than a comment the reader has to infer.

## What you'll be able to do

- Model a cache as cells plus per-sequence spans, and say why a position and a cell are different numbers.
- Share a prefix between two sequences and keep the share sound: pin the donor, honour it in compaction, refuse
  the store that would overwrite the lent rows.
- Compact an arena and prove the contract - addresses change, contents do not - with a checksum before and after.
- Choose a KV format and say what it buys: bytes read per step, not bytes reserved.
- Version a session file so an older engine refuses it instead of misreading it, and say what a refusal is worth.

## Lessons

- [ ] 1. Cells and spans: why a sequence is not a contiguous array
- [ ] 2. Prefix sharing: two sessions, one cache, and the invariant that keeps it sound
- [ ] 3. Growing, shrinking and defragmenting under a context limit
- [ ] 4. Stored formats: f16, packed 8-bit, and what an integer cache costs
- [ ] 5. Sessions as files: version, checksum, and the refusal that protects the user

### Lesson 1 · Cells and spans: why a sequence is not a contiguous array

A layer's arena is one allocation. What a sequence owns inside it is a **reservation** at an offset - and that is
already a lie the moment two sequences share, because the second one wants rows at *positions* 0..34 that live at
*cells* 0..34 of a run it does not own. The engine keeps the two ideas apart in the layer's own struct:

<!-- code: src/graph/kvcache.rs::KvLayer@e844591 -->
```rust
pub struct KvLayer {
    pub k: BufRef,
    pub v: BufRef,
    /// Element count of one region (`row_elems * n_ctx`, where `row_elems` is the
    /// logical `n_kv_embd` for f32/f16 and a packed Q8_0 cell's word count for
    /// `MINFER_CACHE_TYPE=q8_0`) — the value `ensure_kv` compares on a rebuild so a
    /// changed `n_ctx` fails loudly. `elems / n_ctx` is one cell's width for every
    /// copy the C3/C8b machinery makes.
    pub elems: usize,
    /// Rows the arena can hold (`n_ctx`).
    pub n_ctx: usize,
    /// C4: the **logical** row width (`n_kv_embd`) — what the store's K/V input
    /// carries, and what a session header records so a file can be matched to the
    /// model that wrote it. `elems / n_ctx` is the stored width instead.
    pub n_embd: usize,
    /// C4: whether a cell is a packed Q8_0 block row rather than f32 words. The
    /// host-side paths that read rows as f32 (the C2 shift's re-rope) refuse a packed
    /// region instead of reinterpreting its bytes.
    pub packed: bool,
    /// `owner[cell]` — one entry per cell, `FREE` when unowned. Read by C2's
    /// resolver and the tests, and by C1's own `release_seq` (which finds the
    /// donor's cells) — so "C1 only writes it" is not the whole story.
    pub owner: Vec<SeqId>,
    /// Highest written row + 1 (the "used" prefix). Positions at or beyond it
    /// have never been written; attention above it would read zeroes.
    pub n_used: usize,
}
```
<!-- /code -->

`elems` is the whole region in f32 words, `n_ctx` is how many cells it holds, and `elems / n_ctx` is one cell's
stored width. Three of those fields exist only because of this chapter: `n_embd` keeps the *logical* width (what
the store's input means, and what a session file records), `packed` says whether a cell is a f32 row or a packed
Q8_0 block (lesson 4), and `elems` is the number `ensure_kv` compares on a rebuild so a changed context fails
loudly instead of writing off the end.

The translation from position to cell is one function, and its doc comment is where a reader learns that the
identity is a *special case*:

<!-- code: src/graph/kvcache.rs::KvCache::cells_for@e844591 -->
```rust
    /// Host-side `position -> cell` resolution for `layer`.
    ///
    /// C1: the identity, with a bounds check — a position at or beyond `n_ctx`
    /// would index past the arena. C2 replaces exactly this function's body
    /// (consulting `owner` and the layer's window) without touching its
    /// callers.
    pub fn cells_for(&self, layer: usize, positions: &[usize]) -> Result<Vec<u32>, String> {
        let l = self
            .layers
            .get(&layer)
            .ok_or_else(|| format!("no KV arena for layer {layer}"))?;
        let mut cells = Vec::with_capacity(positions.len());
        for &p in positions {
            // In C1 the cell is the position, so the check is also the arena
            // bound; C2 keeps this the arena bound and checks ownership instead.
            if p >= l.n_ctx {
                return Err(format!(
                    "KV layer {layer}: position {p} >= n_ctx {} (arena overflow)",
                    l.n_ctx
                ));
            }
            cells.push(p as u32);
```
<!-- /code -->

Two things to notice. The bounds check is against `n_ctx`, not against a sequence's reservation: a position past
the arena is an error, while a position inside the arena but outside *your* run is a question for the span table,
not this function. And the comment says what replaces its body (a version that consults ownership) without
touching its callers - which is how a position-indexed cache becomes a cell-based one without rewriting the
kernels that read it.

The linearity that makes any of this tractable is worth measuring rather than assuming. The allocator asks for the
whole window up front, so the region is exactly context times cell width - no lazy growth, no reallocation:

<!-- facts: engine-measure.kv-growth -->
| Context | Region bytes | MiB | x the 1k row |
|---|---|---|---|
| 1,024 tokens | 1,048,576 | 1.0 | 1.0 |
| 4,096 tokens | 4,194,304 | 4.0 | 4.0 |
| 8,192 tokens | 8,388,608 | 8.0 | 8.0 |
| 32,768 tokens | 33,554,432 | 32.0 | 32.0 |

One layer, 128 f32 words per cell, f32 region. Every row is exactly its context times the row above it: the allocator asks for the whole window up front, so a doubling of context is a doubling of bytes, and nothing grows lazily at the page level.
<!-- /facts -->

That last column is why a long-context engine is a memory-sizing problem and not a paging one: there is no "it
will grow as needed" to fall back on.

### Lesson 2 · Prefix sharing: two sessions, one cache, and the invariant that keeps it sound

A second sequence may **read** a donor's rows in place. The slot records how many rows it borrows and which cell
they start at, and - this is the field that makes the rest of the chapter work - how many positions it has
*written*, because a borrowed row carries the donor's stamp and not its own:

<!-- code: src/graph/kvcache.rs::SeqSlot@e844591 -->
```rust
pub struct SeqSlot {
    pub start: usize,
    pub cap: usize,
    /// C8b S2: rows read in place from another sequence (positions `[0, rows)`).
    pub shared: SharedPrefix,
    /// Positions written so far — the shared prefix plus the rows this sequence
    /// wrote itself. Explicit rather than derived from `owner`, because a shared
    /// row carries the **donor's** stamp.
    pub written: usize,
}
```
<!-- /code -->

`share_prefix` sets that up. It refuses the cases that would make the share unsound before it does anything else -
zero rows, a sequence sharing with itself, a donor that has not written the rows asked for, a destination that is
already borrowing:

<!-- code: src/graph/kvcache.rs::KvCache::share_prefix@e844591 -->
```rust
    /// C8b S2: let `dst` read the first `rows` positions of `src` **in place**.
    ///
    /// The rows are not copied: `dst`'s span list gains an entry naming the donor's
    /// cells, so both sequences read one copy — which is the whole point, and the
    /// reason the read path needs a span map (`attn_map`) instead of one range.
    ///
    /// Refused loudly rather than approximated when the sharing cannot be expressed:
    /// an empty share, a donor that has not written that far, a destination that
    /// already shares or has written rows of its own (its positions would be
    /// displaced), and a donor prefix that is not **one contiguous cell range** —
    /// that last case is a donor which itself shares a prefix plus private rows, and
    /// it needs a multi-entry map the caller has to fall back from.
    pub fn share_prefix(&mut self, src: SeqId, dst: SeqId, rows: usize) -> Result<(), String> {
        if rows == 0 {
            return Err("share_prefix: asked to share 0 rows".to_string());
        }
        if src == dst {
            return Err(format!(
                "share_prefix: sequence {src} cannot share with itself"
            ));
```
<!-- /code -->

Now the measurement. Two sequences, a 64-cell arena, the donor writing 34 rows and the sharer borrowing all 34 and
writing 8 of its own - and then the operation every engine eventually performs, releasing the donor:

<!-- facts: engine-measure.kv-sharing -->
| | donor A | sharer B | arena |
|---|---|---|---|
| Reservation | cells [0, 34) | cells [34, 42) | 64 cells |
| Rows read in place from the other sequence | - | 34 (positions 0-33) | |
| Rows written | 34 | 8 | |
| Spans | own run | (0, 0, 34), (34, 34, 8) | |
| `reserved_cells` / `shared_cells` / `owned_cells` | | | 42 / 34 / 42 |
| `free_cells` / `free_runs` | | | **22 / 1 x 22** |
| After `release_seq(A)` while B still reads | **0 cells freed** | unchanged (34 shared rows) | 42 / 34 / 42, free 22 |
| After releasing both | - | - | free 64, reserved 0 |
<!-- /facts -->

Read the last two rows together, because they are the whole lesson. Releasing the donor **frees nothing**: the
engine returns 0 cells, because a live share pins its donor's run. Release both, and the arena is empty. And
`free_cells` (22) equals the single free run (22): the two ways of counting the same arena agree, which is the
first thing to check on any cache you write - and it is the first thing that was wrong here.

The release path is four lines, and they are the contract:

<!-- code: src/graph/kvcache.rs::KvCache::release_seq@e844591 -->
```rust
    /// Release everything `seq` reserved and owned; returns the freed capacity.
    ///
    /// #831: a run a **live share** points into is refused (0, state unchanged). Its
    /// rows are the sharer's data, and dropping the reservation would leave them covered
    /// by a span no run holds — the "orphan" state where a compaction cannot move the
    /// rows and may place another run over them. The release succeeds once the last
    /// sharer releases, which is why the pin needs no reference count: it is derived
    /// from the span list.
    pub fn release_seq(&mut self, seq: SeqId) -> usize {
        if self.shared_from(seq) > 0 {
            return 0;
        }
        let Some(slot) = self.seqs.remove(&seq) else {
```
<!-- /code -->

**A measurement wrote this section.** The tests behind the table were built to supply numbers for this chapter, and
they reported a cache whose two views of free space disagreed:

- the counter said **56** free cells after the donor was released; the run table said **22**. The counter summed
  reservations, which do not include a borrowed prefix, while the runs came from the span table, which does. An
  operator reading `minfer_kv_free_cells` would have been told there was room for a request that had none, and
  the server's slot-growth loop would have stopped reclaiming idle slots one release too early.
- Compaction had the second. It packs reservations downward and renumbers a sharer's cells only for a move that
  carries the donor's *written* rows; once the donor's run was gone there was no such move, so a live share could
  be placed over. The code now pins the donor instead, and `set_cap` refuses the release that would leave a share
  uncovered:

<!-- code: src/graph/kvcache.rs::KvCache::set_cap@e844591 -->
```rust
    /// Set `seq`'s reservation to `cap` cells and return the compaction plan that
    /// reconciles the arena (C7b).
    ///
    /// Growing is refused when every reservation cannot fit at once — capacity has
    /// to come from somewhere, which is the caller's decision (the server releases
    /// idle runs first) — and shrinking is refused below the rows already written,
    /// so space can never be traded for data. The run table is updated **before**
    /// planning, which is what lets the planner see the new size; the rows travel
    /// with the plan (the allocator copies them before [`Self::apply_moves`]
    /// renumbers), in whichever direction the plan needs.
    pub fn set_cap(&mut self, seq: SeqId, cap: usize) -> Result<Vec<KvMove>, String> {
        let Some(slot) = self.seqs.get(&seq).copied() else {
            return Err(format!("set_cap: sequence {seq} holds no run"));
        };
        if cap == slot.cap {
            return Ok(Vec::new());
        }
        if cap == 0 {
            let lent = self.shared_from(seq);
            if lent > 0 {
                return Err(format!(
                    "set_cap: sequence {seq} holds {} cells and {lent} of them are read in \
```
<!-- /code -->

- The third was a store. Nothing stopped a donor slot from being handed a new prompt and pre-filling from position
  0, over the rows it had lent. The placement layer now refuses that request; the store path itself keeps a
  backstop that names the violation, because the next caller (speculative decoding, a new server path) would
  inherit no protection from the placement check.

The three fixes are one recorded decision, [ADR-0083](https://github.com/yusiwen/minfer/blob/master/docs/adr/0083-a-live-share-pins-its-donor-s-run-and-a-store-may-not-write-a-lent-prefix.md):
a live share pins its donor's run, and a store may not write a lent prefix. What the book took from that is the
sentence it repeats on every measurement page - **a number two parts of the same system disagree about is a bug in
the system, not a rounding difference to explain away in prose.**

<figure class="book-figure">{{#include ../figures/18-cells-and-spans.svg}}</figure>

*Figure 18.1 - the experiment from the table, drawn to scale: a 64-cell arena holding a 34-cell donor run, an 8-cell sharer run, and the borrowed prefix that lives inside the donor's cells. The sharer's spans are two ranges, the free space is one run of 22, and releasing the donor frees 0.*

### Lesson 3 · Growing, shrinking and defragmenting under a context limit

`set_cap` is where a request's appetite meets the arena. Two refusals matter more than the success path: a shrink
may not go below the rows already written (space can never be traded for data), and growing is refused when the
arena cannot hold every reservation at once - capacity has to come from *somewhere*, and choosing what to reclaim
is the caller's decision, not the allocator's:

<!-- code: src/graph/kvcache.rs::KvCache::set_cap@e844591 -->
```rust
        if cap < slot.cap {
            // Only the rows in its *own* run bound the shrink: a shared prefix lives
            // in cells this reservation never covered (C8b S2).
            let written = slot.private_written();
            if cap < written {
                return Err(format!(
                    "set_cap: sequence {seq} has written {written} rows and cannot shrink to {cap}"
                ));
            }
        } else {
            let others: usize = self
                .seqs
                .iter()
                .filter(|(&s, _)| s != seq)
```
<!-- /code -->

When the arena has holes the fix is compaction: pack the reservations down from cell 0, carry the rows, and report
what moved. The planner walks the runs in cell order and emits a move only when a run is not already where the
cursor is:

<!-- code: src/graph/kvcache.rs::KvCache::compaction_plan@e844591 -->
```rust
    /// The moves that compact the live runs **downward**, in application order.
    ///
    /// Runs are packed from cell 0 in ascending `start` order, each keeping its
    /// `cap`, so free space coalesces at the top of the arena. `need`: `None`
    /// compacts fully; `Some(n)` keeps only the shortest prefix of the plan whose
    /// last move leaves a free gap of at least `n` cells, which is what makes
    /// `reserve_seq(n)` succeed — moving runs that buy nothing is exactly what a
    /// defragmentation should not do.
    ///
    /// Pure: no mutation and no backend call, so the policy is unit-tested
    /// without a device. The data copy is the allocator's job
    /// (`Backend::copy_cells`), driven by the returned `rows`.
    pub fn compaction_plan(&self, need: Option<usize>) -> Vec<KvMove> {
        let mut runs: Vec<(SeqId, SeqSlot)> = self
            .seqs
            .iter()
            .filter(|(_, s)| s.cap > 0)
            .map(|(&seq, &slot)| (seq, slot))
            .collect();
        runs.sort_by_key(|&(seq, slot)| (slot.start, seq));
        let mut cursor = 0usize;
        let mut moves = Vec::new();
        for (i, &(seq, slot)) in runs.iter().enumerate() {
            if slot.start != cursor {
```
<!-- /code -->

`rows: slot.private_written()` is the subtle part: a move carries the rows a sequence *wrote*, not the rows it
borrows, because a borrowed prefix lives in the donor's run and travels (or does not) with the donor.

The allocator wraps the planner, copies the cells it names, and reports before and after - which is what the
measurement below reads. The arena is deliberately tiny (16 cells, two sequences interleaved) so the numbers can be
checked by hand:

<!-- facts: engine-measure.kv-defrag -->
| | Before | After |
|---|---|---|
| Free cells | 8 | 8 |
| Free runs | 2 | 1 |
| Largest free run | **4** | **8** |
| Cells in the run that moves | 4 | 4 |
| Sequences | 2 | 2 |
| Rows moved | - | 4 |

Arena 16 cells, 1 layer, cell 4 wide. Moves: seq 3: cells [8, 12) -> [4, 8). Every moved row's checksum before and after: seq 1 unchanged, seq 3 unchanged.
<!-- /facts -->

Largest free run 4 -> 8, four rows moved, **every checksum identical before and after**. That is the whole contract
of a compaction in one table: the addresses changed, the bytes did not. A defragmenter that reports fewer runs and a
bigger hole while changing a row is a data-corruption bug that surfaces as fluently wrong text three chapters
later, so the invariant is not "it ran" but "the rows are the same rows".

<!-- code: src/graph/alloc.rs::GraphAllocator::kv_defrag@e844591 -->
```rust
    pub fn kv_defrag(&mut self, need: Option<usize>) -> Result<KvDefragReport, String> {
        let before = self.kv.arena_stats();
        let moves = self.kv.compaction_plan(need);
        if moves.is_empty() {
            return Ok(KvDefragReport {
                moves,
                rows_moved: 0,
                before: before.clone(),
                after: before,
            });
        }
        // Snapshot the regions: `kv.iter()` borrows the store, and the copies
        // below need `&mut self`.
        let regions: Vec<(BufRef, BufRef, usize)> = self
            .kv
            .iter()
            .map(|(_, l)| (l.k, l.v, (l.elems / l.n_ctx.max(1)).max(1)))
            .collect();
        for &(k, v, elems_per_cell) in &regions {
            for m in &moves {
                if m.rows == 0 {
                    continue;
```
<!-- /code -->

### Lesson 4 · Stored formats: f16, packed 8-bit, and what an integer cache costs

Chapter 10 established the surprising part: a cell is a fixed number of **f32 words**, so f32 and f16 reserve the
same region and f16 only halves the *payload* read. This chapter adds the operational half - what the engine will
and will not run, and who decides:

<!-- code: src/graph/kvformat.rs::KvFormat::supports@e844591 -->
```rust
    /// Whether this device has kernels that read a region in this format.
    ///
    /// F4: the packed answer is the backend registry's
    /// `BackendCaps::reads_packed_kv` — the same field `GraphAllocator::ensure_kv`
    /// reads, so the format gate and the region-sizing gate cannot disagree. The
    /// CPU's attention kernel dots the stored K blocks against the quantized query
    /// and accumulates V out of the cell; CUDA's kernels are layout-tagged
    /// (`KV_LAYOUT_F32/F16/Q8_0` plus the byte-addressed `kv4<LAYOUT>` load) since
    /// C4 S2b, so it reads a packed region too. [#310] gave Metal the same answer:
    /// its packed store and read are enabled — a direct in-kernel read where the
```
<!-- /code -->

The format is therefore a **capability question asked of the registry**, not a switch the caller flips and hopes: a
packed Q8_0 cell is only offered where the backend's kernels actually read one, and the device policy next to that
function says which models get f16 automatically (the 7B class, where decode is KV-bandwidth-bound; the 0.5B
measured ~3 per cent *slower* with f16, being dispatch-latency-bound). Two consequences for a cache author:

- **The width of a cell is not the width of a value.** Q8_0 packs 34 words into the space of 128 elements, so a
  cell move copies words (`copy_within`), not elements, and every byte offset in the arena is a word offset.
- **The format is part of the arena's identity.** A rebuild that changes it must reallocate the regions, and a
  session file that changes it must say so in its header - which is lesson 5.

The measured gap between the three formats, on one model at one context, is why that complexity is worth it: at a
4k context the packed region is 25.5 MiB against 96.0 MiB for f32 or f16, and a decode step reads the whole region
- 25.5 MiB against 48.0 MiB of f16 payload (Chapter 10's table has all three columns).

### Lesson 5 · Sessions as files: version, checksum, and the refusal that protects the user

A KV session is not a memory dump. It is a file with a magic number, a version, a flag word, the geometry it was
written for, the rows, the arena bookkeeping that says who owns what, and a checksum over all of it:

<!-- code: src/graph/kvsession.rs::MAGIC@e844591 -->
```rust
pub const MAGIC: [u8; 8] = *b"MINFERKV";
/// Container version. A reader refuses anything else, loudly.
///
/// - **1** — the header, one K/V blob per layer and the KV bookkeeping.
/// - **2** — adds the **host-state blob** (C5 S2): an opaque, length-prefixed section
///   after the bookkeeping, carried and checksummed exactly like the rest of the file.
///   The KV rows belong to a host state — the CLI's conversation, a server slot — and a
///   restore that brought the rows back without it would be a different session, so the
///   container carries both or neither. A version-1 file is refused loudly and the
///   caller falls back to re-seeding, which is what the version byte is for.
///
/// The version was deliberately **not** bumped when the element type gained the
/// `FLAG_F16` bit (`#130`): that is an additive flag, exactly as `FLAG_PACKED` was,
/// and an older v2 build refuses the unknown bit rather than guessing — so no file
/// on disk changes meaning. A version bump is for a *layout* change; a new refusal
/// on a bit nobody wrote before is not one.
pub const VERSION: u32 = 2;
/// Flags bit 0: the regions are packed Q8_0 cells (C4).
const FLAG_PACKED: u32 = 1 << 0;
/// Flags bit 1: the regions are f16 cells (C4's GPU bandwidth policy).
///
/// Kept as a **new bit** rather than a format field, so the flag word's byte
/// layout — and therefore every file already on disk — is untouched: a Q8_0 file
/// still reads bit 0 exactly as before, and an f32 file is still `flags == 0`.
/// A pre-#130 build reading an f16 file does not know this bit, so it refuses it
/// loudly at the unknown-flags check instead of decoding the region as f32 — the
/// same property [`FLAG_PACKED`] had when it landed, and the reason no version
/// bump is needed.
const FLAG_F16: u32 = 1 << 1;
/// Every bit this reader understands. A file with any other bit set was written
/// by a newer build, and its layout is not ours to guess.
const KNOWN_FLAGS: u32 = FLAG_PACKED | FLAG_F16;
```
<!-- /code -->

The version comment is the part to read twice: version 2 added the host-state blob, and the element type was added
later as a **flag bit** rather than a version bump, on the explicit argument that an older build refuses an unknown
bit loudly and no file on disk changes meaning. A version bump is for a *layout* change; a new refusal for a bit
nobody wrote before is not one. That is the distinction to carry into your own formats.

The writer's field order *is* the layout, and it is short enough to read as an offset table:

<!-- code: src/graph/kvsession.rs::KvSessionWriter::create@e844591 -->
```rust
    pub fn create(path: &Path, header: &KvSessionHeader) -> Result<Self, String> {
        let file = File::create(path)
            .map_err(|e| format!("KV session: cannot create {}: {e}", path.display()))?;
        let mut sink = Sink::new(BufWriter::new(file));
        sink.bytes(&MAGIC)?;
        sink.u32(header.version)?;
        sink.u32(flags_of(header.format))?;
        sink.u32(tag_of(header.backend))?;
        sink.u32(header.n_layer as u32)?;
        sink.u32(header.n_ctx as u32)?;
        sink.u32(header.n_embd as u32)?;
        sink.u32(header.row_elems as u32)?;
        Ok(Self {
            sink,
            header: header.clone(),
            layers: 0,
            host: Vec::new(),
        })
    }
```
<!-- /code -->

Which makes the whole file arithmetic on the model's geometry:

<!-- facts: engine-measure.kv-session -->
| Section | Bytes | Field order |
|---|---|---|
| Magic and header | 36 | `MINFERKV` (8 bytes), then version, flags, backend tag, n_layer, n_ctx, n_embd, row_elems - seven 4-byte fields |
| One layer's record | 4,194,316 | layer index, K words, V words (3 x u32 of bookkeeping, then 2 x 524,288 f32 words) |
| All 24 layer records | 100,663,584 | 4,194,316 x 24 |
| Arena bookkeeping | 393,600 | identity, n_ctx, per-sequence slots and spans, then **each layer's owner table** (4,096 u32 = 16,384 B per layer), then four u64 counters |
| Host state | 4 + n | length-prefixed, opaque, inside the checksum |
| Checksum | 8 | FNV-1a over everything before it |
| **Total for this model, empty host state** | **101,057,232** | 96.4 MiB, of which 96.0 MiB is KV rows |

Read the fourth row again: the file carries the **owner table**, 16,384 bytes per layer, because a restored arena must know not only the rows but which sequence holds each cell. That is 384 KiB of bookkeeping against 96.0 MiB of rows at this context - small, and not optional.
<!-- /facts -->

**That table is arithmetic over quoted code, not a measurement, and the chapter says so rather than implying
otherwise.** The CLI's single-turn mode writes no `.kv` file, so there is no run to point at; the offsets come from
`create` and `finish`, and the geometry comes from the engine's ledger at the baseline. When you build this, run the
save/restart/reload test and the table stops being arithmetic.

The owner table is the line that surprises people: 16,384 bytes per layer, 384 KiB for 24 layers. A file carrying
only the rows would restore bytes into cells nobody owns, and the first request after the restore could write over a
shared prefix. What the file carries is the **arena**, not a slab of memory.

<figure class="book-figure">{{#include ../figures/18-session-file.svg}}</figure>

*Figure 18.2 - the same file as a byte map: a 36-byte header, 24 layer records of 4.19 MB each, 384 KiB of arena bookkeeping with one owner table per layer, the caller's host state, and a trailing FNV-1a checksum. 96.0 MiB of the 96.4 MiB total is KV rows.*

## Cache operations

| Operation | Invariant it must preserve | Cost | Test that catches it |
|---|---|---|---|
| append a token | a written position is inside the sequence's own run, and a borrowed row is never written | one cell write per layer | a store into a lent prefix is refused by name ([#837](https://github.com/yusiwen/minfer/issues/837)) |
| share a prefix | the donor holds the rows, the sharer owns no cells for them, and the donor cannot be released while the share is live | no copies, one span and one counter | release the donor, then assert 0 cells freed and `free_cells == sum(free_runs)` ([#829](https://github.com/yusiwen/minfer/issues/829), [#831](https://github.com/yusiwen/minfer/issues/831)) |
| resize | shrinking never drops a written row; growing never silently reclaims someone else's | a compaction plan | grow past the arena with one idle and one lent slot; only the idle one may be taken |
| defragment | addresses change, contents do not | one cell copy per row per region | row checksums before and after (the table above) |
| save / load | a file is refused on magic, version, unknown flag, geometry or checksum mismatch, and never silently reinterpreted | 8 bytes of checksum plus the rows | flip one byte and require a refusal that names what failed |

## Project

**Build step S18 · storage, not an array.** Write the invariant list before the operations, then one test per
invariant. Cache bugs do not crash: they produce fluently wrong text, which is the hardest class of bug in this book
to attribute - so the tests are the artifact, and the operations are written to them.

Order the work so each step fails when it is wrong:

1. Cells and spans with a single sequence. Test: a position past the reservation but inside the arena is refused by
   the owner check, not silently accepted.
2. Sharing. Test: release the donor and assert nothing is freed; then release the sharer and assert everything is.
3. Compaction. Test: checksums of every moved row before and after, plus the number of runs.
4. Packing, if your backend can read it. Test: a cell move under the packed stride, and the refusal on a backend
   that cannot read one.
5. Save, restart, reload. Test: the next token is identical, and a one-byte corruption is refused with a message
   that names the version or the checksum.

## Deliverable

- A cell-based cache with append, share, resize and defragment, each with an invariant test
- A versioned session file with a checksum, an owner table, and a refusal path for every mismatch
- A measured bytes-per-token table for two formats at one context length, and the read cost per decode step

## How you know you passed

- Releasing a donor that has a live share frees **zero** cells, and `free_cells` equals the sum of `free_runs` in
  every state you can reach - including the one where a prefix is shared.
- Defragmentation changes addresses and nothing else: a test asserts identical row checksums before and after.
- A save, restart and reload yields an identical next token; an old, unknown-flag or corrupt file is refused with a
  message naming the version, the bit or the checksum.
- You can say what f16 buys (bytes read) and what it does not (bytes reserved), with both numbers.

## Reading

- code: `src/graph/kvcache.rs`, `src/graph/kvformat.rs`, `src/graph/kvsession.rs` at the book's [pinned baseline](../appendix/baseline.md)
- decisions: [ADR-0006](https://github.com/yusiwen/minfer/blob/master/docs/adr/0006-kv-format-is-a-per-engine-gate.md) (the KV format is a per-engine gate), [ADR-0014](https://github.com/yusiwen/minfer/blob/master/docs/adr/0014-a-kv-session-is-a-versioned-file.md) (a KV session is a versioned file), [ADR-0032](https://github.com/yusiwen/minfer/blob/master/docs/adr/0032-the-packed-kv-staging-window-is-f32.md) (the packed-KV staging window is f32), [ADR-0083](https://github.com/yusiwen/minfer/blob/master/docs/adr/0083-a-live-share-pins-its-donor-s-run-and-a-store-may-not-write-a-lent-prefix.md) (a live share pins its donor's run)
- docs: `docs/KV-CACHE-DESIGN.md`
- [paged-attention] Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023) - arXiv:2309.06180

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)

<!-- facts: engine-measure.provenance -->
*`scripts/engine-measure.py`; runs the engine's own measurement tests (`cargo test --release measure`) and reads the KV geometry and footprint groups out of them, then recomputes every number from `KvFormat::row_elems` and refuses to write the artifact when the two disagree; on **Darwin 27.2.0, arm64**; one run of the engine's own measurement tests; the derived column is recomputed from `KvFormat::row_elems` and must equal the measured one. Command: see [the provenance index](../appendix/provenance.md#engine-measure). Raw numbers: `facts/engine-measure.json`.*
<!-- /facts -->
