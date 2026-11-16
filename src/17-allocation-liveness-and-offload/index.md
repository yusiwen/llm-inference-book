# Chapter 17 · Allocation: Liveness, Arena, and Offload

> - **Compute tier**: ✅ any laptop
> - **Status**: draft
> - **Build step S17**: A liveness-based allocator that reuses one arena across a graph, keeps KV regions persistent, and decides what fits on a device before running.
> - **Rule of this book**: The KV cache's internal structure is Chapter 18. This chapter is about memory as a whole: who owns it, when it can be reused, and what happens when it does not fit.

## First, the gist

A transformer's activation memory is almost entirely transient. Each intermediate is born at one node, read by its
consumers, and dead a few nodes later - and the next layer wants a buffer of exactly the same shape. An allocator
that gives every node its own buffer pays for the graph's *sum*. An allocator that knows when each buffer dies pays
for the *peak of the live set*, and on this model's prefill graph that is **28.2x** less memory: 84.7 MiB of activations
become a 3.00 MiB pool, of which at most 2.05 MiB is live at any instant.

The measured summary, from this machine, on the graph the engine executed:

<!-- facts: alloc.al-liveness -->
| | Prefill (34 tokens) | Decode (1 token) |
|---|---|---|
| Nodes in the graph | 443 | 440 |
| Transient nodes (the rest name persistent regions) | 395 | 392 |
| Sum of every transient buffer, no reuse (**the book's arithmetic**) | 88,778,548 B (84.67 MiB) | 3,293,720 B (3.14 MiB) |
| The engine's pool, high-water (**the engine's**) | 3,147,776 B (3.00 MiB) | 740,352 B (0.71 MiB) |
| Live when the phase ends | 632,832 B (0.60 MiB) | 631,808 B (0.60 MiB) |
| Peak live inside the phase | 2,152,448 B (2.05 MiB) | 631,808 B (0.60 MiB) |
| Naive / pool (what reuse bought) | 28.2x | 4.45x |
| Size classes still reserved at the end | 4 | 3 |
| Idle slots in them | 11 | 7 |
| Largest transient buffer | `matmul` 4864x34x1x1 = 661,504 B | `matmul` 151936x1x1x1 = 607,744 B |
<!-- /facts -->

Two numbers in that table are not the ones people expect. The pool is **1.46x the peak live set** on prefill
(3.00 MiB against 2.05 MiB), so fragmentation is real and worth its own lesson - and it is size classes, not
garbage. And the decode graph reuses better still (1.17x) while its naive bill is 27x smaller than prefill's,
because a decode step builds the same few shapes over and over, so nearly every reuse is an exact class match.

The second half of the chapter is the memory that never dies: the KV regions persist across a rebuild, which is
what lets prefill and decode share one process. The third is the question an allocator has to answer *before* it
allocates a byte on a device - will this fit - and what it does when the device will not answer.

## What you'll be able to do

- Separate transient activations from persistent regions, and say which allocator each one belongs to.
- Compute liveness spans over a graph and reuse buffers safely, with the three rules that make reuse sound.
- Size an arena for a real model, compare it with the live-set floor, and explain the gap.
- Make a budget decision from a device memory query, and refuse cleanly when the query fails.
- Predict peak memory, then read the number the process actually held, and explain the difference.

## Lessons

- [ ] 1. Transient versus persistent: which buffers outlive a single node
- [ ] 2. Liveness spans, and the three rules that make reuse sound
- [ ] 3. One arena, many classes: rounding, fragmentation, and the gate
- [ ] 4. Offloading: the budget question, and why a failed query is not a zero budget
- [ ] 5. Peak-memory prediction versus observation

### Lesson 1 · Transient versus persistent

`alloc_graph` runs on every graph (re)build, and its doc comment draws the line everything else follows:

<!-- code: src/graph/alloc.rs::alloc_graph@e844591 -->
```rust
    /// Runs on every graph (re)build: previous liveness buffers are released
    /// back to their pools, while **persistent regions (KV cache) survive** —
    /// they are the KV cache and must persist across prefill→decode rebuilds.
    pub fn alloc_graph(&mut self, graph: &ComputeGraph) -> Result<(), String> {
        let prev: Vec<(Backend, usize)> = self.buf_alive.keys().copied().collect();
        for (b, id) in prev {
            self.free_in_pool(b, id);
        }
        self.buf_alive.clear();
```
<!-- /code -->

**Previous liveness buffers are released, while persistent regions survive.** A rebuild is not a new process:
prefill builds a 443-node graph, runs it, and then the decoder builds a 440-node graph against the *same*
allocator, and the KV that prefill wrote has to still be there.

That is why the two kinds of memory take different paths. Transients go through `alloc_graph`, are keyed by node
id, and die when the graph is replaced. Persistent regions go through `alloc_persistent`, are keyed by **name**
(`kv.{layer}.k`, `kv.{layer}.v`), and outlive everything: a rebuild that produces different node ids still finds
its cache, because a layer's name does not change.

The division is visible in the measurement, because the `kvcache_store` and `kvcache_load` nodes carry the shape of
the region they *view*, not a buffer of their own:

| | Prefill graph |
|---|---|
| Nodes that write or read a persistent region | 48 (24 layers x store and load) |
| Bytes the trace names for one such node | 2,097,152 B - **one** of a layer's two arrays |
| Bytes the engine allocates per layer | 4,194,304 B - K and V, two regions |
| Region total, all layers | 100,663,296 B (96.0 MiB) |
| Charged to the transient arena | 0 |

The trace's 2,097,152 B is worth reading twice, because it is where a careful reader can go wrong: it is the shape
of *one* of the two arrays a layer holds, so a book that summed these 48 nodes would report half the cache. The
engine never sums them - `KvCache::region_bytes` multiplies by two by construction (Chapter 10 quotes it) - and
the number in the table above is the engine's.

96.0 MiB of KV against 463.0 MiB of weights is 20.7% at a 4,096-token context, and that ratio is what moves as
context grows - Chapter 18's subject. What matters here is that the two accounts never mix: an allocator that
charged the KV region to the arena would count it twice on every rebuild.

### Lesson 2 · Liveness spans, and the three rules that make reuse sound

Here is the whole algorithm, from the same function:

<!-- code: src/graph/alloc.rs::liveness@e844591 -->
```rust
        // The scheduler executes nodes in BUILD order (node id order — the
        // builder appends sources before consumers), so liveness must use the
        // same order: topo_order() can reorder srcless nodes (kv_load) ahead,
        // which would let a later consumer's buffer reuse clobber an input the
        // scheduler has not yet read (G3 tail get_rows regression). Validate
        // acyclicity, but keep build order.
        graph.topo_order()?;
        let order: Vec<NodeId> = (0..graph.n_nodes()).collect();
        let n = graph.n_nodes();

        let mut exec = vec![0usize; n];
        for (i, &id) in order.iter().enumerate() {
            exec[id] = i;
        }
        let mut last_use = exec.clone();
        for (i, &id) in order.iter().enumerate() {
            for &s in &graph.node(id).src {
                if last_use[s] < i {
                    last_use[s] = i;
                }
            }
        }
        for &o in &graph.outputs {
            last_use[o] = order.len();
        }
        // Inputs are filled on the host BEFORE execution starts, so every
        // input buffer is live at fill time; liveness (which tracks execution
        // order) must never reuse an input's buffer for another input — the
        // later fill would clobber the earlier one. Treat inputs like outputs.
        for &i in &graph.inputs {
            last_use[i] = order.len();
        }
```
<!-- /code -->

`last_use[s]` starts as the index at which `s` executes, and every consumer pushes it forward. A buffer is dead
once the walk has passed its `last_use`, and its bytes can be handed to someone else. Three details are not
obvious, and each one was a bug:

1. **The walk order is build order, not topological order.** The comment says why: `topo_order()` may hoist
   srcless nodes (`kv_load`) to the front, which would let a later consumer's buffer be reused for something the
   scheduler has not read yet. This is the G3 tail-`get_rows` regression, written down where the next person
   reads it.
2. **Outputs and inputs are pinned to the end.** An output is read after the graph finishes. An input is *filled
   before execution starts*, so its buffer must never be handed to another input - the second fill would
   overwrite the first. A liveness rule that only follows consumers gets that wrong, and passes every test that
   does not use two inputs.
3. **A view extends its parent's life.** A `view` node owns no bytes; it is an offset into another buffer. The
   parent must therefore stay alive as long as any reader of the view, so its `last_use` is pushed forward
   *through* the view:

<!-- code: src/graph/alloc.rs::extend_through_views@e844591 -->
```rust
/// Returns the nodes whose liveness actually grew (in walk order, the node
/// first), so the caller can keep the pool's own deadline in step: `buf_alive`
/// is set from `last_use` when a buffer is allocated, and a later extension that
/// does not touch it would let `sweep` hand the buffer to another node while the
/// alias still reads it (E4 S2: rounding makes such a hand-over likely, because
/// a class match no longer needs an exact size coincidence — the bug was latent
/// before it).
fn extend_through_views(
    graph: &ComputeGraph,
    last_use: &mut [usize],
    from: NodeId,
    to: usize,
) -> Vec<NodeId> {
    let mut grown = Vec::new();
    let mut cur = from;
    loop {
        if last_use[cur] >= to {
            break;
        }
```
<!-- /code -->

The function returns the nodes whose liveness grew, and the doc comment explains why: the pool keeps its own
`buf_alive` deadline, and an extension that does not reach it would let the sweeper hand the buffer to another
node while the alias still reads it. That bug was latent for a long time and only became reachable when
size-class rounding stopped requiring an exact size coincidence - the kind of thing that is invisible until
something unrelated makes it likely.

### Lesson 3 · One arena, many classes: rounding, fragmentation, and the gate

A real allocator does not hand out bytes; it hands out **size classes**. On this engine the ladder is a 1 KiB
floor, powers of two up to 16 KiB, then 16 KiB steps:

<!-- code: src/graph/allocplan.rs::class_size@e844591 -->
```rust
/// The size class `elems` is rounded up to.
///
/// Below the grain the ladder is powers of two (the many small activations of a decode
/// step are cheap to share and expensive to round coarsely); above it, multiples of the
/// grain, so a 1.8 M-element prefill buffer wastes at most 16 KiB instead of up to 2x.
pub fn class_size(elems: usize) -> usize {
    let elems = elems.max(1);
    if elems <= CLASS_MIN {
        return CLASS_MIN;
    }
```
<!-- /code -->

The reason for two regimes is in the doc comment: below the grain the small activations of a decode step are cheap
to share and expensive to round coarsely; above it, a coarse ladder would waste up to half a large buffer, so the
grain wins instead. What the ladder costs on this graph:

<!-- facts: alloc.al-classes -->
| | Prefill arena | Decode arena (fresh) |
|---|---|---|
| Nodes allocated | 443 | 440 |
| Pool bytes | 3,147,776 | 740,352 |
| Size classes reserved | 4 | 3 |
| Idle slots kept for the next graph | 11 | 7 |
<!-- /facts -->

That table is the engine's own report rather than the book's arithmetic, and its two columns say different things:
the prefill arena keeps four classes and eleven idle slots after its graph is done, the decode arena three and
seven. It is also why the pool's high-water mark is not the peak live set - classes are reserved **per class**, so
two buffers of different classes cannot share even when their live intervals never overlap. The allocator reports
those numbers itself, per backend:

<!-- code: src/graph/alloc.rs::memory_report@e844591 -->
```rust
    /// E4's accounting for one backend: what is resident, what is live, the peak, and the
    /// budget the allocator checks against.
    pub fn memory_report(&self, backend: Backend) -> MemoryReport {
        let (mut idle_slots, mut reserved_classes) = (0usize, 0usize);
        for ((b, _), ids) in &self.slots {
            if *b != backend || ids.is_empty() {
                continue;
            }
            idle_slots += ids.len();
            reserved_classes += 1;
        }
        MemoryReport {
            weights_bytes: self.weights_bytes(backend),
            pool_bytes: self.pool_bytes.get(&backend).copied().unwrap_or(0),
            live_bytes: self.live_bytes.get(&backend).copied().unwrap_or(0),
            peak_live_bytes: self.peak_bytes.get(&backend).copied().unwrap_or(0),
            budget: self.memory_budget(backend),
            idle_slots,
            reserved_classes,
        }
    }
```
<!-- /code -->

`pool_bytes`, `live_bytes` and `peak_live_bytes` are exactly the three numbers this chapter computes by hand, and
the CLI does not print them - they are reached from tests today. So read lesson 5's ledger carefully: its
*predictions* are this book's model and its *observation* is the process's own RSS, and the chapter says which is
which, because the difference between "the engine reported" and "I computed" is the thing this book is about.

Fragmentation, as pool over peak live, is 1.46x on prefill and 1.17x on decode. The explanation is worth keeping:
prefill's 395 transient nodes have many different shapes, so neighbouring intervals rarely land in the same class;
decode's 392 repeat one shape per layer, so almost every reuse is an exact class match. **Shape diversity, not node
count, is what creates fragmentation.**

<figure class="book-figure">{{#include ../figures/17-liveness.svg}}</figure>

*Figure 17.1 - the same graph's activations counted two ways. The top bar is the book's arithmetic: every transient buffer with an allocation of its own, 84.7 MiB. The three bars below are the engine's pool, drawn at a 28x larger scale: a 3.00 MiB high-water mark, a 2.05 MiB peak live set inside the phase, and 0.60 MiB still live when the phase ends.*

### Lesson 4 · Offloading: the budget question, and why a failed query is not a zero

The KV and the arena are what the host holds; the weights are what a device must hold, and they do not always fit.
The engine's answer separates three states - a measurement, a failed query, and no device at all:

<!-- code: src/graph/allocplan.rs::budget_decision@e844591 -->
```rust
/// Resolve a backend's activation budget from an explicit override and the device's
/// own memory answer. Pure, so the whole mapping is unit-tested with no device.
///
/// The rules, in order:
/// - an explicit `set_memory_budget` wins and is a measurement the caller chose; no note;
/// - a **reported** free read keeps the pre-existing default, three quarters of it
///   (unchanged on the happy path, including a genuine `free == 0`);
/// - a **failed** query is not a measurement: it falls back to weights-only accounting
///   (an unbounded activation budget) and carries a note naming the real error. Refusing
///   every allocation instead would turn a broken accounting query into a total outage
///   of a device that may be perfectly usable — and the backend's own allocation is the
///   authority on whether memory exists;
/// - no device state is unbounded and silent (a CUDA graph without device state is a
///   configuration error the assignment pass catches).
pub fn budget_decision(explicit: Option<usize>, mem: &DeviceMemory) -> BudgetDecision {
    if let Some(b) = explicit {
        return BudgetDecision {
            budget: Some(b),
            note: None,
        };
    }
    match mem {
        DeviceMemory::Reported { free, .. } => BudgetDecision {
            budget: Some(free / 4 * 3),
            note: None,
        },
        DeviceMemory::QueryFailed { code, name } => BudgetDecision {
            budget: Some(usize::MAX),
            note: Some(format!(
                "the device free-memory query failed with {name} (code {code}); the E4 \
                 activation gate has no measured budget for this backend, so it charges \
                 weights only. The backend's own allocation is now the authority and will \
                 report the real error if the context is unusable"
            )),
        },
        DeviceMemory::NoDevice => BudgetDecision {
            budget: None,
            note: None,
        },
    }
}
```
<!-- /code -->

The doc comment above the function is this lesson in the engine's own words. A **reported** free read becomes
three quarters of it, unchanged from the previous behaviour. A **failed** query is not a measurement: it falls
back to weights-only accounting and carries a note naming the real error, because "refusing every allocation would
turn a broken accounting query into a total outage of a device that may be perfectly usable". And no device state
is unbounded and silent - it is `None`, and the assignment pass from Chapter 16 catches it.

That is Chapter 16's distinction one layer down: **an absent measurement and a measured zero are different
facts**, and code that collapses them - the classic `free.unwrap_or(0)` - turns a broken query into a refusal to
run. This machine has no device to ask, so the CPU path never consults it; the lesson is quoted, not measured, and
the chapter says so rather than inventing a number.

The offload decision itself - moving a prefix of layers to host memory - reads the same `budget_decision`, so the
feasibility gate and the auto-fit cannot disagree about the device's number. That is the only property worth
testing here: two components that each ask the driver separately will eventually disagree, and the disagreement
surfaces as an out-of-memory at the worst possible moment.

### Lesson 5 · Peak-memory prediction versus observation

<!-- no-code: this lesson is about the gap between two numbers, and only one of them comes from the engine. The engine's own ledger (`memory_report`) is quoted in lesson 3; the process's peak RSS is the kernel's number; the prediction between them is this book's model, and quoting a function here would suggest the engine produced all three. -->

Now the ledger: every region the run needs, next to the number the operating system reports for the same run.

<!-- facts: alloc.al-ledger -->
| Region | Bytes | MiB | Whose number |
|---|---|---|---|
| Weights - for the life of the process | 485,452,288 | 463.0 | engine, `GraphAllocator::memory_report` |
| KV regions, 24 layers - persistent across rebuilds | 100,663,296 | 96.0 | engine, `KvCache::region_bytes` |
| Transient pool, high-water - dies with the graph | 3,147,776 | 3.0 | engine, `memory_report` |
| **Predicted resident - the three rows above, added** | **589,263,360** | **562.0** | book, `weights + KV + pool` |
| **Observed peak RSS - the whole process** | **496,959,488** | **473.9** | kernel, `/usr/bin/time -l` |
| Prediction minus observation | 92,303,872 | 88.0 | book, `predicted - observed` |

The prediction is not a bound in either direction, and the reason is in the same table: the cache is allocated for 4,096 tokens and the run wrote 34, so most of the 96.00 MiB of KV is address space the kernel never had to back. Observed peak RSS is a median of 3 runs (495,845,376 to 498,548,736 B).
<!-- /facts -->

<figure class="book-figure">{{#include ../figures/17-memory-ledger.svg}}</figure>

*Figure 17.2 - the ledger. The prediction is the sum of the three regions; the observation is the process's peak RSS. The prediction is 88.0 MiB too high, and that gap is the lesson.*

The prediction is **too high**, by 92,303,872 bytes (88.0 MiB). Not because the model is wrong about what the run
*needs*, but because "needs" and "resident" are different words:

- Weights are read from the file, and pages that are never touched are never resident. A Q4_K_M file is not
  463.0 MiB of dirty pages.
- The KV region is *allocated* for 4,096 tokens and *written* for 34 prefill tokens plus three decode steps. That
  gap is 96.0 MiB of address space the kernel never had to back - larger than the whole 88.0 MiB the prediction is
  off by, which is why the sign was predictable and the size was not.
- The pool is the one exact number: it was allocated and written, so it is resident.

So a peak-memory prediction is not a bound in either direction. It is a **list of regions with their lifetimes**,
and its value is the list. When the observed number comes out *above* the prediction, you have found a region you
did not model - a staging buffer that was never freed, an arena that grew across rebuilds, a leak. When it comes
out below, you have found regions that are lazily touched, which is fine until a longer context touches them.

That is the method of this chapter and of the book: compute the regions, measure the process, explain the
difference. A number nobody can explain is not a measurement.

## Project

**Build step S17 · the arena.** Write the allocator after the graph is stable, because liveness is a property of
the graph, not of the op: a node's death is decided by its consumers, and a forward pass that allocates as it goes
cannot see them.

Order the work so every step has a test that fails when the step is wrong:

1. Naive: one buffer per node, sum the bytes. This is your *reference number*, and you need it to know whether
   reuse buys anything.
2. Liveness: `last_use` by build order, outputs and inputs pinned. Test: reuse a buffer one step too early and
   watch a two-input graph produce a wrong answer.
3. Views: an alias extends its parent. Test: `a -> view(a) -> add(view, b)` with a third node reusing `a`.
4. Classes: round to a ladder, and measure what the rounding cost against exact sizes.
5. The gate: ask the device before allocating, and write down the three states of the answer.

Then measure both graphs. If your decode arena is not much smaller than your prefill arena, you have not reused
anything: decode's nodes all have the same shape, and that is the case where reuse is supposed to be *perfect*.

## Deliverable

- A liveness computation over build order, with outputs and inputs pinned and views extending their parents
- One arena per backend with size classes, and a report of pool bytes, live bytes and peak live bytes
- Persistent regions allocated by name, surviving a graph rebuild
- A budget decision with three states - measured, failed, absent - and a note when the query fails
- A ledger that puts your prediction next to the process's own peak RSS

## How you know you passed

- Your naive/pool ratio is in the same order as this chapter's (28.2x), and your pool/peak-live ratio is not 1.0.
- A graph with two inputs and one reuse computes correctly - the input rule is the one that breaks silently.
- Your decode arena is a small fraction of your prefill arena, because identical shapes share exactly.
- A failed device query leaves the run *running*, with a note, instead of refusing every allocation.
- You can explain the sign and size of the gap between your prediction and the observed RSS.

## Reading

- code: `src/graph/alloc.rs`, `src/graph/allocplan.rs`, `src/graph/kvcache.rs` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/MEMORY-POLICY-DESIGN.md` (the accounting and the budget question), `docs/ARCHITECTURE.md` §"Memory"
- [roofline] Williams, Waterman and Patterson, *Roofline: an insightful visual performance model for multicore architectures* (CACM, 2009)
- [paged-attention] Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023)

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)

<!-- facts: alloc.provenance -->
*`scripts/alloc.py`; runs the engine with its per-node trace and re-implements `alloc_graph`'s liveness rule over the graph it executed (build order, consumers extend a buffer's life, outputs and inputs pinned to the end, views extending their parent); charges the transient nodes to a first-fit arena and the `kvcache_*` nodes to the persistent region they view; measures the process's peak RSS with `/usr/bin/time -l`; on **{'platform': 'macOS-27.2.1-arm64-arm-64bit-Mach-O', 'machine': 'arm64', 'python': '3.15.2'}**; peak RSS: median of 3 runs with the range beside it; the engine's ledger comes from `cargo test --release measure` at the same baseline; the naive column is arithmetic over one trace. Command: see [the provenance index](../appendix/provenance.md#alloc). Raw numbers: `facts/alloc.json`.*
<!-- /facts -->
