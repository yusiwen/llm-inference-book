# Chapter 18 · The KV Cache Subsystem

> **Compute tier**: ✅ any laptop (the packed formats matter on devices; nothing here needs one) | **Status**: stub | **Build step S18**: A KV cache that can append, share a prefix between sessions, shrink, defragment, and be saved to and loaded from a versioned file. | **Rule of this book**: The attention that reads the cache is Chapter 10 and its device kernels are Chapter 25. This chapter is storage: cells, spans, layout, persistence.

## First, the gist

The cache that made attention affordable is also the part of an engine that grows, fragments and must be reconciled with a context window. Real engines therefore treat it as a subsystem rather than an array: memory is handed out in **cells**, a sequence is a list of **spans** over those cells, a prefix can be **shared** by several sessions, and the whole thing can be compacted when the allocator runs out of contiguous space. Two details separate a toy from a usable cache: the stored format (more bits per value is more accurate and reads more slowly) and the fact that a saved session is a versioned, checksummed file, not a memory dump.

## What you'll be able to do

- Implement append, share, resize and defragment over a cell-based cache, with an invariant test after each operation.
- Choose a KV format and defend it: bytes per token against the accuracy the identity gate requires.
- Version a session file so an older engine refuses it instead of misreading it.

## Lessons

- [ ] 1. Cells and spans: why a sequence is not a contiguous array
- [ ] 2. Prefix sharing: two sessions, one cache, and the reference counting that keeps it sound
- [ ] 3. Growing, shrinking and defragmenting under a context limit
- [ ] 4. Stored formats: f16, packed 8-bit, and what an integer cache does to attention
- [ ] 5. Sessions as files: version, checksum, and the refusal that protects the user (payoff: a memory dump reinterpreted is a silent wrong answer)

## Cache operations

| Operation | Invariant it must preserve | Cost | Test that catches it | Your status |
|---|---|---|---|---|
| append a token |  |  |  |  |
| share a prefix |  |  |  |  |
| resize |  |  |  |  |
| defragment |  |  |  |  |
| save / load |  |  |  |  |

## Project

**Build step S18 · storage, not an array.** Write the invariant list before the operations, then write one test per invariant. Cache bugs do not crash: they produce fluently wrong text, which is the hardest class of bug in this book to attribute.

Save a session, restart the process, reload it, and require the next token to be identical. That single test covers versioning, checksums and layout.

## Deliverable

- A cell-based cache with append, share, resize and defragment, each with an invariant test
- A versioned session file with a checksum and a refusal path for a mismatched version
- A measured bytes-per-token table for two formats at one context length

## How you know you passed

- A save, restart and reload yields an identical next token.
- An old or corrupt session file is refused with a message naming the version or the checksum.
- Defragmentation changes addresses and nothing else: a test asserts identical outputs before and after.

## Reading

- code: `src/graph/kvcache.rs`, `src/graph/kvformat.rs`, `src/graph/kvsession.rs` at the book's [pinned baseline](../appendix/baseline.md)
- decisions: ADR-0006 (the KV format is a per-engine gate), ADR-0014 (a KV session is a versioned file), ADR-0032 (the packed-KV staging window is f32)
- docs: `docs/KV-CACHE-DESIGN.md`
- [paged-attention] Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023) - arXiv:2309.06180

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
