# Chapter 17 · Allocation: Liveness, Arena, and Offload

> - **Compute tier**: ✅ any laptop
> - **Status**: stub
> - **Build step S17**: A liveness-based allocator that reuses one arena across a graph, keeps KV regions persistent, and decides what fits on a device before running.
> - **Rule of this book**: The KV cache's internal structure is Chapter 18. This chapter is about memory as a whole: who owns it, when it can be reused, and what happens when it does not fit.

## First, the gist

A transformer's activation memory is mostly transient: each intermediate is born, used, and dead a few nodes later. A naive allocator gives each one its own buffer and multiplies peak memory by the depth of the model. A **liveness** allocator computes, per node, the span in which a buffer is still needed, then hands out overlapping regions from one arena, which is what makes a 4 GB model run in a few hundred megabytes of activations. The second half is the hard case: the weights do not fit on the device, so some layers stay in host memory and the graph is cut at a layer boundary - a decision that must be made before execution, from a real query, never from a guess.

## What you'll be able to do

- Compute liveness spans over a graph and reuse buffers safely, with a test that would catch an early reuse.
- Size an arena for a real model and compare the prediction with the observed peak.
- Make an offload decision from a device memory query, and refuse cleanly when the query fails.

## Lessons

- [ ] 1. Transient versus persistent: which buffers outlive a single node
- [ ] 2. Liveness spans, and the aliasing rules that make reuse sound
- [ ] 3. One arena, many views: alignment, padding, and why fragmentation still happens
- [ ] 4. Offloading a prefix of layers: the auto-fit, its assumptions, and where it stops (payoff: a device query that fails is not a zero budget)
- [ ] 5. Peak-memory prediction versus observation: how to find the leak you did not model

## Memory ledger

| Region | Size (computed) | Size (observed) | Lifetime | Notes |
|---|---|---|---|---|
| weights resident |  |  |  |  |
| weights host-mapped |  |  |  |  |
| activations arena |  |  |  |  |
| KV cache |  |  |  |  |
| staging buffers |  |  |  |  |

## Project

**Build step S17 · the ledger that predicts the run.** Predict, then allocate, then compare. A memory ledger you can compute before the run is what lets you size a machine instead of crashing on it.

Test the reuse by deliberate misuse: shrink the arena below the requirement and require an error, not memory corruption or a wrong answer.

## Deliverable

- A liveness allocator with a reuse test and a shrink-too-far test
- A filled memory ledger for one real model, predicted versus observed
- An offload path that refuses when the device query fails

## How you know you passed

- Peak memory is within a few percent of your prediction, and the largest single region is the one you expected.
- An arena smaller than the true requirement fails loudly; no test accepts a corrupted result.
- You can state how many layers fit on your device and which layer boundary the fit chose.

## Reading

- code: `src/graph/alloc.rs`, `src/graph/allocplan.rs`, `src/graph/offload.rs` at the book's [pinned baseline](../appendix/baseline.md)
- decisions: ADR-0015 (the offload auto-fit takes a prefix), ADR-0016 (a failed device query is not a zero budget), ADR-0033 (the CUDA pool recycles exact byte lengths)
- docs: `docs/MEMORY-POLICY-DESIGN.md`, `docs/inference_e2e_walkthrough/07-allocator-liveness-kv.md`
- [paged-attention] Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023) - arXiv:2309.06180
- [scaling-inference] Pope et al., *Efficiently Scaling Transformer Inference* (2022) - arXiv:2211.05102

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
