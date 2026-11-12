# Chapter 10 · Attention and the KV Cache by Hand

> **Compute tier**: ✅ any laptop | **Status**: stub
> **Build step S10**: Causal attention with grouped queries over an explicitly managed KV cache: prefill writes, decode reads, and the results match a naive recompute.
> **Rule of this book**: This chapter establishes the semantics: what attention computes and how the cache makes it cheap. The storage subsystem (cells, spans, defragmentation) is Chapter 18, and the device kernels are Chapters 21 and 25.

## First, the gist

Attention lets every token look at the tokens before it and mix their values into its own representation. Naively that means recomputing the keys and values of the whole prefix at every step, which is quadratic work per token. The **KV cache** (key-value cache) removes that: keys and values are computed once, stored, and read back. **GQA** (grouped-query attention) then reuses one key/value head across several query heads, which is what makes the cache small enough to hold. The whole chapter is one idea applied twice: prefill fills the cache, decode reads it, and both must agree exactly.

## What you'll be able to do

- Write attention that matches a naive re-computation on the same inputs, at prefill and at decode.
- Lay out a KV cache for grouped-query attention and compute its size for a real model before allocating it.
- Explain why positions must be data passed to the kernel rather than part of a graph's structure, and what breaks when they are not.

## Lessons

- [ ] 1. Queries, keys, values: attention as three matmuls and a mask
- [ ] 2. Causal masking at prefill, and the off-by-one that survives every test that does not compare logits
- [ ] 3. Grouped-query attention: head reuse, cache size, and the arithmetic that decides it
- [ ] 4. Prefill attention versus decode attention: one big masked matmul against a streamed reduction
- [ ] 5. The cache as data: positions, spans, and the layout your kernels will assume (payoff: why a structural position means a rebuild per token)

## Cache arithmetic for one model

| Quantity | Formula | Value for your model | Measured |
|---|---|---|---|
| bytes per token per layer |  |  |  |
| bytes per 4k context |  |  |  |
| share of memory at 8k context |  |  |  |
| decode attention reads per step |  |  |  |

## Project

**Build step S10 · attention you can trust.** The naive implementation is the oracle: recompute attention over the whole prefix every step and compare. This costs a few seconds of runtime and saves days of debugging later in the book.

Deliberately break the mask in your own code and observe which test catches it. If none does, the test suite is the deliverable, not the kernel.

## Deliverable

- A correct prefill and decode attention with a naive-recompute oracle test
- A filled cache-size table for one real model, with the config fields you used
- A positions-as-data note: what the kernel receives, and what would break if positions were baked into the graph

## How you know you passed

- Prefill and decode agree with the naive recompute at the tolerance your precision ledger set.
- Your cache-size prediction is within a few percent of the observed allocation.
- You can produce a wrong answer by changing the mask and show the test that fails.

## Reading

- code: `src/graph/kvcache.rs`, `src/graph/kvformat.rs`, `src/graph/cpu_backend.rs` (attention) at commit `88ffa5d`
- docs: `docs/KV-CACHE-DESIGN.md`, `docs/inference_e2e_walkthrough/11-attention-vecops-kv.md`
- decisions: ADR-0002 (topology is a function of graph params, so positions cannot be structure), ADR-0006 (the KV format is a per-engine gate)
- [gqa] Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023) - arXiv:2305.13245

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
