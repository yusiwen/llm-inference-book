# Chapter 8 · GEMV, GEMM, and the Thread Pool

> **Compute tier**: ✅ any laptop | **Status**: stub
> **Build step S8**: A matrix-vector path for decode and a matrix-matrix path for prefill, both using a persistent thread pool with an explicit work split.
> **Rule of this book**: The attention-specific arithmetic is Chapter 10. What this chapter owns is the general matmul shape problem: how work is divided, and why the two regimes want different kernels.

## First, the gist

**GEMV** (matrix times vector) is what decode does: one row of activations, thousands of rows of weights, no reuse of the weights. **GEMM** (matrix times matrix) is what prefill does: many tokens at once, so each weight row read serves several activations and the machine finally has enough arithmetic to keep its units busy. The same model therefore wants two different kernels, and choosing wrong costs several times the throughput. The third ingredient is the thread pool: spawning threads per call is fine until a token takes milliseconds, and then the spawn dominates the work.

## What you'll be able to do

- Choose between GEMV and GEMM for a given shape, with the arithmetic to justify it.
- Split work across threads by rows, with a work queue that has a bound and a policy for a slow consumer.
- Explain why a persistent pool beats thread-per-call at decode sizes, and what it costs in scheduling control.

## Lessons

- [ ] 1. Two shapes, two kernels: the arithmetic intensity of each
- [ ] 2. Row splitting and the false-sharing traps between worker threads
- [ ] 3. A persistent thread pool: spawn once, feed work, wait for completion
- [ ] 4. The bound on a queue, and what happens when a consumer lags (payoff: backpressure is a design decision, not an error message)
- [ ] 5. When one thread is faster: tiny models and the overhead floor

## Shapes and kernels

| Shape | Tokens | Weight reuse | Kernel | Your throughput |
|---|---|---|---|---|
| prefill, 2k prompt | 2000+ |  |  |  |
| decode, step N | 1 |  |  |  |
| speculative verify, k=4 | 5 |  |  |  |
| batched decode, 8 slots | 8 |  |  |  |

## Project

**Build step S8 · the work split.** Write the pool yourself before reading the reference one. The lesson is in the failure mode you hit without a bound: a slow worker, an unbounded queue, and memory that grows until the run dies.

Time it at decode sizes, not at prefill sizes. A pool that looks free at 2000 tokens can dominate at 1.

## Deliverable

- GEMV and GEMM paths with a test that both produce the same logits on the same weights
- A pool with a bounded queue and a documented policy for a lagging consumer
- A measured cross-over point: at how many tokens does GEMM overtake repeated GEMV on your machine

## How you know you passed

- Both paths produce identical logits on the same fixture, and the slower one is the oracle.
- The cross-over point is measured, quoted with the command, and you can explain its shape.
- A single-threaded run is available and matches, which is how you check the split rather than the pool.

## Reading

- code: `src/kernel/pool.rs`, `src/kernel.rs`, `src/graph/scheduler.rs` at commit `88ffa5d`
- docs: `docs/inference_e2e_walkthrough/10-cpu-matmul-kernels.md`, `08-scheduler-execute.md`
- decision: ADR-0059 (the batched worker queues against a bound and fails a lagging run)
- [scaling-inference] Pope et al., *Efficiently Scaling Transformer Inference* (2022) - arXiv:2211.05102

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
