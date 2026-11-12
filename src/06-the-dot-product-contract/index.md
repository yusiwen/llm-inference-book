# Chapter 6 · The Dot Product Is the Engine

> **Compute tier**: ✅ any laptop | **Status**: stub
> **Build step S6**: A matmul that computes one row of a quantized weight matrix against an activation vector, with the activation quantized to Q8_0 first.
> **Rule of this book**: This chapter is about the arithmetic contract and its scalar version. SIMD is Chapter 7; splitting the work across threads and choosing between matrix-vector and matrix-matrix shapes is Chapter 8.

## First, the gist

Almost every floating-point operation in a transformer is a dot product between a weight row and an activation vector, and there are billions of them per token. An engine's core trick is to stop multiplying floats: it quantizes the activation to 8-bit integers once per row, then computes the dot product in integer arithmetic with a single scale correction at the end. That is faster because integer multiply-add is cheaper and because the weight is already quantized on disk. This chapter states the contract precisely, because every kernel in the rest of the book - scalar, SIMD, Metal, CUDA - implements the same one.

## What you'll be able to do

- Write a scalar quantized dot product that matches a dequantize-then-dot reference within a stated bound.
- Explain the Q8_0 activation policy and where its rounding error enters, without hand-waving.
- Name the dispatch a given `(weight dtype, activation dtype)` pair requires, and what an unsupported pair must do (refuse, never fall back silently).

## Lessons

- [ ] 1. One operation, billions of times: where the dot product appears in a transformer
- [ ] 2. Quantize the activation, once: Q8_0 blocks, scales, and the error budget
- [ ] 3. The integer dot product and the single scale correction at the end
- [ ] 4. The dispatch contract: dtype pair in, kernel out, and the refusal rule (payoff: why a silent CPU fallback is a bug, not a safety net)
- [ ] 5. A scalar reference you can trust: slow, obvious, and the oracle for every kernel after it

## The contract

| Weight dtype | Activation dtype | Kernel family | Same result as dequantize-then-dot? | Your status |
|---|---|---|---|---|
| Q4_0 | Q8_0 |  |  |  |
| Q8_0 | Q8_0 |  |  |  |
| Q4_K | Q8_K |  |  |  |
| Q4_K | f32 |  |  |  |
| f32 | f32 |  |  |  |

## Project

**Build step S6 · one row, then one matrix.** Build the scalar path first and use it as the oracle for everything else. Every later speed number in this book is a claim that the fast path agrees with this one.

Keep the reference implementation in the repository, behind a test-only feature, for the life of the project. The moment it is deleted, the fast paths become their own oracle.

## Deliverable

- A scalar quantized dot product with a reference cross-check test
- A dispatch table in code (dtype pair to function) with an explicit error for unsupported pairs
- A measured error bound for Q8_0 activations on one real row, stated as a number

## How you know you passed

- Your kernel and dequantize-then-dot agree within the bound you wrote down, and the bound is justified, not copied.
- An unsupported dtype pair produces an error naming the pair, and no test accepts a fallback.
- You can point at the exact line where the scale correction happens.

## Reading

- code: `src/kernel.rs`, `src/kernel/dispatch.rs`, `src/quants.rs` at commit `88ffa5d`
- docs: `docs/inference_e2e_walkthrough/10-cpu-matmul-kernels.md`
- decision: ADR-0013 (the CPU quantizes activations to Q8_0; a device reads f32)

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
