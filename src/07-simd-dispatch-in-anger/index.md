# Chapter 7 · SIMD Dispatch in Anger

> **Compute tier**: ✅ any laptop (this chapter is CPU only; no GPU needed) | **Status**: stub
> **Build step S7**: AVX2 and NEON versions of your dot product, selected at compile time, each checked against the scalar oracle.
> **Rule of this book**: This chapter is about making one operation fast on one core. Threads, cache blocking and the matrix-vector versus matrix-matrix decision are Chapter 8.

## First, the gist

A modern CPU can multiply eight or sixteen pairs of integers per instruction if you ask it to, and the compilers will not do it for you reliably in a kernel this branchy. **SIMD** (single instruction, multiple data) is the technique: load a vector of weights and a vector of activations, multiply and accumulate eight lanes at a time, then fold the lanes together. Two instruction sets matter here - **AVX2** on x86 and **NEON** with its `SDOT` integer dot-product extension on ARM - and the interesting part is not writing either one, it is keeping them honest against the scalar oracle and knowing which quantized formats each one can actually accelerate.

## What you'll be able to do

- Write an AVX2 and a NEON kernel for one quantized format and dispatch between them at compile time.
- Explain why ARM's `SDOT` changes what is worth vectorizing compared with x86, and which formats have no fast path on AVX2.
- Measure the speedup you actually got, on your own machine, with the test that proves it did not change the answer.

## Lessons

- [ ] 1. Lanes, masks and horizontal sums: SIMD without intrinsics anxiety
- [ ] 2. AVX2 in practice: nibble unpacking, FMA, and the formats that fit
- [ ] 3. NEON and SDOT: the integer dot product ARM gives you for free
- [ ] 4. Dispatch: compile-time selection, and the fallback that must stay correct
- [ ] 5. Proving a fast path: the identity test, then the timer (payoff: a fast wrong answer is the most expensive bug in this book)

## Coverage by instruction set

| Format | Scalar | AVX2 + FMA | NEON + SDOT | Measured speedup (yours) |
|---|---|---|---|---|
| Q4_0 |  |  |  |  |
| Q5_0 |  |  |  |  |
| Q8_0 |  |  |  |  |
| Q4_K |  |  |  |  |
| Q6_K |  |  |  |  |

## Project

**Build step S7 · the dispatch you will regret skipping.** Write one format's kernel by hand, get it identical to the oracle, and only then look at how the reference engine organizes the other seven. The temptation is to copy eight kernels at once, and it produces eight untested kernels.

Keep every kernel behind the same signature: the caller should never branch on instruction set.

## Deliverable

- At least one hand-written SIMD kernel per instruction set, checked against the scalar oracle
- A build-time dispatch (target feature or architecture) with a scalar fallback
- A measured single-core speedup table for your machine, with the command quoted

## How you know you passed

- Every SIMD kernel passes the same identity test as the scalar path, on the same fixture.
- The measured speedup is quoted with the test conditions (block size, dataset size) and is reproducible.
- Removing the SIMD path does not change results: the fallback is genuinely equivalent.

## Reading

- code: `src/quants/avx2.rs`, `src/quants/avx2_correctness.rs`, `src/quants/neon.rs`, `src/quants/neon_correctness.rs` at commit `88ffa5d`
- docs: `docs/CPU_OPTIMIZATIONS.md`
- [arm-sdot] *Arm Architecture Reference Manual* and the ACLE integer dot-product instructions (`SDOT`, `SMMLA`) - no arXiv ID (vendor manual)

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
