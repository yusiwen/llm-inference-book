# Chapter 24 · CUDA III: The Kernel Ladder

> **Compute tier**: ⚠️ needs an NVIDIA GPU; every kernel here is a re-implementation you can run on rented time | **Status**: stub | **Build step S24**: Four kernels in order: elementwise, dequantize-and-gather, a tiled f16 GEMM using tensor-core fragments, and an integer matmul for prefill plus a matrix-vector path for decode. | **Rule of this book**: Attention kernels are Chapter 25. This chapter is the matmul family, and the ladder is cumulative: each rung is written only after the previous one passes the identity gate.

## First, the gist

The matmul ladder is the core of a GPU inference engine, and it is climbed in a fixed order. Elementwise and gather kernels teach the memory hierarchy on work where arithmetic does not hide anything. A tiled **GEMM** (general matrix multiply) then shows why shared memory exists: a tile of the weight matrix is loaded once and reused by many threads, which is how a GPU stops being bandwidth-bound. The last rung is the one that matters for inference: the weights are quantized on disk, so the kernel dequantizes inside the multiply - 8-bit integer arithmetic, tensor-core instructions, and one scale correction - and prefill runs a matrix-matrix version while decode runs a matrix-vector version of the same idea.

## What you'll be able to do

- Implement a tiled GEMM and show the shared-memory reuse with counts, not adjectives.
- Write an integer matmul over quantized weights that passes the CPU identity gate, and a matrix-vector sibling for decode.
- Choose the right kernel for a shape (1 token, 5 tokens, 2000 tokens) from measured numbers.

## Lessons

- [ ] 1. Elementwise and gather: the kernels where memory hierarchy is the whole lesson
- [ ] 2. Dequantize inside the multiply: why an engine never materializes floats
- [ ] 3. Shared memory tiling, and the tile arithmetic that decides reuse
- [ ] 4. Tensor cores: fragments, layouts, and the f16 path that prefill uses (payoff: what a warp-level matrix instruction buys and demands)
- [ ] 5. The integer matmul: prefill with matrix-matrix, decode with matrix-vector, and the small-M problem

## The ladder

| Kernel | Shape | Bound by | Measured (yours) | Passes the identity gate? |
|---|---|---|---|---|
| elementwise | row |  |  |  |
| gather / embedding | row |  |  |  |
| tiled f16 GEMM | N x K |  |  |  |
| int8 matmul (prefill) | M>16 |  |  |  |
| matrix-vector (decode) | M=1..8 |  |  |  |

## Project

**Build step S24 · climb one rung at a time.** Each kernel is written against the previous rung's output, and the CPU path is the oracle for all of them. Skipping a rung does not save time; it moves the debugging into a larger kernel.

Record the shape, the tile size and the throughput for every kernel. A number without a shape is not a measurement.

## Deliverable

- Four kernels, each with a shape-annotated measurement and an identity test
- A table naming the kernel your engine chooses per shape, and the threshold where it switches
- One kernel's profiler reading, quoted, with the bottleneck it names

## How you know you passed

- Every kernel matches the CPU oracle at the tolerance your precision ledger allows.
- The tile arithmetic predicts the observed speedup within a factor you can explain.
- The dispatch threshold is measured, not copied from another engine's documentation.

## Reading

- code: `src/cuda/kernels/ops_elementwise.cu`, `gemm_wmma.cu`, `mmq_int8.cu`, `mmq_raw.cu`, `mmq_bt_q6k.cu`, `mmvq_q6k.cu`, `mmvq_multi.cu` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/cuda_tutorial/03-kernels-elementwise.md`, `04-kernels-matmul.md` (the audit of these two chapters is in [the CUDA docs audit](../appendix/cuda-docs-audit.md))
- docs: `docs/LLAMA-CPP-MMQ-ANALYSIS.md` for the reference implementation's design, `docs/CUDA-TECH-PRIMER.md` for terms
- [ptx] *PTX ISA*, NVIDIA - no arXiv ID (vendor manual; the `mma` and `cp.async` instructions)
- [sze] Sze et al., *Efficient Processing of Deep Neural Networks* (2020) - no arXiv ID (book)

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
