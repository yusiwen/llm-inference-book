# Chapter 22 · CUDA I: What Kind of Machine a GPU Is

> - **Compute tier**: ⚠️ needs an NVIDIA GPU with a toolkit (the DGX Spark in this book's notes, or rented time); the toys run anywhere with `nvcc`
> - **Status**: stub
> - **Build step S22**: A compiled and run kernel of your own: vector add, then a branch-divergent variant you can explain, with the launch arithmetic written down first.
> - **Rule of this book**: This chapter is the machine model and the first kernel. The Rust seam is Chapter 23, the kernel families are Chapter 24, attention and graphs are Chapter 25, the campaign is Chapter 26.

## First, the gist

A GPU is not a fast CPU. It is a machine with hundreds of simple cores arranged in fixed-size groups, a deep memory hierarchy, and a scheduler that hides latency by switching between warps instead of by predicting. **SIMT** (single instruction, multiple threads) means a group of 32 threads executes one instruction in lockstep, so a branch that splits the group costs both paths. Once you can predict, from the counts alone, what a launch will do and which memory level it will be waiting on, the rest of CUDA is engineering. This chapter builds that prediction and then compiles a kernel to check it.

## What you'll be able to do

- Predict a launch: blocks, warps, registers, occupancy, and the memory level a kernel will stall on.
- Write, compile and run a kernel, with error checking that names the failing call.
- Explain divergence, coalescing and the shared-memory bank structure from the hardware's side, not from a slogan.

## Lessons

- [ ] 1. Host and device: two address spaces and one program
- [ ] 2. Threads, warps, blocks, grids: the launch arithmetic you write down before the code
- [ ] 3. The memory hierarchy and what each level costs in cycles
- [ ] 4. Coalescing, divergence and bank conflicts: three ways a correct kernel runs ten times slow (payoff: why decode is a bandwidth problem and prefill is not)
- [ ] 5. Your first kernel, compiled and run, with the counts predicted first

## Launch sheet

| Kernel | Blocks | Threads/block | Warps | Bytes moved | Predicted bound |
|---|---|---|---|---|---|
| vector add |  |  |  |  |  |
| divergent variant |  |  |  |  |  |
| elementwise gate |  |  |  |  |  |
| one decode matmul row |  |  |  |  |  |

## Project

**Build step S22 · predict, then launch.** Write the launch sheet before you write the code, every time, for the whole chapter. The habit is what makes Chapter 26's profiling readable instead of mysterious.

Use the machine you will actually measure on, and record its CUDA version, driver and SM count. A toy output from another machine is not evidence about yours.

## Deliverable

- Two working kernels with error-checked launches and observed output recorded
- A filled launch sheet for each, with predicted and observed timings
- A one-page note on what your GPU's memory bandwidth implies for decode

## How you know you passed

- Your predicted bound is within a factor of two of the measured time, and you say why it errs.
- Nsight Compute (`ncu`) confirms the bottleneck you named, and the counter it reports is the one you predicted.
- The divergent variant is measurably slower and you can explain the mechanism without the manual.

## Reading

- code: `src/cuda/kernels/*.cu` (18 files, `common.cuh` first), `src/cuda/ffi_runtime.rs` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/cuda_tutorial/01-gpu-mental-model.md`, `02-minimal-cuda.md` (the book's own audit of the tutorial is in [the CUDA docs audit](../appendix/cuda-docs-audit.md))
- docs: `docs/CUDA-TECH-PRIMER.md` for terminology at reference depth
- docs: `docs/GPU_SAFETY.md` before you touch any GPU code you did not write
- [cuda-guide] *CUDA C++ Programming Guide*, NVIDIA - no arXiv ID (vendor manual)
- [pmpp] Kirk and Hwu, *Programming Massively Parallel Processors* (4th edition, 2022) - no arXiv ID (book)

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
