# Chapter 22 · CUDA I: What Kind of Machine a GPU Is

> - **Compute tier**: ⚠️ needs an NVIDIA GPU with a toolkit (the DGX Spark in this book's notes, or rented time); the toys run anywhere with `nvcc`
> - **Status**: written - the launch sheet is measured on a GB10 by `scripts/cuda/launch_sheet.cu`, predictions included and two of them deliberately written to be slow; what is *not* here is the Rust seam (Chapter 23) or the kernels that carry a model (Chapter 24)
> - **Build step S22**: A compiled and run kernel of your own: vector add, then a branch-divergent variant you can explain, with the launch arithmetic written down first.
> - **Rule of this book**: This chapter is the machine model and the first kernel. The Rust seam is Chapter 23, the kernel families are Chapter 24, attention and graphs are Chapter 25, the campaign is Chapter 26.

## First, the gist

A GPU is not a fast CPU. It is a machine with tens of simple cores arranged in fixed-size groups, a deep memory hierarchy, and a scheduler that hides latency by switching between warps instead of by predicting. **SIMT** (single instruction, multiple threads) means a group of 32 threads executes one instruction in lockstep, so a branch that splits the group can cost both paths.

That is the standard paragraph, and this chapter exists to check it. The machine:

<!-- facts: cuda-launch.cuda-device -->
| | |
|---|---|
| Device | **NVIDIA GB10** (compute capability 12.1) |
| Multiprocessors | 48 |
| Peak SM clock, reported | 2,418 MHz |
| Streaming read/write bandwidth, measured here | **247 GB/s** |
| Blocks x threads, every kernel | 131,072 x 256 |
| Repetitions per kernel | 20, medians of the per-repetition event timings |
<!-- /facts -->

Forty-eight multiprocessors, 2,418 MHz, and one number that matters more than both: a streaming copy reads and writes at about a quarter of a terabyte per second, which is the speed limit every kernel in the next four chapters is judged against.

The chapter's method is a **launch sheet**: bytes, a prediction from those bytes, and then the measurement - written down before the code, every time:

<!-- facts: cuda-launch.cuda-launch-sheet -->
| Kernel | Bytes moved | Predicted | Measured | GB/s | of stream | Measured / predicted |
|---|---|---|---|---|---|---|
| `copy` | 268 MB | 1086 us | **1086 us** | 247 | 100% | 1.00x |
| `vecadd` | 403 MB | 1629 us | **1655 us** | 243 | 98% | 1.02x |
| `vecadd_divergent` | 403 MB | 1629 us | **1626 us** | 248 | 100% | 1.00x |
| `vecadd_branchy` | 403 MB | 1629 us | **1662 us** | 242 | 98% | 1.02x |
| `relu` | 268 MB | 1086 us | **1084 us** | 248 | 100% | 1.00x |
| `gemv_row` | 34 MB | 136 us | **212 us** | 159 | 64% | 1.56x |

Prediction is `bytes / 247 GB/s`, and that rate is the first row of the table - this machine's own streaming copy, not a number borrowed from another chapter. Read the last column: **5 of 6 kernels land within a tenth of their byte count**, including the two that were written to be slow, and only `gemv_row` escapes at 1.56x the prediction - 64 per cent of the streaming rate.
<!-- /facts -->

Two of the kernels in that table were written to be slow, and neither of them is. The row that is slow was written to be ordinary, and its problem is not arithmetic. That reversal is the chapter.

<figure class="book-figure">{{#include ../figures/22-memory-hierarchy.svg}}</figure>

*Figure 22.1 - the GPU's memory hierarchy from registers to device memory, what each level costs, and the two levels a decode matmul spends its life in.*

<figure class="book-figure">{{#include ../figures/22-access-patterns.svg}}</figure>

*Figure 22.2 - three warps reading the same 32 values: coalesced, strided, diverged - and the transactions each pattern turns them into.*

## What you'll be able to do

- Predict a launch: blocks, warps, registers, occupancy, and the memory level a kernel will stall on.
- Write, compile and run a kernel, with error checking that names the failing call.
- Explain divergence, coalescing and the shared-memory bank structure from the hardware's side - including the cases where the slogan is wrong.

## Lessons

- [ ] 1. Host and device: two address spaces and one program
- [ ] 2. Threads, warps, blocks, grids: the launch arithmetic you write down before the code
- [ ] 3. The memory hierarchy and what each level costs in cycles
- [ ] 4. Coalescing, divergence and bank conflicts: three ways a correct kernel runs ten times slow - and the measurement that separates them
- [ ] 5. Your first kernel, compiled and run, with the counts predicted first

### Lesson 1 · Host and device: two address spaces and one program

A CUDA program is one program with two halves. The host half is ordinary code - it allocates, it launches, it waits. The device half is compiled to PTX and runs on cores that cannot see the host's pointers. A pointer is either a host pointer or a device pointer, and dereferencing the wrong one is a crash or, worse, a plausible wrong answer.

Everything that crosses the boundary is explicit, and the engine's FFI layer is that boundary written down:

<!-- code: src/cuda/ffi_runtime.rs::cudaMemcpy@e844591 -->
```rust
    pub(crate) fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    pub(crate) fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
    ) -> i32;
```
<!-- /code -->

Two functions and a `kind` argument that says which direction the bytes go. That is the whole model: `cudaMalloc` produces memory the host cannot read, `cudaMemcpy` is the only way across, and the `kind` is the price of forgetting that a device has its own address space.

The machine in this chapter's measurements is *almost* simpler, and the difference is worth knowing: a GB10 has unified memory, so a pointer from `cudaMalloc` is physically the memory the CPU already touches. It is still two address spaces from the program's side - the copies still happen - but they cost less than a PCIe transfer, and a prediction that assumes a discrete card's copy price will be wrong here. It is the same distinction Chapter 21 drew for Apple silicon, from the other direction: shared memory makes a copy cheap, not unnecessary.

### Lesson 2 · Threads, warps, blocks, grids: the arithmetic you write down first

A launch is three numbers - threads per block, blocks per grid, registers per thread - and everything else follows from them.

- A **warp** is 32 threads. It is the unit of scheduling: one instruction is issued for the whole warp, and its 32 lanes execute it in lockstep.
- A **block** is a group of warps that can share memory and synchronise, up to 1,024 threads. It is the unit of co-residency: every warp of a block runs on one multiprocessor.
- A **grid** is every block of one launch. Its blocks may run in any order, on any multiprocessor, at any time - which is why a kernel must not depend on which block got there first.

So `blocks x threads_per_block` has to cover the work, and the warps per block decide how much latency the scheduler can hide: with one warp per block there is nothing to switch to when it stalls. A block of 128 or 256 threads is the common choice - eight warps to rotate through, without exhausting the register file.

The engine writes the geometry where the kernel's contract is visible, in the launch-bounds annotation:

<!-- code: src/cuda/kernels/ops_elementwise.cu::rms_norm_quant_pad40@e844591 -->
```cpp
__global__ void __launch_bounds__(128) rms_norm_quant_pad40(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    uint8_t* __restrict__ q8,
    int d, float eps, int n
) {
    int row = blockIdx.x;
    if (row >= n) return;
```
<!-- /code -->

`__launch_bounds__(128)` is a promise to the compiler - this kernel is never launched with more than 128 threads - which lets it spend registers freely without spilling. The `if (row >= n) return` is the other half of the arithmetic: the grid covers rows, so the grid size is `ceil(n / threads)` and the last block is usually partly outside. The ratio of blocks to rows is what tells you whether a launch is one wave over the machine or two, and it is the first number to look at when a kernel is slower than its byte count says it should be.

`__restrict__` on every pointer is the third part of the same promise: it tells the compiler the pointers do not alias, so a load from one does not have to be re-issued after a store through another. On a GPU, where the compiler decides what stays in registers across a loop, that permission is worth real time.

### Lesson 3 · The memory hierarchy and what each level costs

The hierarchy is a funnel, and the ratios between its levels are why a GPU needs thousands of threads:

| Level | Roughly | Who can see it | Who fills it |
|---|---|---|---|
| registers | an access | one thread | the compiler |
| shared memory / L1 | tens of cycles | the threads of one block | you, explicitly |
| L2 | hundreds of cycles | the whole device | hardware |
| device memory (HBM) | 500-800 cycles | the whole device | `cudaMemcpy`, kernels |

The absolute cycle counts move by generation; the ratios do not. A single thread waiting on HBM loses hundreds of cycles, and the scheduler hides that by switching to another warp that is ready - which is why **occupancy** (how many warps a multiprocessor keeps resident) is a first-class number, and why a kernel with one warp per block can hide nothing at all.

Shared memory is the level you program, and six lines of the engine's reduction show all three levels at once:

<!-- code: src/cuda/kernels/common.cuh::mmvq_block_reduce@e844591 -->
```cpp
static __device__ __forceinline__ void mmvq_block_reduce(
    float acc, float* __restrict__ output, int od, int t
) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    __shared__ float warp_sums[8];
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) v += warp_sums[k];
        output[(size_t)t * od + (size_t)blockIdx.x] = v;
    }
}
```
<!-- /code -->

Read it as a descent through the funnel. The first loop is a **warp reduction**: `__shfl_xor_sync` swaps values between lanes of one warp, five steps for 32 values, and never touches memory - the data is in registers. Then one lane per warp writes its partial sum into `__shared__` memory, `__syncthreads()` makes that visible to the whole block, and lane 0 sums eight floats and writes one result to global memory. Registers, then shared, then global: three levels, and the fast ones do the work.

**Bank conflicts** live in that `__shared__ float warp_sums[8]`. Shared memory is served in 32 banks of 4 bytes, and two lanes hitting the same bank in one cycle serialise. Eight consecutive floats land in eight different banks, so the write is conflict-free; an array indexed by `threadIdx.x * 8` would put all eight writes in one bank and cost eight cycles instead of one. The declaration is one line, and it is the difference between a free store and a serialised one.

### Lesson 4 · Coalescing, divergence and bank conflicts - measured, and mostly invisible

<!-- no-code: the evidence in this lesson is the chapter's own launch sheet (`scripts/cuda/launch_sheet.cu`), quoted below; the engine's answer to a strided read is the `mmvq` family, which is Chapter 24's subject and would be a different lesson here -->

This is where the chapter stops repeating the folklore, because it can test it. Three access patterns, 32 lanes each:

- **Coalesced**: lane `i` reads element `i`. The 32 loads fall inside one 128-byte line per 32 floats, so the hardware issues one transaction and every lane gets what it asked for.
- **Strided**: lane `i` reads element `i * K`. Now the 32 loads touch 32 different lines, and the hardware issues 32 transactions for 128 bytes of useful data.
- **Diverged**: the lanes take different sides of a branch, so the warp executes both sides serially and the cost is the sum of the paths.

All three are in the launch sheet, and the measurement refused to cooperate with the first and the third:

<!-- facts: cuda-launch.cuda-launch-sheet -->
| Kernel | Bytes moved | Predicted | Measured | GB/s | of stream | Measured / predicted |
|---|---|---|---|---|---|---|
| `copy` | 268 MB | 1086 us | **1086 us** | 247 | 100% | 1.00x |
| `vecadd` | 403 MB | 1629 us | **1655 us** | 243 | 98% | 1.02x |
| `vecadd_divergent` | 403 MB | 1629 us | **1626 us** | 248 | 100% | 1.00x |
| `vecadd_branchy` | 403 MB | 1629 us | **1662 us** | 242 | 98% | 1.02x |
| `relu` | 268 MB | 1086 us | **1084 us** | 248 | 100% | 1.00x |
| `gemv_row` | 34 MB | 136 us | **212 us** | 159 | 64% | 1.56x |

Prediction is `bytes / 247 GB/s`, and that rate is the first row of the table - this machine's own streaming copy, not a number borrowed from another chapter. Read the last column: **5 of 6 kernels land within a tenth of their byte count**, including the two that were written to be slow, and only `gemv_row` escapes at 1.56x the prediction - 64 per cent of the streaming rate.
<!-- /facts -->

The `vecadd_divergent` and `vecadd_branchy` rows sit within two per cent of plain `vecadd`. The first has every warp straddling a symmetric if/else; the second has half of every warp running a 32-iteration loop while the other half does one multiply. **Both are genuinely divergent and neither is slower**, and the reason is Lesson 3's model: this kernel is waiting on memory, not on instructions. A warp that executes both sides of a branch still spends most of its time waiting for the loads that both sides need, and the scheduler fills the gap with other warps.

So divergence is not a myth - it is **conditional on being the bottleneck**. The same branch inside a compute-bound kernel, one whose operands are already in registers, serialises two paths for real, which is why Chapter 24's integer matmuls are written to avoid branches inside their inner loops. A chapter that says "divergence costs ten times" without saying *when* is teaching a slogan.

Coalescing has no such excuse, and the kernel that shows it is four lines long:

```cuda
__global__ void k_gemv_row(const float *w, const float *x, float *y, int m, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    const float *r = w + (size_t)row * k;
    float acc = 0.0f;
    for (int j = 0; j < k; j++) acc += r[j] * x[j];
    y[row] = acc;
}
```

Every line of it looks correct, and every line is: the index is bounded, the row pointer is computed once, the accumulation is a register, and `x[j]` is served from cache for all 32 lanes at once. Only the *warp* is wrong. Each thread gives itself one row of a matrix and walks that row in order - contiguous within a thread, but across a warp the 32 lanes are reading 32 *different* rows, `K` floats apart, so every load is its own transaction and the kernel reaches two thirds of the streaming rate while doing nothing but read. That is not a toy result: it is the shape of a decode matmul (`y = W x`), and it is why Chapter 1's ceiling is a bandwidth ceiling and not a compute one. The art of a decode kernel is arranging those reads so the transactions come back full; Chapter 24's `mmvq` family is that art, and this chapter is the reason it has to exist.

### Lesson 5 · Your first kernel, compiled and run

The first kernel should be the smallest thing that can be wrong, and then it should be checked against the arithmetic from Lesson 2. Element-wise add is that kernel:

<!-- code: src/cuda/kernels/ops_elementwise.cu::add_f32@e844591 -->
```cpp
__global__ void add_f32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ z,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    z[tid] = x[tid] + y[tid];
}
```
<!-- /code -->

Six lines with all four previous lessons in them: a global index built from the block and thread ids, the bound check that covers the last partial block, and pointers the compiler is allowed to assume are distinct.

Its launch sheet, written before the code:

| | `vecadd`, 32 Mi floats |
|---|---|
| Bytes moved | 12 per element - two reads, one write |
| Blocks x threads | 131,072 x 256, eight warps resident per block |
| Predicted | 384 MB / 247 GB/s = 1.59 ms |
| Measured | 1.65 ms, **1.02x the prediction** |

Landing within two per cent is the *normal* outcome for a kernel whose access pattern is regular, and that is worth saying plainly: when your first measurement is far from your prediction, the odds are that you counted the bytes wrong, not that the GPU is mysterious. Check the byte column first; it is the one that catches the mistakes.

**Errors must name the call.** Every CUDA call returns a status, and ignoring it turns a failed launch into a wrong answer three layers later. The instrument in this chapter checks every call and prints the expression, the file and the line:

```cuda
#define CHECK(call)                                                                  \
    do {                                                                             \
        cudaError_t err = (call);                                                    \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error: %s at %s:%d (%s)\n", #call, __FILE__,        \
                    __LINE__, cudaGetErrorString(err));                              \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)
```

`#call` is the whole trick: the message says `cudaMalloc(&w, ...)` failed at line 121, not that "something" failed.

## Launch sheet

The table the chapter asks you to fill in before every kernel, filled in here for the four it measures:

| Kernel | Blocks | Threads/block | Warps | Bytes moved | Predicted bound | Measured |
|---|---|---|---|---|---|---|
| vector add | 131,072 | 256 | 8 | 384 MB | 1.59 ms (bandwidth) | 1.65 ms |
| divergent variant | 131,072 | 256 | 8 | 384 MB | 1.59 ms (bandwidth) | 1.63 ms - the branch is free |
| elementwise gate | 131,072 | 256 | 8 | 256 MB | 1.04 ms (bandwidth) | 1.08 ms |
| one decode matmul row | 32 | 256 | 8 | 33 MB | 0.14 ms (bandwidth) | 0.21 ms - strided, 59% of stream at this size |

The last row is the one to keep. It is a decode matmul's access pattern, and it is the only one of the four that misses its own byte count - not because it computes more, but because it asks for its bytes one row at a time.

### The counterfactual: why that row is slow, and how far you can push the claim

"Because it asks for its bytes one row at a time" is an *explanation*, and an explanation needs a second implementation to carry it. One kernel measuring 64 per cent of stream shows only that it is slow. So the same GEMV was written twice - the sheet's version, and a version where one warp owns a row and its 32 lanes split the columns, reducing with `__shfl_down` - and both were run at three sizes:

<!-- facts: gemv-scale.gemv-scale -->
| one GEMV, two access patterns | one thread per row | share of stream | one warp per row | share of stream | strided is |
|---|---|---|---|---|---|
| 1024 x 1024 weights | 37.4 GB/s | 15.7% | **629.1 GB/s** | 263.4% | **16.81x** slower |
| 4096 x 4096 weights | 140.7 GB/s | 58.9% | **270.6 GB/s** | 113.3% | **1.92x** slower |
| 8192 x 8192 weights | 109.5 GB/s | 45.8% | **262.6 GB/s** | 109.9% | **2.40x** slower |

The identical arithmetic, written two ways, differs by **1.92x to 16.81x** - and the two agree numerically (relative deviation in the 1e-5 range), so the difference is the access pattern and nothing else. That is the counterfactual the launch sheet was missing: one kernel can show that a shape is slow, only a second implementation can show *why*.

Two things this table taught that the single-size measurement could not. The stream share is **not a constant** - it moves from 15.7 per cent to 58.9 per cent across a factor of eight in problem size, so quoting it without the size is quoting a coincidence. And the coalesced kernel reaches **263 per cent of the streaming rate** at the smallest size, which is impossible unless those weights never left cache - so for a matrix that fits in L2, 'fraction of stream' is the wrong denominator entirely. The number that survives every size is the ratio.
<!-- /facts -->

The arithmetic is identical and the outputs agree to 1e-5, so the difference is the access pattern and nothing else. **That is the evidence the 64 per cent was standing in for**, and reading it changes two things about how the number should be quoted.

**It is not a constant.** Across a factor of eight in problem size the strided kernel's share of stream moves between 15.7 and 58.9 per cent, so "64 per cent of stream" without the size beside it is a coincidence dressed as a property. What survives every size is the *ratio*: 1.92x to 16.81x slower than the same arithmetic coalesced.

**And the denominator can be wrong.** The coalesced kernel reaches 263 per cent of the streaming rate at 1024 square - which is impossible unless those weights never left cache. For a matrix that fits in L2, "fraction of stream" is not a meaningful unit at all; the stream is what you pay when you actually go to DRAM, and that kernel did not.

The practical form of both findings is the same: when a kernel misses its byte-count prediction, **write the counterfactual before you name the cause**, and say what size you measured at. Chapter 24 runs into the identical trap from the other direction, with a tile that saves traffic nobody was waiting for.

<!-- facts: gemv-scale.provenance -->
*`scripts/probe/gemv_scale.cu`; runs the same GEMV twice - one thread per row, then one warp per row with a __shfl_down reduction - at 1024, 4096 and 8192 square weights, and measures the machine's own streaming rate in the same process as the denominator; milliseconds and GB/s for both kernels at each size, their share of the measured streaming rate, the ratio between them, and the maximum relative difference between their outputs. Command: see [the provenance index](../appendix/provenance.md#gemv-scale). Raw numbers: `facts/gemv-scale.json`.*
<!-- /facts -->

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
- You can explain why the divergent variant of a *bandwidth-bound* kernel is not slower, and name the kind of kernel in which it would be.

<!-- facts: cuda-launch.provenance -->
*`scripts/cuda/launch_sheet.cu`; compiles five kernels with the machine's own `nvcc` and times each one with CUDA events over 20 repetitions, the streaming copy first because every later prediction divides by it; 20 runs; per-kernel milliseconds, the byte count it moved, and the implied GB/s; the prediction is computed from the first row, so the table is self-referential by construction. Command: see [the provenance index](../appendix/provenance.md#cuda-launch). Raw numbers: `facts/cuda-launch.json`.*
<!-- /facts -->

## Reading

- code: `src/cuda/kernels/*.cu` (17 files, `common.cuh` first), `src/cuda/ffi_runtime.rs` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/cuda_tutorial/01-gpu-mental-model.md`, `02-minimal-cuda.md` (the book's own audit of the tutorial is in [the CUDA docs audit](../appendix/cuda-docs-audit.md))
- docs: `docs/CUDA-TECH-PRIMER.md` for terminology at reference depth
- docs: `docs/GPU_SAFETY.md` before you touch any GPU code you did not write
- [cuda-guide] *CUDA C++ Programming Guide*, NVIDIA - no arXiv ID (vendor manual)
- [pmpp] Kirk and Hwu, *Programming Massively Parallel Processors* (4th edition, 2022) - no arXiv ID (book)

## Backfill

- [x] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - the launch sheet's last column: five kernels at the streaming rate, one at two thirds of it, and the difference is the access pattern.
- [x] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - a decode step reads every weight once, so the bytes are the time; Chapter 24 turns that sentence into kernels.
- [x] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - Lesson 1's two address spaces, with `cudaMalloc` and `cudaMemcpy` as the entire interface between them.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - the prediction written before the run, and the `CHECK` macro that makes a failed launch name its call.
