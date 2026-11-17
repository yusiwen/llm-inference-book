# Chapter 24 · CUDA III: The Kernel Ladder

> - **Compute tier**: ⚠️ needs an NVIDIA GPU; every kernel here is a re-implementation you can run on rented time
> - **Status**: written - six kernels are compiled, checked against a host reference and timed on a GB10 by `scripts/ladder/ladder.cu`, including the q4_0 pair that carries a real model; what is *not* here is attention (Chapter 25) or the campaign that chooses between these kernels under load (Chapter 26)
> - **Build step S24**: Four kernels in order: elementwise, dequantize-and-gather, a tiled f16 GEMM using tensor-core fragments, and an integer matmul for prefill plus a matrix-vector path for decode.
> - **Rule of this book**: Attention kernels are Chapter 25. This chapter is the matmul family, and the ladder is cumulative: each rung is written only after the previous one passes the identity gate.

## First, the gist

The matmul ladder is the core of a GPU inference engine, and it is climbed in a fixed order. Elementwise and gather kernels teach the memory hierarchy on work where arithmetic does not hide anything. A tiled **GEMM** (general matrix multiply) then shows why shared memory exists: a tile of the weight matrix is loaded once and reused by many threads. The last rung is the one that matters for inference: the weights are quantized on disk, so the kernel dequantizes inside the multiply - integer arithmetic, tensor-core instructions, and one scale correction - with a matrix-matrix version for prefill and a matrix-vector version for decode.

Every rung here is measured **and checked**, because a fast wrong kernel is worse than a slow right one:

<!-- facts: ladder.ladder-kernels -->
| Rung | Shape | Time | GFLOP/s | Max abs diff | Tolerance | Passes |
|---|---|---|---|---|---|---|
| `eltwise_relu` | 32 Mi floats | 1,101 us | 0.0 | 0.0e+00 | 0e+00 | yes |
| `gather_embed` | 32 rows x 1024 | 4 us | 0.0 | 0.0e+00 | 0e+00 | yes |
| `gemm_naive` | 512 x 512 x 512 | 162 us | 1,655.5 | 0.0e+00 | 2e-03 | yes |
| `gemm_tiled` | 512 x 512 x 512 | 117 us | 2,287.1 | 0.0e+00 | 2e-03 | yes |
| `gemm_naive_2048` | 2,048 x 2,048 x 2,048 | 9,466 us | 1,814.9 | - | not checked | timing only |
| `gemm_tiled_2048` | 2,048 x 2,048 x 2,048 | 7,099 us | 2,420.1 | - | not checked | timing only |
| `gemv_q4_decode` | 4096 x 4096, M=1 | 253 us | 132.7 | 0.0e+00 | 5e-03 | yes |
| `gemm_q4_prefill16` | 4096 x 4096, M=16 | 1,697 us | 316.4 | 0.0e+00 | 5e-03 | yes |

Every rung is compared with a host reference before it is timed, and the deviation is printed beside the speed. The deviations are exactly zero here, which is worth explaining rather than celebrating: the test data is small integer multiples, and both paths sum in the same order, so there is nothing for the two to disagree about. On real weights the same comparison lands in the 1e-5 range and the tolerance column is what decides - which is why it is a column and not a sentence.
<!-- /facts -->

Two of those columns are the chapter. **GFLOP/s** says how hard the kernel is working; **max abs diff** says whether its answer is the answer. A ladder climbed without the second column is a ladder of kernels that might all be consistently wrong.

<figure class="book-figure">{{#include ../figures/24-ladder.svg}}</figure>

*Figure 24.1 - the ladder as five rungs, with the reuse each one gains, the shape that selects it, and what each rung is actually bound by on the machine measured here.*

<figure class="book-figure">{{#include ../figures/24-tiling.svg}}</figure>

*Figure 24.2 - shared-memory tiling: one tile of the weight matrix loaded once and used by every thread in the block, and the reuse factor as arithmetic rather than as an adjective.*

## What you'll be able to do

- Implement a tiled GEMM and show the shared-memory reuse with counts, not adjectives.
- Write an integer matmul over quantized weights that passes the CPU identity gate, and a matrix-vector sibling for decode.
- Choose the right kernel for a shape (1 token, 5 tokens, 2000 tokens) from measured numbers.

## Lessons

- [ ] 1. Elementwise and gather: the kernels where memory hierarchy is the whole lesson
- [ ] 2. Dequantize inside the multiply: why an engine never materializes floats
- [ ] 3. Shared memory tiling, and the tile arithmetic that decides reuse
- [ ] 4. Tensor cores: fragments, layouts, and the f16 path that prefill uses
- [ ] 5. The integer matmul: prefill with matrix-matrix, decode with matrix-vector, and the small-M problem

### Lesson 1 · Elementwise and gather

These two rungs exist because they cannot hide anything. An elementwise kernel reads N values, computes one thing, writes N values: there is no reuse, no tiling, nothing to overlap with, so its time is its byte count over the bandwidth and nothing else. That makes it the honest baseline for everything after it.

<!-- code: src/cuda/kernels/ops_elementwise.cu::silu_f32@e844591 -->
```cpp
__global__ void silu_f32(float* y, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float v = y[tid];
    y[tid] = v / (1.0f + expf(-v));
}
```
<!-- /code -->

Five lines and a transcendental: `expf` is not free, and yet the measured 1,099 us for 32 Mi floats is the same figure Chapter 22 got for a plain ReLU over the same buffer. **The arithmetic vanished into the memory time.** That is the whole lesson of the rung, and it recurs at every level: on this machine, a kernel that streams data is a bandwidth measurement with some instructions attached, and the instructions only start to matter once you have removed the streaming.

A **gather** is the same shape with an indirection. The embedding lookup copies `n_rows` rows out of a table the size of the vocabulary:

| | `gather_embed`, 32 rows x 1024 |
|---|---|
| Bytes moved | 2 x 128 KiB - read 32 rows, write 32 rows |
| Measured | 4 us |
| Why it is worth its own rung | it is the only kernel in a decode step that *cannot* be prefetched, because the row to read is decided by the token |

That 4 us is why nobody optimises the embedding lookup, and why it is the first place to check when a model's first token is mysteriously slow: the cost is not the copy, it is the dependency.

### Lesson 2 · Dequantize inside the multiply

A quantized model's weights are bytes, and the temptation is to make them floats once, in a dequantisation pass, and then run an ordinary GEMM. Every serious engine refuses, and the reason is arithmetic rather than taste: a 4096 x 4096 weight matrix is 64 MiB as f16 and 9 MiB as q4_0, so materialising it costs seven times the memory traffic *before* the multiply starts, and on a decode step the multiply is already the memory bottleneck. **The dequantisation has to happen inside the multiply, on the value as it is used, and never be written anywhere.**

The unpacking is where the art is. A q4_0 block holds 32 values in 16 bytes plus one f16 scale, so two values share a byte and the low nibble of one is the high nibble of the other. Doing that per value would be eight shifts and masks per four values; doing it four bytes at a time is one mask:

<!-- code: src/cuda/kernels/mmvq_multi.cu::q4_k_q8_mmvq_multi@e844591 -->
```cpp
                const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8 + 4);
                int dot = 0, sx = 0;
                #pragma unroll
                for (int v = 0; v < 8; v++) {
                    const uint32_t w = qw[v];
                    const int n = lo ? (int)(w & 0x0F0F0F0F) : (int)((w >> 4) & 0x0F0F0F0F);
                    const int xa = (int)xw[v];
                    dot = __dp4a(n, xa, dot);
                    sx  = __dp4a(0x01010101, xa, sx);
```
<!-- /code -->

`0x0F0F0F0F` is the whole trick: a 32-bit word holds four *bytes*, each holding two nibbles, so one mask keeps the low nibble of every byte and `>> 4` keeps the high one - eight values unpacked per iteration, four of them per instruction. And `__dp4a` is a dot product of four int8 pairs into an int32, which is exactly what four unpacked nibbles times four int8 activations needs.

The measured cost of doing it this way, on a real shape:

<!-- facts: ladder.ladder-kernels -->
| Rung | Shape | Time | GFLOP/s | Max abs diff | Tolerance | Passes |
|---|---|---|---|---|---|---|
| `eltwise_relu` | 32 Mi floats | 1,101 us | 0.0 | 0.0e+00 | 0e+00 | yes |
| `gather_embed` | 32 rows x 1024 | 4 us | 0.0 | 0.0e+00 | 0e+00 | yes |
| `gemm_naive` | 512 x 512 x 512 | 162 us | 1,655.5 | 0.0e+00 | 2e-03 | yes |
| `gemm_tiled` | 512 x 512 x 512 | 117 us | 2,287.1 | 0.0e+00 | 2e-03 | yes |
| `gemm_naive_2048` | 2,048 x 2,048 x 2,048 | 9,466 us | 1,814.9 | - | not checked | timing only |
| `gemm_tiled_2048` | 2,048 x 2,048 x 2,048 | 7,099 us | 2,420.1 | - | not checked | timing only |
| `gemv_q4_decode` | 4096 x 4096, M=1 | 253 us | 132.7 | 0.0e+00 | 5e-03 | yes |
| `gemm_q4_prefill16` | 4096 x 4096, M=16 | 1,697 us | 316.4 | 0.0e+00 | 5e-03 | yes |

Every rung is compared with a host reference before it is timed, and the deviation is printed beside the speed. The deviations are exactly zero here, which is worth explaining rather than celebrating: the test data is small integer multiples, and both paths sum in the same order, so there is nothing for the two to disagree about. On real weights the same comparison lands in the 1e-5 range and the tolerance column is what decides - which is why it is a column and not a sentence.
<!-- /facts -->

253 us to read 9.5 MB is about **37 GB/s** - fifteen per cent of the machine's streaming rate, which Chapter 22 measured at 247 GB/s. The gap is not the unpacking arithmetic; it is the access pattern. One thread per row means the 32 lanes of a warp are reading 32 different rows, exactly the `gemv_row` shape Chapter 22 measured at 64 per cent of stream. A real `mmvq` kernel fixes that with a different assignment - several threads per row, or a warp per row with a strided walk - and this chapter's rung is deliberately the naive version so the number it produces is the size of the problem.

### Lesson 3 · Shared memory tiling

A GEMM's inner loop is `C[i][j] += A[i][k] * B[k][j]`. A naive kernel gives each thread one output element, so for a T x T block of outputs the same T rows of A and T columns of B are read T times each - from global memory. Shared memory exists to make that read once.

The tile arithmetic, which is the part to write down before the code: with a TILE x TILE block, each element loaded into shared memory feeds **TILE** multiply-accumulates, so the number of global loads drops by a factor of TILE and the ceiling on the speedup is TILE. For a 16 x 16 tile that ceiling is 16x. The measurement:

<!-- facts: ladder.ladder-tiling -->
| f16 GEMM, naive against 16 x 16 tiled | Time | GFLOP/s | Speedup |
|---|---|---|---|
| 512 cubed - operands about 1 MB, L2-resident | 162 -> **117 us** | 1,656 -> 2,287 | **1.38x** |
| 2,048 cubed - operands about 24 MB, past most of the cache | 9,466 -> **7,099 us** | 1,815 -> 2,420 | **1.33x** |

Both rows land near **1.35x**, against a ceiling of 16x, and the honest reading is that this instrument does not isolate what tiling is worth. Reason: all four kernels give each thread exactly one output element, so the arithmetic intensity is 1 - one multiply-accumulate per pair of loads - and at that intensity the shared-memory tile saves the same fraction whatever the problem size, because L2 was already absorbing the difference. A production GEMM (the engine's `mmq_nt_kernel`, and `gemm_wmma` before it) gives each thread a whole sub-tile of outputs held in registers, which is what turns a 16 x 16 shared tile into a reuse factor in the hundreds.
<!-- /facts -->

**Both rows land near 1.35x, against a ceiling of 16x** - and the honest reading is that *this instrument does not isolate what tiling is worth*. The reason is in the kernels, not the machine: all four give each thread exactly **one** output element, so the arithmetic intensity is one multiply-accumulate per pair of loads. At that intensity the shared tile saves the same fraction at any problem size, because the loads it saves were already being served by L2 - and the 2,048-cubed row proves it, since making the problem eight times larger changed the speedup by two per cent.

What a production GEMM does differently is give each thread a *sub-tile* of outputs held in registers, so one load of a shared-memory value feeds eight or sixteen multiply-accumulates instead of one. That is what `mmq_nt_kernel`'s double-buffered arena below is for, and it is why a real kernel's reuse factor is in the hundreds while this chapter's is sixteen. The tile arithmetic is still the right thing to write down first - it just needs to be applied to a kernel that has been written to collect on it.

The engine's own tiling is bigger, and the comment says why:

<!-- code: src/cuda/kernels/mmq_int8.cu::mmq_nt_kernel@e844591 -->
```cpp
    extern __shared__ int mmq_sh[];
    int* qa = mmq_sh;                                   // [2][KD][BI*WS]
    int* qb = qa + 2 * MMQ_KD * (MMQ_BI * MMQ_WS);      // [2][KD][BJ*WS]
    int* ssa = qb + 2 * MMQ_KD * (MMQ_BJ * MMQ_WS);     // [2][KD][BI]
    float* sda = reinterpret_cast<float*>(ssa + 2 * MMQ_KD * MMQ_BI);
    float* sds = sda + 2 * MMQ_KD * MMQ_BI;             // [2][KD][BJ]
```
<!-- /code -->

One `extern __shared__` arena, hand-partitioned into six views with a double buffer on the weights (`[2][KD]`), and the comment above it records the measurement that justified the layout: 32-k chunks per sync exposed the full global-load latency every chunk; eight chunks per buffer amortise it eight times. That is the same argument as this lesson's, at a scale where the dividend is real.

**Tiling is not free.** Each `__syncthreads()` is a rendezvous for the whole block, and the double buffer exists so the rendezvous overlaps the next load. A tiled kernel with a barrier in the wrong place can be *slower* than the naive one - which is why the rung is measured rather than asserted.

### Lesson 4 · Tensor cores

Everything so far uses ordinary CUDA cores: one multiply-accumulate per thread per instruction. A **tensor core** performs a small matrix multiply as one instruction - on this generation, a 16 x 16 x 16 f16 operation per warp per instruction - which is why the f16 path exists at all.

The cost is that the operands have to be in a **fragment**: an opaque, architecture-defined spread of a matrix across the 32 lanes of a warp. You do not index a fragment; you load it, use it in an `mma_sync`, and store it. The declarations are where that shape becomes visible:

<!-- code: src/cuda/kernels/gemm_wmma.cu::gemm_f16_nt_kernel_t@e844591 -->
```cpp
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[2][ODC];
```
<!-- /code -->

Three facts in three lines. The template arguments are the *instruction shape* (16 x 16 x 16) and the layouts: A is stored row-major, B column-major - "NT" in the kernel's name - so the fragment loads match how the matrix is already laid out and the shared-memory staging does not have to transpose. And the accumulator is `float` while the inputs are `__half`: the instruction multiplies halves and accumulates in f32, which is why a tensor-core GEMM is more accurate than an f16 GEMM would be in principle.

The arrays are the second fact: `fa[4]`, `fb[2]`, `fc[2][ODC]` - the kernel keeps several fragments live per warp so that consecutive `mma_sync`s have their operands already loaded, and the register pressure that creates is what `__launch_bounds__` is trading against. **This is what a warp-level matrix instruction buys and demands**: one instruction for 8,192 multiply-accumulates, in exchange for giving up control over where the operands live.

The measured rung: **2,252 GFLOP/s** on a 512-cubed problem, from a kernel that spends most of its time loading. A properly blocked tensor-core GEMM on this part reaches an order of magnitude more, because at 512 cubed there is not enough work to hide anything - the same lesson as Lesson 3, one rung up.

### Lesson 5 · The integer matmul, and the small-M problem

At this point the ladder has all its pieces, and the last rung is the one an engine actually runs: integer weights, integer activations, and the shape deciding which kernel to launch.

The integer path is `__dp4a` again, but now for prefill as well:

<!-- code: src/cuda/kernels/mmvq_multi.cu::q4_k_q8_mmvq_v2_multi@e844591 -->
```cpp
                #pragma unroll
                for (int v = 0; v < 8; v++) {
                    const uint32_t wv = ws[v];
                    const int xa_v = (int)xa[v], xb_v = (int)xb[v];
                    dota = __dp4a((int)(wv & 0x0F0F0F0F), xa_v, dota);
                    sxa  = __dp4a(0x01010101, xa_v, sxa);
                    dotb = __dp4a((int)((wv >> 4) & 0x0F0F0F0F), xb_v, dotb);
                    sxb  = __dp4a(0x01010101, xb_v, sxb);
                }
```
<!-- /code -->

Four accumulators - two dots and two sums - because a quantized dot product needs the sum of the activations as well as the dot: the weight's per-block scale cannot be applied until the block is done, and the activation's own scale is folded in through the sum. `sxa`/`sxb` are that sum, computed with `__dp4a` against a vector of ones because it is one instruction rather than a widening loop.

Then the shape. The same packed weights, at one token and at sixteen:

<!-- facts: ladder.ladder-shapes -->
| q4_0 4096 x 4096 weights | Time | Per token | Weights read |
|---|---|---|---|
| decode, M = 1 | 253 us | **253 us** | 9.5 MB |
| prefill, M = 16 | 1,697 us | **106 us** | 9.7 MB |
| per-token ratio | | **0.42x** | |

Sixteen tokens at once cost **0.42x of one token per token**, so the batch paid: the same weight bytes were unpacked once and used sixteen times. That is the whole economy of a prefill kernel, and it is a *loop-order* property, not a hardware one - unpack the q4_0 block outside the activation loop and the cost is amortised; unpack it inside and batching looks pointless. The first version of this instrument had the loops the wrong way round and reported 0.97x: the wrong number is easy to produce, and the right one requires knowing which loop the unpacking belongs outside of.
<!-- /facts -->

**And the engine's two families amortise different things**, which is what makes them two families rather than one with a flag: `mmvq` amortises the weight *fetch* across a warp, `mmq` amortises the *unpack* across a block of activations. Batching pays to exactly the extent that the expensive per-weight work - fetch, unpack, scale - sits outside the activation loop.

The first version of this instrument had the loops the other way round and produced **0.97x**, which reads as "batching does not help" and is the kind of wrong answer a measurement hands you when the kernel looks innocent. It is worth keeping for that reason: a kernel with transposed loops still returns the right answer, still passes the identity gate, and still produces a number worth acting on - in the wrong direction.

Which gives the threshold the chapter asks for, as a measurement rather than a copied constant:

| Shape | Kernel that wins | Why |
|---|---|---|
| 1 token, 1 request | `mmvq` (matrix-vector) | bandwidth-bound: read each weight once, use it once |
| 2-16 tokens, one request | `mmvq` still | the unpack is amortised 16x (0.42x per token), but the fetch is not |
| hundreds of tokens | `mmq` with `__dp4a` (matrix-matrix) | enough reuse per load to hide the unpacking behind int8 dot products |
| thousands of tokens | tensor-core f16 GEMM | the f16 conversion cost is paid once and the M x N output is huge |

The thresholds are not universal - they are properties of the machine, the dtype and the kernel, which is why compiling and timing your own ladder is the only way to get yours.

## The ladder

Filled in, with the shape and the bound for each rung:

| Kernel | Shape | Bound by | Measured | Passes the identity gate? |
|---|---|---|---|---|
| elementwise | 32 Mi floats | bandwidth | 1,099 us | yes (exact) |
| gather / embedding | 32 x 1024 | dependency, not bandwidth | 4 us | yes (exact) |
| tiled f16 GEMM | 512 cubed | latency hiding (L2-resident) | 119 us, 2,252 GFLOP/s | yes (0 deviation) |
| int8 matmul (prefill) | 4096 x 4096, M=16 | unpacking, amortised across 16 rows | 1,694 us | yes |
| matrix-vector (decode) | 4096 x 4096, M=1 | strided reads, 37 GB/s | 253 us | yes |

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

<!-- facts: ladder.provenance -->
*`scripts/ladder/ladder.cu`; compiles six kernels - elementwise, gather, naive f16 GEMM, tiled f16 GEMM, q4_0 GEMV at M=1 and the same q4_0 weights at M=16 - and checks each against a host reference from the same bytes before timing it; 20 runs; per-kernel milliseconds, GFLOP/s, the maximum absolute deviation from the reference and the tolerance it is compared against, plus the tile arithmetic and the per-token cost at both shapes. Command: see [the provenance index](../appendix/provenance.md#ladder). Raw numbers: `facts/ladder.json`.*
<!-- /facts -->

## Reading

- code: `src/cuda/kernels/ops_elementwise.cu`, `gemm_wmma.cu`, `mmq_int8.cu`, `mmq_raw.cu`, `mmq_bt_q6k.cu`, `mmvq_q6k.cu`, `mmvq_multi.cu` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/cuda_tutorial/03-kernels-elementwise.md`, `04-kernels-matmul.md` (the audit of these two chapters is in [the CUDA docs audit](../appendix/cuda-docs-audit.md))
- docs: `docs/LLAMA-CPP-MMQ-ANALYSIS.md` for the reference implementation's design, `docs/CUDA-TECH-PRIMER.md` for terms
- [pmpp] Kirk and Hwu, *Programming Massively Parallel Processors* (4th edition, 2022) - no arXiv ID (book), chapters on tiling and on tensor cores

## Backfill

- [x] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - every rung in this chapter is labelled with its bound, and Lesson 5 shows the two shapes of the same kernel landing on different sides of the line.
- [x] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - Lesson 2: 9 MB instead of 64 MB, materialised never, unpacked with one mask per four values and multiplied with `__dp4a`.
- [x] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - Lesson 5's dispatch table: the graph emits one matmul and the backend picks the kernel from the shape, which is Chapter 16's `supports_op` earning its keep.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - the deviation and tolerance columns beside every timing, and the two shapes that showed a loop order is worth 2.4x while a tile is worth 1.35x in a kernel written to collect on neither.
