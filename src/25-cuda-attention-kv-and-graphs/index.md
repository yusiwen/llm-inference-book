# Chapter 25 · CUDA IV: Attention, KV, and CUDA Graphs on Device

> - **Compute tier**: ⚠️ needs an NVIDIA GPU
> - **Status**: written - the score matrix is materialised and then avoided on a GB10 by `scripts/attn/attn_dev.cu`, decode attention is timed at two context lengths, and 100 launches are compared with one graph replay; what is *not* here is the campaign that chose between these paths under load (Chapter 26)
> - **Build step S25**: Device-side attention for prefill and a fused decode tail over a device-resident KV cache, plus graph capture and replay for the decode step.
> - **Rule of this book**: The cache's storage design is Chapter 18 and its semantics Chapter 10. This chapter is the device implementation, and the campaign that optimized it is Chapter 26.

## First, the gist

Attention on a GPU looks different from attention on a CPU for one reason: **the score matrix does not fit in registers or in shared memory**. For a 1,024-token prompt it is 1,024 x 1,024 floats - 4 MB - and for 8,192 tokens it is 256 MB. Materialising it means writing that to device memory and reading it back three times (max, sum, weighted values), which is pure traffic for a result that is 64 numbers per row.

So the kernel computes the row in pieces and normalises on the fly, keeping a **running max** and a **running sum** in registers. That is the flash attention family, and the measurement of what it saves is this chapter's first table:

<!-- facts: attn-dev.attn-prefill -->
| Prefill attention, 1,024 keys, head dim 64 | Time | Score buffer | Agrees with the other path |
|---|---|---|---|
| materialised - the row written to device memory and read back | 3,580 us | **4.2 MB** | - |
| streamed - running max and running sum, the row never leaves registers | **390 us** | 0 | max abs diff 0.0e+00 |
| speedup | **9.18x** | | |

The buffer is the whole story: 4.2 MB of scores for a 1,024-token prompt with one head, written once and read back three times, to compute an output of 64 floats per row. The streamed version computes the same numbers - the last column says the two agree exactly - and never allocates the matrix at all, which is what makes **9.18x** a memory-traffic result rather than an arithmetic one. At 8,192 tokens the buffer would be 256 MB and the ratio would be worse; that growth is why flash attention is not an optimisation but the only way prefill works at long context.
<!-- /facts -->

**9.2x, and the two paths agree exactly.** The buffer is not a detail of the implementation; it is the implementation, and at 8,192 tokens it would be 256 MB - sixty times larger than the thing it computes.

Decode has the opposite shape: one query against a long cache, so there is no score matrix at all, just a reduction. And the third piece is not a kernel: a decode step launches a hundred small kernels, and a **CUDA graph** records that sequence once and replays it as one submission.

<figure class="book-figure">{{#include ../figures/25-flash-tiling.svg}}</figure>

*Figure 25.1 - attention prefill tiled so the score matrix never materializes: the causal triangle, the tiles that cover it, and the running max and sum that replace a full row.*

<figure class="book-figure">{{#include ../figures/25-cuda-graph.svg}}</figure>

*Figure 25.2 - a hundred small launches per decode step against one captured graph replayed, on the same time axis.*

## What you'll be able to do

- Implement attention prefill with on-the-fly normalization, and a decode path that reduces over a device cache.
- Keep KV in device memory with a defined format, and show the bytes it moves per step.
- Capture a decode step as a CUDA graph and replay it, with a test that replay equals a fresh run.

## Lessons

- [ ] 1. Why the score matrix does not fit: tiling attention and normalizing on the fly
- [ ] 2. The prefill attention kernel, read line by line against its CPU counterpart
- [ ] 3. The decode tail: fusing bias, RoPE, cache store and attention into one launch
- [ ] 4. KV on the device: layout, packed formats, and the split-KV reduction
- [ ] 5. CUDA Graphs: capture once, replay, and the conditions under which capture is unsafe

### Lesson 1 · Why the score matrix does not fit

The arithmetic of attention is three matmuls and a softmax: `S = Q K^T`, `S = softmax(S)`, `O = S V`. The softmax is a *row* operation - every element of a row needs the row's max and sum - which is why the naive implementation has to keep the whole row, and why keeping a whole row is what breaks at long context.

The fix is to notice that the softmax's two statistics can be maintained incrementally. If you have processed some prefix of the row and hold its max `m` and sum `l`, and a new block arrives with max `m'`, then the new max is `max(m, m')`, and the old sum has to be rescaled by `exp(m - m_new)` because every score already exponentiated was exponentiated relative to the old max. That single identity turns a row-wide barrier into a running update, and it is the whole of flash attention's algorithm.

Here it is in the engine, operating on tensor-core fragments rather than on a shared array:

<!-- code: src/cuda/kernels/attention_prefill.cu::fa_prefill_kv@e844591 -->
```cpp
        // Fragment-resident online softmax. Each lane holds fragment rows r0/r1
        // at cols c0, c0+1, c0+8, c0+9 in every fc[cc]; the 4 lanes (l=0..3)
        // sharing a row hold all FA_TKV columns. Within-lane reduce over the
        // fragments, then __shfl_xor over offsets 1,2 (the 4-lane group).
        float sm[FA_TKV / 16 * 4], sm1_[FA_TKV / 16 * 4]; // per (cc, quad)
```
<!-- /code -->

Two things to read out of that comment. **Where the row lives**: not in shared memory, not in a `[seq]` array, but scattered across the registers of four lanes - `wmma` fragments have an architecture-defined layout, so a "row" is a specific set of fragment elements per lane, and the comment says exactly which (`c0, c0+1, c0+8, c0+9`). **How the row is reduced**: within a lane over its own fragment elements, then `__shfl_xor` across the four lanes that share the row - a shuffle, not a `__syncthreads()`, because a row's four lanes are all in the same warp.

That is the difference between the version this chapter measures (a shared-memory block, one block per query) and the engine's (a tensor-core tile with a fragment-resident softmax). Both avoid the score matrix; the engine's also avoids ever writing the row to shared memory.

### Lesson 2 · The prefill kernel against its CPU counterpart

Chapter 10 built prefill attention in Rust, and the comparison is worth making explicitly because the CPU version is the oracle: same three matmuls, same causal mask, same softmax - and the two must agree, since the whole book is built on "the device is a faster way to compute the same function".

The device version's inner step is one tensor-core instruction per tile:

<!-- code: src/cuda/kernels/attention_prefill.cu::fa_prefill_kv@e844591 -->
```cpp
                wmma::mma_sync(fc[cc], fa, fb[cc], fc[cc]);
```
<!-- /code -->

`fc[cc]` is the accumulator (a 16 x 16 tile of scores, kept in fragments), `fa` is a Q tile, `fb[cc]` is a K tile - transposed by the layout, which is what makes the multiply `Q K^T` without an explicit transpose. One line, and the tile is scored.

The differences from the CPU version are all consequences of that one line:

| CPU (Chapter 10) | Device (this chapter) | Why |
|---|---|---|
| scores in a `Vec<f32>` of length `seq` | fragments across 4 lanes | a fragment is not indexable |
| `max` and `sum` over the whole row | running max and sum per tile | the row is never whole |
| causal mask applied by skipping | causal decided by which tiles exist | geometry replaces comparison |
| softmax in `f64` accumulation | `float` accumulate, f32 online correction | the tensor core accumulates in f32 |

The last row is a precision decision worth naming: the online softmax multiplies the accumulated values by `exp(m_old - m_new)` every time the max moves, and each of those multiplications is a rounding. Chapter 9's ledger is what says the total is acceptable - and the measured `max abs diff` of the two paths in this chapter's first table is how you check that the ledger is right rather than hoping.

### Lesson 3 · The decode tail

Decode is one query against a cache of `n_ctx` keys, so there is no score matrix, no tiling of the query dimension, and no reuse of anything: every key is read once, scored once, and used once. The work is a reduction, and its cost is the cache's bytes.

<!-- facts: attn-dev.attn-decode -->
| Decode attention, one query | Time | K read | Effective bandwidth |
|---|---|---|---|
| cache of 128 keys | 12 us | 0.03 MB | 2.7 GB/s |
| cache of 8,192 keys | 220 us | 2.10 MB | 9.5 GB/s |
| 8,192 keys against 128 | | 64x the bytes | 18x the time |

Sixty-four times the cache costs sixty-four times the time: decode attention is linear in context, with no economy of scale, because every key must be read and scored once. The bandwidth column is the interesting one - **9.5 GB/s** against the machine's 247 - and it is not a bug in the kernel: one block on one multiprocessor cannot saturate a device's memory system. That is the case for split-KV (Lesson 4): cut the cache into pieces, give each piece a block on its own SM, and reduce the partials afterwards.
<!-- /facts -->

Sixty-four times the cache costs eighteen times the time here, and both ends of that table are dominated by the same thing: **the kernel is too small to fill the machine**. One block on one multiprocessor cannot saturate a device's memory system, so the effective bandwidth is a few per cent of the 247 GB/s Chapter 22 measured.

That is not a defect in the kernel; it is the shape of the problem, and it is why a real engine **fuses** the decode tail. The tokens for one layer's attention are: apply the bias, apply RoPE to the query, write the new key and value into the cache, then attend. Done as four launches, three of them are tiny and the fourth waits on them - and every one of them pays the launch cost from Lesson 5. Done as one kernel, the intermediates never leave registers.

The engine keeps the dtype variants of this path as *structural mirrors*, which is the discipline that makes a fused tail testable at all - the f16 cache version differs from the f32 one only in how a cache row is loaded:

<!-- code: src/cuda/kernels/attention_decode.cu::gqa_attn_f32_f16kv@e844591 -->
```cpp
template <bool CAUSAL, bool MAP>
__global__ void gqa_attn_f32_f16kv(
    const float* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    float* __restrict__ o,
    const int* bound,
    int nh, int nk, int hd,
    float scale, int nt
) {
```
<!-- /code -->

Two template parameters (`CAUSAL`, `MAP`) instead of two kernels, an f16 cache behind a `float` accumulate, and a `bound` pointer that carries the causal extent - so the same body serves a cold prompt and a long-running conversation without a second implementation to keep in step. When the tail is fused (Chapter 26), this is the shape it is fused *from*.

Fusing costs you the ability to test the pieces separately, which is why the chapter's project says to get the unfused path correct and identical *first*: three separate measurements, three separate gates. A fused kernel that disagrees with the CPU path is a mystery; an unfused one that disagrees tells you which of the four steps is wrong.

### Lesson 4 · KV on the device, and the split-KV reduction

The cache's layout is Chapter 18's subject; what changes on a device is that it lives in device memory and its bytes are the step's cost. The number to carry: at `n_ctx` = 8,192 with a head dim of 64 and one head, one decode step reads 2.1 MB of keys per layer - and a 24-layer model reads that twenty-four times.

The lesson's own measurement is the argument for the next trick. 9.5 GB/s of effective bandwidth on a device that can do 247 says the kernel is latency-bound with one block; the fix is to give the work to every SM at once:

<!-- code: src/cuda/kernels/attention_decode.cu::gqa_attn_split_partial@e844591 -->
```cpp
template <int LAYOUT, bool CAUSAL, bool MAP, bool Q8DP4A = false, bool Q8WIDE = false>
__global__ void gqa_attn_split_partial(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ partial,
    const int* bound,
    int nh, int nk, int hd, float scale, int pstr, size_t row_bytes,
    int rpw_gate
) {
```
<!-- /code -->

**Split-KV**: cut the cache into `ATTN_SPLITS` spans, let each block own one span, and write a *partial* answer per split. The partial cannot be just a weighted sum, because softmax is not linear - each split has to record its own max, its own sum, and its own weighted values. That is exactly the online-softmax state from Lesson 1, and the reduction after it is the same correction applied to whole splits:

<!-- code: src/cuda/kernels/attention_decode.cu::gqa_attn_split_combine@e844591 -->
```cpp
    float gmx = -INFINITY;
    for (int sp = 0; sp < ATTN_SPLITS; sp++)
        gmx = fmaxf(gmx, partial[((size_t)sp * nh + h) * pstr]);
    float S = 0.0f, acc = 0.0f;
    for (int sp = 0; sp < ATTN_SPLITS; sp++) {
        const float* p = partial + ((size_t)sp * nh + h) * pstr;
        float w = expf(p[0] - gmx);
        S += p[1] * w;
```
<!-- /code -->

`gmx` is the global max, `w` is the rescale for each split, `p[1] * w` accumulates the split's sum into the global one. Same identity as the tiled prefill, one level up: **the online-softmax correction is scale-free**, so it works on tiles inside a kernel, on splits across kernels, and (Chapter 27) on requests across a server.

Two things split-KV costs. A second launch (the combine), and a temporary buffer of `ATTN_SPLITS x nh x pstr` floats - which is why the split count is a tuned constant, not a maximum. And the split has to respect the causal bound: a split whose span is entirely in the future must be skipped, not computed as zero, or the softmax's max will be taken over garbage.

### Lesson 5 · CUDA Graphs

A decode step of a real model issues a hundred or more kernel launches: one per quantization block, per layer, per elementwise op. Each launch is a host-side round trip to the driver, and the GPU cannot start the next kernel until the host has asked for it. **A CUDA graph records that sequence once and submits it as a single unit**, which removes the round trips without changing a single kernel.

<!-- facts: attn-dev.attn-graph -->
| 100 small kernels | Time for all 100 | Per kernel |
|---|---|---|
| launched one at a time | 244 us | 2.44 us |
| captured once, replayed as one graph | **79 us** | **0.79 us** |
| saved | **67.7%** | |

The per-launch figures are the host-side cost of asking the driver to start a kernel - 2.44 us of it, before the GPU does any work - and a decode step of a real model issues a hundred or more. The graph removes **67.7 per cent** of that, because the sequence is recorded once as a dependency DAG and submitted as a single unit: the driver walks a list instead of a hundred round trips. It is not a kernel optimisation at all, which is why it stacks with every other chapter in this part.
<!-- /facts -->

**2.44 us per launch, before the GPU does any work.** For a hundred-launch step that is a quarter of a millisecond of pure host overhead, and at decode speeds of a few thousand tokens per second it is a substantial fraction of the step. The graph cuts it by two thirds, and it does so for *any* sequence of kernels, which is why it stacks with everything else in this part.

The mechanics: `cudaStreamBeginCapture` on a stream, launch the kernels as usual (they are recorded, not executed), `cudaStreamEndCapture` to get a `cudaGraph_t`, `cudaGraphInstantiate` to compile it into a `cudaGraphExec_t`, then `cudaGraphLaunch` on every step afterwards. The recorded kernels must be launched on the capturing stream, and the memory they use must not be reallocated while the graph is live - which is the constraint that makes capture **unsafe** in three specific situations the chapter asks you to name:

| Situation | Why capture is unsafe | What the engine must do |
|---|---|---|
| the buffer pool reallocates between steps | the graph holds raw pointers from the step it was recorded on | re-capture, or pin the buffers (Chapter 23's ledger) |
| a kernel's arguments depend on the step (a position, a sequence length) | they are baked into the recorded node | write them through a device pointer the graph reads, not as launch arguments |
| a kernel is launched with a host-side synchronization in the middle | the sync would be captured as an error, or silently dropped | keep syncs outside the captured region |

minfer's own code records the failure mode of the graph's *lifetime*, which is the same discipline Chapter 23 was about:

<!-- code: src/cuda.rs::graph_destroy_failure_message@e844591 -->
```rust
pub fn graph_destroy_failure_message(err: i32) -> String {
    format!(
        "CUDA: cudaGraphDestroy(cudaGraph_t from cudaStreamEndCapture) failed: {} ({err}); the \
         graph handle leaks, the instantiated exec stays valid, and the error was named and \
         cleared here so it cannot resurface as a latched API error (issue #147)",
```
<!-- /code -->

Two handles, two destroy functions, and mixing them returns `cudaErrorInvalidValue` and leaks. The message names which function took which handle because that is the mistake the next person will make.

**Test replay against a fresh run.** The chapter's project says it and it is worth repeating here, because it is a trap: a graph that replays stale memory agrees with itself perfectly. The only test that catches it compares the replay's output with the output of a non-captured run at the same position - never with the previous replay.

## Attention on device

Filled in with the measured paths:

| Path | Kernel | Shape | Bytes read per step | Measured |
|---|---|---|---|---|
| prefill attention | tiled, online softmax | 1,024 keys, hd 64, causal | 0 (nothing materialised) | 390 us |
| prefill, naively materialised | score matrix written and read back | same | 4.2 MB | 3,580 us |
| decode attention (short ctx) | one block, reduction over cache | 128 keys | 0.03 MB | 12 us |
| decode attention (8k ctx) | one block, reduction over cache | 8,192 keys | 2.10 MB | 220 us |
| fused decode tail | one kernel: bias, RoPE, store, attend | 8,192 keys | 2.10 MB | not measured here - Chapter 26 |
| 100-kernel step | graph replay | - | - | 79 us vs 244 us launched |

## Project

**Build step S25 · fuse, then replay.** Get the unfused decode path correct and identical first, then fuse, then capture. Three separate measurements, three separate gates; collapsing them turns a bug into a mystery.

Test graph replay by comparing against a non-captured run at the same position, not against the previous replay: a graph that replays stale memory agrees with itself.

## Deliverable

- Prefill and decode attention kernels with identity tests against the CPU path
- A CUDA-graph capture and replay path with an equality test against a fresh run
- A per-step byte count for attention at two context lengths, quoted with the command

## How you know you passed

- Fused and unfused decode produce identical ids, and replay produces identical ids to a fresh run.
- Capture is refused (with an error) in the conditions your design says are unsafe.
- Your decode kernel's effective bandwidth rises when you split the KV, and you can say by how much on your machine.

<!-- facts: attn-dev.provenance -->
*`scripts/attn/attn_dev.cu`; computes one causal prefill row twice - once materialising the score matrix and once with a running max and sum - then runs decode attention against 128 and 8,192 keys, then times 100 small kernels as launches and as one captured CUDA graph replayed; milliseconds for each path, the score buffer's size, the maximum deviation between the two prefill paths, the bytes read by decode at both context lengths, and the per-launch and per-replay microseconds. Command: see [the provenance index](../appendix/provenance.md#attn-dev). Raw numbers: `facts/attn-dev.json`.*
<!-- /facts -->

## Reading

- code: `src/cuda/kernels/attention_prefill.cu`, `attention_decode.cu`, `kv_store.cu`, `src/cuda.rs` (the graph handles) at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/cuda_tutorial/05-kernels-attention-host.md`, `06-optimization-methods.md`; the audit of both is in [the CUDA docs audit](../appendix/cuda-docs-audit.md)
- papers: [online-softmax] the running-max identity derived in Chapter 9 and used here at tile level; [fa2] Dao, *FlashAttention-2* (2023), arXiv:2307.08691 - the tiling and work-partitioning this chapter's kernel follows
- [cuda-guide] *CUDA C++ Programming Guide*, NVIDIA - no arXiv ID (vendor manual), the "CUDA Graphs" chapter

## Backfill

- [x] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - the prefill table's 9.2x is pure traffic, and the decode table's 9.5 GB/s says the kernel is too small to be either: it is latency-bound until you split it.
- [x] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - the KV cache is read once per step and never reused, so its bytes are the step's cost; Chapter 18's packed format is what keeps that read small.
- [x] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - Lesson 3's fused tail is one kernel where the graph would otherwise emit four, and Lesson 5's graph is the boundary moved from per-op to per-step.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - the streamed prefill agrees with the materialised one exactly, and replay is tested against a fresh run rather than against itself.
