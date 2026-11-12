# Chapter 25 · CUDA IV: Attention, KV, and CUDA Graphs on Device

> **Compute tier**: ⚠️ needs an NVIDIA GPU | **Status**: stub
> **Build step S25**: Device-side attention for prefill and a fused decode tail over a device-resident KV cache, plus graph capture and replay for the decode step.
> **Rule of this book**: The cache's storage design is Chapter 18 and its semantics Chapter 10. This chapter is the device implementation, and the campaign that optimized it is Chapter 26.

## First, the gist

Attention on a GPU looks different from attention on a CPU for one reason: the score matrix for a long prompt does not fit in registers or in shared memory, so it is tiled and normalized on the fly. That is the **flash attention** family of kernels, and it is what makes prefill practical. Decode has the opposite shape - one query against a long cache - so it wants a reduction over the cache rather than a matrix, and a good engine fuses the whole tail (bias, RoPE, store, attend) into one launch. The last ingredient is dispatch overhead: a decode step launches a hundred small kernels, and a **CUDA Graph** records them once and replays the whole sequence, which is why an engine measures microseconds of overhead instead of hundreds.

## What you'll be able to do

- Implement attention prefill with on-the-fly normalization, and a decode path that reduces over a device cache.
- Keep KV in device memory with a defined format, and show the bytes it moves per step.
- Capture a decode step as a CUDA graph and replay it, with a test that replay equals a fresh run.

## Lessons

- [ ] 1. Why the score matrix does not fit: tiling attention and normalizing on the fly
- [ ] 2. The prefill attention kernel, read line by line against its CPU counterpart
- [ ] 3. The decode tail: fusing bias, RoPE, cache store and attention into one launch
- [ ] 4. KV on the device: layout, packed formats, and the split-KV reduction
- [ ] 5. CUDA Graphs: capture once, replay, and the conditions under which capture is unsafe (payoff: the per-step overhead a graph removes)

## Attention on device

| Path | Kernel | Shape | Bytes read per step | Measured (yours) |
|---|---|---|---|---|
| prefill attention |  |  |  |  |
| decode attention (short ctx) |  |  |  |  |
| decode attention (8k ctx) |  |  |  |  |
| fused decode tail |  |  |  |  |

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
- The measured per-step overhead drop is quoted with its measurement window.

## Reading

- code: `src/cuda/kernels/attention_prefill.cu`, `attention_decode.cu`, `kv_store.cu`, `src/cuda/methods/capture.rs`, `src/graph/cuda_backend.rs` at commit `88ffa5d`
- docs: `docs/cuda_tutorial/05-kernels-attention-host.md`, `docs/inference_e2e_walkthrough/15-cuda-backend.md`
- decisions: ADR-0031 (capture runs in thread-local mode), ADR-0032 (the packed-KV staging window is f32), ADR-0034 (the smem opt-in is an eager pre-warm)

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
