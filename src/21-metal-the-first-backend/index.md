# Chapter 21 · Metal: The First Backend

> **Compute tier**: ✅ on any Apple-silicon Mac; ⚠️ elsewhere this is reading only (the seam is the lesson, the shaders are not) | **Status**: stub
> **Build step S21**: A first non-CPU backend behind the registry from Chapter 16: device buffers, a command buffer per split, and three shaders that pass the identity gate.
> **Rule of this book**: CUDA is Chapters 22-26 and is not a translation of this chapter; the shared lesson (device layer, launch layer, kernels, executor) is stated once here and assumed afterwards.

## First, the gist

The first GPU backend is mostly a lesson about seams. Apple's **Metal** gives a process a device, a queue, buffers that may be shared with the CPU's address space, and command buffers that batch work. Because Apple silicon shares memory between CPU and GPU, this is the friendliest possible device to learn on: a weight buffer can be created over the mapped model file and no copy happens at all. What the chapter really teaches is the four-layer split - runtime, launch, kernels, executor - that every backend in the book follows, and the rule that the backend never sees the graph's own bookkeeping.

## What you'll be able to do

- Stand up a device backend behind the registry: buffers, queues, one command buffer per split, errors surfaced.
- Write three shaders (a copy, a norm, a quantized matmul) and pass the identity gate against the CPU path.
- Explain zero-copy buffers and what changes when a device does not share memory with the host.

## Lessons

- [ ] 1. Four layers, one seam: runtime, launch, kernels, executor
- [ ] 2. A device queue and a command buffer: batching work instead of submitting per op
- [ ] 3. Zero-copy weights: what Apple silicon allows and why a discrete GPU cannot
- [ ] 4. Three shaders, then the gate: norm, copy, and one quantized matmul (payoff: the identity gate is the entry ticket, not the last step)
- [ ] 5. What the backend must never do: allocate behind the allocator's back, or fall back silently

## Backend layer checklist

| Layer | Metal artifact | What it owns | What it must not know | Your status |
|---|---|---|---|---|
| runtime |  |  |  |  |
| launch |  |  |  |  |
| kernels |  |  |  |  |
| executor |  |  |  |  |

## Project

**Build step S21 · the friendliest device first.** Do this backend before buying GPU time. Every structural mistake in a backend is cheaper to find here, where the memory is shared and the profiler is local.

If you have no Mac, read this chapter for the four-layer split and start at Chapter 22; the seam is the transferable part.

## Deliverable

- A Metal backend registered behind the registry, executing a real graph
- Three shaders with identity tests against the CPU path
- A measured tok/s for one model on this backend, quoted with conditions

## How you know you passed

- The full graph runs on the device and produces tokens at the identity gate's tolerance.
- Disabling the backend via the registry flag returns the CPU result unchanged.
- You can name the one thing that would have to change for a discrete GPU, and it is the copy.

## Reading

- code: `src/metal.rs`, `src/metal/{runtime,encode,ops,policy}.rs`, `src/metal/kernels/*.metal`, `src/graph/metal_backend.rs` at commit `88ffa5d`
- decisions: ADR-0005 (Metal becomes a first-class backend), ADR-0037 (a weight dtype a kernel cannot consume is refused), ADR-0080 (the backend is a Cargo feature and a build without it refuses the name)
- docs: `docs/METAL-BACKEND-DESIGN.md`, `docs/METAL_OPTIMIZATIONS.md`, `docs/inference_e2e_walkthrough/14-metal-backend.md`

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
