# Chapter 3 · Tensors, Strides, and Bytes

> **Compute tier**: ✅ any laptop | **Status**: stub
> **Build step S3**: A tensor type that owns a byte buffer plus shape, strides and dtype, and can produce a zero-copy view of any 2-D sub-block.
> **Rule of this book**: This chapter builds the container for numbers inside the process. Which numbers the bytes encode is Chapter 4; how to multiply by them is Chapter 6.

## First, the gist

Every tensor in an engine is four things: a byte buffer, a shape, a stride per dimension, and a type that says how to read the bytes. Keeping the type separate from the buffer is what lets one code path move weights, on the CPU or on a device, without knowing whether a number is f32, f16 or a 4-bit block. The other half of the chapter is views: a slice of a tensor that shares memory with its parent, which is how an engine reads one row of an embedding table or one attention head without copying.

## What you'll be able to do

- Implement shape/stride arithmetic for row-major data, including the padding GGML inserts for alignment.
- Produce a view of a sub-tensor and prove it does not copy (pointer equality, not intuition).
- Explain the bytes-in, bytes-out rule: why weight tensors stay untyped byte slices until a kernel is chosen.

## Lessons

- [ ] 1. A tensor is a buffer, a shape, strides and a dtype - and nothing else
- [ ] 2. Row-major strides, and the alignment GGML needs between rows
- [ ] 3. Views, aliases and the lifetime rules that keep them sound in Rust
- [ ] 4. Bytes in, bytes out: why the engine keeps weights opaque until dispatch (payoff: one loader, every quantization)
- [ ] 5. Four-dimensional shapes are a story about batches and heads, not a data layout

## Layout decisions

| Decision | What it buys | What it costs | Where the book takes it up again |
|---|---|---|---|
| row-major, no packing |  |  |  |
| alignment between rows |  |  |  |
| untyped weight bytes |  |  |  |
| views instead of copies |  |  |  |

## Project

**Build step S3 · the tensor crate.** Write the type and its arithmetic first, with unit tests that compare against hand-computed offsets. A tensor library you cannot test on paper is a tensor library you cannot debug in Chapter 6.

Resist adding a `dtype` enum with per-dtype accessors everywhere. One opaque buffer plus a type tag is the shape that survives to Chapter 24.

## Deliverable

- A `Tensor` type with shape, strides, dtype and an owned or borrowed buffer
- A view constructor with a test proving the parent's memory is shared
- A written note stating your alignment rule and where it is enforced

## How you know you passed

- Offsets for three hand-drawn shapes match the test expectations, including a non-contiguous view.
- You can point at the line that would go wrong if weights were typed, and say what breaks.
- No kernel code was needed to make a view: if it copies, it is not a view.

## Reading

- code: `src/tensor.rs`, `src/block.rs` (`repr(C)` block structs) at commit `88ffa5d`
- docs: `docs/inference_e2e_walkthrough/03-model-dispatch-weights.md`
- docs: `docs/PARAMETER_AUDIT.md` for the real shapes of a Qwen3 model

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
