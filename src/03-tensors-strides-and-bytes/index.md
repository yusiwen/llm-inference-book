# Chapter 3 · Tensors, Strides, and Bytes

> - **Compute tier**: ✅ any laptop
> - **Status**: written - the shape, stride and byte arithmetic is derived from the book's 0.5B file (`facts/gguf-qwen2.5-0.5b.json`) and rendered by `scripts/facts.py`; the block sizes were cross-checked against `gguf-py`. Nothing here is timed: this chapter is about addresses.
> - **Build step S3**: A tensor type that owns a byte buffer plus shape, strides and dtype, and can produce a zero-copy view of any 2-D sub-block.
> - **Rule of this book**: This chapter builds the container for numbers inside the process. Which numbers the bytes encode is Chapter 4; how to multiply by them is Chapter 6.

## First, the gist

A tensor is four things and no numbers: **a buffer, a shape, a stride per dimension, and a dtype.** That is the whole type. Everything else - matrix multiplication, attention, dispatch - belongs to code that consumes a tensor, not to the tensor itself.

Two consequences matter, and they are this chapter.

**The buffer is shared, not owned, by anything that looks at it.** A "view" - two rows of an embedding table, one attention head, a batch slice - is the same bytes with a different shape and stride. No copy, no allocation, no function call. In an engine that reads an embedding row per token, that is the difference between 616 bytes touched and 93 megabytes moved.

**The dtype is separate from the buffer, so the bytes stay opaque.** A quantized weight tensor is 144,643,072 bytes of Q8_0 that no code has decoded. The loader does not decode it; the kernel that will use it does, and the loader's job is only to hand over the right `(pointer, shape, strides, dtype)`. Glue the type to the buffer - `Vec<f32>` everywhere - and the engine cannot load a Q5_0 file at all, and pays 4 bytes per weight for the privilege: 2.52 GB instead of 0.49 GB for the file in this book.

The one place where the byte arithmetic gets interesting, and where nearly every beginner implementation breaks, is that **a stride in a quantized tensor is not "elements times bytes per element".** A row of 896 Q5_0 weights is 28 blocks of 22 bytes. Get the block geometry wrong by one and you read the right number of bytes at the wrong place, which produces output that is fluent, wrong, and rarely crashes.

## What you'll be able to do

- Derive a tensor's byte layout from its shape and dtype alone: blocks per row, bytes per row, total bytes - and check the arithmetic against the file (291 of 291 tensors close exactly).
- Explain why the contiguous dimension must be a multiple of the dtype's block width, and what makes that true in the file you are reading (all 291 tensors satisfy it, so the dimension order cannot simply be reversed).
- Produce a view of a sub-tensor and prove it does not copy - by pointer equality and by slice length, not by reading the code.
- State the Rust rules a view needs (parent outlives view, no aliased mutation) and the extra rule a *quantized* view needs (the byte offset must land on a block boundary).
- Say what the engine would cost if weights were typed `f32` instead of opaque bytes, with the arithmetic.

## Lessons

- [ ] 1. A tensor is a buffer, a shape, strides and a dtype - and nothing else
- [ ] 2. Row-major strides, and the alignment GGML needs between rows
- [ ] 3. Views, aliases and the lifetime rules that keep them sound in Rust
- [ ] 4. Bytes in, bytes out: why the engine keeps weights opaque until dispatch
- [ ] 5. Four-dimensional shapes are a story about batches and heads, not a data layout

### Lesson 1 · A tensor is a buffer, a shape, strides and a dtype - and nothing else

Here is the whole type, in the smallest form that stays useful:

```rust
pub struct Tensor<'a> {
    bytes: Bytes<'a>,      // owned, or borrowed from a mapping
    shape: [usize; 4],     // ne[0] is contiguous; unused dims are 1
    strides: [usize; 4],   // in elements, per dimension
    dtype: DType,          // F32, F16, Q5_0, Q8_0, Q6_K, Q4_K, ...
}
```

Four fields, none of them a number of the model. A `f32` matrix is not "a tensor with values"; it is a buffer whose dtype says how to read it. That indirection is the design, and the rest of the book depends on it:

- **The loader is dtype-agnostic.** It reads the file's own type tag and stores it; it never branches on it except to compute a byte length. One loader, every quantization (Lesson 4).
- **The kernel is dtype-specific, chosen at the last moment.** Dispatch picks `q6_k_matmul` over `q5_0_matmul` because the tensor says so (Chapter 16).
- **Nothing in the type says "matrix".** A four-dimensional attention tensor and a one-dimensional bias are the same struct with different shape and stride fields (Lesson 5).

Two numbers from the file keep this honest. It holds **291 tensors, and only two ranks**: 170 are two-dimensional weights, 121 are one-dimensional norms and biases. There is no third kind in the file. Whatever four-dimensional thing you build for attention will be a *view* over these bytes, not a different storage format - which is the last lesson of this chapter, arriving early.

**What a tensor is not.** It is not a matrix class: it has no `mul`, no transpose, no arithmetic at all. It is not a promise about values: a `Q4_K` tensor has no elements you can index individually without decoding a block, which is why the type deliberately exposes bytes. And it is not a statement about ownership: the same type covers a `Vec<u8>` it owns and a slice of a memory-mapped file it does not.

That last point is worth writing down as an invariant, because Rust will make you express it: **a tensor borrows its bytes or owns them, and both must be expressible in the same type** without the arithmetic code knowing which.

### Lesson 2 · Row-major strides, and the alignment GGML needs between rows

Start with a shape and no quantization. A tensor of shape `[896, 151936]` in GGML's convention - `ne[0] = 896` is the *contiguous* dimension - stores row 0 in bytes `[0, 896 * bpe)`, row 1 immediately after it, and so on. The strides in elements are:

```text
stride[0] = 1              # one step along the contiguous dim is one element
stride[1] = ne[0] = 896    # one row is 896 elements further along
offset(i, j) = i * stride[0] + j * stride[1] = i + 896 * j
```

Now quantize it and the arithmetic changes shape. A Q5_0 value does not exist on its own: 32 of them are packed into one 22-byte block. So a row of 896 weights is not 896 values with a byte each; it is **28 blocks** of 22 bytes:

```text
blocks_per_row = ne[0] / elements_per_block = 896 / 32 = 28
row_bytes      = blocks_per_row * bytes_per_block = 28 * 22 = 616
offset(row)    = row * row_bytes = row * 616
```

That is the whole trick, and it is also the trap. `ne[0] / elements_per_block` must be an **integer**: if 896 were not a multiple of 32, the last block of a row would run into the next row. In the book's file every single tensor satisfies it - all 291, checked, zero exceptions - and that is not luck. It is why the format stores the contiguous dimension first, and why you cannot "just transpose" a quantized tensor: reversing the dimensions would need a re-quantization along a different axis, not a stride change.

Here are the two tables the engine actually needs. First, what one block costs:

<!-- facts: gguf-qwen2.5-0.5b.block-sizes -->
| dtype | Elements per block | Bytes per block | Bits per weight | Bytes for a 896-wide row |
|---|---|---|---|---|
| Q5_0 | 32 | 22 | 5.50 | 616 (28 blocks) |
| Q8_0 | 32 | 34 | 8.50 | 952 (28 blocks) |
| Q6_K | 256 | 210 | 6.56 | 630 (3 blocks) |
| Q4_K | 256 | 144 | 4.50 | 432 (3 blocks) |
| F32 | 1 | 4 | 32.00 | 3,584 (896 blocks) |
<!-- /facts -->

Then the geometry of real tensors, derived from their shapes and those block sizes:

<!-- facts: gguf-qwen2.5-0.5b.strides -->
| tensor | dtype | ne0 (contiguous) | ne1 | blocks per row | bytes per row | rows | bytes = rows x row | the file says |
|---|---|---|---|---|---|---|---|---|
| `output.weight` | Q8_0 | 896 | 151,936 | 28 | 952 | 151,936 | 144,643,072 | 144,643,072 (same) |
| `token_embd.weight` | Q5_0 | 896 | 151,936 | 28 | 616 | 151,936 | 93,592,576 | 93,592,576 (same) |
| `blk.0.ffn_down.weight` | Q6_K | 4,864 | 896 | 19 | 3,990 | 896 | 3,575,040 | 3,575,040 (same) |
| `blk.0.ffn_gate.weight` | Q5_0 | 896 | 4,864 | 28 | 616 | 4,864 | 2,996,224 | 2,996,224 (same) |
| `blk.5.attn_v.weight` | Q5_0 | 896 | 128 | 28 | 616 | 128 | 78,848 | 78,848 (same) |
| `blk.0.attn_k.bias` | F32 | 128 | 1 | 128 | 512 | 1 | 512 | 512 (same) |
<!-- /facts -->

Read the last two columns of that table, because they are a check and not decoration: for every tensor, `rows x bytes per row` equals what the file says the tensor occupies. The arithmetic closes exactly, which means the shape, the dtype and the byte count are three views of one fact. When your loader disagrees with any of the three, it is wrong about the other two as well.

**What about alignment between rows?** The most common guess is that GGML pads each row up to a multiple of 16 or 32 bytes so that vector loads are always aligned. It does not. Rows are packed back to back, and the evidence is in the same table: a Q5_0 row is 616 bytes, a Q6_K row is 3,990 bytes. Take 616 modulo 16 and you get 8 - so row 1 begins 8 bytes past a 16-byte boundary, row 2 lands exactly on one, and the pattern alternates down the tensor. Across every two-dimensional tensor in this file, the row length modulo 16 takes exactly three values: **0, 6 and 8**.

That single fact explains a class of kernel code you will meet in Chapter 8: why a "vectorised" inner loop still needs a scalar prologue and epilogue, why llama.cpp repacks some formats into 8x8 tiles before multiplying, and why a kernel that assumes alignment works on the model it was tested with and produces silent garbage on the next one.

**Where the alignment field does apply is between tensors.** The `general.alignment` metadata key (32 in this file) is used once, to compute where the blob starts, and it is what makes every tensor's *base* address a multiple of 32 in a mapping. Nothing inside a tensor is padded. So the correct sentence for your notes is: *alignment is a property of tensor starts, not of rows* - and if you write it the other way round, your byte counts will be off by up to 31 bytes per tensor and every kernel will read shifted weights.

**The stride is in bytes, not elements, for quantized dtypes.** Keep `strides` in elements for shape arithmetic, and compute a byte stride where you need it:

```rust
impl Tensor<'_> {
    /// Bytes between the start of one row and the next. "Elements" are meaningless for a block dtype.
    fn row_bytes(&self, dim: usize) -> usize {
        let per_block = self.dtype.block_elements() as usize;
        let bytes_per_block = self.dtype.block_bytes() as usize;
        self.strides[dim] / per_block * bytes_per_block
    }
}
```

Note what the function does *not* do: ask what the dtype means, or decode anything. It needs two integers per dtype, which is Chapter 4's table.

### Lesson 3 · Views, aliases and the lifetime rules that keep them sound in Rust

A view is a tensor whose bytes are a slice of another tensor's bytes. That is the entire definition; if your implementation copies, it is a tensor and not a view, and the cost it hides is the reason this lesson exists.

<figure class="book-figure">{{#include ../figures/03-views-and-stride.svg}}</figure>

*Figure 3.1 - the parent tensor's rows packed with no padding, and a two-row view that shares them: the view's pointer is the parent's plus 616 bytes, its stride is 616 bytes, and it owns nothing.*

In Rust the type above already expresses it, because `Bytes<'a>` may be borrowed:

```rust
impl<'a> Tensor<'a> {
    /// `count` rows of the parent starting at `row`. Shares the parent's bytes.
    pub fn rows(&self, row: usize, count: usize) -> Tensor<'a> {
        let start = row * self.row_bytes(1);
        let len = count * self.row_bytes(1);
        Tensor {
            bytes: self.bytes.slice(start..start + len),
            shape: [self.shape[0], count, 1, 1],
            strides: self.strides,      // unchanged: the geometry inside a row is the same
            dtype: self.dtype,
        }
    }
}
```

Three rules make this sound, and each one has a failure behind it.

**The parent outlives the view.** The borrow checker enforces it while the view borrows `&self`. If you hand out a raw pointer instead - which you will want to, for a device buffer - the rule becomes yours to keep, and the bug it prevents is a weight matrix that was unmapped while a kernel was reading it.

**Nothing writes to the parent while a view is borrowed.** `&self` gives shared borrows only, which is why the signature is `&self` and not `&mut self`. It also means a view is read-only for free: there is no `&mut` view into a weight tensor anywhere in this book.

**For a quantized dtype, the view's byte offset must land on a block boundary.** This rule has no equivalent for `f32`, and it is where Lesson 2's arithmetic stops being pedantic. A Q5_0 view may start at 616 bytes (616 / 22 = 28, integral) and at 308 bytes (308 / 22 = 14). But those same bytes *reinterpreted as Q4_K* would not be: a Q4_K block is 144 bytes, and 616 / 144 is 4.28. The bytes are there; the block structure is not. Your type must refuse that, and the refusal is a `Result`, not a silent reinterpretation:

```rust
fn reinterpret(&self, dtype: DType, offset: usize) -> Result<Tensor<'a>, ViewError> {
    if offset % dtype.block_bytes() != 0 {
        return Err(ViewError::UnalignedBlock { offset, block: dtype.block_bytes() });
    }
    // ...
}
```

**What the engine gets for these rules.** One function reads an embedding row per token: `token_embd.weight` is 151,936 rows of 616 bytes, so the row for token `id` is `bytes[616*id .. 616*(id+1)]` - 616 bytes touched per token, out of 93.6 MB in the tensor. The same mechanism gives you one attention head's keys, one layer's weights, and a batch slice of activations, all without a copy and all through the same `Tensor` type. That is why the type has four fields rather than four classes.

**What it costs.** Every consumer of a view must agree on the stride convention, and the compiler helps less than you would like: `shape` and `strides` are two arrays that have to stay consistent, and nothing stops you from building a view whose strides say one thing and whose slice length says another. The mitigation is dull and effective - the constructor computes the length from the strides and asserts it fits, once, in one place:

```rust
let need = count * self.row_bytes(1);        // what the geometry says
assert_eq!(need, slice.len(), "view length disagrees with stride geometry");
```

That assertion is the tensor equivalent of Chapter 2's tiling check: cheap, mechanical, and it catches the class of bug that otherwise surfaces as wrong numbers six chapters later.

### Lesson 4 · Bytes in, bytes out: why the engine keeps weights opaque until dispatch

The loader in `myfer-file` knows three things about a weight: where it starts, how many bytes it has, and what type the file claims. It does **not** know how to read a value out of it. That separation is not fastidiousness; it is the only design that survives contact with a file like this one.

Consider what the alternative costs, with arithmetic on the real file. It holds 630,167,424 parameters in 485,452,288 bytes of blob - 6.16 bits per parameter, as Chapter 2 derived. Decode every tensor to `f32` at load time and you hold 630,167,424 x 4 bytes = **2,520,669,696 bytes, or 2.52 GB**, for a file of 0.49 GB: **5.1 times the file**, plus a full pass over the weights before the first token. Decode lazily, per tensor at first use, and you get the same 2.52 GB with a better first token - but you still cannot hold a 7B model's decoded weights in a laptop's memory. The file stays bytes because the *arithmetic* is what needs values, and only a slice of it at a time does.

So the interface between the loader and everything else is:

```rust
// What the loader hands over for a weight tensor: no values, no decoding.
pub struct Weight<'a> {
    pub bytes: &'a [u8],       // exactly the tensor's byte range, from the mapping
    pub shape: [usize; 2],     // as the file stores it: ne0 contiguous
    pub dtype: DType,          // the file's type tag, uninterpreted
}
```

Three things this buys, in the order they matter:

1. **One loader for every quantization.** The file has five dtypes; the loader has no `match` on them beyond a byte-size lookup. Adding Q2_K later is a table entry and a kernel, not a loader change.
2. **Mapping instead of reading.** The bytes are a slice of an `mmap`, so "loading" the weights costs nothing until a kernel touches them (Chapter 2 measured 31-36 ms to fault in the whole file, warm).
3. **The dispatch decision happens where the information is.** Which kernel runs is decided by `dtype` at dispatch (Chapter 16), next to the shape that determines the tiling - not at the top of the program by an `if` on the file name.

What it costs is the thing this chapter keeps insisting on: **every kernel must know the block layout of its dtype**, and every piece of code that computes an offset must use `row_bytes` rather than a value size. There is no type-level guarantee that a kernel's expected layout matches the tensor it is handed; the guarantee is a test that reads one row through the kernel and compares it against a scalar decode of the same bytes (Chapter 6) - the pattern the rest of the book uses.

The reference engine's `src/tensor.rs` and `src/block.rs` are worth reading only *after* you have written your own: the interesting part is not that it stores bytes, it is which few places are allowed to look inside them.

### Lesson 5 · Four-dimensional shapes are a story about batches and heads, not a data layout

Attention wants `[batch, heads, sequence, head_dim]`. The file has no such tensor. It has 170 two-dimensional tensors and 121 one-dimensional ones, and that is all - so the four-dimensional thing you build for attention is a **view with four strides**, over bytes the file laid out flat.

The reason to internalise this is that it changes where bugs live. If `[batch, heads, seq, head_dim]` were a storage format, a wrong number there would be a wrong file. Because it is a view, a wrong number there is a wrong *stride*, and it shows up as attention that attends to the wrong positions: correct shape, correct bytes, plausible output, wrong answer.

Two habits follow, and they are the practical content of this lesson.

**Keep weights two-dimensional, and describe four dimensions at the point of use.** `blk.5.attn_v.weight` is `[896, 128]` in the file - 896 contiguous values per row, 128 rows - because that is how the quantizer wrote it. The forward pass that needs 14 heads of 64 dimensions from it does not reshape the tensor; it computes eight offsets and hands each kernel a view. When a shape is asserted in a test, assert the **offset**, not the shape: `row_bytes(k) == 616 * k` is a claim the file settles, while "shape is [14, 64]" is a claim your own code made up.

**One array of strides, used everywhere, beats four special cases.** Separate code paths for one-, two- and four-dimensional tensors are where implementations of this kind start disagreeing with themselves about which dimension is contiguous. `ne[0]` is contiguous for a bias, for a weight row, and for the last axis of an activation; the same `offset = sum(i[k] * stride[k])` works for all three.

**And one warning about where this abstraction stops.** A transposed weight is not a stride trick when the dtype is quantized: the blocks are laid out along the contiguous axis, so "swap the two strides" gives you a tensor whose blocks no longer describe anything. Real engines handle it by quantizing a second, transposed copy of the matrix at load time, or by making the kernel read the non-contiguous axis - never by pretending the strides are free. That trade-off is Chapter 6's, and it is why `nn.Linear`'s `weight.T` is a one-word operation in PyTorch and a design decision here.

## Layout decisions

| Decision | What it buys | What it costs | Where the book takes it up again |
|---|---|---|---|
| row-major, no packing | the file's own layout, so weights load by mapping with no transform; one offset formula for every rank | no row-level alignment, so vectorised kernels need a prologue and epilogue (a 616-byte row is 8 mod 16) | Chapter 8 (SIMD inner loops) |
| alignment to the tensor start only | every tensor base is 32-byte aligned in a mapping, so kernels can assume the base pointer | rows inside a tensor are unaligned, so a kernel that assumes row alignment is wrong from the second row | Chapter 15 (allocation), Chapter 22 (device buffers) |
| untyped weight bytes | one loader for five dtypes, mapping instead of reading, 0.49 GB instead of 2.52 GB | every kernel must know its block layout, and no compiler check that it does | Chapter 4 (block layouts), Chapter 6 (the dot-product contract) |
| views instead of copies | 616 bytes per token instead of 93.6 MB, and one type for biases, weights and heads | the parent must outlive the view, and a quantized view's offset must land on a block boundary | Chapter 18 (KV cache views), Chapter 20 (a row of logits) |

## Project

**Build step S3 · the tensor crate.** Write the type and its arithmetic first, with unit tests that compare against hand-computed offsets. A tensor library you cannot test on paper is a tensor library you cannot debug in Chapter 6.

The order of work, and the first two steps are the ones people skip:

1. **Write the offset function and test it with no I/O.** Three shapes on paper: a one-dimensional bias of 128 `f32`, a 616-byte row, and a two-dimensional tensor with a two-row view. Hand-compute the byte offsets and assert them.
2. **Add the block geometry table, for two dtypes only** (`F32` and `Q5_0`). One unquantized and one quantized is enough to expose every mistake the abstraction can make; the other three arrive in Chapter 4 with the full table.
3. **Write the view constructor and prove sharing** - assert the view's pointer equals the parent's at that offset, and assert that the view's length is what the stride geometry says.
4. **Add the block-boundary refusal** - `reinterpret` returns `Err` for an offset that is not a multiple of the target dtype's block size, with a test that names the offset.
5. **Point it at the real file** from Chapter 2's loader: for `token_embd.weight`, view row 12345 and print its first four bytes; then check they equal bytes `616*12345 .. +4` of the mapping.

Resist adding a `dtype` enum with per-dtype accessors everywhere. One opaque buffer plus a type tag is the shape that survives to Chapter 24.

## Deliverable

- A `Tensor` type with shape, strides, dtype and an owned or borrowed buffer.
- A view constructor with a test proving the parent's memory is shared, by pointer equality.
- A written note stating your alignment rule and where it is enforced - the rule is "tensor starts are aligned, rows are not", plus the block-boundary rule for views.
- One line in your notes: the row geometry of `token_embd.weight` (blocks per row, bytes per row, total) as your code computes it, next to the file's byte count.

## How you know you passed

- Offsets for three hand-drawn shapes match the test expectations, including a non-contiguous view, and the byte arithmetic of a real tensor closes: `rows x bytes per row` equals the file's byte count for all 291 tensors, not just the one you tried.
- You can point at the line that would go wrong if weights were typed, and say what breaks: 2.52 GB instead of 0.49 GB, and no Q5_0 file loadable at all without five decoders at load time.
- No kernel code was needed to make a view: if it copies, it is not a view - and your test proves it by pointer, not by comment.
- Your view constructor refuses a quantized reinterpretation at a non-block-aligned offset, and the error message names the offset and the block size.

## Reading

- code: `src/tensor.rs`, `src/block.rs` (`repr(C)` block structs) at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/inference_e2e_walkthrough/03-model-dispatch-weights.md`
- docs: `docs/PARAMETER_AUDIT.md` for the real shapes of a Qwen3 model
- [ggml-quant] The ggml quantized block layouts and the k-quant design notes, in the ggml repository - no arXiv ID (specification) - the block structs behind Lesson 2's table. Every element count and byte count there is checked against `gguf-py` by the book's cross-check; the field order *inside* a block is Chapter 4's subject.

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - not yet; this chapter is about addresses. It does set up the measurement: the 616 bytes a token costs in the embedding table are read once per token, and that is one term of the decode budget.
- [x] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - the cost is now arithmetic instead of assertion: 630,167,424 parameters at 4 bytes each is 2.52 GB against a 0.49 GB file, and 6.16 bits per parameter in the blob. What the blocks contain is Chapter 4.
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - partly: the seam starts here as a `Weight` of bytes plus a dtype, with no decoding taken, which is exactly what a device upload needs. The dispatch decision itself is Chapter 16.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - partly: this chapter adds two mechanical checks that cost nothing - the byte arithmetic closes for all 291 tensors, and a view is proven shared by pointer equality. Neither is speed; both are the kind of invariant that makes a later wrong answer findable.
