# Chapter 4 · Quantized Blocks: The Base Formats

> - **Compute tier**: ✅ any laptop
> - **Status**: written - the block layouts, the reconstructed values and the error figures come from two real files (`qwen2.5-0.5b-instruct-q4_0.gguf` and `...-q4_k_m.gguf`) and two decoders, recorded in `facts/dequant-check.json`; speed is not measured here, that waits for Chapter 6
> - **Build step S4**: A dequantizer for Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 that turns one block into 32 floats, plus a test that reconstructs a known vector.
> - **Rule of this book**: This chapter covers the legacy block formats only (the `Q4_0` family and `Q8_0`). The K-quants get Chapter 5 because their super-block structure is a different idea, not a bigger version of this one.

## First, the gist

A 7B model in 16-bit floats is about 14 GB; the same model at four and a half bits per weight is about 4 GB. Quantization is the trade that buys those ten gigabytes, and it is made in one place: **weights are stored in small blocks, each block carries its own scale, and a kernel reconstructs the values while multiplying instead of before.**

The legacy formats are the right place to start because one block is a struct you can hold in your hand: 32 weights, one 16-bit scale, 16 or 32 bytes of payload. Everything after this - K-quants, quantized activations, device kernels - is a variation on that block.

Here is the whole chapter in one comparison, measured on the book's own model rather than argued:

```text
qwen2.5-0.5b-instruct-fp16.gguf    1.27 GB
qwen2.5-0.5b-instruct-q4_0.gguf    0.43 GB     2.96x smaller
```

Both files hold the same 630,167,424 parameters, and the second reconstructs them, one block at a time, from 4-bit integers and a scale. What it costs is measurable too - 7.98 per cent relative RMS error per weight on real weights, in the table further down - and the interesting question is not "why is that acceptable" but "why does it not compound into nonsense when you multiply 4,096 of them together".

## What you'll be able to do

- Read one block of Q4_0, Q4_1, Q5_0, Q5_1 or Q8_0 as fields, and reconstruct its 32 weights by hand - including Q5_0's high-bit plane, which is the one everybody gets wrong.
- Compute bits per weight from a block's size, and explain the 0.5-bit overhead a scale costs at 32 weights versus 16 or 64.
- Verify a decoder against an independent implementation by *values*, not by byte counts, and say precisely what that comparison does and does not prove.
- Rank the five formats by measured reconstruction error on real weights, and explain why one extra bit per weight roughly halves it.
- Explain why a single outlier in a block damages its 31 neighbours, and why a dot product over thousands of terms recovers most of the accuracy the per-weight error suggests it should lose.

## Lessons

- [ ] 1. Why 32 weights per block: the scale overhead arithmetic
- [ ] 2. Q4_0 and Q4_1: nibbles, offsets, and a scale that is a half float
- [ ] 3. Q5_0 and Q5_1: the high-bit plane, and what one extra bit per weight buys
- [ ] 4. Q8_0: the format activations are quantized into, and why the pair Q8_0 x Q4_0 matters
- [ ] 5. Outliers, clipping, and the error a block sees: a worked example

### Lesson 1 · Why 32 weights per block: the scale overhead arithmetic

A block format has to store a scale next to the weights, and that scale is pure overhead: 16 bits that describe 32 values rather than being one of them. So block size is a trade you can compute before you ever look at a model.

A nibble carries 4 bits per weight and the block pays 16 bits for its scale, so the cost of that scale is `16 / block_size` bits per weight:

| Block size | Scale overhead per weight | Bits per weight | Where the format is used |
|---|---|---|---|
| 16 | 16 / 16 = 1.00 | 5.00 | nothing in this book's file |
| 32 | 16 / 32 = 0.50 | 4.50 | Q4_0, Q5_0, Q8_0 |
| 64 | 16 / 64 = 0.25 | 4.25 | not in the GGUF base formats |
| 256 | 16 / 256 = 0.06 | 4.06 | the K-quants, where the scale is 6 bits, not 16 |

The last row is the exception that proves the rule: a 256-weight super-block cannot afford a 16-bit scale per sub-block, which is why the K-quants spend 6 bits on a scale and share one 16-bit multiplier across all 256 weights. Chapters 4 and 5 measure both.

Read down that column and the obvious move is "use a huge block": the file would be smaller and the kernel would do less work per byte. The reason nobody does it is the other half of the trade, which is *error*: one scale has to fit every weight in its block, so the block's value range is the block's problem. A block of 256 weights containing one large value sets a step size that ruins the other 255; a block of 32 confines that damage.

So the real argument for 32 is empirical, and there is a paper behind it rather than a rule of thumb: Dettmers and Zettlemoyer ran 35,000 experiments across 3 to 8 bits and 19M to 176B parameters, and found **block size and data type were the only two levers that improved the bit-level trade-off** - which is why a format designed later (the K-quants, Chapter 5) kept the idea and moved to a 256-weight super-block of eight 32-weight sub-blocks, each with its own scale.

The book's own file gives the arithmetic a check. This model ships at several formats, and the smaller they get the more visible the metadata becomes:

<!-- facts: gguf-qwen2.5-0.5b-q4_0.dtype-mix -->
| dtype | Bytes | Share of blob | Tensors in this file |
|---|---|---|---|
| Q4_0 | 277,853,184 | 65.7% | `attn_k.weight`, `attn_output.weight`, `attn_q.weight`, `attn_v.weight`, `ffn_down.weight`, `ffn_gate.weight`, `ffn_up.weight`, `token_embd.weight` (169) |
| Q8_0 | 144,643,072 | 34.2% | `output.weight` (1) |
| F32 | 286,208 | 0.1% | `attn_k.bias`, `attn_norm.weight`, `attn_q.bias`, `attn_v.bias`, `ffn_norm.weight`, `output_norm.weight` (121) |
<!-- /facts -->

<!-- facts: gguf-qwen2.5-0.5b-q4_0.provenance -->
*`scripts/gguf-inspect.py`; parses the same model at Q4_0; read once from the file. Read from `qwen2.5-0.5b-instruct-q4_0.gguf`; its sha256 is recorded in `facts/gguf-qwen2.5-0.5b-q4_0.json`.*
<!-- /facts -->

Look at that table before moving on. The file is called **q4_0**, and it is **5.37 bits per weight** in the blob, because 34 per cent of it is one Q8_0 tensor: the output matrix (`output.weight`, 896 x 151,936, the largest tensor in any of these files), which quantizers keep at higher precision because it is the last multiplication before the logits. That is Chapter 2's mixed-recipe lesson arriving from the other direction: **the format name describes the payload, not the file.**

The block size is not a constant in the format - it is a property of the type, and the engine answers it in one match:

<!-- code: src/quantize.rs::QuantTarget::blck_size@e844591 -->
```rust

/// The block size (elements per block) of the encoded layout.
pub fn blck_size(self) -> usize {
    match self {
        QuantTarget::F32 | QuantTarget::F16 => 1,
        QuantTarget::Q4_K | QuantTarget::Q5_K | QuantTarget::Q6_K => K_QUANT_BLOCK,
        _ => 32,
    }
}
```
<!-- /code -->

`1` for the float types, `K_QUANT_BLOCK` (256) for the K-quants, `32` for everything else. That is the arithmetic of this lesson in code: 32 weights per block is what makes a 2-byte scale cost half a bit per weight, and a 256-weight super-block with 12 bytes of scales costs 0.375. Note the `_` arm - the legacy quants share a block size, and the engine states that rather than listing them.

### Lesson 2 · Q4_0 and Q4_1: nibbles, offsets, and a scale that is a half float

<figure class="book-figure">{{#include ../figures/04-block-anatomy.svg}}</figure>

*Figure 4.1 - a Q4_0 block: 2 bytes of f16 scale and 16 bytes of packed nibbles, which is 32 weights in 18 bytes, or 4.5 bits each.*

Q4_0 is 18 bytes for 32 weights:

```text
d      f16      2 B    the scale, chosen from the block's largest weight
qs     u8[16]  16 B    32 nibbles: byte j holds weight j in its low nibble and weight j+16 in its high nibble
```

The engine's dequantizer, and the Q4_0 arm is the whole of this lesson:

<!-- code: src/quantize.rs::dequantize_block@e844591 -->
```rust
GgmlType::Q4_0 => {
    let d = f16_at(b, 0);
    (0..32)
        .map(|j| {
            let q = if j < 16 {
                (b[2 + j] & 0x0F) as i32 - 8
            } else {
                (b[2 + j - 16] >> 4) as i32 - 8
            };
            d * q as f32
        })
        .collect()
}
GgmlType::Q4_1 => {
    let d = f16_at(b, 0);
    let m = f16_at(b, 2);
    (0..32)
        .map(|j| {
            let q = if j < 16 {
                (b[4 + j] & 0x0F) as f32
            } else {
                (b[4 + j - 16] >> 4) as f32
            };
            d * q + m
        })
        .collect()
}
```
<!-- /code -->

Watch where the nibble index goes: for `j >= 16` the byte is `b[2 + j - 16]` - the *same* sixteen bytes read again with a shift, not the next byte. The `- 8` centres the unsigned nibble. Q4_1 is the same shape with `d * q + m` and no centring, because its quanta are unsigned and its minimum is explicit: exactly the trade this lesson's table prices. Both arms are four lines and a `collect`, with no branch inside the map beyond the `j < 16` split the compiler hoists.

Three details in those four lines carry the format, and each is a trap.

**The nibble order is not "two in a row".** Low nibble is weight `j`, high nibble is weight `j + 16`, not `j + 1`. Write the naive interleaved version and you get 32 plausible floats in a permuted order: the block is the right length, the values are the right magnitudes, and every dot product using it is wrong. There is no error message for this; the model simply produces worse text than it should - which is why this chapter verifies by value.

**The offset is 8, not 0.** Q4_0 stores unsigned nibbles 0 to 15 and reconstructs `q - 8`, so the range is `-8d` to `+7d`: asymmetric by one step, deliberately, because the scale comes from the block's largest magnitude. Q4_1 adds an `m` field (another f16) and drops the offset: `w = d * q + m` with `q` in 0 to 15, so the range is `m` to `m + 15d` - less symmetric, more expressive, and 20 bytes instead of 18.

**The scale is an f16, not an f32.** Two reasons, and only one is the obvious one. The obvious one is size: an f32 scale would be 32 bits per block, taking the overhead from 0.5 to 1.0 bits per weight - a 22 per cent bigger file at Q4. The subtler one is that the scale does not need f32's precision: it is a single number, and f16's 11-bit significand gives three or four decimal digits, more than the 16 quantization levels being multiplied by it. What f16 *does* need is range, and it has it: roughly 6e-5 to 65,504 brackets every weight scale a trained model produces.

Here is the whole family, generated from the block table that the decoders use:

<!-- facts: dequant-check.anatomy -->
| Format | Weights per block | Bytes per block | Bits per weight | Fields, in file order |
|---|---|---|---|---|
| Q4_0 | 32 | 18 | 4.50 | `d` f16 (2 B), `qs` u8[16] (16 B) |
| Q4_1 | 32 | 20 | 5.00 | `d` f16 (2 B), `m` f16 (2 B), `qs` u8[16] (16 B) |
| Q5_0 | 32 | 22 | 5.50 | `d` f16 (2 B), `qh` u8[4] (4 B), `qs` u8[16] (16 B) |
| Q5_1 | 32 | 24 | 6.00 | `d` f16 (2 B), `m` f16 (2 B), `qh` u8[4] (4 B), `qs` u8[16] (16 B) |
| Q8_0 | 32 | 34 | 8.50 | `d` f16 (2 B), `qs` i8[32] (32 B) |
<!-- /facts -->

<!-- facts: dequant-check.provenance -->
*`scripts/dequant-check.py`; decodes each block format back to f32 and compares it against a second decoder (gguf-py) on the same bytes; max relative error per format; the bit-exact counts are exhaustive over the blocks sampled. Command: see [the provenance index](../appendix/provenance.md#dequant-check). Read from `qwen2.5-0.5b-instruct-q4_0.gguf`; its sha256 is recorded in `facts/dequant-check.json`.*
<!-- /facts -->

Every element count and byte count in that table was checked twice: by this book's reader against the specification, and against `gguf-py`'s constants. That is Chapter 3's rule carried forward - the table is data, so it is checked rather than trusted.

### Lesson 3 · Q5_0 and Q5_1: the high-bit plane, and what one extra bit per weight buys

Q5_0 asks for one more bit per weight and pays 4 bytes per block for it: 22 bytes instead of 18, 5.5 bits per weight instead of 4.5. The mechanism is the part of the legacy formats that is genuinely clever and genuinely easy to get wrong.

Q5_0, where the high bit lives in a plane of its own:

<!-- code: src/quantize.rs::dequantize_block@e844591 -->
```rust
GgmlType::Q5_0 => {
    let d = f16_at(b, 0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    (0..32)
        .map(|j| {
            let lo = if j < 16 {
                (b[6 + j] & 0x0F) as u32
            } else {
                (b[6 + j - 16] >> 4) as u32
            };
            let hi = (qh >> j) & 1;
            d * (((lo | (hi << 4)) as i32) - 16) as f32
        })
        .collect()
}
```
<!-- /code -->

`qh` is read **before** the loop and indexed by `j`, so the high bit for weight `j` is bit `j` of a 32-bit word the block carries separately from its sixteen nibble bytes. The reconstruction is `(lo | (hi << 4)) - 16`: five bits, centred, range -16..15 where Q4_0 had -8..7. Q5_1 is the same with `+ m` in place of `- 16` - which is the whole difference between a format with a minimum and one without.


Read the shift arithmetic carefully, because both lines look wrong at first. `(qh >> j) << 4 & 0x10` moves bit `j` of `qh` into position 4, which is where the fifth bit of a nibble belongs. `(qh >> (j + 12)) & 0x10` takes bit `j + 16` and leaves it *in* position 4 - no shift left, because the 16 has to be subtracted back out. The asymmetry between the two lines is the whole trick, and it exists because 32 bits have to address two sets of 16 nibbles living in the same 16 bytes.

The bias is 16 rather than 8, because `q` now ranges over 0 to 31: the range is `-16d` to `+15d`. Q5_1 replaces the bias with an `m` field, exactly as Q4_1 does with Q4_0, giving 24 bytes and 6.0 bits per weight.

**What the extra bit buys, measured.** The same real weights, quantized into all five formats and decoded back:

<!-- facts: dequant-check.error -->
| Format | Bits per weight | Relative RMS error | Largest single error | Where it lands |
|---|---|---|---|---|
| Q4_0 | 4.50 | 7.98% | 0.02130 | 1.00x of Q4_0 |
| Q4_1 | 5.00 | 6.94% | 0.01642 | 0.87x of Q4_0 |
| Q5_0 | 5.50 | 4.15% | 0.01044 | 0.52x of Q4_0 |
| Q5_1 | 6.00 | 3.40% | 0.00755 | 0.43x of Q4_0 |
| Q8_0 | 8.50 | 0.51% | 0.00129 | 0.06x of Q4_0 |
<!-- /facts -->

Two things in that table are worth internalizing. Q5_0's relative error is **4.15 per cent against Q4_0's 7.98** - almost exactly half, from one bit and 22 per cent more bytes. And the ordering Q4_0 > Q4_1 > Q5_0 > Q5_1 > Q8_0 is not an accident of this sample: an affine format (with the `m` offset) beats its symmetric twin at the same bit width, because it does not have to represent zero as an exact multiple of the step.

That is also the honest answer to "why not just use Q5 everywhere". You can: it costs 22/18 = 1.22 times the file, which in this 0.5B model is 0.43 GB becoming 0.52 GB - at some sizes the difference between fitting beside a 4k context and not. The five formats exist because the trade is real at every point along it.

### Lesson 4 · Q8_0: the format activations are quantized into, and why the pair Q8_0 x Q4_0 matters

Q8_0 is not clever at all: one f16 scale and 32 signed bytes, 34 bytes for 32 weights, 8.5 bits per weight. Its error is 0.51 per cent - sixteen times better than Q4_0 - and for a long time it was the format you used when you cared about accuracy and not about size.

Its real role in an engine is different, and it is why this lesson exists. **Q8_0 is the format activations are quantized into at multiply time.** The trick, which Chapter 6 implements:

1. Take a vector of activations (f32, from the previous layer), find its largest magnitude, divide by 127, and round into 32 signed bytes - a Q8_0 block, computed on the fly, per row.
2. Then the dot product of a Q4_0 weight row with that activation row becomes an integer dot product: unpack the 4-bit values, multiply by signed bytes, accumulate in i32, and apply the two scales once at the end.

That is worth reading twice. The dequantization this chapter teaches is **not** what a fast kernel does - it is what the oracle does, and what you should write first. A production kernel never materialises 32 floats from a block; it multiplies integers and keeps the scale out of the inner loop, which is both faster and more accurate (no intermediate rounding to f32). The block layouts are identical either way, which is why learning them here is not wasted.

The pair Q8_0 x Q4_0 is the historical core of llama.cpp's CPU performance: a Q8_0 activation block times a Q4_0 weight block, `i8` times `u4`, accumulated in `i32`. Later formats (Q4_K, Q6_K in Chapter 5, the IQ-types) changed the weight side, not the activation side.

**Where Q8_0 shows up in this book's files.** In the q4_0 file, exactly one tensor is Q8_0 - `output.weight`, 144,643,072 bytes, 34 per cent of the whole file. That is not an accident of the quantizer: the LM head is the last multiplication before the logits, its error is not averaged over anything downstream, and quantizers in practice leave it at 8 bits (or 16) while compressing everything else to 4. When you write your loader, do not assume a file has one dtype (Chapter 2's tensor table already told you it does not), and do not assume the largest tensor is the embedding: here it is the output matrix, and it is big because it is stored at a higher precision than the embedding it duplicates.

Q8_0 in full, which is the shortest block in the format family:

<!-- code: src/quantize.rs::dequantize_block@e844591 -->
```rust
GgmlType::Q8_0 => {
    let d = f16_at(b, 0);
    (0..32).map(|j| d * b[2 + j] as i8 as f32).collect()
}
```
<!-- /code -->

One half-float scale and thirty-two signed bytes, no packing, no offset, no high-bit plane. `b[2 + j] as i8 as f32` is the entire reconstruction step. That brevity is the argument for the pair: activations are quantized to Q8_0 because decoding it is free, weights are Q4_0 (or a K-quant) because that is where the bytes are, and the kernel in Chapter 6 consumes the Q8_0 bytes without decoding them at all.

### Lesson 5 · Outliers, clipping, and the error a block sees: a worked example

The error table above says a Q4_0 weight is off by about 8 per cent RMS. If that propagated into dot products unchanged, a 7B model at Q4_0 would be unusable: a single attention score would be 8 per cent wrong, and after 28 layers of compounding nothing would survive. It works anyway, for two reasons: **the error is roughly zero-mean and uncorrelated with the activation**, and **it averages out over the sum**.

That averaging is worth doing as arithmetic, because it explains the whole design:

```text
a sum of n terms, each with relative error e, errors independent:
    error of the sum     ~  e * sqrt(n) * rms(term)
    magnitude of the sum ~  sqrt(n) * rms(term)        (random signs)
    relative error of the sum ~ e / sqrt(n)

n = 4096 (one row of 4,096 weights):
    7.98% / 64 = 0.125%
```

One eighth of a per cent per dot product is survivable; eight per cent is not. That square-root factor is why 4-bit weights became practical at the same moment language models got wide.

**Now the part the average hides: outliers.** Inside a block the scale is set by the largest weight, so the other 31 are rounded to a grid chosen for that one value. Here is a real Q4_0 block from this book's file - the worst of 4,096 scanned:

<!-- facts: dequant-check.outlier -->
```text
blocks scanned             4,096  (real Q4_0 blocks, from the file)
worst block                #2477
  largest weight           0.126953
  the block's step         0.015869   (f16 scale; Q4_0 sets it from the largest weight)
  mean of the other 31     0.009726
  step / mean of others    1.63   (a step wider than the values it rounds)
  largest / mean of others 13.1x
median block               step / mean of others 0.39, largest / mean 3.1x
```
<!-- /facts -->

Read the two numbers against each other: the step is **1.63 times the mean magnitude of the other 31 weights** in that block. A weight of magnitude 0.0097 rounded to a grid of step 0.0159 can only be 0, +0.0159 or -0.0159 - one bit of information, not four. The block is not "quantized to 4 bits"; it is quantized to **about one bit for most of its values and four bits for the outlier**. In the median block the ratio is 0.39 instead of 1.63, which is the difference between "the grid fits the values" and "the grid is twice as coarse as the values".

Two engineering responses, both outside this chapter but worth naming so Chapter 5's trade-offs make sense:

- **Clip the block.** Choose the scale from a high percentile instead of the maximum, so the outlier saturates and the other 31 keep their resolution. That is what activation quantization does by default (Chapter 6), and it is why a naive quantizer's first attempt is usually worse than the file you downloaded.
- **Do not put outliers in the same block.** The K-quants' 256-weight super-block with eight sub-block scales is exactly this: eight chances to fit a grid instead of one, so an outlier ruins 32 values rather than 256.

**What to check in your own decoder.** Not the byte count - that is Chapter 3's work and it is exact - but the values. The comparison the book uses:

<!-- facts: dequant-check.agreement -->
| Format | Blocks decoded | Weights compared | Exact matches | Max abs difference | Agreement |
|---|---|---|---|---|---|
| Q4_0 | 8 | 256 | 256 of 256 | 0.0e+00 | bit-exact |
| Q4_1 | 8 | 256 | 256 of 256 | 0.0e+00 | bit-exact |
| Q5_0 | 8 | 256 | 256 of 256 | 0.0e+00 | bit-exact |
| Q5_1 | 8 | 256 | 256 of 256 | 0.0e+00 | bit-exact |
| Q8_0 | 8 | 256 | 256 of 256 | 0.0e+00 | bit-exact |
<!-- /facts -->

Two decoders, the same bytes, **1,280 reconstructed weights** each, all bit-exact. Three of those formats (Q4_0, Q5_0, Q8_0) came from real tensors in a file written by llama.cpp's quantizer, so the bytes are not the witness's own output. The other two (Q4_1, Q5_1) do not appear in any file this book ships; their blocks were produced by the witness quantizing real weights, which means that comparison can catch a transcription error but **cannot** catch a shared misreading of the specification. Say that out loud when you report your own results: the difference between "verified" and "verified against itself" is the difference between this chapter's method and a test that always passes.

The quantizer that produced this lesson's error table, where the clipping and the tie-breaking live:

<!-- code: src/quantize.rs::quantize_q4_0@e844591 -->
```rust
fn quantize_q4_0(x: &[f32], contract: FmaContract) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 18);
    for blk in x.chunks_exact(32) {
        let (mut amax, mut max) = (0.0f32, 0.0f32);
        for &v in blk {
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        put_f16(&mut out, d);
        let mut qs = [0u8; 16];
        for j in 0..16 {
            // llama.cpp's reference is compiled with `-ffp-contract=fast`, so
            // `x*id + 8.5f` is contracted to an FMA on aarch64/x86. Without the
            // FMA the encoder differs from `llama-quantize` in a handful of
            // nibbles (12 of 64512 on one 0.5B tensor) — byte parity needs it.
            let xi0 = trunc_i8(fma(contract, blk[j], id, 8.5)).min(15) as u8;
            let xi1 = trunc_i8(fma(contract, blk[j + 16], id, 8.5)).min(15) as u8;
            qs[j] = xi0 | (xi1 << 4);
        }
        out.extend_from_slice(&qs);
    }
    out
}
```
<!-- /code -->

Three things to take from it. `amax` is the largest **magnitude** but `max` keeps its **sign** - `d = max / -8.0`, so one block's scale is set by that block's single worst weight, which is this lesson's clip. `trunc_i8(...).min(15)` is a truncation toward zero and then a clamp, not a round: the reference truncates, and a rounder quantizer would drift from it. And the comment is the byte-parity obligation in its most concrete form - without the fused multiply-add, twelve nibbles out of 64,512 on one 0.5B tensor differ from `llama-quantize`, which is Chapter 5's Lesson 4 arriving early.

## Block anatomy and cost

The same table, with the arithmetic spelled out - bytes per block divided by weights per block, times 8. Nothing here is a rounded marketing number; the file sizes elsewhere in the book are computed from these figures.

<!-- facts: dequant-check.anatomy -->
| Format | Weights per block | Bytes per block | Bits per weight | Fields, in file order |
|---|---|---|---|---|
| Q4_0 | 32 | 18 | 4.50 | `d` f16 (2 B), `qs` u8[16] (16 B) |
| Q4_1 | 32 | 20 | 5.00 | `d` f16 (2 B), `m` f16 (2 B), `qs` u8[16] (16 B) |
| Q5_0 | 32 | 22 | 5.50 | `d` f16 (2 B), `qh` u8[4] (4 B), `qs` u8[16] (16 B) |
| Q5_1 | 32 | 24 | 6.00 | `d` f16 (2 B), `m` f16 (2 B), `qh` u8[4] (4 B), `qs` u8[16] (16 B) |
| Q8_0 | 32 | 34 | 8.50 | `d` f16 (2 B), `qs` i8[32] (32 B) |
<!-- /facts -->

| What the arithmetic is | Formula | Q4_0 | Q5_0 | Q8_0 |
|---|---|---|---|---|
| bits per weight | bytes x 8 / elements | 4.50 | 5.50 | 8.50 |
| payload for 630,167,424 weights | bits x parameters / 8 | 0.35 GB | 0.43 GB | 0.67 GB |
| the same weights in f16 | 16 bits per weight | 1.26 GB | 1.26 GB | 1.26 GB |

## Project

**Build step S4 · dequantize, then believe it.** Dequantization is the first place where being approximately right is indistinguishable from being wrong: a rounding mode or a mask off by one produces plausible numbers. Compare against a reference on the same file, not against your own earlier output.

Order of work:

1. **Write the scalar decoder for Q4_0 only, and test it by hand.** Take one 18-byte block from the real file, decode it with your code, then decode the same block with a calculator from the pointer and the scale. Sixteen values are enough to catch a nibble-order error, because a permuted decode disagrees immediately.
2. **Add Q8_0** - one multiplication - and check the pair. Q8_0 activations times Q4_0 weights is the dot product you write in Chapter 6, so having both decoders now lets you test that kernel against a scalar path you already trust.
3. **Add Q4_1, Q5_0, Q5_1**, in that order. Slow down on Q5_0: write the high-bit extraction from the specification, not from memory, and test the two halves separately (`j` and `j + 16`).
4. **Build the value-level check**, not the length check: compare your 32 floats against `gguf-py`'s `dequantize` on the same 18/20/22/24/34 bytes. Anything other than exact equality is a bug - these are integer operations followed by one multiply, so there is no rounding freedom to hide in.
5. **Write the error harness**: quantize the same real values into all five formats, decode, report relative RMS. If your Q4_0 error is not near 8 per cent on real weights, your decoder is wrong even when it matches the file - matching the file only proves you agree with that file's encoder.

Write the dequantizer twice: once scalar and obviously correct, once with the SIMD path you will need in Chapter 7. The scalar one is the oracle.

## Deliverable

- A dequantizer for all five formats with a per-format round-trip test.
- A table of measured file sizes for one model at each format, and the bits per weight you computed for each - from your own arithmetic, then checked against the file.
- A note naming the format your engine will use for weights and for activations, and why they differ: 4-bit weights are cheaper to *read*, 8-bit activations are cheaper to *multiply*, and the inner loop wants both.
- One line in your notes: your measured relative RMS error per format, next to the book's numbers in Lesson 3.

## How you know you passed

- Reconstructed blocks match the reference implementation's f32 output *exactly*, and you state which comparison you used and what it cannot prove - a shared misreading of the specification.
- Your bits-per-weight numbers agree with the block sizes you wrote down, and the model-size arithmetic closes against a real file: 630,167,424 x 4.5 bits = 0.35 GB of payload, while the file you have is 0.43 GB because of the Q8_0 output matrix.
- A single outlier weight in a block visibly degrades its neighbours in your worked example, and you can say by how much: in the worst block of this file the step is 1.63 times the mean of the other 31.
- You can explain, with the square-root arithmetic, why 8 per cent per weight does not mean 8 per cent per dot product.

## Reading

- code: `src/block.rs`, `src/quants.rs` and `src/quants/dot_q4_0.rs`, `dot_q4_1.rs`, `dot_q5.rs`, `dot_q8_0.rs` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/GGUF-TOOLING.md` (convert and quantize), `docs/inference_e2e_walkthrough/10-cpu-matmul-kernels.md`
- [kbit] Dettmers and Zettlemoyer, *The case for 4-bit precision: k-bit Inference Scaling Laws* (2022) - arXiv:2212.09720 - the empirical case for small blocks: 35,000 experiments across 3 to 8 bits, and the finding that block size and data type are the only levers that improve the trade-off
- [ggml-quant] The ggml quantized block layouts and the k-quant design notes, in the ggml repository - no arXiv ID (specification) - the field order this chapter implements, and the `dequantize_row_*` functions the book's decoder was written from

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - not answered here, but the shape appears: a 4-bit weight is four times cheaper to *read* than an f32 one and no cheaper to multiply by an f32 activation, which is why Chapter 6's inner loop quantizes the activation as well.
- [x] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - answered in both directions, measured: 2.96x smaller than f16, 7.98 per cent relative RMS error per weight at Q4_0 and 0.51 per cent at Q8_0. The reason blocks exist is the outlier arithmetic in Lesson 5.
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - nothing new; a block is bytes wherever it lives.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - this chapter adds the value-level gate that Chapter 27 generalises: compare reconstructed values against an independent decoder, and state what the comparison cannot prove. It also adds the book's first measurement that is about accuracy rather than speed.
