# Chapter 5 · K-Quants and Byte Identity

> - **Compute tier**: ✅ any laptop
> - **Status**: written - the super-block layouts are decoded twice against real and randomised bytes, the 6-bit packing is compared field by field with the reference implementation, and the file's own composition is counted tensor by tensor; what is *not* here is an encoder diff, because the reference Python package cannot encode a K-quant at all - the chapter says what that gate needs instead
> - **Build step S5**: A Q4_K / Q5_K / Q6_K dequantizer, plus a quantizer whose output is byte-identical to the reference tool's.
> - **Rule of this book**: The super-block idea is explained here, once, in full. What the kernels do with it on a GPU is Chapter 24, and the campaign that optimized those kernels is Chapter 26.

## First, the gist

K-quants put eight 32-weight blocks inside one 256-weight super-block. The payload stays 4 bits per weight; what changes is **how many numbers describe the block**: Q4_0 spends 16 bits on one scale per 32 weights, and Q4_K spends the same 128 bits per 256 weights on **sixteen** numbers - eight 6-bit scales and eight 6-bit minimums, packed three-to-a-byte-and-a-half, plus one shared `d` and `dmin` pair. Same size. Twice the description. That is the trick, and it is made of bit packing, not of arithmetic.

The sting is in that packing: a sub-block's scale is not stored where its number suggests. Sub-block 5's step lives in bytes 5, 9 and 1 of the twelve-byte plane. Get it wrong and you decode a file that looks perfect - right length, right magnitudes - and produces subtly wrong text.

And the second sting is the one this chapter is named for. A quantizer that is *close* is useless: a file that differs from the reference by one byte cannot be diffed, bisected, or compared against a published run. Byte identity is the only test that turns "my quantizer seems right" into a fact - which is why it is a gate in this book's project and an ADR in the engine it studies.

## What you'll be able to do

- Dequantize a K-quant super-block by hand, including the 6-bit scale unpacking where one field is split across three bytes.
- Verify a decoder with *randomised* super-blocks, and explain why random bytes catch what real weights cannot.
- Say what Q4_K actually spends its half-bit of overhead on, and why the answer is not "fewer bits".
- Explain why a tolerance gate is the wrong test for a file format, and what to run instead.
- Read a quantized file's provenance out of its metadata - and name what the metadata does *not* record, which is most of it.

## Lessons

- [ ] 1. Super-blocks: eight sub-blocks, one scale pair, and the arithmetic that motivates it
- [ ] 2. The 6-bit scale packing, unpacked on paper before it is unpacked in code
- [ ] 3. Q6_K and the high-bit planes: 6.5625 bits per weight and why anybody pays it
- [ ] 4. Byte-identical quantization: why a tolerance gate is the wrong test for a file format
- [ ] 5. Reading a quantized file's provenance: what the metadata does and does not record

### Lesson 1 · Super-blocks: eight sub-blocks, one scale pair, and the arithmetic that motivates it

The plan for this chapter said the super-block makes "the scale overhead per weight drop". Measured, it does not. Q4_0 is 4.50 bits per weight and Q4_K is 4.50 bits per weight:

```text
Q4_0   32 weights:  16 B of which 2 B is the scale   -> 4 + 16/32   = 4.50 bits
Q4_K  256 weights: 144 B of which 16 B is description -> 4 + 128/256 = 4.50 bits
```

The overhead is identical - half a bit per weight, both times. What the super-block buys is **resolution inside that half bit**:

```text
Q4_0 spends 128 bits per 256 weights on:  8 scales, 16 bits each
Q4_K spends 128 bits per 256 weights on:  8 scales,  6 bits each
                                          8 minimums, 6 bits each
                                          2 shared values (d, dmin), 16 bits each
```

Sixteen numbers instead of eight, for the same price. That is only possible because 6-bit fields pack: 96 bits of scale and min fit in 12 bytes, where the same 96 bits as 16-bit floats would need 6 fields, not 16.

Why does the extra description help? Because one scale has to cover every weight in its window, and the error a block suffers is set by the *spread* of its values relative to the step. Here is that spread measured on real data, in the only unit that compares across formats - how many of a format's own quantization levels the data actually reaches:

<!-- facts: kquant-check.k-levels -->
| Format | File | Blocks | Grid groups | Levels spanned, median | p10 | p90 | Of |
|---|---|---|---|---|---|---|---|
| Q4_K | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 256 | 2,048 | 15.0 | 14.0 | 15.0 | 16 |
| Q6_K | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 256 | 4,096 | 55.0 | 46.0 | 62.0 | 64 |
| Q4_0 | `qwen2.5-0.5b-instruct-q4_0.gguf` | 256 | 256 | 14.0 | 12.0 | 15.0 | 16 |
<!-- /facts -->

Read the two Q4 lines against each other, because they are the two designs under comparison. Q4_0's median block **reaches 14 of its 16 levels**; Q4_K's median sub-block reaches **15 of 16**, and its tenth percentile is 14 where Q4_0's is 12. The gain is real but modest - and that is the honest finding: the affine minimum buys about a level of range use, which is the same mechanism Chapter 4 measured on the legacy pair: Q4_1 is 13 per cent better than Q4_0 at the same bit width (7.98 per cent to 6.94 per cent), purely because it carries the minimum.

Where the super-block earns its place is not the median block at all - it is the tail. Chapter 4 found a real block where one weight was 13 times the mean of the other 31, and the step chosen for it was 1.63 times wider than the other 31 values. In a super-block that damage is capped at 32 weights, and the neighbouring 224 keep their own grids. The eight scales are eight chances to fit, and the design exists for the blocks that need the ninth.

**One shape fact to carry into Lesson 5.** A 256-weight block can only describe a row whose length is a multiple of 256. This model's hidden size is 896: fine for a 32-weight block (896 = 28 x 32), impossible for a 256-weight one (896 / 256 = 3.5). Every tensor whose row is 896 wide therefore *cannot* be stored as a K-quant, no matter what the mixture asks for. The file in this book's cache is named `q4_k_m` and most of its bytes are not K-quants at all - Lesson 5 counts them.

### Lesson 2 · The 6-bit scale packing, unpacked on paper before it is unpacked in code

<figure class="book-figure">{{#include ../figures/05-superblock.svg}}</figure>

*Figure 5.1 - one Q4_K super-block: eight sub-blocks of 32 weights, each with its own 6-bit scale and minimum, the twelve bytes those sixteen fields live in, and the two 16-bit values shared by all 256 weights.*

Twelve bytes, sixteen fields:

```text
byte  0   EEAAAAAA      A = scale 0, low 6 bits      E = scale 4, high 2 bits
byte  1   FFBBBBBB      B = scale 1                F = scale 5
byte  2   GGCCCCCC      C = scale 2                G = scale 6
byte  3   HHDDDDDD      D = scale 3                H = scale 7
byte  4   eeaaaaaa      a = min 0, low 6 bits      e = min 4, high 2 bits
byte  5   ffbbbbbb      b = min 1                  f = min 5
byte  6   ggcccccc      c = min 2                  g = min 6
byte  7   hhdddddd      d = min 3                  h = min 7
byte  8   eeeeEEEE      scale 4 and min 4, low 4 bits each
byte  9   ffffFFFF      scale 5 and min 5
byte 10   ggggGGGG      scale 6 and min 6
byte 11   hhhhHHHH      scale 7 and min 7
```

Sub-block 5's two numbers live in three different bytes: the low four bits of its scale at byte 9, the high two bits of its scale at byte 1, the low four bits of its min at byte 9 again, and the high two bits of its min at byte 5. If you write the unpacking from intuition you will get sub-blocks 0-3 right and 4-7 wrong, which is exactly the failure mode a real file hides: half the block decodes correctly, so the first spot-check passes.

The code, with the asymmetry made explicit rather than clever:

```rust
fn scale_min(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 0x3F, q[j + 4] & 0x3F)          // low six bits, straight out of the byte
    } else {
        // low four bits from the tail, high two bits from the TOP of an earlier byte
        ((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
         (q[j + 4] >> 4)      | ((q[j]     >> 6) << 4))
    }
}
```

**Verified against the reference, on every bit pattern.** The packing is pure integer work, so there is no rounding to hide behind and the comparison can be exact. The test does not read a real file at all: it feeds 4,096 random twelve-byte planes to both implementations and compares all eight `(scale, min)` pairs on each:

<!-- facts: kquant-check.k-packing -->
```text
planes examined            4,096  (12 random bytes each)
scale fields compared      32,768 of 65,536 of them exact
min fields compared        32,768 of 65,536 of them exact
distinct scale values      64 of 64 exercised
verdict                    bit-exact
```
<!-- /facts -->

32,768 scales and 32,768 minimums, all exact, with every one of the 64 possible scale values exercised. Random planes are the right instrument here for a reason worth stating plainly: a real file's scales only take the values that quantizer chose, so a real-file comparison can pass with two bits of the field never exercised at all.

### Lesson 3 · Q6_K and the high-bit planes: 6.5625 bits per weight and why anybody pays it

Q6_K is 210 bytes for 256 weights - 6.5625 bits, a quarter of a bit short of Q8_0's 8.5 for a third of the size. Its field order is different from the Q4_K/Q5_K pair, and the difference is a deliberate design choice rather than an accident of history:

```text
ql      u8[128]     low 4 bits of every weight
qh      u8[64]      the 2 high bits of every weight, four weights' worth per byte
scales  i8[16]      one signed 8-bit scale per 16 weights
d       f16         one scale for the whole super-block
```

Note the scale count: **sixteen** 16-weight sub-blocks, not eight 32-weight ones, and the scales are signed bytes rather than packed 6-bit fields. At 6 bits per weight there is room to store a scale per 16 weights as a plain `i8`; the packing gymnastics of Lesson 2 are a Q4_Q5 economy, not a K-quant family trait.

The unpacking has its own trap, and this book's first version walked into it:

```rust
for n in 0..2 {                                  // the super-block is decoded in two 128-weight halves
    let base_s = n * 8;                          // <- the missing line in the first draft
    for l in 0..32 {
        let is = l / 16;
        let h = qh[n*32 + l];
        let q1 = ((ql[n*64 + l]      & 0x0F) | ((h & 3) << 4)) as i32 - 32;   // two bits each, four
        let q2 = ((ql[n*64 + l + 32] & 0x0F) | (((h >> 2) & 3) << 4)) as i32 - 32;
        let q3 = ((ql[n*64 + l]      >> 4)   | (((h >> 4) & 3) << 4)) as i32 - 32;
        let q4 = ((ql[n*64 + l + 32] >> 4)   | (((h >> 6) & 3) << 4)) as i32 - 32;
        out[n*128 + l]      = d * sc[base_s + is + 0] as f32 * q1 as f32;
        out[n*128 + l + 32] = d * sc[base_s + is + 2] as f32 * q2 as f32;
        out[n*128 + l + 64] = d * sc[base_s + is + 4] as f32 * q3 as f32;
        out[n*128 + l + 96] = d * sc[base_s + is + 6] as f32 * q4 as f32;
    }
}
```

Without `base_s`, the second half of every super-block uses the **first half's scales**. The values stay in range, the magnitudes stay plausible, and half the block is right - the decoder produced fluent-looking nonsense, which in this chapter's terms is worse than a panic. The unit test above caught it on the first run, at 4,182 exact matches out of 8,192, and the failure was a clean bisection: weights 0-127 exact, 128-255 not.

That is the argument for randomised bytes, stated as a rule: **a decoder is not verified by the data you happen to have.** Random super-blocks exercise bit patterns a quantizer would never emit, and they cost nothing to generate.

**Why pay 6.5625 bits for anything.** Because of where the tensor sits. In this file the Q6_K tensors are twelve `ffn_down` matrices - the ones that project the 4,864-wide feed-forward result back to the residual stream. Two reasons to spend extra bits exactly there: their row length is 4,864, which is the one width in this model that a 256-weight block fits, and their output is added straight into the residual stream, so their error is not averaged away by a later sum. That is the same argument as Chapter 4's Q8_0 output matrix, one layer earlier.

### Lesson 4 · Byte-identical quantization: why a tolerance gate is the wrong test for a file format

Here is the failure a tolerance gate cannot see. Two quantizers differ by one rounding decision in one scale. Decoded, the two files produce weights that differ by 1e-6 relative - far inside any tolerance you would write. Every accuracy test passes, every perplexity number matches to three digits, and the two files are **different files**. Which means:

- You cannot `cmp` your output against the reference tool's, so you cannot bisect a difference to a field. You are back to guessing.
- You cannot reproduce a published run. Whatever the paper or the bug report measured, it measured bytes you do not have.
- You cannot cache or ship by hash. Two files that should be identical are different artifacts.

Quantization is a deterministic integer program over the same input, so **exactness is achievable** - which removes the excuse a tolerance gate provides. The engine this book studies makes it a rule rather than a habit: a quantized file is byte-identical to `llama-quantize`'s, or it is wrong (ADR-0020).

**The boundary of what this book measured, stated plainly.** The reference Python package (`gguf-py`) has a decoder for every K-quant and an encoder for none:

<!-- facts: kquant-check.k-encoders -->
| Format | Reference Python package |
|---|---|
| Q4_K | decode only (no encoder in gguf-py) |
| Q5_K | decode only (no encoder in gguf-py) |
| Q6_K | decode only (no encoder in gguf-py) |
<!-- /facts -->

So the byte-identity gate cannot be run from Python at all - it needs the reference *binary*. The recipe, which the book's project gate runs on a machine that has it:

```text
1. quantize a small tensor with your encoder            -> mine.gguf
2. quantize the same tensor with llama-quantize         -> ref.gguf
3. cmp mine.gguf ref.gguf                               -> must be silent
4. if it is not: cmp -l | head, then map the byte offset back to (field, block, sub-block)
```

And the test to write *before* the gate, so you know the gate works: flip one byte of `mine.gguf` by hand and confirm step 3 fails. A gate nobody has broken on purpose is a gate that might be checking nothing - this book's own history has three of those (a CSS selector that matched no element, a `--strict` flag that defaulted to lenient, a probe that matched its own name in the log).

Do not weaken the test when it fails. The interesting failures all look like "the file is nearly identical": a scale chosen from `max` where the reference chose from `-min`, a rounding tie broken to even instead of away from zero, the iterative scale search stopping one step early. Each of those is a *bisectable* difference only because the gate is exact.

### Lesson 5 · Reading a quantized file's provenance: what the metadata does and does not record

The metadata of the file in this book's cache says, in full:

```text
general.file_type              UINT32   15      (LLAMA_FTYPE_MOSTLY_Q4_K_M)
general.quantization_version   UINT32   2
```

Two integers. From those a reader is supposed to infer what the file contains. Here is what it actually contains:

<!-- facts: gguf-qwen2.5-0.5b.dtype-mix -->
| dtype | Bytes | Share of blob | Tensors in this file |
|---|---|---|---|
| Q5_0 | 266,742,784 | 54.9% | `attn_k.weight`, `attn_output.weight`, `attn_q.weight`, `attn_v.weight`, `ffn_gate.weight`, `ffn_up.weight`, `token_embd.weight` (133) |
| Q8_0 | 146,105,344 | 30.1% | `attn_v.weight`, `output.weight` (13) |
| Q6_K | 42,900,480 | 8.8% | `ffn_down.weight` (12) |
| Q4_K | 29,417,472 | 6.1% | `ffn_down.weight` (12) |
| F32 | 286,208 | 0.1% | `attn_k.bias`, `attn_norm.weight`, `attn_q.bias`, `attn_v.bias`, `ffn_norm.weight`, `output_norm.weight` (121) |
<!-- /facts -->

The file is called **q4_K_M** and **6.1 per cent of its bytes are Q4_K**. Four types, five ways of being wrong if you assume the name. The arithmetic that explains every row:

- **Rows 896 wide cannot take a 256-weight block.** 896 = 28 x 32, so legacy blocks fit exactly and K-quants do not (896 / 256 = 3.5). The `token_embd`, `attn_q`, `attn_k`, `attn_output`, `ffn_gate` and `ffn_up` matrices are all 896 wide, and all of them hold **Q5_0** - 133 tensors, 54.9 per cent of the blob.
- **`ffn_down` is 4,864 wide** (4,864 = 19 x 256), so K-quants survive there - and only there: 12 tensors of **Q4_K** and 12 of **Q6_K**.
- **`attn_v` is split 12/12** between Q5_0 and Q8_0 - the same twelve layers that got Q6_K `ffn_down` got Q8_0 `attn_v`, and the other twelve got Q5_0 for both. So a per-layer rule drove both categories, and `attn_v` never reaches a K-quant at all: 896 is not a multiple of 256, whatever the rule asked for.

Now compare that split with the rule the engine implements, which is llama.cpp's `use_more_bits`:

<!-- facts: gguf-qwen2.5-0.5b.layer-split -->
```text
layers in the file     24
ffn_down, Q6_K         0, 1, 3, 6, 7, 8, 9, 10, 13, 16, 19, 21
ffn_down, Q4_K         2, 4, 5, 11, 12, 14, 15, 17, 18, 20, 22, 23
attn_v,  Q8_0          0, 1, 3, 6, 7, 8, 9, 10, 13, 16, 19, 21
attn_v,  Q5_0          2, 4, 5, 11, 12, 14, 15, 17, 18, 20, 22, 23

the current rule at the baseline (i < n/8, i >= 7n/8, or (i - n/8) % 3 == 2)
would pick             0, 1, 2, 5, 8, 11, 14, 17, 20, 21, 22, 23
that is a DIFFERENT set.
```
<!-- /facts -->

The two sets are **different**. Every layer in the file's Q6_K set is a "needs more bits" layer by some rule, but not the rule at this book's baseline - which means this file was produced by a llama.cpp that chose differently, and the twelve-layer split it actually used is not recoverable from anything except its tensor table.
- **`output.weight` is Q8_0** - the LM head, last multiplication before the logits, the same decision Chapter 4 found in the q4_0 file.
- **121 F32 tensors** are the norms and biases, which nobody quantizes.

Now the provenance argument. The metadata records *which mixture was requested*. It does not record:

- the selection policy - the `use_more_bits` layer rule, the fused-qkv rule, the output-matrix rule, the row-length demotion, and the order in which those rules run;
- the encoder's version, or the reference tool's;
- the source file the weights came from, or its hash;
- the per-tensor type decisions at all, except implicitly, in the tensor table you have to decode anyway.

So two files with `general.file_type = 15` can contain different bytes, and there is no field in either that says so. This book has one in front of it: the twelve-layer split above does not match the current rule, which is proof by example that the integer alone does not identify the bytes. The only provenance you can trust is the one you record yourself: the command, the tool version, the input hash, and the output hash - plus a gate that fails when any of them drifts.

This is why Lesson 4's byte identity is not pedantry. Byte identity is the *only* property that makes a quantized file's name mean something.

## Super-block cost

The same three formats, with the arithmetic spelled out - and against the legacy formats, which is where the surprise lives: Q4_K is the same 4.50 bits as Q4_0, because it spends its description budget differently rather than spending less.

<!-- facts: kquant-check.k-anatomy -->
| Format | Weights per super-block | Sub-blocks | Bytes per super-block | Bits per weight | Fields, in file order |
|---|---|---|---|---|---|
| Q4_K | 256 | 8 | 144 | 4.5000 | `d` f16, `dmin` f16, `scales` u8[12], 6-bit packed, `qs` u8[128] |
| Q5_K | 256 | 8 | 176 | 5.5000 | `d` f16, `dmin` f16, `scales` u8[12], 6-bit packed, `qh` u8[32], `qs` u8[128] |
| Q6_K | 256 | 8 | 210 | 6.5625 | `ql` u8[128], `qh` u8[64], `scales` i8[16], `d` f16 |
<!-- /facts -->

| Format | Bytes per block | Weights | Bits per weight | Description bytes, per 256 weights | Numbers described |
|---|---|---|---|---|---|
| Q4_0 | 18 | 32 | 4.50 | 16 | 8 scales |
| Q4_K | 144 | 256 | 4.50 | 16 | 8 scales + 8 mins + shared d, dmin |
| Q5_K | 176 | 256 | 5.50 | 16 | 8 scales + 8 mins + shared d, dmin |
| Q6_K | 210 | 256 | 6.5625 | 18 | 16 signed scales + shared d |

## Project

**Build step S5 · the quantizer nobody can argue with.** Take one small source tensor, quantize it with your implementation and with the reference tool, and diff the bytes. This is the first identity gate in the book; Chapter 27 makes it a habit.

When the bytes differ, do not weaken the test. Bisect the difference to a field - a scale, a nibble order, a rounding mode - and fix the field.

Order of work:

1. **Port the decoder for Q4_K first**, and test it against randomised super-blocks before you test it against a file. If your unpacking is wrong for sub-blocks 4-7 only, a real file's first spot-check will not tell you.
2. **Do Q6_K second, deliberately**, because it is the one whose field order differs from Q4_K's - the half-super-block scale offset is a bug this book wrote and its own test caught at 4,182 of 8,192 weights exact.
3. **Add Q5_K last**: it is Q4_K's layout plus a 32-byte high-bit plane, and if you write it first you will copy Q4_K's assumptions without testing them.
4. **Write the encoder.** This is the hard half, and none of it is in this book's Python witness: the scale search (`make_qkx2_quants` and friends) is an iterative optimisation per sub-block, not a closed form. Expect to spend your time here, and expect the failures to be one-bit and one-rounding-decision sized.
5. **Gate on bytes, and break the gate on purpose first.** Perturb one byte, confirm the gate fails, restore it. Then run it against `llama-quantize` on a small tensor.

## Deliverable

- Q4_K, Q5_K and Q6_K dequantizers with per-format tests, at least one of which uses randomised super-blocks rather than a model file.
- A quantize-then-diff run producing zero differing bytes on one tensor, with the command, the tool version and both hashes recorded.
- A one-page decision note: which K-quant your engine defaults to and on what evidence - including what happens to a 896-wide row.

## How you know you passed

- `cmp` reports the two quantized files identical, and the command is in your notes with the reference tool's version.
- Your decoder agrees with the reference implementation on random super-blocks, not only on a real file: every one of the 64 possible 6-bit scale values exercised.
- You can unpack a 6-bit scale field on paper, without the code, and point at the three bytes sub-block 5 lives in.
- You can name a case where Q6_K is the wrong choice, and it is not "it is bigger": on this model no `attn_v` tensor can be a K-quant at all, because 896 is not a multiple of 256 - so the file contains Q8_0 and Q5_0 `attn_v` and not one Q6_K.

## Reading

- code: `src/quants/kquant.rs`, `src/quantize.rs`, `src/q4k_dsc.rs` at the book's [pinned baseline](../appendix/baseline.md); the mixture's selection policy is `src/quantize/mix.rs::use_more_bits` and `src/quantize/mix.rs::Target::file_type`
- docs: `docs/GGUF-TOOLING.md`, `docs/inference_e2e_walkthrough/10-cpu-matmul-kernels.md`
- decision: ADR-0020 (a quantized file is byte-identical to llama-quantize, or it is wrong); ADR-0041 (a row that cannot take the target's block is demoted)
- [ggml-quant] The ggml quantized block layouts and the k-quant design notes, in the ggml repository - no arXiv ID (specification) - the twelve-byte scale plane this chapter unpacks, and the `dequantize_row_q4_K` family the decoder was written from
- [kbit] Dettmers and Zettlemoyer, *The case for 4-bit precision: k-bit Inference Scaling Laws* (2022) - arXiv:2212.09720 - why block size is one of the two levers that matter, and what happens to the trade-off below 4 bits

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - still not answered, but the shape sharpens: a K-quant weight is not one memory access but one super-block, and the kernel's inner loop has to unpack 6-bit fields before it can multiply. Chapter 6 measures whether that unpacking costs more than the bytes it saves.
- [x] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - completed. The super-block does not buy smaller files than Q4_0 at the same bit width; it buys sixteen numbers of description instead of eight, a level of range use (14 of 16 to 15 of 16), and damage control for outliers (32 weights, not 256). What it costs is complexity: three places for one scale, and a byte-identity gate you cannot run without the reference binary.
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - nothing new; a super-block is bytes wherever it lives, though the *unpacking* is where device kernels differ most.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - the chapter's spine. Three instruments, three kinds of evidence: randomised bytes for the decoder (catches field order a file hides), exact integer comparison for the packing (no rounding to hide in), and byte identity for the encoder (the only test that makes a file name mean something). Plus the boundary, stated rather than papered over: the reference Python package cannot encode a K-quant, so that gate needs the binary.
