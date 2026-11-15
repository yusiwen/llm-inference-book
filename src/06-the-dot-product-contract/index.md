# Chapter 6 · The Dot Product Is the Engine

> - **Compute tier**: ✅ any laptop
> - **Status**: written - the error decomposition and the accumulator bounds are measured on real rows of the book's files, and the three scalar kernels are timed on this machine (median of seven runs); what is *not* here is vector code, which is Chapter 7, or any thread, which is Chapter 8
> - **Build step S6**: A matmul that computes one row of a quantized weight matrix against an activation vector, with the activation quantized to Q8_0 first.
> - **Rule of this book**: This chapter is about the arithmetic contract and its scalar version. SIMD is Chapter 7; splitting the work across threads and choosing between matrix-vector and matrix-matrix shapes is Chapter 8.

## First, the gist

Almost every floating-point operation in a transformer is a dot product between a weight row and an activation vector, and there are billions of them per token. An engine's core trick is to stop multiplying floats: it quantizes the activation to 8-bit integers once per row, then computes the dot product in integer arithmetic with a single scale correction at the end. That is faster because integer multiply-add is cheaper and because the weight is already quantized on disk. This chapter states the contract precisely, because every kernel in the rest of the book - scalar, SIMD, Metal, CUDA - implements the same one.

The claim "it is faster" is measurable even without a single SIMD instruction, and the number is bigger than it has any right to be:

<!-- facts: dotbench.dot-speed -->
| Path | ns per row | Weight throughput | Multiply-accumulates | Speed vs dequantize-then-dot | Run range |
|---|---|---|---|---|---|
| `dq` - dequantize to f32, then f32 fma | 387.6 | 1.30 GB/s | 2.31 GMAC/s | 1.00x | 2.71-4.91 ms |
| `i32` - integers, i32 accumulator | 69.9 | 7.21 GB/s | 12.81 GMAC/s | 5.54x | 0.48-0.92 ms |
| `i16` - integers, i16 block accumulator | 43.0 | 11.73 GB/s | 20.85 GMAC/s | 9.02x | 0.31-0.51 ms |
<!-- /facts -->

<!-- facts: dotbench.provenance -->
*`scripts/dotbench.c`; times three scalar kernels on 8,192 real rows of the book's file; on **Apple M4 Pro, clang -O2**; 7 runs; median of the runs, with the full range beside it. Command: see [the provenance index](../appendix/provenance.md#dotbench). Raw numbers: `facts/dotbench.json`.*
<!-- /facts -->

Two kernels, the same 896 multiply-accumulates per row, the same bytes read, the same answer (the checksums agree exactly). The integer one is **5.5 times faster** in plain C, and the 16-bit variant - which is legal only for certain formats, as Lesson 3 shows - is **9 times** faster. Nothing was vectorised. The work itself changed shape.

## What you'll be able to do

- Write a scalar quantized dot product that matches a dequantize-then-dot reference within a stated bound - and say which of the two errors you are looking at.
- Explain the Q8_0 activation policy and where its rounding error enters, without hand-waving.
- Prove whether a 16-bit accumulator is legal for a given format, from the format's own bounds rather than from a test that happened to pass.
- Name the dispatch a given `(weight dtype, activation dtype)` pair requires, and what an unsupported pair must do (refuse, never fall back silently).

## Lessons

- [ ] 1. One operation, billions of times: where the dot product appears in a transformer
- [ ] 2. Quantize the activation, once: Q8_0 blocks, scales, and the error budget
- [ ] 3. The integer dot product and the single scale correction at the end
- [ ] 4. The dispatch contract: dtype pair in, kernel out, and the refusal rule (payoff: why a silent CPU fallback is a bug, not a safety net)
- [ ] 5. A scalar reference you can trust: slow, obvious, and the oracle for every kernel after it

### Lesson 1 · One operation, billions of times: where the dot product appears in a transformer

Count the dot products in one token of the book's 0.5B model, because the arithmetic explains every design decision that follows. For a hidden size of 896 and a feed-forward width of 4,864, each layer does:

<figure class="book-figure">{{#include ../figures/06-layer-weights.svg}}</figure>

*Figure 6.1 - one layer's weights, and where they go. The four attention matrices are 1.83M weights (12 per cent); the three feed-forward matrices are 13.07M (88 per cent), and `attn_k` and `attn_v` are narrow because this model shares keys and values across query heads (2 KV heads of head_dim 64, which Chapter 10 needs).*

Twenty-four layers of that, plus the output matrix (896 x 151,936). Which is why a decode step reads essentially the entire model and performs one multiply-accumulate per weight: **the arithmetic intensity of token generation is about one MAC per byte of weights**. That ratio is the whole economic argument for quantizing weights (four times fewer bytes to read) and the whole reason Chapter 1's bandwidth arithmetic looked the way it did.

Two consequences to keep in your head:

1. **The dot product is the engine.** There is no other operation with this volume. If your dot product is 5x slow, everything is 5x slow.
2. **The weight is read once per token and reused zero times.** In prefill that changes - the same weight row serves many tokens (Chapter 8) - and that difference is why engines have two shapes of kernel, not one.

The activation side is small by comparison: 896 floats per row, and each is reused by every row of the matrix it multiplies. That asymmetry is the opening for Lesson 2: you can afford to spend arithmetic on the *weights* (which are read once) but you cannot afford to re-read the *activations* (which are tiny and hot). Quantizing the activation is not about saving bandwidth; it is about making the multiply cheap.

### Lesson 2 · Quantize the activation, once: Q8_0 blocks, scales, and the error budget

The policy is one sentence: take the f32 activation row, and for every 32 values find the largest magnitude, divide by 127, and round into signed bytes. That is a Q8_0 row - the same block format Chapter 4 decoded, except that it never exists on disk.

The engine's quantizer is that policy in code, and three details in it are this lesson:

<!-- code: src/quants/avx2.rs::quantize_scalar@88ffa5d -->
```rust
pub(super) fn quantize_scalar(x: &[f32], y: &mut [u8], k: usize) {
    let nb = k / 32;
    for i in 0..nb {
        let mut am = 0.0f32;
        for j in 0..32 {
            am = am.max(x[i * 32 + j].abs());
        }
        let d = am / 127.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
        let db = half::f16::from_f32(d).to_bits().to_le_bytes();
        let yo = i * Q8B;
        y[yo] = db[0];
        y[yo + 1] = db[1];
        for j in 0..32 {
            y[yo + 2 + j] = (x[i * 32 + j] * id).round_ties_even().clamp(-128.0, 127.0) as i8 as u8;
        }
    }
}
```
<!-- /code -->

`am / 127`, not `/ 128`: the scale maps the largest magnitude onto the symmetric range the byte can hold without a separate zero point. `round_ties_even`, not `round`: the reference rounds half to even, while half-away-from-zero differs on exactly the values that land on .5 - one ulp of the quantizer, and enough to break a bitwise comparison against the reference's output. The `d != 0.0` branch is the one that saves you: an all-zero block gets a zero scale and zero quanta instead of a `NaN` from zero times `inf`.

And the subtle one this lesson's error table does not show: the quanta are computed against `1/d` in f32, while the **stored** scale is `f16(d)`. Dequantizing multiplies by `f16(d)`, so the round trip carries an extra relative error of up to $2^{-11}$ that the quantizer itself never sees. That is a ledger entry, and Chapter 9 is where it gets counted.

**What it costs.** Quantizing a real activation vector and decoding it back:

<!-- facts: dotcheck.dot-errors -->
| Pair | Rows | Activation, Q8_0 rounding | Float path, relative | Integer path, median relative | Integer path, over the norm product | Integer arithmetic alone |
|---|---|---|---|---|---|---|
| Q4_0 x Q8_0 | 512 | 0.534% | 2.2e-07 | 1.3e-03 | 7.7e-04 | 0.0e+00 |
| Q8_0 x Q8_0 | 512 | 0.000% | 1.6e-06 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
<!-- /facts -->

<!-- facts: dotcheck.provenance -->
*`scripts/dotcheck.py`; decomposes the quantized dot product's error into activation, integer and scale parts, and bounds each format's accumulator; max relative error against f64; the bounds are arithmetic, the worst partial is measured. Command: see [the provenance index](../appendix/provenance.md#dotcheck). Read from `qwen2.5-0.5b-instruct-q4_0.gguf`; its sha256 is recorded in `facts/dotcheck.json`.*
<!-- /facts -->

Read the first column: the activation's own rounding is **0.534 per cent** relative RMS. That is the price of 8 bits, and it is the same order as the 4-bit weights' 7.98 per cent from Chapter 4 - smaller because 8 bits is 16 times finer than 4, not twice.

The fourth column is the one that matters for the contract: the error the integer path actually delivers, measured against the norm product (the scale-free form, because a relative error on a dot product that happens to be near zero is meaningless - 512 rows include some of those, and the raw relative figure reaches 21 per cent on the worst one while the norm-product figure stays at **7.7e-04**).

**Why 7.7e-04 is smaller than 5.3e-03.** Because the activation's errors are independent across the 896 terms and the sum does not grow with them: the dot product's error is the weighted sum of 896 independent roundings, which grows like the square root of 896, about 30, while the dot product itself grows like the norm product. So a per-element 0.53 per cent arrives at the dot product as roughly 0.53/30 per cent. Chapter 4 made the same argument for weights; this is the same argument on the other side of the multiply, and it is measured here rather than asserted.

**The second row is a free identity.** When the weights are already Q8_0 and the activation is a dequantized row of that same matrix, re-quantizing it is **exact** - the Q8_0 round trip is idempotent, because the values it decodes to are exactly representable in Q8_0. That is worth knowing for a different reason: it means a Q8_0 x Q8_0 test cannot detect an activation-quantization bug at all. The test that can is Q4_0 weights with real activations, which is why the first row exists.

### Lesson 3 · The integer dot product and the single scale correction at the end

<figure class="book-figure">{{#include ../figures/06-quantized-dot.svg}}</figure>

*Figure 6.2 - the contract as dataflow: the activation row quantized once, per-block integer dot products accumulated exactly, and one f32 multiply per block to restore the scale.*

Here is the whole kernel, scalar version, with the parts named:

```rust
fn dot_q4_0_q8_0(w: &[u8], a: &[u8], blocks: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let wb = &w[b * 18..];                    // d_w (f16), 16 nibble bytes
        let ab = &a[b * 34..];                    // d_a (f16), 32 i8
        let d = f16(wb, 0) * f16(ab, 0);           // <- the only float multiply per block
        let mut acc = 0i32;                        // <- the accumulator, exact
        for j in 0..16 {
            let q = wb[2 + j];
            acc += ((q & 0x0F) as i32 - 8) * (ab[2 + j] as i8 as i32);
            acc += ((q >> 4) as i32 - 8) * (ab[2 + j + 16] as i8 as i32);
        }
        sum += acc as f32 * d;                     // <- the scale correction, once per 32 weights
    }
    sum
}
```

Three properties of that code are the contract, and the third is the one people get wrong.

**The accumulation is exact.** `acc` is an i32 and every term is an integer, so there is no rounding anywhere inside the loop. The measured column in Lesson 2 proves it: the difference between the integer path and the *same* quantized activations computed in f64 is **0.00e+00** on every row. All of the error is upstream, in what was rounded before the kernel ran.

**The scale is applied once per block, in f32.** 28 multiplies per 896 weights, not 896. That is where the speed comes from, and it is also where the naive path pays: the dequantize-then-dot version writes 32 floats into memory and multiplies each of them, so it does roughly 29 times more floating-point work for the same answer.

**The accumulator width is a property of the format, not of your data.** A block's partial sum is bounded by construction: 32 weights of magnitude at most 8 (Q4_0), times activations of at most 127:

<!-- facts: dotcheck.dot-bounds -->
| Weight format | Weights per block | Largest absolute activation | Worst-case block partial | Fits in i16 (32,767)? |
|---|---|---|---|---|
| Q4_0 | 32 | 8 | 32,512 | yes |
| Q5_0 | 32 | 16 | 65,024 | no |
| Q6_K | 32 | 32 | 130,048 | no |
| Q8_0 | 32 | 127 | 516,128 | no |
<!-- /facts -->

For Q4_0 that worst case is 32,512 and an i16 holds 32,767 - **legal, with 255 to spare.** For Q5_0 (16 levels of magnitude) it is 65,024 and an i16 does not. This is not a detail you can decide by testing: the measured worst partial in 512 real rows is 9,484, so a 16-bit accumulator would pass every test you write and then overflow on one unlucky block in someone else's model. The bound is the test.

Which is why the engine's two vector paths do different things and both are correct: the AVX2 kernel multiplies unsigned nibbles by signed bytes into i16 *pairs* (`_mm256_maddubs_epi16`, at most 2,032 per pair) and widens immediately, while the NEON kernel goes straight into i32 with `sdot` (`src/quants/neon.rs::dot16`). Neither relies on the block-level i16 headroom, because one of the formats it must serve does not have it.

**And the 16-bit path is worth having anyway**: 1.63x faster than i32 in the scalar benchmark, because half the accumulator traffic. It is legal for Q4_0, which is the format most weights in the world are stored in. It is illegal for the rest, and the dispatch has to know which is which - which is Lesson 4.

The engine's own version of those three lines, and it is shorter than the sketch because it needs no part names:

<!-- code: src/quants/dot_q8_0.rs::dot_q8_0_q8_0_scalar@88ffa5d -->
```rust
fn dot_q8_0_q8_0_scalar(x: &[u8], y: &[u8], nb: usize) -> f32 {
    let mut s = 0.0f32;
    for ib in 0..nb {
        let xb = &x[ib * Q8B..];
        let yb = &y[ib * Q8B..];
        let dx = block::fp16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
        let dy = block::fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
        let mut si = 0i32;
        for j in 0..32 {
            si += (xb[2 + j] as i8 as i32) * (yb[2 + j] as i8 as i32);
        }
        s += si as f32 * dx * dy;
    }
    s
}
```
<!-- /code -->

`i32` is this lesson's accumulator table in code, and the scale lands **once per block, at the end** - two multiplies for 32 values, not 32 multiplies and a rounding each.

### Lesson 4 · The dispatch contract: dtype pair in, kernel out, and the refusal rule

A kernel is not a function of a tensor; it is a function of a **pair** of tensors. The same weight bytes need different code depending on what the activations are, and the same is true in reverse. Here is the table this book's engine keeps, with what this chapter measured in the last column:

| Weight dtype | Activation dtype | Kernel family | Same result as dequantize-then-dot? | Status in this book |
|---|---|---|---|---|
| Q4_0 | Q8_0 | integer dot, i32 partial, f32 scale | yes, to 7.7e-04 of the norm product | **measured**, 512 real rows |
| Q8_0 | Q8_0 | integer dot, i32 partial, f32 scale | yes, exactly (activation rounding is nil) | **measured**, 512 real rows |
| Q4_K | Q8_K | super-block dot; activation is 256-wide with per-16 i8 scales and an f32 `d` | by the same construction, not measured here | not measured: the activation format itself is different |
| Q4_K | f32 | dequantize to f32, then f32 fma | exact by definition | trivial |
| f32 | f32 | f32 fma | this *is* the definition | measured against f64: 2.2e-07 |

Two rules follow from that table, and both are enforced in code rather than in a comment.

**Rule one: the pair determines the kernel, and the table is exhaustive.** `(Q4_K, Q8_0)` is not in it - and it is not a valid pair, because a Q8_0 activation row has a 32-value scale and a Q4_K super-block has 256-value scales; pairing them would apply the wrong scale to 224 of every 256 weights. A dispatch table that silently *widens* to a supported pair is a bug factory: it will produce plausible numbers with a scale mismatch that no accuracy test catches, because the model is at its least surprising when the error is systematic.

**Rule two: an unsupported pair refuses, naming the pair.** Not a fallback, not a warning on stderr, not "slower but correct". In the engine this book studies the dispatch returns an error carrying both dtypes; the tests assert on the error text. The reason a silent CPU fallback is worse than a crash is arithmetic: a fallback turns a *coverage* failure (you did not implement this pair, and you know it) into a *correctness* failure (you ran something that computes a different function, and nothing tells you). Chapter 16 will show the same rule at a bigger scale, when a graph gets split across devices.

The dispatch contract as the engine writes it:

<!-- code: src/kernel/dispatch.rs::cpu_quant_matmul_f32@88ffa5d -->
```rust
pub fn cpu_quant_matmul_f32(
    w: &Tensor,
    x: &[f32],
    out: &mut [f32],
    od: usize,
    id: usize,
    nt: usize,
) {
    match w.ttype {
        TensorType::Q4_K | TensorType::Q5_K | TensorType::Q6_K => {
            let n_super = id / 256;
            let mut qb = vec![0u8; nt * n_super * Q8KB];
            crate::quants::quantize_row_q8_k_buf(x, nt, id, &mut qb);
            cpu_quant_matmul(w, &qb, out, od, id, nt)
        }
        _ => {
            let nbe = id / 32;
            let mut qb = vec![0u8; nt * nbe * Q8B];
            crate::quants::quantize_row_q8_0_buf(x, nt, id, &mut qb);
            cpu_quant_matmul(w, &qb, out, od, id, nt)
        }
    }
}
```
<!-- /code -->

The match is on the **weight's** dtype, and it decides the *activation* format as well as the kernel: the K-quants quantize activations to Q8_K (256-element blocks carrying precomputed block sums, so the dots never re-reduce the activations), and the simple types keep Q8_0. That is the contract - a dtype pair in, a kernel out - and it has a second half that is easy to miss: choosing the weight format silently chooses the activation format too, and the two must agree block for block or every offset after the first is wrong.

### Lesson 5 · A scalar reference you can trust: slow, obvious, and the oracle for every kernel after it

Every number in the rest of this book - every tok/s, every "the kernel got 2.4x faster" - is a claim that some fast path agrees with this one. That makes the scalar path the single most load-bearing code in the engine, and it needs three properties, only one of which is about being slow.

**It must be obvious.** The version in Lesson 3 is 15 lines with no intrinsics, no unsafe pointer arithmetic, and no knowledge of how weights are laid out beyond the block format. Anyone can audit it against Chapter 4's decoder. If the reference needs a comment to explain what it computes, it is not a reference.

**It must be kept, not deleted.** Keep it in the repository behind a test-only feature for the life of the project. The moment the fast paths become the only implementation, they become their own oracle - and a shared misreading of the block layout (Chapter 5 had one, in the second half of a super-block) then passes every test.

**It must be measured, so you know what it costs.** The oracle is not only a correctness device; it is the calibration point for "how much did SIMD buy". Scalar, on this machine:

<figure class="book-figure">{{#include ../figures/06-bandwidth-headroom.svg}}</figure>

*Figure 6.3 - 8,192 rows of 896 Q4_0 weights, 504 bytes each, against the machine's measured streaming bandwidth. The best scalar kernel manages 11.73 GB/s where the memory system delivers 198, so the kernel is 17x short of bandwidth-bound - and the dequantize-then-dot path is 152x short, which is the same sentence said about the wrong implementation.*

<!-- facts: dotbench.dot-headroom -->
| Quantity | Value | Where it comes from |
|---|---|---|
| Rows benchmarked | 8,192 rows of 896 weights | this artifact |
| Weight bytes per row | 504 B (28 blocks x 18 B) | the format, not a measurement |
| Best scalar throughput | 11.73 GB/s | the `i16` kernel below |
| Dequantize-then-dot throughput | 1.30 GB/s | the `dq` kernel below |
| Measured memory bandwidth | 198 GB/s | Chapter 1, same machine |
| Headroom | 17x short of bandwidth-bound | the two rows above |
<!-- /facts -->

Which turns into an honest prediction before Chapter 7 writes a single intrinsic. If your engine ran this scalar kernel over every weight of a token:

<div class="math-display">
$$
\begin{aligned}
\text{0.5B model, } 422.8\ \text{MB per token}: &\quad \frac{422.8\times10^{6}}{11.73\times10^{9}}\ \text{s} = 36.0\ \text{ms} \;\Rightarrow\; 27.7\ \text{tokens/s} \\[4pt]
\text{7B model, } 4.68\ \text{GB per token}: &\quad \frac{4.68\times10^{9}}{11.73\times10^{9}}\ \text{s} = 399\ \text{ms} \;\Rightarrow\; 2.5\ \text{tokens/s}
\end{aligned}
$$
</div>

Those are the numbers a scalar engine gets, and they are the reason the next two chapters exist: **9x** from using the lanes (Chapter 7) and roughly **8x** more from using the cores (Chapter 8) is the difference between 2.5 tok/s and 150 tok/s. Nothing in that chain changes the answer to within 7.7e-04 of the norm product - which is the contract this chapter spent its words on.

The oracle itself - the most obvious code in the engine, and the definition every fast kernel is checked against:

<!-- code: src/quants/dot_q4_0.rs::dot_q4_0_q8_0_scalar@88ffa5d -->
```rust
fn dot_q4_0_q8_0_scalar(x: &[u8], y: &[u8], nb: usize) -> f32 {
    let mut s = 0.0f32;
    for ib in 0..nb {
        let xb = &x[ib * Q4B..];
        let yb = &y[ib * Q8B..];
        let dx = block::fp16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
        let dy = block::fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
        let mut si = 0i32;
        for j in 0..16 {
            let v0 = (xb[2 + j] & 0x0F) as i8 - 8;
            let v1 = (xb[2 + j] >> 4) as i8 - 8;
            si += (v0 as i32) * (yb[2 + j] as i8 as i32);
            si += (v1 as i32) * (yb[2 + j + 16] as i8 as i32);
        }
        s += si as f32 * dx * dy;
    }
    s
}
```
<!-- /code -->

The second nibble's partner is **sixteen positions later** (`yb[2 + j + 16]`), not the next byte: that is the interleaved layout Figure 4.1 drew, and the reason a scalar loop that reads `j + 1` produces plausible, wrong numbers.

## The contract

| Weight dtype | Activation dtype | Kernel family | Same result as dequantize-then-dot? | Your status |
|---|---|---|---|---|
| Q4_0 | Q8_0 | integer dot, i32 partial, one f32 multiply per block | yes, 7.7e-04 of the norm product; integer arithmetic contributes 0 | measured here |
| Q8_0 | Q8_0 | the same shape, 127 x 127 magnitudes (i16 illegal) | yes, exactly | measured here |
| Q4_K | Q8_K | 256-wide super-block; activation carries per-16 i8 scales and an f32 d | by construction, unmeasured here | not measured |
| Q4_K | f32 | dequantize, then f32 fma | exact by definition | trivial |
| f32 | f32 | f32 fma | this is the definition | measured to 2.2e-07 vs f64 |

The middle column is the one to fill in for your own engine, and the honest entries are the ones that say *which* comparison was made. "Same result" is not a yes/no; it is a number with a norm under it.

## Project

**Build step S6 · one row, then one matrix.** Build the scalar path first and use it as the oracle for everything else. Every later speed number in this book is a claim that the fast path agrees with this one.

Order of work:

1. **Quantize one activation row to Q8_0 and decode it back.** You should see about half a per cent relative RMS. If you see zero, you quantized something that was already quantized (Lesson 2's second row) and you have learned nothing.
2. **Write the scalar integer dot** for `(Q4_0, Q8_0)` and compare it against dequantize-then-dot in f64. Expect agreement to within the activation's rounding and *no* further - if your integer path disagrees by 1e-3 relative, you have a nibble-order or scale bug, not a rounding difference.
3. **Write the dequantize-then-dot path too, and keep it.** It is the second witness: two implementations of the same contract, one of them obviously correct.
4. **Prove the accumulator bound for the formats you support**, on paper, before you pick i16 anywhere. 32 x |q|max x 127 against 32,767. Then, if you do use i16, add a test that the *bound* holds, not just that a run passed.
5. **Build the dispatch table with a refusal.** Map `(weight, activation)` to a function; return an error for anything not in the table, naming both dtypes, and assert on that error in a test. Do this before you have more than two pairs, because a table with a hole is easy to write and impossible to notice.
6. **Measure your scalar kernel** on a real weight matrix (not synthetic data), median of at least five runs, and write the number down next to the machine's bandwidth. You are looking for the ratio: if your scalar path is within 2x of bandwidth-bound, you have a different machine or a different problem than this book's.

## Deliverable

- A scalar quantized dot product with a reference cross-check test, and the reference kept in-tree.
- A dispatch table in code (dtype pair to function) with an explicit error for unsupported pairs, carrying both dtypes.
- A measured error bound for Q8_0 activations on one real row, stated as a number with its norm under it - for this book, 7.7e-04 of the norm product, with the activation's own rounding at 0.53 per cent relative RMS.
- The accumulator bound for each supported weight format: weights per block, largest magnitude, worst-case partial sum, and whether 32,767 covers it.

## How you know you passed

- Your kernel and dequantize-then-dot agree within the bound you wrote down, the bound is justified, and it names the *source* of the difference - the activation's rounding, not the arithmetic.
- Your integer path's difference against the same quantized activations is zero (or f32-rounding small), and you can explain why it has to be.
- An unsupported dtype pair produces an error naming the pair, and no test accepts a fallback.
- You can point at the exact line where the scale correction happens, and say how many float multiplies per 896 weights it costs: 28, not 896.
- You can say whether a 16-bit block accumulator is legal for each format you support, from the format's bounds, and what would have to be true for it to be legal.

## Reading

- code: `src/kernel.rs`, `src/kernel/dispatch.rs`, `src/quants.rs` at the book's [pinned baseline](../appendix/baseline.md); the two vector shapes are `src/quants/dot_q4_0.rs::dot_q4_0_q8_0_avx2` and `src/quants/neon.rs::dot16`
- docs: `docs/inference_e2e_walkthrough/10-cpu-matmul-kernels.md`
- decision: ADR-0013 (the CPU quantizes activations to Q8_0; a device reads f32)
- [ggml-quant] The ggml quantized block layouts and the k-quant design notes, in the ggml repository - no arXiv ID (specification) - the block formats this kernel consumes, and the reference scalar implementations its vector paths are checked against

## Backfill

- [x] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - **answered, with a caveat that matters.** For scalar code this operation is compute-bound by a factor of 17: 11.73 GB/s of weight traffic against 198 GB/s of measured bandwidth. The telling is the ratio, not the absolute number. But the answer is about *this kernel on this machine*: once Chapter 7 uses the lanes, the same op moves 4-8x more bytes per second, and the crossover to bandwidth-bound is exactly where the next chapter starts being about memory instead of arithmetic.
- [x] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - completed across Chapters 4-6. The weights are read once per token and multiplied once, so bytes-per-weight is the cost of a token; 4.5 bits against 16 is 3.5x fewer bytes and 7.98 per cent per-weight error, which arrives at the dot product as 7.7e-04 of the norm product.
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - **started.** The dispatch table is the first place the boundary appears, and this chapter's refusal rule is the ancestor of Chapter 16's device split: an unsupported pair must fail loudly, and a "fallback" that computes a different function is worse than an error.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - both halves now have instruments. Correctness: three implementations that must agree, with the integer-vs-same-activations difference required to be zero and the activation's rounding named as the only error source. Speed: three scalar kernels timed on real weight rows, median of seven runs, with the run range printed so a noisy laptop shows up as noise rather than as a result.
