# Chapter 7 · SIMD Dispatch in Anger

> - **Compute tier**: ✅ any laptop (this chapter is CPU only; no GPU needed)
> - **Status**: written - four kernels across two machines (Apple M4 Pro with NEON `SDOT`, Intel i7-11700B with AVX2 and AVX-512 VNNI), one source file, one data set, every kernel checked against the scalar oracle before it was timed; what is *not* here is threads or cache blocking, which are Chapter 8
> - **Build step S7**: AVX2 and NEON versions of your dot product, selected at compile time, each checked against the scalar oracle.
> - **Rule of this book**: This chapter is about making one operation fast on one core. Threads, cache blocking and the matrix-vector versus matrix-matrix decision are Chapter 8.

## First, the gist

A modern CPU can multiply eight or sixteen pairs of integers per instruction if you ask it to, and the compilers will not do it for you reliably in a kernel this branchy. **SIMD** (single instruction, multiple data) is the technique: load a vector of weights and a vector of activations, multiply and accumulate eight lanes at a time, then fold the lanes together. Two instruction sets matter here - **AVX2** on x86 and **NEON** with its `SDOT` integer dot-product extension on ARM - and the interesting part is not writing either one, it is keeping them honest against the scalar oracle and knowing which quantized formats each one can actually accelerate.

Four kernels, two machines, one data set:

<!-- facts: dotsimd.simd-table -->
| Machine | Kernel | ns per row | Weight throughput | vs scalar | Max difference, over the norm product | Requires |
|---|---|---|---|---|---|---|
| Apple M4 Pro (arm64) | `scalar` | 97.8 | 5.15 GB/s | 1.00x | 0.0e+00 | always |
| Apple M4 Pro (arm64) | `neon-sdot` | 25.9 | 19.48 GB/s | 3.78x | 0.0e+00 | armv8.2 dotprod |
| Apple M4 Pro (arm64) | `neon-acc` | 22.6 | 22.32 GB/s | 4.33x | 1.1e-07 | armv8.2 dotprod, no per-block reduce |
| Intel i7-11700B (x86_64) | `scalar` | 265.8 | 1.90 GB/s | 1.00x | 0.0e+00 | always |
| Intel i7-11700B (x86_64) | `avx2` | 43.6 | 11.56 GB/s | 6.10x | 1.2e-07 | avx2+fma |
| Intel i7-11700B (x86_64) | `avx512-vnni` | 58.1 | 8.67 GB/s | 4.57x | 1.2e-07 | avx512f+vnni+bw+vl |
<!-- /facts -->

Three things in that table are worth more than the headline speedups.

**The vector width was not the biggest lever.** The first version of the ARM kernel measured 48.1 ns per row; the final one measures 25.9. Nothing about the multiply changed between them - the difference was that the f16 *scale* was being decoded by a branchy software routine instead of one hardware conversion instruction. Lesson 3 takes that apart, because it is the most common way a hand-written kernel ends up slower than the scalar code it replaced.

**Wider lost.** AVX-512 VNNI is the x86 answer to `SDOT`, two blocks per instruction, and on this CPU it is **4.57x** against AVX2's **6.10x**. Assuming "512 beats 256" would have made this kernel slower. Lesson 4 is about a dispatch table that can only be built from measurements.

**Agreement is not equality.** Only one kernel in that table is bit-identical to the scalar path, and it is the one that keeps the scalar's accumulation order. The others differ by about 1e-07 of the norm product, which is not an error to fix - it is what f32 addition does when you reorder it. Lesson 5 is about knowing which of the two comparisons you are entitled to.

## What you'll be able to do

- Write an AVX2 and a NEON kernel for one quantized format and dispatch between them at compile time.
- Explain why ARM's `SDOT` changes what is worth vectorizing compared with x86, and which formats have no fast path on AVX2.
- Say why a vector kernel can be *slower* than the scalar code it replaced, and find the cause with a measurement rather than a guess.
- Measure the speedup you actually got, on your own machine, with the test that proves it did not change the answer - and know which equality that test can and cannot claim.

## Lessons

- [ ] 1. Lanes, masks and horizontal sums: SIMD without intrinsics anxiety
- [ ] 2. AVX2 in practice: nibble unpacking, FMA, and the formats that fit
- [ ] 3. NEON and SDOT: the integer dot product ARM gives you for free
- [ ] 4. Dispatch: compile-time selection, the fallback that must stay correct, and why the widest ISA may lose
- [ ] 5. Proving a fast path: the identity test, the tolerance class, then the timer (payoff: a fast wrong answer is the most expensive bug in this book)

### Lesson 1 · Lanes, masks and horizontal sums: SIMD without intrinsics anxiety

<figure class="book-figure">{{#include ../figures/07-lanes.svg}}</figure>

*Figure 7.1 - the same 32 weights in two register shapes: AVX2's pair sums in 16-bit lanes, and NEON's four SDOT lanes.*

Three ideas cover almost every kernel in this book, and none of them is about which instruction set you are on.

**A register is a row of numbers, not a number.** A 256-bit AVX2 register holds 32 signed bytes, or 16 16-bit values, or 8 floats - the same 32 bytes of hardware, reinterpreted. Chapter 6's block is 18 bytes: 2 for the scale and 16 holding 32 nibbles. The payload fits in half a register, which is why one load plus one shift gets you a whole block's weights.

**Masks are unpacking.** Q4_0's nibbles are not a type; they are four bits that mean "this weight plus eight". The unpack is two operations on a vector of bytes:

```text
b      = byte j                  (16 bytes: two weights each)
low    = (b & 0x0F) - 8          weights j      (j = 0..15)
high   = (b >> 4)   - 8          weights j + 16
```

On a scalar core that is 32 shifts and 32 subtracts. On a vector core it is one AND and one shift for sixteen bytes at a time - and this is the part every implementation does the same way.

**Horizontal sums are the expensive part.** Multiplying lanes is cheap; adding five or eight lanes together at the end is a chain of shuffles, and doing it *per block* means paying that chain 28 times per row. The right shape keeps a vector accumulator and folds once:

```text
per block:   acc_vec += scale * (vector of i32 partials)     (no horizontal work)
at the end:  sum(acc_vec)                                    (one fold, 28 blocks later)
```

The measurement that proves this matters is in the table at the top: `neon-sdot` folds per block and gets 3.78x; `neon-acc` keeps a float vector and gets 4.33x. Same instructions, same 32 weights per iteration, 14 per cent apart, and the difference is one horizontal add that moved out of the loop.

### Lesson 2 · AVX2 in practice: nibble unpacking, FMA, and the formats that fit

AVX2's integer dot product for our shape is two instructions wide, plus the unpack:

```rust
let raw = _mm_loadu_si128(w.add(2) as *const __m128i);        // 16 bytes of nibbles
let bytes = _mm256_set_m128i(_mm_srli_epi16(raw, 4), raw);    // [low nibbles | high nibbles]
let qw = _mm256_sub_epi8(_mm256_and_si256(bytes, mask15), bias8);
let qa = _mm256_loadu_si256(a.add(2) as *const __m256i);      // 32 int8 activations
let uw = _mm256_sign_epi8(qw, qw);                            // |q_w|: unsigned, for maddubs
let sa = _mm256_sign_epi8(qa, qw);                            // q_a with q_w's sign folded in
let p16 = _mm256_maddubs_epi16(uw, sa);                       // 16 pair sums, i16
let p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));       // widen, i32
acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32), acc);
```

Four things to notice, in order of how much trouble they cause.

**`maddubs` is unsigned by signed.** (u8 x i8, then add adjacent pairs.) Our weights are signed - nibbles biased by eight - so the sign has to go somewhere. The trick is to move it onto the *activation*: take the absolute value of the weight (`sign_epi8(qw, qw)` gives `|q_w|`), which makes it a legal unsigned operand, and apply the same sign to `qa`. The product of the pair is unchanged. This is the single most-transcribed-wrong sequence in quantized kernels: get the two `sign_epi8` arguments backwards and every product comes out negated in half the lanes, which looks like a scale bug and is not.

**The nibble halves are not interleaved.** `set_m128i(high, low)` puts the 16 low nibbles in the low half and the 16 high nibbles in the high half, so lane `i` is weight `i` and lane `16 + i` is weight `16 + i` - exactly the layout Chapter 4 decoded scalar. The activations are loaded in order 0..31 and line up without shuffling, but only because of that layout. Interleave the nibbles instead and every dot product is a permutation.

**`madd` with ones is the widening step.** `_mm256_madd_epi16(p16, ones)` adds adjacent i16 pairs into i32. It is also where the 16-bit headroom question from Chapter 6 gets settled in practice: each i16 lane holds at most 2 x 8 x 127 = 2,032, so no saturation is possible for Q4_0, and the i32 lanes are exact.

**The scale multiplies the vector, not the block.** `fmadd` adds `d * partials` into the float accumulator, one instruction for eight lanes. 28 blocks, 28 fmadds, no horizontal work until the end.

AVX2's coverage is wider than its reputation: `maddubs` covers Q4_0 (after the sign trick), Q5_0 (add the high-bit plane before maddubs), and Q8_0 (signed bytes hit the same trick). The K-quants need the 6-bit scale plane unpacked into vectors first, which is more shuffling but not a different idea - and this chapter does not implement it, so the coverage table says "not attempted" rather than "impossible".

**One more thing about AVX2 that matters for the dispatch**: `_mm256_maddubs_epi16` and `_mm256_fmadd_ps` need `avx2` and `fma`. On a CPU without FMA the same code cannot be run at all, which is why the function carries `#[target_feature(enable = "avx2,fma")]` and the call site must be guarded by runtime detection.

### Lesson 3 · NEON and SDOT: the integer dot product ARM gives you for free

NEON's answer is one instruction, and it is a better fit for this problem than x86's:

```rust
let bytes = vld1q_u8(wb.add(2));                                    // 16 nibble bytes
let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, m4)), s8);    // weights 0..15
let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), s8);   // weights 16..31
let a0 = vld1q_s8(ab.add(2) as *const i8);
let a1 = vld1q_s8(ab.add(18) as *const i8);
let acc = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, a0), hi, a1);     // 32 weights, 4 i32 lanes
acc_f = vfmaq_n_f32(acc_f, vcvtq_f32_s32(acc), d);                  // no horizontal add
```

`SDOT` is **signed by signed**, so the abs/sign dance of Lesson 2 does not exist here. It also folds four products into each i32 lane directly, so there is no widening step either: two instructions reduce 32 weights to four i32 lanes, where AVX2 needs two `maddubs`, one `madd`, and the sign fix-up around them.

Two measurements turn this into a lesson rather than a claim.

**The kernel that was slower than scalar.** The first version of this kernel measured **48.1 ns per row** - barely better than the scalar path's 91, and *worse* than Chapter 6's best scalar kernel at 41.9. The cause was not the vector code: it was the f16 scale decoder, a branchy software routine called twice per block, for 56 calls per row. Replacing it with the hardware conversion - `(float)(__fp16)bits` on ARM, `_cvtsh_ss` on x86 - took the same kernel to **25.9 ns per row**, a factor of 1.85, without touching a single multiply. On this data the scale decode was costing more than the arithmetic it fed.

```text
the same SDOT kernel, two versions of one helper:
  fp16 decoded in software      48.1 ns/row
  fp16 decoded by fcvt          25.9 ns/row     1.85x, from one instruction
```

**The horizontal add that moved out of the loop.** With the fold kept inside the block loop, the kernel gets 3.78x over scalar; with a vector accumulator and one fold at the end, 4.33x. Both are correct; the second one is what the reference engines do, and the reason is Lesson 1's third point.

**Which formats NEON covers.** Q4_0, Q5_0 and Q8_0 are all `SDOT`-shaped, and the K-quants are the awkward case for a different reason than on x86: their 256-weight super-block gives eight times more weights per iteration, which is *good* for a kernel that pays per-block overhead - but the 6-bit scale plane has to be unpacked first. This chapter implements Q4_0 only, and the coverage table says so.

### Lesson 4 · Dispatch: compile-time selection, the fallback that must stay correct, and why the widest ISA may lose

A dispatcher has three jobs and the third one is the one that gets skipped: pick a kernel, prove which one it picked, and refuse the pairs it cannot serve.

**Compile-time selection is not enough.** `#[cfg(target_arch = "aarch64")]` gives you a binary that runs the NEON path on any ARM - including ARM chips without `dotprod`, where those instructions do not exist. The guard has to be the *feature*, not the architecture:

```rust
#[cfg(target_arch = "aarch64")]
if std::arch::is_aarch64_feature_detected!("dotprod") {
    return unsafe { dot_neon(q4, q8, nb) };      // SDOT
}
dot_scalar(q4, q8, nb)                            // the fallback, always compiled
```

On x86 the same shape uses `__builtin_cpu_supports("avx512vnni")` (or `is_x86_feature_detected!`), and the answer must come from the *runtime*, because the same binary will run on machines that differ.

**The fallback is not optional and not lazy.** It must be compiled in, always, and it must be the scalar path from Chapter 6 - not a "slower SIMD path" that shares the same bugs. The dispatch in this chapter's benchmark program picks the fastest available kernel and prints it, which is the smallest possible version of the requirement that a dispatch be *observable*:

```text
Apple M4 Pro:   dispatch would pick neon-acc
Intel 11700B:   dispatch would pick avx2
```

If your engine cannot answer "which kernel just ran, on this machine, for this dtype pair", you cannot interpret any speed number you measure - and on a machine whose fastest kernel is the *narrower* one, you also cannot explain it.

**Why the narrower one wins.** On the i7-11700B, AVX-512 VNNI is the theoretically better instruction: `vpdpbusd` does what `SDOT` does, on 64 bytes at a time. Measured, it is **4.57x** against AVX2's **6.10x**. Two reasons, both structural:

- The VNNI version needs more non-multiply work per iteration: 64 bytes have to be assembled from two 18-byte blocks (an insert each side), and because AVX-512 has no `sign_epi8` the sign of the weight is applied to the activation with a mask extract, a negate and a blend. At 32 weights per block that prologue is as long as the multiply it feeds.
- Sustained 512-bit code can also cost clock, which this book did not measure here - one attempt to read `scaling_cur_freq` on the NUC while the kernel ran produced mostly idle samples, and a number produced that way would have been worse than none. The code-level argument is enough to explain the result, so it is the one on record.

That last point is the general lesson and it is uncomfortable: **at 32 weights per block you are not vectorising a big loop, you are vectorising a small loop with a large prologue.** Which is exactly why the formats in Chapter 5 moved to 256-weight super-blocks, and why Chapter 8's block-over-rows layout is the next real win.

### Lesson 5 · Proving a fast path: the identity test, the tolerance class, then the timer

The order is not negotiable: identity first, timer second. A fast wrong answer is the most expensive bug in this book because it is *invisible* - the model still produces text, the accuracy tests still pass, and the only symptom is that the output is different from the reference in a way nobody is measuring.

But the identity test has a subtlety that Chapter 6 did not have to face, and the table at the top of this chapter is the evidence:

| Kernel | Max difference from the scalar path, over the norm product | Why |
|---|---|---|
| `neon-sdot` | `0.0e+00` | it accumulates one float per block, in the same order as the scalar loop |
| `neon-acc` | `1.1e-07` | the four float lanes are summed at the end, so the additions are associated differently |
| `avx2` | `1.2e-07` | eight lanes, per block, reduced once |
| `avx512-vnni` | `1.2e-07` | the same, with two blocks per instruction |

So "the fast path must equal the scalar path" is false as stated, and a test that demands bitwise equality will fail on a *correct* kernel. The two honest tests are:

1. **Bitwise, when the kernel preserves the accumulation order.** This is the strongest claim available, and it is worth preserving deliberately: it is why the first NEON version (fold per block, float accumulator, same order as scalar) reads `0.0e+00` and is the one to trust under refactoring.
2. **Bounded, when it does not**, with the bound stated over the norm product rather than as a relative error: `1e-07` of `||w||*||a||` is f32 rounding, `1e-03` would be a bug. Chapter 6 measured the *quantization* error at 7.7e-04 of the same scale; a reordering difference three orders of magnitude below it is not a correctness question.

**And the timer has its own trap, which this chapter walked into.** The scalar kernel in this chapter's benchmark program measures 97.8 ns per row. The *same integer kernel* in Chapter 6's program measures 69.9. Same machine, same bytes, same source shape, 40 per cent apart:

```text
Chapter 6's harness, integer i32 kernel     69.9 ns/row
Chapter 7's harness, scalar kernel          97.8 ns/row
```

That is not noise, and no amount of repetition makes it go away; it is what happens when the code around the loop changes - alignment, inlining, what else is in the binary. The rule that follows is the one this book keeps relearning: **a speedup is a property of a harness, not of a kernel.** Quote the command, quote the baseline, and never divide a Chapter 7 number by a Chapter 6 number.

Which is why every speedup in this chapter is computed inside one program, against one baseline, on one dataset, with the run range printed beside it.

## Coverage by instruction set

`implemented` means this chapter's source has the kernel and the table at the top has its measurement. `shape fits` means the instruction exists for it and the layout permits it, but this book did not write it - and saying so is cheaper than implying it.

| Format | Scalar | AVX2 + FMA | NEON + SDOT | Measured speedup (this machine) |
|---|---|---|---|---|
| Q4_0 | implemented | **implemented, 6.10x** (i7-11700B) | **implemented, 4.33x** (M4 Pro) | AVX2 43.6 ns/row, NEON 22.6 ns/row |
| Q5_0 | implemented | shape fits: add the 4-byte high-bit plane before `maddubs` | shape fits: `SDOT` on the fifth bit's plane | not attempted in this chapter |
| Q8_0 | implemented | shape fits: same abs/sign trick, i8 both sides | shape fits: `SDOT` signed by signed, no trick at all | not attempted in this chapter |
| Q4_K | implemented | needs the 6-bit scale plane unpacked into vectors (`vpermb`, `vpsrlvd`) | needs the 6-bit plane and the 256-weight loop structure | not attempted: Chapter 5's super-block, Chapter 8's shape |
| Q6_K | implemented | 2-bit high plane plus `i8` scales: two shuffles per 64 weights | same, with `SDOT` on the low nibbles | not attempted |

## Project

**Build step S7 · the dispatch you will regret skipping.** Write one format's kernel by hand, get it identical to the oracle, and only then look at how the reference engine organizes the other seven. The temptation is to copy eight kernels at once, and it produces eight untested kernels.

Keep every kernel behind the same signature: the caller should never branch on instruction set.

Order of work:

1. **Build the harness before the kernel.** A program that loads real blocks from a real file, runs the scalar path over N rows, times it, and prints the number. You cannot tell an improvement from noise without a baseline you trust.
2. **Write the AVX2 (or NEON) kernel for Q4_0**, and assert it against the scalar path over *every* row before you time it. Expect either exact agreement (if you keep the scalar's order) or ~1e-07 of the norm product, and decide which one your kernel claims.
3. **Time it. If it is not at least 2x, the problem is probably not the vector width.** Check, in this order: is the f16 scale decoded in software, is there a horizontal sum inside the block loop, is the loop bound recomputed per iteration.
4. **Add the second instruction set** if you have the second machine - this chapter's AVX-512 result is the reason to measure rather than assume. If you only have one machine, write the dispatch anyway and make it print what it picked.
5. **Build the dispatch table** with `(weight dtype, activation dtype)` keys, runtime detection, and a scalar fallback that is always compiled. Print the chosen kernel on demand, and make an unsupported pair an error naming both dtypes.
6. **Break it on purpose**: force the dispatch to choose the scalar path and confirm the numbers move; corrupt one lane's worth of nibbles and confirm the identity test fails. A test that has never failed is a test you have not run.

## Deliverable

- At least one hand-written SIMD kernel per instruction set, checked against the scalar oracle, with the tolerance class stated (bitwise or bounded).
- A runtime dispatch (feature-detected, not architecture-assumed) with a scalar fallback that is always compiled, and a way to ask which kernel ran.
- A measured single-core speedup table for your machine, with the command quoted, the baseline's number beside it, and the run range.
- One sentence on the vector-width result you did *not* get: what you tried, and what it cost.

## How you know you passed

- Every SIMD kernel passes the same identity test as the scalar path, on the same fixture, with the comparison class named - and the kernel that can claim bitwise equality is marked as such.
- The measured speedup is quoted with the test conditions (block size, dataset size, harness) and a run range, and is reproducible by the command in your notes.
- Removing the SIMD path does not change results: the fallback is genuinely equivalent, and you have run the engine with it forced.
- You can say which kernel the dispatch picks on your machine, and why - including a case where the wider instruction set lost.

## Reading

- code: `src/quants/avx2.rs`, `src/quants/avx2_correctness.rs`, `src/quants/neon.rs` (see `dot16` and `sdot_vec`), `src/quants/neon_correctness.rs` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/CPU_OPTIMIZATIONS.md`
- [arm-sdot] *Arm Architecture Reference Manual* and the ACLE integer dot-product instructions (`SDOT`, `SMMLA`) - no arXiv ID (vendor manual) - the instruction this chapter's NEON kernel uses, and the four-lane-by-four-product semantics Figure 7.1 draws

## Backfill

- [x] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - **answered on two machines, and the answer changed with the kernel.** Chapter 6 measured the scalar path at 11.73 GB/s against 198 GB/s of bandwidth and called it compute-bound by 17x. With the best vector kernel on the M4 Pro it is 22.3 GB/s, so the gap is 8.9x; on the i7-11700B it is 11.6 GB/s against 45.1 GB/s, a gap of 3.9x. Still compute-bound, both times - and the *second* lesson is that the number moves by 2x when the kernel changes and by 40 per cent when only the harness changes, so "where does the time go" has to be asked of a named kernel in a named harness.
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - nothing new here; the vector kernels consume the same bytes. What this chapter adds is the cost side of the formats in a different currency: 32 weights per block is too small a loop to hide a prologue behind, which is the measured argument for Chapter 5's 256-weight super-blocks.
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - **the seam becomes a table.** `(weight dtype, activation dtype)` in, kernel out, error if the pair is unknown, and a scalar fallback that is always compiled. Chapter 16 will widen this same table to include devices; the refusal rule is already here.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - the two halves are now permanently coupled and both have a stated class. Correctness: bitwise where the accumulation order is preserved, bounded over `||w||*||a||` where it is not - because demanding bitwise equality from a reordered f32 sum fails on correct kernels. Speed: same harness, same baseline, median of independent runs, range printed. And the 40 per cent harness effect measured here is why speedups do not travel between chapters.
