# Chapter 10 · Attention and the KV Cache by Hand

> - **Compute tier**: ✅ any laptop
> - **Status**: written - the score magnitudes, the mask's cost, one decode step at four prefix lengths and the cache against a rebuild are measured on this model's own weights by `scripts/attn.c`; what is *not* here is any device kernel (Chapter 25) or the storage subsystem (Chapter 18)
> - **Build step S10**: Causal attention with grouped queries over an explicitly managed KV cache: prefill writes, decode reads, and the results match a naive recompute.
> - **Rule of this book**: This chapter establishes the semantics: what attention computes and how the cache makes it cheap. The storage subsystem (cells, spans, defragmentation) is Chapter 18, and the device kernels are Chapters 21 and 25.

## First, the gist

Attention lets every token look at the tokens before it and mix their values into its own representation. Naively that means recomputing the keys and values of the whole prefix at every step, which is quadratic work per token. The **KV cache** (key-value cache) removes that: keys and values are computed once, stored, and read back. **GQA** (grouped-query attention) then reuses one key/value head across several query heads, which is what makes the cache small enough to hold. The whole chapter is one idea applied twice: prefill fills the cache, decode reads it, and both must agree exactly.

The sizes are worth having in front of you before the definitions, because they are what the rest of the book optimises. On this model, at a 1,024-token prefix, one layer's attention for one token - fourteen heads, no SIMD, f32 - takes **212 microseconds**, and **11 per cent** of that is the softmax; the rest is two matrix multiplies. Rebuilding the keys and values for that prefix instead of reading them takes **105 milliseconds**, which is 500 times the whole attention step. A four-thousand-token context costs **96 MiB** of cache across the model's 24 layers, and at eight thousand it is **192 MiB**, two fifths of the weights - because a cell is a fixed number of f32 words, so storing f16 values does not make the cache smaller, only cheaper to read. Packed to q8_0 the same four-thousand-token cache is 25.5 MiB.

## What you'll be able to do

- Write attention that matches a naive re-computation on the same inputs, at prefill and at decode.
- Lay out a KV cache for grouped-query attention and compute its size for a real model before allocating it.
- Explain why positions must be data passed to the kernel rather than part of a graph's structure, and what breaks when they are not.
- Say which mask bug your test would catch, and which one it would not.

## Lessons

- [ ] 1. Queries, keys, values: attention as three matmuls and a mask
- [ ] 2. Causal masking at prefill, and the off-by-one that survives every test that does not compare logits
- [ ] 3. Grouped-query attention: head reuse, cache size, and the arithmetic that decides it
- [ ] 4. Prefill attention versus decode attention: one big masked matmul against a streamed reduction
- [ ] 5. The cache as data: positions, spans, and the layout your kernels will assume

### Lesson 1 · Queries, keys, values: attention as three matmuls and a mask

Three learned projections, then two multiplies and a softmax. For one head at one position:

<div class="math-display">
$$
q = W_q x,\quad k_j = W_k x_j,\quad v_j = W_v x_j
\qquad
s_j = \frac{q \cdot k_j}{\sqrt{d}}
\qquad
p = \operatorname{softmax}(s)
\qquad
o = \sum_j p_j\, v_j
$$
</div>

$d$ is the head dimension - 64 here - and the scale is not decoration: a dot of $d$ independent products has a standard deviation that grows with $\sqrt{d}$, so without the division the scores would widen as the head widens and the softmax would saturate.

<figure class="book-figure">{{#include ../figures/10-attention-shapes.svg}}</figure>

*Figure 10.1 - one head's decode step. The query is one row of 64; the keys and values are the whole cached prefix, 1,024 rows each; the output is one row of 64. The half of the score matrix that a causal model never computes is drawn as the half it is.*

The function that computes it is one signature, and the signature is the shape contract:

<!-- code: src/graph/cpu_backend.rs::cpu_gqa_attn_runs@e844591 -->
```rust
/// GQA attention over each query's **list of `(cell, len)` runs** (C8b S2), with
/// `off[t]..off[t + 1]` naming query `t`'s runs — the union of the runs is the
/// query's window, in order.
pub(crate) fn cpu_gqa_attn_runs(
    q: &[f32],
    ka: &[f32],
    va: &[f32],
    runs: &[(usize, usize)],
    off: &[usize],
    nt: usize,
    nh: usize,
    nk: usize,
    hd: usize,
    hd_kv: usize,
    nkt: usize,
    out: &mut [f32],
    scale: f32,
) -> Result<(), String> {
    if hd < hd_kv {
```
<!-- /code -->

Seven parameters are geometry and the rest is data. Read them against this model: `nt` is how many queries this call carries (1,024 at prefill, 1 for a decode step), `nh` is 14, `nk` is 2, `hd` is 64 (one query head), `hd_kv` is 64 (one cached head), `nkt` is 128 (both cached heads - the width of one cache row), and `scale` is $1/8$, the $1/\sqrt{64}$ from the formula. There is no `Vec<Vec<..>>` and no shape struct: the strides are implied by those numbers, which is why the same function serves a batch of one and a batch of a thousand.

Then the body, reduced to the three steps of the formula - and note that all three are Chapter 9's operators:

<!-- code: src/graph/cpu_backend.rs::attn_heads@e844591 -->
```rust
                let ks = kv * c.nkt + hk * c.hd_kv;
                let s = crate::vec_ops::vec_dot_f32(
                    c.hd_kv,
                    std::slice::from_raw_parts(c.q.add(qs), c.hd_kv),
                    std::slice::from_raw_parts(c.ka.add(ks), c.hd_kv),
                ) * c.scale;
                scrs[i] = s;
                i += 1;
                if s > mx {
                    mx = s;
                }
// ...
        let sm = crate::vec_ops::vec_soft_max_inplace_f32(vl, &mut scrs, mx);
        let is = (1.0 / sm) as f32;
        crate::vec_ops::vec_scale_f32(vl, &mut scrs, is);
// ...
                crate::vec_ops::vec_muladd_f32(
                    c.hd_kv,
                    std::slice::from_raw_parts_mut(c.out.add(os), c.hd_kv),
                    std::slice::from_raw_parts(c.va.add(kv * c.nkt + vs_base), c.hd_kv),
                    scrs[i],
                );
```
<!-- /code -->

`vec_dot_f32` is the score, `vec_soft_max_inplace_f32` is the exponent and its normalisation, `vec_muladd_f32` is the weighted sum. The elisions are the window walk, which lesson 4 comes back to. Nothing here is attention-specific arithmetic: the chapter's three matmuls are three vector operations the book already measured, called in the right order.

The arithmetic on this model's own weights, over every head and every cached position at a 1,024-token prefix:

<!-- facts: attn.attention-scores -->
| Quantity | Value |
|---|---|
| (head, cached position) pairs | 14,336 - 14 heads over a 24-layer model's first layer, 1,024 positions |
| Mean absolute score | 0.437 |
| Root mean square | 0.656 |
| Largest absolute score, layer 0 | 4.67 |
| Pairs above the f16 exp bound (11.09), layer 0 | 0 |
| Pairs above the f32 exp bound (88.72), layer 0 | 0 |
| Largest absolute score, all 24 layers | **17.32**, at layer 16 |
| Pairs above the f16 exp bound, all layers | **127** of 344,064, in layers 13, 16, 17 |
| Pairs above the f32 exp bound, all layers | 0 |
<!-- /facts -->

Two things to read out of that table, and then a third that the first reading got wrong.

The scores at layer 0 are **small**: a mean magnitude of 0.44 and a maximum of 4.67, below the f16 `exp` bound
of 11.09 that Chapter 9 measured, with zero pairs over it. It is tempting to stop there and conclude that the
max subtraction is unnecessary for this model. The last three rows are why that conclusion is wrong: swept over
**all 24 layers**, the largest score is **17.32, at layer 16**, and **127 pairs cross the f16 bound**, in layers
13, 16 and 17 - 75 of them in layer 13 alone. Layer 0 is one of the *safest* layers in the model, not a typical
one, which is exactly the mistake a single-layer measurement invites.

So the bound does not belong to the data - it belongs to the format, as Chapter 9 said, and this model crosses
it in three of its twenty-four layers. Without the max subtraction those 127 exponents overflow to `inf` and
propagate into every layer above them. The subtraction costs one pass and one register, and this is the
measurement that says the pass is not optional: the same weights that look harmless at the front of the network
are not harmless in the middle of it.

What the numbers *do* say is where the time goes. The softmax over a 1,024-wide row, measured alone, is 24 microseconds for fourteen heads; the step it belongs to is 212. So the exponent is **11 per cent** of a decode step, and the two matmuls are the other 89 per cent. Compare that with Chapter 9, where the softmax was 95 per cent of the *pointwise* work at 4,096 tokens: both statements are true, because "the pointwise work" and "the step" are different denominators. The chapter that measures has to say which one it means.

### Lesson 2 · Causal masking at prefill, and the off-by-one that survives every test that does not compare logits

A decoder may not look forward. At prefill the whole prompt arrives at once, so the score matrix is $L \times L$ and the mask keeps the lower triangle: query $i$ may read key $j$ when $j \le i$. The arithmetic is exact - at a 1,024-token prefix that keeps 524,800 of 1,048,576 pairs, **50.05 per cent** - and it is the reason prefill does half the work a naive implementation would.

The mask itself is three lines in the engine, and they are test-only on purpose:

<!-- code: src/graph/cpu_backend.rs::causal_span@e844591 -->
```rust
/// The causal span for a single sequence, in the `attn_span` layout E1 defines:
/// token `t` may see the cells `[0, pos[t] + 1)`. Test/reference helper — the
/// model path resolves the span from the KV store's ownership instead.
// ...
#[cfg(test)]
pub(crate) fn causal_span(pos: &[usize]) -> Vec<(usize, usize)> {
    pos.iter().map(|&p| (0, p + 1)).collect()
}
```
<!-- /code -->

`(0, p + 1)` is the entire mask: a query at position `p` may read cells `0..p+1`. Write `p` instead of `p + 1` and you have this lesson's bug, in one character.

**Why is the definition only compiled in tests?** Not because it is missing - the opposite. The doc comment says it - *"Test/reference helper - the model path resolves the span from the KV store's ownership instead"* - and the line the elision dropped says the rest: *"Test-only (#238): ... `#[cfg(test)]` keeps it out of production builds."* This function exists so that tests can turn a list of positions into a causal window, and it is compiled **only** when the test harness is built; in a release binary it is not there at all. The limitation that keeps it out of production is the interesting part: it produces **one contiguous range per query**, and a sequence that shares a prefix reads several - the shared runs plus its own, up to four (`KV_MAP_MAX_SPANS`). The production path resolves the window from the cache's own ownership and can carry that list, so the simple form is not merely redundant, it is *wrong* for the case that matters. Keeping it behind `#[cfg(test)]` means the compiler, not a review comment, enforces that: an engine build cannot call the oracle by accident. Issue #238 did this deliberately, and it is a stronger version of the discipline this chapter asks for - the naive path is not discouraged, it is unrepresentable in the artifact you ship.

Two ways to get it wrong, and they are not equally expensive:

<!-- facts: attn.mask-effect -->
| Mask | Largest output change | Relative | Caught by a 1e-2 tolerance test? |
|---|---|---|---|
| Off by one (`j &lt; i` instead of `j &lt;= i`) | 0.000406 | 0.46 per cent | no |
| Missing entirely (no causal mask) | 0.019560 | 22.21 per cent | yes |

The query's own position carries **0.24 per cent** of the attention mass at a 512-token prefix, which is the size of what the off-by-one throws away: the largest output element is 0.088, and dropping a position that holds a quarter of a per cent of the weight moves it in the fourth decimal place.
<!-- /facts -->

Read those rows again, because they are the chapter's most useful measurement. **Dropping the mask entirely changes the output by 22 per cent** - enormous, caught by any test with a sane tolerance, caught by a golden-output comparison, often caught by watching the generated text. **An off-by-one changes it by 0.46 per cent** - inside the rounding of a `1e-2` relative tolerance, inside the noise of a different accumulation order, and invisible in a token sample, because a perturbation that small rarely moves an argmax.

The reason is in the last line: the query's own position carries **0.24 per cent** of the attention mass at a 512-token prefix. The off-by-one drops exactly that position and renormalises. Its effect is bounded by the weight it removes, which shrinks as the prefix grows - at 32,000 tokens the self weight is smaller still, and the bug gets *harder* to see the longer you run.

The production kernel writes this down where it counts. This is a comment in `attn_heads`, immediately above the softmax:

<!-- code: src/graph/cpu_backend.rs::attn_heads@e844591 -->
```rust
// Softmax and accumulate over the token's OWN window `[lo, hi)`, the
// one the span input names, rather than over a batch-wide range.
// Padding the reduction and softmaxing over it makes every
// reduction's *length* depend on how many tokens the batch holds,
// which makes a token's result depend on `nt` (measured: a 6-token
// and a 13-token batch diverge from layer 3 on). With the window a
// function of the token alone, incremental prefill and decode
// reproduce a single-shot prefill bitwise — and with one sequence
// the window is exactly `[0, pos + 1)`, which is what the old
// derivation produced.
let sm = crate::vec_ops::vec_soft_max_inplace_f32(vl, &mut scrs, mx);
let is = (1.0 / sm) as f32;
crate::vec_ops::vec_scale_f32(vl, &mut scrs, is);
```
<!-- /code -->

That comment records a measured failure of the same family, and a nastier one: if the reduction is padded to a batch-wide length and the softmax runs over the padding, then a token's result depends on **how many other tokens happen to be in the call** - the comment says a 6-token and a 13-token batch diverge from layer 3 on. `vl` is the token's own window length, summed from its runs, and it is what makes incremental prefill bitwise-equal to a single-shot prefill. A mask that is "applied" by zeroing scores after a batch-wide softmax has this bug even when every score is right.

So the test that matters is not "does the output look right". It is a comparison at the level the bug lives at: the score matrix, or the probabilities, before they are mixed into values, against a reference implementation of the same shapes. That is what the build step asks for, and it is why the naive recompute oracle is written before the fast path rather than after.

**A mask is also a decision about work.** A kernel that computes the masked half and then multiplies by zero has spent twice the arithmetic to produce the same answer; a kernel that skips it is not an optimisation but the correct implementation of what the mask means. Chapter 25 comes back to this on a device where the skip has to be expressed as a block-level test.

### Lesson 3 · Grouped-query attention: head reuse, cache size, and the arithmetic that decides it

Fourteen query heads and two key/value heads. Every query head still has its own $W_q$ slice and its own output, but heads 0-6 read key head 0 and heads 7-13 read key head 1:

<figure class="book-figure">{{#include ../figures/10-gqa-heads.svg}}</figure>

*Figure 10.2 - grouped-query attention as it is: a mapping from fourteen query heads onto two cached key/value heads. The queries are unchanged; what divides by seven is the cache.*

In the code that is one line, and the line has a wrong version that looks reasonable:

```rust
let kv_head = h / (n_head / n_kv);      // 14 / 2 = 7: heads 0-6 use kv 0, 7-13 use kv 1
// let kv_head = h % n_kv;              // wrong: interleaves the heads instead of grouping them
```

Both compile, both run, both produce finite numbers, and the second is a different model - it would be correct only if the checkpoint had been trained with that mapping. This is the bug a tensor shape cannot catch: `attn_k` is `[896, 128]`, and 128 is 2 x 64 whether the two heads are groups or interleaves. The file's metadata says `head_count_kv = 2` and says nothing about which query heads belong to which key head, because in every model that ships GQA the answer is "consecutive groups".

The engine's own lines, and they do two jobs at once:

<!-- code: src/graph/cpu_backend.rs::attn_heads@e844591 -->
```rust
unsafe fn attn_heads(ctx: *const (), h0: usize, h1: usize) {
    let c = &*(ctx as *const AttnCtx);
    let gqa = c.nh / c.hk;
    let ne_q = c.nh * c.hd;
    let mut scrs = vec![0.0f32; c.max_vl.max(1)];
    for h in h0..h1 {
        let hk = h / gqa;
```
<!-- /code -->

`hk` is the answer to "which cache head does this query read", and it is also an address: `hk * c.hd_kv` is the offset of that head's first key inside a cache row (line 1131), and the same product is the value row's base (line 1160). One number, three jobs - mapping, key offset, value offset. Use `h % n_kv` instead and every read is still *inside* the region, which is exactly why nothing crashes: the addresses are valid and the model is different.

Why it matters is the cache, and the cache is what turns long context into a memory problem rather than a compute problem:

<!-- facts: engine-measure.kv-cache-size -->
| Quantity | f32 | f16 | q8_0 |
|---|---|---|---|
| f32 words one cell occupies | 128 | 128 | 34 |
| Bytes per token per layer | 1,024 B | 1,024 B | 272 B |
| Bytes per token, all 24 layers | 24.0 KiB | 24.0 KiB | 6.4 KiB |
| Region at 4k tokens | 96.0 MiB | 96.0 MiB | 25.5 MiB |
| Region at 8k tokens | 192.0 MiB | 192.0 MiB | 51.0 MiB |
| Region at 32k tokens | 768.0 MiB | 768.0 MiB | 204.0 MiB |
| Share of the weights at 8k | 41.0 per cent | 41.0 per cent | 10.9 per cent |
| Read per decode step at 4k | 96.0 MiB | 48.0 MiB | 25.5 MiB |

The first three rows are `row_elems x 4 x 2` - f32 words per cell, bytes per word, and the two regions K and V. The region rows are the engine's own `kv_footprint` measurement, which `scripts/kv-engine.py` recomputes from `row_elems` and refuses to write when the two disagree. The share divides by the 468.6 MiB file. The last row is the payload actually read per decode step, not the words reserved for it.
<!-- /facts -->

The engine computes this table at load time in two functions, and every value in it is checked against the engine's own measurement of the same quantity:

<!-- code: src/graph/kvformat.rs::KvFormat::row_elems@e844591 -->
```rust
/// f32 words one cell occupies in the allocator's pool.
///
/// F32 and F16 occupy one word per element; Q8_0 occupies
/// `ceil(blocks * 34 / 4)` words — rounded up so every cell starts on a word
/// boundary, which is what lets a cell move be a plain `copy_within`.
pub fn row_elems(self, nkt: usize) -> usize {
    match self {
        KvFormat::F32 | KvFormat::F16 => nkt,
        KvFormat::Q8_0 => {
            let bytes = (nkt / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
            bytes.div_ceil(WORD_BYTES)
        }
    }
}
```
<!-- /code -->

<!-- code: src/graph/kvcache.rs::KvCache::region_bytes@e844591 -->
```rust
/// Bytes the persistent KV regions occupy (`elems * 4` per region), summed over
/// layers — the footprint C4 measures a packed region against.
pub fn region_bytes(&self) -> usize {
    self.layers
        .values()
        .map(|l| (l.elems * std::mem::size_of::<f32>()) * 2)
        .sum()
}
```
<!-- /code -->

Read `row_elems` closely, because it is where a format decision becomes an allocator decision: a Q8_0 cell is `ceil(blocks * 34 / 4)` **f32 words**, rounded up to a word boundary, and the comment says why - so that moving a cell is a plain `copy_within`. Packed cells are therefore slightly larger than their payload, and every subsequent byte offset in the engine is in words, not in quantized blocks. `region_bytes` is the doubling for K and V that the table folds into its per-token row.

Read that table by columns, because the columns are the lesson. **f32 and f16 reserve exactly the same bytes.** `row_elems` returns `nkt` for both, so a cell is 128 f32 words per region whether the value inside it is 32 bits or 16: f16 writes half a cell and leaves the other half unused - 256 unused bytes per cell per region, which is the `unused_bytes_per_cell` the engine reports. What f16 buys is **bytes read**, not bytes reserved: 48 MiB per decode step at a 4k context against 96 for f32, on the same 96 MiB reservation. Only q8_0 shrinks the reservation, to 34 words a cell - 272 bytes per token per layer, a quarter of f32's - which is what the packed-region path and its format gate exist for.

The reservation is the row that decides whether a long conversation fits. At 32,000 tokens f32 and f16 need **768 MiB**, which is **164 per cent of the model's 468 MiB of weights**: the cache stops being a footnote and becomes the resident set, and the weights are no longer the thing you are short of. q8_0 brings the same conversation to 204 MiB, 43 per cent. Attention does not become expensive because the arithmetic grows; it becomes expensive because the state it reads grows, and it is read on every single token.

### Lesson 4 · Prefill attention versus decode attention: one big masked matmul against a streamed reduction

Prefill and decode compute the same function with different shapes, and the shapes decide which resource runs out.

**Prefill** is a matmul: $L$ queries against $L$ keys and $L$ values, all heads, one pass. The keys and values are read once and stay in cache; the work per byte is high; a CPU core or a GPU is doing what it is designed for.

**Decode** is a reduction: one query against $L$ cached keys, per head, per layer, per token. The bytes are the same ones you read last step, and they are read again, because the query changed and the keys did not. Nothing is reused across steps except the cache itself.

Both arrive at the same kernel through one function, and it is where "a position, not a structure" meets a graph:

<!-- code: src/graph/cpu_backend.rs::decode_window@e844591 -->
```rust
/// Decode a query-window input into `(cell, len)` runs plus per-query offsets.
///
/// Two layouts reach the kernel, and their **sizes tell them apart** (an input's
/// size is graph topology, so this is fixed at build time, not a per-step choice):
/// `attn_span` carries one `(lo, hi)` per query (E1), and `kv_map` carries
/// `KV_MAP_MAX_SPANS` `(cell, len)` runs per query, zero-padded (C8b S2 — what a
/// sequence reads when it shares a prefix: the shared runs plus its own). Both are
/// validated here, and a length that is neither is an error rather than a guess.
fn decode_window(
    input: &[f32],
    nt: usize,
    n_ctx: usize,
) -> Result<(Vec<(usize, usize)>, Vec<usize>), String> {
    let mut runs: Vec<(usize, usize)> = Vec::with_capacity(nt);
    let mut off: Vec<usize> = Vec::with_capacity(nt + 1);
    off.push(0);
    if input.len() == 2 * nt {
        for t in 0..nt {
            let (lo, hi) = (
                input[t].to_bits() as usize,
                input[nt + t].to_bits() as usize,
            );
            if hi > n_ctx || hi <= lo {
                return Err(format!(
                    "attn: query {t} has span [{lo}, {hi}) — outside the {n_ctx}-cell arena, \
                     or empty (a span input that was never filled is all zeros)"
                ));
```
<!-- /code -->

The input's **length** decides which layout it is - `2 * nt` values is one `(lo, hi)` per query, `nt * KV_MAP_MAX_SPANS * 2` is a list of `(cell, len)` runs per query - and the length is fixed when the graph is built, so this is topology rather than a per-step branch. Anything else is refused with the query number in the message. Note what the bounds check catches: an input that was never filled is all zeros, and `hi <= lo` is exactly the shape of that mistake.

The dispatch, meanwhile, is four lines - the one place this chapter meets Chapter 8:

<!-- code: src/graph/cpu_backend.rs::cpu_gqa_attn_runs@e844591 -->
```rust
    unsafe { attn_heads(&ctx as *const _ as *const (), 0, nh) };
    return Ok(());
}
crate::kernel::par_for(nh, &ctx as *const _ as *const (), attn_heads);
Ok(())
```
<!-- /code -->

Heads are the unit of work because they are independent: each worker owns a head range and a private score buffer, so the result is bit-identical to the single-threaded order because there is no cross-head reduction to reorder. That is why a thread pool can be bolted on here without moving a single number - and why Chapter 8 could claim a thread speedup without a tolerance.

The measurement that separates them is the next table read the other way: one decode step at a 1,024-token prefix costs 212 microseconds, one at 512 costs 106, one at 256 costs 53. **The step is linear in the prefix**, which is what a reduction over cached rows should be - and because a run to $L$ tokens performs $L$ such steps with the average prefix at $L/2$, the run is quadratic in total. This is the honest form of "attention is quadratic": no single step is, and the sequence of steps is.

<!-- facts: attn.decode-step -->
| Cached prefix | One decode step | Per 100 tokens of prefix | Softmax share of the step |
|---|---|---|---|
| tokens | microseconds | nanoseconds | per cent |
| 128 | 26.7 | 21 | 104 |
| 256 | 53.3 (2.0x the previous row) | 21 | 52 |
| 512 | 108.0 (2.0x the previous row) | 21 | 26 |
| 1,024 | 218.0 (2.0x the previous row) | 21 | 13 |
<!-- /facts -->

The softmax column is the one to keep. Its share of the step **falls** as the prefix grows - most of a step at 128 tokens, 11 per cent at 1,024 - because the exponential is linear in the row while the two matmuls are linear in the row for every element. At small prefixes the softmax dominates a step; at long ones the multiplies do. A chapter that optimised the softmax and stopped would have optimised the part that stops mattering.

Two consequences the rest of the book depends on. First, prefill and decode want different kernels - one compute-bound, one bandwidth-bound - which is why Chapter 8's matrix/vector split is not a detail but the shape of the whole engine. Second, the arithmetic intensity of decode is about one multiply-accumulate per byte moved, so the per-token cost is set by memory rather than by the multiplier: the same reason Chapter 1's ceiling was bandwidth and Chapter 6's quantisation was worth 5.5x.

### Lesson 5 · The cache as data: positions, spans, and the layout your kernels will assume

The cache turns a computation into a data structure, and the data structure has a layout that every later kernel depends on.

**Positions are data.** A query's position decides which keys it may read (the mask compares indices) and where its rotation comes from (Chapter 9 rotated by position). So the kernel receives positions as arguments - `start_pos`, a length, an array - and never derives them from where a token sits in a graph. ADR-0002 makes this a rule because the alternative is worse than it sounds: if position were structure, a cached conversation could not start at a non-zero offset without rebuilding the graph, and every token would pay for a topology change.

The signature is the argument, made concrete:

<!-- code: src/graph/cpu_backend.rs::cpu_rope@e844591 -->
```rust
/// RoPE per-head (same math as forward.rs's apply_rope).
///
/// #724: `freqs` is the `llama3` per-pair factor table the file carries, and the
/// per-pair inverse frequency comes from `vec_ops::rope_inv_freq` — the same
/// expression the device kernels mirror — so an unscaled model (`None`) computes
/// exactly what this function computed before the table existed.
pub(crate) fn cpu_rope(
    x: &mut [f32],
    pos: &[usize],
    nh: usize,
    hd: usize,
    freq_base: f32,
    freq_scale: f32,
    freqs: Option<&crate::vec_ops::RopeFreqs>,
    style: RopeStyle,
) {
```
<!-- /code -->

`pos: &[usize]` sits between the data and the geometry, and nothing inside the function can tell whether this call is a 1,024-token prefill or a one-token decode: it rotates row `t` by `pos[t]`, whatever those numbers are. That is what "positions are data" means mechanically - the kernel's shape does not change when the position does.

**The layout is a stride contract.** The cache is addressed as layer, then position, then key/value head, then dimension. That ordering makes one head's slice over positions a walk with a constant stride, which lets a decode kernel read the keys for one head while ignoring the other head's, and turns a per-layer loop into a pointer increment instead of a division. Swap the two inner dimensions and every read becomes a gather.

Positions become cells in one function, and its doc comment is the chapter's claim in the author's own words:

<!-- code: src/graph/kvcache.rs::KvCache::cells_for@e844591 -->
```rust
/// True while `cell == pos` for every live row, i.e. while a backend may
/// keep indexing the arena with the raw positions input.
pub fn is_identity(&self) -> bool {
    self.identity
}

/// Host-side `position -> cell` resolution for `layer`.
///
/// C1: the identity, with a bounds check — a position at or beyond `n_ctx`
/// would index past the arena. C2 replaces exactly this function's body
/// (consulting `owner` and the layer's window) without touching its
/// callers.
pub fn cells_for(&self, layer: usize, positions: &[usize]) -> Result<Vec<u32>, String> {
        // ...
    for &p in positions {
        // In C1 the cell is the position, so the check is also the arena
        // bound; C2 keeps this the arena bound and checks ownership instead.
        if p >= l.n_ctx {
            return Err(format!(
                "KV layer {layer}: position {p} >= n_ctx {} (arena overflow)",
                l.n_ctx
            ));
        }
        cells.push(p as u32);
```
<!-- /code -->

Today the body is the identity plus a bounds check - `cells.push(p as u32)` - and `is_identity()` is the flag that lets a backend skip the indirection while the two agree. The comment describes the replacement: consult the owner table and the layer's window *without touching a caller*. That is what a seam looks like when it is built before it is needed, and it is why "the cache is data" is a design commitment rather than a description.

**The growth is the allocator's problem.** The cache grows one token at a time and nobody knows the final length in advance. Preallocate to the maximum context and you pay 96 MiB for a greeting; grow amortised, and every kernel holding a pointer into the cache has to survive a reallocation. Chapter 18 solves this properly, with cells and spans; here it is enough to notice that the choice exists and that the layout above forces it.

**The format is a gate, and it reaches all the way into the kernel.** A packed Q8_0 cache computes the same attention with a different read:

<!-- code: src/graph/cpu_backend.rs::attn_heads_q8@e844591 -->
```rust
let kbytes = kvformat::region_bytes(ka);
let block_bytes = (c.hd_kv / kvformat::Q8_0_BLOCK) * kvformat::Q8_0_BLOCK_BYTES;
// One Q8_0 block row per worker (not per query): the query is re-quantized per
// (head, token) into this buffer and the K dot reads it back.
let mut qq = vec![0u8; block_bytes];
let mut scrs = vec![0.0f32; c.max_vl.max(1)];
for h in h0..h1 {
    let hk = h / gqa;
    let head_elem = hk * c.hd_kv;
    for t in 0..c.nt {
        let qs = t * ne_q + h * c.hd;
        let qrow = std::slice::from_raw_parts(c.q.add(qs), c.hd_kv);
        crate::quants::quantize_row_q8_0_buf(qrow, 1, c.hd_kv, &mut qq);
```
<!-- /code -->

The query is quantized per (head, token) into one row of Q8_0 blocks, and the K dot becomes an integer dot against the packed cell instead of a float dot against a decoded one; V is accumulated out of the cell (`accumulate_q8_0_row`). The window walk, the softmax and the stride arithmetic do not move - which is why the cache format is a **gate** in ADR-0006: the region width, the graph node's metadata and the kernel must agree before a load succeeds, and a backend that cannot read packed cells refuses the format rather than decoding it behind your back.

Now the honest part, and it is a negative result this chapter keeps rather than rounds up. **The memory win is real; the speed win never arrived.** A Q8_0 cell is 34 bytes per 32 values against 128 for f32, so a region is 3.76x smaller - but measured on the DGX Spark's GB10, `q8_0` decode and prefill are **1.23x and 1.24x slower than f16** at head dimension 64, and 1.01x / 1.04x at 128. The residual has been attributed to a long-scoreboard dependency chain in the packed decode kernel, and the attribution concludes that closing it needs an aligned block layout or more warps - not a kernel tweak - so it is open. Nor is `q8_0` ever chosen for you: the engine's own auto policy only ever picks f16 or f32, and `MINFER_CACHE_TYPE=q8_0` is opt-in. If you are reading this to decide whether to switch it on: switch it on for memory, not for speed, and measure on your own device. When that gap closes this paragraph changes, and the obligation is a ticket rather than a good intention: [this book's issue 2](https://github.com/yusiwen/llm-inference-book/issues/2) lists exactly what has to be re-measured, and [minfer's issue 212](https://github.com/yusiwen/minfer/issues/212) carries the reciprocal note. A book that quotes numbers has to name the day they expire.

**And the cache must match a rebuild exactly.** This is the correctness gate for the whole chapter: attention over a cache that was written token by token must equal attention over a prefix recomputed from scratch. The measurement below is why that gate is not free, and why it is worth building anyway.

<!-- facts: attn.rebuild -->
| Operation | Time | Bytes | What it replaces |
|---|---|---|---|
| Recompute K and V for 1,024 positions | 110.3 ms | 1.00 MiB written | what an engine without a cache does on every token |
| Read the same K and V | 64 us | 1.00 MiB read | 1,733x cheaper than recomputing |
| Attend over the cached prefix | 218 us | the same bytes | 506x cheaper than recomputing |
<!-- /facts -->

105 milliseconds to rebuild a 1,024-token prefix against 64 microseconds to read it back, and 212 microseconds to do the entire attention step over it. The cache is worth 500x on this model at this length, and the ratio grows with the prefix because the rebuild is quadratic while the read is linear. That is also the honest cost of the correctness gate: the naive oracle costs 105 milliseconds per step if you run it on every token, so the oracle is a test and not a mode. Run it on a short prompt, compare the cache path against it, and measure performance with the cache alone.

## Cache arithmetic for one model

<!-- facts: engine-measure.kv-cache-size -->
| Quantity | f32 | f16 | q8_0 |
|---|---|---|---|
| f32 words one cell occupies | 128 | 128 | 34 |
| Bytes per token per layer | 1,024 B | 1,024 B | 272 B |
| Bytes per token, all 24 layers | 24.0 KiB | 24.0 KiB | 6.4 KiB |
| Region at 4k tokens | 96.0 MiB | 96.0 MiB | 25.5 MiB |
| Region at 8k tokens | 192.0 MiB | 192.0 MiB | 51.0 MiB |
| Region at 32k tokens | 768.0 MiB | 768.0 MiB | 204.0 MiB |
| Share of the weights at 8k | 41.0 per cent | 41.0 per cent | 10.9 per cent |
| Read per decode step at 4k | 96.0 MiB | 48.0 MiB | 25.5 MiB |

The first three rows are `row_elems x 4 x 2` - f32 words per cell, bytes per word, and the two regions K and V. The region rows are the engine's own `kv_footprint` measurement, which `scripts/kv-engine.py` recomputes from `row_elems` and refuses to write when the two disagree. The share divides by the 468.6 MiB file. The last row is the payload actually read per decode step, not the words reserved for it.
<!-- /facts -->

<!-- facts: engine-measure.provenance -->
*`scripts/engine-measure.py`; runs the engine's own measurement tests (`cargo test --release measure`) and reads the KV geometry and footprint groups out of them, then recomputes every number from `KvFormat::row_elems` and refuses to write the artifact when the two disagree; on **Darwin 27.2.0, arm64**; one run of the engine's own measurement tests; the derived column is recomputed from `KvFormat::row_elems` and must equal the measured one. Command: see [the provenance index](../appendix/provenance.md#engine-measure). Raw numbers: `facts/engine-measure.json`.*
<!-- /facts -->

<!-- facts: attn.provenance -->
*`scripts/attn.c`; runs one layer of real attention on the model's own weights: scores, one decode step at four prefix lengths, the softmax alone, and rebuilding the cache against reading it; on **arm64, Darwin 27.2.0, clang -O2**; 5 runs; median of 5 runs of 3 repetitions; the score scan is exhaustive over every head and position. Command: see [the provenance index](../appendix/provenance.md#attn). Raw numbers: `facts/attn.json`.*
<!-- /facts -->

## Project

**Build step S10 · attention you can trust.** The naive implementation is the oracle: recompute attention over the whole prefix every step and compare. This costs a few seconds of runtime and saves days of debugging later in the book.

Deliberately break the mask in your own code and observe which test catches it. If none does, the test suite is the deliverable, not the kernel.

## Deliverable

- A correct prefill and decode attention with a naive-recompute oracle test
- A filled cache-size table for one real model, with the config fields you used
- A positions-as-data note: what the kernel receives, and what would break if positions were baked into the graph

## How you know you passed

- Prefill and decode agree with the naive recompute at the tolerance your precision ledger set - and you can say which mask bug your test would have missed, with the number from `attn.json` that says why.
- Your cache-size prediction is within a few percent of the observed allocation, and you can name the row of the table you derived rather than measured.
- You can produce a wrong answer by changing the mask and show the test that fails - and the second test, which catches the off-by-one that the first one does not.

## Reading

- code: `src/graph/kvcache.rs`, `src/graph/kvformat.rs`, `src/graph/cpu_backend.rs` (attention) at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/KV-CACHE-DESIGN.md`, `docs/inference_e2e_walkthrough/11-attention-vecops-kv.md`
- decisions: ADR-0002 (topology is a function of graph params, so positions cannot be structure), ADR-0006 (the KV format is a per-engine gate)
- [gqa] Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023) - arXiv:2305.13245

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
