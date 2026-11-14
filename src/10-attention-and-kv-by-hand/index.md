# Chapter 10 · Attention and the KV Cache by Hand

> - **Compute tier**: ✅ any laptop
> - **Status**: written - the score magnitudes, the mask's cost, one decode step at four prefix lengths and the cache against a rebuild are measured on this model's own weights by `scripts/attn.c`; what is *not* here is any device kernel (Chapter 25) or the storage subsystem (Chapter 18)
> - **Build step S10**: Causal attention with grouped queries over an explicitly managed KV cache: prefill writes, decode reads, and the results match a naive recompute.
> - **Rule of this book**: This chapter establishes the semantics: what attention computes and how the cache makes it cheap. The storage subsystem (cells, spans, defragmentation) is Chapter 18, and the device kernels are Chapters 21 and 25.

## First, the gist

Attention lets every token look at the tokens before it and mix their values into its own representation. Naively that means recomputing the keys and values of the whole prefix at every step, which is quadratic work per token. The **KV cache** (key-value cache) removes that: keys and values are computed once, stored, and read back. **GQA** (grouped-query attention) then reuses one key/value head across several query heads, which is what makes the cache small enough to hold. The whole chapter is one idea applied twice: prefill fills the cache, decode reads it, and both must agree exactly.

The sizes are worth having in front of you before the definitions, because they are what the rest of the book optimises. On this model, at a 1,024-token prefix, one layer's attention for one token - fourteen heads, no SIMD, f32 - takes **212 microseconds**, and **11 per cent** of that is the softmax; the rest is two matrix multiplies. Rebuilding the keys and values for that prefix instead of reading them takes **105 milliseconds**, which is 500 times the whole attention step. A four-thousand-token context costs **48 MiB** of cache across the model's 24 layers, and at eight thousand it is **96 MiB** - a fifth of the weights.

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

The arithmetic on this model's own weights, over every head and every cached position at a 1,024-token prefix:

<!-- facts: attn.attention-scores -->
| Quantity | Value |
|---|---|
| (head, cached position) pairs | 14,336 - 14 heads over a 24-layer model's first layer, 1,024 positions |
| Mean absolute score | 0.437 |
| Root mean square | 0.656 |
| Largest absolute score | 4.67 |
| Pairs above the f16 exp bound (11.09) | 0 |
| Pairs above the f32 exp bound (88.72) | 0 |
<!-- /facts -->

Two things to read out of that table. The scores are **small**: a mean magnitude of 0.44 and a maximum of 4.67. And that means the f16 softmax overflow Chapter 9 measured the bound for - `exp` of 11.09 - **does not happen here**. Zero pairs, on real weights, at layer 0. That is not a licence to skip the max subtraction: Chapter 9's point was that the bound belongs to the format, not to the data, and this model at this layer is one sample. A different model, a different layer, or a prompt whose logits concentrate can cross it; the subtraction costs one pass and one register, and the alternative is an `inf` that propagates into every layer above.

What the numbers *do* say is where the time goes. The softmax over a 1,024-wide row, measured alone, is 24 microseconds for fourteen heads; the step it belongs to is 212. So the exponent is **11 per cent** of a decode step, and the two matmuls are the other 89 per cent. Compare that with Chapter 9, where the softmax was 95 per cent of the *pointwise* work at 4,096 tokens: both statements are true, because "the pointwise work" and "the step" are different denominators. The chapter that measures has to say which one it means.

### Lesson 2 · Causal masking at prefill, and the off-by-one that survives every test that does not compare logits

A decoder may not look forward. At prefill the whole prompt arrives at once, so the score matrix is $L \times L$ and the mask keeps the lower triangle: query $i$ may read key $j$ when $j \le i$. The arithmetic is exact - at a 1,024-token prefix that keeps 524,800 of 1,048,576 pairs, **50.05 per cent** - and it is the reason prefill does half the work a naive implementation would.

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

Why it matters is the cache, and the cache is what turns long context into a memory problem rather than a compute problem:

<!-- facts: attn.cache-size -->
| Quantity | Formula | Value for this model | Measured or derived |
|---|---|---|---|
| Bytes per token per layer | 2 x n_kv x head_dim x 2 (f16) | 512 B | derived from 2 KV heads x 64 dims |
| Bytes per token, all layers | previous x n_layer | 12288 B (12.0 KiB) | 24 layers |
| Bytes per 4k context | previous x 4096 | 48.0 MiB | the buffer the instrument allocates |
| Bytes per 8k context | previous x 2 | 96.0 MiB | doubling, not measured |
| Share of the weights at 8k | cache / 469 MiB file | 20.5 per cent | both are facts about this file |
| Attention reads per decode step at 4k | the whole cache, every layer | 48.0 MiB | the step measured below |
<!-- /facts -->

Two of those rows are worth deriving rather than reading. The per-token figure is $2 \times n_{kv} \times d \times 2$ - two tensors, two heads, 64 dimensions, two bytes each - which is 512 bytes per token per layer, or 12 KiB per token across 24 layers. And the last row is the one that decides whether a 32,000-token conversation fits: at that length the cache is 375 MiB, which is **80 per cent of the model's 468 MiB of weights**. Attention does not become expensive because the arithmetic grows; it becomes expensive because the state it reads grows, and it is read on every single token.

### Lesson 4 · Prefill attention versus decode attention: one big masked matmul against a streamed reduction

Prefill and decode compute the same function with different shapes, and the shapes decide which resource runs out.

**Prefill** is a matmul: $L$ queries against $L$ keys and $L$ values, all heads, one pass. The keys and values are read once and stay in cache; the work per byte is high; a CPU core or a GPU is doing what it is designed for.

**Decode** is a reduction: one query against $L$ cached keys, per head, per layer, per token. The bytes are the same ones you read last step, and they are read again, because the query changed and the keys did not. Nothing is reused across steps except the cache itself.

The measurement that separates them is the next table read the other way: one decode step at a 1,024-token prefix costs 212 microseconds, one at 512 costs 106, one at 256 costs 53. **The step is linear in the prefix**, which is what a reduction over cached rows should be - and because a run to $L$ tokens performs $L$ such steps with the average prefix at $L/2$, the run is quadratic in total. This is the honest form of "attention is quadratic": no single step is, and the sequence of steps is.

<!-- facts: attn.decode-step -->
| Cached prefix | One decode step | Per 100 tokens of prefix | Softmax share of the step |
|---|---|---|---|
| tokens | microseconds | nanoseconds | per cent |
| 128 | 26.3 | 21 | 90 |
| 256 | 53.0 (2.0x the previous row) | 21 | 45 |
| 512 | 105.7 (2.0x the previous row) | 21 | 22 |
| 1,024 | 211.7 (2.0x the previous row) | 21 | 11 |
<!-- /facts -->

The softmax column is the one to keep. Its share of the step **falls** as the prefix grows - most of a step at 128 tokens, 11 per cent at 1,024 - because the exponential is linear in the row while the two matmuls are linear in the row for every element. At small prefixes the softmax dominates a step; at long ones the multiplies do. A chapter that optimised the softmax and stopped would have optimised the part that stops mattering.

Two consequences the rest of the book depends on. First, prefill and decode want different kernels - one compute-bound, one bandwidth-bound - which is why Chapter 8's matrix/vector split is not a detail but the shape of the whole engine. Second, the arithmetic intensity of decode is about one multiply-accumulate per byte moved, so the per-token cost is set by memory rather than by the multiplier: the same reason Chapter 1's ceiling was bandwidth and Chapter 6's quantisation was worth 5.5x.

### Lesson 5 · The cache as data: positions, spans, and the layout your kernels will assume

The cache turns a computation into a data structure, and the data structure has a layout that every later kernel depends on.

**Positions are data.** A query's position decides which keys it may read (the mask compares indices) and where its rotation comes from (Chapter 9 rotated by position). So the kernel receives positions as arguments - `start_pos`, a length, an array - and never derives them from where a token sits in a graph. ADR-0002 makes this a rule because the alternative is worse than it sounds: if position were structure, a cached conversation could not start at a non-zero offset without rebuilding the graph, and every token would pay for a topology change.

**The layout is a stride contract.** The cache is addressed as layer, then position, then key/value head, then dimension. That ordering makes one head's slice over positions a walk with a constant stride, which lets a decode kernel read the keys for one head while ignoring the other head's, and turns a per-layer loop into a pointer increment instead of a division. Swap the two inner dimensions and every read becomes a gather.

**The growth is the allocator's problem.** The cache grows one token at a time and nobody knows the final length in advance. Preallocate to the maximum context and you pay 96 MiB for a greeting; grow amortised, and every kernel holding a pointer into the cache has to survive a reallocation. Chapter 18 solves this properly, with cells and spans; here it is enough to notice that the choice exists and that the layout above forces it.

**And the cache must match a rebuild exactly.** This is the correctness gate for the whole chapter: attention over a cache that was written token by token must equal attention over a prefix recomputed from scratch. The measurement below is why that gate is not free, and why it is worth building anyway.

<!-- facts: attn.rebuild -->
| Operation | Time | Bytes | What it replaces |
|---|---|---|---|
| Recompute K and V for 1,024 positions | 105.1 ms | 1.00 MiB written | what an engine without a cache does on every token |
| Read the same K and V | 64 us | 1.00 MiB read | 1,651x cheaper than recomputing |
| Attend over the cached prefix | 212 us | the same bytes | 497x cheaper than recomputing |
<!-- /facts -->

105 milliseconds to rebuild a 1,024-token prefix against 64 microseconds to read it back, and 212 microseconds to do the entire attention step over it. The cache is worth 500x on this model at this length, and the ratio grows with the prefix because the rebuild is quadratic while the read is linear. That is also the honest cost of the correctness gate: the naive oracle costs 105 milliseconds per step if you run it on every token, so the oracle is a test and not a mode. Run it on a short prompt, compare the cache path against it, and measure performance with the cache alone.

## Cache arithmetic for one model

<!-- facts: attn.cache-size -->
| Quantity | Formula | Value for this model | Measured or derived |
|---|---|---|---|
| Bytes per token per layer | 2 x n_kv x head_dim x 2 (f16) | 512 B | derived from 2 KV heads x 64 dims |
| Bytes per token, all layers | previous x n_layer | 12288 B (12.0 KiB) | 24 layers |
| Bytes per 4k context | previous x 4096 | 48.0 MiB | the buffer the instrument allocates |
| Bytes per 8k context | previous x 2 | 96.0 MiB | doubling, not measured |
| Share of the weights at 8k | cache / 469 MiB file | 20.5 per cent | both are facts about this file |
| Attention reads per decode step at 4k | the whole cache, every layer | 48.0 MiB | the step measured below |
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
