# Chapter 9 · Vector Ops and the Precision Ledger

> - **Compute tier**: ✅ any laptop (CPU only; reading two tensors from the file is the whole data requirement)
> - **Status**: written - five ops implemented twice (once the way an engine writes them, once in f64 as an oracle) and witnessed a third time by numpy; every number in the ledger is measured on the book's own file, and the cost table corrects the plan's assumption that these ops are free
> - **Build step S9**: RMSNorm, RoPE, softmax, SiLU and the elementwise ops your transformer needs, each with a precision note and a reference value.
> - **Rule of this book**: Attention itself is Chapter 10; this chapter is the pointwise and reduction ops that surround it. Device-specific versions of these same ops appear in Chapters 21 and 24 but are not re-derived there.

## First, the gist

Between the matmuls a transformer does a handful of cheap operations: normalize a vector, rotate pairs of coordinates by position, exponentiate and normalize a score row, and gate a value through a sigmoid. Each is a few lines of arithmetic and each has a precision decision attached - what dtype the input is, what accumulates, what the output must be. Those decisions are where a correct engine quietly stops matching a reference, so this chapter keeps a ledger: for every op, the dtype in, the dtype of the accumulation, the dtype out, and the tolerance the reference demands.

The ledger, filled in from measurements on the book's own weights:

<!-- facts: vecops.ledger -->
| Op | Input | Accumulate | Output | Max difference, f32 path against f64 | Comparison class |
|---|---|---|---|---|---|
| RMSNorm | f32 | f32, 8 lanes (3.6e-07) or f32 sequential (6.2e-07) | f32 | 3.6e-07 | bounded, 1e-6 |
| softmax | f32 | f64 sum (1.3e-06 if f32) | f32 | 1.3e-06 | bounded, 1e-5 |
| SiLU | f32 | f32 | f32 | 1.3e-07 | bounded, 1e-6 |
| SwiGLU | f32 | f32 | f32 | 1.3e-07 | bounded, 1e-6 |
| RoPE | f32 | f32 (sin/cos in f32) | f32 | 5.3e-08 | bounded, 1e-6 |
| residual add | f32 | f32 | f32 | 2.6e-07 over 24 adds | bounded, 1e-5 over a chain |

Witness: vecops-witness.json (numpy f64 on the same bytes) - the widest disagreement across all five ops is 2.5e-07.
<!-- /facts -->

Every f32 path lands within about **1e-6** of a double-precision oracle, which is what f32 arithmetic costs and not a defect to fix. So the interesting part of this chapter is not the ops that lose a bit; it is the three that do something else entirely:

<!-- facts: vecops.failures -->
| Failure | What happens | Measured threshold |
|---|---|---|
| softmax without max subtraction, f32 | exp overflows, sum is inf, output is inf/inf = NaN | first NaN above logit 88.72 |
| softmax without max subtraction, f16 | the same, but f16 tops out at 65,504 | first NaN above logit 11.09 |
| residual stream kept in f16 | every add rounds the whole stream to 11 bits | 3.0e-03 of the stream's peak after 24 layers |
| residual stream kept in bf16 | 8 mantissa bits, and the drift compounds | 2.5e-02 of the stream's peak after 24 layers |
| RoPE paired the wrong way | the scores are a different function of distance, and every invariant still holds | 7.3e-02 of the norm product |
<!-- /facts -->

**Softmax without the max subtraction is not inaccurate, it is NaN** - above logit 88.72 in f32, and above logit **11.09** if the scores are kept in f16, which real attention scores reach in a long context. **A residual stream kept in bf16 drifts 2.5 per cent of its own peak over 24 layers**, because every add rounds the whole stream to 8 mantissa bits. And **RoPE paired the wrong way still satisfies every invariant you would test it with**: the score for the same query, key and positions differs by 7.3 per cent of the norm product, and no property of rotations catches it.

And the cost table, which is the part of this chapter the plan got wrong:

<!-- facts: vecops.per-token -->
| Context | softmax per head | attention softmax, per token (24 layers x 14 heads) | everything else, per token | softmax share |
|---|---|---|---|---|
| 4096 | 7.2 us | 2.42 ms | 0.14 ms | 95 per cent |
| 32768 | 58.8 us | 19.75 ms | 0.14 ms | 99 per cent |

Per layer: 2 RMSNorm (260 ns), 3 residual adds (135 ns), SiLU over 4,864 (5076 ns), RoPE (260 ns, priced as two 896-wide passes), and 14 softmax rows of the context length.
<!-- /facts -->

RMSNorm, the residual adds, SiLU and RoPE together come to about **0.14 ms per token** - noise next to Chapter 8's 0.35-3.2 ms per pass. The attention softmax is not noise. At 4,096 context it is **2.42 ms per token**, ninety-five per cent of the pointwise work, because it is 24 layers x 14 heads = 336 rows of 4,096 exponentials: 1.4 million of them per token. That is the reason Chapter 10 fuses the softmax into the attention kernel, and the reason engines carry a polynomial `exp`.

## What you'll be able to do

- Implement RMSNorm, RoPE, softmax and SiLU from the equations, without a framework, and match an f64 oracle to 1e-6.
- Explain why the max subtraction in softmax is a correctness requirement rather than a nicety, with the logit at which each dtype breaks.
- Explain the two RoPE conventions in the wild, why the model's config decides which one it needs, and why no invariant test can tell them apart.
- Keep and read a precision ledger: which op widens, which narrows, and where a comparison must be bitwise instead of approximate.
- Predict which of these ops actually cost time at a long context - and be right, which means pointing at `exp`, not at the norm.

## Lessons

- [ ] 1. RMSNorm: the equation, the epsilon, and why there is no mean subtraction
- [ ] 2. RoPE: pairs, rotations and positions as data rather than structure
- [ ] 3. Softmax: the max subtraction that is not optional, and its cost on a long row
- [ ] 4. SiLU, gating, and the elementwise ops that fuse into one pass (payoff: why fusion is a graph decision, not a kernel decision)
- [ ] 5. The precision ledger: f32 activations, f16 / bf16 weights, and the ops that must not narrow

### Lesson 1 · RMSNorm: the equation, the epsilon, and why there is no mean subtraction

<figure class="book-figure">{{#include ../figures/09-precision-ledger.svg}}</figure>

*Figure 9.1 - the ledger as a flow: an accumulator may be wider than its operands, and the residual stream never narrows.*

Root-mean-square normalisation divides a vector by the root of the mean of its squares, then scales it elementwise:

```text
rms   = sqrt( mean(x[i]^2) )          over i = 0..n-1
y[i]  = x[i] / rms * w[i]             where w is the learned gain
```

There is no mean subtraction, and that is the whole difference from LayerNorm: the vector is not re-centred, only rescaled to unit RMS. It costs one pass for the sum of squares and one pass to scale - which is why an engine fuses the gain multiply into the scaling pass instead of materialising `x/rms` and multiplying afterwards.

**The epsilon, read out of the book's file.** The GGUF metadata carries the value the model was trained with:

<!-- facts: gguf-qwen2.5-0.5b.model-meta -->
| Key | Type | Value |
|---|---|---|
| `general.architecture` | STRING | qwen2 |
| `general.name` | STRING | qwen2.5-0.5b-instruct |
| `qwen2.block_count` | UINT32 | 24 |
| `qwen2.context_length` | UINT32 | 32768 |
| `qwen2.embedding_length` | UINT32 | 896 |
| `qwen2.feed_forward_length` | UINT32 | 4864 |
| `qwen2.attention.head_count` | UINT32 | 14 |
| `qwen2.attention.head_count_kv` | UINT32 | 2 |
| `qwen2.rope.freq_base` | FLOAT32 | 1000000.0 |
| `qwen2.attention.layer_norm_rms_epsilon` | FLOAT32 | 9.999999974752427e-07 |
| `general.file_type` | UINT32 | 15 |
<!-- /facts -->

`qwen2.attention.layer_norm_rms_epsilon` reads `9.999999974752427e-07`, which is the f32 nearest to 1e-6 - the value is not exactly representable, which is the first place a "just use 1e-6 in f64" implementation can diverge from one that reads the file. It is added to the *mean square* before the square root, so what it protects against is a division by zero for a vector of zeros, not an overflow.

For this file's first embedding row the RMS is `0.0154`, so the mean square is about `2.36e-4` and the epsilon is **0.4 per cent** of it: it moves the scale factor by `2.11e-3`, measured. Small, but not decoration - an engine that "helpfully" normalised with 1e-5 instead would shift every activation by about a per cent.

**What the accumulation dtype costs.** The sum of squares is the one place where the order of additions is visible, because every term is positive and the sum is a single scalar:

| Accumulation | Max relative difference from f64, over the 896-element row |
|---|---|
| f32, one accumulator, sequential | `6.2e-07` |
| f32, eight lanes, folded at the end | `3.6e-07` |
| f64 | `0.0e+00` (it is the reference) |

Both f32 variants are inside the ledger's 1e-6 bound, so for an 896-wide vector an engine may accumulate in f32 and be done. The reference engine goes further and sums in **f64** anyway, and after this table you can see the argument is not accuracy: the f64 sum is *order-independent* to well below the f32 noise, so the same kernel produces the same bits on a machine with four lanes and a machine with eight. That is a property an identity gate can use, whereas "3.6e-07 versus 6.2e-07" is a difference you would otherwise have to explain in every comparison.

### Lesson 2 · RoPE: pairs, rotations and positions as data rather than structure

Rotary position embedding rotates the query and key vectors by an angle proportional to their position. Nothing is stored per position: the position *is* the angle. For a head dimension `hd`, pair `d` (of `hd/2`) rotates by

```text
theta(d, pos) = pos / base^(2d/hd)          base from the file's rope.freq_base
q'[i0] = q[i0] * cos(theta) - q[i1] * sin(theta)
q'[i1] = q[i0] * sin(theta) + q[i1] * cos(theta)
```

Every part of that is unambiguous except **which two coordinates are `i0` and `i1`**, and that is where implementations disagree:

- **adjacent pairs**: `(2d, 2d+1)` - neighbours rotate together. This is the GPT-J convention.
- **half split**: `(d, d + hd/2)` - the vector is treated as two halves folded onto each other. This is what llama.cpp does, and what the Hugging Face Llama and NeoX implementations do.

Same equation, same frequencies, different grouping. The model's file does not record the choice; the architecture name in the metadata implies it, which is why a converter that maps a checkpoint to GGUF has to know it and a kernel that reads the GGUF has to agree.

**The test that does not work.** Both conventions are rotations, so both preserve everything a rotation preserves. Measured on two real 64-wide vectors from the file's embedding matrix, the property "a query at `p` and a key at `p + delta` score by the distance alone" holds to **2.6e-08** with adjacent pairs and **1.7e-08** with the half split. An invariance test cannot fail either one; it is the wrong tool, and shipping it gives you a green tick over a broken model.

**The test that does work** is a comparison against a known-good engine on the same bytes, because the two conventions' scores genuinely differ:

```text
same (q, k, positions), scored under the two pairings:
  difference = 7.3e-02 of the norm product          7 per cent, not a rounding
```

Seven per cent on every score in every head in every layer is not a garbled answer. It is a model that produces fluent, slightly-wrong text - the failure class this book keeps warning about, and the reason the reference engine keeps the RoPE table (ADR-0074: the Llama-3 frequency table is *read from the file*, not recomputed) and `rope_inv_freq` in one place, so the CPU reference and the device kernels cannot drift apart.

**Positions as data.** The other half of RoPE is that `pos` is an argument, not a memory layout. That is what makes KV-cache reuse possible (Chapter 11): a cached key does not have to be stored rotated, because a caller that knows its position can rotate on read. It also makes a *re-rope* a legitimate operation - rotating a cached key by a different position when a batch is reordered - and the reference engine's `RopeParams` struct exists precisely so a caller cannot pass the frequency base without also passing the scale and the table.

### Lesson 3 · Softmax: the max subtraction that is not optional, and its cost on a long row

Softmax turns a row of scores into a distribution:

```text
m     = max(x)
y[i]  = exp(x[i] - m) / sum(exp(x[j] - m))
```

The subtraction of the row maximum is the whole reason this is two passes rather than one, and it is a **correctness** measure rather than a numerical nicety. Without it, `exp(x)` overflows at some logit and the row becomes `inf / inf`:

| Scores kept in | First logit at which the naive path is NaN | Why that number |
|---|---|---|
| f32 | `88.72` | just under the logit where `exp` leaves `FLT_MAX` |
| f16 | `11.09` | `exp(11.09) = 65,504`, f16's largest finite value |

Eleven is not a large logit. Attention scores are `q·k / sqrt(hd)` and reach the twenties at a few thousand tokens, so **a f16 softmax without the max subtraction is a real bug that shows up on real prompts**, and its symptom is a NaN inside a layer rather than an error message.

With the subtraction the largest exponent is `exp(0) = 1`, so the largest term is exactly 1 and the sum is at most `n`. That is why the reference engine's `vec_soft_max_f32` takes `max` as a *parameter*: the caller has the maximum already (or is streaming it, for online softmax) and the kernel does not re-scan the row.

**The sum's dtype.** Summing 4,096 positive terms of similar size in f32 loses about `1.3e-06` relative to f64 - inside the ledger's 1e-5 bound, so f32 is enough here, unlike the norm's sum of squares where the f64 choice is about reproducibility rather than accuracy. The reference engine sums in f64 anyway, for that same reason: the result stops depending on the vector width.

**What it costs.** This is the number that surprised the plan:

<!-- facts: vecops.vecops-cost -->
| Op | Length | Median | Range | ns per element |
|---|---|---|---|---|
| `rms_norm(896)` | 896 | 130 ns | 130-155 ns | 0.15 |
| `silu(896)` | 896 | 935 ns | 920-1035 ns | 1.04 |
| `add(896)` | 896 | 45 ns | 40-45 ns | 0.05 |
| `softmax(896)` | 896 | 1500 ns | 1460-1750 ns | 1.67 |
| `softmax(4096)` | 4096 | 7190 ns | 7130-7780 ns | 1.76 |
| `softmax(32768)` | 32768 | 58770 ns | 57530-60380 ns | 1.79 |
| `softmax(131072)` | 131072 | 232540 ns | 231650-232620 ns | 1.77 |
| `expf` (libm) | 1 | 1.17 ns | 1.10-1.40 ns | 1.17 |
| `exp` (degree-5 polynomial) | 1 | 0.61 ns | 0.60-0.70 ns | 0.61 |
<!-- /facts -->

`softmax` costs **1.7 ns per element** and is essentially flat in the row length: 1.5 us for 896, 7.2 us for 4,096, 232 us for 131,072. The flatness is the important part. 1.7 ns per element is not a memory number - a row of 32,768 floats is 128 KB, which at Chapter 1's 198 GB/s is 0.66 us, and this takes 58.8 us, ninety times longer. **The op is bounded by `exp`, not by bytes**: one transcendental per element, at 1.17 ns through libm.

That also names the fix, and it is measured: a degree-5 polynomial with the `2^k` trick does the same `exp` in **0.61 ns** - 1.9x faster than libm at a max relative error of `1.1e-07`, below f32's own resolution. Every serious engine has one of these, for exactly this reason. The two passes over the row then become the next target, and that is the online-softmax computation in the reading list: one pass carrying a running maximum and a running sum, which is what you need anyway when the row arrives as a stream of attention blocks you have not finished computing.

### Lesson 4 · SiLU, gating, and the elementwise ops that fuse into one pass

SiLU is the activation in a Llama-style feed-forward block:

```text
silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

The block is gated: the feed-forward network computes two projections of the same input, applies SiLU to one and multiplies them elementwise, then projects back -

```text
h = silu(W_gate * x) * (W_up * x)          then    y = W_down * h
```

- which is why the intermediate is called SwiGLU: SiLU as the gate, GLU as the shape. The multiply is elementwise over the 4,864-wide intermediate, and it costs what an elementwise op costs: `add(896)` measures **0.05 ns per element**, so a SwiGLU multiply over 4,864 elements is about 250 ns. SiLU itself is 1.04 ns per element because of the `exp` - the same transcendental that dominates softmax.

**Fusion is a graph decision, not a kernel decision.** Two ways to compute SwiGLU:

```text
unfused:  t       = silu(gate)                   write n
          dst[i]  = t[i] * up[i]                 read n, read n, write n
fused:    dst[i]  = silu(gate[i]) * up[i]        read n, read n, write n - once
```

The unfused version materialises a whole intermediate tensor. At 4,864 floats that is 19 KB - it fits in L1, which is why the difference is invisible in a microbenchmark and *visible* in a real engine: the intermediate is not one tensor, it is one per layer per token, and in a batched prefill it is `n_tokens x 19 KB`. The reference engine's `vec_swiglu_f32` is the fused form, and its comment makes the property that allows the fusion explicit - *same formula, same per-element order, bit-identical to the pair* - which is the property that makes introducing it safe at all. An optimisation that changes the bits is a new kernel and needs its own identity test.

The general rule: **fuse when the intermediate has no other reader, and decide that at the graph level so the scheduler can see it** (Chapter 16). A kernel that internally decides to fuse cannot be scheduled, profiled, or compared against anything.

### Lesson 5 · The precision ledger: f32 activations, f16 / bf16 weights, and the ops that must not narrow

The ledger has two kinds of entry, and the difference between them is this chapter's real content.

**Entries that lose about 1e-6 and are fine.** Every f32 path in the table at the top of the page. They cost nothing to fix because nothing is broken: f32 arithmetic with a sensible accumulator lands within 1e-6 of the equations, and an identity gate should state that tolerance rather than demand bits that do not exist.

**Entries that must not narrow.** The residual stream is the one that matters:

| Stream dtype | Max drift from an f64 reference, over 24 residual adds |
|---|---|
| f32 | `2.6e-07` |
| f16 | `3.0e-03` |
| bf16 | `2.5e-02` |

Two and a half per cent of the stream's own peak, after 24 adds, from bf16 alone. That is not a rounding difference you average out downstream; it is a different model. The reason is structural - a residual add *rewrites every element of the stream*, so the rounding error is not multiplied by a small weight and added once, it is re-injected into everything the next layer reads. An engine may keep weights in bf16 (one rounding, at read time) and must keep the stream in f32 (one rounding per layer, compounding), and this table is the measured size of the difference.

That is the shape of a real precision policy, and the reference engine states it as an ADR rather than a convention: bf16 rounds to nearest even, and 1-D tensors stay f32. The second half - norms and biases stay f32 - follows from the same measurement: those tensors are read by ops that feed the stream, so rounding them to bf16 buys a rounding error and no bandwidth worth having.

**The witness.** A ledger that only compares an implementation against itself proves nothing, so each op here is checked a third way: numpy recomputes it in f64 from the same bytes the C program used, and the disagreement is the number.

<!-- facts: vecops-witness.witness -->
| Op | Max relative difference, C f32 path against numpy f64 | Note |
|---|---|---|
| RMSNorm output | 2.5e-07 | scale 64.9683075 vs 64.9683054 |
| softmax (first 16 of 4096) | 7.4e-08 | sum 4066.0963 vs 4066.0963 |
| SiLU over [-4, 3.75] | 6.4e-08 | 32 probe points |
| RoPE, half-split, positions 1..3 | 5.3e-08 | the C kernel's own inv_freq expression, in f64 |

Verdict: **agree within 1e-6**.
<!-- /facts -->

Read the RoPE row carefully, because it shows the limit of this method: the witness re-implements the **half-split pairing**, because that is what the C kernel claims to do. Two implementations that agree about a wrong pairing agree perfectly. That is why the pairing itself is tested against a golden vector from a known engine - or, in the reader's case, against the reference implementation's own output - and never against a second implementation of your own.

## Project

**Build step S9 · the ops you will reuse everywhere.** Write each op against a reference output captured from a known-good engine, not against your own expectations. The equations are simple; the conventions are not.

Record a golden vector per op - input bytes and expected output bytes - with the tolerance class the ledger assigns it. These become the identity fixtures in Chapter 27, so capture them as bytes (hex or base64), not as printed decimals that lose the low bits.

Order of work:

1. **Read the model's own numbers before writing any op**: the epsilon, the rope base, the head dimension, out of the metadata. An op that hard-codes 1e-5, or 10,000 for the base, is wrong for this file and right for a model nobody runs.
2. **Write RMSNorm and check two things separately**: the output against an f64 reference (expect ~1e-6), and the *scale factor* against a hand computation for one vector. A factor-of-two error in the mean passes the first check when both sides use the same wrong formula.
3. **Write softmax with the max subtraction, then write the version without it and run both** on a row with logits near 90. Watch the naive one produce NaN. A stability property you have never seen fail is a property you do not have.
4. **Write RoPE twice, once per pairing**, and score the same query/key pair both ways. Print the difference: that number is what a mismatch costs. Then pick the pairing from the model's architecture and pin it with a golden vector.
5. **Write SiLU and the fused SwiGLU**, and assert they are bit-identical. If they are not, the fusion changed the arithmetic and needs its own test.
6. **Fill the ledger for your own code**, with the comparison class per op. Then run the witness: a second implementation, in a different language and a different precision, reading the same input bytes.

## Deliverable

- Five ops with golden-vector tests at the tolerances you chose and justified, plus one deliberate failure test (softmax without the max subtraction) proving the guard is load-bearing.
- A precision ledger table, filled in, for your own implementation - including the accumulator dtype per op and why it is that dtype.
- A note on the RoPE convention your target model uses and the config field that decides it.
- A per-token cost estimate for the pointwise ops at two context lengths, with the arithmetic shown, so you know which op to optimise first.

## How you know you passed

- Each op matches its golden vector within a stated, justified tolerance - and bitwise where the ledger says bitwise.
- You can state the epsilon in your norm, in the file's own f32 value, and how much it moves the scale for a real vector.
- A wrong RoPE convention is caught by a test, not by a garbled answer twenty chapters later - and you know why your invariant test cannot do it.
- Your cost table says `exp` is the expensive part of softmax, with a number, instead of assuming these ops are free.

## Reading

- code: `src/vec_ops.rs`, `src/vec_ops/` (`rms_norm.rs`, `rope.rs`, `softmax.rs`, `silu.rs`, `bf16.rs`, `f16.rs`), `src/models/rope_scaling.rs` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/inference_e2e_walkthrough/11-attention-vecops-kv.md`
- decisions: ADR-0021 (bf16 rounds to nearest even, 1-D tensors stay f32), ADR-0074 (the Llama-3 RoPE table is read from the file)
- [online-softmax] Milakov and Gimelshein, *Online normalizer calculation for softmax* (2018) - arXiv:1805.02867 - the single-pass softmax that this chapter's measured `exp` cost motivates
- [ieee754] *IEEE 754-2019, Standard for Floating-Point Arithmetic* - no arXiv ID (standard) - the rounding rules behind the f16 and bf16 drift table

## Backfill

- [x] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - **answered for the pointwise ops, and the answer is not bandwidth.** A 32,768-element softmax row is 128 KB, which the machine could stream in 0.66 us; it takes 58.8 us, so the op is ninety times off its memory floor and bounded by one transcendental per element (1.17 ns through libm, 0.61 ns through a polynomial). The other ops are 0.05-0.2 ns per element and simply do not matter: RMSNorm, the three residual adds, SiLU and RoPE together are 0.14 ms per token against 2.42 ms of attention softmax at 4k context.
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - this chapter adds the other end of the same ledger: the *weights* may be 4-bit bytes because each is read once and rounded once, while the stream they accumulate into is f32 because it is rewritten every layer and the error compounds. The measured gap is `2.6e-07` for an f32 stream against `2.5e-02` for a bf16 one, after 24 adds.
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - the seam gains a precision axis, not only a dtype axis: which ops may run in f16 on a device, and which must widen. The "must not narrow" rows of the ledger are the ones a device backend has to honour or refuse.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - correctness gains a third implementation and a named limit: numpy in f64 agrees with the C f32 paths to `2.5e-07` on the widest op, and the witness is explicitly *unable* to catch a wrong RoPE pairing because it re-implements the same pairing. Measurement gains a failure test kept in the tree: the softmax without its max subtraction is written and run, so the guard has been seen working rather than assumed.
