# Chapter 9 · Vector Ops and the Precision Ledger

> **Compute tier**: ✅ any laptop | **Status**: stub
> **Build step S9**: RMSNorm, RoPE, softmax, SiLU and the elementwise ops your transformer needs, each with a precision note and a reference value.
> **Rule of this book**: Attention itself is Chapter 10; this chapter is the pointwise and reduction ops that surround it. Device-specific versions of these same ops appear in Chapters 21 and 24 but are not re-derived there.

## First, the gist

Between the matmuls a transformer does a handful of cheap operations: normalize a vector, rotate pairs of coordinates by position, exponentiate and normalize a score row, and gate a value through a sigmoid. Each is a few lines of arithmetic and each has a precision decision attached - what dtype the input is, what accumulates, what the output must be. Those decisions are where a correct engine quietly stops matching a reference, so this chapter keeps a ledger: for every op, the dtype in, the dtype of the accumulation, the dtype out, and the tolerance the reference demands.

## What you'll be able to do

- Implement RMSNorm, RoPE, softmax and SiLU from the equations, without a framework.
- Explain the two RoPE conventions in the wild and why a model's config decides which one it needs.
- Keep and read a precision ledger: which op widens, which narrows, and where a comparison must be bitwise instead of approximate.

## Lessons

- [ ] 1. RMSNorm: the equation, the epsilon, and why there is no mean subtraction
- [ ] 2. RoPE: pairs, rotations and positions as data rather than structure
- [ ] 3. Softmax: the max subtraction that is not optional, and its cost on a long row
- [ ] 4. SiLU, gating, and the elementwise ops that fuse into one pass (payoff: why fusion is a graph decision, not a kernel decision)
- [ ] 5. The precision ledger: f32 activations, f16 / bf16 weights, and the ops that must not narrow

## Precision ledger

| Op | Input dtype | Accumulate | Output dtype | Comparison against reference |
|---|---|---|---|---|
| RMSNorm |  |  |  |  |
| RoPE |  |  |  |  |
| softmax |  |  |  |  |
| SiLU / SwiGLU |  |  |  |  |
| residual add |  |  |  |  |

## Project

**Build step S9 · the ops you will reuse everywhere.** Write each op against a reference output captured from a known-good engine, not against your own expectations. The equations are simple; the conventions are not.

Record a golden vector per op (input bytes and expected output bytes). These become the identity fixtures in Chapter 27.

## Deliverable

- Five ops with golden-vector tests at the tolerances you chose and justified
- A precision ledger table, filled in, for your own implementation
- A note on the RoPE convention your target model uses and the config field that decides it

## How you know you passed

- Each op matches its golden vector within a stated, justified tolerance - and bitwise where the ledger says bitwise.
- You can state the epsilon in your norm and where it came from.
- A wrong RoPE convention is caught by a test, not by a garbled answer twenty chapters later.

## Reading

- code: `src/vec_ops.rs`, `src/vec_ops/` (rms_norm, rope, softmax, silu, bf16, f16), `src/models/rope_scaling.rs` at commit `88ffa5d`
- docs: `docs/inference_e2e_walkthrough/11-attention-vecops-kv.md`
- decisions: ADR-0021 (bf16 rounds to nearest even, 1-D tensors stay f32), ADR-0074 (the Llama-3 RoPE table is read from the file)
- [online-softmax] Milakov and Gimelshein, *Online normalizer calculation for softmax* (2018) - arXiv:1805.02867
- [ieee754] *IEEE 754-2019, Standard for Floating-Point Arithmetic* - no arXiv ID (standard)

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
