# Chapter 4 · Quantized Blocks: The Base Formats

> **Compute tier**: ✅ any laptop | **Status**: stub
> **Build step S4**: A dequantizer for Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 that turns one block into 32 floats, plus a test that reconstructs a known vector.
> **Rule of this book**: This chapter covers the legacy block formats only (the `Q4_0` family and `Q8_0`). The K-quants get Chapter 5 because their super-block structure is a different idea, not a bigger version of this one.

## First, the gist

A 7B model in 16-bit floats is 14 GB; the same model at 4.5 bits per weight is about 4 GB. Quantization is how that trade is made: weights are stored in small blocks, each block carries a scale, and a kernel reconstructs the values while multiplying instead of before. The legacy formats are the right place to start because one block is one struct you can hold in your hand: 32 weights, one 16-bit scale, 16 or 32 bytes of payload. Everything that follows - K-quants, activation quantization, device kernels - is a variation on that block.

## What you'll be able to do

- Read a GGUF tensor's bytes as blocks and reconstruct exact f32 values, matching a reference implementation bit for bit.
- Compute bits per weight and the resulting file size for a model, and say what a 4-bit file actually costs in accuracy terms.
- Explain why a block scale has to be a power-of-two-friendly f16, and what goes wrong when a block's range is an outlier.

## Lessons

- [ ] 1. Why 32 weights per block: the scale overhead arithmetic
- [ ] 2. Q4_0 and Q4_1: nibbles, offsets, and a scale that is a half float
- [ ] 3. Q5_0 and Q5_1: the high-bit plane, and what one extra bit per weight buys
- [ ] 4. Q8_0: the format activations are quantized into, and why the pair Q8_0 x Q4_0 matters
- [ ] 5. Outliers, clipping, and the error a block sees: a worked example (payoff: why 4-bit models are usable at all)

## Block anatomy and cost

| Format | Block size | Bytes per block | Bits per weight | Payload layout |
|---|---|---|---|---|
| Q4_0 | 32 |  |  |  |
| Q4_1 | 32 |  |  |  |
| Q5_0 | 32 |  |  |  |
| Q5_1 | 32 |  |  |  |
| Q8_0 | 32 |  |  |  |

## Project

**Build step S4 · dequantize, then believe it.** Dequantization is the first place where being approximately right is indistinguishable from being wrong: a rounding mode or a mask off by one produces plausible numbers. Compare against a reference on the same file, not against your own earlier output.

Write the dequantizer twice: once scalar and obviously correct, once with the SIMD path you will need in Chapter 7. The scalar one is the oracle.

## Deliverable

- A dequantizer for all five formats with a per-format round-trip test
- A table of measured file sizes for one model at each format, and the bits-per-weight you computed for each
- A note naming the format your engine will use for weights and for activations, and why they differ

## How you know you passed

- Reconstructed blocks match the reference implementation's f32 output within one unit in the last place, and you state which comparison you used.
- Your bits-per-weight numbers agree with the block sizes you wrote down, and the model-size arithmetic closes.
- A single outlier weight in a block visibly degrades its neighbours in your worked example, and you can say by how much.

## Reading

- code: `src/block.rs`, `src/quants.rs` and `src/quants/dot_q4_0.rs`, `dot_q4_1.rs`, `dot_q5.rs`, `dot_q8_0.rs` at commit `88ffa5d`
- docs: `docs/GGUF-TOOLING.md` (convert and quantize), `docs/inference_e2e_walkthrough/10-cpu-matmul-kernels.md`
- [kbit] Dettmers and Zettlemoyer, *The case for 4-bit precision: k-bit Inference Scaling Laws* (2022) - arXiv:2212.09720
- [ggml-quant] The ggml quantized block layouts and the k-quant design notes, in the ggml repository - no arXiv ID (specification)

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
