# Chapter 5 · K-Quants and Byte Identity

> **Compute tier**: ✅ any laptop | **Status**: stub | **Build step S5**: A Q4_K / Q5_K / Q6_K dequantizer, plus a quantizer whose output is byte-identical to the reference tool's. | **Rule of this book**: The super-block idea is explained here, once, in full. What the kernels do with it on a GPU is Chapter 24, and the campaign that optimized those kernels is Chapter 26.

## First, the gist

K-quants put eight 32-weight blocks inside one 256-weight super-block, and store the eight scales as 6-bit fields packed into 12 bytes. The point is not compression for its own sake: a super-block shares one scale pair across a wider window, so the scale overhead per weight drops while the precision stays usable. The sting is in the packing - two 12-byte planes sharing 6-bit scalars - and in the fact that a quantizer that is merely close is useless, because a file that differs by one byte cannot be compared against a reference run.

## What you'll be able to do

- Dequantize a K-quant super-block by hand, including the 6-bit scale unpacking.
- Prove a quantizer correct by byte equality against the reference tool, not by tolerance.
- Explain when Q6_K earns its extra 2 bits per weight, and when Q4_K is the better file.

## Lessons

- [ ] 1. Super-blocks: eight sub-blocks, one scale pair, and the arithmetic that motivates it
- [ ] 2. The 6-bit scale packing, unpacked on paper before it is unpacked in code
- [ ] 3. Q6_K and the high-bit planes: 6.5625 bits per weight and why anybody pays it
- [ ] 4. Byte-identical quantization: why a tolerance gate is the wrong test for a file format (payoff: how a one-byte drift destroys comparability)
- [ ] 5. Reading a quantized file's provenance: what the metadata does and does not record

## Super-block cost

| Format | Super-block | Bytes per super-block | Bits per weight | Scale encoding |
|---|---|---|---|---|
| Q4_K | 256 |  |  |  |
| Q5_K | 256 |  |  |  |
| Q6_K | 256 |  |  |  |

## Project

**Build step S5 · the quantizer nobody can argue with.** Take one small source tensor, quantize it with your implementation and with the reference tool, and diff the bytes. This is the first identity gate in the book; Chapter 27 makes it a habit.

When the bytes differ, do not weaken the test. Bisect the difference to a field (a scale, a nibble order, a rounding mode) and fix the field.

## Deliverable

- Q4_K, Q5_K and Q6_K dequantizers with per-format tests
- A quantize-then-diff run producing zero differing bytes on one tensor, with the command recorded
- A one-page decision note: which K-quant your engine defaults to and on what evidence

## How you know you passed

- `cmp` reports the two quantized files identical, and the command is in your notes.
- You can unpack a 6-bit scale field on paper, without the code.
- You can name a case where Q6_K is the wrong choice, and it is not "it is bigger".

## Reading

- code: `src/quants/kquant.rs`, `src/quantize.rs`, `src/q4k_dsc.rs` at commit `88ffa5d`
- docs: `docs/GGUF-TOOLING.md`, `docs/inference_e2e_walkthrough/10-cpu-matmul-kernels.md`
- decision: ADR-0020 (a quantized file is byte-identical to llama-quantize, or it is wrong)
- [ggml-quant] The ggml quantized block layouts and the k-quant design notes, in the ggml repository - no arXiv ID (specification)
- [kbit] Dettmers and Zettlemoyer, *The case for 4-bit precision: k-bit Inference Scaling Laws* (2022) - arXiv:2212.09720

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
