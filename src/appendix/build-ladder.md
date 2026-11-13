# The Build Ladder

> **Status**: written - the table is the ordering contract the chapters were written against; `scripts/probe.py` fails when a chapter and its row disagree.
> **Rule of this book**: a build step names what exists in *your* tree when it is done. Every chapter owns exactly one, and this page is the only place they are all visible at once.

Thirty steps, one per page, in the order they compile. Each row is a promise about your own engine, not about minfer: the reference implementation is read *after* the step is attempted, which is what "the hard way" means.

The order is not a suggestion. The failure this table prevents is the one nobody notices until chapter 20: a graph builder written before the tensor type has stable strides, or an allocator written before the registry that tells it how many devices exist, ends up being rewritten rather than extended.

| Step | Page | Produces | Needs | If you skip it |
|---|---|---|---|---|
| S0 | Before you start | a layered cargo workspace, created empty, and a notes file with four sections | - | every chapter has somewhere to put its code |
| S1 | One Token at a Time | a loop against a placeholder model, and a decode ceiling computed by hand | S0 | no sense of what a step costs before you write one |
| S2 | The File Is the Model | a GGUF reader printing the header, every metadata key and the tensor table | S0 | nothing to load: every later step needs weights |
| S3 | Tensors, Strides, and Bytes | a tensor type with views, plus the alignment rule written down | S0 | matmuls get hand-rolled offsets and silently read the wrong row |
| S4 | Quantized Blocks: The Base Formats | dequantizers for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | S2, S3 | no way to turn file bytes into numbers |
| S5 | K-Quants and Byte Identity | Q4_K/Q5_K/Q6_K dequantizers and a quantizer that matches the reference byte for byte | S4 | the 7B-class files this book measures cannot be read |
| S6 | The Dot Product Is the Engine | one quantized dot product, scalar, with a dispatch table and the refusal rule | S3, S4 | the oracle every faster kernel is checked against never exists |
| S7 | SIMD Dispatch in Anger | AVX2 and NEON kernels for one format, dispatched at build time | S6 | the book's speed numbers stay one instruction set wide |
| S8 | GEMV, GEMM, and the Thread Pool | two matmul shapes, a bounded pool, and a measured cross-over point | S6, S7 | prefill and decode share one kernel that suits neither |
| S9 | Vector Ops and the Precision Ledger | RMSNorm, RoPE, softmax, SiLU, and the ledger of what widens and what narrows | S3 | attention has no parts to be built from |
| S10 | Attention and the KV Cache by Hand | causal, grouped-query attention over an explicit cache, prefill and decode | S9, S3 | no transformer, and no cache to make it affordable |
| S11 | The Tokenizer | encode and decode driven by the model file, with a conformance fixture set | S2 | prompts cannot become ids |
| S12 | Chat Templates and Special Tokens | a renderer that matches the reference byte for byte, and refuses when it cannot | S2, S11 | the model answers, badly, and nobody can say why |
| S13 | Sampling and Constrained Decoding | the sampler pipeline, seeded and reproducible, plus one grammar mask | S1 | logits never become tokens |
| S14 | From Imperative Loop to Declarative Graph | build and execute as two passes, with a graph you can dump and read | S3 | nothing can be assigned, allocated, fused or reused |
| S15 | Ops, Nodes, and Fusion | a comparable op payload and one guarded fusion with an unfused oracle | S14 | no pattern to match, and no way to prove the rewrite safe |
| S16 | Backend Assignment and the Registry | a registry, a deterministic assignment, split boundaries and two refusals | S14, S15 | a second device becomes a second engine |
| S17 | Allocation: Liveness, Arena, and Offload | liveness-based buffer reuse, an offload fit, and a memory ledger you can predict | S16 | peak memory is set by model depth, not by need |
| S18 | The KV Cache Subsystem | cells, spans, prefix sharing, defragmentation, and a versioned session file | S10, S17 | long contexts fragment and sessions cannot resume |
| S19 | Architectures as Graph Construction | two model families from one graph vocabulary, with the differences named | S14, S18 | the engine fits exactly one model file |
| S20 | The Engine Loop | prefill, decode, graph reuse, append-only multi-turn, cancellation and overflow | S8, S13, S19 | the pieces exist but nothing generates |
| S21 | Metal: The First Backend | a device backend behind the registry, with three shaders past the identity gate | S16, S20 | the engine is CPU-only on a Mac that has a GPU |
| S22 | CUDA I: What Kind of Machine a GPU Is | two compiled kernels and a launch sheet you wrote before the code | a GPU + nvcc | kernel timing is a mystery rather than a prediction |
| S23 | CUDA II: The Rust-CUDA Seam | owned device buffers, errors as results, and a bounded wait | S22 | the first device fault hangs the process instead of reporting |
| S24 | CUDA III: The Kernel Ladder | elementwise, dequant, tiled GEMM, integer matmul and matrix-vector kernels | S6, S8, S23 | prefill runs at a fraction of the device's arithmetic throughput |
| S25 | CUDA IV: Attention, KV, and CUDA Graphs on Device | flash-style prefill attention, a fused decode tail, and graph capture and replay | S10, S18, S24 | decode pays per-launch overhead on every one of its hundred kernels |
| S26 | CUDA V: The Optimization Campaign | a cost model, three step records including one reverted, and a decision rule | S25, S8 | optimizations land on vibes and cannot be re-audited |
| S27 | Correctness: The Identity Gate | hashed fixtures, per-op oracles, tolerance classes, and a bisection that names the node | S20 | no change can be called safe, so none can be called faster |
| S28 | Speed You Can Defend | a harness with warmup, interleaved A/B, medians and a stated noise floor | S20, S27 | numbers exist but cannot be compared to anything |
| S29 | Serving, and the Speculation Trade | slots, a bounded worker, metrics, a drain path, and a speculative decoder behind a flag | S20, S27, S28 | one user at a time, and every speedup unmetered |

## Three lanes through the ladder

- **The core ladder (S0-S20, S27-S29)** runs on any laptop. It is the book, and it is enough to have a working CPU engine that loads a real GGUF file, generates tokens, and can prove both.
- **The device lane (S21-S26)** needs hardware: S21 wants a Mac, S22-S26 want an NVIDIA GPU. A reader without one reads these chapters and skips the probes - the seam is the transferable part, and Chapter 21 says so in its first line.
- **The proof lane (S27-S28)** is deliberately late. It could be moved earlier, and if your engine is going to grow past a toy you probably should: everything after S20 is easier to get wrong when there is no identity gate to catch it.

## What each step leaves undone

A row's "produces" is a floor, not a finish line. Three deliberate gaps recur, and each chapter names its own:

- **One error path, not all of them.** A step handles the failure it teaches; the rest are named in the chapter's "how you know you passed".
- **One shape, not every shape.** GEMV and GEMM are written for the shapes the book measures; a general dispatch comes later or never.
- **No performance claim.** Speed is S28's job. A step can be slow and correct, and that is the intended order.

## Reading

- [The reference machines](machines.md) - which step runs where, and on what number
- [The minfer Baseline](baseline.md) - the tree each step is compared against
- [Writing and Formatting Rules](conventions.md) - why the build step is a fixed part of the chapter skeleton
