# Summary

[Home](index.md)

[Part 0 · Before you start](start/index.md)

---

# Part 1 · The Ground Truth

- [Chapter 1 · One Token at a Time](01-one-token-at-a-time/index.md)
- [Chapter 2 · The File Is the Model](02-the-file-is-the-model/index.md)
- [Chapter 3 · Tensors, Strides, and Bytes](03-tensors-strides-and-bytes/index.md)
- [Chapter 4 · Quantized Blocks: The Base Formats](04-quantized-blocks-the-base-formats/index.md)
- [Chapter 5 · K-Quants and Byte Identity](05-k-quants-and-byte-identity/index.md)

# Part 2 · Arithmetic

- [Chapter 6 · The Dot Product Is the Engine](06-the-dot-product-contract/index.md)
- [Chapter 7 · SIMD Dispatch in Anger](07-simd-dispatch-in-anger/index.md)
- [Chapter 8 · GEMV, GEMM, and the Thread Pool](08-gemv-gemm-and-the-thread-pool/index.md)
- [Chapter 9 · Vector Ops and the Precision Ledger](09-vector-ops-and-precision/index.md)
- [Chapter 10 · Attention and the KV Cache by Hand](10-attention-and-kv-by-hand/index.md)

# Part 3 · Language

- [Chapter 11 · The Tokenizer](11-the-tokenizer/index.md)
- [Chapter 12 · Chat Templates and Special Tokens](12-chat-templates/index.md)
- [Chapter 13 · Sampling and Constrained Decoding](13-sampling-and-constrained-decoding/index.md)

# Part 4 · The Graph

- [Chapter 14 · From Imperative Loop to Declarative Graph](14-the-declarative-graph/index.md)
- [Chapter 15 · Ops, Nodes, and Fusion](15-ops-nodes-and-fusion/index.md)
- [Chapter 16 · Backend Assignment and the Registry](16-backend-assignment-and-registry/index.md)
- [Chapter 17 · Allocation: Liveness, Arena, and Offload](17-allocation-liveness-and-offload/index.md)
- [Chapter 18 · The KV Cache Subsystem](18-the-kv-cache-subsystem/index.md)
- [Chapter 19 · Architectures as Graph Construction](19-architectures-as-graph-construction/index.md)
- [Chapter 20 · The Engine Loop](20-the-engine-loop/index.md)

# Part 5 · Devices

- [Chapter 21 · Metal: The First Backend](21-metal-the-first-backend/index.md)
- [Chapter 22 · CUDA I: What Kind of Machine a GPU Is](22-cuda-the-machine/index.md)
- [Chapter 23 · CUDA II: The Rust-CUDA Seam](23-cuda-the-rust-seam/index.md)
- [Chapter 24 · CUDA III: The Kernel Ladder](24-cuda-the-kernel-ladder/index.md)
- [Chapter 25 · CUDA IV: Attention, KV, and CUDA Graphs on Device](25-cuda-attention-kv-and-graphs/index.md)
- [Chapter 26 · CUDA V: The Optimization Campaign](26-the-optimization-campaign/index.md)

# Part 6 · Proving It

- [Chapter 27 · Correctness: The Identity Gate](27-correctness-the-identity-gate/index.md)
- [Chapter 28 · Speed You Can Defend](28-speed-you-can-defend/index.md)
- [Chapter 29 · Serving, and the Speculation Trade](29-serving-and-speculation/index.md)

---

# Appendix

- [Writing and Formatting Rules](appendix/conventions.md)
- [The Decision Map](appendix/adr-map.md)
- [Measurements](appendix/measurements.md)
- [Audit: the CUDA tutorial against this commit](appendix/cuda-docs-audit.md)
- [Glossary](appendix/glossary.md)
- [Limits: what this engine cannot do](appendix/limits.md)
- [Where the code lives](appendix/reading.md)
