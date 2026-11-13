# Glossary

> - **Status**: stub - the entries are written; the cross-references will be added as chapters land.
> - **Rule of this book**: a term appears here once it is used by a chapter that leans on it. One-line entries for coinages with a single use are not entries, they are jargon.

| Term | Plain meaning | First used |
|---|---|---|
| **ADR** (architecture decision record) | One page per decision: what was chosen, what it rules out, what would reopen it. | [Decision map](adr-map.md) |
| **arena** | One large allocation handed out in overlapping slices, instead of one allocation per buffer. | [Ch 17](../17-allocation-liveness-and-offload/index.md) |
| **bits per weight** | File size divided by parameter count, in bits. 4.5 means a 7B model is about 4 GB. | [Ch 4](../04-quantized-blocks-the-base-formats/index.md) |
| **block** (quantization) | The group of weights that share one scale. 32 weights for legacy formats, 256 for K-quants. | [Ch 4](../04-quantized-blocks-the-base-formats/index.md) |
| **block** (CUDA) | A group of up to 1024 threads launched together; its threads can share memory. | [Ch 22](../22-cuda-the-machine/index.md) |
| **coalescing** | Neighbouring threads reading neighbouring addresses, so one memory transaction serves them all. | [Ch 22](../22-cuda-the-machine/index.md) |
| **CUDA Graph** | A recorded sequence of kernel launches replayed as one submission, removing per-launch overhead. | [Ch 25](../25-cuda-attention-kv-and-graphs/index.md) |
| **decode** | Producing one token from the whole prefix. Limited by memory bandwidth. | [Ch 1](../01-one-token-at-a-time/index.md) |
| **dp4a** | A GPU instruction that computes a 4-way 8-bit integer dot product and accumulates it. | [Ch 24](../24-cuda-the-kernel-ladder/index.md) |
| **GGUF** (GGML universal file format) | The container: header, metadata, tensor table, aligned data blob. | [Ch 2](../02-the-file-is-the-model/index.md) |
| **GEMM / GEMV** | General matrix-matrix / matrix-vector multiply. Prefill wants the first, decode the second. | [Ch 8](../08-gemv-gemm-and-the-thread-pool/index.md) |
| **GQA** (grouped-query attention) | Several query heads share one key/value head, shrinking the cache by that factor. | [Ch 10](../10-attention-and-kv-by-hand/index.md) |
| **identity gate** | The test that decides whether a change altered the answers, bitwise where the contract says so. | [Ch 27](../27-correctness-the-identity-gate/index.md) |
| **KV cache** | Stored keys and values for the prefix, so each step does not recompute them. | [Ch 10](../10-attention-and-kv-by-hand/index.md) |
| **liveness** | The span of nodes in which a buffer is still needed; what makes buffer reuse sound. | [Ch 17](../17-allocation-liveness-and-offload/index.md) |
| **MMQ / MMVQ** | Quantized matrix-matrix (prefill) and matrix-vector (decode) kernels on a GPU. | [Ch 24](../24-cuda-the-kernel-ladder/index.md) |
| **noise floor** | The spread an unchanged run shows on a machine; a delta below it is not a result. | [Ch 28](../28-speed-you-can-defend/index.md) |
| **occupancy** | Fraction of a streaming multiprocessor's capacity a kernel actually keeps busy. | [Ch 22](../22-cuda-the-machine/index.md) |
| **offload** | Keeping some layers' weights in host memory and running the graph across the boundary. | [Ch 17](../17-allocation-liveness-and-offload/index.md) |
| **prefill** | One pass over the whole prompt. Limited by arithmetic, not bandwidth. | [Ch 1](../01-one-token-at-a-time/index.md) |
| **quantization** | Storing weights (and activations) in blocks with a scale, instead of 16-bit floats. | [Ch 4](../04-quantized-blocks-the-base-formats/index.md) |
| **roofline** | The chart, and the arithmetic, that says whether a workload is bound by compute or by bytes moved. | [Ch 1](../01-one-token-at-a-time/index.md) |
| **SDOT** | An ARM instruction computing a 4-way 8-bit integer dot product. | [Ch 7](../07-simd-dispatch-in-anger/index.md) |
| **SIMD / SIMT** | One instruction over many data lanes (CPU) / over many threads in lockstep (GPU). | [Ch 7](../07-simd-dispatch-in-anger/index.md), [Ch 22](../22-cuda-the-machine/index.md) |
| **speculation** | A small model proposes tokens, the big model verifies several in one pass, survivors are kept. | [Ch 29](../29-serving-and-speculation/index.md) |
| **staging buffer** | A host-side or pinned buffer that holds data between two memory spaces. | [Ch 25](../25-cuda-attention-kv-and-graphs/index.md) |
| **tensor core** | A GPU unit that multiplies small matrices in one instruction; used by the f16 and int8 paths. | [Ch 24](../24-cuda-the-kernel-ladder/index.md) |
| **tolerance class** | A named, written-down comparison bound for cases where bitwise equality is not the contract. | [Ch 27](../27-correctness-the-identity-gate/index.md) |
| **warp** | 32 CUDA threads that execute one instruction in lockstep; divergence splits them. | [Ch 22](../22-cuda-the-machine/index.md) |
| **WMMA** | The CUDA interface to tensor-core matrix fragments, with warp-level layouts. | [Ch 24](../24-cuda-the-kernel-ladder/index.md) |

## Reading

- `docs/GLOSSARY.md` in the reference repository - the campaign's own vocabulary, classified by layer
- `docs/CUDA-TECH-PRIMER.md` - reference depth for every GPU technique named above
