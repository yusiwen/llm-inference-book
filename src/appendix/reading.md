# Where the Code Lives

> **Status**: draft - the pointers are correct at the [pinned baseline](baseline.md); the annotated tour of each directory grows as chapters cite it.
> **Rule of this book**: every code pointer is a path plus a symbol plus the commit it was read at. Line numbers are never published.

## The reference engine

- Repository: `github.com/yusiwen/minfer`
- Commit this book describes: the [pinned baseline](baseline.md), recorded in `minfer.lock`
- Documentation site: `yusiwen.cn/minfer`

Later commits exist. When the tree and a chapter disagree, the chapter is the thing that needs updating, and the symbol is how you find where.

## How the engine is laid out

| Path | What it holds | Book chapter |
|---|---|---|
| `src/gguf.rs`, `src/tensor.rs`, `src/block.rs` | the container, tensors, block layouts | 2-4 |
| `src/quants/`, `src/kernel/`, `src/vec_ops.rs` | quantized dots, dispatch, the thread pool, vector ops | 5-9 |
| `src/graph/` | the compute graph: IR, builder, scheduler, allocator, KV, backends, fusion, cache | 14-18 |
| `src/models/qwen2/`, `src/models/qwen3/` | architecture as graph construction | 19 |
| `src/tokenizer.rs`, `src/template.rs`, `src/sampler.rs`, `src/grammar.rs` | text and token choice | 11-13 |
| `src/engine/`, `src/conversation.rs`, `src/spec.rs` | the generation loop and conversation state | 20, 29 |
| `src/metal/`, `src/metal/kernels/*.metal` | the Apple backend | 21 |
| `src/cuda.rs`, `src/cuda/methods/`, `src/cuda/kernels/*.cu` | the NVIDIA backend | 22-25 |
| `src/device_tier.rs` | per-architecture tables and the gates they feed | 26 |
| `crates/minfer-cli/src/` | CLI, server, bench, viz host, tooling | 20, 28, 29 |
| `docs/adr/` | 82 decision records | [Decision map](adr-map.md) |
| `docs/cuda_optimization_steps/` | 110 optimization step records | 26 |
| `docs/inference_e2e_walkthrough/` | one run, one document per stage | all parts |
| `docs/cuda_tutorial/` | the contributor CUDA tutorial this book audits | [Audit](cuda-docs-audit.md) |

## Reading the project's documentation without being misled

Three habits, learned from the audit in this appendix:

1. **A date, not a version.** The project's documents each name the commit their line numbers were verified against, and those commits differ by weeks. Treat every line number as dated; use the symbol.
2. **A count needs its command.** The project's own files disagree about how many step records, kernels and formats there are. Re-derive, and say how.
3. **A gap list is the most perishable page.** Read `docs/SUPPORT-MATRIX.md` as of the commit you are on, never from memory.

## Keeping up

```bash
git clone https://github.com/yusiwen/minfer
cd minfer && git checkout $(grep -oE '[0-9a-f]{40}' ../llm-inference-book/minfer.lock | head -1)
cargo build --release                  # CPU + Metal on macOS
cargo build --release --features cuda  # on an NVIDIA machine
```

## Reading

- `docs/BUILD.md` - toolchain, features, and the linking choices behind `cuda` and `cuda_static`
- `docs/GATE-CONTRACT.md` - what counts as done in that repository
- [Measurements](measurements.md) - how this book's counts were produced
