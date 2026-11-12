# Audit: the CUDA Tutorial Against This Commit

> **Status**: draft - the mechanical findings are re-derivable from the commands below at the [pinned baseline](baseline.md); the chapter-by-chapter judgement is still being written.
> **Rule of this book**: the CUDA tutorial in the reference project is a contributor onboarding series. This book re-derives its content from the current tree instead of copying it, because a tutorial's line references and counts drift while its prose keeps reading as true.

This book's CUDA part (Chapters 22-26) covers the same ladder as the project's tutorial (`docs/cuda_tutorial/`: seven chapters plus an index and a writing contract, 9 files, 5250 lines) and differs in three ways: it teaches the Rust boundary as its own chapter, it treats the optimization campaign as a method rather than a technique catalogue, and it cites **symbols plus a commit** instead of line numbers. Before writing any of it, the tutorial was checked against the tree. What follows is what that check found.

## Method

1. Extract every `src/...` path with an optional line number from the tutorial's seven chapters.
2. For each, check the file exists, and that the cited line is inside the file.
3. For a sample of named kernels, check the symbol still exists in `src/cuda/kernels/`.
4. For counts the tutorial states about the project, re-derive them.

```bash
# anchors: paths and optional line numbers, per chapter
rg -o 'src/[A-Za-z0-9_./-]+\.(rs|cu|cuh|metal)(:[0-9]+)?' docs/cuda_tutorial/*.md | sort -u | wc -l
# kernel symbols cited
rg -o '[a-z_0-9]+_[a-z_0-9]+_kernel[a-z_0-9]*' docs/cuda_tutorial/*.md | sort -u
rg -l '__global__' src/cuda/kernels/*.cu | wc -l
# counts the tutorial states about the project
rg -n '70\+' docs/cuda_tutorial/ docs/CUDA_OPTIMIZATION.md
ls docs/cuda_optimization_steps/[0-9]*.md | wc -l
```

## Finding 1 - three source references point at files that do not exist

142 distinct source anchors were found in the seven chapters. 139 resolve. The three that do not are all the same class of drift, and all in the chapters a new contributor reads first:

| Chapter | Reference | What the tree has |
|---|---|---|
| `02-minimal-cuda.md` | `src/cuda_kernels.cu` | no such file: the kernels live in `src/cuda/kernels/*.cu`, and the Rust side in `src/cuda.rs` plus `src/cuda/methods/*.rs` |
| `03-kernels-elementwise.md` | `src/cuda/kernels/common.cu` | `src/cuda/kernels/common.cuh` (a header, not a translation unit) |
| `04-kernels-matmul.md` | `src/cuda/kernels/common.cu` | same |

This is the failure mode a book has to design against rather than fix once: the tutorial's own convention already says the function name is the stable address and the line number is a convenience, but a *path* has the same problem and no such warning. Chapters 22-25 therefore cite `src/cuda/kernels/<file>.cu` plus a symbol, and name the commit they were read at.

## Finding 2 - a stale count, stated three times

The tutorial describes the optimization campaign as holding `70+` step records in three places: its index, its writing contract (`STYLE.md`), and the reading-order table in `07-where-next.md`. The tree holds **110**:

```bash
ls docs/cuda_optimization_steps/[0-9]*.md | wc -l   # 110
rg -n '70\+' docs/cuda_tutorial/                     # three files
```

The number is not wrong in the sense of a broken fact - 110 is more than 70 - but a reader who plans by it underestimates the record by a third. Three copies of one count, none of them derived, is how a document tells a reader it has not been re-read. This book states the count with its command on the [Measurements](measurements.md) page.

## Finding 3 - what the tutorial gets right, and this book keeps

Checked, not assumed:

- Every kernel symbol sampled (`fa_prefill_kv`, `attn_bias_rope_store_f32`, `gemm_f16_nt_kernel_t`, `embed_rows_q4_0`, `dequant_q4_0_f16`, `add_f32`) exists in the tree, each in the file the tutorial names.
- The `__global__` census still works: 108 kernels across 17 `.cu` files (`grep -h '__global__' src/cuda/kernels/*.cu | wc -l`).
- Its stated machine (an NVIDIA GB10 with CUDA 13.0) matches the machine the project's campaign records were taken on, and the toy programs' outputs are presented as observed runs rather than quoted from documentation.

The tutorial's toys (`toy1_vec_add.cu` and friends) are embedded in the chapters and are **not** expected in the tree; a scan for them reports them missing, which is not a defect. Recording that here is the point: a finding list that includes its own false positives is usable.

## How this book's CUDA part maps onto the tutorial

| Tutorial chapter | This book | What changes |
|---|---|---|
| 01 - what a GPU is | [Chapter 22](../22-cuda-the-machine/index.md) | keeps the mental model, adds the launch sheet the reader fills before coding |
| 02 - minimal CUDA | [Chapter 23](../23-cuda-the-rust-seam/index.md) | becomes a chapter about the Rust boundary: ownership, pools, bounded waits, linking |
| 03-05 - reading kernels | [Chapters 24-25](../24-cuda-the-kernel-ladder/index.md) | reordered into a ladder (elementwise, dequant, GEMM, integer matmul, attention, graphs), each rung gated on identity |
| 06 - optimization methods | [Chapter 26](../26-the-optimization-campaign/index.md) | becomes a campaign method with a land/revert decision rule, and the reverted levers are the content |
| 07 - where next | [Appendix: reading](reading.md) | merged into the general reading page |

## Reading

- `docs/cuda_tutorial/` at the pinned baseline - the tutorial this page audits, still the best short on-ramp in the project
- `docs/CUDA-TECH-PRIMER.md`, `docs/GLOSSARY.md` - reference depth, and the campaign's own vocabulary
- `docs/cuda_optimization_steps/77-verification-methodology.md` - the method Chapter 26 teaches
