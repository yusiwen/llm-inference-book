# Limits: What This Engine Cannot Do

> - **Status**: stub - the structure and the pointers are written; the per-row verdicts wait on the chapters that would discover them.
> - **Rule of this book**: this page names the *shapes* of gap and points at the project's own inventories. It does not copy their rows: a copied gap list drifts against its source and nothing says which copy is stale.

A book that builds an engine has to say what the finished thing is not, or the reader discovers it on their own time. There are two different lists here, and they should not be confused.

## 1. What this book deliberately does not cover

| Not covered | Why | Where to go instead |
|---|---|---|
| Training, fine-tuning, LoRA | A different pipeline, different arithmetic, different failure modes | the author's other handbook on post-training |
| Mixture-of-experts models | A routing layer between the graph and the kernels; a book of its own | the project's model roadmap lists it as unbuilt |
| Multi-modal (vision, audio) input | A second encoder and a different prompt path | not planned in this book |
| State-space and hybrid architectures | Different attention semantics, so different kernels | ditto |
| Multi-node serving, disaggregated prefill/decode | A distributed systems problem, not an engine one | outside scope |
| Model quality evaluation | This book measures whether the engine is *correct*, never whether the model is *good* | any evaluation harness |

## 2. What the reference engine knows it cannot do

The reference project keeps its own inventory, and it is the authoritative one - it has dated entries, issue numbers, and a maintainer who can be wrong in public. Read it there, not here:

| Question | The project's own answer lives in |
|---|---|
| Which quantized formats work on which backend? | `docs/SUPPORT-MATRIX.md` |
| Which model families are supported, and which are planned? | `docs/MODEL-SUPPORT-ROADMAP.md` |
| What is known to be broken right now? | `docs/KNOWN-CPU-ISSUES-*.md` (dated filenames) |
| What is the system-level backlog? | `docs/ARCHITECTURE-ROADMAP.md` |
| What did the last campaign fail to fix? | `docs/CUDA_OPTIMIZATION.md` section 0, the REVERTED rows |

**One number from that inventory, because it is the most misread one**: the CPU path does not have a fast kernel for every format on every instruction set. The support matrix's per-instruction-set columns are the accurate account, and the count has changed more than once. Do not quote a count from this page; quote the command from [Measurements](measurements.md) that derives it.

## 3. What the book's own build steps will not give you

Each build step leaves something undone on purpose, and the chapter names it. The classes, collected here so a reader can see the pattern before starting:

- **Accuracy**: a worked engine that passes the identity gate against itself is not a correct engine. Only a comparison against an independent implementation (the reference engine, or a reference run's recorded output) settles that, and that comparison is Chapter 27's deliverable.
- **Speed**: a chapter measures one shape on one machine. Nothing in this book is a throughput claim about your hardware until you run the harness of Chapter 28 on it.
- **Robustness**: a build step typically has one error path. The book says which failure modes are handled and which are deliberately not, per chapter.
- **Generality**: two dense model families appear. A third family is an exercise, and Chapter 19 estimates its cost rather than pretending it is free.

## Reading

- `docs/ARCHITECTURE-ROADMAP.md`, `docs/MODEL-SUPPORT-ROADMAP.md`, `docs/SUPPORT-MATRIX.md` - the project's inventories
- [Chapter 19](../19-architectures-as-graph-construction/index.md) - what a new architecture costs
- [Chapter 28](../28-speed-you-can-defend/index.md) - why no number here is a claim about your machine
