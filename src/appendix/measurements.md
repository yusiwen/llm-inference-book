# Measurements

> **Status**: stub - the numbers below are real and were produced by the commands beside them, on the date and commit named. The page will grow as chapters report their own measurements.
> **Rule of this book**: a number lives on this page only with the command that produced it and the commit it was read at. A number without those is a claim, not a measurement.

## The book itself

Read 2026-11-12 at commit of this repository that first contains this page.

| Quantity | Value | Command |
|---|---|---|
| Chapters | 29 | `ls -d src/[0-9][0-9]-*/ \| wc -l` |
| Lessons (checklist items under `## Lessons`) | 146 | counted per page by `scripts/probe.py` |
| Pages under `src/` | 43 | `find src -name '*.md' \| wc -l` |
| Prose bytes under `src/` (all stubs) | 208515 | `find src -name '*.md' \| xargs wc -c \| tail -1` |
| Status | all 29 chapter pages `stub` | `rg -l 'Status\*\*: stub' src/[0-9][0-9]-*/index.md \| wc -l` |
| Sources in the reference index | 29 | `python3 scripts/references.py` (prints the count) |
| minfer citations tracked | 142 | `python3 scripts/minfer-drift.py --check-inventory reference/baseline-citations.json` |
| Behaviour probes recorded | 1 run, 1 probe executed / 7 defined | `ls runs/*.json`, `bash scripts/contract-probes.sh --list` |
| Figures planned / drawn | 46 / 0 | `FIGURES.md`, `ls src/figures/*.svg \| wc -l` |
| Reader starter skeleton | 6 crates, 0 dependencies, builds offline | `cp -R starter /tmp/x && cd /tmp/x && cargo build && cargo test` |

None of these are performance numbers. A chapter count is a fact about an outline, not evidence that the book teaches anything.

## The reference project

Read at the [pinned baseline](baseline.md) (2026-11-11) in `github.com/yusiwen/minfer`. The counts are re-derived by `scripts/minfer-drift.py`, which compares them against `minfer.lock`.

| Quantity | Value | Command |
|---|---|---|
| Rust lines in the engine | 107726 | `find src -name '*.rs' \| xargs wc -l \| tail -1` |
| Rust lines in the CLI package | 26662 | `find crates/minfer-cli/src -name '*.rs' \| xargs wc -l \| tail -1` |
| Architecture decision records | 82 | `ls docs/adr/[0-9]*.md \| wc -l` |
| CUDA step records | 110 | `ls docs/cuda_optimization_steps/[0-9]*.md \| wc -l` |
| CUDA kernel sources | 17 `.cu` (+1 `.cuh` header) | `ls src/cuda/kernels/*.cu \| wc -l` |
| `__global__` kernels in those files | 108 | `grep -h '__global__' src/cuda/kernels/*.cu \| wc -l` |
| Unit tests in the engine crate | 749 | `grep -rn '#\[test\]' src --include=*.rs \| wc -l` |
| Integration tests | 24 | `grep -rn '#\[test\]' tests --include=*.rs \| wc -l` |

**These numbers were re-derived, not copied.** The project's own README, its architecture document and its support matrix disagree with each other about several counts, which is exactly why this page states its commands. When a chapter quotes one of these numbers, this page is the source.

## What this page does not have yet

| Missing | Why it is missing | Where it will come from |
|---|---|---|
| Throughput on any machine | no chapter is written, and this book's author has not run a benchmark for it | Chapter 28's harness, on the machine it names |
| Any accuracy or tolerance figure | the identity gate is Chapter 27's deliverable | that chapter's fixtures |
| Build time and binary size | not measured | `time cargo build --release`, recorded with the toolchain |
| Anything marked *reported* from the project's campaign | quoted from `docs/cuda_optimization_steps/`, never re-run here | the step record it cites |

## Reading

- `docs/TEST-BASELINES.md` and `docs/CUDA_OPTIMIZATION.md` in the reference repository - what the project measures about itself
- [Chapter 28](../28-speed-you-can-defend/index.md) - the harness every future number on this page must come from
