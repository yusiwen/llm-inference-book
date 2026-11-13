# Learning LLM Inference the Hard Way

### Building an Inference Engine from Scratch in Rust

<p align="center">

[![mdBook 0.5](https://img.shields.io/badge/mdBook-0.5-informational?logo=rust&logoColor=white)](#)
[![Language: English](https://img.shields.io/badge/text-English-blue)](#)
[![Chapters: 29](https://img.shields.io/badge/chapters-29-lightgrey)](#)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-green)](LICENSE)
[![Publish the book](https://github.com/yusiwen/llm-inference-book/actions/workflows/pages.yml/badge.svg)](https://github.com/yusiwen/llm-inference-book/actions/workflows/pages.yml)
[![Reference engine](https://img.shields.io/badge/reference-minfer-orange?logo=rust&logoColor=white)](https://github.com/yusiwen/minfer)

</p>

**Read it online: [yusiwen.cn/llm-inference-book](https://yusiwen.cn/llm-inference-book/)** - the built
site is published from this repository on every push to `main`, so that address is always the current
commit.

This is the source of an mdBook that walks a reader from a quantized model file to a serving engine:
the file format, the arithmetic, the declarative graph, the devices, and the measurements that prove
it. Every chapter ends with something to build in the reader's own tree, not with a summary to read.

The reference implementation read beside it is **[`minfer`](https://github.com/yusiwen/minfer)**. It
is a personal hobby project, not a product: it started in 2021 as a way to learn how inference really
works, and has been rewritten along several different lines since - different graph designs, different
backends, different opinions about where the boundary between the engine and a device belongs. What
survives in the code is the part that was measured, and that habit is what this book teaches. This
book describes minfer at the commit recorded in [`minfer.lock`](minfer.lock); its kernels are hand-written, from the AVX2 and NEON dot
products to the CUDA matmuls, with no machine-learning framework underneath.

**Status: writing.** Part 0 and Chapters 1-2 are written; the other 27 chapters are stubs whose outline, lesson checklists and
acceptance criteria exist; the prose is being written one chapter at a time, and each chapter's
figures ship with it. What is finished is the shape of the book, the
[reference index](src/appendix/references.md), and the [audit](src/appendix/cuda-docs-audit.md) of the
documentation this book follows.

## Build

```bash
nix develop                       # mdbook 0.5 + ripgrep + python3, pinned by flake.lock
mdbook serve --open               # http://localhost:3000
mdbook build                      # writes the site to book/
bash scripts/verify.sh            # the gate: build-with-no-warnings, character scan,
                                  # offline assets, figures, maths wrappers, page floor,
                                  # reference index, skeleton and links
nix build                         # reproducible site into result/
```

Without Nix: `brew install mdbook` (0.5.x) and run the same commands.

## Layout

| Path | What it is |
|---|---|
| `src/index.md` | the landing page: what the book is, the four questions, the map |
| `src/start/` | Part 0: the contract, machines, the workspace, the routing map |
| `src/NN-<slug>/index.md` | one chapter per directory, English slug, fixed skeleton |
| `src/figures/` | hand-authored SVGs, included inline so they follow the page theme |
| `src/appendix/` | conventions, the decision map (82 ADRs), the reference index, measurements, the CUDA-tutorial audit, glossary, limits, where the code lives |
| `FIGURES.md` | the figure plan: 46 planned figures, their batch order and acceptance test |
| `starter/` | the reader's engine skeleton: six empty crates, a notes template, builds offline |
| `minfer.lock` | the pinned minfer commit and the counts the book quotes - the only place the hash lives |
| `reference/` | the citation inventory, resolved at that commit, plus the citations that must stay broken |
| `runs/` | behaviour probe records; two of them are compared by hand when the engine moves |
| `scripts/verify.sh` | the gate CI runs; a green local run is a green deploy |
| `scripts/probe.py` | the checks the build cannot see: skeleton sections, links, counts |
| `scripts/references.py` | derives the index's "used in" column from the chapters; fails on a dangling key |
| `scripts/figcheck.py` | figure lint (structure, theme classes, geometry, captions), copied from the `mdbook-figures` skill |
| `scripts/minfer-drift.py` | compares the book's citations and counts against minfer at the baseline or at master |
| `scripts/contract-probes.sh` | records what the engine actually does (flags, graph size, determinism, server surface) |
| `scripts/membw.c`, `scripts/gpubw.cu`, `scripts/metalbw.swift` | the three bandwidth measurements behind [the reference machines](src/appendix/machines.md) |
| `mathjax/`, `css/` | vendored renderer, the figure palette (`figures.css`) and the page furniture (`book.css`); the site fetches nothing remotely |

## Rules that are enforced, not just stated

- **English only.** `scripts/verify.sh` rejects every character outside ASCII and Latin-1 apart from
  a short allowlist of typographic marks.
- **One skeleton per chapter.** `scripts/probe.py` fails a chapter whose fixed sections are missing or
  out of order, and a landing page whose chapter or lesson count disagrees with the tree.
- **Counts are derived.** The counts on the landing page, the appendix and this file come from the
  tree; `src/appendix/measurements.md` records the command behind each.
- **Every page cites a source.** `scripts/references.py` fails on a page with no external reference, or
  on a cite key that the index does not hold.
- **Code pointers are path + symbol.** Never a bare line number, and never a commit hash in prose:
  the hash lives in `minfer.lock`, and every citation is checked against the committed inventory.
- **Drift is tracked, not chased.** A daily advisory job reports what moved against the baseline and
  against master; the book moves its baseline deliberately, following the ledger's procedure.

## License

[CC BY-NC-SA 4.0](LICENSE) - the same licence as the author's other handbook: reuse with
attribution, no commercial reuse, share-alike. The prose is the author's; the reference
implementation `minfer` keeps its own MIT licence.
