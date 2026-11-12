# Part 0 · Before you start

> **Compute tier**: ✅ any laptop | **Status**: stub | **Build step S0**: a repository of your own, a model file, a toolchain, and a notes file with four empty sections. | **Rule of this book**: this page sets up the work. It teaches no engine internals; the routing map at the end tells you which chapter answers which question.

## First, the gist

You are going to write an inference engine. The reference engine is in Rust, and so is yours; it runs on a laptop CPU first, and only later on a GPU. Before any of that, three things have to exist: a copy of a real quantized model, a toolchain that compiles Rust and (for Part 5) CUDA, and a notes file where the numbers you measure will live. The last one sounds like bookkeeping and is the most important: a build book whose numbers are not written down as you go produces a reader who cannot tell an improvement from a coincidence.

## What you'll be able to do

- Say which model file, which machine and which toolchain you will use for the whole book, and why that choice will not change halfway.
- Explain the difference between the labels this book puts on claims - measured, reported, arithmetic, judgement - and label one of your own claims correctly.
- Find the chapter that answers a given question from the routing map, without reading the book in order.

## Lessons

- [ ] 1. The contract: write first, read second, and never publish an unattributed number
- [ ] 2. Pick your machines: one laptop, one optional GPU, and what each part of the book needs
- [ ] 3. Get a model: one small file for iteration, one 7B file for the numbers
- [ ] 4. Set up the tree: a workspace whose crates you will add to for 29 chapters
- [ ] 5. The notes file: four sections that every later chapter writes into (payoff: the pages that make Chapter 28 possible)

## The routing map

| If you are stuck on... | Read |
|---|---|
| "What is actually happening when I run a model?" | [Chapter 1](../01-one-token-at-a-time/index.md) |
| "How do I even read this file?" | [Chapter 2](../02-the-file-is-the-model/index.md), [3](../03-tensors-strides-and-bytes/index.md), [4](../04-quantized-blocks-the-base-formats/index.md) |
| "Why is my multiplication wrong / slow?" | [Chapters 6-8](../06-the-dot-product-contract/index.md) |
| "Why is the output garbage?" | [Chapters 9-10](../09-vector-ops-and-precision/index.md), [11-13](../11-the-tokenizer/index.md) |
| "How do I structure the forward pass?" | [Chapters 14-17](../14-the-declarative-graph/index.md) |
| "How do I add a backend?" | [Chapter 16](../16-backend-assignment-and-registry/index.md), then [21](../21-metal-the-first-backend/index.md) |
| "How do I make it fast on a GPU?" | [Chapters 22-26](../22-cuda-the-machine/index.md) |
| "How do I know it is right / fast?" | [Chapters 27-28](../27-correctness-the-identity-gate/index.md) |

## Project

**Build step S0 · the workspace.** A cargo workspace with one crate per layer you will add (file, numbers, graph, engine, backends), created empty. Creating them now is a promise about layering; filling them is the book.

| Decision | Your choice | Why (one line) |
|---|---|---|
| Small model for iteration | | |
| 7B-class model for numbers | | |
| CPU machine | | |
| GPU machine (or "none") | | |
| Rust toolchain | | |

## Deliverable

- A repository with an empty but layered workspace, and a first commit
- A notes file with four sections: machine, model, measurements, decisions
- One sentence per machine stating what it is and what you will not measure on it

## How you know you passed

- `cargo build` succeeds on the empty workspace, and the layer boundaries are visible in the crate names.
- Every chapter's compute tier is satisfiable by the machines you wrote down, or you know which chapters you will read instead of run.
- Your notes file has a first entry with a date, a command and an output, even if that output is `0 tokens`.

## Reading

- `https://github.com/yusiwen/minfer` - the reference engine; read `README.md` and `docs/BUILD.md` for the toolchain
- `docs/SUPPORT-MATRIX.md` in the same repository - which quantized formats and model families are in scope
- `docs/GATE-CONTRACT.md` - what the reference project requires before a change counts as done
- [trpl] Klabnik and Nichols, *The Rust Programming Language* - no arXiv ID (book)

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
