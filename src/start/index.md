# Part 0 · Before you start

> - **Compute tier**: ✅ any laptop
> - **Status**: written - the model numbers here are the sizes listed in the two Hugging Face repositories named below (read on 2026-11-13, `reported`), and the arithmetic on them is labelled as such; the toolchain guidance is my judgement from running it on the three machines in the appendix
> - **Build step S0**: a repository of your own, a model file, a toolchain, and a notes file with four sections that start empty.
> - **Rule of this book**: this page sets up the work. It teaches no engine internals; the routing map at the end tells you which chapter answers which question.

## First, the gist

You are going to write an inference engine. Not study one: write one, in Rust, in your own tree, until it loads a real quantized model and produces tokens - and then keeps working while you make it fast.

Four things have to exist before Chapter 1 is worth reading:

- **A model file.** A real one, in GGUF format, quantized, on your disk. Everything the book claims about bytes, blocks and ceilings is about that file, not about a diagram of one.
- **A machine you will measure on.** One is enough. A second one (a GPU) changes which chapters you *run* rather than *read*, and the book says which is which on every chapter's first line.
- **A toolchain that compiles Rust.** Nothing else, until Part 5.
- **A notes file with four sections.** This sounds like stationery and is the single most load-bearing thing on the page. Every number in this book that is worth anything came with its machine, its command and its date, and the reason Chapter 28 exists is that most people do not write those three things down until it is too late to recover them.

That is the setup. It should take an afternoon, and most of the afternoon is the model download.

The rest of this page is the contract you are agreeing to, the machines and files to pick, the tree to create, the notes file to keep - and a routing map so that the other twenty-nine chapters are a reference you can enter anywhere, not a novel you must read in order.

## What you'll be able to do

- Say which model file, which machine and which toolchain you will use for the whole book, and why that choice will not change halfway through.
- Label a claim correctly as `measured`, `reported`, `arithmetic` or `my judgement` - and recognise the moment you are about to upgrade one of them into another.
- Find the chapter that answers a given question from the routing map, without reading the book in order.
- State, in one sentence per machine, what you will not measure on it - which is the sentence that keeps a benchmark honest.

## Lessons

- [ ] 1. The contract: write first, read second, and never publish an unattributed number
- [ ] 2. Pick your machines: one laptop, one optional GPU, and what each part of the book needs
- [ ] 3. Get a model: one small file for iteration, one 7B file for the numbers
- [ ] 4. Set up the tree: a workspace whose crates you will add to for 29 chapters
- [ ] 5. The notes file: four sections that every later chapter writes into

### Lesson 1 · The contract: write first, read second, and never publish an unattributed number

Three rules. They are the whole method, and the book is only as good as your adherence to them.

**Write it before you read it.** Every build step in this book comes before the reference implementation is opened. A first version that is slow and slightly wrong teaches you something a correct version you read never will: it teaches you where the difficulty actually is. The reference engine is there to be read *after* your attempt, when you have a specific question - why is their buffer allocator eight lines shorter than mine, why does their kernel not need my bounds check - rather than as a source of code to transcribe. Transcribing produces a working engine and no understanding, and this book has no value to you in that mode.

**Every claim carries its evidence.** A number is one of four things, and the label is part of the sentence:

| Label | What it means | How it reads on the page |
|---|---|---|
| **measured** | produced by a command on a named machine | "read 198 GB/s (184-202) on the M4 Pro, five runs" |
| **reported** | someone else's number, quoted | "the vendor specifies 273 GB/s" |
| **arithmetic** | computed from stated assumptions | "(arithmetic: 4.68 GB / 198 GB/s = 23.6 ms per token)" |
| **my judgement** | an opinion about a trade | "(my judgement: a CPU decode step has no slack in either dimension)" |

The rule exists because of one specific failure, and it is the failure that makes a technical book untrustworthy rather than merely imperfect: **upgrading a number's label without doing the work.** An arithmetic estimate quietly becomes "the measured performance". A vendor specification becomes "the bandwidth". A number from one machine becomes "on this hardware". Each of those upgrades is a small lie that every later number inherits, and after three of them the reader cannot tell which sentences were checked.

Two of the labels are worth practising now, because you will use them in your own notes within an hour:

- `273 GB/s` for the memory on two of the reference machines is **reported** - it is the vendor's figure for the bus, and no kernel achieves it. The **measured** figure is 198 GB/s on one and 242 on the other, which is what [the reference machines page](../appendix/machines.md) records.
- `4.68 GB` for the 7B model's file is **reported** - it is the size Hugging Face lists - but `4.68 GB / 198 GB/s = 23.6 ms` is **arithmetic** on it, and the resulting 42 tokens per second is a ceiling, not a forecast.

**Being right comes before being fast.** The correctness gate arrives in Chapter 27, which is late, and that is deliberate: by then you have something worth gating. But the *habit* starts in Chapter 1, with the rule that every kernel gets an obviously-correct scalar version first, and every optimisation is checked against it. An optimisation without an oracle is indistinguishable from a bug that makes the numbers smaller.

### Lesson 2 · Pick your machines: one laptop, one optional GPU, and what each part of the book needs

Every chapter opens with a compute tier, and the tier is a promise about what you need in order to *do* the chapter rather than read it:

| Tier | Meaning |
|---|---|
| ✅ | runs on any laptop, no special hardware |
| ⚠️ | needs the hardware named in the chapter (an NVIDIA GPU, or a Mac for Metal) |
| ❌ | not used in this book at all |

One machine is enough for the whole core ladder (Part 0 through Part 4, plus Part 6): any CPU that runs Rust. A second machine is optional and unlocks Part 5. The book was written and measured on three, and [the machines page](../appendix/machines.md) names them with the bandwidth each one actually delivered:

| Machine | What it is for | Bandwidth it delivered |
|---|---|---|
| a MacBook Pro (M4 Pro) | CPU chapters, and the Metal backend | 198 GB/s CPU, 212 GB/s Metal |
| an NVIDIA DGX Spark (GB10) | the CUDA chapters | 118 GB/s CPU, 242 GB/s CUDA |
| a small x86 server (NUC11) | the AVX2 half of the SIMD chapters | 45 GB/s |

Write down your own version of that table in your notes, and - this is the part people skip - **one sentence per machine saying what you will not measure on it.** "The laptop: not for CUDA claims, and not for anything where the browser is open, because the spread goes to eight per cent." That sentence is what stops a benchmark from being run on a machine that was busy, and it is the same discipline Chapter 28 formalises.

If you have only a laptop, you are not locked out: Part 5 becomes reading, and the book tells you at each step which claims you cannot check yourself. What you must not do is run a CUDA chapter's numbers on a CPU and call the result a comparison.

### Lesson 3 · Get a model: one small file for iteration, one 7B file for the numbers

Two files, and they have different jobs.

**The small one** is for the loop: it loads fast, fails fast, and is small enough that you can hold its whole structure in your head. **The big one** is for the numbers: 7B-class is where the bandwidth ceiling stops being swamped by per-step overhead, so it is the file every speed claim in this book is about.

Both are public, ungated, and quantized in a format this book parses. Sizes below are from the repositories' file listings (reported, read 2026-11-13); confirm them with `ls -l` after downloading, because that is the number your engine will actually read:

| File | Parameters | Size on disk | Bits per weight |
|---|---|---|---|
| `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 494,032,768 | 0.49 GB | 7.9 |
| `qwen2.5-7b-instruct-q4_k_m` (2 shards) | 7,615,616,512 | 3.99 GB + 0.69 GB | 4.9 |

```bash
# Hugging Face CLI: pip install -U "huggingface_hub[cli]"
mkdir -p ~/.cache/myfer
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
           qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir ~/.cache/myfer
hf download Qwen/Qwen2.5-7B-Instruct-GGUF --include "*q4_k_m*" --local-dir ~/.cache/myfer
ls -l ~/.cache/myfer
```

Three things in that table are worth understanding now, because they are the first real lessons about the file format:

- **The small file is not "4.5 bits per weight".** Its name says Q4_K_M, but 0.49 GB over 494 million parameters is 7.9 bits per weight (arithmetic). Small models keep their embedding and output matrices at higher precision, and at half a billion parameters those matrices are most of the model. The name of a quantization describes a recipe, not a uniform result: the file's size is the truth, and reading the file is Chapter 2's job. (Chapter 2 repeats the division against the parameter count the file itself stores - 630 million, because the vocabulary matrix is written twice - and gets 6.2. Same file, same size, different denominator.)
- **The big file arrives in two pieces.** `-00001-of-00002` and `-00002-of-00002`, because the format was designed before files were routinely multi-gigabyte. A loader resolves the parts by name; the reference engine does exactly that in `src/gguf.rs::resolve_splits`, and Chapter 2 does the same.
- **Neither file is "the model".** Each is a table of tensors plus the metadata to interpret them: 28 layers, 4 key/value heads of 128 dimensions each, a 152,064-entry vocabulary, a 32,768-token context (reported from the model's config). Every one of those numbers turns into code you write, and into bytes you will read.

A word on trust: these are the exact files this book's arithmetic uses, so using them keeps your numbers comparable to the book's. If you prefer a different family - Llama, Gemma, Mistral - the mechanics are identical and the numbers are not; say so in your notes, and do not compare your tokens per second to a table computed for a different file.

### Lesson 4 · Set up the tree: a workspace whose crates you will add to for 29 chapters

The engine is one cargo workspace with six crates, and the crate boundaries are the book's table of contents:

| Crate | Holds | Filled by |
|---|---|---|
| `myfer-file` | GGUF parsing, tensors, strides, quantized block layouts | Chapters 2-5 |
| `myfer-nums` | quantized dot products, SIMD, matmul shapes, vector ops, attention | Chapters 6-10 |
| `myfer-graph` | the declarative graph, fusion, backend assignment, allocation, KV cache | Chapters 14-19 |
| `myfer-engine` | tokenizer, templates, sampler, the generation loop | Chapters 11-13, 20 |
| `myfer-backends` | Metal, then CUDA, behind one registry | Chapters 21-26 |
| `myfer-cli` | the binary, the benchmark harness, the server | Chapters 20, 28, 29 |

Six crates on day zero, all empty, is not premature architecture: it is the cheapest possible test of whether the layering survives contact with the chapters. If chapter 14 makes you want `file` to depend on `graph`, you find out with an empty function and one line of `Cargo.toml`, not with forty thousand lines of code.

The rules, and they are the whole design:

- **Dependencies point one way.** `nums` may use `file`; `graph` may use both; `engine` may use those three; `backends` may use `file` and `graph`; `cli` may use everything. Nothing points back.
- **No dependencies at all, at the start.** No `serde`, no `clap`, no `rayon`. Every dependency you add is a decision a chapter will ask you to justify, and the first three chapters exist partly to make the point that a parser and an allocator do not need a framework.
- **One commit per build step.** Thirty commits, each named for the step. When chapter 22 breaks something you will want to know what the tree looked like when it last worked.

This repository ships the skeleton in [`starter/`](https://github.com/yusiwen/llm-inference-book/tree/main/starter) - six crates, the dependency arrows already pointing the right way, a `notes.md`, and nothing else. Copy it, or type it yourself; the copy takes ten seconds and the typing teaches you the layout.

```bash
cp -R starter ~/myfer && cd ~/myfer
cargo build && cargo test     # six crates, no dependencies, six tests, offline
git init -b main && git add -A && git commit -m "S0: the workspace"
```

### Lesson 5 · The notes file: four sections that every later chapter writes into

`notes.md` has four tables. They are not for tidiness; each one is a question a later chapter will ask and cannot answer for you.

| Section | Columns | Why it exists |
|---|---|---|
| **Machines** | machine, what it is, what you will not measure on it | Chapter 28's noise floor, and the sentence that keeps a benchmark honest |
| **Models** | file, quantization, size on disk, why this one | every ceiling and every comparison is per-file |
| **Measurements** | date, what, command, result, machine state | the only thing that makes a later number comparable to an earlier one |
| **Decisions** | date, decision, why, what would reopen it | the reverse of every choice a chapter will push back on |

The reason is not bureaucracy. Here is the same binary, on the same machine, five times each, on two occasions two hours apart:

```text
with a Jupyter kernel drifting between 57% and 410% of a core:
  read  96.7  81.7  90.3  67.7  66.3 GB/s
on an idle box, same command, same array size:
  read 101.9 101.2 100.9 101.3 108.0 GB/s
```

Thirty per cent, from a process the benchmark never mentions. If you write down "101 GB/s" and not the conditions, you have written an anecdote that will silently disagree with your own later measurement - and you will spend an afternoon looking for a bug in your kernel. Write the state: the load average, whether the browser was open, whether the GPU had a co-tenant. One line, every time.

Two habits worth forming in the first hour:

- **Record the command, not just the number.** `./membw 2048 20` and "101 GB/s" are one fact; the number alone is unreproducible.
- **Date every entry, and never edit an old one.** A corrected log stops being a log. Add a new row that supersedes the old one and say why.

## The routing map

Read straight through if you are new to this; the six parts are a ladder and each build step assumes the previous one compiles. If you are here to solve a problem, enter by the problem:

| If you are stuck on... | Read |
|---|---|
| "What is actually happening when I run a model?" | [Chapter 1](../01-one-token-at-a-time/index.md) |
| "How do I even read this file?" | [Chapter 2](../02-the-file-is-the-model/index.md), [3](../03-tensors-strides-and-bytes/index.md), [4](../04-quantized-blocks-the-base-formats/index.md) |
| "Why is my multiplication wrong, or slow?" | [Chapters 6-8](../06-the-dot-product-contract/index.md) |
| "Why is the output garbage?" | [Chapters 9-10](../09-vector-ops-and-precision/index.md), [11-13](../11-the-tokenizer/index.md) |
| "How do I structure the forward pass?" | [Chapters 14-17](../14-the-declarative-graph/index.md) |
| "How do I add a backend?" | [Chapter 16](../16-backend-assignment-and-registry/index.md), then [21](../21-metal-the-first-backend/index.md) |
| "How do I make it fast on a GPU?" | [Chapters 22-26](../22-cuda-the-machine/index.md) |
| "How do I know it is right, and fast?" | [Chapters 27-28](../27-correctness-the-identity-gate/index.md) |
| "What does this word mean?" | [The glossary](../appendix/glossary.md) |
| "Which step am I on?" | [The build ladder](../appendix/build-ladder.md) |

Three routes worth naming:

1. **The core ladder** (Chapters 1-20, then 27-28) runs on any laptop and ends with a CPU engine that loads a real file, generates tokens and can prove both. If you read nothing else, read this.
2. **The device lane** (Chapters 21-26) needs hardware. Read it anyway if you have none: the transferable part is the seam - which buffers the engine owns, what crosses the boundary, where the errors come from - and Chapter 21 is written so that it is legible without a GPU.
3. **The proof lane** (Chapters 27-28) is deliberately late. If your engine is going to outlive the book, pull it earlier: everything after Chapter 20 is easier to get wrong when there is no identity gate to catch it.

**What you need to know first.** You should be able to write Rust well enough to fight the borrow checker and mostly win, and you should have used a language model. You do **not** need to know how one works: the book assumes zero knowledge of inference internals, expands every term where it first appears, and treats "it is just matrix multiplication" as a sentence that explains nothing.

**What this book is not.** It is not a survey of serving systems - there is one server, in Chapter 29, and it is small. It is not a quantization cookbook - Part 2 derives the formats the file actually uses rather than cataloguing all of them. And it is not a guide to tuning an engine you already have: every chapter assumes you are building the one in front of you.

## Project

**Build step S0 · the workspace, the file and the notes.** No engine code: the point of this step is that the boring prerequisites are done, written down, and committed.

Do this in order, and do not skip the table:

1. Copy [`starter/`](https://github.com/yusiwen/llm-inference-book/tree/main/starter) to a directory of your own, `cargo build`, `cargo test`, `git init`, first commit.
2. Download both model files into one directory and record their exact sizes with `ls -l`.
3. Measure your machine's memory bandwidth with the appendix's `membw.c` (five runs, and note the load average), and paste the output into your notes.
4. Fill the decision table below. Every row has to have an answer, and "the same as the book" is a valid answer only if you say why.

| Decision | Your choice | Why (one line) |
|---|---|---|
| Small model for iteration | | |
| 7B-class model for numbers | | |
| CPU machine | | |
| GPU machine (or "none") | | |
| Rust toolchain (channel, version) | | |
| What I will not measure on the CPU machine | | |

## Deliverable

- A repository with an empty but layered workspace, and a first commit named for the step (`S0: the workspace`).
- Both model files on disk, with their sizes recorded.
- A `notes.md` with four sections, at least one measurement in it, and the decision table above filled.
- One sentence per machine stating what it is for and what you will not measure on it.

## How you know you passed

- `cargo build` succeeds on the empty workspace, and the crate names alone tell a stranger which layer does what.
- Every chapter's compute tier is satisfiable by the machines you wrote down - or you have written down which chapters you will read instead of run, and what you will not claim about them.
- Your notes file has a first entry with a date, a command and an output, even if the output is `0 tokens` or `398 GB/s` or an error message you did not expect.
- Your bandwidth measurement names the machine and the load average, and you can say what would make it 30 per cent wrong.

## Reading

- `https://github.com/yusiwen/minfer` - the reference engine; read `README.md` and `docs/BUILD.md` for the toolchain
- `docs/SUPPORT-MATRIX.md` in the same repository - which quantized formats and model families are in scope
- `docs/GATE-CONTRACT.md` - what the reference project requires before a change counts as done
- [trpl] Klabnik and Nichols, *The Rust Programming Language* - no arXiv ID (book) - chapters 1-10 are enough for everything before Part 5
- [caqa] Hennessy and Patterson, *Computer Architecture: A Quantitative Approach* (6th edition, 2017) - no arXiv ID (book) - chapter 2, for why a bandwidth figure is a ceiling rather than a rate

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - stated here, answered in Chapter 1.
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - stated here; Part 2 is where the bytes are actually read.
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - stated here; Chapters 14-16 and 21 are where it is drawn.
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - this page only sets up the instrument: the notes file, and the habit of writing the machine state beside the number.
