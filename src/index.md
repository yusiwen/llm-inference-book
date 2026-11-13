# Learning LLM Inference the Hard Way

### Building an Inference Engine from Scratch in Rust

This is a build book, not a survey. It takes you from a quantized file on disk to a serving engine that answers requests on the CPU, on Apple silicon and on an NVIDIA GPU - and it makes you write each part yourself, in your own tree, with one real engine open beside you as the reference.

> **This build**: the frame is finished and the prose is starting. Part 0 and Chapters 1-2 are written; the other twenty-seven chapters are still stubs - the outline, the lesson checklists and the acceptance criteria exist, the prose does not - and 3 of the 46 planned figures are drawn, in Chapters 1 and 2. What *is* finished is the shape of the book, [the build ladder](appendix/build-ladder.md), the [reference machines](appendix/machines.md), the [reference index](appendix/references.md) and [the audit](appendix/cuda-docs-audit.md) of the documentation this book follows.

The reference engine is **minfer**, a pure-Rust inference engine with no machine-learning framework underneath it: every kernel is hand-written, from the AVX2 and NEON dot products to the CUDA matmuls. This book describes minfer at the commit recorded on [the baseline page](appendix/baseline.md), and every code pointer in it is a file plus a symbol, not a line number, because line numbers move.

## What "the hard way" means

Three rules, and they are the whole method.

1. **You write it before you read it.** Each chapter's build step comes before the reference implementation is opened. A first version that is slow and slightly wrong teaches more than a correct one you read.
2. **Every claim carries its evidence.** A number is *measured* (with the machine and the command), *arithmetic on stated assumptions*, *reported* by someone else, or *my judgement*. The label is part of the sentence.
3. **Being right comes before being fast.** The correctness gate is built in Chapter 27, before any speed claim, and every optimization afterwards has to pass through it.

## The four questions

The book circles four questions. They are also its acceptance test: if you can answer them about your own engine, you learned the material.

1. **Where does the time go?** Prefill and decode are different problems - one is limited by arithmetic, the other by memory bandwidth - and most wrong optimizations come from confusing them.
2. **Why are the weights bytes instead of floats?** Quantization is a file format, a kernel design and an accuracy budget at the same time.
3. **Where does the boundary between the engine and the device live?** The declarative graph, the backend registry and the refusal rules are what keep three backends from becoming three engines.
4. **How do I know it is correct, and how do I know it is faster?** An identity gate and a measurement harness you can defend, or the other three answers are unfalsifiable.

## Who this book is for

You are in the right place if:

- You write Rust well enough to fight the borrow checker and win, and you have used a language model.
- You have read that inference is "just matrix multiplication" and found that explanation useless.
- You want to end up with an engine of your own that loads a real GGUF file and generates tokens, and you want to understand every layer between the file and the token.

**Who this book is not for.** Four exclusions, said plainly so nobody spends a month finding out:

- **You want to train or fine-tune models.** This book is inference only: no gradients, no optimizer, no dataset.
- **You want a framework to use.** The deliverable is your own engine, and the reference is one you read - not a library you import.
- **You want a tour of every model architecture.** Two dense families appear (Qwen2/Qwen2.5 and Qwen3, with Llama 3.x noted where it differs). Mixture-of-experts, multi-modal and state-space models are out of scope.
- **You have no GPU and no way to rent one.** Parts 1-4, and Chapters 21 (on a Mac), 27 and 28, run on a laptop. Chapters 22-26 need an NVIDIA GPU with a toolkit; the book says where the boundary is instead of pretending it does not exist.

## How to read it

1. **Straight through.** The six parts are a ladder: each chapter's build step assumes the previous one compiles.
2. **As a checklist.** Every chapter ends with a deliverable and a "how you know you passed" list. Reading is not finishing; shipping the deliverable is.
3. **As a checklist of what exists.** [The build ladder](appendix/build-ladder.md) lists the 30 steps, what each produces, and what breaks if it is skipped, with [the reference machines](appendix/machines.md) that every number in the book was taken on beside it.
4. **As an index.** [The decision map](appendix/adr-map.md) links 82 architecture decision records to the chapters that use them, [the reference index](appendix/references.md) files every paper, book and manual the book sends you to, and the [glossary](appendix/glossary.md) is the backstop for any term.

The compute tier on each chapter is honest about what hardware it needs: ✅ runs on any laptop, ⚠️ needs the hardware named in the chapter, ❌ is not used in this book at all.

## Book map

| Part | Chapters | Lessons | Compute | What it buys you |
|---|---|---|---|---|
| **1 The ground truth** | 1-5 | 25 | ✅ | A GGUF parser, a tensor type, and dequantizers for nine formats |
| **2 Arithmetic** | 6-10 | 25 | ✅ | Quantized dot products in scalar and SIMD, matmul shapes, ops, attention |
| **3 Language** | 11-13 | 15 | ✅ | Text to ids, ids to prompts, scores to sampled tokens |
| **4 The graph** | 14-20 | 36 | ✅ | A declarative graph with allocator, KV subsystem and a generation loop |
| **5 Devices** | 21-26 | 30 | ⚠️ | A Metal backend, then CUDA from the machine model to a measured campaign |
| **6 Proving it** | 27-29 | 15 | ✅⚠️ | An identity gate, a defensible benchmark, and a server |

**29 chapters, 146 lessons** in total: roughly 25-30 hours of reading plus 120-200 hours of hands-on work, spread over months rather than weekends. *(arithmetic on stated assumptions: the same per-chapter load as the author's previous handbook.)*

## Why the book is shaped this way

Three decisions explain most of the distance between this book and the project's own documentation, which is excellent and is a different kind of artifact.

1. **It is ordered by build dependency, not by module.** The project's walkthrough follows one run through the engine, which is the right order for understanding a finished system. A book you build along with has to follow what compiles onto what: file format, then numbers, then text, then the graph, then devices, then the proof.
2. **The CUDA part is five chapters, not one.** The machine model, the Rust boundary, the kernel ladder, attention and graphs, and the optimization campaign are five different skills. The project's own CUDA tutorial teaches them well (see [the CUDA audit](appendix/cuda-docs-audit.md) for how this book differs and what it corrected), but a tutorial written for contributors can assume the rest of the engine; a book cannot.
3. **Correctness and measurement are chapters, not appendices.** They are the two disciplines that separate an engine from a pile of kernels, and the project learned that the hard way - its own optimization record contains as many reverted levers as landed ones.

## Where the code lives

- **The reference engine**: `github.com/yusiwen/minfer`, at the pinned baseline in [`minfer.lock`](https://github.com/yusiwen/llm-inference-book/blob/main/minfer.lock). Its documentation site is `yusiwen.cn/minfer`.
- **The decision record**: 82 ADRs in `docs/adr/`, indexed by [the decision map](appendix/adr-map.md).
- **The optimization record**: 110 step records in `docs/cuda_optimization_steps/`, referenced from Chapter 26 and quoted in [Measurements](appendix/measurements.md).

Every measurement in this book will name its machine. Numbers taken from the project's records are labelled *reported*, never silently upgraded to something this book measured.
