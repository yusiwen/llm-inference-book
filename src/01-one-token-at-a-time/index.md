# Chapter 1 · One Token at a Time

> **Compute tier**: ✅ any laptop (arithmetic on paper plus a stub loop that prints ids) | **Status**: stub
> **Build step S1**: A loop that reads a prompt, calls a placeholder `model.forward(ids) -> logits`, and prints one token per step. No arithmetic yet: the skeleton every later chapter hangs code on.
> **Rule of this book**: This chapter measures the shape of the problem and nothing else. The model file is Chapter 2, the arithmetic starts in Chapter 6, the graph that schedules it in Chapter 14.

## First, the gist

An inference engine has one job: take a sequence of token ids, produce one score for every entry of the vocabulary, pick one, append it, repeat. The first pass over the prompt (**prefill**) is a matrix-matrix job and is limited by how much arithmetic the machine can do. Every pass after it (**decode**) is a matrix-vector job that must read the whole model to produce a single token, so it is limited by how fast the weights can be streamed out of memory. That asymmetry explains most of an engine's design, and it gives you a ceiling you can compute on paper: file size divided by memory bandwidth.

## What you'll be able to do

- Predict, from a model's file size and a machine's memory bandwidth, roughly how many tokens per second decode can reach, before running anything.
- Say which of prefill and decode an optimization is supposed to help, and name an optimization that helps one and hurts the other.
- Sketch the loop and its three pieces of state (the KV cache, the position counter, the sampler window) well enough to know where each later chapter plugs in.

## Lessons

- [ ] 1. Ids in, ids out: the loop, and the three pieces of state it carries
- [ ] 2. Prefill and decode are different jobs: one prompt, two regimes
- [ ] 3. The roofline in one page, without calculus: arithmetic per byte
- [ ] 4. A decode ceiling you can compute by hand: file size divided by bandwidth (payoff: the number every later speed claim is measured against)
- [ ] 5. What an engine may be lazy about, and the four things it never may

## The two regimes

| Regime | What it multiplies | Bound by | What helps | What does not help |
|---|---|---|---|---|
| Prefill (a prompt of N tokens) |  |  |  |  |
| Decode (one token per step) |  |  |  |  |

## Project

**Build step S1 · the loop, before arithmetic.** Three empty subsections in your notes, filled as you go: the state the loop carries, the quantity each regime is bound by, and the one number you will measure yourself in Chapter 28.

Write the loop against a placeholder model first. Everything this chapter asserts is checkable with a fake `logits` vector of zeros.

## Deliverable

- `loop-stub` in your own tree: reads a prompt from argv, prints ids, then prints one token per iteration until a stop id
- A one-page ledger: the decode ceiling for one model on one machine, with both inputs (file size, measured bandwidth) named
- A written answer to "prefill or decode?" for four optimizations you have heard of

## How you know you passed

- Your ceiling is within a factor of two of what a real engine achieves on the same machine, and you can say which way it errs and why.
- You can name the regime each optimization targets without looking at the answer.
- No kernel code exists yet, and none is needed: the test is the arithmetic.

## Reading

- code: `src/engine.rs`, `crates/minfer-cli/src/main.rs` (the decode loop) - for shape only, at commit `88ffa5d`
- docs: `docs/ARCHITECTURE.md` section 3 (the inference pipeline)
- background: Williams, Waterman and Patterson, the roofline model (2009) - no arXiv ID; two pages carry the idea

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
