# Chapter 28 · Speed You Can Defend

> **Compute tier**: ⚠️ for the throughput numbers (a real model and a quiet machine); the method needs nothing | **Status**: stub
> **Build step S28**: A benchmark harness with warmup budgets, interleaved A/B runs inside one window, medians, and a written statement of what the number does not cover.
> **Rule of this book**: Chapter 1 computed a ceiling; this chapter measures against it. Chapter 26 ran a campaign; this is the measurement discipline that campaign depends on.

## First, the gist

Almost every performance number you will read is incomparable to almost every other one, because the machine state, the prompt length, the quantization and the warmup all changed. Defensible measurement is mostly refusal: refuse to quote a number without its shape and configuration; refuse to compare numbers taken in different sessions; refuse to trust a delta smaller than the window's own noise. This chapter builds a harness that enforces those refusals, then uses it to price one real optimization end to end, including a correction - because a baseline can be contaminated, and a campaign that cannot correct itself is a campaign that publishes the wrong number confidently.

## What you'll be able to do

- Measure prefill and decode separately, with warmup, and report medians over interleaved A/B runs.
- State a noise floor for a given machine and window, and refuse deltas below it.
- Detect a contaminated baseline and correct a published delta honestly.

## Lessons

- [ ] 1. Two numbers, never one: prefill throughput and decode throughput
- [ ] 2. Warmup budgets, steady state, and the first ten seconds that are not the thing
- [ ] 3. Interleaved A/B inside one window: why cross-session comparisons are invalid
- [ ] 4. Medians, spread, and the noise floor you must state before claiming a win (payoff: the delta that survives re-measurement)
- [ ] 5. Correcting a number: what to do when the baseline was contaminated

## A measurement record

| Field | Value | Why it is required |
|---|---|---|
| machine + driver |  |  |
| model + quantization |  |  |
| prompt length / output length |  |  |
| warmup and window |  |  |
| median + spread |  |  |
| what was not measured |  |  |

## Project

**Build step S28 · a harness that refuses.** Write the harness so that it refuses to print a number without its configuration. Then use it on your own engine and compare against the ceiling from Chapter 1.

Re-measure one of your earlier claims with the new harness. If it moves, that is the lesson: the old number was not wrong, it was unattributed.

## Deliverable

- A harness producing prefill and decode medians with configuration printed alongside
- A stated noise floor for your machine, measured by repeating an unchanged run
- One before/after with interleaved runs, plus one corrected earlier claim

## How you know you passed

- Every published number carries its configuration and its window.
- A null change measures as null under your harness (the negative control).
- At least one earlier number of yours was corrected, and the correction is documented with the reason.

## Reading

- code: `crates/minfer-cli/src/bench.rs`, `crates/minfer-cli/src/spec_verify.rs` at commit `88ffa5d`
- docs: `docs/cuda_optimization_steps/101-steady-state-method.md`, `63-r59b-clean-remeasure.md`, `77-verification-methodology.md`
- docs: `docs/TEST-BASELINES.md`, `docs/PERF-QWEN3-4B-VS-LLAMACPP.md`
- docs: `docs/CUDA_OPTIMIZATION.md` section 0 for how the campaign recorded deltas across sessions
- [rigorous-bench] Georges, Buytaert and Eeckhout, *Statistically Rigorous Java Performance Evaluation* (OOPSLA 2007) - no arXiv ID (conference paper)

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
