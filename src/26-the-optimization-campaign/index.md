# Chapter 26 · CUDA V: The Optimization Campaign

> - **Compute tier**: ⚠️ needs an NVIDIA GPU for the measurements; the method is readable anywhere
> - **Status**: stub
> - **Build step S26**: A campaign of your own: a cost model, one lever per experiment, a gate that decides land-or-revert, and a written record you can defend a year later.
> - **Rule of this book**: This chapter is the method and one worked campaign, not a catalogue of every kernel. The techniques themselves appear where they are needed, in Chapters 7, 24 and 25; the measurement discipline they depend on is Chapter 28.

## First, the gist

A GPU optimization campaign is not a list of tricks, it is a process that survives being wrong. You build a cost model from counts, identify the wall a kernel is waiting on, change exactly one thing, and let a gate decide whether the change lands - where the gate includes correctness, bitwise where the contract demands it, and a throughput delta that is larger than the noise of the measurement. Most levers fail, and the failures are the valuable part: an honest record of a reverted change stops the next engineer from spending a week on it. The project this book follows kept 110 step records of exactly that shape, with landed, reverted and measurement-only rounds side by side.

## What you'll be able to do

- Build a wall-clock cost model for a workload and name the lever that would move it.
- Design an experiment with one variable, a stated noise floor, and a land/revert decision rule fixed in advance.
- Write a step record that a stranger could reproduce, including the reverted ones.

## Lessons

- [ ] 1. Attribute before you optimize: decomposing the wall clock into kernels and waits
- [ ] 2. One lever per experiment, and a decision rule written before the measurement
- [ ] 3. The gate chain: correctness, then identity, then a delta above the noise floor
- [ ] 4. Why most levers fail: the four failure shapes (wall-inert, parity-breaking, occupancy-negative, artifact)
- [ ] 5. Device tiers and formula gates: what generalizes to a GPU you do not own (payoff: a table of measured tiers beats a hard-coded guess)

## A step record

| Field | What it must contain | Your example | Why a reader needs it |
|---|---|---|---|
| hypothesis |  |  |  |
| counts predicted |  |  |  |
| gate result |  |  |  |
| wall-clock delta |  |  |  |
| verdict |  |  |  |

## Project

**Build step S26 · a campaign you can audit.** Pick one kernel from Chapter 24 that is off its roofline and run three experiments: one that lands, one that is wall-inert, and one that breaks parity. Writing up the failures is the exercise; anyone can publish a win.

Fix the decision rule in the record before the first measurement. A rule chosen afterwards is not a gate, it is a rationalization.

## Deliverable

- A cost model for one kernel with predicted counts and the measured wall clock
- Three step records in the campaign's format, including at least one reverted lever
- A one-page verdict: what the campaign taught about this GPU, and what it did not measure

## How you know you passed

- Every record states the measurement window, the binary, and the machine - a reader could re-run it.
- The landed change's delta exceeds the noise floor you stated in advance.
- One of your reverted levers is reproducible from the record alone, and you show the reproduction.

## Reading

- code: `src/device_tier.rs`, `src/graph/cuda_backend.rs`, `src/cuda/methods/policy.rs` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/CUDA_OPTIMIZATION.md` (the master history table), `docs/cuda_optimization_steps/` (110 step records plus index and writing contract), `77-verification-methodology.md`, `101-steady-state-method.md`
- docs: `docs/cuda_tutorial/06-optimization-methods.md`, `07-where-next.md`
- decisions: ADR-0035, ADR-0059, ADR-0072 (draft proposals), ADR-0034 (smem pre-warm)
- [nsight] *Nsight Compute* user guide, NVIDIA - no arXiv ID (vendor manual)

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
