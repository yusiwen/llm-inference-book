# Chapter 13 · Sampling and Constrained Decoding

> - **Compute tier**: ✅ any laptop
> - **Status**: stub
> - **Build step S13**: A sampler pipeline with penalties, top-k, top-p, temperature and a seed you control, plus a grammar mask that makes one class of outputs impossible.
> - **Rule of this book**: This chapter turns scores into an id. The loop that calls it per step is Chapter 20; speculative decoding, which changes how many ids arrive at once, is Chapter 29.

## First, the gist

The model outputs one number per vocabulary entry, and something has to choose. Greedy choice is the obvious pick and it loops, repeats and bores; a sampler reshapes the distribution first - penalize recent tokens, cut the tail with top-k and top-p, rescale with temperature - then draws from it with a seeded generator so the run is reproducible. The second half of the chapter is the strictest form of choice: a grammar that masks every token that would make the output invalid, which turns a request for JSON into a guarantee.

## What you'll be able to do

- Implement the sampler pipeline in the order the reference uses, and prove reproducibility with a fixed seed.
- Explain what top-p does that top-k does not, and give a case where each one misbehaves.
- Compile a schema to a mask over the vocabulary and show that an invalid token is impossible, not unlikely.

## Lessons

- [ ] 1. Scores to a distribution: penalties, top-k, top-p, temperature
- [ ] 2. Order matters: the pipeline is a sequence of decisions, and swapping two stages changes the output
- [ ] 3. Seeded sampling and reproducibility, and what a seed does not fix
- [ ] 4. Stop conditions: stop strings, stop ids, and the byte-level matching nobody gets right first (payoff: why your answer keeps the stop text in it)
- [ ] 5. Grammar masks: one pipeline stage over the vocabulary (payoff: constraints that cannot be violated, not merely discouraged)

## The pipeline

| Stage | Input | Effect on the distribution | Cost per step | Reproducible with a seed? |
|---|---|---|---|---|
| penalties |  |  |  |  |
| top-k |  |  |  |  |
| top-p |  |  |  |  |
| temperature |  |  |  |  |
| grammar mask |  |  |  |  |
| draw |  |  |  |  |

## Project

**Build step S13 · determinism you can defend.** Fix the seed, fix the prompt, and require the same ids twice. Then add a penalty stage and watch a naive implementation break reproducibility because it consumed randomness in a different order.

For the grammar, test the negative: assert that a token outside the constraint never appears across a hundred samples.

## Deliverable

- A sampler with each stage independently testable and a seeded end-to-end reproduction test
- A stop-string matcher that handles multi-token stop sequences
- One grammar-constrained run whose output provably cannot violate the schema

## How you know you passed

- The same seed and prompt produce identical ids across two runs and across a restart.
- Swapping two stages changes the output, which shows the order is real and not decorative.
- No sampled output violates the grammar across your test batch, and the mask cost is measured.

## Reading

- code: `src/sampler.rs`, `src/grammar.rs` at the book's [pinned baseline](../appendix/baseline.md)
- decisions: ADR-0018 (the grammar mask is one stage inside the single sampler pipeline)
- docs: `docs/GRAMMAR-DESIGN.md`, `docs/inference_e2e_walkthrough/12-sampler.md`
- [nucleus] Holtzman et al., *The Curious Case of Neural Text Degeneration* (2019) - arXiv:1904.09751

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
