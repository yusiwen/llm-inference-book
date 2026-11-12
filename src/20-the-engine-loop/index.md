# Chapter 20 · The Engine Loop

> **Compute tier**: ✅ any laptop | **Status**: stub
> **Build step S20**: A generation loop with prefill, decode, graph reuse, incremental multi-turn prefill and cancellation, driven from a command line.
> **Rule of this book**: Sampling is Chapter 13 and the graph is Chapter 14. This chapter is the loop that ties them together and the state it keeps between steps. Serving several sessions at once is Chapter 29.

## First, the gist

With a graph, weights, a cache and a sampler, the engine is a loop: build once for the prompt's shape, execute over all prompt tokens, take the last logits, sample, append, and repeat with a graph that differs only in its input data. The discipline is in what changes: the first decode step learns the decode shape and builds a second graph, every later step reuses it, and the cache grows underneath without any rebuild. Multi-turn conversation is the same idea with an append-only prefix, and the failure modes are cancellation and context overflow - the two events that must leave the cache in a state the next step can still use.

## What you'll be able to do

- Implement a loop whose per-step overhead is measured, and say what part of a step is not the model.
- Reuse a graph across steps and prove the reuse is valid (same structure, different inputs).
- Handle cancellation and context overflow without corrupting the cache.

## Lessons

- [ ] 1. Prefill, then decode: two shapes, two builds, one cache
- [ ] 2. The reuse cache: what must be equal, what may differ, and the invalidation you must not skip
- [ ] 3. Multi-turn: incremental prefill with an append-only prefix (payoff: why re-prefilling the whole conversation is quadratic in turns)
- [ ] 4. Cancellation, stop ids and a drain that finishes the token in flight
- [ ] 5. Overflow: what happens at the context limit, and the three policies an engine can have
- [ ] 6. The engine as a library: two API tiers, and a configuration installed once (payoff: the environment stops being the configuration)

## The loop's own cost

| Phase | What dominates | Measured (yours) | What would reduce it |
|---|---|---|---|
| first prefill (cold) |  |  |  |
| prefill (warm) |  |  |  |
| first decode step (build) |  |  |  |
| steady decode step |  |  |  |
| multi-turn append |  |  |  |

## Project

**Build step S20 · the loop you will live in.** Instrument the loop before optimizing anything: time per step, tokens per step, cache size after each step. Every performance claim in Chapter 28 depends on this instrumentation being honest.

Test cancellation by cancelling mid-generation and then continuing the same session. If the next token differs from an uncancelled run at the same point, the cache was corrupted.

## Deliverable

- A generating engine from the command line with per-step timing
- A graph-reuse test showing a rebuild happens only when the shape changes
- Cancellation and overflow tests, each asserting the cache stays usable

## How you know you passed

- Steady-state decode overhead (non-model time) is measured and reported, not guessed.
- Cancelling and resuming produces the same tokens as an uninterrupted run.
- At the context limit the engine behaves as your chosen policy says, and the policy is written down.

## Reading

- code: `crates/minfer-cli/src/main.rs`, `src/engine/generate.rs`, `src/graph/cache.rs`, `src/conversation.rs`, `src/advanced.rs`, `src/config.rs` at commit `88ffa5d`
- docs: `docs/inference_e2e_walkthrough/09-prefill-forward-path.md`, `13-decode-loop-graph-reuse.md`
- docs: `docs/CLI-CONVERSATION-PLAN.md`, `docs/CRATE-API-DESIGN.md`
- decisions: ADR-0017 (speculative decoding refuses what its identity contract cannot carry), ADR-0079 (the engine is a crate with two API tiers), ADR-0082 (the application layer is its own package)
- [scaling-inference] Pope et al., *Efficiently Scaling Transformer Inference* (2022) - arXiv:2211.05102
- [paged-attention] Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023) - arXiv:2309.06180

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
