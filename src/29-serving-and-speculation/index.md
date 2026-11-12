# Chapter 29 · Serving, and the Speculation Trade

> **Compute tier**: ⚠️ for measurements under load; the design and the cost model need nothing | **Status**: stub
> **Build step S29**: An HTTP server with slots, a bounded batch worker, metrics and a graceful drain - and a speculative decoder that pays for itself or is switched off.
> **Rule of this book**: The loop is Chapter 20 and the sampler Chapter 13. This chapter is concurrency plus one throughput idea whose cost model decides whether it is worth having.

## First, the gist

Serving turns an engine into a shared resource: several conversations at once, arriving and leaving unpredictably, each expecting a stream. That means slots with their own cache, one worker that batches work against a bound, metrics that answer "why is it slow now", and a shutdown path that drains instead of dropping requests. The second half is speculation, a different way to go faster: a small draft model guesses several tokens, the big model verifies them in one pass, and any prefix that survives is free. It is a real speedup only when the acceptance rate and the verification cost multiply to more than one, which is a fact about a model pair on a machine - not a property of the technique.

## What you'll be able to do

- Serve several sessions with per-slot caches, a bounded worker, and metrics that name the bottleneck.
- Build a speculative decoder and compute, from measurements, whether it pays on a given model pair.
- Shut down cleanly: refuse new work, finish what is in flight, and leave no session in an unknown state.

## Lessons

- [ ] 1. Slots: several conversations, one engine, one device
- [ ] 2. A bounded worker and what a lagging slot does to the ones behind it
- [ ] 3. Metrics and cancellation: the three numbers that explain a slow request
- [ ] 4. Speculation: draft, verify, accept - and the identity contract it must respect (payoff: greedy identity, or the whole idea is a lie)
- [ ] 5. The cost model: acceptance rate times tokens saved against verify cost (payoff: the model pair where speculation loses)

## Speculation cost model

| Quantity | Symbol | Your measurement | Break-even value |
|---|---|---|---|
| draft tokens per round |  |  |  |
| acceptance rate |  |  |  |
| verify cost vs decode step |  |  |  |
| net tokens per second |  |  |  |

## Project

**Build step S29 · concurrency and a trade.** Build the server before the speculation: speculation changes the loop's shape, and a loop you have not made concurrent yet is not ready for it.

Test the drain: send a request, start shutting down, and require the in-flight token to arrive and the connection to close cleanly.

## Deliverable

- A server with slots, a bounded worker, a metrics endpoint and a drain path
- A speculative decoder behind a flag, with an identity test and a cost model measured on one model pair
- A written verdict: enable it or not, with the numbers that decided

## How you know you passed

- Two concurrent sessions produce the same tokens as running them one after the other.
- Speculation's greedy output is identical to non-speculative greedy output; if it is not, the feature is off.
- The verdict quotes both sides of the trade, including the case where it loses.

## Reading

- code: `crates/minfer-cli/src/server/{mod,slot,worker,metrics,batch,toolcall}.rs`, `src/spec.rs`, `crates/minfer-cli/src/spec_verify.rs` at commit `88ffa5d`
- docs: `docs/OPENAI-CHAT-API-PLAN.md`, `docs/SPECULATIVE-DECODING-PLAN.md`, `docs/LLAMA-CPP-SPECULATIVE-ANALYSIS.md`
- docs: `docs/cuda_optimization_steps/83-d5-r-stage1-spec-loop.md`, `88-d5-r-stage5-final-battery.md`, `94-greedy-identity-and-d8-crossing.md`
- decisions: ADR-0017 (speculation refuses what its identity contract cannot carry), ADR-0059 (the batched worker queues against a bound), ADR-0072 (a draft proposal is the target's own filtered pick)
- [spec-decoding] Leviathan, Kalman and Matias, *Fast Inference from Transformers via Speculative Decoding* (2022) - arXiv:2211.17192
- [paged-attention] Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023) - arXiv:2309.06180

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
