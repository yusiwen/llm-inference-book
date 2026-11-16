# Chapter 20 · The Engine Loop

> - **Compute tier**: ✅ any laptop
> - **Status**: draft
> - **Build step S20**: A generation loop with prefill, decode, graph reuse, incremental multi-turn prefill and cancellation, driven from a command line.
> - **Rule of this book**: Sampling is Chapter 13 and the graph is Chapter 14. This chapter is the loop that ties them together and the state it keeps between steps. Serving several sessions at once is Chapter 29.

## First, the gist

With a graph, weights, a cache and a sampler, the engine is a loop: build once for the prompt's shape, execute
over all prompt tokens, take the last logits, sample, append, and repeat with a graph that differs only in its
input data. The discipline is in what changes. The first decode step meets a new shape, so it builds a second
graph; every later step reuses it; the cache grows underneath without any rebuild. Multi-turn conversation is the
same idea over an append-only prefix, and the two events that interrupt it - cancellation and context overflow -
are where the loop's bookkeeping becomes visible.

The measured loop on this engine and a 0.5B model:

- a **prompt token costs 4.94 ms** in prefill, where 34 tokens go through one graph;
- a **generated token costs 12.61 ms** in steady decode, and **99.2 per cent of that is the forward pass** - the
  sampler is 0.08 ms, which is why optimising it is optimising noise;
- a **graph build costs 0.6 ms and 0.4 ms**, once for the prompt shape and once for the decode shape, and every
  run builds exactly two of them whether it generates one token or twenty-four;
- the **second turn of a conversation prefills 14 tokens**, not the history it inherits, off a 101,058,161-byte
  session file whose header is the 36 bytes Chapter 18 predicted.

That is the chapter: the loop is cheap to describe and easy to get wrong in the same three places - what may be
reused, what "stop" means, and what the cache looks like after an interruption.

## What you'll be able to do

- Implement a loop whose per-step overhead is measured, and say which part of a step is not the model.
- Reuse a graph across steps and prove the reuse is valid: equal parameters, different input data, one build.
- Handle cancellation and context overflow without corrupting the cache - and know which of the two the engine
  decides and which one it merely detects.
- Put the loop behind two API tiers, and install its configuration once instead of reading the environment from
  twenty places.

## Lessons

- [ ] 1. Prefill, then decode: two shapes, two builds, one cache
- [ ] 2. The reuse cache: what must be equal, what may differ, and the invalidation you must not skip
- [ ] 3. Multi-turn: incremental prefill with an append-only prefix
- [ ] 4. Cancellation, stop ids and a drain that finishes the token in flight
- [ ] 5. Overflow: what happens at the context limit, and the three policies an engine can have
- [ ] 6. The engine as a library: two API tiers, and a configuration installed once

### Lesson 1 · Prefill, then decode: two shapes, two builds, one cache

<figure class="book-figure">{{#include ../figures/20-engine-loop.svg}}</figure>

*Figure 20.1 - the loop, with the measured costs on it. One graph for the prompt's shape, one for the shape of a single token, and the second one reused for every token after the first. The cache sits underneath both and is never rebuilt; the stop box is the only place the loop can leave.*

The single-shot loop lives in one module, and its doc comment is the shortest honest description of what it does:

<!-- code: src/engine/generate.rs::generate@e844591 -->
```rust
//! The CLI keeps the presentation (prompt rendering, the think-block
//! highlighter, the stats lines); this module keeps the inference: the prefill,
//! the sampler pipeline (penalties, grammar, mirostat, the speculative round
//! loop), stop strings, the trace marks and the KV writes.

```
<!-- /code -->

Everything the loop needs arrives in one struct, and the two fields that matter most here are `n_predict` (when
to stop asking) and `cancel` (whether someone has asked it to). What comes back is not a string but a **record of
how the run ended**, which is lesson 4's subject.

Run it and the loop narrates its own structure. `MINFER_REBUILD_TRACE=1` prints a line for every graph the engine
builds, and `MINFER_TIMING=1` splits each decode step into sample and forward time:

<!-- facts: loop.loop-runs -->
| Run | Shapes built | Build cost | Prefill | Decode |
|---|---|---|---|---|
| prefill-only, 34 prompt tokens, `-n 1` | nt=34, nt=1 | 0.6 / 0.4 ms | 202.5 tok/s | - |
| run 1: 36 prompt + 24 generated | nt=36, nt=1 | 0.5 ms / 0.5 ms | 240.3 tok/s | 79.3 tok/s |
| run 2: 36 prompt + 24 generated | nt=36, nt=1 | 0.6 ms / 0.5 ms | 210.4 tok/s | 77.2 tok/s |
| run 3: 36 prompt + 24 generated | nt=36, nt=1 | 0.6 ms / 0.4 ms | 236.1 tok/s | 80.8 tok/s |

Every run builds **exactly two graphs**, whether it generates one token or twenty-four: the prompt's shape and the decode shape. That is the reuse cache working - 24 decode steps, one build for decode - and it is why the second rebuild's `nt=1` is the interesting line: the shape of a step that repeats.
<!-- /facts -->

Two builds, always two: `nt=34` for the prompt and `nt=1` for the step that repeats. The second number is the
whole point of a reuse cache - the decode shape is built once and then used for every remaining token, so a
24-token generation pays one build, not twenty-four.

What that costs and what it does not:

<!-- facts: loop.loop-cost -->
| Phase | Per token | What it is |
|---|---|---|
| Prefill, 34 tokens in one graph | **4.94 ms** | one forward over the whole prompt |
| Decode, steady | **12.61 ms** | one forward over one token, re-reading the weights |
| of which: the forward pass | 12.81 ms (99.2%) | the model's arithmetic and memory traffic |
| of which: the sampler | 0.08 ms (0.8%) | top-k, top-p, the penalties, one token |
| Graph build, once per shape | 0.6 / 0.4 ms | build + backend assign + alloc, for the prompt shape then the decode shape |

Two numbers to keep. The build is **3.1% of one decode step** even though it happens once, so it cannot be what makes a decode step slow - a graph that never got reused would cost 24 builds for 24 tokens, which is 9.6 ms of 303 ms. And the sampler is **0.8%**: a decode step on this engine is the forward pass, and optimising the sampler is optimising noise.
<!-- /facts -->

Read the last line of that table twice. A build is **3.1 per cent of one decode step**, so a loop that rebuilt
every step would lose about 3 per cent - annoying, not fatal - and the reason to build a reuse cache is not the
3 per cent: it is that the build also **re-allocates the transient pool** (Chapter 17), and an allocator that is
re-entered every step cannot keep the pool warm, cannot serve a second session from the same arena, and cannot
capture a device graph. The reuse cache is a structural precondition for everything in Part 6, and its direct
saving is the smallest thing about it.

<figure class="book-figure">{{#include ../figures/20-step-budget.svg}}</figure>

*Figure 20.2 - the same numbers as one picture: a decode step is the forward pass and almost nothing else, the sampler is six pixels of a 348-pixel bar, and the build that a reuse cache exists to avoid is 3 per cent of a single step.*

A prefill token is cheaper than a decode token by a factor of 2.5, and that is the whole reason a prompt is sent
in one graph: 34 tokens in one pass read the 463 MiB of weights **once**. A decode step reads them again for every
single token, which is Chapter 5's bandwidth argument arriving as a wall-clock number.

### Lesson 2 · The reuse cache: what must be equal, what may differ, and the invalidation you must not skip

The cache's contract is one sentence, and it is worth reading as a specification rather than a comment:

<!-- code: src/graph/cache.rs::GraphCache@e844591 -->
```rust
//!
//! Mirrors llama.cpp's `llm_graph_params::allow_reuse` + `llm_graph_result`
//! reuse path. Key invariant: **graph topology is a deterministic function of
// ...
//! as-is, only input data is refreshed. `n_past` never appears here (it is
//! execution data).
//!
//! The allocator lives inside the cache and **survives graph rebuilds**: the
//! persistent KV regions are exactly the KV cache, so a prefill→decode
// ...
//! Only the node/buffer mapping is recomputed on rebuild.
//!
//! E4 S3: the cache holds **several** graphs, one per distinct `GraphParams`
//! (MRU first, bounded), and switching between them is the *assign* half of
//! reserve/assign: the target graph already carries its backend assignment, so a
//! switch is `alloc_graph` — liveness plus a re-map onto the allocator's reserved
//! slots — with no graph build, no fusion pass and no pool traffic. That is what
//! stops a server alternating 1-wide and N-wide decode steps (roadmap §14 row 3)
//! and a chunked prefill whose chunk sizes repeat (E3) from rebuilding every
```
<!-- /code -->

**Graph topology is a deterministic function of `GraphParams`.** Equal parameters mean identical topology, so the
graph is reused as-is and only the input *data* is refreshed. (Two places in that comment join their clauses with a
double arrow this book's print cannot carry; they say *equal parameters imply identical topology* and *a shape
change implies a rebuild*. The `// ...` marks stand for those two lines.) `n_past` - the position every decode step advances -
never appears in that function, which is exactly why a decode step can reuse a graph while everything that makes
it a *step* changes. If the cache key had included the position, the cache would never hit.

Two things follow, and both are easy to get wrong:

- **The allocator survives rebuilds.** The persistent KV regions are the cache (Chapter 18), not the pool; a
  prefill-to-decode transition rebuilds the graph and must not lose them. The module says so because a reader who
  assumed "rebuild" meant "start over" would write a loop that re-prefills every step.
- **The cache holds several graphs**, one per distinct parameter set, most-recently-used first. That is what stops
  a server alternating a 1-wide and an N-wide decode step from rebuilding every time - and it is the reason the
  measured `nt=1` build reappears only when the shape changes, not when the data does.

The chapter's own test for this is a count, not an opinion: generate N tokens, trace the builds, and require the
number of builds to be independent of N. Anything else means the parameters carry something per-step.

### Lesson 3 · Multi-turn: incremental prefill with an append-only prefix

A conversation is the same loop with a longer-lived cache and a harder invariant:

<!-- code: src/conversation.rs::conversation@e844591 -->
```rust
//! Core of the CLI multi-turn conversation session (docs/CLI-CONVERSATION-PLAN.md).
//!
//! Strategy: **append-only KV + incremental template rendering** (the llama.cpp legacy `examples/main` route).
//! The session's token stream accumulates in the KV region (position-addressed, persistent across graph rebuilds); each turn only
//! prefills/decodes the delta of the new message — long sessions cost O(turn delta) instead of O(full history).
//!
//! Key abstractions:
//! - [`Engine`]: the inference backend (`forward` + `reset_cache`). The real implementation [`GraphEngine`]
//!   holds a `ModelDef` reference + a session-private `GraphCache` + `n_ctx`; the mock implementation lets the state machine
//!   be fully tested without a model (§8.2 L1).
//! - [`TokenCodec`]: byte-level encode/decode (trait-ified `Tokenizer`), mock-friendly.
//! - [`Conversation`]: holds only logical state (messages, the token-stream mirror, the sampler window, the rollback point),
//!   **no cache** — the cache belongs to `GraphEngine`, avoiding the `&mut self` vs `&mut cache`
//!   borrow conflict, and keeps `Conversation` fully testable.
//!
//! KV consistency invariant (§5.4): after each turn's EOT insertion and before the delta append,
//! `stream_tokens` (the host-side mirror of the KV) == a token prefix of `tokenize(render(messages, false))`
//! (possibly missing the trailing template newline). `/regen` rollback (`turn_pos` pointer rewind) and
//! full re-render both rely on it.

```
<!-- /code -->

The invariant is the part to keep: after a turn's end-of-turn token and before the next delta is appended, the
host-side token mirror **must equal a token prefix** of what the template would render for the whole history. Two
features depend on it - `/regen` rollback, which rewinds a pointer, and the fallback path, which re-renders
everything. If the mirror and the cache disagree by one token, both silently produce a different conversation.

The measured behaviour of two turns through one session file:

<!-- facts: loop.loop-turns -->
| Turn | Prefilled | Generated | Reply |
|---|---|---|---|
| 1: "My name is Ada." | **34 tokens** | 8 | Hello Ada! It's nice to meet |
| 2: "What is my name?" | **14 tokens** | 5 | Your name is Ada. |

The second turn prefilled **14 tokens**, not the 56 of history it inherited: the prefix is append-only and the KV was resumed from the file. The cache that makes that possible is `101,058,161 B` on disk, whose header reads `MINFERKV` version 2, flags 0 (f32), backend tag 0, 24 layers, `n_ctx` 4096, `n_embd` 128, `row_elems` 128 - the same 36 bytes Chapter 18's byte map predicted.
<!-- /facts -->

14 tokens, not 46. The first turn's prompt and its reply are already in the cache; the second turn renders the
template for the whole history, diffs it against the mirror, and prefills only the new suffix. That is the
difference between a conversation that costs O(turn) and one that costs O(history) per turn: re-prefilling
everything is quadratic in the number of turns, and it is the single most common way a hand-written chat loop
becomes unusable after twenty messages.

The session file is also the honest end of Chapter 18's story: the byte map that chapter had to *derive* from the
writer's field order is here read out of a file the engine wrote, and its header matches - `MINFERKV`, version 2,
f16 flag clear, backend tag 0, 24 layers, `n_ctx` 4096, `n_embd` 128, `row_elems` 128.

### Lesson 4 · Cancellation, stop ids and a drain that finishes the token in flight

A loop can end for more reasons than "it finished", and an engine that returns a string has thrown that
information away. This one returns a record:

<!-- code: src/engine/generate.rs::GenerateOutcome@e844591 -->
```rust
/// What the run produced, for the caller's report.
#[derive(Debug, Default)]
pub struct GenerateOutcome {
    pub generated: usize,
    pub prefill_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub stopped_by_eog: bool,
    pub stopped_by_string: bool,
    pub hit_n_predict: bool,
    /// A caller's [`CancelToken`] fired: the run stopped at a token boundary
    /// before `n_predict`, and `hit_n_predict` is false by construction.
    pub cancelled: bool,
    /// A grammar refusal (the emitted text is a valid prefix).
    pub grammar_error: Option<String>,
    /// `(sample_secs, forward_secs, counted_tokens)` when `timing` was on.
    pub timing: Option<(f64, f64, usize)>,
    /// Speculative-decoding counters, when a draft engine was supplied.
    pub spec: Option<SpecStatsOut>,
}
```
<!-- /code -->

Five `bool`s and an `Option`, and between them they answer every question a host asks after a cancelled run: did
we produce what was asked (`hit_n_predict`), did the model stop (`stopped_by_eog`), did a stop string fire
(`stopped_by_string`), did the grammar refuse (`grammar_error`), and - the one that matters here - was this
**cancelled** rather than failed. The doc comment on `cancelled` spells out the cross-invariant: it is true only
before `n_predict`, and `hit_n_predict` is false by construction in that case.

Cancellation itself is a shared token with a documented boundary:

<!-- code: src/cancel.rs::CancelToken@e844591 -->
```rust
//! The cancellation handle both generation paths share (#796, #800).
//!
//! The single-shot loop (`crate::engine::generate`) and the conversation turn
//! loop (`crate::conversation`) stop at the same boundary — between decoded
//! tokens, and between speculative rounds — and both report the stop through a
//! named `cancelled` field rather than an error. One type, one home: the token
//! is an L1 primitive (the conversation layer needs it too), so it lives here
//! rather than in the facade module that first used it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A cancellation handle a host holds across threads (#796).
///
/// The loops check it once per decoded token — and once per speculative round,
/// so a draft round cannot outrun it — and stop at that token boundary. It
/// cannot interrupt a forward pass, which is a single call into the model; a
/// cancel that arrives mid-pass is honoured at the next token.
///
/// Cheap to clone: every clone shares one flag, so a signal handler or a UI
/// thread can hold its own. There is no reset: a cancelled token stays
/// cancelled, so a host that wants to run again makes a new one.
```
<!-- /code -->

*Cannot interrupt a forward pass ... honoured at the next token.* That is the honest design for a synchronous
loop: the alternative is to abandon a forward pass mid-flight, which leaves the cache holding a partially written
layer, and no later step can tell whether the region is consistent. The signal handler lives on its own thread
and only flips an atomic flag, because the loop itself is synchronous and cannot host it.

The four ends of this loop, with the evidence each leaves behind:

<!-- facts: loop.loop-stops -->
| Stop | Exit | Evidence |
|---|---|---|
| the model chose its end-of-generation token | 0 | `Generated:` line in every ordinary run |
| `n_predict` reached | 0 | `[turn 1] stopped by n_predict / context limit` - printed by the conversation frontend |
| cancelled (SIGINT) | 130 | `[cancelled] SIGINT - 0 of 600 requested tokens` |
| context overflow | 101 | `position 59 exceeds n_ctx 59 (KV region overflow)` |

Exit 130 is 128 + SIGINT, and it is a *decision*: the CLI prints its summary, says how many of the requested tokens it produced (with an em dash the book prints as a hyphen, because its figures and pages are ASCII), and exits with the shell's cancellation code so a script can tell a cancelled run from a finished one. Exit 101 is Rust's panic code, and it is not a decision the loop made - it is the KV region's own assertion firing, because the prompt filled the region exactly and the first generated token had nowhere to go.
<!-- /facts -->

The exit codes are the API for scripts. `130` is 128 + SIGINT, the shell's own convention, and the CLI prints its
summary and a `[cancelled]` line before exiting with it - a cancelled run is not a failed one, but it must not look
like a completed generation either. `101` is Rust's panic code, and the next lesson is about why that one is a
different kind of event.

The project step for this lesson is worth stating as a test rather than a feature: **cancel mid-generation, then
continue the same session, and require the next token to equal an uncancelled run's token at the same position.**
A cancellation that drops a half-written layer passes every other test and fails that one.

### Lesson 5 · Overflow: what happens at the context limit, and the three policies an engine can have

The KV region is sized when the model is loaded, and it is sized to hold the *context* - so what happens when the
conversation tries to exceed it is a policy question, and this engine has three answers in three frontends.

The single-shot path sizes the region to at least the prompt, and then asserts inside the builder rather than
writing past the end:

<!-- code: src/engine.rs::Engine::generate@e844591 -->
```rust
        // the regions are actually built for.
        let ctx = params
            .n_ctx
            .max(params.tokens.len())
            .min(self.model.max_seq_len());
        generate(
            self.model.as_ref(),
            &self.tokenizer,
```
<!-- /code -->

and the assertion it can reach:

<!-- code: src/models/qwen2/graph.rs::Qwen2Graph::forward_cached@e844591 -->
```rust
        debug_assert!(n_out <= nt);
        // Out-of-range positions would write past the KV regions (which are
        // sized n_kv_embd * n_ctx): fail loudly instead of corrupting memory.
        if let Some(&maxp) = positions.iter().max() {
            assert!(
                maxp < n_ctx,
                "position {maxp} exceeds n_ctx {n_ctx} (KV region overflow)"
            );
```
<!-- /code -->

Run that: `--n-ctx 32` with a 59-token prompt gives exit **101** and

```
position 59 exceeds n_ctx 59 (KV region overflow)
```

The region was clamped up to 59 to hold the prompt, and the first generated token wanted cell 59. The assert is
right - *fail loudly instead of corrupting memory* - and the policy that would prevent it belongs to the caller:
either refuse the request up front, or size the region for prompt **plus** `n_predict`.

The conversation path takes the second policy, and says so in its outcome struct: on overflow it **drops the
oldest turns** and continues, counting them in `dropped_turns` (0 means nothing was truncated). A server takes the
third: it refuses or queues the request, which is Chapter 29's subject.

Three policies, one engine, and the chapter's point is that none of them is wrong: what is wrong is leaving the
choice implicit, so that a user meets it as a panic in one frontend and as a shortened conversation in another
without either saying which one it is.

### Lesson 6 · The engine as a library: two API tiers, and a configuration installed once

A loop that only exists inside `main.rs` is a program, not an engine. The line between the two is drawn in the
crate root:

<!-- code: src/lib.rs::tiers@e844591 -->
```rust
//!
//! # Tiers
//!
//! - [`engine`] — the facade: [`Engine`](engine::Engine), [`Session`](engine::Session),
//!   `chat`/`complete`, `generate`, and the error types. The only semver-relevant surface before 1.0.
//! - [`advanced`] — the primitives the server drives: the graph cache and its params, the batch, the
//!   KV format gate, the tokenizer, the sampler, the grammar, the speculative engine and
//!   [`CancelToken`](cancel::CancelToken). Public, **not** semver-stable before 1.0.
```
<!-- /code -->

Two tiers, one stable: the facade (`Engine`, `Session`, `chat`, `complete`, `generate`) is the semver-relevant
surface, and the primitives the server drives are public but not stable before 1.0. The payoff is that the CLI,
the HTTP server and the visualizer are *consumers* rather than privileged insiders - and the loop that chapter 1
described as "one token at a time" is the same loop a library host runs.

The second half of that payoff is configuration, and it is where engines usually rot:

<!-- code: src/config.rs::EngineConfig@e844591 -->
```rust
//! Process-wide engine configuration (#763 S4).
//!
//! The engine used to read `MINFER_*` variables from arbitrary depth — a SIMD
//! dispatch, a graph builder, a model loader — which made the environment the
//! only way to configure inference. A library consumer cannot use that, and two
//! engines cannot differ.
//!
//! The rule now: **[`EngineConfig::from_env`] is the application's job** (the CLI
//! calls it once at startup and [`install`]s it); the engine reads [`get`]. A
//! consumer that installs nothing gets [`EngineConfig::default`], never the
//! host's environment.
//!
//! Scope: the whole engine. #763 S4 converted the platform-neutral core
//! (`graph`, `models`, `quants`, `vec_ops`, `conversation`); #769 converted the
//! device backends' own policy knobs (`src/cuda/`, `src/metal/`,
//! `src/graph/metal_backend.rs`, `src/models/*/graph.rs`), which used to read
//! `MINFER_*` directly at 60 sites. Nothing under `src/` reads the environment
```
<!-- /code -->

*The environment stops being the configuration.* One value, read once by the application, installed once; the
engine reads `get()`, and a consumer that installs nothing gets the defaults rather than its host's environment.
That is what makes two engines in one process possible - and it is why the measurements in this book can set
`MINFER_TIMING` in one run and not another without the engine having a mutable global.

## The loop's own cost

| Phase | What dominates | Measured on this model | What would reduce it |
|---|---|---|---|
| first prefill (cold) | the prompt's forward pass, one graph | 4.94 ms per prompt token (34 tokens, 202.5 tok/s) | a shorter prompt, or chunked prefill across steps |
| prefill (warm) | the same; the weights are resident | the same 4.94 ms within run-to-run noise | nothing in the loop: this is the model |
| first decode step (build) | build + assign + allocate, once | 0.4-0.6 ms, 3.1 per cent of a step | fewer parameter changes (Chapter 16's assignment is the cost) |
| steady decode step | the forward pass, 99.2 per cent of it | 12.61 ms: 12.81 forward, 0.08 sampler | nothing in the loop; Chapter 28 is the memory-traffic answer |
| multi-turn append | the new suffix's forward pass | 14 prompt tokens on turn 2, not 46 | an append-only prefix, which is what this is |

Every row is either the model or the build, and the loop's own share - the sampler, the bookkeeping, the string
matching - does not appear in the table because it does not appear in the measurement: 0.8 per cent.

## Project

**Build step S20 · the loop you will live in.** Instrument the loop before optimising anything: time per step,
tokens per step, cache size after each step. Every performance claim in Chapter 28 depends on this instrumentation
being honest, and the cheapest version of it is the one this chapter used: print a line per graph built, and a
line per thousand decode steps that splits forward from sampler.

Test cancellation by cancelling mid-generation and then continuing the same session. If the next token differs
from an uncancelled run at the same point, the cache was corrupted - and that test is the only way to know,
because a dropped layer does not crash.

## Deliverable

- A generating engine from the command line with per-step timing and a build log
- A graph-reuse test showing the number of builds does not grow with the number of generated tokens
- Cancellation and overflow tests, each asserting what the cache looks like afterwards, plus the written policy
  for each frontend

## How you know you passed

- Steady-state decode overhead (non-model time) is measured and reported, not guessed - and it is small enough
  that you can say what the next optimisation must be.
- A 24-token generation builds two graphs, and a 1-token generation builds two graphs.
- Cancelling and resuming produces the same tokens as an uninterrupted run.
- At the context limit each frontend behaves as its documented policy says, and no path panics by accident.
- Two `Engine`s in one process can hold different configurations, and neither reads the environment.

## Reading

- code: `crates/minfer-cli/src/main.rs`, `src/engine/generate.rs`, `src/graph/cache.rs`, `src/conversation.rs`, `src/advanced.rs`, `src/config.rs` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/inference_e2e_walkthrough/09-prefill-forward-path.md`, `docs/inference_e2e_walkthrough/13-decode-loop-graph-reuse.md`
- docs: `docs/CLI-CONVERSATION-PLAN.md`, `docs/CRATE-API-DESIGN.md`
- decisions: [ADR-0017](https://github.com/yusiwen/minfer/blob/master/docs/adr/0017-speculative-decoding-refuses-what-its-identity-contract-cannot-carry.md) (speculative decoding refuses what its identity contract cannot carry), [ADR-0079](https://github.com/yusiwen/minfer/blob/master/docs/adr/0079-the-engine-is-a-crate-with-two-api-tiers.md) (the engine is a crate with two API tiers), [ADR-0082](https://github.com/yusiwen/minfer/blob/master/docs/adr/0082-the-application-layer-is-its-own-package.md) (the application layer is its own package)
- [scaling-inference] Pope et al., *Efficiently Scaling Transformer Inference* (2022) - arXiv:2211.05102
- [paged-attention] Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023) - arXiv:2309.06180

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)

<!-- facts: loop.provenance -->
*`scripts/loop.py`; runs the binary with `MINFER_REBUILD_TRACE` and `MINFER_TIMING` on and counts the graph builds per run, reads the `Prefill:`/`Generated:` lines, runs two conversation turns through one `--session` file, decodes the 36-byte session header out of the `.kv` companion, sends SIGINT mid-decode, and drives a context overflow against `--n-ctx 32`; on **{'platform': 'macOS-27.2.1-arm64-arm-64bit-Mach-O', 'machine': 'arm64', 'python': '3.15.2'}**; the engine's own `Prefill:`/`Generated:` lines, median of 3 runs; graph builds counted from `MINFER_REBUILD_TRACE`; the session file is read as bytes (`<8s7I` header) and sized on disk. Command: see [the provenance index](../appendix/provenance.md#loop). Raw numbers: `facts/loop.json`.*
<!-- /facts -->
