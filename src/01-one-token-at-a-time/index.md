# Chapter 1 · One Token at a Time

> **Compute tier**: ✅ any laptop (arithmetic on paper, plus a loop against a placeholder model) | **Status**: written - every number below is arithmetic on the measured bandwidths in [the reference machines](../appendix/machines.md) or on a stated assumption; nothing here was measured on a real model file, because no file has been parsed yet | **Build step S1**: A loop that reads a prompt, calls a placeholder `model.forward(ids) -> logits`, and prints one token per step. No arithmetic in the code yet: the skeleton every later chapter hangs a kernel on. | **Rule of this book**: This chapter measures the shape of the problem and nothing else. The model file is Chapter 2, the arithmetic starts in Chapter 6, the graph that schedules it in Chapter 14.

## First, the gist

A language model is a function. You give it a sequence of integers, and it gives you back one score per integer it knows how to produce. You pick one of those integers, append it, and call the function again. Repeat until it produces whatever your engine was told means "stop".

<div class="math-display">
$$
\text{ids} \;\rightarrow\; \text{model} \;\rightarrow\; \text{scores} \;\rightarrow\; \text{one id} \;\rightarrow\; \text{append} \;\rightarrow\; \text{ids}
$$
</div>

That is an inference engine. Not a summary of one: the whole loop is about forty lines of Rust, and you will have written it by the end of this chapter. Everything else in this book applies to those forty lines. The other twenty-eight chapters exist because each of the four things inside them has a cost, and the costs are not obvious.

The first cost to understand is that the loop runs in two regimes that behave nothing alike. Your whole prompt arrives at once, so the first pass multiplies the model by many token rows simultaneously - a matrix times a matrix - and the machine is paid per unit of arithmetic. Every token after that arrives alone, so each later pass multiplies the model by a single row, a matrix times a vector, and to produce one token the engine must read the entire model out of memory. The first regime is limited by how much arithmetic the machine can do. The second is limited by how many bytes it can move.

That asymmetry is the spine of the book. Since the second regime is bandwidth-bound, and the weights are what has to be moved, the number of tokens per second a single request can produce has a ceiling you can compute on paper before writing a kernel: *the model's size divided by the machine's memory bandwidth*. For a 4 GB Q4 model on a laptop measuring 198 GB/s, that ceiling is about 50 tokens per second. No amount of clever kernel work goes through it; the only ways past are reading fewer bytes or producing several tokens per read.

That number also decides the book's structure. Quantization (Part 2) attacks the numerator. Batching and speculation (Part 5, and Chapter 29) attack the implied "per token" in the denominator. The graph and its allocation (Part 4) decide whether each byte is read once or twice. Correctness and measurement (Part 6) are how you know any of it worked.

## What you'll be able to do

- Predict, from a model's file size and a machine's measured bandwidth, roughly how many tokens per second a single decode stream can reach - before running anything, and to within a factor of two of what a real engine achieves.
- Say which of prefill and decode an optimization is supposed to help, and name one optimization that helps prefill while doing nothing for decode (and one that does the reverse).
- Sketch the loop and the three pieces of state it carries - the KV cache, the position counter, the sampler window - and say what breaks when each is forgotten.
- Explain, without calculus, why arithmetic per byte decides which regime you are in, and why batching moves the line.

## Lessons

- [ ] 1. Ids in, ids out: the loop, and the three pieces of state it carries
- [ ] 2. Prefill and decode are different jobs: one prompt, two regimes
- [ ] 3. The roofline in one page, without calculus: arithmetic per byte
- [ ] 4. A decode ceiling you can compute by hand: file size divided by bandwidth
- [ ] 5. What an engine may be lazy about, and the four things it never may

### Lesson 1 · Ids in, ids out: the loop, and the three pieces of state it carries

The interface is integers, not text. A model never sees a `String`; it sees a `Vec<u32>` of token ids, each an index into a fixed vocabulary. Qwen3's vocabulary holds about 151,000 entries and Llama 3's about 128,000, and every token in both is a byte sequence of between one and a handful of bytes (textbook fact - the real number is in the model file's metadata, which we read in Chapter 2, and the mapping is built in Chapter 11).

The fact that a token is not a word is the first thing beginners trip over. "Unbelievable" is four tokens in one model and three in another; " 42" with its leading space is a different id from "42"; one emoji can be several tokens. None of that matters yet. What matters is that the unit of work is an integer, and that integers are the only thing crossing the model boundary - which is what makes the loop small and the arithmetic countable.

So one step has this shape:

```rust
// The whole engine, more or less. Everything else is making this cheap and honest.
let mut ids: Vec<u32> = tokenizer.encode(prompt);       // Chapter 11
let mut position: usize = ids.len();                    // state 2
let mut kv = KvCache::new(config);                      // state 1
let mut sampler = Sampler::new(seed);                   // part of state 3

model.prefill(&mut kv, &ids, 0);                        // Chapter 10: fill the cache
let mut last_id = *ids.last().unwrap();
loop {
    let logits: Vec<f32> = model.forward(&mut kv, last_id, position);  // one row
    let next = sampler.pick(&logits, &ids);             // Chapter 13
    if next == stop_id || ids.len() == max_tokens { break; }
    ids.push(next);
    last_id = next;
    position += 1;
}
tokenizer.decode(&ids)                                  // Chapter 11, byte-exact
```

Every later chapter replaces one line of that while keeping the loop's shape. Chapter 2 turns `model` from a placeholder into something that reads a file. Chapter 10 gives it attention. Chapter 14 rewrites how it dispatches. Chapter 20 gives it cancellation, streaming and a session. Chapter 29 puts several of them behind a socket.

The loop looks stateless - `forward(ids)` - but it is not, and this is the second thing beginners trip over: **the model function is stateless, so the engine must carry the state.** Three pieces, and each exists because recomputing what it holds would cost more than storing it.

**State 1: the KV cache.** Attention needs, for every token produced so far, the key and value vectors that token's layer computed. Without a cache, producing token 300 of a context would recompute the keys and values of the first 299 tokens at every step. With a cache, each token's keys and values are computed once and read thereafter. The size is worth computing now, because it is the first number in this book that grows with the conversation (arithmetic on stated assumptions):

```text
bytes per token = 2 (K and V) x layers x kv_heads x head_dim x bytes_per_element
                = 2 x 32 x 8 x 128 x 2
                = 131,072 bytes = 128 KiB per token
```

At 4,096 tokens that is 512 MiB for one conversation - half a gigabyte a naive engine does not budget for, allocated per layer and reused across turns. On a machine with 128 GB of unified memory it sounds free; multiply by eight concurrent slots and it is 4 GiB, which is the difference between a model that fits and one that does not. Chapter 18 is entirely about this structure: cells, spans, prefix sharing, defragmentation. Nothing in this chapter needs any of that - a `Vec<f32>` per layer is enough - but the state has to live somewhere named, because the loop will not get smaller later.

**State 2: the position counter.** A token's meaning depends on where it is. The positional encoding (RoPE in every model this book builds) is a function of the absolute position, so the engine must know it, and must keep it correct across turns of a conversation, across a prefix shared between two requests, and across the moment a cache is truncated. The failure mode is nasty because it is quiet: an off-by-one gives output that is slightly worse - `the the` instead of `the`, a wrong date, a broken list - rather than a crash. This is the first place where "it runs" and "it is right" diverge, and the divergence is invisible without a reference output.

**State 3: the sampler window.** Sampling from logits needs more than the logits: a repetition penalty needs recent ids, a grammar constraint needs the parser's state, a seeded sampler needs the RNG's state, and a stop-string check needs a window of decoded text. That state is *not* part of the model's computation and must not be baked into it - Chapter 13's point is that sampling is a swappable policy over logits - but it does have to be carried, seeded and reset per request.

A loop that forgets any of the three still compiles. It even produces plausible text for a while. That is why this chapter exists before the first kernel: the cheap version of the engine gets the *state* right and leaves speed for later.

**The placeholder is part of the deliverable.** `forward` returns a `Vec<f32>` of vocabulary length. Make it return zeros, take the argmax, and you get id `0` forever - useless output and a perfect test. The loop is correct if it produces exactly `max_tokens` ids, stops on the stop id, and prints the positions it used. Every later chapter can then be judged against it: does the loop still behave identically when the placeholder is replaced by real arithmetic?

### Lesson 2 · Prefill and decode are different jobs: one prompt, two regimes

Here is the same weight matrix, used two ways.

<figure class="book-figure">{{#include ../figures/01-two-regimes.svg}}</figure>

*Figure 1.1 - prefill multiplies the weights by every prompt row at once and is limited by arithmetic; decode multiplies them by one row and is limited by the bytes it reads.*

Take one layer of a 7B model: a weight matrix of roughly 4,096 by 4,096 values. During prefill, a 500-token prompt arrives as a 500 by 4,096 activation block, and the layer produces a 500 by 4,096 output block. Each weight is multiplied by 500 different activation values. The cost is arithmetic, and the machine's job is keeping its floating-point units busy. This is what compute-bound means: the bytes are a small fraction of the work.

During decode, the same matrix meets a 1 by 4,096 row. Each weight is used exactly once, contributes to one output number, and is discarded until the next token asks for it again. The arithmetic is 16.7 million multiply-adds - microseconds of work - and the bytes are 16.7 million values, about 9.4 MB at four and a half bits each. The floating-point units idle; the memory system does not. That is bandwidth-bound.

For a whole model and one token (arithmetic on stated assumptions): decode does about **2 x params** FLOPs - 14 GFLOP for a 7B - and reads **one file's worth** of bytes, 4 GB at 4.5 bits. Prefill of an N-token prompt does 14N GFLOP and reads the same 4 GB once. So prefilling a 100-token prompt reads 4 GB and does 1.4 TFLOP; a hundred decode steps read 400 GB and do the same 1.4 TFLOP. Identical arithmetic, one hundred times the bytes.

Two consequences follow, and they are the two things most worth knowing before you optimize anything.

**Batching helps decode, and does not help one user's prefill.** If eight requests decode together, the weights are read once for eight tokens: cost per token falls by up to eight, and the regime drifts back toward the prefill shape. That is the whole economic argument for a serving engine, and it is why Chapter 29's server has slots instead of one loop.

**The two regimes are two different latency numbers.** Time to first token (TTFT) is paid by prefill and grows with prompt length, quadratically in its attention part. Inter-token latency (ITL) is paid by decode, is nearly flat in prompt length for the weight read, and grows with context length because the KV cache must be read too. A user who says "it is slow" is complaining about one of these two, and the fixes do not overlap. Asking which one is not pedantry; it is the first question of every optimization in this book.

### Lesson 3 · The roofline in one page, without calculus

Engineers compare the two regimes with one ratio: **arithmetic intensity**, the FLOPs performed per byte moved. It has no units worth arguing about - it is FLOP/byte - and everything else is multiplication and division.

For a decode step whose weights are quantized to b bits, every parameter contributes two FLOPs and b/8 bytes:

<div class="math-display">
$$
\text{intensity} \;=\; \frac{2\ \text{FLOP}}{\text{parameter}} \Big/ \frac{b/8\ \text{byte}}{\text{parameter}} \;=\; \frac{16}{b}\ \text{FLOP per byte}
$$
</div>

| Weight format | Bits per weight | Intensity (FLOP/byte) |
|---|---|---|
| f16 | 16 | 1.0 |
| Q8_0 | 8.5 | 1.9 |
| Q4_0 | 4.5 | 3.6 |
| Q4_K_M (mixed) | ~4.8 | 3.3 |
| Q2_K | ~2.6 | 6.2 |

A machine, separately, has a **ridge point**: the intensity at which its arithmetic capacity and its bandwidth run out at the same moment. Above the ridge a kernel is compute-bound and faster arithmetic helps; below it a kernel is memory-bound and only fewer bytes help. The ridge is the machine's peak FLOP/s divided by its bandwidth, so it is a property of the machine, not of your model.

You cannot place your machine's ridge point yet - you will have both inputs measured by Part 6, and the bandwidth half is already on [the reference machines page](../appendix/machines.md) - but the shape is enough to reason with, and two conclusions are robust enough to state now.

- **A GPU's ridge is far above 3.6.** Its arithmetic is enormous relative to its bandwidth, so a quantized decode step on a GPU is deeply memory-bound: more FLOPs are nearly free and the bytes are the whole problem. This is why integer dot products (Chapter 24) matter more there than any tile size.
- **A CPU's ridge sits close to 3.6** (my judgement, and the reason is structural: a CPU has a handful of vector lanes per core, not thousands, so its FLOP/s is small against the same memory system). A CPU decode step has no slack in either dimension, so a kernel that is careless in either one shows up immediately. That is the honest reason a CPU engine is a harder engineering problem than a GPU one, and Chapter 8's measurements are where it becomes checkable.

Two corollaries the rest of the book leans on.

**Batching multiplies intensity by the batch size.** B decode steps sharing one weight read have intensity 16B/b, so a batch of eight moves a Q4 model from 3.6 to 29 FLOP/byte - past many CPUs' ridge points, and into the range where arithmetic starts to matter. This is the arithmetic behind "the same engine is memory-bound for one user and compute-bound for sixty".

**The KV cache is bytes too.** Nothing in the formula counts the cache, but decode reads it every step and it grows with context. At 128 KiB per token, an 8,000-token conversation reads an extra gigabyte per step - which at 198 GB/s is 5 milliseconds, a quarter of a token's budget at the 4 GB model's ceiling. Long context moves the decode bound from the weights to the cache, which is why Chapter 18 treats the cache as a subsystem and not a `Vec`.

### Lesson 4 · A decode ceiling you can compute by hand: file size divided by bandwidth

The ceiling follows from Lesson 2's accounting and needs nothing but a file size and a bandwidth:

<div class="math-display">
$$
\text{tokens per second} \;\le\; \frac{\text{bandwidth}}{\text{bytes read per token}} \;\approx\; \frac{\text{bandwidth}}{\text{file size}}
$$
</div>

The approximation in the second step is the important one: it assumes every byte of the file is read exactly once per token, which is what a well-written decode kernel does - no byte left in memory, no byte read twice. A kernel that reads a weight twice halves the ceiling; a kernel that skips a weight is simply wrong.

<figure class="book-figure">{{#include ../figures/01-decode-ceiling.svg}}</figure>

*Figure 1.2 - the ceiling as a nomogram: file size on one axis, measured bandwidth on the other, iso-lines of tokens per second between them, and this book's three machines marked where they fall.*

Worked once, slowly, because this is the calculation every later speed claim is checked against.

A 7B model at four and a half bits per weight is `7e9 x 4.5 / 8 = 3.94e9` bytes, about 4 GB (arithmetic, and the real number comes from the file's own metadata in Chapter 2 - quantized formats pad and mix, so 4 GB is an estimate to within about ten per cent). The M4 Pro's CPU-side measured read bandwidth is 198 GB/s. One token therefore costs at least `4 / 198` seconds = 20.2 milliseconds, and `1 / 0.0202` = **49.5 tokens per second** (arithmetic on a measured input).

Do the same division on the appendix's machines and it becomes a table worth keeping:

| Where the engine runs | Bandwidth | 4 GB model | 0.34 GB model (0.6B at 4.5 bits) |
|---|---|---|---|
| M4 Pro, CPU | 198 GB/s | ~50 tok/s | ~580 tok/s |
| M4 Pro, Metal | 212 GB/s | ~53 tok/s | ~620 tok/s |
| GB10, CUDA | 242 GB/s | ~61 tok/s | ~710 tok/s |
| NUC11, CPU | 45 GB/s | ~11 tok/s | ~130 tok/s |

Four things this table is not.

**It is not a forecast.** Real engines land at 70-85 per cent of their ceiling: the KV cache is read too, every step pays launch and dispatch overheads, and the logits must reach the sampler. A ceiling of 49.5 that an engine reaches 38 of is a healthy engine, not a disappointing one.

**It does not apply to small models.** A 0.6B model's 1.7-millisecond step is shorter than the cost of dispatching a few hundred kernels, so at that size the ceiling stops describing anything real and the engine becomes launch-bound (my judgement; Chapter 20 is where per-step overhead becomes measurable). That is why the book measures with a 7B-class file and treats sub-2B numbers as a different problem.

**It says nothing about prefill.** For a prompt of N tokens the cost is arithmetic, and the bound is a FLOP bound instead: `2 x params x N` FLOPs against the machine's measured FLOP/s.

**It is per stream, not per machine.** Add a second concurrent request and, if the engine batches, the two streams together can beat one stream's ceiling, because the same bytes serve both.

Now the check that keeps this honest instead of decorative. The reference machine's own serving record reports a 27B NVFP4 model at 11.8 tokens per second (*reported* - the author's earlier measurement on that machine, not reproduced here). That model's weights are 21.8 GB, so `21.8 GB x 11.8 tok/s = 257 GB/s` of implied read bandwidth - against the 242 GB/s the CUDA kernel measured on the same box. The real engine runs at 94 per cent of the kernel's streaming rate, which is what a bandwidth-bound decode loop should look like. It is also why this book treats "within a factor of two of the ceiling" as a pass condition rather than a coincidence.

Your own version of that check is this chapter's deliverable: pick a model file you actually have, one machine, measure the bandwidth with the appendix's `membw.c`, divide, then compare against any engine you can run. If your number is off by more than 2x in the *optimistic* direction, you have a bug in the arithmetic or in the measurement. If the real engine is far under the ceiling, the interesting work starts.

### Lesson 5 · What an engine may be lazy about, and the four things it never may

An engine that tries to be complete on day one never runs. A great deal can be postponed without lying to yourself about it, provided the postponement is written down.

Fair game to postpone, and when each one bites:

- **The tokenizer's edge cases.** Start with the common path; the merge table's awkward entries (multi-byte characters split across merges, trailing whitespace) bite in Chapter 11, where a fixture set - not intuition - is the fix.
- **Sampling features.** Greedy and top-k are twenty lines. Top-p, min-p, mirostat, XTC and grammars are Chapter 13, and each is a policy you can add without touching the model.
- **Multiple architectures.** One architecture until Chapter 19. The generic graph is what makes the second one cheap, so building for two on day one buys nothing.
- **Threading.** Single-threaded until the kernel exists and is correct. A parallel implementation of a wrong kernel is a faster wrong kernel; Chapter 8 is where the pool arrives, with a scalar oracle already in place.
- **Every GPU.** Zero device code until Chapter 21. Metal and CUDA are worth roughly half the book, and none of it is a prerequisite for a working engine.

Now the four that cannot be postponed, because each is a decision that is expensive to reverse and cheap to get right the first time.

1. **The mapping between ids and bytes.** Decoding must reproduce the original bytes exactly - including bytes that came from a chat template - and must be idempotent across a round trip. A tokenizer that "mostly" round-trips produces an engine that is subtly wrong in a way no benchmark catches.
2. **The bookkeeping of state: positions and the cache.** Both are one integer and one buffer, and both fail silently. The rule: the engine owns them, the model never does, and the identity gate in Chapter 27 hashes them alongside the logits.
3. **The numeric contract.** For every operation, the engine says what precision it computes in and what it stores - f32 accumulation over quantized inputs, f16 storage for the cache, f32 for the norm. Not because any single choice is sacred, but because an undocumented one drifts per backend, and then Metal and CUDA disagree with each other and with the CPU (Chapter 9 builds the ledger, Chapter 27 enforces it).
4. **An oracle before an optimization.** The scalar, obviously-correct version of every kernel comes first, and every faster version is checked against it. The order is not taste: an optimization without an oracle cannot be distinguished from a bug that makes numbers faster.

All four are about correctness, and none is about speed. That is the chapter's parting claim, and the book's: the loop is small, the arithmetic is simple, and what actually constrains an inference engine is that the cheap version has to stay *right* while it gets fast.

## The two regimes

| Regime | What it multiplies | Bound by | What helps | What does not help |
|---|---|---|---|---|
| Prefill (a prompt of N tokens) | an N x d activation block by each d x d weight matrix | arithmetic: FLOPs, with an attention part that grows with N squared | larger tiles, more threads, integer dot products on the wide units; batching to fill the machine | a streaming memory layout, if there is no compute behind it |
| Decode (one token per step) | a 1 x d activation row by each d x d weight matrix | bytes: the weights plus the KV cache | fewer bytes (quantization), one fused pass that reads each weight exactly once, batching to amortize the read | more FLOPs, more threads than the bandwidth can feed, tiles larger than the row |

## Project

**Build step S1 · the loop, before arithmetic.** Three empty subsections in your notes, filled as you go: the state the loop carries, the quantity each regime is bound by, and the one number you will measure yourself in Chapter 28.

Write the loop against a placeholder model first. Everything this chapter asserts is checkable with a fake `logits` vector of zeros, and a loop that is right against zeros is the thing you debug the real model against - so the placeholder is not a throwaway.

Concretely, in your own tree (the [starter skeleton](https://github.com/yusiwen/llm-inference-book/tree/main/starter) already has the crates):

- `myfer-engine/src/lib.rs`: a `Model` trait with one method, `fn forward(&mut self, id: u32, position: usize) -> Vec<f32>`, and a `ZeroModel` that returns a vocabulary-sized vector of zeros.
- `myfer-engine/src/loop.rs`: the loop above - prompt in as ids, one token per iteration, a stop id, a maximum length, and a print of `(position, id)` per step so the bookkeeping is visible.
- `notes.md`: the three subsections. The state you carry, the quantity each regime is bound by, and the ceiling you computed for your own machine, with both inputs named.

## Deliverable

- `loop-stub` in your own tree: reads a prompt from argv, prints the ids it was given, then prints one token per iteration until a stop id or a limit.
- A one-page ledger: the decode ceiling for one model on one machine, with both inputs (file size, and *measured* bandwidth) named, and the command used to measure the bandwidth.
- A written answer to "prefill or decode?" for four optimizations you have heard of - quantization, batching, flash attention and speculative decoding are a fair set.

## How you know you passed

- Your ceiling is within a factor of two of what a real engine achieves on the same machine, and you can say which way it errs and why.
- You can name the regime each of your four optimizations targets without looking at the answer.
- Your loop prints a position sequence with no gaps and no repeats, and stops on exactly the condition you wrote down - checked against the zero model, not against a real one.
- No kernel code exists yet, and none is needed: the test is the arithmetic and the bookkeeping.

## Reading

- code: `src/engine.rs`, `crates/minfer-cli/src/main.rs` (the decode loop) - for shape only, at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/ARCHITECTURE.md` section 3 (the inference pipeline)
- [scaling-inference] Pope et al., *Efficiently Scaling Transformer Inference* (2022) - arXiv:2211.05102 - the analysis-versus-batch trade, and the memory-bound decode argument this chapter leans on
- [roofline] Williams, Waterman and Patterson, *Roofline: an insightful visual performance model for multicore architectures* (CACM, 2009) - no arXiv ID (journal paper) - where intensity and the ridge point come from
- [caqa] Hennessy and Patterson, *Computer Architecture: A Quantitative Approach* (6th edition, 2017) - no arXiv ID (book) - chapters 2 and 4, on why a bandwidth figure is a ceiling and not a rate

## Backfill

- [x] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - **prefill is arithmetic, decode is bytes**; the ratio that decides between them is intensity, `16/b` for a quantized decode step, against the machine's ridge point. The table above is the first attempt at the answer.
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - named here as the numerator of the ceiling; the mechanism is Part 2.
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - not yet: this chapter's only device is a `Vec<f32>`.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - partly: **the ceiling is the yardstick every later speed claim is measured against**, and Lesson 5's four non-negotiables are the correctness side in miniature.
