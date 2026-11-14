# Chapter 8 · GEMV, GEMM, and the Thread Pool

> - **Compute tier**: ✅ any laptop (CPU only; this chapter is about dividing work, not about a device)
> - **Status**: written - one instrument (`scripts/threadbench.c`) run on both reference machines: the thread sweep, the token sweep, the per-call overhead sweep and the false-sharing comparison are measured rather than assumed, and the headline result contradicts the plan's claim
> - **Build step S8**: A matrix-vector path for decode and a matrix-matrix path for prefill, both using a persistent thread pool with an explicit work split.
> - **Rule of this book**: The attention-specific arithmetic is Chapter 10. What this chapter owns is the general matmul shape problem: how work is divided, and why the two regimes want different kernels.

## First, the gist

**GEMV** (matrix times vector) is what decode does: one row of activations, thousands of rows of weights, no reuse of the weights. **GEMM** (matrix times matrix) is what prefill does: many tokens at once, so each weight row read serves several activations. The textbook conclusion is that prefill is much cheaper per token because the weights are read once for many tokens, and that the same model therefore wants two kernels. The third ingredient is the thread pool: spawning threads per call is fine until a call takes milliseconds, and then the spawn dominates the work.

Two of those three claims survived measurement. The third did not:

<!-- facts: threadbench.ridge -->
| Machine | Arithmetic ceiling, GMAC/s | Streaming bandwidth | Ridge, MAC per byte | Intensity per token | Tokens where the pass stops being bandwidth-bound | Best measured gain over N passes |
|---|---|---|---|---|---|---|
| Apple M4 Pro (arm64) | 394 | 198.0 GB/s | 1.99 | 1.78 | 1.12 | 1.29x at 32 tokens |
| Intel i7-11700B (x86_64) | 132 | 45.1 GB/s | 2.92 | 1.78 | 1.64 | 1.34x at 2 tokens |
<!-- /facts -->

Read the last three columns together. **A token carries 1.78 MAC per byte** - that number is a property of the format (32 weights in 18 bytes), not of the model or the machine. The **ridge** is where the machine's arithmetic ceiling meets its bandwidth ceiling: 1.99 MAC per byte on the M4 Pro, 2.92 on the i7-11700B. So on the Mac a *second* token's arithmetic is already past what the bandwidth can absorb - the crossover is at 1.12 tokens - and the largest measured gain from batching 32 tokens into one pass is **1.29x**. On the NUC the ridge sits at 1.64 tokens, so batching 2 tokens buys 1.34x, which is the whole win, banked immediately.

That is not the "several times the throughput" the plan predicted, and the reason is not the shape. It is the kernel: this one reaches 88 per cent of the machine's streaming bandwidth but only about 30 per cent of its arithmetic peak, which pushes the ridge down to a single token. A kernel with three times the arithmetic efficiency would move the ridge to three tokens and make batching pay. **Which is the useful lesson: whether GEMM beats GEMV is a property of your kernel's efficiency ratio, and you cannot read it off the shape.**

Two things did hold up, both measured:

<!-- facts: threadbench.threads-sweep -->
| Machine | Threads | Pool, ms | Spawn per call, ms | Weight throughput, pool | Throughput, spawn | Speedup vs 1 thread |
|---|---|---|---|---|---|---|
| Apple M4 Pro (arm64) | 1 | 3.17 | 3.27 | 24.2 GB/s | 23.4 GB/s | 1.00x |
| Apple M4 Pro (arm64) | 2 | 1.55 | 1.61 | 49.5 GB/s | 47.6 GB/s | 2.05x |
| Apple M4 Pro (arm64) | 4 | 0.79 | 0.84 | 96.9 GB/s | 90.8 GB/s | 4.01x |
| Apple M4 Pro (arm64) | 8 | 0.44 | 0.54 | 174.0 GB/s | 142.9 GB/s | 7.20x |
| Apple M4 Pro (arm64) | 10 | 0.49 | 0.57 | 156.3 GB/s | 133.2 GB/s | 6.47x |
| Intel i7-11700B (x86_64) | 1 | 7.98 | 8.00 | 9.6 GB/s | 9.6 GB/s | 1.00x |
| Intel i7-11700B (x86_64) | 2 | 4.09 | 4.24 | 18.7 GB/s | 18.1 GB/s | 1.95x |
| Intel i7-11700B (x86_64) | 4 | 2.23 | 2.50 | 34.3 GB/s | 30.6 GB/s | 3.57x |
| Intel i7-11700B (x86_64) | 8 | 1.42 | 2.13 | 54.0 GB/s | 35.9 GB/s | 5.63x |
| Intel i7-11700B (x86_64) | 16 | 1.37 | 2.44 | 56.0 GB/s | 31.4 GB/s | 5.84x |
<!-- /facts -->

Splitting the rows across threads turns a 3.17 ms pass into 0.44 ms - **7.20x on eight threads** - and then stops. The tenth thread on the Mac is *slower* than the eighth (156 GB/s against 174), and on the NUC sixteen threads are indistinguishable from eight (56.0 against 54.0). That pass is at 174 GB/s against a 198 GB/s streaming measurement, so it is bandwidth-bound and there is nothing left for more threads to win.

And the pool: a per-call spawn costs **75 us** of overhead on the Mac and **372 us** on the NUC, where a dispatch through a persistent pool costs single-digit microseconds. That is the whole argument for build step S8, and it gets stronger the smaller the pass.

## What you'll be able to do

- Choose between GEMV and GEMM for a given shape, with the arithmetic that decides it - and say which side of the ridge you are on instead of quoting a ratio.
- Split work across threads by rows, explain why rows and not columns, and make the split contiguous so two workers never share a cache line.
- Build a persistent pool: generation handoff, completion counter, one submission at a time, main thread as a participant.
- State the bound on a queue and the policy for a lagging consumer as a design decision with a name, not as an error message.
- Find the shape below which the split stops paying, on your own machine, with the per-call numbers to prove it.

## Lessons

- [ ] 1. Two shapes, two kernels: the arithmetic intensity of each
- [ ] 2. Row splitting and the false-sharing traps between worker threads
- [ ] 3. A persistent thread pool: spawn once, feed work, wait for completion
- [ ] 4. The bound on a queue, and what happens when a consumer lags (payoff: backpressure is a design decision, not an error message)
- [ ] 5. When one thread is faster: tiny models and the overhead floor

### Lesson 1 · Two shapes, two kernels: the arithmetic intensity of each

<figure class="book-figure">{{#include ../figures/08-gemv-vs-gemm.svg}}</figure>

*Figure 8.1 - the same weight bytes streamed for one token or for four: the second pass reads the matrix once and multiplies it four times.*

A matmul in this book takes Q4_0 weights and Q8_0 activation rows and produces one logit per weight row: `od` rows, `id` columns, `nt` tokens.

```text
weights read   = od rows x (id/32) blocks x 18 bytes      one pass, whatever nt is
activations    = nt rows x (id/32) blocks x 34 bytes      small next to the weights
multiply-adds  = od x id x nt
```

For one token the ratio of arithmetic to weight traffic is `od*id / (od*id/32*18)` = **32/18 = 1.78 MAC per byte**, with no `od` or `id` in it at all. Batch `nt` tokens through the same pass and the intensity becomes `1.78 * nt` - the weight bytes are amortised, which is the entire idea behind GEMM.

Whether amortising them is worth anything depends on the machine, and the comparison has a name: the **ridge point**, the arithmetic rate the machine can sustain divided by the bandwidth it can sustain. Above the ridge, more arithmetic per byte is free and batching wins; below it, the pass is bandwidth-bound and the arithmetic you add is what you pay for.

The table at the top of this chapter computes both sides from measurements, which is the only way the comparison means anything:

- **the arithmetic ceiling** is not the CPU's peak specification, it is what *this kernel* achieved at the largest batch (`od*id*nt / time`), because that is the part you can actually use. On the M4 Pro that is 394 GMAC/s, against a ten-core NEON peak somewhere near four times that.
- **the bandwidth** is Chapter 1's measured streaming number for the same machine, not the memory's specification.

Their ratio, divided by 1.78, is the token count at which a pass stops being bandwidth-bound: **1.12 on the Mac, 1.64 on the NUC**. That one derived number explains the whole token sweep:

<!-- facts: threadbench.token-sweep -->
| Machine | Tokens | One GEMM, ms | N one-token passes, ms | Per token, ms | Ratio (N passes / 1 GEMM) |
|---|---|---|---|---|---|
| Apple M4 Pro (arm64) | 1 | 0.354 | 0.355 | 0.354 | 1.00x |
| Apple M4 Pro (arm64) | 2 | 0.734 | 0.776 | 0.367 | 1.06x |
| Apple M4 Pro (arm64) | 4 | 1.474 | 1.513 | 0.368 | 1.03x |
| Apple M4 Pro (arm64) | 8 | 2.933 | 3.585 | 0.367 | 1.22x |
| Apple M4 Pro (arm64) | 16 | 5.783 | 6.987 | 0.361 | 1.21x |
| Apple M4 Pro (arm64) | 32 | 11.045 | 14.299 | 0.345 | 1.29x |
| Intel i7-11700B (x86_64) | 1 | 1.341 | 1.320 | 1.341 | 0.98x |
| Intel i7-11700B (x86_64) | 2 | 1.955 | 2.614 | 0.977 | 1.34x |
| Intel i7-11700B (x86_64) | 4 | 3.981 | 5.306 | 0.995 | 1.33x |
| Intel i7-11700B (x86_64) | 8 | 7.991 | 10.678 | 0.999 | 1.34x |
| Intel i7-11700B (x86_64) | 16 | 16.247 | 21.349 | 1.015 | 1.31x |
| Intel i7-11700B (x86_64) | 32 | 33.115 | 43.210 | 1.035 | 1.30x |
<!-- /facts -->

Look at the per-token column on the Mac: 0.354 ms for one token, 0.345 for 32. Batching 32 tokens into one pass saved **three per cent per token**, and the ratio column - 32 one-token passes against one 32-token pass - tops out at 1.29x. On the NUC the per-token cost drops from 1.341 to 0.977 as soon as there are two tokens (1.34x) and then stays flat, which is exactly what a ridge at 1.64 tokens predicts.

The honest reading is uncomfortable, and it is the chapter's main result: **on a laptop CPU with a 32-weight block format, "prefill is much faster per token because of weight reuse" is not true, and the measurement says why.** The pass is already at 88 per cent of bandwidth with one token, so the only thing a second token adds is arithmetic. To make batching pay you have to reduce the arithmetic per token - wider blocks, better register reuse inside the thread, lower precision, a device with a higher ridge - which is Chapters 9 and 13, not this one.

Two caveats a careful reader will want, stated here rather than buried:

- **The weight-throughput column can exceed the streaming number.** The Mac pass reports 216 GB/s at one token against a 198 GB/s streaming measurement. That is not a contradiction: the streaming figure was measured on a 1 GiB array where nothing fits in cache, and this pass is 76.6 MB repeated five times, so part of it is served from cache. The NUC reports 54.0 GB/s against a 45.1 GB/s streaming figure for the same reason. Read both as "saturated", not as "peak".
- **The token sweep is one matmul, not a model.** `token_embd` is 76.6 MB and 896 wide; a model is two dozen layers of these plus attention. The *ratio* is a property of the format and the machine, so the shape of the result carries over; the absolute per-token milliseconds do not.

### Lesson 2 · Row splitting and the false-sharing traps between worker threads

The unit of work is a **row range**, because the weight row is the one thing in this computation that no two workers need to share:

```rust
fn chunk(parts: usize, idx: usize, total: usize) -> (usize, usize) {
    let a = total * idx / parts;
    let b = total * (idx + 1) / parts;
    (a, b)
}
```

That is the reference engine's split, and the multiplication before the division matters: `total * idx / parts` distributes the remainder across the later chunks instead of giving every worker a floor and leaving `total % parts` rows unassigned. With 151,936 rows over eight workers every chunk gets 18,992 rows exactly, but seven workers leave a remainder, and `total / parts` alone would silently drop it - a matmul that is wrong only for some thread counts, which is the worst kind.

Each worker computes `out[t*od + o]` for its own `o` range, so two workers write different floats. Different floats are not different cache lines, and that is the first trap:

<!-- facts: threadbench.sharing -->
| Machine | Threads | Contiguous ranges, ms | Interleaved rows, ms | Cost of sharing a line |
|---|---|---|---|---|
| Apple M4 Pro (arm64) | 10 | 0.023 | 0.099 | 4.3x slower |
| Intel i7-11700B (x86_64) | 16 | 0.051 | 0.786 | 15.4x slower |
<!-- /facts -->

The interleaved arm does the same dots on the same rows and writes the same outputs, only in a different order: worker `k` takes rows `k, k+parts, k+2*parts, ...`. On the NUC that is **15.4x slower**. The mechanism is the cache line, not the arithmetic: a core writes 64 bytes at a time, so two threads writing neighbours in the same line both claim ownership and the line bounces between their caches - coherency traffic per write instead of per line.

The lesson generalises past this kernel: **when you split for-work across threads, split it into contiguous ranges and give each range a whole number of cache lines.** Rows are convenient here because a row of 896 Q4_0 weights is 504 bytes - seven lines and a bit - so a contiguous row range is contiguous in the array. Column splits, and outputs of length `od` written interleaved, are where this bites; it is also why the reference engine's `chunk` returns row ranges and nothing else.

The second trap is the tail. Contiguous chunks of unequal size make the slowest worker the pass's duration: 1,001 rows over eight workers gives 125 rows to seven and 126 to one, which is fine, but the *first* call is not - the pages of a 76.6 MB weight matrix have to be faulted in, and whichever workers touch a region first pay for all of it. This benchmark runs the pass once before timing for exactly that reason, and so must any engine that reports a first-token latency.

### Lesson 3 · A persistent thread pool: spawn once, feed work, wait for completion

Thread-per-call is the obvious design and it is wrong by a measured amount:

<!-- facts: threadbench.overhead -->
| Machine | Shape | One thread, us | Persistent pool, us | Spawn per call, us | Pool speedup | Spawn overhead, us |
|---|---|---|---|---|---|---|
| Apple M4 Pro (arm64) | 4096 rows, 10 threads | 85.2 | 16.8 | 91.8 | 5.1x | 75.0 |
| Intel i7-11700B (x86_64) | 4096 rows, 16 threads | 373.2 | 29.4 | 401.6 | 12.7x | 372.3 |
<!-- /facts -->

At 4,096 rows (2.06 MB of weights, 3.7 MMAC) one thread takes 85 us on the Mac and the pool takes 17 - a 5.1x win, of which about 75 us of the spawn arm is `pthread_create` and `pthread_join` and only the remainder is the split itself. On the NUC the same comparison is 12.7x, with 372 us of spawn overhead. The reference engine's own source comment records the same finding from the other end: *spawning threads per matmul costs ~170 us (measured) - a persistent pool with an atomic generation handoff costs ~1-3 us per dispatch instead.*

A pool is therefore a small state machine with four parts, and each part exists because of a failure mode:

```text
gen     an atomic counter the caller increments to publish a job
done    an atomic counter the workers increment as they finish
job     the work descriptor, read by every worker after it sees gen change
gate    a mutex the caller holds across submit -> wait
```

**The handoff is by generation, not by notification.** Workers spin on `gen`, and after 8,000 spins they yield. That is a deliberate trade: a spin loop wastes a core while the pool is idle and wins microseconds when it is busy, and a matmul-heavy workload is busy. The cost is that an idle pool burns a little power forever; the alternative, a condition variable, costs a syscall per dispatch and shows up as several microseconds in a pass that may only take 17.

**The main thread is a participant.** With eight hired workers the split is into nine chunks: waiting costs as much as working, so the caller does the last range itself instead of blocking. This is why the pool's thread count is *workers + 1* everywhere in the code, and why a benchmark comparing "eight threads" against a pool has to compare the same participation count or it is measuring the difference in workers.

**The gate exists because a pool is global and a job is stack-local.** The descriptor carries raw pointers into a caller's buffers, and the caller returns as soon as its own range is done and the counter reaches `n`. Two concurrent submitters would clobber `job` and share `done`, so one caller could observe completion of the *other's* work and return while its own buffers die under late workers - undefined behaviour, not just a wrong answer. The mutex makes submission single-file and workers never take it, so there is no deadlock. That is the difference between a pool that passes tests and a pool that is correct; a reader building their own should copy the reasoning, not the code.

### Lesson 4 · The bound on a queue, and what happens when a consumer lags

A matmul pool has a cheap discipline: a job either fits the machine or it does not, and the caller waits. A **serving** queue does not - requests arrive from outside, and the engine must decide what to do when there are more of them than it can hold. The reference engine's ADR-0059 settles this with four decisions worth reading as a template:

- **Queue, with one number as the bound.** `serve_loop` admits at most one job per idle slot, oldest first, and keeps the rest in its own deque bounded by `MINFER_QUEUE_MAX` (default 64, the job channel's own capacity). One knob, so a client refused by the worker's bound and one refused by the transport's block see the same depth.
- **A refusal has a name.** A job beyond the bound is answered `503` with the reason, not dropped and not held forever. The predecessor decision - "reject loudly; queueing is a feature" - had left the two transports disagreeing about what "all slots busy" means, which is the real bug the ADR fixes: not the number, the inconsistency.
- **No wall-clock bounds.** Liveness bounds are counts of steps, never seconds, so a verdict is a property of the run rather than of how loaded the machine was. A run that advances its step counter without ever completing is still caught, by a per-run bound, so a wedged slot costs one request instead of the worker.
- **The unbounded queue is the rejected alternative**, for a reason worth quoting: a client would wait while the engine did work it could have answered quickly, and the refusal would be unnameable. A bound is what makes the refusal a sentence you can put in a log.

In this chapter's pool the same principle appears in miniature as a **threshold**: below some shape the split is not worth the dispatch, and the code must decide that by measuring rather than by taste. The reference engine uses one million MACs (`MIN_PARALLEL_MACS`) and runs the 512-row attention matmuls inline. That threshold is conservative against what this benchmark measures - at 128 rows the pool still wins on both machines - and understanding why is more useful than copying the constant: a decode step runs a hundred-plus matmuls, each with a barrier, so a pool entered for a 100 us shape pays its dispatch a hundred times, and the single-submission gate means those barriers serialise. **The threshold protects the pipeline, not the call.**

<!-- facts: threadbench.threshold -->
| Machine | Rows in the pass | One thread, us | Persistent pool, us | Spawn per call, us | Pool against one thread |
|---|---|---|---|---|---|
| Apple M4 Pro (arm64) | 128 | 2.62 | 1.53 | 78.65 | 1.71x |
| Apple M4 Pro (arm64) | 256 | 5.30 | 1.85 | 80.93 | 2.86x |
| Apple M4 Pro (arm64) | 512 | 10.55 | 3.89 | 82.47 | 2.71x |
| Apple M4 Pro (arm64) | 1024 | 21.04 | 6.25 | 82.95 | 3.37x |
| Apple M4 Pro (arm64) | 2048 | 42.13 | 9.17 | 83.89 | 4.59x |
| Apple M4 Pro (arm64) | 4096 | 84.64 | 13.46 | 89.77 | 6.29x |
| Apple M4 Pro (arm64) | 16384 | 338.90 | 39.95 | 122.45 | 8.48x |
| Intel i7-11700B (x86_64) | 128 | 11.25 | 2.12 | 321.59 | 5.31x |
| Intel i7-11700B (x86_64) | 256 | 22.80 | 3.35 | 325.42 | 6.81x |
| Intel i7-11700B (x86_64) | 512 | 43.76 | 4.75 | 326.34 | 9.21x |
| Intel i7-11700B (x86_64) | 1024 | 88.80 | 7.74 | 355.07 | 11.47x |
| Intel i7-11700B (x86_64) | 2048 | 176.85 | 14.50 | 380.18 | 12.20x |
| Intel i7-11700B (x86_64) | 4096 | 373.29 | 30.59 | 447.49 | 12.20x |
| Intel i7-11700B (x86_64) | 16384 | 2850.88 | 167.00 | 997.68 | 17.07x |
<!-- /facts -->

### Lesson 5 · When one thread is faster: tiny models and the overhead floor

The threshold table is the answer to this lesson's question, and it says something different from what the plan expected: **on both machines the pool wins at every shape measured**, down to 128 rows - 1.71x on the Mac, 5.31x on the NUC. So "when is one thread faster" is not answered by rows. It is answered by the *fixed cost*, and the fixed cost is not what you would guess.

**The dispatch cost is not constant.** At 4,096 rows the pool's dispatch plus imbalance is about 8 us on the Mac (17 us measured, minus 85/10 of work). At 128 rows the whole pool call takes 1.53 us while one thread takes 2.62 - so the dispatch at that size is nearer one microsecond than eight. The workers were already spinning when the job was published, and a job that finishes in a microsecond finishes before the first `yield`. Quote one number for "the dispatch cost" and you will be wrong by an order of magnitude at the other end of the range; the honest statement is a range that depends on how long the workers were idle.

**What is genuinely faster on one thread is spawning.** At 128 rows the spawn-per-call arm takes 79 us of wall clock to do 2.6 us of work - thirty times worse than one thread alone, and fifty times worse than the pool. So the real answer to "when one thread is faster" is: *always*, if your alternative is creating threads per call. The pool is not a way to make small work parallel; it is a way to make the parallel path affordable.

And there is a third case, which is not about size at all:

<!-- facts: threadbench.threads-sweep -->
| Machine | Threads | Pool, ms | Spawn per call, ms | Weight throughput, pool | Throughput, spawn | Speedup vs 1 thread |
|---|---|---|---|---|---|---|
| Apple M4 Pro (arm64) | 1 | 3.17 | 3.27 | 24.2 GB/s | 23.4 GB/s | 1.00x |
| Apple M4 Pro (arm64) | 2 | 1.55 | 1.61 | 49.5 GB/s | 47.6 GB/s | 2.05x |
| Apple M4 Pro (arm64) | 4 | 0.79 | 0.84 | 96.9 GB/s | 90.8 GB/s | 4.01x |
| Apple M4 Pro (arm64) | 8 | 0.44 | 0.54 | 174.0 GB/s | 142.9 GB/s | 7.20x |
| Apple M4 Pro (arm64) | 10 | 0.49 | 0.57 | 156.3 GB/s | 133.2 GB/s | 6.47x |
| Intel i7-11700B (x86_64) | 1 | 7.98 | 8.00 | 9.6 GB/s | 9.6 GB/s | 1.00x |
| Intel i7-11700B (x86_64) | 2 | 4.09 | 4.24 | 18.7 GB/s | 18.1 GB/s | 1.95x |
| Intel i7-11700B (x86_64) | 4 | 2.23 | 2.50 | 34.3 GB/s | 30.6 GB/s | 3.57x |
| Intel i7-11700B (x86_64) | 8 | 1.42 | 2.13 | 54.0 GB/s | 35.9 GB/s | 5.63x |
| Intel i7-11700B (x86_64) | 16 | 1.37 | 2.44 | 56.0 GB/s | 31.4 GB/s | 5.84x |
<!-- /facts -->

The Mac's tenth thread is **slower** than its eighth (156 GB/s against 174, ranges barely overlapping), and the NUC's sixteenth is indistinguishable from its eighth (56.0 against 54.0). Two machines, two reasons, one conclusion:

- On the Mac the pass is saturated on eight P-cores: 174 GB/s is 88 per cent of what the memory system will give, so a ninth and tenth worker have nothing to fetch, and the extra participation adds barrier cost and cache contention.
- On the NUC sixteen threads are eight cores with simultaneous multithreading: the hyperthreads share load and store units, so a compute-heavy kernel gains nothing and pays a little in shared-cache pressure.

A thread count is therefore a **measured** parameter with a shape, not a constant inherited from `nproc`. The reference engine reads the P-core count on macOS (`hw.perflevel0.logicalcpu`) precisely because the E-cores measurably hurt matmul throughput, and that choice is the same finding from the other direction: more threads is a hypothesis, and this table is the test.

## Shapes and kernels

The stub asked for this table filled with your own throughput. The measured column is per token for **one** matrix - `token_embd`, 76.6 MB, 896 wide - so it is a shape's cost, not a model's: a 0.5B model runs two dozen of these per token.

| Shape | Tokens | Weight reuse | Kernel | Measured per-token cost for this matrix |
|---|---|---|---|---|
| prefill, 2k prompt | 2000+ | weights read once for every token | GEMM, blocked | 0.345 ms (measured at 32 tokens; flat to 32, so this matrix alone would be about 690 ms for 2,000) |
| decode, step N | 1 | none - every byte read for one token | GEMV, row split | 0.354 ms on ten threads, 3.17 ms on one |
| speculative verify, k=4 | 5 | 5x per weight byte | GEMM, same kernel with `nt` = 5 | 0.368 ms (measured at 4 tokens) |
| batched decode, 8 slots | 8 | 8x per weight byte | GEMM, `nt` = 8 | 0.367 ms (measured at 8 tokens) |

The column that matters is the last one, and what it says is that on this machine the four rows cost nearly the same per token. That is not a property of the shapes; it is the ridge point sitting at 1.12 tokens.

## Project

**Build step S8 · the work split.** Write the pool yourself before reading the reference one. The lesson is in the failure modes you hit without a discipline: a slow worker with an unbounded queue, a job descriptor that outlives its caller, and a split worth less than its dispatch.

Time it at decode sizes, not at prefill sizes. A pool that looks free at 2,000 tokens can dominate at 1, and the threshold table is what that looks like when it does.

Order of work:

1. **Write the row kernel and the serial path first**, and time one pass. Everything else is a ratio against this number.
2. **Split by contiguous row ranges** with `total * idx / parts`, verify the output against the serial path element for element, then time 1, 2, 4, 8 and `nproc` threads. Expect the curve to flatten or turn down before `nproc`; find the turn and explain it.
3. **Write the pool**: generation counter, done counter, spin-then-yield, main thread as the last participant. Verify by forcing it to one worker and comparing outputs.
4. **Add the gate** and a test that submits from two threads at once. Without the gate that test should fail; make it fail first, then fix it, or you have not proved the gate does anything.
5. **Measure the per-call overhead** at shrinking shapes: 128, 256, 512, 1,024 rows. Find where the pool stops winning and write that number next to the code as the threshold, with the machine named.
6. **Compare GEMV against GEMM** on the same weights: `nt` tokens in one call against `nt` calls. Report the ratio *and* the ridge arithmetic - the ratio is the portable part, the milliseconds are the machine.

## Deliverable

- GEMV and GEMM paths, both producing identical output on the same fixture, with the serial path as the oracle and a test that proves it.
- A pool with a bounded submission discipline and a documented policy for a lagging consumer, plus a threshold that came from a measurement on a named machine.
- A measured cross-over: at how many tokens GEMM overtakes repeated GEMV, and the ridge arithmetic that explains the number rather than merely reporting it.

## How you know you passed

- Both paths produce identical numbers on the same fixture, and the slower one is the oracle.
- The thread sweep has a turn in it that you can explain - saturated bandwidth, SMT, or a barrier - rather than a monotone line you did not look at.
- The cross-over is quoted with the command, the shape, the machine and a run range, and it is consistent with `intensity per token` divided by `ridge`.
- A single-threaded run is available and matches, which is how you check the split rather than the pool.

## Reading

- code: `src/kernel/pool.rs` (`chunk`, `mm_rows`, `Pool`, `par_for`), `src/kernel.rs`, `src/graph/scheduler.rs` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/inference_e2e_walkthrough/10-cpu-matmul-kernels.md`, `08-scheduler-execute.md`
- decision: ADR-0059 (the batched worker queues against a bound and fails a lagging run)
- [scaling-inference] Pope et al., *Efficiently Scaling Transformer Inference* (2022) - arXiv:2211.05102 - the ridge-point analysis this chapter derives numerically: latency-optimal partitioning trades weight traffic against activation traffic, and the ratio decides

## Backfill

- [x] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - **answered for both shapes, and the answer is the ridge.** One pass over 76.6 MB at eight threads reads 174 GB/s of a 198 GB/s streaming ceiling: bandwidth-bound, 88 per cent. The same pass with 32 tokens moves the same bytes with 32 times the arithmetic and lands at 394 GMAC/s, which is arithmetic-bound. The ridge - 1.99 MAC per byte against the format's 1.78 per token - puts the crossover at 1.12 tokens, and every number in the token sweep follows from it.
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - this chapter contributes one constant on the cost side: a Q4_0 block is 32 weights in 18 bytes, so the format's intensity is 1.78 MAC per byte and **changes with the bit width on both sides of the ratio at once** - a narrower format reads fewer bytes but does not reduce the MACs. That is why quantization moves the ridge rather than the shape, and why a 4-bit engine is more likely to be bandwidth-bound than a 16-bit one.
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - the seam gains a second axis: not only `(weight dtype, activation dtype)` but `(shape, thread count)` with a measured threshold. A scheduler that dispatches by dtype alone will thread a 512-row matmul and pay for it a hundred times per token.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - correctness is unchanged from Chapter 7 (the serial path is the oracle, and the split must be element-identical), but the measurement protocol gains two traps: the dispatch cost is not a constant (8 us at 4,096 rows, about 1 us at 128), and a throughput number above the machine's streaming figure means the working set is partly cached - not that the machine beat its own memory.
