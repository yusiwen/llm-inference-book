# Chapter 13 · Sampling and Constrained Decoding

> - **Compute tier**: ✅ any laptop
> - **Status**: draft
> - **Build step S13**: A sampler pipeline with penalties, top-k, top-p, temperature and a seed you control, plus a grammar mask that makes one class of outputs impossible.
> - **Rule of this book**: This chapter turns scores into an id. The loop that calls it per step is Chapter 20; speculative decoding, which changes how many ids arrive at once, is Chapter 29.

## First, the gist

The model ends every step with one number per vocabulary entry - 151,936 floats for the model this book
measures. None of them is a token. Something has to pick one, and the obvious pick (take the biggest) has a
failure mode you have already seen if you have ever run a greedy model for two hundred tokens: it loops,
repeats an opening phrase, and walks into a sentence it cannot leave.

A **sampler** is the component that decides. It reshapes the scores before choosing: a penalty on tokens that
already appeared, a cap at the top few candidates, a cut at the smallest set that carries most of the
probability, a temperature that flattens or sharpens what is left. Then one token is drawn from what survives,
with a generator you seed so the run can be repeated.

The second half of the chapter is the strictest form of choice. A **grammar** masks every token that would make
the output invalid, which turns "please answer in JSON" from a request into a guarantee: the model cannot emit a
token that breaks the schema, because those tokens have probability zero before the draw.

Every number in this chapter comes from a real run of the engine on the model file this book measures: the CLI is
executed, it dumps the logits it computed, and this book's own re-implementation of the pipeline replays the
engine's token stream from those logits. Three replays - one greedy, one with the repeat penalty on, one on a
second prompt - come out **byte-identical** to what the engine printed. That is the strongest claim available
here: not "the pipeline looks right", but "the engine's logits plus this pipeline reproduce the engine's output,
and here is the code that does it".

## What you'll be able to do

- Implement the sampler pipeline in the order the reference uses, and prove reproducibility with a fixed seed.
- Explain what top-p does that top-k does not, and give a case where each one misbehaves.
- Say what a seed fixes and what it cannot, and why a greedy run ignores your top-k.
- Compile a schema to a mask over the vocabulary and show that an invalid token is impossible, not unlikely.

## Lessons

- [ ] 1. Scores to a distribution: penalties, top-k, top-p, and what temperature does not do
- [ ] 2. Order matters: the pipeline is a sequence of decisions, and swapping two stages changes the output
- [ ] 3. Seeded sampling and reproducibility, and what a seed does not fix
- [ ] 4. Stop conditions: stop strings, stop ids, and the byte-level matching nobody gets right first
- [ ] 5. Grammar masks: one stage over the vocabulary, and a constraint that cannot be violated

### Lesson 1 · Scores to a distribution: penalties, top-k, top-p, and what temperature does not do

One step of a real run, from the logits to the single id that was drawn:

<!-- facts: sampler.smp-pipeline -->
Step `logits_prefill_00000.f32`, prompt `My favourite programming language is`, top-1 mass before any stage: 0.394.

| Stage | Candidates left | What it did |
|---|---|---|
| raw logits | 151,936 | every vocabulary entry |
| repeat penalty 1.0 | 151,936 | 0 logit(s) moved - nothing repeats |
| top-k 40 | 40 | the tail is set to -inf |
| top-p 0.95 | 21 | the smallest set reaching the mass |
| temperature 0.8 | 21 | the set does not change; the weights do |
| draw | 1 | one id, by the generator |
<!-- /facts -->

Read the two middle rows together. **Top-k is a lid.** It keeps exactly forty candidates, always, because it is
a count. **Top-p is the cut.** It keeps the smallest set whose probability adds up to 95 percent, and the size of
that set is a property of the distribution, not of the configuration - twenty-one here, one on a step where the
model is certain. A sampler configured with both is saying "at most forty, and once you have covered 95 percent,
stop" - and in this run the second clause did all the work.

<figure class="book-figure">{{#include ../figures/13-sampler-pipeline.svg}}</figure>

*Figure 13.1 - the funnel on the step measured above (left) and the same top-p count on all 24 steps of the run (right). Top-k kept 40 on every step and never bound; top-p kept between 1 and 21, median 2.5. Bar lengths are log10 of the candidates left, because 151,936 and 40 are not comparable on a linear axis.*

That step is not the first one in the run, and it was not chosen by taste. The pipeline's decisions are only
visible on a step where the model is actually undecided, so five candidate prompts were measured and the flattest
one won:

<!-- facts: sampler.smp-prompts -->
| Candidate prompt | Top-1 mass | Tokens for 90% | Entropy |
|---|---|---|---|
| `The capital of France is` | 0.780 | 2 | 0.87 |
| `Once upon a time` | 0.584 | 15 | 2.20 |
| `The three primary colours are` | 0.839 | 5 | 1.10 |
| `My favourite programming language is` **chosen** | 0.394 | 26 | 2.88 |
| `Q: What is 17 times 23? A:` | 0.421 | 3 | 1.33 |
<!-- /facts -->

The spread is the point. On `The capital of France is` the top candidate already carries 0.78 of the probability,
so every stage below it is decoration: whatever you cut, the same token wins. On the prompt that was chosen, the
top candidate carries 0.39 and twenty-six tokens are needed to cover ninety percent - a distribution with room
for a pipeline to change something.

The nucleus cut itself - the engine's own words for it are in the doc comment above the function, which this
chapter elides because it contains a `less-than-or-equal` sign this book's character gate refuses; the sentence
it sits in says the survivor set is at most `k` entries:

<!-- code: src/sampler.rs::apply_top_p@88ffa5d -->
```rust
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p <= 0.0 || p >= 1.0 {
        return;
    }
    let survivors: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > f32::NEG_INFINITY)
        .map(|(i, &v)| (i, v))
        .collect();
    if survivors.is_empty() {
        return;
    }
    if survivors.len() > 1024 {
        apply_top_p_full(logits, p);
        return;
    }

    // Softmax over the survivors (identical to the old full-array softmax since
    // the masked entries contribute exp(-INF)=0 and 0.0 doesn't change the f64
    // running sum).
    let max_val = survivors
        .iter()
        .fold(f32::NEG_INFINITY, |a, &(_, v)| a.max(v));
    let sum: f64 = survivors
        .iter()
        .map(|&(_, v)| ((v - max_val) as f64).exp())
        .sum();
    let mut cand: Vec<(usize, f32)> = survivors
        .iter()
        .map(|&(i, v)| (i, (v - max_val).exp() / sum as f32))
        .collect();

    // Stable descending sort by probability (ties keep index order, matching
    // the full-array sort the survivor set was derived from).
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0f32;
    let mut keep = cand.len();
    for (i, &(_, prob)) in cand.iter().enumerate() {
        cumulative += prob;
        if cumulative > p {
            keep = i + 1;
            break;
        }
    }
    for &(idx, _) in &cand[keep..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}
```
<!-- /code -->

Three details are worth the attention. The comparison is `cumulative > p`, so a set whose mass lands exactly on
`p` keeps going for one more token - the boundary is inclusive in the wrong direction if you were hoping for
"at least p, no more". The softmax runs over the *survivors* of top-k, which the comment defends as bit-identical
to the full-array version and is what keeps this stage off the 151,936-element path. And the survivors are then
masked to `-inf` rather than replaced by probabilities, so the temperature stage's own softmax stays correct.

Now temperature, which is the stage most often described as something it does not do:

<!-- facts: sampler.smp-temperature -->
| Temperature | Top-1 mass | Top-5 mass | Tokens for 90% | Entropy | Argmax |
|---|---|---|---|---|---|
| 0.5 | 0.863 | 0.977 | 3 | 0.66 | `2121` |
| 0.8 | 0.585 | 0.861 | 7 | 1.66 | `2121` |
| 1.2 | 0.351 | 0.688 | 12 | 2.37 | `2121` |
| 2.0 | 0.185 | 0.499 | 17 | 2.82 | `2121` |
<!-- /facts -->

One real step, four temperatures, and the **argmax never moves**: token 2121 is the most likely token at 0.5 and
at 2.0. What temperature changes is how much of the mass the top token carries (0.86 down to 0.18) and how many
tokens you need to cover ninety percent (three, then seventeen). Temperature is a *measurement* choice about how
adventurous the draw should be; it cannot promote a token that the model did not rank first, and it cannot demote
one either. A sampler with `temperature = 0` is not "temperature turned down"; it is a different code path that
never looks at the distribution at all.

Which is the subject of the next lesson.

### Lesson 2 · Order matters: the pipeline is a sequence of decisions, and swapping two stages changes the output

The five stages are not a set, they are a sequence, and the engine writes the sequence down in one place:

<!-- code: src/sampler.rs::sample_with_config_grammar@88ffa5d -->
```rust
/// The complete pipeline, grammar-aware (F2, #47):
///
/// ```text
/// logit bias → penalties → DRY → [GRAMMAR MASK] → (greedy shortcut) → top-k →
/// typical → top-p → min-p → XTC → temperature | mirostat
/// ```
///
/// The order is llama.cpp's `common_sampler_init` chain, with three documented
/// decisions:
///
/// * the `temp < 1e-6` greedy shortcut keeps the exact pre-F3 position (after
///   the deterministic penalties, before the stochastic filters), so a greedy
///   request is bit-identical to today and `min_p`/`typical`/`xtc` cannot
///   perturb it;
/// * in mirostat mode the temperature is *ignored* — mirostat truncates the
///   distribution at `mu`, which subsumes temperature (llama.cpp sets
///   `temp = 1.0` when mirostat is on). The greedy shortcut still wins at
///   `temp == 0`.
/// * **the grammar mask sits between DRY and the greedy shortcut.** Everything
///   before it only *shifts* logits (bias, penalties, DRY add finite values), so
///   it cannot lift a masked `-inf` back to finite; everything after it only
///   *removes* candidates or reweights the survivors, so no forbidden token can
///   be selected. Being before the shortcut is what makes `--greedy` respect the
///   grammar. The mask consumes no RNG and writes nothing the other stages read,
///   so mirostat's `mu` and DRY's penalties are unchanged for the same token
///   sequence (the no-grammar path is pinned bitwise by
///   `test_default_pipeline_matches_the_pinned_pre_f2_sequence`).
///
/// The caller trims `prev_tokens` to its own window (the CLI keeps 64); DRY
/// additionally trims to `dry_penalty_last_n` internally. `grammar` is the
/// run's automaton state, one per run exactly like `mirostat`.
// ...
    let sampled = if cfg.temp < 1e-6 {
        sample_greedy(logits)
    } else {
        apply_top_k(logits, cfg.top_k);
        apply_typical(logits, cfg.typical_p);
        apply_top_p(logits, cfg.top_p);
        apply_min_p(logits, cfg.min_p);
        apply_xtc(logits, cfg.xtc_probability, cfg.xtc_threshold, rng);
        match cfg.mirostat {
            MirostatMode::Off => sample_temperature(logits, cfg.temp, rng),
            MirostatMode::V2 => sample_mirostat_v2(
                logits,
        // ...
    };
```
<!-- /code -->

The comment is doing the work a specification would do: it names the order, then defends the three positions that
are not obvious. The greedy shortcut comes **after** the deterministic penalties and **before** the stochastic
filters, so a `--temp 0` request is bit-identical to a run that never had the later stages. Mirostat ignores
temperature because it truncates at its own statistic. And the grammar mask sits between DRY and the shortcut, for
a reason worth reading twice: everything before it only *shifts* finite logits, so it cannot lift a masked `-inf`
back to finite; everything after it only *removes* candidates or reweights survivors, so a forbidden token cannot
come back. That single sentence is why `--greedy --json-schema` respects the schema.

To see that the order is real rather than decorative, put the same three stages in three orders on the step from
Lesson 1 and count what survives:

<!-- facts: sampler.smp-order -->
| Order of the same stages | Candidates kept | Top-1 mass | Argmax |
|---|---|---|---|
| top-k -> top-p -> temperature (engine) | 21 | 0.585 | `2121` |
| temperature -> top-k -> top-p | 12 | 0.737 | `2121` |
| top-p -> top-k -> temperature | 40 | 0.574 | `2121` |
| top-p alone (top-k off) | 69 | 0.570 | `2121` |
<!-- /facts -->

Four orders, four different sets - 12, 21, 40, 69 - from the same logits and the same configuration. The
temperature-first order keeps **fewer** candidates than the engine's order, not more, and that is not a bug: 0.8
is below 1, so scaling by `1/temp` sharpens the distribution before the cut is taken, and a sharper distribution
needs fewer tokens to cover 95 percent. Set `temp = 2.0` and the same order would keep more. The lesson is not
"one of these is right" - the engine's order is right *for the engine*, because it is the order the model was
tuned against. The lesson is that "top-p 0.95" is not a specification until you say what runs before it.

There is one more consequence of the order that surprises people, and it is measurable: under the greedy shortcut
**top-k and top-p never execute**. Not "have no effect" - the code path does not reach them, so a greedy run
cannot be made to respect a top-k, and a bug report that says "greedy ignores my top-k" is reporting the design.

### Lesson 3 · Seeded sampling and reproducibility, and what a seed does not fix

The draw is the only stage that consumes randomness, and it is one comparison per candidate:

<!-- code: src/sampler.rs::sample_temperature@88ffa5d -->
```rust
pub fn sample_temperature<R: Rng>(logits: &mut [f32], temp: f32, rng: &mut R) -> SampledToken {
    if temp < 1e-6 {
        return sample_greedy(logits);
    }

    let inv_temp = 1.0 / temp;
    for v in logits.iter_mut() {
        *v *= inv_temp;
    }

    // Softmax. Masked (-INF) logits map to exp(-INF)=0 and contribute nothing
    // to the running max or sum; skipping the exp() call for them avoids
    // ~n_vocab transcendental evaluations per token while staying bit-identical
    // (exp(-INF) == +0.0 exactly).
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f64;
    for v in logits.iter_mut() {
        if *v > f32::NEG_INFINITY {
            *v = (*v - max_val).exp();
        } else {
            *v = 0.0;
        }
        sum += *v as f64;
    }
    let inv_sum = (1.0 / sum) as f32;
    for v in logits.iter_mut() {
        *v *= inv_sum;
    }
        // ...
    let r: f32 = rng.gen();
    let mut cumulative = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        if v <= 0.0 {
            continue;
        }
        cumulative += v;
        if r <= cumulative {
            return SampledToken { token_id: i as u32 };
        }
    }
    SampledToken {
        token_id: (logits.len() - 1) as u32,
    }
}
```
<!-- /code -->

The last block is the whole sampler: walk the probabilities, accumulate, stop when the uniform draw is at or
below the running total. Three comment lines in the softmax above are elided for the same reason as Lesson 1's -
a `less-than-or-equal` sign - and they say only that the loop skips the masked entries, which is the `if v <= 0.0`
you can see ten lines later. Note the `if temp < 1e-6` at the top - that is the greedy shortcut, in the same function
whose name says temperature. Note also what the function *does* to its input: it multiplies the logits in place
and then overwrites them with probabilities, which is why every caller passes a scratch buffer.

What reproducibility looks like, measured - five runs, two of them identical by construction:

<!-- facts: sampler.smp-repro -->
| Run | Bytes | Same as the first run? | Shared prefix | Output |
|---|---|---|---|---|
| seed 42 | 126 | yes | 126 B | `As an AI language model, I don't have person...` |
| seed 42, second run | 126 | yes | 126 B | `As an AI language model, I don't have person...` |
| seed 43 | 105 | **no** | 0 B | `Sure, I'd be happy to help you with your fav...` |
| seed 44 | 137 | **no** | 6 B | `As an artificial intelligence, I don't have ...` |
| seed 42, top-k 5 | 126 | yes | 126 B | `As an AI language model, I don't have person...` |
<!-- /facts -->

Three claims, each with a number behind it:

- **The same seed twice gives byte-identical output.** 126 bytes, twice. That is a weak claim on its own;
  what makes it worth printing is that the two runs are separate processes with separate model loads.
- **A different seed diverges, and where it diverges is not predictable.** Seed 43 differed from the first byte.
  Seed 44 shared six bytes - `As an ` - and then left. A seed does not give you "the same story with different
  words"; it gives you a different walk through the same distribution, and the first step where two walks
  disagree can be the first step.
- **The same seed with a different top-k produced the same output.** Narrowing the candidate set from 40 to 5
  changed nothing, because the drawn token was inside both sets. A seed plus a config is the contract; a seed
  alone is not.

The strongest form of this claim is not a comparison between two runs of the same binary - it is a replay of one
run by a different implementation. The engine dumps the logits it computed at each step; this chapter's script
re-implements the pipeline and replays the token stream from those logits:

<!-- facts: sampler.smp-verify -->
| Run | Steps replayed | Python replay against the engine's text |
|---|---|---|
| greedy no penalty | 7 | byte-identical |
| greedy repeat 1 1 | 24 | byte-identical |
| greedy flat prompt | 24 | byte-identical |

The replay produced `The capital of France is Paris.` - the engine's own text, from the engine's own logits.
<!-- /facts -->

Three runs, three byte-identical replays, one of them twenty-four steps long with the repeat penalty on. That is
what "the sampler is deterministic given the logits" means as an executable statement - and it also tells you
where to look when two runs with the same seed differ: not at the sampler, at the logits. Different batch
shapes, a different backend, a different number of threads and the logits move in the last bits, which is enough
to change a knife-edge draw.

### Lesson 4 · Stop conditions: stop strings, byte-level matching, and where the run is cut

A run ends when the model emits an end-of-sequence token, or when the text it has produced so far ends with one
of the caller's stop strings. The second condition is a byte problem, not a token problem, and the engine handles
it as bytes:

<!-- code: src/sampler.rs::match_stop_suffix@88ffa5d -->
```rust
/// Byte-wise suffix match of `buf` against any stop string in `stops`.
///
/// Returns the byte index where the matched stop string starts — the
/// truncation point for the generated text (text up to that index is kept).
/// When several stop strings match, the earliest start wins (the longest stop
/// truncates the most). Empty stop strings are ignored. Multi-byte stop
/// strings split across tokens match because the comparison is byte-wise.
pub fn match_stop_suffix(buf: &[u8], stops: &[&[u8]]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for s in stops {
        if s.is_empty() || s.len() > buf.len() {
            continue;
        }
        if &buf[buf.len() - s.len()..] == *s {
            let start = buf.len() - s.len();
            best = Some(best.map_or(start, |b| b.min(start)));
        }
    }
    best
}
```
<!-- /code -->

Three properties, each of which someone gets wrong on the first attempt:

1. **Byte-wise, not token-wise.** The stop string is compared against the tail of the decoded buffer, so a stop
   string whose bytes are split across two tokens still matches. Measured: the stop string `pple\n` is **three
   tokens** in this vocabulary (`602, 273, 198`) and it still cuts the run. A matcher that compares whole tokens
   against the stop string finds nothing here.
2. **The earliest start wins.** If several stop strings match, the one that begins earliest truncates the most,
   and `best.map_or(start, |b| b.min(start))` is that rule in one line.
3. **The stop text is not part of the answer.** The return value is the *index* where the match begins, and the
   caller keeps everything before it:

<!-- facts: sampler.smp-stop -->
| Quantity | Value |
|---|---|
| Stop string | `pple\n` |
| Tokens it spans | 3 (`602, 273, 198`) |
| Full greedy output | `1. Apple\n2. Orange\n3. Banana` |
| Output with the stop | `1. Apple` |
| Stop text in the output | no |
<!-- /facts -->

The full greedy run was `1. Apple\n2. Orange\n3. Banana`; with `pple\n` as a stop string the output is
`1. Apple`, and the stop text is gone. That is the answer to the question this lesson opens with ("why does my
answer keep the stop text in it?"): because the truncation index was not applied, or was applied to the token
list after the text had already been assembled.

One design note that is easy to miss: this function matches against the *tail* of the buffer, which means the
caller has to feed it the accumulated bytes, not each token's bytes alone. Feed it one token at a time and a
three-token stop never matches.

### Lesson 5 · Grammar masks: one stage over the vocabulary, and a constraint that cannot be violated

The grammar is compiled into an automaton over the vocabulary, and at each decoding state the automaton produces
a **bit mask**: one bit per vocabulary entry, set if that token would keep the output on a path to a valid
string. The sampler applies it as one more stage, with the same `-inf` convention every other stage uses:

<!-- code: src/sampler.rs::apply_token_mask@88ffa5d -->
```rust
/// Mask the logits to the allowed tokens (`-inf` elsewhere, the same convention
/// `apply_top_k`/`apply_min_p` use, so every downstream survivor test keeps
/// working). Returns how many allowed tokens are still finite — 0 means the
/// grammar permits nothing and the caller must stop loudly.
fn apply_token_mask(logits: &mut [f32], mask: &[u64]) -> usize {
    let mut allowed = 0usize;
    for (i, v) in logits.iter_mut().enumerate() {
        let ok = mask
            .get(i / 64)
            .map_or(false, |w| w & (1u64 << (i % 64)) != 0);
        if ok {
            if *v > f32::NEG_INFINITY {
                allowed += 1;
            }
        } else {
            *v = f32::NEG_INFINITY;
        }
    }
    allowed
}
```
<!-- /code -->

The return value is the count of tokens still finite, and it exists for one reason: zero means the grammar permits
nothing, and the caller must stop loudly rather than draw from an empty set. A run that reaches such a state is
reporting a bug in the grammar or in the prompt, and the engine turns it into an error naming the automaton
state.

What that buys, measured on the same prompt across eight seeds:

<!-- facts: sampler.smp-grammar -->
| Condition | Runs | Output parses as the schema | Example |
|---|---|---|---|
| no schema | 8 | **0 of 8** | `To provide the weather in Paris with a temperature in Ce...` |
| schema | 8 | **8 of 8** | `{\n  "city": "Paris",\n  "temperature_c": 15\n}` |
| schema, greedy | 1 | **1 of 1** | `{\n  "city": "Paris",\n  "temperature_c": 10\n}` |
<!-- /facts -->

Zero of eight schema-less runs produced anything a JSON parser would accept - they produced prose, as models do.
Eight of eight constrained runs parsed, and the required keys were present in all eight. The greedy run under the
same schema also parsed, which is the point of the mask's position in the pipeline (Lesson 2): the guarantee does
not depend on the draw being lucky.

**What the guarantee does not cover** is worth one of the eight samples. Here is a run that satisfies the schema
and is still nonsense:

```
{"city": "Paris, " \t,"temperature_c": 13}
```

Valid JSON. Both required keys. A city name with a stray comma and a tab inside it, and a temperature that the
model invented. The grammar constrains the *shape* of the output, not its *content* - it can tell you that your
parser will not crash, and it cannot tell you that the answer is right. Every system that advertises
"structured output" is selling you the first thing and hoping you do not ask about the second.

A grammar is also a program with limits, and the engine bounds its own:

<!-- code: src/grammar.rs::MAX_STACK_DEPTH@88ffa5d -->
```rust
/// Bound on the automaton's call-stack depth. A stack that grows past it means a
/// rule that recurses without consuming input (left recursion), which a pushdown
/// parser cannot simulate — reported as such rather than looping.
const MAX_STACK_DEPTH: usize = 256;
/// Bound on the number of live nondeterministic stacks. Every in-scope grammar
/// stays far below it; exceeding it is an error, never a silent truncation.
const MAX_STACKS: usize = 64;
/// Bound on the number of visited stacks in one epsilon closure (a safety net
/// against an epsilon-heavy grammar, independent of `MAX_STACK_DEPTH`).
const MAX_CLOSURE_STEPS: usize = 65_536;
/// The per-run mask cache holds this many states, then clears. JSON revisits a
/// handful of states per member, so the hit rate matters; the bound keeps a long
/// generation from growing without limit.
const MAX_MASK_CACHE: usize = 64;
/// Upper bound on an explicit GBNF repetition (`{m,n}`) and on array-length
/// repetitions. An explicit cap keeps the compiled program finite and the
/// refusal names the limit.
const MAX_REPEAT: usize = 1024;
```
<!-- /code -->

There is also a parity trap in *how* the mask is built - the interesting part is not the loop over the
vocabulary, it is that a token's bytes can be a prefix of a valid string, so a token is allowed when it lies on
*some* completion path, not only when it is complete. That is why the mask is built from a walk of the automaton
over the token's byte string, and why the walk has a closure step budget.

## The pipeline

| Stage | Input | Effect on the distribution | Cost per step | Reproducible with a seed? |
|---|---|---|---|---|
| penalties | logits, the last 64 ids | shifts the logits of tokens already seen (multiplicative repeat penalty, additive frequency/presence), so they lose rank without being forbidden | one pass over the distinct window tokens | yes - no randomness |
| top-k | logits | keeps exactly *k* candidates, sets the rest to `-inf`; a count, so it never adapts | O(n) partial select on the vocabulary | yes |
| top-p | the survivors | keeps the smallest set whose probability reaches *p*; the size follows the distribution | sort over the survivors (<= k entries) | yes |
| temperature | the survivors | rescales logits before the final softmax; changes the weights, never the ranking | one pass over the survivors | no - it precedes the draw, and the draw consumes the generator |
| grammar mask | logits, the automaton state | sets every token the grammar forbids to `-inf`; the mask is cached per state | one pass over the vocabulary, amortised by the mask cache | yes - no randomness |
| draw | probabilities | picks one id by cumulative comparison against one uniform draw | one pass over the survivors | no - this *is* the randomness |

## Project

**Build step S13 · determinism you can defend.** Fix the seed, fix the prompt, and require the same ids twice.
Then add a penalty stage and watch a naive implementation break reproducibility because it consumed randomness in
a different order - the failure is not in the penalty, it is in *when* the generator was advanced.

Then reorder your stages deliberately. Apply temperature before top-p, and compare the surviving set against the
reference order on the same logits. If nothing changes on your test prompt, your test prompt is too easy: find one
where the top candidate carries less than half the probability before you decide the order does not matter.

For the grammar, test the negative: assert that a token outside the constraint never appears across a hundred
samples, and that the same prompt without the constraint fails to parse in most of them. The second half is the
one that proves the constraint is doing something.

## Deliverable

- A sampler with each stage independently testable and a seeded end-to-end reproduction test
- A stop-string matcher that handles multi-token stop sequences, with a fixture whose stop spans a token boundary
- One grammar-constrained run whose output provably cannot violate the schema, and one unconstrained run of the
  same prompt that does

## How you know you passed

- The same seed and prompt produce identical ids across two runs and across a restart.
- Swapping two stages changes the output, which shows the order is real and not decorative.
- No sampled output violates the grammar across your test batch, and the mask cost is measured.
- Your pipeline, fed the reference engine's logits, reproduces the reference engine's text byte for byte.

## Reading

- code: `src/sampler.rs`, `src/grammar.rs` at the book's [pinned baseline](../appendix/baseline.md)
- decisions: ADR-0018 (the grammar mask is one stage inside the single sampler pipeline)
- docs: `docs/GRAMMAR-DESIGN.md`, `docs/inference_e2e_walkthrough/12-sampler.md`
- [nucleus] Holtzman et al., *The Curious Case of Neural Text Degeneration* (2019) - arXiv:1904.09751

## Backfill

- [ ] Question 1: Why can the same text produce two different ids, and two runs diverge from one seed? (normalizer, pre-tokenizer, and what the file does not record)
- [ ] Question 2: Where does the algorithm end and the dialect begin? (one merge loop, one regex table)
- [ ] Question 3: What must never be silent? (a missing byte token, an unsupported dialect, a special token inferred from its name)
- [ ] Question 4: What does a round trip actually prove? (bytes survive, text may not)

<!-- facts: sampler.provenance -->
*`scripts/smp.py`; runs the engine's CLI on the model, keeps the logits it dumps, and replays the deterministic part of the sampler pipeline (penalties, top-k, top-p, temperature) in Python so the engine's own token stream can be reproduced from the engine's own logits; on **arm64, Darwin 27.2.0, python 3.15.2, numpy 2.5.4**; byte equality between the replay and the engine's text; the funnel, mass and entropy figures are computed from those logits, one step at a time, never averaged. Command: see [the provenance index](../appendix/provenance.md#sampler). Raw numbers: `facts/sampler.json`.*
<!-- /facts -->
