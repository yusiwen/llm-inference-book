# Chapter 26 · CUDA V: The Optimization Campaign

> - **Compute tier**: ⚠️ needs an NVIDIA GPU for the measurements; the method is readable anywhere
> - **Status**: written - the campaign's own record set is counted by `scripts/campaign.py` (113 step records, no GPU required), and the two tables below are that audit rather than a summary of its wins
> - **Build step S26**: A campaign of your own: a cost model, one lever per experiment, a gate that decides land-or-revert, and a written record you can defend a year later.
> - **Rule of this book**: This chapter is the method and one audited campaign, not a catalogue of every kernel. The techniques themselves appear where they are needed, in Chapters 7, 24 and 25; the measurement discipline they depend on is Chapter 28.

## First, the gist

A GPU optimization campaign is not a list of tricks, it is a process that survives being wrong. You build a cost model from counts, identify the wall a kernel is waiting on, change exactly one thing, and let a gate decide whether the change lands - where the gate includes correctness, bitwise where the contract demands it, and a throughput delta larger than the measurement's own noise floor.

Most levers fail. That sentence is usually a consolation offered by someone whose lever failed; here it is a count, taken from the engine this book follows:

<!-- facts: campaign.campaign-verdicts -->
| Verdict in the record | Count | Share of 113 |
|---|---|---|
| `LANDED` | 43 | 38.1% |
| `REVERTED` | 15 | 13.3% |
| `REVERTED / MEAS-ONLY` | 1 | 0.9% |
| `MEAS-ONLY` | 3 | 2.7% |
| `CLOSED` | 2 | 1.8% |
| no verdict in the title | 49 | 43.4% |
| **total** | **113** | |

**38.1 per cent landed.** That is the number a reader should take from a campaign record, and it is the one an article about optimisation never prints: 70 of 113 levers did not survive, 16 of them because they were measured and reverted on purpose, and a further 49 carry no verdict in their title at all - the record set is itself an artifact with a defect rate.
<!-- /facts -->

And the second table is the one that makes this chapter worth writing, because it audits the *method* the same way the method audits the kernels:

<!-- facts: campaign.campaign-discipline -->
| What the record set shows | Count | Share |
|---|---|---|
| states a decision bar **before** the measurement | **20** | 17.7% |
| no pre-registered bar | 93 | 82.3% |
| carries a verification section | 102 | 90.3% |
| opens with a `Result` block (the claim, before the prose) | 101 | 89.4% |
| median length | 230 lines | |

This is the chapter's uncomfortable table, and it is here because the campaign states its own house rule - *a bar named before measuring* - and a rule that is stated is not a rule that is followed. **17.7 per cent** of the records can show the bar. Verification is in far better shape (90 per cent), which is the expected asymmetry: checking that a kernel is still correct is mechanical, and deciding in advance how much faster it has to be is a judgement that is easy to make after the fact and impossible to distinguish from an excuse.
<!-- /facts -->

**Seventeen per cent.** The campaign states its own house rule - *a bar named before measuring* - and only a sixth of its records can show one. Correctness, by contrast, is checked in nine records out of ten, because checking correctness is mechanical and deciding in advance how much faster a kernel has to be is a judgement that is trivially made afterwards and indistinguishable from an excuse.

That asymmetry is the chapter. A campaign's record has two defect rates: the rate at which levers fail, and the rate at which the records of them fail to be evidence.

<figure class="book-figure">{{#include ../figures/26-gate-chain.svg}}</figure>

*Figure 26.1 - the gate chain a lever has to pass: hypothesis, predicted counts, one change, correctness, identity, then a delta above the noise floor - with the two exits that make a record worth keeping.*

<figure class="book-figure">{{#include ../figures/26-campaign-verdicts.svg}}</figure>

*Figure 26.2 - the campaign's record by verdict: the bar of landed levers is shorter than the remainder, and the failures are the documentation rather than an embarrassment.*

## What you'll be able to do

- Build a wall-clock cost model for a workload and name the lever that would move it.
- Design an experiment with one variable, a stated noise floor, and a land/revert decision rule fixed in advance.
- Write a step record that a stranger could reproduce, including the reverted ones.

## Lessons

- [ ] 1. Attribute before you optimize: decomposing the wall clock into kernels and waits
- [ ] 2. One lever per experiment, and a decision rule written before the measurement
- [ ] 3. The gate chain: correctness, then identity, then a delta above the noise floor
- [ ] 4. Why most levers fail: the four failure shapes (wall-inert, parity-breaking, occupancy-negative, artifact)
- [ ] 5. Device tiers and formula gates: what generalizes to a GPU you do not own

### Lesson 1 · Attribute before you optimize

A decode step of a small model spends its time in a handful of places, and the first job is to say which one *without changing anything*. The decomposition that matters is not "which kernel is slowest" but "which resource is each kernel waiting on", because the two answers point at different levers: a kernel waiting on bytes needs fewer bytes or better access; a kernel waiting on latency needs more parallelism; a kernel waiting on a serial dependency needs its algorithm restructured.

Chapter 22 gives you the two instruments for that classification without a profiler at all: **predict the time from the byte count** and compare. A kernel that lands on `bytes / streaming_rate` is bandwidth-bound and has nothing left to win from better arithmetic; a kernel that misses it is waiting on something else, and the size of the miss is the size of the opportunity. Chapter 24 then shows the same instrument at the next level down, where the comparison is against a tile arithmetic rather than a byte count.

<!-- no-code: This chapter's evidence is the campaign's record set, counted by scripts/campaign.py, not a kernel. The instruments that make the decomposition are quoted where they are built - the launch sheet in Chapter 22, the ladder in Chapter 24, the attention paths in Chapter 25. -->

What the engine's records add is the accounting *between* kernels. A thousand launches of a 3-microsecond kernel spend more time being launched than running, which is why Chapter 25 measured 2.44 microseconds of host time per launch and 0.79 when the same hundred kernels are replayed as a graph. **A campaign that only profiles kernels will never see that**, because the cost is in the gaps, and the gaps are where a decode step at high token rates actually goes.

### Lesson 2 · One lever per experiment, and a bar written first

The rule is one change per experiment, and it is not about discipline for its own sake - it is about what you can conclude afterwards. Two changes that both help leave you unable to say which helped and by how much; two changes where one helps and one hurts leave you unable to say anything at all.

The second rule is harder and it is the one the campaign only follows a sixth of the time: **state the bar before you measure**. Not "faster", a number, chosen while you still do not know the answer, because a threshold chosen after the measurement is not a gate - it is a rationalization with a number on it.

The record set contains the counter-example and it is instructive. One ticket (minfer#633) writes the rule into its own body - "a bar named before measuring ... a smaller bar is fine but must be stated" - and then names one: parity with f16, which would need the kernel at f16's own duration, a 2.1x cut. The two step records that later closed that ticket both open with the result and neither carries the bar. The levers landed; the *evidence that they cleared a previously stated threshold* did not survive.

That is worth sitting with, because it is the failure mode of method rather than of code: nothing was wrong, the numbers were real, the identity gate was byte-exact, and a reader a year later still cannot tell whether the bar was cleared or moved.

<!-- no-code: the evidence in this lesson is the absence of a field in a prose record, and the book's code gate verifies source symbols rather than markdown. What can be counted is counted - `scripts/campaign.py` reports how many of the 113 records state a bar before their measurement, and that number is the lesson. -->

### Lesson 3 · The gate chain

A lever is not a change; it is a change plus the evidence that it is allowed to stay. The chain has four links and each one can end the experiment:

| Link | The question | If it fails |
|---|---|---|
| **correctness** | does the output equal the reference? | revert; this is not a trade-off |
| **identity** | is it byte-equal where the contract demands bitwise? | revert, or state the new tolerance and why |
| **delta above noise** | is the change larger than the measurement's own spread? | inconclusive - re-measure, do not land |
| **cost** | what did it make worse? | land only if the trade is named |

The identity link is the one a GPU campaign gets wrong most often, because floating-point equality is usually a fine requirement on a CPU and a trap on a device: a matmul that reorders its summation is bitwise different while being numerically fine, and a greedy decode then amplifies one flipped argmax into a completely different sentence. Chapter 21's first gate was therefore *sha256 equality of the generated tokens*, not a tolerance, and everything downstream inherited that.

<!-- no-code: this lesson is the chain itself, and each link is already demonstrated where it was built - the sha256 identity gate in Chapter 21, the tolerance column in Chapter 24, the noise floor in Chapter 22's interleaved rounds. Quoting a fourth copy of the same excerpts here would make the chapter longer without making the chain any clearer. -->

The noise floor is the link that is easiest to skip and hardest to apologize for. A three-run median with a two-per-cent spread cannot resolve a one-per-cent improvement, and the honest outcome of such an experiment is "inconclusive", which is a verdict a campaign should be willing to record. The engine's records do exactly that: some levers are `MEAS-ONLY` because the measurement answered the question about the mechanism without producing a change to keep.

### Lesson 4 · Why most levers fail

Counted from the same record set, failures have four shapes, and naming them is what stops the next engineer from rediscovering the same one:

| Shape | What it looks like | Why it happens |
|---|---|---|
| **wall-inert** | the kernel got faster; the token rate did not move | the kernel was not on the critical path - Amdahl, but measured rather than assumed |
| **parity-breaking** | faster, and a different token comes out | the change was numerically real, not just slower arithmetic |
| **occupancy-negative** | slower after adding parallelism | smaller blocks or more registers cut resident warps, and the added work was not the bottleneck |
| **artifact** | the speedup belongs to the instrument | the comparison measured something else - a warm cache, a different shape, a recompiled reference |

The fourth is the one this book has spent the most chapters on, because it is the only one that *looks* like a win. Chapter 24's tiling experiment is a worked example: a 16x predicted ceiling and a 1.37x measurement, where the honest reading is that the instrument does not isolate the effect at all. Chapter 22 has a second: an access-pattern claim carried by one implementation for weeks, until a second implementation showed the share of stream was not even a constant across sizes.

<!-- no-code: the four shapes are counted from the record set, not demonstrated with a kernel - the artifact's verdict table is what this lesson rests on. Each shape has a worked example earlier in the book: wall-inert in Chapter 7's dispatch, parity-breaking in Chapter 21's identity gate, occupancy-negative in Chapter 22's launch sheet, artifact in Chapter 24's tiling. -->

**A campaign whose failures are not written down is a campaign that will re-run them.** The 15 reverted records in the count are the most valuable 13 per cent of the corpus, and they are the only part of it that gets cheaper to write and more expensive to omit.

### Lesson 5 · What generalizes to a GPU you do not own

Two things from a campaign transfer to hardware you have never touched, and one does not.

**Transfers: the byte arithmetic.** A kernel's cost from its byte count, a tile's reuse factor, a model's bandwidth floor, the number of launches per step - all of these are counts, and counts do not care whose device they run on. Chapter 22 wrote the prediction from `bytes / streaming_rate` on one machine and Chapter 24 reused the method on the same one; the arithmetic would have been identical on a laptop.

**Transfers: the gate discipline.** Identity before tolerance, a bar before a measurement, a verdict per record. None of that is device-specific, and all of it is what makes a number usable by someone who was not there.

**Does not transfer: the ratios.** A measured speedup is a property of a kernel *and* a device *and* a shape *and* a context length. The campaign's own headline numbers make this concrete: a packed Q8_0 cache's decode penalty went 1.239x, then 1.0139x, then 0.9705x across three levers on one device, while the same cache's prefill penalty never moved from 1.224x - and on a second backend the same idea started at 1.210x and settled at 1.041x. **A table of measured tiers beats a hard-coded guess**, which is why the engine keeps its device constants beside the code that uses them rather than in a comment about a GPU that has since been replaced.

<!-- no-code: the ratios this lesson rejects are quoted in Chapter 10's Q8_0 table with their commit, device and command, which is exactly the "publish the conditions" rule being argued for; a second excerpt of the same numbers here would be a copy, and a copy is the failure mode the lesson is about. -->

The practical form: when you publish a number, publish the shape, the device, the context length and the command beside it, or expect it to be quoted as if it were universal. Chapter 22's counterfactual and Chapter 24's two-size tiling experiment are both that lesson, learned the hard way inside this book.

## A step record

Every record in the corpus has this shape, and a stranger should be able to act on it without you in the room:

| Field | What it must contain | Why a reader needs it |
|---|---|---|
| hypothesis | the mechanism you believe is the bottleneck, in one sentence | so a later refutation can name what was wrong, not just that the number moved |
| counts predicted | the arithmetic, before the measurement | so a surprise is visible as a surprise |
| bar | the decision threshold, stated before the measurement | so landing is a decision and not a description |
| one lever | the single change, named | so the delta is attributable |
| gate result | correctness and identity, with the test that ran | so the next person can re-run it |
| wall-clock delta | the token rate or latency, with the shape and command | so it can be compared to something else |
| verdict | landed, reverted, or measurement-only, with the reason | so the failure is documentation instead of folklore |

## Project

**Build step S26 · a campaign you can audit.** Pick one kernel from Chapter 24 that is off its roofline and run three experiments: one that lands, one that is wall-inert, and one that breaks parity. Writing up the failures is the exercise; anyone can publish a win.

Fix the decision rule in the record before the first measurement. A rule chosen afterwards is not a gate, it is a rationalization - and the campaign this chapter audits can show you exactly what that costs, at a compliance rate of one record in six.

## Deliverable

- Three step records in the shape above: one landed, one wall-inert, one parity-breaking
- A cost model for one kernel that predicts its time from counts before you measure it
- A noise floor for your own harness, stated as a number, so "inconclusive" is available as a verdict

## How you know you passed

- Every record states a bar before its measurement, and at least one experiment uses the verdict *inconclusive*.
- Your wall-inert experiment names the resource the kernel was *not* waiting on, and how you know.
- Your parity-breaking experiment says which token changed and why the tolerance you were using did not catch it.

<!-- facts: campaign.provenance -->
*`docs/cuda_optimization_steps/ in minfer (parsed by scripts/campaign.py; no GPU needed)`; parses every step record in the engine's campaign directory for its verdict, whether it records a decision bar before the measurement, and whether it carries a verification section; the count of step records by verdict, the share that landed, reverted or carry no verdict, the number that record a pre-registered bar, and the median length of a record. Raw numbers: `facts/campaign.json`.*
<!-- /facts -->

## Reading

- code: `docs/cuda_optimization_steps/` in minfer - 113 step records, one per lever, with `STYLE.md` as the writing contract - at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/CUDA_OPTIMIZATION.md` (current state) and `docs/cuda_optimization_steps/77-verification-methodology.md` (the methodology record the others follow)
- [roofline] Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures* (2009), DOI 10.1145/1498765.1498785 - the arithmetic-intensity model Chapter 22 and this chapter both use
- [amdahl] Amdahl, *Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities* (1967), DOI 10.1145/1465482.1465560 - why a wall-inert lever is the most common result in any campaign

## Backfill

- [x] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - Lesson 1's decomposition, plus the gaps between kernels that a profiler cannot attribute to any of them.
- [x] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - the campaign's own longest argument: a packed cache that got its decode penalty from 1.239x to 0.9705x over three levers while its prefill penalty sat still, which is a trade with a shape, not a win.
- [x] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - Lesson 5: counts transfer across devices, ratios do not, and the seam is the place where a ratio becomes a policy.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - Lesson 3's four links, and the audit that found the campaign's own records follow the identity rule nine times in ten and the bar rule once in six.
