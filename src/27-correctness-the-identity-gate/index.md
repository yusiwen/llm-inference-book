# Chapter 27 · Correctness: The Identity Gate

> - **Compute tier**: ✅ any laptop (the discipline is the point; the GPU parts need a GPU)
> - **Status**: stub
> - **Build step S27**: A gate that decides, mechanically, whether a change to your engine changed its answers: bitwise where the contract demands it, tolerance-classed where it does not.
> - **Rule of this book**: Measurement is Chapter 28. This chapter is about being right; Chapter 28 is about being fast, and the gate is what stops the second from silently destroying the first.

## First, the gist

An inference engine fails in a way most software does not: a wrong answer is a fluent sentence. Testing therefore has to compare against a recorded truth at a precision the design chose, and it has to say which precision - bitwise for anything with a file-format or cache-layout contract, a named tolerance for anything with a reordered reduction. This chapter builds that gate: golden fixtures with content hashes, per-op and end-to-end comparisons, debug dumps and per-node traces, and the rule the project this book follows adopted - the identity gate is bitwise by default, and any tolerance is a named class a human wrote down, not a convenient epsilon.

## What you'll be able to do

- Build a golden-fixture suite with hashes so a fixture cannot drift silently.
- Decide per comparison whether it must be bitwise, and defend the choice in writing.
- Diagnose a mismatch: per-op dumps, per-node traces, and the bisection that finds the first divergent node.

## Lessons

- [ ] 1. What a wrong answer looks like here: fluent text, no exception
- [ ] 2. Golden fixtures and content hashes: the gate that cannot be talked out of
- [ ] 3. Bitwise by default: why a tolerance is a decision, not a default
- [ ] 4. Dumps and traces: per-layer hidden states, per-node timings, and the first divergent node (payoff: bisection beats guessing)
- [ ] 5. Test baselines as measurements too: what a suite records about itself

## Comparison classes

| Comparison | Must be bitwise? | Tolerance | Why | Where it is enforced |
|---|---|---|---|---|
| quantized file vs reference |  |  |  |  |
| per-op CPU vs oracle |  |  |  |  |
| backend vs CPU |  |  |  |  |
| end-to-end ids |  |  |  |  |
| session reload |  |  |  |  |

## Project

**Build step S27 · the gate you cannot bypass.** Build the gate before the second backend, not after. A gate written after two implementations exist tends to be calibrated to whatever they happen to disagree by.

Keep one fixture you can regenerate with a single command, and hash it. A fixture nobody can regenerate is a fossil.

## Deliverable

- A gate script that runs the suite and fails on any unclassified mismatch
- Fixtures with content hashes, regenerable by a documented command
- One mismatch diagnosed to a single node, written up

## How you know you passed

- A deliberately broken kernel is caught, and the failure names the fixture and the class.
- No comparison in the suite uses an epsilon that is not written down as a class.
- The fixture's hash is checked before the comparison runs, and the test would fail if it drifted.

## Reading

- code: `src/testfail.rs`, `docs/GATE-CONTRACT.md`, `src/graph/dot.rs`, `src/graph/json.rs`, `src/dump.rs`, `src/trace.rs`, `src/live.rs`, `src/optiming.rs` at the book's [pinned baseline](../appendix/baseline.md)
- decisions: ADR-0010 (the identity gate: bitwise by default), ADR-0020 (a quantized file is byte-identical), ADR-0023 (machine ledgers live beside their checkers)
- docs: `docs/TEST-BASELINES.md`, `docs/debug-dump.md`
- [goldberg] Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic* (ACM Computing Surveys, 1991) - no arXiv ID (journal paper)
- [ieee754] *IEEE 754-2019, Standard for Floating-Point Arithmetic* - no arXiv ID (standard)

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
