# The minfer Baseline

> - **Status**: written - the mechanism is in place and exercised; two baselines are recorded so far.
> - **Rule of this book**: a sentence about what minfer *currently* does is dated by the baseline below. A sentence about what an engine *must* do is not.

minfer is not finished, and pretending otherwise would make this book wrong within a month. It also moves fast: **1364 commits in the last 90 days** and 35 in the last week, measured at the baseline with

```bash
git -C <minfer> rev-list --count --since="90 days ago" HEAD   # 1364
ls docs/adr/[0-9]*.md | wc -l                                 # 83, all added in that window
```

Of the **139 distinct minfer paths this book cites** (143 citations), 103 - 74 per cent - were touched in the last 30 days, and 138 (99 per cent) in the last 90 (`git -C <minfer> log --since="30 days ago" --oneline -- <path>` per citation). Chasing `master` would mean rewriting citations every week and never finishing a chapter. So the book pins a baseline and moves it deliberately.

## The pinned baseline

| Field | Value |
|---|---|
| Commit | `e8445913064e2871ee3edb1582cb8af49e1c8cbd` |
| Describe | `v0.0.9-134-ge844591` |
| Date | 2026-11-16 |
| Why this one | the KV prefix-sharing contract stopped being a comment and became an invariant (a live share pins its donor's run, a compaction cannot place a run over one, and the store refuses to write a lent prefix); the six measurement tests the KV chapter is built on landed in the same window |
| Recorded in | [`minfer.lock`](https://github.com/yusiwen/llm-inference-book/blob/main/minfer.lock) - the only file that holds the hash |

```bash
git clone https://github.com/yusiwen/minfer && cd minfer
git checkout e8445913064e2871ee3edb1582cb8af49e1c8cbd
cargo build --release                 # CPU + Metal on macOS
cargo build --release --features cuda # on an NVIDIA machine
```

No other page prints this hash. A chapter's `## Reading` section says "at the book's pinned baseline" and links here, so a bump is one edit rather than thirty - and a citation can never disagree with the cover page about which tree it came from.

## What is tracked, and what is not

| Drift | Example | Covered by | Automatic? |
|---|---|---|---|
| **Address** | `src/graph/alloc.rs` split into a directory; `MmapState` renamed | `scripts/minfer-drift.py` | yes |
| **Number** | ADR count 82 to 95; kernels 108 to 121; LOC | same tool, against `minfer.lock` | yes |
| **Behaviour** | KV format gate moved from process-wide to per-engine; template fallback replaced by a refusal | `scripts/contract-probes.sh` + a human reading the diff | no - recorded, then judged |

The tool checks every cited path with `git cat-file -e <ref>:<path>` and every `path::symbol` citation with `git grep -w <symbol> <ref> -- <path>`. A citation that must *not* resolve - two findings of [the CUDA audit](cuda-docs-audit.md) are exactly that - is listed in `reference/intentional-broken.txt`, so "this file does not exist" can be a claim the book defends instead of an error it hides.

The behaviour probes are a handful of commands whose output the book quotes: a graph dump's node count, a seeded generation's token ids, a server's metric keys. Each run is recorded under `runs/`, and a difference between two records is a question for a person, never an automatic edit.

## Where the book is coupled to minfer

Not every chapter goes stale at the same rate, and the coupling is what decides how much reading a bump costs.

| Chapters | Coupling | On a bump |
|---|---|---|
| 1, 8, 9, 10, 11, 12, 13, 19 | **concept** | read nothing: the mechanisms (roofline, matmul shapes, tokenization, sampling, architecture families) are stable |
| 2, 3, 4, 5, 6, 7 | **format** | re-check the cited paths and the block layouts; the file formats change slowly, the kernel files around them weekly |
| 14, 15, 16, 17, 18, 20 | **mirrors-code** | re-read the drift findings for these chapters; `src/graph/` is the most active directory in the project |
| 21, 22, 23, 24, 25, 26 | **mirrors-code (device)** | expect renames: `src/cuda/` and `src/metal/` move fastest, and the campaign record grows by a step at a time |
| 27, 28, 29 | **measured** | re-run the probes; the gates and the bench harness are what the chapters claim |

## Bumping the baseline

Six steps, in this order, and the whole thing is one commit.

1. `git -C <minfer> fetch --tags` and pick a ref: a tag when one is close, the tip of `master` otherwise. Not the tip by default - a tag is a fact, the tip is a Tuesday.
2. `python3 scripts/minfer-drift.py --repo <minfer> --ref <new-sha> --report /tmp/drift.md` and read the findings in the order the coupling table above ranks them.
3. Fix what the report calls address drift, chapter by chapter. Where a symbol moved, the fix is a new citation, not a tolerance for a dead one.
4. Re-read the `mirrors-code` chapters' passages the report names. This is the only step a tool cannot do, and skipping it is how a book becomes confidently wrong.
5. Update `minfer.lock` (commit, describe, date, counts) and the counts on [Measurements](measurements.md).
6. `python3 scripts/minfer-drift.py --repo <minfer> --ref <new-sha> --write-inventory reference/baseline-citations.json`, append a row to the ledger below, run `bash scripts/verify.sh`, commit as `docs(baseline): move to <tag or sha>`.

## How you are reminded

The daily advisory job prints the baseline's **age in days**, how many commits `master` is ahead, and the latest tag, at the top of the rolling issue - then adds a line when the age passes 30 days, which is the monthly fallback this book promised itself. Nothing acts on that reminder automatically: an age is a prompt to run the six steps above, not evidence that anything is wrong. A baseline can be a month old and still correct, and one that is a day old can already be behind on a renamed kernel.

Locally the same three numbers come out of the tool, so you never have to open GitHub to ask "how far behind am I?":

```bash
python3 scripts/minfer-drift.py --repo <minfer> --ref <baseline>
# baseline 88ffa5d dated 2026-11-11: 1 day(s) old, master is 5 commit(s) ahead, latest tag v0.0.9
```

## Ledger (append-only)

One row per move. Never edit an old row: the ledger is a log, and a log that gets corrected stops being one.

| # | Baseline | Describe | Date | Why | What the book had to change |
|---|---|---|---|---|---|
| 1 | `88ffa5d` | `v0.0.9-118` | 2026-11-11 | initial baseline: the tree the outline, the reference index and the CUDA audit were written against | - |
| 2 | `e844591` | `v0.0.9-134` | 2026-11-16 | prefix sharing stopped being a comment and became an enforced invariant (a live share pins its donor's run, a compaction cannot place a run over one, the store refuses to write a lent prefix), and the six `measure:` tests the KV chapter is built on landed | no citation moved (the drift report found 0); the counts above and on [Measurements](measurements.md) were re-derived |

## The rule this page exists for

- **A current-state claim is dated.** "minfer indexes one session file per engine" is a claim about the baseline; it belongs next to a link to this page or is not written at all.
- **A capability claim is never timeless.** "minfer cannot run this architecture" goes in [Limits](limits.md) as a pointer to the project's own inventory, with a date, never as a property of the world.
- **A count is re-derived, not remembered.** The numbers in `minfer.lock` are the only ones the book may quote, and the tool is what re-derives them.

## Reading

- [`minfer.lock`](https://github.com/yusiwen/llm-inference-book/blob/main/minfer.lock) - the record itself
- [Measurements](measurements.md) - the counts, with the command behind each
- [The CUDA audit](cuda-docs-audit.md) - what a drift checklist looks like when it is done by hand
- [Limits](limits.md) - why the book points at inventories instead of copying them
