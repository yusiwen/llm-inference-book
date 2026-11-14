# Working in this repository

An mdBook. The book describes the `minfer` inference engine at a pinned commit and teaches a reader
to build the same thing. These rules are enforced by `scripts/verify.sh` unless stated otherwise.

## The gate

```bash
nix develop --command bash scripts/verify.sh
```

Each page opens with a four-item list in a blockquote (compute tier, status, build step, rule of the book);
`css/book.css` sets it smaller than body text, and it is targeted as `main blockquote:first-of-type` because
mdBook 0.5's content container is `#mdbook-content`, not `#content`.

Run it before reporting any work as done. It builds, rejects a build warning, scans characters,
checks that the site fetches nothing remotely, lints the figures, checks display-maths wrappers,
checks for a page mdBook silently stubbed, checks that the measurements page still carries its
commands, re-derives the book's own counts (`scripts/selfcounts.py`), checks every minfer citation
against the committed inventory, runs `scripts/probe.py` (skeleton sections, link resolution,
lesson-item pairing, the build ladder's coverage, the starter skeleton's shape, count drift), and
renders every `<!-- facts: ... -->` block against `facts/*.json` (`scripts/facts.py --check`).

## Numbers come from artifacts, not from the author

A sentence that quotes a type id, an offset or a byte count is quoting a file, and a wrong number typed
once is invisible to every checker in this repository - it disagrees with nothing, because nothing else
in the tree knows the answer. So numbers are generated:

| Piece | Role |
|---|---|
| `scripts/gguf-inspect.py --json` | reads a real GGUF and emits every provable fact as JSON |
| `scripts/facts.py --write FILE --name NAME` | writes `facts/NAME.json` (artifact path, sha256, header, metadata, all tensors, dtype mix) |
| `scripts/facts.py --render` | fills every `<!-- facts: NAME.block -->` block from the facts file |
| `scripts/facts.py --check` | gate: page blocks equal the render, every facts file is used, and the artifact's sha256 still matches |
| `scripts/gguf-crosscheck.py` | a *second* implementation (`gguf-py`) reads the same file; the verdict is committed in `reference/gguf-crosscheck.json` |

**A table row is not prose.** mdBook splits a table cell on every pipe, including pipes inside a code
span, so `||w||*||a||` in a table header silently destroys the table and the page still builds clean. Gate
step 12 now flags any table row containing `||`; spell the norm product as words.

**A mechanism claim is a fact too.** "Layer *i* got Q6_K because `use_more_bits` picks it" is the same
kind of assertion as a byte count, and it fails the same way: the file was produced by *some* version of the
reference tool, and that version's rule may not be the one at the baseline. Chapter 5 asserted the current
rule explains this file's twelve-layer split; the file's own tensor table showed a different twelve. Render
the split from the artifact (`layer-split`) and let the page say "different set" when it is.

Rules: never type a number that a block can render; when a number must stay in prose, it either comes
from a block or carries its command. A cross-check proves structure only - two parsers agreeing says
nothing about whether a kernel reads the bytes correctly.

## Tracking minfer

The engine this book describes is unfinished and changes daily, so it is pinned. The machinery:

| File | Role |
|---|---|
| `minfer.lock` | **the only place the baseline commit and the quoted counts live** |
| `reference/baseline-citations.json` | every citation, resolved at the baseline; regenerated only on a bump |
| `reference/intentional-broken.txt` | citations that must NOT resolve (the audit's findings) |
| `scripts/minfer-drift.py` | static comparator: paths, `path::symbol`, ADR numbers, counts |
| `scripts/contract-probes.sh` | behaviour probes; records under `runs/` for a human to diff |
| `src/appendix/baseline.md` | the ledger, the coupling table and the bump procedure |
| `.github/workflows/drift.yml` | daily advisory report in one rolling issue; never fails the build |

Rules that follow:

1. **Never write the commit hash in prose.** Say "at the book's pinned baseline" and link.
2. **A statement about minfer's current behaviour is dated** by the baseline page; a statement about
   what an engine must do is not dated at all.
3. **A new citation must be added to the inventory** - the gate fails otherwise, by design.
4. **A baseline bump follows the six steps on the baseline page**, reads the drift report, and commits
   as `docs(baseline): move to <tag or sha>`.

## Non-negotiables

1. **English only.** Anything outside ASCII and Latin-1 fails the gate apart from the allowlist in
   `scripts/verify.sh` (dashes, arrows, the tier marks, a few Greek notation letters).
2. **The chapter skeleton is fixed** and its order is fixed: gist, abilities, lessons, one
   chapter-specific ledger, Project, Deliverable, How you know you passed, Reading, Backfill.
   Reference pages (appendices) carry a status block and `## Reading` instead.
3. **A checklist item and its `### Lesson N` heading open with the same clause** once the prose
   exists. `scripts/probe.py` compares them.
4. **Status vocabulary is two words**: `stub` (skeleton only), `draft` (prose exists, a claim is
   known missing or wrong), `written` (prose complete; the qualifier says what the evidence covers).
5. **Claims are labelled**: measured / arithmetic on stated assumptions / reported / textbook fact /
   my judgement. Never upgrade a reported number into something this book measured.
6. **Numbers live in `src/appendix/measurements.md`** with the command and the commit. A count
   anywhere else must be derivable and must agree with the tree.
7. **Code citations are path + symbol + commit**, never a bare line number, because line numbers
   drift (see `src/appendix/cuda-docs-audit.md` for what that drift cost the reference project).
8. **Do not copy the reference project's documentation.** Link it. Mechanics are re-derived for a
   reader who is building; depth stays in the project.

## Writing a chapter

Two questions decide what a chapter owes before a word of prose: **does it quote an artifact**, and **does it
make a claim about the world that a second implementation could contradict**? The answers pick the evidence,
and the gate enforces the evidence.

1. **Skeleton first.** `src/NN-english-slug/index.md` from the stub: five lessons, the build step, the four
   sections that `scripts/probe.py` requires.
2. **An artifact means the numbers are generated, not typed.** Add `--json`-style facts from an instrument,
   commit `facts/<name>.json`, and put every table of counts, offsets or type ids behind
   `<!-- facts: NAME.block -->` markers that `scripts/facts.py --render` fills. Gate step 12 fails if a page
   and its facts disagree, if a facts file is unused, or if the artifact's sha256 no longer matches.
   Never type a number a block can render; a number that stays in prose either comes from a block or carries
   its command.
3. **A claim about a format or a behaviour needs a witness.** Use a second implementation
   (`scripts/gguf-crosscheck.py` against `gguf-py`), commit the verdict under `reference/`, and reconcile
   every difference in writing rather than normalising it away. Agreement on structure is not agreement on
   semantics - say which one you got.
4. **Figures ship with the chapter**: draw its P0 figures from `FIGURES.md` in the same sitting, include them
   where the prose needs them, mark them drawn.
5. **Cite at least one source by key** in `## Reading`, then `python3 scripts/references.py` (the gate fails
   on a stale used-in column).
6. **Re-run the gate**; `probe.py` names what the page is still missing. A new gate needs a negative test
   (break the thing, confirm exit 1) before it is trusted.
7. **Ladder and counts**: add the chapter's row to `src/appendix/build-ladder.md`, and let
   `scripts/selfcounts.py` tell you which numbers in `src/index.md` and `src/appendix/measurements.md` moved.
   Both go in the same commit.

**Chapters with no artifact** (the graph, the seam, the backend designs, Chapters 14-19 and 21-26) have a
different obligation, and it is not a weaker one: every claim is labelled `my judgement` and tied to an ADR
via `path::symbol`, or to a cited source, or to a run recorded in `runs/`. What they must not do is borrow
the authority of a measured number they do not have.

## Formulas, figures and monospace

Three carriers, and the choice between them is a rule rather than a taste. A block that looks like a
picture, in characters, is the one defect class the build cannot see: the page compiles, the site
deploys, and the reader is handed an ASCII sketch that should have been drawn.

- **MathJax for anything that looks like typeset mathematics.** `$$` for a display formula, `$` inline,
  and **every display block wrapped in `<div class="math-display">`** (gate step 5 counts the wrappers
  against the `$$` markers). No formula numbering: `tags: "none"` in `mathjax/config.js`, because TeX
  numbering cannot be referenced across mdBook pages without breaking.
- **SVG for anything that looks like a sketch** - a bit layout, a byte plan, a pipeline, a dataflow.
  Never characters in a monospace block; see `FIGURES.md` for the drawing conventions.
- **Monospace (`text` / `rust` / `bash`) for exactly three things**: program output and terminal
  sessions, source code, and a numeric derivation whose point is that the reader can check the digits.
  A single-row derivation with an `=` chain belongs in math, with `\text{}` for the units; several
  aligned rows of numbers belong in a table, so the columns are structure rather than spaces.
- Inline code spans stay for identifiers (`f32`, `rope.freq_base`, `vec_soft_max_f32`), not for
  mathematics: `q·k/sqrt(hd)` in a code span renders as a line of symbols the reader has to parse,
  where `$q\cdot k/\sqrt{h_d}$` renders as a product over a root.
- **A formula that MathJax cannot parse does not fail the build.** Gate step 13 (`scripts/texlint.py`)
  catches unbalanced braces, `\left` without `\right`, unclosed environments, unknown control
  sequences and stray `$`; the rendered page still has to be looked at (below).
- **A table that the parser rejects does not fail the build either.** GFM refuses a table whose header
  row has a different cell count than its delimiter row, and ships the whole block as a paragraph of
  pipe characters - which is how a header cell reading `Largest |q|` reached a reader with every check
  green. Gate step 12 counts the cells now; `||` inside a row is the same defect with two pipes.
- **A sketch is counted, not trusted.** `scripts/sketch-probe.py` (gate step 14) lists the monospace
  blocks that are pictures or hand-aligned ledgers, and `--allow` is a ratchet: the backlog may only go
  down. Four sketches are waiting to be drawn as SVG and four ledgers to become facts tables; the
  allowance in `scripts/verify.sh` is the number, and it is lowered in the commit that converts one.

Math is typeset client-side, so `mdbook build` proves nothing about it. **Before publishing a page that
contains formulas, load the built page and look at it** - the same obligation as the figures, and the
reason the `|q|` table in Chapter 6 shipped broken while every gate stayed green.

## Do not

- Do not hand-edit `book/` (generated) or add a remote asset (the site must work offline).
- Do not draw a diagram in characters. A sketch of bytes, fields or dataflow goes into an SVG, a formula
  into a maths block, and a multi-row ledger into a facts table; `scripts/sketch-probe.py` counts what is
  left, and its allowance only goes down.
- Do not add a figure outside `FIGURES.md` without adding its row there first, and do not introduce
  a cite key without a row in `src/appendix/references.md`.
- Do not cite an arXiv ID from memory, and do not mark an unchecked one "to verify": resolve it or
  drop the source.
- Do not weaken a gate to make a page pass; fix the page, or change the rule here and in
  `src/appendix/conventions.md` together.
- Do not touch the `minfer` tree from this repository's work.
- Do not type a number into a page that a facts block could render, and do not put opinion inside a
  generated block - numbers and judgement live in different places, or the numbers start drifting again.
