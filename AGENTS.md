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

## Adding a chapter

1. `src/NN-english-slug/index.md`, following an existing stub exactly.
2. Add the entry to `src/SUMMARY.md` under its part.
3. **Figures ship with the chapter**: draw the chapter's P0 figures from `FIGURES.md` in the same
   sitting, include them where the prose needs them, and mark them drawn in `FIGURES.md`.
4. **Cite at least one source by key** in `## Reading`, and regenerate the reference index with
   `python3 scripts/references.py` (the gate fails on a stale "used in" column).
5. Re-run the gate; `probe.py` will tell you what the page is missing.
6. Add the chapter to `src/appendix/build-ladder.md` (step, produces, needs, what skipping costs) -
   the probe fails when a chapter and the ladder disagree.
7. Update the counts on `src/index.md` and `src/appendix/measurements.md` in the same commit;
   `scripts/selfcounts.py` tells you which ones moved.

## Do not

- Do not hand-edit `book/` (generated) or add a remote asset (the site must work offline).
- Do not add a figure outside `FIGURES.md` without adding its row there first, and do not introduce
  a cite key without a row in `src/appendix/references.md`.
- Do not cite an arXiv ID from memory, and do not mark an unchecked one "to verify": resolve it or
  drop the source.
- Do not weaken a gate to make a page pass; fix the page, or change the rule here and in
  `src/appendix/conventions.md` together.
- Do not touch the `minfer` tree from this repository's work.
