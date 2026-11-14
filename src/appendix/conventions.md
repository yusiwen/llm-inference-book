# Writing and Formatting Rules

> - **Status**: written - these are the rules every chapter of this book is checked against.
> - **Rule of this book**: the rules live here, not in the author's head, because a consistent shape is what lets a reader pick up any chapter on its own.

Two rules are non-negotiable: **everything is written in English**, and **everything is written for someone who is building, not browsing**.

## Chapter skeleton (fixed)

```text
# Chapter N · Title
> - Compute tier
> - Status
> - Build step SN
> - Rule of this book          (a blockquote holding a four-item list)
## First, the gist
## What you'll be able to do
## Lessons          (- [ ] checklist; one ### Lesson N heading per item once written)
## <one chapter-specific ledger or map section>
## Project          (the build step: name, problem, what to keep empty)
## Deliverable
## How you know you passed
## Reading
## Backfill         (the four book questions)
```

**The opening block is a four-item list, not a paragraph.** Each item is a bold label and its value, and the list sits inside a blockquote so it reads as apparatus rather than prose:

```text
> - **Compute tier**: ✅ any laptop
> - **Status**: written - what the evidence covers, and what it does not
> - **Build step SN**: what exists in the reader's tree when the page is done
> - **Rule of this book**: one sentence on what this page is and is not
```

The list is what makes it scannable; `css/book.css` sets it a little smaller than the body text, so the page's first screen belongs to the title and the gist. An earlier version put all four facts on one line separated by `|`, which read as a single dense sentence - four facts a reader parses once, formatted so that parsing them is work.

**Section names are fixed and their order is fixed.** A chapter that needs a second track labels it inside `## Lessons` or gives it a section *after* the fixed ones. A renamed standard section makes one chapter navigate differently from the other twenty-eight, and it drops that page out of the mechanical check.

**The lesson checklist item and its `### Lesson N` heading must open with the same clause.** The item may add a parenthetical promise; the heading stays short. A reader who ticks item 4 must land on a section that is recognisably item 4.

**Which pages carry the skeleton.** Every chapter and the opening part carry the full skeleton, because each is a unit of work with a deliverable. The reference pages (glossary, decision map, measurements, audit, limits, reading) carry a status block and a `## Reading` section instead: a glossary with a Lessons checklist and a "how you know you passed" is ceremony, and a rule that demands ceremony gets ignored. `scripts/probe.py` enforces exactly this split, page class by page class.

## The five labels on a claim

Every factual sentence in this book is one of five things, and the reader is told which:

| Label | Meaning | How it reads |
|---|---|---|
| **measured** | produced by a command on a named machine, quoted with the command | "median 41.2 tok/s decode on the machine named below" |
| **arithmetic** | computed from stated assumptions, no run involved | *(arithmetic on stated assumptions: 7B at 4.5 bits is about 4 GB)* |
| **reported** | someone else's number, quoted from a document or paper | "the project's record reports 1.080x at this anchor (*reported, not reproduced*)" |
| **textbook fact** | a definition or a mechanism that a reader can look up | "(textbook fact)" |
| **my judgement** | an opinion about a design trade | "(my judgement, and the reason)" |

An arithmetic estimate is not a measurement, and a reported number is not something this book reproduced. Upgrading one into the other is the one mistake that makes the whole artifact untrustworthy, because every other number inherits it.

## Code citations

- **Path plus symbol.** Write `src/graph/alloc.rs::GraphAllocator` inside backticks. The path alone is allowed only when the sentence really is about a whole file (the kernel sources, the ADR directory); a sentence about a behaviour names the symbol, because the symbol is what the sentence is actually claiming.
- **Never a bare line number.** [The CUDA audit](cuda-docs-audit.md) found three references to files that no longer exist, in a tutorial whose own rule was already "the symbol is the stable address" - line numbers drift the same way, only faster.
- **No commit hash outside [`minfer.lock`](https://github.com/yusiwen/llm-inference-book/blob/main/minfer.lock) and [the baseline page](baseline.md).** Every other page says "at the book's pinned baseline" and links there. One hash to bump, not thirty.
- **Every citation is checked.** `scripts/minfer-drift.py --check-inventory` fails the gate when a page cites an address that is not in the committed inventory, so a new pointer cannot reach a reader unchecked.
- **Symbol names are re-verified when the chapter is written**, not when the outline is written. An outline's citation is a plan.
- **Where the behaviour is inferred from a name and not read**, the sentence says *assumed*.

## Counts are claims

"The book has 29 chapters" is a claim about a number somebody can count. Every count in this book is derived by a command and recorded on the [Measurements](measurements.md) page:

- a chapter count, a lesson count, a page count - derived from the tree, not typed from memory;
- a project count (kernels, ADRs, step records) - derived at the commit named, with the command quoted;
- **a number from another document is attributed to that document**, and never repeated as if this book had verified it.

The failure this rule exists for: a stub's numbers are not facts either. A lesson title drafted before the work carries counts nobody measured, and a chapter written from that title inherits them.

## Figures and maths

- **A figure is SVG produced at build time**, with its source as text in the page (`src/figures/NN-slug.svg`, included inside `<figure class="book-figure">`). A client-side diagram library renders blank wherever scripts are blocked and is lost from print.
- **No LaTeX inside a figure** - the SVG cannot see MathJax. Write `V(s)` as plain text.
- **Maths is vendored**: MathJax lives in `mathjax/`, loaded through `additional-js`. The built-in injects a CDN copy, which leaves every page showing raw dollar signs when the CDN is blocked - and the build stays green while it happens.
- **Every display equation is wrapped** in `<div class="math-display">`. Markdown parses the inside of a display block as ordinary text, so a subscript pair can be read as emphasis and that one equation silently does not render.
- **Three carriers, chosen by what the content would look like if printed.** Typeset mathematics goes to MathJax: `$$` display, `$` inline, no equation numbering, `\text{}` for units so `ms` is upright and `x` is not. A sketch - a bit layout, a byte plan, a pipeline, a dataflow - goes to an SVG; never characters in a monospace block, because a sketch drawn in `+` and `|` reads as code and dies at a narrow window width. A monospace block carries exactly three things: program output and terminal sessions, source code, and a numeric derivation whose point is that the reader can check the digits. A single `=` chain is maths with real numbers in it; several aligned rows of numbers are a table, so the columns are structure rather than spaces.
- **Inline code spans are for identifiers, not for mathematics.** `f32`, `rope.freq_base` and `vec_soft_max_f32` are names the reader will type; `q·k/sqrt(hd)` is an expression, and as a code span it ships a line of symbols instead of a product over a root.
- **A formula that does not parse fails the reader, not the build.** `scripts/texlint.py` (gate step 13) checks braces, `\left`/`\right`, environments, control sequences and stray `$`; the rest needs a rendered page, which is why every page carrying maths is loaded once before it is published.


## Reading and citations

- **Every page's `## Reading` section cites at least one source from the [reference index](references.md), by key.** The key is written in square brackets: `- [gqa] Ainslie et al., ... - arXiv:2305.13245`. `scripts/references.py` fails the build when a page cites nothing external, or cites a key the index does not hold.
- **The index owns the metadata, the page owns the reason.** Title, year and identifier live once, on the index; the chapter's Reading line says what the reader is being sent there *for* when that is not obvious.
- **An arXiv ID is resolved, never recalled.** The index records the date each ID was checked against the API. A source with no ID says so in words; an unchecked ID is removed rather than marked pending.
- **A reported number stays reported.** A figure or claim taken from a paper is labelled as such and is never upgraded into something this book measured.

## Statements about minfer itself

Two kinds of sentence, and they age differently.

- **What an engine must do** is timeless: "decode is limited by bytes read" does not go stale. No date needed.
- **What minfer currently does** is dated by the baseline. Write it next to a link to [the baseline page](baseline.md), or do not write it. A capability sentence ("minfer cannot ...") belongs in [Limits](limits.md) as a pointer to the project's own inventory, never as a property of the world.

When minfer moves, `scripts/minfer-drift.py` finds the addresses and the counts, and `scripts/contract-probes.sh` records the behaviour. Neither of them edits prose: a difference is a question for whoever knows the chapter.

## Status vocabulary (two words, defined)

- **stub**: the skeleton exists (gist, checklist, ledgers, acceptance criteria); the prose does not.
- **draft**: the prose exists and at least one claim in it is known to be missing or wrong; the qualifier after the dash says which.
- **written**: the prose is complete; the qualifier after the dash says what the evidence covers and what it does not.

The word describes the text, not how much of the world it measured. A chapter with no hardware numbers is `written` plus a qualifier, never `draft`. The release stage ("v0.1", "early access") is a book-level fact that belongs on the landing page and in the tag, not stamped on fifteen chapter headers, where it would carry no information and bury the per-chapter evidence.

## Division of labour with the reference project's docs

This book is not a second copy of `docs/`. The project's documentation is a reference for people who already work in that tree; this book is a build sequence for someone who does not. When both could say the same thing:

- mechanics go in the chapter, at teaching depth;
- the project's own files are **linked, not quoted**, and the reader is sent there for depth;
- a place where the project's doc is stale is *corrected in this book and recorded in the audit appendix*, never silently patched over.

## Backfill

Every chapter's `## Backfill` re-asks the book's four questions. Ticking a box means the chapter advanced an answer to that question, and the answer is written in the chapter: where the time goes, why weights are bytes, where the engine/device boundary is, how correctness and speed are known.

## Reading

- [Part 0](../start/index.md) - the contract a reader accepts before Chapter 1
- [Measurements](measurements.md) - how the book's own numbers are produced
- [The CUDA audit](cuda-docs-audit.md) - the code-citation rule, with the evidence that motivated it
