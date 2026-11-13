# Figure plan

> **Status**: planning only - **0 of 46 figures drawn**. This file is repository material, not book content: it is the drawing order and the acceptance test, and it never appears in the built site.
> **Rule**: a figure must be cheaper to author than the paragraph it replaces, and must carry a claim the prose cannot make as quickly. A diagram nobody remembers is a table.
> **Ship rule**: a chapter's P0 figures are written *with* the chapter, in the same sitting, and are part of what "the chapter is drafted" means - a stub may have none, a `draft` may not owe a P0. P1 figures are added when the chapter is revised, or dropped.

46 figures across the 30 pages that have figures: **31 core (P0)** and **15 supporting (P1)**. P1 figures are drawn only after every P0 exists; a P1 that never gets drawn costs nothing, because its lesson is written to stand without it.

## How a figure is built here (and why)

Three properties, each of which the build stays green through when it is broken. The mechanics live in `mdbook-figures`; the short version:

1. **SVG produced at build time, never in the browser.** No Mermaid, no diagram JavaScript: a client-side renderer is blank wherever scripts are blocked and absent from print/PDF.
2. **Inline, never `<img>`.** CSS custom properties do not cross an image boundary, so an image reference loses the theme in the other four themes.
3. **The source is text in git** - `src/figures/NN-slug.svg`, hand-authored with the shared class vocabulary, diffable, reviewed in the same commit as the prose.

The include idiom, with no blank line anywhere inside the SVG:

```markdown
<figure class="book-figure">{{#include ../figures/01-two-regimes.svg}}</figure>

*Figure 1.1 - prefill is limited by arithmetic, decode by bytes read; the same weights, two different walls.*
```

- **File name = chapter prefix + slug** (`01-two-regimes.svg`), **caption number = `Figure <chapter>.<n>`** in reading order within that page. `figcheck` fails when the two disagree.
- **`viewBox` 810 wide** (or 800 where the content needs it) - the page scales a figure to the column, so a 920-wide box shrinks 11.5 px labels to an illegible 8.6 px.
- **Class vocabulary, shared with the author's other handbook**: `fig-card`, `fig-alt` (band), `fig-t` (title text), `fig-s` (secondary text), `fig-n` (numbers), `fig-cap` (the letterspaced header line), `fig-grid`, `fig-edge`, and accents `fig-a-*` named for meaning (`pos`, `neg`, `acc`, `cheap`, `mid`, `dear`). Fill and stroke come from `var(--quote-bg, #...)`, `var(--quote-border, #...)`, `var(--fg, #...)`; the accents are the only literal colours and must clear 3:1 against both card fills.
- **No LaTeX in a figure** - the SVG cannot see MathJax. Plain text: `A = -0.15`, `gamma`, `V(s)`.
- **`role="img"` and an `aria-label`** that states what a reader should take away. The sentence in the table below *is* that label: if it cannot be written in one clause, the figure is not understood yet.
- **Placement**: immediately after the paragraph or table it explains, never in a gallery.

## The numbers rule

A figure is a claim. Until a chapter measures something, every number in a figure is either a **definition** (32 weights per block, 4.5 bits per weight) or **arithmetic on stated assumptions** (a cache size computed from a model's config), and the caption says which. A figure never carries a *reported* number from the reference project's campaign without naming it as reported. This is the same rule the prose follows, and no gate checks it - the caption is read by a human or not at all.

### Part 0 · Before you start

| Fig | File | Pri | Shape | Lesson | What it teaches (the aria-label draft) |
|---|---|---|---|---|---|
| 0.1 | `00-book-map.svg` | P0 | ladder | Part 0 · routing | Six parts and 29 chapters as a build ladder: each rung compiles on top of the one below, and the GPU rungs are marked as needing hardware. |
| 0.2 | `00-four-questions.svg` | P0 | map | Part 0 · the contract | The four book questions and which part answers each, so a stuck reader can route instead of re-reading. |

### Part 1 · The ground truth

| Fig | File | Pri | Shape | Lesson | What it teaches (the aria-label draft) |
|---|---|---|---|---|---|
| 1.1 | `01-two-regimes.svg` | P0 | panel | L2 | Prefill multiplies a matrix by many activation rows and is limited by arithmetic; decode multiplies it by one row and is limited by the bytes it must read. |
| 1.2 | `01-decode-ceiling.svg` | P1 | nomogram | L4 | The decode ceiling as a nomogram: model file size, bytes read per token, and memory bandwidth, with no machine-specific number baked in. |
| 2.1 | `02-gguf-layout.svg` | P0 | byte map | L3 | A GGUF file from byte zero: header, metadata key-values, tensor table, alignment padding, data blob, and the byte range one tensor occupies. |
| 3.1 | `03-views-and-stride.svg` | P0 | grid | L3 | One 2-D tensor whose rows are packed with **no** padding (616 B each), and a two-row view sharing those bytes: the view's pointer is the parent plus 616, its stride is 616, and it owns nothing. The figure corrects the plan - GGML pads between tensors, never between rows. |
| 4.1 | `04-block-anatomy.svg` | P0 | byte map | L1 | A Q4_0 block: one 16-bit scale and 16 packed nibble bytes for 32 weights, which is the 4.5 bits per weight the file pays. |
| 4.2 | `04-block-error.svg` | P1 | bar | L5 | One outlier weight forces a larger block scale, so the other 31 values are quantized more coarsely: the error a block sees. |
| 5.1 | `05-superblock.svg` | P0 | byte map | L2 | A 256-weight K-quant super-block: eight sub-blocks, one shared d/dmin pair, and the 12 bytes holding eight 6-bit scales and eight 6-bit mins - drawn as it is, not as planned: the packing puts sub-block 5's step in three different bytes. |

### Part 2 · Arithmetic

| Fig | File | Pri | Shape | Lesson | What it teaches (the aria-label draft) |
|---|---|---|---|---|---|
| 6.1 | `06-quantized-dot.svg` | P0 | flow | L3 | The dot-product contract as dataflow: f32 activations quantized to Q8_0 once, an integer dot against the weight row, and the scale corrections that restore f32. |
| 7.1 | `07-lanes.svg` | P0 | lanes | L2 | Eight AVX2 lanes and four NEON SDOT lanes doing the same quantized dot, with the nibble unpack and horizontal fold that differ between them. |
| 8.1 | `08-gemv-vs-gemm.svg` | P0 | matrix | L1 | One activation row against a weight matrix (decode, no reuse) versus N rows against the same weights (prefill, each weight read serving N tokens). |
| 9.1 | `09-precision-ledger.svg` | P1 | flow | L5 | Where each op widens and narrows: f16 or bf16 weights entering f32 accumulation, and the ops whose output must stay f32 for the identity gate. |
| 10.1 | `10-attention-shapes.svg` | P0 | matrix | L1 | Queries, keys and values per head, the causal mask as a triangle, and the three matmuls attention performs over the cached prefix. |
| 10.2 | `10-gqa-heads.svg` | P1 | heads | L3 | Grouped-query attention: four query heads reading one key/value head, and the cache size that sharing divides. |

### Part 3 · Language

| Fig | File | Pri | Shape | Lesson | What it teaches (the aria-label draft) |
|---|---|---|---|---|---|
| 11.1 | `11-dialects.svg` | P0 | panel | L2 | One string tokenized under four pre-tokenization dialects, giving different ids, because each family marks spaces and normalizes text differently. |
| 12.1 | `12-prompt-anatomy.svg` | P1 | byte map | L1 | A rendered chat prompt annotated: role markers, the reasoning block, tool definitions, and the generation prefix the model must see last. |
| 13.1 | `13-sampler-pipeline.svg` | P0 | funnel | L1 | The sampler pipeline as a funnel: penalties, top-k, top-p and temperature each remove candidates before one token is drawn. |
| 13.2 | `13-grammar-mask.svg` | P1 | matrix | L5 | A grammar masking the vocabulary at one decoder state, so tokens that would break the schema are impossible rather than unlikely. |

### Part 4 · The graph

| Fig | File | Pri | Shape | Lesson | What it teaches (the aria-label draft) |
|---|---|---|---|---|---|
| 14.1 | `14-two-passes.svg` | P0 | panel | L2 | The imperative forward against build-then-execute, and the four things the split makes possible: assignment, allocation, fusion, reuse. |
| 14.2 | `14-graph-reuse.svg` | P1 | timeline | L4 | Prefill and decode as two graph shapes, the decode one reused for every later step while the cache grows underneath and nothing rebuilds. |
| 15.1 | `15-fusion.svg` | P0 | flow | L4 | Three separate elementwise nodes folded into one pass: three reads and three writes of the same buffer become one of each. |
| 16.1 | `16-assignment.svg` | P0 | graph | L3 | A graph coloured by device with two split boundaries, and the cross-device copy each boundary costs. |
| 17.1 | `17-liveness.svg` | P0 | ledger | L2 | The liveness spans of a graph's buffers, and how overlapping spans share one arena instead of each getting its own allocation. |
| 17.2 | `17-memory-ledger.svg` | P1 | bar | L5 | A memory ledger as stacked bars: resident weights, host-mapped weights, the activation arena, the KV cache and staging, predicted against observed. |
| 18.1 | `18-cells-and-spans.svg` | P0 | grid | L2 | A cell-based cache holding two sequences as interleaved spans, one sharing a prefix with the other, and the same cache after defragmentation. |
| 18.2 | `18-session-file.svg` | P1 | byte map | L5 | A saved KV session as a versioned file: version, checksum, per-layer extents, so an older engine refuses it instead of reinterpreting memory. |
| 19.1 | `19-family-deltas.svg` | P0 | block | L3 | One dense transformer layer with the four places model families differ marked: head dimension, query/key norm, attention bias, tied output embedding. |
| 20.1 | `20-engine-loop.svg` | P0 | loop | L1 | The generation loop: prefill builds and runs one graph, decode reuses a second, the cache grows, and only cancellation or overflow interrupts it. |
| 20.2 | `20-step-budget.svg` | P1 | bar | L1 | What a steady decode step spends its time on: the model's arithmetic against graph dispatch, sampler and host overhead. |

### Part 5 · Devices

| Fig | File | Pri | Shape | Lesson | What it teaches (the aria-label draft) |
|---|---|---|---|---|---|
| 21.1 | `21-four-layers.svg` | P0 | stack | L1 | The four layers every backend has - runtime, launch, kernels, executor - drawn for CPU, Metal and CUDA, with the only polymorphic layer marked. |
| 22.1 | `22-memory-hierarchy.svg` | P0 | stack | L3 | The GPU memory hierarchy from registers to device memory with the cost of each level, and the levels a decode kernel actually touches. |
| 22.2 | `22-access-patterns.svg` | P0 | grid | L4 | Three warps reading the same 32 values - coalesced, strided, diverged - and the memory transactions each one costs. |
| 23.1 | `23-seam.svg` | P0 | stack | L1 | The Rust-CUDA seam as five crossings - allocation, ownership, error, wait, link - and what each costs when it is left unowned. |
| 24.1 | `24-ladder.svg` | P0 | ladder | L1 | The matmul ladder as five rungs from elementwise to integer matrix-vector, with the reuse each rung gains and the shape that selects it. |
| 24.2 | `24-tiling.svg` | P1 | matrix | L3 | Shared-memory tiling: one weight tile loaded once and used by every thread in the block, and the reuse factor as arithmetic. |
| 25.1 | `25-flash-tiling.svg` | P0 | matrix | L1 | Attention prefill tiled so the score matrix never materializes: the causal triangle, the tiles on it, and the running max and sum replacing a full row. |
| 25.2 | `25-cuda-graph.svg` | P1 | timeline | L5 | A hundred small launches per decode step against one captured graph replayed, on the same time axis. |
| 26.1 | `26-gate-chain.svg` | P0 | flow | L3 | The campaign's gate chain: hypothesis, predicted counts, one lever, correctness, identity, a delta above the noise floor, then land or revert. |
| 26.2 | `26-campaign-verdicts.svg` | P1 | bar | L4 | The campaign's own record by verdict - landed, reverted, measurement-only, closed - showing most levers fail and the failures are the documentation. |

### Part 6 · Proving it

| Fig | File | Pri | Shape | Lesson | What it teaches (the aria-label draft) |
|---|---|---|---|---|---|
| 27.1 | `27-identity-chain.svg` | P0 | flow | L2 | The correctness chain from a hashed fixture through per-op oracles to end-to-end ids, each comparison marked bitwise or tolerance-classed. |
| 27.2 | `27-bisect.svg` | P1 | graph | L4 | Bisecting a mismatch to the first divergent node by comparing per-node dumps down the graph instead of guessing at the last one that looks wrong. |
| 28.1 | `28-ab-window.svg` | P0 | timeline | L3 | Interleaved A and B runs inside one measurement window with their medians and the noise floor, and the cross-session comparison that is invalid. |
| 28.2 | `28-warmup.svg` | P1 | curve | L2 | Throughput against elapsed time from a cold start to steady state, and where a warmup budget must end before a median means anything. |
| 29.1 | `29-slots.svg` | P0 | queue | L1 | Serving slots against a bounded worker queue: two conversations in flight, one lagging, and the drain that finishes in-flight tokens at shutdown. |
| 29.2 | `29-speculation.svg` | P0 | flow | L5 | Speculative decoding: a draft proposing k tokens, the target verifying them in one pass, the accepted prefix kept, and the acceptance rate where the trade stops paying. |

## Drawing batches

One batch per sitting, drawn in chapter order, each batch committed on its own so a broken figure is one commit to revert:

| Batch | Figures | Why this order |
|---|---|---|
| 1 | 0.1-0.2, 1.1-1.2, 2.1, 3.1 | the front door and the file layer: these set the visual vocabulary every later figure reuses |
| 2 | 4.1-4.2, 5.1, 6.1, 7.1 | quantization and the dot product: the diagrams with the most arithmetic in them |
| 3 | 8.1, 9.1, 10.1-10.2, 11.1, 12.1, 13.1-13.2 | the shape diagrams (matrices, funnels) where a picture beats a table |
| 4 | 14.1-14.2, 15.1, 16.1, 17.1-17.2, 18.1-18.2, 19.1, 20.1-20.2 | the graph part: layout-heavy, one shared grid idiom |
| 5 | 21.1, 22.1-22.2, 23.1, 24.1-24.2, 25.1-25.2 | devices: the only batch that needs no measured numbers at all |
| 6 | 26.1-26.2, 27.1-27.2, 28.1-28.2, 29.1-29.2 | the proof part: timelines and gate chains, drawn last because the chapters define the gates |

26.2 (`26-campaign-verdicts`) is the only figure whose numbers must be **derived from the reference project's record** (`docs/CUDA_OPTIMIZATION.md` section 0, one row per step, each labelled landed / reverted / measurement-only / closed). It is drawn after that count is taken with a command and recorded on `src/appendix/measurements.md`, not before.

## Accepting a figure

```bash
python3 scripts/figcheck.py --root . --strict   # structure, theme classes, geometry, captions
mdbook build && grep -o '<svg' book/<page>/index.html | wc -l
```

Then, because the build cannot see any of it: open the page, switch every shipped theme (`light`, `rust`, `coal`, `navy`, `ayu` - `rust` is the one that bites), and look at the rendered figure. One figure per check, read back before the next: a figure can pass `figcheck`, pass the character gate and still have a label colliding with a bar.

## Current state

| Quantity | Value | How |
|---|---|---|
| Planned figures | 46 | this file |
| Drawn | 7 | `ls src/figures/*.svg \| wc -l` |
| Pages that will carry at least one figure | 30 | every page in the table above |
| Chapters with no figure | 0 | the five-lesson chapters were checked one by one; a chapter only appears here where its shape is the lesson |
