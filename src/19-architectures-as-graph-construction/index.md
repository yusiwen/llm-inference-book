# Chapter 19 · Architectures as Graph Construction

> - **Compute tier**: ✅ any laptop
> - **Status**: draft
> - **Build step S19**: Two model architectures built from the same graph vocabulary, plus a loader that dispatches on the file's declared architecture.
> - **Rule of this book**: The graph machinery is Chapters 14-17. This chapter is the shape of a particular family of models: what a Qwen3 layer contains that a Qwen2 layer does not, and what adding a third costs.

## First, the gist

A transformer architecture is a small, opinionated list of operations with specific shapes: how many heads, how
wide, which normalization, whether a projection has a bias, whether the output embedding is tied to the input. An
engine worth calling one turns the file's *declared* architecture into graph nodes whose shapes are read from
metadata, and reuses one vocabulary for every family. A program copies the graph and edits it, and the copy then
inherits every absence silently.

The numbers say where the line falls. A Qwen2.5 layer builds **18 nodes** in this engine, out of a vocabulary of
**23 op kinds**, of which this family uses 11. The whole prefill graph is 443 nodes for 24 layers. The second
family (Qwen3) is **1,854 lines against Qwen2's 2,876** - 64 per cent - and **6 test functions against 29**,
because it inherits the vocabulary, the allocator, the scheduler, the fusion pass and the weight registry. And the
third entry in the dispatch table, Llama, **calls the Qwen2 loader and builder** rather than owning either: what
is Llama-specific is the stop set, resolved from the file.

That is the chapter's claim, and it is checkable rather than rhetorical: **a family owns shapes, flags and a stop
set.** Everything else is the vocabulary, and the vocabulary is where a new op goes - with its own tests, on every
backend that claims it.

## What you'll be able to do

- Build a transformer layer from the model file's parameters, with shapes derived rather than hard-coded.
- Name the differences between the Qwen2, Qwen3 and Llama-3 style layers, point at the line each one lives on, and
  say which of them costs an optimisation rather than a line.
- Refuse an unknown architecture by name, and read a refusal that names a *group* of fields for what it is.
- Estimate what adding a fourth family actually costs in an engine of this shape, in lines and in tests.

## Lessons

- [ ] 1. A layer, in ops: what every dense transformer has in common
- [ ] 2. Shapes from metadata: head counts, head dimension, hidden width, and the derived quantities
- [ ] 3. Where families differ: decoupled head dimension, per-head query/key norm, tied embeddings, attention bias
- [ ] 4. Dispatch on the declared architecture, and the refusal when it is unknown
- [ ] 5. Adding a fourth family: the checklist, and the tests that must exist first

### Lesson 1 · A layer, in ops: what every dense transformer has in common

Before the differences, the common case. Run one real file through the engine with the per-node trace on
(Chapter 14's switch) and the layer stops being an abstraction - it is a numbered list of nodes with shapes:

<!-- facts: family.fam-layer -->
| # | op | tensor | shape |
|---|---|---|---|
| 0 | `matmul` | `matmul_blk.0.attn_q.weight` | 896 x 34 |
| 1 | `matmul` | `matmul_blk.0.attn_k.weight` | 128 x 34 |
| 2 | `matmul` | `matmul_blk.0.attn_v.weight` | 128 x 34 |
| 3 | `rope` | `rope` | 896 x 34 |
| 4 | `rope` | `rope` | 128 x 34 |
| 5 | `input` | `cells` | 34 |
| 6 | `kvcache_store` | `kv_store.0` | 128 x 4096 |
| 7 | `kvcache_load` | `kv_load.0` | 128 x 4096 |
| 8 | `input` | `attn_span` | 68 |
| 9 | `input` | `seq_ids` | 34 |
| 10 | `attn` | `attn` | 896 x 34 |
| 11 | `matmul` | `matmul_blk.0.attn_output.weight` | 896 x 34 |
| 12 | `add` | `add` | 896 x 34 |
| 13 | `rms_norm` | `rms_norm` | 896 x 34 |
| 14 | `matmul` | `matmul_blk.0.ffn_gate.weight` | 4864 x 34 |
| 15 | `matmul` | `matmul_blk.0.ffn_up.weight` | 4864 x 34 |
| 16 | `silu` | `silu` | 4864 x 34 |
| 17 | `swiglu` | `mul` | 4864 x 34 |
| 18 | `matmul` | `matmul_blk.0.ffn_down.weight` | 896 x 34 |
| 19 | `add` | `add` | 896 x 34 |
| 20 | `rms_norm` | `rms_norm` | 896 x 34 |

The slice holds 21 nodes, but 3 of them are the `input` nodes (`cells`, `attn_span`, `seq_ids`) that the graph creates **once** and every layer references - so a layer contributes **18 nodes of its own**. The prefill graph is 18 x 24 = 432 plus 11 model-level nodes (the embedding, the final norm, one matmul for the vocabulary, and those shared inputs).
<!-- /facts -->

Read that list for its *shape*, not its contents: seven matmuls, one attention, two RMS norms, two residual adds, a
RoPE pair, a SiLU and its SwiGLU multiply, one cache store and one cache load. The KV node carries `128 x 4096` -
the whole arena, not this step's rows - which is why Chapter 17 counted it separately from the transient buffers.
And the three `input` nodes near the middle are a trap for the reader who counts the window: `cells`,
`attn_span` and `seq_ids` are created **once** for the whole graph and referenced by all 24 layers, so a layer
contributes 18 nodes of its own and the window shows 21.

The same run settles how big a vocabulary has to be:

<!-- facts: family.fam-ops -->
| Op | Prefill nodes | Decode nodes | Per layer |
|---|---|---|---|
| `matmul` | 169 | 169 | 7 + 1 at model level |
| `rms_norm` | 49 | 49 | 2 + 1 at model level |
| `add` | 48 | 48 | 2 |
| `rope` | 48 | 48 | 2 |
| `attn` | 24 | 24 | 1 |
| `kvcache_load` | 24 | 24 | 1 |
| `kvcache_store` | 24 | 24 | 1 |
| `silu` | 24 | 24 | 1 |
| `swiglu` | 24 | 24 | 1 |
| `input` | 6 | 5 | **not per layer** |
| `get_rows` | 3 | 1 | **not per layer** |
| **total** | **443** | **440** | **18, plus 3 shared inputs at model level** |

11 op kinds cover the whole model. That number is the chapter's real subject: a family is a *recipe over a vocabulary*, so the vocabulary's size is what a new family is measured against - and the shared vocabulary in this engine has 23 variants.
<!-- /facts -->

Eleven kinds for a dense family, and the per-layer column is exact: `matmul` at 169 for 24 layers is seven per
layer plus one for the vocabulary. That is why a builder is short - the layer body is a sequence of calls into a
vocabulary, not a sequence of kernels:

<!-- code: src/models/qwen2/graph.rs::Qwen2Graph::build@e844591 -->
```rust
            let residual = h;

            // pre-norm
            let normed = b.rms_norm(h, l.attn_norm.as_ref(), eps);

            // Q/K/V projections (+ biases). decode (nt==1) with GPU QKV
            // fusion (G4 + D3-8) has two classes:
            // - concat class (wq|wk|wv same quant type): `Op::FusedQKV` — one
            //   concat matmul + one fused bias+rope+store kernel;
            // - mixed-quant class (e.g. Q6_K attn_v among Q4_K q/k): three
            //   separate matmuls (no bias) + one `Op::QkvBiasRopeStore`
            //   epilogue (CUDA-only today; Metal keeps the unfused chain for
            //   these layers). Both replace 3 matmul + 3 bias + 2 rope +
            //   2 store dispatches.
            // #144 item 1: a packed cache builds the fused epilogue too — CUDA's
            // `attn_bias_rope_store_q8_0` gives one thread a whole (head,
```
<!-- /code -->

That is the whole of the "common" layer: pre-norm, three projections, attention, an output projection, and then the
same shape again for the feed-forward. Everything a family changes is a substitution inside it.

### Lesson 2 · Shapes from metadata: head counts, head dimension, hidden width, and the derived quantities

The builder opens by turning the file's hyperparameters into local numbers, and every later shape is one of them:

<!-- code: src/models/qwen2/graph.rs::Qwen2Graph::build@e844591 -->
```rust
    pub fn build(model: &Qwen2Model, params: &GraphParams) -> ComputeGraph {
        let hp = &model.hparams;
        // Namespaced fused-weight names must match the loader's registration
        // keys (Qwen2Model::ns; empty for the primary model).
        let wns = &model.ns;
        let nt = params.n_tokens;
        let ne = hp.n_embd as usize;
        let nh = hp.n_head as usize;
        let nk = hp.n_head_kv as usize;
        let hd = hp.n_embd_head() as usize;
        let nkt = hp.n_kv_embd as usize;
        let hd_kv = nkt / nk;
        let nf = hp.n_ff as usize;
        let eps = hp.f_norm_rms_eps;
        let n_ctx = params.cparams.n_ctx;
```
<!-- /code -->

Six lines, each a decision. `nh` and `nk` come from metadata (14 and 2 on the 0.5B this chapter measured, which is
why q's width is 896 and k/v's is 128). `hd` is *not* `ne / nh`: it is asked for, because a family may decouple the
head width from the embedding width. `hd_kv` is derived from the KV row width instead, because with grouped-query
attention the KV head width is not the query head width. `nkt` - the logical KV row width - is what the cache sizes
a cell from (Chapters 10 and 18).

The trait forces that request to be answered rather than inherited:

<!-- code: src/models/mod.rs::ModelDef::n_embd_head@e844591 -->
```rust
    fn n_layer(&self) -> usize;
    fn n_head_kv(&self) -> usize;
    fn n_embd_head(&self) -> usize;
    fn n_kv_embd(&self) -> usize;
    fn n_vocab(&self) -> usize;

```
<!-- /code -->

`Required, not defaulted` is the pattern to copy. A default here would be exactly the silent bug this chapter is
about: a family whose head width is decoupled would inherit `n_embd / n_head`, build a graph with the wrong
shapes, and not crash - it would produce plausible text with a corrupted attention pattern.

### Lesson 3 · Where families differ: decoupled head dimension, per-head query/key norm, tied embeddings, attention bias

<figure class="book-figure">{{#include ../figures/19-family-deltas.svg}}</figure>

*Figure 19.1 - one dense layer with the four places these families diverge marked, and what each divergence costs: a shape (head width), an op (per-head q/k norm), a loader rule (a bias, or its absence; a tied output), and - the one people miss - a fusion that stops being applicable.*

The Qwen3 builder states its own deltas in the module comment, and states the cost of one of them:

<!-- code: src/models/qwen3/graph.rs::Qwen3Graph@e844591 -->
```rust
//! Qwen3 (dense) compute-graph construction + graph forward path.
//!
//! Mirrors `qwen2/graph.rs` with the two Qwen3 deltas
//! (docs/QWEN3-SUPPORT-PLAN.md §2):
//!   1. head dim `hd` comes from `hparams.n_embd_head` (128), not
//!      `n_embd / n_head` (64) — this drives the Q/K/V/wo widths, the RoPE dims,
//!      the attention scale and the KV row stride.
//!   2. per-head Q/K RMSNorm (`Op::QkNorm`) after the projections, before RoPE.
//!
//! The decode fused-QKV path (`Op::FusedQKV`) is intentionally NOT used for
//! Qwen3: the `attn_bias_rope_store` kernel does bias+rope+store in one pass
//! and cannot express the per-head norm between projection and rope (Phase E
//! follow-up). The unfused 3-matmul path is correct on both backends. The
//! fused-FFN decode path (`Op::FusedFFN`) IS reused (FFN is Qwen2-identical).
```
<!-- /code -->

Two deltas and a *sacrifice*. The second delta is an op that must be inserted *between* two ops a fused kernel
already performs in one pass, so on the fused path it cannot be expressed, and Qwen3 decode falls back to the
unfused chain of three matmuls. That is the real shape of a new family's cost: not lines, but an optimisation you
can no longer take. The fused-FFN path *is* reused, because the feed-forward is Qwen2-identical.

Where the two deltas live is four lines:

<!-- code: src/models/qwen3/graph.rs::Qwen3Graph::build@e844591 -->
```rust
            } else {
                let q = b.matmul(normed, l.wq.as_ref().unwrap(), None);
                let k = b.matmul(normed, l.wk.as_ref().unwrap(), None);
                let v = b.matmul(normed, l.wv.as_ref().unwrap(), None);

                // Qwen3: per-head RMSNorm on Q and K before RoPE (llama.cpp
                // `build_norm(Qcur, attn_q_norm, ...)` on the [hd, n_head, nt]
                // reshaped buffer — contiguous rows in our flat layout).
                let q = b.qk_norm(q, l.q_norm.as_ref(), hd, nh, eps);
                let k = b.qk_norm(k, l.k_norm.as_ref(), hd, nk, eps);

                let q = b.rope(
                    q,
                    inp_pos,
                    hp.rope_style,
                    RoPEMeta {
                        freq_base: hp.rope_freq_base,
                        freq_scale: hp.rope_freq_scale,
                        freqs: hp.rope_freqs.map(Box::new),
                        n_head: nh,
                        hd,
                    },
```
<!-- /code -->

`b.qk_norm(...)` twice - between the projections and the rope, each over `hd` and over its own head count. A copied
Qwen2 builder would have had three matmuls and a rope here, and would have been wrong in a way that produces text.

The four differences, each with a count behind it so the claim is checkable rather than remembered:

<!-- facts: family.fam-deltas -->
| Difference | Anchor string | Qwen2 | Qwen3 |
|---|---|---|---|
| per-head RMSNorm on q and k after the projections, before RoPE | `QkNorm` | 0 | 2 |
| head width read from metadata instead of derived from `n_embd / n_head` | `n_embd_head` | 1 | 2 |
| an attention projection with a bias term | `attn_q_bias` | 1 | 0 |
| `output.weight` absent, so the input embedding is reused | `tok_embd.clone()` | 1 | 1 |

Four differences, one per row, and the counts say which family implements each. A zero is the interesting entry: it is the absence that a copied graph inherits silently.
<!-- /facts -->

**A zero is the interesting entry**, and the tied-output row is the one to think about: both families tie, so the
counts are 1 and 1 - but tying is not a delta *between* them, it is a delta from a model that ships an `lm_head`.
The row stays because the question a new family raises is "does this one tie?", and the answer lives in the loader
rather than in the graph:

<!-- code: src/models/qwen2/loader.rs::qwen2::loader::load@e844591 -->
```rust
    // Output norm
    let output_norm = load_one(tn::OUTPUT_NORM);

    // Output weight (with weight tying fallback)
    let output = load_one(tn::OUTPUT).unwrap_or_else(|| tok_embd.clone());

    // Output bias (optional)
```
<!-- /code -->

### Lesson 4 · Dispatch on the declared architecture, and the refusal when it is unknown

The file says what it is in `general.architecture`, and the engine's dispatch is one `match`:

<!-- code: src/models/mod.rs::load_model_configured@e844591 -->
```rust
    let ctx = &model.parts[0].ctx;
    let arch = ctx
        .get_key_val_str("general.architecture")
        .ok_or_else(|| ModelLoadError {
            arch: String::new(),
            reason: "GGUF metadata has no `general.architecture`".to_string(),
        })?;
    let mut loaded: Box<dyn ModelDef> =
        match arch.as_str() {
            "qwen2" => Box::new(qwen2::loader::load(model, ns, offload).map_err(|reason| {
                ModelLoadError {
                    arch: arch.clone(),
                    reason,
                }
            })?),
            "qwen3" => Box::new(qwen3::loader::load(model, ns, offload).map_err(|reason| {
                ModelLoadError {
                    arch: arch.clone(),
                    reason,
                }
            })?),
            // #723: Llama 3.1/3.2/3.3 dense. The loader and graph are the dense
            // Qwen2 ones by construction: the loader reads `llama.*` metadata, ties
            // `output.weight` to `token_embd` when the file has no lm_head, and
            // tolerates a missing attention bias; the graph already runs the
            // unfused chain when a bias is absent. What is Llama-specific is the
            // stop set (`<|eot_id|>` / `<|eom_id|>`), resolved from the file.
            "llama" => Box::new(qwen2::loader::load(model, ns, offload).map_err(|reason| {
                ModelLoadError {
                    arch: arch.clone(),
                    reason,
                }
            })?),
            other => {
                return Err(ModelLoadError {
                    arch: other.to_string(),
                    reason: "unsupported architecture (supported: qwen2, qwen3, llama)".to_string(),
                });
            }
        };
```
<!-- /code -->

Three arms and a default. Note the third arm's comment: `llama` gets neither its own loader nor its own builder - it
is the dense Qwen2 pair, with the loader reading `llama.*` metadata, tying the output weight when the file has no
`lm_head`, and tolerating a missing attention bias. Three families, two implementations, one vocabulary; the
chapter's "what a family owns" claim is visible in the code rather than asserted.

The default arm is where a book about measurement gets its best example, because a refusal's text is an interface.
Three byte-patched copies of one working file, three real runs:

<!-- facts: family.fam-refusals -->
| What was changed in the file | Exit | The engine's message |
|---|---|---|
| `general.architecture` renamed to `qwen9` | 1 | `Error: qwen9 model: unsupported architecture (supported: qwen2, qwen3, llama)` |
| the key `qwen2.block_count` renamed to `qwenX.block_count` | 1 | `Error: qwen2 model: GGUF metadata: cannot determine hparams (qwen2.*/llama.* embedding_length, attention.head_count, block_count, feed_forward_length)` |
| the key `qwen2.attention.head_count` renamed to `qwenX.…` | 1 | `Error: qwen2 model: GGUF metadata: cannot determine hparams (qwen2.*/llama.* embedding_length, attention.head_count, block_count, feed_forward_length)` |

Three byte-patched copies of one working file, and three refusals: the first names the architecture, the second and third name the *group* of fields rather than the one that is missing - the resolver tries `qwen2.*` and then `llama.*`, so by the time it gives up it cannot say which of the four was absent. That is the difference between an error that diagnoses and an error that only refuses, and the chapter says which one this is.
<!-- /facts -->

The first message names the architecture and the supported set: a user holding a Qwen2.5-VL file learns immediately
which field was wrong and what the engine does know. The other two deliberately do **not** name the missing field,
and the reason is in the resolver - it tries `qwen2.*` and then `llama.*`, so by the time it gives up it knows only
that a group of four is unsatisfied. That is an error that refuses but does not diagnose, and the honest thing for
a chapter to do is say so and name the fix: the resolver would have to record which candidate key it last looked
for, which means the lookup cannot stay a chain of `or_else`.

### Lesson 5 · Adding a fourth family: the checklist, and the tests that must exist first

The engine has already paid for this lesson once. The two loaders used to carry their own copy of the CUDA
per-tensor registration rule, and the copies drifted:

<!-- code: src/models/weight_reg.rs::register_cuda_weight@e844591 -->
```rust
//! The one CUDA per-tensor weight-registration rule shared by the qwen2 and qwen3
//! loaders (issue #167).
//!
//! Before #167 each loader carried its own copy of this dispatch, and the two copies had
//! already drifted twice: the qwen3 copy had neither the q4_K `W_dsc` plane registration
//! (r59, the rule #165 hardened into [`crate::q4k_dsc::q4k_dsc_plane_admitted`]) nor the
//! `TensorType::F16` branch (#141). A q4_K Qwen3 weight therefore fell back to
//! `mmq_raw_nb_bt`'s in-kernel scalar dsc decode, and an f16 Qwen3 model was dropped to
//! the CPU by the all-or-nothing `weights_on_cuda` gate even on a build with the f16
//! kernels. Both loaders now call [`register_cuda_weight`], so the type coverage has one
//! authority and cannot diverge by edit.
//!
//! The **decision** ([`cuda_weight_reg`]) is pure on purpose — no `CudaState`, no
//! environment. CI's CUDA job only compile-checks the device modules (a hosted runner has
//! no GPU), so a predicate that lived behind `#[cfg(feature = "cuda")]` would never be
//! *run* in CI; keeping it pure means the CPU job executes its tests. [`register_cuda_weight`]
//! is the thin `cuda`-gated half that carries the decision out.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]
```
<!-- /code -->

*Already drifted twice* is the sentence to keep. The first drift cost a Q4_K Qwen3 weight its device kernel and
fell back silently to a scalar decode; the second dropped an f16 model to the CPU entirely. Neither crashed, and
neither could have been found by reading one loader alone: they are found by comparing two implementations of the
same rule. That is why the rule now has one authority and the family modules call it.

So the checklist for a fourth family is short, and every line is a thing that needs a test before the family lands:

1. **A loader** that reads the family's metadata prefixes and refuses a missing required field by name.
2. **A builder** that reads shapes from metadata - especially the head width - and substitutes only inside the
   common layer body.
3. **A stop set** resolved from the file rather than hard-coded, because it is the one thing family-specific about
   generation that is not a shape.
4. **A dispatch arm**, plus rows in the support matrix (Chapter 16) for every op the family uses on every backend.
5. **An identity test** against a reference implementation on one file of that family, and a shape test that
   deletes one metadata field and requires the refusal - then reads the message to see whether it names the field.

What that costs here, measured rather than guessed:

<!-- facts: family.fam-cost -->
| Module | Lines (without tests) | Lines of tests | `#[test]` fns |
|---|---|---|---|
| `src/models/qwen2/` | 2,876 | 4,197 | 29 |
| `src/models/qwen3/` | 1,854 | 621 | 6 |
| `src/graph/ops.rs` (shared) | 385 | - | - |
| `src/graph/alloc.rs` (shared) | 2,879 | - | - |
| `src/graph/registry.rs` (shared) | 567 | - | - |
| `src/models/weight_reg.rs` (shared) | 209 | - | - |

The second family is 64% of the first in code and 6 tests against 29 - not because Qwen3 is simpler, but because the vocabulary, the allocator, the scheduler, the fusion pass and the weight registry were already there. What a family owns is shapes, flags and a stop set.
<!-- /facts -->

The second family is 64 per cent of the first in code and a fifth of it in tests, and the shared modules it leans on
are 4,040 lines. A fourth family should be *cheaper* than the second was if the vocabulary is stable - and if it is
not cheaper, the reason will be that the family needs a new op, which is a vocabulary decision with its own review,
not a family decision.

## Family differences

| Feature | Qwen2 / Qwen2.5 | Qwen3 | Llama 3.x | Where it shows in the graph |
|---|---|---|---|---|
| head dimension | `n_embd / n_head` | from metadata (`n_embd_head`) | from metadata | width of q/k/v and of a KV cell; the RoPE dims; the attention scale |
| query/key norm | none | per-head RMSNorm after the projections | none | an `Op::QkNorm` pair, and the loss of the fused qkv+rope path |
| attention bias | optional, read when present | none | tolerated absent | a bias+rope+store kernel on decode, or three matmuls plus an `add` |
| tied output embedding | tie when `output.weight` is absent | tie when absent | tie when absent | one matmul against the embedding instead of a second tensor |
| RoPE table | from `rope.freq_base` | same, wider table | `llama3` scaling factors | the `rope` node's inputs, and the size of the table it reads |

## Project

**Build step S19 · two families, one vocabulary.** Build the second architecture only after the first one's graph
vocabulary is stable. The second one is the test of whether the first was an engine or a program.

Keep the per-family code to shapes, flags and a stop set. If a family needs a new op, the op goes in the shared
vocabulary with its own tests - and if the op has to sit *between* two that a fused kernel already fuses, say so
out loud, because the family has just cost you a fast path.

## Deliverable

- Two architectures built from one vocabulary, with shape tests derived from metadata
- A dispatch that refuses an unknown architecture by name, and a named refusal for each required metadata group
- A short written estimate - lines, tests, and any fusion lost - for a fourth family

## How you know you passed

- Both architectures produce logits on a real file of that family, and their graphs differ only where the family
  table predicts.
- Deleting one metadata field makes the build fail, and you can say whether the message names the field or the
  group it belongs to - and why.
- Adding a new op does not require touching either family module: you can point at the one place in the vocabulary
  where it belongs.
- You can point at the one place a new family would need new code, and it is not a copy of the whole graph.

## Reading

- code: `src/models/mod.rs`, `src/models/qwen2/graph.rs`, `src/models/qwen3/graph.rs`, `src/models/weight_reg.rs` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/inference_e2e_walkthrough/03-model-dispatch-weights.md`, `docs/MODEL-SUPPORT-ROADMAP.md`, `docs/SUPPORT-MATRIX.md`
- decisions: [ADR-0038](https://github.com/yusiwen/minfer/blob/master/docs/adr/0038-the-per-tensor-registration-dispatch-is-one-shared-rule.md) (the per-tensor registration dispatch is one shared rule), [ADR-0075](https://github.com/yusiwen/minfer/blob/master/docs/adr/0075-a-model-owns-the-mapping-it-reads.md) (a model owns the mapping it reads)
- [qwen3] Yang et al., *Qwen3 Technical Report* (2025) - arXiv:2505.09388
- [llama3] Grattafiori et al., *The Llama 3 Herd of Models* (2024) - arXiv:2407.21783
- [qwen2] Yang et al., *Qwen2 Technical Report* (2024) - arXiv:2407.10671

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)

<!-- facts: family.provenance -->
*`scripts/family.py`; runs the binary with its per-node trace on one real file and counts the nodes by op kind, slices one transformer layer out of the graph, runs three byte-patched copies of the same file through the loader to capture its refusals verbatim, and counts lines and `#[test]` functions per family module in the pinned checkout; writes `facts/family.json`; on **Darwin 27.2.0, arm64**; node counts and one layer's slice from the engine's own `MINFER_TRACE` dump; refusals are the binary's stderr on byte-patched copies of the same file; source counts are line and `#[test]` counts in the pinned checkout. Command: see [the provenance index](../appendix/provenance.md#family). Raw numbers: `facts/family.json`.*
<!-- /facts -->
