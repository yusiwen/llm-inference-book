# Chapter 19 · Architectures as Graph Construction

> **Compute tier**: ✅ any laptop | **Status**: stub | **Build step S19**: Two model architectures built from the same graph vocabulary, plus a loader that dispatches on the file's declared architecture. | **Rule of this book**: The graph machinery is Chapters 14-17. This chapter is the shape of a particular family of models: what a Qwen3 layer contains that a Qwen2 layer does not, and what adding a third costs.

## First, the gist

A transformer architecture is a small, opinionated list of operations with specific shapes: how many heads, how wide, which normalization, whether a projection has a bias, whether the output embedding is tied to the input. The engine's job is to turn the file's declared architecture into graph nodes with the right shapes, and to reuse as much machinery as possible across families. The interesting content is the differences, because they are where a copied graph goes wrong: a decoupled head dimension, per-head normalization on queries and keys, a tied output embedding, a different RoPE table, a different stop-token set.

## What you'll be able to do

- Build a transformer layer from the model file's parameters, with shapes derived rather than hard-coded.
- Name the specific differences between the Qwen2, Qwen3 and Llama-3 style layers, and where each shows up in the graph.
- Estimate what adding a new architecture actually costs in an engine of this shape.

## Lessons

- [ ] 1. A layer, in ops: what every dense transformer has in common
- [ ] 2. Shapes from metadata: head counts, head dimension, hidden width, and the derived quantities
- [ ] 3. Where families differ: decoupled head dimension, per-head query/key norm, tied embeddings, attention bias (payoff: the four places a copied graph breaks)
- [ ] 4. Dispatch on the declared architecture, and the refusal when it is unknown
- [ ] 5. Adding a fourth family: the checklist, and the tests that must exist first

## Family differences

| Feature | Qwen2 / Qwen2.5 | Qwen3 | Llama 3.x | Where it shows in the graph |
|---|---|---|---|---|
| head dimension |  |  |  |  |
| query/key norm |  |  |  |  |
| attention bias |  |  |  |  |
| tied output embedding |  |  |  |  |
| RoPE table |  |  |  |  |

## Project

**Build step S19 · two families, one vocabulary.** Build the second architecture only after the first one's graph vocabulary is stable. The second one is the test of whether the first was an engine or a program.

Keep the per-family code to shapes, flags and a stop set. If a family needs a new op, the op goes in the shared vocabulary with its own tests.

## Deliverable

- Two architectures built from one vocabulary, with shape tests derived from metadata
- A dispatch that refuses an unknown architecture by name
- A short written estimate: the files and tests a fourth family would need

## How you know you passed

- Both architectures produce logits on a real file of that family.
- Deleting one metadata field makes the build fail with a message naming that field.
- You can point at the one place a new family would need new code, and it is not a copy of the whole graph.

## Reading

- code: `src/models/mod.rs`, `src/models/qwen2/graph.rs`, `src/models/qwen3/graph.rs`, `src/models/weight_reg.rs` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/inference_e2e_walkthrough/03-model-dispatch-weights.md`, `docs/MODEL-SUPPORT-ROADMAP.md`, `docs/SUPPORT-MATRIX.md`
- decisions: ADR-0038 (the per-tensor registration dispatch is one shared rule), ADR-0075 (a model owns the mapping it reads)
- [qwen3] Yang et al., *Qwen3 Technical Report* (2025) - arXiv:2505.09388
- [llama3] Grattafiori et al., *The Llama 3 Herd of Models* (2024) - arXiv:2407.21783
- [qwen2] Yang et al., *Qwen2 Technical Report* (2024) - arXiv:2407.10671

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
