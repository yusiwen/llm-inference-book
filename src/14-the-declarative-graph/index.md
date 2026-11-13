# Chapter 14 · From Imperative Loop to Declarative Graph

> - **Compute tier**: ✅ any laptop
> - **Status**: stub
> - **Build step S14**: Your forward pass rebuilt as a pure data structure: nodes, inputs, outputs, built without touching memory, then executed by a separate walk.
> - **Rule of this book**: This chapter is the graph's shape and why it exists. The op vocabulary and fusion are Chapter 15; who executes a node is Chapter 16; where buffers come from is Chapter 17.

## First, the gist

The obvious way to write a forward pass is a function that computes each layer in turn, calling kernels as it goes. It works, and it traps you: nothing can be inspected before it runs, nothing can be reused between steps, and the choice of which device runs which operation is buried inside the code. The **compute graph** inverts that. Building the graph is pure - it allocates nothing and computes nothing - and execution is a separate pass that can assign operations to devices, reuse buffers, and cache the whole structure when only the input data changes. This is the single most consequential design decision in the book, and it is the reason a decode step can cost microseconds of overhead rather than milliseconds.

## What you'll be able to do

- Rebuild a forward pass as graph construction, keeping build and execute strictly separate.
- Explain why a graph whose topology depends only on shapes (not on positions, not on data) can be reused across every token of a generation.
- Export the graph and read it: what a node is, what its inputs are, and how the topology validates.

## Lessons

- [ ] 1. The imperative forward, and the four things it makes impossible
- [ ] 2. Build and execute as two passes: purity as a testable property, not a style preference
- [ ] 3. Topology as a function of parameters: why positions and ids are data, never structure
- [ ] 4. Graph identity and reuse: what has to be equal for the cached graph to be valid (payoff: the microseconds that make decode usable)
- [ ] 5. Inspecting a graph: a textual dump, a JSON dump, and the live visualizer (payoff: seeing the topology is how a shape bug becomes obvious)

## What moved out of the forward pass

| Concern | Where it lived imperatively | Where it lives in a graph | What it now enables |
|---|---|---|---|
| device choice |  |  |  |
| buffer lifetime |  |  |  |
| fusion |  |  |  |
| reuse across steps |  |  |  |
| inspection |  |  |  |

## Project

**Build step S14 · build, do not compute.** The test that keeps you honest: build the graph twice with the same shapes and prove the two structures are equal, with no allocation and no kernel call in between. If building touches memory, the design has already drifted.

Start with a two-layer toy model rather than a real one. The graph machinery is easier to see when the topology is short enough to draw.

## Deliverable

- A graph builder for a toy transformer with a topologically valid node list
- A structural-equality test proving build is pure and deterministic
- A dump format you can read, plus one graph printed and annotated

## How you know you passed

- Two builds of the same shapes produce identical graphs, and a shape change produces a different one.
- No allocation or kernel call happens during build, and you can show the test that would catch one.
- You can point at the exact node where the last-token logits come out, and why the tail rows are trimmed.

## Reading

- code: `src/graph/mod.rs`, `src/graph/builder.rs`, `src/graph/params.rs` at the book's [pinned baseline](../appendix/baseline.md)
- decisions: ADR-0001 (inference runs through one declarative compute graph), ADR-0002 (topology is a function of graph params)
- docs: `docs/COMPUTE-GRAPH-DESIGN.md`, `crates/minfer-cli/src/server/viz.rs`, `docs/inference_e2e_walkthrough/05-graph-builder-ir.md`
- [tvm] Chen et al., *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning* (2018) - arXiv:1802.04799
- [scaling-inference] Pope et al., *Efficiently Scaling Transformer Inference* (2022) - arXiv:2211.05102

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
