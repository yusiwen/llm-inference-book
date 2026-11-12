# Chapter 15 · Ops, Nodes, and Fusion

> **Compute tier**: ✅ any laptop | **Status**: stub
> **Build step S15**: An op vocabulary with payloads, node metadata, and one fusion rule that rewrites a three-node pattern into one node.
> **Rule of this book**: This chapter defines what an op is and how patterns are folded. Which backend runs a node is Chapter 16; the memory consequences of fusion are Chapter 17.

## First, the gist

A graph is only as precise as its vocabulary. Each op has a payload - shapes, strides, flags - and the graph's correctness depends on that payload comparing equal when it should and unequal when it should not. Once ops are explicit, patterns become visible: a multiply feeding a sigmoid feeding a multiply is the gated activation every modern transformer uses, and it can run as one pass over memory instead of three. **Fusion** is that rewrite, and it is only safe because the graph is a data structure you can match on.

## What you'll be able to do

- Design an op payload that is comparable, serializable and complete enough to re-execute the graph faithfully.
- Implement a pattern-matching fusion and prove the fused graph computes what the unfused one did.
- Say when fusion is the wrong call: the cases where a fused kernel is slower than two plain ones.

## Lessons

- [ ] 1. Ops and payloads: what a node must carry to be executable twice
- [ ] 2. Node metadata: shapes, dtypes, and the annotations a scheduler needs
- [ ] 3. Pattern matching on a graph, and the guarded rewrite
- [ ] 4. The gated activation fusion: three nodes become one pass (payoff: the bandwidth it saves, computed)
- [ ] 5. When fusion loses: register pressure, occupancy, and the measurement that decides it

## Fusion candidates

| Pattern | Nodes before | Nodes after | Bandwidth saved (computed) | Measured effect (yours) |
|---|---|---|---|---|
| gated activation |  |  |  |  |
| residual + norm |  |  |  |  |
| query/key/value projection |  |  |  |  |
| bias + rope + store |  |  |  |  |

## Project

**Build step S15 · rewrite with proof.** Fuse one pattern, then run both graphs on the same inputs and require identical output. A fusion that changes the answer by a few bits is admissible only if your identity gate says so, and you must be the one who decided that.

Keep the unfused path alive behind a flag: it is the oracle, and it is also the escape hatch when a fused kernel misbehaves on a new device.

## Deliverable

- A comparable op payload with a structural-equality test on the graph
- One guarded fusion with an unfused oracle and a flag to switch it off
- A measured before/after for the fused pattern on your machine

## How you know you passed

- The fused and unfused graphs agree within a bound you chose and justified.
- Disabling the fusion via the flag produces the unfused graph, and a test asserts it.
- You have at least one pattern you decided **not** to fuse, with the number that decided it.

## Reading

- code: `src/graph/ops.rs`, `src/graph/fusion.rs`, `src/graph/op_matrix.rs` at commit `88ffa5d`
- docs: `docs/inference_e2e_walkthrough/06-assign-fusion.md`
- docs: `docs/COMPUTE-GRAPH-DESIGN.md`
- [tvm] Chen et al., *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning* (2018) - arXiv:1802.04799

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
