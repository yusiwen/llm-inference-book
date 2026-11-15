# Chapter 15 · Ops, Nodes, and Fusion

> - **Compute tier**: ✅ any laptop
> - **Status**: draft
> - **Build step S15**: An op vocabulary with payloads, one pattern rewritten in place under a capability guard, and the measurement that says what the rewrite is worth on your machine - which is not what the byte count says.
> - **Rule of this book**: This chapter is the vocabulary and the rewrite. Which backend runs a node is Chapter 16; the memory consequences of fusion are Chapter 17.

## First, the gist

Chapter 14 ended with a program as a data structure: a vector of nodes, each an op plus the ids of its
inputs. This chapter is about the op itself - what it carries, what it promises, and what a compiler is
allowed to do once ops are first-class values.

Two things follow from making the op a value. First, the vocabulary can be *larger than the model*: an op
exists so a backend can refuse it, and the doc comment on the variant says what would ever emit it. Second,
patterns become matchable: a multiply fed by a sigmoid-multiply is the gated activation every modern
transformer uses, and if you can see that shape in a data structure you can rewrite it into one node that
reads two buffers and writes one.

The rewrite is where this chapter stops agreeing with the textbook. Fusing three nodes into one removes
40% of the bytes at this model's FFN width - and on the machine this book measures, it makes the kernel
**1.06x faster**, not 1.67x, because the kernel was never waiting on memory. The same rewrite on an
activation without a transcendental in it is 1.64x faster. Both numbers are in this chapter, measured the
same way, and the difference between them is the whole of lesson 5.

## What you'll be able to do

- Design an op payload that is comparable, serializable, and complete enough to re-execute the graph.
- Say why a vocabulary contains variants no supported model emits, and what refuses them.
- Implement a pattern match with a capability guard, including the two-phase commit that keeps the rewrite
  from corrupting the nodes it has not visited yet.
- Measure whether a fusion is worth anything **on your machine** before you claim it is, and explain a
  1.06x where the arithmetic promised 1.67x.

## Lessons

- [ ] 1. Ops and payloads: what a node must carry to be executable twice
- [ ] 2. Node metadata: what the backend must answer before a node may run
- [ ] 3. Pattern matching on a graph, and the guarded rewrite
- [ ] 4. The gated activation: three nodes become one, and the orphan that stays behind
- [ ] 5. When fusion loses: the measurement that decides it, and the three rewrites this engine refused

### Lesson 1 · Ops and payloads: what a node must carry to be executable twice

An op is an enum, and the payload is the part that makes two nodes with the same op *different*. Read the
vocabulary this engine settled on, with the doc comments that explain the variants it does not emit:

<!-- code: src/graph/ops.rs::Op@88ffa5d -->
```rust
/// Operator type. Implements full `PartialEq` (payloads included) so debug
/// builds can verify graph-rebuild structural identity; the production graph
/// reuse decision is params-only (see docs/COMPUTE-GRAPH-DESIGN.md §6).
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// Leaf input node (token ids, positions, KV idx, ...). Filled externally
    /// each step via `GraphAllocator::fill_input`; never part of the topology.
    Input,

    // ---- element-wise ----
    Add,
    Mul,
    /// Part of the full op vocabulary (ggml parity) so a backend can refuse it;
    /// no supported architecture emits a scale node. What would construct it: a
    /// builder that scales an activation in place instead of folding the scale
    /// into the producing op (llama.cpp's `ggml_scale`), for a temperature/1.0
    /// style rescaled graph.
    #[cfg_attr(not(test), allow(dead_code))]
    Scale(f32),
    Silu,

    // ---- reduction ----
    /// Softmax is part of the op vocabulary; the attention kernels fuse the
    /// softmax internally, so no standalone softmax node is emitted. What would
    /// construct it: an attention or sampling graph that exposes the normalized
    /// probabilities as a node output instead of fusing them into the kernel.
    #[cfg_attr(not(test), allow(dead_code))]
    Softmax {
        dim: usize,
    },

    // ---- normalization ----
    RmsNorm {
        eps: f32,
    },
    /// Per-head RMSNorm (Qwen3 `attn_q_norm`/`attn_k_norm`): the input is a
    /// flat token-major buffer `[nt * nh * hd]`; each contiguous `hd`-wide row
    /// (`t*nh + h`) is RMS-normalized with a weight of length `hd`. The buffer
    /// layout makes this a contiguous `[nt*nh, hd]` matrix, so execution reuses
    /// the RMSNorm kernels with `d = hd`, `n = nt*nh`.
    QkNorm {
        hd: usize,
        nh: usize,
        eps: f32,
    },

    // ---- linear algebra ----
    MatMul {
        transpose_b: bool,
    },

    // ---- indexing ----
    GetRows,

    // ---- positional encoding ----
    RoPE {
        style: RopeStyle,
    },

    // ---- attention ----
    /// Attention. `explicit_span` is set when the batch covers more than one
    /// sequence: `positions` alone cannot bound a query's window then, so a
    /// backend that has not been ported to the explicit span input must refuse
    /// the op (`Backend::supports_attn_span`) instead of deriving a wrong bound.
    Attn {
        mode: AttnMode,
        explicit_span: bool,
    },
```
<!-- /code -->

Three things in that type are worth more than the list.

**The vocabulary is wider than the model.** `Scale` and `Softmax` are in the enum with a doc comment
naming the builder that would emit them (a rescaled graph; an attention that exposes its probabilities as a
node). Neither is emitted by any supported architecture - and both are there so a backend has something to
*refuse*, which is a stronger statement than "the case cannot happen". The measurement below says the model
uses 11 of the 23 variants.

**The payload is part of the identity.** `PartialEq` is derived over the payload, which the type's own doc
comment states as a debug-build property: two graphs built from the same params must compare equal including
every `eps`, every `layer`, every `explicit_span` flag. Chapter 14's reuse decision is *params-only* - it
does not compare nodes at all - so this equality exists to catch the case where the params agree and the
topology does not.

**Some payloads are contracts, not parameters.** `RmsNorm { eps }` is arithmetic; `Attn { mode,
explicit_span }` is a promise about which input a backend must read, and the next lesson is about what
happens when a backend cannot keep it.

<!-- facts: fs-vocab -->
<!-- /facts -->

23 variants, 15 of them carrying a payload, and one fusion target:

<!-- code: src/graph/ops.rs::FusedOp@88ffa5d -->
```rust
/// Fused-op capability tag: drives the fusion pass (Phase 4) — a fusion is only
/// applied when the target backend reports `supports_fused(FusedOp)`.
///
/// `SwiGLU` is the only fusion produced. The plan also sketched `BiasRope`
/// (removed: no backend ever claimed the capability — bias+rope is covered by
/// the build-time fused decode nodes) and `BatchMatMul` (deferred together with
/// its `Op` variant: the single-output IR cannot express it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedOp {
    SwiGLU, // silu(gate) * up
}
```
<!-- /code -->

The doc comment on `FusedOp` is the honest part of this chapter's design record. The plan sketched two more
fusion tags - `BiasRope`, removed because no backend ever claimed the capability, and `BatchMatMul`, deferred
because the single-output IR cannot express a multi-output node. What is left is one tag and one rule, and
lesson 5 is about why the tag list is short rather than about why it should have been longer.

### Lesson 2 · Node metadata: what the backend must answer before a node may run

A node carries more than an op. It carries a shape, a dtype, a backend assignment, and - the part this lesson
is about - a set of questions a backend has to answer about it. Those questions are the trait:

<!-- code: src/graph/backend.rs::Backend@88ffa5d -->
```rust
pub trait Backend: Send + Sync {
    /// Op support by (op, dtype). `supports_fused` gates the fusion pass
    /// (Phase 4) so fused IR nodes are only produced when a kernel exists.
    fn supports_op(&self, op: &Op, dtype: DType) -> bool;
    fn supports_fused(&self, fused: &FusedOp) -> bool;

    /// Whether this backend bounds attention from the explicit `attn_span` input
    /// (E1) instead of deriving the causal window from `positions`.
    ///
    /// The default is `false` on purpose: a backend that has not been ported
    /// must never receive a multi-sequence attention node, and
    /// `GraphAllocator::supports` reads this instead of the backend's own
    /// source deciding (Metal is the unported one today — Phase G).
    fn supports_attn_span(&self) -> bool {
        false
    }
```
<!-- /code -->

`supports_op(op, dtype)` is the plain one: a backend either has a kernel for this op at this dtype or it does
not, and the answer is per-dtype because the same op may exist for f32 and not for f16. `supports_fused` is
the guard the previous lesson's `FusedOp` exists for - it is what makes the fusion pass's rewrite *safe*
rather than optimistic, because a fusion is only applied where a kernel backs it.

The third method deserves its doc comment quoted twice, because it is the pattern the other two follow. The
default is `false`, deliberately: *"a backend that has not been ported must never receive a multi-sequence
attention node"*. A capability is not a promise you make in a comment; it is a method whose default answer is
no, so that forgetting to port a backend produces a refusal at assignment time rather than a wrong answer at
step 300.

That is also why the CPU backend's answer matters for this chapter: it claims the `SwiGLU` fusion,

<!-- code: src/graph/cpu_backend.rs::supports_fused@88ffa5d -->
```rust
/// F4: the fusion-pass capability, as a free function (see [`supports_op`]).
pub fn supports_fused(fused: &FusedOp) -> bool {
    // The SwiGLU rewrite IS applied to CPU nodes (CPU is first in the
    // fusion pass's backend list); `Op::SwiGLU` below executes it as a
    // single pass. bias+rope and batch-matmul are not fused (batch QKV
    // quantize-sharing is a Phase 5+ win).
    matches!(fused, FusedOp::SwiGLU)
}
```
<!-- /code -->

which is the sentence that decides everything measured in lessons 4 and 5. The rewrite is not a GPU
optimization in this engine. It is applied to CPU nodes too, and `Op::SwiGLU` on CPU executes as a single
pass - so the graph this book has been measuring since Chapter 14 was already fused, and the CPU is where its
payoff can be measured at all.

### Lesson 3 · Pattern matching on a graph, and the guarded rewrite

The pass is one method and one rule, and reading it is the best way to see how much of a compiler fits in a
hundred lines:

<!-- code: src/graph/fusion.rs::FusionPass::fuse_swiglu@88ffa5d -->
```rust
    /// Pattern: `Mul(Silu(X), Y)` → `SwiGLU(X, Y)`.
    /// Gate: backend supports `FusedOp::SwiGLU`.
    fn fuse_swiglu(
        &self,
        graph: &mut ComputeGraph,
        backends: &[&dyn Backend],
        backend_of: &dyn Fn(&ComputeGraph, usize) -> Option<usize>,
    ) -> usize {
        let n = graph.n_nodes();
        let mut replaced = 0usize;
        let mut new_ops: Vec<Option<Op>> = vec![None; n];
        for id in 0..n {
            if !matches!(graph.node(id).op, Op::Mul) {
                continue;
            }
            let mul = graph.node(id);
            if mul.src.len() != 2 {
                continue;
            }
            let (s, y) = (mul.src[0], mul.src[1]);
            let is_silu = |x: usize| matches!(graph.node(x).op, Op::Silu);
            let (silu_in, gate, up) = if is_silu(s) {
                (graph.node(s).src[0], graph.node(s).src[0], y)
            } else if is_silu(y) {
                (graph.node(y).src[0], graph.node(y).src[0], s)
            } else {
                continue;
            };
            let _ = silu_in;
            // gate the fusion on the mul node's backend capability
            let ok = match backend_of(graph, id) {
                Some(bi) => backends[bi].supports_fused(&FusedOp::SwiGLU),
                None => false,
            };
            if !ok {
                continue;
            }
            // replace Mul with SwiGLU(gate, up)
            new_ops[id] = Some(Op::SwiGLU);
            if let Some(node) = graph.nodes.get_mut(id) {
                node.src = vec![gate, up];
            }
            replaced += 1;
        }
        for (id, op) in new_ops.into_iter().enumerate() {
            if let Some(op) = op {
                graph.nodes[id].op = op;
            }
        }
        replaced
    }
```
<!-- /code -->

Four details in those fifty lines are the ones that separate a working rewrite from a plausible one.

**The match is on nodes, not on text.** It looks for `Op::Mul` with exactly two inputs, then asks whether
either input is an `Op::Silu`. Matching both sides rather than the left one is the difference between
handling `silu(x) * y` and `y * silu(x)`, and the code pays for it with a two-branch `if` instead of a
pattern in a single arm.

**The guard is the backend, not the device.** `backend_of(graph, id)` gives the node's assigned backend index,
and `backends[bi].supports_fused(&FusedOp::SwiGLU)` decides. A fusion pass that ran before assignment could
not ask this question; that is why Chapter 14's order - build, assign, *then* fuse - is load-bearing and not
a matter of taste.

**The rewrite is two-phase on purpose.** `new_ops` is collected in one pass and applied in a second, and the
reason is in the loop: rewriting `graph.nodes[id].op` while scanning forward would change what later
iterations match. Two nodes can share a `Mul` id only once, but a `Mul` whose input is itself a `Mul` becomes
visible or invisible depending on visit order if you mutate in place - so the pass separates "decide" from
"apply", which is the same discipline as Chapter 14's build-then-execute.

**What it does not do: delete the node.** The rewrite changes the `Mul`'s op and rewires its inputs to skip
the `Silu` - and then stops, because deleting a node from the vector would renumber every id after it, and
the ids are the edges. The consequences are lesson 4.

### Lesson 4 · The gated activation: three nodes become one, and the orphan that stays behind

So the rewrite happened. Whether it happened is not a matter of reading the pass; it is a matter of reading
the graph the engine actually executed, which its per-node trace hands over:

<!-- facts: fs-structure -->
<!-- /facts -->

Read the `mul` row and the `swiglu` row together. There is no `mul` node in either graph type and there are
24 `swiglu` nodes - one per transformer block - so every multiply that fed the output projection was rewritten.
Then read the `silu` row: **24 nodes are still there**, and the last row says how many of them the trace
recorded as executed: none.

That is the mechanism, and it lives in the scheduler rather than in the fusion pass:

<!-- code: src/graph/scheduler.rs::node_buffer@88ffa5d -->
```rust
                // dead nodes (no consumers, not outputs) get no buffer — the
                // fusion pass can orphan them (e.g. silu folded into SwiGLU);
                // they are skipped, not executed
                let Some(br) = alloc.node_buffer(id) else {
                    continue;
                };
```
<!-- /code -->

No dead-code pass exists. A node with no consumers and not in the graph's outputs simply gets no buffer from
the allocator, and the scheduler's `let Some(..) else { continue }` skips whatever has no buffer. The graph
keeps its 443 nodes and its ids; the execution runs 371 of them. Two of the three silent groups in that table
are there for a different reason - `kvcache_store` and `kvcache_load` run on a dedicated path that never
enters the traced node loop - which is worth knowing before you read a missing row as a missing computation.

<figure class="book-figure">{{#include ../figures/15-fusion.svg}}</figure>

*Figure 15.1 - the rewrite and its orphan. Before: the activation writes a temporary and the multiply reads it back, five passes over the row. After: one node reads both operands and writes once, three passes - and the node that used to feed the multiply is still in the graph, with no consumer, no buffer, and no execution.*

The byte count is the reason to do this at all, and it is arithmetic rather than measurement:

<!-- facts: fs-bill -->
<!-- /facts -->

97,280 bytes per row become 58,368, which is 40% of the traffic for one FFN activation, 933,888 bytes saved
per token across 24 layers. At 4864 floats the row is 19 kB, so a single token's worth of this operation sits
in L1 and bytes are not what bounds it - which is the bridge to the last lesson, and the reason the next
number is smaller than the 1.67x the bill implies.

### Lesson 5 · When fusion loses: the measurement that decides it

The fusion's claim is that three nodes became one pass. The bill says 40% fewer bytes. What it is *worth*
depends on what the kernel was waiting for, so the instrument measures the same rewrite twice: once on `silu`,
which needs an exponential per element, and once on `relu`, which needs a compare. Both are activations with
the same signature and the same three-node-to-one-node shape.

<!-- facts: fs-kernel -->
<!-- /facts -->

At 4096 rows the fused form is **1.06x** faster on silu and **1.64x** faster on relu. Same rewrite, same
40% byte reduction, an order of magnitude apart in payoff. The table also carries the answer to why: the
achieved bandwidth. silu moves 10 to 17 GB/s; relu moves 115 to 132 GB/s; and this machine's single-core
streaming copy rate is in the bill table above, at 126.3 GB/s. relu sits *on* that line - it is waiting on
memory, and removing 40% of the memory work removes about 40% of the time. silu sits at an eighth of it, which
means the exponential, not the traffic, is the limit; the fusion removes work the kernel was not doing.

<figure class="book-figure">{{#include ../figures/15-fusion-cost.svg}}</figure>

*Figure 15.2 - measured bandwidth against this machine's single-core streaming rate. The two forms of the same rewrite differ by an order of magnitude in what they achieve, and the gap between silu and the purple line is the gap between a kernel waiting on the ALU and one waiting on memory.*

This is the roofline argument [roofline] applied to a fusion: a rewrite only pays if it removes the resource
that is actually saturated. It is also why an engine keeps the unfused path alive - and why this chapter's
instrument had to be written twice.

Two things the measurement changed about what this chapter says. The first is the title of the table in the
stub this page replaced: it promised a 3-to-1 node reduction with a bandwidth saving, and the engine delivers
a 3-node-to-3-node rewrite with a 5-pass-to-3-pass saving and a downstream skip. The second is smaller and
more embarrassing: the first version of `scripts/fuse.c` selected its activation through a function pointer,
which is a call per element, and reported both forms twice as slow. The numbers above are from the version
that generates both kernels by macro. An instrument that measures its own dispatch is the exact failure this
book keeps warning about, so it is written into the file's header comment.

The engine's own record says the same thing more quietly. Three rewrites were considered and only one is
applied:

| Pattern | Decision | Reason, in the engine's words |
|---|---|---|
| gated activation `Mul(Silu(x), y)` | **applied**, on CPU, Metal and CUDA | `SwiGLU` is the only rule; the capability is claimed by all three backends |
| `RoPE(Add(x, b))` → `FusedBiasRope` | **removed** | no backend ever claimed the capability; bias+rope is covered by the build-time fused decode nodes instead |
| `BatchMatMul` | **deferred** | the single-output IR cannot express a multi-output node |
| `FusedQKV`, `QkvBiasRopeStore`, `FusedFFN`, `FusedQkvNorm` | not fusion-pass targets | they are build-time fused nodes in the builder, and the Metal ones refuse `n_tokens != 1` |

That last row is a cost worth naming: a fused node is not free, it is a *sharper* node. The Metal fused
builders refuse any batch except one, which is why the engine's own tests include four refusals next to the two
fusion tests - a rewrite that widens the set of shapes a graph can express is a rewrite that narrows what a
kernel accepts. Here is one of those refusals, in the Metal backend's own execute path:

<!-- code: src/graph/metal_backend.rs::Op::FusedFFN@88ffa5d -->
```rust
            Op::FusedFFN => {
                // #39: the fused FFN kernel is decode-only, and `debug_assert!`
                // is compiled out in release. Shape validation is cheap and does
                // not depend on the weights, so it is the first thing checked
                // (kernel-invariant violations return Err, never assume).
                let nt = node.out_shape[1];
                if nt != 1 {
                    return Err(format!(
                        "metal: {}: FusedFFN is decode (nt==1) only, got nt={nt}",
                        node.name
                    ));
                }
                let meta = match &node.meta {
                    NodeMeta::FusedFfn(m) => m,
                    other => return Err(format!("fused_ffn node missing FusedFfnMeta: {other:?}")),
                };
                let (wb, w_off) = self
                    .state
                    .weight_buf(&meta.gu_weight)
                    .ok_or_else(|| format!("gate+up weight '{}' not on GPU", meta.gu_weight))?;
```
<!-- /code -->

The comment is the part to keep: `debug_assert!` is compiled out in release, so a shape check that only exists
in debug is a check the release binary does not have. This arm validates before it touches a weight, and
returns an error instead of assuming - which is what "a fused node is a sharper contract" costs the kernel
author: one more arm that can refuse.

## Project

**Build step S15 · rewrite with proof.** Fuse one pattern, then run both graphs on the same inputs and require
identical output. This chapter's instrument asserts exactly that, and the witness behind it recomputes the
same values in f64:

<!-- facts: fs-witness -->
<!-- /facts -->

A fusion that changes the answer by a few bits is admissible only if your identity gate says so, and you must
be the one who decided that. Here the two forms are byte-identical, because the fused expression multiplies
the same rounded f32 activation - prove the same thing for yours or state the tolerance you accepted.

Then measure the payoff the way lesson 5 does, not the way the byte count does. Write the unfused path as a
second kernel, take the median of nine runs, and compare against your machine's streaming bandwidth. If your
fused kernel is at a fifth of that line, the fusion is not your problem and no amount of rewriting will move
it.

## Deliverable

- An op enum with at least one payload-carrying variant, deriving full-payload equality
- A capability query per backend, defaulting to `false`, and at least one op that a backend refuses
- A guarded rewrite with a two-phase apply, plus a test for the refused case
- The measured speedup of your rewrite on two activations, with the bandwidth each one achieved

## How you know you passed

- Your fused graph produces bit-identical output to the unfused one, on the same inputs.
- Your rewrite skips a node instead of deleting it, and your execution path skips whatever has no consumer.
- You can state what your rewrite is worth on your machine, in one number, and say what would have made it
  bigger.
- You can name a rewrite you decided *not* to do, and why.

## Reading

- code: `src/graph/ops.rs`, `src/graph/fusion.rs`, `src/graph/backend.rs`, `src/graph/scheduler.rs` at the book's [pinned baseline](../appendix/baseline.md)
- decisions: `docs/COMPUTE-GRAPH-DESIGN.md` §5 (the IR and fusion), §5.4 (why `BatchMatMul` needs multi-output)
- tests: `src/graph/fusion/tests.rs` - two tests, one per branch of the guard; run on this machine for this chapter:

<!-- facts: fs-tests -->
<!-- /facts -->

- [tvm] Chen et al., *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning* (2018) - arXiv:1802.04799
- [roofline] Williams, Waterman and Patterson, *Roofline: an insightful visual performance model for multicore architectures* (CACM, 2009)
- [scaling-inference] Pope et al., *Efficiently Scaling Transformer Inference* (2022) - arXiv:2211.05102

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)

<!-- facts: fusion.provenance -->
*`scripts/fuse.py`; runs the engine's CLI with its per-node trace on (the fused graph it actually executed), reads the op vocabulary out of the pinned source, runs the engine's own fusion tests, and compiles and runs scripts/fuse.c, which measures the same 3-to-1 rewrite on two activations at four row counts; scripts/fuse-check.py recomputes the C's sampled outputs in f64; on **{'platform': 'macOS-27.2.1-arm64-arm-64bit-Mach-O', 'machine': 'arm64', 'python': '3.15.2'}**; node counts and the set of executed node ids from one trace; timings are the median of nine runs of the same kernel; output equality is compared byte for byte, and the numpy witness agrees within float32 rounding. Command: see [the provenance index](../appendix/provenance.md#fusion). Raw numbers: `facts/fusion.json`.*
<!-- /facts -->
