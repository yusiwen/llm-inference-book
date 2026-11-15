# Chapter 14 · From Imperative Loop to Declarative Graph

> - **Compute tier**: ✅ any laptop
> - **Status**: draft
> - **Build step S14**: Your forward pass rebuilt as a pure data structure: nodes, inputs, outputs, built without touching memory, then executed by a separate walk.
> - **Rule of this book**: This chapter is the graph's shape and why it exists. The op vocabulary and fusion are Chapter 15; who executes a node is Chapter 16; where buffers come from is Chapter 17.

## First, the gist

The obvious way to write a forward pass is a function: for each layer, compute the norm, call the matmul, add
the residual, call attention, go on. It works. It also traps you, and the trap is not about speed at first - it is
about what you can no longer *say* about the computation. There is no object that represents "this model's forward
pass for this shape", so there is nothing to inspect, nothing to compare between two runs, nothing to hand to
another component, and nothing to cache.

The **compute graph** replaces the function with data. Building the graph allocates no buffers and computes
nothing; it appends nodes to a vector - an op, the ids of its inputs, an output shape, a dtype. A separate pass
then decides which device runs each node, assigns every node a buffer, and executes the nodes in order. Every
question the imperative function could not answer becomes a property of a data structure: what are the inputs, what
is the output, which nodes read the same buffer, which nodes could be fused, and - the one that pays for the whole
redesign - is this the same graph as last step, so can I skip building it?

This chapter measures that structure on a real model: **443 nodes** for a prefill, **440** for a decode, built
once each and reused for every token after. Every number below was read out of the engine's own per-node trace,
the same file its visualizer draws.

## What you'll be able to do

- Rebuild a forward pass as graph construction, keeping build and execute strictly separate.
- Explain why a graph whose topology depends only on shapes (not on positions, not on data) can be reused across
  every token of a generation.
- Export the graph and read it: what a node is, what its inputs are, and how the topology validates.
- Say which of the four things the imperative loop made impossible, and point at the row of this chapter's own
  data that shows the graph making it possible.

## Lessons

- [ ] 1. The imperative forward, and the four things it makes impossible
- [ ] 2. Build and execute as two passes: purity as a testable property, not a style preference
- [ ] 3. Topology as a function of parameters: why positions and ids are data, never structure
- [ ] 4. Graph identity and reuse: what has to be equal for the cached graph to be valid
- [ ] 5. Inspecting a graph: a textual dump, a JSON dump, and the live visualizer

### Lesson 1 · The imperative forward, and the four things it makes impossible

An honest note first: this engine no longer contains the imperative forward it replaced. The CLI still accepts
`--graph`, and the comment where it does says why - *"`--graph` is accepted for compatibility; the graph path is
now the default forward"*. The before is deleted, so this chapter cannot show you a before/after diff of the same
model. What it can do is show the after as data, and name the four questions the before could not answer:

| Question | Why the imperative loop cannot answer it | What answers it once the pass is a graph |
|---|---|---|
| Which device runs this op? | the call site is the decision; there is no op object to ask | a per-node `backend` field, assigned by a pass |
| Who owns this buffer, and for how long? | buffers are locals in a call stack that has already returned | liveness over node ids; owners are nodes |
| Could these two ops be one op? | they are two statements in a function body | two nodes with a shared input and a rewriter |
| Is this the same computation as last token? | there is nothing to compare; the function just runs | structural equality over a params struct |

The engine's answer to all four is one type:

<!-- code: src/graph/mod.rs::CNode@88ffa5d -->
```rust
/// Single compute node.
#[derive(Debug, Clone)]
pub struct CNode {
    pub id: NodeId,
    pub name: String,
    pub op: Op,
    pub src: Vec<NodeId>,
    pub out_shape: [usize; 4],
    pub out_dtype: DType,
    /// Backend assigned by the scheduler (Phase 4); None = undecided.
    pub backend: Option<Backend>,
    pub meta: NodeMeta,
    /// D1: `Some` when this node does not own its output — it is a window into
    /// another node's buffer. Set by `GraphBuilder::node` for the view-like ops
    /// (`View`/`Reshape`/`Permute`); the allocator maps it onto the parent's
    /// buffer and keeps the parent alive instead of copying.
    pub view: Option<ViewAlias>,
    /// E5: the transformer block this node belongs to (`None` for everything
    /// outside a block: the embedding, the final norm, `lm_head`, and the
    /// standalone test graphs). Stamped by `GraphBuilder::set_layer`, which the
    /// model builders set once per block; the offload policy reads it to keep a
    /// non-offloaded block's nodes off the device.
    pub layer: Option<usize>,
}
```
<!-- /code -->

Read the fields as a list of the questions above, answered in advance. `op` is the operation, `src` is the
computation's edges, `out_shape`/`out_dtype` are known before anything runs, `backend` is `Option<Backend>` because
at build time nobody has decided yet, and `view` marks the nodes that will not own a buffer at all. The graph
itself is the vector, in an order the doc comment states as a promise:

<!-- code: src/graph/mod.rs::ComputeGraph@88ffa5d -->
```rust
/// Compute graph: topologically ordered node sequence + input/output sets.
#[derive(Debug, Clone, Default)]
pub struct ComputeGraph {
    pub nodes: Vec<CNode>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    /// Graph identifier for reuse detection (CUDA Graph caching etc.).
    /// Populated by `GraphCache::replace_graph` (monotonic per process);
    /// a reused graph keeps its uid.
    pub uid: u64,
}
```
<!-- /code -->

"Topologically ordered node sequence" is the invariant every later pass depends on: the scheduler walks the vector
front to back, the allocator walks it for liveness, and the exporters index into it by id. If a builder ever
appended a node before one of its inputs, nothing would crash - the walk would simply read a buffer that had not
been written yet.

What that vector contains for the model this book measures:

<!-- facts: graph.gr-shape -->
| Quantity | Prefill graph | Decode graph |
|---|---|---|
| Nodes | 443 | 440 |
| Leaf inputs | 6 | 5 |
| Output node | 442 | 439 |
| Steps executed | 1 | 7 |
| Backends assigned | cpu | cpu |

| Op | Prefill | Decode |
|---|---|---|
| `matmul` | 169 | 169 |
| `rms_norm` | 49 | 49 |
| `add` | 48 | 48 |
| `rope` | 48 | 48 |
| `attn` | 24 | 24 |
| `kvcache_load` | 24 | 24 |
| `kvcache_store` | 24 | 24 |
| `silu` | 24 | 24 |
| `swiglu` | 24 | 24 |
| `input` | 6 | 5 |
| `get_rows` | 3 | 1 |
<!-- /facts -->

Read the op counts against the model's shape, because they are arithmetic rather than design. 24 transformer blocks
with 7 matmuls each is 168, plus the output projection is 169. Two norms per block plus the final one is 49. Two
rope applications and two residual adds per block are 48 each. One attention, one gate, one silu and one KV store
and load per block are 24 each. The graph is the model's parameter count written as a program - and the fact that
you can *check* that arithmetic against the node list is the first thing the graph buys you.

<figure class="book-figure">{{#include ../figures/14-two-passes.svg}}</figure>

*Figure 14.1 - the op histogram of the two graph types, measured. Every block contributes the same seven matmuls, two norms, two ropes, two adds, and one each of attention, silu, swiglu, KV store and KV load. The three bars on the right are the entire difference between the two graphs.*

### Lesson 2 · Build and execute as two passes: purity as a testable property

Building appends nodes. That is the whole of it, and the function that does it is worth reading because of what it
does *not* do - there is no allocation, no kernel, no branching on data:

<!-- code: src/graph/builder.rs::GraphBuilder::node@88ffa5d -->
```rust
    /// Create an operator node; returns its id.
    pub fn node(
        &mut self,
        name: &str,
        op: Op,
        src: &[NodeId],
        out_shape: [usize; 4],
        out_dtype: DType,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.graph.nodes.len();
        // D1: the view-like ops do not own their output — they are windows into
        // their single source's buffer. Recorded here (the one construction
        // point, so hand-built graphs get it too) and honoured by the allocator,
        // which maps the node onto the parent's buffer and extends the parent's
        // liveness instead of allocating a copy.
        let view = match (&op, src.len()) {
            (Op::View { offset, .. }, 1) => Some(ViewAlias {
                src: src[0],
                offset: *offset,
            }),
            (Op::Reshape { .. }, 1) | (Op::Permute { .. }, 1) => Some(ViewAlias {
                src: src[0],
                offset: 0,
            }),
            _ => None,
        };
        self.graph.nodes.push(CNode {
            id,
            name: name.to_string(),
            op,
            src: src.to_vec(),
            out_shape,
            out_dtype,
            backend: None,
            meta,
            view,
            // E5: the block this node belongs to (set by the model builders per block).
            layer: self.cur_layer,
        });
        id
    }

    /// Leaf input node — filled externally every step; never part of the
    /// topology (so `n_past`/positions changes never force a graph rebuild).
    pub fn input(&mut self, name: &str, shape: [usize; 4], dtype: DType) -> NodeId {
        let id = self.node(name, Op::Input, &[], shape, dtype, NodeMeta::None);
        self.graph.inputs.push(id);
        id
    }
```
<!-- /code -->

Three things in those forty lines matter more than they look.

**The id is the vector length.** `let id = self.graph.nodes.len();` - a node's identity is where it happens to be
appended. That is why the topological promise is load-bearing: an id is a position, and every edge in the graph is
a position too, so "input 18" means "the node appended before me that produced what I need".

**Views are decided at construction.** The `view` field is set by a `match` on the op, at the single point where
nodes are created, so *no* allocator can forget that a reshape does not own its output. A later chapter allocates
buffers; this line is why it can map a reshape onto its parent's buffer instead of copying a tensor.

**Inputs are leaves, not topology.** The `input()` method is the one place where a node is created with no edges,
and its doc comment states the invariant that makes reuse possible at all: *"never part of the topology (so
`n_past`/positions changes never force a graph rebuild)"*. The KV position of the next token changes on every step.
The graph does not.

The graph's structure, read back from the edges the builder wrote:

<!-- facts: graph.gr-dag -->
| Quantity | Value |
|---|---|
| Longest chain of nodes | 268 |
| Nodes at that depth | 1 |
| Most inputs one node takes | 4 |
| Most consumers one node feeds | 72 |
| Leaf inputs | 6 |

Nodes by input count - 0 input(s): 30, 1 input(s): 243, 2 input(s): 122, 3 input(s): 24, 4 input(s): 24.
<!-- /facts -->

A chain 268 nodes deep, in a graph of 443: the longest path is the layer stack itself, walked end to end. Thirty
nodes take no input at all, and they are exactly the six leaf inputs plus the 24 KV loads - the loads read a
persistent region through a meta field rather than an edge, because the cache is not a tensor produced by this
graph. And one node feeds 72 others: the embedding output, which every block's first norm reads. That single
number is the clearest statement of what a graph gives a scheduler - the fan-out is *known*, before execution,
which is what lets a buffer for it be sized once and kept alive across the whole run.

**Purity, as a test.** "Build is pure" is a claim you can check cheaply, and the project's check is the one this
chapter asks you to write: build twice with the same params, compare the two structures, and assert that no
allocation or kernel call happened in between. The equality is structural - the same ops, the same names, the same
edges, the same shapes - and it has to hold while the *data* differs, which is the next lesson.

### Lesson 3 · Topology as a function of parameters: positions and ids are data

The claim that makes reuse possible is one sentence long, and the engine keeps it in the type that the cache
compares:

<!-- code: src/graph/params.rs::GraphParams@88ffa5d -->
```rust
/// Reuse-relevant graph parameters (see module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphParams {
    pub n_tokens: usize,
    /// Number of output (tail) rows: the last layer's FFN + lm_head run on the
    /// last `n_out` rows only (llama `inp_out_ids`). Part of the topology —
    /// a change forces a rebuild.
    pub n_out: usize,
    pub gtype: GraphType,
    pub cparams: CParams,
    /// Bumped by the model whenever weights change (LoRA switch, reload).
    pub weights_version: u64,
}
```
<!-- /code -->

Everything in that struct is a *shape decision*; nothing in it is a token, a position, or a KV index. `n_tokens` is
how wide the batch is. `n_out` is how many rows come out of the head - the "tail rows" of a prefill, one row for a
decode. `gtype` is which of the two graphs this is. `cparams` carries the choices that change the topology without
changing the shape: fusion on or off, GPU in or out, the KV cell format, the offload plan. And `weights_version`
exists so that reloading the weights invalidates a cached graph even when every shape is identical.

That the params really do determine the topology is measurable, and the measurement is the interesting part: two
prompts of different length produce **the same node list** and different tensor sizes.

<!-- facts: graph.gr-topology -->
| Quantity | Short prompt (34 tokens) | Long prompt (58 tokens) |
|---|---|---|
| Nodes | 443 | 443 |
| Node sequence (op, name, edges) identical | - | **yes** |
| Node 0 shape | `[34, 1, 1, 1]` | `[58, 1, 1, 1]` |
<!-- /facts -->

Two prompts, 34 tokens and 58 tokens, 443 nodes each, and the sequence of `(op, name, edges)` is identical node for
node - checked as a value, not eyeballed. What moved is the first dimension of every tensor: node 0 is
`[34, 1, 1, 1]` in one graph and `[58, 1, 1, 1]` in the other.

This is the sentence people misread. "The graph is the same" does not mean "the shapes are the same". It means the
*program* is the same and its inputs are wider, which is exactly the condition under which the same buffer plan,
the same backend assignment and the same fused nodes stay valid.

The other half of the lesson is that a different `gtype` is *not* a shape change - it is a different program, and
the difference is small enough to read:

<!-- facts: graph.gr-delta -->
| Op | Prefill | Decode | Difference |
|---|---|---|---|
| `get_rows` | 3 | 1 | +2 |
| `input` | 6 | 5 | +1 |

The nodes that exist only in the prefill graph: `input tail_ids`, `get_rows get_rows`, `get_rows get_rows`.
<!-- /facts -->

Three nodes: one input and two embedding lookups. The prefill graph carries a `tail_ids` input and two `get_rows`
nodes that select which rows reach the head; the decode graph carries neither, because its batch *is* one row and
there is nothing to select. Every other one of the 440 nodes is shared, in the same order, with different widths.

Which gives the acceptance criterion of this build step its sharpest form: point at the node where the last
token's logits come out - id 442 in this prefill graph, `[1, 151936]` - and say why the tail rows are trimmed.
Because the head is 151,936 wide and the answer needs one row of it, and the two `get_rows` nodes are the
difference between multiplying 34 rows and multiplying 1.

### Lesson 4 · Graph identity and reuse: the microseconds that make decode usable

A decode step changes one thing: the position. The KV cache has grown by one cell, the input token is new, the
sampler will draw from a different distribution. None of that is topology - so on step 300 of a generation, the
engine should not be building a graph.

It does not. The cache that makes that true states its invariant in its module docs - *graph topology is a
deterministic function of `GraphParams`*: equal params give an identical topology, so the graph is reused as-is and
only the input data is refreshed, and `n_past` never appears in the comparison because it is execution data. (The
doc line writes that implication with a double-arrow sign, which this book's character gate refuses, so the
sentence is paraphrased here rather than quoted.) The implementation is shorter than the sentence:

<!-- code: src/graph/cache.rs::GraphCache::try_reuse@88ffa5d -->
```rust
    pub fn try_reuse(&mut self, params: &GraphParams) -> Result<bool, String> {
        let Some(pos) = self
            .graphs
            .iter()
            .position(|(p, _)| Self::params_match(p, params))
        else {
            return Ok(false);
        };
        let (_, graph) = self.graphs.remove(pos);
        // The re-map can fail (the budget gate), and a failed switch must not silently leave
        // the allocator pointing at the old graph's buffers — the error is returned to the
        // caller, which then rebuilds.
        self.alloc.alloc_graph(&graph)?;
        self.graphs.insert(0, (params.clone(), graph));
        self.reuses += 1;
        Ok(true)
    }
```
<!-- /code -->

`try_reuse` compares **params, never the node sequence** - that is the whole trick, and it is only sound because of
Lesson 3. If the params are equal, the topology is identical *by construction*, so the cached graph is not merely
"probably fine", it is the same program. The comment about the allocator is the other half: the allocator lives
inside the cache and survives a rebuild, because the persistent KV regions *are* the cache, and a prefill→decode
transition must not lose them.

Measured, on an eight-step run:

<!-- facts: graph.gr-reuse -->
| Build | Tokens in that graph | Cost (build + assign + alloc) |
|---|---|---|
| 1 | 34 | 0.5 ms |
| 2 | 1 | 0.4 ms |

Steps executed: **8** (1 prefill + 7 decode). Graphs built: **2**.
<!-- /facts -->

Two builds for eight forward passes. The prefill graph is built once (0.5 ms), the decode graph once (0.4 ms), and
the seven decode steps after the first reuse it. Those two numbers are the ones that matter for the argument this
chapter is making: a decode step that rebuilt its graph would pay half a millisecond of bookkeeping to do a job
that takes tens of microseconds of actual arithmetic.

<figure class="book-figure">{{#include ../figures/14-graph-reuse.svg}}</figure>

*Figure 14.2 - the same run as a timeline. One prefill graph, then one decode graph, then six more steps over that same decode graph: the topology line is flat while the KV cache underneath grows by one cell per step. The two marks are the only two builds in the run.*

The reuse rule has a cost, and it is worth stating: `try_reuse` must be *conservative*. Anything that could change
the topology has to be in `GraphParams`, or a cached graph will be silently wrong. That is why `cparams` carries
the fusion flags, the KV format and the offload plan, and why the module doc for `CParams` says, of the number of
sequences in a batch, that it is deliberately absent: it is data, not topology, and carrying it "cost a rebuild
whenever the batching changed shape with the topology unchanged".

### Lesson 5 · Inspecting a graph: the dump is the design

The last thing the imperative loop could not do is show you what it was going to do. The graph can, and the export
is twenty lines because the data structure already holds everything:

<!-- code: src/graph/json.rs::ComputeGraph::export_json@88ffa5d -->
```rust
    /// Export the graph as the web-visualizer JSON document.
    /// `model` = display name (e.g. "qwen2.5-0.5b-instruct-q4_k_m"),
    /// `kind`  = "prefill" | "decode" (the graph type that produced it).
    pub fn export_json(&self, model: &str, kind: &str) -> Value {
        let nodes: Vec<Value> = self
            .nodes
            .iter()
            .map(|n| {
                json!({
                    "id": n.id,
                    "name": n.name,
                    "op": op_name(&n.op),
                    "detail": op_detail(&n.op),
                    "shape": n.out_shape,
                    "dtype": dtype_name(n.out_dtype),
                    "backend": n.backend.map(backend_name),
                    "src": n.src,
                    "meta": meta_json(&n.meta),
                })
            })
            .collect();
        json!({
            "format": "minfer-graph",
            "version": 1,
            "model": model,
            "kind": kind,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": nodes,
        })
    }
```
<!-- /code -->

Every field of a node is exported, and two of them are the ones you will actually use. `src` is the edge list, so a
reader can rebuild the DAG and ask the questions Lesson 2 asked - depth, fan-out, which node is the bottleneck
structurally. And `shape` is the tensor size, which is where shape bugs live: a `[34, 896]` where you expected
`[896, 34]` is invisible in a stack trace and obvious in a dump.

The per-node trace goes further, because it records what each node *produced*:

<!-- facts: graph.gr-trace -->
| Trace field | Value |
|---|---|
| Nodes recorded in one decode step | 368 |
| Per-node statistics | absmean, max, mean, min |
| Largest sample kept per node | 64 values |
| Output node | `439`, 151,936 values, f32 |
| Output range | min -16.84, max 24.03, mean -2.94 |

| Rank | Id | Probability | Token |
|---|---|---|---|
| 1 | 6722 | 0.9959 | `\u0120capital` |
| 2 | 3946 | 0.0014 | `\u0120official` |
| 3 | 1482 | 0.0009 | `\u0120current` |
| 4 | 18374 | 0.0002 | `\u0120Capital` |
| 5 | 3146 | 0.0002 | `\u0120country` |
<!-- /facts -->

368 of the 440 nodes are recorded in a decode step, each with four statistics and a strided sample of up to 64
values. The output node has 151,936 values, f32, ranging from -16.84 to 24.03 - and the trace also carries the
step's token and the softmax top-5 of those logits, which is how the previous chapter's sampler numbers and this
chapter's graph numbers end up in the same file.

That is the inspection story end to end: one JSON file, produced by a run, containing the program, every node's
output statistics, and the token that came out. A shape bug shows up as a wrong `shape` in the node list; a
numerical bug shows up as a `max` where you expected a `min`; an assignment bug shows up as a `backend` field you
did not expect. None of those three sentences could be written about a function.

## What moved out of the forward pass

| Concern | Where it lived imperatively | Where it lives in a graph | What it now enables |
|---|---|---|---|
| device choice | the call site: a kernel was called on a device because that is where the code called it | a per-node `backend` field, filled by `assign_backends` | the same graph can run with a different assignment, and the assignment is inspectable (measured: every node in this run is `cpu`) |
| buffer lifetime | stack locals inside the function, freed when it returned | liveness over node ids, owned by the allocator, with `view` nodes aliasing their parent | buffers can be reused across steps and across nodes, and a reshape costs no copy |
| fusion | writing the fused kernel call instead of the two calls | a pass over nodes with a shared input, gated on the assigned backend | fusion becomes an option you can A/B (on this CPU-only run the two fusion flags are no-ops: no node carries a device) |
| reuse across steps | impossible: the function is the computation | structural equality over `GraphParams`, with the graph kept in a cache | two builds in an eight-step run instead of eight |
| inspection | a stack trace, if you were unlucky | the JSON export and the per-node trace | the topology, the shapes, the stats and the token in one artifact |

## Project

**Build step S14 · build, do not compute.** The test that keeps you honest: build the graph twice with the same
shapes and prove the two structures are equal, with no allocation and no kernel call in between. If building
touches memory, the design has already drifted.

Start with a two-layer toy model rather than a real one. The graph machinery is easier to see when the topology is
short enough to draw - and the arithmetic in Lesson 1 is easier to check when the op counts are 14, not 443.

Then do the measurement that makes the design worth its cost: run your model for twenty tokens with a counter in
the builder, and print how many graphs you built. If the answer is more than two, your params struct is missing a
field or carrying one it should not.

## Deliverable

- A graph builder for a toy transformer with a topologically valid node list
- A structural-equality test proving build is pure and deterministic
- A dump format you can read, plus one graph printed and annotated
- A build counter for a twenty-token generation that prints 2, not 20

## How you know you passed

- Two builds of the same shapes produce identical graphs, and a shape change produces a different one.
- No allocation or kernel call happens during build, and you can show the test that would catch one.
- You can point at the exact node where the last-token logits come out, and why the tail rows are trimmed.
- Your generation builds two graphs and reuses the decode one for the rest of the run.

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

<!-- facts: graph.provenance -->
*`(unknown)`; on **arm64, Darwin 27.2.0, python 3.15.2**. Command: see [the provenance index](../appendix/provenance.md#graph). Raw numbers: `facts/graph.json`.*
<!-- /facts -->
