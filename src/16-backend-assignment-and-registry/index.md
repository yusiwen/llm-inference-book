# Chapter 16 · Backend Assignment and the Registry

> - **Compute tier**: ✅ any laptop
> - **Status**: draft
> - **Build step S16**: A backend registry with stable ids, one capability question per op, a splitter that cuts the graph into contiguous same-backend runs, and three refusals you can reproduce in three commands.
> - **Rule of this book**: The GPU backends themselves are Part 5; the CPU one is what you have been writing since Chapter 6, and it goes into this same registry. This chapter is the seam: how an op ends up on a device, and what the engine owes you when it cannot.

## First, the gist

Chapter 14 built a graph; Chapter 15 rewrote parts of it. Now the question the graph makes unavoidable: **where
does each node run?** Every backend answers one question - can you run this op, at this dtype, with these
shapes - and the assignment pass asks it in priority order and writes the answer into the node.

Two design decisions carry the rest. The backend set is *data*, not an enum: a handle with a fixed id space, a
name, a priority and a capability record, so adding a device stops meaning editing the allocator, the
scheduler, the fusion wiring and the exporters. And the graph is cut into **contiguous same-backend runs**, so a
device receives a stretch of work instead of a node at a time.

The part people get wrong is the failure mode. A stub version of this chapter said an engine must never fall
back to the CPU. The engine in this book does fall back - *when you did not name a backend* - and refuses
instead when you did. Lesson 4 is that distinction, with three refusals reproduced on this machine, verbatim.

## What you'll be able to do

- Implement a registry whose ids are a file-format contract and whose assignment order is a separate,
  pinned number.
- Assign a whole graph deterministically, and explain why the assignment pass refuses to depend on hash
  iteration order.
- Cut that graph into splits and price a boundary.
- State the three refusals an engine owes its user, and reproduce each one.

## Lessons

- [ ] 1. One question per backend: can you run this op, at this dtype, on this device
- [ ] 2. Two orders, and the determinism a stable ordering buys
- [ ] 3. Splitting: contiguous runs, boundary copies, and the cost of a bad assignment
- [ ] 4. Refusals: unknown name, missing backend, unavailable device - and when a fallback is correct
- [ ] 5. Ids as a contract: append-only, and what breaks when they are not

### Lesson 1 · One question per backend

The question is a method on the backend trait (`supports_op`, Chapter 15), and the *asking* is a walk over the
registry in priority order. The walk is worth reading in full, because every rule this chapter states is in it:

<!-- code: src/graph/alloc.rs::supports_for@e844591 -->
```rust
    /// F4: the device order is the registry's **priority** order, not a statement
    /// order in this function — Metal (300), CUDA (200), CPU (100) — and a backend
    /// the run fenced off (`--backend` / `MINFER_BACKENDS`) or whose pool is not
    /// enabled is skipped. The CPU is tried last and is always allowed: it is the
    /// universal fallback (`BackendFilter::from_names` re-admits it), so the order
    /// and the answer are identical to the pre-F4 chain.
    pub fn supports_for(
        &self,
        op: &Op,
        dtype: crate::graph::DType,
        layer: Option<usize>,
    ) -> Option<Backend> {
        let eligible = |b: &dyn BackendTrait| -> bool { super::backend_takes(b, op, dtype) };
        let device_ok = match (self.offload, layer) {
            (None, _) => true, // no policy: the pre-E5 all-or-nothing behaviour
            (Some(p), Some(l)) => p.on_device(l),
            (Some(p), None) => p.device_holds_unblocked(),
        };
        if device_ok {
            for &backend in super::registry::registry().by_priority() {
                if backend == Backend::CPU {
                    continue; // the fallback, tried below even when device_ok is false
                }
                if !self.filter.allows(backend) {
                    continue;
                }
                if let Some(pool) = self.pool(backend) {
                    if eligible(pool) {
                        return Some(backend);
                    }
                }
            }
        }
        if self.filter.allows(Backend::CPU) && eligible(&self.cpu) {
            return Some(Backend::CPU);
        }
        None
```
<!-- /code -->

Read the order of the clauses. A node whose block the offload plan left on the CPU never reaches a device
(`device_ok`). The loop skips the CPU on purpose - `continue; // the fallback, tried below even when device_ok
is false` - because the CPU is tried after the fence, unconditionally. A fenced-off backend is skipped. A pool
that is not enabled is skipped. And the last three lines are the fallback: if the CPU is allowed and can run the
op, the CPU gets it.

What that produces, per operand, is the engine's documented contract. Every row is `supports_op`'s answer for one
backend, and the engine ships a test - `support_table_matches_support_matrix_doc` - that fails when any column
drifts from it:

<!-- facts: dispatch.ds-matrix -->
| Operator | CPU | Metal | CUDA |
|---|---|---|---|
| `Input, KvcacheLoad, View/Reshape/Permute` | yes | yes | yes |
| `Add, Mul, Silu` | yes | yes | yes |
| `RmsNorm, QkNorm` | yes | yes | yes |
| `MatMul` | yes | yes | yes |
| `GetRows (embedding, tail rows)` | yes | yes | yes |
| `View with offset != 0 or a partial window (D1)` | yes | no | yes |
| `Attn` | yes | yes | yes |
| `Attn with explicit_span (a window that starts at a non-zero cell, or s` | yes | yes | yes |
| `KvcacheStore` | yes | yes | yes |
| `SwiGLU (fused)` | yes | yes | yes |
| `RoPE non-interleaved` | yes | yes | yes |
| `RoPE llama3 scaling` | yes | yes | yes |
| `RoPE interleaved` | yes | yes | yes |
| `FusedQKV (decode)` | no | yes | yes |
| `FusedFFN (decode)` | no | yes | yes |
| `FusedQkvNorm (Qwen3 decode)` | no | yes | yes |
| `QkvBiasRopeStore (mixed-quant decode)` | no | no | yes |
| `Scale, Softmax, BatchMatMul` | yes | no | no |

18 documented rows; 14 accept the op on CPU, 4 refuse it. `supports_op` is what decides, and the engine's own test `support_table_matches_support_matrix_doc` fails when a column drifts from this table.
<!-- /facts -->

Note the shape of the asymmetries, because they are all deliberate: the fused decode ops (`FusedQKV`, `FusedFFN`,
`FusedQkvNorm`) exist **only** on the devices, `QkvBiasRopeStore` only on CUDA, and one row is a *standing
design limit* rather than a pending port - Metal cannot express a `View` with a non-zero element offset, because
its kernels take a buffer and a length. That row is why the hand-written `Op::FusedFFN` stays on Metal.

The three tests behind that table were run for this chapter on this machine:

<!-- facts: dispatch.ds-tests -->
| Test | Result |
|---|---|
| `graph::op_matrix::every_op_has_a_matrix_decision` | ok |
| `graph::op_matrix::matrix_cases_match_their_reference` | ok |
| `graph::op_matrix::support_table_matches_support_matrix_doc` | ok |
<!-- /facts -->

The interesting one is `every_op_has_a_matrix_decision`: `op_label` matches exhaustively with no wildcard arm, so
adding an `Op` variant is a **compile error** until the matrix covers it or excuses it with a reason. That is the
same trick as Chapter 10's identity gate - a rule enforced by the compiler rather than by a reviewer's memory.

### Lesson 2 · Two orders, and the determinism a stable ordering buys

The registry holds two orders and they are not the same one:

<!-- code: src/graph/registry.rs::Two orders@e844591 -->
```rust
//! ## Two orders, and the determinism rule
//!
//! The registry has two orders and they are deliberately not the same:
//!
//! - **identity** — [`Backend::index`], the id order. It fixes the on-disk
//!   KV-session tag, the exported graph's backend index, `Ord`, and the order of
//!   the fusion pass's backend list.
//! - **priority** — [`Registry::by_priority`], descending
//!   [`BackendEntry::priority`]. It is the *assignment preference*: which
//!   backend `GraphAllocator::supports_for` offers first. Before F4 it was
//!   implicit in the statement order of that function (Metal, then CUDA, then
//!   CPU); it is preserved exactly and is now a number the ordering gate pins.
//!
//! Nothing here may depend on `HashMap` iteration order: the assignment pass
//! fixes a graph's topology, and topology is part of the reuse identity
//! (standing rule 3). The table is a fixed-size array indexed by the handle id,
//! `by_priority` is sorted by `(Reverse(priority), index)` so no two entries can
//! tie, and [`BackendFilter`] is a `[bool; N_BACKENDS]` indexed by id rather
//! than an iterated set.
```
<!-- /code -->

Identity is the id order - `cpu` = 0, `metal` = 1, `cuda` = 2 - and it is fixed because it is *on disk*. Priority
is the assignment preference, and it is a number so that it can be pinned by a test instead of by statement order
in a function. The doc comment names the consequence: before this, the order was implicit in the sequence of
`if` statements inside `supports_for`, which means a refactor could silently change which device won.

The table, read out of the source:

<!-- facts: dispatch.ds-registry -->
| Backend | Registry id | Assignment priority |
|---|---|---|
| `cpu` | 0 | 100 |
| `metal` | 1 | 300 |
| `cuda` | 2 | 200 |

Backend ids in the space: **3**. The id order is the on-disk tag order; the assignment order is priority, and the two are deliberately not the same.
<!-- /facts -->

The determinism rule is the second half of that quote: nothing may depend on `HashMap` iteration order, because
the assignment pass decides a graph's topology, and topology is part of the reuse identity from Chapter 14. So
the registry is a fixed-size array indexed by handle id, `by_priority` is sorted by `(Reverse(priority), index)` so
no two entries can tie, and the run's `--backend` fence is a `[bool; N_BACKENDS]` indexed by id rather than an
iterated set.

A test pins the whole thing - `the_registered_set_and_priority_order_are_pinned` - and the values in the table
above are its subject. "Arbitrary but pinned" is the honest description of 300/200/100: the numbers carry no
meaning, the *order* does, and the test is what stops a well-meaning edit from reordering Metal below CPU.

### Lesson 3 · Splitting: contiguous runs, boundary copies

Assignment writes a backend into every node. Then the scheduler cuts the graph into runs:

<!-- code: src/graph/scheduler.rs::assign_backends@e844591 -->
```rust
    /// Assign every node to the best backend that supports it (capability
    /// driven via the allocator's backend registry).
    ///
    /// E5: the node's block is part of the question — `supports_for` keeps a node whose
    /// block the offload plan left on the CPU off the device, so a partially offloaded
    /// model never runs a block whose weights were never registered there.
    pub fn assign_backends(&self, graph: &mut ComputeGraph, alloc: &GraphAllocator) {
        for node in &mut graph.nodes {
            if node.backend.is_some() {
                continue; // keep explicit assignments
            }
            node.backend = alloc
                .supports_for(&node.op, node.out_dtype, node.layer)
                .or(Some(BackendTag::CPU));
        }
    }

    /// Partition the graph into contiguous same-backend splits.
    /// Cross-split inputs/outputs are derived from the src edges.
    pub fn split_graph(&self, graph: &ComputeGraph) -> Vec<Split> {
        let n = graph.n_nodes();
        let mut splits: Vec<Split> = Vec::new();
        let mut cur: Option<BackendTag> = None;
        let mut split = Split::new(BackendTag::CPU, 0);
        for id in 0..n {
            let node = graph.node(id);
            let b = node
                .backend
                .unwrap_or_else(|| cur.unwrap_or(BackendTag::CPU));
            if let Some(c) = cur {
                if b != c {
                    split.node_range.1 = id;
                    splits.push(std::mem::replace(&mut split, Split::new(b, id)));
                }
            } else {
                split = Split::new(b, id);
            }
            cur = Some(b);
        }
        split.node_range.1 = n;
        splits.push(split);
```
<!-- /code -->

Two things in those forty lines. `assign_backends` keeps explicit assignments (`if node.backend.is_some() {
continue }`) - the graph builder can pin a node - and otherwise asks and falls back, `.or(Some(BackendTag::CPU))`.
`split_graph` walks ids in order and starts a new split whenever the backend changes, then derives cross-split
edges by asking, for every node, whether a `src` input lives on a different backend.

What this machine produced, counted from the graph that actually ran rather than from a diagram:

<!-- facts: dispatch.ds-assignment -->
| | Prefill graph | Decode graph |
|---|---|---|
| Nodes | 443 | 440 |
| Backends assigned | cpu: 443 | cpu: 440 |
| Contiguous same-backend splits | 1 | 1 |
| Cross-split edges (each one a copy) | 0 | 0 |
| How the dump names a backend | a name, not an id | a name, not an id |
<!-- /facts -->

<figure class="book-figure">{{#include ../figures/16-assignment.svg}}</figure>

*Figure 16.1 - the assignment on this machine: three registry ids, the priority order they are tried in, and the graph as one contiguous CPU run. The last line prices what this run did not have to pay.*

One split, no cross-split edges, because only one backend could participate: this build has Metal compiled in but
the run disabled it, and CUDA is not in the build at all. The honest statement is not "the splitter works" but
"the splitter had nothing to split" - and that is why the figure prices a boundary instead of drawing one. A
residual-stream tensor crossing a boundary is [34, 896] f32 here: 121,856 bytes, per crossing, on top of the
synchronisation that makes the copy necessary in the first place.

Assume the arithmetic: a graph cut into `k` runs crosses at least `k - 1` boundaries, and a boundary in the middle
of the layer stack costs a full activation copy in *both* directions if the runs alternate. That is the argument
for a priority order that keeps the stack together rather than one that picks the best backend per op - a per-op
choice with a device that wins some ops and loses others can produce a boundary per layer, and each one is a
round trip that costs more than the op it was optimising.

### Lesson 4 · Refusals: three messages, and the one fallback that is correct

A user names a backend with `--backend`. Three things can then be wrong, and the engine distinguishes all three:

<!-- code: src/graph/registry.rs::NameError@e844591 -->
```rust
/// Why a name could not be resolved. The two cases are deliberately distinct: a
/// name that does not exist is always an error, and so is a name this build does
/// not contain — a compiled-out backend must never look absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NameError {
    /// No backend has this name (on any configuration).
    Unknown { name: String },
    /// The name exists, but this build does not contain the backend.
    NotCompiled {
        name: &'static str,
        why: &'static str,
    },
}

impl NameError {
    /// The message a startup refusal prints.
    pub fn message(&self) -> String {
        match self {
            NameError::Unknown { name } => format!(
                "unknown backend '{name}'; known backends are: {}",
                NAMES.join(", ")
            ),
            NameError::NotCompiled { name, why } => {
                format!("backend '{name}' is known but not compiled into this build: {why}")
            }
        }
    }
}
```
<!-- /code -->

The compile-time case carries its own reason, because a compiled-out backend must never look absent - the message
for CUDA on a macOS build names the feature flag rather than pretending the name is a typo. The runtime case is the
third:

<!-- code: src/graph/registry.rs::check_available@e844591 -->
```rust
/// Startup stage 2: every **named** backend that is compiled in must also be
/// usable on this machine. Run once the device layer is up, before the model is
/// loaded — a named backend that cannot run is refused, never dropped.
///
/// Only a named backend is checked. The default request ("whatever this build
/// can use") must keep the pre-F4 behaviour, where an unavailable device is
/// simply not used: `MINFER_DISABLE_CUDA=1` means "run on the CPU", not "refuse
/// to start".
pub fn check_available(filter: &BackendFilter) -> Result<(), String> {
    for &backend in registry().by_priority() {
        if !filter.is_requested(backend) {
            continue;
        }
        if let Some(why) = unavailable_reason(backend) {
            return Err(format!(
                "backend '{}' is compiled in but not available on this machine: {why}",
                backend.name()
            ));
        }
    }
    Ok(())
}
```
<!-- /code -->

And then the sentence that decides the question this chapter's stub got wrong: *"Only a named backend is checked.
The default request ('whatever this build can use') must keep the pre-F4 behaviour, where an unavailable device is
simply not used: `MINFER_DISABLE_CUDA=1` means 'run on the CPU', not 'refuse to start'."*

So the rule is not "never fall back". It is:

| Request | Behaviour | Why |
|---|---|---|
| no `--backend` | an unavailable device is **not used**; the run proceeds on the CPU | the engine has to run everywhere, and the CPU is the universal backend |
| `--backend cuda` on a build without CUDA | **refused** at startup | a build flag is a build fact; silently ignoring the request wastes a benchmark run |
| `--backend metal` with no usable device | **refused** at startup, with the reason | you asked for a device; running somewhere else without saying so is the bug |

All three refusals, run on this machine for this chapter, with their exit codes:

<!-- facts: dispatch.ds-refusals -->
| Case | Command | Exit | Message |
|---|---|---|---|
| unknown name | `--backend foo` | 1 | Error: unknown backend 'foo'; known backends are: cpu, metal, cuda |
| known but not compiled | `--backend cuda` | 1 | Error: backend 'cuda' is known but not compiled into this build: the CUDA backend is compiled only with --features cuda |
| compiled but unavailable | `--backend metal` | 1 | Error: backend 'metal' is compiled in but not available on this machine: no Metal device, or MPS is unavailable (MINFER_DISABLE_MPS) |
<!-- /facts -->

Two details in that table are worth more than the words. The third case is preceded by `MPS: not available, using
CPU fallback` on the same run - those two lines are *both* correct: the device layer falls back, and the fence
refuses. They answer different questions. And every case exits non-zero, which is what makes them usable in a
script: a refusal that exits 0 is a warning, and a warning in a benchmark harness is a wrong number later.

### Lesson 5 · Ids as a contract

The handle is a `u16` newtype with three constants, and the doc comment on the type says the only thing that
matters about the number:

<!-- code: src/graph/registry.rs::Backend@e844591 -->
```rust
/// A registered backend, as a cheap index into the registry.
///
/// The constants are the only handles the code should name; [`Backend::from_index`]
/// exists for the id spaces that persist one (the KV-session tag).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Backend(u16);

impl Backend {
    pub const CPU: Backend = Backend(0);
    pub const METAL: Backend = Backend(1);
    pub const CUDA: Backend = Backend(2);

    /// The registry id (and the on-disk KV-session backend tag).
    pub const fn index(self) -> usize {
        self.0 as usize
    }
```
<!-- /code -->

The number is persisted in exactly one place - the KV-session file, which stores a cache so a later run can resume
from it:

<!-- code: src/graph/kvsession.rs::tag_of@e844591 -->
```rust
/// The on-disk backend tag: the registry id, so the file format is the handle's
/// id space and nothing else (F4). `backend_of_tag` is its inverse.
fn tag_of(backend: Backend) -> u32 {
    backend.index() as u32
}

fn backend_of_tag(tag: u32) -> Result<Backend, String> {
    Backend::from_index(tag as usize).ok_or_else(|| {
        format!("KV session: unknown backend tag {tag} (this file was written by a newer build?)")
    })
}
```
<!-- /code -->

`backend_of_tag` is the inverse, and its error message is the whole lesson in one line: *"unknown backend tag
`{tag}` (this file was written by a newer build?)"*. A file written by this build is read by this build; a file
written by a build with a fourth backend is refused with a message that names the likely cause, instead of
returning a cache whose contents are attributed to the wrong device.

That gives the append-only rule its teeth:

- **Append, never renumber.** `cpu` = 0, `metal` = 1, `cuda` = 2 are on disk. Inserting a backend at 1 in a later
  version silently reinterprets every session file that mentions 1.
- **Names are for humans, ids are for files.** The per-node trace and the graph export write `"cpu"`, not `0` -
  which is why the table in lesson 3 says the dump names a backend rather than numbering it. A debug dump is
  read by a person; a cache file is read by a program.
- **A new id must be refused by old builds, not ignored.** `from_index` returns `Option`, and the session reader
  turns `None` into an error. A `bool` return would have turned "a device I do not know" into "not the CPU".

## Project

**Build step S16 · the seam.** Build the registry before you build the second backend. A seam designed by one
implementation fits exactly one implementation.

Write the refusal tests first: an op nothing supports, a `--backend` value that does not exist, and a
`--backend` naming a backend that exists in the source but not in this build. They are cheap, and they pin the
contract before there is a second device to disagree with it.

Then do the measurement that keeps the splitter honest: run a graph whose nodes are all one backend and count the
splits. If your answer is not 1, your splitter is wrong or your assignment is not deterministic - and the second
case is the one that will look like a scheduling bug for a week.

## Deliverable

- A backend handle with a fixed id space, a name, and a priority separate from its id
- A capability query per (op, dtype) with a documented table, and a test that keeps the table honest
- An assignment pass whose result depends on nothing but the graph, the registry and the fence
- A splitter that reports its splits and its cross-split edges
- Three refusal messages, each with its own exit code

## How you know you passed

- Two runs of the same graph produce byte-identical assignments, and a reordered registry with the same
  priorities still assigns the same way.
- You can point at a node whose backend differs from its input's and say what crosses the boundary.
- Every refusal exits non-zero and names what was wrong: the name, the build, or the device.
- Your on-disk cache refuses an id it does not know, and says why that might be.

## Reading

- code: `src/graph/registry.rs`, `src/graph/scheduler.rs`, `src/graph/alloc.rs`, `src/graph/kvsession.rs` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/BACKEND-REGISTRY-DESIGN.md` (the design record this chapter follows), `docs/SUPPORT-MATRIX.md` §"Operator Coverage by Backend"
- decisions: ADR-0007 (the backend registry's id space), `docs/BACKENDS.md`
- [roofline] Williams, Waterman and Patterson, *Roofline: an insightful visual performance model for multicore architectures* (CACM, 2009)
- [scaling-inference] Pope et al., *Efficiently Scaling Transformer Inference* (2022) - arXiv:2211.05102

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)

<!-- facts: dispatch.provenance -->
*`scripts/dispatch.py`; runs the engine's CLI three ways with an impossible or unavailable `--backend` value and records each refusal verbatim, runs the engine's own op-support contract tests, and reads the backend assignment out of the per-node trace of a real run; the registry ids, names and priorities are parsed from the pinned source, and the op-by-backend table from `docs/SUPPORT-MATRIX.md`; on **{'platform': 'macOS-27.2.1-arm64-arm-64bit-Mach-O', 'machine': 'arm64', 'python': '3.15.2'}**; counts of nodes per backend and of contiguous same-backend runs, from one graph per phase; the refusals are exit codes and messages, compared as values, not summarised. Command: see [the provenance index](../appendix/provenance.md#dispatch). Raw numbers: `facts/dispatch.json`.*
<!-- /facts -->
