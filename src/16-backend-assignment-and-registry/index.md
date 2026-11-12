# Chapter 16 · Backend Assignment and the Registry

> **Compute tier**: ✅ any laptop | **Status**: stub
> **Build step S16**: A backend registry with stable ids, a capability query per op, and a splitter that cuts the graph into contiguous same-backend runs.
> **Rule of this book**: The device implementations themselves are Part 5. This chapter is the seam: how an op ends up on a device, and what an engine must never do at that boundary.

## First, the gist

With a graph in hand, the question becomes: where does each node run? Every backend answers one question - can you run this op with these shapes - and the scheduler asks it in priority order, then cuts the graph into contiguous runs so a device sees a long stretch of work instead of a node at a time. Two rules keep this honest. The first is that a failure is an error: an op no backend supports must refuse, because a silent fallback to the CPU is how a GPU path ends up quietly ten times slower. The second is that backend ids are a file-format contract: numbers that appear in saved files are appended, never renumbered.

## What you'll be able to do

- Implement a registry with priorities and a per-op capability query, and assign a whole graph deterministically.
- Explain the split step and why cross-backend copies happen only at boundaries.
- State the two refusals an engine owes its user: an unsupported op, and a backend name it does not have.

## Lessons

- [ ] 1. One question per backend: can you run this op, with these shapes, on this device
- [ ] 2. Priority order, and the determinism a stable ordering buys
- [ ] 3. Splitting: contiguous runs, boundary copies, and the cost of a bad assignment
- [ ] 4. Refusals: unsupported op, unknown backend name, and the failure that must never be a fallback (payoff: a silent CPU path is a performance bug with no error message)
- [ ] 5. Ids as a contract: append-only, and what breaks when they are not

## Assignment matrix

| Op | CPU | Metal | CUDA | Chosen | Why |
|---|---|---|---|---|---|
| matmul (quantized) |  |  |  |  |  |
| attention (prefill) |  |  |  |  |  |
| RMSNorm |  |  |  |  |  |
| copy |  |  |  |  |  |

## Project

**Build step S16 · the seam.** Build the registry before you build the second backend. A seam designed by one implementation fits exactly one implementation.

Write the refusal tests first: an op nothing supports, and a `--backend` value that does not exist. They are cheap and they pin the contract.

## Deliverable

- A registry with ids, priorities and capability queries, plus a deterministic assignment over a real graph
- A splitter producing contiguous runs, with a test that a boundary copy is inserted exactly once
- Two refusal tests, each asserting the error text names the op or the backend

## How you know you passed

- Assigning the same graph twice yields the same assignment, and the split boundaries are the same.
- An unsupported op fails the run; no test tolerates a silent CPU execution.
- Changing the priority order changes the assignment, which shows the priority is doing work.

## Reading

- code: `src/graph/registry.rs`, `src/graph/backend.rs`, `src/graph/scheduler.rs` at commit `88ffa5d`
- decisions: ADR-0011 (backend ids are append-only), ADR-0009 (a failure is an error, never a silent fallback), ADR-0012 (device is the first axis)
- docs: `docs/BACKEND-REGISTRY-DESIGN.md`, `docs/BACKENDS.md`

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
