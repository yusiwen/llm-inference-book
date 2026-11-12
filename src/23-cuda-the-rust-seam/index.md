# Chapter 23 · CUDA II: The Rust-CUDA Seam

> **Compute tier**: ⚠️ needs an NVIDIA GPU and `nvcc`; the seam itself is compile-time work you can read anywhere | **Status**: stub | **Build step S23**: A safe Rust wrapper over the CUDA runtime: device buffers with ownership, a stream, error states that cannot be ignored, and a bounded wait. | **Rule of this book**: Kernel families are Chapter 24. This chapter never looks inside a kernel: it is about everything around them, which is where a GPU backend usually breaks.

## First, the gist

Rust does not compile CUDA, so a CUDA backend is two languages and one hand-written boundary. The boundary is where the interesting engineering lives: allocating and freeing device memory without leaking it, associating an allocation with the model that owns it, translating error codes into results nobody can ignore, and never waiting forever on a device that has stopped responding. Every one of these is a place where the ergonomic choice (silently return, reuse a buffer, wait indefinitely) produces a bug that shows up as a wrong answer or a hung process a thousand tokens later.

## What you'll be able to do

- Wrap device memory in a type whose lifetime is tied to the model that owns it.
- Translate every runtime call's error code into a result, and prove no path ignores one.
- Bound every wait, and design the failure path a hung device takes.

## Lessons

- [ ] 1. Two languages, one boundary: FFI, opaque handles and the types that own them
- [ ] 2. Allocation, pooling and release: why the pool recycles exact byte lengths and never frees
- [ ] 3. Errors as results: the launch-severity helper, and the 120 call sites it prevents
- [ ] 4. Bounded waits, stream synchronisation, and the failure a hung device must produce (payoff: a watchdog, not a hang)
- [ ] 5. Linking: static and shared runtimes, and what the binary then requires at run time

## Seam checklist

| Concern | Naive choice | What it costs later | Book's rule | Your status |
|---|---|---|---|---|
| allocation |  |  |  |  |
| ownership |  |  |  |  |
| error handling |  |  |  |  |
| waits |  |  |  |  |
| linking |  |  |  |  |

## Project

**Build step S23 · the part you cannot debug later.** Write this layer as if every call fails, because on someone's machine it will. The tests for this chapter are failure-injection tests, not happy paths.

Do not add a convenience `unsafe` escape hatch until Chapter 24 is written; the escape hatch is how the error discipline dies.

## Deliverable

- A safe wrapper with owned buffers, a stream, and results instead of ignored codes
- A bounded wait plus a test for the timeout path
- A note recording how your build links the runtime and what the binary needs at run time

## How you know you passed

- A deliberately failed allocation returns an error naming the request, and the test asserts the message.
- A wedged stream produces a timeout error, and the process exits rather than hanging.
- `ldd` (or `otool -L`) on your binary shows only the libraries you intended.

## Reading

- code: `src/cuda.rs`, `src/cuda/ffi_runtime.rs`, `src/cuda/methods/{init,buffers,copy,stream,events,capture}.rs` at the book's [pinned baseline](../appendix/baseline.md)
- decisions: ADR-0008 (GPU safety: bounded waits, runtime device limits), ADR-0009 (a failure is an error), ADR-0030 (launch severity lives in the helper), ADR-0033 (the pool recycles exact byte lengths)
- docs: `docs/GPU_SAFETY.md`, `docs/BUILD.md` (the `cuda` and `cuda_static` features)
- [cuda-guide] *CUDA C++ Programming Guide*, NVIDIA - no arXiv ID (vendor manual)

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
