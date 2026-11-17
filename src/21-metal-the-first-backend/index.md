# Chapter 21 · Metal: The First Backend

> - **Compute tier**: ✅ on any Apple-silicon Mac; ⚠️ elsewhere this is reading only (the seam is the lesson, the shaders are not)
> - **Status**: written - the two backends are run against each other on one model and one prompt by `scripts/metal.py`, the identity gate is a sha256 equality, and the price of zero-copy is measured by making the engine copy instead; what is *not* here is a device without shared memory (Chapter 22) or the CUDA kernels themselves (Chapters 23-26)
> - **Build step S21**: A first non-CPU backend behind the registry from Chapter 16: device buffers, a command buffer per split, and three shaders that pass the identity gate.
> - **Rule of this book**: CUDA is Chapters 22-26 and is not a translation of this chapter; the shared lesson (device layer, launch layer, kernels, executor) is stated once here and assumed afterwards.

## First, the gist

The first GPU backend is mostly a lesson about seams. Apple's **Metal** gives a process a device, a queue, buffers that may be shared with the CPU's address space, and command buffers that batch work. Because Apple silicon puts the CPU and the GPU on the same memory, this is the friendliest possible device to learn on: a weight buffer can be created *over* the mapped model file, and no copy of the weights happens at all.

<figure class="book-figure">{{#include ../figures/21-four-layers.svg}}</figure>

*Figure 21.1 - the four layers every backend has, drawn for the three machines this book runs on. The executor is the only layer that is polymorphic; the other three differ in their APIs, not in their shape.*

What the chapter really teaches is the split that every backend in this book follows - **runtime, launch, kernels, executor** - and the rule that keeps it honest: the backend never sees the graph's own bookkeeping. A backend that knows about node ids, liveness or the free list is a backend that cannot be replaced, and this book replaces one (Chapter 16's registry) for a living.

The measured difference on one machine, one model, one binary:

<!-- facts: metal.metal-vs-cpu -->
| Per token | CPU | Metal | Ratio |
|---|---|---|---|
| Prefill | 252.8 tok/s | **693.0 tok/s** | 2.74x |
| Decode | 90.5 tok/s | **259.2 tok/s** | 2.86x |
| of which: the forward pass | 10.54 ms | 3.80 ms | 2.77x |
| of which: the sampler | 0.08 ms | 0.08 ms | 1.0x |

Read the last two rows together, because they are the reason this chapter is short. The GPU takes the forward pass from 10.54 to 3.80 ms and leaves the sampler at 0.08 ms, untouched - **2.77x on the half that runs on the device, 1.00x on the half that does not**. Chapter 20 measured the sampler at under one per cent of a CPU decode step; on a device it is two per cent and rising, and no kernel you write will move it. A backend can only accelerate what it is handed.
<!-- /facts -->

That table is a target, not the lesson. The lesson is that it took four layers, three shaders, a submit path with a timeout, and an equality gate to get there - and that three of those four layers are the same on a CUDA device, which is why Part 5 is six chapters and not two.

## What you'll be able to do

- Stand up a device backend behind the registry: buffers, queues, one command buffer per split, errors surfaced.
- Write three shaders (a copy, a norm, a quantized matmul) and pass the identity gate against the CPU path.
- Explain zero-copy buffers and what changes when a device does not share memory with the host.

## Lessons

- [ ] 1. Four layers, one seam: runtime, launch, kernels, executor
- [ ] 2. A device queue and a command buffer: batching work instead of submitting per op
- [ ] 3. Zero-copy weights: what Apple silicon allows and why a discrete GPU cannot
- [ ] 4. Three shaders, then the gate: a copy, a norm, and one quantized matmul
- [ ] 5. What the backend must never do: allocate behind the allocator's back, or fall back silently

### Lesson 1 · Four layers, one seam

Every backend in this book is the same four layers, in the same order, and naming them is what makes the rest of Part 5 legible:

| Layer | In minfer | What it owns | What it must not know |
|---|---|---|---|
| **runtime** | `src/metal/runtime.rs`, `src/metal.rs` | the device, the queue, the shader library, the zero-copy weight registry | that a graph exists |
| **launch** | `src/metal/encode.rs`, `src/metal/policy.rs` | encoders, barriers, submission, and every "which kernel for this shape" decision | who called it, or why |
| **kernels** | `src/metal/kernels/*.metal` | one function per operation, on buffers and integers | anything about the model |
| **executor** | `src/graph/metal_backend.rs` | implementing `Backend`: buffer pool, name to (buffer, offset), per-op dispatch | the device's API |

The split is not decoration. `src/metal/` is the only place that touches Objective-C or Metal, and `metal_backend.rs` is the only place that knows what a graph node is - which is the same split CUDA uses, one part later, and the reason a chapter on CUDA kernels does not have to re-explain the executor.

The executor's own state is where the seam becomes visible:

<!-- code: src/graph/metal_backend.rs::MetalBackend@e844591 -->
```rust
pub struct MetalBackend {
    state: &'static MpsState,
    /// f32-element pool: id → shared MTLBuffer (size * 4 bytes)
    pool: Vec<crate::metal::MetalBuffer>,
    free: Vec<usize>,
    /// P2/P3 capture staging: host-readable buffers written by per-split blits.
    /// Only allocated while a trace/live capture is armed.
    staging: Vec<crate::metal::MetalBuffer>,
    free_staging: Vec<usize>,
    /// Pending command buffer for the current split. Stored as a leaked box
    /// pointer (null = none) because MpsCommandBuffer is !Send/!Sync; all
    /// access happens sequentially through &self/&mut self methods on the
    /// scheduler thread, so the raw pointer is contained.
    cb_ptr: *mut crate::metal::MpsCommandBuffer<'static>,
```
<!-- /code -->

Two things to notice in nine fields. The pool is a `Vec` of buffers indexed by id, and the free list holds *lengths* - not graph concepts, just a bump-and-reuse allocator that the graph decides how to use. And `state` is `&'static`: the device, the queue and the shader library are process-wide singletons, because a GPU is one resource even when two engines share it. That is the one place where a design decision from Chapter 16 (one registry, many engines) shows up as a lifetime.

**The seam rule.** The executor may ask the runtime for a buffer, a kernel or a completion; it may not ask it anything about the graph, and the runtime may not ask the executor anything at all. Every backend in this book obeys this, and the CUDA one will make you glad of it: the same executor shape, a different runtime, and no chapter needs to re-derive who owns the free list.

### Lesson 2 · A device queue and a command buffer

A GPU dispatch is cheap; a *submission* is not. Committing a command buffer has a fixed cost on the CPU side and a wakeup cost on the driver side, and a single decode step of a 24-layer model issues hundreds of dispatches. Submitting each one would spend more time in the driver than in the model, so the engine does what every serious GPU backend does: **one command buffer per split**, filled with dispatches, submitted once at the boundary.

The command buffer is created lazily on the first operation of a split, and the returned reference is `'static` even though it lives in the backend - a leaked box, because the borrow checker and the encoder's own lifetime do not want to agree:

<!-- code: src/graph/metal_backend.rs::cb@e844591 -->
```rust
    /// The current split's command buffer (created on first op of a split).
    /// The box is leaked, so the returned reference is 'static and does not
    /// borrow `self` — callers can freely touch the pool afterwards.
    fn cb(&mut self) -> &'static mut crate::metal::MpsCommandBuffer<'static> {
        if self.cb_ptr.is_null() {
            let cb = Box::new(self.state.cmd_buffer());
            self.cb_ptr = Box::into_raw(cb);
        }
        // SAFETY: cb_ptr is null or points to a live box created here; all
        // callers hold &mut self, so no concurrent mutation.
        unsafe { &mut *self.cb_ptr }
    }
```
<!-- /code -->

A leaked box is a real cost - the memory is never reclaimed - and here it is the right trade: there is exactly one live command buffer per backend at a time, it is replaced on every split boundary, and the alternative is threading a lifetime through every dispatch helper. This is the kind of decision a book should show rather than hide, because the next person to touch it will wonder.

**Barriers, or the illusion of order.** Dispatches in one command buffer do not necessarily run in the order they were encoded - the GPU may overlap them, and the memory they touch may not be visible to the next kernel without help. Every dispatch helper therefore ends with a barrier scoped to buffers, which costs little and removes an entire class of "it works until I add a second op" bug. On a device with separate memory the same barrier is not optional; it is the difference between a result and a race.

**Submission has to be able to fail.** This is the part every tutorial skips, and it is the part that costs days when it is missing. A GPU can hang - a driver fault, a bad shader, a hardware problem - and the naive submit waits forever, which turns one broken kernel into a process that cannot be killed politely. minfer waits with a bound, then asks the command buffer what happened, and reports the last few dispatches it encoded:

<!-- code: src/metal/encode.rs::submit@e844591 -->
```rust
    pub fn submit(self) -> Result<(), String> {
        if self.enc_open {
            self.enc.endEncoding();
        }

        // dispatch_semaphore_t is already a reference-counted opaque pointer.
        let sem = unsafe { dispatch_semaphore_create(0) };
        let sem_val = sem as usize;

        let blk = RcBlock::new(
            move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| unsafe {
                dispatch_semaphore_signal(sem_val as *mut c_void);
            },
        );
        unsafe {
            self.cmd_buf.addCompletedHandler(RcBlock::into_raw(blk));
        }
        self.cmd_buf.commit();

        // Bounded wait (10 s). If the GPU hangs (hardware fault), the completion
        // handler never fires and we bail out instead of blocking forever.
        let timeout = unsafe { dispatch_time(0, 10_000_000_000i64) }; // 10 s from now
        let rc = unsafe { dispatch_semaphore_wait(sem, timeout) };
```
<!-- /code -->

The rest of that function is the part that makes the error *useful*: on a timeout it formats the last dispatches from a per-device trace, so the message names the op that hung rather than the fact that something did. That trace is cheap - a bounded ring of strings behind a mutex - and it is the difference between a bug report that says "the GPU hung" and one that says "the fused FFN on layer 17 hung".

### Lesson 3 · Zero-copy weights

On Apple silicon, the CPU and the GPU share memory. That single fact removes the largest cost a backend has on a discrete card, and it is worth being precise about what the engine does with it.

The GGUF file is mapped, not read: the loader holds an `mmap` of each shard, and the weights are slices of it. Metal can wrap a range of host memory as a GPU buffer *without copying*, provided the base address is page-aligned - which `mmap` guarantees, and which the code asserts rather than hopes:

<!-- code: src/metal/runtime.rs::register_part@e844591 -->
```rust
    pub fn register_part(&self, data: &[u8], owner: crate::gguf::PartBytes) {
```
<!-- /code -->

Forty lines follow, and three of them are the whole idea:

<!-- code: src/metal/runtime.rs::register_part@e844591 -->
```rust
                    let page = 16384; // macOS page size on Apple Silicon
                    debug_assert!(base % page == 0, "mmap'd GGUF part not page-aligned");
                    self.inner
                        .device
                        .newBufferWithBytesNoCopy_length_options_deallocator(
                            NonNull::new(data.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (data.len() as u64) as usize,
                            MTLResourceOptions::StorageModeShared,
                            None,
                        )
                        .unwrap()
```
<!-- /code -->

`newBufferWithBytesNoCopy` means what it says: the `MTLBuffer` is a view over those pages, and every weight becomes a `(buffer, byte offset)` pair rather than a copy. `StorageModeShared` says the same memory is visible to both sides, so reading a weight on the CPU and reading it in a shader are the same read.

Three consequences follow, and each is a trap if a backend forgets it:

- **The mapping must outlive the buffer.** The GPU may touch those pages at any time, so the `mmap` handle is moved into the registry beside the buffer rather than dropped after registration. That is what the `owner` argument is for, and it is model-scoped: dropping the model removes the buffers first, then the mapping.
- **A part that is not a mapping cannot be zero-copy.** A `Vec`'s allocation is arbitrarily aligned and owned by an `Arc`, so the code takes the other branch and copies once per part - one allocation per part rather than one per tensor, which is the difference between a slow first load and a pathological one.
- **A discrete GPU has to copy, and the copy is the whole chapter.** There is no shared address space to alias, so the weights go over a bus into device memory once per model load. That is why Chapter 22 begins with "what kind of machine a GPU is" rather than with a port of these shaders.

What zero-copy is worth, measured by making the engine do the other thing:

<!-- facts: metal.metal-weight-copy -->
| Decode | tok/s |
|---|---|
| Zero-copy weights - the mmap'd GGUF part aliased into the buffer | **259.2** |
| Copied per weight (`MINFER_WEIGHT_COPY=1`) | 248.4 |
| What the copy costs | **-4.2%** |

So the copy costs 4.2 per cent of decode throughput on a 0.5B model whose weights are qwen2.5-0.5b-instruct-q4_k_m.gguf on disk. That is the honest size of the prize: zero-copy is free where the memory is shared, and a discrete GPU has to pay it - once per weight per model load, into memory it cannot see from the host - which is why Chapter 22 starts with what a CUDA device is rather than with a port of these shaders.
<!-- /facts -->

A few per cent, and only at load plus whatever the cache does with two copies of the same pages. That is the honest size of the prize on a machine that already shares memory - **and it is not the reason to do zero-copy**. The reason is that on a device that does *not* share memory the same decision has a completely different price, and a backend written to assume a copy is a backend that will make that copy forever.

### Lesson 4 · Three shaders, then the gate

The temptation with a first GPU backend is to port everything at once. The book's method is the opposite: write the smallest set of shaders that can run a model, and put the identity gate in front of it before adding a fourth.

The smallest set is three: a **copy** (the embedding lookup and KV moves), a **norm** (RMS norm, which is a reduction over one row), and one **quantized matmul** (the operation that reads the file's packed weights). Everything else in the model is these three, composed - which is Chapter 15's observation about the graph, arriving again on the device.

Here is the copy, in full, because a shader is short enough to read:

<!-- code: src/metal/kernels/f32.metal::kernel_get_rows_f32_embed@e844591 -->
```rust
kernel void kernel_get_rows_f32_embed(
    device const float * weights [[buffer(0)]],
    device const int   * ids     [[buffer(1)]],
    device       float * dst     [[buffer(2)]],
    constant    int    & ne      [[buffer(3)]],
    constant    int    & nt      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int total = nt * ne;
    int idx = (int)tid;
    if (idx >= total) return;
    int token_id = ids[idx / ne];
    dst[idx] = weights[(size_t)token_id * ne + (idx % ne)];
}
```
<!-- /code -->

Read the shape of it, because every kernel in this book has this shape: a global thread index, a bound check, a couple of integer divisions, and one load-store. `nt * ne` threads are dispatched, each copies one element, and the `if (idx >= total) return` is not defensive programming - GPU work is dispatched in fixed-size groups, so the last group is almost always partly outside the grid.

Now the gate. Before any speed number means anything, the device has to produce *the same tokens* as the CPU:

<!-- facts: metal.metal-identity -->
| | |
|---|---|
| Prompt | `The quick brown fox` |
| Tokens, greedy | 24 |
| CPU output, sha256 | `fd5d17e4a5b9a859` |
| Metal output, sha256 | `fd5d17e4a5b9a859` |
| Identical | **yes** |

> The quick brown fox jumps over the lazy dog. This is a classic English saying that has been used for centuries to describe

The two hashes are equal, and the words beside them are what a reader can check by eye. This is the gate the chapter is built around, and it is stricter than the quantized matmul's tolerance on purpose: a shader that reorders a sum changes the last float, and after thirty-two greedy steps that is a different sentence. Equality is the only test that fails at the right moment.
<!-- /facts -->

**Why equality and not a tolerance.** A quantized matmul that sums its blocks in a different order produces a slightly different float - and Chapter 9 measured how slightly, in the last digit. A tolerance would accept that, and the engine would still be wrong, because greedy decoding amplifies: the wrong last digit can flip an argmax, and one flipped token changes every token after it. The gate is byte equality on the *text*, not on the logits, because the text is what the reader gets.

That is the entry ticket. A backend that runs the graph but produces different tokens has implemented a different model, and no amount of throughput makes that acceptable - which is the subject of the next lesson, from the other side.

### Lesson 5 · What the backend must never do

Three rules, each of which is a bug somebody has shipped.

**Never allocate behind the allocator's back.** The buffer pool in Lesson 1 is indexed by the graph's own ids, so a backend that allocates its own scratch memory breaks the accounting Chapter 17 built: liveness analysis computes a peak, the planner reserves it once, and a hidden allocation turns a bounded step into an unbounded one. The device layer is allowed to own the *device*; it is not allowed to own a cache of its own.

**Never compute the wrong thing quietly.** A backend is handed operations it may not be able to express, and the choice is between refusing and approximating. minfer refuses, in a function whose comments read like a design record:

<!-- code: src/graph/metal_backend.rs::supports_op@e844591 -->
```rust
        // D1: Metal's kernels take a buffer and a length, with no element
        // offset, so it can only express an *exact* view (offset 0 == the
        // parent's own buffer, which is what the allocator maps). An offset
        // window would read the wrong bytes, so it is refused here rather
        // than silently mis-computed — the allocator backstops the partial
        // case, which `supports_op` cannot see (the parent's length is not
        // in the op). A standing design limit, not a pending port (G5 did not add it).
        Op::View { offset, .. } => *offset == 0,
```
<!-- /code -->

This is the per-engine gate of ADR-0006 in its smallest form: the backend answers a question about *itself*, the assignment step believes the answer (Chapter 16), and the graph never emits an op the backend cannot run. A `false` here does not mean "slow"; it means "not here", and the registry finds somewhere else.

**Never hide a switch.** Every launch decision in this backend is a named policy with a measured justification and an environment variable that turns it off:

<!-- code: src/metal/policy.rs::rms_norm_256_enabled@e844591 -->
```rust
pub fn rms_norm_256_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::config::get().rms_256)
}
```
<!-- /code -->

One line of body, and the doc comment above it carries the reason ("P1 2026-08-10 A/B gate; ON by default after it measured ~2x faster than the 32-thread kernel"). That is the pattern the book asks for everywhere: a heuristic is a claim, a claim has a measurement, and a measurement has an off switch so the next person can re-run the A/B instead of trusting it.

## Backend layer checklist

Filled in for this chapter, and the same table works for every backend in Part 5:

| Layer | Metal artifact | What it owns | What it must not know | Your status |
|---|---|---|---|---|
| runtime | `MpsState` (device, queue, library), `register_part` (zero-copy registry) | device lifetime, shader library, weight buffers | the graph | ✅ |
| launch | `MpsCommandBuffer` (encode, barrier, submit), `policy.rs` | batching, ordering, per-shape kernel choice | who called it | ✅ |
| kernels | `src/metal/kernels/*.metal` | one op each, on buffers | the model | ✅ (the three this chapter asks for) |
| executor | `MetalBackend` | pool, free list, name→offset, dispatch | the device's API | ✅ |

## Project

**Build step S21 · the friendliest device first.** Do this backend before buying GPU time. Every structural mistake in a backend is cheaper to find here, where the memory is shared and the profiler is local: no copy to debug, no bus to blame, no device-memory budget to overrun. What you learn here about the seam is what makes Chapter 22 a chapter about a machine rather than a chapter about an API.

If you have no Mac, read this chapter for the four-layer split and start at Chapter 22; the seam is the transferable part, and the shaders are not.

## Deliverable

- A Metal backend registered behind the registry, executing a real graph
- Three shaders with identity tests against the CPU path
- A measured tok/s for one model on this backend, quoted with conditions

## How you know you passed

- The full graph runs on the device and produces tokens at the identity gate's tolerance - which is to say, the same bytes.
- Disabling the backend via the registry flag returns the CPU result unchanged.
- You can name the one thing that would have to change for a discrete GPU, and it is the copy.

<!-- facts: metal.provenance -->
*`scripts/metal.py`; runs one binary against itself with `--backend cpu` and `--backend metal` on one model and one prompt, three runs each, compares the generated bytes, and repeats the Metal arm with `MINFER_WEIGHT_COPY=1` to price the copy that zero-copy avoids; on **Darwin 27.2.0, arm64**; medians of three runs for the rates; a sha256 equality for the gate; one timed run per backend for the sampler/forward split. Command: see [the provenance index](../appendix/provenance.md#metal). Raw numbers: `facts/metal.json`.*
<!-- /facts -->

## Reading

- code: `src/metal.rs`, `src/metal/{runtime,encode,ops,policy}.rs`, `src/metal/kernels/*.metal`, `src/graph/metal_backend.rs` at the book's [pinned baseline](../appendix/baseline.md)
- decisions: ADR-0005 (Metal becomes a first-class backend), ADR-0037 (a weight dtype a kernel cannot consume is refused), ADR-0080 (the backend is a Cargo feature and a build without it refuses the name)
- docs: `docs/METAL-BACKEND-DESIGN.md`, `docs/METAL_OPTIMIZATIONS.md`, `docs/inference_e2e_walkthrough/14-metal-backend.md`
- [metal] *Metal Shading Language Specification*, Apple - no arXiv ID (vendor manual)

## Backfill

- [x] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - the sampler/forward split in Lesson 2's table, and Chapter 1's ceiling as the bound to check it against.
- [x] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - Lesson 3's zero-copy argument only works because the file is bytes that are aliased, not floats that are parsed.
- [x] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - Lesson 1's four layers, and Lesson 5's `supports_op` as the seam's contract.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - Lesson 4's sha256 equality and Lesson 2's measured rates.
