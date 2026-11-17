# Chapter 23 · CUDA II: The Rust-CUDA Seam

> - **Compute tier**: ⚠️ needs an NVIDIA GPU and `nvcc`; the seam itself is compile-time work you can read anywhere
> - **Status**: written - the three failures are made to happen on a GB10 by `scripts/seam/seam.cu` (a terabyte that will not allocate, a two-second kernel behind two different waits, and an `ldd` on the result); what is *not* here is anything inside a kernel (Chapter 24)
> - **Build step S23**: A safe Rust wrapper over the CUDA runtime: device buffers with ownership, a stream, error states that cannot be ignored, and a bounded wait.
> - **Rule of this book**: Kernel families are Chapter 24. This chapter never looks inside a kernel: it is about everything around them, which is where a GPU backend usually breaks.

## First, the gist

Rust does not compile CUDA, so a CUDA backend is two languages and one hand-written boundary. The kernels are the interesting half to read about and the *easy* half to get right; the boundary is where the bugs live, because every crossing has an ergonomic wrong answer that costs a day a thousand tokens later:

| Crossing | The cheap wrong answer | What it costs |
|---|---|---|
| allocate | `cudaMalloc` and hope | a failed allocation surfaces as a wrong number, three layers up |
| own | free whenever the pointer goes out of scope | a captured graph holds a lease on a buffer someone just freed |
| report | ignore the return code | an error latches and reappears at the next sync, blaming the wrong kernel |
| wait | `cudaStreamSynchronize` | a wedged kernel hangs the process, with no way to ask it to stop |
| link | "it built, ship it" | the binary needs a toolkit the deployment does not have |

This chapter takes the five in order, and measures the ones that can be made to fail on purpose.

<!-- facts: cuda-seam.seam-failures -->
| Crossing | What was asked of it | What came back |
|---|---|---|
| allocate | `cudaMalloc` for 1 TiB on a device with none to give | `cudaErrorMemoryAllocation` - "out of memory" |
| wait (blocking) | one kernel that busy-waits 2,000 ms, behind `cudaStreamSynchronize` | the host blocked for **1,924 ms**, and no API to ask it to come back sooner |
| wait (polled) | the same kernel, behind `cudaStreamQuery` against a 500 ms deadline | **500 ms**, then *gave up* |

The first row is the argument for wrapping every call: CUDA's own message is "out of memory" - true, and useless. It does not say what was asked for, by which model, or how much was left. A seam that returns `cudaErrorMemoryAllocation` upward has thrown away the only three facts that make the failure actionable, so the wrapper's job is to put them back.

The last two rows are the same kernel with two different waits, and they disagree by 1,424 ms. There is no `cudaStreamSynchronizeWithTimeout` to call, so a backend that must survive a wedged device owns a clock and polls; one that calls the blocking version has decided that a hung kernel is allowed to hang the process.
<!-- /facts -->

The first row is the argument for wrapping every call, and it is worth reading twice: CUDA's own message is **"out of memory"** - true, and useless. It does not say what was asked for, by which model, or how much was left. A seam that passes `cudaErrorMemoryAllocation` upward has thrown away the only three facts that make the failure actionable.

<figure class="book-figure">{{#include ../figures/23-seam.svg}}</figure>

*Figure 23.1 - the five crossings of the seam, the cheap choice at each one, and what that choice costs when it is discovered a thousand tokens later.*

## What you'll be able to do

- Wrap device memory in a type whose lifetime is tied to the model that owns it.
- Translate every runtime call's error code into a result, and prove no path ignores one.
- Bound every wait, and design the failure path a hung device takes.

## Lessons

- [ ] 1. Two languages, one boundary: FFI, opaque handles and the types that own them
- [ ] 2. Allocation, pooling and release: why the pool recycles exact byte lengths and never frees
- [ ] 3. Errors as results: the launch-severity helper, and the 144 call sites it prevents
- [ ] 4. Bounded waits, stream synchronisation, and the failure a hung device must produce
- [ ] 5. Linking: static and shared runtimes, and what the binary then requires at run time

### Lesson 1 · Two languages, one boundary

Rust can call C, and CUDA's runtime is a C API, so the boundary is `extern "C"` and a pile of raw pointers. Two problems follow immediately, and both are solved by types rather than by discipline.

**First, raw pointers are neither `Send` nor `Sync`.** A device pointer must cross threads - the engine runs one thread per request, and they share a device - so the pointer has to be wrapped in something that says so, out loud, in one place:

<!-- code: src/cuda.rs::CudaPtr@e844591 -->
```rust
/// Wrapper to make `*mut c_void` Send+Sync for use in Mutex.
#[derive(Clone, Copy)]
struct CudaPtr(*mut std::ffi::c_void);
unsafe impl Send for CudaPtr {}
unsafe impl Sync for CudaPtr {}
```
<!-- /code -->

That `unsafe impl` is a promise the compiler cannot check: *the code that owns this pointer serialises access*. Putting it on a four-line newtype means the promise is made once, at a place a reviewer can see, instead of at every function that touches a pointer.

**Second, freeing is not `Drop`.** A device allocation must outlive every kernel that reads it, and the natural Rust answer - free when the owner drops - is wrong as soon as two things share the buffer, or as soon as a captured graph holds it. So the *model* owns a registration, and the registration's `Drop` is what releases:

<!-- code: src/cuda.rs::CudaModelRegistration@e844591 -->
```rust
pub struct CudaModelRegistration {
    id: u64,
}

impl CudaModelRegistration {
    /// The id the ledger tags this model's allocations with.
    pub(crate) fn new(id: u64) -> Self {
        Self { id }
    }
```
<!-- /code -->

Nine bytes of state and a `Drop` (the next lines) that calls `release_registration(self.id)`. The type carries no pointer at all - it carries an *id into a ledger*, which is the whole trick: the ledger decides when bytes actually go back, because the ledger is the only thing that knows how many owners a buffer has. Chapter 21's Metal registration is the same shape (`ModelRegistration`, ADR-0077); this is its device-memory counterpart (ADR-0078).

### Lesson 2 · Allocation, pooling and release

A decode loop asks for the same shapes over and over: an activation of `n_embd` floats, a KV row, a logits vector. Allocating and freeing those per token would spend the budget in `cudaMalloc`, so the pool recycles - and the first surprise is how strict the recycling key is.

<!-- code: src/cuda.rs::DeviceAlloc@e844591 -->
```rust
struct DeviceAlloc {
    ptr: CudaPtr,
    size: usize,
    /// Registration ids that own this buffer. Same-name+same-size+same-fingerprint
    /// reuse adds the current registration instead of allocating a second copy.
    owners: Vec<u64>,
    /// A registration released this allocation while a captured graph was live; the
    /// bytes stay resident until [`CudaState::reap_retired`] frees them.
    retired: bool,
}
```
<!-- /code -->

Four fields, and each one is a bug that was fixed:

- **`size`** - the pool recycles *exact byte lengths*, not size classes. A request for 4,096 bytes cannot be served by a 4,352-byte buffer, because a kernel that reads 4,096 floats and a kernel that reads 4,352 are the same kernel with a different length argument, and a silent mismatch is a wrong answer rather than a slow one.
- **`owners`** - a `Vec`, not a flag, because two models can legitimately share one weight buffer (same name, same size, same fingerprint). The second registration is appended rather than the buffer duplicated.
- **`retired`** - a registration released this buffer while a captured CUDA graph was still live. Freeing it then would leave the graph pointing at returned memory, so the bytes stay resident until `reap_retired` runs.

That last one has a *named* outcome, and the naming is the point. A release does not return `()`, it returns which of two things happened:

```rust
pub enum ReleaseOutcome {
    Freed { bytes: usize },
    Retained { bytes: usize, captures: usize },
}
```

`Retained` is a refusal with a reason attached - "I did not free your 33 MB because 2 captures still hold it" - and it is what the negative test asserts. A backend that returned `()` here would have to log the distinction and hope somebody read the log.

**The pool never frees.** That is by design and it is worth saying plainly: the free list grows to the high-water mark of a workload and stays there, because a decode loop's peak is stable and a `cudaFree` in the middle of one is a synchronisation point. The bytes are accounted, they are returned to the *pool*, and the pool's peak is the number Chapter 17's liveness analysis is trying to predict.

### Lesson 3 · Errors as results

CUDA's C API returns an `int` status from every call, and the second-most-popular way to write a CUDA backend is to ignore it. The most popular way is to check it *late* - call `cudaGetLastError()` after a launch and attribute whatever it says to the kernel you just launched - which is worse, because an error from a previous call then gets blamed on the wrong kernel and the trace points at innocent code.

minfer's answer is a C-side helper that names the failure where it is made, and a Rust-side check that turns the sticky record into an `Err`:

<!-- code: src/cuda/kernels/guard.cu::minfer_launch_ok@e844591 -->
```cpp
// Follow-on to #145 (docs/GPU_SAFETY.md rules 4-5). A discarded
// `cudaFuncSetAttribute` return, a `<<<>>>` launch whose error is only seen by a
// *later* `cudaGetLastError()` (attribution by position), and an unchecked
// `cudaGraphDestroy` all let an error latch and resurface at a sync as a phantom
// "kernel launch error". Every dynamic-smem opt-in and launch below goes through
// `minfer_smem_optin()` / `minfer_launch_ok()`, which:
```
<!-- /code -->

The comment goes on to list four properties, and they are the design of the whole layer: read the call's own return value and name the failure where it is made (the site, the instantiation, the requested bytes, the device's queried limit); **skip** a request the device's own limit already excludes rather than making a call that can only fail; **clear the latch** you named, so nothing is left for a later sync to mis-attribute; and return `false` so the caller refuses the launch instead of launching into a checked error.

The Rust side is one check, not 144 signature changes. `minfer_launch_ok` records a sticky failure; `CudaBackend::execute_node` drains it and returns `Err`. That is the payoff the lesson is named for: the discipline lives in one place on each side of the boundary, and no kernel launcher has to be rewritten to participate.

**The honest label for an error you did not cause.** There is a case that cannot be fixed by better checking: an error that was already latched by some earlier call, observed at a sync. minfer prints it as what it is - `"CUDA: latched API error observed by cudaGetLastError: … NOT attributed to a kernel"` - and counts it, rather than blaming the kernel in flight. A message that admits it does not know is worth more than one that guesses.

### Lesson 4 · Bounded waits

This is the lesson where this book's two backends disagree, and the disagreement is instructive rather than an oversight.

Chapter 21's Metal `submit()` waits on a semaphore with a ten-second bound, checks the command buffer's status, and reports the last dispatches it encoded. That is possible because Metal's API has a completion handler and a status property. **CUDA has no synchronise-with-timeout.** There is no argument to `cudaStreamSynchronize` that says "come back in five seconds", so a backend that calls it has decided to wait for as long as the kernel runs:

<!-- code: src/cuda/methods/events.rs::sync@e844591 -->
```rust
        let err = unsafe { cudaGetLastError() };
        if err != 0 {
            LATCHED_API_ERRORS.fetch_add(1, Ordering::Relaxed);
            eprintln!("{}", latched_api_error_message(err));
        }
        let err = unsafe { cudaStreamSynchronize(self.stream()) };
        if err != 0 {
            eprintln!("CUDA stream sync error: {} ({err})", cuda_error_name(err));
        }
```
<!-- /code -->

Two things to notice. The `cudaGetLastError()` before the sync is the attribution fix from Lesson 3 - anything already latched is reported as latched, not as this kernel's fault. And the sync itself is unbounded: if the kernel never returns, this line never returns either. The error path prints to stderr rather than returning a `Result`, which is a second, smaller decision worth naming: at this layer the engine has already decided the step is over, and there is no caller left to hand a `Result` to.

So what does a bounded wait look like on this machine? The measurement is unambiguous:

<!-- facts: cuda-seam.seam-failures -->
| Crossing | What was asked of it | What came back |
|---|---|---|
| allocate | `cudaMalloc` for 1 TiB on a device with none to give | `cudaErrorMemoryAllocation` - "out of memory" |
| wait (blocking) | one kernel that busy-waits 2,000 ms, behind `cudaStreamSynchronize` | the host blocked for **1,924 ms**, and no API to ask it to come back sooner |
| wait (polled) | the same kernel, behind `cudaStreamQuery` against a 500 ms deadline | **500 ms**, then *gave up* |

The first row is the argument for wrapping every call: CUDA's own message is "out of memory" - true, and useless. It does not say what was asked for, by which model, or how much was left. A seam that returns `cudaErrorMemoryAllocation` upward has thrown away the only three facts that make the failure actionable, so the wrapper's job is to put them back.

The last two rows are the same kernel with two different waits, and they disagree by 1,424 ms. There is no `cudaStreamSynchronizeWithTimeout` to call, so a backend that must survive a wedged device owns a clock and polls; one that calls the blocking version has decided that a hung kernel is allowed to hang the process.
<!-- /facts -->

The difference is **1,424 ms of not being able to do anything about it**. The polled version is four lines and owns a clock:

```cuda
while (true) {
    cudaError_t q = cudaStreamQuery(0);
    if (q == cudaSuccess) break;                 // done
    if (q != cudaErrorNotReady) break;            // a real error, not "still running"
    if (now_ms() - t0 >= deadline_ms) {           // the host's own limit
        outcome = "gave up";
        break;
    }
}
```

`cudaErrorNotReady` is the important branch: "the stream is still busy" is not an error, and treating it as one is how a slow kernel becomes a spurious failure. Everything else is a real error and should stop the loop.

**What "gave up" must mean.** A backend that gives up cannot then call `cudaDeviceSynchronize` to clean up - that is the same unbounded wait wearing a different name. The choices, in increasing order of severity: refuse the step and surface an error while the kernel drains on its own; destroy the context (`cudaDeviceReset`), which kills the kernel and every allocation in it, and is only safe if the backend can rebuild everything; or exit. minfer's CUDA path currently takes the first option implicitly - the sync returns when it returns - which is the honest thing to write down and the thing this lesson asks you to improve. The book's requirement is the one from the stub: **a wedged stream produces a timeout error, and the process exits rather than hanging.**

The instrument at the end of this chapter does exactly that: it prints `"gave up"`, exits with code 2, and calls `cudaDeviceReset` on the way out so the next run starts on a clean device.

### Lesson 5 · Linking

The last crossing is not in the source at all. It is in the binary, and it decides what the deployment needs:

<!-- facts: cuda-seam.seam-link -->
| `ldd` on the built binary |
|---|
| `linux-vdso.so.1 (0x0000e4e489c08000)` |
| `libstdc++.so.6 => /lib/aarch64-linux-gnu/libstdc++.so.6 (0x0000e4e489860` |
| `libgcc_s.so.1 => /lib/aarch64-linux-gnu/libgcc_s.so.1 (0x0000e4e48982000` |
| `libc.so.6 => /lib/aarch64-linux-gnu/libc.so.6 (0x0000e4e489660000)` |
| `/lib/ld-linux-aarch64.so.1 (0x0000e4e489bc0000)` |
| `libm.so.6 => /lib/aarch64-linux-gnu/libm.so.6 (0x0000e4e4895b0000)` |

Six lines, and **not one of them is a CUDA library**. No `libcudart.so`, and no `libcuda.so` either - the runtime is inside the executable and the driver is opened by name at run time. That is not luck: it is the default `nvcc` behaviour, and it is the reason a binary built here runs on a machine with a driver and no toolkit. The cost is that a missing driver becomes a *runtime* error instead of a load error, which is exactly the trade this chapter's last lesson is about.
<!-- /facts -->

Six system libraries and **not one CUDA entry**. No `libcudart.so`, and no `libcuda.so` either. The runtime is inside the executable, and the driver is opened by name at run time - which is why the same binary runs on a machine with a driver and no CUDA toolkit installed.

That is a decision, and minfer makes it explicitly in `build.rs`, where the reasoning is written down next to the flag:

<!-- code: build.rs::main@e844591 -->
```rust
    // How is the CUDA runtime linked? Exactly like llama.cpp: a single build
    // flag toggles the cudart archive vs the shared library (there GGML_STATIC
    // picks CUDA::cudart_static vs CUDA::cudart). A static cudart removes the
    // libcudart.so NEEDED dependency, so the binary can run on a host that has
    // only the NVIDIA driver + libstdc++ (no CUDA toolkit runtime). The CUDA
    // driver libcuda.so.1 is not a link dependency in either case — cudart and
    // CudaState::preload_driver() both load it lazily via dlopen at runtime.
```
<!-- /code -->

Three requirements are being traded against each other, and it is worth naming them:

| Link style | Needs at build time | Needs at run time | Fails how |
|---|---|---|---|
| dynamic `libcudart.so` | toolkit + headers | the toolkit's runtime library | **at load**: won't start without it |
| static cudart (this build) | toolkit + headers | only the driver | **at first call**, if the driver is missing |
| `dlopen` by name | nothing | only the driver | **at first call**, with a message you wrote |

minfer lands on the second for the kernels and the third for the driver, which is why `src/cuda/ffi_runtime.rs` declares `dlopen` in its `extern "C"` block at all - and why a build without the CUDA feature still compiles on a machine that has no CUDA: nothing links against it until something asks for it.

The trade is not free. A load-time failure is the *better* failure: the deployment process dies immediately, with the missing soname in the message, instead of a request failing an hour later on a host whose driver is one major version too old. The book's requirement is not "pick static" - it is **know which one you picked**, and the way you know is `ldd`:

```
ldd target/release/minfer | grep -i cuda
```

If that prints nothing, your binary has taken this chapter's route, with this chapter's trade.

## Seam checklist

The table the chapter asks you to fill in, filled in for the engine this book reads:

| Concern | Naive choice | What it costs later | Book's rule | Engine |
|---|---|---|---|---|
| allocation | `cudaMalloc` per op | thousands of sync points per token | pool, keyed on exact byte length | ✅ `DeviceAlloc` |
| ownership | free on drop | a live capture reads freed memory | model-scoped registration with a ledger | ✅ `CudaModelRegistration`, `ReleaseOutcome::Retained` |
| error handling | ignore the code, or check it late | errors blamed on the wrong kernel | name the failure where it is made; one op-level `Err` | ✅ `minfer_launch_ok` + `execute_node` |
| waits | `cudaStreamSynchronize` | a wedged device hangs the process | bound it, and make giving up a first-class outcome | ⚠️ unbounded today - the improvement this chapter asks for |
| linking | whatever the default did | a deployment missing a soname | know it, and check with `ldd` | ✅ static cudart + `dlopen` for the driver |

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

<!-- facts: cuda-seam.provenance -->
*`scripts/seam/seam.cu`; asks the device for a terabyte, then runs one two-second kernel twice - once behind `cudaStreamSynchronize`, once behind a `cudaStreamQuery` poll with a host-owned deadline - and reads `ldd` off the resulting binary; the error name and string for the failed allocation, the host's blocked milliseconds, the poll's elapsed milliseconds and outcome, and the library list. Command: see [the provenance index](../appendix/provenance.md#cuda-seam). Raw numbers: `facts/cuda-seam.json`.*
<!-- /facts -->

## Reading

- code: `src/cuda.rs` (the registration, the ledger, `CudaPtr`), `src/cuda/ffi_runtime.rs`, `src/cuda/methods/events.rs`, `src/cuda/kernels/guard.cu`, `build.rs` at the book's [pinned baseline](../appendix/baseline.md)
- decisions: ADR-0077 (Metal's model registration), ADR-0078 (its device-memory counterpart), ADR-0006 (the per-engine capability gate)
- docs: `docs/GPU_SAFETY.md` rules 4-5, which is where the launch-check discipline was written down first
- [cuda-guide] *CUDA C++ Programming Guide*, NVIDIA - no arXiv ID (vendor manual), the "Error Handling" and "Linking" sections

## Backfill

- [x] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - a decode step's cost is a byte count over a bandwidth, so the seam's job is to not add copies; Lesson 2's pool exists because a `cudaMalloc` in the middle of a step is a sync point.
- [x] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - the device reads the same bytes the file holds, which is why a weight buffer can be registered once and shared by two models (Lesson 2's `owners` list).
- [x] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - this chapter is that boundary from the Rust side: `extern "C"`, the ledger, and `execute_node` as the one place an error becomes an `Err`.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - failure injection: a terabyte that must not allocate, a kernel that must time out, and an `ldd` that must show no CUDA.
