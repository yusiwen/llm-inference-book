// The three failures Chapter 23 says a seam must survive, made to happen on purpose.
//
//   1. allocate      ask for more memory than the device has, and report the request, not just "an error"
//   2. wait          a kernel that runs for two seconds, seen two ways: a blocking synchronise, and a
//                    poll-with-deadline that gives up and says so
//   3. link          (done from the host side by scripts/cuda_seam.py: `ldd` on this binary)
//
// The second is the one worth measuring. `cudaStreamSynchronize` has no timeout argument - there is no API to
// ask a stream to come back in n seconds - so the choice is between blocking for as long as the kernel runs,
// or not using it and polling `cudaStreamQuery` against a deadline you own. This program does both, on the
// same kernel, so the difference is a measurement rather than a recommendation.
//
//   nvcc -O3 -arch=sm_121 -o seam seam.cu && ./seam 2000 500
//
// Arguments: kernel busy-wait in milliseconds, then the host-side deadline in milliseconds.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>

#define CHECK(call)                                                                  \
    do {                                                                             \
        cudaError_t err = (call);                                                    \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error: %s at %s:%d (%s)\n", #call, __FILE__,       \
                    __LINE__, cudaGetErrorString(err));                              \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Busy-wait for `ms` milliseconds on the device, measured in SM clock cycles so the wait does not depend on
// the host's timers. Deliberately not infinite: this program is about a *bounded* wait, and a kernel that
// never returns would make the experiment unmeasurable on a shared machine.
__global__ void k_busy(long long clocks) {
    long long start = clock64();
    while (clock64() - start < clocks) {
        // spin: one warp occupying one SM, the smallest legitimate wedge
    }
}

int main(int argc, char **argv) {
    long long kernel_ms = argc > 1 ? atoll(argv[1]) : 2000;
    double deadline_ms = argc > 2 ? atof(argv[2]) : 500.0;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
    CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));

    printf("{\n");
    printf("  \"device\": \"%s\",\n", prop.name);
    printf("  \"compute_cap\": \"%d.%d\",\n", prop.major, prop.minor);
    printf("  \"sm_mhz\": %d,\n", clock_khz / 1000);

    // ---- 1. a failed allocation that names its request -------------------------------------------------
    size_t want = (size_t)1 << 40;  // 1 TiB: no device this book runs on has it
    void *p = NULL;
    cudaError_t err = cudaMalloc(&p, want);
    const char *name = cudaGetErrorName(err);
    const char *str = cudaGetErrorString(err);
    printf("  \"alloc\": {\"requested_bytes\": %zu, \"ok\": %s, \"error_name\": \"%s\", \"error_string\": \"%s\"},\n",
           want, err == cudaSuccess ? "true" : "false", name ? name : "?", str ? str : "?");
    if (err == cudaSuccess) cudaFree(p);  // a machine with 1 TiB would surprise everyone, including this program
    cudaGetLastError();                   // clear the latch so experiment 2 starts clean

    // ---- 2a. the blocking wait -------------------------------------------------------------------------
    long long clocks = (long long)((double)kernel_ms / 1000.0 * (double)clock_khz * 1000.0);
    double t0 = now_ms();
    k_busy<<<1, 32>>>(clocks);
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(0));
    double blocked_ms = now_ms() - t0;
    printf("  \"blocking_wait\": {\"requested_kernel_ms\": %lld, \"blocked_ms\": %.1f},\n", kernel_ms, blocked_ms);

    // ---- 2b. the same kernel, with a deadline the host owns --------------------------------------------
    t0 = now_ms();
    k_busy<<<1, 32>>>(clocks);
    CHECK(cudaGetLastError());
    double elapsed = 0.0;
    const char *outcome = "completed";
    while (true) {
        cudaError_t q = cudaStreamQuery(0);
        if (q == cudaSuccess) {
            outcome = "completed";
            break;
        }
        if (q != cudaErrorNotReady) {  // a real error, not "still running"
            outcome = cudaGetErrorName(q);
            break;
        }
        elapsed = now_ms() - t0;
        if (elapsed >= deadline_ms) {
            outcome = "gave up";  // the kernel is still running; the host stops waiting for it
            break;
        }
    }
    printf("  \"polled_wait\": {\"deadline_ms\": %.0f, \"elapsed_ms\": %.1f, \"outcome\": \"%s\"},\n",
           deadline_ms, elapsed > 0 ? elapsed : now_ms() - t0, outcome);

    // A real engine would now refuse the step and surface an error. This program has already reported what it
    // needs to; the exit code is the signal that the deadline was hit.
    int gave_up = (outcome[0] == 'g' && outcome[1] == 'a');
    printf("  \"exit\": %d\n}\n", gave_up ? 2 : 0);
    fflush(stdout);
    // The wedged kernel is still resident. There is deliberately no cudaDeviceSynchronize() here - that is the
    // call this experiment is about not making - but the context is torn down before exit so the next run of
    // this program starts from a clean device rather than on top of a kernel still spinning.
    cudaDeviceReset();
    return gave_up ? 2 : 0;
}
