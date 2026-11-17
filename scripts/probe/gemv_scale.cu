// Does Chapter 22's claim survive a second look?
//
// Chapter 22 measured a row-per-thread GEMV at 64 per cent of the machine's streaming rate and attributed the
// gap to the access pattern: the 32 lanes of a warp walk 32 different rows, so every load is a strided read.
// That explanation was never tested against an alternative kernel, and the measurement was taken at one size.
// This probe does both:
//
//   per_row     one thread per output row, walking that row in order - exactly the shape Chapter 22 measured
//   per_warp    one warp per output row, the 32 lanes splitting the columns and reducing with __shfl_down -
//               the same arithmetic, a coalesced access pattern
//
// Each is run at three sizes and reported against the streaming rate measured on the same device in the same
// process, so a 64 per cent can be compared with a 64 per cent on equal terms.
//
//   nvcc -O3 -arch=sm_121 -o gemv_scale gemv_scale.cu && ./gemv_scale

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>

#define CHECK(call)                                                                  \
    do {                                                                             \
        cudaError_t err = (call);                                                    \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error: %s at %s:%d (%s)\n", #call, __FILE__,        \
                    __LINE__, cudaGetErrorString(err));                              \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// The streaming rate, measured in this process rather than imported from a chapter.
__global__ void k_copy(const float *src, float *dst, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = src[i];
}

// A: one thread per row. Lane l of warp w works on row (w*32 + l); each reads its own row in order, so at any
// instant the 32 lanes are 4 KB apart.
__global__ void k_per_row(const float *w, const float *x, float *y, int rows, int k) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float *row = w + (size_t)r * k;
    float acc = 0.0f;
    for (int j = 0; j < k; j++) acc += row[j] * x[j];
    y[r] = acc;
}

// B: one warp per row. Lane l takes columns l, l+32, l+64 ... so the 32 lanes read 32 consecutive floats on
// every step - one 128-byte transaction instead of 32 separate ones - then the lanes reduce with __shfl_down.
__global__ void k_per_warp(const float *w, const float *x, float *y, int rows, int k) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const float *row = w + (size_t)warp * k;
    float acc = 0.0f;
    for (int j = lane; j < k; j += 32) acc += row[j] * x[j];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) y[warp] = acc;
}

static double bench_copy(const float *a, float *b, size_t n, int reps) {
    int g = (int)((n + 255) / 256);
    if (g > 65535) g = 65535;
    CHECK(cudaDeviceSynchronize());
    double t0 = now_ms();
    for (int r = 0; r < reps; r++) k_copy<<<g, 256>>>(a, b, n);
    CHECK(cudaDeviceSynchronize());
    return (now_ms() - t0) / reps;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("{\n  \"device\": \"%s\", \"sm_count\": %d,\n", prop.name, prop.multiProcessorCount);

    // streaming rate: a 256 MB copy, as Chapter 22 did
    size_t cn = 64u * 1024u * 1024u;  // floats -> 256 MB read + 256 MB written
    float *ca, *cb;
    CHECK(cudaMalloc(&ca, cn * 4));
    CHECK(cudaMalloc(&cb, cn * 4));
    CHECK(cudaMemset(ca, 0, cn * 4));
    double copy_ms = bench_copy(ca, cb, cn, 20);
    double copy_gbs = (double)cn * 4 * 2 / copy_ms / 1e6;
    printf("  \"copy_256mb_ms\": %.4f, \"stream_gbs\": %.2f,\n", copy_ms, copy_gbs);
    CHECK(cudaFree(ca));
    CHECK(cudaFree(cb));

    const int sizes[3] = {1024, 4096, 8192};
    printf("  \"sizes\": [1024, 4096, 8192],\n  \"results\": [\n");
    int first = 1;
    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        size_t words = (size_t)n * n;
        float *w, *x, *y, *y2;
        CHECK(cudaMalloc(&w, words * 4));
        CHECK(cudaMalloc(&x, (size_t)n * 4));
        CHECK(cudaMalloc(&y, (size_t)n * 4));
        CHECK(cudaMalloc(&y2, (size_t)n * 4));
        // a pattern that is not all zeroes, so the dot products are not trivially equal
        CHECK(cudaMemset(w, 0x3c, words * 4));
        CHECK(cudaMemset(x, 0x3c, (size_t)n * 4));
        int reps = n <= 1024 ? 40 : (n <= 4096 ? 10 : 4);

        double t0 = now_ms();
        for (int r = 0; r < reps; r++) k_per_row<<<(n + 255) / 256, 256>>>(w, x, y, n, n);
        CHECK(cudaDeviceSynchronize());
        double row_ms = (now_ms() - t0) / reps;

        double t1 = now_ms();
        for (int r = 0; r < reps; r++) k_per_warp<<<((n * 32) + 255) / 256, 256>>>(w, x, y2, n, n);
        CHECK(cudaDeviceSynchronize());
        double warp_ms = (now_ms() - t1) / reps;

        // do the two agree?
        float *hy = (float *)malloc((size_t)n * 4), *hy2 = (float *)malloc((size_t)n * 4);
        CHECK(cudaMemcpy(hy, y, (size_t)n * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hy2, y2, (size_t)n * 4, cudaMemcpyDeviceToHost));
        float worst = 0.0f, denom = 0.0f;
        for (int i = 0; i < n; i++) {
            float d = fabsf(hy[i] - hy2[i]);
            if (d > worst) worst = d;
            if (fabsf(hy[i]) > denom) denom = fabsf(hy[i]);
        }
        double bytes = (double)words * 4;
        if (!first) printf(",\n");
        first = 0;
        printf("    {\"size\": %d, \"per_row_ms\": %.4f, \"per_row_gbs\": %.1f, \"per_row_of_stream\": %.1f, "
               "\"per_warp_ms\": %.4f, \"per_warp_gbs\": %.1f, \"per_warp_of_stream\": %.1f, "
               "\"ratio_row_over_warp\": %.2f, \"max_rel_diff\": %.2e}",
               n, row_ms, bytes / row_ms / 1e6, bytes / row_ms / 1e6 / copy_gbs * 100,
               warp_ms, bytes / warp_ms / 1e6, bytes / warp_ms / 1e6 / copy_gbs * 100,
               row_ms / warp_ms, worst / (denom > 0 ? denom : 1));
        free(hy); free(hy2);
        CHECK(cudaFree(w)); CHECK(cudaFree(x)); CHECK(cudaFree(y)); CHECK(cudaFree(y2));
    }
    printf("\n  ]\n}\n");
    return 0;
}
