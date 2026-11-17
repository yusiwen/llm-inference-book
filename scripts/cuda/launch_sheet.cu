// The launch sheet of Chapter 22, as an executable.
//
// Four kernels, one per row of the chapter's table, plus the streaming copy that sets the bandwidth every
// prediction is divided by. Each one is launched with the geometry written down in the chapter first:
//
//   copy              N floats in, N out            bytes = 8N
//   vecadd            three arrays                  bytes = 12N
//   vecadd_divergent  the same three arrays, one branch that splits every warp   bytes = 12N
//   relu              two arrays                    bytes = 8N
//   gemv_row          M rows of K f32, one thread per row   bytes = 4MK + 4K + 4M
//
// The point of the exercise is the last two columns of the table: a prediction from bytes and bandwidth, and
// the measured time. Where they disagree, the reason is in the access pattern, not in the arithmetic - which
// is the whole of Lesson 4.
//
// Build and run (CUDA 13, sm_121 on the DGX Spark):
//   nvcc -O3 -arch=sm_121 -o launch_sheet launch_sheet.cu
//   ./launch_sheet            # JSON on stdout

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                                  \
    do {                                                                             \
        cudaError_t err = (call);                                                    \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error: %s at %s:%d (%s)\n", #call, __FILE__,        \
                    __LINE__, cudaGetErrorString(err));                              \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

__global__ void k_copy(const float *a, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i];
}

__global__ void k_vecadd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// The divergent twin: neighbouring threads take different sides of one branch, so a warp executes both.
__global__ void k_vecadd_divergent(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (a[i] > 0.0f) {
            c[i] = a[i] * 2.0f;
        } else {
            c[i] = a[i] * b[i];
        }
    }
}

// The same data, but now the two sides are not symmetric: half of every warp runs a 32-iteration loop and the
// other half does one multiply. This is the divergence the first variant does *not* show, because a compiler
// turns a symmetric if-else into two predicated instructions and the warp pays nothing.
__global__ void k_vecadd_branchy(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i];
    if (v > 0.0f) {
        for (int r = 0; r < 32; r++) v = v * 1.000001f + b[i];
        c[i] = v;
    } else {
        c[i] = v * b[i];
    }
}

__global__ void k_relu(const float *a, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] > 0.0f ? a[i] : 0.0f;
}

// One thread per row, each walking K contiguous floats. Reads are coalesced *within* a row and strided
// *across* a warp, which is exactly the shape of a decode matmul: it is the chapter's honest mismatch.
__global__ void k_gemv_row(const float *w, const float *x, float *y, int m, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    const float *r = w + (size_t)row * k;
    float acc = 0.0f;
    for (int j = 0; j < k; j++) acc += r[j] * x[j];
    y[row] = acc;
}

static void report(const char *name, float ms, size_t bytes, int block) {
    double gb_s = bytes / ms / 1e6;  // bytes/ms -> GB/s
    printf("  {\"name\": \"%s\", \"ms\": %.5f, \"bytes\": %zu, \"gb_s\": %.1f, \"block\": %d},\n",
           name, ms, bytes, gb_s, block);
}

#define BENCH(label, bytes, launches)                                                     \
    do {                                                                                  \
        for (int warm = 0; warm < 5; warm++) { launches; }                                \
        CHECK(cudaDeviceSynchronize());                                                   \
        cudaEvent_t t0, t1;                                                               \
        CHECK(cudaEventCreate(&t0));                                                      \
        CHECK(cudaEventCreate(&t1));                                                      \
        CHECK(cudaEventRecord(t0));                                                       \
        for (int r = 0; r < reps; r++) { launches; }                                      \
        CHECK(cudaEventRecord(t1));                                                       \
        CHECK(cudaEventSynchronize(t1));                                                  \
        float total = 0.0f;                                                               \
        CHECK(cudaEventElapsedTime(&total, t0, t1));                                      \
        report(label, total / reps, bytes, 256);                                     \
        cudaEventDestroy(t0);                                                             \
        cudaEventDestroy(t1);                                                             \
    } while (0)

int main(int argc, char **argv) {
    const int n = 32 << 20;   // 32 Mi floats: 128 MiB per array
    const int m = 8192, k = 1024;
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    const int block = 256;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("{\n");
    printf("  \"device\": \"%s\",\n", prop.name);
    printf("  \"compute_cap\": \"%d.%d\",\n", prop.major, prop.minor);
    printf("  \"sm_count\": %d,\n", prop.multiProcessorCount);
    int clock_khz = 0;
    CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    printf("  \"max_sm_mhz\": %d,\n", clock_khz / 1000);
    printf("  \"n\": %d, \"m\": %d, \"k\": %d, \"reps\": %d, \"block\": %d,\n", n, m, k, reps, block);

    float *a, *b, *c;
    CHECK(cudaMalloc(&a, sizeof(float) * n));
    CHECK(cudaMalloc(&b, sizeof(float) * n));
    CHECK(cudaMalloc(&c, sizeof(float) * n));
    float *h = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) h[i] = (i % 2 == 0) ? 1.0f : -1.0f;  // every warp straddles the branch
    CHECK(cudaMemcpy(a, h, sizeof(float) * n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b, h, sizeof(float) * n, cudaMemcpyHostToDevice));

    float *w, *x, *y;
    CHECK(cudaMalloc(&w, sizeof(float) * (size_t)m * k));
    CHECK(cudaMalloc(&x, sizeof(float) * k));
    CHECK(cudaMalloc(&y, sizeof(float) * m));
    CHECK(cudaMemset(w, 0, sizeof(float) * (size_t)m * k));
    CHECK(cudaMemset(x, 0, sizeof(float) * k));
    CHECK(cudaMemset(y, 0, sizeof(float) * m));

    int grid_n = (n + block - 1) / block;
    int grid_m = (m + block - 1) / block;

    printf("  \"kernels\": [\n");
    BENCH("copy", (size_t)sizeof(float) * n * 2, (k_copy<<<grid_n, block>>>(a, c, n)));
    BENCH("vecadd", (size_t)sizeof(float) * n * 3, (k_vecadd<<<grid_n, block>>>(a, b, c, n)));
    BENCH("vecadd_divergent", (size_t)sizeof(float) * n * 3,
          (k_vecadd_divergent<<<grid_n, block>>>(a, b, c, n)));
    BENCH("vecadd_branchy", (size_t)sizeof(float) * n * 3,
          (k_vecadd_branchy<<<grid_n, block>>>(a, b, c, n)));
    BENCH("relu", (size_t)sizeof(float) * n * 2, (k_relu<<<grid_n, block>>>(a, c, n)));
    BENCH("gemv_row", (size_t)sizeof(float) * ((size_t)m * k + k + m),
          (k_gemv_row<<<grid_m, block>>>(w, x, y, m, k)));
    printf("  {\"name\": \"end\", \"ms\": 0, \"bytes\": 0, \"gb_s\": 0, \"block\": 0}\n");
    printf("  ]\n}\n");

    CHECK(cudaFree(a)); CHECK(cudaFree(b)); CHECK(cudaFree(c));
    CHECK(cudaFree(w)); CHECK(cudaFree(x)); CHECK(cudaFree(y));
    free(h);
    return 0;
}
