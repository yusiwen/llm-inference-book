// gpubw.cu - device-side streaming bandwidth, the number Chapter 1 actually needs for a GPU path.
//
// The CPU-side number (scripts/membw.c) answers "how fast can the host move bytes". A GPU backend
// does not move bytes that way: it reads weights through the device's own memory system. On a
// unified-memory part like the GB10 the two numbers differ by more than a factor of two, and the
// decode ceiling has to use the one the kernels actually run against.
//
// Build and run:
//   nvcc -O3 -arch=sm_121 -o gpubw scripts/gpubw.cu     # sm_121 is the GB10
//   ./gpubw 1024                                        # 1024 MiB, prints read and copy GB/s
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)

__global__ void copy_k(const float *__restrict__ in, float *__restrict__ out, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = in[i];
}

__global__ void read_k(const float *__restrict__ in, float *__restrict__ sink, size_t n) {
    float acc = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        acc += in[i];
    if (acc == 1234.5678f) sink[0] = acc;   // never true: keeps the loop, costs no write
}

int main(int argc, char **argv) {
    size_t mib = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 1024;
    size_t n = mib * 1024 * 1024 / sizeof(float);
    float *a, *b, *sink;
    CHECK(cudaMalloc(&a, n * sizeof(float)));
    CHECK(cudaMalloc(&b, n * sizeof(float)));
    CHECK(cudaMalloc(&sink, sizeof(float)));
    CHECK(cudaMemset(a, 1, n * sizeof(float)));

    int dev = 0, blocks = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaGetDeviceProperties(&prop, dev));
    blocks = prop.multiProcessorCount * 8;
    const int threads = 256;

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // Two passes, second reported: the first pays for page mapping and clock ramp.
    float copy_gbs = 0, read_gbs = 0;
    for (int pass = 0; pass < 2; pass++) {
        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 5; r++) copy_k<<<blocks, threads>>>(a, b, n);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms = 0; CHECK(cudaEventElapsedTime(&ms, t0, t1));
        copy_gbs = (2.0f * (float)n * sizeof(float) * 5) / (ms * 1e6f);

        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 5; r++) read_k<<<blocks, threads>>>(a, sink, n);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        read_gbs = ((float)n * sizeof(float) * 5) / (ms * 1e6f);
    }

    std::printf("gpubw %zu MiB on %s (%d SMs, %d blocks x %d threads): read %.1f GB/s, copy %.1f GB/s\n",
                mib, prop.name, prop.multiProcessorCount, blocks, threads, read_gbs, copy_gbs);
    return 0;
}
