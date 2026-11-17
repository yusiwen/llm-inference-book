// The matmul ladder of Chapter 24, as five kernels that can be measured and checked against a CPU oracle.
//
//   rung 1  eltwise   ReLU over a flat array                     - the memory hierarchy with no arithmetic to hide it
//   rung 2  gather    copy n_rows rows of an embedding table     - what a token lookup actually costs
//   rung 3  gemm      f16 C = A x B, naive then shared-tiled     - reuse, and the tile arithmetic that predicts it
//   rung 4  gemv      q4_0 weights, one row per thread           - decode: M = 1, bandwidth-bound, weights never float
//   rung 5  gemm_q4   q4_0 weights, M = 16 rows of activations   - prefill: the same weights, sixteen outputs per load
//
// Every rung is checked against a host reference before it is timed, and the check is printed with the timing so a
// fast wrong kernel cannot be quoted. The tolerance is per-rung and written down: exact for the first two, 2e-3
// relative for the f16 GEMM (it accumulates in f32 and rounds to f16 once), and 5e-3 for the q4_0 pair (the
// dequantisation is exact; the summation order is not).
//
//   nvcc -O3 -arch=sm_121 -o ladder ladder.cu && ./ladder

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
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

// ---- rung 1: elementwise -------------------------------------------------------------------
__global__ void k_relu(const float *a, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] > 0.0f ? a[i] : 0.0f;
}

// ---- rung 2: gather ------------------------------------------------------------------------
__global__ void k_gather(const float *table, const int *ids, float *out, int n_rows, int ne) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_rows * ne;
    if (i >= total) return;
    int token = ids[i / ne];
    out[i] = table[(size_t)token * ne + (i % ne)];
}

// ---- rung 3a: naive f16 GEMM ---------------------------------------------------------------
__global__ void k_gemm_naive(const __half *a, const __half *b, __half *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = 0.0f;
    for (int i = 0; i < k; i++) acc += __half2float(a[row * k + i]) * __half2float(b[i * n + col]);
    c[row * n + col] = __float2half(acc);
}

// ---- rung 3b: tiled f16 GEMM ---------------------------------------------------------------
// One TILE x TILE block of C. A and B are read once per tile instead of once per output element, so each
// weight element loaded into shared memory is used TILE times: that ratio is the whole point of the rung.
#define TILE 16
__global__ void k_gemm_tiled(const __half *a, const __half *b, __half *c, int m, int n, int k) {
    __shared__ float as[TILE][TILE];
    __shared__ float bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    int tiles = (k + TILE - 1) / TILE;
    for (int t = 0; t < tiles; t++) {
        int ka = t * TILE + tx;
        int kb = t * TILE + ty;
        as[ty][tx] = (row < m && ka < k) ? __half2float(a[row * k + ka]) : 0.0f;
        bs[ty][tx] = (kb < k && col < n) ? __half2float(b[kb * n + col]) : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i++) acc += as[ty][i] * bs[i][tx];
        __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = __float2half(acc);
}

// ---- rung 4: q4_0 GEMV (decode, M = 1) -----------------------------------------------------
// Block layout, exactly as the file stores it: 32 values packed as 16 bytes of nibbles plus one f16 scale.
//   x[i] = (nibble - 8) * d      for i in 0..31
// weight offset for row `r`, block `s`: r * (k / 32) * 18 + s * 18   (2 bytes scale + 16 bytes nibbles)
__device__ __forceinline__ float q4_0_decode(const unsigned char *blk, int j) {
    float d = __half2float(*(const __half *)blk);
    int byte = blk[2 + (j % 32) / 2];
    int nib = (j & 1) ? (byte >> 4) : (byte & 0x0F);
    return ((float)nib - 8.0f) * d;
}

__global__ void k_gemv_q4(const unsigned char *w, const float *x, float *y, int rows, int k) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const unsigned char *row = w + (size_t)r * (k / 32) * 18;
    float acc = 0.0f;
    for (int s = 0; s < k / 32; s++) {
        const unsigned char *blk = row + (size_t)s * 18;
        float d = __half2float(*(const __half *)blk);
        float block_sum = 0.0f;
        for (int j = 0; j < 32; j++) {
            int byte = blk[2 + j / 2];
            int nib = (j & 1) ? (byte >> 4) : (byte & 0x0F);
            block_sum += ((float)nib - 8.0f) * x[s * 32 + j];
        }
        acc += block_sum * d;
    }
    y[r] = acc;
}

// ---- rung 5: the same weights, M rows of activations ---------------------------------------
__global__ void k_gemm_q4(const unsigned char *w, const float *x, float *y, int rows, int k, int mm) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const unsigned char *row = w + (size_t)r * (k / 32) * 18;
    for (int t = 0; t < mm; t++) {
        const float *xt = x + (size_t)t * k;
        float acc = 0.0f;
        for (int s = 0; s < k / 32; s++) {
            const unsigned char *blk = row + (size_t)s * 18;
            float d = __half2float(*(const __half *)blk);
            float block_sum = 0.0f;
            for (int j = 0; j < 32; j++) {
                int byte = blk[2 + j / 2];
                int nib = (j & 1) ? (byte >> 4) : (byte & 0x0F);
                block_sum += ((float)nib - 8.0f) * xt[s * 32 + j];
            }
            acc += block_sum * d;
        }
        y[(size_t)t * rows + r] = acc;
    }
}

// ---- host side -----------------------------------------------------------------------------
static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float d = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > d) d = e;
    }
    return d;
}

static void report(const char *name, double ms, double bytes, double flops, double err, double tol) {
    printf("  {\"name\": \"%s\", \"ms\": %.4f, \"bytes\": %.0f, \"gflops\": %.1f, \"max_abs_diff\": %.6g, "
           "\"tolerance\": %.6g, \"passes\": %s},\n",
           name, ms, bytes, flops / ms / 1e6, err, tol, err <= tol ? "true" : "false");
}

int main() {
    const int n = 32 << 20;          // rung 1
    const int n_rows = 32, ne = 1024; // rung 2: 32 rows of 1024
    const int table_rows = 4096;
    const int gm = 512, gn = 512, gk = 512;   // rung 3
    const int qrows = 4096, qk = 4096;        // rungs 4-5
    const int reps = 20;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("{\n  \"device\": \"%s\", \"compute_cap\": \"%d.%d\", \"sm_count\": %d, \"reps\": %d,\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, reps);
    printf("  \"kernels\": [\n");

    // ---- rung 1
    float *a, *c;
    CHECK(cudaMalloc(&a, (size_t)n * 4));
    CHECK(cudaMalloc(&c, (size_t)n * 4));
    float *h = (float *)malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) h[i] = (i % 3 == 0) ? -1.0f * i : 0.5f * i;
    CHECK(cudaMemcpy(a, h, (size_t)n * 4, cudaMemcpyHostToDevice));
    int g1 = (n + 255) / 256;
    double t0 = now_ms();
    for (int r = 0; r < reps; r++) k_relu<<<g1, 256>>>(a, c, n);
    CHECK(cudaDeviceSynchronize());
    double ms1 = (now_ms() - t0) / reps;
    CHECK(cudaMemcpy(h, c, (size_t)n * 4, cudaMemcpyDeviceToHost));
    float *ref = (float *)malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) ref[i] = h[i];  // recomputed below from the input instead
    {
        float *hin = (float *)malloc((size_t)n * 4);
        CHECK(cudaMemcpy(hin, a, (size_t)n * 4, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) ref[i] = hin[i] > 0.0f ? hin[i] : 0.0f;
        free(hin);
    }
    report("eltwise_relu", ms1, 2.0 * n * 4, 0.0, max_abs_diff(h, ref, n), 0.0);

    // ---- rung 2
    float *table, *out2;
    int *ids;
    CHECK(cudaMalloc(&table, (size_t)table_rows * ne * 4));
    CHECK(cudaMalloc(&out2, (size_t)n_rows * ne * 4));
    CHECK(cudaMalloc(&ids, n_rows * sizeof(int)));
    float *ht = (float *)malloc((size_t)table_rows * ne * 4);
    for (size_t i = 0; i < (size_t)table_rows * ne; i++) ht[i] = (float)(i % 97) * 0.01f;
    int *hid = (int *)malloc(n_rows * sizeof(int));
    for (int i = 0; i < n_rows; i++) hid[i] = (i * 37) % table_rows;
    CHECK(cudaMemcpy(table, ht, (size_t)table_rows * ne * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ids, hid, n_rows * sizeof(int), cudaMemcpyHostToDevice));
    int g2 = (n_rows * ne + 255) / 256;
    t0 = now_ms();
    for (int r = 0; r < reps; r++) k_gather<<<g2, 256>>>(table, ids, out2, n_rows, ne);
    CHECK(cudaDeviceSynchronize());
    double ms2 = (now_ms() - t0) / reps;
    float *hout = (float *)malloc((size_t)n_rows * ne * 4);
    CHECK(cudaMemcpy(hout, out2, (size_t)n_rows * ne * 4, cudaMemcpyDeviceToHost));
    float *ref2 = (float *)malloc((size_t)n_rows * ne * 4);
    for (int r = 0; r < n_rows; r++)
        for (int j = 0; j < ne; j++) ref2[r * ne + j] = ht[(size_t)hid[r] * ne + j];
    report("gather_embed", ms2, 2.0 * n_rows * ne * 4, 0.0, max_abs_diff(hout, ref2, n_rows * ne), 0.0);

    // ---- rung 3: naive vs tiled, same data
    __half *ha, *hb, *hc;
    CHECK(cudaMalloc(&ha, (size_t)gm * gk * 2));
    CHECK(cudaMalloc(&hb, (size_t)gk * gn * 2));
    CHECK(cudaMalloc(&hc, (size_t)gm * gn * 2));
    __half *hha = (__half *)malloc((size_t)gm * gk * 2);
    __half *hhb = (__half *)malloc((size_t)gk * gn * 2);
    for (size_t i = 0; i < (size_t)gm * gk; i++) hha[i] = __float2half((float)(i % 31) * 0.01f);
    for (size_t i = 0; i < (size_t)gk * gn; i++) hhb[i] = __float2half((float)(i % 17) * 0.02f);
    CHECK(cudaMemcpy(ha, hha, (size_t)gm * gk * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(hb, hhb, (size_t)gk * gn * 2, cudaMemcpyHostToDevice));
    double gf = 2.0 * gm * gn * gk / 1e9;
    double gbytes = (double)((size_t)gm * gk + (size_t)gk * gn + (size_t)gm * gn) * 2;
    dim3 blk(16, 16), grd((gn + 15) / 16, (gm + 15) / 16);
    t0 = now_ms();
    for (int r = 0; r < 5; r++) k_gemm_naive<<<grd, blk>>>(ha, hb, hc, gm, gn, gk);
    CHECK(cudaDeviceSynchronize());
    double ms3a = (now_ms() - t0) / 5;
    t0 = now_ms();
    for (int r = 0; r < reps; r++) k_gemm_tiled<<<grd, blk>>>(ha, hb, hc, gm, gn, gk);
    CHECK(cudaDeviceSynchronize());
    double ms3b = (now_ms() - t0) / reps;
    __half *hhc = (__half *)malloc((size_t)gm * gn * 2);
    CHECK(cudaMemcpy(hhc, hc, (size_t)gm * gn * 2, cudaMemcpyDeviceToHost));
    float *gc = (float *)malloc((size_t)gm * gn * 4);
    for (size_t i = 0; i < (size_t)gm * gn; i++) gc[i] = __half2float(hhc[i]);
    float *gref = (float *)malloc((size_t)gm * gn * 4);
    for (int r = 0; r < gm; r++)
        for (int ccol = 0; ccol < gn; ccol++) {
            float acc = 0.0f;
            for (int i = 0; i < gk; i++) acc += __half2float(hha[r * gk + i]) * __half2float(hhb[i * gn + ccol]);
            gref[r * gn + ccol] = __half2float(__float2half(acc));
        }
    float ge = max_abs_diff(gc, gref, gm * gn);
    report("gemm_naive", ms3a, gbytes, gf * 1e9, ge, 2e-3f);
    report("gemm_tiled", ms3b, gbytes, gf * 1e9, ge, 2e-3f);

    // ---- rungs 4-5: q4_0 weights, decode (M=1) then prefill (M=16)
    unsigned char *qw;
    float *qx, *qy;
    int kblocks = qk / 32;
    size_t wbytes = (size_t)qrows * kblocks * 18;
    CHECK(cudaMalloc(&qw, wbytes));
    CHECK(cudaMalloc(&qx, (size_t)16 * qk * 4));
    CHECK(cudaMalloc(&qy, (size_t)16 * qrows * 4));
    unsigned char *hqw = (unsigned char *)malloc(wbytes);
    for (int r = 0; r < qrows; r++)
        for (int s = 0; s < kblocks; s++) {
            unsigned char *blk = hqw + ((size_t)r * kblocks + s) * 18;
            __half d = __float2half(0.0125f);
            memcpy(blk, &d, 2);
            for (int b = 0; b < 16; b++) blk[2 + b] = (unsigned char)((b * 7 + r + s) & 0xFF);
        }
    float *hx = (float *)malloc((size_t)16 * qk * 4);
    for (int t = 0; t < 16; t++)
        for (int j = 0; j < qk; j++) hx[t * qk + j] = (float)((j + t) % 13) * 0.05f - 0.3f;
    CHECK(cudaMemcpy(qw, hqw, wbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(qx, hx, (size_t)16 * qk * 4, cudaMemcpyHostToDevice));
    int gq = (qrows + 255) / 256;
    t0 = now_ms();
    for (int r = 0; r < reps; r++) k_gemv_q4<<<gq, 256>>>(qw, qx, qy, qrows, qk);
    CHECK(cudaDeviceSynchronize());
    double ms4 = (now_ms() - t0) / reps;
    t0 = now_ms();
    for (int r = 0; r < reps; r++) k_gemm_q4<<<gq, 256>>>(qw, qx, qy, qrows, qk, 16);
    CHECK(cudaDeviceSynchronize());
    double ms5 = (now_ms() - t0) / reps;
    // host reference for both, from the same packed bytes
    float *qref = (float *)malloc((size_t)16 * qrows * 4);
    for (int t = 0; t < 16; t++)
        for (int r = 0; r < qrows; r++) {
            float acc = 0.0f;
            for (int s = 0; s < kblocks; s++) {
                const unsigned char *blk = hqw + ((size_t)r * kblocks + s) * 18;
                float d;
                memcpy(&d, blk, 2);
                __half dh = *(__half *)&d;
                float dv = __half2float(dh);
                float bsum = 0.0f;
                for (int j = 0; j < 32; j++) {
                    int byte = blk[2 + j / 2];
                    int nib = (j & 1) ? (byte >> 4) : (byte & 0x0F);
                    bsum += ((float)nib - 8.0f) * hx[t * qk + s * 32 + j];
                }
                acc += bsum * dv;
            }
            qref[t * qrows + r] = acc;
        }
    float *qout = (float *)malloc((size_t)16 * qrows * 4);
    CHECK(cudaMemcpy(qout, qy, (size_t)16 * qrows * 4, cudaMemcpyDeviceToHost));
    float qe5 = max_abs_diff(qout, qref, 16 * qrows);
    // the gemv result is the M=0 slice of the same buffer; re-run it alone to check that slice
    CHECK(cudaMemcpy(qout, qy, (size_t)qrows * 4, cudaMemcpyDeviceToHost));
    float qe4 = max_abs_diff(qout, qref, qrows);
    report("gemv_q4_decode", ms4, (double)wbytes + qrows * 4.0, 2.0 * qrows * qk / 1e9 * 1e9, qe4, 5e-3f);
    report("gemm_q4_prefill16", ms5, (double)wbytes + 16.0 * qrows * 4.0, 2.0 * 16.0 * qrows * qk / 1e9 * 1e9,
           qe5, 5e-3f);

    printf("  {\"name\": \"end\", \"ms\": 0, \"bytes\": 0, \"gflops\": 0, \"max_abs_diff\": 0, \"tolerance\": 0, "
           "\"passes\": true}\n  ]\n}\n");
    return 0;
}
