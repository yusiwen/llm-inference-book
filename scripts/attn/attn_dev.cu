// Chapter 25's three device measurements: the score matrix that does not fit, decode attention over a growing
// cache, and the launch overhead a CUDA graph removes.
//
//   1. prefill, materialised   one thread per query row, the whole score row written to global memory and read
//                              back - the thing flash attention exists to avoid. The artifact records both the
//                              time and the bytes of that score buffer, because the buffer is the point.
//   2. prefill, streamed       the same row computed in blocks with a running max and running sum, so the row
//                              never leaves registers. Equal results, no score buffer.
//   3. decode attention        one query against a cache of 128 then 8,192 keys: the per-step byte count and
//                              the bandwidth it actually reaches, which is the number Chapter 18's cache exists
//                              to keep small.
//   4. graph replay            100 small launches, timed as launches and then as one captured graph replayed -
//                              the per-step overhead a graph removes, measured instead of asserted.
//
//   nvcc -O3 -arch=sm_121 -o attn_dev attn_dev.cu && ./attn_dev

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
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

// ---- 1. prefill, materialised: S goes to global memory --------------------------------------
__global__ void k_prefill_materialised(const float *q, const float *k, const float *v, float *o,
                                       float *s_out, int seq, int hd) {
    int row = blockIdx.x;
    if (row >= seq) return;
    float *s = s_out + (size_t)row * seq;
    for (int j = 0; j <= row; j++) {  // causal: only the past
        float acc = 0.0f;
        for (int d = 0; d < hd; d++) acc += q[row * hd + d] * k[j * hd + d];
        s[j] = acc;
    }
    float mx = -INFINITY;
    for (int j = 0; j <= row; j++) mx = fmaxf(mx, s[j]);
    float sum = 0.0f;
    for (int j = 0; j <= row; j++) { s[j] = expf(s[j] - mx); sum += s[j]; }
    for (int d = 0; d < hd; d++) {
        float acc = 0.0f;
        for (int j = 0; j <= row; j++) acc += s[j] * v[j * hd + d];
        o[row * hd + d] = acc / sum;
    }
}

// ---- 2. prefill, streamed: running max and running sum, nothing materialised ----------------
#define BC 64
__global__ void k_prefill_streamed(const float *q, const float *k, const float *v, float *o,
                                   int seq, int hd) {
    int row = blockIdx.x;
    if (row >= seq) return;
    __shared__ float s_score[BC];
    __shared__ float s_corr;
    __shared__ float s_acc[256];
    float m = -INFINITY, l = 0.0f;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) s_acc[d] = 0.0f;
    __syncthreads();
    for (int j0 = 0; j0 <= row; j0 += BC) {
        int nb = min(BC, row - j0 + 1);
        if (threadIdx.x < nb) {
            int j = j0 + threadIdx.x;
            float acc = 0.0f;
            for (int d = 0; d < hd; d++) acc += q[row * hd + d] * k[j * hd + d];
            s_score[threadIdx.x] = acc;
        }
        __syncthreads();
        // one thread updates the running max and sum, then scales the accumulator - the online softmax
        if (threadIdx.x == 0) {
            float mx = m;
            for (int i = 0; i < nb; i++) mx = fmaxf(mx, s_score[i]);
            float corr = expf(m - mx);
            float lsum = 0.0f;
            for (int i = 0; i < nb; i++) {
                s_score[i] = expf(s_score[i] - mx);
                lsum += s_score[i];
            }
            l = l * corr + lsum;
            s_corr = corr;  // hand the correction to the whole block, in its own slot
            m = mx;
        }
        __syncthreads();
        float corr = s_corr;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) {
            float acc = s_acc[d] * corr;
            for (int i = 0; i < nb; i++) acc += s_score[i] * v[(j0 + i) * hd + d];
            s_acc[d] = acc;
        }
        __syncthreads();
    }
    __shared__ float s_l;
    if (threadIdx.x == 0) s_l = l;
    __syncthreads();
    for (int d = threadIdx.x; d < hd; d += blockDim.x) o[row * hd + d] = s_acc[d] / s_l;
}

// ---- 3. decode attention: one query, a long cache --------------------------------------------
__global__ void k_decode_attn(const float *q, const float *k, const float *v, float *o,
                              int ctx, int hd) {
    // The cache is split across the block: each thread owns a slice of the keys and keeps its own running max
    // and sum; lane 0 then merges the partials with the same online-softmax correction the prefill kernel uses.
    __shared__ float s_q[256];
    __shared__ float s_m[256], s_l[256];
    if (threadIdx.x < hd) s_q[threadIdx.x] = q[threadIdx.x];
    __syncthreads();
    float m = -INFINITY, l = 0.0f;
    for (int j = threadIdx.x; j < ctx; j += blockDim.x) {
        float s = 0.0f;
        for (int d = 0; d < hd; d++) s += s_q[d] * k[j * hd + d];
        float nm = fmaxf(m, s);
        l = l * expf(m - nm) + expf(s - nm);
        m = nm;
    }
    s_m[threadIdx.x] = m;
    s_l[threadIdx.x] = l;
    __syncthreads();
    if (threadIdx.x == 0) {
        float M = -INFINITY, L = 0.0f;
        for (int i = 0; i < (int)blockDim.x; i++) {
            float nm = fmaxf(M, s_m[i]);
            L = L * expf(M - nm) + s_l[i] * expf(s_m[i] - nm);
            M = nm;
        }
        o[0] = L;  // the softmax denominator, so the loop cannot be optimised away
    }
}

// ---- 4. a hundred small kernels, launched then replayed --------------------------------------
__global__ void k_tiny(float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = a[i] * 1.000001f + 0.5f;
}

static void report(const char *name, double ms, double bytes, double extra) {
    printf("  {\"name\": \"%s\", \"ms\": %.4f, \"bytes\": %.0f, \"value\": %.6f},\n", name, ms, bytes, extra);
}

int main() {
    const int seq = 1024, hd = 64;
    const int reps = 10;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("{\n  \"device\": \"%s\", \"compute_cap\": \"%d.%d\", \"sm_count\": %d,\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("  \"head_dim\": %d, \"prefill_seq\": %d, \"block\": %d,\n", hd, seq, BC);
    printf("  \"results\": [\n");

    float *q, *k, *v, *o, *s, *o2;
    CHECK(cudaMalloc(&q, (size_t)seq * hd * 4));
    CHECK(cudaMalloc(&k, (size_t)seq * hd * 4));
    CHECK(cudaMalloc(&v, (size_t)seq * hd * 4));
    CHECK(cudaMalloc(&o, (size_t)seq * hd * 4));
    CHECK(cudaMalloc(&o2, (size_t)seq * hd * 4));
    CHECK(cudaMalloc(&s, (size_t)seq * seq * 4));  // the score buffer, 4 MB at seq = 1024
    float *h = (float *)malloc((size_t)seq * hd * 4);
    for (size_t i = 0; i < (size_t)seq * hd; i++) h[i] = sinf((float)i * 0.017f) * 0.5f;
    CHECK(cudaMemcpy(q, h, (size_t)seq * hd * 4, cudaMemcpyHostToDevice));
    for (size_t i = 0; i < (size_t)seq * hd; i++) h[i] = cosf((float)i * 0.011f) * 0.5f;
    CHECK(cudaMemcpy(k, h, (size_t)seq * hd * 4, cudaMemcpyHostToDevice));
    for (size_t i = 0; i < (size_t)seq * hd; i++) h[i] = sinf((float)i * 0.007f) * 0.25f;
    CHECK(cudaMemcpy(v, h, (size_t)seq * hd * 4, cudaMemcpyHostToDevice));

    double t0 = now_ms();
    for (int r = 0; r < reps; r++) k_prefill_materialised<<<seq, 1>>>(q, k, v, o, s, seq, hd);
    CHECK(cudaDeviceSynchronize());
    double ms_mat = (now_ms() - t0) / reps;
    report("prefill_materialised", ms_mat, (double)seq * seq * 4, 0.0);

    t0 = now_ms();
    for (int r = 0; r < reps; r++) k_prefill_streamed<<<seq, hd>>>(q, k, v, o2, seq, hd);
    CHECK(cudaDeviceSynchronize());
    double ms_str = (now_ms() - t0) / reps;

    // correctness: the two must agree
    float *ha = (float *)malloc((size_t)seq * hd * 4);
    float *hb = (float *)malloc((size_t)seq * hd * 4);
    CHECK(cudaMemcpy(ha, o, (size_t)seq * hd * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hb, o2, (size_t)seq * hd * 4, cudaMemcpyDeviceToHost));
    double worst = 0.0;
    for (size_t i = 0; i < (size_t)seq * hd; i++) {
        double d = fabs((double)ha[i] - (double)hb[i]);
        if (d > worst) worst = d;
    }
    printf("  {\"name\": \"prefill_streamed\", \"ms\": %.4f, \"bytes\": 0, \"value\": %.6f},\n", ms_str, worst);

    // decode at two context lengths
    const int ctxs[2] = {128, 8192};
    float *kb, *vb;
    CHECK(cudaMalloc(&kb, (size_t)ctxs[1] * hd * 4));
    CHECK(cudaMalloc(&vb, (size_t)ctxs[1] * hd * 4));
    CHECK(cudaMemcpy(kb, k, (size_t)seq * hd * 4, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(vb, v, (size_t)seq * hd * 4, cudaMemcpyDeviceToDevice));
    for (int c = 0; c < 2; c++) {
        double ds = now_ms();
        for (int r = 0; r < reps; r++) k_decode_attn<<<1, 256>>>(q, kb, vb, o, ctxs[c], hd);
        CHECK(cudaDeviceSynchronize());
        double ms = (now_ms() - ds) / reps;
        char label[64];
        snprintf(label, sizeof label, "decode_attn_ctx%d", ctxs[c]);
        report(label, ms, (double)ctxs[c] * hd * 4, 0.0);  // the K rows this step had to read
    }

    // a hundred launches, then one graph replaying them
    const int n = 4096, nk = 100;
    float *buf;
    CHECK(cudaMalloc(&buf, (size_t)n * 4));
    CHECK(cudaMemset(buf, 0, (size_t)n * 4));
    int g = (n + 255) / 256;
    double l0 = now_ms();
    for (int r = 0; r < reps; r++)
        for (int i = 0; i < nk; i++) k_tiny<<<g, 256>>>(buf, n);
    CHECK(cudaDeviceSynchronize());
    double ms_launch = (now_ms() - l0) / reps;

    cudaStream_t cap;
    CHECK(cudaStreamCreate(&cap));
    CHECK(cudaStreamBeginCapture(cap, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < nk; i++) k_tiny<<<g, 256, 0, cap>>>(buf, n);
    cudaGraph_t graph;
    CHECK(cudaStreamEndCapture(cap, &graph));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
    double r0 = now_ms();
    for (int r = 0; r < reps; r++) CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    double ms_graph = (now_ms() - r0) / reps;
    report("hundred_launches", ms_launch, 0.0, (double)nk);
    report("graph_replay", ms_graph, 0.0, (double)nk);
    printf("  {\"name\": \"end\", \"ms\": 0, \"bytes\": 0, \"value\": 0}\n  ]\n}\n");
    return 0;
}
