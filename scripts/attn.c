// attn.c - what attention costs, measured rather than asserted.
//
// Chapter 10 makes four claims a reader should be able to check on their own file: the causal mask is
// arithmetic rather than a convention, the score magnitudes are large enough that the softmax's max
// subtraction is load-bearing (Chapter 9 measured the bound), grouped-query attention divides the cache by
// the number of query heads per key head, and the KV cache replaces a matrix multiply with a read. This
// program measures each one on the book's own weights, in f32, with no SIMD - so the numbers describe the
// shape of the work, not one compiler's vectorisation.
//
//   clang -O2 -o attn scripts/attn.c -lm
//   ./attn qwen2.5-0.5b-instruct-q4_k_m.gguf attn.conf --json facts/attn.json
//
// The config file is written by scripts/attn-run.py out of the facts file, so no offset in here is typed:
// each line is `key value`, and the keys are the tensor offsets plus the model's geometry.
//
// What it does not measure: any device kernel, any quantized activation, any thread. Chapter 7 owns SIMD
// and Chapter 8 the threads; this chapter owns the semantics and the shape.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LMAX 1024
#define RUNS 5
#define REPS 3

static uint8_t *slurp(const char *path, size_t off, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); exit(1); }
    if (fseek(f, (long)off, SEEK_SET)) { perror("seek"); exit(1); }
    uint8_t *b = malloc(n);
    if (!b) { fprintf(stderr, "oom %zu\n", n); exit(1); }
    if (fread(b, 1, n, f) != n) { fprintf(stderr, "short read at %zu\n", off); exit(1); }
    fclose(f);
    return b;
}

static float h2f(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FFu, out;
    float r;
    if (exp == 0) {
        if (man == 0) { out = sign; }
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; exp--; }
            man &= 0x3FFu;
            out = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    memcpy(&r, &out, 4);
    return r;
}

// Q5_0: 32 weights in 22 bytes - f16 d, a 32-bit high-bit plane, then 16 low-nibble bytes.
static void deq_q5_0(const uint8_t *p, float *out) {
    float d = h2f((uint16_t)(p[0] | (p[1] << 8)));
    uint32_t qh;
    memcpy(&qh, p + 2, 4);
    const uint8_t *qs = p + 6;
    for (int j = 0; j < 16; j++) {
        uint8_t x0 = (uint8_t)((qs[j] & 0x0F) | (((qh >> j) & 1u) << 4));
        uint8_t x1 = (uint8_t)((qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4));
        out[j] = d * (float)((int)x0 - 16);
        out[j + 16] = d * (float)((int)x1 - 16);
    }
}

// Q8_0: 32 weights in 34 bytes - f16 d then 32 signed bytes.
static void deq_q8_0(const uint8_t *p, float *out) {
    float d = h2f((uint16_t)(p[0] | (p[1] << 8)));
    for (int j = 0; j < 32; j++) out[j] = d * (float)((int8_t)p[2 + j]);
}

// Dequantize a weight matrix whose contiguous dimension is `cols`, into a row-major f32 buffer.
static void load_matrix(const char *gguf, size_t abs_off, const char *fmt, int rows, int cols, float *out) {
    int blocks = cols / 32;
    size_t block_bytes = strcmp(fmt, "Q8_0") == 0 ? 34 : 22;
    size_t per_row = (size_t)blocks * block_bytes;
    uint8_t *raw = slurp(gguf, abs_off, per_row * (size_t)rows);
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < blocks; b++) {
            const uint8_t *p = raw + (size_t)r * per_row + (size_t)b * block_bytes;
            if (block_bytes == 34) deq_q8_0(p, out + (size_t)r * cols + b * 32);
            else deq_q5_0(p, out + (size_t)r * cols + b * 32);
        }
    }
    free(raw);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double median(double *v, int n) {
    for (int i = 1; i < n; i++) {
        double x = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > x) { v[j + 1] = v[j]; j--; }
        v[j + 1] = x;
    }
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void rmsnorm(const float *x, const float *w, float *y, int n, double eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float rms = (float)sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] / rms * w[i];
}

// out[r] = dot(W[r], x) with a double accumulator, so the dot is not the thing under test here.
static void matvec(const float *W, const float *x, float *out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *row = W + (size_t)r * cols;
        double acc = 0.0;
        for (int c = 0; c < cols; c++) acc += (double)row[c] * x[c];
        out[r] = (float)acc;
    }
}

static float dot_f32(const float *a, const float *b, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += (double)a[i] * b[i];
    return (float)acc;
}

static void softmax(const float *s, float *p, int n) {
    float m = s[0];
    for (int i = 1; i < n; i++) if (s[i] > m) m = s[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) { p[i] = expf(s[i] - m); sum += p[i]; }
    for (int i = 0; i < n; i++) p[i] = (float)(p[i] / sum);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: attn GGUF CONFIG [--json out]\n"); return 2; }
    const char *gguf = argv[1], *conf = argv[2];
    const char *json_out = NULL;
    for (int i = 3; i + 1 < argc; i++) if (strcmp(argv[i], "--json") == 0) json_out = argv[i + 1];

    size_t off_q = 0, off_k = 0, off_v = 0, off_norm = 0, off_embd = 0;
    char fmt_q[16] = "Q5_0", fmt_k[16] = "Q5_0", fmt_v[16] = "Q8_0", fmt_embd[16] = "Q5_0";
    int n_embd = 896, n_head = 14, n_kv = 2, n_layer = 24, head_dim = 64;
    FILE *cf = fopen(conf, "r");
    if (!cf) { perror("config"); return 1; }
    char key[64], val[64];
    while (fscanf(cf, "%63s %63s", key, val) == 2) {
        if (!strcmp(key, "q_off")) off_q = strtoull(val, NULL, 10);
        else if (!strcmp(key, "k_off")) off_k = strtoull(val, NULL, 10);
        else if (!strcmp(key, "v_off")) off_v = strtoull(val, NULL, 10);
        else if (!strcmp(key, "norm_off")) off_norm = strtoull(val, NULL, 10);
        else if (!strcmp(key, "embd_off")) off_embd = strtoull(val, NULL, 10);
        else if (!strcmp(key, "q_fmt")) strncpy(fmt_q, val, 15);
        else if (!strcmp(key, "k_fmt")) strncpy(fmt_k, val, 15);
        else if (!strcmp(key, "v_fmt")) strncpy(fmt_v, val, 15);
        else if (!strcmp(key, "embd_fmt")) strncpy(fmt_embd, val, 15);
        else if (!strcmp(key, "n_embd")) n_embd = atoi(val);
        else if (!strcmp(key, "n_head")) n_head = atoi(val);
        else if (!strcmp(key, "n_kv")) n_kv = atoi(val);
        else if (!strcmp(key, "n_layer")) n_layer = atoi(val);
        else if (!strcmp(key, "head_dim")) head_dim = atoi(val);
    }
    fclose(cf);

    int kv_dim = n_kv * head_dim;
    float *Wq = malloc(sizeof(float) * (size_t)n_embd * n_embd);
    float *Wk = malloc(sizeof(float) * (size_t)kv_dim * n_embd);
    float *Wv = malloc(sizeof(float) * (size_t)kv_dim * n_embd);
    float *nw = malloc(sizeof(float) * n_embd);
    load_matrix(gguf, off_q, fmt_q, n_embd, n_embd, Wq);
    load_matrix(gguf, off_k, fmt_k, kv_dim, n_embd, Wk);
    load_matrix(gguf, off_v, fmt_v, kv_dim, n_embd, Wv);
    {
        uint8_t *nb = slurp(gguf, off_norm, (size_t)n_embd * 4);
        memcpy(nw, nb, (size_t)n_embd * 4);
        free(nb);
    }

    // LMAX real embedding rows, dequantized and normalised the way a residual stream would be.
    int embd_blocks = n_embd / 32;
    size_t block_bytes = strcmp(fmt_embd, "Q8_0") == 0 ? 34 : 22;
    size_t embd_row = (size_t)embd_blocks * block_bytes;
    uint8_t *eraw = slurp(gguf, off_embd, embd_row * (size_t)LMAX);
    float *X = malloc(sizeof(float) * (size_t)LMAX * n_embd);
    float *normed = malloc(sizeof(float) * n_embd);
    for (int t = 0; t < LMAX; t++) {
        float *row = X + (size_t)t * n_embd;
        for (int b = 0; b < embd_blocks; b++) {
            const uint8_t *p = eraw + (size_t)t * embd_row + (size_t)b * block_bytes;
            if (block_bytes == 34) deq_q8_0(p, row + b * 32);
            else deq_q5_0(p, row + b * 32);
        }
        rmsnorm(row, nw, normed, n_embd, 1e-6);
        memcpy(row, normed, sizeof(float) * n_embd);
    }
    free(eraw); free(normed);

    // Q, K and V for every cached position: the cache is what the rest of the program measures.
    float *Q = malloc(sizeof(float) * (size_t)LMAX * n_embd);
    float *K = malloc(sizeof(float) * (size_t)LMAX * kv_dim);
    float *V = malloc(sizeof(float) * (size_t)LMAX * kv_dim);
    for (int t = 0; t < LMAX; t++) {
        matvec(Wq, X + (size_t)t * n_embd, Q + (size_t)t * n_embd, n_embd, n_embd);
        matvec(Wk, X + (size_t)t * n_embd, K + (size_t)t * kv_dim, kv_dim, n_embd);
        matvec(Wv, X + (size_t)t * n_embd, V + (size_t)t * kv_dim, kv_dim, n_embd);
    }

    // 1. Score magnitudes on the model's own weights: every (head, cached position) pair at LMAX. The two
    //    bounds are Chapter 9's measurements - exp() overflows in f16 above 11.09, in f32 above 88.72.
    double ssum = 0.0, sq = 0.0, smax = -1e30;
    long n_scores = 0, over_f16 = 0, over_f32 = 0;
    for (int h = 0; h < n_head; h++) {
        const float *qh = Q + h * head_dim;                 // token 0's query, head h
        int kh = h / (n_head / n_kv);
        for (int t = 0; t < LMAX; t++) {
            float s = dot_f32(qh, K + (size_t)t * kv_dim + kh * head_dim, head_dim) / sqrtf((float)head_dim);
            double f = s < 0 ? -s : s;
            ssum += f; sq += f * f;
            if (f > smax) smax = f;
            if (f > 11.09) over_f16++;
            if (f > 88.72) over_f32++;
            n_scores++;
        }
    }
    double smean = ssum / n_scores, srms = sqrt(sq / n_scores);

    // 2. One decode step at several prefix lengths: the last token's query over the cached prefix. If the
    //    step is linear in the prefix, the time doubles when the prefix does - which is what makes the sum
    //    over all steps quadratic, and the cache worth its memory.
    volatile double sink_so = 0.0;
    int lens[4] = {128, 256, 512, 1024};
    double step_s[4];
    float *scores = malloc(sizeof(float) * LMAX);
    float *probs = malloc(sizeof(float) * LMAX);
    float *acc = malloc(sizeof(float) * (size_t)head_dim);
    for (int li = 0; li < 4; li++) {
        int L = lens[li];
        double runs[RUNS];
        for (int r = 0; r < RUNS; r++) {
            double t0 = now_s();
            for (int rep = 0; rep < REPS; rep++) {
                for (int h = 0; h < n_head; h++) {
                    int kh = h / (n_head / n_kv);
                    const float *qh = Q + (size_t)(L - 1) * n_embd + h * head_dim;
                    for (int j = 0; j < L; j++)
                        scores[j] = dot_f32(qh, K + (size_t)j * kv_dim + kh * head_dim, head_dim)
                                    / sqrtf((float)head_dim);
                    softmax(scores, probs, L);
                    for (int d = 0; d < head_dim; d++) {
                        double a = 0.0;
                        for (int j = 0; j < L; j++)
                            a += (double)probs[j] * V[(size_t)j * kv_dim + kh * head_dim + d];
                        acc[d] = (float)a;
                    }
                }
            }
            runs[r] = (now_s() - t0) / REPS;
        }
        step_s[li] = median(runs, RUNS);
    }
    free(scores); free(probs); free(acc);

    // 2a. The mask, as three answers to one question. The query sits at position L/2 with the cache holding
    //     all L positions, so "no mask" is genuinely different from "causal": exactly half the cache is in
    //     the future. Three answers - correct (j <= i), off by one (j < i), unmasked (j < L) - and the
    //     weight the correct answer puts on the query's own position, which is what the off-by-one drops.
    double d_off = 0.0, d_full = 0.0, base_max = 0.0, self_w = 0.0;
    {
        int L = LMAX, i = L / 2;
        float *s2 = malloc(sizeof(float) * L), *p2 = malloc(sizeof(float) * L);
        float *o_ok = malloc(sizeof(float) * head_dim), *o_off = malloc(sizeof(float) * head_dim);
        float *o_full = malloc(sizeof(float) * head_dim);
        for (int h = 0; h < n_head; h++) {
            int kh = h / (n_head / n_kv);
            const float *qh = Q + (size_t)i * n_embd + h * head_dim;
            for (int j = 0; j < L; j++)
                s2[j] = dot_f32(qh, K + (size_t)j * kv_dim + kh * head_dim, head_dim) / sqrtf((float)head_dim);
            for (int d = 0; d < head_dim; d++) o_ok[d] = o_off[d] = o_full[d] = 0.0f;

            softmax(s2, p2, i + 1);                    // correct: j <= i
            self_w += p2[i];
            for (int d = 0; d < head_dim; d++) {
                double a = 0.0;
                for (int j = 0; j <= i; j++) a += (double)p2[j] * V[(size_t)j * kv_dim + kh * head_dim + d];
                o_ok[d] = (float)a;
            }
            softmax(s2, p2, i);                        // off by one: j < i
            for (int d = 0; d < head_dim; d++) {
                double a = 0.0;
                for (int j = 0; j < i; j++) a += (double)p2[j] * V[(size_t)j * kv_dim + kh * head_dim + d];
                o_off[d] = (float)a;
            }
            softmax(s2, p2, L);                        // no mask: every cached position, future included
            for (int d = 0; d < head_dim; d++) {
                double a = 0.0;
                for (int j = 0; j < L; j++) a += (double)p2[j] * V[(size_t)j * kv_dim + kh * head_dim + d];
                o_full[d] = (float)a;
            }
            for (int d = 0; d < head_dim; d++) {
                double b = fabs((double)o_ok[d]);
                if (b > base_max) base_max = b;
                double d1 = fabs((double)o_ok[d] - (double)o_off[d]);
                double d2 = fabs((double)o_ok[d] - (double)o_full[d]);
                if (d1 > d_off) d_off = d1;
                if (d2 > d_full) d_full = d2;
            }
        }
        self_w /= n_head;
        free(s2); free(p2); free(o_ok); free(o_off); free(o_full);
    }

    // 2b. The softmax alone at the longest prefix: what share of a decode step is the exponential.
    double t_softmax, softmax_runs[RUNS];
    for (int r = 0; r < RUNS; r++) {
        double t0 = now_s();
        for (int rep = 0; rep < REPS; rep++) {
            for (int h = 0; h < n_head; h++) {
                float *s2 = malloc(sizeof(float) * LMAX), *p2 = malloc(sizeof(float) * LMAX);
                for (int j = 0; j < LMAX; j++) s2[j] = 1.0f + 0.001f * j;
                softmax(s2, p2, LMAX);
                sink_so += p2[0];
                free(s2); free(p2);
            }
        }
        softmax_runs[r] = (now_s() - t0) / REPS;
    }
    t_softmax = median(softmax_runs, RUNS);

    // 3. Cache versus no cache, at the longest prefix: rebuilding K and V for every cached position (what an
    //    engine without a cache does on every token) against reading the bytes it already has.
    double t_recompute[RUNS], t_read[RUNS];
    float *tmp = malloc(sizeof(float) * (size_t)LMAX * kv_dim);
    volatile double sink = 0.0;
    for (int r = 0; r < RUNS; r++) {
        double t0 = now_s();
        for (int rep = 0; rep < REPS; rep++) {
            for (int t = 0; t < LMAX; t++) {
                matvec(Wk, X + (size_t)t * n_embd, tmp + (size_t)t * kv_dim, kv_dim, n_embd);
                matvec(Wv, X + (size_t)t * n_embd, tmp + (size_t)t * kv_dim, kv_dim, n_embd);
            }
            // Read one element per position into a volatile: without a use, clang deletes the whole loop
            // and reports a recompute that takes zero seconds. The first version of this program did
            // exactly that, which is a measurement lesson worth more than the number it was asked for.
            for (int t = 0; t < LMAX; t++) sink += tmp[(size_t)t * kv_dim];
        }
        t_recompute[r] = (now_s() - t0) / REPS;
    }
    for (int r = 0; r < RUNS; r++) {
        double t0 = now_s();
        for (int rep = 0; rep < REPS; rep++) {
            double a = 0.0;
            for (size_t i = 0; i < (size_t)LMAX * kv_dim; i++) a += K[i];
            sink += a;
        }
        t_read[r] = (now_s() - t0) / REPS;
    }
    (void)sink; free(tmp);

    // 4. The mask, as arithmetic rather than convention.
    double masked_pairs = 0.0;
    for (int t = 0; t < LMAX; t++) masked_pairs += t + 1;
    double full_pairs = (double)LMAX * LMAX;

    // 5. The cache size for a real model, from its own geometry: two tensors, every layer, every token.
    double per_token_f16 = 2.0 * n_kv * head_dim * 2.0;
    double all_layers_4k = per_token_f16 * n_layer * 4096;
    double all_layers_8k = per_token_f16 * n_layer * 8192;
    double weights = 491400032.0;   // the Q4_K_M file this program reads; its byte count is in facts/gguf-*.json

    if (json_out) {
        FILE *o = fopen(json_out, "w");
        fprintf(o, "{\n");
        fprintf(o, " \"geometry\": {\"n_embd\": %d, \"n_head\": %d, \"n_kv\": %d, \"head_dim\": %d, \"n_layer\": %d},\n",
                n_embd, n_head, n_kv, head_dim, n_layer);
        fprintf(o, " \"scores\": {\"n\": %ld, \"abs_mean\": %.4f, \"abs_rms\": %.4f, \"abs_max\": %.4f, "
                   "\"over_f16_exp\": %ld, \"over_f32_exp\": %ld},\n",
                n_scores, smean, srms, smax, over_f16, over_f32);
        fprintf(o, " \"decode_step\": {");
        for (int li = 0; li < 4; li++) fprintf(o, "%s\"%d\": %.9f", li ? ", " : "", lens[li], step_s[li]);
        fprintf(o, "},\n");
        fprintf(o, " \"softmax_only_s\": %.9f,\n", t_softmax);
        fprintf(o, " \"mask_effect\": {\"max_abs_out\": %.6f, \"off_by_one_max_abs\": %.6f, "
                   "\"off_by_one_relative\": %.6f, \"unmasked_max_abs\": %.6f, "
                   "\"unmasked_relative\": %.6f, \"self_weight\": %.6f},\n",
                base_max, d_off, base_max > 0 ? d_off / base_max : 0.0,
                d_full, base_max > 0 ? d_full / base_max : 0.0, self_w);
        fprintf(o, " \"cache\": {\"per_token_per_layer_f16\": %.0f, \"all_layers_4k\": %.0f, "
                   "\"all_layers_8k\": %.0f, \"share_of_weights_at_8k\": %.4f},\n",
                per_token_f16, all_layers_4k, all_layers_8k, all_layers_8k / weights);
        fprintf(o, " \"rebuild\": {\"recompute_kv_s\": %.9f, \"read_kv_s\": %.9f, \"runs\": %d, \"reps\": %d},\n",
                median(t_recompute, RUNS), median(t_read, RUNS), RUNS, REPS);
        fprintf(o, " \"mask\": {\"prefix\": %d, \"masked_pairs\": %.0f, \"full_pairs\": %.0f, \"ratio\": %.6f}\n",
                LMAX, masked_pairs, full_pairs, masked_pairs / full_pairs);
        fprintf(o, "}\n");
        fclose(o);
        printf("wrote %s: %ld scores, |s|max %.2f, %ld over the f16 bound, step@1024 %.3f ms "
               "(softmax alone %.3f ms), rebuild %.3f s vs read %.6f s\n",
               json_out, n_scores, smax, over_f16, step_s[3] * 1e3, t_softmax * 1e3,
               median(t_recompute, RUNS), median(t_read, RUNS));
    }
    return 0;
}
