/* vecops.c - Chapter 9's instrument.
 *
 * The pointwise ops around a transformer: RMSNorm, RoPE, softmax, SiLU/SwiGLU, the residual
 * add. Four questions, one program:
 *
 *   1. How much does the accumulation dtype matter, per op, against a double-precision oracle?
 *   2. Which ops break outright (not "lose a bit") when the formula is applied naively?
 *   3. Which RoPE pairing convention did the model's file assume, and does the wrong one fail a
 *      test rather than quietly producing a bad answer?
 *   4. What do these ops cost, per call and per token, next to the matmul they sit between?
 *
 * Fixtures are real bytes from the book's q4_K_M file: an embedding row (token_embd, Q5_0) as
 * the vector, and blk.0.attn_norm.weight (f32) as the gain. The vectors used are written into
 * the JSON so a second implementation (scripts/vecops-check.py, numpy, f64) can witness the
 * same bytes instead of re-deriving its own.
 *
 *   cc -O2 -o vecops scripts/vecops.c -lm
 *   ./vecops FILE --json out.json
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EMB_OFF 150590816L   /* token_embd.weight, Q5_0, 896 x 151936 */
#define NORM_OFF 244183392L  /* blk.0.attn_norm.weight, F32, 896 */
#define NEMB 896
#define Q5B 22
#define HEAD_DIM 64

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y);
}

static float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 0x1F, m = h & 0x3FF, b;
    float f;
    if (e == 0) {
        if (m == 0) { b = s; memcpy(&f, &b, 4); return f; }
        while (!(m & 0x400)) { m <<= 1; e--; }
        e++; m &= 0x3FF;
    } else if (e == 31) {
        b = s | 0x7F800000u | (m << 13); memcpy(&f, &b, 4); return f;
    }
    b = s | ((e - 15 + 127) << 23) | (m << 13);
    memcpy(&f, &b, 4);
    return f;
}

static uint16_t f32_to_f16(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t s = (b >> 31) & 1, e = (b >> 23) & 0xFF, m = b & 0x7FFFFF;
    if (e == 0xFF) return (uint16_t)((s << 15) | 0x7C00 | (m ? 0x200 : 0));
    int he = (int)e - 127 + 15;
    if (he >= 31) return (uint16_t)((s << 15) | 0x7C00);              /* overflow -> inf */
    if (he <= 0) {                                                     /* subnormal, round to nearest even */
        if (he < -10) return (uint16_t)(s << 15);
        m |= 0x800000;
        int shift = 14 - he;
        uint32_t hm = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (hm & 1))) hm++;
        return (uint16_t)((s << 15) | hm);
    }
    uint32_t hm = m >> 13, rem = m & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (hm & 1))) { hm++; if (hm == 0x400) { hm = 0; he++; if (he >= 31) return (uint16_t)((s << 15) | 0x7C00); } }
    return (uint16_t)((s << 15) | ((uint32_t)he << 10) | hm);
}

static float round_f16(float f) { return f16_to_f32(f32_to_f16(f)); }

/* bfloat16: the top 16 bits of an f32, round to nearest even (ADR-0021's rounding rule). */
static float round_bf16(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t lsb = (b >> 16) & 1, rem = b & 0xFFFF;
    b += 0x7FFF + lsb;
    (void)rem;
    b &= 0xFFFF0000u;
    float r; memcpy(&r, &b, 4);
    return r;
}

static uint8_t *load_bytes(const char *path, long off, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); exit(2); }
    uint8_t *p = malloc(n);
    if (fseek(f, off, SEEK_SET) || fread(p, 1, n, f) != n) { fprintf(stderr, "short read\n"); exit(2); }
    fclose(f);
    return p;
}

/* One Q5_0 row of 896 weights: 22 bytes per 32 weights, the fifth bit in a separate plane. */
static void q5_row(const uint8_t *row, int n, float *out) {
    for (int b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * Q5B;
        float d = f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        uint32_t qh;
        memcpy(&qh, blk + 2, 4);
        for (int j = 0; j < 16; j++) {
            int lo = (blk[6 + j] & 0x0F) | (((qh >> j) & 1) << 4);
            int hi = (blk[6 + j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
            out[b * 32 + j] = d * (float)(lo - 16);
            out[b * 32 + j + 16] = d * (float)(hi - 16);
        }
    }
}

/* ------------------------------------------------------------------ the ops
 *
 * Each op is written twice: once the way an engine writes it (f32 in, f32 or f64 accumulate),
 * once the way a double-precision oracle writes it. The difference is the ledger's number.
 */

/* sum of squares: 1 accumulator, 8 lanes, and 1 in f64 */
static double ss_seq_f32(const float *x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return (double)s;
}
static double ss_lanes_f32(const float *x, int n) {
    float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int i = 0;
    for (; i + 7 < n; i += 8)
        for (int k = 0; k < 8; k++) s[k] += x[i + k] * x[i + k];
    float t = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    for (; i < n; i++) t += x[i] * x[i];
    return (double)t;
}
static double ss_f64(const float *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)x[i] * (double)x[i];
    return s;
}

static void rms_norm(const float *x, const float *w, int n, float eps, double *ss, int mode, float *y) {
    /* mode 0: f32 sequential sum, mode 1: f32 lanes, mode 2: f64 sum */
    double s = mode == 0 ? ss_seq_f32(x, n) : mode == 1 ? ss_lanes_f32(x, n) : ss_f64(x, n);
    if (ss) *ss = s;
    float mean = (float)(s / (double)n);
    float scale = 1.0f / sqrtf(mean + eps);
    for (int i = 0; i < n; i++) y[i] = w ? x[i] * scale * w[i] : x[i] * scale;
}

static void softmax(const float *x, int n, int subtract_max, int sum_f64, float *y, double *sum_out, float *max_out) {
    float mx = -INFINITY;
    if (subtract_max) {
        for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    } else {
        mx = 0.0f;
    }
    if (max_out) *max_out = mx;
    if (sum_f64) {
        double s = 0.0;
        for (int i = 0; i < n; i++) { y[i] = expf(x[i] - mx); s += (double)y[i]; }
        for (int i = 0; i < n; i++) y[i] = (float)((double)y[i] / s);
        if (sum_out) *sum_out = s;
    } else {
        float s = 0.0f;
        for (int i = 0; i < n; i++) { y[i] = expf(x[i] - mx); s += y[i]; }
        for (int i = 0; i < n; i++) y[i] = y[i] / s;
        if (sum_out) *sum_out = (double)s;
    }
}


/* exp is what softmax and SiLU actually spend their time on, so the engine carries an
 * approximation. This is the standard reduce-to-2^k trick: n = round(x/ln2), r = x - n*ln2,
 * then a degree-5 polynomial for exp(r) on |r| <= ln2/2, then scale by 2^n by bit assembly. */
static float fast_exp(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) return INFINITY;
    const float ln2 = 0.6931471805599453f;
    float n = roundf(x * 1.4426950408889634f);
    float r = x - n * ln2;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.0f;
    /* multiply by 2^n */
    union { float f; uint32_t i; } u;
    u.i = (uint32_t)((int)n + 127) << 23;
    return p * u.f;
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

/* RoPE, two conventions. `split` = 0 pairs (2i, 2i+1); `split` = 1 pairs (i, i + hd/2),
 * which is what llama.cpp and HF's Llama/NeoX implementations do. */
static void rope(const float *in, int hd, int pos, float base, int split, float *out) {
    memcpy(out, in, sizeof(float) * (size_t)hd);
    for (int d = 0; d < hd / 2; d++) {
        int i0 = split ? d : 2 * d, i1 = split ? d + hd / 2 : 2 * d + 1;
        float inv = 1.0f / powf(base, (float)(2 * d) / (float)hd);
        float th = (float)pos * inv;
        float c = cosf(th), s = sinf(th);
        out[i0] = in[i0] * c - in[i1] * s;
        out[i1] = in[i0] * s + in[i1] * c;
    }
}

/* ------------------------------------------------------------------ reporting */

static void jprint_floats(FILE *j, const char *name, const float *v, int n) {
    fprintf(j, " \"%s\": [", name);
    for (int i = 0; i < n; i++) fprintf(j, "%s%.9g", i ? ", " : "", (double)v[i]);
    fprintf(j, "],\n");
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s FILE [--json out.json]\n", argv[0]); return 2; }
    const char *path = argv[1];
    const char *json = NULL;
    for (int i = 2; i < argc; i++) if (!strcmp(argv[i], "--json") && i + 1 < argc) json = argv[++i];

    /* ---- fixtures: a real embedding row and the real gain vector */
    uint8_t *emb_raw = load_bytes(path, EMB_OFF, (size_t)(NEMB / 32) * Q5B);
    float *emb = malloc(sizeof(float) * NEMB);
    q5_row(emb_raw, NEMB, emb);
    float *gain = (float *)load_bytes(path, NORM_OFF, sizeof(float) * NEMB);

    /* ---- 1. RMSNorm: what the accumulation dtype costs */
    float *y_ref = malloc(sizeof(float) * NEMB), *y = malloc(sizeof(float) * NEMB);
    double ss_ref = 0;
    rms_norm(emb, gain, NEMB, 1e-6f, &ss_ref, 2, y_ref);
    double norm_rel[3];
    int modes[3] = {0, 1, 2};
    float scale_by_mode[3];
    for (int m = 0; m < 3; m++) {
        double ss = 0;
        rms_norm(emb, gain, NEMB, 1e-6f, &ss, modes[m], y);
        double worst = 0;
        for (int i = 0; i < NEMB; i++) {
            if (y_ref[i] == 0.0f) continue;
            double rel = fabs((double)y[i] - (double)y_ref[i]) / fabs((double)y_ref[i]);
            if (rel > worst) worst = rel;
        }
        norm_rel[m] = worst;
        float mean = (float)(ss / (double)NEMB);
        scale_by_mode[m] = 1.0f / sqrtf(mean + 1e-6f);
    }
    /* the sum of squares, both ways, and what epsilon is worth against it */
    double rms = sqrt(ss_ref / (double)NEMB);
    float scale_no_eps = 1.0f / sqrtf((float)(ss_ref / (double)NEMB));
    float scale_eps = 1.0f / sqrtf((float)(ss_ref / (double)NEMB) + 1e-6f);
    double eps_effect = fabs((double)scale_eps - (double)scale_no_eps) / (double)scale_no_eps;

    /* ---- 2. softmax: the max subtraction, and the sum's dtype */
    int ns = 4096;
    float *logits = malloc(sizeof(float) * ns), *sp = malloc(sizeof(float) * ns);
    float *sp_naive = malloc(sizeof(float) * ns), *sp_f32sum = malloc(sizeof(float) * ns);
    /* real scores: the embedding row dotted with the first ns rows of the same matrix */
    {
        uint8_t *rows = load_bytes(path, EMB_OFF, (size_t)ns * (NEMB / 32) * Q5B);
        float *row = malloc(sizeof(float) * NEMB);
        for (int r = 0; r < ns; r++) {
            q5_row(rows + (size_t)r * (NEMB / 32) * Q5B, NEMB, row);
            double acc = 0.0;
            for (int i = 0; i < NEMB; i++) acc += (double)row[i] * (double)emb[i];
            logits[r] = (float)(acc / sqrt((double)NEMB));
        }
        free(rows); free(row);
    }
    float mx_real = -INFINITY;
    for (int i = 0; i < ns; i++) if (logits[i] > mx_real) mx_real = logits[i];
    double sum64 = 0, sum32 = 0;
    float m1 = 0, m2 = 0;
    softmax(logits, ns, 1, 1, sp, &sum64, &m1);
    softmax(logits, ns, 1, 0, sp_f32sum, &sum32, &m2);
    double softsum_rel = 0;
    for (int i = 0; i < ns; i++) {
        if (sp[i] == 0.0f) continue;
        double rel = fabs((double)sp_f32sum[i] - (double)sp[i]) / (double)sp[i];
        if (rel > softsum_rel) softsum_rel = rel;
    }
    /* Without the max subtraction the row is exp(x)/sum(exp(x)), and exp overflows: at some
     * logit the sum is inf and every output is inf/inf. The threshold is a property of the
     * dtype, so measure it for f32 (this op's natural dtype) and for f16 (where an engine might
     * keep scores): a binary search on the logit at which the naive path first produces a NaN. */
    double naive_break_f32 = 0.0;
    {
        double lo = 0.0, hi = 200.0;
        for (int it = 0; it < 60; it++) {
            double mid = 0.5 * (lo + hi);
            float probe[2] = {0.0f, (float)mid};
            float out[2];
            softmax(probe, 2, 0, 1, out, NULL, NULL);
            int nan = !(out[1] == out[1]);
            if (nan) hi = mid; else lo = mid;
        }
        naive_break_f32 = hi;
    }
    double naive_break_f16 = 0.0;
    {
        /* the same search with the exponentials rounded to f16 and summed in f16 */
        double lo = 0.0, hi = 200.0;
        for (int it = 0; it < 60; it++) {
            double mid = 0.5 * (lo + hi);
            float e0 = round_f16(fast_exp(0.0f)), e1 = round_f16(fast_exp((float)mid));
            float sum = round_f16(e0 + e1);
            float o1 = sum > 0.0f ? round_f16(e1 / sum) : 0.0f;
            int bad = !(o1 == o1) || !isfinite((double)o1) || o1 == 0.0f;
            if (bad) hi = mid; else lo = mid;
        }
        naive_break_f16 = hi;
    }
    /* with the subtraction, the same row is fine at any logit: the shift is exact */

    /* ---- 3. SiLU and the residual chain: where the dtype of the stream shows up */
    double silu_rel = 0.0;
    {
        float lo = -20.0f, hi = 20.0f;
        for (int i = 0; i < 4001; i++) {
            float x = lo + (hi - lo) * (float)i / 4000.0f;
            double ref = (double)x / (1.0 + exp(-(double)x));
            double got = (double)silu(x);
            double den = fabs(ref) > 1e-6 ? fabs(ref) : 1e-6;
            double rel = fabs(got - ref) / den;
            if (rel > silu_rel) silu_rel = rel;
        }
    }
    /* a 24-layer residual stream: y = y + delta, delta a fixed pattern, the same in f32, f16, bf16 */
    int layers = 24;
    double drift[3];
    {
        float *y32 = malloc(sizeof(float) * NEMB), *y16 = malloc(sizeof(float) * NEMB), *yb = malloc(sizeof(float) * NEMB);
        double *ref = malloc(sizeof(double) * NEMB);
        for (int i = 0; i < NEMB; i++) {
            y32[i] = emb[i]; y16[i] = emb[i]; yb[i] = emb[i]; ref[i] = (double)emb[i];
        }
        for (int l = 0; l < layers; l++) {
            for (int i = 0; i < NEMB; i++) {
                float d = gain[i] * 0.01f * (float)((i % 7) - 3);
                y32[i] += d;                        /* f32 accumulate */
                y16[i] = round_f16(y16[i] + d);     /* f16 stream: every sum rounded to f16 */
                yb[i] = round_bf16(yb[i] + d);      /* bf16 stream */
                ref[i] += (double)d;
            }
        }
        double w32 = 0, w16 = 0, wb = 0, base = 0;
        for (int i = 0; i < NEMB; i++) {
            double d32 = fabs((double)y32[i] - ref[i]), d16 = fabs((double)y16[i] - ref[i]),
                   db = fabs((double)yb[i] - ref[i]);
            if (d32 > w32) w32 = d32;
            if (d16 > w16) w16 = d16;
            if (db > wb) wb = db;
            if (fabs(ref[i]) > base) base = fabs(ref[i]);
        }
        drift[0] = w32 / base; drift[1] = w16 / base; drift[2] = wb / base;
        free(y32); free(y16); free(yb); free(ref);
    }

    /* ---- 4. RoPE: which pairing, and does the wrong one fail a test?
     *
     * Two questions, and the second one is the interesting one. (a) Does each pairing preserve the
     * property that a query at p and a key at p+delta score by the distance only? Yes - both are
     * rotations, so that test passes for either, which is exactly why it cannot catch a mismatch
     * between a model's file and the kernel. (b) How far apart are the two pairings' scores for
     * the same bytes? That distance is the size of the bug you would ship. */
    float qvec[HEAD_DIM], kvec[HEAD_DIM];
    memcpy(qvec, emb, sizeof(float) * HEAD_DIM);
    memcpy(kvec, emb + HEAD_DIM, sizeof(float) * HEAD_DIM);
    int positions = 16;
    double worst_split[2] = {0, 0};
    double cross_dev = 0.0;
    for (int split = 0; split < 2; split++) {
        float kb[HEAD_DIM], q0r[HEAD_DIM], k0r[HEAD_DIM];
        for (int delta = 1; delta < positions; delta++) {
            rope(qvec, HEAD_DIM, 0, 1000000.0f, split, q0r);       /* q at 0 */
            rope(kvec, HEAD_DIM, delta, 1000000.0f, split, kb);    /* k at delta */
            rope(kvec, HEAD_DIM, 0, 1000000.0f, split, k0r);       /* k at 0, for the invariant */
            double s_rel = 0, s_zero = 0, den = 0, nq = 0, nk = 0;
            for (int i = 0; i < HEAD_DIM; i++) {
                s_rel += (double)q0r[i] * (double)kb[i];
                s_zero += (double)q0r[i] * (double)k0r[i];
                nq += (double)q0r[i] * (double)q0r[i];
                nk += (double)kb[i] * (double)kb[i];
            }
            den = sqrt(nq * nk);
            double mag = den > 0 ? fabs(s_zero) / den : 0;
            /* the rotated score at distance delta against the unrotated score: they differ, and
             * that difference is what the rotation is for. The invariant that matters is that
             * the rotated score depends on delta alone, which is checked by comparing the
             * rotation of the pair against rotating only the key by delta from a shared origin. */
            double dev = den > 0 ? fabs(s_rel - s_zero) / den : 0;
            (void)dev;
            (void)mag;
            /* the real check: q at p and k at p+delta, against q at 0 and k at delta */
            float qa2[HEAD_DIM], kb2[HEAD_DIM];
            rope(qvec, HEAD_DIM, delta + 3, 1000000.0f, split, qa2);
            rope(kvec, HEAD_DIM, delta + 3 + delta, 1000000.0f, split, kb2);
            double s_shift = 0, ns = 0;
            for (int i = 0; i < HEAD_DIM; i++) {
                s_shift += (double)qa2[i] * (double)kb2[i];
                ns += (double)kb2[i] * (double)kb2[i];
            }
            double d2 = sqrt(nq * ns);
            if (d2 > 0) {
                double gap2 = fabs(s_shift - s_rel) / d2;
                if (gap2 > worst_split[split]) worst_split[split] = gap2;
            }
        }
    }
    /* the same (q, k, p, delta) scored under the two pairings */
    {
        float qa[HEAD_DIM], kb[HEAD_DIM], qb[HEAD_DIM], kbb[HEAD_DIM];
        for (int delta = 1; delta < positions; delta++) {
            rope(qvec, HEAD_DIM, 0, 1000000.0f, 0, qa);
            rope(kvec, HEAD_DIM, delta, 1000000.0f, 0, kb);
            rope(qvec, HEAD_DIM, 0, 1000000.0f, 1, qb);
            rope(kvec, HEAD_DIM, delta, 1000000.0f, 1, kbb);
            double s0 = 0, s1 = 0, nq = 0, nk = 0;
            for (int i = 0; i < HEAD_DIM; i++) {
                s0 += (double)qa[i] * (double)kb[i];
                s1 += (double)qb[i] * (double)kbb[i];
                nq += (double)qb[i] * (double)qb[i];
                nk += (double)kbb[i] * (double)kbb[i];
            }
            double den = sqrt(nq * nk);
            if (den > 0) {
                double gap = fabs(s0 - s1) / den;
                if (gap > cross_dev) cross_dev = gap;
            }
        }
    }

    /* ---- 4b. exp: the transcendental inside softmax and SiLU */
    double exp_libm_ns = 0, exp_fast_ns = 0, exp_err = 0;
    {
        double r[5];
        for (int k = 0; k < 5; k++) {
            double t0 = now_s();
            float acc = 0.0f;
            for (int rep = 0; rep < 200; rep++) for (int i = 0; i < ns; i++) acc += expf(logits[i] * 0.001f);
            r[k] = (now_s() - t0) / (200.0 * ns) * 1e9;
            if (acc == 1234.0f) printf(" ");
        }
        qsort(r, 5, sizeof(double), cmp_double);
        exp_libm_ns = r[2];
        for (int k = 0; k < 5; k++) {
            double t0 = now_s();
            float acc = 0.0f;
            for (int rep = 0; rep < 200; rep++) for (int i = 0; i < ns; i++) acc += fast_exp(logits[i] * 0.001f);
            r[k] = (now_s() - t0) / (200.0 * ns) * 1e9;
            if (acc == 1234.0f) printf(" ");
        }
        qsort(r, 5, sizeof(double), cmp_double);
        exp_fast_ns = r[2];
        for (int i = 0; i < 4001; i++) {
            float x = -20.0f + 40.0f * (float)i / 4000.0f;
            double ref = exp((double)x), got = (double)fast_exp(x);
            double rel = ref > 0 ? fabs(got - ref) / ref : 0;
            if (rel > exp_err) exp_err = rel;
        }
    }

    /* ---- 5. what these ops cost, at model shapes */
    int shapes[4] = {896, 4096, 32768, 131072};
    double cost_norm[4], cost_soft[4], cost_silu[4], cost_add[4];
    float *big = malloc(sizeof(float) * 131072), *bigy = malloc(sizeof(float) * 131072);
    for (int i = 0; i < 131072; i++) big[i] = logits[i % ns];
    for (int s = 0; s < 4; s++) {
        int n = shapes[s];
        double r[5];
        for (int k = 0; k < 5; k++) {
            double t0 = now_s();
            for (int rep = 0; rep < 200; rep++) rms_norm(big, NULL, NEMB, 1e-6f, NULL, 1, bigy);
            r[k] = (now_s() - t0) / 200 * 1e9;
        }
        qsort(r, 5, sizeof(double), cmp_double);
        cost_norm[s] = r[2];
        for (int k = 0; k < 5; k++) {
            double t0 = now_s();
            for (int rep = 0; rep < 100; rep++) softmax(big, n, 1, 1, bigy, NULL, NULL);
            r[k] = (now_s() - t0) / 100 * 1e9;
        }
        qsort(r, 5, sizeof(double), cmp_double);
        cost_soft[s] = r[2];
        for (int k = 0; k < 5; k++) {
            double t0 = now_s();
            for (int rep = 0; rep < 200; rep++) for (int i = 0; i < NEMB; i++) bigy[i] = silu(big[i]);
            r[k] = (now_s() - t0) / 200 * 1e9;
        }
        qsort(r, 5, sizeof(double), cmp_double);
        cost_silu[s] = r[2];
        for (int k = 0; k < 5; k++) {
            double t0 = now_s();
            for (int rep = 0; rep < 200; rep++) for (int i = 0; i < NEMB; i++) bigy[i] += big[i];
            r[k] = (now_s() - t0) / 200 * 1e9;
        }
        qsort(r, 5, sizeof(double), cmp_double);
        cost_add[s] = r[2];
    }

    printf("VECTOR OPS | %d-wide embedding, 24 layers, head_dim %d\n", NEMB, HEAD_DIM);
    printf("rms_norm  rms %.6f  eps %.6g  scale %.6f  eps changes it by %.2e\n", rms, 1e-6, scale_eps, eps_effect);
    printf("          max rel vs f64: seq f32 %.2e | 8 lanes %.2e | f64 %.2e\n", norm_rel[0], norm_rel[1], norm_rel[2]);
    printf("softmax   max logit %.4f  sum %.6f (f64) %.6f (f32)  max rel %.2e\n", mx_real, sum64, sum32, softsum_rel);
    printf("          naive path goes NaN above logit %.2f in f32, %.2f in f16\n", naive_break_f32, naive_break_f16);
    printf("exp       libm expf %.2f ns/elem | polynomial %.2f ns/elem | max rel err %.2e\n",
           exp_libm_ns, exp_fast_ns, exp_err);
    printf("silu      max rel vs double over [-20,20]: %.2e\n", silu_rel);
    printf("residual  drift after %d adds, over max |y|: f32 %.2e | f16 %.2e | bf16 %.2e\n", layers,
           drift[0], drift[1], drift[2]);
    printf("rope      relative-position deviation: adjacent pairs %.3e | half split %.3e\n", worst_split[0], worst_split[1]);
    printf("          the same (q,k) scored under the two pairings differs by %.3e of the norm product\n", cross_dev);
    printf("cost      ns:  rms_norm(896) %.0f | silu(896) %.0f | add(896) %.0f\n", cost_norm[0], cost_silu[0], cost_add[0]);
    for (int s = 0; s < 4; s++)
        printf("          softmax(%d) %.0f ns\n", shapes[s], cost_soft[s]);

    if (json) {
        FILE *j = fopen(json, "w");
        if (!j) { perror("json"); return 2; }
        fprintf(j, "{\n \"artifact\": \"vecops\",\n");
        fprintf(j, " \"fixtures\": {\n  \"note\": \"real bytes: an embedding row (token_embd, Q5_0, row 0) and "
                   "blk.0.attn_norm.weight (f32); the logits are that row dotted with the first 4096 rows\",\n");
        fprintf(j, "  \"emb_off\": %ld,\n  \"norm_off\": %ld,\n  \"embeddings\": %d,\n  \"head_dim\": %d,\n",
                EMB_OFF, NORM_OFF, NEMB, HEAD_DIM);
        jprint_floats(j, "emb", emb, NEMB);
        jprint_floats(j, "gain", gain, NEMB);
        jprint_floats(j, "logits", logits, ns);
        fprintf(j, "  \"logits_len\": %d,\n  \"logits_max\": %.9g,\n", ns, (double)mx_real);
        fprintf(j, "  \"q_head\": [");
        for (int i = 0; i < HEAD_DIM; i++) fprintf(j, "%s%.9g", i ? ", " : "", (double)qvec[i]);
        fprintf(j, "],\n  \"k_head\": [");
        for (int i = 0; i < HEAD_DIM; i++) fprintf(j, "%s%.9g", i ? ", " : "", (double)kvec[i]);
        fprintf(j, "]\n },\n");
        /* outputs, so a second implementation can witness the arithmetic rather than the intent */
        rms_norm(emb, gain, NEMB, 1e-6f, NULL, 1, y);
        jprint_floats(j, "norm_out", y, NEMB);
        softmax(logits, ns, 1, 1, sp, NULL, NULL);
        jprint_floats(j, "softmax_out", sp, 16);
        {
            float sx[32], sy[32];
            for (int i = 0; i < 32; i++) { sx[i] = -4.0f + 0.25f * (float)i; sy[i] = silu(sx[i]); }
            jprint_floats(j, "silu_in", sx, 32);
            jprint_floats(j, "silu_out", sy, 32);
            float rk[3][HEAD_DIM];
            for (int d = 0; d < 3; d++) rope(kvec, HEAD_DIM, d + 1, 1000000.0f, 1, rk[d]);
            fprintf(j, " \"rope_out\": [");
            for (int d = 0; d < 3; d++)
                for (int i = 0; i < HEAD_DIM; i++)
                    fprintf(j, "%s%.9g", (d || i) ? ", " : "", (double)rk[d][i]);
            fprintf(j, "],\n");
        }
        fprintf(j, " \"norm\": {\"rms\": %.9g, \"eps\": 1e-06, \"scale\": %.9g, \"eps_effect\": %.6e,\n"
                   "          \"max_rel_vs_f64\": {\"seq_f32\": %.6e, \"lanes_f32\": %.6e, \"f64\": %.6e},\n"
                   "          \"scale_by_mode\": [%.9g, %.9g, %.9g], \"sum_sq_f64\": %.9g},\n",
                rms, (double)scale_eps, eps_effect, norm_rel[0], norm_rel[1], norm_rel[2],
                (double)scale_by_mode[0], (double)scale_by_mode[1], (double)scale_by_mode[2], ss_ref);
        fprintf(j, " \"softmax\": {\"len\": %d, \"max_logit\": %.9g, \"sum_f64\": %.9g, \"sum_f32\": %.9g,\n"
                   "             \"max_rel_f32_sum\": %.6e, \"naive_breaks_above_logit\": {\"f32\": %.4f, "
                   "\"f16\": %.4f}},\n",
                ns, (double)mx_real, sum64, sum32, softsum_rel, naive_break_f32, naive_break_f16);
        fprintf(j, " \"exp\": {\"libm_ns_per_elem\": %.3f, \"polynomial_ns_per_elem\": %.3f, "
                   "\"polynomial_max_rel\": %.6e},\n", exp_libm_ns, exp_fast_ns, exp_err);
        fprintf(j, " \"silu\": {\"max_rel_vs_double\": %.6e},\n", silu_rel);
        fprintf(j, " \"residual\": {\"layers\": %d, \"max_drift_over_max_abs\": {\"f32\": %.6e, \"f16\": %.6e, "
                   "\"bf16\": %.6e}},\n", layers, drift[0], drift[1], drift[2]);
        fprintf(j, " \"rope\": {\"base\": 1000000.0, \"positions\": %d, \"head_dim\": %d,\n"
                   "          \"relative_position_deviation\": {\"adjacent_pairs\": %.6e, \"half_split\": %.6e},\n"
                   "          \"score_gap_between_pairings\": %.6e},\n",
                positions, HEAD_DIM, worst_split[0], worst_split[1], cross_dev);
        fprintf(j, " \"cost_ns\": {\n  \"shapes\": [%d, %d, %d, %d],\n", shapes[0], shapes[1], shapes[2], shapes[3]);
        fprintf(j, "  \"rms_norm_896\": %.1f,\n  \"silu_896\": %.1f,\n  \"add_896\": %.1f,\n", cost_norm[0],
                cost_silu[0], cost_add[0]);
        fprintf(j, "  \"softmax\": [{\"len\": %d, \"ns\": %.1f}, {\"len\": %d, \"ns\": %.1f}, {\"len\": %d, \"ns\": %.1f}, "
                   "{\"len\": %d, \"ns\": %.1f}]\n },\n", shapes[0], cost_soft[0], shapes[1], cost_soft[1],
                shapes[2], cost_soft[2], shapes[3], cost_soft[3]);
        fprintf(j, " \"invocation\": \"./vecops qwen2.5-0.5b-instruct-q4_k_m.gguf --json facts/vecops.json\"\n}\n");
        fclose(j);
    }
    free(emb_raw); free(emb); free(gain); free(y); free(y_ref);
    free(logits); free(sp); free(sp_naive); free(sp_f32sum); free(big); free(bigy);
    return 0;
}
