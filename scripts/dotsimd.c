// dotsimd.c - the same quantized dot product, four ways, with the dispatch a chapter is about.
//
//   scalar        the oracle (Chapter 6)
//   avx2          _mm256_maddubs_epi16 + _mm256_madd_epi16, 32 weights per iteration   (x86)
//   avx512-vnni   _mm512_dpbusd_epi32, 64 weights per iteration                       (x86, Tiger Lake+)
//   neon-sdot     vdotq_s32, 32 weights per iteration                                  (ARMv8.2 dotprod)
//
// The point is not "SIMD is faster" - it is that every one of these must return the same number as the
// scalar path, and that the dispatch which chooses among them must be able to prove which one it chose.
// The program therefore does three things in order: run every available kernel over every row and compare
// against the scalar path *exactly*, time each one, and print what the runtime dispatch would pick on this
// machine and why.
//
//   cc -O2 -o /tmp/dotsimd scripts/dotsimd.c          # x86 or ARM, it adapts
//   /tmp/dotsimd model.gguf <offset> <row_len> <rows> [runs] [--json out]
//
// Offset and row length come from the facts file (scripts/gguf-inspect.py --json), not from this file.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

#define QK 32
#define Q4B 18
#define Q8B 34

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    if (exp == 0) {
        if (man == 0) { uint32_t b = sign; float f; memcpy(&f, &b, 4); return f; }
        while (!(man & 0x400)) { man <<= 1; exp--; }
        exp++; man &= 0x3FF;
    } else if (exp == 31) {
        uint32_t b = sign | 0x7F800000u | (man << 13); float f; memcpy(&f, &b, 4); return f;
    }
    uint32_t b = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f; memcpy(&f, &b, 4); return f;
}

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

// The scale is an f16, and every kernel pays to convert it. The software decoder above is branchy; both of
// these machines have a single instruction for it, and using it moves the measured number more than the
// vector width does (Lesson 3).
static float scale_f16(uint16_t h) {
#if defined(__aarch64__)
    __fp16 v;
    memcpy(&v, &h, 2);
    return (float)v;                              // fcvt, one instruction
#elif defined(__F16C__)
    return _cvtsh_ss(h);                          // vcvtph2ps, one instruction
#else
    return fp16_to_f32(h);
#endif
}

// ---------------------------------------------------------------- the oracle

static float dot_scalar(const uint8_t *w, const uint8_t *a, int nb) {
    float s = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *wb = w + b * Q4B, *ab = a + b * Q8B;
        float d = scale_f16(rd16(wb)) * scale_f16(rd16(ab));
        const int8_t *q = (const int8_t *)(ab + 2);
        int32_t acc = 0;
        for (int j = 0; j < 16; j++) {
            acc += ((int)(wb[2 + j] & 0x0F) - 8) * (int)q[j];
            acc += ((int)(wb[2 + j] >> 4) - 8) * (int)q[j + 16];
        }
        s += (float)acc * d;
    }
    return s;
}

// ---------------------------------------------------------------- x86

#if defined(__x86_64__)
#define HAVE_X86 1

__attribute__((target("avx2,fma")))
static float dot_avx2(const uint8_t *w, const uint8_t *a, int nb) {
    __m256 acc = _mm256_setzero_ps();
    const __m256i mask = _mm256_set1_epi8(0x0F), bias = _mm256_set1_epi8(8);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int b = 0; b < nb; b++) {
        const uint8_t *wb = w + b * Q4B, *ab = a + b * Q8B;
        float d = scale_f16(rd16(wb)) * scale_f16(rd16(ab));
        __m128i raw = _mm_loadu_si128((const __m128i *)(wb + 2));
        // low nibbles become weights 0-15, high nibbles weights 16-31: one 256-bit register of 32 bytes
        __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(raw, 4), raw);
        __m256i qw = _mm256_sub_epi8(_mm256_and_si256(bytes, mask), bias);
        __m256i qa = _mm256_loadu_si256((const __m256i *)(ab + 2));
        // maddubs wants unsigned x signed, so fold the weight's sign into the activation instead
        __m256i uw = _mm256_sign_epi8(qw, qw);      // |q_w|
        __m256i sa = _mm256_sign_epi8(qa, qw);      // q_a with q_w's sign
        __m256i p16 = _mm256_maddubs_epi16(uw, sa); // 16 pair sums, i16
        __m256i p32 = _mm256_madd_epi16(p16, ones); // widen to i32
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32), acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    return _mm_cvtss_f32(s4);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")))
static float dot_avx512_vnni(const uint8_t *w, const uint8_t *a, int nb) {
    __m256 facc = _mm256_setzero_ps();
    const __m256i mask = _mm256_set1_epi8(0x0F), bias = _mm256_set1_epi8(8);
    const __m512i zero512 = _mm512_setzero_si512();
    int b = 0;
    // Two blocks per instruction: 64 weights, 64 activations. VNNI (dpbusd) multiplies unsigned by signed
    // and adds four products into each i32 lane - ARM's SDOT under another name. There is no AVX-512
    // sign_epi8, so the sign of the weight is applied to the activation through a mask instead.
    for (; b + 1 < nb; b += 2) {
        __m128i p0 = _mm_loadu_si128((const __m128i *)(w + b * Q4B + 2));
        __m128i p1 = _mm_loadu_si128((const __m128i *)(w + (b + 1) * Q4B + 2));
        __m256i r0 = _mm256_set_m128i(_mm_srli_epi16(p0, 4), p0);   // [low nibbles | high nibbles]
        __m256i r1 = _mm256_set_m128i(_mm_srli_epi16(p1, 4), p1);
        __m256i q0 = _mm256_sub_epi8(_mm256_and_si256(r0, mask), bias);
        __m256i q1 = _mm256_sub_epi8(_mm256_and_si256(r1, mask), bias);
        __m512i qw = _mm512_inserti64x4(_mm512_castsi256_si512(q0), q1, 1);
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + b * Q8B + 2));
        __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + (b + 1) * Q8B + 2));
        __m512i qa = _mm512_inserti64x4(_mm512_castsi256_si512(a0), a1, 1);
        __m512i uw = _mm512_abs_epi8(qw);
        __mmask64 neg = _mm512_movepi8_mask(qw);
        __m512i sa = _mm512_mask_blend_epi8(neg, qa, _mm512_sub_epi8(zero512, qa));
        __m512i acc = _mm512_dpbusd_epi32(zero512, uw, sa);
        float d0 = scale_f16(rd16(w + b * Q4B)) * scale_f16(rd16(a + b * Q8B));
        float d1 = scale_f16(rd16(w + (b + 1) * Q4B)) * scale_f16(rd16(a + (b + 1) * Q8B));
        facc = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(_mm512_castsi512_si256(acc)), facc);
        facc = _mm256_fmadd_ps(_mm256_set1_ps(d1),
                              _mm256_cvtepi32_ps(_mm512_extracti64x4_epi64(acc, 1)), facc);
    }
    for (; b < nb; b++) {                        // the odd trailing block, upper lanes zeroed
        __m128i p = _mm_loadu_si128((const __m128i *)(w + b * Q4B + 2));
        __m256i r = _mm256_set_m128i(_mm_srli_epi16(p, 4), p);
        __m256i qw = _mm256_sub_epi8(_mm256_and_si256(r, mask), bias);
        __m256i qa = _mm256_loadu_si256((const __m256i *)(a + b * Q8B + 2));
        __m256i uw = _mm256_abs_epi8(qw);
        __mmask32 neg = _mm256_movepi8_mask(qw);
        __m256i sa = _mm256_mask_blend_epi8(neg, qa, _mm256_sub_epi8(_mm256_setzero_si256(), qa));
        __m512i acc = _mm512_dpbusd_epi32(zero512, _mm512_zextsi256_si512(uw),
                                          _mm512_zextsi256_si512(sa));
        float d = scale_f16(rd16(w + b * Q4B)) * scale_f16(rd16(a + b * Q8B));
        facc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm512_castsi512_si256(acc)), facc);
    }
    __m128 lo = _mm256_castps256_ps128(facc), hi = _mm256_extractf128_ps(facc, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    return _mm_cvtss_f32(s4);
}

#endif  // x86

// ---------------------------------------------------------------- ARM

#if defined(__aarch64__)
#define HAVE_ARM 1
#if defined(__ARM_FEATURE_DOTPROD)
#define HAVE_SDOT 1
#endif
#endif  // aarch64

// SDOT folds 16 weights into an i32 accumulator per instruction, but the per-block scales differ, so the
// kernel keeps one i32 accumulator per block and one float multiply per block.
#if defined(HAVE_SDOT)
// The same SDOT kernel with the per-block horizontal add removed: keep four float lanes as the accumulator
// and turn each block's i32 dot into a float vector, folding the four lanes once at the end. ggml's kernels
// have this shape, and the difference against `neon-sdot` is what the chapter measures.
static float dot_neon_acc(const uint8_t *w, const uint8_t *a, int nb) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8x16_t s8 = vdupq_n_s8(8);
    float32x4_t facc = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; b++) {
        const uint8_t *wb = w + b * Q4B, *ab = a + b * Q8B;
        float d = scale_f16(rd16(wb)) * scale_f16(rd16(ab));
        uint8x16_t bytes = vld1q_u8(wb + 2);
        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, m4)), s8);
        int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), s8);
        int8x16_t a0 = vld1q_s8((const int8_t *)(ab + 2));
        int8x16_t a1 = vld1q_s8((const int8_t *)(ab + 18));
        int32x4_t acc = vdotq_s32(vdupq_n_s32(0), lo, a0);
        acc = vdotq_s32(acc, hi, a1);
        facc = vfmaq_n_f32(facc, vcvtq_f32_s32(acc), d);   // no horizontal add here
    }
    return vaddvq_f32(facc);
}
#endif

#if defined(HAVE_SDOT)
static float dot_neon_scaled(const uint8_t *w, const uint8_t *a, int nb) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8x16_t s8 = vdupq_n_s8(8);
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *wb = w + b * Q4B, *ab = a + b * Q8B;
        float d = scale_f16(rd16(wb)) * scale_f16(rd16(ab));
        uint8x16_t bytes = vld1q_u8(wb + 2);
        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, m4)), s8);
        int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), s8);
        int8x16_t a0 = vld1q_s8((const int8_t *)(ab + 2));
        int8x16_t a1 = vld1q_s8((const int8_t *)(ab + 18));
        int32x4_t acc = vdotq_s32(vdupq_n_s32(0), lo, a0);
        acc = vdotq_s32(acc, hi, a1);
        sum += (float)vaddvq_s32(acc) * d;
    }
    return sum;
}
#endif

// ---------------------------------------------------------------- dispatch

typedef float (*kernel)(const uint8_t *, const uint8_t *, int);

typedef struct {
    const char *name;
    kernel fn;
    const char *requires;
} entry;

static void quantize_q8_0(const float *v, int n, uint8_t *out) {
    for (int b = 0; b < n / QK; b++) {
        const float *x = v + b * QK;
        float amax = 0.0f;
        for (int i = 0; i < QK; i++) if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
        float d = amax / 127.0f;
        uint32_t bits; memcpy(&bits, &d, 4);
        int e = (int)((bits >> 23) & 0xFF) - 127 + 15;
        uint16_t dh = (uint16_t)((e << 10) | ((bits >> 13) & 0x3FF));
        out[b * Q8B + 0] = (uint8_t)(dh & 0xFF);
        out[b * Q8B + 1] = (uint8_t)(dh >> 8);
        float ds = fp16_to_f32(dh);
        int8_t *q = (int8_t *)(out + b * Q8B + 2);
        for (int i = 0; i < QK; i++) {
            int r = (int)lrintf(ds ? x[i] / ds : 0.0f);
            q[i] = (int8_t)(r > 127 ? 127 : (r < -127 ? -127 : r));
        }
    }
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y);
}

static double bench(kernel fn, const uint8_t *w, const uint8_t *a, int nb, int rows, int reps, float *sink) {
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        float s = 0.0f;
        for (int i = 0; i < rows; i++) s += fn(w + (size_t)i * nb * Q4B, a, nb);
        double t1 = now_s();
        *sink += s;
        if (t1 - t0 < best) best = t1 - t0;
    }
    return best;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.gguf offset row_len rows [runs] [--json out]\n", argv[0]);
        return 2;
    }
    long offset = atol(argv[2]);
    int row_len = atoi(argv[3]), rows = atoi(argv[4]);
    int runs = argc > 5 ? atoi(argv[5]) : 7;
    const char *json_path = (argc > 7 && strcmp(argv[6], "--json") == 0) ? argv[7] : NULL;
    int nb = row_len / QK;

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    size_t span = (size_t)nb * Q4B * rows;
    uint8_t *w = malloc(span);
    if (!w) { fprintf(stderr, "out of memory\n"); return 1; }
    if (fseek(f, offset, SEEK_SET) || fread(w, 1, span, f) != span) { perror("read"); return 1; }
    fclose(f);

    float *act = malloc(sizeof(float) * row_len);
    for (int b = 0; b < nb; b++) {
        float d = scale_f16(rd16(w + b * Q4B));
        for (int j = 0; j < 16; j++) {
            act[b * QK + j] = d * ((float)(w[b * Q4B + 2 + j] & 0x0F) - 8.0f);
            act[b * QK + j + 16] = d * ((float)(w[b * Q4B + 2 + j] >> 4) - 8.0f);
        }
    }
    uint8_t *a = calloc((size_t)nb * Q8B, 1);
    quantize_q8_0(act, row_len, a);

    entry table[6];
    int n = 0;
    table[n++] = (entry){"scalar", dot_scalar, "always"};
#if defined(HAVE_X86)
    table[n++] = (entry){"avx2", dot_avx2, "avx2+fma"};
    if (__builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512bw"))
        table[n++] = (entry){"avx512-vnni", dot_avx512_vnni, "avx512f+vnni+bw+vl"};
#endif
#if defined(HAVE_SDOT)
    table[n++] = (entry){"neon-sdot", dot_neon_scaled, "armv8.2 dotprod"};
    table[n++] = (entry){"neon-acc", dot_neon_acc, "armv8.2 dotprod, no per-block reduce"};
#endif

    if (runs > 64) runs = 64;
    double med[6], lo[6], hi[6];
    double maxrel[6];
    // Identity before timing: a fast wrong answer is the most expensive bug in the book. The comparison is
    // per row and relative, because different vector kernels sum the per-block products in different
    // orders - exact equality is only available to kernels that keep the scalar's order.
    float *ref_rows = malloc(sizeof(float) * rows);
    double norm_a = 0.0;
    for (int i = 0; i < row_len; i++) norm_a += (double)act[i] * act[i];
    norm_a = sqrt(norm_a);
    // Each row's reference is normalised by ||w|| * ||a||, the same scale Chapter 6 used: a relative error on
    // a dot product that happens to be near zero is not a number worth quoting.
    double *den_rows = malloc(sizeof(double) * rows);
    for (int i = 0; i < rows; i++) {
        const uint8_t *row = w + (size_t)i * nb * Q4B;
        ref_rows[i] = dot_scalar(row, a, nb);
        double nw = 0.0;
        for (int b = 0; b < nb; b++) {
            float d = scale_f16(rd16(row + b * Q4B));
            for (int j = 0; j < 16; j++) {
                double lo = d * ((double)(row[b * Q4B + 2 + j] & 0x0F) - 8.0);
                double hi = d * ((double)(row[b * Q4B + 2 + j] >> 4) - 8.0);
                nw += lo * lo + hi * hi;
            }
        }
        den_rows[i] = sqrt(nw) * norm_a + 1e-30;
    }
    for (int k = 0; k < n; k++) {
        double worst = 0.0;
        for (int i = 0; i < rows; i++) {
            float got = table[k].fn(w + (size_t)i * nb * Q4B, a, nb);
            double rel = fabs((double)got - (double)ref_rows[i]) / den_rows[i];
            if (rel > worst) worst = rel;
        }
        maxrel[k] = worst;
    }
    int all_exact = 1;
    for (int k = 0; k < n; k++) if (maxrel[k] != 0.0) all_exact = 0;

    for (int k = 0; k < n; k++) {
        double r[64];
        float sink = 0.0f;
        for (int i = 0; i < runs; i++) r[i] = bench(table[k].fn, w, a, nb, rows, 2, &sink);
        qsort(r, runs, sizeof(double), cmp_double);
        med[k] = r[runs / 2]; lo[k] = r[0]; hi[k] = r[runs - 1];
        if (sink == 12345.0f) printf(" ");        // keep the sink alive without printing it
    }

    const char *machine = 
#if defined(__aarch64__)
        "Apple M4 Pro (arm64)";
#else
        "Intel i7-11700B (x86_64)";
#endif
    int fastest = 0;
    for (int k = 1; k < n; k++) if (med[k] < med[fastest]) fastest = k;

    printf("%s | row_len %d, rows %d, runs %d, median\n", machine, row_len, rows, runs);
    printf("%-12s %10s %10s %10s %10s %10s %10s  %s\n", "kernel", "ns/row", "GB/s", "GMAC/s",
           "vs scalar", "range ms", "max rel vs scalar", "requires");
    for (int k = 0; k < n; k++) {
        double wbytes = (double)nb * Q4B;
        printf("%-12s %10.1f %10.2f %10.2f %9.2fx %6.2f-%-6.2f %10.1e  %s\n", table[k].name,
               med[k] / rows * 1e9, wbytes * rows / med[k] / 1e9, (double)row_len * rows / med[k] / 1e9,
               med[0] / med[k], lo[k] * 1e3, hi[k] * 1e3, maxrel[k], table[k].requires);
    }
    printf("dispatch would pick: %s\n", table[fastest].name);
    printf("identity: %s\n", all_exact ? "every kernel bit-identical to the scalar path" :
           "differences over ||w||*||a||, in the last column: reordered f32 sums, not wrong results");

    if (json_path) {
        FILE *j = fopen(json_path, "w");
        if (j) {
            fprintf(j, "{\n \"machine\": \"%s\",\n", machine);
            fprintf(j, " \"compiler\": \"%s\",\n",
#if defined(__aarch64__)
                    "Apple clang -O2"
#else
                    __VERSION__
#endif
                    );
            fprintf(j, " \"row_len\": %d,\n \"rows\": %d,\n \"runs\": %d,\n", row_len, rows, runs);
            fprintf(j, " \"identity\": \"%s\",\n", all_exact ? "bit-identical" : "within-tolerance");
            fprintf(j, " \"dispatch_picks\": \"%s\",\n", table[fastest].name);
            fprintf(j, " \"kernels\": [\n");
            for (int k = 0; k < n; k++) {
                fprintf(j, "  {\"name\": \"%s\", \"requires\": \"%s\", \"ns_per_row\": %.1f, "
                           "\"gb_per_s\": %.2f, \"gmac_per_s\": %.2f, \"speedup_vs_scalar\": %.2f, "
                           "\"range_ms\": [%.3f, %.3f], \"max_rel_vs_scalar\": %.2e}%s\n",
                        table[k].name, table[k].requires, med[k] / rows * 1e9,
                        (double)nb * Q4B * rows / med[k] / 1e9, (double)row_len * rows / med[k] / 1e9,
                        med[0] / med[k], lo[k] * 1e3, hi[k] * 1e3, maxrel[k], k + 1 < n ? "," : "");
            }
            fprintf(j, " ]\n}\n");
            fclose(j);
            printf("wrote %s\n", json_path);
        }
    }
    free(w); free(a); free(act); free(ref_rows); free(den_rows);
    return 0;
}
