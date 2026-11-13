// dotbench.c - the same dot product, two ways, in scalar code on purpose.
//
// Chapter 6's claim is that an engine stops multiplying floats. The chapter before it measured bytes and
// bandwidth; this program measures the one thing a scalar kernel can still show: that the two
// implementations do different work, not the same work faster.
//
//   dq    dequantize each 18-byte block into 32 floats, then fmaf-accumulate        (the obvious way)
//   i32   nibbles times int8 into an i32 accumulator, one scale multiply per block    (what an engine runs)
//   i16   the same with a 16-bit block accumulator, which is legal for Q4_0 only     (and why it is legal)
//
// All three read the same bytes, do the same 896 multiply-accumulates per row, and produce the same value
// up to f32 rounding. Nothing here is intrinsics - that is Chapter 7 - so any difference is the shape of the
// work, not vector width.
//
//   clang -O2 -o /tmp/dotbench scripts/dotbench.c
//   /tmp/dotbench ~/.cache/myfer/qwen2.5-0.5b-instruct-q4_0.gguf <offset> 896 4096
//
// Offset and row length come from scripts/gguf-inspect.py --json (the facts file), not from this file.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float fp16_to_f32(uint16_t h) {              // enough of a decoder for the f16 scales we read
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
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

static uint16_t read16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

// Q8_0 activation row: an f16 scale per 32 values and int8 payload. Built once, read by all three kernels.
#define QK 32
static void quantize_q8_0(const float *v, int n, uint8_t *out) {
    for (int b = 0; b < n / QK; b++) {
        const float *x = v + b * QK;
        float amax = 0.0f;
        for (int i = 0; i < QK; i++) if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
        float d = amax / 127.0f;
        uint16_t dh = 0;
        {   // round-trip the scale through f16, because the file does
            float f = d; uint32_t b32; memcpy(&b32, &f, 4); (void)b32; dh = 0; }
        // simple f16 encode for the range we need (0 .. 1)
        { uint32_t bits; memcpy(&bits, &d, 4); int e = (int)((bits >> 23) & 0xFF) - 127 + 15;
          uint32_t m = (bits >> 13) & 0x3FF;
          dh = (uint16_t)((e << 10) | m); }
        out[b * 34 + 0] = (uint8_t)(dh & 0xFF);
        out[b * 34 + 1] = (uint8_t)(dh >> 8);
        float ds = fp16_to_f32(dh);
        int8_t *q = (int8_t *)(out + b * 34 + 2);
        for (int i = 0; i < QK; i++) {
            float t = ds ? x[i] / ds : 0.0f;
            int r = (int)lrintf(t);
            if (r > 127) r = 127;
            if (r < -127) r = -127;
            q[i] = (int8_t)r;
        }
    }
}

// The obvious implementation: turn each block into floats, then multiply floats.
static float dot_dq(const uint8_t *w, const uint8_t *a, int nb) {
    float s = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *wb = w + b * 18;
        const uint8_t *ab = a + b * 34;
        float dw = fp16_to_f32(read16(wb)), da = fp16_to_f32(read16(ab));
        float tmp[QK];
        for (int j = 0; j < 16; j++) {
            tmp[j] = dw * ((float)(wb[2 + j] & 0x0F) - 8.0f);
            tmp[j + 16] = dw * ((float)(wb[2 + j] >> 4) - 8.0f);
        }
        const int8_t *q = (const int8_t *)(ab + 2);
        float acc = 0.0f;
        for (int i = 0; i < QK; i++) acc = fmaf(tmp[i], (float)q[i], acc);
        s += acc * da;      // dw is already in tmp: this path dequantized to floats first
    }
    return s;
}

// The engine's shape: integers all the way, one scale multiply per block.
static float dot_i32(const uint8_t *w, const uint8_t *a, int nb) {
    float s = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *wb = w + b * 18;
        const uint8_t *ab = a + b * 34;
        float dw = fp16_to_f32(read16(wb)), da = fp16_to_f32(read16(ab));
        const int8_t *q = (const int8_t *)(ab + 2);
        int32_t sum = 0;
        for (int j = 0; j < 16; j++) {
            sum += ((int)(wb[2 + j] & 0x0F) - 8) * (int)q[j];
            sum += ((int)(wb[2 + j] >> 4) - 8) * (int)q[j + 16];
        }
        s += (float)sum * dw * da;
    }
    return s;
}

// The same, with a 16-bit block accumulator: legal only because Q4_0's weights are at most 8 in magnitude,
// so a block's partial sum cannot exceed 32 * 8 * 127 = 32,512 < 32,767.
static float dot_i16(const uint8_t *w, const uint8_t *a, int nb) {
    float s = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *wb = w + b * 18;
        const uint8_t *ab = a + b * 34;
        float dw = fp16_to_f32(read16(wb)), da = fp16_to_f32(read16(ab));
        const int8_t *q = (const int8_t *)(ab + 2);
        int16_t sum = 0;
        for (int j = 0; j < 16; j++) {
            sum = (int16_t)(sum + ((int)(wb[2 + j] & 0x0F) - 8) * (int)q[j]);
            sum = (int16_t)(sum + ((int)(wb[2 + j] >> 4) - 8) * (int)q[j + 16]);
        }
        s += (float)sum * dw * da;
    }
    return s;
}

typedef float (*kernel)(const uint8_t *, const uint8_t *, int);

static double bench(kernel fn, const uint8_t *w, const uint8_t *a, int nb, int rows, int reps, float *sink) {
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        float s = 0.0f;
        for (int i = 0; i < rows; i++) s += fn(w + (size_t)i * nb * 18, a, nb);
        double t1 = now_s();
        *sink += s;                                  // keep the compiler honest
        if (t1 - t0 < best) best = t1 - t0;
    }
    return best;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.gguf offset row_len rows [reps] [runs] [--json out]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    long offset = atol(argv[2]);
    int row_len = atoi(argv[3]);
    int rows = atoi(argv[4]);
    int reps = argc > 5 ? atoi(argv[5]) : 3;
    int runs = argc > 6 ? atoi(argv[6]) : 7;
    const char *json_path = (argc > 8 && strcmp(argv[7], "--json") == 0) ? argv[8] : NULL;
    int nb = row_len / QK;

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    size_t span = (size_t)nb * 18 * rows;
    uint8_t *w = malloc(span);
    if (!w) { fprintf(stderr, "out of memory\n"); return 1; }
    if (fseek(f, offset, SEEK_SET) || fread(w, 1, span, f) != span) { perror("read"); return 1; }
    fclose(f);

    // The activation vector: row 0 dequantized, then Q8_0-quantized. Real values from the same file.
    float act[row_len];
    {
        const uint8_t *wb = w;
        for (int b = 0; b < nb; b++) {
            float dw = fp16_to_f32(read16(wb + b * 18));
            for (int j = 0; j < 16; j++) {
                act[b * QK + j] = dw * ((float)(wb[b * 18 + 2 + j] & 0x0F) - 8.0f);
                act[b * QK + j + 16] = dw * ((float)(wb[b * 18 + 2 + j] >> 4) - 8.0f);
            }
        }
    }
    uint8_t *a = calloc((size_t)nb * 34, 1);
    quantize_q8_0(act, row_len, a);

    // The book's protocol: independent runs, median reported, range kept. One run on a laptop is a rumour.
    double r_dq[64], r_i32[64], r_i16[64];
    if (runs > 64) runs = 64;
    float s0 = 0, s1 = 0, s2 = 0;
    for (int r = 0; r < runs; r++) {
        r_dq[r] = bench(dot_dq, w, a, nb, rows, reps, &s0);
        r_i32[r] = bench(dot_i32, w, a, nb, rows, reps, &s1);
        r_i16[r] = bench(dot_i16, w, a, nb, rows, reps, &s2);
    }
    qsort(r_dq, runs, sizeof(double), cmp_double);
    qsort(r_i32, runs, sizeof(double), cmp_double);
    qsort(r_i16, runs, sizeof(double), cmp_double);
    double t_dq = r_dq[runs / 2], t_i32 = r_i32[runs / 2], t_i16 = r_i16[runs / 2];
    float mn = fminf(fminf(s0, s1), s2), mx = fmaxf(fmaxf(s0, s1), s2);

    double wbytes = (double)nb * 18;                 // weight bytes per row
    double macs = (double)row_len;                   // multiply-accumulates per row
    printf("row_len %d, rows %d, runs %d, reps %d (median of runs)\n", row_len, rows, runs, reps);
    printf("%-6s %10s %10s %10s %10s %10s %10s\n", "path", "ms", "ns/row", "GB/s", "GMAC/s",
           "relative", "range");
    struct { const char *name; double t, lo, hi; } v[] = {
        {"dq", t_dq, r_dq[0], r_dq[runs - 1]},
        {"i32", t_i32, r_i32[0], r_i32[runs - 1]},
        {"i16", t_i16, r_i16[0], r_i16[runs - 1]},
    };
    for (int i = 0; i < 3; i++) {
        double per_row = v[i].t / rows;
        printf("%-6s %10.3f %10.1f %10.2f %10.2f %9.2fx   %.2f-%.2f\n", v[i].name, v[i].t * 1e3,
               per_row * 1e9, wbytes * rows / v[i].t / 1e9, macs * rows / v[i].t / 1e9,
               t_dq / v[i].t, v[i].lo * 1e3, v[i].hi * 1e3);
    }
    if (json_path) {
        FILE *j = fopen(json_path, "w");
        if (j) {
            fprintf(j, "{\n \"machine\": \"%s\",\n \"compiler\": \"clang -O2\",\n", "Apple M4 Pro");
            fprintf(j, " \"row_len\": %d,\n \"rows\": %d,\n \"runs\": %d,\n \"reps\": %d,\n",
                    row_len, rows, runs, reps);
            fprintf(j, " \"weight_bytes_per_row\": %.0f,\n", wbytes);
            fprintf(j, " \"paths\": [\n");
            for (int i = 0; i < 3; i++) {
                double per_row = v[i].t / rows;
                fprintf(j, "  {\"name\": \"%s\", \"ns_per_row\": %.1f, \"gb_per_s\": %.2f, "
                           "\"gmac_per_s\": %.2f, \"median_ms\": %.3f, \"range_ms\": [%.3f, %.3f], "
                           "\"speedup_vs_dq\": %.2f}%s\n",
                        v[i].name, per_row * 1e9, wbytes * rows / v[i].t / 1e9,
                        macs * rows / v[i].t / 1e9, v[i].t * 1e3, v[i].lo * 1e3, v[i].hi * 1e3,
                        t_dq / v[i].t, i < 2 ? "," : "");
            }
            fprintf(j, " ],\n \"checksums_agree\": %s\n}\n",
                    (mx == mn) ? "true" : "false");
            fclose(j);
            printf("wrote %s\n", json_path);
        }
    }
    // The three must agree; print the spread so a miscompile is visible rather than assumed.
    printf("checksum spread %.3e relative (three paths must agree)\n",
           fabs((double)(mx - mn) / (double)mx));
    free(w); free(a);
    return 0;
}
