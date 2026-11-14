/* threadbench.c - Chapter 8's instrument.
 *
 * Three questions, one program:
 *   1. how much of a matmul's cost survives when the rows are split across threads,
 *   2. what a thread costs per call - spawn-and-join versus a persistent pool,
 *   3. how many tokens a matrix-matrix pass needs before the arithmetic (not the weights)
 *      becomes the thing that limits it.
 *
 * The row kernel is deliberately the Chapter 7 scalar kernel: this chapter is about the
 * division of work, and a plain kernel keeps the measurement about the pool. Everything is
 * Q4_0 weights x Q8_0 activations from the book's own file, so the byte counts are the real
 * ones.
 *
 *   cc -O2 -pthread -o threadbench scripts/threadbench.c -lm
 *   ./threadbench FILE OFFSET ID ROWS --json out.json
 */

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

#define Q4B 18
#define Q8B 34
#define MAX_THREADS 64

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y);
}

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

static float scale_f16(uint16_t h) {
#if defined(__aarch64__)
    __fp16 v;
    memcpy(&v, &h, 2);
    return (float)v;
#elif defined(__F16C__)
    return _cvtsh_ss(h);
#else
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
#endif
}

/* One Q4_0 weight row dotted with one Q8_0 activation row. Chapter 7's scalar path. */
static float dot(const uint8_t *w, const uint8_t *a, int nb) {
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


/* ---------------------------------------------------------------- the row kernel
 *
 * The same two instruction sets as Chapter 7, compiled in and chosen once. The point of
 * this chapter is the division of work, but the *shape* of that division depends on whether
 * the row kernel is fast enough to make a pass bandwidth-bound - so the benchmark carries
 * both and says which one it used.
 */
#define HAVE_FAST (defined(__ARM_FEATURE_DOTPROD) || (defined(__x86_64__) && defined(__AVX2__)))

#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
static float dot_neon(const uint8_t *w, const uint8_t *a, int nb) {
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
        facc = vfmaq_n_f32(facc, vcvtq_f32_s32(acc), d);
    }
    return vaddvq_f32(facc);
}
#define DOT_FAST dot_neon
#define KERNEL_NAME "neon-sdot"
#elif defined(__x86_64__) && defined(__AVX2__)
__attribute__((target("avx2,fma")))
static float dot_avx2(const uint8_t *w, const uint8_t *a, int nb) {
    __m256 facc = _mm256_setzero_ps();
    const __m256i mask = _mm256_set1_epi8(0x0F), bias = _mm256_set1_epi8(8);
    for (int b = 0; b < nb; b++) {
        const uint8_t *wb = w + b * Q4B, *ab = a + b * Q8B;
        __m128i raw = _mm_loadu_si128((const __m128i *)(wb + 2));
        __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(raw, 4), raw);
        __m256i qw = _mm256_sub_epi8(_mm256_and_si256(bytes, mask), bias);
        __m256i qa = _mm256_loadu_si256((const __m256i *)(ab + 2));
        __m256i uw = _mm256_sign_epi8(qw, qw);
        __m256i sa = _mm256_sign_epi8(qa, qw);
        __m256i p16 = _mm256_maddubs_epi16(uw, sa);
        __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
        float d = scale_f16(rd16(wb)) * scale_f16(rd16(ab));
        facc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32), facc);
    }
    __m128 lo = _mm256_castps256_ps128(facc), hi = _mm256_extractf128_ps(facc, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    return _mm_cvtss_f32(s4);
}
#define DOT_FAST dot_avx2
#define KERNEL_NAME "avx2"
#else
#define DOT_FAST dot
#define KERNEL_NAME "scalar"
#endif

static int USE_FAST = 1;
static float row_dot(const uint8_t *w, const uint8_t *a, int nb) {
    return USE_FAST ? DOT_FAST(w, a, nb) : dot(w, a, nb);
}

/* ---------------------------------------------------------------- the work
 *
 * One matmul: od rows of weights, id columns, nt tokens. Rows are split because
 * the weight row is the unit that no thread shares with another: row o's bytes are
 * read by exactly one worker, and out[t*od + o] is written by exactly one worker.
 */
typedef struct {
    const uint8_t *w;
    const uint8_t *x;      /* nt activation rows, each nb*Q8B bytes */
    float *out;            /* nt * od */
    int od, id, nb, nt;
} mm_job;

static void mm_rows(const mm_job *j, int r0, int r1) {
    size_t ws = (size_t)j->nb * Q4B, rowb = (size_t)j->nb * Q8B;
    for (int o = r0; o < r1; o++) {
        const uint8_t *wrow = j->w + (size_t)o * ws;
        for (int t = 0; t < j->nt; t++) {
            j->out[(size_t)t * j->od + o] = row_dot(wrow, j->x + (size_t)t * rowb, j->nb);
        }
    }
}

static void chunk(int parts, int idx, int total, int *a, int *b) {
    *a = (int)((long)total * idx / parts);
    *b = (int)((long)total * (idx + 1) / parts);
}

/* ------------------------------------------------------- style 1: spawn per call */

typedef struct { const mm_job *j; int r0, r1; } spawn_arg;

static void *spawn_worker(void *p) {
    spawn_arg *a = (spawn_arg *)p;
    mm_rows(a->j, a->r0, a->r1);
    return NULL;
}

/* Spawns (threads-1) threads and takes the last chunk itself - the same participation
 * count as the pool below, so the only difference between them is the spawn. */
static void run_spawn(const mm_job *j, int threads, spawn_arg *args, pthread_t *tids) {
    for (int i = 0; i < threads - 1; i++) {
        int a, b;
        chunk(threads, i, j->od, &a, &b);
        args[i] = (spawn_arg){j, a, b};
        pthread_create(&tids[i], NULL, spawn_worker, &args[i]);
    }
    int a, b;
    chunk(threads, threads - 1, j->od, &a, &b);
    mm_rows(j, a, b);
    for (int i = 0; i < threads - 1; i++) pthread_join(tids[i], NULL);
}

/* --------------------------------------------------- style 2: persistent pool
 *
 * Mirror of the reference engine's design: a generation counter for the handoff, a done
 * counter for completion, workers that spin briefly and then yield, and the main thread
 * as the last participant. Single submission at a time - the caller holds `gate` across
 * submit and wait, which is what makes a stack-local job safe to hand out.
 */
typedef struct pool {
    int n;                          /* workers, excluding the main thread */
    _Atomic int gen, done, stop;
    mm_job job;
    pthread_mutex_t gate;
    pthread_t *tid;
    struct pool *self;
} pool_t;

static pool_t *POOL;                /* the pool the experiments run against */

typedef struct { pool_t *p; int idx; } worker_arg;

static void *pool_worker(void *arg) {
    worker_arg *wa = (worker_arg *)arg;
    pool_t *p = wa->p;
    int my = wa->idx, seen = 0, spins = 0;
    for (;;) {
        while (atomic_load_explicit(&p->gen, memory_order_seq_cst) == seen) {
            if (atomic_load_explicit(&p->stop, memory_order_seq_cst)) return NULL;
            if (++spins >= 8000) { spins = 0; sched_yield(); }
        }
        seen = atomic_load_explicit(&p->gen, memory_order_seq_cst);
        mm_job j = p->job;
        int a, b;
        chunk(p->n + 1, my, j.od, &a, &b);
        mm_rows(&j, a, b);
        atomic_fetch_add_explicit(&p->done, 1, memory_order_seq_cst);
    }
}

static pool_t *pool_new(int workers) {
    pool_t *p = calloc(1, sizeof(pool_t));
    p->n = workers;
    p->tid = calloc((size_t)workers, sizeof(pthread_t));
    p->self = p;
    atomic_store(&p->gen, 0);
    atomic_store(&p->done, 0);
    atomic_store(&p->stop, 0);
    pthread_mutex_init(&p->gate, NULL);
    static worker_arg args[MAX_THREADS * 4];
    for (int i = 0; i < workers; i++) {
        args[i] = (worker_arg){p, i};
        pthread_create(&p->tid[i], NULL, pool_worker, &args[i]);
    }
    return p;
}

static void pool_free(pool_t *p) {
    atomic_store(&p->stop, 1);
    atomic_fetch_add(&p->gen, 1);            /* wake them so they see stop */
    for (int i = 0; i < p->n; i++) pthread_join(p->tid[i], NULL);
    free(p->tid);
    free(p);
}

static void pool_run(pool_t *p, const mm_job *j) {
    pthread_mutex_lock(&p->gate);
    p->job = *j;
    atomic_fetch_add_explicit(&p->gen, 1, memory_order_seq_cst);
    int a, b;
    chunk(p->n + 1, p->n, j->od, &a, &b);
    mm_rows(j, a, b);
    int spins = 0;
    while (atomic_load_explicit(&p->done, memory_order_seq_cst) < p->n) {
        if (++spins >= 8000) { spins = 0; sched_yield(); }
    }
    atomic_store(&p->done, 0);
    pthread_mutex_unlock(&p->gate);
}

/* ------------------------------------------------------------------ fixtures */

static uint8_t *load_rows(const char *path, long off, size_t bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); exit(2); }
    uint8_t *w = malloc(bytes);
    if (!w) { fprintf(stderr, "alloc %zu\n", bytes); exit(2); }
    if (fseek(f, off, SEEK_SET) || fread(w, 1, bytes, f) != bytes) {
        fprintf(stderr, "short read\n");
        exit(2);
    }
    fclose(f);
    return w;
}

/* Activation rows: real weights, quantized to Q8_0 the same way Chapter 6 did. */
static uint8_t *make_acts(const uint8_t *w, int nb, int nt, size_t rowb) {
    uint8_t *a = calloc((size_t)nt, rowb);
    float *f = malloc(sizeof(float) * (size_t)nb * 32);
    for (int t = 0; t < nt; t++) {
        const uint8_t *src = w + (size_t)t * (size_t)nb * Q4B;
        for (int b = 0; b < nb; b++) {
            float d = scale_f16(rd16(src + b * Q4B));
            for (int j = 0; j < 16; j++) {
                f[b * 32 + j] = d * ((float)(src[b * Q4B + 2 + j] & 0x0F) - 8.0f);
                f[b * 32 + j + 16] = d * ((float)(src[b * Q4B + 2 + j] >> 4) - 8.0f);
            }
        }
        for (int b = 0; b < nb; b++) {
            float amax = 0.0f;
            for (int i = 0; i < 32; i++) if (fabsf(f[b * 32 + i]) > amax) amax = fabsf(f[b * 32 + i]);
            float d = amax / 127.0f;
            uint32_t bits;
            memcpy(&bits, &d, 4);
            int e = (int)((bits >> 23) & 0xFF) - 127 + 15;
            uint16_t dh = (uint16_t)((e << 10) | ((bits >> 13) & 0x3FF));
            uint8_t *dst = a + (size_t)t * rowb + (size_t)b * Q8B;
            dst[0] = (uint8_t)(dh & 0xFF);
            dst[1] = (uint8_t)(dh >> 8);
            float ds = scale_f16(dh);
            for (int i = 0; i < 32; i++) {
                int q = (int)lrintf(ds > 0.0f ? f[b * 32 + i] / ds : 0.0f);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                ((int8_t *)(dst + 2))[i] = (int8_t)q;
            }
        }
    }
    free(f);
    return a;
}

/* ---------------------------------------------------------------- experiments */

static double median_of(const double *v, int n) { return v[n / 2]; }

typedef struct {
    int threads;
    double pool_ms, spawn_ms, gbs_pool, gbs_spawn, speedup_pool;
    double pool_lo, pool_hi;
} row_t;

/* One point per thread count. The pool is created and destroyed for each point, outside the
 * timed region: a pool is a resource you build once, but the question "does a tenth thread
 * help" needs a pool of that size, not a pool of ten used by two. */
static void experiment_threads(const char *path, long off, int id, int rows, const int *counts,
                               int ncounts, row_t *out, int *nout) {
    int nb = id / 32;
    size_t ws = (size_t)nb * Q4B, bytes = ws * (size_t)rows;
    uint8_t *w = load_rows(path, off, bytes);
    uint8_t *x = make_acts(w, nb, 1, (size_t)nb * Q8B);
    float *o = malloc(sizeof(float) * (size_t)rows);
    mm_job job = {w, x, o, rows, id, nb, 1};
    spawn_arg *args = calloc(MAX_THREADS, sizeof(spawn_arg));
    pthread_t *tids = calloc(MAX_THREADS, sizeof(pthread_t));
    mm_rows(&job, 0, rows);                                  /* touch the pages first */

    *nout = 0;
    for (int c = 0; c < ncounts; c++) {
        int t = counts[c];
        double mp[5], ms[5];
        pool_t *pool = t > 1 ? pool_new(t - 1) : NULL;
        if (pool) { pool_run(pool, &job); pool_run(pool, &job); }
        for (int r = 0; r < 5; r++) {
            double t0 = now_s();
            if (pool) pool_run(pool, &job); else mm_rows(&job, 0, rows);
            mp[r] = (now_s() - t0) * 1e3;
            t0 = now_s();
            if (t == 1) mm_rows(&job, 0, rows); else run_spawn(&job, t, args, tids);
            ms[r] = (now_s() - t0) * 1e3;
        }
        if (pool) pool_free(pool);
        qsort(mp, 5, sizeof(double), cmp_double);
        qsort(ms, 5, sizeof(double), cmp_double);
        row_t *r = &out[(*nout)++];
        r->threads = t;
        r->pool_ms = median_of(mp, 5);
        r->spawn_ms = median_of(ms, 5);
        r->pool_lo = mp[0];
        r->pool_hi = mp[4];
        r->gbs_pool = bytes / (r->pool_ms * 1e-3) / 1e9;
        r->gbs_spawn = bytes / (r->spawn_ms * 1e-3) / 1e9;
        r->speedup_pool = out[0].pool_ms / r->pool_ms;
    }
    free(args); free(tids); free(o); free(x); free(w);
}

typedef struct {
    int tokens;
    double gemm_ms, gemv_ms_total, gbs, per_token_ms;
} tok_t;

static void experiment_tokens(const char *path, long off, int id, int rows, int threads,
                              const int *tokens, int ntok, tok_t *out, int *nout) {
    int nb = id / 32;
    int maxt = 1;
    for (int i = 0; i < ntok; i++) if (tokens[i] > maxt) maxt = tokens[i];
    size_t ws = (size_t)nb * Q4B, bytes = ws * (size_t)rows, rowb = (size_t)nb * Q8B;
    uint8_t *w = load_rows(path, off, bytes);
    uint8_t *x = make_acts(w, nb, maxt, rowb);
    float *o = calloc((size_t)maxt * rows, sizeof(float));
    spawn_arg *args = calloc(MAX_THREADS, sizeof(spawn_arg));
    pthread_t *tids = calloc(MAX_THREADS, sizeof(pthread_t));

    pool_t *pool = threads > 1 ? pool_new(threads - 1) : NULL;
    if (pool) pool_run(pool, &(mm_job){w, x, o, rows, id, nb, 1});   /* warm */

    *nout = 0;
    for (int i = 0; i < ntok; i++) {
        int nt = tokens[i];
        double mg[5], mv[5];
        for (int r = 0; r < 5; r++) {
            mm_job j = {w, x, o, rows, id, nb, nt};
            double t0 = now_s();
            if (pool) pool_run(pool, &j); else mm_rows(&j, 0, rows);
            mg[r] = (now_s() - t0) * 1e3;
            /* the same nt tokens, one at a time through a GEMV-shaped call */
            t0 = now_s();
            for (int t = 0; t < nt; t++) {
                mm_job g = {w, x + (size_t)t * rowb, o + (size_t)t * rows, rows, id, nb, 1};
                if (pool) pool_run(pool, &g); else mm_rows(&g, 0, rows);
            }
            mv[r] = (now_s() - t0) * 1e3;
        }
        qsort(mg, 5, sizeof(double), cmp_double);
        qsort(mv, 5, sizeof(double), cmp_double);
        tok_t *t = &out[(*nout)++];
        t->tokens = nt;
        t->gemm_ms = median_of(mg, 5);
        t->gemv_ms_total = median_of(mv, 5);
        t->per_token_ms = t->gemm_ms / nt;
        t->gbs = bytes / (t->gemm_ms * 1e-3) / 1e9;
    }
    if (pool) pool_free(pool);
    free(args); free(tids); free(o); free(x); free(w);
}

/* How much a *call* costs when there is almost no work in it. Rows are few on purpose:
 * this measures the dispatch, and a dispatch that is invisible at 1,000 tokens can be the
 * whole cost at one. */
static void experiment_overhead(const char *path, long off, int id, int rows, int threads,
                                double *pool_us, double *spawn_us, double *serial_us) {
    int nb = id / 32;
    size_t ws = (size_t)nb * Q4B, bytes = ws * (size_t)rows;
    uint8_t *w = load_rows(path, off, bytes);
    uint8_t *x = make_acts(w, nb, 1, (size_t)nb * Q8B);
    float *o = malloc(sizeof(float) * (size_t)rows);
    mm_job job = {w, x, o, rows, id, nb, 1};
    spawn_arg *args = calloc(MAX_THREADS, sizeof(spawn_arg));
    pthread_t *tids = calloc(MAX_THREADS, sizeof(pthread_t));
    pool_t *pool = pool_new(threads - 1);
    pool_run(pool, &job);

    const int calls = 200;
    double a[5], b[5], c[5];
    for (int r = 0; r < 5; r++) {
        double t0 = now_s();
        for (int i = 0; i < calls; i++) pool_run(pool, &job);
        a[r] = (now_s() - t0) / calls * 1e6;
        t0 = now_s();
        for (int i = 0; i < calls; i++) run_spawn(&job, threads, args, tids);
        b[r] = (now_s() - t0) / calls * 1e6;
        t0 = now_s();
        for (int i = 0; i < calls; i++) mm_rows(&job, 0, rows);
        c[r] = (now_s() - t0) / calls * 1e6;
    }
    qsort(a, 5, sizeof(double), cmp_double);
    qsort(b, 5, sizeof(double), cmp_double);
    qsort(c, 5, sizeof(double), cmp_double);
    *pool_us = median_of(a, 5);
    *spawn_us = median_of(b, 5);
    *serial_us = median_of(c, 5);
    pool_free(pool);
    free(w); free(x); free(o); free(args); free(tids);
}


/* Where threading stops paying: the same call at shrinking shapes. Below some row count the
 * dispatch is the whole cost, and the pool's fixed overhead loses to one thread that simply
 * starts working. */
typedef struct { int rows; double serial_us, pool_us, spawn_us, pool_lo, pool_hi; } thr_t;

static void experiment_threshold(const char *path, long off, int id, int threads, const int *shapes,
                                 int nshapes, thr_t *out, int *nout) {
    int nb = id / 32;
    size_t ws = (size_t)nb * Q4B;
    int maxrows = 0;
    for (int i = 0; i < nshapes; i++) if (shapes[i] > maxrows) maxrows = shapes[i];
    uint8_t *w = load_rows(path, off, ws * (size_t)maxrows);
    uint8_t *x = make_acts(w, nb, 1, (size_t)nb * Q8B);
    float *o = malloc(sizeof(float) * (size_t)maxrows);
    spawn_arg *args = calloc(MAX_THREADS, sizeof(spawn_arg));
    pthread_t *tids = calloc(MAX_THREADS, sizeof(pthread_t));
    pool_t *pool = pool_new(threads - 1);

    *nout = 0;
    for (int i = 0; i < nshapes; i++) {
        int rows = shapes[i];
        mm_job job = {w, x, o, rows, id, nb, 1};
        int calls = rows <= 4096 ? 200 : 20;
        pool_run(pool, &job);
        double a[5], b[5], c[5];
        for (int r = 0; r < 5; r++) {
            double t0 = now_s();
            for (int k = 0; k < calls; k++) mm_rows(&job, 0, rows);
            c[r] = (now_s() - t0) / calls * 1e6;
            t0 = now_s();
            for (int k = 0; k < calls; k++) pool_run(pool, &job);
            a[r] = (now_s() - t0) / calls * 1e6;
            t0 = now_s();
            for (int k = 0; k < calls; k++) run_spawn(&job, threads, args, tids);
            b[r] = (now_s() - t0) / calls * 1e6;
        }
        qsort(a, 5, sizeof(double), cmp_double);
        qsort(b, 5, sizeof(double), cmp_double);
        qsort(c, 5, sizeof(double), cmp_double);
        thr_t *t = &out[(*nout)++];
        t->rows = rows;
        t->serial_us = median_of(c, 5);
        t->pool_us = median_of(a, 5);
        t->spawn_us = median_of(b, 5);
        t->pool_lo = a[0];
        t->pool_hi = a[4];
    }
    pool_free(pool);
    free(w); free(x); free(o); free(args); free(tids);
}

/* Interleaved split: worker k takes rows k, k+parts, k+2*parts, ... Same bytes read, same
 * dots, disjoint outputs - but two workers' writes now share cache lines. */
typedef struct { const uint8_t *w; const uint8_t *x; float *o; int od, nb, parts, k; } sj;

static void *strided_worker(void *p) {
    sj *q = (sj *)p;
    size_t wsx = (size_t)q->nb * Q4B;
    for (int o = q->k; o < q->od; o += q->parts) q->o[o] = row_dot(q->w + (size_t)o * wsx, q->x, q->nb);
    return NULL;
}

/* False sharing: two ways to write the same output vector with the same total work.
 * Adjacent rows in one array are the good case; a stride that puts two threads' writes in
 * the same 64-byte line is the bad one. */
static void experiment_sharing(const char *path, long off, int id, int rows, int threads,
                               double *contig_ms, double *strided_ms) {
    int nb = id / 32;
    size_t ws = (size_t)nb * Q4B, bytes = ws * (size_t)rows;
    uint8_t *w = load_rows(path, off, bytes);
    uint8_t *x = make_acts(w, nb, 1, (size_t)nb * Q8B);
    float *o = malloc(sizeof(float) * (size_t)rows);
    pool_t *pool = pool_new(threads - 1);

    int parts = threads;
    double a[3], b[3];
    for (int r = 0; r < 3; r++) {
        double t0 = now_s();
        mm_job j = {w, x, o, rows, id, nb, 1};
        pool_run(pool, &j);
        a[r] = (now_s() - t0) * 1e3;

        /* Same rows, same dots, but each worker writes into a different slice of every
         * line: worker k takes rows k, k+parts, k+2*parts, ... The bytes read are the same
         * and the outputs are disjoint - only the cache lines are shared. */
        t0 = now_s();
        sj s = {w, x, o, rows, nb, parts, 0};
        pthread_t t[MAX_THREADS];
        sj sa[MAX_THREADS];
        for (int k = 0; k < parts; k++) {
            sa[k] = s;
            sa[k].k = k;
            pthread_create(&t[k], NULL, strided_worker, &sa[k]);
        }
        for (int k = 0; k < parts; k++) pthread_join(t[k], NULL);
        b[r] = (now_s() - t0) * 1e3;
    }
    qsort(a, 3, sizeof(double), cmp_double);
    qsort(b, 3, sizeof(double), cmp_double);
    *contig_ms = median_of(a, 3);
    *strided_ms = median_of(b, 3);
    pool_free(pool);
    free(w); free(x); free(o);
}

static int detect_cores(void) {
    long n = 0;
#if defined(__APPLE__)
    FILE *p = popen("sysctl -n hw.perflevel0.logicalcpu 2>/dev/null", "r");
    if (p) { if (fscanf(p, "%ld", &n) != 1) n = 0; pclose(p); }
    if (n <= 0) { p = popen("sysctl -n hw.ncpu", "r"); if (p) { if (fscanf(p, "%ld", &n) != 1) n = 0; pclose(p); } }
#else
    n = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    if (n <= 0) n = 1;
    if (n > MAX_THREADS) n = MAX_THREADS;
    return (int)n;
}

static void machine_name(char *buf, size_t len) {
#if defined(__aarch64__) && defined(__APPLE__)
    FILE *p = popen("sysctl -n machdep.cpu.brand_string", "r");
    if (p) { if (!fgets(buf, (int)len, p)) snprintf(buf, len, "arm64"); pclose(p); }
    char *nl = strchr(buf, '\n');
    if (nl) *nl = 0;
    snprintf(buf + strlen(buf), len - strlen(buf), " (arm64)");
#elif defined(__x86_64__)
    FILE *f = fopen("/proc/cpuinfo", "r");
    char line[512];
    snprintf(buf, len, "x86_64");
    if (f) {
        while (fgets(line, sizeof line, f)) {
            if (!strncmp(line, "model name", 10)) {
                char *c = strchr(line, ':');
                if (c) {
                    char *v = c + 2;
                    char *nl = strchr(v, '\n');
                    if (nl) *nl = 0;
                    snprintf(buf, len, "%s (x86_64)", v);
                }
                break;
            }
        }
        fclose(f);
    }
#else
    snprintf(buf, len, "unknown");
#endif
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s FILE OFFSET ID ROWS [--json out.json] [--threads N]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    long off = atol(argv[2]);
    int id = atoi(argv[3]), rows = atoi(argv[4]);
    const char *json = NULL;
    int threads = 0;
    for (int i = 5; i < argc; i++) {
        if (!strcmp(argv[i], "--json") && i + 1 < argc) json = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--scalar")) USE_FAST = 0;
    }
    int cores = detect_cores();
    if (threads > 0) cores = threads;
    if (cores < 2) cores = 2;

    char machine[256];
    machine_name(machine, sizeof machine);
    int nb = id / 32;
    size_t ws = (size_t)nb * Q4B;

    printf("%s | %d cores | id %d, %d rows, %.1f MB per pass | row kernel: %s\n", machine, cores, id,
           rows, (double)ws * rows / 1e6, USE_FAST ? KERNEL_NAME : "scalar");

    /* 1. the thread sweep, on the whole matrix */
    int counts[8], ncounts = 0;
    for (int t = 1; t <= cores; t *= 2) counts[ncounts++] = t;
    if (counts[ncounts - 1] != cores) counts[ncounts++] = cores;
    row_t rs[8];
    int nrs = 0;
    experiment_threads(path, off, id, rows, counts, ncounts, rs, &nrs);

    printf("\n%-8s %12s %12s %10s %10s %9s\n", "threads", "pool ms", "spawn ms", "pool GB/s",
           "spawn GB/s", "speedup");
    for (int i = 0; i < nrs; i++)
        printf("%-8d %12.2f %12.2f %10.2f %10.2f %8.2fx\n", rs[i].threads, rs[i].pool_ms,
               rs[i].spawn_ms, rs[i].gbs_pool, rs[i].gbs_spawn, rs[i].speedup_pool);

    /* 2. one pass at increasing token counts: the flat region and where it bends */
    int tk[10] = {1, 2, 4, 8, 16, 32, 64, 128, 0, 0};
    int ntk = 8;
    if (rows > 32768) { tk[4] = 16; tk[5] = 32; ntk = 6; }   /* keep the output array sane */
    tok_t ts[10];
    int nts = 0;
    experiment_tokens(path, off, id, rows, cores, tk, ntk, ts, &nts);
    printf("\n%-8s %10s %10s %12s %12s %10s\n", "tokens", "gemm ms", "gemv ms", "per token ms",
           "GB/s", "gemv/gemm");
    for (int i = 0; i < nts; i++)
        printf("%-8d %10.3f %10.3f %12.3f %12.2f %9.2fx\n", ts[i].tokens, ts[i].gemm_ms,
               ts[i].gemv_ms_total, ts[i].per_token_ms, ts[i].gbs,
               ts[i].gemm_ms > 0 ? ts[i].gemv_ms_total / ts[i].gemm_ms : 0.0);

    /* 3. per-call overhead at a small shape */
    int small = rows < 4096 ? rows : 4096;
    double pool_us = 0, spawn_us = 0, serial_us = 0;
    experiment_overhead(path, off, id, small, cores, &pool_us, &spawn_us, &serial_us);
    printf("\nper call, %d rows (%.2f MB): serial %.1f us | pool %.1f us | spawn %.1f us\n", small,
           (double)ws * small / 1e6, serial_us, pool_us, spawn_us);

    /* 3b. where threading stops paying */
    int shapes[8] = {128, 256, 512, 1024, 2048, 4096, 16384, 0};
    int nshapes = 7;
    thr_t th[8];
    int nth = 0;
    experiment_threshold(path, off, id, cores, shapes, nshapes, th, &nth);
    printf("\n%-8s %12s %12s %12s %10s\n", "rows", "serial us", "pool us", "spawn us", "pool/serial");
    for (int i = 0; i < nth; i++)
        printf("%-8d %12.2f %12.2f %12.2f %9.2fx\n", th[i].rows, th[i].serial_us, th[i].pool_us,
               th[i].spawn_us, th[i].serial_us / th[i].pool_us);

    /* 4. false sharing */
    double contig = 0, strided = 0;
    experiment_sharing(path, off, id, small, cores, &contig, &strided);
    printf("row split, %d threads: contiguous %.2f ms | interleaved %.2f ms\n", cores, contig,
           strided);

    if (json) {
        FILE *j = fopen(json, "w");
        if (!j) { perror("json"); return 2; }
        fprintf(j, "{\n \"artifact\": \"threadbench\",\n \"machine\": \"%s\",\n \"cores\": %d,\n",
                machine, cores);
        fprintf(j, " \"shape\": {\"offset\": %ld, \"id\": %d, \"rows\": %d, \"weight_bytes_per_pass\": %zu},\n",
                off, id, rows, ws * rows);
        fprintf(j, " \"row_kernel\": \"%s\",\n \"small_shape\": %d,\n", KERNEL_NAME, small);
        fprintf(j, " \"invocation\": \"cc -O2 -pthread -o threadbench scripts/threadbench.c -lm ; "
                   "./threadbench FILE %ld %d %d --json facts/threadbench.json\",\n", off, id, rows);
        fprintf(j, " \"thread_sweep\": [\n");
        for (int i = 0; i < nrs; i++)
            fprintf(j, "  {\"threads\": %d, \"pool_ms\": %.3f, \"pool_range_ms\": [%.3f, %.3f], "
                       "\"spawn_ms\": %.3f, \"pool_gbs\": %.2f, \"spawn_gbs\": %.2f, \"speedup\": %.2f}%s\n",
                    rs[i].threads, rs[i].pool_ms, rs[i].pool_lo, rs[i].pool_hi, rs[i].spawn_ms,
                    rs[i].gbs_pool, rs[i].gbs_spawn, rs[i].speedup_pool, i + 1 < nrs ? "," : "");
        fprintf(j, " ],\n \"token_sweep\": [\n");
        for (int i = 0; i < nts; i++)
            fprintf(j, "  {\"tokens\": %d, \"gemm_ms\": %.4f, \"gemv_ms\": %.4f, \"per_token_ms\": %.4f, "
                       "\"gbs\": %.2f, \"gemv_over_gemm\": %.2f}%s\n",
                    ts[i].tokens, ts[i].gemm_ms, ts[i].gemv_ms_total, ts[i].per_token_ms, ts[i].gbs,
                    ts[i].gemm_ms > 0 ? ts[i].gemv_ms_total / ts[i].gemm_ms : 0.0,
                    i + 1 < nts ? "," : "");
        fprintf(j, " ],\n \"overhead_us\": {\"rows\": %d, \"serial\": %.2f, \"pool\": %.2f, "
                   "\"spawn\": %.2f},\n", small, serial_us, pool_us, spawn_us);
        fprintf(j, " \"threshold_sweep\": [\n");
        for (int i = 0; i < nth; i++)
            fprintf(j, "  {\"rows\": %d, \"serial_us\": %.2f, \"pool_us\": %.2f, \"pool_range_us\": [%.2f, %.2f], "
                       "\"spawn_us\": %.2f}%s\n",
                    th[i].rows, th[i].serial_us, th[i].pool_us, th[i].pool_lo, th[i].pool_hi,
                    th[i].spawn_us, i + 1 < nth ? "," : "");
        fprintf(j, " ],\n \"false_sharing_ms\": {\"rows\": %d, \"threads\": %d, \"contiguous\": %.3f, "
                   "\"interleaved\": %.3f}\n}\n", small, cores, contig, strided);
        fclose(j);
    }
    return 0;
}
