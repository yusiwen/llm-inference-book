/* membw.c - a streaming memory-bandwidth microbenchmark, small enough to read.
 *
 * Why this exists: the book's Chapter 1 turns "bytes read per token" into a decode ceiling, and
 * Chapter 28 refuses a number without its machine. Both need a bandwidth figure that was measured on
 * the machine in question, with the command recorded - not a vendor spec quoted from a web page.
 *
 * What it measures: how fast this machine can stream a large array through one core, and through N
 * cores. Two kernels, because they disagree by a factor of two on most machines:
 *
 *   read    sum a[]                     1 byte moved per byte read
 *   copy    b[i] = a[i]                 2 bytes moved per byte read (one in, one out)
 *
 * Build and run (no flags, no dependencies, works on macOS and Linux):
 *
 *   cc -O2 -pthread -o membw scripts/membw.c
 *   ./membw 1024 8          # 1024 MiB per array, 8 threads
 *
 * It reports the best of three passes for each kernel, in GB/s, using the decimal GB the rest of the
 * book uses. Single-thread numbers are not the machine's bandwidth - a memory controller needs
 * several cores asking at once - which is exactly why the thread count is an argument and why it is
 * printed in the output line.
 *
 * Not a benchmark harness: no warmup budget, no median over runs, no NUMA handling. Chapter 28 is
 * where the discipline lives; this program is the smallest thing that can produce a defensible number
 * for the machines page.
 */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    double *a;
    double *b;
    size_t n;
    int tid;
    int nthreads;
    double sink; /* keeps the compiler from deleting the read */
} job;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *read_kernel(void *arg) {
    job *j = (job *)arg;
    size_t chunk = j->n / (size_t)j->nthreads;
    size_t lo = chunk * (size_t)j->tid;
    size_t hi = (j->tid == j->nthreads - 1) ? j->n : lo + chunk;
    /* Four accumulators, not one: a single `acc` makes the loop latency-bound on the add
     * (about 4 cycles per element), and the number then measures the FPU's dependency chain rather
     * than the memory system. This is the first thing that goes wrong when anyone hand-writes a
     * streaming kernel, so it is spelled out here instead of hidden behind -ffast-math. */
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    size_t i = lo;
    for (; i + 3 < hi; i += 4) {
        a0 += j->a[i];
        a1 += j->a[i + 1];
        a2 += j->a[i + 2];
        a3 += j->a[i + 3];
    }
    for (; i < hi; i++) a0 += j->a[i];
    j->sink = a0 + a1 + a2 + a3;
    return NULL;
}

static void *copy_kernel(void *arg) {
    job *j = (job *)arg;
    size_t chunk = j->n / (size_t)j->nthreads;
    size_t lo = chunk * (size_t)j->tid;
    size_t hi = (j->tid == j->nthreads - 1) ? j->n : lo + chunk;
    memcpy(j->b + lo, j->a + lo, (hi - lo) * sizeof(double));
    return NULL;
}

static double run(void *(*fn)(void *), double *a, double *b, size_t n, int threads) {
    pthread_t th[256];
    job jobs[256];
    double t0 = now();
    for (int i = 0; i < threads; i++) {
        jobs[i] = (job){.a = a, .b = b, .n = n, .tid = i, .nthreads = threads, .sink = 0};
        pthread_create(&th[i], NULL, fn, &jobs[i]);
    }
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
    return now() - t0;
}

int main(int argc, char **argv) {
    size_t mib = (argc > 1) ? (size_t)strtoull(argv[1], NULL, 10) : 1024;
    int threads = (argc > 2) ? atoi(argv[2]) : 8;
    if (threads < 1 || threads > 256) threads = 8;
    size_t n = mib * 1024 * 1024 / sizeof(double);

    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    if (!a || !b) {
        fprintf(stderr, "allocation failed: try a smaller size\n");
        return 1;
    }
    for (size_t i = 0; i < n; i++) a[i] = (double)(i & 1023);
    memset(b, 0, n * sizeof(double));

    double best_read = 0, best_copy = 0;
    for (int pass = 0; pass < 3; pass++) {
        double tr = run(read_kernel, a, b, n, threads);
        double tc = run(copy_kernel, a, b, n, threads);
        double gb = (double)(n * sizeof(double)) / 1e9;
        double read_gbs = gb / tr, copy_gbs = 2.0 * gb / tc;
        if (read_gbs > best_read) best_read = read_gbs;
        if (copy_gbs > best_copy) best_copy = copy_gbs;
    }

    printf("membw %zu MiB x 2 arrays, %d thread(s): read %.1f GB/s, copy %.1f GB/s\n",
           mib, threads, best_read, best_copy);
    return 0;
}
