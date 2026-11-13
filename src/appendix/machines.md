# The Reference Machines

> **Status**: draft - two of the three machines are measured, one is waiting for its address; the numbers below are one afternoon of measurements, not a benchmark campaign.
> **Rule of this book**: a number names its machine, and a machine names its measured bandwidth. A spec sheet is not a measurement, and this page says which is which.

Every performance claim in this book is either *measured on one of these machines* or *arithmetic on stated assumptions*. The arithmetic needs one number more than people expect: for decode, the ceiling is the model's bytes per token divided by how fast this machine can actually stream memory - not by the bandwidth printed on the box. So the machines come first, with the measurement that produced their number.

## The three machines, and what each one is for

| Machine | Path it represents | CPU | Memory | Chapters |
|---|---|---|---|---|
| **MacBook Pro, M4 Pro** | CPU + Metal | 14 cores (10 performance, 4 efficiency), arm64 | 48 GiB unified | 1-20, 21, 27, 28 |
| **DGX Spark, NVIDIA GB10** | CUDA | 20 cores, aarch64 | 121 GiB unified (Grace + Blackwell) | 22-26, and every CUDA number in 24-25 |
| **NUC11** | x86-64 with AVX2, the CPU path on a non-ARM machine | x86-64 | to record | 6, 7 (the AVX2 halves of the SIMD chapters) |

Three paths, because they fail differently: ARM's NEON with `SDOT` does integer dot products in one instruction, x86 needs AVX2 plus `dp4a`-style tricks, and the two GPUs disagree about memory entirely - one shares it with the CPU, the other does not.

## How the bandwidth was measured

A 60-line C program, in this repository: [`scripts/membw.c`](https://github.com/yusiwen/llm-inference-book/blob/main/scripts/membw.c). It streams a large array through N threads with two kernels - a sum over the array (1 byte moved per byte read) and a copy (2 bytes per byte) - and reports the best of three passes. No benchmark harness, no warmup budget: that discipline is Chapter 28, and this page only needs the number to be honest about its conditions.

```bash
cc -O2 -pthread -o membw scripts/membw.c
./membw 1024 10        # 1024 MiB per array, 10 threads
```

Two decisions in that command that a reader should not have to guess:

- **10 threads, not 14.** A memory controller needs several cores asking at once, but not all of them: on this chip the efficiency cores add contention rather than bandwidth. `sysctl hw.perflevel0.physicalcpu` reports the performance-core count.
- **Two kernels, because they disagree.** A copy moves twice the bytes of a sum, and on most machines it also achieves a higher fraction of the peak - it is the friendlier access pattern.

### Measured

| Machine | Command | Read (GB/s) | Copy (GB/s) | Measured on |
|---|---|---|---|---|
| M4 Pro | `./membw 1024 10` | 190-207 | 204-211 | 2026-11-13, repeated on a machine at load average ~4.4 (my own session) |
| GB10 | `./membw 2048 30` | 123 | 151 | 2026-11-13, **with a co-tenant process at 410% CPU** - a floor, to re-measure on an idle machine |
| NUC11 | - | not measured | not measured | pending: the machine is not reachable from the author's laptop by name |

Three honest notes, because the numbers are worse than the spec sheets and that is the point:

1. **The M4 Pro's LPDDR5X is specified well above this.** The gap is what a streaming kernel actually achieves without hand-tuned prefetching and page placement. The book uses the measured number, because that is what a kernel gets to use.
2. **The GB10 number is contaminated.** A stray Jupyter kernel was burning four cores during the run, and on a unified-memory part that costs bandwidth. It is a floor. The row will be replaced when it has been measured on an idle machine, and until then no chapter may quote it as the machine's bandwidth.
3. **A spec number is `reported`, never `measured`.** If a chapter needs a figure for a machine not listed here, it says so and labels it.

## What the numbers are for

Chapter 1 divides them. For decode, the ceiling in tokens per second is

<div class="math-display">
$$
\text{tokens per second} \;\le\; \frac{\text{bandwidth}}{\text{bytes read per token}}
$$
</div>

where the denominator is essentially the model's file size, because every weight is read once per token. On the M4 Pro at 200 GB/s, a 4 GB quantized model cannot exceed about 50 tokens per second no matter what the kernels do; the same model on the GB10 at 123 GB/s cannot exceed about 30. Both numbers are arithmetic on this page's measurements, not results - which is exactly why the page exists before Chapter 1 is written.

## Adding a machine

One row, one command, one date. If the command was not run, the row says "not measured" - an empty cell that nobody can mistake for a number is worth more than a spec sheet pasted in.

## Reading

- [`scripts/membw.c`](https://github.com/yusiwen/llm-inference-book/blob/main/scripts/membw.c) - the measurement, in full
- [Measurements](measurements.md) - the numbers about the book itself
- [The minfer Baseline](baseline.md) - the other pin this book depends on
