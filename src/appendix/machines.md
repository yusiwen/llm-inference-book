# The Reference Machines

> **Status**: draft - two of the three machines are measured, one is waiting for its address; the numbers below are one afternoon of measurements, not a benchmark campaign.
> **Rule of this book**: a number names its machine, and a machine names its measured bandwidth. A spec sheet is not a measurement, and this page says which is which.

Every performance claim in this book is either *measured on one of these machines* or *arithmetic on stated assumptions*. The arithmetic needs one number more than people expect: for decode, the ceiling is the model's bytes per token divided by how fast this machine can actually stream memory - not by the bandwidth printed on the box. So the machines come first, with the measurement that produced their number.

## The three machines, and what each one is for

| Machine | Path it represents | CPU | Memory | Chapters |
|---|---|---|---|---|
| **MacBook Pro, M4 Pro** | CPU + Metal | 14 cores (10 performance, 4 efficiency), arm64 | 48 GiB unified | 1-20, 21, 27, 28 |
| **DGX Spark, NVIDIA GB10** | CUDA | 20 cores, aarch64 | 121 GiB unified (Grace + Blackwell) | 22-26, and every CUDA number in 24-25 |
| **NUC11** | x86-64 with AVX2, the CPU path on a non-ARM machine | Intel i7-11700B, 8 cores / 16 threads | 62 GiB DDR4 | 6, 7 (the AVX2 halves of the SIMD chapters) |

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

The same protocol on all three machines: **five runs, median with the spread in brackets, and the machine state written beside it.** A single run is not a measurement; a median without its spread hides the only thing that decides whether a later comparison means anything.

| Machine | Command | Read (GB/s) | Copy (GB/s) | Conditions, 2026-11-13 |
|---|---|---|---|---|
| M4 Pro | `./membw 2048 10` | **198** (184-202) | **205** (202-207) | the author's own session running, load average 4.4 |
| NUC11 | `./membw 2048 8` | **45.1** (45.0-45.2) | **41.4** (41.3-41.4) | idle, load average 0.07 |
| GB10 | `./membw 2048 20` | **101** (101-108) | **125** (125-134) | idle: load average 1.0, GPU at 0%; 40 threads (SMT) reaches 117 / 150 |

Three honest notes, because the numbers are below the spec sheets and that is the point:

1. **Every machine lands well below its specification.** The M4 Pro's LPDDR5X is specified far above 200 GB/s and the GB10's unified memory above 270; a streaming kernel without hand-tuned prefetching gets what is in this table. The book uses the measured number, because that is what a kernel gets to use.
2. **The spreads differ by a factor of twenty, and that is information.** The NUC11, sitting idle on a shelf, repeats to within half a per cent. The M4 Pro, with an editor and a browser and this build running on it, moves eight per cent. The GB10 moves seven. A number from a machine you are also using is a number with a floor under it - and on your own laptop, that floor is the one you have.
3. **This table had a fourth column's worth of history.** The GB10's first series, taken while a Jupyter kernel on the box drifted between 57 and 410 per cent of a core, read 66-97 GB/s: same binary, same array size, same machine, thirty per cent slower than the idle run above. Nothing about the kernel changed - the machine did. That is why every row here carries its state, and why Chapter 28 refuses to compare two measurements before stating a noise floor.

## What the numbers are for

Chapter 1 divides them. For decode, the ceiling in tokens per second is

<div class="math-display">
$$
\text{tokens per second} \;\le\; \frac{\text{bandwidth}}{\text{bytes read per token}}
$$
</div>

where the denominator is essentially the model's file size, because every weight is read once per token. On the M4 Pro at 198 GB/s, a 4 GB quantized model cannot exceed about 50 tokens per second no matter what the kernels do. The same model on the GB10, at 101 GB/s, cannot exceed about 25 - or about 29 if the SMT run is believed. The NUC11's 45 GB/s puts the ceiling near 11 tokens per second, which is what a server without a GPU really is for this job.

Every one of those numbers is arithmetic on this page's measurements, not a result - which is exactly why the page exists before Chapter 1 is written.

## Adding a machine

One row, one command, one date. If the command was not run, the row says "not measured" - an empty cell that nobody can mistake for a number is worth more than a spec sheet pasted in.

## Reading

- [`scripts/membw.c`](https://github.com/yusiwen/llm-inference-book/blob/main/scripts/membw.c) - the measurement, in full
- [Measurements](measurements.md) - the numbers about the book itself
- [The minfer Baseline](baseline.md) - the other pin this book depends on
