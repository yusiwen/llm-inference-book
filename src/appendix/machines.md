# The Reference Machines

> - **Status**: draft - two of the three machines are measured, one is waiting for its address; the numbers below are one afternoon of measurements, not a benchmark campaign.
> - **Rule of this book**: a number names its machine, and a machine names its measured bandwidth. A spec sheet is not a measurement, and this page says which is which.

Every performance claim in this book is either *measured on one of these machines* or *arithmetic on stated assumptions*. The arithmetic needs one number more than people expect: for decode, the ceiling is the model's bytes per token divided by how fast this machine can actually stream memory - not by the bandwidth printed on the box. So the machines come first, with the measurement that produced their number.

## The three machines, and what each one is for

| Machine | Path it represents | CPU | Memory | Chapters |
|---|---|---|---|---|
| **MacBook Pro, M4 Pro** | CPU + Metal | 14 cores (10 performance, 4 efficiency), arm64 | 48 GiB unified | 1-20, 21, 27, 28 |
| **DGX Spark, NVIDIA GB10** | CUDA | 20 cores, aarch64 | 121 GiB unified (Grace + Blackwell) | 22-26, and every CUDA number in 24-25 |
| **NUC11** | x86-64 with AVX2, the CPU path on a non-ARM machine | Intel i7-11700B, 8 cores / 16 threads | 62 GiB DDR4 | 6, 7 (the AVX2 halves of the SIMD chapters) |

Three paths, because they fail differently: ARM's NEON with `SDOT` does integer dot products in one instruction, x86 needs AVX2 plus `dp4a`-style tricks, and the two GPUs disagree about memory entirely - one shares it with the CPU, the other does not.

## Three numbers per machine, not one

A phrase like "the GB10 does 273 GB/s" is three different claims wearing one number. This page keeps them apart, because the book's arithmetic gets the wrong answer if it confuses them:

- **The specification** - bus width times transfer rate. It is a property of the memory interface, `reported` from the vendor, and no program achieves it.
- **The CPU-side achievable** - what a streaming kernel on the CPU cores gets. [`scripts/membw.c`](https://github.com/yusiwen/llm-inference-book/blob/main/scripts/membw.c).
- **The GPU-side achievable** - what a kernel *on the device* gets from the same memory. [`scripts/gpubw.cu`](https://github.com/yusiwen/llm-inference-book/blob/main/scripts/gpubw.cu) for CUDA, [`scripts/metalbw.swift`](https://github.com/yusiwen/llm-inference-book/blob/main/scripts/metalbw.swift) for Metal.

On an Apple machine and on an x86 server, CPU and GPU sit close enough that one number is a fair shorthand. On the GB10 they differ by more than a factor of two, and the difference is the whole reason Chapter 1's ceiling has a path in it.

## How they were measured

The same protocol on every machine: **a streaming sum (1 byte moved per byte read) and a streaming copy (2 bytes per byte), repeated, with the machine state written beside the result.** No benchmark harness, no median-of-many discipline: that is Chapter 28, and this page only needs the conditions to be honest.

```bash
cc -O2 -pthread -o membw scripts/membw.c && ./membw 2048 10   # CPU side
nvcc -O3 -arch=sm_121 -o gpubw scripts/gpubw.cu && ./gpubw 4096   # CUDA
swiftc -O -o metalbw scripts/metalbw.swift && ./metalbw 4096     # Metal
```

Two decisions a reader should not have to guess:

- **The thread count is an argument, and it matters.** A memory controller needs several cores asking at once, but not all of them: on the M4 Pro the efficiency cores add contention instead of bandwidth, so the CPU-side number uses the 10 performance cores. On the GB10 the CPU is 15 Cortex-X925 plus 5 Cortex-A725, and that mixture behaves oddly enough to be measured separately below.
- **Read and copy disagree**, so both are reported. A copy moves twice the bytes of a sum and usually reaches a higher fraction of the peak, because its access pattern is friendlier - but not always: on the M4 Pro's GPU the copy is *slower*.

## Measured

| Machine | Path | Specification (`reported`) | Read (GB/s) | Copy (GB/s) | Share of spec |
|---|---|---|---|---|---|
| M4 Pro | CPU, 10 threads | 273 | **198** (184-202) | **205** (202-207) | 73% |
| M4 Pro | Metal, 20-core GPU | 273 | **212** (202-218) | **186** (175-189) | 78% |
| GB10 | CPU, 24 threads | 273 | **118** (101-118) | **146** (125-152) | 43% |
| GB10 | CUDA, 48 SMs | 273 | **242** (241-243) | **207** (207) | 89% |
| NUC11 | CPU, 8 threads | 51.2 | **45.1** (45.0-45.2) | **41.4** (41.3-41.4) | 88% |

Measured 2026-11-13. The M4 Pro ran with the author's own session on it (load average 4.4); the NUC11 and the GB10 were idle (load 0.07 and 1.0, GPU at 0%). Five runs each; the spread is in brackets where more than one run exists.

## Why these disagree with the numbers on the box

Six reasons, in the order they matter when someone asks why your measurement is lower than the web page's.

1. **The specification is a peak, and peaks are not reachable.** 273 GB/s is 256 bits of LPDDR5X at 8533 MT/s - the bus, at its maximum, with no refresh, no page miss, no command overhead. Streaming code lands at 70-90% of it on a good day. Two of the three machines sit exactly there (M4 Pro 73%, NUC11 88% of DDR4-3200's 51.2), which is what a healthy measurement looks like.
2. **The path matters more than the peak.** The GB10's CPU side reaches 43% while its GPU side reaches 89% of the same 273 GB/s. The number on the box describes the LPDDR5X fabric; the CPU cluster reaches it through a coherent link and saturates near 120 GB/s. Two independent confirmations: the CUDA kernel above (243 GB/s), and a real 27B NVFP4 model decoding at 11.8 tokens per second on that machine, which at 21.8 GB of weights implies 257 GB/s of effective read - within six per cent of the kernel.
3. **The core mixture matters.** On the GB10, pinning to the 15 Cortex-X925 cores at one thread each gives only 79.7 GB/s; 30 threads with SMT on the same cores gives 117; the 5 slow Cortex-A725 cores alone give 86.8. The ceiling is the fabric, not the cores - which is why "use all the threads" and "use the fast threads" both fail, and why the table above says 24 threads for that machine.
4. **The kernel matters.** A sum written with one accumulator is latency-bound on the add - 15 GB/s instead of 46 on the M4 Pro - because it measures a dependency chain, not memory. Four accumulators fix it. This is the first thing that goes wrong in anyone's first bandwidth test, including this book's.
5. **The co-tenant matters.** The GB10's first series - identical binary, identical array, identical machine, only a Jupyter kernel drifting between 57 and 410 per cent of a core in the background - read 66-97 GB/s. Thirty per cent slower, from a process the benchmark never mentions.
6. **The size and the clock ramp matter.** The Metal figure is 100 GB/s at 1 GiB and 212 at 4 GiB: unified-memory pages are mapped on first touch, and an Apple GPU core does not sit at its boost clock while idle. Under one gigabyte, a GPU bandwidth number says more about the driver than about the memory.

None of this makes the vendor number wrong. It makes it a different claim: **what the interface can do, not what your kernel will get.** The book always uses the second.

## What the numbers are for

Chapter 1 divides them. For decode, the ceiling in tokens per second is

<div class="math-display">
$$
\text{tokens per second} \;\le\; \frac{\text{bandwidth}}{\text{bytes read per token}}
$$
</div>

where the denominator is essentially the model's file size, because every weight is read once per token - and where the bandwidth is *the path the kernels run on*, which is why this page grew a column.

| Where the engine runs | Bandwidth used | Ceiling for a 4 GB quantized model |
|---|---|---|
| M4 Pro, CPU backend | 198 GB/s | about 50 tokens per second |
| M4 Pro, Metal backend | 212 GB/s | about 53 |
| GB10, CUDA backend | 243 GB/s | about 61 |
| GB10, CPU backend | 118 GB/s | about 30 |
| NUC11, CPU backend | 45 GB/s | about 11 |

The last row is why a server without a GPU is not a substitute, and the GB10's two rows are why "the machine is fast" is not a statement about anything until you say which memory path the kernel uses. Every number in this table is arithmetic on a measurement above, not a result - which is exactly why the page exists before Chapter 1 is written.

## Adding a machine

One row, one command, one date. If the command was not run, the row says "not measured" - an empty cell that nobody can mistake for a number is worth more than a spec sheet pasted in.

## Reading

- [`scripts/membw.c`](https://github.com/yusiwen/llm-inference-book/blob/main/scripts/membw.c) - the measurement, in full
- [Measurements](measurements.md) - the numbers about the book itself
- [The minfer Baseline](baseline.md) - the other pin this book depends on
