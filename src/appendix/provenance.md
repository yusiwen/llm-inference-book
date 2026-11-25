# How Every Number Was Produced

> - **Status**: written - generated from `facts/*.json` by `scripts/provenance.py`, so the commands and machines below are the ones recorded next to the numbers themselves
> - **Rule of this book**: a measured number names its instrument. Each chapter prints one line under the first table it renders; this page is the unabridged version of those lines, with the full command and the file the numbers were written to.

Chapters show tables of measured values: byte counts, error bounds, nanoseconds. A table is easy to read and easy to fake, so every one of them comes from a program in this repository that writes its output to `facts/<name>.json`, and the chapter renders the table from that file rather than typing it. `scripts/facts.py --check` fails if a page and its artifact disagree, so a number cannot drift from the run that produced it without the build saying so.

What that gate can check is internal agreement. Whether the *measurement* is honest is a different question, and the answer is the command: run it yourself on your own copy of the model and you should get the same shape of answer - the same ordering, the same bounds, a timing in the same neighbourhood if your machine is comparable. Every command below is the one that produced the committed artifact, with one substitution worth knowing about: the paths are shown with `~` for the home directory.

<!-- provenance-index -->
| Artifact | Instrument | What it measures | Machine | Runs | Statistic |
|---|---|---|---|---|---|
| `alloc` | `scripts/alloc.py` | runs the engine with its per-node trace and re-implements `alloc_graph`'s liveness rule over the graph it executed (build order, consumers extend a buffer's life, outputs and inputs pinned to the end, views extending their parent); charges the transient nodes to a first-fit arena and the `kvcache_*` nodes to the persistent region they view; measures the process's peak RSS with `/usr/bin/time -l` | {'platform': 'macOS-27.2.1-arm64-arm-64bit-Mach-O', 'machine': 'arm64', 'python': '3.15.2'} | per machine | bytes: the sum over every buffer, the first-fit arena high-water mark, the live-set floor at the busiest step, and the class-rounded total; the RSS is the OS's own number for the same run |
| `attn-dev` | `scripts/attn/attn_dev.cu` | computes one causal prefill row twice - once materialising the score matrix and once with a running max and sum - then runs decode attention against 128 and 8,192 keys, then times 100 small kernels as launches and as one captured CUDA graph replayed | not machine-dependent | - | milliseconds for each path, the score buffer's size, the maximum deviation between the two prefill paths, the bytes read by decode at both context lengths, and the per-launch and per-replay microseconds |
| `attn` | `scripts/attn.c` | runs one layer of real attention on the model's own weights: scores, one decode step at four prefix lengths, the softmax alone, and rebuilding the cache against reading it | arm64, Darwin 27.2.0 | 5 | median of 5 runs of 3 repetitions; the score scan is exhaustive over every head and position |
| `campaign` | `docs/cuda_optimization_steps/ in minfer (parsed by scripts/campaign.py; no GPU needed)` | parses every step record in the engine's campaign directory for its verdict, whether it records a decision bar before the measurement, and whether it carries a verification section | not machine-dependent | - | the count of step records by verdict, the share that landed, reverted or carry no verdict, the number that record a pre-registered bar, and the median length of a record |
| `cuda-launch` | `scripts/cuda/launch_sheet.cu` | compiles five kernels with the machine's own `nvcc` and times each one with CUDA events over 20 repetitions, the streaming copy first because every later prediction divides by it | not machine-dependent | 20 | per-kernel milliseconds, the byte count it moved, and the implied GB/s; the prediction is computed from the first row, so the table is self-referential by construction |
| `cuda-seam` | `scripts/seam/seam.cu` | asks the device for a terabyte, then runs one two-second kernel twice - once behind `cudaStreamSynchronize`, once behind a `cudaStreamQuery` poll with a host-owned deadline - and reads `ldd` off the resulting binary | not machine-dependent | - | the error name and string for the failed allocation, the host's blocked milliseconds, the poll's elapsed milliseconds and outcome, and the library list |
| `dequant-check` | `scripts/dequant-check.py` | decodes each block format back to f32 and compares it against a second decoder (gguf-py) on the same bytes | not machine-dependent | - | max relative error per format; the bit-exact counts are exhaustive over the blocks sampled |
| `dispatch` | `scripts/dispatch.py` | runs the engine's CLI three ways with an impossible or unavailable `--backend` value and records each refusal verbatim, runs the engine's own op-support contract tests, and reads the backend assignment out of the per-node trace of a real run; the registry ids, names and priorities are parsed from the pinned source, and the op-by-backend table from `docs/SUPPORT-MATRIX.md` | Darwin 27.2.0, arm64 | per machine | counts of nodes per backend and of contiguous same-backend runs, from one graph per phase; the refusals are exit codes and messages, compared as values, not summarised |
| `dotbench` | `scripts/dotbench.c` | times three scalar kernels on 8,192 real rows of the book's file | Apple M4 Pro | 7 | median of the runs, with the full range beside it |
| `dotcheck` | `scripts/dotcheck.py` | decomposes the quantized dot product's error into activation, integer and scale parts, and bounds each format's accumulator | not machine-dependent | - | max relative error against f64; the bounds are arithmetic, the worst partial is measured |
| `dotsimd-freq` | `scripts/dotsimd.c --hold` | runs one kernel alone on a pinned core for eight seconds while `turbostat` samples `Bzy_MHz` every 200 ms, and keeps only the samples inside the window the binary stamps to stderr | Intel i7-11700B @ 3.20GHz, nuc11btmi7 (16 threads) | per machine | median of about seventy samples per kernel; the clock question Chapter 7 left open |
| `dotsimd` | `scripts/dotsimd.c` | times four SIMD kernels and the scalar reference, one source file compiled for ARM and for x86 | Apple M4 Pro (arm64) and Intel i7-11700B (x86_64) | per machine | median of the runs per machine |
| `engine-measure` | `scripts/engine-measure.py` | runs the engine's own measurement tests (`cargo test --release measure`) and reads the KV geometry and footprint groups out of them, then recomputes every number from `KvFormat::row_elems` and refuses to write the artifact when the two disagree | Darwin 27.2.0, arm64 | per machine | one run of the engine's suite; the derived column must equal the measured one, format by format and context by context |
| `family` | `scripts/family.py` | runs the binary with its per-node trace on one real file and counts the nodes by op kind, slices one transformer layer out of the graph, runs three byte-patched copies of the same file through the loader to capture its refusals verbatim, and counts lines and `#[test]` functions per family module in the pinned checkout; writes `facts/family.json` | Darwin 27.2.0, arm64 | per machine | node counts and one layer's slice from one trace; the refusals are exit codes and stderr read as values; line and test counts are recomputed from the pinned checkout, not remembered |
| `fusion` | `scripts/fuse.py` | runs the engine's CLI with its per-node trace on (the fused graph it actually executed), reads the op vocabulary out of the pinned source, runs the engine's own fusion tests, and compiles and runs scripts/fuse.c, which measures the same 3-to-1 rewrite on two activations at four row counts; scripts/fuse-check.py recomputes the C's sampled outputs in f64 | Darwin 27.2.0, arm64 | per machine | node counts and the set of executed node ids from one trace; timings are the median of nine runs of the same kernel; output equality is compared byte for byte, and the numpy witness agrees within float32 rounding |
| `gemv-scale` | `scripts/probe/gemv_scale.cu` | runs the same GEMV twice - one thread per row, then one warp per row with a __shfl_down reduction - at 1024, 4096 and 8192 square weights, and measures the machine's own streaming rate in the same process as the denominator | not machine-dependent | - | milliseconds and GB/s for both kernels at each size, their share of the measured streaming rate, the ratio between them, and the maximum relative difference between their outputs |
| `gguf-qwen2.5-0.5b-q4_0` | `scripts/gguf-inspect.py` | parses the same model at Q4_0 | not machine-dependent | - | read once from the file |
| `gguf-qwen2.5-0.5b` | `scripts/gguf-inspect.py` | parses the file: header, every metadata key, every tensor record, and the byte range each tensor occupies | not machine-dependent | - | read once from the file; the counts are structure, not samples |
| `graph` | `scripts/graph.py` | runs the engine's CLI with its per-node trace and its rebuild log on, then reads the node list, the shapes, the edges and the per-node statistics out of the trace the visualizer consumes | arm64, Darwin 27.2.0 | per machine | a structural comparison of two node sequences (equal or not), and node counts read one run at a time; nothing here is averaged |
| `kquant-check` | `scripts/kquant-check.py` | decodes K-quant super-blocks from adversarial random bytes, from real weights, and from the format's own packing | not machine-dependent | - | bit-exact comparisons, exhaustive over the blocks sampled |
| `ladder` | `scripts/ladder/ladder.cu` | compiles six kernels - elementwise, gather, naive f16 GEMM, tiled f16 GEMM, q4_0 GEMV at M=1 and the same q4_0 weights at M=16 - and checks each against a host reference from the same bytes before timing it | not machine-dependent | 20 | per-kernel milliseconds, GFLOP/s, the maximum absolute deviation from the reference and the tolerance it is compared against, plus the tile arithmetic and the per-token cost at both shapes |
| `loop` | `scripts/loop.py` | runs the binary with `MINFER_REBUILD_TRACE` and `MINFER_TIMING` on and counts the graph builds per run, reads the `Prefill:`/`Generated:` lines, runs two conversation turns through one `--session` file, decodes the 36-byte session header out of the `.kv` companion, sends SIGINT mid-decode, and drives a context overflow against `--n-ctx 32` | Darwin 27.2.0, arm64 | per machine | medians of three runs for the two rates; graph builds and their milliseconds are counts the engine narrated; the session header is read as `<8s7I` from the file on disk; the overflow and cancellation rows are exit codes and messages compared as values |
| `membw` | `scripts/membw-run.py` | compiles and runs `scripts/membw.c` - a streaming read of two arrays with no model involved - at each thread count in `--threads`, five runs each, and keeps the median of the reads and of the copies with every run beside them | Darwin 27.2.0, arm64 | {'read_gb_s': 210.2, 'copy_gb_s': 224.7, 'arrays': 2}; {'read_gb_s': 214.4, 'copy_gb_s': 227.2, 'arrays': 2}; {'read_gb_s': 220.7, 'copy_gb_s': 225.5, 'arrays': 2}; {'read_gb_s': 218.7, 'copy_gb_s': 224.6, 'arrays': 2}; {'read_gb_s': 225.2, 'copy_gb_s': 223.5, 'arrays': 2} | medians of five runs per thread count, with the minimum and maximum kept because a bandwidth measured once on a laptop is a rumour |
| `metal` | `scripts/metal.py` | runs one binary against itself with `--backend cpu` and `--backend metal` on one model and one prompt, three runs each, compares the generated bytes, and repeats the Metal arm with `MINFER_WEIGHT_COPY=1` to price the copy that zero-copy avoids | Darwin 27.2.0, arm64 | per machine | medians of three runs for the rates; a sha256 equality for the gate; one timed run per backend for the sampler/forward split |
| `sampler` | `scripts/smp.py` | runs the engine's CLI on the model, keeps the logits it dumps, and replays the deterministic part of the sampler pipeline (penalties, top-k, top-p, temperature) in Python so the engine's own token stream can be reproduced from the engine's own logits | arm64, Darwin 27.2.0 | per machine | byte equality between the replay and the engine's text; the funnel, mass and entropy figures are computed from those logits, one step at a time, never averaged |
| `template` | `scripts/tmpl.py` | renders the model's own `tokenizer.chat_template` (read out of the GGUF) with Jinja2, and compares every case byte for byte with the reference renderer's committed output (HuggingFace `transformers` 5.17.0 `apply_chat_template`); the token counts come from the Chapter 11 tokenizer on the same bytes | arm64, Darwin 27.2.0 | per machine | byte equality on every case; a byte or token count that differs is a failure, not a statistic |
| `threadbench` | `scripts/threadbench.c` | sweeps thread count, batch size, spawn-per-call against a pool, and two write layouts on one matrix | Apple M4 Pro (arm64) and 11th Gen Intel(R) Core(TM) i7-11700B @ 3.20GHz (x86_64) | per machine | median of the runs per machine, with the range where the runs disagree |
| `tokenizer-witness` | `scripts/tok-check.py` | runs the model family's reference tokenizer on the same fixtures and compares ids with the book's instrument | not machine-dependent | - | 15 of 15 fixtures agree; a mismatch is a failure, not a statistic |
| `tokenizer` | `scripts/tok.py` | encodes a fixture set of awkward strings from the vocabulary and merges inside the GGUF, and decodes the ids back; the ids are compared one by one against the reference tokenizer implementation (scripts/tok-check.py, HuggingFace `tokenizers` on the model family's own `tokenizer.json`) | arm64, Darwin 27.2.0 | per machine | exact id equality on every fixture string; nothing here is averaged |
| `vecops-witness` | `scripts/vecops-check.py` | recomputes every op in numpy f64 from the C program's own fixture bytes | not machine-dependent | - | max relative difference, one pass |
| `vecops` | `scripts/vecops.c` | computes each vector op the way an engine writes it, reports the error against an f64 oracle, and times the op | Apple M4 Pro | 3 | median of three runs for timing; the errors are deterministic and identical across runs |

The command behind each row, in full:

### alloc

`scripts/alloc.py` - runs the engine with its per-node trace and re-implements `alloc_graph`'s liveness rule over the graph it executed (build order, consumers extend a buffer's life, outputs and inputs pinned to the end, views extending their parent); charges the transient nodes to a first-fit arena and the `kvcache_*` nodes to the persistent region they view; measures the process's peak RSS with `/usr/bin/time -l`.

- **Machine**: {'platform': 'macOS-27.2.1-arm64-arm-64bit-Mach-O', 'machine': 'arm64', 'python': '3.15.2'}
- **Runs**: per machine; bytes: the sum over every buffer, the first-fit arena high-water mark, the live-set floor at the busiest step, and the class-rounded total; the RSS is the OS's own number for the same run
- **Raw numbers**: `facts/alloc.json`

```bash
python3 scripts/alloc.py ~/git/mine/minfer/target/release/minfer ~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --minfer ~/git/mine/minfer
```

### attn-dev

`scripts/attn/attn_dev.cu` - computes one causal prefill row twice - once materialising the score matrix and once with a running max and sum - then runs decode attention against 128 and 8,192 keys, then times 100 small kernels as launches and as one captured CUDA graph replayed.

- **Machine**: not machine-dependent
- **Runs**: -; milliseconds for each path, the score buffer's size, the maximum deviation between the two prefill paths, the bytes read by decode at both context lengths, and the per-launch and per-replay microseconds
- **Raw numbers**: `facts/attn-dev.json`

```bash
scp scripts/attn/attn_dev.cu yusiwen@192.168.2.40:/tmp/ch25-attn/ ; /usr/local/cuda/bin/nvcc -O3 -arch=sm_121 -o attn_dev attn_dev.cu ; ./attn_dev
```

### attn

`scripts/attn.c` - runs one layer of real attention on the model's own weights: scores, one decode step at four prefix lengths, the softmax alone, and rebuilding the cache against reading it.

- **Machine**: arm64, Darwin 27.2.0
- **Runs**: 5; median of 5 runs of 3 repetitions; the score scan is exhaustive over every head and position
- **Raw numbers**: `facts/attn.json`

```bash
clang -O2 -o attn scripts/attn.c -lm ; ./attn qwen2.5-0.5b-instruct-q4_k_m.gguf attn.conf --json facts/attn.json
```

### campaign

`docs/cuda_optimization_steps/ in minfer (parsed by scripts/campaign.py; no GPU needed)` - parses every step record in the engine's campaign directory for its verdict, whether it records a decision bar before the measurement, and whether it carries a verification section.

- **Machine**: not machine-dependent
- **Runs**: -; the count of step records by verdict, the share that landed, reverted or carry no verdict, the number that record a pre-registered bar, and the median length of a record
- **Raw numbers**: `facts/campaign.json`

```bash
(no command recorded)
```

### cuda-launch

`scripts/cuda/launch_sheet.cu` - compiles five kernels with the machine's own `nvcc` and times each one with CUDA events over 20 repetitions, the streaming copy first because every later prediction divides by it.

- **Machine**: not machine-dependent
- **Runs**: 20; per-kernel milliseconds, the byte count it moved, and the implied GB/s; the prediction is computed from the first row, so the table is self-referential by construction
- **Raw numbers**: `facts/cuda-launch.json`

```bash
scp scripts/cuda/launch_sheet.cu yusiwen@192.168.2.40:/tmp/ch22-launch-sheet/ ; /usr/local/cuda/bin/nvcc -O3 -arch=sm_121 -o launch_sheet launch_sheet.cu ; ./launch_sheet 20
```

### cuda-seam

`scripts/seam/seam.cu` - asks the device for a terabyte, then runs one two-second kernel twice - once behind `cudaStreamSynchronize`, once behind a `cudaStreamQuery` poll with a host-owned deadline - and reads `ldd` off the resulting binary.

- **Machine**: not machine-dependent
- **Runs**: -; the error name and string for the failed allocation, the host's blocked milliseconds, the poll's elapsed milliseconds and outcome, and the library list
- **Raw numbers**: `facts/cuda-seam.json`

```bash
scp scripts/seam/seam.cu yusiwen@192.168.2.40:/tmp/ch23-seam/ ; /usr/local/cuda/bin/nvcc -O3 -arch=sm_121 -o seam seam.cu ; ./seam 2000 500 ; ldd ./seam
```

### dequant-check

`scripts/dequant-check.py` - decodes each block format back to f32 and compares it against a second decoder (gguf-py) on the same bytes.

- **Machine**: not machine-dependent
- **Runs**: -; max relative error per format; the bit-exact counts are exhaustive over the blocks sampled
- **Raw numbers**: `facts/dequant-check.json` (read from `qwen2.5-0.5b-instruct-q4_0.gguf`; its sha256 is in the facts file)

```bash
uv run --with gguf python scripts/dequant-check.py --real facts/gguf-qwen2.5-0.5b-q4_0.json:~/.cache/myfer/qwen2.5-0.5b-instruct-q4_0.gguf --real facts/gguf-qwen2.5-0.5b.json:~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --synthetic-from facts/gguf-qwen2.5-0.5b.json:~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --out facts/dequant-check.json
```

### dispatch

`scripts/dispatch.py` - runs the engine's CLI three ways with an impossible or unavailable `--backend` value and records each refusal verbatim, runs the engine's own op-support contract tests, and reads the backend assignment out of the per-node trace of a real run; the registry ids, names and priorities are parsed from the pinned source, and the op-by-backend table from `docs/SUPPORT-MATRIX.md`.

- **Machine**: Darwin 27.2.0, arm64
- **Runs**: per machine; counts of nodes per backend and of contiguous same-backend runs, from one graph per phase; the refusals are exit codes and messages, compared as values, not summarised
- **Raw numbers**: `facts/dispatch.json`

```bash
python3 scripts/dispatch.py ~/git/mine/minfer/target/release/minfer ~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --minfer ~/git/mine/minfer --out facts/dispatch.json
```

### dotbench

`scripts/dotbench.c` - times three scalar kernels on 8,192 real rows of the book's file.

- **Machine**: Apple M4 Pro
- **Runs**: 7; median of the runs, with the full range beside it
- **Raw numbers**: `facts/dotbench.json`

```bash
clang -O2 -o dotbench scripts/dotbench.c ; ./dotbench qwen2.5-0.5b-instruct-q4_0.gguf <offset> 896 4096
```

### dotcheck

`scripts/dotcheck.py` - decomposes the quantized dot product's error into activation, integer and scale parts, and bounds each format's accumulator.

- **Machine**: not machine-dependent
- **Runs**: -; max relative error against f64; the bounds are arithmetic, the worst partial is measured
- **Raw numbers**: `facts/dotcheck.json` (read from `qwen2.5-0.5b-instruct-q4_0.gguf`; its sha256 is in the facts file)

```bash
uv run --with gguf python scripts/dotcheck.py --weights facts/gguf-qwen2.5-0.5b-q4_0.json:~/.cache/myfer/qwen2.5-0.5b-instruct-q4_0.gguf --pairs Q4_0,Q8_0 --rows 512
```

### dotsimd-freq

`scripts/dotsimd.c --hold` - runs one kernel alone on a pinned core for eight seconds while `turbostat` samples `Bzy_MHz` every 200 ms, and keeps only the samples inside the window the binary stamps to stderr.

- **Machine**: Intel i7-11700B @ 3.20GHz, nuc11btmi7 (16 threads)
- **Runs**: per machine; median of about seventy samples per kernel; the clock question Chapter 7 left open
- **Raw numbers**: `facts/dotsimd-freq.json`

```bash
gcc -O3 -mf16c -o dotsimd dotsimd.c -lm; sudo turbostat --cpu 0 --show Time_Of_Day_Seconds,Bzy_MHz,PkgWatt --interval 0.2 & ; for k in avx2 avx512-vnni; do taskset -c 0 ./dotsimd q4_0.gguf 150590816 896 8192 40 --only $k --hold 8; done
```

### dotsimd

`scripts/dotsimd.c` - times four SIMD kernels and the scalar reference, one source file compiled for ARM and for x86.

- **Machine**: Apple M4 Pro (arm64) and Intel i7-11700B (x86_64)
- **Runs**: per machine; median of the runs per machine
- **Raw numbers**: `facts/dotsimd.json`

```bash
clang -O2 -o dotsimd scripts/dotsimd.c  (arm);  gcc -O3 -mf16c -o dotsimd scripts/dotsimd.c -lm  (x86);  then each machine runs dotsimd q4_0.gguf 150590816 896 8192 7 --json ...
```

### engine-measure

`scripts/engine-measure.py` - runs the engine's own measurement tests (`cargo test --release measure`) and reads the KV geometry and footprint groups out of them, then recomputes every number from `KvFormat::row_elems` and refuses to write the artifact when the two disagree.

- **Machine**: Darwin 27.2.0, arm64
- **Runs**: per machine; one run of the engine's suite; the derived column must equal the measured one, format by format and context by context
- **Raw numbers**: `facts/engine-measure.json`

```bash
cargo test --release measure -- --nocapture   # MINFER_MEASURE_MODEL=~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf (in ~/git/mine/minfer)
```

### family

`scripts/family.py` - runs the binary with its per-node trace on one real file and counts the nodes by op kind, slices one transformer layer out of the graph, runs three byte-patched copies of the same file through the loader to capture its refusals verbatim, and counts lines and `#[test]` functions per family module in the pinned checkout; writes `facts/family.json`.

- **Machine**: Darwin 27.2.0, arm64
- **Runs**: per machine; node counts and one layer's slice from one trace; the refusals are exit codes and stderr read as values; line and test counts are recomputed from the pinned checkout, not remembered
- **Raw numbers**: `facts/family.json`

```bash
(no command recorded)
```

### fusion

`scripts/fuse.py` - runs the engine's CLI with its per-node trace on (the fused graph it actually executed), reads the op vocabulary out of the pinned source, runs the engine's own fusion tests, and compiles and runs scripts/fuse.c, which measures the same 3-to-1 rewrite on two activations at four row counts; scripts/fuse-check.py recomputes the C's sampled outputs in f64.

- **Machine**: Darwin 27.2.0, arm64
- **Runs**: per machine; node counts and the set of executed node ids from one trace; timings are the median of nine runs of the same kernel; output equality is compared byte for byte, and the numpy witness agrees within float32 rounding
- **Raw numbers**: `facts/fusion.json`

```bash
python3 scripts/fuse.py ~/git/mine/minfer/target/release/minfer ~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --minfer ~/git/mine/minfer --out facts/fusion.json
```

### gemv-scale

`scripts/probe/gemv_scale.cu` - runs the same GEMV twice - one thread per row, then one warp per row with a __shfl_down reduction - at 1024, 4096 and 8192 square weights, and measures the machine's own streaming rate in the same process as the denominator.

- **Machine**: not machine-dependent
- **Runs**: -; milliseconds and GB/s for both kernels at each size, their share of the measured streaming rate, the ratio between them, and the maximum relative difference between their outputs
- **Raw numbers**: `facts/gemv-scale.json`

```bash
scp scripts/probe/gemv_scale.cu yusiwen@192.168.2.40:/tmp/ch22-gemv-scale/ ; /usr/local/cuda/bin/nvcc -O3 -arch=sm_121 -o gemv_scale gemv_scale.cu ; ./gemv_scale
```

### gguf-qwen2.5-0.5b-q4_0

`scripts/gguf-inspect.py` - parses the same model at Q4_0.

- **Machine**: not machine-dependent
- **Runs**: -; read once from the file
- **Raw numbers**: `facts/gguf-qwen2.5-0.5b-q4_0.json` (read from `qwen2.5-0.5b-instruct-q4_0.gguf`; its sha256 is in the facts file)

```bash
(no command recorded)
```

### gguf-qwen2.5-0.5b

`scripts/gguf-inspect.py` - parses the file: header, every metadata key, every tensor record, and the byte range each tensor occupies.

- **Machine**: not machine-dependent
- **Runs**: -; read once from the file; the counts are structure, not samples
- **Raw numbers**: `facts/gguf-qwen2.5-0.5b.json` (read from `qwen2.5-0.5b-instruct-q4_k_m.gguf`; its sha256 is in the facts file)

```bash
(no command recorded)
```

### graph

`scripts/graph.py` - runs the engine's CLI with its per-node trace and its rebuild log on, then reads the node list, the shapes, the edges and the per-node statistics out of the trace the visualizer consumes.

- **Machine**: arm64, Darwin 27.2.0
- **Runs**: per machine; a structural comparison of two node sequences (equal or not), and node counts read one run at a time; nothing here is averaged
- **Raw numbers**: `facts/graph.json`

```bash
python3 scripts/graph.py <engine binary> ~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --out facts/graph.json
```

### kquant-check

`scripts/kquant-check.py` - decodes K-quant super-blocks from adversarial random bytes, from real weights, and from the format's own packing.

- **Machine**: not machine-dependent
- **Runs**: -; bit-exact comparisons, exhaustive over the blocks sampled
- **Raw numbers**: `facts/kquant-check.json` (read from `qwen2.5-0.5b-instruct-q4_k_m.gguf`; its sha256 is in the facts file)

```bash
uv run --with gguf python scripts/kquant-check.py --real facts/gguf-qwen2.5-0.5b.json:~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --real facts/gguf-qwen2.5-0.5b-q4_0.json:~/.cache/myfer/qwen2.5-0.5b-instruct-q4_0.gguf
```

### ladder

`scripts/ladder/ladder.cu` - compiles six kernels - elementwise, gather, naive f16 GEMM, tiled f16 GEMM, q4_0 GEMV at M=1 and the same q4_0 weights at M=16 - and checks each against a host reference from the same bytes before timing it.

- **Machine**: not machine-dependent
- **Runs**: 20; per-kernel milliseconds, GFLOP/s, the maximum absolute deviation from the reference and the tolerance it is compared against, plus the tile arithmetic and the per-token cost at both shapes
- **Raw numbers**: `facts/ladder.json`

```bash
scp scripts/ladder/ladder.cu yusiwen@192.168.2.40:/tmp/ch24-ladder/ ; /usr/local/cuda/bin/nvcc -O3 -arch=sm_121 -o ladder ladder.cu ; ./ladder
```

### loop

`scripts/loop.py` - runs the binary with `MINFER_REBUILD_TRACE` and `MINFER_TIMING` on and counts the graph builds per run, reads the `Prefill:`/`Generated:` lines, runs two conversation turns through one `--session` file, decodes the 36-byte session header out of the `.kv` companion, sends SIGINT mid-decode, and drives a context overflow against `--n-ctx 32`.

- **Machine**: Darwin 27.2.0, arm64
- **Runs**: per machine; medians of three runs for the two rates; graph builds and their milliseconds are counts the engine narrated; the session header is read as `<8s7I` from the file on disk; the overflow and cancellation rows are exit codes and messages compared as values
- **Raw numbers**: `facts/loop.json`

```bash
(no command recorded)
```

### membw

`scripts/membw-run.py` - compiles and runs `scripts/membw.c` - a streaming read of two arrays with no model involved - at each thread count in `--threads`, five runs each, and keeps the median of the reads and of the copies with every run beside them.

- **Machine**: Darwin 27.2.0, arm64
- **Runs**: {'read_gb_s': 210.2, 'copy_gb_s': 224.7, 'arrays': 2}; {'read_gb_s': 214.4, 'copy_gb_s': 227.2, 'arrays': 2}; {'read_gb_s': 220.7, 'copy_gb_s': 225.5, 'arrays': 2}; {'read_gb_s': 218.7, 'copy_gb_s': 224.6, 'arrays': 2}; {'read_gb_s': 225.2, 'copy_gb_s': 223.5, 'arrays': 2}; medians of five runs per thread count, with the minimum and maximum kept because a bandwidth measured once on a laptop is a rumour
- **Raw numbers**: `facts/membw.json`

```bash
clang -O2 -o membw scripts/membw.c -lm ; ./membw 1024 N  for N in 8,10, x5 runs each
```

### metal

`scripts/metal.py` - runs one binary against itself with `--backend cpu` and `--backend metal` on one model and one prompt, three runs each, compares the generated bytes, and repeats the Metal arm with `MINFER_WEIGHT_COPY=1` to price the copy that zero-copy avoids.

- **Machine**: Darwin 27.2.0, arm64
- **Runs**: per machine; medians of three runs for the rates; a sha256 equality for the gate; one timed run per backend for the sampler/forward split
- **Raw numbers**: `facts/metal.json`

```bash
/tmp/mf21-target/release/minfer qwen2.5-0.5b-instruct-q4_k_m.gguf 'The quick brown fox' --backend {cpu,metal} -n 24 --greedy  (x3 each; plus MINFER_TIMING=1 once per backend and MINFER_WEIGHT_COPY=1 x3 on metal)
```

### sampler

`scripts/smp.py` - runs the engine's CLI on the model, keeps the logits it dumps, and replays the deterministic part of the sampler pipeline (penalties, top-k, top-p, temperature) in Python so the engine's own token stream can be reproduced from the engine's own logits.

- **Machine**: arm64, Darwin 27.2.0
- **Runs**: per machine; byte equality between the replay and the engine's text; the funnel, mass and entropy figures are computed from those logits, one step at a time, never averaged
- **Raw numbers**: `facts/sampler.json`

```bash
python3 scripts/smp.py <engine binary> ~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --out facts/sampler.json
```

### template

`scripts/tmpl.py` - renders the model's own `tokenizer.chat_template` (read out of the GGUF) with Jinja2, and compares every case byte for byte with the reference renderer's committed output (HuggingFace `transformers` 5.17.0 `apply_chat_template`); the token counts come from the Chapter 11 tokenizer on the same bytes.

- **Machine**: arm64, Darwin 27.2.0
- **Runs**: per machine; byte equality on every case; a byte or token count that differs is a failure, not a statistic
- **Raw numbers**: `facts/template.json`

```bash
python3 scripts/tmpl.py ~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --out facts/template.json
```

### threadbench

`scripts/threadbench.c` - sweeps thread count, batch size, spawn-per-call against a pool, and two write layouts on one matrix.

- **Machine**: Apple M4 Pro (arm64) and 11th Gen Intel(R) Core(TM) i7-11700B @ 3.20GHz (x86_64)
- **Runs**: per machine; median of the runs per machine, with the range where the runs disagree
- **Raw numbers**: `facts/threadbench.json`

```bash
clang -O2 -pthread -o threadbench scripts/threadbench.c -lm  (arm); gcc -O3 -mavx2 -mfma -mf16c -pthread -o threadbench scripts/threadbench.c -lm  (x86); then each machine runs ./threadbench q4_0.gguf 150590816 896 151936 --json ...
```

### tokenizer-witness

`scripts/tok-check.py` - runs the model family's reference tokenizer on the same fixtures and compares ids with the book's instrument.

- **Machine**: not machine-dependent
- **Runs**: -; 15 of 15 fixtures agree; a mismatch is a failure, not a statistic
- **Raw numbers**: `facts/tokenizer-witness.json`

```bash
(no command recorded)
```

### tokenizer

`scripts/tok.py` - encodes a fixture set of awkward strings from the vocabulary and merges inside the GGUF, and decodes the ids back; the ids are compared one by one against the reference tokenizer implementation (scripts/tok-check.py, HuggingFace `tokenizers` on the model family's own `tokenizer.json`).

- **Machine**: arm64, Darwin 27.2.0
- **Runs**: per machine; exact id equality on every fixture string; nothing here is averaged
- **Raw numbers**: `facts/tokenizer.json`

```bash
python3 scripts/tok.py ~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --out facts/tokenizer.json
```

### vecops-witness

`scripts/vecops-check.py` - recomputes every op in numpy f64 from the C program's own fixture bytes.

- **Machine**: not machine-dependent
- **Runs**: -; max relative difference, one pass
- **Raw numbers**: `facts/vecops-witness.json`

```bash
uv run --with numpy python scripts/vecops-check.py facts/vecops.json --write facts/vecops-witness.json
```

### vecops

`scripts/vecops.c` - computes each vector op the way an engine writes it, reports the error against an f64 oracle, and times the op.

- **Machine**: Apple M4 Pro
- **Runs**: 3; median of three runs for timing; the errors are deterministic and identical across runs
- **Raw numbers**: `facts/vecops.json`

```bash
clang -O2 -o vecops scripts/vecops.c -lm ; ./vecops qwen2.5-0.5b-instruct-q4_k_m.gguf --json facts/vecops.json
```
<!-- /provenance-index -->

## What these numbers are not

- **They are one machine, not a specification.** A timing here is a fact about the machine named beside it, on the date of the commit that recorded it. Chapter 7 measures the same kernels on two machines precisely because a single-machine number invites the wrong generalisation.
- **They are medians with a range, not best cases.** Where the runs disagreed, the range is printed next to the median, in the chapter and in the artifact.
- **A reported number stays reported.** Where a chapter quotes a figure from a paper or another project, it says so and does not relabel it as something this book measured.
- **An unchanged artifact is not a re-run.** Committing `facts/dotbench.json` again does not mean the benchmark was re-run on your machine, and the `artifact_sha256_check` field records which copy of the model file the numbers came from.

## Reading

- [measurements.md](measurements.md) - the counts about the book itself, with the command beside each one
- [machines.md](machines.md) - the three reference machines, their bandwidth, and why those three
- [conventions.md](conventions.md) - the rule that a number lives on the measurements page only with its command
