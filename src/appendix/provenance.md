# How Every Number Was Produced

> - **Status**: written - generated from `facts/*.json` by `scripts/provenance.py`, so the commands and machines below are the ones recorded next to the numbers themselves
> - **Rule of this book**: a measured number names its instrument. Each chapter prints one line under the first table it renders; this page is the unabridged version of those lines, with the full command and the file the numbers were written to.

Chapters show tables of measured values: byte counts, error bounds, nanoseconds. A table is easy to read and easy to fake, so every one of them comes from a program in this repository that writes its output to `facts/<name>.json`, and the chapter renders the table from that file rather than typing it. `scripts/facts.py --check` fails if a page and its artifact disagree, so a number cannot drift from the run that produced it without the build saying so.

What that gate can check is internal agreement. Whether the *measurement* is honest is a different question, and the answer is the command: run it yourself on your own copy of the model and you should get the same shape of answer - the same ordering, the same bounds, a timing in the same neighbourhood if your machine is comparable. Every command below is the one that produced the committed artifact, with one substitution worth knowing about: the paths are shown with `~` for the home directory.

<!-- provenance-index -->
| Artifact | Instrument | What it measures | Machine | Runs | Statistic |
|---|---|---|---|---|---|
| `attn` | `scripts/attn.c` | runs one layer of real attention on the model's own weights: scores, one decode step at four prefix lengths, the softmax alone, and rebuilding the cache against reading it | arm64, Darwin 27.2.0 | 5 | median of 5 runs of 3 repetitions; the score scan is exhaustive over every head and position |
| `dequant-check` | `scripts/dequant-check.py` | decodes each block format back to f32 and compares it against a second decoder (gguf-py) on the same bytes | not machine-dependent | - | max relative error per format; the bit-exact counts are exhaustive over the blocks sampled |
| `dotbench` | `scripts/dotbench.c` | times three scalar kernels on 8,192 real rows of the book's file | Apple M4 Pro | 7 | median of the runs, with the full range beside it |
| `dotcheck` | `scripts/dotcheck.py` | decomposes the quantized dot product's error into activation, integer and scale parts, and bounds each format's accumulator | not machine-dependent | - | max relative error against f64; the bounds are arithmetic, the worst partial is measured |
| `dotsimd` | `scripts/dotsimd.c` | times four SIMD kernels and the scalar reference, one source file compiled for ARM and for x86 | Apple M4 Pro (arm64) and Intel i7-11700B (x86_64) | per machine | median of the runs per machine |
| `gguf-qwen2.5-0.5b-q4_0` | `scripts/gguf-inspect.py` | parses the same model at Q4_0 | not machine-dependent | - | read once from the file |
| `gguf-qwen2.5-0.5b` | `scripts/gguf-inspect.py` | parses the file: header, every metadata key, every tensor record, and the byte range each tensor occupies | not machine-dependent | - | read once from the file; the counts are structure, not samples |
| `graph` | `(not declared)` |  | arm64, Darwin 27.2.0 | per machine |  |
| `kquant-check` | `scripts/kquant-check.py` | decodes K-quant super-blocks from adversarial random bytes, from real weights, and from the format's own packing | not machine-dependent | - | bit-exact comparisons, exhaustive over the blocks sampled |
| `sampler` | `scripts/smp.py` | runs the engine's CLI on the model, keeps the logits it dumps, and replays the deterministic part of the sampler pipeline (penalties, top-k, top-p, temperature) in Python so the engine's own token stream can be reproduced from the engine's own logits | arm64, Darwin 27.2.0 | per machine | byte equality between the replay and the engine's text; the funnel, mass and entropy figures are computed from those logits, one step at a time, never averaged |
| `template` | `scripts/tmpl.py` | renders the model's own `tokenizer.chat_template` (read out of the GGUF) with Jinja2, and compares every case byte for byte with the reference renderer's committed output (HuggingFace `transformers` 5.17.0 `apply_chat_template`); the token counts come from the Chapter 11 tokenizer on the same bytes | arm64, Darwin 27.2.0 | per machine | byte equality on every case; a byte or token count that differs is a failure, not a statistic |
| `threadbench` | `scripts/threadbench.c` | sweeps thread count, batch size, spawn-per-call against a pool, and two write layouts on one matrix | Apple M4 Pro (arm64) and 11th Gen Intel(R) Core(TM) i7-11700B @ 3.20GHz (x86_64) | per machine | median of the runs per machine, with the range where the runs disagree |
| `tokenizer-witness` | `scripts/tok-check.py` | runs the model family's reference tokenizer on the same fixtures and compares ids with the book's instrument | not machine-dependent | - | 15 of 15 fixtures agree; a mismatch is a failure, not a statistic |
| `tokenizer` | `scripts/tok.py` | encodes a fixture set of awkward strings from the vocabulary and merges inside the GGUF, and decodes the ids back; the ids are compared one by one against the reference tokenizer implementation (scripts/tok-check.py, HuggingFace `tokenizers` on the model family's own `tokenizer.json`) | arm64, Darwin 27.2.0 | per machine | exact id equality on every fixture string; nothing here is averaged |
| `vecops-witness` | `scripts/vecops-check.py` | recomputes every op in numpy f64 from the C program's own fixture bytes | not machine-dependent | - | max relative difference, one pass |
| `vecops` | `scripts/vecops.c` | computes each vector op the way an engine writes it, reports the error against an f64 oracle, and times the op | Apple M4 Pro | 3 | median of three runs for timing; the errors are deterministic and identical across runs |

The command behind each row, in full:

### attn

`scripts/attn.c` - runs one layer of real attention on the model's own weights: scores, one decode step at four prefix lengths, the softmax alone, and rebuilding the cache against reading it.

- **Machine**: arm64, Darwin 27.2.0
- **Runs**: 5; median of 5 runs of 3 repetitions; the score scan is exhaustive over every head and position
- **Raw numbers**: `facts/attn.json`

```bash
clang -O2 -o attn scripts/attn.c -lm ; ./attn qwen2.5-0.5b-instruct-q4_k_m.gguf attn.conf --json facts/attn.json
```

### dequant-check

`scripts/dequant-check.py` - decodes each block format back to f32 and compares it against a second decoder (gguf-py) on the same bytes.

- **Machine**: not machine-dependent
- **Runs**: -; max relative error per format; the bit-exact counts are exhaustive over the blocks sampled
- **Raw numbers**: `facts/dequant-check.json` (read from `qwen2.5-0.5b-instruct-q4_0.gguf`; its sha256 is in the facts file)

```bash
uv run --with gguf python scripts/dequant-check.py --real facts/gguf-qwen2.5-0.5b-q4_0.json:~/.cache/myfer/qwen2.5-0.5b-instruct-q4_0.gguf --real facts/gguf-qwen2.5-0.5b.json:~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --synthetic-from facts/gguf-qwen2.5-0.5b.json:~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --out facts/dequant-check.json
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

### dotsimd

`scripts/dotsimd.c` - times four SIMD kernels and the scalar reference, one source file compiled for ARM and for x86.

- **Machine**: Apple M4 Pro (arm64) and Intel i7-11700B (x86_64)
- **Runs**: per machine; median of the runs per machine
- **Raw numbers**: `facts/dotsimd.json`

```bash
clang -O2 -o dotsimd scripts/dotsimd.c  (arm);  gcc -O3 -mf16c -o dotsimd scripts/dotsimd.c -lm  (x86);  then each machine runs dotsimd q4_0.gguf 150590816 896 8192 7 --json ...
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

`(not declared)` - .

- **Machine**: arm64, Darwin 27.2.0
- **Runs**: per machine; 
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
