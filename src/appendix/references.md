# Reference Index

> **Status**: draft - every entry is real and every arXiv ID was resolved against the arXiv API on 2026-11-12; the "used in" column is generated from the chapters, not typed.
> **Rule of this book**: this page lists what the book sends you to read, and where each source is used. It is the one page allowed to keep growing, because the field it describes keeps growing.

29 sources on four shelves: **16 papers**, **5 books**, **8 specifications or vendor manuals**, plus the reference implementation's own documentation, which is not a citation but is the largest single body of reading this book assigns.

## How this page works

- **Every source has a key.** A chapter's `## Reading` section names the key in square brackets, for example `[gqa]`. The key is what the two ends agree on; the title, the year and the identifier live here, once.
- **The "used in" column is derived.** `scripts/references.py` scans every page for the keys, recomputes the column and fails the gate when it is stale. Add a citation to a chapter and this page must be regenerated in the same commit.
- **arXiv IDs were resolved, not recalled.** Each editorial ID below returned its own title from `export.arxiv.org/api/query` on the date named; the title printed is the title the API returned. A source with no ID says so, and inventing one is the failure mode this note exists to prevent.
- **A key pointing at nothing is an error.** If a chapter cites a key that is not on this page, `references.py` fails rather than letting the citation dangle.
- **Only some sources are papers.** For a build book, the specifications are as load-bearing as the papers: a kernel that disagrees with the PTX manual is wrong, and no paper will say so.

## Reading order, if you want one

Not required, and not a syllabus. If you read three things before Chapter 1: the [roofline](#shelf-1--papers-and-articles) idea (one page of a journal paper, and the arithmetic this book repeats in Chapter 1), the GGUF specification next to the file you are about to parse, and the tokenizer paper's merge loop. Everything else is best read *after* the chapter that needs it, which is why each chapter lists its own.


## Shelf 1 · Papers and articles

| Key | Work | Checked | Used in |
|---|---|---|---|
| `[roofline]` | Williams, Waterman and Patterson, *Roofline: an insightful visual performance model for multicore architectures* (CACM, 2009) - no arXiv ID (journal paper) | no arXiv ID | 1 |
| `[scaling-inference]` | Pope et al., *Efficiently Scaling Transformer Inference* (2022) - arXiv:2211.05102 | arXiv API | 1, 8, 14, 17, 20 |
| `[kbit]` | Dettmers and Zettlemoyer, *The case for 4-bit precision: k-bit Inference Scaling Laws* (2022) - arXiv:2212.09720 | arXiv API | 4, 5 |
| `[gqa]` | Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023) - arXiv:2305.13245 | arXiv API | 10 |
| `[fa2]` | Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023) - arXiv:2307.08691 | arXiv API | 25 |
| `[online-softmax]` | Milakov and Gimelshein, *Online normalizer calculation for softmax* (2018) - arXiv:1805.02867 | arXiv API | 9, 25 |
| `[paged-attention]` | Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023) - arXiv:2309.06180 | arXiv API | 17, 18, 20, 29 |
| `[spec-decoding]` | Leviathan, Kalman and Matias, *Fast Inference from Transformers via Speculative Decoding* (2022) - arXiv:2211.17192 | arXiv API | 29 |
| `[bpe]` | Sennrich, Haddow and Birch, *Neural Machine Translation of Rare Words with Subword Units* (arXiv 2015, ACL 2016) - arXiv:1508.07909 | arXiv API | 11 |
| `[nucleus]` | Holtzman et al., *The Curious Case of Neural Text Degeneration* (2019) - arXiv:1904.09751 | arXiv API | 13 |
| `[tvm]` | Chen et al., *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning* (2018) - arXiv:1802.04799 | arXiv API | 14, 15, 16 |
| `[qwen2]` | Yang et al., *Qwen2 Technical Report* (2024) - arXiv:2407.10671 | arXiv API | 19 |
| `[qwen3]` | Yang et al., *Qwen3 Technical Report* (2025) - arXiv:2505.09388 | arXiv API | 19 |
| `[llama3]` | Grattafiori et al., *The Llama 3 Herd of Models* (2024) - arXiv:2407.21783 | arXiv API | 19 |
| `[goldberg]` | Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic* (ACM Computing Surveys, 1991) - no arXiv ID (journal paper) | no arXiv ID | 27 |
| `[rigorous-bench]` | Georges, Buytaert and Eeckhout, *Statistically Rigorous Java Performance Evaluation* (OOPSLA 2007) - no arXiv ID (conference paper) | no arXiv ID | 28 |

## Shelf 2 · Books

| Key | Work | Checked | Used in |
|---|---|---|---|
| `[pmpp]` | Kirk and Hwu, *Programming Massively Parallel Processors* (4th edition, 2022) - no arXiv ID (book) | no arXiv ID | 22 |
| `[sze]` | Sze et al., *Efficient Processing of Deep Neural Networks* (2020) - no arXiv ID (book) | no arXiv ID | 24 |
| `[caqa]` | Hennessy and Patterson, *Computer Architecture: A Quantitative Approach* (6th edition, 2017) - no arXiv ID (book) | no arXiv ID | 0, 1 |
| `[jm]` | Jurafsky and Martin, *Speech and Language Processing* (3rd edition draft) - no arXiv ID (book; the tokenization chapter) | no arXiv ID | 11 |
| `[trpl]` | Klabnik and Nichols, *The Rust Programming Language* - no arXiv ID (book) | no arXiv ID | 0 |

## Shelf 3 · Specifications and vendor manuals

| Key | Work | Checked | Used in |
|---|---|---|---|
| `[ieee754]` | *IEEE 754-2019, Standard for Floating-Point Arithmetic* - no arXiv ID (standard) | no arXiv ID | 9, 27 |
| `[gguf]` | The GGUF file-format specification, in the ggml repository - no arXiv ID (specification) | no arXiv ID | 2, 12 |
| `[ggml-quant]` | The ggml quantized block layouts and the k-quant design notes, in the ggml repository - no arXiv ID (specification) | no arXiv ID | 3, 4, 5, 6 |
| `[cuda-guide]` | *CUDA C++ Programming Guide*, NVIDIA - no arXiv ID (vendor manual) | no arXiv ID | 22, 23 |
| `[ptx]` | *PTX ISA*, NVIDIA - no arXiv ID (vendor manual; the `mma` and `cp.async` instructions) | no arXiv ID | 24 |
| `[nsight]` | *Nsight Compute* user guide, NVIDIA - no arXiv ID (vendor manual) | no arXiv ID | 26 |
| `[arm-sdot]` | *Arm Architecture Reference Manual* and the ACLE integer dot-product instructions (`SDOT`, `SMMLA`) - no arXiv ID (vendor manual) | no arXiv ID | 7 |
| `[metal]` | *Metal Shading Language Specification*, Apple - no arXiv ID (vendor manual) | no arXiv ID | 21 |

## Shelf 4 · The reference implementation

minfer is not a citation, and this book reads a specific commit of it, the [pinned baseline](baseline.md). Its documentation tree is the largest body of reading the book assigns, and it is listed per chapter rather than here, because the same document often serves two chapters for different reasons.

| Question | Where it is answered |
|---|---|
| Where do I start in that repository? | [Where the code lives](reading.md) - the directory-to-chapter table |
| What were the decisions, and why? | [The Decision Map](adr-map.md) - 82 ADRs, with the chapters that use them |
| What has the project measured about itself? | [Measurements](measurements.md), and `docs/TEST-BASELINES.md` there |
| What does it not do? | [Limits](limits.md), which points at the project's own inventories |

## Empty slots

A shelf that names only what it has read has stopped being honest about the field. These are sources this book wants and has not yet read closely enough to cite; naming them is the to-read list:

| Slot | Why it is empty | What would fill it |
|---|---|---|
| Integer tensor-core matmul inside a quantized inference kernel | the technique is documented in code and vendor manuals, not in a paper the author has read end to end | a paper (or the PTX section) that explains the int8 `mma` fragment layout for a mixed-precision quantized GEMM |
| Continuous batching and serving schedulers | Chapter 29 describes the mechanism from the implementation, not from the literature | a paper on iteration-level scheduling for LLM serving |
| KV cache compression (quantized, evicted, or shared caches) | Chapter 18 covers one packed format and prefix sharing; the research space is wider | a survey, plus one method the author has reproduced |
| Grammar-constrained decoding beyond a JSON schema | Chapter 13 compiles one schema to a mask; the theory of constrained generation is unread here | the constrained-decoding literature |
| GPU memory-bandwidth measurement methodology | Chapter 28 uses a noise floor it measures itself; no standard reference is cited | a benchmarking paper specific to GPU bandwidth, not CPU |

## Checked

```bash
# every ID on this page, resolved in one request each (the API rate-limits the default user agent)
python3 - <<'EOF'
import urllib.request, re
ids = re.findall(r"arXiv:(\d4\.\d(4, 5))", open("src/appendix/references.md").read())
print(len(ids), "arXiv IDs")
EOF
```

An ID that fails to resolve is removed from the page, not marked pending: a citation this book cannot check is a citation it should not print.

## Reading

- [Writing and Formatting Rules](conventions.md) - the labels this book puts on a claim, and why a reported number is never upgraded
- [Where the code lives](reading.md) - the reference implementation, by chapter
