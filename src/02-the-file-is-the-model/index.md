# Chapter 2 · The File Is the Model

> - **Compute tier**: ✅ any laptop
> - **Status**: written - every number and offset below was read out of one real file, `qwen2.5-0.5b-instruct-q4_k_m.gguf` (491,400,032 bytes), with `scripts/gguf-inspect.py`; the 7B file is not parsed here, and the block layouts behind the byte lengths are Chapter 4's
> - **Build step S2**: A GGUF reader that prints the header, every metadata key, and the tensor table (name, dtype, shape, byte range) of a real quantized file.
> - **Rule of this book**: This chapter stops at the container: bytes and their map. Tensor strides, block layouts and the meaning of the numbers inside a tensor are Chapters 3 and 4; loading weights into an engine is Chapter 19.

## First, the gist

A language model is not a program. It is a file of numbers plus a short description of what those numbers mean - and "short" is doing quiet work in that sentence, because in the file this chapter reads, the description runs to 5.9 megabytes.

**GGUF** (GGML universal file format) is that container, and it is the first thing your engine has to read: a 24-byte header, a dictionary of key-value metadata, a table of tensors with their shapes, types and offsets, then one large data blob. The whole design exists to make one operation cheap: an engine takes a slice of its address space, calls it a weight matrix, and never copies a byte. That is only possible if the file's own table says exactly where every tensor starts and what shape it has, which is why the table is the point of the format and why parsing it wrong poisons everything downstream.

The numbers in this chapter come from one file, and they are worth meeting early because they are surprising in a specific way: **291 tensors, 26 metadata entries, 485 MB of data, and a name - "Q4_K_M" - that describes a recipe rather than the contents.** Almost 55 per cent of this file's bytes are Q5_0, and only 6 per cent are Q4_K. A reader who trusts the file name and skips the tensor table will be wrong about the single most important number in inference: how many bytes a token costs.

<div class="math-display">
$$
\text{file} \;=\; \text{header} \;+\; \text{metadata} \;+\; \text{tensor table} \;+\; \text{padding} \;+\; \text{blob}
$$
</div>

## What you'll be able to do

- Parse a GGUF v3 file end to end: header, metadata key-values of every type, tensor table, alignment padding, and the data-blob offset - and say for this file that the blob starts at byte 5,947,744, and why.
- Check the parse against the file itself: the tensor ranges must tile the blob exactly, and the last one must end at the file's end.
- Explain why an engine memory-maps weights instead of reading them, with measured numbers for both, and why the *parse* rather than the load is what startup time is made of.
- Diagnose the four ways loading fails in practice: truncated file, unaligned blob, split model, metadata naming an architecture you do not implement - and refuse each one with the field named.

## Lessons

- [ ] 1. What a model file has to contain, and why a container format exists at all
- [ ] 2. The header, the metadata key-values, and the value types a parser must handle
- [ ] 3. The tensor table: name, dimensions, dtype, offset, and the alignment that makes zero-copy possible
- [ ] 4. Memory-mapping, and what a page fault costs compared with a read
- [ ] 5. Split files, and refusing a file instead of guessing: the parse failures worth naming

### Lesson 1 · What a model file has to contain, and why a container format exists at all

Three things have to travel together for a file to be a usable model:

1. **The weights**, as a few hundred arrays of bytes.
2. **The shapes** - how to interpret each array as a matrix, and in what order the dimensions go. A 896 by 151,936 array of bytes is meaningless without knowing which dimension is which.
3. **The conventions** - what architecture this is, how many layers, how the tokenizer splits text, what the chat template looks like, where the end-of-sequence id is. This is the part beginners underestimate: it is not decoration, it is the specification the engine's code has to match, and it is the part that makes a file *self-describing*.

A directory of `.npy` files has the first and second and none of the third. A Python pickle has all three and is unsafe to load from a stranger. Safetensors solves the first two cleanly for unquantized weights and leaves the third to the model card - which is why the same model appears in three different repository layouts depending on who shipped it.

GGUF's answer is to put all three in one file, in one order, with **offsets that are absolute enough to allow zero-copy loading**:

<figure class="book-figure">{{#include ../figures/02-gguf-layout.svg}}</figure>

*Figure 2.1 - a GGUF file from byte zero: a 24-byte header, 5.93 MB of metadata, a 16.5 KB tensor table, 3 bytes of padding to reach the alignment, and a 485 MB blob that the table describes exactly.*

Two properties of that layout are worth naming before the details, because everything else follows from them.

**The data comes last, and nothing in it is self-describing.** There are no per-tensor headers in the blob, no lengths, no separators. A tensor is 144,643,072 bytes of Q8_0 at some offset, and the only thing that knows it is a tensor is the table 5.9 megabytes earlier. This is deliberate: a kernel wants to read `[offset, offset + len)` and start multiplying, with no parsing in the inner loop.

**The table is small; the metadata is not.** Figure 2.1 is drawn not to scale because the metadata is 5.9 MB against a 16.5 KB table. The reason is the tokenizer: `tokenizer.ggml.tokens` is an array of 151,936 strings and `tokenizer.ggml.merges` is 151,387 more, and both live in the metadata section as ordinary key-values. So "metadata" in GGUF does not mean "a few hyperparameters"; it means "everything that is not a weight", which includes a 4 MB vocabulary.

### Lesson 2 · The header, the metadata key-values, and the value types a parser must handle

The first 24 bytes are the whole header, and they are the cheapest part of the format to get right:

<!-- facts: gguf-qwen2.5-0.5b.header -->
```text
bytes 0-23   47 47 55 46  03 00 00 00  23 01 00 00 00 00 00 00  1a 00 00 00 00 00 00 00
decoded      magic "GGUF" | version 3 | tensor count 291 | metadata count 26
```
<!-- /facts -->

- `47 47 55 46` is `GGUF` in ASCII. A file that fails this is not a GGUF file and the only useful response is to say so.
- `03 00 00 00` is version 3 as a little-endian u32. Version matters because the tensor-info record changed shape between v1 and v2; a v1 file parsed by v3 rules produces plausible nonsense.
- `23 01 00 00 00 00 00 00` is 291: the number of tensor-info records that follow the metadata.
- `1a 00 00 00 00 00 00 00` is 26: the number of metadata key-values.

Both counts are u64 rather than u32, which looks wasteful for a count that fits in nine bits. It is a format decision in the other direction: element counts and string lengths are also u64, so nothing in the format needs a "your file is too big" rule later.

Then come the 26 metadata entries. Each one is a key, a type tag, and a value:

```rust
// The loop, which is the whole of the metadata section.
for _ in 0..metadata_count {
    let key = read_string();          // u64 length, then that many UTF-8 bytes
    let vtype = read_u32();           // one of twelve
    let value = read_value(vtype);    // fixed size, a string, or an array of either
}
```

Two traps live in those four lines.

**A GGUF string is length-prefixed, not null-terminated.** The u64 comes first, then exactly that many bytes of UTF-8. A parser that assumes C strings will read the length bytes as text; a parser that confuses bytes for characters will mis-slice every non-ASCII token and then, from that point on, read the file at the wrong offset and produce a table of nonsense with no error at all.

**The value types are twelve, and arrays nest one level.** `UINT8` through `FLOAT64`, plus `BOOL`, `STRING`, `ARRAY`. An array carries its own element type and count, then the elements; an array of strings is what the tokenizer needs, and it is the reason 5.9 MB of this file is "metadata". A parser can be complete with a 60-line match - but it must be complete, because one skipped type turns the rest of the file into garbage.

Here is what those 26 entries actually are in this file, abbreviated to the ones with consequences:

| Key | Type | Value here | What the engine does with it |
|---|---|---|---|
| `general.architecture` | string | `qwen2` | picks the graph builder; an unknown value is a refusal (Chapter 19) |
| `general.size_label` | string | `630M` | the file's own claim about its size - and the arithmetic later agrees |
| `qwen2.block_count` | u32 | 24 | how many layers to build |
| `qwen2.context_length` | u32 | 32768 | the KV cache's maximum (Chapter 18) |
| `qwen2.embedding_length` | u32 | 896 | the width of every activation row |
| `qwen2.feed_forward_length` | u32 | 4864 | the MLP's hidden width |
| `qwen2.attention.head_count` | u32 | 14 | query heads |
| `qwen2.attention.head_count_kv` | u32 | 2 | key/value heads - the grouped-query ratio, 7:1 here |
| `qwen2.rope.freq_base` | f32 | 1000000.0 | the rotation base in RoPE |
| `qwen2.attention.layer_norm_rms_epsilon` | f32 | 1e-6 | the epsilon in every RMSNorm |
| `tokenizer.ggml.tokens` | array of string | 151,936 entries | the token table (Chapter 11) |
| `tokenizer.ggml.merges` | array of string | 151,387 entries | byte-pair merges (Chapter 11) |
| `tokenizer.ggml.eos_token_id` | u32 | 151645 | where generation stops |
| `tokenizer.chat_template` | string | a Jinja template | how messages become a prompt (Chapter 12) |
| `general.alignment` | u32 | 32 | where the blob starts (Lesson 3) |

Fourteen keys that decide everything from the shape of the graph to where generation stops, and the parser has to hand all of them to the engine without interpretation. This is the practical meaning of the book's rule that the engine's job starts with a faithful read: `qwen2.attention.head_count_kv = 2` is not a hint, it is the architecture, and an engine that assumes 1 (multi-head) or 14 (full multi-head) will allocate the wrong KV cache and read the wrong weight slice.

**Cost.** Parsing those 26 entries - including 303,323 strings - takes 0.16 s in Python on the M4 Pro (measured; `time python3 scripts/gguf-inspect.py FILE`), and most of it is decoding strings, not arithmetic. Keep that number in mind for Lesson 4: it is larger than any other cost on the load path except the page faults.

### Lesson 3 · The tensor table: name, dimensions, dtype, offset, and the alignment that makes zero-copy possible

After the metadata comes one record per tensor. Each record is a name, a dimension count, the dimensions, a type tag, and an offset:

<!-- facts: gguf-qwen2.5-0.5b.tensor-sample -->
```text
output.weight           n_dims=2  dims=[896, 151936]  type=8 (Q8_0)  offset=0
token_embd.weight       n_dims=2  dims=[896, 151936]  type=6 (Q5_0)  offset=144643072
blk.0.ffn_down.weight   n_dims=2  dims=[4864, 896]  type=14 (Q6_K)  offset=238239232
blk.0.attn_q.weight     n_dims=2  dims=[896, 896]  type=6 (Q5_0)  offset=248445184
```
<!-- /facts -->

Three details across those lines carry the chapter.

**The offset is relative to the blob, not to the file.** `output.weight` at offset 0 means the first byte of the data section, which is byte 5,947,744 of the file. That single indirection is what lets the same file be memory-mapped at any address: the loader adds one base value to every offset and never re-reads a table.

**The dimensions are stored innermost-first.** The array is `[896, 151936]` and that does not mean 896 rows of 151,936. In GGML's own terms it means `ne[0] = 896, ne[1] = 151936`: the first element is the *contiguous* dimension, so this tensor is 151,936 rows of 896 values each, laid out row-major over the second dimension. Get this backwards and every matrix multiply reads a transposed matrix - which, for a weight matrix that is nearly square, produces output that is wrong in a way that still looks like text.

**The type tag indexes a block table, not a byte size.** `type=8` is Q8_0, which stores 32 weights in 34 bytes; `type=6` is Q5_0, 32 weights in 22 bytes; `type=12` is Q4_K, 256 weights in 144 bytes; `type=14` is Q6_K, 256 in 210. The type numbers are not contiguous and not in any order that a beginner would guess - 4 and 5 are retired formats - so the table is data you copy from the specification, never arithmetic you derive.

With that table, a tensor's byte length is `elements / elements_per_block * bytes_per_block`, which is only computable if you know the block layout. Chapter 4 is that table in full; here the parser needs just enough of it to compute each tensor's range.

The blob's start is the one place where the format allows slack. The table ends at byte 5,947,741 in this file; the alignment is 32; the blob therefore starts at the next multiple of 32, which is 5,947,744:

<div class="math-display">
$$
\text{blob} \;=\; \left\lceil \frac{\text{end of table}}{\text{alignment}} \right\rceil \times \text{alignment} \;=\; \lceil 5{,}947{,}741 / 32 \rceil \times 32 \;=\; 5{,}947{,}744
$$
</div>

Those three padding bytes are not laziness. In this file all 291 tensor offsets are multiples of 32, and so is every tensor's byte length (checked, not assumed: the inspector prints the property and it holds with zero exceptions). A vectorised kernel can therefore load 32 bytes at a time without a tail case, and a memory-mapped slice can be handed to SIMD code without alignment fixups - which is the whole reason the format has an alignment field at all. An engine that ignores the alignment reads *almost* the right bytes, which is the worst possible outcome: the first few elements are wrong by a shift and the rest are noise.

**And then the check that makes a parser trustworthy.** The tensor ranges must tile the blob: sorted by offset, each tensor must begin exactly where the previous one ended, and the total must equal the blob size. For this file that arithmetic closes exactly - 291 tensors, 485,452,288 bytes, no gaps, last tensor ending at the file's end:

```text
$ python3 scripts/gguf-inspect.py ~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf
  size on disk      491400032 bytes (0.491 GB)
  magic/version     GGUF v3
  tensor count      291
  metadata entries  26
  metadata ends at  byte 5931189
  alignment         32
  tensor table ends byte 5947741
  data blob starts  byte 5947744  (aligned up from 5947741, padding 3)
  data blob size    485452288 bytes
  tensors tile the  blob: yes
```

A parse that is wrong by one byte in the metadata section will still print *something* for every one of those lines, and the tiling check is what catches it. Make it an assertion in your loader, not a print in your debug tool: it costs one pass over 291 records and it is the difference between "the file is wrong" and "your engine is reading a shuffled model".

**And a parser needs a witness that is not itself.** The tiling check proves the file is internally consistent; it cannot prove your *interpretation* is right. That is the failure this chapter's own numbers had: a draft called Q5_0 `type=5` in a table, which is what the specification says it is not, and nothing in the repository disagreed, because the wrong number had only ever been typed once. What settled it was a second implementation - `gguf-py`, the reference Python reader that ships with ggml - reading the same file and producing the same 291 tensor records. The book keeps that comparison as evidence (`reference/gguf-crosscheck.json`: 1,463 field-by-field comparisons, all agreeing, plus three synthetic fields `gguf-py` invents from the header, which is itself the kind of difference you have to explain rather than paper over).

Do the same for your parser, and do it before you believe your own dump: llama.cpp's `llama-gguf-dump` or `gguf-py` will disagree with you, and the file settles the dispute. What such a comparison proves is structure - counts, names, dtypes, shapes, offsets. It says nothing about whether your kernel reads the bytes correctly; that is Chapter 27's problem, and it needs a different witness.

**The dtype mix is the chapter's quiet bombshell.** The file is called Q4_K_M. Here is where its bytes actually go:

<!-- facts: gguf-qwen2.5-0.5b.dtype-mix -->
| dtype | Bytes | Share of blob | Tensors in this file |
|---|---|---|---|
| Q5_0 | 266,742,784 | 54.9% | `attn_k.weight`, `attn_output.weight`, `attn_q.weight`, `attn_v.weight`, `ffn_gate.weight`, `ffn_up.weight`, `token_embd.weight` (133) |
| Q8_0 | 146,105,344 | 30.1% | `attn_v.weight`, `output.weight` (13) |
| Q6_K | 42,900,480 | 8.8% | `ffn_down.weight` (12) |
| Q4_K | 29,417,472 | 6.1% | `ffn_down.weight` (12) |
| F32 | 286,208 | 0.1% | `attn_k.bias`, `attn_norm.weight`, `attn_q.bias`, `attn_v.bias`, `ffn_norm.weight`, `output_norm.weight` (121) |
<!-- /facts -->

So the file name is a recipe, and the recipe landed mostly on Q5_0 here - not because the model is small, but because whatever produced this file chose per tensor, and the file records the choice rather than the reason. Put the arithmetic on it and the name stops being a summary of anything:

<!-- facts: gguf-qwen2.5-0.5b.bits -->
```text
parameters in the file        630,167,424
blob bytes per parameter      6.16 bits
whole file per parameter      6.24 bits  (metadata included)
by dtype, bits per weight     Q5_0 5.50, Q8_0 8.50, Q6_K 6.56, Q4_K 4.50, F32 32.00
```
<!-- /facts -->

Six and a fifth bits per parameter, in a file whose name says 4.5 - and, if you divide by the model card's 494 million instead of the 630 million the file stores, 7.9, which is the figure Part 0 quotes. Both divisions are correct; they answer different questions, and the gap between them is the duplicated embedding described in the next paragraph. Every byte-per-token figure in Chapter 1's ceiling was therefore arithmetic on the *file size*, which is the only number that already contains all of this. That is why the book keeps saying file size and never "bits per weight": the second is a summary the file itself contradicts, and it contradicts it by almost 40 per cent.

One more, because it explains the file's own size label. The 291 tensors add up to 630,167,424 parameters, and `general.size_label` says `630M`. But the model card for Qwen2.5-0.5B counts 494,032,768. The difference is exactly 136,134,656 = 896 x 151,936: the file stores the vocabulary matrix twice, once as `token_embd.weight` and once as `output.weight`, at different precisions. A model whose embeddings are tied in training is untied in the file. Neither count is a lie; they are counts of different things, and **the only parameter count that matters is the one belonging to the bytes you are going to read.**

### Lesson 4 · Memory-mapping, and what a page fault costs compared with a read

An engine has two ways to get the weights in front of its kernels.

**Read them.** `File::read` copies 485 MB from the kernel's page cache into your process's heap. You now own two copies of the weights in RAM in the worst case (the cache's and yours), you paid a real memory bandwidth cost for the copy, and you cannot start computing until the last byte arrives.

**Map them.** `mmap` creates a virtual mapping of the file into your address space. Nothing is copied. When a kernel touches an address that has no physical page behind it yet, the CPU faults, the kernel reads that page from disk (or finds it in the page cache), maps it in, and the instruction retries. The file's bytes reach your computation without ever being an owned buffer.

The difference is measurable, so measure it rather than asserting it. On the M4 Pro, with the file already in the page cache (it had just been downloaded and parsed), touching one byte per 4 KiB page of 469 MiB:

```text
read() into a heap buffer, then touch:   first 64.2 ms, second 46.7 ms
mmap, then touch the same pages:         first 35.9 ms, second 31.3 ms
read the 24-byte header and nothing else: first 0.1 ms, second 0.0 ms
```

Three readings, three conclusions.

**Mapping is not free, it is cheaper.** 31-36 ms versus 47-64 ms is a factor of 1.5, not a factor of infinity, because both paths fault the same pages and pay the same cache reads; what `mmap` saves is the copy. Anyone who tells you mapping is instant is describing the *open*, not the first pass over the weights.

**Those numbers are warm, and I cannot honestly present them as cold.** The kernel was already holding the file. A cold run pays disk latency per page - on this machine an SSD, so tens of microseconds each, which would dominate everything above. Measuring cold requires purging the file cache, which needs privileges the book does not assume; the honest form is "warm: 31-36 ms; cold: dominated by the device, and you should measure it on yours".

**The load was never the expensive part.** Reading and parsing the whole map takes 0.16 s in Python, and the header alone takes 0.1 ms. An engine can print its model's architecture, tensor count and total size before a single weight page is faulted in - which is why `llama-cli --verbose` answers instantly while the first token takes seconds. The cost of "loading a model" is not the file; it is the first forward pass faulting in the pages it needs, and after that, the memory bandwidth.

What mapping costs instead of time: **address space** (harmless on 64-bit, where 485 MB is a rounding error), **predictable-ish faults at unpredictable moments** (the first token pays them; Chapter 20 is where that shows up as a latency spike), and **a contract with the loader**: the file must not change under the mapping, or your process takes SIGBUS instead of an error. The reference engine's `MmapFile` in `src/gguf.rs` is the small type that owns that contract, and the reason it exists rather than a `Vec<u8>` is the three numbers above.

One practical consequence worth writing into your loader now: **map, do not read, and fault lazily.** Then the memory your process reports at startup is a few megabytes, and the numbers you record in your notes are about the machine rather than about your allocator.

### Lesson 5 · Split files, and refusing a file instead of guessing

Two things remain, and both are about what happens when the file is not the file you expected.

**Splits.** The format has a 64-bit offset and no size limit, but distribution does not: the 7B file in this book arrives as `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` and `...-00002-of-00002.gguf`, 3.99 GB and 0.69 GB. The convention is in the name, and the loader's job is to notice the pattern, find the siblings, and treat the parts as one logical blob whose tensors happen to live in different files. The reference engine does exactly that in `src/gguf.rs::resolve_splits`, and it is a good example of a function that is twenty lines of string handling and saves a class of bug: an engine that opens only part 1 will load a model with two thirds of its tensors and fail somewhere deep in a matrix multiply, far from the cause.

**Refusals.** Every real failure in this chapter has one thing in common: it is silent if you let it be. Here is what this chapter's own tool does with three broken files, which is the behaviour to copy:

```text
$ head -c 100 model.gguf > trunc.gguf && python3 scripts/gguf-inspect.py trunc.gguf
EOFError: wanted 8 bytes at 93, file ends at 100

$ printf 'NOTGGUFF and some padding here' > bad.gguf && python3 scripts/gguf-inspect.py bad.gguf
bad.gguf: not a GGUF file (magic is b'NOTG')

$ printf 'GGUF' > four.gguf && python3 scripts/gguf-inspect.py four.gguf
EOFError: wanted 4 bytes at 4, file ends at 4
```

Note *where* the first error is raised: 93 bytes in, looking for an 8-byte string length, not at the end of the file. A bounds-checked reader reports the failure at the offset where the structure stopped making sense, which is almost always more useful than "the file is truncated". That is the reason the parser's error type should carry the position and the expected field, not just a message.

The five refusals this chapter's build step asks for, each with the field it names:

| Condition | Why it must refuse rather than continue |
|---|---|
| magic is not `GGUF` | you are not reading this format at all; every subsequent offset is a guess |
| version greater than 3 | the tensor-info layout is versioned; newer files may parse into plausible nonsense |
| any read runs past the end of the file | a truncated download is the most common real failure, and zeros are worse than an error |
| `alignment` is not a power of two in `[1, 2^31]` | the blob offset becomes ambiguous; several different starts look "aligned" |
| a tensor's range ends past the file, or the ranges do not tile the blob | the table and the data disagree - one of them is corrupt |

Two more that are judgement calls rather than corruption:

- **Unknown dtype** (`type` not in your table): refuse when you need the tensor, not when you parse the table. A file with one IQ4_XS tensor you never read is still usable for the layers you do read - but the moment a forward pass needs it, refusing is the only honest option.
- **Unknown architecture** (`general.architecture = "mamba"`): parse everything, then refuse at graph construction with the architecture name in the message. The distinction matters because the file is still readable, and a reader diagnosing a bad download benefits from the table printing before the refusal.

That is the chapter's contract in one sentence: **parse faithfully, assert the layout's arithmetic, and name the field when you stop.** The temptation to be forgiving - to pad a short read with zeros, to round an odd alignment, to skip a metadata type you do not know - produces an engine that runs and lies, which is the failure mode the rest of this book is built to avoid.

## The container, field by field

The numbers come first because they are not opinions: this table is *rendered from the file* by `scripts/facts.py`, out of `facts/gguf-qwen2.5-0.5b.json`, and the repository's gate fails if the page and the facts file disagree. That is a deliberate consequence of this chapter's most embarrassing failure - an earlier draft of the table below had two tensor type ids wrong, and no checker could see it, because the wrong numbers were only ever typed once. Now they are typed nowhere:

The judgement is mine, and it comes after the facts:

<!-- facts: gguf-qwen2.5-0.5b.regions -->
| Region | Bytes | Decimal range |
|---|---|---|
| header | 24 | 0-23 |
| metadata key-values | 5,931,165 | 24-5,931,188 |
| tensor table | 16,552 | 5,931,189-5,947,740 |
| alignment padding | 3 | 5,947,741-5,947,743 |
| data blob | 485,452,288 | 5,947,744-491,400,031 |
| **the file** | **491,400,032** | **0-491,400,031** |
<!-- /facts -->

What breaks when each region is misread:

- **magic and version** - you are parsing a different format, or a different vintage of this one. The tensor-info record changed shape between v1 and v2, so a version check is not politeness.
- **tensor count** - you read 291 records where none exist, or stop after 26. Every count in the header is a loop bound, and a wrong loop bound is a wrong file offset.
- **metadata count** - you parse the tensor table as metadata. This is the single most damaging error in the chapter, because everything after it shifts and the output is still plausible.
- **metadata key-values** - a value type you do not handle silently consumes the wrong number of bytes, and the error surfaces somewhere else.
- **tensor table** - the blob offset is wrong by the difference, so every kernel reads a shifted view of every weight.
- **alignment padding** - the blob starts off by up to `alignment - 1` bytes; a vectorised load gets a shifted view and the first elements are noise.
- **data blob** - the only region that is data rather than description, and the only one where a mistake is invisible: the engine runs and lies.

## Project

**Build step S2 · a parser you can point at.** One small real file, one parser, and printed output rather than asserted beliefs. The small file is enough: everything structural in this chapter is in the first six megabytes of any GGUF, and the 0.5B is small enough to iterate on in a second.

Work in `myfer-file`, and keep the API small enough to test:

```rust
// The shape of the build step, not its implementation.
pub struct GgufFile { pub header: Header, pub metadata: Metadata, pub tensors: Vec<TensorInfo>, pub blob_at: u64 }
impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, LoadError>;   // header + metadata + table, no weights
    pub fn tensor<'a>(&'a self, name: &str) -> Option<&'a TensorInfo>;
    pub fn byte_range(&self, t: &TensorInfo) -> Result<Range<u64>, LoadError>;  // needs the block table
}
```

Order of work, and the order matters:

1. **Read the header and print the four numbers.** Compare with `scripts/gguf-inspect.py` on the same file. If the counts disagree, stop - everything after this is arithmetic on them.
2. **Read the metadata, skipping values you do not need but not types you do not know.** Print the 26 keys with their types. A parser that can print a key it does not understand is a parser that will not corrupt the stream.
3. **Read the tensor table, computing byte lengths from the block table.** Print name, dtype, element count, byte length, and absolute range.
4. **Assert the tiling.** Sorted by offset, no gaps, sum equals blob size. This is your loader's invariant, not your dump tool's feature.
5. **Refuse three malformed files** - truncated, wrong magic, and a hand-edited offset in the table - each with the position and field in the message.

The reference loader is `src/gguf.rs` (`load_gguf_model`, `MmapFile`). Open it after step 4 works, not before. The one thing worth comparing first is how it reports a bad tensor offset, because that is where a parser either helps its author or abandons them.

## Deliverable

- `gguf-dump <file>` printing version, key count, tensor count, and one line per tensor with dtype, shape, element count and absolute byte range.
- A tiling assertion in the loader, plus one malformed file per refusal in the table above, each rejected with the field named.
- A one-paragraph note on which metadata keys your engine will refuse to load without, and why - start from `general.architecture`, the four shape keys, and the tokenizer's token and merge arrays.
- One line in your notes: this file's size, blob offset, tensor count, and dtype mix, measured on your machine.

## How you know you passed

- Your tensor table matches `scripts/gguf-inspect.py`'s output on the same file, line for line - including the byte lengths, which means your block table is right for every dtype this file uses (five of them).
- You can state the file's data-blob offset from memory *and* how it is computed, and your number is 5,947,744 for this file.
- Feeding it a truncated file produces an error naming the offset, never a panic and never zeros.
- Your dump reports a dtype mix that is not the file's name: if your tool says this file is mostly Q4_K, it is not reading the tensor table.
- Your table also agrees with a second implementation - `gguf-py` or llama.cpp's dumper - on every tensor's name, dtype, shape and offset. Agreement with your own dump tool is not evidence; it is the same code twice.

## Reading

- code: `src/gguf.rs` (`load_gguf_model`, `MmapFile`), `src/gguf/tests.rs` at the book's [pinned baseline](../appendix/baseline.md)
- docs: `docs/inference_e2e_walkthrough/02-gguf-load.md`
- [gguf] The GGUF file-format specification, in the ggml repository - no arXiv ID (specification) - the header, the value types and the tensor-info record; read this before writing the parser, and read it again when the tiling check fails
- cross-check: `scripts/gguf-crosscheck.py` in this repository, and `gguf-py` in the ggml project - a second reader, so that "my parser agrees with my dump tool" is never the evidence you submit
- [caqa] Hennessy and Patterson, *Computer Architecture: A Quantitative Approach* (6th edition, 2017) - no arXiv ID (book) - chapter 5, for what a page fault actually charges and why a mapping is cheaper than a copy but not free

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?) - touched here from an unexpected direction: the load path is 0.16 s of parsing and 31-36 ms of page faults, and neither is bandwidth-bound work; the weights only become a bandwidth question once a forward pass reads them.
- [x] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs) - partly: this chapter shows what the bytes *are*. One file is 55 per cent Q5_0 and 6 per cent Q4_K under a name that says Q4_K_M, which is the strongest available argument for reading the tensor table instead of trusting a label. The arithmetic of blocks is Chapter 4.
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch) - not yet; this chapter's boundary is between a file and an address space.
- [x] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement) - partly: the tiling assertion is a correctness check that costs nothing and catches the one class of bug this chapter creates, and the mmap-versus-read numbers are the first place in the book where two implementations of the same thing are measured against each other rather than argued about.
