# Chapter 2 · The File Is the Model

> **Compute tier**: ✅ any laptop | **Status**: stub
> **Build step S2**: A GGUF reader that prints the header, every metadata key, and the tensor table (name, dtype, shape, byte range) of a real quantized file.
> **Rule of this book**: This chapter stops at the container: bytes and their map. Tensor strides, block layouts and the meaning of the numbers inside a tensor are Chapters 3 and 4; loading weights into an engine is Chapter 19.

## First, the gist

A language model is not a program: it is a file of numbers plus a short description of what those numbers mean. **GGUF** (GGML universal file format) is that container, and everything an engine does starts by reading it faithfully: a header, a list of key-value metadata, a table of tensors with their shapes and offsets, then one large data blob. Because the blob is laid out exactly as the kernels want it, a good engine never copies it: it memory-maps the file and hands kernels slices of it. Get the parse wrong and nothing downstream can be right, which is why this is the first thing the book makes you write.

## What you'll be able to do

- Parse a GGUF v3 file end to end: header, metadata key-values of every type, tensor table, alignment padding, and the data-blob offset.
- Explain why an engine memory-maps weights instead of reading them, and what that costs in address space and page faults.
- Diagnose the four ways loading fails in practice: truncated file, unaligned blob, split model, metadata naming an architecture you do not implement.

## Lessons

- [ ] 1. What a model file has to contain, and why a container format exists at all
- [ ] 2. The header, the metadata key-values, and the value types a parser must handle
- [ ] 3. The tensor table: name, dimensions, dtype, offset, and the alignment that makes zero-copy possible
- [ ] 4. Memory-mapping, and what a page fault costs compared with a read (payoff: why a cold 7B model can load in milliseconds)
- [ ] 5. Split files, and refusing a file instead of guessing: the parse failures worth naming

## The container, field by field

| Field | Where it lives | What breaks if you misread it | Your parser |
|---|---|---|---|
| magic and version |  |  |  |
| metadata key-values |  |  |  |
| tensor infos |  |  |  |
| alignment padding |  |  |  |
| data-blob offset |  |  |  |

## Project

**Build step S2 · a parser you can point at.** Work against one small real file (a 0.6B model or a tiny generated fixture). Print, do not guess: every failure in this chapter is visible in the first 64 bytes.

The engine's loader (`gguf.rs`, `load_gguf_model` and `MmapFile`) is the reference. Read it after your parser reads the same file, never before: the point of the hard way is to have a wrong version first.

## Deliverable

- `gguf-dump <file>` printing version, key count, tensor count, and one line per tensor with dtype and element count
- A one-paragraph note on which metadata keys your engine will refuse to load without, and why
- A refusal for at least one malformed file, with the message naming the field that failed

## How you know you passed

- Your tensor table matches `gguf-dump`'s output on the same file, line for line.
- You can state the file's data-blob offset from memory and how it is computed.
- Feeding it a truncated file produces an error, never a panic and never zeros.

## Reading

- code: `src/gguf.rs` (`load_gguf_model`, `MmapFile`), `src/gguf/tests.rs` at commit `88ffa5d`
- docs: `docs/inference_e2e_walkthrough/02-gguf-load.md`
- spec: the GGUF file-format specification in the ggml repository - no arXiv ID

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
