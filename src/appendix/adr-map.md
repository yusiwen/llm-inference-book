# The Decision Map

> - **Status**: stub - the table is generated from the project at the [pinned baseline](baseline.md); the book's prose about *why* each decision matters is still to write.
> - **Rule of this book**: this page lists what the book's contract depends on. The arguments live in the chapters; the record lives here.

minfer keeps **82 architecture decision records** (ADRs) in `docs/adr/`. An ADR is one page that says: here is a decision, here is what it rules out, here is what would make us revisit it. They are the project's most reusable artifact, because an engine is mostly a pile of decisions that were expensive to make and cheap to forget.

This map is generated, not typed: the titles come from the ADR files, and the "used in" column is derived from the chapters that cite each number. **38 of the 82 are cited somewhere in this book**; the rest govern parts of the project this book does not follow (tooling, packaging, the visualizer) and are left visible so the map does not pretend to be complete.

| ADR | Title | First chapter that uses it |
|---|---|---|
| 0001 | 0001. Inference runs through one declarative compute graph | 14 |
| 0002 | 0002. Topology is a function of `GraphParams` alone, so `positions` cannot be structure | 10, 14 |
| 0003 | 0003. Metal is out of scope for this round | - |
| 0004 | 0004. The batching default follows the device | - |
| 0005 | 0005. Metal becomes a first-class backend | 21 |
| 0006 | 0006. The KV storage format is a per-engine gate, not a process-wide global | 10, 18 |
| 0007 | 0007. No ML frameworks: every operator is hand-written | - |
| 0008 | 0008. GPU safety: bounded waits, no early return past a barrier, runtime device limits | 23 |
| 0009 | 0009. A failure is an error, never a silent fallback | 16, 23 |
| 0010 | 0010. The identity gate: bitwise by default, a named tolerance class otherwise | 27 |
| 0011 | 0011. Backend ids are a file-format contract: appended, never renumbered | 16 |
| 0012 | 0012. Device is the first axis, the layer the second — and no premature `common` | 16 |
| 0013 | 0013. The CPU quantizes activations to Q8_0; a device reads f32 | 6 |
| 0014 | 0014. A KV session is a versioned, checksummed file — never a memory dump | 18 |
| 0015 | 0015. The offload `auto` fit takes a prefix, not a knapsack | 17 |
| 0016 | 0016. A failed device-memory query is not a zero budget | 17 |
| 0017 | 0017. Speculative decoding refuses the features its identity contract cannot carry | 20, 29 |
| 0018 | 0018. The grammar mask is one stage inside the single sampler pipeline | 13 |
| 0019 | 0019. A chat template that cannot be rendered refuses the load | 12 |
| 0020 | 0020. A quantized file is byte-identical to `llama-quantize`, or it is wrong | 5, 27 |
| 0021 | 0021. bf16 is a round-to-nearest-even cast, and 1-D tensors stay f32 | 9 |
| 0022 | 0022. A defect in a frozen ADR is corrected by a new ADR, not by editing it | - |
| 0023 | 0023. Each machine ledger lives beside its checker, one per prose target | 27 |
| 0024 | 0024. A machine-read record is not book content | - |
| 0025 | 0025. A docs-only change runs the docs gate, not the compilers | - |
| 0026 | 0026. The classifier carries no docs-path exception, and its list is ratcheted | - |
| 0027 | 0027. A citation is dated too — paths, symbols and counts, not only capabilities | - |
| 0028 | 0028. A document's ADR map lists what its contract depends on | - |
| 0029 | 0029. An ADR's prose count is dated too, and its References are map evidence | - |
| 0030 | 0030. Launch severity lives in the helper, not in 120 call sites | 23 |
| 0031 | 0031. Capture runs in thread-local mode, because Global lets a foreign thread's call join the window | 25 |
| 0032 | 0032. The packed-KV staging window is f32, not f16 | 18, 25 |
| 0033 | 0033. The CUDA pool recycles exact byte lengths, never frees, and reports OOM as an error | 17, 23 |
| 0034 | 0034. The smem opt-in is an eager pre-warm by construction, with the lazy path as defence in depth | 25, 26 |
| 0035 | 0035. The q4_K dsc plane is admitted by two gates, and the payload test is equality | 26 |
| 0036 | 0036. bf16 weights get their own device kernels, not a dtype flag on the f16 ones | - |
| 0037 | 0037. A Metal weight dtype a kernel cannot consume is refused, never run as a wrong kernel | 21 |
| 0038 | 0038. The per-tensor registration dispatch is one shared rule, not a copy per loader | 19 |
| 0039 | 0039. The corpus is entered through a question at PR time, and its maps are checked | - |
| 0040 | 0040. A decision is bound to the surfaces it governs, so the question names it | - |
| 0041 | 0041. A row that cannot take the target's block is demoted, the legacy targets included | - |
| 0042 | 0042. The tokenizer applies the model's declared NFC normalizer | 11 |
| 0043 | 0043. A special token is declared by the file, never inferred from its name | 11 |
| 0044 | 0044. The input normalizer follows the pre-tokenizer family's declaration | - |
| 0045 | 0045. The tokenizer engine is a dispatch on `tokenizer.ggml.model` | - |
| 0046 | 0046. WordPiece is the third engine, and it follows BERT's normalizer | - |
| 0047 | 0047. The unigram engine reads the file's charsmap, and SentencePiece is its reference | - |
| 0048 | 0048. gemma4 is a BPE over raw UTF-8, and its rule comes from the model value | - |
| 0049 | 0049. rwkv is a trie over decoded bytes, and its unknown role is conditional | - |
| 0050 | 0050. plamo2 is a backward DP over the file's own scores, and llama.cpp's automaton is an index | - |
| 0051 | 0051. hybriddna is the gpt2 engine plus a DNA k-mer block, and its vocabulary is escaped in place | - |
| 0052 | 0052. `none` is a vocabulary-less file, and the checked entry points are how it refuses | 11 |
| 0053 | 0053. `whitespace` is the gpt2 engine behind a whitespace cut, and it has no artifact gate | - |
| 0054 | 0054. `jina-v1-en` is the plain GPT-2 pre rule, not a Qwen variant | - |
| 0055 | 0055. `ignore_merges` is a whole-piece shortcut, and the llama3 pre group sets it | - |
| 0056 | 0056. A missing byte token is accepted only when no token can reach it | - |
| 0057 | 0057. A template file overrides the GGUF copy, and time enters a render through one injected clock | 12 |
| 0058 | 0058. A request's tools reach the prompt as the client wrote them, and a template that drops them refuses by name | 12 |
| 0059 | 0059. The batched worker queues a request against a bounded depth, and a run that makes no progress is failed rather than waited on | 8, 26, 29 |
| 0060 | 0060. A mixture is a per-tensor policy over the same encoders, and the `OUTPUT`/tied branch runs for every quantizing ftype | - |
| 0061 | 0061. The Q6_K dot accumulates in eight lanes, and the scalar kernel implements that order | - |
| 0062 | 0062. A request message reaches the prompt as the client wrote it, and a replayed tool call that reaches nothing refuses by name | 12 |
| 0063 | 0063. A completion's tool call is extracted, and `tool_choice` is a grammar on the model's own call format | - |
| 0064 | 0064. A hash that does not resolve is history only where the document or the ledger says so | - |
| 0065 | 0065. The hash marker covers the block below it, not the document | - |
| 0066 | 0066. A tool-call format is a table row | - |
| 0067 | 0067. Qwen3.5's XML call is parsed, and its arguments are serialized JSON | - |
| 0068 | 0068. Ministral 3 puts the name in the anchor, and the grammar ends at the arguments | - |
| 0069 | 0069. Gemma 4's call is a lenient JSON body, and its arguments are serialized | - |
| 0070 | 0070. tool_choice constrains the arguments to the offered tool's schema | - |
| 0071 | 0071. Qwen3.5's XML call gets its own tool_choice grammar | - |
| 0072 | 0072. A draft proposal is the target's own filtered pick, computed from one shared prefix | 26, 29 |
| 0073 | 0073. Gemma 4's lenient body gets a dialect tool_choice grammar | - |
| 0074 | 0074. The llama3 rope table is read from the GGUF, not re-derived from four numbers | 9 |
| 0075 | 0075. A model owns the mapping it reads, and a tensor carries a handle into it | 19 |
| 0076 | 0076. A tensor's bytes are a mapping handle plus a range, never a `'static` view of a mapping | - |
| 0077 | 0077. A Metal registration is released with its model, and the cache belongs to the engine | - |
| 0078 | 0078. A CUDA weight is released with its model, and a live captured graph holds a lease | - |
| 0079 | 0079. The engine is a crate with two API tiers, and the apps are its first consumers | - |
| 0080 | 0080. The Metal backend is a cargo feature, and a metal-less build refuses the name | 21 |
| 0081 | 0081. A tensor's bytes are a backing handle plus a range, and the backing may be an owned buffer | - |
| 0082 | 0082. The application layer is its own package, and the engine's four application features retire | - |

## How to read this page

- A dash in the last column means the decision is real but this book never touches it. That is information about the book's scope, not a defect in the record.
- Numbers are append-only in this project (ADR-0011), so a citation is stable: `ADR-0033` means the same thing in five years. That is why the book cites numbers and not section titles.
- A defect in a frozen ADR is corrected by a new ADR, never by editing the old one (ADR-0022). When you find the record contradicting the code, look for the later number.

## Reading

- `docs/adr/README.md` in the reference repository - the index and the format
- [Chapter 26](../26-the-optimization-campaign/index.md) - the campaign whose decisions most of Part 5 cites
