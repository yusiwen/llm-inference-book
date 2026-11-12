# Chapter 11 · The Tokenizer

> **Compute tier**: ✅ any laptop | **Status**: stub
> **Build step S11**: A tokenizer that loads its vocabulary from the model file and encodes and decodes text exactly as the reference does, including the awkward cases.
> **Rule of this book**: This chapter is text to ids and back. What wraps those ids into a prompt (templates, special tokens) is Chapter 12; what turns logits into the next id is Chapter 13.

## First, the gist

A model does not see characters, it sees integer ids from a fixed vocabulary, and the split is learned. **BPE** (byte-pair encoding) is the usual algorithm: start from bytes, repeatedly merge the most frequent adjacent pair into a new symbol, and keep the merge list. That is the easy half. The hard half is that every model family disagrees about what text it feeds the merge list - how spaces are marked, whether a normalizer runs first, which token is special - and those disagreements are recorded in the model file, not in the algorithm. An engine that gets any of them wrong produces token ids that look plausible and answers that are subtly broken.

## What you'll be able to do

- Load a BPE vocabulary and merges from the model file and encode text with no external tokenizer library.
- Identify which pre-tokenization family a file declares, and what each family does to spaces, accents and bytes.
- Explain why a special token is declared by the file rather than inferred from its name, and what a missing byte token means.

## Lessons

- [ ] 1. Vocabulary, merges, and the merge loop on a real sentence
- [ ] 2. Pre-tokenization families: GPT-2, Llama-3, whitespace, SentencePiece-style, and the odd ones (payoff: one algorithm, many dialects)
- [ ] 3. Normalization before splitting: NFC, and why it changes the ids
- [ ] 4. Special tokens, byte fallback, and unknown characters that must not be silently dropped
- [ ] 5. Decoding is not the inverse: what round-tripping can and cannot prove

## Dialects seen in the wild

| Family | Space handling | Normalizer | Special-token rule | Your implementation status |
|---|---|---|---|---|
| GPT-2 style |  |  |  |  |
| Llama-3 style |  |  |  |  |
| SentencePiece / Unigram |  |  |  |  |
| WordPiece |  |  |  |  |
| vocabulary-less |  |  |  |  |

## Project

**Build step S11 · the tokenizer conformance suite.** Build a fixture set of awkward strings: leading spaces, doubled spaces, non-ASCII, emoji, a lone combining mark, a string longer than the context. Compare your ids with the reference tokenizer's, one string at a time.

Do not test by decoding the model's answers. Test ids directly; a tokenizer bug that shows up as a bad answer is a week of your life.

## Deliverable

- `encode` and `decode` driven from the model file, with a fixture-based conformance test
- A table naming which pre-tokenization families your engine supports and which it refuses
- A recorded diff against the reference tokenizer over your awkward fixture set

## How you know you passed

- Every fixture string produces identical ids to the reference, including the empty string and a single space.
- An unsupported dialect is refused with a message naming the field, not tokenized approximately.
- You can explain one real case where the same text tokenizes differently across two model files, and why.

## Reading

- code: `src/tokenizer.rs`, `src/tokenizer/nfc.rs` at commit `88ffa5d`
- decisions: ADR-0042 (the tokenizer applies the model's declared NFC normalizer), ADR-0043 (a special token is declared by the file), ADR-0052 (vocabulary-less files refuse at the checked entry points)
- docs: `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md`
- [bpe] Sennrich, Haddow and Birch, *Neural Machine Translation of Rare Words with Subword Units* (arXiv 2015, ACL 2016) - arXiv:1508.07909
- [jm] Jurafsky and Martin, *Speech and Language Processing* (3rd edition draft) - no arXiv ID (book; the tokenization chapter)

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
