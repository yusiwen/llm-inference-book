#!/usr/bin/env python3
"""tok.py - the book's tokenizer instrument: GGUF vocabulary in, token ids out.

This is a measuring instrument, not the deliverable. The chapter's exercise is a tokenizer of your own inside
your engine; this script exists so that the chapter's ids, counts and round-trip results are produced from a
real model file instead of typed, and so the reader has a second opinion when their own encoder disagrees.

It reads the two arrays that decide everything - `tokenizer.ggml.tokens` and `tokenizer.ggml.merges` - plus the
three metadata keys that say what to do with them (`model`, `pre`, `token_type`), and implements the two halves
of byte-level BPE:

  * the pre-tokenizer: one regular expression, selected by `tokenizer.ggml.pre`, that cuts the text into
    pieces no merge may cross;
  * the merge loop: repeatedly join the adjacent pair with the lowest merge rank, then look each surviving
    piece up in the vocabulary.

Byte-level encoding matters before either: every byte of the pre-tokenized piece is mapped to a printable
character (`!` stays `!`, a space becomes `\u0120`, and so on), because the file's vocabulary is a list of
strings and a byte value like 0x0A has no printable form of its own.

Usage:
    python3 scripts/tok.py FILE [--out facts/tokenizer.json] [--fixtures]

`--fixtures` prints one line per fixture string (the escape form, the ids, the decoded text) and is what a
reader runs when a fixture in their own engine disagrees with the table in the chapter.

Requires the `regex` module for `\\p{L}` / `\\p{N}`: Python's built-in `re` has no Unicode property classes.
Everything else is the standard library - the GGUF is parsed here, so no third-party GGUF reader is involved.
"""

import argparse
import hashlib
import importlib.util
import json
import pathlib
import platform
import sys

import regex
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The pre-tokenizer rule this book measures, as `tokenizer.json` spells it for `tokenizer.ggml.pre = "qwen2"`
# (Qwen2/Qwen2.5/Qwen3). Read it as seven alternatives, tried left to right at each position:
#   contraction suffixes | optional non-letter + letters | a single digit | optional space + symbols + newlines
#   | newline runs | whitespace not followed by more text | whitespace
# The engine keeps the same list as variants of one enum (`PreTokenizer::Qwen2` .. `Whitespace`); what differs
# between dialects is this string and nothing else.
QWEN2_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def bytes_to_unicode():
    """The GPT-2 byte encoder: a printable stand-in for every one of the 256 byte values."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_CHAR = bytes_to_unicode()
CHAR_TO_BYTE = {c: b for b, c in BYTE_TO_CHAR.items()}


class Reader:
    """Just enough GGUF to read the three tokenizer arrays; `scripts/gguf-inspect.py` is the full reader."""

    def __init__(self, buf, pos=0):
        self.buf, self.pos = buf, pos

    def take(self, n):
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self):
        return int.from_bytes(self.take(4), "little")

    def u64(self):
        return int.from_bytes(self.take(8), "little")

    def string(self):
        return self.take(self.u64()).decode("utf-8", "replace")

    def value(self, vtype):
        simple = {0: (1, False), 1: (1, True), 2: (2, False), 3: (2, True), 4: (4, False), 5: (4, True),
                  6: (4, None), 7: (1, False)}
        if vtype in simple:
            size, signed = simple[vtype]
            raw = self.take(size)
            if signed is None:
                import struct
                return struct.unpack("<f", raw)[0]
            return int.from_bytes(raw, "little", signed=signed)
        if vtype == 8:
            return self.string()
        if vtype == 9:  # array: element type, then count, then the elements
            etype, count = self.u32(), self.u64()
            return [self.value(etype) for _ in range(count)]
        if vtype == 10:
            return int.from_bytes(self.take(8), "little")
        if vtype == 11:
            return int.from_bytes(self.take(8), "little", signed=True)
        if vtype == 12:
            import struct
            return struct.unpack("<d", self.take(8))[0]
        raise SystemExit(f"gguf: unknown value type {vtype} at byte {self.pos}")


def read_tokenizer(path):
    """Pull the tokenizer arrays out of a GGUF, in full - the chapter quotes their counts and their first rows."""
    buf = open(path, "rb").read()
    r = Reader(buf)
    if r.take(4) != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file")
    r.u32(), r.u64()            # version, n_tensors
    n_kv = r.u64()
    want = {"tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens",
            "tokenizer.ggml.merges", "tokenizer.ggml.token_type", "tokenizer.chat_template"}
    meta = {}
    for _ in range(n_kv):
        key = r.string()
        vtype = r.u32()
        val = r.value(vtype)
        if key in want:
            meta[key] = val
    missing = {"tokenizer.ggml.tokens", "tokenizer.ggml.merges"} - set(meta)
    if missing:
        raise SystemExit(f"{path}: no tokenizer arrays ({', '.join(sorted(missing))})")
    return meta


def load_vocab(path):
    meta = read_tokenizer(path)
    tokens = meta["tokenizer.ggml.tokens"]
    vocab = {tok: i for i, tok in enumerate(tokens)}
    merges = {}
    for rank, line in enumerate(meta["tokenizer.ggml.merges"]):
        a, b = line.split(" ")
        merges[(a, b)] = rank
    types = meta.get("tokenizer.ggml.token_type", [])
    return {
        "tokens": tokens,
        "vocab": vocab,
        "merges": merges,
        "types": types,
        "model": meta.get("tokenizer.ggml.model"),
        "pre": meta.get("tokenizer.ggml.pre"),
    }


def pre_tokenize(text):
    """Cut the text where no merge may cross. One regex, seven alternatives, leftmost-longest per branch."""
    return [m.group(0) for m in regex.finditer(QWEN2_PATTERN, text)]


def bpe(piece, merges):
    """The merge loop: join the lowest-ranked adjacent pair, repeat until no pair is in the table."""
    word = list(piece)
    while True:
        best = None
        for i in range(len(word) - 1):
            rank = merges.get((word[i], word[i + 1]))
            if rank is not None and (best is None or rank < best[0]):
                best = (rank, i)
        if best is None:
            return word
        _, i = best
        word[i:i + 2] = [word[i] + word[i + 1]]


SPECIAL_TYPE_NAMES = {2: "unknown", 3: "control", 4: "user-defined", 5: "unused", 6: "byte"}


def special_tokens(tok):
    """The tokens the file declares special, by `tokenizer.ggml.token_type` - never by their spelling."""
    out = []
    for i, t in enumerate(tok["types"]):
        if t in (3, 4, 5, 6):
            out.append((tok["tokens"][i], i, SPECIAL_TYPE_NAMES[t]))
    return out


def encode(tok, text, specials=()):
    """Text to ids: the byte-level step, then the merge loop, with declared specials matched first.

    The order is the engine's (`Tokenizer::encode`): the earliest special token wins, the text before it is
    encoded normally, the special contributes its own id, and the scan resumes after it. With `specials`
    empty this is a pure BPE pass - which is how the chapter measures what "declared" is worth.
    """
    if specials:
        best = None
        for s, i in specials:
            pos = text.find(s)
            if pos >= 0 and (best is None or pos < best[0] or (pos == best[0] and len(s) > len(best[1]))):
                best = (pos, s, i)
        if best is not None:
            pos, s, i = best
            return (encode(tok, text[:pos], specials) + [i]
                    + encode(tok, text[pos + len(s):], specials))
    ids = []
    for piece in pre_tokenize(text):
        marked = "".join(BYTE_TO_CHAR[b] for b in piece.encode("utf-8"))
        for merged in bpe(marked, tok["merges"]):
            ids.append(tok["vocab"][merged])
    return ids


def decode(tok, ids):
    """Ids to bytes. Ids that name no token are kept as a placeholder rather than dropped."""
    out = bytearray()
    for i in ids:
        if i >= len(tok["tokens"]):
            out += b"?"
            continue
        for c in tok["tokens"][i]:
            out.append(CHAR_TO_BYTE.get(c, ord("?")))
    return bytes(out)


def merge_trace(tok, piece, limit=12):
    """The merge loop with its decisions kept, for the figure: the units after each round, not the string.

    The joined text never changes - only where the cuts are - so a state recorded as a string is the same six
    characters every round. The units are the thing.
    """
    marked = "".join(BYTE_TO_CHAR[b] for b in piece.encode("utf-8"))
    word = list(marked)
    trace = [{"round": 0, "pair": None, "rank": None, "units": list(word)}]
    while len(trace) <= limit:
        best = None
        for i in range(len(word) - 1):
            rank = tok["merges"].get((word[i], word[i + 1]))
            if rank is not None and (best is None or rank < best[0]):
                best = (rank, i)
        if best is None:
            break
        rank, i = best
        pair = (word[i], word[i + 1])
        word[i:i + 2] = [word[i] + word[i + 1]]
        trace.append({"round": len(trace), "pair": [pair[0], pair[1]], "rank": rank, "units": list(word)})
    return trace


def complete_utf8_prefix_len(b):
    """How many leading bytes of `b` are whole UTF-8 characters - the streaming decoder's holdback rule."""
    end, i = 0, 0
    while i < len(b):
        c = b[i]
        n = 1 if c < 0x80 else 2 if c & 0xE0 == 0xC0 else 3 if c & 0xF0 == 0xE0 else 4 if c & 0xF8 == 0xF0 else 1
        if i + n > len(b):
            break
        i += n
        end = i
    return end


def holdback(tok):
    """One character split across tokens: what a streamer may print after each id, and what it must hold.

    The holdback is only interesting when a UTF-8 sequence straddles two ids, so the fixture is chosen by
    measurement - the first candidate whose bytes end mid-character after some id.
    """
    for text in ("\U0001f9cb", "\U0001d54f", "\u0e01\u0e33", "\U0001f642"):
        ids = encode(tok, text)
        rows, acc = [], bytearray()
        for i in ids:
            acc += decode(tok, [i])
            whole = complete_utf8_prefix_len(bytes(acc))
            rows.append({"id": i, "token": tok["tokens"][i] if i < len(tok["tokens"]) else None,
                         "bytes": len(acc), "whole": whole, "held_back": len(acc) - whole})
        if any(r["held_back"] > 0 for r in rows):
            return {"text": escaped(text), "ids": ids, "rows": rows, "splits": True}
    return {"text": "none", "ids": [], "rows": [], "splits": False}


# The fixture set: the awkward strings a tokenizer fails on, in the order the chapter's table lists them.
# Every entry is (a short name, the text). Escapes are printed, never the raw control characters.
FIXTURES = [
    ("empty", ""),
    ("one space", " "),
    ("plain word", "hello"),
    ("leading space", " hello"),
    ("two spaces", "hello  world"),
    ("precomposed", "H\u00e9llo"),
    ("combining mark", "He\u0301llo"),
    ("cjk", "\u4e2d\u6587"),
    ("emoji", "\U0001f642"),
    ("newlines", "line\n\nbreak"),
    ("digits", "1234567890"),
    ("contraction", "don't"),
    ("special text", "<|im_start|>system"),
    ("tab", "a\tb"),
    ("long", "a" * 200),
]


def escaped(text):
    """An ASCII form of a fixture, so a control character is visible in a table rather than invisible."""
    return text.encode("unicode_escape").decode("ascii")


def collect(path):
    tok = load_vocab(path)
    type_hist = {}
    for t in tok["types"]:
        type_hist[str(t)] = type_hist.get(str(t), 0) + 1
    specials = [(s, i) for s, i, _ in special_tokens(tok)]
    fixtures = []
    for name, text in FIXTURES:
        raw = encode(tok, text)                                  # no normalizer, no declared specials
        nfc = encode(tok, unicodedata.normalize("NFC", text))     # the file's declared normalizer
        engine = encode(tok, unicodedata.normalize("NFC", text), specials)   # what the engine does
        back = decode(tok, engine)
        try:
            roundtrip = back.decode("utf-8")
        except UnicodeDecodeError:
            roundtrip = None
        fixtures.append({
            "name": name,
            "text": escaped(text),
            "chars": len(text),
            "bytes": len(text.encode("utf-8")),
            "ids_raw": raw,
            "ids_nfc": nfc,
            "ids": engine,
            "n_ids": len(engine),
            "decoded": escaped(roundtrip) if roundtrip is not None else None,
            "decodes_as_text": roundtrip is not None,
            "roundtrip_same": roundtrip == text,
            "nfc_same": unicodedata.normalize("NFC", text) == text,
            "nfc_changes_ids": raw != nfc,
            "specials_change_ids": nfc != engine,
        })
    return {
        "script": "scripts/tok.py",
        "machine": f"{platform.machine()}, {platform.system()} {platform.release()}",
        "compiler": f"python {platform.python_version()}, regex {regex.__version__}",
        "invocation": f"python3 scripts/tok.py ~/.cache/myfer/{pathlib.Path(path).name} --out facts/tokenizer.json",
        "gguf": {
            "file": pathlib.Path(path).name,
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "tokens": len(tok["tokens"]),
            "merges": len(tok["merges"]),
            "model": tok["model"],
            "pre": tok["pre"],
            "token_type_counts": type_hist,
            "first_tokens": tok["tokens"][:8],
            "first_merges": [f"{a} {b}" for (a, b) in list(tok["merges"])[:8]],
            "byte_tokens": sum(1 for t in tok["tokens"] if len(t) == 1 and t in CHAR_TO_BYTE),
            "special_tokens": [{"token": s, "id": i, "type": ty} for s, i, ty in special_tokens(tok)][:12],
            "n_special": len(special_tokens(tok)),
        },
        "pre_pattern": QWEN2_PATTERN,
        "fixtures": fixtures,
        "merge_trace": merge_trace(tok, " hello"),
        "holdback": holdback(tok),
        "summary": {
            "fixtures": len(fixtures),
            "roundtrip_differs": [f["name"] for f in fixtures if not f["roundtrip_same"]],
            "ids_total": sum(f["n_ids"] for f in fixtures),
            "nfc_changes": [f["name"] for f in fixtures if f["nfc_changes_ids"]],
            "engine_vs_raw": [f["name"] for f in fixtures if f["ids"] != f["ids_raw"]],
            "specials_change": [f["name"] for f in fixtures if f["specials_change_ids"]],
            "bytes_total": sum(f["bytes"] for f in fixtures),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", help="path to a .gguf file")
    ap.add_argument("--out", default=str(ROOT / "facts" / "tokenizer.json"))
    ap.add_argument("--fixtures", action="store_true")
    args = ap.parse_args()
    if not args.model:
        ap.error("a .gguf file is required")
    facts = collect(args.model)
    if args.fixtures:
        for f in facts["fixtures"]:
            print(f"{f['name']:16s} {f['text'][:30]:32s} ids={f['ids']}")
        return 0
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(facts, indent=1, ensure_ascii=True) + "\n")
    shown = out.relative_to(ROOT) if out.is_absolute() else out
    print(f"wrote {shown}: {facts['gguf']['tokens']} tokens, {facts['gguf']['merges']} merges, "
          f"{len(facts['fixtures'])} fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
