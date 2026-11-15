#!/usr/bin/env python3
"""tok-check.py - the witness for `scripts/tok.py`: the same strings, the reference implementation, one verdict.

The book's rule is that a number produced by an instrument needs a second implementation before it is printed.
For a tokenizer the second implementation has to be the *reference* one, not a second reading of the same
notes: a pre-tokenizer regex misread in two places by the same person agrees with itself and disagrees with
every real model. So this script loads the model family's own `tokenizer.json` with HuggingFace `tokenizers`,
encodes the same fixture set, and fails loudly on the first id that differs.

What it needs, and where it comes from:

  * `tokenizer.json` for the model the GGUF was made from (`Qwen/Qwen2.5-0.5B-Instruct@main`, 7,031,645 bytes,
    sha256 c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539). It is **not** committed: it is a
    7 MB third-party artifact, and the ids it produced are, in `facts/tokenizer.json`.

Usage:
    python3 scripts/tok-check.py TOKENIZER_JSON [--facts facts/tokenizer.json] [--out facts/tokenizer-witness.json]

The output is the witness half of the chapter's evidence: per fixture, the reference ids, my ids, and whether
they agree - plus the reference's own view of its configuration (vocab size, merge count, normalizer, the
number of added tokens), which is what the tables compare the GGUF's numbers against.
"""

import argparse
import hashlib
import json
import pathlib
import sys

import tokenizers

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer_json")
    ap.add_argument("--facts", default=str(ROOT / "facts" / "tokenizer.json"))
    ap.add_argument("--out", default=str(ROOT / "facts" / "tokenizer-witness.json"))
    args = ap.parse_args()

    raw = open(args.tokenizer_json, "rb").read()
    ref = tokenizers.Tokenizer.from_file(args.tokenizer_json)
    ref.no_padding()
    ref.no_truncation()

    facts = json.loads(pathlib.Path(args.facts).read_text())
    specs = ref.to_str()  # round-trips the configuration, including normalizer and pre_tokenizer
    rows, mismatches = [], []
    for f in facts["fixtures"]:
        text = f["text"].encode("ascii").decode("unicode_escape")
        got = ref.encode(text, add_special_tokens=False).ids
        ok = got == f["ids"]
        if not ok:
            mismatches.append({"fixture": f["name"], "reference": got, "instrument": f["ids"]})
        rows.append({"name": f["name"], "reference_ids": got, "instrument_ids": f["ids"], "match": ok,
                     "n_reference": len(got)})

    witness = {
        "script": "scripts/tok-check.py",
        "witness": {
            "tool": f"tokenizers {tokenizers.__version__}",
            "file": pathlib.Path(args.tokenizer_json).name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "vocab": len(json.loads(raw)["model"]["vocab"]),
            "merges": len(json.loads(raw)["model"]["merges"]),
            "added_tokens": len(json.loads(raw).get("added_tokens", [])),
            "normalizer": json.loads(raw).get("normalizer", {}).get("type"),
            "pre_tokenizer": json.loads(raw).get("pre_tokenizer", {}).get("pretokenizers", [{}])[0].get("type"),
        },
        "fixtures": rows,
        "answer": {
            "compared": len(rows),
            "agree": sum(1 for r in rows if r["match"]),
            "mismatches": mismatches,
        },
    }
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(witness, indent=1, ensure_ascii=True) + "\n")
    print(f"wrote {out}: {witness['answer']['agree']}/{witness['answer']['compared']} fixtures agree with "
          f"tokenizers {tokenizers.__version__}")
    if mismatches:
        for m in mismatches:
            print(f"[E] {m['fixture']}: reference {m['reference']} != instrument {m['instrument']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
