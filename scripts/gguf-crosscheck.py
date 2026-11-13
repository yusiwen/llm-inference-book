#!/usr/bin/env python3
"""gguf-crosscheck.py - read the same file with a second, independent implementation.

Every number in Chapter 2 comes from `scripts/gguf-inspect.py`, which is this book's own parser. A parser
checking itself proves nothing: the failure this script exists for is the one where the author *believes*
type id 6 is Q5_0 and both the code and the prose agree with the belief. Only a second implementation,
written by someone else, can contradict that.

The witness is `gguf-py`, the reference Python implementation that ships with the ggml project (and that
llama.cpp's own tooling uses). It is installed on demand, not vendored:

    uv run --with gguf python scripts/gguf-crosscheck.py \
        --facts facts/gguf-qwen2.5-0.5b.json --model ~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf

The comparison is written to `reference/gguf-crosscheck.json` and committed, so the evidence travels with
the book: the gate does not need network access, it checks that the recorded verdict is `agree` and that
the artifact's sha256 still matches the facts file.

Two honest notes about what a cross-check like this can and cannot do:

  * It compares *structure*: counts, names, dtypes, shapes, offsets, sizes. Two implementations agreeing on
    the container says nothing about whether a kernel reads the bytes correctly - that is Chapter 27's job.
  * gguf-py reports three synthetic fields the file does not contain (`GGUF.version`, `GGUF.tensor_count`,
    `GGUF.kv_count`), so its field count is the metadata count plus three. A cross-check that does not know
    this reports a false disagreement, which is exactly the trap this script warns about in its own notes.
"""

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "reference" / "gguf-crosscheck.json"

SYNTHETIC = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}


def field_int(reader, key, default=None):
    """gguf-py returns a scalar for a single-valued field and a sequence for arrays; normalise to int."""
    if key not in reader.fields:
        return default
    value = reader.fields[key].contents()
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    try:
        import gguf  # noqa: F401  (installed on demand; the import error is the instruction to install)
    except ImportError:
        sys.exit("gguf-py is not importable. Run this script as:\n"
                 "  uv run --with gguf python scripts/gguf-crosscheck.py --facts ... --model ...")

    facts = json.loads(pathlib.Path(args.facts).read_text())
    model = pathlib.Path(args.model)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != facts["artifact"]["sha256"]:
        sys.exit(f"this model is not the one the facts describe (sha256 {digest[:12]} != "
                 f"{facts['artifact']['sha256'][:12]})")

    reader = gguf.GGUFReader(str(model))
    agree, disagree, notes = [], [], []

    def compare(what, ours, theirs):
        (agree if ours == theirs else disagree).append({"field": what, "book": ours, "gguf_py": theirs})

    compare("file bytes", facts["artifact"]["bytes"], model.stat().st_size)
    compare("version", facts["header"]["version"], field_int(reader, "GGUF.version"))
    compare("tensor count", facts["header"]["tensor_count"], len(reader.tensors))
    theirs_fields = set(reader.fields)
    compare("metadata count (file's own)",
            facts["metadata"]["count"], len(theirs_fields - SYNTHETIC))
    notes.append(f"gguf-py reports {len(theirs_fields)} fields against the file's "
                 f"{facts['metadata']['count']}: the difference is {sorted(SYNTHETIC)}, which it synthesises "
                 f"from the header. Subtracting them is the comparison, not a fudge.")

    compare("alignment", facts["alignment"], field_int(reader, "general.alignment", 32))
    compare("data blob offset", facts["blob"]["offset"], int(reader.data_offset))
    compare("data blob bytes", facts["blob"]["bytes"],
            model.stat().st_size - int(reader.data_offset))

    # The block table: elements and bytes per block, for every dtype this file uses. It is the table that
    # makes a tensor's byte length computable, and the one a reader is most likely to mistype (Q6_K is 210,
    # not 208; Q4_K is 144, not 128). Compare it against the reference implementation's constants.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from facts import BLOCK_LAYOUT  # noqa: E402

    for name in sorted({t["type"] for t in facts["tensors"]}):
        enum = getattr(gguf.GGMLQuantizationType, name, None)
        if enum is None:
            notes.append(f"{name}: absent from gguf-py's GGMLQuantizationType enum, so not compared")
            continue
        theirs = tuple(int(x) for x in gguf.GGML_QUANT_SIZES[enum])
        compare(f"block layout {name} (elements, bytes)", BLOCK_LAYOUT.get(name), theirs)

    ours = {t["name"]: t for t in facts["tensors"]}
    theirs = {t.name: t for t in reader.tensors}
    compare("tensor names", sorted(ours), sorted(theirs))

    for name in sorted(set(ours) & set(theirs)):
        a, b = ours[name], theirs[name]
        compare(f"{name}: dtype", a["type"], b.tensor_type.name)
        compare(f"{name}: dims", list(a["dims"]), [int(d) for d in b.shape])
        compare(f"{name}: elements", a["numel"], int(b.n_elements))
        compare(f"{name}: bytes", a["bytes"], int(b.n_bytes))
        compare(f"{name}: absolute offset", a["abs"], int(b.data_offset))

    report = {
        "artifact": {"path": str(model), "bytes": int(model.stat().st_size), "sha256": digest},
        "facts": str(pathlib.Path(args.facts).name),
        "witness": {
            "implementation": "gguf-py (the ggml project's reference Python reader)",
            "invocation": "uv run --with gguf python scripts/gguf-crosscheck.py "
                          f"--facts {pathlib.Path(args.facts).name} --model {model.name}",
        },
        "compared": len(agree) + len(disagree),
        "agreements": len(agree),
        "disagreements": disagree,
        "notes": notes,
        "verdict": "agree" if not disagree else "disagree",
    }
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"{report['agreements']}/{report['compared']} fields agree; verdict {report['verdict']}")
    for d in disagree[:10]:
        print(f"  [E] {d['field']}: book {d['book']!r} vs gguf-py {d['gguf_py']!r}")
    print(f"wrote {pathlib.Path(args.out).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
