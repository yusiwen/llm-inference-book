#!/usr/bin/env python3
"""facts.py - keep the book's numbers out of the author's hands.

A page that quotes a tensor type id, a byte offset or a byte count is quoting an artifact. Typing those
numbers into prose is how a chapter ended up calling Q5_0 `type=5` when the file says 6: the sentence read
perfectly, the table was wrong, and no gate in this repository could have caught it, because nothing in the
repository disagreed.

So the numbers are generated instead. A page carries a marker and a block:

    <!-- facts: gguf-qwen2.5-0.5b.dtype-mix -->
    ... generated markdown ...
    <!-- /facts -->

and `--render` rewrites the block from `facts/<name>.json`, which is itself produced by
`scripts/gguf-inspect.py --json` from a real file. The gate (`--check`) then verifies three things:

  1. every block in the tree equals what the renderer produces (no hand edits inside a block),
  2. every marker names a facts file that exists, and every facts file is used at least once,
  3. when the artifact is present on this machine, its sha256 still matches the one recorded in the
     facts file - so "the facts are stale against the file you actually have" is a failure, not a mystery.

Usage:
    python3 scripts/facts.py --write FILE --name NAME   # derive facts/NAME.json from a model file
    python3 scripts/facts.py --render                  # rewrite every block from the facts files
    python3 scripts/facts.py --check                   # gate mode
"""

import json
import pathlib
import re
import subprocess
import sys
import hashlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
FACTS = ROOT / "facts"
SRC = ROOT / "src"

MARKER = re.compile(r"<!-- facts: (?P<name>[a-z0-9._-]+)\.(?P<block>[a-z-]+) -->")
END = "<!-- /facts -->"


def fmt(n):
    return f"{n:,}"


def render_dtype_mix(f):
    """Bytes per dtype, with the share that Chapter 2's prose argues about."""
    total = f["blob"]["bytes"]
    rows = sorted(f["dtype_bytes"].items(), key=lambda kv: -kv[1])
    out = ["| dtype | Bytes | Share of blob | Tensors in this file |", "|---|---|---|---|"]
    for name, n in rows:
        kinds = f["dtype_kinds"][name]
        count = f["dtype_count"][name]
        shown = ", ".join(f"`{k}`" for k in kinds[:8])
        if len(kinds) > 8:
            shown += f", and {len(kinds) - 8} more"
        out.append(f"| {name} | {fmt(n)} | {n / total:.1%} | {shown} ({count}) |")
    return out


def render_tensor_sample(f):
    """The same four records the chapter reads aloud, taken from the table by name."""
    by_name = {t["name"]: t for t in f["tensors"]}
    names = f.get("samples") or [t["name"] for t in f["tensors"][:4]]
    out = ["```text"]
    width = max(len(n) for n in names) + 2
    for n in names:
        t = by_name[n]
        dims = "[" + ", ".join(str(d) for d in t["dims"]) + "]"
        out.append(f"{n.ljust(width)} n_dims={len(t['dims'])}  dims={dims}  "
                   f"type={t['type_id']} ({t['type']})  offset={t['offset']}")
    out.append("```")
    return out


def render_regions(f):
    """The file's regions with real byte ranges - the table that must not be typed."""
    regions = [
        ("header", 24, 0, 23),
        ("metadata key-values", f["metadata"]["bytes"], 24, f["metadata"]["end"] - 1),
        ("tensor table", f["tensor_table"]["bytes"], f["metadata"]["end"], f["tensor_table"]["end"] - 1),
        ("alignment padding", f["blob"]["padding"], f["tensor_table"]["end"], f["blob"]["offset"] - 1),
        ("data blob", f["blob"]["bytes"], f["blob"]["offset"], f["blob"]["offset"] + f["blob"]["bytes"] - 1),
    ]
    out = ["| Region | Bytes | Decimal range |", "|---|---|---|"]
    for name, n, start, end in regions:
        span = f"{fmt(start)}-{fmt(end)}" if n else f"{fmt(start)} (empty)"
        out.append(f"| {name} | {fmt(n)} | {span} |")
    out.append(f"| **the file** | **{fmt(f['artifact']['bytes'])}** | "
               f"**0-{fmt(f['artifact']['bytes'] - 1)}** |")
    return out


def grouped_hex(flat):
    """24 bytes as four groups: magic (4), version (4), tensor count (8), metadata count (8)."""
    b = flat.split()
    return "  ".join([" ".join(b[0:4]), " ".join(b[4:8]), " ".join(b[8:16]), " ".join(b[16:24])])


def render_header(f):
    """The 24-byte header: the hex, then the same bytes decoded."""
    h, m = f["header"], f["metadata"]
    return [
        "```text",
        f"bytes 0-23   {grouped_hex(h['first_24_hex'])}",
        f"decoded      magic \"GGUF\" | version {h['version']} | tensor count {h['tensor_count']} "
        f"| metadata count {m['count']}",
        "```",
    ]


def render_bits(f):
    """The one number that settles what a quantization name means: bits per parameter actually stored."""
    params = sum(t["numel"] for t in f["tensors"])
    blob_bits = f["blob"]["bytes"] * 8 / params
    file_bits = f["artifact"]["bytes"] * 8 / params
    rows = sorted(f["dtype_bytes"].items(), key=lambda kv: -kv[1])
    per_dtype = []
    for name, n in rows:
        numel = sum(t["numel"] for t in f["tensors"] if t["type"] == name)
        per_dtype.append(f"{name} {n * 8 / numel:.2f}")
    return [
        "```text",
        f"parameters in the file        {params:,}",
        f"blob bytes per parameter      {blob_bits:.2f} bits",
        f"whole file per parameter      {file_bits:.2f} bits  (metadata included)",
        f"by dtype, bits per weight     {', '.join(per_dtype)}",
        "```",
    ]


BLOCK_LAYOUT = {
    "F32": (1, 4), "F16": (1, 2), "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22),
    "Q5_1": (32, 24), "Q8_0": (32, 34), "Q8_1": (32, 36), "Q2_K": (256, 84), "Q3_K": (256, 110),
    "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
}


def row_geometry(t):
    """(blocks per row, bytes per row, rows) for a tensor, or for a 1-D tensor its single row."""
    be, bb = BLOCK_LAYOUT[t["type"]]
    ne0 = t["dims"][0]
    ne1 = t["dims"][1] if len(t["dims"]) > 1 else 1
    blocks = ne0 // be
    return blocks, blocks * bb, ne1


STRIDE_SAMPLES = ["output.weight", "token_embd.weight", "blk.0.ffn_down.weight",
                  "blk.0.ffn_gate.weight", "blk.5.attn_v.weight", "blk.0.attn_k.bias"]


def render_strides(f):
    """Shape, stride and byte arithmetic for real tensors - the table Chapter 3 is built on."""
    by_name = {t["name"]: t for t in f["tensors"]}
    out = ["| tensor | dtype | ne0 (contiguous) | ne1 | blocks per row | bytes per row | rows | "
           "bytes = rows x row | the file says |", "|---|---|---|---|---|---|---|---|---|"]
    for name in STRIDE_SAMPLES:
        if name not in by_name:
            continue
        t = by_name[name]
        blocks, row, rows = row_geometry(t)
        ok = "same" if row * rows == t["bytes"] else f"MISMATCH ({row * rows})"
        out.append(f"| `{name}` | {t['type']} | {t['dims'][0]:,} | {rows:,} | {blocks:,} | "
                   f"{row:,} | {rows:,} | {row * rows:,} | {t['bytes']:,} ({ok}) |")
    return out


def render_block_sizes(f):
    """What one block of each dtype in this file costs, and what it implies per weight."""
    out = ["| dtype | Elements per block | Bytes per block | Bits per weight | Bytes for a 896-wide row |",
           "|---|---|---|---|---|"]
    for name in sorted(f["dtype_bytes"], key=lambda n: -f["dtype_bytes"][n]):
        if name not in BLOCK_LAYOUT:
            continue
        be, bb = BLOCK_LAYOUT[name]
        out.append(f"| {name} | {be:,} | {bb:,} | {bb * 8 / be:.2f} | "
                   f"{(896 // be) * bb:,} ({(896 // be)} blocks) |")
    return out



def render_anatomy(f):
    """The block table: fields, bytes, and what each format pays per weight."""
    out = ["| Format | Weights per block | Bytes per block | Bits per weight | Fields, in file order |",
           "|---|---|---|---|---|"]
    for fmt in sorted(f["formats"], key=lambda x: x["bits_per_weight"]):
        fields = ", ".join(f"`{x['name']}` {x['type']} ({x['bytes']} B)" for x in fmt["fields"])
        per_block = fmt["elements"] // fmt["blocks"]              # weights in one block
        block_bytes = sum(x["bytes"] for x in fmt["fields"])       # what the block costs on disk
        out.append(f"| {fmt['dtype']} | {per_block} | {block_bytes} | "
                   f"{fmt['bits_per_weight']:.2f} | {fields} |")
    return out


def render_agreement(f):
    """Two decoders, the same bytes: what the witness agreed on."""
    out = ["| Format | Blocks decoded | Weights compared | Exact matches | Max abs difference | Agreement |",
           "|---|---|---|---|---|---|"]
    for fmt in sorted(f["formats"], key=lambda x: x["dtype"]):
        out.append(f"| {fmt['dtype']} | {fmt['blocks']} | {fmt['elements']:,} | "
                   f"{fmt['exact_matches']:,} of {fmt['compared']:,} | {fmt['max_abs_diff']:.1e} | "
                   f"{fmt['agreement']} |")
    return out


def render_error(f):
    """What a format costs against the same real weights - the arm the chapter's argument rests on."""
    if not f.get("error"):
        return ["(no error measurements in this facts file)"]
    out = ["| Format | Bits per weight | Relative RMS error | Largest single error | Where it lands |",
           "|---|---|---|---|---|"]
    base = next((e for e in f["error"] if e["dtype"] == "Q4_0"), None)
    for e in sorted(f["error"], key=lambda x: -x["relative_rms"]):
        ratio = f"{e['relative_rms'] / base['relative_rms']:.2f}x of Q4_0" if base else "-"
        out.append(f"| {e['dtype']} | {_bits_for(f, e['dtype']):.2f} | {e['relative_rms']:.2%} | "
                   f"{e['max_abs_error']:.5f} | {ratio} |")
    return out


def _bits_for(f, dtype):
    for fmt in f["formats"]:
        if fmt["dtype"] == dtype:
            return fmt["bits_per_weight"]
    return float("nan")


def render_outlier(f):
    """One real block where a single weight set the step for the other 31."""
    o = f.get("outlier")
    if not o:
        return ["(no outlier scan in this facts file)"]
    w = o["worst"]
    return [
        "```text",
        f"blocks scanned             {o['blocks_scanned']:,}  (real Q4_0 blocks, from the file)",
        f"worst block                #{w['block']}",
        f"  largest weight           {w['max_abs']:.6f}",
        f"  the block's step         {w['step']:.6f}   (f16 scale; Q4_0 sets it from the largest weight)",
        f"  mean of the other 31     {w['mean_others']:.6f}",
        f"  step / mean of others    {w['step_over_mean']:.2f}   (a step wider than the values it rounds)",
        f"  largest / mean of others {w['ratio']:.1f}x",
        f"median block               step / mean of others {o['median_step_over_mean']:.2f}, "
        f"largest / mean {o['median_ratio']:.1f}x",
        "```",
    ]



BLOCKS = {
    "anatomy": render_anatomy,
    "agreement": render_agreement,
    "error": render_error,
    "outlier": render_outlier,
    "strides": render_strides,
    "block-sizes": render_block_sizes,
    "bits": render_bits,
    "dtype-mix": render_dtype_mix,
    "tensor-sample": render_tensor_sample,
    "regions": render_regions,
    "header": render_header,
}


def load(name):
    path = FACTS / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"facts/{name}.json does not exist - run `facts.py --write FILE --name {name}`")
    return json.loads(path.read_text())


def rewrite(text, name, block):
    """Replace every body for (name, block) in text; return (new_text, count)."""
    renderer = BLOCKS[block]
    f = load(name)
    body = "\n".join(renderer(f))
    # The body may be empty (a freshly added marker), so the pattern must not require a line between
    # the marker and the end tag - an earlier version did, and every new marker was reported as
    # "no <!-- /facts -->" instead of being filled in.
    pattern = re.compile(
        r"(<!-- facts: " + re.escape(name) + r"\." + re.escape(block) + r" -->)\n(.*?)<!-- /facts -->",
        re.S)
    new, n = pattern.subn(lambda m: m.group(1) + "\n" + (body + "\n" if body else "") + "<!-- /facts -->",
                          text)
    return new, n, body


def pages():
    return sorted(p for p in SRC.rglob("*.md"))


def render(write):
    changed = 0
    for p in pages():
        text = p.read_text()
        new = text
        for m in MARKER.finditer(text):
            new, n, _ = rewrite(new, m.group("name"), m.group("block"))
            if n == 0:
                raise SystemExit(f"{p.relative_to(ROOT)}: marker for {m.group('name')}.{m.group('block')} "
                                 f"has no matching `{END}`")
            changed += n
        if new != text and write:
            p.write_text(new)
    print(f"rendered {changed} block(s) across {len(pages())} page(s)")
    return changed


def check():
    findings = []
    used, seen = set(), 0
    for p in pages():
        text = p.read_text()
        for m in MARKER.finditer(text):
            name, block = m.group("name"), m.group("block")
            seen += 1
            used.add(name)
            if block not in BLOCKS:
                findings.append(f"{p.relative_to(ROOT)}: unknown block type '{block}'")
                continue
            if not (FACTS / f"{name}.json").exists():
                findings.append(f"{p.relative_to(ROOT)}: marker names facts/{name}.json, which does not exist")
                continue
            _, n, want = rewrite(text, name, block)
            if n == 0:
                findings.append(f"{p.relative_to(ROOT)}: marker for {name}.{block} has no `{END}`")
                continue
            # compare the block body only, so the prose around a block stays free
            pat = re.compile(r"<!-- facts: " + re.escape(name) + r"\." + re.escape(block) +
                             r" -->\n(.*?)<!-- /facts -->", re.S)
            m2 = pat.search(text)
            got = (m2.group(1) if m2 else "").strip("\n")
            if got != want:
                first = next((i for i, (a, b) in enumerate(zip(got.split("\n"), want.split("\n"))) if a != b), 0)
                findings.append(f"{p.relative_to(ROOT)}: block {name}.{block} differs from the facts file "
                                f"(first differing line {first + 1}):\n      page:  {got.split(chr(10))[first][:100]}\n"
                                f"      facts: {want.split(chr(10))[first][:100]}")
    for f in sorted(FACTS.glob("*.json")):
        if f.stem not in used:
            findings.append(f"facts/{f.name}: no page uses these facts (stale file?)")
        data = json.loads(f.read_text())
        art = data["artifact"]
        path = pathlib.Path(art["path"])
        if not path.exists():
            continue  # the artifact is not on this machine; the recorded hash cannot be re-derived
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != art["sha256"]:
            findings.append(f"facts/{f.name}: the artifact at {path} is no longer the one these facts "
                            f"describe (sha256 {digest[:12]} != {art['sha256'][:12]})")
    # The second implementation's verdict, recorded when it was run (it needs network and uv, so the
    # gate cannot re-run it). What the gate can check is that the record still describes these facts.
    xc = ROOT / "reference" / "gguf-crosscheck.json"
    if xc.exists():
        rep = json.loads(xc.read_text())
        target = rep.get("facts", "")
        if not (FACTS / target).exists():
            findings.append(f"reference/gguf-crosscheck.json: describes {target}, which no longer exists")
        else:
            data = json.loads((FACTS / target).read_text())
            if rep["artifact"]["sha256"] != data["artifact"]["sha256"]:
                findings.append("reference/gguf-crosscheck.json: the cross-check was run against a different "
                                f"artifact ({rep['artifact']['sha256'][:12]} vs {data['artifact']['sha256'][:12]})")
            if rep.get("verdict") != "agree":
                bad = rep.get("disagreements", [])[:3]
                findings.append(f"reference/gguf-crosscheck.json: verdict is '{rep.get('verdict')}': {bad}")

    for finding in findings:
        print(f"[E] {finding}")
    covers = ", ".join(sorted(used)) or "none"
    print(("FAIL: " + f"{len(findings)} finding(s)" if findings else "clean") +
          f" ({seen} block(s) in {len(pages())} page(s), facts: {covers})")
    return 1 if findings else 0


def write_facts(model, name):
    FACTS.mkdir(exist_ok=True)
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "gguf-inspect.py"), model, "--json"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stderr or out.stdout)
    data = json.loads(out.stdout)
    names = {t["name"] for t in data["tensors"]}
    preferred = ["output.weight", "token_embd.weight", "blk.0.ffn_down.weight", "blk.0.attn_q.weight"]
    data["samples"] = [n for n in preferred if n in names] or [t["name"] for t in data["tensors"][:4]]
    path = FACTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")
    print(f"wrote {path.relative_to(ROOT)}: {data['header']['tensor_count']} tensors, "
          f"{data['blob']['bytes']} byte blob, sha256 {data['artifact']['sha256'][:12]}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--write" in args:
        model = args[args.index("--write") + 1]
        name = args[args.index("--name") + 1] if "--name" in args else pathlib.Path(model).stem
        write_facts(model, name)
    elif "--render" in args:
        render(write=True)
    elif "--check" in args:
        sys.exit(check())
    else:
        raise SystemExit(__doc__)
