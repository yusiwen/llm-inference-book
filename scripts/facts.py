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




def render_k_anatomy(f):
    """The super-block table: what a K-quant costs and which fields it has."""
    out = ["| Format | Weights per super-block | Sub-blocks | Bytes per super-block | Bits per weight | Fields, in file order |",
           "|---|---|---|---|---|---|"]
    for dt, (elems, size, fields) in K_FORMATS.items():
        fs = ", ".join(f"`{n}` {t}" for n, t in fields)
        out.append(f"| {dt} | {elems} | {elems // 32} | {size} | {size * 8 / elems:.4f} | {fs} |")
    return out


def render_k_agreement(f):
    """Two decoders on adversarial bytes, then on real ones."""
    out = ["| Bytes | Format | Blocks | Weights compared | Exact matches | Agreement |",
           "|---|---|---|---|---|---|"]
    for r in f.get("randomised", []):
        out.append(f"| random super-block | {r['dtype']} | {r['blocks']} | {r['weights']:,} | "
                   f"{r['exact']:,} | {r['agreement']} |")
    for r in f.get("real", []):
        if r["agreement"].startswith("n/a"):
            continue
        out.append(f"| {r['file']} | {r['dtype']} | {r['blocks']} | {r['weights']:,} | "
                   f"{r['exact']:,} | {r['agreement']} |")
    return out


def render_k_packing(f):
    """The 6-bit plane, against the reference implementation, on every bit pattern."""
    p = f["packing"]
    half = p["fields_compared"] // 2
    return ["```text",
            f"planes examined            {p['planes']:,}  (12 random bytes each)",
            f"scale fields compared      {p['scale_exact']:,} of {p['fields_compared']:,} of them exact",
            f"min fields compared        {p['min_exact']:,} of {p['fields_compared']:,} of them exact",
            f"distinct scale values      {p['distinct_scale_values_seen']} of 64 exercised",
            f"verdict                    {p['agreement']}",
            "```"]


def render_k_levels(f):
    """How much of its own grid each format's data actually uses - the affine argument, measured."""
    out = ["| Format | File | Blocks | Grid groups | Levels spanned, median | p10 | p90 | Of |",
           "|---|---|---|---|---|---|---|---|"]
    levels = {"Q4_0": 16, "Q4_K": 16, "Q5_K": 32, "Q6_K": 64}
    for r in f.get("real", []):
        out.append(f"| {r['dtype']} | `{r['file']}` | {r['blocks']} | {r['groups']:,} | "
                   f"{r['levels_spanned_median']:.1f} | {r['levels_spanned_p10']:.1f} | "
                   f"{r['levels_spanned_p90']:.1f} | {levels.get(r['dtype'], 0)} |")
    return out


def render_k_encoders(f):
    out = ["| Format | Reference Python package |", "|---|---|"]
    for dt, status in f.get("encoders", {}).items():
        out.append(f"| {dt} | {status} |")
    return out




def render_layer_split(f):
    """Which layers got which type for the two tensors the mixture treats per-layer.

    The file's own tensor table is the only record of the selection policy: the metadata holds a single
    file-type integer. This renders the split as it is, next to the layers the current rule would pick.
    """
    def layers(dtype, cat):
        return sorted(int(t["name"].split(".")[1]) for t in f["tensors"]
                      if t["type"] == dtype and cat in t["name"])

    more = layers("Q6_K", "ffn_down")
    less = layers("Q4_K", "ffn_down")
    if not more:
        return ["(no per-layer split in this file)"]
    n = max(more + less) + 1
    n8 = n // 8
    rule = [i for i in range(n) if i < n8 or i >= 7 * n // 8 or (i - n8) % 3 == 2]
    same = "the same set" if rule == sorted(more) else "a DIFFERENT set"
    return [
        "```text",
        f"layers in the file     {n}",
        f"ffn_down, Q6_K         {', '.join(map(str, more))}",
        f"ffn_down, Q4_K         {', '.join(map(str, less))}",
        f"attn_v,  Q8_0          {', '.join(map(str, layers('Q8_0', 'attn_v')))}",
        f"attn_v,  Q5_0          {', '.join(map(str, layers('Q5_0', 'attn_v')))}",
        "",
        f"the current rule at the baseline (i < n/8, i >= 7n/8, or (i - n/8) % 3 == 2)",
        f"would pick             {', '.join(map(str, rule))}",
        f"that is {same}.",
        "```",
    ]



def render_dot_errors(f):
    """The decomposition: what each stage of the integer dot product costs."""
    out = ["| Pair | Rows | Activation, Q8_0 rounding | Float path, relative | Integer path, median relative | Integer path, over the norm product | Integer arithmetic alone |",
           "|---|---|---|---|---|---|---|"]
    for p in f.get("pairs", []):
        out.append(f"| {p['pair']} | {p['weights']['rows_used']} | "
                   f"{p['activation']['q8_relative_rms_error']:.3%} | "
                   f"{p['float_path_max_rel_error']:.1e} | "
                   f"{p['integer_path_median_rel_error']:.1e} | "
                   f"{p['integer_error_over_norm_product_max']:.1e} | "
                   f"{p['integer_vs_same_activations_max_rel_error']:.1e} |")
    return out


def render_dot_bounds(f):
    """Is a 16-bit block accumulator legal? The answer is a property of the format, not of the data."""
    out = ["| Weight format | Weights per block | Largest |q| | Worst-case block partial | Fits in i16 (32,767)? |",
           "|---|---|---|---|---|"]
    order = ["Q4_0", "Q5_0", "Q6_K", "Q8_0"]
    b = f.get("accumulator_bounds", {})
    for name in order:
        if name not in b:
            continue
        e = b[name]
        out.append(f"| {name} | {e['weights_per_block']} | {e['max_weight']} | "
                   f"{e['worst_case_partial']:,} | {'yes' if e['fits_i16'] else 'no'} |")
    return out


def render_dot_speed(f):
    """Three scalar kernels, the same arithmetic, median of independent runs."""
    out = ["| Path | ns per row | Weight throughput | Multiply-accumulates | Speed vs dequantize-then-dot | Run range |",
           "|---|---|---|---|---|---|"]
    label = {"dq": "dequantize to f32, then f32 fma", "i32": "integers, i32 accumulator",
             "i16": "integers, i16 block accumulator"}
    for p in f.get("paths", []):
        lo, hi = p["range_ms"]
        out.append(f"| `{p['name']}` - {label.get(p['name'], p['name'])} | {p['ns_per_row']:.1f} | "
                   f"{p['gb_per_s']:.2f} GB/s | {p['gmac_per_s']:.2f} GMAC/s | "
                   f"{p['speedup_vs_dq']:.2f}x | {lo:.2f}-{hi:.2f} ms |")
    return out


def render_dot_headroom(f):
    """The gap between what a scalar kernel reaches and what the machine can deliver."""
    paths = {p["name"]: p for p in f.get("paths", [])}
    if "i16" not in paths:
        return ["(no scalar measurement in this facts file)"]
    best = paths["i16"]["gb_per_s"]
    return ["```text",
            f"rows benchmarked            {f['rows']:,} of {f['row_len']} weights each",
            f"weight bytes per row       {f['weight_bytes_per_row']:.0f} B (28 blocks x 18 B)",
            f"best scalar throughput     {best:.2f} GB/s",
            f"measured memory bandwidth  198 GB/s (Chapter 1, same machine)",
            f"headroom                   {198 / best:.0f}x short of bandwidth-bound",
            "```"]



def render_simd_table(f):
    """Every kernel, both machines, in one table: speed, agreement, and what it needs to exist."""
    out = ["| Machine | Kernel | ns per row | Weight throughput | vs scalar | Max difference, over the norm product | Requires |",
           "|---|---|---|---|---|---|---|"]
    for m in f.get("machines", []):
        for k in m["kernels"]:
            out.append(f"| {m['machine']} | `{k['name']}` | {k['ns_per_row']:.1f} | {k['gb_per_s']:.2f} GB/s | "
                       f"{k['speedup_vs_scalar']:.2f}x | {k['max_rel_vs_scalar']:.1e} | {k['requires']} |")
    return out


def render_simd_headroom(f):
    """How far the best kernel still is from the machine's measured bandwidth - the case for Threads."""
    bw = f.get("bandwidth_gb_s", {})
    rows = []
    for m in f.get("machines", []):
        best = max(m["kernels"], key=lambda k: k["gb_per_s"])
        limit = bw.get(m["machine"])
        head = f"{limit / best['gb_per_s']:.1f}x" if limit else "-"
        rows.append((m["machine"], best["name"], best["gb_per_s"], limit, head, m["dispatch_picks"]))
    out = ["```text",
           "machine                          best kernel     GB/s   bandwidth   still short   dispatch picks"]
    for name, kern, gbs, limit, head, picks in rows:
        out.append(f"{name:32s} {kern:14s} {gbs:6.2f}  {('%.1f' % limit) if limit else '-':>8s}  "
                   f"{head:>11s}   {picks}")
    out.append("```")
    return out



def render_threads_sweep(f):
    """Rows split across threads: what the split buys, and what a spawn costs."""
    out = ["| Machine | Threads | Pool, ms | Spawn per call, ms | Weight throughput, pool | Throughput, spawn | Speedup vs 1 thread |",
           "|---|---|---|---|---|---|---|"]
    for m in f.get("machines", []):
        for r in m["thread_sweep"]:
            out.append(f"| {m['machine_key']} | {r['threads']} | {r['pool_ms']:.2f} | {r['spawn_ms']:.2f} | "
                       f"{r['pool_gbs']:.1f} GB/s | {r['spawn_gbs']:.1f} GB/s | {r['speedup']:.2f}x |")
    return out


def render_token_sweep(f):
    """The same pass with N tokens: one GEMM call against N one-token calls."""
    out = ["| Machine | Tokens | One GEMM, ms | N one-token passes, ms | Per token, ms | Ratio (N passes / 1 GEMM) |",
           "|---|---|---|---|---|---|"]
    for m in f.get("machines", []):
        for r in m["token_sweep"]:
            out.append(f"| {m['machine_key']} | {r['tokens']} | {r['gemm_ms']:.3f} | {r['gemv_ms']:.3f} | "
                       f"{r['per_token_ms']:.3f} | {r['gemv_over_gemm']:.2f}x |")
    return out


def render_ridge(f):
    """Where the roof line is: the machine's arithmetic ceiling over its bandwidth ceiling,
    against the intensity this format can offer per token."""
    out = ["| Machine | Arithmetic ceiling, GMAC/s | Streaming bandwidth | Ridge, MAC per byte | Intensity per token | "
           "Tokens where the pass stops being bandwidth-bound | Best measured gain over N passes |",
           "|---|---|---|---|---|---|---|"]
    for m in f.get("machines", []):
        last = max(m["token_sweep"], key=lambda r: r["tokens"])
        rows = m["shape"]["rows"]
        macs = rows * m["shape"]["id"] * last["tokens"]
        gmac = macs / (last["gemm_ms"] * 1e-3) / 1e9
        bw = f["streaming_gb_s"][m["machine_key"]]
        ridge = gmac / bw
        intensity = 32.0 / 18.0        # 32 weights per 18-byte block: one MAC per weight, 1.78 per byte
        n_ridge = ridge / intensity
        best = max(m["token_sweep"], key=lambda r: r["gemv_over_gemm"])
        out.append(f"| {m['machine_key']} | {gmac:.0f} | {bw:.1f} GB/s | {ridge:.2f} | {intensity:.2f} | "
                   f"{n_ridge:.2f} | {best['gemv_over_gemm']:.2f}x at {best['tokens']} tokens |")
    return out


def render_overhead(f):
    """A call that is mostly dispatch: the floor a persistent pool puts under every pass."""
    out = ["| Machine | Shape | One thread, us | Persistent pool, us | Spawn per call, us | Pool speedup | Spawn overhead, us |",
           "|---|---|---|---|---|---|---|"]
    for m in f.get("machines", []):
        o = m["overhead_us"]
        fss = m["false_sharing_ms"]
        out.append(f"| {m['machine_key']} | {o['rows']} rows, {fss['threads']} threads | {o['serial']:.1f} | "
                   f"{o['pool']:.1f} | {o['spawn']:.1f} | {o['serial'] / o['pool']:.1f}x | "
                   f"{o['spawn'] - o['pool']:.1f} |")
    return out


def render_sharing(f):
    """Contiguous row ranges against interleaved rows: same work, shared cache lines."""
    out = ["| Machine | Threads | Contiguous ranges, ms | Interleaved rows, ms | Cost of sharing a line |",
           "|---|---|---|---|---|"]
    for m in f.get("machines", []):
        s = m["false_sharing_ms"]
        out.append(f"| {m['machine_key']} | {s['threads']} | {s['contiguous']:.3f} | {s['interleaved']:.3f} | "
                   f"{s['interleaved'] / s['contiguous']:.1f}x slower |")
    return out



def render_threshold(f):
    """Where a call stops having enough work in it to be worth splitting."""
    out = ["| Machine | Rows in the pass | One thread, us | Persistent pool, us | Spawn per call, us | Pool against one thread |",
           "|---|---|---|---|---|---|"]
    for m in f.get("machines", []):
        for r in m["threshold_sweep"]:
            out.append(f"| {m['machine_key']} | {r['rows']} | {r['serial_us']:.2f} | {r['pool_us']:.2f} | "
                       f"{r['spawn_us']:.2f} | {r['serial_us'] / r['pool_us']:.2f}x |")
    return out



def render_ledger(f):
    """The precision ledger: per op, what the engine accumulates in, and what that costs against
    a double-precision oracle. The comparison column is measured, not asserted."""
    n = f["norm"]
    witness = {}
    wpath = ROOT / "facts" / "vecops-witness.json"
    if wpath.exists():
        for row in json.loads(wpath.read_text())["ops"]:
            witness[row["op"]] = row["max_rel_diff"]
    out = ["| Op | Input | Accumulate | Output | Max difference, f32 path against f64 | Comparison class |",
           "|---|---|---|---|---|---|"]
    out.append(f"| RMSNorm | f32 | f32, 8 lanes ({n['max_rel_vs_f64']['lanes_f32']:.1e}) "
               f"or f32 sequential ({n['max_rel_vs_f64']['seq_f32']:.1e}) | f32 | "
               f"{n['max_rel_vs_f64']['lanes_f32']:.1e} | bounded, 1e-6 |")
    out.append(f"| softmax | f32 | f64 sum ({f['softmax']['max_rel_f32_sum']:.1e} if f32) | f32 | "
               f"{f['softmax']['max_rel_f32_sum']:.1e} | bounded, 1e-5 |")
    out.append(f"| SiLU | f32 | f32 | f32 | {f['silu']['max_rel_vs_double']:.1e} | bounded, 1e-6 |")
    out.append(f"| SwiGLU | f32 | f32 | f32 | {f['silu']['max_rel_vs_double']:.1e} | bounded, 1e-6 |")
    out.append(f"| RoPE | f32 | f32 (sin/cos in f32) | f32 | {witness.get('RoPE, half-split, positions 1..3', 0.0):.1e} "
               f"| bounded, 1e-6 |")
    r = f["residual"]
    out.append(f"| residual add | f32 | f32 | f32 | {r['max_drift_over_max_abs']['f32']:.1e} over 24 adds "
               f"| bounded, 1e-5 over a chain |")
    out.append("")
    out.append(f"Witness: {wpath.name if wpath.exists() else 'not run'} (numpy f64 on the same bytes) - "
               f"the widest disagreement across all five ops is "
               f"{max(witness.values()) if witness else 0.0:.1e}.")
    return out


def render_failures(f):
    """The ops that do not lose a bit - they break, or they drift far enough to matter."""
    s, r, rope = f["softmax"], f["residual"], f["rope"]
    out = [
        "| Failure | What happens | Measured threshold |",
        "|---|---|---|",
        f"| softmax without max subtraction, f32 | exp overflows, sum is inf, output is inf/inf = NaN | "
        f"first NaN above logit {s['naive_breaks_above_logit']['f32']:.2f} |",
        f"| softmax without max subtraction, f16 | the same, but f16 tops out at 65,504 | "
        f"first NaN above logit {s['naive_breaks_above_logit']['f16']:.2f} |",
        f"| residual stream kept in f16 | every add rounds the whole stream to 11 bits | "
        f"{r['max_drift_over_max_abs']['f16']:.1e} of the stream's peak after {r['layers']} layers |",
        f"| residual stream kept in bf16 | 8 mantissa bits, and the drift compounds | "
        f"{r['max_drift_over_max_abs']['bf16']:.1e} of the stream's peak after {r['layers']} layers |",
        f"| RoPE paired the wrong way | the scores are a different function of distance, and every "
        f"invariant still holds | {rope['score_gap_between_pairings']:.1e} of the norm product |",
    ]
    return out


def render_cost(f):
    """What the pointwise ops cost, and what they add up to per token next to the matmul."""
    ops = {o["op"]: o for o in f["cost_ns"]["ops"]}
    out = ["| Op | Length | Median | Range | ns per element |", "|---|---|---|---|---|"]
    for o in f["cost_ns"]["ops"]:
        per = o["median_ns"] / o["len"]
        out.append(f"| `{o['op']}` | {o['len']} | {o['median_ns']:.0f} ns | "
                   f"{o['range_ns'][0]:.0f}-{o['range_ns'][1]:.0f} ns | {per:.2f} |")
    e = f.get("exp", {})
    if e:
        out.append(f"| `expf` (libm) | 1 | {e['libm_ns_per_elem']:.2f} ns | "
                   f"{e['libm_range'][0]:.2f}-{e['libm_range'][1]:.2f} ns | {e['libm_ns_per_elem']:.2f} |")
        out.append(f"| `exp` (degree-5 polynomial) | 1 | {e['polynomial_ns_per_elem']:.2f} ns | "
                   f"{e['polynomial_range'][0]:.2f}-{e['polynomial_range'][1]:.2f} ns | "
                   f"{e['polynomial_ns_per_elem']:.2f} |")
    return out


def render_per_token(f):
    """The same numbers assembled into a per-token budget for the book's model: 24 layers, 14 query
    heads, 896 wide. Arithmetic, in the open, with every input named."""
    ops = {o["op"]: o["median_ns"] for o in f["cost_ns"]["ops"]}
    layers, heads, hidden = 24, 14, 896
    rms, add, silu = ops["rms_norm(896)"], ops["add(896)"], ops["silu(896)"]
    silu_ffn = silu * (4864 / hidden)          # the FFN gate is 4,864 wide
    rope = 2 * rms                             # two rotations of one hidden-size vector, priced as a norm
    out = ["| Context | softmax per head | attention softmax, per token (24 layers x 14 heads) | "
           "everything else, per token | softmax share |",
           "|---|---|---|---|---|"]
    for ctx in (4096, 32768):
        per_head = ops[f"softmax({ctx})"]
        attn = per_head * heads * layers
        other = (2 * rms + 3 * add + silu_ffn + rope) * layers
        share = attn / (attn + other)
        out.append(f"| {ctx} | {per_head / 1e3:.1f} us | {attn / 1e6:.2f} ms | {other / 1e6:.2f} ms | "
                   f"{share * 100:.0f} per cent |")
    out.append("")
    out.append(f"Per layer: 2 RMSNorm ({2 * rms:.0f} ns), 3 residual adds ({3 * add:.0f} ns), "
               f"SiLU over 4,864 ({silu_ffn:.0f} ns), RoPE ({rope:.0f} ns, priced as two 896-wide passes), "
               f"and {heads} softmax rows of the context length.")
    return out



def render_witness(f):
    """The second implementation's verdict, op by op."""
    out = ["| Op | Max relative difference, C f32 path against numpy f64 | Note |",
           "|---|---|---|"]
    for o in f["ops"]:
        out.append(f"| {o['op']} | {o['max_rel_diff']:.1e} | {o['note']} |")
    out.append("")
    out.append(f"Verdict: **{f['verdict']}**.")
    return out



def render_model_meta(f):
    """The handful of metadata keys a kernel has to read: shapes, epsilon, rope base."""
    keys = f["metadata"]["keys"]
    want = ["general.architecture", "general.name", "qwen2.block_count", "qwen2.context_length",
            "qwen2.embedding_length", "qwen2.feed_forward_length", "qwen2.attention.head_count",
            "qwen2.attention.head_count_kv", "qwen2.rope.freq_base",
            "qwen2.attention.layer_norm_rms_epsilon", "general.file_type"]
    out = ["| Key | Type | Value |", "|---|---|---|"]
    for k in want:
        if k not in keys:
            continue
        v = keys[k]["value"]
        shown = f"{v!r}" if k == "qwen2.attention.layer_norm_rms_epsilon" else v
        out.append(f"| `{k}` | {keys[k]['type']} | {shown} |")
    return out


K_FORMATS = {
    "Q4_K": (256, 144, [("d", "f16"), ("dmin", "f16"), ("scales", "u8[12], 6-bit packed"),
                        ("qs", "u8[128]")]),
    "Q5_K": (256, 176, [("d", "f16"), ("dmin", "f16"), ("scales", "u8[12], 6-bit packed"),
                        ("qh", "u8[32]"), ("qs", "u8[128]")]),
    "Q6_K": (256, 210, [("ql", "u8[128]"), ("qh", "u8[64]"), ("scales", "i8[16]"), ("d", "f16")]),
}

BLOCKS = {
    "anatomy": render_anatomy,
    "agreement": render_agreement,
    "error": render_error,
    "outlier": render_outlier,
    "layer-split": render_layer_split,
    "threads-sweep": render_threads_sweep,
    "token-sweep": render_token_sweep,
    "ridge": render_ridge,
    "overhead": render_overhead,
    "sharing": render_sharing,
    "threshold": render_threshold,
    "ledger": render_ledger,
    "witness": render_witness,
    "failures": render_failures,
    "vecops-cost": render_cost,
    "per-token": render_per_token,
    "simd-table": render_simd_table,
    "simd-headroom": render_simd_headroom,
    "dot-errors": render_dot_errors,
    "dot-bounds": render_dot_bounds,
    "dot-speed": render_dot_speed,
    "dot-headroom": render_dot_headroom,
    "k-anatomy": render_k_anatomy,
    "k-agreement": render_k_agreement,
    "k-packing": render_k_packing,
    "k-levels": render_k_levels,
    "k-encoders": render_k_encoders,
    "strides": render_strides,
    "block-sizes": render_block_sizes,
    "bits": render_bits,
    "dtype-mix": render_dtype_mix,
    "tensor-sample": render_tensor_sample,
    "regions": render_regions,
    "header": render_header,
    "model-meta": render_model_meta,
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
    # A markdown table row containing "||" is a cell boundary that ate a code span: mdBook splits on every
    # pipe, so `||w||*||a||` inside a table silently destroys the table. Caught by eye once; caught here now.
    for page in sorted((ROOT / "src").rglob("*.md")):
        for n, line in enumerate(page.read_text().split("\n"), 1):
            if line.lstrip().startswith("|") and "||" in line:
                findings.append(f"{page.relative_to(ROOT)}:{n}: a table row contains \"||\" - the markdown "
                                f"parser will split the cell; spell the norm product as words")
    for f in sorted(FACTS.glob("*.json")):
        if f.stem not in used:
            findings.append(f"facts/{f.name}: no page uses these facts (stale file?)")
        data = json.loads(f.read_text())
        art = data.get("artifact")
        if not isinstance(art, dict):
            # A measurement artifact (a timing run) describes a machine, not a file: there is no bytes hash
            # to re-derive, and the run range inside it is what a reader should look at instead. Some of
            # them name the instrument in a string instead of carrying a path, which is the same case.
            continue
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
