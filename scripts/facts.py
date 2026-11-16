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
    # A table, not a hand-aligned column: the fields are columns and the per-dtype rows are a list, so a
    # reader reads down one column instead of counting spaces to find which number belongs to which word.
    out = ["| Quantity | Value | Note |", "|---|---|---|",
           f"| Parameters in the file | {params:,} | every tensor's element count, summed |",
           f"| Blob bytes per parameter | {blob_bits:.2f} bits | the tensor data only |",
           f"| Whole file per parameter | {file_bits:.2f} bits | metadata included |",
           "", "| Weight dtype | Bits per weight | Bytes in the file |", "|---|---|---|"]
    for name, n in rows:
        numel = sum(t["numel"] for t in f["tensors"] if t["type"] == name)
        out.append(f"| {name} | {n * 8 / numel:.2f} | {n:,} |")
    return out


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
    return ["| Check | Result | What it rules out |", "|---|---|---|",
            f"| Planes examined | {p['planes']:,}, 12 random bytes each | a real file only exercises the values "
            f"its own quantizer happened to choose |",
            f"| Scale fields | {p['scale_exact']:,} of {p['fields_compared']:,} exact | a shift or mask "
            f"error in the low six bits |",
            f"| Min fields | {p['min_exact']:,} of {p['fields_compared']:,} exact | the same error in the "
            f"high two bits, which live in a different byte |",
            f"| Distinct scale values | {p['distinct_scale_values_seen']} of 64 exercised | a test that "
            f"never sets the top of the field |",
            f"| Verdict | {p['agreement']} | - |"]


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
    same = "the same set" if rule == sorted(more) else "a **different** set"
    return [
        "| Tensor | Weight dtype | Layers | Layer indices |", "|---|---|---|---|",
        f"| `ffn_down` | Q6_K | {len(more)} | {', '.join(map(str, more))} |",
        f"| `ffn_down` | Q4_K | {len(less)} | {', '.join(map(str, less))} |",
        f"| `attn_v` | Q8_0 | {len(layers('Q8_0', 'attn_v'))} | "
        f"{', '.join(map(str, layers('Q8_0', 'attn_v')))} |",
        f"| `attn_v` | Q5_0 | {len(layers('Q5_0', 'attn_v'))} | "
        f"{', '.join(map(str, layers('Q5_0', 'attn_v')))} |",
        "",
        f"**{n} layers in the file.** The rule the engine implements at the book's baseline - `i < n/8`, "
        f"`i >= 7n/8`, or `(i - n/8) % 3 == 2` - would pick {', '.join(map(str, rule))}: {same}.",
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
    # The header cell once read `Largest |q|`. Two unescaped pipes made the header 7 cells against a
    # 5-cell delimiter row, so GFM refused the table and shipped the whole block as a paragraph - the
    # defect a reader found before any check did. Name it in words; a cell is not a maths block.
    out = ["| Weight format | Weights per block | Largest absolute activation | Worst-case block partial "
           "| Fits in i16 (32,767)? |",
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
    dq = paths.get("dq", {}).get("gb_per_s", 0.0)
    # A table, not an aligned monospace block: the columns are structure, and the four numbers below are
    # the ones Figure 6.3 draws as bars. The figure shows the shape; this keeps the values citable and
    # carries the provenance line under it.
    return ["| Quantity | Value | Where it comes from |",
            "|---|---|---|",
            f"| Rows benchmarked | {f['rows']:,} rows of {f['row_len']} weights | this artifact |",
            f"| Weight bytes per row | {f['weight_bytes_per_row']:.0f} B (28 blocks x 18 B) | the format, not a measurement |",
            f"| Best scalar throughput | {best:.2f} GB/s | the `i16` kernel below |",
            f"| Dequantize-then-dot throughput | {dq:.2f} GB/s | the `dq` kernel below |",
            "| Measured memory bandwidth | 198 GB/s | Chapter 1, same machine |",
            f"| Headroom | {198 / best:.0f}x short of bandwidth-bound | the two rows above |"]



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


# ---------------------------------------------------------------------------------------------------
# Provenance: where a table's numbers came from.
#
# The review that started this asked one question about every measured table: who produced it, and how
# would a reader get the source. A facts block alone answers neither - it shows the numbers and hides the
# instrument. So every page that renders a facts block also renders one line naming the script, the
# command, the machine, the number of runs and the statistic, and a gate step fails when that line is
# missing. Nothing here retypes a number: the command and the machine are read out of the artifact that
# already carries them, and only the script path and the statistic are declared, once, below.

INSTRUMENTS = {
    "alloc": ("scripts/alloc.py",
              "runs the engine with its per-node trace and re-implements `alloc_graph`'s liveness rule over the "
              "graph it executed (build order, consumers extend a buffer's life, outputs and inputs pinned to the "
              "end, views extending their parent); charges the transient nodes to a first-fit arena and the "
              "`kvcache_*` nodes to the persistent region they view; measures the process's peak RSS with "
              "`/usr/bin/time -l`",
              "bytes: the sum over every buffer, the first-fit arena high-water mark, the live-set floor at the "
              "busiest step, and the class-rounded total; the RSS is the OS's own number for the same run",
              ),
    "dispatch": ("scripts/dispatch.py",
                 "runs the engine's CLI three ways with an impossible or unavailable `--backend` value and records "
                 "each refusal verbatim, runs the engine's own op-support contract tests, and reads the backend "
                 "assignment out of the per-node trace of a real run; the registry ids, names and priorities are "
                 "parsed from the pinned source, and the op-by-backend table from `docs/SUPPORT-MATRIX.md`",
                 "counts of nodes per backend and of contiguous same-backend runs, from one graph per phase; the "
                 "refusals are exit codes and messages, compared as values, not summarised",
                 ),
    "fusion": ("scripts/fuse.py",
               "runs the engine's CLI with its per-node trace on (the fused graph it actually executed), reads "
               "the op vocabulary out of the pinned source, runs the engine's own fusion tests, and compiles "
               "and runs scripts/fuse.c, which measures the same 3-to-1 rewrite on two activations at four "
               "row counts; scripts/fuse-check.py recomputes the C's sampled outputs in f64",
               "node counts and the set of executed node ids from one trace; timings are the median of nine "
               "runs of the same kernel; output equality is compared byte for byte, and the numpy witness "
               "agrees within float32 rounding",
               ),
    "graph": ("scripts/graph.py",
              "runs the engine's CLI with its per-node trace and its rebuild log on, then reads the node list, "
              "the shapes, the edges and the per-node statistics out of the trace the visualizer consumes",
              "a structural comparison of two node sequences (equal or not), and node counts read one run at a "
              "time; nothing here is averaged",
              ),
    "sampler": ("scripts/smp.py",
                "runs the engine's CLI on the model, keeps the logits it dumps, and replays the deterministic "
                "part of the sampler pipeline (penalties, top-k, top-p, temperature) in Python so the engine's "
                "own token stream can be reproduced from the engine's own logits",
                "byte equality between the replay and the engine's text; the funnel, mass and entropy figures "
                "are computed from those logits, one step at a time, never averaged",
                ),
    "template": ("scripts/tmpl.py",
                 "renders the model's own `tokenizer.chat_template` (read out of the GGUF) with Jinja2, and "
                 "compares every case byte for byte with the reference renderer's committed output "
                 "(HuggingFace `transformers` 5.17.0 `apply_chat_template`); the token counts come from the "
                 "Chapter 11 tokenizer on the same bytes",
                 "byte equality on every case; a byte or token count that differs is a failure, not a statistic",
                 ),
    "tokenizer": ("scripts/tok.py",
                  "encodes a fixture set of awkward strings from the vocabulary and merges inside the GGUF, "
                  "and decodes the ids back; the ids are compared one by one against the reference "
                  "tokenizer implementation (scripts/tok-check.py, HuggingFace `tokenizers` on the model "
                  "family's own `tokenizer.json`)",
                  "exact id equality on every fixture string; nothing here is averaged"),
    "tokenizer-witness": ("scripts/tok-check.py",
                          "runs the model family's reference tokenizer on the same fixtures and compares "
                          "ids with the book's instrument",
                          "15 of 15 fixtures agree; a mismatch is a failure, not a statistic"),

    # facts stem -> (script, what it measures, statistic)
    "gguf-qwen2.5-0.5b": ("scripts/gguf-inspect.py",
                          "parses the file: header, every metadata key, every tensor record, and the byte "
                          "range each tensor occupies",
                          "read once from the file; the counts are structure, not samples"),
    "gguf-qwen2.5-0.5b-q4_0": ("scripts/gguf-inspect.py",
                               "parses the same model at Q4_0",
                               "read once from the file"),
    "dequant-check": ("scripts/dequant-check.py",
                      "decodes each block format back to f32 and compares it against a second decoder "
                      "(gguf-py) on the same bytes",
                      "max relative error per format; the bit-exact counts are exhaustive over the "
                      "blocks sampled"),
    "kquant-check": ("scripts/kquant-check.py",
                     "decodes K-quant super-blocks from adversarial random bytes, from real weights, and "
                     "from the format's own packing",
                     "bit-exact comparisons, exhaustive over the blocks sampled"),
    "dotcheck": ("scripts/dotcheck.py",
                 "decomposes the quantized dot product's error into activation, integer and scale parts, "
                 "and bounds each format's accumulator",
                 "max relative error against f64; the bounds are arithmetic, the worst partial is "
                 "measured"),
    "dotbench": ("scripts/dotbench.c",
                 "times three scalar kernels on 8,192 real rows of the book's file",
                 "median of the runs, with the full range beside it"),
    "dotsimd": ("scripts/dotsimd.c",
                "times four SIMD kernels and the scalar reference, one source file compiled for ARM "
                "and for x86",
                "median of the runs per machine"),
    "threadbench": ("scripts/threadbench.c",
                    "sweeps thread count, batch size, spawn-per-call against a pool, and two write "
                    "layouts on one matrix",
                    "median of the runs per machine, with the range where the runs disagree"),
    "vecops": ("scripts/vecops.c",
               "computes each vector op the way an engine writes it, reports the error against an f64 "
               "oracle, and times the op",
               "median of three runs for timing; the errors are deterministic and identical across runs"),
    "engine-measure": ("scripts/engine-measure.py",
                  "runs the engine's own measurement tests (`cargo test --release measure`) and reads "
                  "the KV geometry and footprint groups out of them, then recomputes every number from "
                  "`KvFormat::row_elems` and refuses to write the artifact when the two disagree",
                  "one run of the engine's suite; the derived column must equal the measured one, "
                  "format by format and context by context"),
    "attn": ("scripts/attn.c",
             "runs one layer of real attention on the model's own weights: scores, one decode step at four "
             "prefix lengths, the softmax alone, and rebuilding the cache against reading it",
             "median of 5 runs of 3 repetitions; the score scan is exhaustive over every head and position"),
    "vecops-witness": ("scripts/vecops-check.py",
                       "recomputes every op in numpy f64 from the C program's own fixture bytes",
                       "max relative difference, one pass"),
}


def render_provenance(f, name=None):
    """One italic line: which program, which command, which machine, how many runs, which statistic."""
    script, what, statistic = INSTRUMENTS.get(name, ("(unknown)", "", ""))
    explicit = f.get("provenance") if isinstance(f.get("provenance"), dict) else {}

    command = explicit.get("command") or f.get("invocation") or ""
    if isinstance(command, list):
        command = "; ".join(command)
    machine = explicit.get("machine") or f.get("machine") or ""
    machines = f.get("machines")
    if not machine and isinstance(machines, dict):
        machine = " and ".join(machines.keys())
    if not machine and isinstance(machines, list):
        names = [m.get("machine") for m in machines if isinstance(m, dict) and m.get("machine")]
        machine = " and ".join(dict.fromkeys(names))
    compiler = explicit.get("compiler") or f.get("compiler") or ""
    if machine and compiler:
        machine = f"{machine}, {compiler}"
    runs = explicit.get("runs") or f.get("runs") or f.get("reps")
    statistic = explicit.get("statistic") or statistic
    artifact = explicit.get("artifact")
    if not artifact:
        art = f.get("artifact")
        artifact = art.get("path") if isinstance(art, dict) else (art if isinstance(art, str) else "")
    # Some artifacts name the instrument instead of a file ("vecops"), and some name the model file the
    # numbers were read from. Neither is where the numbers live: they live in facts/<name>.json, so use
    # that whenever the recorded value is not a path.
    note = ""
    if artifact.endswith(".gguf"):
        note = f" Read from `{pathlib.Path(artifact).name}`; its sha256 is recorded in `facts/{name}.json`."
        artifact = ""
    elif not artifact or "/" not in artifact:
        artifact = f"facts/{name}.json"

    bits = [f"`{script}`"]
    if what:
        bits.append(what)
    if machine:
        bits.append(f"on **{machine}**")
    if runs:
        bits.append(f"{runs} runs")
    if statistic:
        bits.append(statistic)
    line = "; ".join(bits) + "."
    # Strip the home directory: the command is meant to be reproducible, not to publish a path that only
    # exists on one machine. Match the shape of a home path rather than this machine's own home - with
    # Path.home() the rendered page differed between the author's laptop and CI, and the gate that compares
    # them caught it (which is the whole reason the page is generated rather than typed).
    command = re.sub(r"(?:/Users/|/home/)[^/\s:;]+/", "~/", command)
    if command and len(command) <= 90:
        line += f" Command: `{command}`."
    elif command:
        # A long command turns the prose into a wall of flags. The appendix page carries the full one,
        # anchored by the facts file's own name.
        line += f" Command: see [the provenance index](../appendix/provenance.md#{name})."
    if artifact:
        line += f" Raw numbers: `{artifact}`."
    line += note
    return [f"*{line}*"]

def render_attention_scores(f):
    """How large the scores actually are, on the model's own weights."""
    sc = f["scores"]
    return ["| Quantity | Value |", "|---|---|",
            f"| (head, cached position) pairs | {sc['n']:,} - 14 heads over a {f.get('geometry', {}).get('n_layer', 24)}-layer "
            f"model's first layer, 1,024 positions |",
            f"| Mean absolute score | {sc['abs_mean']:.3f} |",
            f"| Root mean square | {sc['abs_rms']:.3f} |",
            f"| Largest absolute score | {sc['abs_max']:.2f} |",
            f"| Pairs above the f16 exp bound (11.09) | {sc['over_f16_exp']} |",
            f"| Pairs above the f32 exp bound (88.72) | {sc['over_f32_exp']} |"]


def render_decode_step(f):
    """One decode step against the prefix length: linear in the prefix, quadratic over a whole run."""
    ds = f["decode_step"]
    sm = f.get("softmax_only_s") or 0.0
    rows = ["| Cached prefix | One decode step | Per 100 tokens of prefix | Softmax share of the step |",
            "|---|---|---|---|",
            "| tokens | microseconds | nanoseconds | per cent |"]
    keys = sorted(ds, key=lambda k: int(k))
    prev = None
    for k in keys:
        us = ds[k] * 1e6
        per100 = us / (int(k) / 100.0)
        share = 100.0 * sm / ds[k] if ds[k] else 0.0
        ratio = "" if prev is None else f" ({us / prev:.1f}x the previous row)"
        rows.append(f"| {int(k):,} | {us:.1f}{ratio} | {per100:.0f} | {share:.0f} |")
        prev = us
    return rows



def render_kv_cache_size(f):
    """What the engine allocates for the KV cache, per storage format.

    The allocation is `row_elems` f32 words per cell per region, times two regions (K and V), so
    **f16 and f32 allocate the same region** - f16 halves the *payload* inside those words, which is
    a bandwidth win, not a footprint win. Only q8_0 shrinks the cell (34 words per 128 elements).
    Sizes are the engine's own `kv_footprint` measurement; the arithmetic beside them is `row_elems`
    recomputed in Python by `scripts/kv-engine.py`, which fails if the two disagree.
    """
    d = f["derived"]
    geo = {row["format"]: row for row in f["groups"]["kv_geometry"]["formats"]}
    layers, weights = f["layers"], f["weights_bytes"]
    ctxs = f["contexts"]

    def mib(n):
        return f"{n / 2**20:,.1f} MiB"

    def kib(n):
        return f"{n / 1024:,.1f} KiB"

    read = {fmt: geo[fmt]["payload_bytes"] * 2 for fmt in ("f32", "f16", "q8_0")}

    def region(fmt, ctx):
        return d[fmt]["region_bytes"][str(ctx)]

    rows = ["| Quantity | f32 | f16 | q8_0 |", "|---|---|---|---|",
            f"| f32 words one cell occupies | {d['f32']['row_elems']} | {d['f16']['row_elems']} "
            f"| {d['q8_0']['row_elems']} |",
            f"| Bytes per token per layer | {d['f32']['per_token_per_layer']:,} B "
            f"| {d['f16']['per_token_per_layer']:,} B | {d['q8_0']['per_token_per_layer']:,} B |",
            f"| Bytes per token, all {layers} layers | {kib(d['f32']['per_token_all_layers'])} "
            f"| {kib(d['f16']['per_token_all_layers'])} | {kib(d['q8_0']['per_token_all_layers'])} |"]
    for ctx in ctxs:
        label = f"{ctx // 1024}k" if ctx % 1024 == 0 else str(ctx)
        rows.append(f"| Region at {label} tokens | {mib(region('f32', ctx))} "
                    f"| {mib(region('f16', ctx))} | {mib(region('q8_0', ctx))} |")
    rows.append(f"| Share of the weights at 8k | {region('f32', 8192) / weights * 100:.1f} per cent "
                f"| {region('f16', 8192) / weights * 100:.1f} per cent "
                f"| {region('q8_0', 8192) / weights * 100:.1f} per cent |")
    rows.append(f"| Read per decode step at 4k | {mib(read['f32'] * 4096 * layers)} "
                f"| {mib(read['f16'] * 4096 * layers)} | {mib(read['q8_0'] * 4096 * layers)} |")
    rows.append("")
    rows.append("The first three rows are `row_elems x 4 x 2` - f32 words per cell, bytes per word, and the "
                "two regions K and V. The region rows are the engine's own `kv_footprint` measurement, which "
                "`scripts/kv-engine.py` recomputes from `row_elems` and refuses to write when the two "
                "disagree. The share divides by the 468.6 MiB file. The last row is the payload actually "
                "read per decode step, not the words reserved for it.")
    return rows


def render_rebuild(f):
    """Rebuilding K and V against reading them: the cache's whole justification, timed."""
    r = f["rebuild"]
    step = f["decode_step"][max(f["decode_step"], key=lambda k: int(k))]
    rec, rd = r["recompute_kv_s"], r["read_kv_s"]
    return ["| Operation | Time | Bytes | What it replaces |", "|---|---|---|---|",
            f"| Recompute K and V for 1,024 positions | {rec * 1e3:.1f} ms | {2 * 1024 * 128 * 4 / 2**20:.2f} MiB written | "
            f"what an engine without a cache does on every token |",
            f"| Read the same K and V | {rd * 1e6:.0f} us | {2 * 1024 * 128 * 4 / 2**20:.2f} MiB read | "
            f"{rec / rd:,.0f}x cheaper than recomputing |",
            f"| Attend over the cached prefix | {step * 1e6:.0f} us | the same bytes | "
            f"{rec / step:,.0f}x cheaper than recomputing |"]


def render_mask_effect(f):
    """What each mask bug costs, in the output, on real weights."""
    m = f["mask_effect"]
    return ["| Mask | Largest output change | Relative | Caught by a 1e-2 tolerance test? |",
            "|---|---|---|---|",
            f"| Off by one (`j &lt; i` instead of `j &lt;= i`) | {m['off_by_one_max_abs']:.6f} | "
            f"{m['off_by_one_relative'] * 100:.2f} per cent | no |",
            f"| Missing entirely (no causal mask) | {m['unmasked_max_abs']:.6f} | "
            f"{m['unmasked_relative'] * 100:.2f} per cent | yes |",
            "",
            f"The query's own position carries **{m['self_weight'] * 100:.2f} per cent** of the attention mass "
            f"at a 512-token prefix, which is the size of what the off-by-one throws away: the largest output "
            f"element is {m['max_abs_out']:.3f}, and dropping a position that holds a quarter of a per cent of "
            f"the weight moves it in the fourth decimal place."]



def _cell(s):
    """A markdown table cell: pipes become \\| and newlines become spaces."""
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_tok_vocab(f):
    """Chapter 11: what the file itself declares about its tokenizer."""
    g = f["gguf"]
    rows = [
        ("Tokens in the vocabulary", f"{g['tokens']:,}"),
        ("Merges", f"{g['merges']:,}"),
        ("Single-byte tokens", f"{g['byte_tokens']}"),
        ("Tokens declared special", f"{g['n_special']:,}"),
        ("`tokenizer.ggml.model`", f"`{g['model']}`"),
        ("`tokenizer.ggml.pre`", f"`{g['pre']}`"),
        ("Normalizer", "not recorded in the file"),
    ]
    out = ["| Quantity | Value |", "|---|---|"]
    out += [f"| {a} | {b} |" for a, b in rows]
    return out


def render_tok_fixtures(f):
    """Chapter 11: the fixture set, with the ids this instrument produced."""
    out = ["| Fixture | Text (escaped) | Bytes | Ids | Count |", "|---|---|---|---|---|"]
    for x in f["fixtures"]:
        ids = ", ".join(str(i) for i in x["ids"][:10]) + (" ..." if x["n_ids"] > 10 else "") if x["ids"] else "-"
        shown = x["text"] if len(x["text"]) <= 24 else f"{x['text'][:8]}... ({len(x['text'])} chars)"
        out.append(f"| {x['name']} | `{_cell(shown)}` | {x['bytes']} | `{ids}` | {x['n_ids']} |")
    return out


def render_tok_columns(f):
    """Chapter 11: the same string through three passes - raw BPE, NFC, then the engine (NFC + specials)."""
    out = ["| Fixture | No normalizer | NFC first | NFC + declared specials |", "|---|---|---|---|"]
    for x in f["fixtures"]:
        if not (x["nfc_changes_ids"] or x["specials_change_ids"]):
            continue
        a = ", ".join(map(str, x["ids_raw"]))
        b = ", ".join(map(str, x["ids_nfc"]))
        c = ", ".join(map(str, x["ids"]))
        out.append(f"| {x['name']} | `{a}` | `{b}` | `{c}` |")
    out.append("")
    out.append(f"{len(f['summary']['engine_vs_raw'])} of {f['summary']['fixtures']} fixtures change ids "
               f"between the raw pass and the engine's pass.")
    return out


def render_tok_specials(f):
    """Chapter 11: the token_type histogram, which is what makes a token special."""
    g = f["gguf"]
    names = {"1": "normal (text the splitter may produce)", "3": "control", "4": "user-defined",
             "5": "unused", "6": "byte"}
    out = ["| `token_type` | Meaning | Count |", "|---|---|---|"]
    for k, v in sorted(g["token_type_counts"].items(), key=lambda kv: -kv[1]):
        out.append(f"| {k} | {names.get(k, 'other')} | {int(v):,} |")
    out.append("")
    out.append("| Token | Id | Type |")
    out.append("|---|---|---|")
    for s in g["special_tokens"][:6]:
        out.append(f"| `{_cell(s['token'])}` | {s['id']} | {s['type']} |")
    return out


def render_tok_holdback(f):
    """Chapter 11: one character, two tokens, and what a streamer may print."""
    h = f["holdback"]
    out = [f"`{h['text']}` encodes to `{h['ids']}` - one character, {len(h['ids'])} tokens.", "",
           "| Id | Bytes so far | Whole UTF-8 characters | Bytes held back | May print |",
           "|---|---|---|---|---|"]
    for r in h["rows"]:
        may = "nothing" if r["held_back"] else "the character"
        out.append(f"| {r['id']} | {r['bytes']} | {r['whole']} | {r['held_back']} | {may} |")
    return out


def render_tok_merges(f):
    """Chapter 11: the merge loop on one piece, with the rank that won each round."""
    out = ["| Round | Pair joined | Rank | Piece after |", "|---|---|---|---|"]
    for r in f["merge_trace"]:
        units = "`" + " ".join(u.replace("\u0120", "_") for u in r["units"]) + "`"
        if r["round"] == 0:
            out.append(f"| 0 | - | - | {units} |")
        else:
            a, b = r["pair"][0].replace("\u0120", "_"), r["pair"][1].replace("\u0120", "_")
            out.append(f"| {r['round']} | `{a}` + `{b}` | {r['rank']:,} | {units} |")
    return out


def render_tok_roundtrip(f):
    """Chapter 11: what decode(encode(x)) gives back, per fixture - the round trip is not the identity."""
    out = ["| Fixture | Text (escaped) | Ids | Decodes to | Same? |", "|---|---|---|---|---|"]
    for x in f["fixtures"]:
        if x["name"] in ("long",):
            continue
        out.append(f"| {x['name']} | `{_cell(x['text'])}` | {x['n_ids']} id(s) | `{_cell(x['decoded'])}` | "
                   f"{'yes' if x['roundtrip_same'] else '**no**'} |")
    return out


def render_tok_witness(f):
    """Chapter 11: the witness - the reference implementation's own numbers and its verdict on every fixture."""
    w, a = f["witness"], f["answer"]
    out = [f"| Quantity | `{w['file']}` |", "|---|---|",
           f"| Vocabulary entries | {w['vocab']:,} |",
           f"| Merges | {w['merges']:,} |",
           f"| Added tokens | {w['added_tokens']} |",
           f"| Normalizer | `{w['normalizer']}` |",
           f"| Pre-tokenizer starts with | `{w['pre_tokenizer']}` |",
           f"| Tool | `{w['tool']}` |",
           "",
           f"Compared {a['compared']} fixture strings: **{a['agree']} agree**."]
    return out



def _c(s):
    """A markdown table cell."""
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_tmpl_program(f):
    """Chapter 12: the template as a program - size, what it reads, what it calls."""
    prog = f["program"]
    rows = [
        ("Template length", f"{prog['chars']:,} characters, {prog['bytes']:,} bytes, {prog['lines']} lines"),
        ("`{{ ... }}` expressions", f"{prog['expressions']}"),
        ("`{% ... %}` statements", f"{prog['statements']}"),
        ("Variables it reads", ", ".join(f"`{v}`" for v in prog["variables"])),
        ("Filters it calls", ", ".join(f"`{v}`" for v in prog["filters"])),
        ("Roles it branches on", ", ".join(prog["roles"])),
        ("sha256", f"`{prog['sha256'][:16]}...`"),
    ]
    return ["| Quantity | Value |", "|---|---|"] + [f"| {k} | {_c(v)} |" for k, v in rows]


def render_tmpl_copies(f):
    """Chapter 12: the same template shipped twice, and what the two copies disagree about."""
    c = f["copies"]
    out = [f"| Copy | Characters | sha256 |", "|---|---|---|",
           f"| `tokenizer_config.json` (the model card) | {c['upstream']['chars']:,} | `{c['upstream']['sha256'][:16]}...` |",
           f"| `tokenizer.chat_template` in the GGUF | {c['gguf']['chars']:,} | `{c['gguf']['sha256'][:16]}...` |",
           "", f"Identical: **{'yes' if c['identical'] else 'no'}**."
           + (f" {len(c['edits'])} edit(s)." if not c["identical"] else ""), ""]
    if c["edits"]:
        out += ["| At character | Removed | Inserted |", "|---|---|---|"]
        for e in c["edits"]:
            out.append(f"| {e['at']:,} | `{e['removed'] or '-'}` | `{e['inserted'] or '-'}` |")
        out += ["", f"The model-card copy reads `{_c(c['upstream_around'])}`;",
                f"the GGUF copy reads `{_c(c['gguf_around'])}`."]
    return out


def render_tmpl_anatomy(f):
    """Chapter 12: one rendered prompt cut at its role markers - what is the user's and what is the template's."""
    a = f["anatomy"]
    out = [f"{a['prompt_bytes']} bytes, {a['prompt_tokens']} tokens. The segments below are cut at the special",
           "tokens and add up to the whole prompt:", "",
           "| At byte | Segment | Bytes | Tokens |", "|---|---|---|---|"]
    for s in a["segments"]:
        out.append(f"| {s['at']:,} | `{_c(s['text'])}` | {s['bytes']} | {s['tokens']} |")
    out += ["", f"Sum of the segments: **{a['sum_of_segments']}** tokens against **{a['prompt_tokens']}** for the",
            "whole prompt - the cut is at the special tokens, and a special token never merges with its",
            "neighbours, so the counts add up."]
    return out


def render_tmpl_surfaces(f):
    """Chapter 12: the same conversation with one knob moved."""
    out = ["| Variant | Bytes | Tokens | Ends with |", "|---|---|---|---|"]
    for r in f["surfaces"]:
        out.append(f"| {r['variant']} | {r['bytes']:,} | {r['tokens']} | `{_c(r['tail'])}` |")
    return out


def render_tmpl_fixtures(f):
    """Chapter 12: every committed case under three conditions."""
    fx = f["fixtures"]
    out = ["| Case | Tools | Bytes | Tokens | Reference | Jinja2 `tojson` | GGUF copy |",
           "|---|---|---|---|---|---|---|"]
    mark = {True: "same", False: "**differs**"}
    for r in fx["cases"]:
        out.append(f"| {r['name']} | {r['tools']} | {r['bytes']:,} | {r['tokens']} | "
                   f"{mark[r['matches_reference']]} | {mark[r['native_tojson_matches']]} | "
                   f"{mark[r['gguf_copy_matches']]} |")
    out += ["", f"Rendered with the model card's copy: **{fx['agree']} of {fx['total']}** byte-identical to the",
            f"reference renderer. With Jinja2's own `tojson`: **{fx['agree_native']} of {fx['total']}**. With the",
            f"copy inside the GGUF: **{fx['agree_gguf_copy']} of {fx['total']}**."]
    return out


def render_tmpl_tojson(f):
    """Chapter 12: one tool definition, four tojson policies."""
    out = ["| Policy | Characters | Tokens | Output starts |", "|---|---|---|---|"]
    for r in f["tojson"]:
        out.append(f"| {r['policy']} | {r['chars']} | {r['tokens']} | `{_c(r['text'][:64])}` |")
    return out


def render_tmpl_thinking(f):
    """Chapter 12: the reasoning flag, measured."""
    out = [f"The template is {f['thinking']['template_chars']:,} characters; one user turn, generation prompt on:",
           "", "| Variant | Characters | Tokens | Ends with |", "|---|---|---|---|"]
    for r in f["thinking"]["rows"]:
        out.append(f"| {r['variant']} | {r['chars']} | {r['tokens']} | `{_c(r['tail'])}` |")
    return out


def render_tmpl_pystr(f):
    """Chapter 12: the Python the templates call, evaluated by Python."""
    out = ["| Expression | Python returns |", "|---|---|"]
    for r in f["pystr"]:
        out.append(f"| `{_c(r['expression'])}` | `{_c(r['python'])}` |")
    return out


def render_tmpl_refusal(f):
    """Chapter 12: what a renderer does with a template it cannot honour."""
    out = ["| Template | Jinja2 default | Jinja2 strict |", "|---|---|---|"]
    for r in f["refusal"]:
        out.append(f"| `{_c(r['template'])}` | {_c(r['default'])} | {_c(r['strict'])} |")
    return out



def render_smp_verify(f):
    """Chapter 13: the engine's output, replayed from the engine's own logits by this book's pipeline."""
    out = ["| Run | Steps replayed | Python replay against the engine's text |", "|---|---|---|"]
    for name, v in f["verify"].items():
        verdict = "byte-identical" if v["match"] else "**differs**"
        out.append(f"| {name.replace('_', ' ')} | {v['steps']} | {verdict} |")
    first = next(iter(f["verify"].values()))
    out += ["", f"The replay produced `{_c(first['text'])}` - the engine's own text, from the engine's own logits."]
    return out


def render_smp_prompts(f):
    """Chapter 13: the prompt the showcase step comes from, chosen by measuring five candidates."""
    out = ["| Candidate prompt | Top-1 mass | Tokens for 90% | Entropy |", "|---|---|---|---|"]
    chosen = f["prompt_choice"]["chosen"]["prompt"]
    for r in f["prompt_choice"]["candidates"]:
        mark = " **chosen**" if r["prompt"] == chosen else ""
        out.append(f"| `{_c(r['prompt'])}`{mark} | {r['top1']:.3f} | {r['tokens_90']} | {r['entropy']:.2f} |")
    return out


def render_smp_pipeline(f):
    """Chapter 13: one real step, stage by stage."""
    p = f["pipeline"]
    out = [f"Step `{p['step']}`, prompt `{_c(p['prompt'])}`, top-1 mass before any stage: "
           f"{p['top1_raw']:.3f}.", "",
           "| Stage | Candidates left | What it did |", "|---|---|---|"]
    for r in p["rows"]:
        out.append(f"| {_c(r['stage'])} | {r['survivors']:,} | {_c(r['note'])} |")
    return out


def render_smp_temperature(f):
    """Chapter 13: temperature on the same step - the ranking holds, the weights move."""
    out = ["| Temperature | Top-1 mass | Top-5 mass | Tokens for 90% | Entropy | Argmax |", "|---|---|---|---|---|---|"]
    for r in f["temperature"]:
        out.append(f"| {r['temp']} | {r['top1']:.3f} | {r['top5']:.3f} | {r['tokens_90']} | {r['entropy']:.2f} | "
                   f"`{r['argmax']}` |")
    return out


def render_smp_funnel(f):
    """Chapter 13: the two truncating stages across the whole run."""
    rows = f["funnel"]
    after_p = [r["after_p"] for r in rows]
    top1 = [r["top1"] for r in rows]

    def stat(values, fmt="{:.0f}"):
        ordered = sorted(values)
        mid = ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
        return fmt.format(ordered[0]), fmt.format(mid), fmt.format(ordered[-1])

    out = [f"{len(rows)} steps of one greedy run, no penalties:", "",
           "| Quantity | Smallest | Median | Largest |", "|---|---|---|---|"]
    for label, values, fmt in (("Candidates after top-k 40", [r["after_k"] for r in rows], "{:,}"),
                               ("Candidates after top-p 0.95", after_p, "{:,}"),
                               ("Top-1 mass of the final set", top1, "{:.3f}"),
                               ("Entropy of the final set (nats)", [r["entropy"] for r in rows], "{:.2f}")):
        lo, mid, hi = stat(values, fmt)
        out.append(f"| {label} | {lo} | {mid} | {hi} |")
    widest = max(rows, key=lambda r: r["after_p"])
    out += ["", f"Top-k never bound at all: it kept 40 candidates on every step. Top-p did the cutting - "
            f"and the widest step ({_c(widest['step'])}) still kept {widest['after_p']} of the 40."]
    return out


def render_smp_penalty(f):
    """Chapter 13: what the repeat penalty actually changed."""
    p = f["penalty"]
    out = ["| Quantity | Value |", "|---|---|",
           f"| Steps where the penalty moved at least one logit | {p['steps_with_moves']} |",
           f"| Steps where it changed the argmax | {p['steps_flipped']} |"]
    if p["examples"]:
        out += ["", "| Step | Logits moved | Argmax without | Argmax with |", "|---|---|---|---|"]
        for e in p["examples"]:
            out.append(f"| {_c(e['step'])} | {e['moved']} | `{e['plain']}` | `{e['penalised']}` |")
    else:
        out += ["", "Every step's argmax survived the penalty. The penalty is a *shift*, not a veto: it moved "
                "logits on nearly every step and changed nothing about which token won."]
    return out


def render_smp_order(f):
    """Chapter 13: the same three stages in three orders, on one measured step."""
    out = ["| Order of the same stages | Candidates kept | Top-1 mass | Argmax |", "|---|---|---|---|"]
    for r in f["order"]:
        out.append(f"| {_c(r['order'])} | {r['survivors']:,} | {r['top1']:.3f} | `{r['argmax']}` |")
    return out


def render_smp_repro(f):
    """Chapter 13: what a seed buys."""
    out = ["| Run | Bytes | Same as the first run? | Shared prefix | Output |", "|---|---|---|---|---|"]
    for r in f["repro"]:
        out.append(f"| {_c(r['name'])} | {r['bytes']} | {'yes' if r['same_as_first'] else '**no**'} | "
                   f"{r['shared_prefix_bytes']} B | `{_c(r['text'])}` |")
    return out


def render_smp_stop(f):
    """Chapter 13: a stop string that spans more than one token."""
    s = f["stop"]
    out = ["| Quantity | Value |", "|---|---|",
           f"| Stop string | `{_c(s['stop'])}` |",
           f"| Tokens it spans | {s['stop_tokens']} (`{', '.join(str(i) for i in s['stop_ids'])}`) |",
           f"| Full greedy output | `{_c(s['whole_text'])}` |",
           f"| Output with the stop | `{_c(s['cut_text'])}` |",
           f"| Stop text in the output | {'yes' if s['kept_stop_text'] else 'no'} |"]
    return out


def render_smp_grammar(f):
    """Chapter 13: the same prompt with and without a schema, over eight seeds."""
    out = ["| Condition | Runs | Output parses as the schema | Example |", "|---|---|---|---|"]
    for r in f["grammar"]:
        sample = r["samples"][0]
        out.append(f"| {_c(r['condition'])} | {r['seeds']} | **{r['valid']} of {r['seeds']}** | "
                   f"`{_c(sample['text'])}` |")
    return out



def render_fs_vocab(f):
    """Chapter 15: how big the vocabulary is, and how small the model's use of it is."""
    v = f["vocab"]
    used = sorted({op for phase in f["structure"].values() for op in phase["graph_ops"]})
    out = ["| Quantity | Count |", "|---|---|"]
    out.append(f"| `Op` variants | {v['op_variants']} |")
    out.append(f"| ... carrying a payload | {v['payload_variants']} |")
    out.append(f"| ... unit variants | {v['unit_variants']} |")
    out.append(f"| `FusedOp` variants (fusion targets) | {v['fused_op_variants']} |")
    out.append(f"| Variants no supported architecture emits | {len(v['never_emitted'])} "
               f"({', '.join('`' + x + '`' for x in v['never_emitted'])}) |")
    out.append(f"| Variants this model's graph uses | {len(used)} |")
    return out


def render_fs_structure(f):
    """Chapter 15: the rewrite, and what it leaves behind, per graph type."""
    out = ["| | Prefill graph | Decode graph |", "|---|---|---|"]
    pre, dec = f["structure"]["prefill"], f["structure"]["decode"]
    out.append(f"| Nodes | {pre['nodes']} | {dec['nodes']} |")
    out.append(f"| `mul` nodes | {pre['mul']} | {dec['mul']} |")
    out.append(f"| `swiglu` nodes (rewrite output) | {pre['swiglu']} | {dec['swiglu']} |")
    out.append(f"| `silu` nodes left in the graph | {pre['silu']} | {dec['silu']} |")
    out.append(f"| ... of those, executed | {pre['silu_run']} | {dec['silu_run']} |")
    out.append(f"| Nodes recorded by the per-node trace | {pre['recorded']} | {dec['recorded']} |")
    out.append(f"| Nodes not recorded | {', '.join(f'{k} {v}' for k, v in pre['not_recorded'].items())} "
               f"| {', '.join(f'{k} {v}' for k, v in dec['not_recorded'].items())} |")
    return out


def render_fs_kernel(f):
    """Chapter 15: the same 3-to-1 rewrite, measured on two activations at four row counts."""
    out = ["| Activation | Rows | Three nodes | One node | Speedup | Output identical |",
           "|---|---|---|---|---|---|"]
    for act in f["kernel"]["acts"]:
        for r in act["rows"]:
            # the relu kernel at one row finishes inside the clock's resolution: there is no rate to
            # divide, so the cell says so instead of printing a ratio computed from a zero
            sp = f"{r['us_three'] / r['us_one']:.2f}x" if r["us_one"] else "n/a"
            out.append(f"| `{act['act']}` | {r['rows']:,} | {r['us_three']:,.0f} us "
                       f"({r['gbs_three']:.1f} GB/s) | {r['us_one']:,.0f} us "
                       f"({r['gbs_one']:.1f} GB/s) | {sp} | "
                       f"{'yes' if r['identical'] else '**no**'} |")
    return out


def render_fs_bill(f):
    """Chapter 15: the byte bill, and the machine's own streaming rate to read it against."""
    b = f["kernel"]["bill"]
    bw = f["kernel"]["bandwidth"]["1"]
    out = ["| Quantity | Bytes |", "|---|---|"]
    out.append(f"| Three nodes, one row | {b['per_row_bytes_three']:,} |")
    out.append(f"| One node, one row | {b['per_row_bytes_one']:,} |")
    out.append(f"| Saved per row | {b['saved_per_row']:,} ({b['saved_fraction']:.0%}) |")
    out.append(f"| Saved per token, 24 layers | {b['per_token_saved']:,} |")
    out.append(f"| This machine, one core, streaming copy | {bw['copy']:.1f} GB/s |")
    return out


def render_fs_witness(f):
    """Chapter 15: the numpy recomputation behind the C samples."""
    w = f["kernel"]["witness"]
    return [f"| {w['agree']}/{w['checked']} sampled outputs | recomputed in f64 from the same seed |",
            "|---|---|",
            f"| Agreement | {w['agree']}/{w['checked']} within float32 rounding |",
            f"| Worst relative difference | {w['worst_relative']:.2e} |"]


def render_fs_tests(f):
    """Chapter 15: the engine's own fusion tests, run on this machine."""
    out = ["| Test | Result |", "|---|---|"]
    for name in f["tests"]["run"]:
        out.append(f"| `{name}` | {'ok' if name not in f['tests']['failed'] else '**FAILED**'} |")
    return out


def _mb(n):
    return f"{n / 1048576:.2f} MiB"


def render_al_liveness(f):
    """Chapter 17: the naive bill against the engine's arena, phase by phase."""
    pre, dec = f["phases"]["prefill"], f["phases"]["decode"]
    ep, ed, reuse = f["prefill"], f["decode"], f["reuse"]
    led = f["ledger"]
    pf_pool, pf_live, pf_peak = (led["pool_bytes"]["value"], led["live_bytes"]["value"],
                                 led["peak_live_bytes"]["value"])
    out = ["| | Prefill (34 tokens) | Decode (1 token) |", "|---|---|---|",
           f"| Nodes in the graph | {pre['nodes']} | {dec['nodes']} |",
           f"| Transient nodes (the rest name persistent regions) | {pre['transient_nodes']} "
           f"| {dec['transient_nodes']} |",
           f"| Sum of every transient buffer, no reuse (**the book's arithmetic**) "
           f"| {pre['naive_bytes']:,} B ({_mb(pre['naive_bytes'])}) "
           f"| {dec['naive_bytes']:,} B ({_mb(dec['naive_bytes'])}) |",
           f"| The engine's pool, high-water (**the engine's**) | {pf_pool:,} B "
           f"({_mb(pf_pool)}) | {ed['pool_bytes']:,} B ({_mb(ed['pool_bytes'])}) |",
           f"| Live when the phase ends | {pf_live:,} B ({_mb(pf_live)}) "
           f"| {ed['live_bytes']:,} B ({_mb(ed['live_bytes'])}) |",
           f"| Peak live inside the phase | {pf_peak:,} B ({_mb(pf_peak)}) "
           f"| {ed['peak_live_bytes']:,} B ({_mb(ed['peak_live_bytes'])}) |",
           f"| Naive / pool (what reuse bought) | {reuse['prefill']}x | {reuse['decode']}x |",
           f"| Size classes still reserved at the end | {ep['reserved_classes']} "
           f"| {ed['reserved_classes']} |",
           f"| Idle slots in them | {ep['idle_slots']} | {ed['idle_slots']} |",
           f"| Largest transient buffer | `{pre['largest_buffer']['op']}` "
           f"{'x'.join(str(d) for d in pre['largest_buffer']['shape'])} = "
           f"{pre['largest_buffer']['bytes']:,} B | `{dec['largest_buffer']['op']}` "
           f"{'x'.join(str(d) for d in dec['largest_buffer']['shape'])} = "
           f"{dec['largest_buffer']['bytes']:,} B |"]
    return out


def render_al_ledger(f):
    """Chapter 17: the ledger, with the party that measured each line named in the table."""
    led, kv, proc = f["ledger"], f["kv"], f["process"]
    total = led["predicted_bytes"]["value"]
    obs = led["observed_rss_bytes"]["value"]
    rows = [("Weights - for the life of the process", "weights_bytes"),
            (f"KV regions, {kv['layers']} layers - persistent across rebuilds", "kv_region_bytes"),
            ("Transient pool, high-water - dies with the graph", "pool_bytes"),
            ("**Predicted resident** - the three rows above, added", "predicted_bytes"),
            ("**Observed peak RSS** - the whole process", "observed_rss_bytes"),
            ("Prediction minus observation", "delta_bytes")]
    out = ["| Region | Bytes | MiB | Whose number |", "|---|---|---|---|"]
    for label, key in rows:
        value = led[key]["value"]
        author = f"{led[key]['author']}, `{led[key]['from'].replace(' high-water', '')}`"
        bold = label.startswith("**")
        name = label.replace("**", "")
        out.append(f"| {'**' if bold else ''}{name}{'**' if bold else ''} "
                   f"| {'**' if bold else ''}{value:,}{'**' if bold else ''} "
                   f"| {'**' if bold else ''}{value / 2**20:.1f}{'**' if bold else ''} "
                   f"| {author} |" if not bold else
                   f"| **{name}** | **{value:,}** | **{value / 2**20:.1f}** | {author} |")
    out.append("")
    out.append(f"The prediction is not a bound in either direction, and the reason is in the same table: "
               f"the cache is allocated for {kv['n_ctx']:,} tokens and the run wrote {kv['n_tokens']}, so "
               f"most of the {_mb(led['kv_region_bytes']['value'])} of KV is address space the kernel never "
               f"had to back. Observed peak RSS is a median of {proc['runs']} runs "
               f"({proc['maxrss_min']:,} to {proc['maxrss_max']:,} B).")
    return out


def render_al_classes(f):
    """Chapter 17: what the pool keeps between phases - the engine's own slot and class counts."""
    pre, dec = f["prefill"], f["decode"]
    out = ["| | Prefill arena | Decode arena (fresh) |", "|---|---|---|",
           f"| Nodes allocated | 443 | {dec['nodes']} |",
           f"| Pool bytes | {f['ledger']['pool_bytes']['value']:,} | {dec['pool_bytes']:,} |",
           f"| Size classes reserved | {pre['reserved_classes']} | {dec['reserved_classes']} |",
           f"| Idle slots kept for the next graph | {pre['idle_slots']} | {dec['idle_slots']} |"]
    return out


def render_ds_registry(f):
    """Chapter 16: one id space, two orders."""
    r = f["registry"]
    out = ["| Backend | Registry id | Assignment priority |", "|---|---|---|"]
    for name in r["names"]:
        out.append(f"| `{name}` | {r['ids'].get(name, '?')} | {r['priority'].get(name, 'not compiled in')} |")
    out.append("")
    out.append(f"Backend ids in the space: **{r['n_backends']}**. The id order is the on-disk tag order; "
               f"the assignment order is priority, and the two are deliberately not the same.")
    return out


def render_ds_matrix(f):
    """Chapter 16: the documented op-by-backend contract, and which columns this machine can check."""
    out = ["| Operator | CPU | Metal | CUDA |", "|---|---|---|---|"]
    for row in f["doc_matrix"]:
        mark = lambda v: "yes" if v == "yes" else "no" if v == "no" else "?"
        out.append(f"| `{row['op']}` | {mark(row['cpu'])} | {mark(row['metal'])} | {mark(row['cuda'])} |")
    rows = len(f["doc_matrix"])
    yes = f["summary"]["cpu_yes_rows"]
    out.append("")
    out.append(f"{rows} documented rows; {yes} accept the op on CPU, "
               f"{rows - yes} refuse it. `supports_op` is what decides, and the engine's own test "
               f"`support_table_matches_support_matrix_doc` fails when a column drifts from this table.")
    return out


def render_ds_assignment(f):
    """Chapter 16: what the assignment pass produced on this machine."""
    out = ["| | Prefill graph | Decode graph |", "|---|---|---|"]
    pre, dec = f["assignment"]["prefill"], f["assignment"]["decode"]
    out.append(f"| Nodes | {pre['nodes']} | {dec['nodes']} |")
    out.append(f"| Backends assigned | {', '.join(f'{k}: {v}' for k, v in pre['backend_histogram'].items())} "
               f"| {', '.join(f'{k}: {v}' for k, v in dec['backend_histogram'].items())} |")
    out.append(f"| Contiguous same-backend splits | {pre['splits']} | {dec['splits']} |")
    out.append(f"| Cross-split edges (each one a copy) | {pre['cross_split_edges']} | "
               f"{dec['cross_split_edges']} |")
    out.append(f"| How the dump names a backend | a {pre['backend_field_kind']}, not an id | "
               f"a {dec['backend_field_kind']}, not an id |")
    return out


def render_ds_refusals(f):
    """Chapter 16: three refusals, three messages, all of them errors."""
    out = ["| Case | Command | Exit | Message |", "|---|---|---|---|"]
    for r in f["refusals"]:
        msg = r["message"].replace("|", "\\|")
        out.append(f"| {r['case']} | `{r['args']}` | {r['exit']} | {msg} |")
    return out


def render_ds_tests(f):
    """Chapter 16: the engine's op-support contract tests, run on this machine."""
    out = ["| Test | Result |", "|---|---|"]
    for name in f["support_tests"]["run"]:
        out.append(f"| `{name}` | {'ok' if name not in f['support_tests']['failed'] else '**FAILED**'} |")
    return out


def render_gr_shape(f):
    """Chapter 14: what the two graph types are made of."""
    pre, dec = f["shape"]["prefill"], f["shape"]["decode"]
    out = ["| Quantity | Prefill graph | Decode graph |", "|---|---|---|"]
    out.append(f"| Nodes | {pre['nodes']:,} | {dec['nodes']:,} |")
    out.append(f"| Leaf inputs | {pre['input_nodes']:,} | {dec['input_nodes']:,} |")
    out.append(f"| Output node | {', '.join(str(i) for i in pre['outputs'])} | "
               f"{', '.join(str(i) for i in dec['outputs'])} |")
    out.append(f"| Steps executed | {pre['steps']:,} | {dec['steps']:,} |")
    out.append(f"| Backends assigned | {', '.join(pre['backends'])} | {', '.join(dec['backends'])} |")
    out += ["", "| Op | Prefill | Decode |", "|---|---|---|"]
    # deterministic order: count first, then the op name - a bare count-only key lets Python's set order (and
    # with it PYTHONHASHSEED) decide the table's row order, which the fact gate correctly refuses.
    ops = sorted(set(pre["ops"]) | set(dec["ops"]), key=lambda o: (-pre["ops"].get(o, 0), o))
    for op in ops:
        out.append(f"| `{op}` | {pre['ops'].get(op, 0):,} | {dec['ops'].get(op, 0):,} |")
    return out


def render_gr_topology(f):
    """Chapter 14: the same topology for two prompts of different length."""
    s = f["same_topology"]
    d = s["first_shape_difference"]
    out = ["| Quantity | Short prompt (34 tokens) | Long prompt (58 tokens) |", "|---|---|---|",
           f"| Nodes | {s['short_prompt_nodes']:,} | {s['long_prompt_nodes']:,} |",
           f"| Node sequence (op, name, edges) identical | - | **{'yes' if s['signature_equal'] else 'no'}** |"]
    if d:
        out.append(f"| Node {d['node']} shape | `{d['short']}` | `{d['long']}` |")
    return out


def render_gr_delta(f):
    """Chapter 14: the three-node difference between the two graph types."""
    delta = f["shape"]["delta"]
    out = ["| Op | Prefill | Decode | Difference |", "|---|---|---|---|"]
    for op, diff in sorted(delta.items(), key=lambda kv: -abs(kv[1])):
        pre = f["shape"]["prefill"]["ops"].get(op, 0)
        dec = f["shape"]["decode"]["ops"].get(op, 0)
        out.append(f"| `{op}` | {pre:,} | {dec:,} | {diff:+,} |")
    out += ["", "The nodes that exist only in the prefill graph: "
            + ", ".join(f"`{_c(n)}`" for n in f["shape"]["delta_names"]) + "."]
    return out


def render_gr_dag(f):
    """Chapter 14: the DAG the edges describe."""
    d = f["dag"]
    out = ["| Quantity | Value |", "|---|---|",
           f"| Longest chain of nodes | {d['depth']} |",
           f"| Nodes at that depth | {d['nodes_at_max_depth']} |",
           f"| Most inputs one node takes | {d['max_fan_in']} |",
           f"| Most consumers one node feeds | {d['max_fan_out']} |",
           f"| Leaf inputs | {d['leaves']} |"]
    hist = ", ".join(f"{k} input(s): {v}" for k, v in d["fan_in_histogram"].items())
    out += ["", f"Nodes by input count - {hist}."]
    return out


def render_gr_reuse(f):
    """Chapter 14: how many graphs a generation builds."""
    r = f["reuse"]
    out = ["| Build | Tokens in that graph | Cost (build + assign + alloc) |", "|---|---|---|"]
    for i, b in enumerate(r["builds"], 1):
        out.append(f"| {i} | {b['n_tokens']} | {b['ms']} ms |")
    out += ["", f"Steps executed: **{r['steps_executed']}** (1 prefill + {r['decode_steps']} decode). "
            f"Graphs built: **{len(r['builds'])}**."]
    return out


def render_gr_trace(f):
    """Chapter 14: what one node of the trace says, and what the graph's output looks like."""
    t = f["trace"]
    logits = t["logits_node"] or {}
    out = ["| Trace field | Value |", "|---|---|",
           f"| Nodes recorded in one decode step | {t['nodes_recorded']:,} |",
           f"| Per-node statistics | {', '.join(t['node_stat_fields'])} |",
           f"| Largest sample kept per node | {t['sample_cap']} values |"]
    if logits:
        out += [f"| Output node | `{logits['id']}`, {logits['n']:,} values, {logits['dtype']} |",
                f"| Output range | min {logits['stats']['min']:.2f}, max {logits['stats']['max']:.2f}, "
                f"mean {logits['stats']['mean']:.2f} |"]
    if t["logits_top"]:
        out += ["", "| Rank | Id | Probability | Token |", "|---|---|---|---|"]
        for i, row in enumerate(t["logits_top"], 1):
            out.append(f"| {i} | {row['id']} | {row['prob']:.4f} | `{_c(row['token'])}` |")
    return out


BLOCKS = {
    "al-liveness": render_al_liveness,
    "al-ledger": render_al_ledger,
    "al-classes": render_al_classes,
    "ds-registry": render_ds_registry,
    "ds-matrix": render_ds_matrix,
    "ds-assignment": render_ds_assignment,
    "ds-refusals": render_ds_refusals,
    "ds-tests": render_ds_tests,
    "fs-vocab": render_fs_vocab,
    "fs-structure": render_fs_structure,
    "fs-kernel": render_fs_kernel,
    "fs-bill": render_fs_bill,
    "fs-witness": render_fs_witness,
    "fs-tests": render_fs_tests,
    "tok-vocab": render_tok_vocab,
    "tok-fixtures": render_tok_fixtures,
    "tok-columns": render_tok_columns,
    "tok-specials": render_tok_specials,
    "tok-holdback": render_tok_holdback,
    "tok-merges": render_tok_merges,
    "tok-witness": render_tok_witness,
    "tok-roundtrip": render_tok_roundtrip,
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
    "provenance": render_provenance,
    "attention-scores": render_attention_scores,
    "decode-step": render_decode_step,
    "kv-cache-size": render_kv_cache_size,
    "rebuild": render_rebuild,
    "mask-effect": render_mask_effect,
    "gr-trace": render_gr_trace,
    "gr-reuse": render_gr_reuse,
    "gr-dag": render_gr_dag,
    "gr-delta": render_gr_delta,
    "gr-topology": render_gr_topology,
    "gr-shape": render_gr_shape,
    "smp-grammar": render_smp_grammar,
    "smp-stop": render_smp_stop,
    "smp-order": render_smp_order,
    "smp-repro": render_smp_repro,
    "smp-penalty": render_smp_penalty,
    "smp-funnel": render_smp_funnel,
    "smp-temperature": render_smp_temperature,
    "smp-pipeline": render_smp_pipeline,
    "smp-prompts": render_smp_prompts,
    "smp-verify": render_smp_verify,
    "tmpl-refusal": render_tmpl_refusal,
    "tmpl-pystr": render_tmpl_pystr,
    "tmpl-thinking": render_tmpl_thinking,
    "tmpl-tojson": render_tmpl_tojson,
    "tmpl-fixtures": render_tmpl_fixtures,
    "tmpl-surfaces": render_tmpl_surfaces,
    "tmpl-anatomy": render_tmpl_anatomy,
    "tmpl-copies": render_tmpl_copies,
    "tmpl-program": render_tmpl_program,
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
    # A renderer may need to know which artifact it is rendering (provenance names the script), so the
    # ones that accept it are called with the name and the rest keep the single-argument signature.
    body = "\n".join(renderer(f, name) if renderer is render_provenance else renderer(f))
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


def unparsed():
    """Every `<!-- facts: ... -->` marker that this script will never render, with the reason.

    The renderer is a substitution over `MARKER`; a marker that does not parse, or that names a block
    nobody registered, is left in the page *as an HTML comment* - invisible to the reader, invisible to
    `--check` (which only walks the markers it could parse), and green in the gate. Two chapters shipped
    that way before this function existed.
    """
    bad = []
    for page in pages():
        text = page.read_text()
        for i, line in enumerate(text.split("\n"), 1):
            if "<!-- facts:" not in line:
                continue
            m = MARKER.search(line)
            if not m:
                bad.append(f"{page.relative_to(ROOT)}:{i}: `{line.strip()}` does not parse as "
                           f"`<!-- facts: <artifact>.<block> -->`")
                continue
            name, block = m.group("name"), m.group("block")
            if block not in BLOCKS:
                bad.append(f"{page.relative_to(ROOT)}:{i}: no renderer registered for block "
                           f"`{block}` (add it to BLOCKS in scripts/facts.py)")
            elif not (FACTS / f"{name}.json").exists():
                bad.append(f"{page.relative_to(ROOT)}:{i}: there is no artifact `facts/{name}.json`")
    return bad


def check():
    findings = []
    used, seen = set(), 0
    for p in pages():
        text = p.read_text()
        names_here = set()
        for m in MARKER.finditer(text):
            name, block = m.group("name"), m.group("block")
            seen += 1
            used.add(name)
            names_here.add(name)
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
        # Every facts file a page quotes owes that page a provenance line. Without this rule the line is
        # written once, in the chapter the author was paying attention to, and the other eight pages keep
        # showing measured numbers with no origin - which is exactly what a review found.
        for name in sorted(names_here):
            if f"<!-- facts: {name}.provenance -->" not in text:
                findings.append(f"{p.relative_to(ROOT)}: quotes facts/{name}.json but never says where the "
                                f"numbers came from - add `<!-- facts: {name}.provenance -->`")
    # A markdown table whose header row has a different number of cells than its delimiter row is not
    # repaired by the parser: GFM refuses to recognise the table at all and ships the whole block as a
    # paragraph of pipe characters. That is how Chapter 6's accumulator table reached a reader while every
    # check stayed green - the header cell `Largest |q|` had an unescaped pipe pair, so 7 cells against 5
    # delimiters. The `||` check below only ever saw doubled pipes; this one counts the cells.
    for page in sorted((ROOT / "src").rglob("*.md")):
        lines = page.read_text().split("\n")
        in_fence = False
        for n, line in enumerate(lines):
            if line.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if line.lstrip().startswith("|") and "||" in line:
                findings.append(f"{page.relative_to(ROOT)}:{n + 1}: a table row contains \"||\" - the markdown "
                                f"parser will split the cell; spell the norm product as words")
            if not line.lstrip().startswith("|") or n + 1 >= len(lines):
                continue
            delimiter = lines[n + 1]
            if not re.match(r"^\s*\|[-: |]+\|\s*$", delimiter):
                continue
            cells = len(re.findall(r"(?<!\\)\|", line)) - 1
            delim_cells = len(re.findall(r"\|", delimiter)) - 1
            if cells != delim_cells:
                findings.append(f"{page.relative_to(ROOT)}:{n + 1}: table header has {cells} cell(s) against "
                                f"{delim_cells} delimiter(s) - an unescaped `|` inside a cell; GFM will ship "
                                f"the whole table as a paragraph")
                continue
            for k in range(n + 2, len(lines)):
                if not lines[k].lstrip().startswith("|"):
                    break
                got = len(re.findall(r"(?<!\\)\|", lines[k])) - 1
                if got != delim_cells:
                    findings.append(f"{page.relative_to(ROOT)}:{k + 1}: table row has {got} cell(s) against "
                                    f"{delim_cells} in the header - the table will not parse")
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

    findings += unparsed()

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
