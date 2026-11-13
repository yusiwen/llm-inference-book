#!/usr/bin/env python3
"""dequant-check.py - decode real blocks with two implementations and compare the numbers.

Chapter 4's risk is not "the block is the wrong length". A block whose fields are in the wrong order still
decodes to 32 floats of plausible magnitude, which means every length check in the book passes and the
model produces fluent nonsense. The only test that catches it compares *values*: take the same bytes, decode
them two ways, and require bit-level agreement.

This script is the book's own decoder, written out field by field from the ggml reference (`dequantize_row_*`
in ggml-quants.c), run against the same bytes through `gguf-py`'s vectorised decoder. Where a format is not in
any file the book ships, the blocks are produced by `gguf-py`'s quantizer from real f32 weights; that is
weaker evidence, and the report says which blocks are which.

    uv run --with gguf python scripts/dequant-check.py \\
        --real facts/gguf-qwen2.5-0.5b.json:~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf \\
        --real facts/gguf-qwen2.5-0.5b-q4_0.json:~/.cache/myfer/qwen2.5-0.5b-instruct-q4_0.gguf \\
        --out facts/dequant-check.json

What the output cannot prove: that a kernel multiplies correctly. It proves the decoder against the file's
bytes; Chapter 6 and Chapter 27 prove the arithmetic built on top of it.
"""

import argparse
import hashlib
import json
import pathlib
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Block geometries, from the ggml specification. Each entry: (elements, bytes, field list).
# The field list is what the chapter's anatomy table prints, so it lives here rather than in prose.
FORMATS = {
    "Q4_0": (32, 18, [("d", "f16", 2), ("qs", "u8[16]", 16)]),
    "Q4_1": (32, 20, [("d", "f16", 2), ("m", "f16", 2), ("qs", "u8[16]", 16)]),
    "Q5_0": (32, 22, [("d", "f16", 2), ("qh", "u8[4]", 4), ("qs", "u8[16]", 16)]),
    "Q5_1": (32, 24, [("d", "f16", 2), ("m", "f16", 2), ("qh", "u8[4]", 4), ("qs", "u8[16]", 16)]),
    "Q8_0": (32, 34, [("d", "f16", 2), ("qs", "i8[32]", 32)]),
}


def f16(b, off):
    return struct.unpack_from("<e", b, off)[0]


def decode_block(dtype, raw):
    """The book's decoder: one block's bytes in, 32 floats out. Scalar on purpose."""
    out = [0.0] * 32
    if dtype == "Q4_0":
        d = f16(raw, 0)
        for j, byte in enumerate(raw[2:18]):
            out[j] = d * ((byte & 0x0F) - 8)
            out[j + 16] = d * ((byte >> 4) - 8)
    elif dtype == "Q4_1":
        d, m = f16(raw, 0), f16(raw, 2)
        for j, byte in enumerate(raw[4:20]):
            out[j] = d * (byte & 0x0F) + m
            out[j + 16] = d * (byte >> 4) + m
    elif dtype in ("Q5_0", "Q5_1"):
        d = f16(raw, 0)
        off = 2
        m = 0.0
        if dtype == "Q5_1":
            m = f16(raw, 2)
            off = 4
        qh = int.from_bytes(raw[off:off + 4], "little")
        qs = raw[off + 4:off + 20]
        bias = 16.0 if dtype == "Q5_0" else 0.0
        for j, byte in enumerate(qs):
            hi_lo = ((qh >> j) << 4) & 0x10           # weight j      uses qh bit j
            hi_hi = (qh >> (j + 12)) & 0x10           # weight j + 16 uses qh bit j + 16
            out[j] = d * (((byte & 0x0F) | hi_lo) - bias) + m
            out[j + 16] = d * (((byte >> 4) | hi_hi) - bias) + m
    elif dtype == "Q8_0":
        d = f16(raw, 0)
        out = [d * float(v) for v in np.frombuffer(raw, dtype=np.int8, count=32, offset=2)]
    else:
        raise SystemExit(f"no decoder for {dtype}")
    return out


def blocks_of(tensor_bytes, dtype, count):
    """First `count` blocks of a tensor's byte range."""
    elems, size, _ = FORMATS[dtype]
    return [tensor_bytes[i * size:(i + 1) * size] for i in range(count)]



def error_of_format(quants, gguf, dtype, values):
    """Quantize the same real values into one format, decode, and measure what it cost."""
    enum = getattr(gguf.GGMLQuantizationType, dtype)
    packed = quants.quantize(np.ascontiguousarray(values, dtype=np.float32), enum)
    back = np.asarray(quants.dequantize(packed.copy(), enum), dtype=np.float64).ravel()
    ideal = np.asarray(values, dtype=np.float64)
    err = back - ideal
    rms = float(np.sqrt(np.mean(err ** 2)))
    scale = float(np.sqrt(np.mean(ideal ** 2))) or 1.0
    return {
        "dtype": dtype,
        "values": int(ideal.size),
        "rms_error": rms,
        "relative_rms": rms / scale,
        "max_abs_error": float(np.max(np.abs(err))),
        "signal_rms": scale,
    }


def outlier_block(raw, dtype="Q4_0", limit=4096):
    """Find the real Q4_0 block whose scale wastes the most on one value, and measure the damage.

    Q4_0 stores 32 weights as 4-bit integers and one f16 scale chosen from the block's largest magnitude,
    so a single outlier sets the step size for all 32. This walks real blocks and reports the worst case
    next to the median case - the arithmetic behind Lesson 5.
    """
    size = 18
    rows = []
    for i in range(min(limit, len(raw) // size)):
        blk = raw[i * size:(i + 1) * size]
        w = np.array(decode_block(dtype, blk), dtype=np.float64)
        d = abs(f16(blk, 0))
        step = d if d else 1e-30
        abs_w = np.abs(w)
        i_max = int(np.argmax(abs_w))
        others = np.delete(w, i_max)                      # the 31 weights the outlier's step applies to
        rounded = np.round(others / step) * step
        mean_others = float(np.mean(np.abs(others))) or 1e-30
        rows.append({
            "block": i,
            "step": step,
            "max_abs": float(abs_w[i_max]),
            "mean_others": mean_others,
            "ratio": float(abs_w[i_max] / mean_others),
            "step_over_mean": float(step / mean_others),
            "rms_others": float(np.sqrt(np.mean((others - rounded) ** 2))),
        })
    worst = max(rows, key=lambda r: r["step_over_mean"])
    ratios = sorted(r["ratio"] for r in rows)
    steps = sorted(r["step_over_mean"] for r in rows)
    return {"blocks_scanned": len(rows), "worst": worst,
            "median_ratio": ratios[len(ratios) // 2],
            "median_step_over_mean": steps[len(steps) // 2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="append", default=[],
                    help="facts.json:model.gguf - take blocks from a real file")
    ap.add_argument("--synthetic-from", default=None,
                    help="facts.json:model.gguf - source of f32 weights for formats no file ships")
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "facts" / "dequant-check.json"))
    args = ap.parse_args()

    import gguf
    from gguf import quants

    # 1. Find, for each format, a real tensor of that dtype and read its first blocks.
    sources, results = {}, []
    for spec in args.real:
        facts_path, model_path = spec.split(":", 1)
        facts = json.loads(pathlib.Path(facts_path).read_text())
        model_path = pathlib.Path(model_path).expanduser()
        raw = model_path.read_bytes()
        for t in facts["tensors"]:
            dt = t["type"]
            if dt in FORMATS and dt not in sources:
                body = raw[t["abs"]:t["abs"] + t["bytes"]]
                sources[dt] = {"kind": "real file", "tensor": t["name"],
                               "file": pathlib.Path(model_path).name, "bytes": body,
                               "artifact_sha256": facts["artifact"]["sha256"]}

    # 2. Formats no file ships: quantize real f32 weights with gguf-py and use those blocks.
    if args.synthetic_from:
        facts_path, model_path = args.synthetic_from.split(":", 1)
        facts = json.loads(pathlib.Path(facts_path).read_text())
        model_path = pathlib.Path(model_path).expanduser()
        raw = model_path.read_bytes()
        # A source of real values: an f32 tensor if the file has a big enough one (this one does not - its
        # f32 tensors are norms and biases), otherwise the first blocks of a weight tensor decoded to f32.
        src_t = next((t for t in facts["tensors"] if t["type"] == "F32" and t["numel"] >= 32 * args.blocks),
                     None)
        if src_t is not None:
            values = np.frombuffer(raw[src_t["abs"]:src_t["abs"] + src_t["bytes"]], dtype=np.float32)[: 32 * args.blocks]
        else:
            src_t = next(t for t in facts["tensors"] if t["type"] in FORMATS and t["numel"] >= 32 * args.blocks)
            _, size, _ = FORMATS[src_t["type"]]
            packed = np.frombuffer(raw[src_t["abs"]:src_t["abs"] + size * args.blocks],
                                   dtype=np.uint8).reshape(args.blocks, size)
            values = np.asarray(quants.dequantize(packed.copy(),
                                                  getattr(gguf.GGMLQuantizationType, src_t["type"]))).ravel()
        for dt in ("Q4_1", "Q5_1"):
            if dt in sources:
                continue
            enum = getattr(gguf.GGMLQuantizationType, dt)
            blocks = quants.quantize(np.ascontiguousarray(values[: 32 * args.blocks], dtype=np.float32), enum)
            sources[dt] = {"kind": "synthetic (gguf-py quantizer over real weights)",
                           "tensor": src_t["name"], "file": pathlib.Path(model_path).name,
                           "bytes": blocks.tobytes(),
                           "artifact_sha256": facts["artifact"]["sha256"]}

    # 3. Decode both ways and compare.
    for dt, src in sources.items():
        elems, size, fields = FORMATS[dt]
        blk = blocks_of(src["bytes"], dt, args.blocks)
        enum = getattr(gguf.GGMLQuantizationType, dt)
        mine, theirs = [], []
        for raw_block in blk:
            mine.extend(decode_block(dt, raw_block))
        packed = np.frombuffer(src["bytes"][: size * args.blocks], dtype=np.uint8).reshape(args.blocks, size)
        theirs.extend(np.asarray(quants.dequantize(packed.copy(), enum), dtype=np.float32).ravel().tolist())
        diffs = np.abs(np.array(mine, dtype=np.float64) - np.array(theirs, dtype=np.float64))
        scale = np.maximum(np.abs(np.array(theirs, dtype=np.float64)), 1e-30)
        exact = int(np.sum(diffs == 0))
        results.append({
            "dtype": dt,
            "source": src["kind"],
            "tensor": src["tensor"],
            "file": src["file"],
            "blocks": args.blocks,
            "elements": args.blocks * elems,
            "fields": [{"name": n, "type": t, "bytes": b} for n, t, b in fields],
            "bits_per_weight": round(size * 8 / elems, 4),
            "max_abs_diff": float(diffs.max()) if len(diffs) else 0.0,
            "max_rel_diff": float((diffs / scale).max()) if len(diffs) else 0.0,
            "exact_matches": exact,
            "compared": len(mine),
            "agreement": "bit-exact" if exact == len(mine) else ("within 1e-6 relative"
                                                                if (diffs / scale).max() < 1e-6 else "DIFFERS"),
        })
        print(f"{dt:5s} {results[-1]['blocks']} blocks  exact {exact}/{len(mine)}  "
              f"max abs {results[-1]['max_abs_diff']:.3e}  max rel {results[-1]['max_rel_diff']:.3e}  "
              f"{results[-1]['agreement']}")

    # 4. What each format costs against the same real values, and what an outlier does to its block.
    if args.synthetic_from:
        report_error = [error_of_format(quants, gguf, dt, values) for dt in FORMATS]
        for e in report_error:
            print(f"error {e['dtype']:5s} rms {e['rms_error']:.5f}  relative {e['relative_rms']:.4%}  "
                  f"max abs {e['max_abs_error']:.5f}")
        q4 = sources.get("Q4_0")
        out_block = outlier_block(q4["bytes"]) if q4 else None
        if out_block:
            print(f"outlier: worst ratio {out_block['worst']['ratio']:.1f}x at block "
                  f"{out_block['worst']['block']}, median {out_block['median_ratio']:.1f}x")
    else:
        report_error, out_block = [], None

    report = {
        "witness": "gguf-py (gguf.quants.dequantize) - the ggml project's reference Python decoder",
        "invocation": "uv run --with gguf python scripts/dequant-check.py " + " ".join(sys.argv[1:]),
        "note": ("Real-file blocks are the strongest evidence available: the bytes were written by "
                 "llama.cpp's quantizer, not by the witness. Synthetic blocks (Q4_1, Q5_1) exist because no "
                 "file this book ships uses those formats; they are decoded by the same witness that "
                 "produced them, so they can catch a transcription error but not a shared misreading."),
        "formats": results,
        "error": report_error,
        "outlier": out_block,
        "verdict": "agree" if all(r["agreement"] != "DIFFERS" for r in results) else "disagree",
    }
    if args.real:
        facts_path, model_path = args.real[0].split(":", 1)
        report["artifact"] = json.loads(pathlib.Path(facts_path).read_text())["artifact"]
        report["artifact_sha256_check"] = hashlib.sha256(
            pathlib.Path(model_path).expanduser().read_bytes()).hexdigest()
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {pathlib.Path(args.out)}")


if __name__ == "__main__":
    main()
