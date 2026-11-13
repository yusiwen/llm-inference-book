#!/usr/bin/env python3
"""dotcheck.py - the dot-product contract, checked by values.

Three ways to compute the same dot product, and the error each one carries:

  exact     weights dequantized to f64, activations f64            - the definition
  float     weights dequantized to f32, activations f32, f32 fma   - the obvious implementation
  integer   activations quantized to Q8_0, nibbles times int8, i32 - what an engine actually runs

The question the chapter answers is not "which is fastest" (that is Chapter 7) but "what does the integer
path cost in accuracy, and where exactly does each part of that cost come from". The answer needs a
decomposition, so the script measures each stage separately:

  * the activation's own quantization error, on real vectors;
  * the dot product's error for the integer path against the f64 definition;
  * the f32 accumulation's error, which the naive path pays and the integer path does not;
  * the integer headroom, which decides whether a 16-bit accumulator is legal.

    uv run --with gguf python scripts/dotcheck.py \\
        --weights facts/gguf-qwen2.5-0.5b-q4_0.json:~/.cache/myfer/qwen2.5-0.5b-instruct-q4_0.gguf \\
        --out facts/dotcheck.json

What this cannot prove: that the f32 activations here are the ones a real forward pass produces. They are
dequantized rows of the model's own embedding matrix - real magnitudes, real distribution, and exactly what
layer 0 sees for a token that is one of those rows. Layer 12's activations are not this, and the chapter
says so rather than implying a measurement it did not make.
"""

import argparse
import hashlib
import json
import pathlib
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def f16(b, off):
    return struct.unpack_from("<e", b, off)[0]


def dequantize_q8_0(raw, nblocks):
    """(nblocks, 32) float32: one f16 scale and 32 signed bytes per block."""
    out = np.zeros((nblocks, 32), dtype=np.float32)
    for i in range(nblocks):
        blk = raw[i * 34:(i + 1) * 34]
        d = np.float32(f16(blk, 0))
        out[i] = d * np.frombuffer(blk, dtype=np.int8, count=32, offset=2).astype(np.float32)
    return out


def dequantize_q4_0(raw, nblocks):
    """(nblocks, 32) float32 array, Chapter 4's decoder, vectorised over blocks."""
    out = np.zeros((nblocks, 32), dtype=np.float32)
    for i in range(nblocks):
        blk = raw[i * 18:(i + 1) * 18]
        d = np.float32(f16(blk, 0))
        b = np.frombuffer(blk, dtype=np.uint8, count=16, offset=2)
        low = (b & 0x0F).astype(np.int32) - 8
        high = (b >> 4).astype(np.int32) - 8
        out[i, :16] = d * low
        out[i, 16:] = d * high
    return out


# Per-format: (block bytes, weights per block, decode, largest |q| the format can store)
LAYOUTS = {
    "Q4_0": (18, 32, dequantize_q4_0, 8),
    "Q8_0": (34, 32, dequantize_q8_0, 127),
}


def quantize_q8_0(values):
    """One Q8_0 row: an f16 scale per 32 values, int8 payload. Returns (int8 array, f16 scale array)."""
    v = np.asarray(values, dtype=np.float32)
    nblocks = v.size // 32
    q = np.zeros(v.size, dtype=np.int8)
    scales = np.zeros(nblocks, dtype=np.float16)
    for i in range(nblocks):
        blk = v[i * 32:(i + 1) * 32]
        amax = float(np.max(np.abs(blk)))
        d = np.float16(amax / 127.0) if amax else np.float16(0.0)
        scales[i] = d
        if d:
            q[i * 32:(i + 1) * 32] = np.round(blk / np.float32(d)).clip(-127, 127).astype(np.int8)
    return q, scales


def measure_pair(facts, raw_all, wdtype, args):
    """Run the accuracy decomposition for one (weight dtype x Q8_0 activation) pair."""
    block_bytes, per_block, decode, wmax = LAYOUTS[wdtype]
    tensor = next(t for t in facts["tensors"] if t["type"] == wdtype and len(t["dims"]) == 2
                  and t["dims"][0] % per_block == 0 and t["dims"][0] >= 512)
    row_len = tensor["dims"][0]
    nblocks = row_len // per_block
    nrows = min(args.rows, tensor["dims"][1])
    row_bytes = nblocks * block_bytes
    body = raw_all[tensor["abs"]:tensor["abs"] + row_bytes * nrows]

    # The activation vector: the dequantized first row of the same matrix. Real values, real scale, and a
    # stand-in for layer 0's input rather than a measurement of it.
    act = decode(body[:row_bytes], nblocks).reshape(-1).astype(np.float32)
    aq, ascale = quantize_q8_0(act)
    q8_back = (aq.reshape(-1, 32).astype(np.float32) * ascale.reshape(-1, 1).astype(np.float32)).reshape(-1)
    act_rel = float(np.linalg.norm(q8_back - act) / np.linalg.norm(act))
    act_max = float(np.max(np.abs(q8_back - act)) / np.max(np.abs(act)))

    exact, q8_act_path, float_path, int_path, norm_prod = [], [], [], [], []
    partial_max = 0
    for r in range(nrows):
        raw_row = body[r * row_bytes:(r + 1) * row_bytes]
        w = decode(raw_row, nblocks).reshape(-1)
        w64 = w.astype(np.float64)
        exact.append(float(w64 @ act.astype(np.float64)))
        norm_prod.append(float(np.linalg.norm(w64) * np.linalg.norm(act.astype(np.float64))))
        q8_act_path.append(float(w64 @ q8_back.astype(np.float64)))
        float_path.append(float(np.float32(np.sum(w.astype(np.float32) * act, dtype=np.float32))))

        total = np.float64(0.0)
        for b in range(nblocks):
            blk = raw_row[b * block_bytes:(b + 1) * block_bytes]
            dw = f16(blk, 0)
            if wdtype == "Q4_0":
                by = np.frombuffer(blk, dtype=np.uint8, count=16, offset=2)
                qw = np.concatenate([(by & 0x0F).astype(np.int32) - 8, (by >> 4).astype(np.int32) - 8])
            else:
                qw = np.frombuffer(blk, dtype=np.int8, count=32, offset=2).astype(np.int32)
            qa = aq[b * 32:(b + 1) * 32].astype(np.int32)
            partial = int(np.dot(qw, qa))               # exact in i32: this is the accumulator
            partial_max = max(partial_max, abs(partial))
            total += np.float64(partial) * np.float64(dw) * np.float64(ascale[b])
        int_path.append(float(total))

    exact_a = np.array(exact)
    scale = np.maximum(np.abs(exact_a), 1e-30)
    err = lambda a: np.abs(np.array(a) - exact_a) / scale
    err_float, err_int, err_act = err(float_path), err(int_path), err(q8_act_path)
    err_arith = np.abs(np.array(int_path) - np.array(q8_act_path)) / scale
    # A scale-free error: the absolute difference over the product of the two norms. The relative error
    # explodes on rows whose dot product happens to be near zero; this one does not, and it is the bound a
    # kernel can actually be held to.
    nrm = np.array(norm_prod)
    e_int_norm = np.abs(np.array(int_path) - exact_a) / nrm
    e_float_norm = np.abs(np.array(float_path) - exact_a) / nrm
    return {
        "pair": f"{wdtype} x Q8_0",
        "weights": {"tensor": tensor["name"], "row_len": row_len, "rows_used": nrows,
                    "blocks_per_row": nblocks},
        "activation": {"source": f"dequantized row 0 of {tensor['name']}",
                       "q8_relative_rms_error": act_rel,
                       "q8_max_error_over_row_max": act_max, "values": int(act.size)},
        "float_path_max_rel_error": float(np.max(err_float)),
        "float_path_median_rel_error": float(np.median(err_float)),
        "activation_only_max_rel_error": float(np.max(err_act)),
        "activation_only_median_rel_error": float(np.median(err_act)),
        "integer_vs_same_activations_max_rel_error": float(np.max(err_arith)),
        "integer_vs_same_activations_median_rel_error": float(np.median(err_arith)),
        "integer_error_over_norm_product_max": float(np.max(e_int_norm)),
        "integer_error_over_norm_product_median": float(np.median(e_int_norm)),
        "float_error_over_norm_product_max": float(np.max(e_float_norm)),
        "integer_path_max_rel_error": float(np.max(err_int)),
        "integer_path_median_rel_error": float(np.median(err_int)),
        "integer_worst_partial_sum": int(partial_max),
        "worst_case_partial_bound": per_block * wmax * 127,
        "fits_i16": per_block * wmax * 127 <= 32767,
        "exact_max_abs": float(np.max(np.abs(exact_a))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="facts.json:model.gguf, a file with Q4_0 weights")
    ap.add_argument("--pairs", default="Q4_0,Q8_0", help="weight dtypes to measure, comma separated")
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--out", default=str(ROOT / "facts" / "dotcheck.json"))
    args = ap.parse_args()

    facts_path, model_path = args.weights.split(":", 1)
    facts = json.loads(pathlib.Path(facts_path).read_text())
    model_path = pathlib.Path(model_path).expanduser()
    raw_all = model_path.read_bytes()

    pairs = []
    for wdtype in [p.strip() for p in args.pairs.split(",") if p.strip()]:
        if not any(t["type"] == wdtype for t in facts["tensors"]):
            print(f"{wdtype}: not in this file, skipped")
            continue
        pairs.append(measure_pair(facts, raw_all, wdtype, args))

    # The accumulator bound, for every format the reader might reach for, not only the measured ones.
    bounds = {}
    for name, (_, per_block, _, wmax) in LAYOUTS.items():
        worst = per_block * wmax * 127
        bounds[name] = {"weights_per_block": per_block, "max_weight": wmax,
                        "worst_case_partial": worst, "fits_i16": worst <= 32767}
    for name, wmax in (("Q5_0", 16), ("Q6_K", 32)):
        worst = 32 * wmax * 127
        bounds[name] = {"weights_per_block": 32, "max_weight": wmax,
                        "worst_case_partial": worst, "fits_i16": worst <= 32767}

    report = {
        "artifact": facts["artifact"],
        "artifact_sha256_check": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "pairs": pairs,
        "accumulator_bounds": bounds,
        "invocation": "uv run --with gguf python scripts/dotcheck.py " + " ".join(sys.argv[1:]),
        "note": ("Each pair is decomposed, because 'the integer path is less accurate' is not one statement: "
                 "the difference against the same quantized activations measures the arithmetic, and the "
                 "difference against f64 measures the activation's rounding. What is not measured: layer-12 "
                 "activations, or any device."),
    }
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    for p in pairs:
        print(f"{p['pair']:12s} rows {p['weights']['rows_used']:4d}  activation {p['activation']['q8_relative_rms_error']:.3%}"
              f"  float {p['float_path_max_rel_error']:.2e}  integer {p['integer_path_max_rel_error']:.2e}"
              f"  (arith {p['integer_vs_same_activations_max_rel_error']:.2e})"
              f"  worst partial {p['integer_worst_partial_sum']}  i16 {p['fits_i16']}"
              f"  e/norm {p['integer_error_over_norm_product_max']:.2e}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
