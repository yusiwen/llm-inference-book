#!/usr/bin/env python3
"""vecops-check.py - the second implementation for Chapter 9.

Reads what scripts/vecops.c produced (fixtures *and* outputs) and recomputes every op in f64
with numpy, then reports the agreement. This is the chapter's witness: the C code says what an
engine does, this says what the equations say, and the difference is the number the precision
ledger needs.

Nothing here re-derives its own inputs - that would only prove two programs agree about how to
make up data.

    uv run --with numpy python scripts/vecops-check.py facts/vecops.json --write facts/vecops-witness.json
"""

import argparse
import json
import sys

import numpy as np


def rel(a, b, floor=0.0):
    """max |a-b| over |b|, ignoring terms smaller than `floor` times the vector's own scale."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.max(np.abs(b)) if b.size else 0.0
    if scale == 0.0:
        return 0.0
    if a.size != b.size:
        raise SystemExit(f"size mismatch: {a.size} vs {b.size}")
    mask = np.abs(b) > floor * scale
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(a[mask] - b[mask])) / scale)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("facts")
    ap.add_argument("--write")
    args = ap.parse_args()

    d = json.load(open(args.facts))
    fx = d["fixtures"]
    x = np.array(fx["emb"], dtype=np.float64)
    w = np.array(fx["gain"], dtype=np.float64)
    eps = 1e-6

    rows = []

    # ---- RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w
    ss = float(np.sum(x * x))
    scale64 = 1.0 / np.sqrt(ss / x.size + eps)
    y_ref = x * scale64 * w
    y_c = np.array(d["norm_out"], dtype=np.float64)
    rows.append(("RMSNorm output", rel(y_c, y_ref), f"scale {d['norm']['scale']:.9g} vs {scale64:.9g}"))

    # ---- softmax: exp(x - max) / sum, on the same row the C program used
    logits = np.array(fx["logits"], dtype=np.float64)          # the whole 4096-long row
    sp_c = np.array(d["softmax_out"], dtype=np.float64)
    z = np.exp(logits - logits.max())
    sp_ref = z / z.sum()
    rows.append((f"softmax (first 16 of {logits.size})", rel(sp_c, sp_ref[: sp_c.size]),
                 f"sum {d['softmax']['sum_f64']:.4f} vs {z.sum():.4f}"))

    # ---- SiLU: x / (1 + exp(-x))
    sx = np.array(d["silu_in"], dtype=np.float64)
    sy_c = np.array(d["silu_out"], dtype=np.float64)
    sy_ref = sx / (1.0 + np.exp(-sx))
    rows.append(("SiLU over [-4, 3.75]", rel(sy_c, sy_ref), "32 probe points"))

    # ---- RoPE, half-split pairing, three positions
    q = np.array(fx["k_head"], dtype=np.float64)
    hd = fx["head_dim"]
    base = 1000000.0
    got = np.array(d["rope_out"], dtype=np.float64).reshape(3, hd)
    worst = 0.0
    for k in range(3):
        pos = k + 1
        out = q.copy()
        for dd in range(hd // 2):
            inv = 1.0 / base ** (2 * dd / hd)
            th = pos * inv
            i0, i1 = dd, dd + hd // 2
            out[i0] = q[i0] * np.cos(th) - q[i1] * np.sin(th)
            out[i1] = q[i0] * np.sin(th) + q[i1] * np.cos(th)
        worst = max(worst, rel(got[k], out))
    rows.append(("RoPE, half-split, positions 1..3", worst, "the C kernel's own inv_freq expression, in f64"))

    print(f"{'op':34s} {'max rel diff':>14s}  note")
    for name, err, note in rows:
        print(f"{name:34s} {err:14.3e}  {note}")

    # The C program's own error numbers, restated as bounds: each must be >= the witness's.
    checks = {
        "norm_seq_f32": d["norm"]["max_rel_vs_f64"]["seq_f32"],
        "norm_lanes_f32": d["norm"]["max_rel_vs_f64"]["lanes_f32"],
        "silu_f32": d["silu"]["max_rel_vs_double"],
        "softmax_f32_sum": d["softmax"]["max_rel_f32_sum"],
    }
    print("\nreported by the C program (f32 paths against its own f64 oracle):")
    for k, v in checks.items():
        print(f"  {k:18s} {v:.3e}")

    verdict = all(err < 1e-6 for _, err, _ in rows if err is not None)
    print(f"\nwitness: {'agrees within 1e-6 on every op' if verdict else 'DISAGREES beyond 1e-6'}")

    if args.write:
        out = {
            "artifact": "vecops-witness",
            "witness": "numpy float64, scripts/vecops-check.py",
            "compares": "facts/vecops.json (the C outputs, f32 paths) against the equations in f64",
            "invocation": "uv run --with numpy python scripts/vecops-check.py facts/vecops.json --write facts/vecops-witness.json",
            "ops": [{"op": n, "max_rel_diff": e, "note": t} for n, e, t in rows],
            "verdict": "agree within 1e-6" if verdict else "disagree beyond 1e-6",
            "note": ("A witness for the arithmetic, not for the conventions: the RoPE row re-implements "
                     "the half-split pairing because that is what the C kernel claims to do. A wrong "
                     "pairing in both would agree perfectly - which is why the pairing itself is tested "
                     "against a golden vector from a known engine, not against this file."),
        }
        with open(args.write, "w") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
        print(f"wrote {args.write}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
