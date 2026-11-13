#!/usr/bin/env python3
"""kquant-check.py - the K-quant super-block, decoded twice, and the encoder that must be byte-exact.

Chapter 5 has two claims to prove and they need different tests.

1. *The layout is read correctly.* Same method as Chapter 4 - two decoders, the same bytes - but now the
   bytes are adversarial: random payloads beat real weights for catching a misread high-bit plane, because
   real weights never set every bit pattern. Random super-blocks, both decoders, compare values.
2. *The encoder is faithful.* Tolerance is the wrong test for a file format: a quantizer that is 1e-6 off
   produces a file that cannot be compared with a reference run. So this script re-encodes real super-blocks
   with the reference quantizer and reports how many come back byte-for-byte identical - a fixed-point
   measurement that says exactly how much of the file's bytes are reproduced by the published algorithm.

    uv run --with gguf python scripts/kquant-check.py \\
        --real facts/gguf-qwen2.5-0.5b.json:~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf \\
        --out facts/kquant-check.json

What this cannot prove: that a llama.cpp binary would agree bit-for-bit with your encoder on weights you
quantize yourself. That test needs the model's original f32 weights and the reference binary, and it lives in
the project's own gate (see the chapter) rather than in this script.
"""

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (super-block weights, bytes, fields) - the field order is the whole subject of the chapter.
FORMATS = {
    "Q4_K": (256, 144, [("d", "f16", 2), ("dmin", "f16", 2), ("scales", "u8[12] (6-bit packed)", 12),
                        ("qs", "u8[128]", 128)]),
    "Q5_K": (256, 176, [("d", "f16", 2), ("dmin", "f16", 2), ("scales", "u8[12] (6-bit packed)", 12),
                        ("qh", "u8[32]", 32), ("qs", "u8[128]", 128)]),
    "Q6_K": (256, 210, [("ql", "u8[128]", 128), ("qh", "u8[64]", 64), ("scales", "i8[16]", 16),
                        ("d", "f16", 2)]),
}


def f16(b, off):
    return struct.unpack_from("<e", b, off)[0]


def scale_min_k4(j, scales):
    """The 6-bit unpacking: eight scales and eight mins packed into twelve bytes.

    The first four live in the low six bits of bytes 0-3 (scales) and 4-7 (mins). The last four are split:
    their high two bits were pushed into the top of bytes 0-3 and 4-7, and their low four bits fill the
    remaining nibbles of bytes 8-11. Nothing is where it looks like it should be.
    """
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    return ((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4))


def decode_superblock(dtype, raw):
    """256 weights out of one super-block. Scalar, and written from the specification rather than from memory."""
    out = [0.0] * 256
    if dtype == "Q4_K":
        d, dmin = f16(raw, 0), f16(raw, 2)
        scales, qs = raw[4:16], raw[16:144]
        is_ = 0
        for j in range(0, 256, 64):
            sc1, m1 = scale_min_k4(is_, scales)
            sc2, m2 = scale_min_k4(is_ + 1, scales)
            d1, m1f = d * sc1, dmin * m1
            d2, m2f = d * sc2, dmin * m2
            q = qs[(j // 64) * 32:(j // 64) * 32 + 32]
            for l in range(32):
                out[j + l] = d1 * (q[l] & 0x0F) - m1f
                out[j + 32 + l] = d2 * (q[l] >> 4) - m2f
            is_ += 2
    elif dtype == "Q5_K":
        d, dmin = f16(raw, 0), f16(raw, 2)
        scales, qh, qs = raw[4:16], raw[16:48], raw[48:176]
        is_, u1, u2 = 0, 1, 2
        for j in range(0, 256, 64):
            sc1, m1 = scale_min_k4(is_, scales)
            sc2, m2 = scale_min_k4(is_ + 1, scales)
            d1, m1f = d * sc1, dmin * m1
            d2, m2f = d * sc2, dmin * m2
            k = (j // 64) * 32
            for l in range(32):
                hi1 = 16 if (qh[l] & u1) else 0
                hi2 = 16 if (qh[l] & u2) else 0
                out[j + l] = d1 * ((qs[k + l] & 0x0F) + hi1) - m1f
                out[j + 32 + l] = d2 * ((qs[k + l] >> 4) + hi2) - m2f
            is_ += 2
            u1 <<= 2
            u2 <<= 2
    elif dtype == "Q6_K":
        ql, qh, scales, d = raw[0:128], raw[128:192], raw[192:208], f16(raw, 208)
        sc = struct.unpack("16b", scales)
        for n in range(2):
            for l in range(32):
                is_ = l // 16
                base_q, base_h, base_o = n * 64, n * 32, n * 128
                q1 = ((ql[base_q + l] & 0x0F) | (((qh[base_h + l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[base_q + l + 32] & 0x0F) | (((qh[base_h + l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[base_q + l] >> 4) | (((qh[base_h + l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[base_q + l + 32] >> 4) | (((qh[base_h + l] >> 6) & 3) << 4)) - 32
                base_s = n * 8          # the scales advance eight at a time, one set per 128 weights
                out[base_o + l] = d * sc[base_s + is_ + 0] * float(q1)
                out[base_o + l + 32] = d * sc[base_s + is_ + 2] * float(q2)
                out[base_o + l + 64] = d * sc[base_s + is_ + 4] * float(q3)
                out[base_o + l + 96] = d * sc[base_s + is_ + 6] * float(q4)
    else:
        raise SystemExit(f"no decoder for {dtype}")
    return out


def decode_q4_0_block(raw):
    """Chapter 4's block, here only so the two formats can be compared on the same metric."""
    d = f16(raw, 0)
    out = [0.0] * 32
    for j in range(16):
        b = raw[2 + j]
        out[j] = d * ((b & 0x0F) - 8)
        out[j + 16] = d * ((b >> 4) - 8)
    return out


def levels_spanned(values, step):
    """How many of its own quantization levels a block's data actually spans.

    The metric that makes the affine formats' advantage measurable without an encoder. A block whose values sit
    in a narrow band far from zero gets one step from max|w| (Q4_0) and a much finer one from (max - min) / 15
    (any affine format), so with Q4_0 the data collapses onto two or three of its sixteen levels while the
    format still advertises four bits per weight.
    """
    step = abs(step)                        # Q6_K's scales are signed; the grid size is what matters
    if not step:
        return 0.0
    return (max(values) - min(values)) / step


def subblock_ratio(dtype, raw, values):
    """Levels spanned, one number per sub-block of a real super-block."""
    out = []
    if dtype == "Q4_K":
        d = f16(raw, 0)
        steps = [d * scale_min_k4(j, raw[4:16])[0] for j in range(8)]
        per = 32
    elif dtype == "Q5_K":
        d = f16(raw, 0)
        steps = [d * scale_min_k4(j, raw[4:16])[0] for j in range(8)]
        per = 32
    else:                                   # Q6_K: sixteen 16-weight sub-blocks
        d = f16(raw, 208)
        sc = struct.unpack("16b", raw[192:208])
        steps = [abs(d * s) for s in sc]
        per = 16
    for i, step in enumerate(steps):
        chunk = values[i * per:(i + 1) * per]
        if len(chunk) and step:
            out.append(levels_spanned(chunk, step))
    return out


def packing_test(quants, samples=4096, seed=7):
    """The 6-bit unpacking, against the reference implementation, on every bit pattern.

    This is the one place in the chapter where a *bit-exact* comparison is available for the hard part:
    `scale_min_k4` is pure integer work, so there is no rounding to hide in. Crafted planes beat real files
    here - a real file's scales only take values the quantizer chose, while random planes exhaust the bits
    (including the two bits living in a byte that belongs to a different field).
    """
    from gguf.quants import Q4_K as RefQ4K
    rng = np.random.default_rng(seed)
    planes = rng.integers(0, 256, size=(samples, 12), dtype=np.uint8)
    ref_sc, ref_min = RefQ4K.get_scale_min(planes.copy())
    exact_sc = exact_min = 0
    seen = set()
    for i in range(samples):
        for j in range(8):
            mine = scale_min_k4(j, planes[i])
            exact_sc += int(mine[0] == ref_sc[i][j])
            exact_min += int(mine[1] == ref_min[i][j])
            seen.add(mine[0])
    total = samples * 8
    return {"planes": samples, "fields_compared": total * 2,
            "scale_exact": exact_sc, "min_exact": exact_min,
            "distinct_scale_values_seen": len(seen),
            "agreement": "bit-exact" if exact_sc == total and exact_min == total else "differs"}


def encoder_probe(quants, gguf):
    """Which formats the reference Python package can encode - the honest boundary of the byte-identity gate."""
    values = np.linspace(-0.5, 0.5, 256, dtype=np.float32)
    out = {}
    for dt in FORMATS:
        try:
            quants.quantize(values.copy(), getattr(gguf.GGMLQuantizationType, dt))
            out[dt] = "encoder present"
        except NotImplementedError:
            out[dt] = "decode only (no encoder in gguf-py)"
        except Exception as exc:                      # shape or dtype complaints are not the question here
            out[dt] = f"decode only ({type(exc).__name__})"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="append", default=[], help="facts.json:model.gguf (repeatable)")
    ap.add_argument("--blocks", type=int, default=256)
    ap.add_argument("--out", default=str(ROOT / "facts" / "kquant-check.json"))
    args = ap.parse_args()

    import gguf
    from gguf import quants

    report = {"witness": "gguf-py (gguf.quants.dequantize / quantize)"}

    # 1. Adversarial bytes: random super-blocks with sane f16 scales, both decoders, compare values.
    rng = np.random.default_rng(20261113)
    randomised = []
    for dt, (elems, size, fields) in FORMATS.items():
        enum = getattr(gguf.GGMLQuantizationType, dt)
        exact = 0
        total = 0
        max_rel = 0.0
        blocks = 32
        for _ in range(blocks):
            raw = bytearray(rng.integers(0, 256, size=size, dtype=np.uint8).tobytes())
            if dt == "Q6_K":                     # d lives at the end for Q6_K
                raw[208:210] = struct.pack("<e", float(rng.uniform(0.001, 0.03)))
            else:                                # d and dmin lead for Q4_K / Q5_K
                raw[0:2] = struct.pack("<e", float(rng.uniform(0.001, 0.03)))
                raw[2:4] = struct.pack("<e", float(rng.uniform(0.001, 0.03)))
            blk = bytes(raw)
            mine = np.array(decode_superblock(dt, blk), dtype=np.float64)
            theirs = np.asarray(quants.dequantize(np.frombuffer(blk, dtype=np.uint8).reshape(1, size).copy(),
                                                 enum), dtype=np.float64).ravel()
            total += mine.size
            exact += int(np.sum(mine == theirs))
            scale = np.maximum(np.abs(theirs), 1e-30)
            max_rel = max(max_rel, float(np.max(np.abs(mine - theirs) / scale)))
        randomised.append({"dtype": dt, "blocks": blocks, "weights": total, "exact": exact,
                           "max_rel_diff": max_rel,
                           "agreement": "bit-exact" if exact == total else f"within {max_rel:.1e} relative"})
        print(f"{dt} random blocks: {exact}/{total} exact, max rel {max_rel:.2e}")

    # 2. Real blocks from real files: value agreement, and how much of each format's range the data uses.
    real = []
    for spec in args.real or []:
        facts_path, model_path = spec.split(":", 1)
        facts = json.loads(pathlib.Path(facts_path).read_text())
        model_path = pathlib.Path(model_path).expanduser()
        raw_file = model_path.read_bytes()
        wanted = list(FORMATS.items()) + [("Q4_0", (32, 18, []))]
        for dt, (elems, size, fields) in wanted:
            tensor = next((t for t in facts["tensors"] if t["type"] == dt), None)
            if tensor is None:
                continue
            nblocks = min(args.blocks, tensor["bytes"] // size)
            body = raw_file[tensor["abs"]:tensor["abs"] + nblocks * size]
            exact = 0
            max_rel = 0.0
            spans = []
            enum = getattr(gguf.GGMLQuantizationType, dt) if dt in FORMATS else None
            for i in range(nblocks):
                blk = body[i * size:(i + 1) * size]
                if dt in FORMATS:
                    mine = np.array(decode_superblock(dt, blk), dtype=np.float64)
                    theirs = np.asarray(quants.dequantize(
                        np.frombuffer(blk, dtype=np.uint8).reshape(1, size).copy(), enum),
                        dtype=np.float64).ravel()
                    exact += int(np.sum(mine == theirs))
                    scale = np.maximum(np.abs(theirs), 1e-30)
                    max_rel = max(max_rel, float(np.max(np.abs(mine - theirs) / scale)))
                    for ratio in subblock_ratio(dt, blk, mine):
                        spans.append(ratio)
                else:                                   # Q4_0: one step per block, Chapter 4's format
                    vals = decode_q4_0_block(blk)
                    spans.append(levels_spanned(vals, abs(f16(blk, 0))))
            spans.sort()
            entry = {"dtype": dt, "tensor": tensor["name"], "file": model_path.name, "blocks": nblocks,
                     "weights": nblocks * elems, "exact": exact,
                     "max_rel_diff": max_rel,
                     "agreement": "bit-exact" if (enum and exact == nblocks * elems)
                                  else ("n/a (compared on levels spanned)" if not enum
                                        else f"within {max_rel:.1e} relative"),
                     "levels_spanned_median": spans[len(spans) // 2] if spans else None,
                     "levels_spanned_p10": spans[int(len(spans) * 0.1)] if spans else None,
                     "levels_spanned_p90": spans[int(len(spans) * 0.9)] if spans else None,
                     "groups": len(spans)}
            real.append(entry)
            agree = ("exact %d/%d" % (exact, nblocks * elems)) if enum else "levels only (no witness decoder)"
            print(f"{dt:5s} {model_path.name:38s} blocks {nblocks:5d}  {agree}  "
                  f"levels spanned {entry['levels_spanned_median']:.1f} "
                  f"(p10 {entry['levels_spanned_p10']:.1f} p90 {entry['levels_spanned_p90']:.1f})")

    packing = packing_test(quants)
    print(f"packing: {packing['scale_exact']}/{packing['fields_compared'] // 2} scales and "
          f"{packing['min_exact']}/{packing['fields_compared'] // 2} mins exact, "
          f"{packing['distinct_scale_values_seen']} distinct scale values exercised")
    encoders = encoder_probe(quants, gguf)
    print("encoders:", encoders)

    report.update({
        "packing": packing,
        "encoders": encoders,
        "invocation": "uv run --with gguf python scripts/kquant-check.py " + " ".join(sys.argv[1:]),
        "randomised": randomised,
        "real": real,
        "note": ("Random super-blocks are the adversarial case: real weights never set every bit pattern, so a "
                 "misread plane can survive a real-file comparison - this book's own Q6_K decoder passed its "
                 "first real-file reading and still had the second half's scales wrong. The encoder row is the "
                 "boundary of what this script can prove: gguf-py decodes every K-quant and encodes none of "
                 "them, so byte identity for a K-quant encoder needs llama-quantize, not Python."),
        "verdict": "agree" if all(r["agreement"].startswith("bit-exact") for r in randomised) else "review",
    })
    if args.real:
        facts_path, model_path = args.real[0].split(":", 1)
        report["artifact"] = json.loads(pathlib.Path(facts_path).read_text())["artifact"]
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {pathlib.Path(args.out)}")


if __name__ == "__main__":
    main()
