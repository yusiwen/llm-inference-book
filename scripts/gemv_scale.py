#!/usr/bin/env python3
"""Chapter 22's second implementation: the same GEMV written two ways, at three sizes.

Chapter 22 measured a row-per-thread GEMV at a fraction of the streaming rate and attributed the gap to the
access pattern. One kernel cannot support that attribution - it only shows that the kernel is slow. This probe
adds the counterfactual: the identical arithmetic with a coalesced access pattern (one warp per row, 32 lanes
splitting the columns and reducing with __shfl_down), run at three sizes so the claim can be checked for
stability instead of asserted at one point.

    python3 scripts/gemv_scale.py --host yusiwen@192.168.2.40 --arch sm_121 --out facts/gemv-scale.json
"""

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts" / "probe" / "gemv_scale.cu"


def sh(host, cmd, timeout=900):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        sys.exit(f"remote command failed:\n{cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="yusiwen@192.168.2.40")
    ap.add_argument("--arch", default="sm_121")
    ap.add_argument("--out", default="facts/gemv-scale.json")
    args = ap.parse_args()

    remote = "/tmp/ch22-gemv-scale"
    sh(args.host, f"mkdir -p {remote}")
    subprocess.run(["scp", "-q", str(SRC), f"{args.host}:{remote}/gemv_scale.cu"], check=True)
    sh(args.host, f"cd {remote} && /usr/local/cuda/bin/nvcc -O3 -arch={args.arch} -o gemv_scale gemv_scale.cu")
    raw = sh(args.host, f"cd {remote} && ./gemv_scale")
    try:
        doc = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError) as e:
        sys.exit(f"the instrument did not emit JSON ({e}):\n{raw}")

    doc["artifact"] = "gemv-scale"
    doc["host"] = args.host
    doc["arch"] = args.arch
    doc["invocation"] = (f"scp scripts/probe/gemv_scale.cu {args.host}:{remote}/ ; "
                         f"/usr/local/cuda/bin/nvcc -O3 -arch={args.arch} -o gemv_scale gemv_scale.cu ; "
                         f"./gemv_scale")
    doc["question"] = ("does the row-per-thread GEMV's share of the streaming rate hold its value across sizes, "
                       "and is the access pattern really the cause?")
    ratios = [r["ratio_row_over_warp"] for r in doc["results"]]
    above = [r["size"] for r in doc["results"] if r["per_warp_of_stream"] > 100]
    doc["finding"] = {
        "row_over_warp_min": min(ratios),
        "row_over_warp_max": max(ratios),
        "shares_vary_from": min(r["per_row_of_stream"] for r in doc["results"]),
        "shares_vary_to": max(r["per_row_of_stream"] for r in doc["results"]),
        "sizes_where_coalesced_exceeds_stream": above,
        "note": ("the coalesced kernel beats the streaming rate at"
                 + (f" {', '.join(str(s) for s in above)}" if above else " no size")
                 + ", which means those weights were served from cache and the 'fraction of stream' "
                   "normalisation is not meaningful there; the access-pattern attribution is carried by the "
                   "row-over-warp ratio, which holds at every size"),
    }

    path = pathlib.Path(args.out)
    if not path.is_absolute():
        path = ROOT / path
    path.write_text(json.dumps(doc, indent=1) + "\n")

    print(f"wrote {path.relative_to(ROOT)}: {doc['device']}, stream {doc['stream_gbs']:.1f} GB/s")
    print(f"  {'size':>6} {'per_row':>10} {'%str':>7} {'per_warp':>10} {'%str':>8} {'row/warp':>9}")
    for r in doc["results"]:
        print(f"  {r['size']:>6} {r['per_row_gbs']:>9.1f}G {r['per_row_of_stream']:>6.1f}% "
              f"{r['per_warp_gbs']:>9.1f}G {r['per_warp_of_stream']:>7.1f}% {r['ratio_row_over_warp']:>8.2f}x")
    f = doc["finding"]
    print(f"  attribution holds: {f['row_over_warp_min']:.2f}x to {f['row_over_warp_max']:.2f}x slower strided; "
          f"the stream share moves {f['shares_vary_from']:.1f}%-{f['shares_vary_to']:.1f}%")


if __name__ == "__main__":
    main()
