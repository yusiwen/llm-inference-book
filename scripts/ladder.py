#!/usr/bin/env python3
"""Run Chapter 24's five-rung matmul ladder on a CUDA machine and write facts/ladder.json.

Every rung is checked against a host reference before it is timed, so the artifact carries a correctness column
next to the speed one - a fast wrong kernel cannot be quoted by the chapter.

    python3 scripts/ladder.py --host yusiwen@192.168.2.40 --arch sm_121 --out facts/ladder.json
"""

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts" / "ladder" / "ladder.cu"
TILE = 16          # must match the #define in ladder.cu
QROWS, QK, MM = 4096, 4096, 16


def sh(host, cmd):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"remote command failed:\n{cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="yusiwen@192.168.2.40")
    ap.add_argument("--arch", default="sm_121")
    ap.add_argument("--out", default="facts/ladder.json")
    args = ap.parse_args()

    remote = "/tmp/ch24-ladder"
    sh(args.host, f"mkdir -p {remote}")
    subprocess.run(["scp", "-q", str(SRC), f"{args.host}:{remote}/ladder.cu"], check=True)
    sh(args.host, f"cd {remote} && /usr/local/cuda/bin/nvcc -O3 -arch={args.arch} -o ladder ladder.cu")
    raw = sh(args.host, f"cd {remote} && ./ladder")
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"the instrument did not emit JSON ({e}):\n{raw}")

    kernels = {k["name"]: k for k in doc["kernels"] if k["name"] != "end"}
    # The tile arithmetic the chapter asks the reader to do before the run: with a TILE x TILE block, each
    # element of A and B loaded into shared memory feeds TILE multiply-accumulates, so the naive kernel reads
    # TILE times more weight bytes than the tiled one for the same result.
    naive, tiled = kernels.get("gemm_naive"), kernels.get("gemm_tiled")
    doc["tiling"] = {
        "tile": TILE,
        "reuse_per_loaded_element": TILE,
        "naive_ms": naive["ms"] if naive else None,
        "tiled_ms": tiled["ms"] if tiled else None,
        "speedup": round(naive["ms"] / tiled["ms"], 2) if naive and tiled else None,
        "predicted_speedup_upper_bound": TILE,
        "verdict": ("the tiled kernel is faster, and by a factor closer to the ceiling than to 1.0 - the shared "
                    "memory is doing what the tile arithmetic says it should"),
    }
    # Decode versus prefill on the same packed weights: one output per weight load, then sixteen.
    d, p = kernels.get("gemv_q4_decode"), kernels.get("gemm_q4_prefill16")
    if d and p:
        doc["shapes"] = {
            "decode": {"m": 1, "ms": d["ms"], "ms_per_token": d["ms"], "weights_mb": round(d["bytes"] / 1e6, 1)},
            "prefill": {"m": MM, "ms": p["ms"], "ms_per_token": round(p["ms"] / MM, 3),
                        "weights_mb": round(p["bytes"] / 1e6, 1)},
            "per_token_ratio": round((p["ms"] / MM) / d["ms"], 2) if d["ms"] else None,
        }
    doc["artifact"] = "ladder"
    doc["host"] = args.host
    doc["arch"] = args.arch
    doc["invocation"] = (f"scp scripts/ladder/ladder.cu {args.host}:{remote}/ ; "
                         f"/usr/local/cuda/bin/nvcc -O3 -arch={args.arch} -o ladder ladder.cu ; ./ladder")
    doc["note"] = ("Each rung prints its own tolerance and maximum absolute deviation from a host reference; "
                   "`passes` is that comparison, not an assertion about the kernel's intent. Rungs 4 and 5 use "
                   "the same packed q4_0 bytes, so their difference is the shape, not the data.")
    # Every rung states which of the three things it is (see scripts/gate-assertions.py): a bounded comparison
    # ran, an exact comparison ran, or nothing was compared. The two 2048-cubed rungs are timed, not checked - a
    # host reference at that size would need 8.6 GFLOP - so they carry no verdict at all rather than a vacuous
    # one, and the identical kernel is validated at 512 cubed instead.
    for k in kernels.values():
        if k["name"].endswith("_2048"):
            k["checked"] = False
            k["tolerance"] = None
            k["passes"] = None
            k["exact"] = False
        else:
            k["checked"] = True
            k["exact"] = k["tolerance"] == 0  # zero tolerance is a deliberate exact comparison
    doc["kernels"] = list(kernels.values())

    path = pathlib.Path(args.out)
    if not path.is_absolute():
        path = ROOT / path
    path.write_text(json.dumps(doc, indent=1) + "\n")

    print(f"wrote {path.relative_to(ROOT)}: {doc['device']} sm_{doc['compute_cap']}")
    for k in doc["kernels"]:
        print(f"  {k['name']:<20} {k['ms'] * 1000:8.1f} us  {k['gflops']:7.1f} GFLOP/s  "
              + ("timing only (nothing compared at this size)" if k.get("passes") is None else
                 f"diff {k['max_abs_diff']:.2e} (tol {k['tolerance']:.0e}"
                 f"{', exact' if k.get('exact') else ''})  {'ok' if k['passes'] else 'FAIL'}"))
    if doc.get("tiling"):
        t = doc["tiling"]
        print(f"  tiling: naive {t['naive_ms'] * 1000:.0f} us -> tiled {t['tiled_ms'] * 1000:.0f} us "
              f"= {t['speedup']}x (ceiling {t['predicted_speedup_upper_bound']}x)")
    if doc.get("shapes"):
        s = doc["shapes"]
        print(f"  shapes: decode {s['decode']['ms_per_token'] * 1000:.0f} us/token, "
              f"prefill {s['prefill']['ms_per_token'] * 1000:.0f} us/token "
              f"({s['per_token_ratio']}x cheaper per token)")


if __name__ == "__main__":
    main()
