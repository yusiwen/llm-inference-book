#!/usr/bin/env python3
"""Compile and run Chapter 22's launch sheet on a CUDA machine and write facts/cuda-launch.json.

The chapter's method is "predict, then launch": every row of the table is a byte count divided by the
machine's streaming bandwidth, written down *before* the kernel runs. This script produces both sides of that
comparison on one machine, so the chapter can print the prediction beside the measurement and talk about the
rows where they disagree.

    python3 scripts/cuda_launch.py --host yusiwen@192.168.2.40 --arch sm_121 --out facts/cuda-launch.json

It compiles `scripts/cuda/launch_sheet.cu` on the remote host with that host's own `nvcc`, runs it, and reads
its JSON. The bandwidth used for the prediction is the machine's own streaming copy, not a number from another
chapter - a prediction that borrows its constant is not a prediction about this machine.
"""

import argparse
import json
import pathlib
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts" / "cuda" / "launch_sheet.cu"


def sh(host, cmd):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"remote command failed:\n{cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="yusiwen@192.168.2.40")
    ap.add_argument("--arch", default="sm_121")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="facts/cuda-launch.json")
    args = ap.parse_args()

    remote_dir = "/tmp/ch22-launch-sheet"
    sh(args.host, f"mkdir -p {remote_dir}")
    subprocess.run(["scp", "-q", str(SRC), f"{args.host}:{remote_dir}/launch_sheet.cu"], check=True)
    build = f"cd {remote_dir} && /usr/local/cuda/bin/nvcc -O3 -arch={args.arch} -o launch_sheet launch_sheet.cu"
    sh(args.host, build)
    raw = sh(args.host, f"cd {remote_dir} && ./launch_sheet {args.reps}")
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"the instrument did not emit JSON ({e}):\n{raw}")

    kernels = [k for k in doc["kernels"] if k["name"] != "end"]
    stream = next(k for k in kernels if k["name"] == "copy")["gb_s"]
    for k in kernels:
        k["predicted_ms"] = round(k["bytes"] / stream / 1e6, 5)  # bytes / (GB/s * 1e6) -> ms
        k["ratio"] = round(k["ms"] / k["predicted_ms"], 2) if k["predicted_ms"] else None
        k["pct_of_stream"] = round(k["gb_s"] / stream * 100, 1)

    out = {
        "artifact": "cuda-launch",
        "device": doc["device"],
        "compute_cap": doc["compute_cap"],
        "sm_count": doc["sm_count"],
        "max_sm_mhz": doc["max_sm_mhz"],
        "host": args.host,
        "arch": args.arch,
        "reps": doc["reps"],
        "block": doc["block"],
        "geometry": {"n": doc["n"], "m": doc["m"], "k": doc["k"]},
        "stream_gb_s": round(stream, 1),
        "kernels": kernels,
        "invocation": (f"scp scripts/cuda/launch_sheet.cu {args.host}:{remote_dir}/ ; "
                       f"/usr/local/cuda/bin/nvcc -O3 -arch={args.arch} -o launch_sheet launch_sheet.cu ; "
                       f"./launch_sheet {args.reps}"),
        "note": ("Predictions are `bytes / stream_gb_s`, where the stream rate is this machine's own copy "
                 "kernel - the first row of the table, measured, not quoted. A ratio above 1.0 means the "
                 "kernel took longer than its bytes allow, which is a statement about its access pattern."),
    }
    path = pathlib.Path(args.out)
    if not path.is_absolute():
        path = ROOT / path
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)}: {out['device']} sm_{out['compute_cap']}, "
          f"stream {out['stream_gb_s']} GB/s")
    for k in kernels:
        print(f"  {k['name']:<18} {k['ms']*1000:7.1f} us  {k['gb_s']:6.1f} GB/s  "
              f"{k['pct_of_stream']:5.1f}% of stream  {k['ratio']:5.2f}x prediction")


if __name__ == "__main__":
    main()
