#!/usr/bin/env python3
"""Run Chapter 23's three seam failures on a CUDA machine and write facts/cuda-seam.json.

The chapter's claim is that a seam has five crossings (allocate, own, report, wait, link) and that each one has a
cheap wrong answer that costs a day later. Two of them can be made to fail on purpose in a single program:

  * allocate - ask for a terabyte and see whether the error names the request or just "an error"
  * wait     - run one kernel for two seconds twice: once behind `cudaStreamSynchronize` (no timeout exists)
               and once behind a `cudaStreamQuery` poll with a deadline the host owns

The third (link) is read off the binary afterwards, because the question is what the executable *needs at run
time*, which is a property of the build, not of the program:

    python3 scripts/cuda_seam.py --host yusiwen@192.168.2.40 --arch sm_121 --out facts/cuda-seam.json
"""

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts" / "seam" / "seam.cu"


def sh(host, cmd, ok_codes=(0,)):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True)
    if r.returncode not in ok_codes:
        sys.exit(f"remote command failed ({r.returncode}):\n{cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout, r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="yusiwen@192.168.2.40")
    ap.add_argument("--arch", default="sm_121")
    ap.add_argument("--kernel-ms", type=int, default=2000)
    ap.add_argument("--deadline-ms", type=int, default=500)
    ap.add_argument("--out", default="facts/cuda-seam.json")
    args = ap.parse_args()

    remote_dir = "/tmp/ch23-seam"
    sh(args.host, f"mkdir -p {remote_dir}")
    subprocess.run(["scp", "-q", str(SRC), f"{args.host}:{remote_dir}/seam.cu"], check=True)
    sh(args.host, f"cd {remote_dir} && /usr/local/cuda/bin/nvcc -O3 -arch={args.arch} -o seam seam.cu")

    # Exit code 2 is this program's "the deadline was hit" signal, not a failure of the harness.
    out, code = sh(args.host, f"cd {remote_dir} && ./seam {args.kernel_ms} {args.deadline_ms}", ok_codes=(0, 2))
    try:
        doc = json.loads(out)
    except json.JSONDecodeError as e:
        sys.exit(f"the instrument did not emit JSON ({e}):\n{out}")

    # Link: what the binary needs at run time, and whether the matmul really lives inside it.
    ldd, _ = sh(args.host, f"cd {remote_dir} && ldd ./seam 2>&1 | sed 's/^[[:space:]]*//'")
    libs = [line for line in ldd.splitlines() if line.strip()]
    doc["link"] = {
        "ldd": libs,
        "libcudart": next((l.split("=>")[0].strip() for l in libs if "libcudart" in l), None),
        "libcuda_driver": next((l.split("=>")[0].strip() for l in libs if "libcuda.so" in l), None),
    }
    doc["kernel_ms"] = args.kernel_ms
    doc["host"] = args.host
    doc["arch"] = args.arch
    doc["process_exit"] = code
    doc["invocation"] = (f"scp scripts/seam/seam.cu {args.host}:{remote_dir}/ ; "
                         f"/usr/local/cuda/bin/nvcc -O3 -arch={args.arch} -o seam seam.cu ; "
                         f"./seam {args.kernel_ms} {args.deadline_ms} ; ldd ./seam")
    doc["note"] = ("The polled wait is the only one that can end on the host's terms: CUDA has no synchronise-"
                   "with-timeout, so a backend that must survive a wedged device polls `cudaStreamQuery` "
                   "against a clock it owns. The blocking wait is measured here as the thing that cannot.")

    path = pathlib.Path(args.out)
    if not path.is_absolute():
        path = ROOT / path
    path.write_text(json.dumps(doc, indent=1) + "\n")

    a, b, w = doc["alloc"], doc["blocking_wait"], doc["polled_wait"]
    print(f"wrote {path.relative_to(ROOT)}: {doc['device']} sm_{doc['compute_cap']}")
    print(f"  alloc        {a['requested_bytes'] / (1 << 40):.0f} TiB -> {a['error_name']}: {a['error_string'][:60]}")
    print(f"  blocking     kernel {doc['kernel_ms']} ms -> host blocked {b['blocked_ms']:.0f} ms")
    print(f"  polled       deadline {w['deadline_ms']:.0f} ms -> {w['elapsed_ms']:.0f} ms, {w['outcome']}")
    print(f"  link         libcudart: {doc['link']['libcudart'] or 'not linked'}")


if __name__ == "__main__":
    main()
