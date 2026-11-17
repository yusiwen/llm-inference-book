#!/usr/bin/env python3
"""Run the streaming-bandwidth instrument and write facts/membw.json.

Chapter 1 computes the decode ceiling as `model bytes / machine bandwidth`, so the bandwidth is an input to the
book's first number - and until this script existed it was the one input with no artifact behind it: the page
said "the M4 Pro's CPU-side measured read bandwidth is 198 GB/s" and nothing in the tree could reproduce it.
`scripts/membw.c` was there all along; no one had made it emit a fact.

    python3 scripts/membw-run.py --out facts/membw.json

The binary is compiled on this machine with clang -O2; the artifact records the compiler, the size, the thread
count and every run, and the chapter renders the median.
"""

import argparse
import json
import pathlib
import platform
import re
import statistics
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
FACTS = ROOT / "facts"
LINE = re.compile(r"membw (\d+) MiB x (\d+) arrays, (\d+) thread\(s\): read ([\d.]+) GB/s, copy ([\d.]+) GB/s")


def head_rev(checkout):
    return subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=1024, help="working set, MiB per array")
    ap.add_argument("--threads", default="8,10",
                    help="comma-separated thread counts; the machine table's history is why this is a sweep")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--cc", default="clang")
    ap.add_argument("--out", default=str(FACTS / "membw.json"))
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        binary = pathlib.Path(td) / "membw"
        build = subprocess.run([args.cc, "-O2", "-o", str(binary), str(ROOT / "scripts" / "membw.c"), "-lm"],
                               capture_output=True, text=True)
        if build.returncode:
            sys.exit(build.stderr)
        configs = []
        for threads in [int(t) for t in str(args.threads).split(",")]:
            runs = []
            for _ in range(args.runs):
                out = subprocess.run([str(binary), str(args.mib), str(threads)],
                                     capture_output=True, text=True)
                m = LINE.search(out.stdout)
                if not m:
                    sys.exit(out.stdout + out.stderr)
                runs.append({"read_gb_s": float(m.group(4)), "copy_gb_s": float(m.group(5)),
                             "arrays": int(m.group(2))})
            configs.append({
                "threads": threads,
                "read_gb_s": round(statistics.median(r["read_gb_s"] for r in runs), 1),
                "copy_gb_s": round(statistics.median(r["copy_gb_s"] for r in runs), 1),
                "read_min": min(r["read_gb_s"] for r in runs),
                "read_max": max(r["read_gb_s"] for r in runs),
                "runs": runs})

    # The highest thread count is the headline; the sweep is what shows how much of the number is the machine
    # and how much is the thread count, which is the question a reader with the same laptop will ask.
    main_cfg = max(configs, key=lambda c: c["threads"])
    read, copy = main_cfg["read_gb_s"], main_cfg["copy_gb_s"]
    doc = {
        "artifact": "membw",
        "geometry": {"mib": args.mib, "threads": main_cfg["threads"], "arrays": main_cfg["runs"][0]["arrays"]},
        "read_gb_s": read,
        "copy_gb_s": copy,
        "read_min": main_cfg["read_min"],
        "read_max": main_cfg["read_max"],
        "runs": main_cfg["runs"],
        "configs": configs,
        "compiler": f"{args.cc} -O2",
        "machine": f"{platform.system()} {platform.release()}, {platform.machine()}",
        "invocation": (f"{args.cc} -O2 -o membw scripts/membw.c -lm ; "
                       f"./membw {args.mib} N  for N in {args.threads}, x{args.runs} runs each"),
        "note": ("A streaming read of two arrays, one pass, no model involved: this is the speed of light the "
                 "kernels in Chapters 6-9 are measured against, and the denominator of Chapter 1's ceiling."),
        "provenance": {"statistic": f"median of {args.runs} runs of the C loop; min and max are kept too, because "
                                    f"a bandwidth measured once on a laptop is a rumour"},
    }
    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}: read {read:.1f} GB/s, copy {copy:.1f} GB/s "
          f"(runs {doc['read_min']:.1f}-{doc['read_max']:.1f})")


if __name__ == "__main__":
    main()
