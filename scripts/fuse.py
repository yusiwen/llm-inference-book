#!/usr/bin/env python3
"""fuse.py - Chapter 15's instrument: what the fusion pass does, and what it is worth.

The chapter makes three claims and this script is what makes them measurable rather than asserted:

  1. **The rewrite is real and it is applied.** The engine's op vocabulary and the
     `Mul(Silu(x), y) -> SwiGLU(x, y)` rule are read out of the pinned source, and the graph the engine
     actually ran is read out of its own trace: the prefill graph carries 24 `swiglu` nodes and no `mul`
     at all.
  2. **The rewrite leaves orphans behind.** Rewriting the `Mul` in place does not delete the `Silu`
     node, because deleting a node would renumber every id after it. So the fused graph still contains
     24 `silu` nodes - and the allocator gives them no buffer, so the scheduler skips them. Both halves
     are measured: the node counts from the graph, the executed ids from the trace.
  3. **The rewrite is a bandwidth claim, and bandwidth is not always the limit.** `scripts/fuse.c`
     measures the same 3-to-1 rewrite on silu and on relu, at four row counts, and the two disagree by
     an order of magnitude in what the fusion is worth - because one of them is waiting on `expf` and
     the other is waiting on memory. `scripts/fuse-check.py` recomputes the C's output in f64.

      python3 scripts/fuse.py BINARY MODEL --minfer /tmp/mf15 --out facts/fusion.json

`BINARY` is the engine's CLI at the book's pinned baseline; `MODEL` is the GGUF the chapter measures;
`--minfer` is a worktree of that baseline, used to read the op vocabulary and to run its own tests.
"""

import argparse
import collections
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import graph as bookgraph  # noqa: E402  (Chapter 14's driver: same CLI invocation, same trace format)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPT = "The capital of France is"
STEPS = 5
SILU = "silu"
SWIGLU = "swiglu"


def matched(pattern, text, what, flags=0):
    """A regex result or a hard stop: a missing match here means the source moved, not that the answer is 0."""
    m = re.search(pattern, text, flags)
    if m is None:
        raise SystemExit(f"could not find {what} in the instrument's output - the source moved")
    return m


def shell(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(map(str, cmd))} failed ({proc.returncode}):\n{proc.stderr[-1500:]}")
    return proc.stdout


def vocabulary(minfer):
    """The op vocabulary: how many variants, how many carry a payload, how many are fusion targets."""
    ops_src = (minfer / "src/graph/ops.rs").read_text()
    start = ops_src.index("pub enum Op {")
    end = ops_src.index("\n}", start)
    variants = []
    for line in ops_src[start:end].split("\n"):
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("#") or s.startswith("pub enum"):
            continue
        m = re.match(r"([A-Z][A-Za-z0-9]*)\b(.*)", s)
        if m:
            variants.append((m.group(1), m.group(2).strip()[:1] in ("{", "(")))
    fused_body = matched(r"pub enum FusedOp \{(.*?)\}", ops_src, "the FusedOp enum", re.S).group(1)
    fused_variants = [v for v in re.findall(r"^\s*([A-Z][A-Za-z0-9]*)", fused_body, re.M)]
    # a variant marked #[cfg_attr(not(test), allow(dead_code))] is in the vocabulary for parity but is
    # never emitted by a supported architecture - the doc comments say what would emit it
    never_emitted = [v for v, payload in variants if not payload and v in ("Scale", "Softmax")] or \
                    [v for v, _ in variants if v in ("Scale", "Softmax")]
    return {
        "op_variants": len(variants),
        "payload_variants": sum(1 for _, payload in variants if payload),
        "unit_variants": sum(1 for _, payload in variants if not payload),
        "variants": [v for v, _ in variants],
        "unit_variant_names": [v for v, payload in variants if not payload],
        "fused_op_variants": len(fused_variants),
        "fused_ops": fused_variants,
        "never_emitted": never_emitted,
    }


def structure(trace):
    """Graph node counts and executed node counts, per phase, read out of the engine's own trace."""
    out = {}
    for phase in trace["phases"]:
        kind = phase["kind"]
        ops = [n["op"] for n in phase["graph"]["nodes"]]
        graph_counts = collections.Counter(ops)
        step = phase["steps"][0]
        executed_ids = [n["id"] for n in step["nodes"]]
        executed = collections.Counter(ops[i] for i in executed_ids)
        in_graph_not_run = collections.Counter(ops[i] for i in range(len(ops))
                                               if i not in set(executed_ids))
        out[kind] = {
            "nodes": len(ops),
            "recorded": len(executed_ids),
            "graph_ops": dict(sorted(graph_counts.items())),
            "executed_ops": dict(sorted(executed.items())),
            "not_recorded": dict(sorted(in_graph_not_run.items())),
            "mul": graph_counts.get("mul", 0),
            "silu": graph_counts.get(SILU, 0),
            "swiglu": graph_counts.get(SWIGLU, 0),
            "silu_run": executed.get(SILU, 0),
            "outputs": phase["graph"]["outputs"],
        }
    return out


def kernel(minfer_cc, work):
    """Compile and run the C instrument, then witness it with the numpy implementation."""
    binary = work / "fuse"
    shell([minfer_cc, "-O2", "-o", str(binary), str(ROOT / "scripts/fuse.c"), "-lm"])
    raw_path = work / "raw-fusion.json"
    shell([str(binary), "--json", str(raw_path)])
    raw = json.loads(raw_path.read_text())
    witness_out = shell([sys.executable, str(ROOT / "scripts/fuse-check.py"), str(raw_path)])

    # the machine's streaming bandwidth, so "16 GB/s" can be read as a fraction of what this box can do
    membw = work / "membw"
    shell([minfer_cc, "-O2", "-o", str(membw), str(ROOT / "scripts/membw.c"), "-lm"])
    bw = {}
    for threads in (1, 4, 8):
        line = shell([str(membw), "1024", str(threads)]).strip().split("\n")[-1]
        bw[str(threads)] = {
            "read": float(matched(r"read ([\d.]+) GB/s", line, "the read rate").group(1)),
            "copy": float(matched(r"copy ([\d.]+) GB/s", line, "the copy rate").group(1)),
            "line": line}

    agree = matched(r"(\d+)/(\d+) samples agree", witness_out, "the witness verdict")
    worst = max(float(m) for m in re.findall(r"max relative difference ([\d.e+-]+)", witness_out))
    return {
        "acts": raw["acts"],
        "bill": raw["bill"],
        "dims": raw["dims"],
        "build": raw["build"],
        "witness": {"agree": int(agree.group(1)), "checked": int(agree.group(2)),
                    "worst_relative": worst, "raw": witness_out.strip().split("\n")},
        "bandwidth": bw,
    }


def tests(minfer):
    """The engine's own fusion tests, run here - the guard's two branches, and the fused-node shape refusals."""
    out = shell(["cargo", "test", "--release", "fusion", "--", "--test-threads", "4"],
                cwd=str(minfer), env={**os.environ, "CARGO_TARGET_DIR": "/tmp/mf15-target"})
    results = [l.strip() for l in out.split("\n") if l.startswith("test ") and " ... " in l]
    named = {}
    for line in results:
        name, verdict = line.split(" ... ")
        named[name.replace("test ", "")] = verdict.split(";")[0].strip()
    return {"run": sorted(named), "failed": [k for k, v in named.items() if v != "ok"],
            "count": len(named)}


def head_rev(checkout):
    """The commit the book is describing, read from the checkout and never typed.

    This field used to be the literal string of whichever commit the script was first written at, which is how
    `facts/dispatch.json` and `facts/fusion.json` came to claim a baseline the book had already moved past.
    """
    return subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("model")
    ap.add_argument("--minfer", required=True, help="worktree of the pinned baseline")
    ap.add_argument("--out", default="facts/fusion.json")
    args = ap.parse_args()

    binary = pathlib.Path(args.binary).resolve()
    model = pathlib.Path(args.model).expanduser()
    minfer = pathlib.Path(args.minfer)
    version = shell([str(binary), "--version"]).strip()

    work = pathlib.Path(tempfile.mkdtemp(prefix="fuse-"))
    trace_path = work / "trace.json"
    trace, _rebuilds, _stdout = bookgraph.run(binary, model, PROMPT, STEPS, trace_path, work)

    cc = os.environ.get("CC", "cc")
    doc = {
        "artifact": "fusion",
        "engine": {"version": version, "baseline": head_rev(minfer), "binary": binary.name,
                   "binary_bytes": binary.stat().st_size, "prompt": PROMPT, "steps": STEPS},
        "vocab": vocabulary(minfer),
        "structure": structure(trace),
        "kernel": kernel(cc, work),
        "tests": tests(minfer),
        # A string, not a dict: the provenance renderer prints this value verbatim into a page, and a dict
        # arrived on the page as `{'platform': ...}` until it did. Python's version is not in it because the
        # number the page reports comes from the C kernel or the engine, not from the interpreter that drove them.
        "machine": f"{platform.system()} {platform.release()}, {platform.machine()}",
        "invocation": f"python3 scripts/fuse.py {binary} {model} --minfer {minfer} --out {args.out}",
    }
    # one number the chapter leans on, computed rather than typed: what fraction of the graph the
    # fusion pass left behind as nodes nobody executes
    pre = doc["structure"]["prefill"]
    doc["summary"] = {
        "rewrites_prefill": pre["swiglu"],
        "orphans_prefill": pre["silu"],
        "orphan_fraction": round(pre["silu"] / pre["nodes"], 4),
        "executed_fraction": round(pre["recorded"] / pre["nodes"], 4),
        "saved_bytes_per_row": doc["kernel"]["bill"]["saved_per_row"],
        "saved_fraction_of_traffic": doc["kernel"]["bill"]["saved_fraction"],
        "speedup_silu_4096": next(r["us_three"] / r["us_one"]
                                  for a in doc["kernel"]["acts"] if a["act"] == "silu"
                                  for r in a["rows"] if r["rows"] == 4096),
        "speedup_relu_4096": next(r["us_three"] / r["us_one"]
                                  for a in doc["kernel"]["acts"] if a["act"] == "relu"
                                  for r in a["rows"] if r["rows"] == 4096),
    }
    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps(doc, indent=1) + "\n")
    s = doc["summary"]
    print(f"wrote {out}: {s['rewrites_prefill']} SwiGLU rewrites, {s['orphans_prefill']} orphan "
          f"silu nodes ({s['orphan_fraction']:.1%} of the graph), fused-node speedup "
          f"{s['speedup_silu_4096']:.2f}x on silu vs {s['speedup_relu_4096']:.2f}x on relu, "
          f"tests {doc['tests']['count']} passed")


if __name__ == "__main__":
    main()
