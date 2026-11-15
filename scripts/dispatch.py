#!/usr/bin/env python3
"""dispatch.py - Chapter 16's instrument: where a node runs, and what refuses to let it run there.

The chapter makes four claims, and each one is read rather than asserted:

  1. **The registry is data with two orders.** The identity order (`cpu` = 0, `metal` = 1,
     `cuda` = 2) is a file-format contract; the assignment order is priority (Metal 300, CUDA 200,
     CPU 100). Both are parsed out of the pinned source, not typed into prose.
  2. **One question per backend, and a documented answer.** `docs/SUPPORT-MATRIX.md` carries the
     "Operator Coverage by Backend" table, and the engine ships a test that fails when a backend's
     `supports_op` disagrees with it. That test is run here - so the table in the chapter is a claim
     the engine checks, not a wish.
  3. **The assignment this machine produces.** The engine's own trace records each node's backend
     index, so the number of splits and the number of cross-split edges are counted from the graph
     that ran.
  4. **Three different refusals, three different messages.** `--backend` is run three ways on this
     machine: an unknown name, a name that exists but is not compiled in, and a compiled-in backend
     the machine cannot provide. Their exact text and exit codes are recorded.

      python3 scripts/dispatch.py BINARY MODEL --minfer /tmp/mf16 --out facts/dispatch.json
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
STEPS = 3


def matched(pattern, text, what, flags=0):
    """A regex result or a hard stop: a missing match here means the source moved, not that the answer is 0."""
    m = re.search(pattern, text, flags)
    if m is None:
        raise SystemExit(f"could not find {what} in the pinned source - the source moved")
    return m


def shell(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return proc


def must(cmd, **kw):
    proc = shell(cmd, **kw)
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(map(str, cmd))} failed ({proc.returncode}):\n{proc.stderr[-1500:]}")
    return proc.stdout


def registry(minfer):
    """The id space and the priority order, as the source states them."""
    src = (minfer / "src/graph/registry.rs").read_text()
    n = int(matched(r"pub const N_BACKENDS: usize = (\d+);", src, "N_BACKENDS").group(1))
    names = matched(r'const NAMES: \[&str; N_BACKENDS\] = \[(.*?)\];', src, "the NAMES array",
                    re.S).group(1)
    names = re.findall(r'"([a-z]+)"', names)
    prio = {}
    for key, name in (("METAL", "metal"), ("CUDA", "cuda"), ("CPU", "cpu")):
        m = re.search(rf"pub const PRIORITY_{key}: u16 = (\d+);", src)
        if m:
            prio[name] = int(m.group(1))
    # the constants the handles are defined by: the id order is the on-disk tag order
    ids = {}
    for name in names:
        m = re.search(rf"pub const {name.upper()}: Backend = Backend\((\d+)\);", src)
        if m:
            ids[name] = int(m.group(1))
    return {"n_backends": n, "names": names, "priority": prio, "ids": ids}


def doc_matrix(minfer):
    """The documented op-by-backend contract (`docs/SUPPORT-MATRIX.md`), row by row."""
    text = (minfer / "docs/SUPPORT-MATRIX.md").read_text()
    section = text[text.index("## Operator Coverage by Backend"):]
    rows = []
    for line in section.split("\n"):
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        op = cells[0].replace("`", "").strip()
        marks = []
        for cell in cells[1:4]:
            if cell.startswith("✅"):
                marks.append("yes")
            elif cell.startswith("❌"):
                marks.append("no")
            else:
                marks.append("?")
        rows.append({"op": op[:70], "cpu": marks[0], "metal": marks[1], "cuda": marks[2]})
    return rows


def support_tests(minfer, target_dir):
    """The engine's own op-support contract tests, run here."""
    out = must(["cargo", "test", "--release", "op_matrix", "--", "--test-threads", "4"],
               cwd=str(minfer), env={**os.environ, "CARGO_TARGET_DIR": target_dir})
    named = {}
    for line in out.split("\n"):
        if line.startswith("test ") and " ... " in line:
            name, verdict = line.split(" ... ")
            named[name.replace("test ", "").strip()] = verdict.split(";")[0].strip()
    passed = sum(1 for line in out.split("\n") if line.startswith("test result: ok"))
    skips = len(re.findall(r"\bSKIP\b", out))
    return {"run": sorted(named), "failed": [k for k, v in named.items() if v != "ok"],
            "count": len(named), "skip_mentions": skips, "result_lines": passed}


def refusals(binary, model):
    """The three refusals, run for real: each returns its own message and its own exit code."""
    cases = [
        ("unknown name", ["--backend", "foo"]),
        ("known but not compiled", ["--backend", "cuda"]),
        ("compiled but unavailable", ["--backend", "metal"]),
    ]
    out = []
    for label, args in cases:
        env = {**os.environ, "MINFER_DISABLE_MPS": "1"}
        proc = shell([str(binary), *args, "-n", "1", str(model), "hi"], env=env)
        lines = [l for l in (proc.stdout + proc.stderr).strip().split("\n") if l.strip()]
        err = next((l for l in lines if l.startswith("Error:") or "unknown backend" in l), lines[-1])
        out.append({"case": label, "args": " ".join(args), "exit": proc.returncode,
                    "message": err.strip()})
    return out


def assignment(trace):
    """Per-phase backend histogram, split count, and cross-split edges, from the engine's own graph."""
    out = {}
    for phase in trace["phases"]:
        nodes = phase["graph"]["nodes"]
        backends = [n.get("backend") for n in nodes]
        hist = collections.Counter("none" if b is None else str(b) for b in backends)
        # a split is a maximal run of nodes with the same backend, in node order
        splits, cur = 1, None
        for b in backends:
            if cur is None:
                cur = b
            elif b != cur:
                splits += 1
                cur = b
        # a cross-split edge is an input on a different backend from its consumer
        cross = 0
        for n in nodes:
            for s in (n.get("src") or []):
                if backends[s] != n.get("backend"):
                    cross += 1
        out[phase["kind"]] = {
            "nodes": len(nodes),
            "backend_histogram": dict(sorted(hist.items())),
            "splits": splits,
            "cross_split_edges": cross,
            # the trace records the backend as its registry *name*, not its id: the id is the
            # durable tag (KV-session files), the name is what a human reads in a dump
            "backend_field_kind": ("empty" if not any(b is not None for b in backends)
                                   else "index" if all(isinstance(b, int) for b in backends
                                                       if b is not None) else "name"),
        }
    return out


def shapes(trace):
    """The two numbers a cross-split copy is priced in, read out of the graph rather than assumed.

    Node 0 is the token-id input, so its first dimension is the batch width; the first `matmul` node's
    first dimension is n_embd (the graph stores out_shape as [rows, ...]). A residual-stream tensor
    crossing a split boundary is f32 of that width, for every token.
    """
    graph = next(p for p in trace["phases"] if p["kind"] == "prefill")["graph"]
    nodes = graph["nodes"]
    n_tokens = nodes[0]["shape"][0]
    n_embd = next(n["shape"][0] for n in nodes if n["op"] == "matmul")
    return {"n_tokens": n_tokens, "n_embd": n_embd,
            "boundary_bytes_f32": n_tokens * n_embd * 4,
            "boundary_bytes_f16": n_tokens * n_embd * 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("model")
    ap.add_argument("--minfer", required=True, help="worktree of the pinned baseline")
    ap.add_argument("--target-dir", default="/tmp/mf16-target")
    ap.add_argument("--out", default="facts/dispatch.json")
    args = ap.parse_args()

    binary = pathlib.Path(args.binary).resolve()
    model = pathlib.Path(args.model).expanduser()
    minfer = pathlib.Path(args.minfer)
    version = must([str(binary), "--version"]).strip()

    work = pathlib.Path(tempfile.mkdtemp(prefix="dispatch-"))
    trace_path = work / "trace.json"
    trace, _rebuilds, _stdout = bookgraph.run(binary, model, PROMPT, STEPS, trace_path, work)

    doc = {
        "artifact": "dispatch",
        "engine": {"version": version, "baseline": "88ffa5d", "binary": binary.name,
                   "prompt": PROMPT, "steps": STEPS},
        "registry": registry(minfer),
        "doc_matrix": doc_matrix(minfer),
        "support_tests": support_tests(minfer, args.target_dir),
        "refusals": refusals(binary, model),
        "assignment": assignment(trace),
        "shapes": shapes(trace),
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "python": platform.python_version()},
        "invocation": f"python3 scripts/dispatch.py {binary} {model} --minfer {minfer} "
                      f"--out {args.out}",
    }
    pre = doc["assignment"]["prefill"]
    doc["summary"] = {
        "backends_in_id_space": doc["registry"]["n_backends"],
        "priority_order": [b for b, _ in sorted(doc["registry"]["priority"].items(),
                                                key=lambda kv: -kv[1])],
        "doc_rows": len(doc["doc_matrix"]),
        "cpu_yes_rows": sum(1 for r in doc["doc_matrix"] if r["cpu"] == "yes"),
        "metal_no_rows": sum(1 for r in doc["doc_matrix"] if r["metal"] == "no"),
        "splits": pre["splits"],
        "cross_split_edges": pre["cross_split_edges"],
        "refusals_recorded": len(doc["refusals"]),
        "all_refused": all(r["exit"] != 0 for r in doc["refusals"]),
    }
    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps(doc, indent=1) + "\n")
    s = doc["summary"]
    print(f"wrote {out}: {s['backends_in_id_space']} backend ids, priority {'>'.join(s['priority_order'])}, "
          f"{s['doc_rows']} documented op rows, {s['splits']} split(s) with "
          f"{s['cross_split_edges']} cross-split edges, {s['refusals_recorded']} refusals all refused: "
          f"{s['all_refused']}")


if __name__ == "__main__":
    main()
