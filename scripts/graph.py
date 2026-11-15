#!/usr/bin/env python3
"""graph.py - the book's graph instrument: one run's graph structure, read out of the engine's own trace.

This is a measuring instrument, not the deliverable. The chapter's exercise is a forward pass rebuilt as a data
structure; this script exists so that the chapter's node counts, op histograms and reuse claims come from a real
run of the engine instead of from a diagram.

It drives the engine's CLI twice - one short prompt and one long one - with `MINFER_TRACE` (the per-node trace the
visualizer uses) and `MINFER_REBUILD_TRACE` (one stderr line per graph build), then answers:

  * what a graph is made of: node count, the op histogram, the inputs and the single output;
  * what changes between the two graph *types* (prefill vs decode) and what does not;
  * what changes when only the prompt length changes - the shapes, or the topology too;
  * the DAG: depth, fan-in, fan-out, and how many nodes are views rather than owners;
  * how many graphs a generation builds (two) and what each build costs;
  * what the trace records per node, so the reader knows what "inspecting a graph" actually gives them.

Usage:
    python3 scripts/graph.py BINARY MODEL [--out facts/graph.json]

`BINARY` is the engine's CLI at the book's pinned baseline, built from it; `MODEL` is the GGUF the chapter
measures.
"""

import argparse
import collections
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tok as booktok  # noqa: E402  (the Chapter 11 instrument, for turning trace token ids back into text)

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHORT = "The capital of France is"
LONG = ("The capital of France is Paris, and the capital of Germany is Berlin, "
        "but the capital of Italy is Rome, and the capital of Spain is")


def run(binary, model, prompt, steps, trace_path, work, timeout=900):
    """One CLI run with the trace and the rebuild log on."""
    env = {**os.environ, "MINFER_DISABLE_MPS": "1", "MINFER_TRACE": str(trace_path),
           "MINFER_REBUILD_TRACE": "1", "MINFER_GRAPH_DUMP": str(work)}
    pathlib.Path(work).mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([str(binary), "--backend", "cpu", "-n", str(steps), "--greedy",
                           "--repeat-penalty", "1.0", str(model), prompt],
                          capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"engine failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    rebuilds = [l for l in proc.stderr.split("\n") if l.startswith("[rebuild]")]
    return json.loads(pathlib.Path(trace_path).read_text()), rebuilds, proc.stdout


def graph_of(trace, kind):
    for phase in trace["phases"]:
        if phase["kind"] == kind:
            return phase["graph"], phase
    raise SystemExit(f"trace has no {kind} phase")


def op_counts(graph):
    return collections.Counter(n["op"] for n in graph["nodes"])


def signature(graph):
    """The topology, with every shape erased: op, name and edges, in node order.

    Two graphs with the same signature differ only in tensor sizes - which is the whole claim this chapter makes
    about prompt length, so it is checked as a value rather than asserted in prose.
    """
    return [(n["op"], n["name"], tuple(n.get("src") or [])) for n in graph["nodes"]]


def dag(graph):
    """Depth and degree of the DAG the `src` edges describe."""
    nodes = graph["nodes"]
    depth = {}
    for n in nodes:
        src = n.get("src") or []
        depth[n["id"]] = 0 if not src else 1 + max(depth.get(s, 0) for s in src)
    fan_in = collections.Counter(len(n.get("src") or []) for n in nodes)
    fan_out = collections.Counter()
    for n in nodes:
        for s in n.get("src") or []:
            fan_out[s] += 1
    longest = max(depth.values()) if depth else 0
    on_longest = [i for i, d in depth.items() if d == longest]
    return {
        "depth": longest,
        "nodes_at_max_depth": len(on_longest),
        "max_fan_in": max(fan_in) if fan_in else 0,
        "max_fan_out": max(fan_out.values()) if fan_out else 0,
        "leaves": len(graph["inputs"]),
        "fan_in_histogram": dict(sorted(fan_in.items())),
    }


def _delta_names(prefill, decode):
    """The named nodes that exist in one graph type and not the other."""
    pre = [(n["op"], n["name"]) for n in prefill["nodes"]]
    dec = [(n["op"], n["name"]) for n in decode["nodes"]]
    gone = [x for x in pre if pre.count(x) > dec.count(x)]
    return [f"{op} {name}" for op, name in gone]


def build_log(rebuilds):
    rows = []
    for line in rebuilds:
        m = re.match(r"\[rebuild\] nt=(\d+) params rebuilt in ([0-9.]+) ms", line)
        if m:
            rows.append({"n_tokens": int(m.group(1)), "ms": float(m.group(2))})
    return rows


def trace_section(model, trace):
    """What the trace holds, and one step's worth of it."""
    step = trace["phases"][1]["steps"][0] if len(trace["phases"]) > 1 else trace["phases"][0]["steps"][0]
    first = (step.get("nodes") or [{}])[0]
    stats_fields = sorted((first.get("stats") or {}).keys())
    top = step.get("logits_top") or []
    decoded = []
    for tok_id, prob in top[:5]:
        text = model["tokens"][int(tok_id)] if int(tok_id) < len(model["tokens"]) else "?"
        decoded.append({"id": int(tok_id), "prob": round(float(prob), 4), "token": escaped(text)})
    sample_cap = max((len(n.get("values") or []) for n in (step.get("nodes") or [])), default=0)
    out_id = (trace["phases"][1]["graph"]["outputs"] if len(trace["phases"]) > 1
              else trace["phases"][0]["graph"]["outputs"])[0]
    logits = next((n for n in (step.get("nodes") or []) if n.get("id") == out_id), None)
    return {"step_keys": sorted(step.keys()), "node_stat_fields": stats_fields,
            "nodes_recorded": len(step.get("nodes") or []), "logits_top": decoded,
            "sample_cap": sample_cap,
            "token": step.get("token"), "token_text": escaped(step.get("text") or ""),
            "logits_node": ({"id": logits["id"], "n": logits["n"], "dtype": logits["dtype"],
                             "stats": logits["stats"]} if logits else None)}


def escaped(text):
    out = []
    for ch in text:
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\x{code:02x}")
        elif code > 0x7E:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    return "".join(out)


def collect(binary, model_path, out_path, steps):
    model = booktok.load_vocab(str(model_path))
    work = pathlib.Path(tempfile.gettempdir()) / "minfer-graph"
    short_trace, short_rb, short_out = run(binary, model_path, SHORT, steps, work / "short.json", work / "d1")
    long_trace, long_rb, _ = run(binary, model_path, LONG, 1, work / "long.json", work / "d2")

    prefill, prefill_phase = graph_of(short_trace, "prefill")
    decode, decode_phase = graph_of(short_trace, "decode")
    prefill_long, _ = graph_of(long_trace, "prefill")

    c_pre, c_dec = op_counts(prefill), op_counts(decode)
    delta = {op: c_pre.get(op, 0) - c_dec.get(op, 0) for op in set(c_pre) | set(c_dec)}
    shapes_pre = {n["id"]: n["shape"] for n in prefill["nodes"]}
    shapes_long = {n["id"]: n["shape"] for n in prefill_long["nodes"]}
    first_shape_diff = next((i for i in shapes_pre if shapes_pre[i] != shapes_long.get(i)), None)

    data = {
        "script": "scripts/graph.py",
        "machine": f"{platform.machine()}, {platform.system()} {platform.release()}",
        "compiler": f"python {platform.python_version()}",
        "invocation": "python3 scripts/graph.py <engine binary> "
                      "~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --out facts/graph.json",
        "measurement": ("the node lists, shapes and edges are read out of the engine's own per-node trace "
                        "(MINFER_TRACE), which the visualizer consumes; the build count comes from "
                        "MINFER_REBUILD_TRACE, one stderr line per build"),
        "engine": {
            "binary": str(binary),
            "built_from": ("the pinned baseline, in a `git worktree` of the engine repo: "
                           "`cargo build --release -p minfer-cli`"),
            "backend": "cpu (--backend cpu, MINFER_DISABLE_MPS=1)",
            "short_prompt": SHORT,
            "long_prompt": LONG,
            "steps": steps,
            "gguf": pathlib.Path(model_path).name,
            "gguf_sha256": hashlib.sha256(open(model_path, "rb").read()).hexdigest(),
        },
        "shape": {
            "prefill": {"nodes": len(prefill["nodes"]), "input_nodes": len(prefill["inputs"]),
                        "outputs": prefill["outputs"], "ops": dict(c_pre.most_common()),
                        "backends": sorted({n["backend"] for n in prefill["nodes"] if n["backend"]}),
                        "steps": len(prefill_phase["steps"])},
            "decode": {"nodes": len(decode["nodes"]), "input_nodes": len(decode["inputs"]),
                       "outputs": decode["outputs"], "ops": dict(c_dec.most_common()),
                       "backends": sorted({n["backend"] for n in decode["nodes"] if n["backend"]}),
                       "steps": len(decode_phase["steps"])},
            "delta": {k: v for k, v in delta.items() if v},
            "delta_names": _delta_names(prefill, decode),
        },
        "same_topology": {
            "short_prompt_nodes": len(prefill["nodes"]),
            "long_prompt_nodes": len(prefill_long["nodes"]),
            "signature_equal": signature(prefill) == signature(prefill_long),
            "first_shape_difference": (
                {"node": first_shape_diff,
                 "short": shapes_pre[first_shape_diff], "long": shapes_long.get(first_shape_diff)}
                if first_shape_diff is not None else None),
        },
        "dag": dag(prefill),
        "reuse": {
            "builds": build_log(short_rb),
            "steps_executed": len(prefill_phase["steps"]) + len(decode_phase["steps"]),
            "decode_steps": len(decode_phase["steps"]),
            "generated_tokens": len(decode_phase["steps"]),
        },
        "trace": trace_section(model, short_trace),
        "output": {"text": escaped(short_out.split("\n---\n")[0].split("Prefill:")[-1].split("\n", 1)[-1]
                                   .strip("\n"))},
    }
    pathlib.Path(out_path).write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")
    print(f"wrote {out_path}: prefill {len(prefill['nodes'])} nodes, decode {len(decode['nodes'])}, "
          f"same topology for both prompts: {data['same_topology']['signature_equal']}, "
          f"{len(data['reuse']['builds'])} build(s) for {data['reuse']['steps_executed']} steps")


def main():
    parser = argparse.ArgumentParser(description="Read an engine's compute graph out of its own trace.")
    parser.add_argument("binary")
    parser.add_argument("model")
    parser.add_argument("--out", default=str(ROOT / "facts" / "graph.json"))
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    collect(pathlib.Path(args.binary).expanduser(), pathlib.Path(args.model).expanduser(), args.out, args.steps)


if __name__ == "__main__":
    main()
