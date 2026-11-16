#!/usr/bin/env python3
"""Chapter 19's instrument: what a family difference costs, and where the graph says so.

Three kinds of evidence, kept apart on purpose:

  1. **The graph, counted.** One trace of a real Qwen2.5 file at the pinned baseline: how many nodes
     the forward step builds, of which kinds, and what one transformer layer contributes. This is the
     engine's own dump (`MINFER_TRACE`), not a reading of the builder.
  2. **The refusals.** Two byte-patched copies of the same file - one with `general.architecture`
     renamed, one with `qwen2.block_count` renamed - run through the binary. The messages are the
     engine's, and they are the reason a loader's error text is an interface and not a log line.
  3. **The source, measured.** Lines and tests per family module, the shared op vocabulary's size, and
     the named places each family differs, found by anchor in the pinned checkout. Counts, not
     impressions: "adding a family costs N lines" needs an N someone can re-derive.

      python3 scripts/family.py BINARY MODEL --minfer ~/git/mine/minfer --ref e844591
"""

import argparse
import collections
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPT = "The capital of France is"
STEPS = 2
ARCH_AT = 64              # offset of the `qwen2` value of `general.architecture` in this file
BLOCK_KEY_AT = 305        # offset of the key `qwen2.block_count`
HEAD_KEY_AT = 453         # offset of the key `qwen2.attention.head_count`


def run(binary, model, extra_env=None, args=("-n", "1", "--greedy")):
    env = {**os.environ, "MINFER_DISABLE_MPS": "1", **(extra_env or {})}
    proc = subprocess.run([str(binary), *args, str(model), "hi"], capture_output=True, text=True, env=env)
    tail = [l for l in proc.stderr.splitlines() if l.strip() and not l.startswith("MPS:")]
    return proc.returncode, (tail[-1] if tail else "")


def patch(source, dest, offset, old, new):
    data = bytearray(source.read_bytes())
    if data[offset:offset + len(old)] != old:
        raise SystemExit(f"{source.name}: expected {old!r} at {offset}, found {data[offset:offset + len(old)]!r}")
    data[offset:offset + len(new)] = new
    dest.write_bytes(bytes(data))
    return dest


def counted_trace(binary, model, work):
    """Node counts by op kind, and one layer's worth of nodes in order."""
    trace = work / "trace.json"
    env = {"MINFER_TRACE": str(trace), "MINFER_GRAPH_DUMP": str(work)}
    proc = subprocess.run([str(binary), "--backend", "cpu", "-n", str(STEPS), "--greedy",
                           "--repeat-penalty", "1.0", str(model), PROMPT],
                          capture_output=True, text=True, env={**os.environ, "MINFER_DISABLE_MPS": "1", **env})
    if proc.returncode != 0:
        raise SystemExit(f"trace run failed: {proc.stderr[-800:]}")
    doc = json.loads(trace.read_text())
    out = {}
    for phase in doc["phases"]:
        nodes = phase["graph"]["nodes"]
        hist = collections.Counter(n["op"] for n in nodes)
        out[phase["kind"]] = {"nodes": len(nodes), "kinds": len(hist),
                              "by_op": dict(sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])))}
    # Layer 0's slice: from the first per-layer matmul to just before the second one.
    nodes = doc["phases"][0]["graph"]["nodes"]
    starts = [i for i, n in enumerate(nodes)
              if n["op"] == "matmul" and n["name"] == "matmul_blk.0.attn_q.weight"]
    if len(starts) != 1:
        raise SystemExit(f"expected one layer-0 q projection, found {len(starts)}")
    i = starts[0]
    j = i
    while j < len(nodes) and not (nodes[j]["op"] == "matmul" and nodes[j]["name"] == "matmul_blk.1.attn_q.weight"):
        j += 1
    layer = [{"op": n["op"], "name": n["name"], "shape": n["shape"], "dtype": n["dtype"]}
             for n in nodes[i:j]]
    # One `attn` node per layer, so the prefill histogram *is* the layer count.
    n_layer = out["prefill"]["by_op"]["attn"]
    shared_inputs = [n for n in layer if n["op"] == "input"]
    per_layer = len(layer) - len(shared_inputs)
    out["layer"] = {"nodes": len(layer), "ops": layer, "layers_in_file": n_layer,
                    "shared_inputs": [n["name"] for n in shared_inputs],
                    "per_layer_nodes": per_layer,
                    "graph_minus_layers": out["prefill"]["nodes"] - per_layer * n_layer}
    return out


def source_facts(checkout):
    """Counts and anchors from the pinned checkout - no impression survives without a line count."""
    def text(rel):
        p = checkout / rel
        if not p.exists():
            raise SystemExit(f"missing at the baseline: {rel}")
        return p.read_text()

    mod = text("src/models/mod.rs")
    arms = re.findall(r'^\s+"([a-z0-9]+)" => Box::new\((qwen\d?)::loader::load', mod, re.M)
    refusal = re.search(r'"(unsupported architecture[^"]*)"', mod)
    modules = {}
    for fam in ("qwen2", "qwen3"):
        lib = sorted((checkout / "src/models" / fam).rglob("*.rs"))
        src_lines = tests = 0
        for f in lib:
            body = f.read_text()
            n = body.count("\n")
            if "/tests" in str(f) or f.name.startswith("tests"):
                tests += n
            else:
                src_lines += n
        modules[fam] = {"files": len(lib), "lines_without_tests": src_lines,
                        "lines_of_tests": tests,
                        "test_fns": sum(f.read_text().count("#[test]") for f in lib)}
    ops = text("src/graph/ops.rs")
    enum = ops[ops.index("pub enum Op {"):]
    enum = enum[:enum.index("\n}")]
    variants = len(re.findall(r"^\s{4}([A-Z][A-Za-z0-9]*)", enum, re.M))
    # One anchor per family difference, counted in both families: a delta with no line count
    # behind it is an impression, and the chapter prints these counts rather than adjectives.
    deltas = {
        "per_head_qk_norm": {"anchor": "QkNorm",
                             "qwen2": text("src/models/qwen2/graph.rs").count("QkNorm"),
                             "qwen3": text("src/models/qwen3/graph.rs").count("QkNorm")},
        "head_dim_from_metadata": {"anchor": "n_embd_head",
                                   "qwen2": text("src/models/qwen2/graph.rs").count("n_embd_head"),
                                   "qwen3": text("src/models/qwen3/graph.rs").count("n_embd_head")},
        "attention_bias": {"anchor": "attn_q_bias",
                           "qwen2": text("src/models/qwen2/loader.rs").count("attn_q_bias"),
                           "qwen3": text("src/models/qwen3/loader.rs").count("attn_q_bias")},
        "tied_output_embedding": {"anchor": "tok_embd.clone()",
                                  "qwen2": text("src/models/qwen2/loader.rs").count("tok_embd.clone()"),
                                  "qwen3": text("src/models/qwen3/loader.rs").count("tok_embd.clone()")},
    }
    shared = {rel: text(rel).count("\n") for rel in
              ("src/graph/ops.rs", "src/graph/alloc.rs", "src/graph/registry.rs", "src/models/weight_reg.rs")}
    return {"arch_arms": arms, "refusal": refusal.group(1) if refusal else None,
            "modules": modules, "op_variants": variants, "deltas": deltas, "shared": shared,
            "mod_lines": mod.count("\n")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("model")
    ap.add_argument("--minfer", required=True, help="a minfer checkout, for the source counts")
    ap.add_argument("--ref", default=None, help="the commit the checkout is at (recorded only)")
    ap.add_argument("--out", default="facts/family.json")
    args = ap.parse_args()

    binary = pathlib.Path(args.binary).resolve()
    model = pathlib.Path(args.model).expanduser()
    checkout = pathlib.Path(args.minfer).expanduser()
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout.strip()
    rev = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    if args.ref and not rev.startswith(args.ref):
        raise SystemExit(f"the checkout is at {rev[:7]}, not at --ref {args.ref}")

    work = pathlib.Path(tempfile.mkdtemp(prefix="family-"))
    refusals = []
    cases = [("unknown architecture", "arch-unknown.gguf", ARCH_AT, b"qwen2", b"qwen9"),
             ("missing block_count", "no-block-count.gguf", BLOCK_KEY_AT, b"qwen2.block_count",
              b"qwenX.block_count"),
             ("missing head_count", "no-head-count.gguf", HEAD_KEY_AT, b"qwen2.attention.head_count",
              b"qwenX.attention.head_count")]
    for label, name, offset, old, new in cases:
        patched = patch(model, work / name, offset, old, new)
        code, message = run(binary, patched)
        refusals.append({"case": label, "file": name, "exit": code, "message": message})

    doc = {
        "artifact": "family",
        "engine": {"version": version, "baseline": rev, "model": model.name},
        "trace": counted_trace(binary, model, work),
        "refusals": refusals,
        "source": source_facts(checkout),
        "machine": f"{platform.system()} {platform.release()}, {platform.machine()}",
        "provenance": {
            "command": f"python3 scripts/family.py {binary} {model} --minfer {checkout}",
            "statistic": "node counts and one layer's slice from the engine's own `MINFER_TRACE` dump; "
                         "refusals are the binary's stderr on byte-patched copies of the same file; "
                         "source counts are line and `#[test]` counts in the pinned checkout",
            "artifact": "facts/family.json",
        },
    }
    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps(doc, indent=1) + "\n")
    t = doc["trace"]["prefill"]
    print(f"wrote {out}")
    print(f"  prefill  {t['nodes']} nodes, {t['kinds']} kinds; one layer = {doc['trace']['layer']['nodes']} nodes")
    print(f"  decode   {doc['trace']['decode']['nodes']} nodes")
    for r in refusals:
        print(f"  exit {r['exit']}  {r['case']:22s} {r['message'][:90]}")
    print(f"  source   {doc['source']['modules']}")


if __name__ == "__main__":
    main()
