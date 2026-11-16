#!/usr/bin/env python3
"""alloc.py - Chapter 17's instrument: the allocation ledger, with each number's author named.

Three different parties measure three different things, and the chapter's argument is that they must
not be confused:

  1. **The book's arithmetic on the graph.** Every *transient* buffer in the graph dump at its own
     size, summed - the bill if nothing were ever reused. Pure arithmetic over `MINFER_TRACE` shapes:
     no liveness model, no allocator.
  2. **The engine's own ledger.** `GraphAllocator::memory_report` (weights, pool high-water, live,
     peak live, idle slots, reserved classes) and `KvCache::region_bytes`, read out of
     `facts/engine-measure.json`, which `scripts/engine-measure.py` produced by running the engine's
     measurement tests at the pinned baseline. Not a model of the allocator: the allocator.
  3. **The kernel's number.** Peak RSS from `/usr/bin/time -l`, three runs, median and range. What
     actually happened in the process, including every page the engine allocated but never touched.

The trace still supplies two things no engine method reports: the node counts, and the persistent (KV)
node vocabulary. The counts are a cross-check against the engine's ledger - if the two disagree the
numbers below would describe different graphs, so the script refuses to write.

      python3 scripts/alloc.py BINARY MODEL --minfer ~/git/mine/minfer --out facts/alloc.json
"""

import argparse
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPT = "The capital of France is"
STEPS = 3
DTYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "i32": 4, "i64": 8, "u32": 4, "u8": 1}
RUNS = 3


def run_trace(binary, model, work, timeout=900):
    """One CLI run with the per-node trace on, and the trace object back.

    Chapter 14's driver (`scripts/graph.py`) does the same thing, but it imports the Chapter 11
    tokenizer to turn trace token ids back into text - a dependency this chapter does not have. The
    trace alone is all chapter 17 needs.
    """
    env = {**os.environ, "MINFER_DISABLE_MPS": "1", "MINFER_TRACE": str(work / "trace.json"),
           "MINFER_REBUILD_TRACE": "1", "MINFER_GRAPH_DUMP": str(work)}
    work.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([str(binary), "--backend", "cpu", "-n", str(STEPS), "--greedy",
                           "--repeat-penalty", "1.0", str(model), PROMPT],
                          capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"engine failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    return json.loads((work / "trace.json").read_text())


def nbytes(shape, dtype):
    n = 1
    for d in shape:
        n *= d
    return n * DTYPE_BYTES[dtype]


def peak_rss_runs(binary, model, runs=RUNS):
    """The process's own number: `/usr/bin/time -l` prints maxrss in bytes on macOS."""
    values = []
    for _ in range(runs):
        proc = subprocess.run(["/usr/bin/time", "-l", str(binary), "--backend", "cpu", "-n", "2",
                               "--greedy", "--repeat-penalty", "1.0", str(model), PROMPT],
                              capture_output=True, text=True,
                              env={**os.environ, "MINFER_DISABLE_MPS": "1"})
        m = re.search(r"(\d+)\s+maximum resident set size", proc.stdout + proc.stderr)
        if m is None:
            raise SystemExit("could not read maxrss from /usr/bin/time -l")
        values.append(int(m.group(1)))
    return sorted(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("model")
    ap.add_argument("--minfer", required=True, help="a minfer checkout (for its version string only)")
    ap.add_argument("--engine-measure", default=str(ROOT / "facts" / "engine-measure.json"),
                    help="the artifact scripts/engine-measure.py wrote at the same baseline")
    ap.add_argument("--out", default="facts/alloc.json")
    args = ap.parse_args()

    binary = pathlib.Path(args.binary).resolve()
    model = pathlib.Path(args.model).expanduser()
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout.strip()

    engine = json.loads(pathlib.Path(args.engine_measure).read_text())
    ledger_engine = engine["groups"]["alloc_ledger"]
    baseline = engine["baseline"]["commit"]

    work = pathlib.Path(tempfile.mkdtemp(prefix="alloc-"))
    trace = run_trace(binary, model, work)

    phases = {}
    for phase in trace["phases"]:
        nodes = phase["graph"]["nodes"]
        # A `kvcache_*` node reads or writes a *view* into a persistent region. That region is not
        # allocated by the transient allocator, so charging these nodes to it would count the same
        # bytes once per store and once per load. They are recorded here as a vocabulary; their bytes
        # are the engine's `region_bytes`, never a sum over these nodes.
        persistent = [i for i, n in enumerate(nodes) if n["op"].startswith("kvcache")]
        transient = [i for i in range(len(nodes)) if i not in persistent]
        naive = sum(nbytes(nodes[i]["shape"], nodes[i]["dtype"]) for i in transient)
        worst = max(transient, key=lambda i: nbytes(nodes[i]["shape"], nodes[i]["dtype"]))
        phases[phase["kind"]] = {
            "nodes": len(nodes),
            "transient_nodes": len(transient),
            "naive_bytes": naive,
            "persistent_nodes": len(persistent),
            "persistent_ops": sorted({nodes[i]["op"] for i in persistent}),
            "largest_buffer": {"node": worst, "op": nodes[worst]["op"], "shape": nodes[worst]["shape"],
                               "bytes": nbytes(nodes[worst]["shape"], nodes[worst]["dtype"])},
        }

    # The trace and the engine's ledger must describe the same graph, or the two halves of the table
    # below describe different runs.
    crosscheck = {}
    for phase in ("prefill", "decode"):
        traced, measured = phases[phase]["nodes"], ledger_engine["isolated"][phase]["nodes"]
        crosscheck[phase] = {"trace_nodes": traced, "engine_nodes": measured, "match": traced == measured}
    if not all(v["match"] for v in crosscheck.values()):
        raise SystemExit(f"the trace and the engine's ledger disagree about the graph: {crosscheck}")

    kv_bytes = ledger_engine["kv"]["region_bytes"]
    weights = ledger_engine["session"]["prefill"]["weights_bytes"]
    pre, dec = ledger_engine["isolated"]["prefill"], ledger_engine["isolated"]["decode"]
    rss = peak_rss_runs(binary, model)
    observed = rss[len(rss) // 2]
    predicted = weights + kv_bytes + pre["pool_bytes"]

    doc = {
        "artifact": "alloc",
        "engine": {"version": version, "baseline": baseline, "binary": binary.name,
                   "prompt": PROMPT, "steps": STEPS},
        "phases": phases,
        "kv": {"region_bytes": kv_bytes, "layers": ledger_engine["kv"]["layers"],
               "n_ctx": ledger_engine["kv"]["n_ctx"], "format": ledger_engine["kv_format"],
               "n_tokens": ledger_engine["n_tokens"]},
        "ledger": {
            # every value carries the party that measured it - the chapter prints this column
            "weights_bytes": {"value": weights, "author": "engine", "from": "GraphAllocator::memory_report"},
            "kv_region_bytes": {"value": kv_bytes, "author": "engine", "from": "KvCache::region_bytes"},
            "pool_bytes": {"value": pre["pool_bytes"], "author": "engine", "from": "memory_report high-water"},
            "live_bytes": {"value": pre["live_bytes"], "author": "engine", "from": "memory_report"},
            "peak_live_bytes": {"value": pre["peak_live_bytes"], "author": "engine", "from": "memory_report"},
            "naive_bytes": {"value": phases["prefill"]["naive_bytes"], "author": "book",
                            "from": "sum of the trace's transient buffers"},
            "predicted_bytes": {"value": predicted, "author": "book", "from": "weights + KV + pool"},
            "observed_rss_bytes": {"value": observed, "author": "kernel", "from": "/usr/bin/time -l"},
            "delta_bytes": {"value": predicted - observed, "author": "book", "from": "predicted - observed"},
        },
        "prefill": {"idle_slots": pre["idle_slots"], "reserved_classes": pre["reserved_classes"]},
        "decode": {"nodes": dec["nodes"], "pool_bytes": dec["pool_bytes"], "live_bytes": dec["live_bytes"],
                   "peak_live_bytes": dec["peak_live_bytes"], "idle_slots": dec["idle_slots"],
                   "reserved_classes": dec["reserved_classes"],
                   "naive_bytes": phases["decode"]["naive_bytes"]},
        "reuse": {"prefill": round(phases["prefill"]["naive_bytes"] / pre["pool_bytes"], 2),
                  "decode": round(phases["decode"]["naive_bytes"] / dec["pool_bytes"], 2)},
        "crosscheck": crosscheck,
        "process": {"maxrss_runs": rss, "maxrss_median": observed, "maxrss_min": rss[0],
                    "maxrss_max": rss[-1], "runs": RUNS, "gguf_bytes": model.stat().st_size,
                    "engine_weights_bytes": weights, "model": model.name},
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "python": platform.python_version()},
        "invocation": f"python3 scripts/alloc.py {binary} {model} --minfer {args.minfer}",
        "provenance": {
            "command": f"python3 scripts/alloc.py {binary} {model} --minfer {args.minfer}",
            "statistic": f"peak RSS: median of {RUNS} runs with the range beside it; the engine's ledger "
                         "comes from `cargo test --release measure` at the same baseline; the naive "
                         "column is arithmetic over one trace",
            "artifact": "facts/alloc.json",
        },
    }

    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out}")
    print(f"  naive (book)      {phases['prefill']['naive_bytes']:>12,} B   pool (engine) "
          f"{pre['pool_bytes']:>10,} B   reuse {doc['reuse']['prefill']}x")
    print(f"  weights (engine)  {weights:>12,} B   KV (engine)   {kv_bytes:>10,} B")
    print(f"  predicted (book)  {predicted:>12,} B   observed RSS  {observed:>10,} B   "
          f"delta {predicted - observed:+,} B")
    print(f"  cross-check       {crosscheck}")


if __name__ == "__main__":
    main()
