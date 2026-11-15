#!/usr/bin/env python3
"""alloc.py - Chapter 17's instrument: liveness, one arena, and what the process actually holds.

A transformer's activations are almost all transient: a node's output is read by its consumers and then
never again. An allocator that gives every node its own buffer pays for the graph's *sum*; an allocator
that knows when each buffer dies pays for the *peak of the live set* instead. This script measures the
difference on the graph the engine actually executed, and then compares the prediction with the process's
own peak RSS.

What it does:

  1. Runs the engine with its per-node trace on, and takes the prefill graph (node id order == build
     order, which is what the engine's `alloc_graph` uses - see the quote in the chapter).
  2. Re-implements the engine's liveness rule faithfully: `last_use[s] = max(index of s's consumers)`,
     outputs and **inputs** pinned to `n` (an input is filled before execution starts, so its buffer must
     never be handed to another node), views extending their parent's life.
  3. Computes three numbers: the naive bill (every node its own buffer), the live-set floor (the smallest
     arena any allocator could use), and a first-fit arena with a free list (what a real allocator of this
     shape needs).
  4. Measures the run's peak RSS with `/usr/bin/time -l`, and puts the GGUF blob next to it.

The model is the book's, not the engine's: the engine's own numbers come from `GraphAllocator::memory_report`,
which the CLI does not print. The chapter says which numbers are which, and lesson 5 is about the gap.

      python3 scripts/alloc.py BINARY MODEL --minfer /tmp/mf17 --out facts/alloc.json
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import graph as bookgraph  # noqa: E402  (Chapter 14's driver: same CLI invocation, same trace format)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPT = "The capital of France is"
STEPS = 3
DTYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "i32": 4, "i64": 8, "u32": 4, "u8": 1}
# `src/graph/allocplan.rs`: CLASS_MIN / CLASS_GRAIN in f32 elements
CLASS_MIN = 256
CLASS_GRAIN = 4096


def class_elems(elems):
    """`allocplan::class_size`, transcribed: a 1 KiB floor, powers of two to 16 KiB, then 16 KiB steps."""
    elems = max(elems, 1)
    if elems <= CLASS_MIN:
        return CLASS_MIN
    if elems <= CLASS_GRAIN:
        return 1 << (elems - 1).bit_length()
    return -(-elems // CLASS_GRAIN) * CLASS_GRAIN


def classes(nodes):
    """What the size-class ladder costs over exact sizes, on this graph's own shapes."""
    exact = rounded = 0
    small = grain = 0
    for n in nodes:
        elems = 1
        for d in n["shape"]:
            elems *= d
        exact += elems * 4
        c = class_elems(elems)
        rounded += c * 4
        if elems <= CLASS_GRAIN:
            small += 1
        else:
            grain += 1
    return {"exact_bytes": exact, "rounded_bytes": rounded,
            "overhead_pct": round((rounded / exact - 1) * 100, 2) if exact else None,
            "nodes_pow2_regime": small, "nodes_grain_regime": grain}


def nbytes(shape, dtype):
    n = 1
    for d in shape:
        n *= d
    return n * DTYPE_BYTES[dtype]


def last_use(nodes, outputs, inputs):
    """The engine's rule, in Python: build order, consumers extend, outputs and inputs pinned to the end."""
    n = len(nodes)
    lu = list(range(n))
    for i, node in enumerate(nodes):
        for s in node.get("src") or []:
            if lu[s] < i:
                lu[s] = i
    for o in outputs:
        lu[o] = n
    for i in inputs:
        lu[i] = n
    # a view aliases its parent's bytes (Chapter 17 lesson 2): the parent must outlive every reader of
    # the view, so pushing a view's last use back pushes its parent's too
    changed = True
    while changed:
        changed = False
        for i, node in enumerate(nodes):
            if node["op"] != "view":
                continue
            parent = (node.get("src") or [None])[0]
            if parent is None:
                continue
            if lu[parent] < lu[i]:
                lu[parent] = lu[i]
                changed = True
    return lu


def naive_and_floor(sizes, lu):
    """The bill if nothing is reused, and the live-set floor any allocator must pay."""
    naive = sum(sizes)
    n = len(sizes)
    # peak of the live set: at each step, which nodes are alive
    alive = []
    floor = 0
    peak_step = 0
    for i in range(n):
        alive = [j for j in alive if lu[j] >= i]
        alive.append(i)
        live = sum(sizes[j] for j in alive)
        if live > floor:
            floor, peak_step = live, i
    return naive, floor, peak_step


def first_fit(sizes, lu):
    """One arena, a free list, allocate-then-free: the shape of the engine's pool, in bytes."""
    n = len(sizes)
    free = []           # (offset, length) of dead holes
    live = {}           # node -> (offset, length)
    high = 0
    reused = 0
    live_offsets_all = {}
    for i in range(n):
        # free everything whose last use has passed
        for j in [j for j, (_, l) in live.items() if lu[j] < i]:
            off, ln = live.pop(j)
            free.append((off, ln))
        size = sizes[i]
        # first fit, then bump
        pick = None
        for k, (off, ln) in enumerate(free):
            if ln >= size:
                pick = k
                break
        if pick is None:
            off = high
            high += size
        else:
            off, ln = free.pop(pick)
            if ln > size:
                free.append((off + size, ln - size))
            reused += 1
        live[i] = (off, size)
        live_offsets_all[i] = off
    return high, reused, len(free), live_offsets_all


def peak_rss(binary, model):
    """The process's own number: `/usr/bin/time -l` prints maxrss in bytes on macOS."""
    proc = subprocess.run(["/usr/bin/time", "-l", str(binary), "--backend", "cpu", "-n", "2",
                           "--greedy", "--repeat-penalty", "1.0", str(model), PROMPT],
                          capture_output=True, text=True,
                          env={**os.environ, "MINFER_DISABLE_MPS": "1"})
    text = proc.stdout + proc.stderr
    m = re.search(r"(\d+)\s+maximum resident set size", text)
    if m is None:
        raise SystemExit("could not read maxrss from /usr/bin/time -l")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("model")
    ap.add_argument("--minfer", required=True)
    ap.add_argument("--out", default="facts/alloc.json")
    args = ap.parse_args()

    binary = pathlib.Path(args.binary).resolve()
    model = pathlib.Path(args.model).expanduser()
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout.strip()

    work = pathlib.Path(tempfile.mkdtemp(prefix="alloc-"))
    trace_path = work / "trace.json"
    trace, _rebuilds, _stdout = bookgraph.run(binary, model, PROMPT, STEPS, trace_path, work)

    phases = {}
    for phase in trace["phases"]:
        g = phase["graph"]
        nodes = g["nodes"]
        # Persistent regions (the KV arenas) are allocated by name and survive a rebuild; the
        # `kvcache_*` nodes read and write *views* into them, so they must not be charged to the
        # transient arena. They keep their place in the liveness walk (a projection feeding a store
        # is still live until the store reads it) but they are given size 0 and counted separately.
        persistent = [i for i, n in enumerate(nodes) if n["op"].startswith("kvcache")]
        sizes = [0 if i in persistent else nbytes(n["shape"], n["dtype"]) for i, n in enumerate(nodes)]
        # every layer has a `kvcache_store` and a `kvcache_load` node, and both carry the shape of
        # the *same* region for that layer - so the region is per-layer bytes times layers, not the
        # sum over nodes (summing would double every layer).
        kv_per_node = max((nbytes(nodes[i]["shape"], nodes[i]["dtype"]) for i in persistent), default=0)
        kv_bytes = kv_per_node * (len(persistent) // 2)
        lu = last_use(nodes, g["outputs"], g.get("inputs") or [])
        naive, floor, peak_step = naive_and_floor(sizes, lu)
        arena, reused, holes, offsets = first_fit(sizes, lu)
        worst = max(range(len(sizes)), key=lambda i: sizes[i])
        phases[phase["kind"]] = {
            "nodes": len(nodes),
            "naive_bytes": naive,
            "live_floor_bytes": floor,
            "arena_bytes": arena,
            "reused_allocations": reused,
            "holes_at_end": holes,
            "peak_step": peak_step,
            # a legible sample of the span picture: the biggest transient buffers, with the interval
            # they were live for and the offset the first-fit arena gave them
            "spans": sorted(
                ({"node": i, "op": nodes[i]["op"], "start": i, "end": lu[i],
                  "offset": offsets.get(i, 0), "bytes": sizes[i]}
                 for i in range(len(nodes)) if sizes[i] and lu[i] < len(nodes)),
                key=lambda s: -s["bytes"])[:36],
            "largest_buffer": {"node": worst, "op": nodes[worst]["op"], "shape": nodes[worst]["shape"],
                               "bytes": sizes[worst]},
            "persistent_nodes": len(persistent),
            "kv_region_bytes": kv_bytes,
            "kv_layers": len(persistent) // 2,
            "largest_kv_node": {"shape": nodes[persistent[0]]["shape"],
                                "bytes": nbytes(nodes[persistent[0]]["shape"], nodes[persistent[0]]["dtype"])}
                               if persistent else None,
            "sum_over_floor": round(naive / floor, 2) if floor else None,
            "arena_over_floor": round(arena / floor, 3) if floor else None,
        }

    doc = {
        "artifact": "alloc",
        "engine": {"version": version, "baseline": "88ffa5d", "binary": binary.name,
                   "prompt": PROMPT, "steps": STEPS},
        "phases": phases,
        "classes": classes(next(p for p in trace["phases"] if p["kind"] == "prefill")["graph"]["nodes"]),
        "process": {
            "maxrss_bytes": peak_rss(binary, model),
            "gguf_bytes": model.stat().st_size,
            "model": model.name,
        },
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "python": platform.python_version()},
        "invocation": f"python3 scripts/alloc.py {binary} {model} --minfer {args.minfer} "
                      f"--out {args.out}",
    }
    pre = phases["prefill"]
    doc["summary"] = {
        "naive_over_arena": round(pre["naive_bytes"] / pre["arena_bytes"], 2),
        "arena_bytes": pre["arena_bytes"],
        "naive_bytes": pre["naive_bytes"],
        "live_floor_bytes": pre["live_floor_bytes"],
        "reused_allocations": pre["reused_allocations"],
        "rss_minus_gguf_bytes": doc["process"]["maxrss_bytes"] - doc["process"]["gguf_bytes"],
        "class_overhead_pct": doc["classes"]["overhead_pct"],
    }
    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out}: naive {pre['naive_bytes']:,} B, arena {pre['arena_bytes']:,} B "
          f"({doc['summary']['naive_over_arena']}x smaller), floor {pre['live_floor_bytes']:,} B, "
          f"{pre['reused_allocations']} reuses, rss {doc['process']['maxrss_bytes']:,} B "
          f"on a {doc['process']['gguf_bytes']:,} B file")


if __name__ == "__main__":
    main()
