#!/usr/bin/env python3
"""Run the engine's own measurement tests and write facts/engine-measure.json.

Why this exists. Chapter 10's cache table used to be the book's arithmetic on a
"2 bytes per value" assumption, and the engine does not implement that: `KvFormat::row_elems`
returns `nkt` for **both** F32 and F16 (one f32 *word* per element, the comment says so), so an
f16 cell keeps the f32 width - f16 buys bytes read, not bytes allocated - and only Q8_0 shrinks
the cell (34 words per 128 elements). The table therefore overstated the packing win by 2x.

So the numbers come from the engine now:

  1. `cargo test --release measure -- --nocapture` in a minfer checkout prints `measure: {...}`
     lines; this script keeps `kv_geometry` (per-format cell geometry, per device) and
     `kv_footprint` (region bytes at three context lengths for a 24-layer, 128-wide cell).
  2. It derives the same numbers in Python straight from the `row_elems` rule.
  3. It fails when the two disagree, so the table on the page is a measurement *with* a second
     witness, never arithmetic on an assumption.

The witness is offline: like `facts/tokenizer-witness.json`, this artifact is written once on a
machine that has the checkout, and CI only checks that the page still matches the artifact.

    python3 scripts/kv-engine.py --json facts/engine-measure.json
"""

import argparse
import json
import os
import math
import pathlib
import platform
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FACTS = ROOT / "facts"
MINFER = pathlib.Path.home() / "git" / "mine" / "minfer"
Q8_0_BLOCK, Q8_0_BLOCK_BYTES, WORD_BYTES = 32, 34, 4


def run_tests(repo, log=None, gguf=None):
    """The engine's measurement suite, verbatim. `--log` re-reads a saved run instead.

    `MINFER_MEASURE_MODEL` is *read* by one test (the allocator ledger needs a real graph); without it
    that test prints a `skipped` line instead of measuring, so the ledger is only collected when a model
    is named here.
    """
    if log:
        text = pathlib.Path(log).read_text()
        command = f"cargo test --release measure -- --nocapture   # re-read from {log}"
    else:
        command = "cargo test --release measure -- --nocapture"
        env = {**os.environ}
        if gguf:
            env["MINFER_MEASURE_MODEL"] = str(gguf)
            command += f"   # MINFER_MEASURE_MODEL={gguf}"
        proc = subprocess.run(["cargo", "test", "--release", "measure", "--", "--nocapture"],
                              cwd=repo, capture_output=True, text=True, env=env)
        text = proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise SystemExit(f"the engine's measure tests failed in {repo}:\n{text[-3000:]}")
    groups = {}
    for line in text.splitlines():
        if not line.startswith("measure: "):
            continue
        payload = json.loads(line[len("measure: "):])
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if isinstance(value, dict):
                groups[key] = value
    skipped = [g["skipped"] for g in groups.values() if "skipped" in g]
    for reason in skipped:
        print(f"  note: the engine skipped one group - {reason}")
    need = ["kv_geometry", "kv_footprint", "kv_context_growth", "kv_sharing", "kv_defrag"] + (
        ["alloc_ledger"] if gguf else [])
    missing = [k for k in need if k not in groups]
    if missing:
        raise SystemExit(f"no {' / '.join(missing)} group in the test output - the tests moved")
    return groups, command


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True).stdout.strip()


def row_elems(fmt, nkt):
    """The engine's rule, transcribed: one f32 word per element for F32 and F16."""
    if fmt in ("f32", "f16"):
        return nkt
    blocks = nkt // Q8_0_BLOCK
    return math.ceil((blocks * Q8_0_BLOCK_BYTES) / WORD_BYTES)


def derive(nkt, layers, ctxs):
    """Bytes per token per layer, and the whole region, from `row_elems` - in Python, on purpose."""
    out = {}
    for fmt in ("f32", "f16", "q8_0"):
        per_cell = row_elems(fmt, nkt) * WORD_BYTES * 2  # two regions: K and V
        out[fmt] = {
            "row_elems": row_elems(fmt, nkt),
            "row_bytes": row_elems(fmt, nkt) * WORD_BYTES,
            "per_token_per_layer": per_cell,
            "per_token_all_layers": per_cell * layers,
            "region_bytes": {ctx: per_cell * ctx * layers for ctx in ctxs},
        }
    return out


def check(measured, derived, nkt, layers, weights):
    """Every derived number against the engine's measurement of the same thing. Fail loudly."""
    bad = []
    geo = {row["format"]: row for row in measured["kv_geometry"]["formats"]}
    for fmt in ("f32", "f16", "q8_0"):
        row, mine = geo[fmt], derived[fmt]
        if row["row_elems"] != mine["row_elems"]:
            bad.append(f"{fmt}: row_elems measured {row['row_elems']}, derived {mine['row_elems']}")
        if row["row_bytes"] != mine["row_bytes"]:
            bad.append(f"{fmt}: row_bytes measured {row['row_bytes']}, derived {mine['row_bytes']}")
        if row["bytes_per_token_per_layer"] != mine["per_token_per_layer"]:
            bad.append(f"{fmt}: per-token-per-layer measured {row['bytes_per_token_per_layer']}, "
                       f"derived {mine['per_token_per_layer']}")
    foot = {(r["format"], r["n_ctx"]): r["region_bytes"] for r in measured["kv_footprint"]["rows"]}
    for (fmt, ctx), value in foot.items():
        mine = derived[fmt]["region_bytes"].get(ctx)
        if mine is None:
            continue
        if mine != value:
            bad.append(f"{fmt} at {ctx}: region measured {value:,}, derived {mine:,}")
    if not bad:
        # The table's last row is a ratio, and a ratio of two facts is still worth pinning.
        for fmt in ("f16", "q8_0"):
            share = derived[fmt]["region_bytes"][8192] / weights
            derived[fmt]["share_of_weights_at_8k"] = share
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minfer", default=str(MINFER))
    ap.add_argument("--log", default=None, help="re-read a saved `cargo test` log instead of running")
    ap.add_argument("--json", default=str(FACTS / "engine-measure.json"))
    ap.add_argument("--gguf", default=os.environ.get("MINFER_MEASURE_MODEL",
                                                     str(pathlib.Path.home() / ".cache" / "myfer" /
                                                         "qwen2.5-0.5b-instruct-q4_k_m.gguf")),
                    help="the model the allocator-ledger test needs (it only *reads* the variable)")
    ap.add_argument("--gguf-bytes", type=int, default=491400032,
                    help="the blob the share-of-weights row divides by")
    args = ap.parse_args()

    repo = pathlib.Path(args.minfer)
    groups, command = run_tests(repo, args.log, args.gguf)
    geo, foot = groups["kv_geometry"], groups["kv_footprint"]
    nkt, layers, ctxs = geo["n_embd"], foot["layers"], sorted({r["n_ctx"] for r in foot["rows"]})
    derived = derive(nkt, layers, ctxs)
    bad = check(groups, derived, nkt, layers, args.gguf_bytes)
    if bad:
        print("derived and measured disagree:", file=sys.stderr)
        for line in bad:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(1)

    artifact = {
        "baseline": {"commit": git(repo, "rev-parse", "HEAD"),
                     "describe": git(repo, "describe", "--tags", "HEAD")},
        "machine": f"{platform.system()} {platform.release()}, {platform.machine()}",
        "command": command + (f" (in {repo})"),
        "n_kv_embd": nkt,
        "layers": layers,
        "contexts": ctxs,
        "weights_bytes": args.gguf_bytes,
        # every group the suite printed, so a chapter can render from the same run
        "groups": groups,
        "derived": derived,
        "checks": {"derived_equals_measured": True, "compared": ["row_elems", "row_bytes",
                                                                 "per_token_per_layer",
                                                                 "region_bytes x format x ctx"]},
        "provenance": {
            "command": (f"cd {repo} && {command}"),
            "statistic": "one run of the engine's own measurement tests; the derived column is "
                         "recomputed from `KvFormat::row_elems` and must equal the measured one",
            "artifact": "facts/engine-measure.json",
        },
    }
    pathlib.Path(args.json).write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"wrote {args.json}")
    for fmt in ("f32", "f16", "q8_0"):
        d = derived[fmt]
        print(f"  {fmt:5s} row_elems {d['row_elems']:3d}  per token/layer {d['per_token_per_layer']:5d} B"
              f"  4k {d['region_bytes'].get(4096, 0) / 2**20:7.1f} MiB"
              f"  8k {d['region_bytes'].get(8192, 0) / 2**20:7.1f} MiB"
              f"  32k {d['region_bytes'].get(32768, 0) / 2**20:7.1f} MiB"
              + (f"  share @8k {d['share_of_weights_at_8k'] * 100:5.1f}%" if "share_of_weights_at_8k" in d else ""))


if __name__ == "__main__":
    main()
