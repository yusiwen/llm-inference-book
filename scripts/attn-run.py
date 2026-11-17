#!/usr/bin/env python3
"""Run the attention instrument and write facts/attn.json.

The offsets are never typed: they are read out of facts/gguf-qwen2.5-0.5b.json, which is itself the output
of scripts/gguf-inspect.py. That chain is the point - a chapter that quotes a byte offset should be able to
say which parse produced it.

    python3 scripts/attn-run.py --gguf ~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --json facts/attn.json

The binary is compiled on this machine with clang -O2 (the flags are recorded in the artifact, and the
chapter's provenance line prints them), then run once for the full measurement on layer 0, and once per
remaining layer for the score sweep. The sweep exists because the chapter used to quote one layer's largest
score as if it were the model's: 4.67 is layer 0's maximum, and a reader has no way to know whether layer 17
is worse without the other 23 rows.
"""

import argparse
import json
import pathlib
import platform
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
FACTS = ROOT / "facts"


def tensors_for(layer):
    """The five tensors one attention measurement needs, named for that layer, short name -> tensor."""
    return {"q": f"blk.{layer}.attn_q.weight", "k": f"blk.{layer}.attn_k.weight",
            "v": f"blk.{layer}.attn_v.weight", "norm": f"blk.{layer}.attn_norm.weight",
            "embd": "token_embd.weight"}


def conf_for(by_name, layer, geometry):
    names = tensors_for(layer)
    lines = [f"{k}_off {by_name[t]['abs']}" for k, t in names.items() if k != "embd"]
    lines += [f"{k}_fmt {by_name[t]['type']}" for k, t in names.items()]
    lines += [f"embd_off {by_name[names['embd']]['abs']}"]
    lines += [f"{k} {v}" for k, v in geometry.items()]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--facts", default=str(FACTS / "gguf-qwen2.5-0.5b.json"),
                    help="the parse that owns the offsets")
    ap.add_argument("--json", default=str(FACTS / "attn.json"))
    ap.add_argument("--cc", default="clang")
    ap.add_argument("--no-sweep", action="store_true",
                    help="measure layer 0 only (the sweep is 24 short runs and takes about a minute)")
    args = ap.parse_args()

    src = json.loads(pathlib.Path(args.facts).read_text())
    meta = {k: v.get("value") for k, v in src["metadata"]["keys"].items()}
    n_embd = int(meta["qwen2.embedding_length"])
    n_head = int(meta["qwen2.attention.head_count"])
    n_kv = int(meta["qwen2.attention.head_count_kv"])
    n_layer = int(meta["qwen2.block_count"])
    if n_embd % n_head:
        raise SystemExit(f"embedding {n_embd} does not divide by {n_head} heads")
    head_dim = n_embd // n_head
    geometry = {"n_embd": n_embd, "n_head": n_head, "n_kv": n_kv, "n_layer": n_layer, "head_dim": head_dim}

    by_name = {t["name"]: t for t in src["tensors"]}

    with tempfile.TemporaryDirectory() as td:
        binary = pathlib.Path(td) / "attn"
        build = subprocess.run([args.cc, "-O2", "-o", str(binary), str(ROOT / "scripts" / "attn.c"), "-lm"],
                               capture_output=True, text=True)
        if build.returncode:
            sys.exit(build.stderr)
        print(f"built with {args.cc} -O2")

        def measure(layer, out_path):
            conf = pathlib.Path(td) / f"attn-{layer}.conf"
            conf.write_text(conf_for(by_name, layer, geometry))
            run = subprocess.run([str(binary), args.gguf, str(conf), "--json", str(out_path)],
                                 capture_output=True, text=True)
            if run.returncode:
                sys.exit(run.stderr or run.stdout)
            return run.stdout.strip()

        print(measure(0, args.json))

        # The sweep: the same score scan on every other layer, so "the largest score" is a statement about
        # the model rather than about the one layer the offsets happen to point at.
        sweep = None
        if not args.no_sweep:
            rows = []
            for layer in range(n_layer):
                tmp = pathlib.Path(td) / f"sweep-{layer}.json"
                measure(layer, tmp)
                s = json.loads(tmp.read_text())["scores"]
                rows.append({"layer": layer, "n": s["n"], "abs_mean": s["abs_mean"], "abs_max": s["abs_max"],
                             "over_f16_exp": s["over_f16_exp"], "over_f32_exp": s["over_f32_exp"]})
            worst = max(rows, key=lambda r: r["abs_max"])
            sweep = {"layers": rows, "max_abs_over_all": worst["abs_max"], "layer_of_max": worst["layer"],
                     "min_abs_max": min(r["abs_max"] for r in rows),
                     "pairs_total": sum(r["n"] for r in rows),
                     "above_f16_total": sum(r["over_f16_exp"] for r in rows),
                     "above_f32_total": sum(r["over_f32_exp"] for r in rows),
                     "layers_above_f16_bound": [r["layer"] for r in rows if r["over_f16_exp"]],
                     "layers_above_f32_bound": [r["layer"] for r in rows if r["over_f32_exp"]]}
            print(f"swept {n_layer} layers: max |score| {worst['abs_max']:.4f} at layer {worst['layer']}, "
                  f"{sweep['above_f16_total']} pairs over the f16 bound in {sweep['pairs_total']}")

    data = json.loads(pathlib.Path(args.json).read_text())
    data["artifact"] = "attn"
    data["machine"] = f"{platform.machine()}, {platform.system()} {platform.release()}"
    data["compiler"] = f"{args.cc} -O2"
    data["runs"] = 5
    data["invocation"] = (f"{args.cc} -O2 -o attn scripts/attn.c -lm ; "
                          f"./attn {pathlib.Path(args.gguf).name} attn.conf --json facts/attn.json")
    data["note"] = ("Weight matrices, activations and the KV cache are all f32 and the dots accumulate in "
                    "double, on purpose: the chapter is about the shape of attention, not about a kernel. "
                    "The prefix is 1,024 tokens - a real embedding row per position, normalised as the "
                    "residual stream would be - and the score scan covers every (head, position) pair. The "
                    "line-by-line measurement is layer 0; `sweep` repeats the score scan on all "
                    f"{n_layer} layers, which is what makes the bound statement model-wide.")
    if sweep:
        data["sweep"] = sweep
    pathlib.Path(args.json).write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {pathlib.Path(args.json).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
