#!/usr/bin/env python3
"""Run the attention instrument and write facts/attn.json.

The offsets are never typed: they are read out of facts/gguf-qwen2.5-0.5b.json, which is itself the output
of scripts/gguf-inspect.py. That chain is the point - a chapter that quotes a byte offset should be able to
say which parse produced it.

    python3 scripts/attn-run.py --gguf ~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --json facts/attn.json

The binary is compiled on this machine with clang -O2 (the flags are recorded in the artifact, and the
chapter's provenance line prints them), then run once; the program does its own repeat/median handling.
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
TENSORS = ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
           "blk.0.attn_norm.weight", "token_embd.weight"]
KEYS = {"blk.0.attn_q.weight": "q", "blk.0.attn_k.weight": "k", "blk.0.attn_v.weight": "v",
        "blk.0.attn_norm.weight": "norm", "token_embd.weight": "embd"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--facts", default=str(FACTS / "gguf-qwen2.5-0.5b.json"),
                    help="the parse that owns the offsets")
    ap.add_argument("--json", default=str(FACTS / "attn.json"))
    ap.add_argument("--cc", default="clang")
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

    by_name = {t["name"]: t for t in src["tensors"]}
    lines = [f"{k}_off {by_name[t]['abs']}" for t, k in KEYS.items() if k != "embd"]
    lines += [f"{k}_fmt {by_name[t]['type']}" for t, k in KEYS.items()]
    lines += [f"embd_off {by_name['token_embd.weight']['abs']}",
              f"n_embd {n_embd}", f"n_head {n_head}", f"n_kv {n_kv}",
              f"n_layer {n_layer}", f"head_dim {head_dim}"]

    with tempfile.TemporaryDirectory() as td:
        conf = pathlib.Path(td) / "attn.conf"
        conf.write_text("\n".join(lines) + "\n")
        binary = pathlib.Path(td) / "attn"
        build = subprocess.run([args.cc, "-O2", "-o", str(binary), str(ROOT / "scripts" / "attn.c"), "-lm"],
                               capture_output=True, text=True)
        if build.returncode:
            sys.exit(build.stderr)
        print(f"built with {args.cc} -O2")
        run = subprocess.run([str(binary), args.gguf, str(conf), "--json", args.json],
                             capture_output=True, text=True)
        if run.returncode:
            sys.exit(run.stderr or run.stdout)
        print(run.stdout.strip())

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
                    "residual stream would be - and the score scan covers every (head, position) pair.")
    pathlib.Path(args.json).write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {pathlib.Path(args.json).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
