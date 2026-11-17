#!/usr/bin/env python3
"""Run the Metal backend against the CPU one and write facts/metal.json.

Chapter 21 makes four claims that a reader could otherwise take on trust, so the instrument measures all four
on one machine, one model and one prompt:

1. **The device path is a real path.** Same binary, same weights, `--backend cpu` against `--backend metal`;
   prefill and decode token rates, three runs each, medians.
2. **The identity gate holds.** Greedy decoding over the same prompt produces byte-identical text on both
   backends. That is the entry ticket for a new backend in this book - not a tolerance, an equality - because
   a quantized matmul whose operand order changed would show up as a different token, not a slower token.
3. **Zero-copy is worth something, and the number is small.** `MINFER_WEIGHT_COPY=1` makes the engine copy each
   weight into its own buffer instead of aliasing the mmap'd GGUF part. The difference is the cost of the copy,
   measured rather than asserted.
4. **The split of a decode step.** `MINFER_TIMING=1` decomposes a token into sampler and forward time, so the
   reader can see which half a GPU is actually shortening.

    python3 scripts/metal.py --binary /tmp/mf21-target/release/minfer \
        --model ~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --minfer /tmp/mf21 --out facts/metal.json

Deliberately not measured here: anything a machine without an Apple GPU cannot reproduce. The chapter says so,
and the identity claim is the one that does not need a GPU to be meaningful.
"""

import argparse
import hashlib
import json
import pathlib
import platform
import re
import statistics
import subprocess
import sys

PREFILL = re.compile(r"Prefill: (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)")
GENERATED = re.compile(r"Generated: (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)")
TIMING = re.compile(r"\[MINFER_TIMING\] over (\d+) tokens: sample\s+([\d.]+) ms/tok.*forward\s+([\d.]+) ms/tok")


def head_rev(checkout):
    return subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def run(binary, model, backend, prompt, n, extra_env=None):
    env = None
    if extra_env:
        import os
        env = {**os.environ, **extra_env}
    p = subprocess.run([str(binary), str(model), prompt, "--backend", backend, "-n", str(n), "--greedy"],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.exit(f"minfer --backend {backend} failed:\n{p.stdout}\n{p.stderr}")
    pre, gen = PREFILL.search(p.stdout), GENERATED.search(p.stdout)
    if not pre or not gen:
        sys.exit(f"could not parse the run's summary:\n{p.stdout}")
    # The generated text sits between the prompt line and the "---" rule.
    body = p.stdout.split("Prefill:")[-1]
    body = body.split("\n", 1)[1] if "\n" in body else ""
    text = body.split("\n---")[0].strip()
    out = {"prefill_tokens": int(pre.group(1)), "prefill_s": float(pre.group(2)),
           "prefill_tok_s": float(pre.group(3)), "generated_tokens": int(gen.group(1)),
           "generated_s": float(gen.group(2)), "decode_tok_s": float(gen.group(3)),
           "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}
    tim = TIMING.search(p.stderr)
    if tim:
        out["timing"] = {"tokens": int(tim.group(1)), "sample_ms": float(tim.group(2)),
                         "forward_ms": float(tim.group(3))}
    return out


def median(rs, key):
    return round(statistics.median(r[key] for r in rs), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--minfer", required=True, help="the checkout the binary was built from")
    ap.add_argument("--prompt", default="The quick brown fox")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="facts/metal.json")
    args = ap.parse_args()

    cpu = [run(args.binary, args.model, "cpu", args.prompt, args.n) for _ in range(args.runs)]
    metal = [run(args.binary, args.model, "metal", args.prompt, args.n) for _ in range(args.runs)]
    # One timed run per backend, because MINFER_TIMING is not free and must not enter the rates above.
    cpu_timed = run(args.binary, args.model, "cpu", args.prompt, args.n, {"MINFER_TIMING": "1"})
    metal_timed = run(args.binary, args.model, "metal", args.prompt, args.n, {"MINFER_TIMING": "1"})
    copied = [run(args.binary, args.model, "metal", args.prompt, args.n, {"MINFER_WEIGHT_COPY": "1"})
              for _ in range(args.runs)]

    identical = len({r["sha256"] for r in cpu + metal}) == 1
    doc = {
        "artifact": "metal",
        "machine": f"{platform.system()} {platform.release()}, {platform.machine()}",
        "baseline": head_rev(args.minfer),
        "model": pathlib.Path(args.model).name,
        "prompt": args.prompt,
        "n_predict": args.n,
        "cpu": {"prefill_tok_s": median(cpu, "prefill_tok_s"), "decode_tok_s": median(cpu, "decode_tok_s"),
                "runs": [{k: r[k] for k in ("prefill_tok_s", "decode_tok_s")} for r in cpu]},
        "metal": {"prefill_tok_s": median(metal, "prefill_tok_s"), "decode_tok_s": median(metal, "decode_tok_s"),
                  "runs": [{k: r[k] for k in ("prefill_tok_s", "decode_tok_s")} for r in metal]},
        "identity": {"identical": identical, "cpu_sha256": cpu[0]["sha256"], "metal_sha256": metal[0]["sha256"],
                     "text": metal[0]["text"], "tokens": metal[0]["generated_tokens"]},
        "timing": {"cpu": cpu_timed.get("timing"), "metal": metal_timed.get("timing")},
        "weight_copy": {"off_tok_s": median(metal, "decode_tok_s"), "on_tok_s": median(copied, "decode_tok_s"),
                        "runs": [{k: r[k] for k in ("decode_tok_s",)} for r in copied]},
        "invocation": (f"{args.binary} {pathlib.Path(args.model).name} {args.prompt!r} --backend {{cpu,metal}} "
                       f"-n {args.n} --greedy  (x{args.runs} each; plus MINFER_TIMING=1 once per backend and "
                       f"MINFER_WEIGHT_COPY=1 x{args.runs} on metal)"),
        "note": ("One machine, one model, one prompt, one binary. The identity column is the gate this chapter "
                 "cares about: greedy decoding must produce the same bytes on both backends before any speed "
                 "number means anything."),
    }
    doc["speedup"] = {
        "prefill": round(doc["metal"]["prefill_tok_s"] / doc["cpu"]["prefill_tok_s"], 2),
        "decode": round(doc["metal"]["decode_tok_s"] / doc["cpu"]["decode_tok_s"], 2),
    }
    doc["weight_copy"]["delta_pct"] = round(
        (doc["weight_copy"]["on_tok_s"] - doc["weight_copy"]["off_tok_s"]) / doc["weight_copy"]["off_tok_s"] * 100, 1)

    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = pathlib.Path(__file__).resolve().parent.parent / out
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out}: cpu {doc['cpu']['decode_tok_s']} / metal {doc['metal']['decode_tok_s']} tok/s decode "
          f"({doc['speedup']['decode']}x), identical={identical}, "
          f"weight copy {doc['weight_copy']['delta_pct']:+.1f}%")


if __name__ == "__main__":
    main()
