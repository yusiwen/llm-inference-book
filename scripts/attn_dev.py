#!/usr/bin/env python3
"""Run Chapter 25's device measurements on a CUDA machine and write facts/attn-dev.json.

Four numbers the chapter needs and cannot assert: the cost of materialising the score matrix, the cost of not
materialising it, what decode attention reads at two context lengths, and what a CUDA graph saves per step.

    python3 scripts/attn_dev.py --host yusiwen@192.168.2.40 --arch sm_121 --out facts/attn-dev.json
"""

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts" / "attn" / "attn_dev.cu"


def sh(host, cmd):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"remote command failed:\n{cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="yusiwen@192.168.2.40")
    ap.add_argument("--arch", default="sm_121")
    ap.add_argument("--out", default="facts/attn-dev.json")
    args = ap.parse_args()

    remote = "/tmp/ch25-attn"
    sh(args.host, f"mkdir -p {remote}")
    subprocess.run(["scp", "-q", str(SRC), f"{args.host}:{remote}/attn_dev.cu"], check=True)
    sh(args.host, f"cd {remote} && /usr/local/cuda/bin/nvcc -O3 -arch={args.arch} -o attn_dev attn_dev.cu")
    raw = sh(args.host, f"cd {remote} && ./attn_dev")
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"the instrument did not emit JSON ({e}):\n{raw}")

    res = {r["name"]: r for r in doc["results"] if r["name"] != "end"}
    mat, stm = res["prefill_materialised"], res["prefill_streamed"]
    doc["prefill"] = {
        "materialised_ms": mat["ms"],
        "streamed_ms": stm["ms"],
        "speedup": round(mat["ms"] / stm["ms"], 2) if stm["ms"] else None,
        "score_buffer_mb": round(mat["bytes"] / 1e6, 1),
        "max_abs_diff": stm["value"],
        "note": ("both compute the same causal row; the difference is that one writes a seq x seq score matrix "
                 "to device memory and reads it back three times, and the other keeps the row in registers with "
                 "a running max and a running sum"),
    }
    dec = {k: v for k, v in res.items() if k.startswith("decode_attn_ctx")}
    doc["decode"] = {
        "contexts": [
            {"ctx": int(k.rsplit("ctx", 1)[1]), "ms": v["ms"], "bytes_read": v["bytes"],
             "gb_s": round(v["bytes"] / v["ms"] / 1e6, 2) if v["ms"] else None}
            for k, v in sorted(dec.items(), key=lambda kv: int(kv[0].rsplit("ctx", 1)[1]))
        ],
    }
    doc["decode"]["byte_ratio_8k_over_128"] = round(
        doc["decode"]["contexts"][-1]["bytes_read"] / doc["decode"]["contexts"][0]["bytes_read"], 1)
    doc["decode"]["time_ratio_8k_over_128"] = round(
        doc["decode"]["contexts"][-1]["ms"] / doc["decode"]["contexts"][0]["ms"], 2)
    doc["graph"] = {
        "kernels": int(res["hundred_launches"]["value"]),
        "launch_ms": res["hundred_launches"]["ms"],
        "replay_ms": res["graph_replay"]["ms"],
        "per_launch_us": round(res["hundred_launches"]["ms"] * 1000 / res["hundred_launches"]["value"], 2),
        "per_replay_us": round(res["graph_replay"]["ms"] * 1000 / res["graph_replay"]["value"], 2),
        "saved_pct": round((1 - res["graph_replay"]["ms"] / res["hundred_launches"]["ms"]) * 100, 1)
        if res["hundred_launches"]["ms"] else None,
    }
    doc["artifact"] = "attn-dev"
    doc["host"] = args.host
    doc["arch"] = args.arch
    doc["invocation"] = (f"scp scripts/attn/attn_dev.cu {args.host}:{remote}/ ; "
                         f"/usr/local/cuda/bin/nvcc -O3 -arch={args.arch} -o attn_dev attn_dev.cu ; ./attn_dev")
    doc["results"] = list(res.values())

    path = pathlib.Path(args.out)
    if not path.is_absolute():
        path = ROOT / path
    path.write_text(json.dumps(doc, indent=1) + "\n")

    p, d, g = doc["prefill"], doc["decode"], doc["graph"]
    print(f"wrote {path.relative_to(ROOT)}: {doc['device']} sm_{doc['compute_cap']}")
    print(f"  prefill      materialised {mat['ms'] * 1000:,.0f} us ({p['score_buffer_mb']} MB of scores) -> "
          f"streamed {stm['ms'] * 1000:,.0f} us = {p['speedup']}x, diff {p['max_abs_diff']:.2e}")
    for c in d["contexts"]:
        print(f"  decode       ctx {c['ctx']:>5}: {c['ms'] * 1000:,.0f} us, {c['bytes_read'] / 1e6:6.2f} MB read, "
              f"{c['gb_s']} GB/s")
    print(f"  graph        {g['kernels']} launches {g['launch_ms'] * 1000:,.0f} us "
          f"({g['per_launch_us']} us each) -> replay {g['replay_ms'] * 1000:,.0f} us "
          f"({g['per_replay_us']} us each), {g['saved_pct']}% saved")


if __name__ == "__main__":
    main()
