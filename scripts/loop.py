#!/usr/bin/env python3
"""Chapter 20's instrument: the loop's own cost, its reuse, and the two ways it stops.

Everything here comes from running the binary and reading what it prints, or from the bytes it leaves on
disk. Nothing is modelled:

  1. **Two shapes, two builds.** `MINFER_REBUILD_TRACE=1` makes the engine narrate every graph it builds;
     one prefill run should narrate two (the prompt's shape, then the decode shape) and a long generation
     should still narrate two. That is the reuse claim, as a count.
  2. **The loop's cost**, from the engine's own `Prefill:` / `Generated:` lines, median of three runs each.
  3. **Multi-turn**, by running the conversation twice with `--session`: the second turn's prefill count is
     the evidence that the prefix is not re-read, and the `.kv` companion's size is the cache it kept.
  4. **The two stops**: `n_predict` (a stop the caller asked for) and context overflow (a stop the engine
     cannot prevent - it asserts, and the assert is a panic).

      python3 scripts/loop.py BINARY MODEL --minfer ~/git/mine/minfer --out facts/loop.json
"""

import argparse
import json
import os
import pathlib
import platform
import re
import signal
import struct
import subprocess
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = 3
PREFILL_PROMPT = "The capital of France is"
GENERATE_PROMPT = "Write one sentence about the sea:"


def run(binary, model, args, env_extra=None, timeout=180):
    env = {**os.environ, "MINFER_DISABLE_MPS": "1", **(env_extra or {})}
    t0 = time.monotonic()
    # The CLI takes positionals as `<model> [prompt]`: the model comes first, then the prompt,
    # and options may sit anywhere among them.
    proc = subprocess.run([str(binary), str(model), *args], capture_output=True, text=True,
                          env=env, timeout=timeout)
    return {"wall_s": round(time.monotonic() - t0, 3), "exit": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr}


def numbers(out):
    """The three timing lines the engine prints, as numbers."""
    p = re.search(r"Prefill:\s+(\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)", out)
    g = re.search(r"Generated:\s+(\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)", out)
    t = re.search(r"Total:\s+(\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)", out)
    def pack(m):
        return None if not m else {"n": int(m.group(1)), "s": float(m.group(2)), "tok_s": float(m.group(3))}
    tim = re.search(r"\[MINFER_TIMING\] over (\d+) tokens: sample\s+([\d.]+) ms/tok \(\s*([\d.]+)%\), "
                    r"forward\s+([\d.]+) ms/tok \(\s*([\d.]+)%\)", out)
    return {"prefill": pack(p), "generated": pack(g), "total": pack(t),
            "timing": None if not tim else {"tokens": int(tim.group(1)),
                                            "sample_ms": float(tim.group(2)),
                                            "sample_pct": float(tim.group(3)),
                                            "forward_ms": float(tim.group(4)),
                                            "forward_pct": float(tim.group(5))}}


def rebuilds(stderr):
    out = []
    for line in stderr.splitlines():
        m = re.search(r"\[rebuild\] nt=(\d+) params rebuilt in ([\d.]+) ms", line)
        if m:
            out.append({"nt": int(m.group(1)), "ms": float(m.group(2))})
    return out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else round((xs[n // 2 - 1] + xs[n // 2]) / 2, 4)


def timings(binary, model, prompt, n_predict, timing=True):
    """`timing=False` is the honest headline number: `MINFER_TIMING` is not free (it slows the run)."""
    runs = []
    for _ in range(RUNS):
        extra = {"MINFER_REBUILD_TRACE": "1"}
        if timing:
            extra["MINFER_TIMING"] = "1"
        r = run(binary, model, ["--backend", "cpu", "-n", str(n_predict), "--greedy",
                                "--repeat-penalty", "1.0", prompt], extra)
        if r["exit"] != 0:
            raise SystemExit(f"run failed: {r['stderr'][-500:]}")
        # The timing line goes to stderr, the stats lines to stdout: read both.
        runs.append({"nums": numbers(r["stdout"] + r["stderr"]), "builds": rebuilds(r["stderr"]),
                     "wall_s": r["wall_s"]})
    return runs


def multi_turn(binary, model):
    """Two turns through one session file: what the second turn re-reads, and what it kept."""
    work = pathlib.Path(tempfile.mkdtemp(prefix="loop-session-"))
    session = work / "s.json"
    turns = []
    for prompt in ("My name is Ada.", "What is my name?"):
        r = run(binary, model, ["--backend", "cpu", "--cnv", "-st", "--session", str(session),
                                "-n", "8", "--greedy", prompt])
        m = re.search(r"(\d+) tokens, generated (\d+) tokens in ([\d.]+)s", r["stderr"])
        replay = re.search(r"resum\w*|no prefill|prefill (\d+)", r["stderr"])
        turns.append({"prompt": prompt, "exit": r["exit"],
                      "prefill_tokens": int(m.group(1)) if m else None,
                      "generated_tokens": int(m.group(2)) if m else None,
                      "reply": [l for l in r["stdout"].splitlines() if l.strip()][-1][:80],
                      "session_note": replay.group(0) if replay else None})
    kv = pathlib.Path(str(session) + ".kv")
    header = {}
    if kv.exists():
        b = kv.read_bytes()[:36]
        magic, version, flags, backend, n_layer, n_ctx, n_embd, row_elems = struct.unpack("<8s7I", b)
        header = {"magic": magic.decode("ascii", "replace"), "version": version, "flags": flags,
                  "backend_tag": backend, "n_layer": n_layer, "n_ctx": n_ctx,
                  "n_embd": n_embd, "row_elems": row_elems}
    return {"turns": turns, "history_bytes": session.stat().st_size if session.exists() else None,
            "kv_bytes": kv.stat().st_size if kv.exists() else None, "header": header,
            "layer_record_bytes": 3 * 4 + 2 * (header.get("n_ctx", 0) * header.get("row_elems", 0) * 4)
            if header else None}


def overflow(binary, model):
    long_prompt = ("The capital of France is a city that has many museums and" + " very old buildings" * 6)
    r = run(binary, model, ["--backend", "cpu", "--n-ctx", "32", "-n", "8", "--greedy", long_prompt])
    panic = [l for l in r["stderr"].splitlines() if "panicked" in l or "exceeds" in l]
    return {"exit": r["exit"], "prompt_tokens": re.search(r"Prompt: (\d+) tokens", r["stdout"]).group(1)
            if "Prompt:" in r["stdout"] else None, "panic": panic}


def n_predict_stop(binary, model):
    r = run(binary, model, ["--backend", "cpu", "-n", "4", "--greedy", "--repeat-penalty", "1.0", GENERATE_PROMPT])
    note = [l for l in r["stderr"].splitlines() if "stopped by" in l]
    return {"exit": r["exit"], "nums": numbers(r["stdout"]), "note": note[-1].strip() if note else None}


def interrupt(binary, model):
    """SIGINT mid-generation: what the process does, and what it leaves behind."""
    env = {**os.environ, "MINFER_DISABLE_MPS": "1"}
    proc = subprocess.Popen([str(binary), str(model), "--backend", "cpu", "-n", "600",
                             "--repeat-penalty", "1.0",
                             "Count from one to six hundred in words, one per line:"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    # The 0.5B finishes this prompt in well under a second, so the signal has to land while the
    # decode loop is still running or the test measures nothing.
    time.sleep(0.25)
    proc.send_signal(signal.SIGINT)
    out, err = proc.communicate(timeout=120)
    note = [l.strip() for l in (out + err).splitlines() if "[cancelled]" in l or "SIGINT" in l]
    return {"exit": proc.returncode, "note": note[-1].strip() if note else None,
            "nums": numbers(out), "tail": [l for l in out.splitlines() if l.strip()][-1][:70]}


def stop_reason(binary, model):
    """The conversation frontend reports why the loop ended; the single-shot one does not."""
    ms = run(binary, model, ["--backend", "cpu", "--cnv", "-st", "-n", "3", "--greedy", GENERATE_PROMPT])
    note = [l.strip() for l in ms["stderr"].splitlines() if "stopped by" in l]
    shot = run(binary, model, ["--backend", "cpu", "-n", "3", "--greedy", GENERATE_PROMPT])
    shot_note = [l.strip() for l in (shot["stdout"] + shot["stderr"]).splitlines() if "stopped by" in l]
    return {"cnv_note": note[-1] if note else None, "single_shot_note": shot_note[-1] if shot_note else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("model")
    ap.add_argument("--minfer", required=True)
    ap.add_argument("--out", default="facts/loop.json")
    args = ap.parse_args()

    binary = pathlib.Path(args.binary).resolve()
    model = pathlib.Path(args.model).expanduser()
    checkout = pathlib.Path(args.minfer).expanduser()
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout.strip()
    rev = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()

    prefill = timings(binary, model, PREFILL_PROMPT, 1)
    generate = timings(binary, model, GENERATE_PROMPT, 24, timing=False)   # the headline
    generate_timed = timings(binary, model, GENERATE_PROMPT, 24)           # the split, and its cost
    turns = multi_turn(binary, model)

    doc = {
        "artifact": "loop",
        "engine": {"version": version, "baseline": rev, "model": model.name},
        "prefill": {"runs": prefill,
                    "median_tok_s": median([r["nums"]["prefill"]["tok_s"] for r in prefill]),
                    "median_tokens": prefill[0]["nums"]["prefill"]["n"],
                    "builds_per_run": sorted({len(r["builds"]) for r in prefill}),
                    "build_ms": [b["ms"] for b in prefill[0]["builds"]]},
        "decode": {"runs": generate,
                   "median_tok_s": median([r["nums"]["generated"]["tok_s"] for r in generate]),
                   "median_tokens": generate[0]["nums"]["generated"]["n"],
                   "builds_per_run": sorted({len(r["builds"]) for r in generate}),
                   "build_ms": [b["ms"] for b in generate[0]["builds"]],
                   "ms_per_token": round(1000 / median([r["nums"]["generated"]["tok_s"] for r in generate]), 2),
                   "timed": {"median_tok_s": median([r["nums"]["generated"]["tok_s"] for r in generate_timed]),
                             "ms_per_token": round(1000 / median([r["nums"]["generated"]["tok_s"]
                                                                  for r in generate_timed]), 2),
                             "sample_ms": median([r["nums"]["timing"]["sample_ms"] for r in generate_timed
                                                  if r["nums"]["timing"]]),
                             "forward_ms": median([r["nums"]["timing"]["forward_ms"] for r in generate_timed
                                                   if r["nums"]["timing"]]),
                             "sample_pct": generate_timed[0]["nums"]["timing"]["sample_pct"],
                             "forward_pct": generate_timed[0]["nums"]["timing"]["forward_pct"]}},
        "prefill_ms_per_token": round(1000 / median([r["nums"]["prefill"]["tok_s"] for r in prefill]), 2),
        "multi_turn": turns,
        "overflow": overflow(binary, model),
        "n_predict_stop": n_predict_stop(binary, model),
        "stop_reason": stop_reason(binary, model),
        "interrupt": interrupt(binary, model),
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "python": platform.python_version()},
        "provenance": {
            "command": f"python3 scripts/loop.py {binary} {model} --minfer {checkout}",
            "statistic": f"the engine's own `Prefill:`/`Generated:` lines, median of {RUNS} runs; graph "
                         "builds counted from `MINFER_REBUILD_TRACE`; the session file is read as bytes "
                         "(`<8s7I` header) and sized on disk",
            "artifact": "facts/loop.json",
        },
    }
    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out}")
    print(f"  prefill {doc['prefill']['median_tok_s']} tok/s ({doc['prefill_ms_per_token']} ms/token), "
          f"builds {doc['prefill']['builds_per_run']} at {doc['prefill']['build_ms']} ms")
    print(f"  decode  {doc['decode']['median_tok_s']} tok/s ({doc['decode']['ms_per_token']} ms/token), "
          f"builds {doc['decode']['builds_per_run']}, {doc['decode']['median_tokens']} tokens")
    print(f"  turns   {[ (t['prefill_tokens'], t['generated_tokens']) for t in turns['turns'] ]}, "
          f"kv {turns['kv_bytes']} B, header {turns['header']}")
    print(f"  stops   n_predict: {doc['n_predict_stop']['note']} | overflow exit {doc['overflow']['exit']} "
          f"{doc['overflow']['panic']}")
    print(f"  SIGINT  exit {doc['interrupt']['exit']} note={doc['interrupt']['note']}")


if __name__ == "__main__":
    main()
