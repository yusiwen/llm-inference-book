#!/usr/bin/env python3
"""smp.py - the book's sampler instrument: real logits in, the pipeline's decisions out.

This is a measuring instrument, not the deliverable. The chapter's exercise is a sampler of your own inside your
engine; this script exists so that the chapter's funnel counts, penalty effects and reproducibility claims come
from a real model and a real engine run instead of from a toy distribution typed into a page.

It drives the engine's CLI, keeps the logits it dumps, and then *re-implements* the deterministic part of the
pipeline in Python (penalties, top-k, top-p, temperature) so the engine's own token stream can be replayed from
the logits alone:

  * the pipeline - how many candidates survive each stage on a real step, and how much probability each stage
    removes;
  * the penalty pass - which logits it moves, and how often it changes the argmax;
  * top-k versus top-p - which stage bound first, per step;
  * temperature - what it does to the ranking (nothing) and to the probabilities (everything);
  * the seed - what two runs with the same seed share, and where a different seed diverges;
  * stop strings - the truncation point, on stop strings that span a token boundary;
  * the grammar - a schema-constrained run against the same prompt unconstrained, over several seeds.

Usage:
    python3 scripts/smp.py BINARY MODEL [--out facts/sampler.json] [--steps 24]

`BINARY` is the engine's CLI at the book's pinned baseline; `MODEL` is the GGUF the chapter measures. The logits
dumps come from `MINFER_GRAPH_DUMP`, which the engine writes one f32 file per step.
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tok as booktok  # noqa: E402  (the Chapter 11 instrument: the tokenizer, for ids and text)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPT = "The capital of France is"
REPEAT_PROMPT = "Repeat this line exactly, as many times as you can: ha ha ha ha ha ha ha ha"
LIST_PROMPT = "List three fruits, one per line:"
SCHEMA = ('{"type":"object","properties":{"city":{"type":"string"},'
          '"temperature_c":{"type":"integer"}},"required":["city","temperature_c"]}')
JSON_PROMPT = "Give the weather in Paris as JSON with a city and a temperature_c: "
# Candidate prompts for the one step the pipeline tables show. A step whose argmax already carries 0.999 of the
# mass has no funnel to draw, so the prompt is chosen by measuring, not by taste: `choose_flat_prompt` runs each
# of these once and keeps the one whose first-step distribution is the flattest.
FLAT_CANDIDATES = (
    "The capital of France is",
    "Once upon a time",
    "The three primary colours are",
    "My favourite programming language is",
    "Q: What is 17 times 23? A:",
)


# ---------------------------------------------------------------------------------------------------------------
# driving the engine
# ---------------------------------------------------------------------------------------------------------------

def run(binary, model, prompt, steps, flags=(), dump=None, timeout=600):
    """One CLI run: returns (stdout text, generated text, stderr, dump directory)."""
    cmd = [str(binary), "--backend", "cpu", "-n", str(steps), *flags, str(model), prompt]
    env = {"MINFER_DISABLE_MPS": "1"}
    if dump:
        dump = pathlib.Path(dump)
        if dump.exists():
            shutil.rmtree(dump)
        dump.mkdir(parents=True, exist_ok=True)
        env["MINFER_GRAPH_DUMP"] = str(dump)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, **env})
    out, err = proc.stdout, proc.stderr
    if proc.returncode != 0:
        raise SystemExit(f"engine failed ({proc.returncode}):\n{err[-2000:]}")
    return out, generated_text(out), err, dump


def generated_text(stdout):
    """The generated continuation: after the prefill line, before the `---` separator.

    The engine prints the prompt's length, not its text, and then the continuation on its own line - so the
    first token of the output is the first token the sampler chose.
    """
    body = stdout.split("\n---\n")[0]
    lines = body.split("\n")
    start = next((i + 1 for i, l in enumerate(lines) if l.startswith("Prefill:")), 0)
    kept = [l for l in lines[start:] if not l.startswith(("[graph dump]", "Decode:"))]
    return "\n".join(kept).rstrip("\n") if kept else ""


def logits_steps(dump):
    """Every dumped logits vector, in step order.

    The names are `logits_prefill_00000.f32` and `logits_decode_0000N.f32`, so sorting them as strings puts the
    decode files first (`d` < `p`) and every chain below would be one step off. Order by the number.
    """
    files = sorted(pathlib.Path(dump).glob("logits_*.f32"),
                   key=lambda f: int(re.search(r"_(\d+)\.f32$", f.name).group(1)))
    return [(f.name, np.fromfile(f, dtype="<f4")) for f in files]


# ---------------------------------------------------------------------------------------------------------------
# the pipeline, re-implemented from the engine's own definitions
# ---------------------------------------------------------------------------------------------------------------

def apply_penalties(logits, prev, repeat=1.0, frequency=0.0, presence=0.0):
    """The engine's penalty pass: one pass over the distinct tokens of the window."""
    if abs(repeat - 1.0) < 1e-6 and abs(frequency) < 1e-6 and abs(presence) < 1e-6:
        return 0, logits
    out = logits.copy()
    counts = {}
    for t in prev:
        counts[int(t)] = counts.get(int(t), 0) + 1
    for t, c in counts.items():
        if t >= out.size:
            continue
        v = float(out[t])
        if abs(frequency) >= 1e-6 or abs(presence) >= 1e-6:
            v -= c * frequency
            if c > 0:
                v -= presence
        if abs(repeat - 1.0) >= 1e-6 and repeat >= 1.0:
            v = v * repeat if v <= 0.0 else v / repeat
        out[t] = np.float32(v)
    return int(np.count_nonzero(out != logits)), out


def apply_top_k(logits, k):
    """Keep the top k; ties at the threshold survive, as the engine's `select_nth` comparison does."""
    if k == 0 or k >= logits.size:
        return logits
    threshold = np.partition(logits, -k)[-k]
    out = logits.copy()
    out[out < threshold] = -np.inf
    return out


def apply_top_p(logits, p):
    """Nucleus: smallest set whose cumulative probability exceeds p, on the survivors of top-k."""
    if p <= 0.0 or p >= 1.0:
        return logits
    finite = np.isfinite(logits)
    if not finite.any():
        return logits
    vals = logits[finite].astype(np.float64)
    probs = np.exp(vals - vals.max())
    probs /= probs.sum()
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    keep = int(np.searchsorted(cumulative, p, side="right")) + 1
    keep = min(keep, order.size)
    survivors = np.flatnonzero(finite)[order[:keep]]
    out = np.full_like(logits, -np.inf)
    out[survivors] = logits[survivors]
    return out


def softmax(logits, temp):
    vals = logits.astype(np.float64) / temp
    finite = np.isfinite(vals)
    out = np.zeros_like(vals)
    if not finite.any():
        return out
    e = np.exp(vals[finite] - vals[finite].max())
    out[finite] = e / e.sum()
    return out


def survive(logits):
    return int(np.isfinite(logits).sum())


def entropy(probs):
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum())


def top_mass(probs, mass):
    """How many tokens carry `mass` of the probability."""
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    return int(np.searchsorted(cumulative, mass, side="right")) + 1


# ---------------------------------------------------------------------------------------------------------------
# the sections the chapter quotes
# ---------------------------------------------------------------------------------------------------------------

def verify_chain(model, steps_files, texts, repeat=1.0):
    """Replay the engine's greedy output from its own dumped logits.

    The point of the exercise: the logits are the engine's, the pipeline is this script's, and the output has to
    come out byte-identical. If it does not, one of the two is wrong and the chapter's other numbers are void.
    """
    prev, produced = [], []
    for _, logits in steps_files[:len(steps_files) - 1]:
        score, penalized = apply_penalties(logits, prev, repeat=repeat)
        tok = int(np.argmax(penalized))
        produced.append(tok)
        prev.append(tok)
        if len(prev) > 64:
            prev = prev[-64:]
    text = decode_ids(model, produced)
    return {"steps": len(produced), "text": escaped(text, 80), "expected": escaped(texts, 80),
            "match": text == texts, "ids": produced[:12]}


def decode_ids(model, ids):
    try:
        return booktok.decode(model, ids).decode("utf-8")
    except UnicodeDecodeError:
        return booktok.decode(model, ids).decode("utf-8", "replace")


def escaped(text, limit=None):
    out = []
    for ch in text:
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\x{code:02x}")
        elif code > 0x7E:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    s = "".join(out)
    return s[:limit] + "..." if limit and len(s) > limit else s


def choose_flat_prompt(binary, model_path, work):
    """The candidate prompt with the flattest first-step distribution, measured at temperature 1."""
    rows = []
    for i, prompt in enumerate(FLAT_CANDIDATES):
        _, _, _, dump = run(binary, model_path, prompt, 1, ("--greedy", "--repeat-penalty", "1.0"),
                            dump=work / f"probe{i}")
        logits = logits_steps(dump)[0][1]
        probs = softmax(logits, 1.0)
        rows.append({"prompt": prompt, "top1": float(probs.max()), "tokens_90": top_mass(probs, 0.90),
                     "entropy": entropy(probs)})
    best = min(rows, key=lambda r: r["top1"])
    return best, rows


def pipeline_section(model, logits, prev, cfg):
    """One real step, stage by stage: what each stage leaves behind."""
    rows = []
    raw = logits.astype(np.float32)
    rows.append(("raw logits", survive(raw), "every vocabulary entry"))
    moved, penalized = apply_penalties(raw, prev, repeat=cfg["repeat"])
    rows.append((f"repeat penalty {cfg['repeat']}", survive(penalized),
                 f"{moved} logit(s) moved" + (" - the window holds those tokens" if moved else " - nothing repeats")))
    top_k = apply_top_k(penalized, cfg["top_k"])
    rows.append((f"top-k {cfg['top_k']}", survive(top_k), "the tail is set to -inf"))
    top_p = apply_top_p(top_k, cfg["top_p"])
    rows.append((f"top-p {cfg['top_p']}", survive(top_p), "the smallest set reaching the mass"))
    probs = softmax(top_p, cfg["temp"])
    rows.append((f"temperature {cfg['temp']}", survive(top_p), "the set does not change; the weights do"))
    rows.append(("draw", 1, "one id, by the generator"))
    return [{"stage": s, "survivors": n, "note": note} for s, n, note in rows], probs


def funnel_section(model, steps_files, cfg):
    """Across the run: how many candidates the two truncating stages keep."""
    rows = []
    for name, logits in steps_files[:-1]:
        _, penalized = apply_penalties(logits, [], repeat=cfg["repeat"])
        after_k = apply_top_k(penalized, cfg["top_k"])
        after_p = apply_top_p(after_k, cfg["top_p"])
        probs = softmax(after_p, cfg["temp"])
        rows.append({"step": name, "after_k": survive(after_k), "after_p": survive(after_p),
                     "mass_kept": float(probs.sum()), "top1": float(probs.max()),
                     "entropy": entropy(probs)})
    return rows


def temperature_section(model, logits, cfg):
    """What temperature does to one step's ranking and weights."""
    _, penalized = apply_penalties(logits, [], repeat=cfg["repeat"])
    top_k = apply_top_k(penalized, cfg["top_k"])
    top_p = apply_top_p(top_k, cfg["top_p"])
    rows = []
    for temp in (0.5, 0.8, 1.2, 2.0):
        probs = softmax(top_p, temp)
        rows.append({"temp": temp, "top1": float(probs.max()), "top5": float(np.sort(probs)[-5:].sum()),
                     "tokens_90": top_mass(probs, 0.90), "entropy": entropy(probs),
                     "argmax": int(np.argmax(probs))})
    return rows


def penalty_section(model, steps_files, cfg):
    """The repeat penalty, measured where it bites: does it change the choice?"""
    prev, rows = [], []
    for name, logits in steps_files[:-1]:
        moved, penalized = apply_penalties(logits, prev, repeat=cfg["repeat"])
        plain = int(np.argmax(logits))
        with_pen = int(np.argmax(penalized))
        if moved:
            rows.append({"step": name, "moved": moved, "window": len(prev),
                         "plain": plain, "penalised": with_pen, "flipped": plain != with_pen})
        tok = with_pen
        prev.append(tok)
        if len(prev) > 64:
            prev = prev[-64:]
    flips = [r for r in rows if r["flipped"]]
    return {"steps_with_moves": len(rows), "steps_flipped": len(flips), "examples": flips[:6]}


def order_section(model, logits, cfg):
    """The same three stages, in three orders, on the same step - one of these orders is the engine's."""
    rows = []

    def summarise(label, kept):
        probs = softmax(kept, cfg["temp"])
        return {"order": label, "survivors": survive(kept), "mass": float(probs.sum()),
                "top1": float(probs.max()), "argmax": int(np.argmax(probs))}

    rows.append(summarise("top-k -> top-p -> temperature (engine)",
                          apply_top_p(apply_top_k(logits, cfg["top_k"]), cfg["top_p"])))
    scaled = (logits.astype(np.float64) / cfg["temp"]).astype(np.float32)
    rows.append(summarise("temperature -> top-k -> top-p",
                          apply_top_p(apply_top_k(scaled, cfg["top_k"]), cfg["top_p"])))
    rows.append(summarise("top-p -> top-k -> temperature",
                          apply_top_k(apply_top_p(logits, cfg["top_p"]), cfg["top_k"])))
    full_p = apply_top_p(logits, cfg["top_p"])
    rows.append({"order": "top-p alone (top-k off)", "survivors": survive(full_p),
                 "mass": float(softmax(full_p, cfg["temp"]).sum()),
                 "top1": float(softmax(full_p, cfg["temp"]).max()),
                 "argmax": int(np.argmax(full_p))})
    return rows


def repro_section(runs):
    """What a seed buys, and what it does not."""
    base = runs[0]["text"]
    rows = []
    for r in runs:
        same = r["text"] == base
        common = 0
        for a, b in zip(r["text"], base):
            if a != b:
                break
            common += 1
        rows.append({"name": r["name"], "flags": r["flags"], "bytes": len(r["text"].encode()),
                     "same_as_first": same, "shared_prefix_bytes": common,
                     "text": escaped(r["text"], 44)})
    return rows


def stop_section(model, binary, model_path, stop_source):
    """Stop strings: where the run is cut, and whether the stop text survives."""
    text = stop_source
    if len(text) < 12:
        return None
    cut = 4
    stop = text[cut:cut + 5]
    ids = booktok.encode(model, stop, [(s, i) for s, i, _ in booktok.special_tokens(model)])
    out = run(binary, model_path, LIST_PROMPT, 32, ("--greedy", "--repeat-penalty", "1.0", "--stop", stop))
    return {"stop": escaped(stop), "stop_tokens": len(ids), "stop_ids": ids,
            "whole_text": escaped(text, 60), "cut_text": escaped(out[1], 60),
            "cut_at": cut, "kept_stop_text": stop in out[1]}


def grammar_section(binary, model_path, seeds):
    """The negative test: the same prompt with and without a schema, over several seeds."""
    rows = []
    for label, flags in (("no schema", ("--temp", "0.8", "--repeat-penalty", "1.0")),
                         ("schema", ("--temp", "0.8", "--repeat-penalty", "1.0",
                                     "--json-schema-str", SCHEMA))):
        ok = 0
        samples = []
        for seed in seeds:
            text = run(binary, model_path, JSON_PROMPT, 40, (*flags, "--seed", str(seed)))[1]
            valid, why = looks_like_json(text)
            ok += valid
            samples.append({"seed": seed, "valid": valid, "why": why, "text": escaped(text.strip(), 56)})
        rows.append({"condition": label, "seeds": len(seeds), "valid": ok, "samples": samples})
    greedy = run(binary, model_path, JSON_PROMPT, 40,
                 ("--greedy", "--repeat-penalty", "1.0", "--json-schema-str", SCHEMA))[1]
    rows.append({"condition": "schema, greedy", "seeds": 1, "valid": int(looks_like_json(greedy)[0]),
                 "samples": [{"seed": "greedy", "valid": looks_like_json(greedy)[0],
                              "why": looks_like_json(greedy)[1], "text": escaped(greedy.strip(), 56)}]})
    return rows


def looks_like_json(text):
    """Whether the output is a JSON object carrying the two keys the schema requires."""
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return False, f"not JSON ({exc.msg})"
    if not isinstance(value, dict):
        return False, "JSON, but not an object"
    missing = [k for k in ("city", "temperature_c") if k not in value]
    return (not missing), ("ok" if not missing else f"missing {', '.join(missing)}")


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def collect(binary, model_path, out_path, steps):
    model = booktok.load_vocab(str(model_path))
    cfg = {"repeat": 1.1, "top_k": 40, "top_p": 0.95, "temp": 0.8}
    work = pathlib.Path(tempfile.gettempdir()) / "minfer-smp"
    work.mkdir(exist_ok=True)

    # A. greedy, no penalty: the run whose logits every other table is derived from.
    out_a, text_a, _, dump_a = run(binary, model_path, PROMPT, steps,
                                   ("--greedy", "--repeat-penalty", "1.0"), dump=work / "greedy")
    steps_files = logits_steps(dump_a)
    verify = verify_chain(model, steps_files, text_a)

    # A2. the step the pipeline tables show: the flattest of five candidate prompts.
    flat, flat_rows = choose_flat_prompt(binary, model_path, work)
    _, flat_text, _, dump_flat = run(binary, model_path, flat["prompt"], steps,
                                     ("--greedy", "--repeat-penalty", "1.0"), dump=work / "flat")
    flat_steps = logits_steps(dump_flat)
    verify_flat = verify_chain(model, flat_steps, flat_text)
    show_step, show_logits = flat_steps[0]
    pipe_rows, probs = pipeline_section(model, show_logits, [], {**cfg, "repeat": 1.0})

    # B. greedy with the penalty on: the run the penalty table replays.
    out_b, text_b, _, dump_b = run(binary, model_path, REPEAT_PROMPT, steps,
                                   ("--greedy",), dump=work / "greedy-penalty")
    pen = penalty_section(model, logits_steps(dump_b), cfg)
    verify_pen = verify_chain(model, logits_steps(dump_b), text_b, repeat=cfg["repeat"])

    # C. seeds: same, same again, different.
    seeded = []
    variants = (("seed 42", 42, "40"), ("seed 42, second run", 42, "40"), ("seed 43", 43, "40"),
                ("seed 44", 44, "40"), ("seed 42, top-k 5", 42, "5"))
    for name, seed, top_k in variants:
        flags = ("--temp", "0.8", "--top-k", top_k, "--repeat-penalty", "1.0", "--seed", str(seed))
        text = run(binary, model_path, flat["prompt"], steps, flags)[1]
        seeded.append({"name": name, "flags": " ".join(flags), "text": text})

    # D. stops: a stop string cut out of a real run, so it is known to span tokens.
    long_text = run(binary, model_path, LIST_PROMPT, 32, ("--greedy", "--repeat-penalty", "1.0"))[1]
    stop = stop_section(model, binary, model_path, long_text)

    data = {
        "script": "scripts/smp.py",
        "machine": f"{platform.machine()}, {platform.system()} {platform.release()}",
        "compiler": f"python {platform.python_version()}, numpy {np.__version__}",
        "invocation": "python3 scripts/smp.py <engine binary> "
                      "~/.cache/myfer/qwen2.5-0.5b-instruct-q4_k_m.gguf --out facts/sampler.json",
        "measurement": ("the engine's own logits dumps are replayed through this script's pipeline; the token "
                        "counts, funnel rows and entropy figures are computed from those logits"),
        "engine": {
            "prompt": PROMPT,
            "steps": steps,
            "seed_default": 42,
            "sampler_defaults": cfg,
            "backend": "cpu (--backend cpu, MINFER_DISABLE_MPS=1)",
            "binary": str(binary),
            "built_from": ("the pinned baseline, in a `git worktree` of the engine repo: "
                           "`cargo build --release -p minfer-cli`, so the numbers belong to the commit the "
                           "chapter quotes rather than to whatever binary happened to be in the tree"),
        },
        "verify": {"greedy_no_penalty": verify, "greedy_repeat_1_1": verify_pen,
                   "greedy_flat_prompt": verify_flat},
        "prompt_choice": {"chosen": flat, "candidates": flat_rows},
        "pipeline": {"step": show_step, "prompt": flat["prompt"],
                     "top1_raw": float(softmax(show_logits, 1.0).max()), "rows": pipe_rows},
        "funnel": funnel_section(model, flat_steps, {**cfg, "repeat": 1.0}),
        "temperature": temperature_section(model, show_logits, {**cfg, "repeat": 1.0}),
        "order": order_section(model, show_logits, {**cfg, "repeat": 1.0}),
        "penalty": pen,
        "repro": repro_section(seeded),
        "stop": stop,
        "grammar": grammar_section(binary, model_path, (42, 43, 44, 45, 46, 47, 48, 49)),
        "texts": {"greedy": escaped(text_a, 120), "greedy_penalty": escaped(text_b, 120),
                  "list": escaped(long_text, 120)},
    }
    out = pathlib.Path(out_path)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")
    print(f"wrote {out}: greedy replay {'matches' if verify['match'] else 'DIFFERS'}, "
          f"penalty replay {'matches' if verify_pen['match'] else 'DIFFERS'}, "
          f"{pipe_rows[-3]['survivors']} candidates after top-p on step 0")


def main():
    parser = argparse.ArgumentParser(description="Measure an engine's sampler pipeline on real logits.")
    parser.add_argument("binary")
    parser.add_argument("model")
    parser.add_argument("--out", default=str(ROOT / "facts" / "sampler.json"))
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()
    collect(pathlib.Path(args.binary).expanduser(), pathlib.Path(args.model).expanduser(),
            args.out, args.steps)


if __name__ == "__main__":
    main()
