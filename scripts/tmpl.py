#!/usr/bin/env python3
"""tmpl.py - the book's chat-template instrument: the GGUF's template in, exact prompt bytes and ids out.

This is a measuring instrument, not the deliverable. The chapter's exercise is a template renderer of your own
inside your engine; this script exists so that the chapter's prompts, byte lengths and token counts are produced
from a real model file instead of typed.

It answers four questions about one model's `tokenizer.chat_template`:

  * what the program is - its size, the variables it reads, the filters it calls;
  * what it renders - the exact bytes for a conversation, and the ids those bytes become, using this book's
    tokenizer (Chapter 11's `scripts/tok.py`) rather than a second tokenizer;
  * what changes the bytes - the generation prompt, the default system message, the tools block, the thinking
    flag - measured as a delta in bytes and tokens;
  * what a renderer must reproduce that a template library does not give it for free (`tojson` spacing and key
    order, Python `str` semantics, and what "undefined variable" does by default).

Usage:
    python3 scripts/tmpl.py GGUF [--out facts/template.json] [--minfer PATH]

`--minfer` points at a checkout of the engine at the book's pinned baseline, whose committed chat fixtures are
the witness: they were rendered by HuggingFace `transformers` 5.17.0, so rendering them here with Jinja2 is a
comparison between two independent implementations of the same template language.
"""

import argparse
import difflib
import hashlib
import json
import pathlib
import platform
import sys

import re

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tok as booktok  # noqa: E402  (the Chapter 11 instrument: GGUF reader and BPE encoder)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MINFER = pathlib.Path.home() / "git" / "mine" / "minfer"


# ---------------------------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------------------------

def transformers_tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    """The `tojson` filter the reference implementation installs: CPython `json.dumps`, insertion order kept.

    Reproduced here because it is *not* Jinja2's own `tojson`, and the two disagree - see `tojson_policies`.
    """
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def environment(reference_filters=True, strict=False):
    """The environment the reference renderer builds: sandboxed, block trimming on, loop controls enabled."""
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
        undefined=jinja2.StrictUndefined if strict else jinja2.Undefined,
    )
    if reference_filters:
        env.filters["tojson"] = transformers_tojson
    return env


def render(env, template, case, extra=None):
    """Render one fixture case. The variables are the ones a chat renderer is expected to expose."""
    variables = {
        "messages": case["messages"],
        "add_generation_prompt": case.get("add_generation_prompt", False),
        "tools": case.get("tools"),
        "bos_token": case.get("bos_token"),
    }
    if extra:
        variables.update(extra)
    return env.from_string(template).render(**variables)


def escaped(text, limit=None, backslash=True):
    """The one-line escape form used in every table: ASCII only, control characters visible."""
    out = []
    for ch in text:
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\" if backslash else "\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\x{code:02x}")
        elif code > 0x7E:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    s = "".join(out)
    if limit and len(s) > limit:
        s = s[:limit] + "..."
    return s


def tokens_for(model, text):
    """Ids for a rendered prompt, from Chapter 11's encoder: NFC, declared specials, byte-level BPE."""
    return booktok.encode(model, text, [(s, i) for s, i, _ in booktok.special_tokens(model)])


# ---------------------------------------------------------------------------------------------------------------
# the sections the chapter quotes
# ---------------------------------------------------------------------------------------------------------------

def program_section(template):
    """What the program is, counted rather than described."""
    exprs = [_strip_strings(part) for part in _inside(template, "{{", "}}")]
    stmts = [_strip_strings(part) for part in _inside(template, "{%", "%}")]
    filters = set()
    words = set()
    for part in exprs + stmts:
        # `| name` with a space is a filter; the templates also contain `<|im_start|>` with no space, which is not.
        filters.update(re.findall(r"\|\s+([a-z_][a-z0-9_]*)", part))
        words.update(_words(part))
    return {
        "chars": len(template),
        "bytes": len(template.encode("utf-8")),
        "lines": template.count("\n") + 1,
        "expressions": template.count("{{"),
        "statements": template.count("{%"),
        "filters": sorted(filters),
        "variables": sorted(w for w in words if w not in ENGINE_NAMES and w not in filters),
        "roles": [r for r in ("system", "user", "assistant", "tool") if f'"{r}"' in template],
        "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }


def _strip_strings(part):
    """One expression with its string literals blanked, so prose inside quotes is not read as a variable."""
    return re.sub(r"'[^']*'|\"[^\"]*\"", "''", part)


# Names the template engine itself supplies or the template body binds locally; everything else is a variable the
# renderer has to pass in, which is the list the chapter prints.
ENGINE_NAMES = {
    "if", "else", "elif", "endif", "for", "endfor", "in", "not", "and", "or", "is", "defined", "undefined",
    "true", "false", "none", "set", "endset", "loop", "break", "continue", "macro", "endmacro", "with",
    "endwith", "block", "endblock", "filter", "endfilter", "raw", "endraw", "index0", "first", "last",
    "length", "revindex", "message", "tool", "tool_call", "content", "role", "function", "arguments", "name",
}

# The names a renderer has to pass in: the four the reference implementation exposes, plus the two flags
# individual templates test for (`enable_thinking` on Qwen3). A template that reads anything else is unrenderable.
VARIABLES = ["messages", "tools", "add_generation_prompt", "bos_token", "enable_thinking"]


def _words(part):
    """Identifier-shaped words in one expression or statement, minus engine names and attribute accesses."""
    out = set()
    for raw in part.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ") \
            .replace("|", " ").replace(",", " ").replace(":", " ").replace("=", " ").split():
        word = raw.strip().strip("'\"")
        if not word or not word[:1].isalpha() or not all(c.isalnum() or c == "_" for c in word):
            continue
        if word in ENGINE_NAMES or word[0].isupper():
            continue
        out.add(word)
    return out


def _inside(template, opening, closing):
    """The text between each `opening` and the following `closing`, in order."""
    out, i = [], 0
    while True:
        a = template.find(opening, i)
        if a < 0:
            return out
        b = template.find(closing, a + len(opening))
        if b < 0:
            return out
        out.append(template[a + len(opening):b])
        i = b + len(closing)


def copies_section(upstream, from_gguf):
    """Two copies of one template: the file the model card ships and the string inside the GGUF."""
    edits = [{"at": i1, "removed": upstream[i1:i2], "inserted": from_gguf[j1:j2]}
             for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, upstream, from_gguf).get_opcodes()
             if tag != "equal"]
    first = edits[0]["at"] if edits else None
    return {
        "upstream": {"chars": len(upstream), "sha256": hashlib.sha256(upstream.encode()).hexdigest()},
        "gguf": {"chars": len(from_gguf), "sha256": hashlib.sha256(from_gguf.encode()).hexdigest()},
        "identical": upstream == from_gguf,
        "edits": [{k: (escaped(v, backslash=False) if isinstance(v, str) else v) for k, v in e.items()}
                  for e in edits],
        "first_difference": first,
        "upstream_around": escaped(upstream[max(0, (first or 0) - 26):(first or 0) + 34], backslash=False) if first is not None else "",
        "gguf_around": escaped(from_gguf[max(0, (first or 0) - 26):(first or 0) + 36], backslash=False) if first is not None else "",
    }


def fixture_section(model, upstream, from_gguf, cases, reference_env, native_env):
    """Every committed case under three conditions.

    The reference renderer's expected bytes came from the template the model card ships (`upstream`), so that is
    the condition that tests *this* renderer. The GGUF carries its own copy of the same program, and the third
    column asks what happens if you render with the copy you actually have. A difference there is not a renderer
    bug, and the two counts are printed separately for that reason.
    """
    rows = []
    for case in cases:
        got = render(reference_env, upstream, case)
        native = render(native_env, upstream, case)
        shipped = render(reference_env, from_gguf, case)
        expected = case["expected"]
        rows.append({
            "name": case["name"],
            "tools": len(case.get("tools") or []),
            "generation_prompt": bool(case.get("add_generation_prompt")),
            "chars": len(got),
            "bytes": len(got.encode("utf-8")),
            "tokens": len(tokens_for(model, got)),
            "matches_reference": got == expected,
            "native_tojson_matches": native == expected,
            "gguf_copy_matches": shipped == expected,
            "first_difference": None if got == expected else _first_difference(got, expected),
            "gguf_copy_difference": None if shipped == expected else _first_difference(shipped, expected),
        })
    return {
        "cases": rows,
        "agree": sum(1 for r in rows if r["matches_reference"]),
        "agree_native": sum(1 for r in rows if r["native_tojson_matches"]),
        "agree_gguf_copy": sum(1 for r in rows if r["gguf_copy_matches"]),
        "total": len(rows),
    }


def _first_difference(got, expected):
    i = next((k for k in range(min(len(got), len(expected))) if got[k] != expected[k]),
             min(len(got), len(expected)))
    return {"at": i, "got": escaped(got[max(0, i - 20):i + 20]), "expected": escaped(expected[max(0, i - 20):i + 20])}


def tojson_section(model, value):
    """Four `tojson` policies on one tool definition: the filter's name is not its specification."""
    policies = [
        ("reference", transformers_tojson(value)),
        ("jinja2-default", json.dumps(value, sort_keys=True, ensure_ascii=False).replace("<", "\\u003c")
         .replace(">", "\\u003e").replace("&", "\\u0026").replace("'", "\\u0027")),
        ("compact", json.dumps(value, separators=(",", ":"))),
        ("ascii", json.dumps(value, ensure_ascii=True)),
    ]
    return [{"policy": name, "chars": len(s), "tokens": len(tokens_for(model, s)), "text": escaped(s, 200)}
            for name, s in policies]


def surface_section(model, template, env):
    """The same conversation with one knob moved: bytes and tokens, nothing else."""
    base = [{"role": "user", "content": "What is the capital of France?"}]
    with_system = [{"role": "system", "content": "You are a concise assistant."}] + base
    tool = [{"type": "function",
             "function": {"name": "get_weather",
                          "parameters": {"type": "object",
                                         "properties": {"city": {"type": "string"}}}}}]
    variants = [
        ("no system message, prompt off", {"messages": base, "add_generation_prompt": False}),
        ("no system message, prompt on", {"messages": base, "add_generation_prompt": True}),
        ("explicit system, prompt off", {"messages": with_system, "add_generation_prompt": False}),
        ("explicit system, prompt on", {"messages": with_system, "add_generation_prompt": True}),
        ("one tool, generation prompt", {"messages": base, "tools": tool, "add_generation_prompt": True}),
        ("one tool, explicit system", {"messages": with_system, "tools": tool, "add_generation_prompt": True}),
    ]
    rows = []
    for name, case in variants:
        text = render(env, template, case)
        rows.append({"variant": name, "chars": len(text), "bytes": len(text.encode()),
                     "tokens": len(tokens_for(model, text)),
                     "head": escaped(text, 60), "tail": escaped(text[-40:])})
    return rows


def anatomy_section(model, template, env):
    """The rendered prompt cut at its role markers: which bytes are yours, which are the template's.

    A special token starts a new segment and is a segment of its own, so the segment token counts add up to the
    whole prompt's - which is checked here rather than assumed, because a table that does not add up is worse
    than no table.
    """
    case = {"messages": [{"role": "user", "content": "What is the capital of France?"}],
            "add_generation_prompt": True}
    text = render(env, template, case)
    marks = [s for s, _, kind in booktok.special_tokens(model) if kind in ("control", "user-defined")]
    cuts, i, segment_start = [], 0, 0
    while i < len(text):
        hit = next((m for m in sorted(marks, key=len, reverse=True) if text.startswith(m, i)), None)
        if hit:
            if segment_start < i:
                cuts.append((segment_start, i))
            cuts.append((i, i + len(hit)))
            i += len(hit)
            segment_start = i
        else:
            i += 1
    if segment_start < len(text):
        cuts.append((segment_start, len(text)))
    rows, total = [], 0
    for start, end in cuts:
        piece = text[start:end]
        ids = tokens_for(model, piece)
        total += len(ids)
        rows.append({"at": start, "bytes": len(piece.encode()), "tokens": len(ids),
                     "ids": ids[:4], "text": escaped(piece, 52)})
    return {"prompt_bytes": len(text.encode()), "prompt_tokens": len(tokens_for(model, text)),
            "sum_of_segments": total, "segments": rows}


def thinking_section(model, template, env):
    """The reasoning flag on a template that has one: what turning it off actually emits."""
    messages = [{"role": "user", "content": "How many r's are in strawberry?"}]
    rows = []
    for label, extra in (("thinking on (default)", {}), ("thinking off", {"enable_thinking": False})):
        text = render(env, template, {"messages": messages, "add_generation_prompt": True}, extra)
        rows.append({"variant": label, "chars": len(text), "tokens": len(tokens_for(model, text)),
                     "tail": escaped(text[-48:])})
    return {"template_chars": len(template), "rows": rows}


PY_EXPRESSIONS = [
    ('"don\'t stop".title()', '"Don\'T Stop"'),
    ('"  a  b  ".split(None, 1)', "['a', 'b  ']"),
    ('"a,b,,c".split(",")', "['a', 'b', '', 'c']"),
    ('"x".ljust(4, "-")', "'x---'"),
    ('"a\\nb\\n".splitlines(True)', "['a\\n', 'b\\n']"),
    ('"  x  ".rstrip()', "'  x'"),
    ('"</think>abc".split("</think>")[-1].lstrip("\\n")', "'abc'"),
]


def pystr_section():
    """The Python the templates call, evaluated by Python: the answers a renderer has to match."""
    rows = []
    for expr, want in PY_EXPRESSIONS:
        got = eval(expr)  # noqa: S307 - the expressions are literals in this file, not input
        rows.append({"expression": expr, "python": repr(got), "recorded": want, "agrees": repr(got) == want})
    return rows


REFUSALS = [
    ("undefined variable", "{{ nosuchvar }}"),
    ("unknown filter", "{{ 'x' | nosuchfilter }}"),
    ("unclosed statement", "{% if tools %}x"),
    ("index into nothing", "{{ messages[0]['content'] }}"),
    ("unknown method", "{{ 'x'.nosuchmethod() }}"),
]


def refusal_section(default_env, strict_env):
    """What a renderer does with a template it cannot honour - and what the model would have seen."""
    rows = []
    for name, template in REFUSALS:
        row = {"case": name, "template": escaped(template)}
        for label, env in (("default", default_env), ("strict", strict_env)):
            try:
                text = env.from_string(template).render(messages=[], tools=None)
                row[label] = f"renders {escaped(text, 24)!r}"
            except Exception as exc:  # noqa: BLE001 - the error class is the measurement
                row[label] = f"raises {type(exc).__name__}"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def collect(gguf_path, minfer):
    meta = booktok.read_tokenizer(str(gguf_path))
    if "tokenizer.chat_template" not in meta:
        raise SystemExit(f"{gguf_path}: no tokenizer.chat_template - this chapter needs the template string")
    model = booktok.load_vocab(str(gguf_path))
    template = meta["tokenizer.chat_template"]

    fixtures = {}
    for name in ("qwen2.5-0.5b-instruct", "tools_qwen2.5-0.5b-instruct", "qwen3-0.6b"):
        path = minfer / "tests" / "fixtures" / "chat" / f"{name}.json"
        if path.exists():
            fixtures[name] = json.loads(path.read_text())
    if "qwen2.5-0.5b-instruct" not in fixtures:
        raise SystemExit(f"no chat fixtures under {minfer}; pass --minfer PATH")

    reference_env, native_env = environment(), environment(reference_filters=False)
    plain = fixtures["qwen2.5-0.5b-instruct"]
    tool_fixture = fixtures.get("tools_qwen2.5-0.5b-instruct")
    qwen3 = fixtures.get("qwen3-0.6b")

    cases = list(plain["cases"]) + list(tool_fixture["cases"] if tool_fixture else [])
    tool_value = (tool_fixture["cases"][0]["tools"][0] if tool_fixture else
                  {"type": "function", "function": {"name": "f"}})

    data = {
        "script": "scripts/tmpl.py",
        "machine": f"{platform.machine()}, {platform.system()} {platform.release()}",
        "compiler": f"python {platform.python_version()}, jinja2 {jinja2.__version__}",
        "invocation": f"python3 scripts/tmpl.py ~/.cache/myfer/{pathlib.Path(gguf_path).name} "
                      f"--out facts/template.json",
        "measurement": (
            "the prompt bytes are compared with the reference renderer's committed output, one case at a time; "
            "the token counts come from this book's Chapter 11 tokenizer on the same bytes"
        ),
        "gguf": {
            "file": pathlib.Path(gguf_path).name,
            "sha256": hashlib.sha256(open(gguf_path, "rb").read()).hexdigest(),
            "template_key": "tokenizer.chat_template",
        },
        "witness": {
            "reference": plain["provenance"]["reference"],
            "template_source": plain["provenance"]["template_source"],
            "fixture_sha256": hashlib.sha256(
                (minfer / "tests" / "fixtures" / "chat" / "qwen2.5-0.5b-instruct.json").read_bytes()).hexdigest(),
            "recorded_template_sha256": plain["template_sha256"],
            "recorded_gguf_template_sha256": plain["gguf_template_sha256"],
        },
        "program": program_section(template),
        "copies": copies_section(plain["template"], template),
        "fixtures": fixture_section(model, plain["template"], template, cases, reference_env, native_env),
        "tojson": tojson_section(model, tool_value),
        "surfaces": surface_section(model, template, reference_env),
        "anatomy": anatomy_section(model, template, reference_env),
        "pystr": pystr_section(),
        "refusal": refusal_section(environment(), environment(strict=True)),
    }
    if qwen3:
        data["thinking"] = thinking_section(model, qwen3["template"], reference_env)
    return data


def main():
    parser = argparse.ArgumentParser(description="Render a model's chat template and measure the result.")
    parser.add_argument("gguf")
    parser.add_argument("--out", default=str(ROOT / "facts" / "template.json"))
    parser.add_argument("--minfer", default=str(DEFAULT_MINFER))
    args = parser.parse_args()

    data = collect(pathlib.Path(args.gguf).expanduser(), pathlib.Path(args.minfer).expanduser())
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")
    f = data["fixtures"]
    print(f"wrote {out}: {f['agree']}/{f['total']} cases byte-identical to the reference renderer, "
          f"{f['agree_native']}/{f['total']} with Jinja2's own tojson, "
          f"{f['agree_gguf_copy']}/{f['total']} with the copy inside the GGUF")


if __name__ == "__main__":
    main()
