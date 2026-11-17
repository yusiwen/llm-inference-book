#!/usr/bin/env python3
"""Third-party requirements of the instruments, and a message that says what to do about them.

The gate (`scripts/verify.sh`) needs nothing but the standard library. The instruments - the scripts that
write `facts/*.json` - need four packages, and they are not part of the gate, so a missing one surfaces as a
traceback in the middle of a measurement run rather than as a failing check. That is how three measurement runs
died on `ModuleNotFoundError: No module named 'regex'` while `verify.sh` was green.

Two ways to provide them. The first is the one this repository is built around, because `flake.lock` pins the
versions and an instrument whose numpy can change under it cannot back a number that claims a commit:

    nix develop --command python3 scripts/fuse.py ...
    uv run --python 3.15 --with regex --with jinja2 --with numpy --with tokenizers python3 scripts/fuse.py ...
"""

import sys

NIX = "nix develop --command python3 <script> ..."
UV = ("uv run --python 3.15 --with regex --with jinja2 --with numpy "
      "--with tokenizers python3 <script> ...")

REQUIREMENTS = {
    "regex": "the Qwen2 pre-tokenizer pattern (scripts/tok.py)",
    "jinja2": "rendering a model's own chat template (scripts/tmpl.py)",
    "numpy": "the f64 witnesses and the logits replay (scripts/smp.py, the -check scripts)",
    "tokenizers": "the HuggingFace reference tokenizer (scripts/tok-check.py)",
}


def need(module):
    """Import a third-party module, or exit 2 with the command that would have provided it."""
    try:
        return __import__(module)
    except ModuleNotFoundError:
        sys.stderr.write(
            "\n"
            "{} is not available in this Python ({}).\n"
            "  needed for: {}\n"
            "  with nix:   {}\n"
            "  without:    {}\n"
            "\n".format(module, sys.executable, REQUIREMENTS.get(module, "an instrument"), NIX, UV))
        raise SystemExit(2)


def missing():
    """Which of the four cannot be imported here. The gate reports this and fails nothing."""
    out = []
    for module in REQUIREMENTS:
        try:
            __import__(module)
        except ModuleNotFoundError:
            out.append(module)
    return out


if __name__ == "__main__":
    gone = missing()
    if gone:
        sys.stderr.write("missing: {}\n  with nix: {}\n  without:  {}\n".format(", ".join(gone), NIX, UV))
        raise SystemExit(1)
    print("all four instrument dependencies import cleanly")
