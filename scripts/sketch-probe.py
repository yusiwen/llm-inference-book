#!/usr/bin/env python3
"""Count the monospace blocks that are secretly pictures or hand-aligned tables.

The rule these stand for: a fence may hold program output, source code, or a numeric derivation whose
point is that the reader can check the digits. A bit layout, a byte plan, a pipeline or a dataflow drawn
with `+ - | >` is a *sketch* and belongs in an SVG; a multi-row ledger aligned with spaces is a *table*,
where the columns are structure instead of padding and the numbers can come from an artifact.

The defect is invisible to the build: an ASCII sketch compiles, deploys, and looks deliberate. So this
runs as a ratchet. `--allow N` is the backlog the book admits to, and the number may only go down: when a
block is converted, lower the allowance in the same commit.

    python3 scripts/sketch-probe.py             # list them, with the line number of each fence
    python3 scripts/sketch-probe.py --allow 10  # gate: fails if the backlog grew

The heuristic is deliberately generous and every hit is a question for a human, not a verdict. A block is
a candidate when it has no code tokens and any of: a box-drawing character, a rule line of `- = + _`, a
`-->` arrow, a column pipe, or three consecutive lines aligned with a run of spaces.

Known keepers, reviewed by hand (they stay monospace on purpose and are counted in the allowance):
`src/02-the-file-is-the-model/index.md:123` is `gguf-inspect.py` output, and
`src/08-gemv-gemm-and-the-thread-pool/index.md:157` is a list of pseudocode steps rather than a sketch.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

CODE = re.compile(r"\b(fn|let|mut|impl|return|def|class|for|while|if|match|const|static|struct|enum)\b"
                  r"|//|#include|=>|;\s*$|\$\s+[a-z]")
BOX = re.compile("[\u2500-\u257f]")
ARROW = re.compile(r"-->|<--|<->")
RULE = re.compile(r"^\s*[-=+_]{4,}\s*$")
PIPE_COLUMN = re.compile(r"^\s*\|")            # a table drawn with pipes at the line start
PIPE_PADDED = re.compile(r"\s{2,}\|\s{2,}")    # an interior pipe with padding, so a column, not a word
ALIGNED = re.compile(r"^[^\s|].{0,40}?\S\s{3,}\S")


def blocks(text):
    """Yield (line number of the first body line, body) per fenced block."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if re.match(r"^\s*(```+|~~~+)", lines[i]):
            start = i + 1
            i += 1
            body = []
            while i < len(lines) and not re.match(r"^\s*(```+|~~~+)", lines[i]):
                body.append(lines[i])
                i += 1
            yield start, "\n".join(body)
        i += 1


def why(body):
    """Return the reason this block is a candidate, or None."""
    if CODE.search(body):
        return None
    lines = body.split("\n")
    if BOX.search(body) or any(RULE.match(line) for line in lines):
        return "rules and boxes"
    if any(ARROW.search(line) for line in lines):
        return "arrows"
    if any(PIPE_COLUMN.match(line) or PIPE_PADDED.search(line) for line in lines):
        return "column pipes"
    run = 0
    for line in lines:
        run = run + 1 if ALIGNED.match(line) else 0
        if run >= 3:
            return "aligned columns"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow", type=int, default=None,
                    help="fail when more candidates than this are found (the committed backlog)")
    args = ap.parse_args()

    hits = []
    for page in sorted(SRC.rglob("*.md")):
        for line_no, body in blocks(page.read_text()):
            reason = why(body)
            if reason:
                first = body.strip().split("\n")[0][:66]
                hits.append(f"{page.relative_to(ROOT)}:{line_no}  [{reason}]  {first}")

    for hit in hits:
        print(f"  {hit}")
    print(f"{len(hits)} monospace block(s) are pictures or hand-aligned ledgers, not program output")

    if args.allow is not None and len(hits) > args.allow:
        print(f"FAIL: {len(hits)} against an allowance of {args.allow} - draw the sketch as SVG, make the "
              f"ledger a facts table, or raise the allowance deliberately in the same commit with a reason",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
