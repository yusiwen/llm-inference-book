#!/usr/bin/env python3
"""A figure's numbers are prose: check that every one of them exists in *that chapter's* artifact.

`scripts/facts.py` regenerates the tables, so a stale number in a table survives about as long as it takes to
run `--render`. A figure does not: its captions are hand-drawn `<text>` like a sentence, and nothing regenerates
them. Chapter 20's two diagrams were still saying 4.31 ms and 12.50 ms after a re-measurement moved the chapter
to 3.98 and 10.49 - in the visible captions *and* in the `aria-label` a screen reader reads.

The check is deliberately narrow. A number with a unit inside `src/figures/NN-*.svg` must appear in one of the
`facts/*.json` artifacts that the chapter `src/NN-*/index.md` renders its tables from. Comparing against every
artifact in the tree looks thorough and is useless: a two-decimal value collides with some float somewhere, and
the check passes exactly when it matters.

It does not check that the number is attached to the right claim - a human reads that - only that the figure was
not left behind by a re-run.

    python3 scripts/fignumbers.py            # the gate's check
    python3 scripts/fignumbers.py --list     # every unit-bearing number and the artifact that has it

Numbers deliberately *not* from that chapter's measurement go in `reference/figure-numbers.txt`, one per line:
this script's only escape hatch, and it should stay short.
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Volume units are deliberately absent: `473.9 MiB` is a division of a byte count the artifact does hold, and a
# checker that cannot do the division would only teach the author to write bytes in a chart.
UNITS = r"(?:ms|us|ns|s|tok/s|GB/s|GiB/s|MB/s|B/s)"
PATTERN = re.compile(r"(?<![\w.])(\d[\d,]*\.?\d*)\s?(" + UNITS + r")(?![\w/])")
ALLOW = ROOT / "reference" / "figure-numbers.txt"


def numbers_in(obj, out):
    """Every number an artifact holds. Strings are scanned: they carry formatted values and quoted messages."""
    if isinstance(obj, dict):
        for v in obj.values():
            numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            numbers_in(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        # Scientific notation is skipped: its mantissa is not a number anyone quotes, and `-4.31538974e-05` was
        # matching a figure's `4.31 ms` in the first version of this check, which passed a stale caption.
        for m in re.findall(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", obj):
            if "e" in m.lower():
                continue
            try:
                out.add(float(m.replace(",", "")))
            except ValueError:
                pass


def chapter_artifacts(index):
    """The artifact names a chapter's own facts blocks render from, e.g. '20' -> {'loop'}."""
    pages = sorted((ROOT / "src").glob(f"{index}-*/index.md"))
    names = set()
    for p in pages:
        names |= set(re.findall(r"<!-- facts: ([a-z0-9-]+)\.", p.read_text()))
    return names


def allowed():
    if not ALLOW.exists():
        return set()
    return {line.strip() for line in ALLOW.read_text().splitlines() if line.strip() and not line.startswith("#")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print every figure number that does resolve")
    args = ap.parse_args()

    cache, allow = {}, allowed()
    figures = sorted((ROOT / "src" / "figures").glob("*.svg"))
    bad, seen, resolved = [], 0, 0
    for p in figures:
        index = p.name.split("-")[0]
        if index not in cache:
            names = chapter_artifacts(index)
            have = set()
            for name in names:
                f = ROOT / "facts" / f"{name}.json"
                if f.exists():
                    numbers_in(json.loads(f.read_text()), have)
            cache[index] = (names, have)
        names, have = cache[index]
        text = p.read_text()
        for m in PATTERN.finditer(text):
            seen += 1
            try:
                v = float(m.group(1).replace(",", ""))
            except ValueError:
                continue
            hit = any(abs(v - x) <= max(0.006, 0.002 * abs(v)) for x in have)
            if hit or m.group(0) in allow:
                resolved += 1
                if args.list:
                    print(f"ok   {p.name}: {m.group(0)}")
                continue
            where = ", ".join(sorted(names)) + ".json" if names else "no artifact"
            ctx = re.sub(r"\s+", " ", text[max(0, m.start() - 60):m.end() + 10]).strip()
            bad.append(f"{p.relative_to(ROOT)}: {m.group(0)!r} is in none of {where}\n      ...{ctx}")

    print(f"figures: {len(figures)} files, {seen} unit-bearing number(s), {resolved} resolved to a chapter artifact")
    if bad:
        print("")
        for b_ in bad:
            print(f"[E] {b_}")
        print("\nA figure caption is prose. Either re-measure, or add the value to "
              "reference/figure-numbers.txt if it is deliberately not a measurement.")
        return 1
    print("every number in a figure comes from the artifact its chapter renders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
