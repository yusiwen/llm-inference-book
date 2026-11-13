#!/usr/bin/env python3
"""Check the book's own counts, the way the drift tool checks minfer's.

The book says "counts are claims" and derives every number about minfer with a command. That rule kept
failing for the book itself: the measurements page went stale twice in one afternoon, because the gate
only checked that the page still carried its commands, not that its numbers were still true.

So this script re-derives the book's inventory and compares it with:

  src/appendix/measurements.md   the "The book itself" table
  src/index.md                   the "N chapters, M lessons" line and the per-part map table

Nothing here is hardcoded except the labels it looks for. Chapters are `src/NN-slug/index.md`, lessons
are checklist items under `## Lessons`, sources are rows of the reference index, citations are entries
of the committed drift inventory, and figures are rows of FIGURES.md against files in src/figures/.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def pages(src):
    return sorted(p for p in src.rglob("*.md"))


def chapter_files(src):
    return sorted(src.glob("[0-9][0-9]-*/index.md"))


def lessons_of(path):
    body = path.read_text().split("## Lessons")[1].split("\n## ")[0]
    return len(re.findall(r"^- \[ \] \d+\. ", body, re.M))


def derive(root):
    src = root / "src"
    chapters = chapter_files(src)
    per_chapter = {int(p.parent.name.split("-")[0]): lessons_of(p) for p in chapters}
    sources = len(re.findall(r"^\| `\[[a-z0-9][a-z0-9-]*\]`", (src / "appendix" / "references.md").read_text(), re.M))
    inv_path = root / "reference" / "baseline-citations.json"
    citations = len(json.loads(inv_path.read_text())["citations"]) if inv_path.exists() else 0
    fig_plan = len(re.findall(r"^\| \d+\.\d+ \|", (root / "FIGURES.md").read_text(), re.M))
    fig_drawn = len(list((src / "figures").glob("*.svg"))) if (src / "figures").exists() else 0
    return {
        "Chapters": len(chapters),
        "Lessons (checklist items under `## Lessons`)": sum(per_chapter.values()),
        "Pages under `src/`": len(pages(src)),
        "Prose bytes under `src/`": sum(p.stat().st_size for p in pages(src)),
        "Sources in the reference index": sources,
        "minfer citations tracked": citations,
        "Figures planned / drawn": f"{fig_plan} / {fig_drawn}",
    }, per_chapter


def check_measurements(root, want):
    path = root / "src" / "appendix" / "measurements.md"
    text = path.read_text()
    bad = []
    for label, expected in want.items():
        m = re.search(r"^\| " + re.escape(label) + r" \| (.+?) \|", text, re.M)
        if not m:
            bad.append(f"measurements.md: no row labelled {label!r}")
            continue
        got = m.group(1).strip()
        if got != str(expected):
            bad.append(f"measurements.md: {label} says {got}, the tree says {expected}")
    return bad, text


def check_index(root, per_chapter):
    path = root / "src" / "index.md"
    text = path.read_text()
    bad = []
    total = sum(per_chapter.values())
    if f"**{len(per_chapter)} chapters, {total} lessons**" not in text:
        bad.append(f"index.md: the map summary is not '**{len(per_chapter)} chapters, {total} lessons**'")
    # per-part rows: | **1 The ground truth** | 1-5 | 25 | ...
    for m in re.finditer(r"^\| \*\*(\d) [^|]+\*\* \| (\d+)-(\d+) \| (\d+) \|", text, re.M):
        part, lo, hi, claimed = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        real = sum(v for k, v in per_chapter.items() if lo <= k <= hi)
        if real != claimed:
            bad.append(f"index.md: part {part} ({lo}-{hi}) claims {claimed} lessons, the chapters hold {real}")
    return bad


def main(*argv):
    root = pathlib.Path(argv[0]).resolve() if argv and not argv[0].startswith("-") else ROOT
    want, per_chapter = derive(root)
    bad, _ = check_measurements(root, want)
    bad += check_index(root, per_chapter)
    for k, v in want.items():
        print(f"  {k}: {v}")
    for b in bad:
        print(f"[E] {b}")
    if bad:
        print(f"FAIL: {len(bad)} count(s) disagree with the tree")
        return 1
    print("clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
