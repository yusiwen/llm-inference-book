#!/usr/bin/env python3
"""Keep the reference index derived from the chapters.

The index in src/appendix/references.md lists every source the book sends a reader to, one row per
cite key. A chapter cites a key by writing it in square brackets in its `## Reading` section:
`- [gqa] Ainslie et al., ... - arXiv:2305.13245`.

What this script enforces (exit 1 on any failure):

  1. a chapter that cites a key not on the index  - a dangling citation
  2. a chapter (or the opening part) that cites no external source at all
  3. a "used in" column that is stale with respect to the pages

Run with `--check` from the gate; run without it after adding a citation, and commit the page.
"""

import pathlib
import re
import sys

ROW = re.compile(r"^\| `\[([a-z0-9][a-z0-9-]*)\]` \| .*? \| .*? \| (.*?) \|$", re.M)
ANY_KEY = re.compile(r"\[([a-z][a-z0-9-]{1,40})\]")


def pages(src):
    """(chapter number, path) for the opening part and every chapter, in order."""
    out = [(0, src / "start" / "index.md")]
    out += [(int(p.parent.name.split("-")[0]), p) for p in sorted(src.glob("[0-9][0-9]-*/index.md"))]
    return [(n, p) for n, p in out if p.exists()]


FENCE = re.compile(r"^```.*?^```", re.M | re.S)
INLINE = re.compile(r"`[^`]*`")


def keys_cited(text):
    """Cite keys in prose only.

    A shell command such as `pip install "huggingface_hub[cli]"` contains a bracketed word that looks
    exactly like a cite key, and an earlier version of this script reported it as a dangling citation
    from Part 0 - the page was right and the checker was wrong. Code is not prose: strip it first.
    """
    return set(ANY_KEY.findall(INLINE.sub("", FENCE.sub("", text))))


def main(*argv):
    check = "--check" in argv
    root = pathlib.Path(".").resolve()
    src = root / "src"
    page = src / "appendix" / "references.md"
    text = page.read_text()
    indexed = {m.group(1) for m in ROW.finditer(text)}
    if not indexed:
        print("FAIL: no reference rows found in src/appendix/references.md", file=sys.stderr)
        return 1

    hard = []
    used = {k: [] for k in indexed}
    for num, path in pages(src):
        body = path.read_text()
        cited = keys_cited(body) & indexed
        if not cited:
            hard.append(f"{path.relative_to(root)}: cites no source from the reference index")
        for k in cited:
            used[k].append(num)
        unknown = keys_cited(body) - indexed
        for k in sorted(unknown):
            hard.append(f"{path.relative_to(root)}: cites [{k}], which is not in the reference index")

    new = text
    stale = []
    for m in ROW.finditer(text):
        k = m.group(1)
        want = ", ".join(str(n) for n in sorted(set(used[k]))) or "-"
        if want != m.group(2):
            stale.append(f"references.md: [{k}] used in {m.group(2)!r}, pages say {want!r}")
            new = new.replace(m.group(0), m.group(0)[: m.start(2) - m.start(0)] + want + " |", 1)

    for s in stale:
        print(f"[E] {s}")
    for s in hard:
        print(f"[E] {s}")
    if hard:
        print(f"FAIL: {len(hard)} hard finding(s)")
        return 1
    if stale:
        if check:
            print(f"FAIL: {len(stale)} stale row(s) - run `python3 scripts/references.py` and commit")
            return 1
        page.write_text(new)
        print(f"rewrote {len(stale)} row(s)")
        return 0
    print(f"clean ({len(indexed)} sources, {len(pages(src))} pages, all cited)")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
