#!/usr/bin/env python3
"""Mechanical checks for this book. Run: python3 scripts/probe.py [repo]

Checks that the build cannot see:
  1. every page under src/ carries the fixed skeleton sections, in order
  2. every relative link resolves to a file that exists
  3. a checklist item pairs with a ### Lesson heading once the chapter has prose
  4. no stub leftover marker in a page whose status is draft or written
  5. the landing page's chapter and lesson counts match the tree
Exit 1 on a hard failure (missing section, broken link, count drift).
"""

import pathlib
import re
import sys

REQUIRED = [
    "First, the gist",
    "What you'll be able to do",
    "Lessons",
    "Deliverable",
    "How you know you passed",
    "Reading",
    "Backfill",
]
STUB_LEFTOVER = re.compile(r"To fill in:|TODO|FIXME")


def strip_fences(text):
    """Drop fenced blocks: the conventions page quotes the skeleton inside one."""
    return re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)


def h2s(text):
    return [m.group(1).strip() for m in re.finditer(r"^## (.+)$", strip_fences(text), re.M)]


def main(root_arg="."):
    root = pathlib.Path(root_arg).resolve()
    src = root / "src"
    hard, soft = [], []
    # SUMMARY.md is the manifest and index.md is the landing page: neither is a reading page.
    pages = sorted(p for p in src.rglob("*.md") if p.name != "SUMMARY.md" and p != src / "index.md")
    chapter_files = sorted(src.glob("[0-9][0-9]-*/index.md"))

    for p in pages:
        rel = p.relative_to(root)
        text = p.read_text()
        h2 = h2s(text)
        # A chapter and the opening part carry the full skeleton. A reference page (appendix,
        # landing page) owes a status block and a Reading section instead: a glossary with a
        # Lessons checklist is ceremony, and a check that demands one gets ignored.
        is_chapter = bool(re.match(r"src/(start/|[0-9]{2}-)", str(rel)))
        required = REQUIRED if is_chapter else ["Reading"]
        missing = [s for s in required if s not in h2]
        if missing:
            hard.append(f"{rel}: missing section(s): {', '.join(missing)}")
        seen = [s for s in h2 if s in required]
        if seen != sorted(seen, key=required.index):
            hard.append(f"{rel}: sections out of order: {' > '.join(seen)}")

        # 2. relative links
        for m in re.finditer(r"\]\(([^)#]+?)(?:#[^)]*)?\)", text):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (p.parent / target).resolve()
            if not resolved.exists():
                hard.append(f"{rel}: broken link -> {target}")

        # 3. checklist item vs ### Lesson heading, when lessons exist
        status = re.search(r"Status\*\*: (\w+)", text)
        status = status.group(1) if status else "?"
        lesson_heads = [m.group(1).strip() for m in re.finditer(r"^### Lesson \d+ · (.+)$", text, re.M)]
        if lesson_heads:
            body = text.split("## Lessons")[1].split("\n## ")[0]
            items = [m.group(1) for m in re.finditer(r"^- \[ \] \d+\. (.+)$", body, re.M)]
            if len(items) != len(lesson_heads):
                soft.append(f"{rel}: {len(items)} checklist item(s) but {len(lesson_heads)} lesson heading(s)")
            for item, head in zip(items, lesson_heads):
                clause = re.split(r"[:(]| -- ", item)[0].strip().lower()
                if not head.lower().startswith(clause[:24]):
                    soft.append(f"{rel}: item \u2194 heading mismatch: {item!r} / {head!r}")

        # 4. stub leftovers on a page that claims prose
        if status in ("draft", "written") and STUB_LEFTOVER.search(text):
            hard.append(f"{rel}: status is {status} but the page still carries a stub leftover")

    # 5. the landing page's counts
    index = (src / "index.md").read_text()
    n_ch = len(list(src.glob("[0-9][0-9]-*/index.md")))
    n_lessons = sum(
        len(re.findall(r"^- \[ \] \d+\. ", open(f).read().split("## Lessons")[1].split("\n## ")[0], re.M))
        for f in chapter_files
    )
    for label, real in (("chapters", n_ch), ("lessons", n_lessons)):
        m = re.search(r"(\d+)\s+" + label + r"\b", index)
        if m and int(m.group(1)) != real:
            hard.append(f"src/index.md: says {m.group(1)} {label}, the tree has {real}")
    if f"**{n_ch} chapters, {n_lessons} lessons**" not in index:
        hard.append(f"src/index.md: the map summary is not '{n_ch} chapters, {n_lessons} lessons'")

    print(f"pages={len(pages)} chapters={n_ch} lessons={n_lessons}")
    for s in soft:
        print(f"[W] {s}")
    for s in hard:
        print(f"[E] {s}")
    if hard:
        print(f"FAIL: {len(hard)} hard finding(s)")
        return 1
    print("clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
