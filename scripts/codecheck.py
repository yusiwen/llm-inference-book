#!/usr/bin/env python3
"""Verify that every quoted line of minfer code in the book still matches the pinned baseline.

A book that quotes source code makes a claim that decays: minfer moves, the prose stays, and a reader who
follows `path::symbol` finds something else. The numbers in this book are already guarded (facts/*.json and a
gate); this is the same guard for quotations.

    python3 scripts/codecheck.py --check     # every block matches the pinned source, or fail
    python3 scripts/codecheck.py --vendor    # re-copy the quoted files out of a local minfer checkout
    python3 scripts/codecheck.py --list      # what is quoted, and from where
    python3 scripts/codecheck.py --coverage  # lessons that quote nothing, ratcheted per page

A block is written as:

    <!-- code: src/graph/cpu_backend.rs::causal_span@88ffa5d -->
    ```rust
    ...lines copied from the file, verbatim...
    ```
    <!-- /code -->

Lines are compared after trimming trailing whitespace and leading indentation, in order, as a subsequence of
the file - so a re-indent is tolerated and a changed token is not. A line that is exactly `// ...` is the
elision marker and is skipped.

What this cannot check: whether the excerpt is the right passage, or whether an elision hides the one line
that would contradict the prose. That is review, not a gate, and the AGENTS.md rule says so.

The sources are vendored under `reference/pinned/<commit>/` so the check runs in CI without a minfer
checkout - and so a history rewrite upstream cannot silently change what the book verified against.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PINNED = ROOT / "reference" / "pinned"
BASELINE = ROOT / "reference" / "baseline-citations.json"
BASE = ROOT / "reference" / "coverage-baseline.json"
MARKER = re.compile(
    r"<!--\s*code:\s*(?P<path>[\w./-]+)::(?P<sym>[^@\n]+?)@(?P<commit>[0-9a-f]{7,40})\s*-->\n"
    r"(?P<fence>```[a-zA-Z]*)\n(?P<body>.*?)\n```\n<!--\s*/code\s*-->",
    re.S,
)
ELISION = re.compile(r"^\s*//\s*\.\.\.\s*$")


def baseline_commit():
    return json.loads(BASELINE.read_text())["baseline"]["commit"]


def blocks():
    """Every marked block in the book: (markdown_file, line_no, path, symbol, commit, body lines)."""
    out = []
    for md in sorted((ROOT / "src").rglob("*.md")):
        text = md.read_text()
        for m in MARKER.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            body = [ln for ln in m.group("body").split("\n") if not ELISION.match(ln)]
            out.append((md, line_no, m.group("path"), m.group("sym"), m.group("commit"), body))
    return out


def vendor(commit, paths, repo):
    """Copy the quoted files out of a minfer checkout at `commit`, so the check needs no checkout later."""
    dest_root = PINNED / commit[:7]
    for path in sorted(paths):
        show = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"],
                              capture_output=True, text=True)
        if show.returncode:
            raise SystemExit(f"cannot read {path}@{commit} from {repo}: {show.stderr.strip()}")
        dest = dest_root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(show.stdout)
        print(f"  vendored {path} ({len(show.stdout.splitlines())} lines) -> {dest.relative_to(ROOT)}")


def unparsed():
    """A marker the regex cannot read is an excerpt nothing verifies - so it is a failure, not a skip."""
    out = []
    for md in sorted((ROOT / "src").rglob("*.md")):
        text = md.read_text()
        for m in re.finditer(r"<!--\s*code:", text):
            if not MARKER.match(text, m.start()):
                out.append(f"{md.relative_to(ROOT)}:{text[:m.start()].count(chr(10)) + 1}: marker does not "
                           f"parse - the excerpt below it would go unchecked (symbol charset, fence, "
                           f"or a missing <!-- /code -->)")
    return out


def check():
    commit = baseline_commit()
    items = blocks()
    bad, checked = unparsed(), 0
    for md, line_no, path, sym, at, body in items:
        where = f"{md.relative_to(ROOT)}:{line_no}"
        if not at.startswith(commit[:len(at)]) and not commit.startswith(at):
            bad.append(f"{where}: excerpt of {sym} is pinned to {at}, the baseline is {commit[:7]}")
            continue
        src_path = PINNED / commit[:7] / path
        if not src_path.exists():
            bad.append(f"{where}: {path} is not vendored at {commit[:7]} - run --vendor")
            continue
        src = [ln.rstrip() for ln in src_path.read_text().split("\n")]
        j = 0
        for k, line in enumerate(body, 1):
            want = line.lstrip().rstrip()
            if not want:
                continue
            found = None
            for n in range(j, len(src)):
                if src[n].lstrip().rstrip() == want:
                    found = n
                    break
            if found is None:
                bad.append(f"{where}: {sym} line {k} of the excerpt is not in {path} at {commit[:7]} "
                           f"(searched from source line {j + 1}): {want[:90]!r}")
                break
            j = found + 1
        checked += 1
    for b in bad:
        print(f"[E] {b}")
    if bad:
        print(f"FAIL: {len(bad)} excerpt(s) no longer match the pinned source")
        return 1
    files = len({p for _, _, p, _, _, _ in items})
    print(f"clean ({checked} excerpt(s) from {files} file(s) at {commit[:7]})")
    return 0


def listing():
    commit = baseline_commit()
    for md, line_no, path, sym, at, body in blocks():
        print(f"  {md.relative_to(ROOT)}:{line_no}  {path}::{sym}@{at[:7]}  {len(body)} line(s)")
    print(f"{len(blocks())} excerpt(s), baseline {commit[:7]}")
    return 0


def coverage(write_baseline):
    """Every lesson should quote the engine, or say in one line why nothing can be quoted.

    A lesson that explains an engine in prose alone cannot be checked against it - the failure this book
    keeps closing. So each `### Lesson` either contains a marked excerpt or carries
    `<!-- no-code: the reason -->`. The count is ratcheted **per page**: a page may only lose bare lessons
    against `reference/coverage-baseline.json`, and a page that is not in that file starts at zero. New
    chapters therefore cannot be written without quotes; old ones burn down.
    """
    lessons = re.compile(r"^### Lesson (\d+)\b", re.M)
    current, total = {}, 0
    for md in sorted((ROOT / "src").rglob("*.md")):
        text = md.read_text()
        heads = list(lessons.finditer(text))
        if not heads:
            continue
        rel = str(md.relative_to(ROOT))
        missing = []
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            nxt = re.search(r"^## ", text[h.end():end], re.M)
            span = text[h.end(): h.end() + nxt.start()] if nxt else text[h.end():end]
            total += 1
            if MARKER.search(span) or re.search(r"<!--\s*no-code:", span):
                continue
            missing.append(h.group(1))
        if missing:
            current[rel] = missing

    if write_baseline:
        BASE.write_text(json.dumps({"bare": {k: len(v) for k, v in sorted(current.items())}}, indent=2) + "\n")
        print(f"wrote {BASE.relative_to(ROOT)}: {sum(len(v) for v in current.values())} bare lesson(s)")
        return 0

    base = json.loads(BASE.read_text())["bare"] if BASE.exists() else {}
    bad = []
    for rel, missing in sorted(current.items()):
        allowed = base.get(rel, 0)
        if len(missing) > allowed:
            bad.append(f"{rel}: lesson(s) {', '.join(missing)} quote nothing and say nothing about why "
                       f"({len(missing)} now, {allowed} allowed)")
    for b in bad:
        print(f"  {b}")
    if bad:
        print(f"FAIL: {len(bad)} page(s) above their bare-lesson baseline - quote the engine, or add "
              f"<!-- no-code: why -->")
        return 1
    n = sum(len(v) for v in current.values())
    print(f"clean ({n} lesson(s) without an excerpt, baseline {sum(base.values())}, {total} lessons)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--vendor", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--repo", default=str(pathlib.Path.home() / "git" / "mine" / "minfer"))
    args = ap.parse_args()
    if args.vendor:
        vendor(baseline_commit(), {b[2] for b in blocks()}, args.repo)
        return 0
    if args.list:
        return listing()
    if args.coverage:
        return coverage(args.write_baseline)
    if args.check:
        return check()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
