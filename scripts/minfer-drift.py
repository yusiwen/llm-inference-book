#!/usr/bin/env python3
"""Track drift between this book and the minfer engine it describes.

Three kinds of drift matter, and only the first two are mechanical:

  address   a path the book cites moved, vanished, or a symbol was renamed
  number    a count the book prints (ADRs, kernels, step records, LOC, tests)
  behaviour  a mechanism changed while every name stayed the same

This script covers address and number. Behaviour is covered by
`scripts/contract-probes.sh`, which records real runs and is compared by a human.

Modes
-----
  --repo <dir> --ref <commit>   compare the book against a check-out of minfer
  --repo <dir> --head           compare against the tip of that check-out (advisory)
  --write-inventory <file>      record the result as the committed inventory
  --check-inventory <file>      offline: every citation must be in the inventory and resolved

The book's gate runs `--check-inventory`: it needs no network and no minfer
check-out, so CI stays hermetic. Bumping the baseline is an author's action and
is the only time `--repo` is used.

Exit codes: 0 clean, 1 hard finding, 2 the repo or ref could not be read.
"""

import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys
import time

# A citation is a backticked minfer path, optionally followed by `::symbol`.
CITATION = re.compile(r"`((?:src|docs|crates|tests|examples)/[A-Za-z0-9_./+-]+)(?:::(?:r#)?([A-Za-z_][A-Za-z0-9_]*))?`")
# The book's own tree shares the `src/` prefix; those are not minfer paths.
BOOK_OWN = re.compile(r"^src/(appendix|figures|start)/|^src/(index|SUMMARY)\.md$|^src/\d\d-")
INTENTIONAL = pathlib.Path("reference/intentional-broken.txt")


def intentional(root):
    """Citations that must NOT resolve - the audit's own findings, one per line."""
    f = root / INTENTIONAL
    if not f.exists():
        return set()
    out = set()
    for line in f.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out
CHAPTER = re.compile(r"/(\d\d)-[^/]+/index\.md$")


def git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r


def ref_exists(repo, ref):
    return git(repo, "cat-file", "-e", f"{ref}^{{commit}}", check=False).returncode == 0


def path_at(repo, ref, path):
    """True when `path` exists at `ref` (file or directory)."""
    if git(repo, "cat-file", "-e", f"{ref}:{path}", check=False).returncode == 0:
        return True
    # a directory that exists only through its children still resolves for a reader
    r = git(repo, "ls-tree", "-d", "--name-only", ref, f"{path}/", check=False)
    return r.returncode == 0 and bool(r.stdout.strip())


def symbol_at(repo, ref, path, symbol):
    r = git(repo, "grep", "-w", "-e", symbol, ref, "--", path, check=False)
    return r.returncode == 0 and bool(r.stdout.strip())


def pages(root):
    src = root / "src"
    out = [(0, src / "start" / "index.md")]
    out += [(int(m.group(1)), p) for p in sorted(src.glob("[0-9][0-9]-*/index.md")) for m in [CHAPTER.search(str(p))] if m]
    out += [(None, p) for p in sorted(src.glob("appendix/*.md"))]
    out += [(None, src / "index.md")]
    return [(n, p) for n, p in out if p.exists()]


def collect(root):
    """Citations as {path, symbol} -> sorted list of page labels."""
    found = {}
    for num, path in pages(root):
        label = "part 0" if num == 0 else (f"ch {num}" if num else "appendix/home")
        for m in CITATION.finditer(path.read_text()):
            p, sym = m.group(1).rstrip("/"), m.group(2)
            if BOOK_OWN.search(p) or "..." in p or "*" in p:
                continue  # the book's own tree, or a pattern rather than a citation
            found.setdefault((p, sym), set()).add(label)
    return found


def counts(repo, ref):
    def sh(cmd):
        return subprocess.run(cmd, shell=True, cwd=repo, capture_output=True, text=True).stdout

    return {
        "adrs": int(sh("ls docs/adr/[0-9]*.md | wc -l").strip() or 0),
        "step_records": int(sh("ls docs/cuda_optimization_steps/[0-9]*.md | wc -l").strip() or 0),
        "cuda_kernels": int(sh("grep -h '__global__' src/cuda/kernels/*.cu 2>/dev/null | wc -l").strip() or 0),
        "rust_lines": int(sh("find src -name '*.rs' 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}'").strip() or 0),
        "tests": int(sh("grep -rn '#\\[test\\]' src --include=*.rs 2>/dev/null | wc -l").strip() or 0),
    }


def compare(root, repo, ref):
    findings = []
    inv = []
    expected_broken = intentional(root)
    for (p, sym), where in sorted(collect(root).items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        ok_path = path_at(repo, ref, p)
        ok_sym = True if sym is None else (ok_path and symbol_at(repo, ref, p, sym))
        place = ", ".join(sorted(where))
        key = f"{p}::{sym}" if sym else p
        if not ok_path or not ok_sym:
            if key in expected_broken:
                print(f"[i] {place}: `{key}` is listed as intentionally broken (an audit finding)")
            elif not ok_path:
                findings.append(f"{place}: `{p}` does not exist at {ref[:7]}")
            else:
                findings.append(f"{place}: `{p}::{sym}` - path resolves but the symbol does not")
        inv.append({"path": p, "symbol": sym, "pages": sorted(where), "resolved": ok_path and ok_sym})
    return findings, inv


def check_inventory(root, inv_path):
    inv = json.loads(pathlib.Path(inv_path).read_text())
    known = {(c["path"], c["symbol"]): c for c in inv["citations"]}
    expected_broken = intentional(root)
    findings = []
    seen = set()
    for (p, sym), where in sorted(collect(root).items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        key = (p, sym)
        seen.add(key)
        entry = known.get(key)
        place = ", ".join(sorted(where))
        if entry is None:
            findings.append(f"{place}: `{p}{'::' + sym if sym else ''}` is not in the inventory - "
                            f"run scripts/minfer-drift.py --repo <minfer> --ref <baseline> --write-inventory")
        elif not entry["resolved"]:
            key = f"{p}::{sym}" if sym else p
            if key not in expected_broken:
                findings.append(f"{place}: `{key}` did not resolve at the baseline")
    stale = [k for k in known if k not in seen]
    baseline = inv.get("baseline", {})
    print(f"inventory: {inv.get('generated')} at {str(baseline.get('commit'))[:7]} "
          f"({len(known)} citations, {len(stale)} no longer cited)")
    if stale:
        print(f"[W] {len(stale)} inventory entr(y/ies) no longer cited - regenerate when the baseline is bumped")
    return findings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="the book repository (default: .)")
    ap.add_argument("--repo", help="a check-out of minfer")
    ap.add_argument("--ref", help="the commit to compare against (default: the lock file's commit)")
    ap.add_argument("--head", action="store_true", help="compare against the tip of --repo")
    ap.add_argument("--write-inventory", metavar="FILE", help="record the result as the committed inventory")
    ap.add_argument("--check-inventory", metavar="FILE", help="offline check against the committed inventory")
    ap.add_argument("--report", metavar="FILE", help="write a markdown report")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()

    if args.check_inventory:
        findings = check_inventory(root, args.check_inventory)
        for f in findings:
            print(f"[E] {f}")
        if findings:
            print(f"FAIL: {len(findings)} citation(s) outside the inventory")
            return 1
        print("clean")
        return 0

    if not args.repo:
        print("FAIL: --repo is required unless --check-inventory is used", file=sys.stderr)
        return 2
    repo = pathlib.Path(args.repo).expanduser().resolve()

    lock = root / "minfer.lock"
    ref = "HEAD" if args.head else args.ref
    if ref is None:
        if lock.exists():
            m = re.search(r'^commit\s*=\s*"([0-9a-f^]+)"', lock.read_text(), re.M)
            ref = m.group(1) if m else None
        if ref is None:
            print("FAIL: no --ref and no commit in minfer.lock", file=sys.stderr)
            return 2

    if not ref_exists(repo, ref):
        print(f"FAIL: {ref} is not in {repo}", file=sys.stderr)
        return 2

    sha = git(repo, "rev-parse", ref).stdout.strip()

    # Baseline age: the one number that says whether the book is behind on purpose or by neglect.
    # The same figures are printed into the daily issue by .github/workflows/drift.yml.
    bdate = git(repo, "show", "-s", "--format=%cs", sha).stdout.strip()
    bepoch = int(git(repo, "show", "-s", "--format=%ct", sha).stdout.strip() or 0)
    age = int((time.time() - bepoch) // 86400) if bepoch else -1
    tags = git(repo, "tag", "--list", "v*", "--sort=-v:refname", check=False).stdout.split()
    latest = tags[0] if tags else "none"
    ahead = git(repo, "rev-list", "--count", f"{sha}..HEAD", check=False).stdout.strip() or "0"
    days = "day" if age == 1 else "days"
    if args.head:
        # In advisory mode the "baseline" is whatever the tip happens to be; saying "0 days old,
        # 0 commits ahead" reads as if nothing had been checked.
        print(f"master tip {sha[:7]} dated {bdate}; latest tag {latest}")
    else:
        print(f"baseline {sha[:7]} dated {bdate}: {age} {days} old, "
              f"master is {ahead} commit(s) ahead, latest tag {latest}"
              + ("  [due: the monthly fallback has passed]" if age >= 30 else ""))

    findings, inv = compare(root, repo, ref)
    derived = counts(repo, ref)
    mode = "HEAD (advisory)" if args.head else f"baseline {sha[:7]}"

    print(f"minfer {mode}: {len(inv)} citations, {sum(1 for c in inv if not c['resolved'])} unresolved")
    print(f"counts at {sha[:7]}: " + ", ".join(f"{k}={v}" for k, v in derived.items()))
    for f in findings:
        print(f"[E] {f}")

    expected = {}
    if lock.exists():
        for m in re.finditer(r"^(adrs|step_records|cuda_kernels|rust_lines|tests)\s*=\s*(\d+)", lock.read_text(), re.M):
            expected[m.group(1)] = int(m.group(2))
    for k, v in expected.items():
        if k in derived and derived[k] != v:
            print(f"[C] count {k}: minfer.lock says {v}, the tree says {derived[k]}")

    if args.write_inventory:
        payload = {
            "baseline": {"commit": sha, "ref": ref},
            "generated": datetime.date.today().isoformat(),
            "citations": inv,
            "counts": derived,
        }
        out = pathlib.Path(args.write_inventory)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.write_inventory}")

    if args.report:
        lines = [f"# minfer drift report", "", f"Compared against `{sha[:7]}` ({mode}) on {datetime.date.today()}.", ""]
        lines += ["## Counts", "", "| Quantity | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in derived.items()]
        lines += ["", f"## Findings ({len(findings)})", ""]
        lines += [f"- {f}" for f in findings] or ["- none"]
        pathlib.Path(args.report).write_text("\n".join(lines) + "\n")

    return 1 if findings or any(k in derived and derived[k] != v for k, v in expected.items()) else 0


if __name__ == "__main__":
    sys.exit(main())
