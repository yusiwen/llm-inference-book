#!/usr/bin/env python3
"""Render the provenance index: every measured number in the book, and the program behind it.

The chapters print one line under their first table ("scripts/dotbench.c; on Apple M4 Pro; 7 runs;
median"), and this page is the full version of those lines: the complete command, the sample size, the
machine, the statistic, and the file the numbers were written to. A reader who wants to reproduce a number
starts here; a reader who wants to check that the book is not making it up starts here too.

    python3 scripts/provenance.py --render   # rewrite the block between the markers
    python3 scripts/provenance.py --check    # the page matches facts/*.json, or fail

Everything is read out of `facts/*.json` and `facts.INSTRUMENTS`. Nothing on this page is typed by hand:
the artifact's own `invocation`, `machine`, `runs` and `artifact` fields are the source, which is the same
field the chapter's one-line provenance reads.
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FACTS = ROOT / "facts"
PAGE = ROOT / "src" / "appendix" / "provenance.md"

sys.path.insert(0, str(ROOT / "scripts"))
from facts import INSTRUMENTS  # noqa: E402  (declared once, next to the renderer that uses it)
START = "<!-- provenance-index -->"
END = "<!-- /provenance-index -->"


HOME_RE = re.compile(r"(?:/Users/|/home/)[^/\s:;]+/")


def home_free(text):
    """Strip a home directory by shape, not by this machine's home: CI runs as /home/runner and the
    rendered page has to come out identical there, or the staleness check fails on a clean tree."""
    return HOME_RE.sub("~/", text)


def field(d, *names):
    for n in names:
        v = d.get(n)
        if isinstance(v, list):
            v = "; ".join(str(x) for x in v)
        if v:
            return v
    return ""


def machines_of(d):
    m = d.get("machine") or ""
    if m:
        return home_free(str(m))
    ms = d.get("machines")
    if isinstance(ms, dict):
        return " and ".join(ms.keys())
    if isinstance(ms, list):
        names = [str(x.get("machine")) for x in ms if isinstance(x, dict) and x.get("machine")]
        return " and ".join(dict.fromkeys(names))
    return ""


def artifact_of(name, d):
    art = d.get("artifact")
    path = str(art.get("path") or "") if isinstance(art, dict) else (art if isinstance(art, str) else "")
    note = ""
    if path.endswith(".gguf"):
        note = f" (read from `{pathlib.Path(path).name}`; its sha256 is in the facts file)"
        return f"`facts/{name}.json`{note}"
    return f"`facts/{name}.json`"


def render():
    rows = []
    sections = []
    for f in sorted(FACTS.glob("*.json")):
        name = f.stem
        d = json.loads(f.read_text())
        script, what, statistic = INSTRUMENTS.get(name, ("(not declared)", "", ""))
        machine = machines_of(d) or "not machine-dependent"
        runs = field(d, "runs", "reps")
        if not runs:
            # a per-machine sweep records its run count inside each machine's block
            runs = "per machine" if machines_of(d) else "-"
        command = home_free(field(d, "invocation", "command")) or "(no command recorded)"
        rows.append(f"| `{name}` | `{script}` | {what} | {machine} | {runs} | {statistic} |")
        sections.append(
            f"### {name}\n\n"
            f"`{script}` - {what}.\n\n"
            f"- **Machine**: {machine}\n"
            f"- **Runs**: {runs}; {statistic}\n"
            f"- **Raw numbers**: {artifact_of(name, d)}\n\n"
            f"```bash\n{command}\n```"
        )
    out = [
        "| Artifact | Instrument | What it measures | Machine | Runs | Statistic |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        "The command behind each row, in full:",
        "",
        *[s + "\n" for s in sections],
    ]
    return "\n".join(out).rstrip() + "\n"


def rewrite(text, body):
    if START not in text or END not in text:
        raise SystemExit(f"{PAGE.relative_to(ROOT)}: missing {START} / {END}")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{body}{END}{tail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    body = render()
    text = PAGE.read_text()
    new = rewrite(text, body)

    if args.check:
        if new != text:
            print(f"{PAGE.relative_to(ROOT)}: the provenance index is stale - run "
                  f"`python3 scripts/provenance.py --render`")
            return 1
        print(f"clean ({body.count(chr(10) + '### ')} instrument(s) indexed)")
        return 0
    if args.render:
        if new != text:
            PAGE.write_text(new)
            print(f"rendered {PAGE.relative_to(ROOT)}: {body.count(chr(10) + '### ')} instrument(s)")
        else:
            print(f"{PAGE.relative_to(ROOT)} is current")
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
