#!/usr/bin/env python3
"""Chapter 26's artifact: what a real optimization campaign's record actually contains.

Channel 26 is about a process, and the claim it makes - "most levers fail, and the record is the deliverable" -
is checkable against minfer's own step record set without running a single kernel. So it is checked rather than
asserted: every file in `docs/cuda_optimization_steps/` is parsed for its verdict, whether it recorded a
pre-registered bar, and whether it carries a verification section.

The third of those is the interesting one. The campaign states its own house rule - *a bar named before
measuring* - and a rule that is stated is not a rule that is followed, so the artifact counts it.

    python3 scripts/campaign.py --repo ../minfer --out facts/campaign.json
"""

import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Verdicts as the record set writes them: in the title's trailing parentheses, e.g. "(LANDED)".
VERDICT_RE = re.compile(r"\((LANDED|REVERTED|REVERTED / MEAS-ONLY|MEAS-ONLY|CLOSED)\)\s*$")
# A bar named before the measurement, in any of the phrasings the records actually use. Deliberately strict:
# the loose word "bar" also appears as "the roofline bar" and "bar +/-1%", which is a tolerance, not a decision
# rule, and counting those as pre-registration would inflate the number that the chapter is about.
BAR_RE = re.compile(r"pre-?regist|bar (?:named|set|fixed) before|^\*\*Bar\*\*", re.IGNORECASE | re.MULTILINE)
VERIFY_RE = re.compile(r"^## .*([Vv]erification|[Gg]ate|[Tt]ests?)", re.MULTILINE)
RESULT_RE = re.compile(r"^> \*\*Result\*\*", re.MULTILINE)
DATE_RE = re.compile(r"\*\*Date\*\*:\s*(\d{4}-\d{2}(?:-\d{2})?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="../minfer")
    ap.add_argument("--out", default="facts/campaign.json")
    args = ap.parse_args()

    steps = pathlib.Path(args.repo).resolve() / "docs" / "cuda_optimization_steps"
    if not steps.is_dir():
        sys.exit(f"no step records at {steps}")

    files = [f for f in sorted(steps.glob("*.md")) if f.stem not in ("index", "STYLE")]
    if not files:
        sys.exit(f"no step records found in {steps}")

    verdicts = collections.Counter()
    unmarked = []
    rows, bar_count, verify_count, result_count = [], 0, 0, 0
    dates = []
    for f in files:
        text = f.read_text()
        title = next((ln for ln in text.splitlines() if ln.startswith("# ")), "")
        m = VERDICT_RE.search(title)
        verdict = m.group(1) if m else None
        if verdict:
            verdicts[verdict] += 1
        else:
            unmarked.append(f.stem)
        has_bar = bool(BAR_RE.search(text))
        has_verify = bool(VERIFY_RE.search(text))
        has_result = bool(RESULT_RE.search(text))
        bar_count += has_bar
        verify_count += has_verify
        result_count += has_result
        d = DATE_RE.search(text)
        if d:
            dates.append(d.group(1))
        rows.append({"doc": f.stem, "verdict": verdict, "bar": has_bar, "verification": has_verify,
                     "result_block": has_result, "lines": len(text.splitlines())})

    total = len(files)
    landed = verdicts.get("LANDED", 0)
    reverted = verdicts.get("REVERTED", 0) + verdicts.get("REVERTED / MEAS-ONLY", 0)
    doc = {
        "artifact": "campaign",
        "source": f"{args.repo}/docs/cuda_optimization_steps",
        "source_rev": subprocess.run(["git", "-C", steps, "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
        "steps": total,
        "verdicts": dict(sorted(verdicts.items(), key=lambda kv: -kv[1])),
        "unmarked": len(unmarked),
        "unmarked_examples": unmarked[:4],
        "landed": landed,
        "not_landed": total - landed,
        "reverted": reverted,
        "landed_pct": round(landed / total * 100, 1),
        "reverted_pct": round(reverted / total * 100, 1),
        "with_preregistered_bar": bar_count,
        "without_preregistered_bar": total - bar_count,
        "bar_pct": round(bar_count / total * 100, 1),
        "with_verification_section": verify_count,
        "with_result_block": result_count,
        "median_lines": sorted(r["lines"] for r in rows)[total // 2],
        "date_range": [min(dates), max(dates)] if dates else None,
        "rows": rows,
        "note": (f"The bar count is deliberately strict. 'bar' also appears in these records as a tolerance "
                 f"('bar +/-1%') or as a roofline reference; counting those would turn {bar_count} into something "
                 f"larger and would answer a different question. What is counted is a decision threshold stated "
                 f"before the measurement, and the campaign's own house rule asks for exactly that."),
    }

    path = pathlib.Path(args.out)
    if not path.is_absolute():
        path = ROOT / path
    path.write_text(json.dumps(doc, indent=1) + "\n")

    print(f"wrote {path.relative_to(ROOT)}: {total} step records at {doc['source_rev']}")
    print(f"  verdicts    {doc['verdicts']}  ({doc['unmarked']} unmarked)")
    print(f"  landed      {landed} ({doc['landed_pct']}%), reverted {reverted} ({doc['reverted_pct']}%)")
    print(f"  bar         {bar_count} of {total} record a pre-registered bar ({doc['bar_pct']}%)")
    print(f"  structure   {verify_count} have a verification section, {result_count} open with a Result block, "
          f"median {doc['median_lines']} lines")


if __name__ == "__main__":
    main()
