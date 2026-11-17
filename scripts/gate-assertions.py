#!/usr/bin/env python3
"""Gate step 20: a verdict in an artifact is a claim that an assertion ran.

Every artifact that reports `passes` is reporting the outcome of a comparison against a reference. The failure
this step exists to catch is subtler than a wrong verdict: it is a *vacuous* one. `passes: true` with
`tolerance: null` means nothing was ever compared, and it renders in the book exactly like a real pass - which is
how a table of two 2,048-cubed GEMMs came to print "yes" in a Passes column for two kernels that were only timed.

The contract, per entry, is three fields:

    checked   true only if a reference comparison actually ran
    tolerance the bound it was compared against (a number), or null if nothing was compared
    exact     required when tolerance is 0, because "zero tolerance" is a legitimate exact comparison and must
              not be confused with "no comparison"

    passes: true,  checked: true, tolerance: 2e-03                 -> a bounded comparison ran
    passes: true,  checked: true, tolerance: 0, exact: true        -> an exact comparison ran
    passes: null,  checked: false, tolerance: null                 -> timed only, and says so
    passes: true,  tolerance: null                                 -> FAIL: a pass with no comparison
    passes: true,  tolerance: 0                                    -> FAIL until `exact: true` says it was intended

    python3 scripts/gate-assertions.py            # the gate's own verdict
    python3 scripts/gate-assertions.py --verbose  # list every entry it accepted
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FACTS = ROOT / "facts"


def entries(node, path=""):
    """Every dict in the tree that makes a verdict, with a dotted path for the message."""
    if isinstance(node, dict):
        if "passes" in node or "tolerance" in node:
            yield path or "<root>", node
        for k, v in node.items():
            child = f"{path}.{k}" if path else str(k)
            yield from entries(v, child)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            name = v.get("name") if isinstance(v, dict) and "name" in v else str(i)
            child = f"{path}[{name}]" if path else f"[{name}]"
            yield from entries(v, child)


def check(path, d):
    """Return a list of findings for one entry."""
    out = []
    has_passes = "passes" in d
    passes = d.get("passes")
    checked = d.get("checked")
    tol = d.get("tolerance", "<absent>")

    if not has_passes:
        out.append(f"{path}: carries `tolerance: {tol!r}` but no `passes` - a bound with no verdict is a "
                   f"comparison whose outcome nobody recorded")
        return out

    if passes is True or passes is False:
        if checked is not True:
            out.append(f"{path}: `passes: {str(passes).lower()}` with no `checked: true` - a verdict without a "
                       f"record that the comparison ran")
        if tol is None:
            out.append(f"{path}: `passes: {str(passes).lower()}` with `tolerance: null` - this is the shape a "
                       f"vacuous assertion takes: nothing was compared, and the table will still print a verdict")
        elif tol == 0 and d.get("exact") is not True:
            out.append(f"{path}: `tolerance: 0` without `exact: true` - zero tolerance is either an exact "
                       f"comparison (say so) or no comparison at all (do not claim a pass)")
    elif passes is None:
        if checked is True:
            out.append(f"{path}: `checked: true` with `passes: null` - a comparison that ran and produced no "
                       f"verdict; record the outcome or drop the field")
        if tol is not None:
            out.append(f"{path}: `tolerance: {tol!r}` with `passes: null` - a bound with no verdict")
    else:
        out.append(f"{path}: `passes` is {passes!r}; expected true, false or null")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="list every accepted entry")
    args = ap.parse_args()

    files = sorted(FACTS.glob("*.json"))
    if not files:
        sys.exit("no artifacts under facts/ - the gate cannot check what does not exist")

    findings, accepted, checked_count = [], [], 0
    for f in files:
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            findings.append(f"{f.name}: not valid JSON ({e})")
            continue
        for path, d in entries(doc):
            name = f"{f.name}:{path}"
            these = check(name, d)
            findings.extend(these)
            if not these:
                accepted.append(name)
                if d.get("checked") is True:
                    checked_count += 1

    if args.verbose:
        for a in accepted:
            print(f"  ok  {a}")

    if findings:
        print(f"gate 20: {len(findings)} finding(s) across {len(files)} artifact(s)")
        for x in findings:
            print(f"[E] {x}")
        print("\nA verdict in an artifact is a claim that an assertion ran. If a rung is timed and not checked,")
        print("say so: `checked: false, tolerance: null, passes: null`. The book renders all three states, and")
        print("the difference between 'passed' and 'never compared' is the difference this book is about.")
        sys.exit(1)

    print(f"gate 20: {checked_count} checked verdict(s), {len(accepted) - checked_count} entry(ies) that say "
          f"nothing was compared; every entry states which it is")


if __name__ == "__main__":
    main()
