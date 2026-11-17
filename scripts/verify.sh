#!/usr/bin/env bash
# The repository's verification gate. Run before every "done":
#     nix develop --command bash scripts/verify.sh
set -euo pipefail
cd "$(dirname "$0")/.."

log=$(mktemp -t llm-inference-book-build.XXXXXX.log)

echo "== 1. build (any WARN is a real defect) =="
mdbook build 2>&1 | tee "$log"
if grep -q ' WARN ' "$log"; then
  echo "FAIL: mdbook printed a warning (unbalanced figure, missing page, bad config key)" >&2
  exit 1
fi

echo
echo "== 2. English only: nothing outside ASCII and Latin-1, apart from the allowlist =="
# LICENSE is excluded: it is the Creative Commons legal code, verbatim. Its curly quotes are the
# licence's own typography, and editing them to satisfy a prose rule is not ours to do.
# reference/pinned/ is excluded for the same reason at one remove: those are minfer sources copied verbatim
# at the baseline, and the whole point of the copy is that it is byte-identical to upstream. Its comments
# carry characters this book's prose may not (`\u{2265}`, `\u{2466}`), and "fixing" them would quietly break
# `scripts/codecheck.py`, which compares those bytes to the quotes in `src/`.
allowlist='—–…→↔►▲─│┌┐└┘✅⚠️❌≈γλδπθβτ'
if rg -n --hidden "[^\x00-\x7F\x{A1}-\x{AC}\x{AE}-\x{FF}${allowlist}]" \
     -g '!book/**' -g '!result/**' -g '!.direnv/**' -g '!.git/**' -g '!mathjax/**' -g '!LICENSE' \
     -g '!reference/pinned/**' . ; then
  echo "FAIL: character outside ASCII and Latin-1 (and not on the allowlist)" >&2
  exit 1
fi
echo "clean"

echo
echo "== 3. the site must not fetch anything remote =="
count() { { rg -o "$1" book/ || true; } | wc -l | tr -d ' '; }
remote_js=$(count '<script[^>]*src="https?://[^"]*"')
remote_css=$(count '<link[^>]*href="https?://[^"]*"')
echo "remote scripts: $remote_js, remote stylesheets: $remote_css"
if [ "$remote_js" -ne 0 ] || [ "$remote_css" -ne 0 ]; then
  echo "FAIL: the built site references a remote asset - it must work offline" >&2
  exit 1
fi

echo
echo "== 4. figures: structure, theme classes, geometry, captions (figcheck --strict) =="
# scripts/figcheck.py is copied from the mdbook-figures skill; the copy in the repo is the one CI
# runs, so it must stay self-contained (stdlib Python only, no toolchain).
python3 scripts/figcheck.py --root . --strict

echo
echo "== 5. display maths: every block wrapped =="
blocks=$(( $( { rg -N -o '^\$\$$' -g '*.md' src/ || true; } | wc -l | tr -d ' ') / 2 ))
wrapped=$( { rg -N -o '^<div class="math-display">$' -g '*.md' src/ || true; } | wc -l | tr -d ' ')
if [ "$blocks" -ne "$wrapped" ]; then
  echo "FAIL: $blocks display block(s) but $wrapped wrapped in <div class=\"math-display\">" >&2
  exit 1
fi
echo "clean ($blocks display block(s))"

echo
echo "== 6. no page is a bare stub the build silently created =="
small=$(find book -name 'index.html' -size -1200c | wc -l | tr -d ' ')
echo "pages under 1.2 KB: $small (mdBook writes a stub for a missing SUMMARY entry)"
if [ "$small" -ne 0 ]; then
  find book -name 'index.html' -size -1200c >&2
  echo "FAIL: a page in SUMMARY.md has no real source file" >&2
  exit 1
fi

echo
echo "== 7. the measurements page still carries its commands =="
if ! rg -q 'wc -l|wc -c' src/appendix/measurements.md; then
  echo "FAIL: src/appendix/measurements.md lost the commands behind its numbers" >&2
  exit 1
fi
echo "clean"

echo
echo "== 8. the book's own counts match the tree =="
# The measurements page is where the book promises that every number has a command behind it. That
# promise kept breaking for the book itself: without this check the page went stale twice in one
# afternoon while the gate stayed green.
python3 scripts/selfcounts.py

echo
echo "== 9. every minfer citation is in the committed inventory =="
# Offline on purpose: the book's gate never reaches the network. The inventory is regenerated only
# when the baseline moves (see src/appendix/baseline.md), so a new citation nobody checked fails here.
python3 scripts/minfer-drift.py --check-inventory reference/baseline-citations.json

echo
echo "== 10. the reference index matches the chapters =="
python3 scripts/references.py --check

echo
echo "== 11. the book's own skeleton and links =="
python3 scripts/probe.py .

echo
echo "== 12. the numbers in the pages still match the artifact =="
python3 scripts/facts.py --check

echo
echo "== 13. the TeX in the pages parses =="
# A formula MathJax cannot parse does not fail `mdbook build`: the page deploys green and the reader
# gets a red error box. Same silent-failure class as a markdown table with the wrong cell count, and
# the same answer - catch it before the browser is asked to.
# scripts/texlint.py is a verbatim copy of the one in the `mdbook-study-handbook` skill. Fix the skill's
# copy first and re-copy it here, or the two drift apart - which is the exact failure this book is about.
python3 scripts/texlint.py

echo
echo "== 14. monospace holds code and output, not pictures (ratchet) =="
# The allowance is two reviewed keepers: a tool's own parse dump (Chapter 2) and a list of pseudocode steps
# (Chapter 8). Everything else in the book is a figure, a maths block, a table or code. The number may only
# go down; a new candidate has to be drawn or turned into a table, not counted here.
python3 scripts/sketch-probe.py --allow 2

echo
echo "== 15. the provenance index matches the artifacts =="
# Every measured table in a chapter is rendered from facts/*.json, and one line under it names the script,
# machine and statistic. This step keeps the appendix page honest: it lists all of them, in full, and it is
# generated from the artifacts rather than typed.
python3 scripts/provenance.py --check

# ...and a provenance line that names no script is worse than none: it looks like evidence. `render_provenance`
# prints `(unknown)` when the artifact's instrument is missing from `INSTRUMENTS`, which the index check above
# cannot see (the appendix would list the artifact and the page would still be anonymous).
if bad=$(grep -rln '\*`(unknown)`' src/ 2>/dev/null) && [ -n "$bad" ]; then
  for f in $bad; do echo "[E] $f: a provenance line names no script - add the artifact to INSTRUMENTS in scripts/facts.py" >&2; done
  echo "FAIL: a provenance line prints (unknown)" >&2
  exit 1
else
  echo "clean (every provenance line names its script)"
fi

echo
echo "== 16. every quoted line of minfer code matches the pinned source =="
# A quoted line is a claim like a number is. The sources are vendored under reference/pinned/<commit>/ so
# this runs without a minfer checkout; a baseline bump means re-reading and re-quoting, and the vendor step
# says which files to re-copy. What it cannot check: whether the excerpt is the right passage, or whether an
# elision hides the line that contradicts the prose. That is review, and AGENTS.md says so.
python3 scripts/codecheck.py --check

echo
echo "== 17. no lesson explains the engine without quoting it =="
# Steps 16 checks the quotes that exist; this one checks the ones that do not. A lesson with neither an
# excerpt nor an explicit `<!-- no-code: why -->` is a lesson the reader cannot check against minfer - the
# exact defect this book spent two passes closing. The count is ratcheted per page against
# reference/coverage-baseline.json, so a page can only lose bare lessons and a new page starts at zero.
python3 scripts/codecheck.py --coverage

echo
echo "== 18. every number in a figure comes from that chapter's artifact =="
# A facts block is regenerated; a figure's caption is hand-drawn text and nothing regenerates it. Chapter 20's
# two diagrams were still saying 4.31 ms after a re-measurement moved the chapter to 3.98 - including in the
# aria-label a screen reader reads. Numbers deliberately not from a measurement are listed in
# reference/figure-numbers.txt, which is the only escape hatch.
python3 scripts/fignumbers.py

echo
echo "== 19. instrument dependencies (advisory; the gate itself needs none of them) =="
# Steps 1-17 need nothing but the standard library. The instruments that write `facts/*.json` need regex,
# jinja2, numpy and tokenizers, and they are not part of this gate - which is how three measurement runs died
# on `ModuleNotFoundError: No module named 'regex'` while this script reported "all checks passed". This step
# never fails: a machine that cannot re-measure the book can still build and check it.
if python3 -c 'import regex, jinja2, numpy, tokenizers' 2>/dev/null; then
  echo "instruments: regex, jinja2, numpy and tokenizers are importable"
else
  echo "note: an instrument dependency is missing in this Python (see scripts/deps.py):"
  python3 scripts/deps.py || true
  echo "     building and checking the book does not need them; re-measuring facts/ does."
fi

echo
echo "all checks passed"
