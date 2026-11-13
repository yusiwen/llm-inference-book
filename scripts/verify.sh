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
allowlist='—–…→↔►▲─│┌┐└┘✅⚠️❌≈γλδπθβτ'
if rg -n --hidden "[^\x00-\x7F\x{A1}-\x{AC}\x{AE}-\x{FF}${allowlist}]" \
     -g '!book/**' -g '!result/**' -g '!.direnv/**' -g '!.git/**' -g '!mathjax/**' -g '!LICENSE' . ; then
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
echo "all checks passed"
