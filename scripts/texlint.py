#!/usr/bin/env python3
"""Lint the TeX in the book's pages before the browser ever sees it.

Why this exists: a formula that MathJax cannot parse does not break `mdbook build`. The page builds
green, the site deploys, and the reader gets a red error box where an equation should be - the same
silent-failure class as a markdown table whose header cell count disagrees with its delimiter row.
This gate is static and offline; it cannot prove the typesetting is right, but it catches every error
that comes from the source text: an unbalanced brace, a \\left without its \\right, an unclosed
environment, an unknown control sequence, and a stray `$` that swallows the rest of a paragraph.

Run from the repository root:  python3 scripts/texlint.py

What it does NOT check: whether the mathematics is true, whether the notation is readable, or whether
the glyphs overlap. Those need eyes on a rendered page - see AGENTS.md, "a page with math gets looked at".
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Control sequences MathJax 3 supports out of the box (base + ams + the handful of constructs a
# mathematical explanation of an inference engine needs). Anything outside this list is reported,
# because in practice it is a typo - \fraq, \theata - and MathJax renders those as red text.
MACROS = """
frac dfrac tfrac sqrt cdot cdots ldots dots times div pm mp odot otimes oplus sum prod int oint
infty partial nabla le leq ge geq ne neq approx sim simeq propto equiv ll gg
left right big Big bigl bigr Bigl Bigr middle
text textrm textit textbf mathrm mathbf mathit mathsf mathtt operatorname mathbb mathcal mathscr
max min exp log ln sup log_ lim sup inf arg argmin argmax det dim ker
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi
pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega
Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
sin cos tan arctan sinh cosh tanh
quad qquad hspace space enspace thinspace
leftarrow rightarrow to mapsto longrightarrow Longrightarrow Rightarrow uparrow downarrow
lvert rvert vert Vert lVert rVert langle rangle lfloor rfloor lceil rceil lgroup
{ } % $ & # _ | ,
begin end
hline cline
overline underline overbrace underbrace hat bar vec tilde widehat widetilde dot ddot
displaystyle textstyle scriptstyle mathstrut phantom boxed
tag tag* label nonumber notag
leqslant geqslant nmid perp parallel
nequiv notin in ni subset supset subseteq supseteq cup cap setminus emptyset varnothing
forall exists nexists neg land lor implies iff
""".split()

ENVIRONMENTS = {"aligned", "align", "align*", "gathered", "gather", "gather*", "array",
                "matrix", "pmatrix", "bmatrix", "vmatrix", "cases", "split", "smallmatrix"}


def strip_code(text):
    """Blank out fenced blocks and inline code spans, keeping line numbering intact."""
    out = []
    fence = None
    for line in text.split("\n"):
        m = re.match(r"^\s*(```+|~~~+)", line)
        if m:
            if fence is None:
                fence = m.group(1)[0]
                out.append("")
                continue
            if line.strip().startswith(fence * 3):
                fence = None
                out.append("")
                continue
        if fence is not None:
            out.append("")
        else:
            out.append(re.sub(r"`[^`]*`", lambda m: " " * len(m.group(0)), line))
    return "\n".join(out)


def check_block(tex, where, problems):
    """Structural checks on one formula body."""
    depth = 0
    for ch in tex:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                problems.append(f"{where}: a closing brace with no opening brace")
                return
    if depth:
        problems.append(f"{where}: {depth} unclosed brace(s)")

    # Count with a boundary: `\rightarrow` contains `\right`, and a substring count reported the
    # first formula in the book as unbalanced when it was not.
    n_left = len(re.findall(r"\\left(?![A-Za-z])", tex))
    n_right = len(re.findall(r"\\right(?![A-Za-z])", tex))
    if n_left != n_right:
        problems.append(f"{where}: {n_left} \\left against {n_right} \\right")

    stack = []
    for env in re.findall(r"\\(begin|end)\{([^}]*)\}", tex):
        kind, name = env
        if name not in ENVIRONMENTS:
            problems.append(f"{where}: unknown environment {name!r}")
            continue
        if kind == "begin":
            stack.append(name)
        else:
            if not stack:
                problems.append(f"{where}: \\end{{{name}}} with no \\begin")
            elif stack[-1] != name:
                problems.append(f"{where}: \\end{{{name}}} closes \\begin{{{stack[-1]}}}")
                stack.pop()
            else:
                stack.pop()
    for name in stack:
        problems.append(f"{where}: \\begin{{{name}}} never closed")

    for name in re.findall(r"\\([A-Za-z]+)", tex):
        if name not in MACROS:
            problems.append(f"{where}: unknown control sequence \\{name}")


def main():
    problems = []
    pages = sorted(SRC.rglob("*.md"))
    total = 0
    for page in pages:
        rel = page.relative_to(ROOT)
        text = strip_code(page.read_text())

        # display math: $$ ... $$ possibly spanning lines
        bodies = re.findall(r"\$\$(.*?)\$\$", text, re.S)
        total += len(bodies)
        for i, body in enumerate(bodies, 1):
            check_block(body, f"{rel}: display math #{i}", problems)
        if len(re.findall(r"\$\$", text)) % 2:
            problems.append(f"{rel}: an odd number of $$ markers - one formula is unclosed")

        # inline math: single $ pairs, with no space just inside the delimiters (MathJax needs that,
        # and a `$5 and $6` in prose would otherwise become one formula spanning the sentence)
        stripped = re.sub(r"\$\$(.*?)\$\$", "", text, flags=re.S)
        for m in re.finditer(r"\$([^$\n]+)\$", stripped):
            body = m.group(1)
            if body[:1] == " " or body[-1:] == " ":
                problems.append(f"{rel}: inline math {body[:30]!r} has a space inside its delimiters")
            check_block(body, f"{rel}: inline math {body[:30]!r}", problems)
        lone = len(re.findall(r"\$", re.sub(r"\$[^$\n]+\$", "", stripped)))
        if lone:
            problems.append(f"{rel}: {lone} unmatched $ outside code - prose has a stray dollar sign")

    if problems:
        for p in problems:
            print(f"[E] {p}")
        print(f"FAIL: {len(problems)} TeX problem(s) in {total} formula(s)")
        return 1
    print(f"clean ({total} formula(s) across {len(pages)} page(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
