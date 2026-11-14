#!/usr/bin/env python3
"""Lint the TeX in a book's pages before the browser ever sees it.

Why this exists: a formula MathJax cannot parse does not break `mdbook build`. The page builds green,
the site deploys, and the reader gets a red error box where an equation should be - the same silent
failure class as a markdown table whose header cell count disagrees with its delimiter row. This check
is static and offline: it cannot prove the typesetting is right, but it catches every error that comes
from the source text - an unbalanced brace, a \\left without its \\right, an unclosed environment, an
unknown control sequence, and a stray `$` that swallows the rest of a paragraph.

Usage:  python3 scripts/texlint.py [src_dir]        (default: src)

Exit 0 = clean, 1 = problems printed one per line. Wire it into the repo's gate as its own step, and
negative-test it by injecting each defect class before trusting it.
"""

import re
import sys
from pathlib import Path

# Control sequences MathJax 3 supports out of the box (base + ams + the constructs a technical
# explanation actually needs). Anything outside this list is reported, because in practice it is a
# typo - \\fraq, \\theata - and MathJax renders those as red text inside the page.
MACROS = """
frac dfrac tfrac sqrt cdot cdots ldots dots times div pm mp odot otimes oplus sum prod int oint
infty partial nabla le leq ge geq ne neq approx sim simeq propto equiv ll gg
left right big Big bigl bigr Bigl Bigr middle
lvert rvert vert Vert lVert rVert langle rangle lfloor rfloor lceil rceil lgroup
leftarrow rightarrow to mapsto longrightarrow Longrightarrow Rightarrow uparrow downarrow
leqslant geqslant nmid perp parallel neg land lor implies iff forall exists nexists
subset supset subseteq supseteq cup cap setminus emptyset varnothing in ni notin
frac text textrm textit textbf mathrm mathbf mathit mathsf mathtt operatorname mathbb mathcal mathscr
textstyle displaystyle scriptstyle mathstrut phantom boxed tag nonumber notag label
quad qquad hspace space enspace thinspace
max min exp log ln sup lim inf arg argmin argmax det dim ker mod bmod pmod
sigma theta alpha beta gamma delta epsilon varepsilon zeta eta vartheta iota kappa lambda mu nu xi
pi varpi rho varrho tau upsilon phi varphi chi psi omega
Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
sin cos tan arctan sinh cosh tanh overline underline overbrace underbrace
hat bar vec tilde widehat widetilde dot ddot\
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
        out.append("" if fence is not None else
                   re.sub(r"`[^`]*`", lambda m: " " * len(m.group(0)), line))
    return "\n".join(out)


def check_block(tex, where, problems):
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

    # Count with a word boundary. `\\rightarrow` contains `\\right`, so a substring count reports
    # a perfectly good formula as unbalanced - a lint that lies is worse than no lint, because it
    # teaches you to ignore its output.
    n_left = len(re.findall(r"\\left(?![A-Za-z])", tex))
    n_right = len(re.findall(r"\\right(?![A-Za-z])", tex))
    if n_left != n_right:
        problems.append(f"{where}: {n_left} \\left against {n_right} \\right")

    stack = []
    for kind, name in re.findall(r"\\(begin|end)\{([^}]*)\}", tex):
        if name not in ENVIRONMENTS:
            problems.append(f"{where}: unknown environment {name!r}")
            continue
        if kind == "begin":
            stack.append(name)
        elif not stack:
            problems.append(f"{where}: \\end{{{name}}} with no \\begin")
        elif stack[-1] != name:
            problems.append(f"{where}: \\end{{{name}}} closes \\begin{{{stack[-1]}}}")
            stack.pop()
        else:
            stack.pop()
    for name in stack:
        problems.append(f"{where}: \\begin{{{name}}} never closed")

    # `\\begin` and `\\end` are validated by the environment stack above, so they are not
    # unknown macros - scanning them here reported every aligned block as two defects (caught by
    # running this copy against a page with a \\begin{aligned} in it).
    for name in re.findall(r"\\([A-Za-z]+)", tex):
        if name not in MACROS and name not in ("begin", "end"):
            problems.append(f"{where}: unknown control sequence \\{name}")


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "src")
    problems, total = [], 0
    pages = sorted(src.rglob("*.md"))
    for page in pages:
        text = strip_code(page.read_text())

        bodies = re.findall(r"\$\$(.*?)\$\$", text, re.S)
        total += len(bodies)
        for i, body in enumerate(bodies, 1):
            check_block(body, f"{page}: display maths #{i}", problems)
        if len(re.findall(r"\$\$", text)) % 2:
            problems.append(f"{page}: an odd number of $$ markers - one formula is unclosed")

        stripped = re.sub(r"\$\$(.*?)\$\$", "", text, flags=re.S)
        for m in re.finditer(r"\$([^$\n]+)\$", stripped):
            body = m.group(1)
            if body[:1] == " " or body[-1:] == " ":
                problems.append(f"{page}: inline maths {body[:30]!r} has a space inside its delimiters")
            check_block(body, f"{page}: inline maths {body[:30]!r}", problems)
        lone = len(re.findall(r"\$", re.sub(r"\$[^$\n]+\$", "", stripped)))
        if lone:
            problems.append(f"{page}: {lone} unmatched $ outside code - prose has a stray dollar sign")

    if problems:
        for p in problems:
            print(f"[E] {p}")
        print(f"FAIL: {len(problems)} TeX problem(s) in {total} formula(s)")
        return 1
    print(f"clean ({total} formula(s) across {len(pages)} page(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
