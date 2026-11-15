# Pinned source, vendored for verification

These are minfer sources at the commit the book is pinned to (`reference/baseline-citations.json`), copied
here so that `scripts/codecheck.py` can verify every quoted line without a minfer checkout - and so that a
history rewrite upstream cannot change what the book was checked against.

- Upstream: <https://github.com/yusiwen/minfer>, MIT licensed. These copies stay MIT; the book's prose is
  CC BY-NC-SA 4.0. Only the files the book quotes are here.
- Refresh after a baseline bump: `python3 scripts/codecheck.py --vendor` (reads the new commit from
  `baseline-citations.json` and re-copies from `$HOME/git/mine/minfer`, or `--repo <path>`).
- Verify: `python3 scripts/codecheck.py --check` - every block in `src/` must be an in-order subsequence of
  its file here, after trimming indentation. A changed token fails; a re-indent does not.

What verification cannot catch, and the review must: whether the excerpt is the right passage, and whether an
elision (`// ...`) hides the one line that would contradict the prose.
