# Learning LLM Inference the Hard Way

### Building an Inference Engine from Scratch in Rust

The source of an mdBook that walks a reader from a quantized model file to a serving engine: the
file format, the arithmetic, the declarative graph, the devices, and the measurements that prove it.
The reference implementation read beside it is [`minfer`](https://github.com/yusiwen/minfer), at
commit `88ffa5d`.

**Status: skeleton.** All 29 chapters are stubs: the outline, lesson checklists, empty ledgers and
acceptance criteria exist; the prose is being written one chapter at a time.

## Build

```bash
nix develop                       # mdbook 0.5 + ripgrep + python3, pinned by flake.lock
mdbook serve --open               # http://localhost:3000
mdbook build                      # writes the site to book/
bash scripts/verify.sh            # the gate: build-with-no-warnings, character scan,
                                  # offline assets, maths wrappers, page floor, skeleton, links
nix build                         # reproducible site into result/
```

Without Nix: `brew install mdbook` (0.5.x) and run the same commands.

## Layout

| Path | What it is |
|---|---|
| `src/index.md` | the landing page: what the book is, the four questions, the map |
| `src/start/` | Part 0: the contract, machines, the workspace, the routing map |
| `src/NN-<slug>/index.md` | one chapter per directory, English slug, fixed skeleton |
| `src/appendix/` | conventions, the decision map (82 ADRs), measurements, the CUDA-tutorial audit, glossary, limits, where the code lives |
| `scripts/verify.sh` | the gate CI runs; a green local run is a green deploy |
| `scripts/probe.py` | the checks the build cannot see: skeleton sections, links, counts |
| `mathjax/`, `css/` | vendored renderer and figure styles; the site fetches nothing remotely |

## Rules that are enforced, not just stated

- **English only.** `scripts/verify.sh` rejects every character outside ASCII and Latin-1 apart from
  a short allowlist of typographic marks.
- **One skeleton per chapter.** `scripts/probe.py` fails a chapter whose fixed sections are missing or
  out of order, and a landing page whose chapter or lesson count disagrees with the tree.
- **Counts are derived.** The counts on the landing page, the appendix and this file come from the
  tree; `src/appendix/measurements.md` records the command behind each.
- **Code pointers are path + symbol + commit.** Never a bare line number.

## Publishing

`.github/workflows/pages.yml` builds the site with the pinned toolchain, runs `scripts/verify.sh`,
and deploys to GitHub Pages on every push to `main`. One-time setup: repository Settings -> Pages ->
Source: **GitHub Actions**.

The site is served under the user site's custom domain, so it lands at
`https://yusiwen.cn/llm-inference-book/`; `site-url` in `book.toml` must equal the repository name,
and a `CNAME` file must never be added.

## License

[CC BY-NC-SA 4.0](LICENSE) - the same licence as the author's other handbook: reuse with
attribution, no commercial reuse, share-alike. The prose is the author's; the reference
implementation `minfer` keeps its own MIT licence.
