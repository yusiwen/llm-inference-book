# Vendored MathJax

The handbook renders mathematics **offline**. This directory holds the renderer, so
`mdbook build` produces a site that needs no CDN, no network and no web-font downloads.

| File | What it is |
|---|---|
| `tex-svg.js` | MathJax 3.2.2, `es5/tex-svg.js` bundle, unmodified |
| `config.js` | This book's configuration (delimiters, skipped tags, sidebar re-typeset) |

`config.js` loads first, then `tex-svg.js` — that order is set by `additional-js` in
`book.toml`, and `mathjax-support` is deliberately **off** there. mdBook's built-in
support injects a MathJax 2.7 script from cdnjs, which silently does nothing when the
CDN is blocked, when a content-security-policy blocks external scripts, or offline;
the page then shows raw LaTeX. Owning the renderer removes that entire class of failure.

## Why the SVG output

MathJax's CHTML output fetches web fonts at run time (hundreds of requests, from the
same CDN). The SVG output embeds glyph outlines directly in the page: one file, no fonts,
identical rendering offline. It costs a few hundred kilobytes of page size.

## Updating

MathJax is Apache-2.0 licensed; `tex-svg.js` carries its own licence header.

```bash
curl -sSL -o /tmp/mathjax.tgz https://registry.npmmirror.com/mathjax/-/mathjax-3.2.2.tgz
tar -xzf /tmp/mathjax.tgz -O package/es5/tex-svg.js > mathjax/tex-svg.js
mdbook build   # then open a page with a formula and confirm it typesets
```

The npm mirror above is used because the registry is reachable from the network this
book is written on; any source of the official package works, but keep the bundle
**unmodified** so the version in the table above stays true.
