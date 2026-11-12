/* MathJax configuration for this handbook.
 *
 * Loaded before mathjax/tex-svg.js (see `additional-js` in book.toml).
 * The SVG output is deliberate: it embeds glyph outlines in the page, so the site
 * needs no web-font downloads and renders correctly with no network at all.
 */
window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    tags: "none"
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    ignoreHtmlClass: "tex2jax_ignore|searchbar"
  },
  svg: {
    fontCache: "global"
  }
};

/* mdBook builds the sidebar client-side from toc-<hash>.js, which can finish after
 * MathJax has already typeset the page. Re-typeset the sidebar once the page has
 * loaded, so any mathematics in a heading title is rendered there too.
 */
window.addEventListener("load", function () {
  var mj = window.MathJax;
  if (!mj || !mj.startup || !mj.startup.promise) return;
  mj.startup.promise
    .then(function () {
      var sidebar = document.querySelector("#mdbook-sidebar") || document.querySelector(".sidebar");
      return sidebar ? mj.typesetPromise([sidebar]) : null;
    })
    .catch(function () {
      /* Rendering falls back to the raw LaTeX; never break the page for this. */
    });
});
