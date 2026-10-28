// Read the Docs-style version flyout (bottom right), after the switcher of docs.fenicsproject.org.
// Served from the root of the site and injected into every built page by
// .github/workflows/deploy-pages.yml, so it also covers old release builds.
// Reads the page's <version>/<rest> from its URL, lists the versions in switcher.json and, on
// selection, opens the same page in the chosen version, falling back to the nearest existing
// parent folder. Older releases get a banner that points to the latest one.
(function () {
  "use strict";

  // Not inside frames
  if (window.self !== window.top) return;

  // Site root, from this script's own URL, so previews served below another path work too
  const root = new URL(".", document.currentScript.src).pathname;
  if (!location.pathname.startsWith(root)) return;
  const parts = location.pathname.slice(root.length).split("/");
  if (parts.length < 2) return;
  const current = decodeURIComponent(parts[0]);
  const rest = parts.slice(1);

  // Candidate URLs in the target version: the page itself, then each parent folder's
  // index.html up to the version root. The first `same` of them are the same page.
  function candidates(version) {
    const base = root + version + "/";
    const out = [];
    const page = rest.join("/");
    if (page !== "" && !page.endsWith("/")) out.push(base + page);
    const same = out.length;
    const dirs = rest.slice(0, -1);
    for (let n = dirs.length; n >= 0; n--) {
      const dir = dirs.slice(0, n).join("/");
      out.push(base + (dir ? dir + "/" : "") + "index.html");
    }
    return { urls: [...new Set(out)], same: same };
  }

  async function exists(url) {
    try {
      const r = await fetch(url, { method: "HEAD" });
      return r.ok;
    } catch (e) {
      return false;
    }
  }

  async function go(version) {
    const { urls, same } = candidates(version);
    // Check all candidates at once, then take the first that exists
    const found = await Promise.all(urls.map(exists));
    const i = found.indexOf(true);
    if (i === -1) {
      location.href = root + version + "/";
      return;
    }
    // The query and fragment only make sense on the same page
    const keep = i < same ? location.search + location.hash : "";
    location.href = urls[i].replace(/(^|\/)index\.html$/, "$1") + keep;
  }

  const css = `
.rbe-vs { position: fixed; right: 1rem; bottom: 1rem; z-index: 10000; font-size: 0.85rem;
  font-family: var(--pst-font-family-base, system-ui, sans-serif);
  border-radius: 0.35rem; border: 1px solid var(--pst-color-border, #ccc);
  background: var(--pst-color-surface, #f5f5f5); color: var(--pst-color-text-base, #222);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2); max-width: 18rem; }
.rbe-vs summary { cursor: pointer; padding: 0.35rem 0.75rem; list-style: none;
  user-select: none; white-space: nowrap; }
.rbe-vs summary::-webkit-details-marker { display: none; }
.rbe-vs summary::before { content: "v: "; color: var(--pst-color-primary, #0a7d91); }
.rbe-vs summary::after { content: " \\25BE"; }
.rbe-vs[open] summary::after { content: " \\25B4"; }
.rbe-vs summary .rbe-vs-old { color: var(--pst-color-warning, #c77700); }
/* The list opens above the button, which stays where it is */
.rbe-vs ul { position: absolute; right: -1px; bottom: calc(100% + 0.4rem); margin: 0;
  padding: 0.3rem 0; list-style: none; min-width: 10rem; max-height: 50vh; overflow-y: auto;
  border-radius: 0.35rem; border: 1px solid var(--pst-color-border, #ccc);
  background: var(--pst-color-surface, #f5f5f5); box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2); }
.rbe-vs li a { display: block; padding: 0.25rem 0.9rem; text-decoration: none;
  color: var(--pst-color-text-base, #222); }
.rbe-vs li a:hover { background: var(--pst-color-background, #fff); }
.rbe-vs li.rbe-vs-current a { font-weight: bold; color: var(--pst-color-primary, #0a7d91); }
.rbe-vs-banner { position: relative; z-index: 9999; padding: 0.5rem 2.5rem; text-align: center;
  background: #fff3cd; color: #533f03; border-bottom: 1px solid #e6d28a;
  font: 14px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.rbe-vs-banner a { color: #0b5394; font-weight: 600; }
.rbe-vs-banner button { position: absolute; right: 0.6rem; top: 0.3rem; border: 0;
  background: none; color: inherit; font-size: 1.2rem; cursor: pointer; line-height: 1; }
@media print { .rbe-vs, .rbe-vs-banner { display: none; } }`;

  function el(tag, attrs, text) {
    const e = document.createElement(tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }

  // The href (used for middle-click and copy-link) is the same page in that version;
  // a plain click resolves the nearest existing page first.
  function link(version, text) {
    const a = el("a", { href: candidates(version).urls[0] }, text);
    a.addEventListener("click", function (ev) {
      if (ev.ctrlKey || ev.metaKey || ev.shiftKey || ev.button !== 0) return;
      ev.preventDefault();
      go(version);
    });
    return a;
  }

  function banner(latest) {
    const key = "rbe-vs-dismissed:" + current;
    try {
      if (sessionStorage.getItem(key)) return;
    } catch (e) {}
    const div = el("div", { class: "rbe-vs-banner", role: "note" });
    div.append("You are viewing the documentation for an older release (" + current + "). The latest release is ");
    div.append(link(latest.version, latest.name));
    div.append(".");
    const close = el("button", { type: "button", "aria-label": "Dismiss" }, "×");
    close.addEventListener("click", function () {
      div.remove();
      try {
        sessionStorage.setItem(key, "1");
      } catch (e) {}
    });
    div.append(close);
    document.body.prepend(div);
  }

  function render(versions) {
    const entry = versions.find((v) => v.version === current);
    if (!entry) return;
    const latest = versions.find((v) => v.preferred);
    const old = latest && current !== "main" && current !== latest.version;

    const style = el("style");
    style.textContent = css;
    document.head.append(style);

    const box = el("details", { class: "rbe-vs" });
    const summary = el("summary", { "aria-label": "Select documentation version" });
    summary.append(el("span", old ? { class: "rbe-vs-old" } : {}, entry.name));
    box.append(summary);
    const list = el("ul");
    for (const v of versions) {
      const li = el("li", v.version === current ? { class: "rbe-vs-current" } : {});
      li.append(link(v.version, v.name));
      list.append(li);
    }
    box.append(list);
    // Close the list when clicking elsewhere
    document.addEventListener("click", (e) => { if (!box.contains(e.target)) box.open = false; });
    document.body.append(box);

    if (old) banner(latest);
  }

  fetch(root + "switcher.json")
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then(render)
    .catch(() => {});
})();
