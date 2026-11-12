/* Cinematic Grammar Tokenizer, static demo: the public build of the interactive demo (app/demo/app.py) as plain HTML, CSS and JavaScript.
   All data comes from data/ (written by scripts/build_static_app.py from the public demo data, public-domain films only); keyframes from
   keyframes/<film>/. Charts are the Vega-Lite specs of app/demo/viz.py, rendered with vega-embed. Every caveat text comes verbatim from
   data/site.json (the demo's own constants) and is drawn as a visible callout or caption, never collapsed. Nothing is sent anywhere. */
(function () {
  "use strict";

  var DATA = "data/";
  var KEYFRAMES = "keyframes/";
  var MODEL_SLOTS = ["size", "lens", "angle", "framing", "light", "comp", "move"];
  var HEURISTIC_SLOTS = ["key", "DoF", "people", "sound", "len", "palette"];
  var HOME_TITLE = "One readable token row per shot";
  var TITLES = { home: "Home", films: "Films", overview: "Film overview", explorer: "Shot explorer", compare: "Compare (side by side)",
                 reports: "Reports", about: "About" };
  var main = document.getElementById("main");
  var cache = {};
  var views = [];
  var observer = null;
  var explorerState = {};
  var arcMode = "equal20";
  var renderToken = 0;

  // ---- small helpers ---------------------------------------------------------------------------------------------------------------------------

  function esc(s) {  // = Python html.escape(s, quote=True)
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#x27;");
  }

  function h(tag, attrs) {
    var el = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        var v = attrs[k];
        if (v === null || v === undefined || v === false) return;
        if (k === "text") el.textContent = v;
        else if (k === "html") el.innerHTML = v;
        else if (k.slice(0, 2) === "on") el.addEventListener(k.slice(2), v);
        else el.setAttribute(k, v === true ? "" : v);
      });
    }
    for (var i = 2; i < arguments.length; i++) add(el, arguments[i]);
    return el;
  }

  function add(el, c) {
    if (c === null || c === undefined || c === false) return;
    if (Array.isArray(c)) { c.forEach(function (x) { add(el, x); }); return; }
    el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }

  function raw(htmlText) {  // a trusted snippet written by the build (style.py / viz.py output)
    var t = document.createElement("template");
    t.innerHTML = htmlText;
    return t.content;
  }

  function fetchJSON(path) {
    if (!cache[path]) {
      cache[path] = fetch(DATA + path, { credentials: "same-origin" }).then(function (r) {
        if (!r.ok) throw new Error(path + ": HTTP " + r.status);
        return r.json();
      });
      cache[path].catch(function () { delete cache[path]; });
    }
    return cache[path];
  }

  function theme() { return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light"; }

  var ICON_WARN = '<svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>';
  var ICON_INFO = '<svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor"><path d="M11 7h2v2h-2zm0 4h2v6h-2zm1-9C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/></svg>';
  var ICON_MOON = '<svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor"><path d="M12 3a9 9 0 1 0 9 9c0-.46-.04-.92-.1-1.36a5.39 5.39 0 0 1-4.4 2.26 5.4 5.4 0 0 1-3.14-9.8c-.44-.06-.9-.1-1.36-.1z"/></svg>';
  var ICON_SUN = '<svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor"><path d="M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10zM2 13h2a1 1 0 0 0 0-2H2a1 1 0 0 0 0 2zm18 0h2a1 1 0 0 0 0-2h-2a1 1 0 0 0 0 2zM11 2v2a1 1 0 0 0 2 0V2a1 1 0 0 0-2 0zm0 18v2a1 1 0 0 0 2 0v-2a1 1 0 0 0-2 0zM5.99 4.58a1 1 0 0 0-1.41 1.41l1.06 1.06a1 1 0 0 0 1.41-1.41L5.99 4.58zm12.37 12.37a1 1 0 0 0-1.41 1.41l1.06 1.06a1 1 0 0 0 1.41-1.41l-1.06-1.06zm1.06-10.96a1 1 0 0 0-1.41-1.41l-1.06 1.06a1 1 0 0 0 1.41 1.41l1.06-1.06zM7.05 18.36a1 1 0 0 0-1.41-1.41l-1.06 1.06a1 1 0 0 0 1.41 1.41l1.06-1.06z"/></svg>';

  function caveat(text) {  // a required caveat: always visible, full text
    var box = h("div", { "class": "callout warn", role: "note", "data-caveat": "" });
    box.appendChild(raw(ICON_WARN));
    box.appendChild(h("p", { text: text }));
    return box;
  }

  function info(text) {
    var box = h("div", { "class": "callout info", role: "note" });
    box.appendChild(raw(ICON_INFO));
    box.appendChild(h("p", { text: text }));
    return box;
  }

  function caption(text) { return h("p", { "class": "caption", text: text }); }

  function footer() {
    return h("footer", { "class": "footer" },
      h("p", null, "Night of the Living Dead (1968) is in the public domain. Every token on these pages is a model prediction or a heuristic of unmeasured accuracy. ",
        h("a", { href: "../" }, "Read the paper"), " · ", h("a", { href: "https://github.com/jahoi12345/cinematic-grammar-tokenizer" }, "Source code"),
        " · The interactive version of this demo runs locally with Streamlit (see the repository README)."));
  }

  // ---- token rows (= style.parse_row / style.chips_html) -----------------------------------------------------------------------------------------

  function parseRow(row) {
    if (typeof row !== "string") return [];
    var s = row.trim();
    if (!(s.charAt(0) === "[" && s.charAt(s.length - 1) === "]")) return [];
    var out = [];
    var parts = s.slice(1, -1).split("|");
    for (var i = 0; i < parts.length; i++) {
      var part = parts[i].trim();
      var at = part.indexOf("=");
      if (at < 0) return [];
      var slot = part.slice(0, at);  // = Python's partition: the slot keeps its own text
      if (!slot) return [];
      var value = part.slice(at + 1).trim();
      var bare = value.replace(/\*+$/, "");
      out.push({ slot: slot, value: value, low: value.slice(-1) === "*", missing: bare === "?" || bare === "n/a",
                 source: MODEL_SLOTS.indexOf(slot) >= 0 ? "model" : HEURISTIC_SLOTS.indexOf(slot) >= 0 ? "heuristic" : "other" });
    }
    return out;
  }

  function chipsHtml(row, size) {
    var slots = parseRow(row);
    if (!slots.length) return "";
    var chips = slots.map(function (t) {
      var val = esc(t.value.replace(/\*+$/, "")) + (t.low ? '<span class="cgt-star" title="start, mid and end keyframes disagree">*</span>' : "");
      var cls = "cgt-chip cgt-" + t.source + (t.low ? " cgt-low" : "") + (t.missing ? " cgt-missing" : "");
      var tip = t.source === "model" ? "model prediction" : t.source === "heuristic" ? "heuristic" : "";
      return '<span class="' + cls + '" role="listitem" title="' + tip + '"><span class="cgt-slot">' + esc(t.slot) + '</span>' +
             '<span class="cgt-val">' + val + "</span></span>";
    });
    return '<div class="cgt-row cgt-' + (size || "md") + '" role="list" aria-label="token row">' + chips.join("") + "</div>";
  }

  function tokenRow(row, chips, size) {  // chips (optional) then the raw row, wrapped, never cut
    var frag = document.createDocumentFragment();
    if (chips) { var c = chipsHtml(row, size); if (c) frag.appendChild(raw(c)); }
    frag.appendChild(h("pre", { "class": "raw" }, h("code", { text: row == null ? "" : String(row) })));
    return frag;
  }

  // ---- charts ----------------------------------------------------------------------------------------------------------------------------------

  function chartBox(film, group, name, extraClass) {
    var el = h("div", { "class": "chart" + (extraClass ? " " + extraClass : ""), "data-film": film, "data-group": group, "data-chart": name });
    el.appendChild(h("div", { "class": "chart-loading", text: "Drawing the chart" }));
    if (observer) observer.observe(el); else embed(el);
    return el;
  }

  function embed(el) {
    var token = renderToken;
    var film = el.getAttribute("data-film"), group = el.getAttribute("data-group"), name = el.getAttribute("data-chart");
    fetchJSON(film + "/" + group + ".json").then(function (charts) {
      if (token !== renderToken) return;
      var t = theme();
      var spec = JSON.parse(JSON.stringify(charts.specs[t][name]));
      var used = spec.usedDatasets || [];
      delete spec.usedDatasets;
      spec.datasets = {};
      used.forEach(function (d) { spec.datasets[d] = JSON.parse(JSON.stringify(charts.datasets[d])); });
      if (typeof window.vegaEmbed !== "function") throw new Error("the chart library did not load");
      return window.vegaEmbed(el, spec, { actions: false, renderer: "svg", tooltip: { theme: t === "dark" ? "dark" : "light" } }).then(function (res) {
        if (token !== renderToken) { res.finalize(); return; }
        views.push(res);
      });
    }).catch(function (err) {
      el.textContent = "";
      el.appendChild(h("p", { "class": "caption", text: "This chart could not be drawn (" + err.message + ")." }));
    });
  }

  function resetCharts() {
    renderToken += 1;
    views.forEach(function (v) { try { v.finalize(); } catch (e) { /* already gone */ } });
    views = [];
    if (observer) observer.disconnect();
    observer = "IntersectionObserver" in window ? new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { observer.unobserve(e.target); embed(e.target); }
      });
    }, { rootMargin: "600px 0px" }) : null;
  }

  // ---- pages -----------------------------------------------------------------------------------------------------------------------------------

  function filmIds(site) { return site.films.map(function (f) { return f.id; }); }
  function filmEntry(site, id) { return site.films.filter(function (f) { return f.id === id; })[0] || null; }
  function frameSrc(film, name) { return KEYFRAMES + encodeURIComponent(film) + "/" + encodeURIComponent(name); }
  function img(film, name, alt, extra) {
    return h("img", Object.assign({ src: frameSrc(film, name), alt: alt, loading: "lazy", decoding: "async", width: "640", height: "480" }, extra || {}));
  }

  function pageHome(site) {
    var t = site.text;
    var out = [h("h1", { text: HOME_TITLE }),
      h("p", { "class": "lede", html: "This project turns a film into <strong>one readable line of tokens per shot</strong>: shot size, lens, angle, framing, light, " +
        "composition, key, depth of field, people, sound, length, palette and camera movement, in fixed slots. The rows can be searched, filtered, counted " +
        "and plotted over a film&#x27;s running time." }),
      caveat(t.predictions)];
    var heroFilm = site.films.filter(function (f) { return f.hero; })[0];
    var tail = function () {
      var list = h("ul", { "class": "plain" });
      site.films.forEach(function (f) { list.appendChild(h("li", { text: f.label })); });
      var links = [["#/films", "Films", "every film with its shots and running time"], ["#/compare", "Compare (side by side)", "within-film shapes of two films"],
                   ["#/reports", "Reports", "the paper and the phase reports"], ["#/about", "About", "how to read a token row, how it was verified, caveats"]];
      return [h("h2", { text: "Films in this build" }), site.films.length ? list : h("p", { text: "No film is available in this mode." }),
        h("h2", { text: "Where to go" }),
        h("div", { "class": "linkgrid" }, links.map(function (l) { return h("a", { "class": "linkcard", href: l[0] }, h("strong", { text: l[1] }), h("span", { text: l[2] })); })),
        caption("Public build: public-domain films only, the public paper and the reports that pass the release checker."), footer()];
    };
    if (!heroFilm) return Promise.resolve(out.concat(tail()));
    return fetchJSON(heroFilm.id + "/film.json").then(function (fj) {
      var hero = fj.hero, parts = hero.home_html[theme()];
      out.push(raw(parts.head));
      out.push(h("div", { "class": "hscroll" }, chartBox(fj.film, "charts_home", "barcode")));
      out.push(raw(parts.legend), raw(parts.note), raw(parts.zoom));
      var right = h("div", null, raw(site.chip_key_html), raw(hero.chips), tokenRow(hero.token_row, false));
      out.push(h("div", { "class": "hero" },
        h("div", null, img(fj.film, hero.kf_mid, "Mid keyframe of shot " + hero.shot_id, { loading: "eager" }), h("div", { "class": "cgt-hero-cap", text: hero.caption })),
        right));
      return out.concat(tail());
    });
  }

  function pageFilms(site) {
    var cards = site.films.map(function (f) {
      return h("div", { "class": "filmcard" },
        f.hero ? img(f.id, f.hero.kf_mid, "A keyframe of " + f.label) : null,
        h("strong", { text: f.label }), h("p", { "class": "caption", text: f.card }),
        h("a", { "class": "btn", href: "#/overview/" + encodeURIComponent(f.id) }, "Film overview"),
        h("a", { "class": "btn", href: "#/explorer/" + encodeURIComponent(f.id) }, "Shot explorer"));
    });
    return Promise.resolve([h("h1", { text: "Films" }),
      caption("One card per film available in this build. Shots and running time come from the heuristic cut table (no labels exist for these films)."),
      h("div", { "class": "films" }, cards), footer()]);
  }

  function filmPicker(site, route, film) {
    if (site.films.length < 2) return null;
    var sel = h("select", { "aria-label": "Film", onchange: function () { location.hash = "#/" + route + "/" + encodeURIComponent(sel.value); } });
    site.films.forEach(function (f) { sel.appendChild(h("option", { value: f.id, selected: f.id === film, text: f.label })); });
    return h("label", { "class": "fgroup" }, h("span", { "class": "flabel", text: "Film" }), sel);
  }

  function pageOverview(site, film) {
    return Promise.all([fetchJSON(film + "/film.json")]).then(function (r) {
      var fj = r[0], t = site.text, th = theme();
      var out = [filmPicker(site, "overview", film), h("h1", { text: "Film overview: " + fj.label }), caveat(t.predictions),
        h("div", { "class": "tiles" }, fj.tiles.map(function (x) {
          return h("div", { "class": "tile" }, h("div", { "class": "tile-label", text: x[0] }), h("div", { "class": "tile-value", text: x[1] }));
        })),
        caption(fj.length_caption), chartBox(film, "charts", "length")];
      if (!fj.has_arc) {
        out.push(info("No v1 token table with the arc columns for this film: the visual arc needs `<film>_tokens_v1.parquet`."), footer());
        return out;
      }
      var hh = fj.html[th];
      out.push(caption(fj.kf_caption), raw(hh.glance));
      out.push(h("section", { "class": "box" }, raw(hh.title_page), chartBox(film, "charts", "page", "page"), raw(hh.legend_lanes), raw(hh.note)));
      out.push(h("section", { "class": "box" }, raw(hh.title_strip),
        h("div", null, h("img", { "class": "strip", src: DATA + encodeURIComponent(film) + "/" + fj.strip, alt: "Contact-sheet barcode: every shot's mid keyframe averaged into one column of tone, as wide as the shot is long", loading: "lazy", width: "1600", height: "160" }),
          chartBox(film, "charts", "size_lane")), raw(hh.legend_size), raw(hh.strip_cap), raw(hh.note)));
      out.push(raw(hh.rhythm_sec));
      out.push(h("section", { "class": "box" }, raw(hh.title_pace), chartBox(film, "charts", "pace_top"), chartBox(film, "charts", "pace_bottom"), raw(hh.legend_pace), raw(hh.note)));
      out.push(h("section", { "class": "box" }, raw(hh.title_rhythm), chartBox(film, "charts", "rhythm"), raw(hh.legend_rhythm), raw(hh.note)));
      out.push(raw(hh.grammar_sec));
      out.push(h("section", { "class": "box" }, raw(hh.title_grammar), chartBox(film, "charts", "grammar"), raw(hh.note)));
      var fp = raw(hh.fingerprint);
      Array.prototype.forEach.call(fp.querySelectorAll(".cgt-vfr"), function (row, k) {  // = the title text of viz.fingerprint_html
        var tip = fj.fingerprint_tips[k];
        if (tip) row.setAttribute("title", tip[0] + " = " + tip[1] + ": " + tip[2] + " of " + fj.stats.shots + " shots (" + tip[3] + "%)" +
          (tip[4] ? "; keyframes disagree on " + tip[4] + "% of them" : ""));
      });
      out.push(h("section", { "class": "box" }, raw(hh.title_fingerprint), fp, raw(hh.note)));
      out.push(h("h2", { text: "Visual arc" }));
      var arcHost = h("div");
      var seg = h("div", { "class": "seg", role: "group", "aria-label": "windows" });
      [["equal20", "20 equal windows of running time (primary)"], ["min5", "5-minute windows (secondary)"]].forEach(function (m) {
        seg.appendChild(h("button", { type: "button", "aria-pressed": String(arcMode === m[0]), text: m[1], onclick: function () {
          arcMode = m[0];
          Array.prototype.forEach.call(seg.children, function (b, i) { b.setAttribute("aria-pressed", String(i === (arcMode === "equal20" ? 0 : 1))); });
          drawArc();
        } }));
      });
      function drawArc() {
        arcHost.textContent = "";
        fj.panels.forEach(function (title, i) {
          arcHost.appendChild(h("section", { "class": "box" }, h("strong", { text: title }), chartBox(film, "charts", "arc_" + arcMode + "_" + i)));
        });
        arcHost.appendChild(caption(fj.window_counts[arcMode]));
        if (fj.has_movement) {
          arcHost.appendChild(h("h2", { text: "Predicted camera movement" }));
          arcHost.appendChild(caveat(fj.movement_caption));
          arcHost.appendChild(h("section", { "class": "box" }, chartBox(film, "charts", "move_" + arcMode + "_0")));
          arcHost.appendChild(h("section", { "class": "box" }, chartBox(film, "charts", "move_" + arcMode + "_1")));
        }
      }
      out.push(h("div", { "class": "fgroup", style: "margin-bottom:12px" }, h("span", { "class": "flabel", text: "windows" }), seg),
        caption(t.intervals), caption(t.shapes), arcHost);
      drawArc();
      out.push(footer());
      return out;
    });
  }

  function multiselect(label, values, selected, onchange) {
    var cnt = h("span", { "class": "cnt" });
    var list = h("div", { "class": "ms-list" });
    function count() { cnt.textContent = selected.size ? selected.size + " selected" : "any"; }
    values.forEach(function (v) {
      var box = h("input", { type: "checkbox", value: v, checked: selected.has(v), onchange: function () {
        if (box.checked) selected.add(v); else selected.delete(v);
        count(); onchange();
      } });
      list.appendChild(h("label", null, box, String(v)));
    });
    count();
    return h("div", { "class": "fgroup" }, h("span", { "class": "flabel", text: label }),
      h("details", { "class": "ms" }, h("summary", null, h("span", { text: "choose" }), cnt), list));
  }

  function pageExplorer(site, film) {
    return fetchJSON(film + "/shots.json").then(function (sj) {
      var t = site.text, C = sj.columns, n = sj.n, o = sj.options;
      var st = explorerState[film] || (explorerState[film] = { sel: {}, people: new Set(), move: new Set(), lo: o.len_range[0], hi: o.len_range[1], kf: "any", page: 1 });
      o.slots.forEach(function (s) { if (!st.sel[s.column]) st.sel[s.column] = new Set(); });
      var byId = {};
      C.shot_id.forEach(function (id, i) { byId[id] = i; });
      var results = h("div");
      var countEl = h("p", { style: "margin:0" });
      var pagerTop = h("div", { "class": "pager" });
      var activeEl = h("span", { "class": "cnt" });

      function matches() {
        var idx = [];
        var full = st.lo <= o.len_range[0] && st.hi >= o.len_range[1];
        for (var i = 0; i < n; i++) {
          var ok = true;
          for (var s = 0; s < o.slots.length && ok; s++) {
            var col = o.slots[s].column, set = st.sel[col];
            if (set.size && !set.has(C[col][i])) ok = false;
          }
          if (ok && !full && !(C.len_s[i] >= st.lo && C.len_s[i] <= st.hi)) ok = false;
          if (ok && st.people.size && !st.people.has(C.people_tok[i])) ok = false;
          if (ok && st.move.size && C.move_atoms) {
            var a = C.move_atoms[i], any = false;
            if (a) for (var k = 0; k < a.length; k++) if (st.move.has(a[k])) { any = true; break; }
            if (!any) ok = false;
          }
          if (ok && C.n_kf_disagree && st.kf === "disagree" && !(C.n_kf_disagree[i] > 0)) ok = false;
          if (ok && C.n_kf_disagree && st.kf === "agree" && !(C.n_kf_disagree[i] === 0)) ok = false;
          if (ok) idx.push(i);
        }
        return idx;
      }

      function activeCount() {
        var c = st.people.size + st.move.size + (st.kf !== "any" ? 1 : 0) + ((st.lo > o.len_range[0] || st.hi < o.len_range[1]) ? 1 : 0);
        Object.keys(st.sel).forEach(function (k) { c += st.sel[k].size; });
        return c;
      }

      function card(i) {
        var id = C.shot_id[i];
        var frames = h("div", { "class": "frames" });
        [["kf_start", "start"], ["kf_mid", "mid"], ["kf_end", "end"]].forEach(function (k) {
          var name = C[k[0]][i];
          frames.appendChild(name ? h("figure", null, img(film, name, k[1] + " keyframe of shot " + id), h("figcaption", { text: k[1] }))
                                  : h("figure", null, h("p", { "class": "caption", text: k[1] + ": frame not available" })));
        });
        var sims = sj.similar[i] || [];
        var det = h("details", { "class": "similar" }, h("summary", { text: "Shots like this" }));
        det.addEventListener("toggle", function () {
          if (!det.open || det.getAttribute("data-done")) return;
          det.setAttribute("data-done", "1");
          det.appendChild(caption(sj.similar_caption));
          sims.forEach(function (s) {
            var j = byId[s[1]];
            var mid = j === undefined ? null : C.kf_mid[j];
            det.appendChild(h("div", { "class": "simrow" },
              h("div", null, mid ? h("figure", { style: "margin:0" }, img(film, mid, "mid keyframe of shot " + s[1]), h("figcaption", { "class": "caption", style: "text-align:center;margin:2px 0 0", text: "mid" }))
                                 : caption("similar shot: mid frame not available")),
              h("div", null, h("p", { "class": "t" }, h("strong", { text: "#" + s[0] + " shot " + s[1] }),
                " · start " + s[2] + " min · gap " + s[3] + " min · similarity " + s[4]),
                tokenRow(j === undefined ? "" : C.token_row[j], false))));
          });
        });
        return h("article", { "class": "shotcard", "aria-label": "shot " + id },
          h("div", { "class": "head" }, h("strong", { text: "shot " + id }), h("span", { text: "· start " + C.start_min_txt[i] + " min · length " + C.len_txt[i] + " s" })),
          frames, tokenRow(C.token_row[i], true),
          C.detail ? caption("p1/margin per attribute (the model's own softmax, not calibrated accuracy): " +
            C.detail[i].map(function (d) { return d[0] + " " + d[1] + "/" + d[2]; }).join(" · ")) : null, det);
      }

      function pager(nPages, where) {
        var input = h("input", { type: "number", min: "1", max: String(nPages), value: String(st.page), "aria-label": "page (1-" + nPages + ")",
          onchange: function () { var v = parseInt(input.value, 10); if (!isNaN(v)) { st.page = Math.min(Math.max(1, v), nPages); draw(where === "bottom"); } } });
        return [h("button", { type: "button", "class": "btn", disabled: st.page <= 1, onclick: function () { st.page -= 1; draw(where === "bottom"); } }, "Previous"),
          h("label", null, "page (1-" + nPages + ") ", input),
          h("button", { type: "button", "class": "btn", disabled: st.page >= nPages, onclick: function () { st.page += 1; draw(where === "bottom"); } }, "Next")];
      }

      function draw(scrollUp) {
        var idx = matches();
        var nPages = Math.max(1, Math.ceil(idx.length / sj.page_size));
        st.page = Math.min(Math.max(1, st.page), nPages);
        countEl.textContent = "";
        add(countEl, [h("strong", { text: idx.length.toLocaleString("en-US") }), " of " + n.toLocaleString("en-US") + " shots match the filters"]);
        var ac = activeCount();
        activeEl.textContent = ac ? ac + " active" : "";
        pagerTop.textContent = "";
        add(pagerTop, pager(nPages, "top"));
        results.textContent = "";
        if (!idx.length) results.appendChild(h("p", { "class": "empty", text: "No shot matches these filters. Clear a filter to see shots again." }));
        idx.slice((st.page - 1) * sj.page_size, st.page * sj.page_size).forEach(function (i) { results.appendChild(card(i)); });
        if (idx.length > sj.page_size) results.appendChild(h("div", { "class": "pager", style: "justify-content:flex-end" }, pager(nPages, "bottom")));
        if (scrollUp) document.getElementById("results-top").scrollIntoView({ block: "start" });
      }
      function changed() { st.page = 1; draw(false); }

      var groups = [];
      o.slots.forEach(function (s) { groups.push(multiselect(s.label, s.values, st.sel[s.column], changed)); });
      groups.push(multiselect("people", o.people, st.people, changed));
      if (o.move) groups.push(multiselect("move (any atom of the predicted set)", o.move, st.move, changed));
      var loIn = h("input", { type: "number", step: "0.1", min: String(o.len_range[0]), max: String(o.len_range[1]), value: String(+st.lo.toFixed(2)) });
      var hiIn = h("input", { type: "number", step: "0.1", min: String(o.len_range[0]), max: String(o.len_range[1]), value: String(+st.hi.toFixed(2)) });
      function lenChanged() {
        var a = parseFloat(loIn.value), b = parseFloat(hiIn.value);
        st.lo = isNaN(a) || a <= +o.len_range[0].toFixed(2) ? o.len_range[0] : a;
        st.hi = isNaN(b) || b >= +o.len_range[1].toFixed(2) ? o.len_range[1] : b;
        changed();
      }
      loIn.addEventListener("change", lenChanged); hiIn.addEventListener("change", lenChanged);
      groups.push(h("div", { "class": "fgroup" }, h("span", { "class": "flabel", text: "shot length (s)" }),
        h("div", { "class": "range" }, h("label", null, "from", loIn), h("label", null, "to", hiIn))));
      if (o.keyframe_modes) {
        var seg = h("div", { "class": "seg", role: "group", "aria-label": "keyframes" });
        o.keyframe_modes.forEach(function (m) {
          seg.appendChild(h("button", { type: "button", "aria-pressed": String(st.kf === m), text: m, onclick: function () {
            st.kf = m;
            Array.prototype.forEach.call(seg.children, function (b) { b.setAttribute("aria-pressed", String(b.textContent === m)); });
            changed();
          } }));
        });
        groups.push(h("div", { "class": "fgroup" }, h("span", { "class": "flabel", text: "keyframes" }), seg));
      }
      groups.push(h("button", { type: "button", "class": "btn fclear", onclick: function () {
        delete explorerState[film];
        route();
      } }, "Clear all filters"));
      var wide = window.matchMedia("(min-width: 900px)").matches;
      var panel = h("details", { "class": "filters", open: wide }, h("summary", null, h("span", null, "Filter by token slot ", activeEl)), groups);

      draw(false);
      var legend = h("details", { "class": "box", style: "gap:0" }, h("summary", { style: "cursor:pointer;font-weight:500", text: "How to read a token row (slots, *, ?)" }), slotTables(site));
      return [filmPicker(site, "explorer", film), h("h1", { text: "Shot explorer: " + filmEntry(site, film).label }), caveat(t.explorer), raw(site.chip_key_html), legend,
        h("div", { "class": "explorer" }, panel,
          h("div", { style: "min-width:0" }, h("div", { "class": "resbar", id: "results-top" }, countEl, pagerTop), results)), footer()];
    });
  }

  function slotTables(site) {
    var t1 = h("table", { "class": "slots" }, h("thead", null, h("tr", null, h("th", { text: "slot" }), h("th", { text: "source" }), h("th", { text: "meaning" }))));
    var b1 = h("tbody");
    site.slot_help.forEach(function (r) { b1.appendChild(h("tr", null, h("td", null, h("code", { text: r[0] })), h("td", { text: r[1] }), h("td", { text: r[2] }))); });
    t1.appendChild(b1);
    var t2 = h("table", { "class": "slots" }, h("thead", null, h("tr", null, h("th", { text: "marker" }), h("th", { text: "meaning" }))));
    var b2 = h("tbody");
    site.marker_help.forEach(function (r) { b2.appendChild(h("tr", null, h("td", null, h("code", { text: r[0] })), h("td", { text: r[1] }))); });
    t2.appendChild(b2);
    return h("div", { style: "padding-top:10px" }, t1, t2);
  }

  function pageCompare(site) {
    var t = site.text;
    var out = [h("h1", { text: "Compare: within-film shapes side by side" }), caveat(t.shapes), caption(t.predictions)];
    if (site.compare_html) out.push(raw(site.compare_html));
    out.push(info(t.compare_info), footer());
    return Promise.resolve(out);
  }

  var DOC_LINKS = { "docs/writeup/paper_public.html": "../", "docs/writeup/paper_public.pdf": "../paper_public.pdf" };

  function pageReports(site) {
    var lib = raw(site.reports_html);
    Array.prototype.forEach.call(lib.querySelectorAll("li.cgt-doc"), function (li) {
      var path = li.querySelector(".cgt-doc-path"), title = li.querySelector(".cgt-doc-title");
      var href = path && DOC_LINKS[path.textContent];
      if (href && title) {
        var a = h("a", { href: href, text: title.textContent });
        title.textContent = "";
        title.appendChild(a);
      }
    });
    return Promise.resolve([h("h1", { text: "Reports" }), lib,
      h("h2", { text: "The paper" }),
      caption("Source: docs/writeup/paper_public.html (the same paper as the report site and its PDF)."),
      h("div", { "class": "btnrow" }, h("a", { "class": "btn primary", href: "../" }, "Open the paper"),
        h("a", { "class": "btn", href: "../paper_public.pdf" }, "Download the paper (PDF)")),
      h("iframe", { "class": "paperframe", src: "../", title: "The project paper", loading: "lazy" }),
      caption(site.text.model),
      h("h2", { text: "Phase reports" }),
      caption("Public build: only reports that pass the release checker are listed; internal results are never shown."),
      info("No report is available in this build."), footer()]);
  }

  function pageAbout(site) {
    var t = site.text;
    var cav = [["Predictions of unmeasured accuracy.", t.predictions], ["Within-film shapes only.", t.shapes], ["Intervals.", t.intervals],
               ["Inflated movement sets.", t.movement_generic], ["Shots like this.", t.similar]];
    var films = h("ul", { "class": "plain" });
    site.films.forEach(function (f) { films.appendChild(h("li", { text: f.label })); });
    var verified = h("ul", { "class": "plain" });
    [["Pre-registration", ": plans, hyper-parameters, selection and stopping rules were committed before each training run; later changes were written down as dated amendments."],
     ["Sealed evaluators", ": test labels sat behind an evaluator that refused uncommitted code and logged every evaluation once."],
     ["Independent checks", ": each change was checked by a separate verification pass that did not write it."],
     ["Mutation checks", ": each guard was broken on purpose to check that a test failed (this demo's public-mode guard included)."],
     ["Byte-identical reproductions", " of model runs and token tables where possible."]].forEach(function (b) {
      verified.appendChild(h("li", null, h("strong", { text: b[0] }), b[1]));
    });
    return Promise.resolve([h("h1", { text: "About this demo" }),
      h("p", { "class": "lede", html: "The <strong>Cinematic Grammar Tokenizer</strong> turns a film into <strong>one readable line of tokens per shot</strong>. Film students " +
        "describe a shot in a shared vocabulary: how big the subject is, where the camera sits, how it moves, how the scene is lit, how long the shot lasts. This " +
        "project writes that description for every shot automatically, with fixed slots in a fixed order, so a film can be searched, counted and plotted without " +
        "watching it again (for example: every low-angle close-up with one person)." }),
      caveat(t.predictions),
      h("h2", { text: "What a shot token row is" }),
      h("p", null, "Each shot gets one row with 13 slots, always in this order (row version v1, ", h("code", { text: "configs/tokens.yaml" }), "). Model slots come from " +
        "trained models; the others come from hand-written rules and pretrained detectors whose thresholds are not checked against labels."),
      raw(site.chip_key_html), slotTables(site),
      h("h2", { text: "Which model writes the static slots" }), h("p", { text: t.model }),
      h("h2", { text: "How it was verified" }), verified,
      h("p", { text: "None of this measures how accurate the tokens are on the films shown here: no labels exist for them yet." }),
      h("h2", { text: "Caveats (read before any page)" }),
      cav.map(function (c) { return [h("p", { style: "margin:0 0 2px" }, h("strong", { text: c[0] })), caption(c[1])]; }),
      h("h2", { text: "Films in this mode" }), films,
      caption("Public mode: only films marked public domain in the committed film registry are listed."),
      h("h2", { text: "Run it locally" }),
      h("p", null, "These pages are a static build of the demo: the data and charts were computed ahead of time from the public token table. The same demo also runs " +
        "as an interactive Streamlit app on your own machine, from a copy of the repository: ", h("code", { text: "pip install -r requirements.txt" }), " and then ",
        h("code", { text: "streamlit run streamlit_app.py" }), " (public mode, the bundled public-domain data only)."),
      footer()]);
  }

  function notFound() {
    return Promise.resolve([h("h1", { text: "Page not found" }), h("p", null, "There is no such page in this demo. ", h("a", { href: "#/", text: "Go to Home" }), "."), footer()]);
  }

  // ---- router ----------------------------------------------------------------------------------------------------------------------------------

  function parseHash() {
    var parts = (location.hash || "#/").replace(/^#\/?/, "").split("/").filter(Boolean).map(function (p) {
      try { return decodeURIComponent(p); } catch (e) { return "\u0000"; }
    });
    return { page: parts[0] || "home", film: parts[1] || null, extra: parts.length > 2 };
  }

  function route() {
    var r = parseHash();
    resetCharts();
    var token = renderToken;
    fetchJSON("site.json").then(function (site) {
      var ids = filmIds(site), job;
      var film = r.film || ids[0];
      if (r.extra) job = notFound();
      else if (r.page === "home" && !r.film) job = pageHome(site);
      else if (r.page === "films" && !r.film) job = pageFilms(site);
      else if (r.page === "overview" && ids.indexOf(film) >= 0) job = pageOverview(site, film);  // only films listed in site.json (public build)
      else if (r.page === "explorer" && ids.indexOf(film) >= 0) job = pageExplorer(site, film);
      else if (r.page === "compare" && !r.film) job = pageCompare(site);
      else if (r.page === "reports" && !r.film) job = pageReports(site);
      else if (r.page === "about" && !r.film) job = pageAbout(site);
      else if ((r.page === "overview" || r.page === "explorer") && !ids.length) job = Promise.resolve([h("h1", { text: TITLES[r.page] }),
        caveat("No film is available in this mode (no token table, or the film registry could not be read)."), footer()]);
      else job = notFound();
      return job.then(function (nodes) {
        if (token !== renderToken) return;
        main.textContent = "";
        add(main, nodes);
        var known = TITLES[r.page] ? r.page : null;
        Array.prototype.forEach.call(document.querySelectorAll(".nav a"), function (a) {
          if (a.getAttribute("data-route") === known) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
        });
        document.title = (known && known !== "home" ? TITLES[known] + " · " : "") + "Cinematic Grammar Tokenizer";
      });
    }).catch(function (err) {
      if (token !== renderToken) return;
      main.textContent = "";
      add(main, [h("h1", { text: "This page could not be loaded" }), h("p", { text: String(err && err.message || err) })]);
    });
  }

  // ---- theme -----------------------------------------------------------------------------------------------------------------------------------

  var themeBtn = document.getElementById("theme-btn");
  function paintThemeButton() {
    var dark = theme() === "dark";
    themeBtn.innerHTML = (dark ? ICON_SUN : ICON_MOON) + "<span>" + (dark ? "Light theme" : "Dark theme") + "</span>";
    themeBtn.setAttribute("aria-label", dark ? "Switch to the light theme" : "Switch to the dark theme");
  }
  function setTheme(t, remember) {
    document.documentElement.setAttribute("data-theme", t);
    if (remember) { try { window.localStorage.setItem("cgt-theme", t); } catch (e) { /* storage blocked: the choice lasts for this page only */ } }
    paintThemeButton();
    var y = window.scrollY;
    route();
    window.requestAnimationFrame(function () { window.scrollTo(0, y); });
  }
  themeBtn.addEventListener("click", function () { setTheme(theme() === "dark" ? "light" : "dark", true); });
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    var follow = function (e) {
      var stored = null;
      try { stored = window.localStorage.getItem("cgt-theme"); } catch (x) { stored = null; }
      if (stored !== "light" && stored !== "dark") setTheme(e.matches ? "dark" : "light", false);
    };
    if (mq.addEventListener) mq.addEventListener("change", follow); else if (mq.addListener) mq.addListener(follow);
  }
  paintThemeButton();

  window.addEventListener("hashchange", function () {
    route();
    window.scrollTo(0, 0);
    main.focus({ preventScroll: true });
  });
  route();

  window.cgtStatic = { parseRow: parseRow, chipsHtml: chipsHtml };  // for the page's own checks (no side effects)
})();
