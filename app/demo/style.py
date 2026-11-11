"""Demo v3 look (docs/design/revamp_plan.md, docs/demo_spec.md section v3): the ONE place for injected CSS and presentational HTML.

Everything the theme in .streamlit/config.toml cannot express lives here: token chips, the nav-row look of the sidebar page radio, the callout shape,
card and metric surfaces. Rules:
- stable selectors only: Streamlit `data-testid` attributes, `data-selected`, the `st-key-<key>` class of keyed containers, and our own `cgt-*` classes;
  never hashed `st-emotion-cache-*` classes;
- presentational only: no caveat, banner, number or filter is produced here; every HTML snippet carries a `cgt-` class so tests can tell it apart;
- colours that must follow light/dark are drawn as translucent overlays on the theme ground (work on both), the accent comes from the active theme type.
Pure helpers (parse_row, chips_html, strip_emoji) have no streamlit dependency; inject() is the only function that calls streamlit."""
import html
import re

ACCENT = {"light": "#146964", "dark": "#3FB5AD"}  # = primaryColor of [theme.light] / [theme.dark]
CHART_COLORS = ["#1F8F88", "#C27A1A", "#C4506A", "#4F78C4", "#6F8A2E", "#8E5FA8"]  # = theme.chartCategoricalColors; each >= 3:1 on both grounds
# v3.2 chart palettes of app/demo/viz.py, selected per theme (dataviz method; values in docs/demo_spec.md v3.2, pinned by tests/test_demo_v32_viz.py).
# Ordinal ramp: one teal hue (OKLCH h 190) in 7 monotone lightness steps, index 0 = the most intense on that theme's ground (closest size / most);
# light L 0.27..0.63, dark L 0.93..0.50 (adjacent dL >= 0.06); every step >= 3:1 against its own theme ground.
VIZ_RAMP = {"light": ["#022D2B", "#073E3B", "#0A504D", "#07625F", "#137571", "#348682", "#4D9893"],
            "dark": ["#93FEF7", "#80E5DE", "#6DCDC6", "#5AB5AF", "#479D98", "#358782", "#21716D"]}
VIZ_GREYS = {"light": ["#27292B", "#56585A", "#8A8C8E"], "dark": ["#616365", "#A1A3A5", "#E6E8EA"]}  # key = brightness: low, mid, high key; >= 3:1 on own ground
VIZ_NOMINAL = ["#1F8F88", "#C27A1A", "#4F78C4", "#6F8A2E", "#8E5FA8"]  # CHART_COLORS re-ordered without rose
VIZ_HIGHLIGHT = "#C4506A"  # = CHART_COLORS[2]: the one highlight (fastest-cut minutes)
VIZ_NEUTRAL = "#808389"  # de-emphasised marks and heuristic bars: >= 3:1 on both grounds (3.5 light, 5.0 dark)
VIZ_MISSING = {"light": "#D9DAD5", "dark": "#2C3035"}  # '?' (no value): the hairline colour, deliberately quiet
VIZ_GROUND = {"light": "#F7F7F5", "dark": "#0F1113"}
VIZ_SURFACE = {"light": "#ECEDEA", "dark": "#191C1F"}
VIZ_LINE = {"light": "#D9DAD5", "dark": "#2C3035"}
VIZ_INK = {"light": "#16181B", "dark": "#ECEBE7"}
VIZ_MUTED = {"light": "#565A61", "dark": "#A2A6AD"}


def luminance(hexc: str) -> float:
    def ch(c):
        c = c / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# v3.3: a live light/dark switch (OS setting, no rerun) must not leave the charts in the other theme's ink. Streamlit sets
# `color-scheme` on the app root from the ACTIVE theme, so CSS light-dark() follows the theme the viewer sees (OS switch or settings menu) without a
# rerun. Each theme-paired chart colour (light value, dark value) is re-mapped in the rendered SVG by its fill/stroke attribute, whichever theme the
# chart was drawn in; colours shared by both themes (CHART_COLORS, VIZ_NOMINAL, VIZ_HIGHLIGHT, VIZ_NEUTRAL, white) need no mapping. Browsers without
# light-dark() ignore these rules and keep the colours of the render theme (the v3.2 behaviour).
STREAMLIT_CHART_GREY = "#808495"  # axis/legend text of the Streamlit-themed charts (length chart, visual arc) in light mode, Streamlit 1.64: 3.46:1
LABEL_ON = {"k": "#16181B", "w": "#FFFFFF"}  # cutting-grammar cell labels (viz.text_on): ink or white, per theme (viz.GRAMMAR_LABEL_LAYERS)


def theme_pairs() -> list[tuple[str, str]]:
    """Every (light, dark) colour pair of the v3.2 charts and their HTML legends, without duplicates; no colour belongs to two pairs."""
    pairs = list(zip(VIZ_RAMP["light"], VIZ_RAMP["dark"], strict=True)) + list(zip(VIZ_GREYS["light"], VIZ_GREYS["dark"], strict=True))
    pairs += [(d["light"], d["dark"]) for d in (VIZ_MISSING, VIZ_GROUND, VIZ_SURFACE, VIZ_LINE, VIZ_INK, VIZ_MUTED, ACCENT)]
    return list(dict.fromkeys(pairs))


def viz_theme_css() -> str:
    """The light-dark() re-mapping rules (see above): SVG fill / stroke / stop-color of every Vega-Lite chart, the swatches of the HTML legends, the
    cutting-grammar label layers, and the axis labels of the Streamlit-themed charts (A8: their own grey is 3.46:1 on the light ground)."""
    out = []
    ch = '[data-testid="stVegaLiteChart"]'
    for lt, dk in theme_pairs():
        ld = f"light-dark({lt},{dk})"
        for attr in ("fill", "stroke", "stop-color"):
            out.append(f'{ch} [{attr}="{lt}" i],{ch} [{attr}="{dk}" i]{{{attr}:{ld} !important}}')
        out.append(f".cgt-vsw.cgt-c-{lt[1:].lower()},.cgt-vsw.cgt-c-{dk[1:].lower()}{{background:{ld} !important}}")
    for a in "kw":
        for b in "kw":
            out.append(f"{ch} .cgt_lab_{a}{b}_marks text{{fill:light-dark({LABEL_ON[a]},{LABEL_ON[b]}) !important}}")
    sel = ",".join(f'{ch} .{r} text[fill="{STREAMLIT_CHART_GREY}" i]' for r in ("role-axis-label", "role-axis-title", "role-legend-label", "role-legend-title"))
    out.append(f'{sel}{{fill:light-dark({VIZ_MUTED["light"]},{VIZ_MUTED["dark"]}) !important}}')
    return "\n".join(out)


MODEL_SLOTS = ("size", "lens", "angle", "framing", "light", "comp", "move")  # trained models (static 4B adapter; movement geometry model)
HEURISTIC_SLOTS = ("key", "DoF", "people", "sound", "len", "palette")  # rules and detectors (core.SLOT_HELP)
STYLE_MARKER = "/* cgt-style v3 */"
# v3.1: Streamlit draws captions in the text colour at opacity 0.6 (light: 4.55:1). Captions now use the muted ink token at full opacity:
# light #565A61 (6.5:1 on the ground, 6.0:1 on the sidebar), dark #A2A6AD (7.6:1 on the ground). Measured values: docs/demo_spec.md v3.1.
CAPTION_COLOR = {t: f'[data-testid="stCaptionContainer"]{{opacity:1;color:{c}}}[data-testid="stCaptionContainer"] p{{color:{c}}}'
                 for t, c in (("light", "#565A61"), ("dark", "#A2A6AD"))}

# Emoji: only what renders as an emoji is removed. Code points of Unicode Extended_Pictographic (15.1, produced with the `regex`
# package's \p{Extended_Pictographic} and pinned by tests/test_demo_v31.py), regional indicators, skin-tone modifiers, emoji tag characters, the
# keycap mark U+20E3, variation selector 16 (U+FE0F) and a zero-width joiner next to any of them. Kept: everything else, in particular the arrows block
# U+2190-21FF (also its six Extended_Pictographic members, which are text-presentation arrows; a following U+FE0F is dropped so they stay text),
# math, dashes, quotes, check marks and stars. The copyright, registered and trade-mark signs (Extended_Pictographic) become "(c)", "(R)", "(TM)":
# no information is lost and the rendered text stays free of pictographs. No caveat or banner contains any of these code points (pinned by tests).
# Written as numbers so this source file itself stays free of them.
EXT_PICT = ((0xA9, 0xA9), (0xAE, 0xAE), (0x203C, 0x203C), (0x2049, 0x2049), (0x2122, 0x2122), (0x2139, 0x2139), (0x2194, 0x2199), (0x21A9, 0x21AA),
            (0x231A, 0x231B), (0x2328, 0x2328), (0x23CF, 0x23CF), (0x23E9, 0x23F3), (0x23F8, 0x23FA), (0x24C2, 0x24C2), (0x25AA, 0x25AB), (0x25B6, 0x25B6),
            (0x25C0, 0x25C0), (0x25FB, 0x25FE), (0x2600, 0x2604), (0x260E, 0x260E), (0x2611, 0x2611), (0x2614, 0x2615), (0x2618, 0x2618), (0x261D, 0x261D),
            (0x2620, 0x2620), (0x2622, 0x2623), (0x2626, 0x2626), (0x262A, 0x262A), (0x262E, 0x262F), (0x2638, 0x263A), (0x2640, 0x2640), (0x2642, 0x2642),
            (0x2648, 0x2653), (0x265F, 0x2660), (0x2663, 0x2663), (0x2665, 0x2666), (0x2668, 0x2668), (0x267B, 0x267B), (0x267E, 0x267F), (0x2692, 0x2697),
            (0x2699, 0x2699), (0x269B, 0x269C), (0x26A0, 0x26A1), (0x26A7, 0x26A7), (0x26AA, 0x26AB), (0x26B0, 0x26B1), (0x26BD, 0x26BE), (0x26C4, 0x26C5),
            (0x26C8, 0x26C8), (0x26CE, 0x26CF), (0x26D1, 0x26D1), (0x26D3, 0x26D4), (0x26E9, 0x26EA), (0x26F0, 0x26F5), (0x26F7, 0x26FA), (0x26FD, 0x26FD),
            (0x2702, 0x2702), (0x2705, 0x2705), (0x2708, 0x270D), (0x270F, 0x270F), (0x2712, 0x2712), (0x2714, 0x2714), (0x2716, 0x2716), (0x271D, 0x271D),
            (0x2721, 0x2721), (0x2728, 0x2728), (0x2733, 0x2734), (0x2744, 0x2744), (0x2747, 0x2747), (0x274C, 0x274C), (0x274E, 0x274E), (0x2753, 0x2755),
            (0x2757, 0x2757), (0x2763, 0x2764), (0x2795, 0x2797), (0x27A1, 0x27A1), (0x27B0, 0x27B0), (0x27BF, 0x27BF), (0x2934, 0x2935), (0x2B05, 0x2B07),
            (0x2B1B, 0x2B1C), (0x2B50, 0x2B50), (0x2B55, 0x2B55), (0x3030, 0x3030), (0x303D, 0x303D), (0x3297, 0x3297), (0x3299, 0x3299),
            (0x1F004, 0x1F004), (0x1F02C, 0x1F02F), (0x1F094, 0x1F09F), (0x1F0AF, 0x1F0B0), (0x1F0C0, 0x1F0C0), (0x1F0CF, 0x1F0D0), (0x1F0F6, 0x1F0FF),
            (0x1F170, 0x1F171), (0x1F17E, 0x1F17F), (0x1F18E, 0x1F18E), (0x1F191, 0x1F19A), (0x1F1AF, 0x1F1E5), (0x1F201, 0x1F20F), (0x1F21A, 0x1F21A),
            (0x1F22F, 0x1F22F), (0x1F232, 0x1F23A), (0x1F23C, 0x1F23F), (0x1F249, 0x1F25F), (0x1F266, 0x1F321), (0x1F324, 0x1F393), (0x1F396, 0x1F397),
            (0x1F399, 0x1F39B), (0x1F39E, 0x1F3F0), (0x1F3F3, 0x1F3F5), (0x1F3F7, 0x1F3FA), (0x1F400, 0x1F4FD), (0x1F4FF, 0x1F53D), (0x1F549, 0x1F54E),
            (0x1F550, 0x1F567), (0x1F56F, 0x1F570), (0x1F573, 0x1F57A), (0x1F587, 0x1F587), (0x1F58A, 0x1F58D), (0x1F590, 0x1F590), (0x1F595, 0x1F596),
            (0x1F5A4, 0x1F5A5), (0x1F5A8, 0x1F5A8), (0x1F5B1, 0x1F5B2), (0x1F5BC, 0x1F5BC), (0x1F5C2, 0x1F5C4), (0x1F5D1, 0x1F5D3), (0x1F5DC, 0x1F5DE),
            (0x1F5E1, 0x1F5E1), (0x1F5E3, 0x1F5E3), (0x1F5E8, 0x1F5E8), (0x1F5EF, 0x1F5EF), (0x1F5F3, 0x1F5F3), (0x1F5FA, 0x1F64F), (0x1F680, 0x1F6C5),
            (0x1F6CB, 0x1F6D2), (0x1F6D5, 0x1F6E5), (0x1F6E9, 0x1F6E9), (0x1F6EB, 0x1F6F0), (0x1F6F3, 0x1F6FF), (0x1F7DC, 0x1F7F0), (0x1F80C, 0x1F80F),
            (0x1F848, 0x1F84F), (0x1F85A, 0x1F85F), (0x1F888, 0x1F88F), (0x1F8AE, 0x1F8AF), (0x1F8BC, 0x1F8BF), (0x1F8C2, 0x1F8CF), (0x1F8D9, 0x1F8FF),
            (0x1F90C, 0x1F93A), (0x1F93C, 0x1F945), (0x1F947, 0x1F9FF), (0x1FA58, 0x1FA5F), (0x1FA6E, 0x1FAFF), (0x1FC00, 0x1FFFD))
KEPT_TEXT = ((0x2190, 0x21FF),)  # arrows: text presentation, never removed
EMOJI_PARTS = ((0x1F1E6, 0x1F1FF), (0x1F3FB, 0x1F3FF), (0xE0020, 0xE007F), (0x20E3, 0x20E3), (0xFE0F, 0xFE0F))  # regional indicators, skin tones, tags, keycap, VS16


def _minus(ranges, cut):
    out = []
    for a, b in ranges:
        for c, d in cut:
            if b < c or a > d:
                continue
            if a < c:
                out.append((a, c - 1))
            a = d + 1
        if a <= b:
            out.append((a, b))
    return tuple(out)


def _cls(ranges) -> str:
    return "[" + "".join(f"\\U{a:08x}-\\U{b:08x}" for a, b in ranges) + "]"


EMOJI_RANGES = _minus(EXT_PICT, KEPT_TEXT) + EMOJI_PARTS  # every code point strip_emoji removes (or, for (c)/(R)/(TM), replaces)
EMOJI = re.compile(_cls(EMOJI_RANGES))
_ZWJ = re.compile(f"\u200d(?={_cls(EMOJI_RANGES)})|(?<={_cls(EMOJI_RANGES)})\u200d")
_TEXT_FOR = {0xA9: "(c)", 0xAE: "(R)", 0x2122: "(TM)"}


def strip_emoji(text: str) -> str:
    """Rendered text without emoji (see EMOJI_RANGES above; arrows and other text symbols are kept; (c)/(R)/(TM) for the three signs). Used on file text
    the app shows (reports, paper, essay), never on a caveat, banner or download."""
    return EMOJI.sub(lambda m: _TEXT_FOR.get(ord(m.group()), ""), _ZWJ.sub("", text))


def parse_row(row) -> list[dict]:
    """Slots of a token row '[size=MCU | lens=medium* | ... ]' in order: {slot, value, low (keyframes disagree: '*'), missing ('?' or 'n/a'), source}.
    The value keeps its literal text (including '*'). An unparsable row gives []: the raw row is always shown next to the chips anyway."""
    if not isinstance(row, str):
        return []
    s = row.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return []
    out = []
    for part in s[1:-1].split("|"):
        slot, sep, value = part.strip().partition("=")
        if not sep or not slot:
            return []
        value = value.strip()
        out.append({"slot": slot, "value": value, "low": value.endswith("*"), "missing": value.rstrip("*") in ("?", "n/a"),
                    "source": "model" if slot in MODEL_SLOTS else "heuristic" if slot in HEURISTIC_SLOTS else "other"})
    return out


def chips_html(row, size: str = "md") -> str:
    """One token row as labelled chips (slot label above the value), model slots teal-tinted, heuristic slots neutral with a dashed edge. '' if unparsable."""
    slots = parse_row(row)
    if not slots:
        return ""
    chips = []
    for t in slots:
        val = html.escape(t["value"].rstrip("*")) + ('<span class="cgt-star" title="start, mid and end keyframes disagree">*</span>' if t["low"] else "")
        cls = f"cgt-chip cgt-{t['source']}" + (" cgt-low" if t["low"] else "") + (" cgt-missing" if t["missing"] else "")
        tip = "model prediction" if t["source"] == "model" else "heuristic" if t["source"] == "heuristic" else ""
        chips.append(f'<span class="{cls}" role="listitem" title="{tip}"><span class="cgt-slot">{html.escape(t["slot"])}</span>'
                     f'<span class="cgt-val">{val}</span></span>')
    return f'<div class="cgt-row cgt-{size}" role="list" aria-label="token row">{"".join(chips)}</div>'


def chip_key_html() -> str:
    """The provenance key of the chips (shown once per page above the first chip row)."""
    return ('<div class="cgt-key"><span class="cgt-keyitem"><span class="cgt-chip cgt-model cgt-mini"><span class="cgt-val">MCU</span></span> model prediction</span>'
            '<span class="cgt-keyitem"><span class="cgt-chip cgt-heuristic cgt-mini"><span class="cgt-val">low-key</span></span> heuristic</span>'
            '<span class="cgt-keyitem"><span class="cgt-chip cgt-model cgt-low cgt-mini"><span class="cgt-val">wide<span class="cgt-star">*</span></span></span> '
            'keyframes disagree</span></div>')


def brand_html() -> str:
    return ('<div class="cgt-brand"><div class="cgt-mark" aria-hidden="true"><i></i><i></i><i></i></div>'
            '<div><div class="cgt-word">Cinematic Grammar Tokenizer</div><div class="cgt-sub">A film as one token row per shot</div></div></div>')


def verdict_html(cls: str) -> str:
    """Essay verdict class as a text label (replaces the old emoji marks)."""
    return f'<span class="cgt-verdict cgt-verdict-{html.escape(cls.replace(" ", "-"))}">{html.escape(cls)}</span>'


# ---- v3.1 Compare and Reports (docs/design/revamp_plan.md section 7) ------------------------------------------------------------------------------

def compare_pair_html(a: str, b: str | None) -> str:
    """The two-film picker summary of Compare, which is also the legend of its charts: film A and film B with the series colour each gets
    (the Compare charts use the domain sorted(labels) with range CHART_COLORS, so the swatches follow the sorted labels). b None: the empty state."""
    order = sorted([x for x in (a, b) if x is not None])
    def side(tag: str, label: str | None) -> str:
        if label is None:
            return (f'<div class="cgt-pick cgt-pick-empty"><span class="cgt-pick-tag">{tag}</span>'
                    '<span class="cgt-pick-film">No second film in this build</span></div>')
        sw = CHART_COLORS[order.index(label)]
        return (f'<div class="cgt-pick"><span class="cgt-pick-tag">{tag}</span><span class="cgt-pick-film">'
                f'<span class="cgt-swatch" style="background:{sw}"></span>{html.escape(label)}</span></div>')
    return f'<div class="cgt-pair" role="group" aria-label="films compared">{side("Film A", a)}{side("Film B", b)}</div>'


# Document library of the Reports page: (title, one line on what it is, kind) per path the app can list. Presentational only: which paths are listed
# is decided by core.paper_files / core.report_list exactly as before; a path without an entry falls back to its file name.
REPORT_INFO = {
    "docs/writeup/paper_public.html": ("Project paper", "The public write-up: the pipeline, the ShotBench results and the film analyses.", "paper"),
    "docs/writeup/paper_public.pdf": ("Project paper (PDF)", "The same paper as a PDF file.", "paper"),
    "data/reports/paper_full.html": ("Project paper, local build", "The full local paper, including the owned film and the user's clip.", "paper"),
    "data/reports/paper_full.pdf": ("Project paper, local build (PDF)", "The full local paper as a PDF file.", "paper"),
    "docs/writeup/draft.md": ("Paper draft", "The Markdown draft of the paper (shown when no HTML paper is built).", "paper"),
    "docs/phase6_report.md": ("Phase 6: analyses, demo, write-up", "What the last phase built and what it dropped, with the reasons.", "project"),
    "results/phase4/PHASE4_FINAL_REPORT.md": ("Phase 4: camera movement", "Movement models on ShotBench movement and the CameraBench public test.", "evaluation"),
    "results/phase3b/shotbench_summary.md": ("Phase 3B: ShotBench result", "The recorded ShotBench score of the 2B LoRA model, with its limits.", "evaluation"),
    "results/phase6/visual_arc_night_of_the_living_dead.md": ("Visual arc: Night of the Living Dead",
                                                              "How rhythm, scale, light and people change over the film's running time.", "film analysis"),
    "results/phase6/similar_shots_night_of_the_living_dead.md": ("Similar shots: Night of the Living Dead",
                                                                 "Which shots have the most similar token rows, and how far apart they are.", "film analysis"),
    "docs/project_log.md": ("Project log", "The running log of the project (local build only).", "project"),
    "results/p3m_matched/shotbench_matched_v1.md": ("Matched-protocol ShotBench result", "Internal comparison under the matched protocol; never published.",
                                                    "internal"),
}
REPORT_GROUPS = (("paper", "Paper"), ("evaluation", "Evaluations"), ("film analysis", "Film analyses"), ("project", "Project"),
                 ("internal", "Internal (local build)"), ("local", "Local clips"))


def report_info(rel: str) -> tuple[str, str, str]:
    if rel in REPORT_INFO:
        return REPORT_INFO[rel]
    if rel.startswith("data/processed/") and rel.endswith("_report.md"):
        return ("Clip report", "The local report of a film or clip added on this machine.", "local")
    return (rel.rsplit("/", 1)[-1], "", "project")


def report_library_html(paths: list[str], current: str | None = None) -> str:
    """The Reports page's library: every listed document (paper first, then the phase reports, in the order given) grouped by kind, each with its
    title, one line on what it is and its source path; `current` (the report open below) is marked. '' when nothing is listed."""
    if not paths:
        return ""
    by = {}
    for rel in paths:
        by.setdefault(report_info(rel)[2], []).append(rel)
    groups = []
    for kind, name in REPORT_GROUPS:
        items = []
        for rel in by.get(kind, []):
            title, what, _ = report_info(rel)
            cur = rel == current
            items.append(f'<li class="cgt-doc{" cgt-doc-open" if cur else ""}"><span class="cgt-doc-title">{html.escape(title)}</span>'
                         + (f'<span class="cgt-doc-what">{html.escape(what)}</span>' if what else "")
                         + f'<code class="cgt-doc-path">{html.escape(rel)}</code>' + ('<span class="cgt-doc-flag">open below</span>' if cur else "") + "</li>")
        if items:
            groups.append(f'<section class="cgt-docgroup"><div class="cgt-docgroup-name">{name}</div><ul class="cgt-docs">{"".join(items)}</ul></section>')
    return f'<div class="cgt-library">{"".join(groups)}</div>'


def css(theme: str = "light") -> str:
    a = ACCENT.get(theme, ACCENT["light"])
    return STYLE_MARKER + """
[data-testid="stMainBlockContainer"]{max-width:1240px;padding-top:3.2rem;padding-bottom:5rem}
h1,h2,h3,h4{letter-spacing:-0.012em;text-wrap:balance}
[data-testid="stMain"] h2{margin-bottom:.25rem}
::selection{background:ACC33}
/* sidebar: brand and page radio drawn as nav rows (the radio, its options and key stay: guard surface) */
.cgt-brand{display:flex;gap:12px;align-items:center;margin:0 0 6px}
.cgt-mark{display:grid;grid-template-columns:repeat(3,6px);gap:3px;padding:7px;border-radius:6px;background:ACC}
.cgt-mark i{display:block;height:18px;border-radius:1px;background:MARKINK}
.cgt-mark i:nth-child(2){height:12px;align-self:end}
.cgt-word{font-family:"Newsreader",serif;font-weight:600;font-size:1.12rem;line-height:1.15}
.cgt-sub{font-size:.78rem;opacity:.72;margin-top:2px}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(> [data-testid="stRadio"]),[data-testid="stSidebar"] [data-testid="stRadio"]{width:100%}
[data-testid="stSidebar"] [data-testid="stRadioGroup"]{gap:2px;width:100%}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div{width:100%}
[data-testid="stSidebar"] [data-testid="stRadioOption"]{width:100%;box-sizing:border-box;margin:0;padding:7px 10px;border-radius:6px;transition:background-color .15s ease-out}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child{display:none}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover{background:rgba(127,127,127,.11)}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]{background:rgba(127,127,127,.13)}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p{font-weight:600;color:ACC}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:focus-visible){outline:2px solid ACC;outline-offset:1px}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"] p{font-size:.78rem;opacity:.72}
/* callouts: required caveats stay full-text and visible; calmer shape (colours: theme yellow keys) */
[data-testid="stAlertContainer"]{border-radius:10px;padding:14px 18px}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]){border:1px solid rgba(154,107,0,.30)}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]){border:1px solid rgba(190,40,40,.35)}
[data-testid="stAlertContentWarning"] p{font-size:.9rem;line-height:1.55}
/* surfaces */
[data-testid="stMetric"]{padding:14px 16px}
[data-testid="stMetricLabel"] p{font-size:.8rem;opacity:.75}
[data-testid="stMetricValue"]{font-family:"Newsreader",serif}
[class*="st-key-shotcard_"]{border-radius:10px}
[class*="st-key-shotcard_"] > div:first-child strong{font-family:"Newsreader",serif;font-size:1.2rem;font-weight:600;margin-right:2px}
[class*="st-key-shotcard_"] [data-testid="stImage"] img{border-radius:4px}
[data-testid="stCode"] pre{border-radius:6px}
[data-testid="stCaptionContainer"]{line-height:1.5}
/* token chips */
.cgt-row{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 6px}
.cgt-chip{display:inline-flex;flex-direction:column;gap:1px;padding:5px 9px 6px;border-radius:6px;line-height:1.2;border:1px solid transparent}
.cgt-model{background:rgba(31,143,136,.16);border-color:rgba(31,143,136,.30)}
.cgt-heuristic{background:rgba(127,127,127,.09);border:1px dashed rgba(127,127,127,.55)}
.cgt-other{background:rgba(127,127,127,.09)}
.cgt-slot{font-size:.75rem;opacity:.78;letter-spacing:.01em}
.cgt-val{font-family:"IBM Plex Mono",monospace;font-size:.86rem;font-weight:500}
.cgt-lg .cgt-chip{padding:7px 12px 8px}
.cgt-lg .cgt-val{font-size:1rem}
.cgt-low .cgt-val{text-decoration:underline dotted;text-underline-offset:3px;text-decoration-thickness:1.5px}
.cgt-star{margin-left:1px}
.cgt-missing .cgt-val{opacity:.65}
.cgt-key{display:flex;flex-wrap:wrap;align-items:center;gap:6px 18px;font-size:.78rem;margin:2px 0 10px}
.cgt-keyitem{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.cgt-mini{flex-direction:row;padding:1px 6px}
.cgt-mini .cgt-val{font-size:.72rem}
.cgt-verdict{display:inline-block;font-size:.75rem;font-weight:600;padding:2px 8px;border-radius:999px;background:rgba(127,127,127,.14)}
.cgt-verdict-agree{background:rgba(31,143,136,.18)}
.cgt-verdict-disagree{background:rgba(196,80,106,.20)}
.cgt-verdict-partly{background:rgba(194,122,26,.20)}
[class*="st-key-homelink_"]{padding:14px 18px;gap:0;transition:border-color .15s ease-out}
[class*="st-key-homelink_"]:hover{border-color:ACC}
[class*="st-key-homelink_"] [data-testid="stButton"] button{padding:0;min-height:0}
[class*="st-key-homelink_"] [data-testid="stButton"] button p{font-weight:600;font-size:1rem;color:ACC}
[class*="st-key-filmcard_"]{padding:18px}
.st-key-ov_metrics [data-testid="stHorizontalBlock"]{flex-wrap:wrap}
.st-key-hero [data-testid="stHorizontalBlock"]{flex-wrap:wrap;row-gap:12px}
.st-key-hero [data-testid="stColumn"]{min-width:min(100%,340px)}
@media (max-width:640px){.st-key-ov_metrics [data-testid="stColumn"]{flex:1 1 calc(50% - .5rem);min-width:calc(50% - .5rem)}}
[class*="st-key-filmcard_"] strong{font-family:"Newsreader",serif;font-size:1.25rem;font-weight:600}
.cgt-hero-cap{font-size:.8rem;opacity:.75;margin:4px 0 12px}
@media (max-width:640px){
  [data-testid="stMainBlockContainer"]{padding-left:16px;padding-right:16px;padding-top:2.4rem}
  .cgt-val{font-size:.8rem}
  .cgt-lg .cgt-val{font-size:.9rem}
  [class*="st-key-shotcard_"] [data-testid="stHorizontalBlock"]{flex-wrap:nowrap;gap:6px}
  [class*="st-key-shotcard_"] [data-testid="stColumn"]{min-width:0;flex:1 1 0}
}
/* v3.1: wide content scrolls inside its own box, never the page or the whole document */
[data-testid="stMarkdownContainer"] table{display:block;max-width:100%;overflow-x:auto;border-collapse:collapse}
[data-testid="stMarkdownContainer"] pre{max-width:100%;overflow-x:auto}
[data-testid="stMarkdownContainer"] img{max-width:100%;height:auto}
[data-testid="stCaptionContainer"]{overflow-wrap:anywhere}
CAPCOLOR
/* v3.1 Compare: picker summary and aligned chart panels */
.cgt-pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:8px 12px;margin:6px 0 14px}
.cgt-pick{display:flex;align-items:baseline;gap:12px;padding:11px 14px;border:1px solid rgba(127,127,127,.30);border-radius:10px}
.cgt-pick-empty{border-style:dashed}
.cgt-pick-empty .cgt-pick-film{font-family:inherit;font-size:.95rem;font-weight:400}
.cgt-pick-tag{flex:0 0 auto;font-size:.8rem;font-weight:600}
.cgt-pick-film{display:flex;align-items:center;gap:8px;font-family:"Newsreader",serif;font-size:1.15rem;font-weight:600;line-height:1.25}
.cgt-swatch{flex:0 0 auto;width:18px;height:4px;border-radius:2px}
[class*="st-key-cmp_"]{gap:4px}
[class*="st-key-cmp_"] [data-testid="stMarkdownContainer"] p{min-height:2.7em;margin:0;line-height:1.35}
@media (min-width:641px) and (max-width:1100px){[class*="st-key-cmp_"] [data-testid="stMarkdownContainer"] p{min-height:4.05em}}
/* v3.1 Reports: document library (an index, not cards) and a readable measure */
.cgt-library{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:8px 32px;margin:4px 0 12px}
.cgt-docgroup-name{font-size:.9rem;font-weight:600;margin:12px 0 2px}
.cgt-docs{list-style:none;margin:0;padding:0}
[data-testid="stMarkdownContainer"] .cgt-docs li.cgt-doc{display:flex;flex-direction:column;gap:2px;margin:0;padding:10px 0 11px;border-top:1px solid rgba(127,127,127,.28)}
.cgt-doc-title{font-family:"Newsreader",serif;font-size:1.08rem;font-weight:600;line-height:1.25}
.cgt-doc-open .cgt-doc-title{color:ACC}
.cgt-doc-what{font-size:.88rem;line-height:1.45}
[data-testid="stMarkdownContainer"] code.cgt-doc-path{align-self:flex-start;font-size:.75rem;padding:0;background:none;overflow-wrap:anywhere}
.cgt-doc-flag{font-size:.75rem;font-weight:600;color:ACC}
.st-key-rep_doc,.st-key-paper_doc{padding:8px clamp(14px,4vw,40px) 24px}
.st-key-rep_doc [data-testid="stMarkdownContainer"],.st-key-paper_doc [data-testid="stMarkdownContainer"]{font-size:1rem;line-height:1.65}
.st-key-rep_doc [data-testid="stMarkdownContainer"] :is(p,ul,ol,h1,h2,h3,h4,blockquote),.st-key-paper_doc [data-testid="stMarkdownContainer"] :is(p,ul,ol,h1,h2,h3,h4,blockquote){max-width:70ch}
.st-key-rep_doc td,.st-key-rep_doc th,.st-key-paper_doc td,.st-key-paper_doc th{min-width:8.5em;vertical-align:top;font-variant-numeric:tabular-nums}
.st-key-rep_doc [data-testid="stMarkdownContainer"] h1,.st-key-paper_doc [data-testid="stMarkdownContainer"] h1{font-size:1.75rem;line-height:1.2;padding-top:.6rem}
.st-key-rep_doc [data-testid="stMarkdownContainer"] h2,.st-key-paper_doc [data-testid="stMarkdownContainer"] h2{font-size:1.35rem;padding-top:1.1rem}
.st-key-rep_doc [data-testid="stMarkdownContainer"] h3,.st-key-paper_doc [data-testid="stMarkdownContainer"] h3{font-size:1.1rem;padding-top:.8rem}
.st-key-rep_doc [data-testid="stMarkdownContainer"] table,.st-key-paper_doc [data-testid="stMarkdownContainer"] table{font-size:.85rem;line-height:1.4}
.st-key-rep_doc [data-testid="stMarkdownContainer"] p,.st-key-rep_doc [data-testid="stMarkdownContainer"] li{overflow-wrap:anywhere}
/* v3.2 charts: section heads, chart titles, HTML legends, the fingerprint, the hero barcode (app/demo/viz.py) */
.cgt-vsec{margin:2.2rem 0 .2rem;padding-top:1.1rem;border-top:1px solid rgba(127,127,127,.28)}
.cgt-vkick{font-size:.78rem;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:ACC;margin:0 0 4px}
.cgt-vsec h3{font-family:"Newsreader",serif;font-size:1.55rem;font-weight:600;line-height:1.2;margin:0 0 4px;padding:0}
.cgt-vsec p:last-child,.cgt-vtitle p,.cgt-vzoom p:last-child{font-size:.92rem;line-height:1.5;max-width:72ch;margin:0;color:MUTEDC}
.cgt-vtitle h4{font-family:"Newsreader",serif;font-size:1.22rem;font-weight:600;line-height:1.25;margin:0 0 3px;padding:0}
.cgt-vhero{margin:6px 0 2px}
[data-testid="stMarkdownContainer"] .cgt-vtitle p,[data-testid="stMarkdownContainer"] .cgt-vsec p:last-child{margin:0 0 4px}
.cgt-vlede{font-size:1rem;line-height:1.55;max-width:78ch;margin:0}
.cgt-vzoom{margin:1.6rem 0 .3rem;padding-top:1rem;border-top:1px solid rgba(127,127,127,.28)}
.cgt-vnote,.cgt-vcap{font-size:.78rem;line-height:1.5;color:MUTEDC;margin:2px 0 0;max-width:90ch}
.cgt-vlegend{display:flex;flex-direction:column;gap:5px;margin:2px 0 0}
.cgt-vlrow{display:flex;flex-wrap:wrap;align-items:center;gap:3px 12px;font-size:.78rem}
.cgt-vlane{min-width:58px;justify-content:center}
.cgt-vli{display:inline-flex;align-items:center;white-space:nowrap;max-width:100%}
.cgt-vlnote .cgt-vli,.cgt-vli-long{white-space:normal}
.cgt-vsw{display:inline-block;flex:0 0 auto;width:11px;height:11px;border-radius:2px;margin-right:5px}
.cgt-vsw + .cgt-vsw{margin-left:-2px}
.cgt-vpin{flex:0 0 auto;margin-right:6px;font-size:.72rem;line-height:1;color:INKC}
.cgt-vswings{font-size:.82rem;line-height:1.5;margin:4px 0 0;padding-left:1.2rem}
.cgt-vfgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(176px,1fr));gap:14px 22px}
.cgt-vfhead{display:flex;align-items:center;gap:6px;margin-bottom:4px}
.cgt-vfsrc{font-size:.72rem;color:MUTEDC}
.cgt-vfr{display:grid;grid-template-columns:minmax(58px,max-content) 1fr 32px;align-items:center;gap:6px;font-size:.76rem;line-height:1.55}
.cgt-vfv{font-family:"IBM Plex Mono",monospace;font-size:.74rem;white-space:nowrap}
.cgt-vft{display:flex;height:9px;min-width:0}
.cgt-vfb{display:block;height:9px}
.cgt-vfb:first-child{border-radius:2px 0 0 2px}
.cgt-vfb:last-child{border-radius:0 2px 2px 0}
.cgt-vfp{text-align:right;font-variant-numeric:tabular-nums;color:MUTEDC}
.cgt-vfgrid-wrap .cgt-vlnote{margin-top:12px}
[class*="st-key-viz_"]{gap:.55rem}
.st-key-viz_page .vega-embed{display:flex !important;flex-direction:column-reverse;gap:6px}
.st-key-viz_strip [data-testid="stImage"] img{border-radius:4px 4px 0 0}
@media (max-width:640px){
  .st-key-viz_hero{overflow-x:auto;overscroll-behavior-x:contain}
  .st-key-viz_hero [data-testid="stVegaLiteChart"]{min-width:760px}
  .cgt-vsec h3{font-size:1.35rem}
  .cgt-vli{white-space:normal}
  .cgt-vfgrid{grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:12px 14px}
}
/* v3.2: chart tooltips show the complete token row, wrapped (Streamlit clamps a tooltip value to five lines of 18rem) */
#vg-tooltip-element{max-width:min(40rem,calc(100vw - 24px)) !important}
#vg-tooltip-element table tr td.value{display:table-cell !important;-webkit-line-clamp:none !important;line-clamp:none !important;max-width:min(30rem,calc(100vw - 140px)) !important;white-space:normal !important;overflow-wrap:anywhere !important;text-overflow:clip !important}
/* v3.3: the tooltip never leaves the viewport. vega-tooltip puts it right of the pointer, else left of it, and never clamps; so below
   1240 px it is at most half the viewport wide (it then always fits on one side), and on phones it is a full-width band 12 px from both edges */
@media (max-width:1240px){
  #vg-tooltip-element{max-width:calc(50vw - 40px) !important}
  #vg-tooltip-element table tr td.value{max-width:calc(50vw - 150px) !important}
}
@media (max-width:640px){
  #vg-tooltip-element{left:12px !important;right:12px !important;max-width:none !important;width:auto !important;box-sizing:border-box !important}
  #vg-tooltip-element table tr td.value{max-width:none !important}
}
VIZTHEME
@media (prefers-reduced-motion:reduce){[data-testid="stSidebar"] [data-testid="stRadioOption"]{transition:none}}
""".replace("CAPCOLOR", CAPTION_COLOR.get(theme, "")).replace("VIZTHEME", viz_theme_css()).replace("ACC33", a + "33").replace("ACC", _ld(ACCENT)).replace(
        "MARKINK", "light-dark(#FFFFFF,#0F1113)").replace("MUTEDC", _ld(VIZ_MUTED)).replace("INKC", _ld(VIZ_INK))


def _ld(pair: dict) -> str:
    """A theme-paired colour as CSS light-dark()."""
    return f"light-dark({pair['light']},{pair['dark']})"


def inject() -> None:
    """Apply the style sheet once per run (st.html with only a <style> block adds no layout)."""
    import streamlit as st
    try:
        theme = st.context.theme.type or "light"
    except Exception:  # noqa: BLE001 - no theme info (tests, old clients): light accent
        theme = "light"
    st.html(f"<style>{css(theme)}</style>")
