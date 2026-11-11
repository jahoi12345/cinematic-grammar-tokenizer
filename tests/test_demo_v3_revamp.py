"""Demo v3 revamp: no emoji anywhere (app sources and rendered text of every page in both
modes), token chips keep every slot of the row, the Home example is never a LOCAL ONLY film, the style sheet is injected once and stays presentational.
Synthetic tmp roots and synthetic film ids only; never touches film data."""
import importlib.util
import re
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if not (REPO / "app/demo/style.py").exists():
    pytest.skip("app/demo v3 not on this branch", allow_module_level=True)

from test_demo_guard import CLIP, NOTLD, OWNED, core, run_app
from test_demo_v2_frontend import (
    _w,
    root,  # noqa: F401 - pytest fixture (synthetic registry, overlay clip, reports, essay)
)

# Independent of app/demo/style.py on purpose: Unicode Extended_Pictographic blocks, regional indicators (inside 1F000-1FAFF), skin tones,
# keycap U+20E3 and variation selector U+FE0F.
_RANGES = ((0xA9, 0xA9), (0xAE, 0xAE), (0x203C, 0x203C), (0x2049, 0x2049), (0x20E3, 0x20E3), (0x2122, 0x2122), (0x2139, 0x2139), (0x2194, 0x2199),
           (0x21A9, 0x21AA), (0x231A, 0x231B), (0x2328, 0x2328), (0x2388, 0x2388), (0x23CF, 0x23CF), (0x23E9, 0x23F3), (0x23F8, 0x23FA), (0x24C2, 0x24C2),
           (0x25AA, 0x25AB), (0x25B6, 0x25B6), (0x25C0, 0x25C0), (0x25FB, 0x25FE), (0x2600, 0x27BF), (0x2934, 0x2935), (0x2B05, 0x2B07), (0x2B1B, 0x2B1C),
           (0x2B50, 0x2B50), (0x2B55, 0x2B55), (0x3030, 0x3030), (0x303D, 0x303D), (0x3297, 0x3297), (0x3299, 0x3299), (0xFE0F, 0xFE0F),
           (0x1F000, 0x1FAFF), (0x1FC00, 0x1FFFD))


def emoji_in(text: str) -> list[str]:
    return sorted({f"U+{ord(c):04X}" for c in text if any(a <= ord(c) <= b for a, b in _RANGES)})


def _style():
    spec = importlib.util.spec_from_file_location("demo_style_under_test", REPO / "app/demo/style.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


style = _style()
PAGES = ["Home", "Films", "Film overview", "Shot explorer", "Side by side", "Reports", "About"]
LOCAL_PAGES = ["Tokens vs my essay", "Add your own clip"]


def rendered_text(at) -> str:
    """Every string AppTest holds for the rendered page: each element's protobuf (bodies, labels, options, code, iframe source, badge, help)."""
    text_format = pytest.importorskip("google.protobuf.text_format")  # with streamlit only (the light CPU venv skips, like run_app)
    out = []

    def walk(n):
        p = getattr(n, "proto", None)
        if p is not None:
            out.append(text_format.MessageToString(p, as_utf8=True))  # as_utf8: non-ASCII stays as characters (str() would octal-escape them)
        for c in getattr(n, "children", {}).values():
            walk(c)
    walk(at._tree)
    return "\n".join(out)


# ---- emoji ---------------------------------------------------------------------------------------------------------------------------------------------

def test_no_emoji_in_app_sources():
    hits = {}
    for p in sorted((REPO / "app").rglob("*")):
        if p.is_file() and p.suffix in (".py", ".html", ".js", ".css", ".md", ".toml", ".json", ".txt"):
            found = emoji_in(p.read_text(encoding="utf-8", errors="replace"))
            if found:
                hits[str(p.relative_to(REPO))] = found
    found = emoji_in((REPO / ".streamlit/config.toml").read_text())
    if found:
        hits[".streamlit/config.toml"] = found
    assert hits == {}


def test_emoji_detector_catches_known_glyphs():  # the scan above is only as good as this detector
    for g in ("✅", "❌", "⚠️", "\U0001F3AC", "\U0001F1EB\U0001F1F7", "©", "⭐", "\U0001F44D\U0001F3FD"):
        assert emoji_in(g), g
    for ok in ("[size=MCU* | lens=?]", "1,077 shots · 95.5 min", "– — … é"):
        assert not emoji_in(ok), ok


def _cases():
    cases = [(True, p, {}) for p in PAGES] + [(False, p, {}) for p in PAGES + LOCAL_PAGES]
    cases += [(False, "Film overview", {"ov_film": OWNED}), (False, "Shot explorer", {"ex_film": CLIP})]
    return cases


@pytest.mark.parametrize("public,page,sel", _cases(), ids=lambda v: str(v))
def test_no_emoji_in_rendered_pages(monkeypatch, root, public, page, sel):  # noqa: F811
    # an emoji planted in a report the app renders must not reach the page (stripped at render time; the downloads keep the file bytes)
    _w(root, "docs/phase6_report.md", "# Phase 6 ✅ done ⚠️ PHASESIXMARK © 2026\n")
    at = run_app(monkeypatch, root, public, page, **sel)
    assert emoji_in(rendered_text(at)) == [], (public, page)


def test_essay_verdicts_are_text_labels(monkeypatch, root):  # noqa: F811
    at = run_app(monkeypatch, root, False, "Tokens vs my essay")
    text = rendered_text(at)
    for cls in ("agree", "partly", "disagree", "not-representable"):
        assert f"cgt-verdict-{cls}" in text
    assert emoji_in(text) == []


def test_strip_emoji_keeps_text():
    assert style.strip_emoji("ok ✅ done © 2026 ⚠️!") == "ok  done (c) 2026 !"
    assert style.strip_emoji("[size=MCU* | len=4-8s]") == "[size=MCU* | len=4-8s]"


# ---- token chips ---------------------------------------------------------------------------------------------------------------------------------------

ROW = ("[size=MCU | lens=medium* | angle=low | framing=single | light=daylight/hard | comp=right-heavy | key=mid-key | DoF=shallow | people=1 | "
       "sound=dialogue | len=4-8s | palette=g20 | move=?]")


def test_parse_row_keeps_every_slot_in_order():
    slots = style.parse_row(ROW)
    assert [t["slot"] for t in slots] == ["size", "lens", "angle", "framing", "light", "comp", "key", "DoF", "people", "sound", "len", "palette", "move"]
    assert "[" + " | ".join(f"{t['slot']}={t['value']}" for t in slots) + "]" == ROW  # lossless
    by = {t["slot"]: t for t in slots}
    assert by["lens"]["low"] and not by["size"]["low"] and by["move"]["missing"]
    assert {t["slot"] for t in slots if t["source"] == "model"} == {"size", "lens", "angle", "framing", "light", "comp", "move"}
    assert {t["slot"] for t in slots if t["source"] == "heuristic"} == {"key", "DoF", "people", "sound", "len", "palette"}


@pytest.mark.parametrize("bad", [None, "", "size=MCU", "[size MCU | lens=wide]", 3.0])
def test_parse_row_unparsable_gives_no_chips(bad):
    assert style.parse_row(bad) == [] and style.chips_html(bad) == ""


def test_chips_html_escapes_and_marks():
    h = style.chips_html("[size=<b>x</b>* | key=low-key]")
    assert "<b>x</b>" not in h and "&lt;b&gt;" in h and "cgt-low" in h and "cgt-heuristic" in h and 'class="cgt-' in h


def test_explorer_cards_show_chips_and_keep_the_raw_row(monkeypatch, root):  # noqa: F811
    at = run_app(monkeypatch, root, True, "Shot explorer")
    chips = [m.value for m in at.markdown if 'class="cgt-row' in m.value]
    codes = [c.proto.code_text for c in at.get("code") if c.proto.code_text.startswith("[")]
    assert len(chips) == 8 and len(codes) == 8 + 8 * 5  # chips on the 8 cards; raw rows of cards and of similar shots unchanged


# ---- Home example --------------------------------------------------------------------------------------------------------------------------------------

def _set_rows(root: Path, film: str, qualify: bool) -> None:  # noqa: F811 - the tmp root, not the fixture
    p = root / f"data/processed/{film}_tokens_v1.parquet"
    df = pd.read_parquet(p)
    size = "MCU" if qualify else "EWS"
    df["token_row"] = [f"[size={size} | lens=medium | angle=low | people=1 | film={film} | shot={i}]" for i in range(len(df))]
    df.to_parquet(p)


def test_home_example_is_a_public_film_row_as_chips(monkeypatch, root):  # noqa: F811
    for f in (NOTLD, OWNED, CLIP):
        _set_rows(root, f, True)
    for public in (True, False):
        at = run_app(monkeypatch, root, public, "Home")
        chips = [m.value for m in at.markdown if 'class="cgt-row' in m.value]
        codes = [c.proto.code_text for c in at.get("code")]
        assert len(chips) == 1 and f">{NOTLD}<" in chips[0] and codes == [f"[size=MCU | lens=medium | angle=low | people=1 | film={NOTLD} | shot=0]"]
        text = rendered_text(at)
        assert OWNED not in text.replace("Owned Film X (2000) (LOCAL ONLY)", "") and f"film={CLIP}" not in text
        assert len(at.image) == 1


def test_home_example_never_falls_back_to_a_local_only_film(monkeypatch, root):  # noqa: F811
    _set_rows(root, NOTLD, False)
    for f in (OWNED, CLIP):
        _set_rows(root, f, True)
    at = run_app(monkeypatch, root, False, "Home")
    assert not [m for m in at.markdown if 'class="cgt-row' in m.value] and len(at.image) == 0 and len(at.get("code")) == 0
    assert any(core.CAPTION_PREDICTIONS in w.value for w in at.warning)  # the caveat stays


def test_home_example_rule_needs_one_person_close_shot_without_star(monkeypatch, root):  # noqa: F811
    p = root / f"data/processed/{NOTLD}_tokens_v1.parquet"
    df = pd.read_parquet(p)
    rows = ["[size=MCU* | people=1]", "[size=CU | people=2]", "[size=WS | people=1]", "[size=CU | people=1]"] + ["[size=EWS | people=0]"] * (len(df) - 4)
    df["token_row"] = rows
    df.to_parquet(p)
    at = run_app(monkeypatch, root, True, "Home")
    assert [c.proto.code_text for c in at.get("code")] == ["[size=CU | people=1]"]


# ---- style sheet ---------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("public", [True, False])
def test_style_injected_once_and_presentational(monkeypatch, root, public):  # noqa: F811
    at = run_app(monkeypatch, root, public, "Film overview")
    text = rendered_text(at)
    assert text.count(style.STYLE_MARKER) == 1
    css = style.css("light") + style.css("dark")
    assert "st-emotion-cache" not in css and "display:none" in css
    hidden = re.findall(r"([^{}]+)\{[^}]*display:none", css)
    assert all('[data-testid="stRadioOption"] > div > div:first-child' in h for h in hidden)  # only the radio dot is ever hidden, never an alert/caveat
    assert "stAlert" not in " ".join(hidden) and "visibility:hidden" not in css and not re.search(r"(?<![-\w])(max-)?height:0", css)


def test_sidebar_state_keeps_page_list_on_desktop():  # never start collapsed; "auto" = expanded above 768 px (checked in Chrome at 800 px)
    src = (REPO / "app/demo/app.py").read_text()
    states = re.findall(r'initial_sidebar_state="(\w+)"', src)
    assert states and set(states) <= {"auto", "expanded"}
