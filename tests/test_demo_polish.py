"""Demo v2.1 polish: every page still shows its caveat VISIBLY (a top-level warning or text, never only inside a collapsed
expander), every page has exactly one page title (no repeated app title), LOCAL ONLY banners stay visible, and the theme file is valid. Synthetic tmp roots only."""
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if not (REPO / "app/demo/core.py").exists():
    pytest.skip("app/demo not on this branch", allow_module_level=True)

from test_demo_guard import CLIP, OWNED, core, run_app
from test_demo_v2_frontend import (
    root,  # noqa: F401 - pytest fixture (synthetic registry, overlay clip, reports, essay)
)

CUT_WARNING = "The default shot-cut threshold is UNCHECKED for a new input"


def _visible(at) -> list[tuple[str, str]]:
    """(element type, text) of every element under the main area or sidebar that is NOT inside an expander (expanders start collapsed unless told otherwise)."""
    out: list[tuple[str, str]] = []

    def walk(node, hidden: bool):
        kind = type(node).__name__
        hidden = hidden or kind == "Expander"
        if not hidden:
            for attr in ("value", "body", "label"):
                try:
                    v = getattr(node, attr, None)
                except Exception:  # noqa: BLE001 - some element types raise for attributes they do not have
                    v = None
                if isinstance(v, str):
                    out.append((kind, v))
        for c in getattr(node, "children", {}).values():
            walk(c, hidden)
    walk(at._tree, False)
    return out


def _warnings(at) -> list[str]:
    return [t for k, t in _visible(at) if k == "Warning"]


def _has(at, text: str, kind: str | None = None) -> bool:
    return any(text.replace("`", "") in t.replace("`", "") for k, t in _visible(at) if kind is None or k == kind)


@pytest.mark.parametrize("public", [True, False])
@pytest.mark.parametrize("page,caveats", [("Home", [core.CAPTION_PREDICTIONS]), ("Film overview", [core.CAPTION_PREDICTIONS]),
                                          ("Shot explorer", [core.CAPTION_PREDICTIONS]), ("Side by side", [core.CAPTION_SHAPES]),
                                          ("About", [core.CAPTION_PREDICTIONS])])
def test_caveat_is_a_visible_warning_on_every_prediction_page(monkeypatch, root, public, page, caveats):  # noqa: F811
    at = run_app(monkeypatch, root, public, page)
    for cap in caveats:
        assert any(cap in w for w in _warnings(at)), (page, cap[:60])


@pytest.mark.parametrize("public", [True, False])
def test_overview_movement_and_interval_caveats_visible(monkeypatch, root, public):  # noqa: F811
    at = run_app(monkeypatch, root, public, "Film overview")
    assert any("INFLATED" in w for w in _warnings(at))  # the movement caption stays a warning
    for cap in (core.CAPTION_INTERVALS, core.CAPTION_SHAPES):
        assert _has(at, cap[:80])


@pytest.mark.parametrize("public", [True, False])
def test_films_and_reports_caveats_visible(monkeypatch, root, public):  # noqa: F811
    assert _has(run_app(monkeypatch, root, public, "Films"), "no labels exist for these films")
    assert _has(run_app(monkeypatch, root, public, "Reports"), core.CAPTION_MODEL[:80])


def test_local_pages_caveats_visible(monkeypatch, root):  # noqa: F811
    assert any(CUT_WARNING in w for w in _warnings(run_app(monkeypatch, root, False, "Add your own clip")))
    assert any(core.CAPTION_ESSAY in w for w in _warnings(run_app(monkeypatch, root, False, "Tokens vs my essay")))


@pytest.mark.parametrize("page,sel", [("Film overview", {"ov_film": OWNED}), ("Shot explorer", {"ex_film": CLIP})])
def test_local_only_banner_visible(monkeypatch, root, page, sel):  # noqa: F811
    at = run_app(monkeypatch, root, False, page, **sel)
    assert _has(at, core.BANNER_LOCAL_ONLY, "Error")


@pytest.mark.parametrize("public", [True, False])
@pytest.mark.parametrize("page", ["Home", "Films", "Film overview", "Shot explorer", "Side by side", "Reports", "About"])
def test_one_page_title_per_page(monkeypatch, root, public, page):  # noqa: F811
    at = run_app(monkeypatch, root, public, page)
    assert len(at.title) == 0  # the repeated app title is gone (v2.1); the brand sits once in the sidebar
    assert len(at.main.header) == 1, [h.value for h in at.main.header]
    text = " ".join(t for _, t in _visible(at))
    assert ("PUBLIC MODE" in text) == public and ("local mode" in text) == (not public)  # the mode badge stays visible


def test_theme_config_is_valid():
    cfg = tomllib.loads((REPO / ".streamlit/config.toml").read_text())
    assert "base" not in cfg["theme"] and "primaryColor" not in cfg["theme"]  # light and dark both defined: the app follows the viewer's setting
    assert cfg["theme"]["light"]["primaryColor"].startswith("#") and cfg["theme"]["dark"]["primaryColor"].startswith("#")
