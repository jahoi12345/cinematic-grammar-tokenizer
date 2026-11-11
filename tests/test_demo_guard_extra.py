"""an earlier review adversarial checks of the Phase 6 demo guard (app/demo, docs/demo_spec.md; review a verification record).
Synthetic tmp roots only; never touches film data. Skipped when app/demo is absent (branch without the demo)."""
import builtins
import importlib.util
import os
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
if not (REPO / "app/demo/core.py").exists():
    pytest.skip("app/demo not on this branch", allow_module_level=True)

try:
    from _private_values import owned_contra_ratio
except ImportError:  # the private loader is not part of an exported copy: nothing private to guard against
    def owned_contra_ratio() -> str:
        return "<no private value>"
from test_demo_guard import CLIP, NOTLD, OWNED, _table, _texts, make_root, run_app


def _core():
    spec = importlib.util.spec_from_file_location("demo_core_vh", REPO / "app/demo/core.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


core = _core()


@pytest.fixture
def root(tmp_path):
    return make_root(tmp_path)


@pytest.mark.parametrize("val", ["1", " 1", "1 ", "true", "TRUE", "false", "False", "no", "NO", "off", "yes", "00", "-0", "0.0", "o"])
def test_env_any_value_is_public(monkeypatch, val):
    monkeypatch.setenv("CGT_DEMO_PUBLIC", val)
    assert core.public_mode() is True


@pytest.mark.parametrize("val", ["", "0"])
def test_env_documented_off_values(monkeypatch, val):
    # spec: only exactly empty or "0" is local (advisory A1 fixed: no whitespace stripping)
    monkeypatch.setenv("CGT_DEMO_PUBLIC", val)
    assert core.public_mode() is False


@pytest.mark.parametrize("val", [" 0", "0 ", "  "])
def test_env_whitespace_variants_are_public(monkeypatch, val):
    monkeypatch.setenv("CGT_DEMO_PUBLIC", val)
    assert core.public_mode() is True


@pytest.mark.parametrize("flag", ["true", "True", "yes", 1, "1", [True], {"x": True}, None])
def test_non_boolean_public_domain_is_not_public(tmp_path, flag):
    r = make_root(tmp_path, [{"id": NOTLD, "public_domain": flag}, {"id": OWNED, "public_domain": False}], [], frames=False)
    assert core.listed_films(r, True) == []


@pytest.mark.parametrize("main_text,overlay_text", [
    ("films: [\n  - {id: x", None),                                        # invalid YAML main
    ("- a\n- b\n", None),                                                   # main is a list, not a mapping
    ("films:\n  - just_a_string\n", None),                                  # entry not a mapping
    ("films:\n  - {public_domain: true}\n", None),                          # entry without id
    (None, "films: [\n  - {id: y"),                                         # invalid overlay YAML
    (None, "films:\n  - {id: night_of_the_living_dead, public_domain: true}\n"),  # overlay re-registers the public id
    (None, "- x\n"),                                                        # overlay is a list
])
def test_registry_edge_cases_list_nothing(root, main_text, overlay_text):
    if main_text is not None:
        (root / "configs/films.yaml").write_text(main_text)
    if overlay_text is not None:
        (root / "configs/films.local.yaml").write_text(overlay_text)
    try:
        listed = core.listed_films(root, True)
    except Exception:  # noqa: BLE001 - raising is also closed (nothing listed)
        listed = []
    assert listed == []


def test_registry_unreadable_file_lists_nothing(root):
    p = root / "configs/films.yaml"
    os.chmod(p, 0)
    try:
        if os.access(p, os.R_OK):
            pytest.skip("running as a user that can read mode-0 files")
        assert core.listed_films(root, True) == []
        assert core.is_local_only(root, NOTLD)
    finally:
        os.chmod(p, 0o644)


@pytest.mark.parametrize("film", [OWNED, CLIP, "stray", f"../{OWNED}", f"{NOTLD}/../{OWNED}", "Night_of_the_living_dead", NOTLD + " ", " " + NOTLD,
                                  str(Path("/") / "data/processed" / OWNED), "", ".", "*", f"{NOTLD}\x00"])
def test_direct_calls_refuse_before_any_file_access(root, monkeypatch, film):
    _table("stray").to_parquet(root / "data/processed/stray_tokens_v1.parquet")
    allowed = core.public_ids(root)  # read the registry first, then forbid every further file access
    assert allowed == {NOTLD}
    opened = []
    monkeypatch.setattr(core, "public_ids", lambda r: allowed)
    monkeypatch.setattr(core.pd, "read_parquet", lambda *a, **k: opened.append(("parquet", a)))
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda *a, **k: opened.append(("open", a)) or real_open(*a, **k))
    monkeypatch.setattr(Path, "exists", lambda self: opened.append(("exists", str(self))) or False)
    with pytest.raises(PermissionError):
        core.load_table(root, film, public=True)
    with pytest.raises(PermissionError):
        core.frame_path(root, film, f"data/interim/{OWNED}/keyframes/00000_mid.jpg", public=True)
    assert opened == []


def test_frame_path_traversal_absolute_and_symlinks(root):
    kd = root / f"data/interim/{NOTLD}/keyframes"
    (root / f"data/interim/{NOTLD}/keyframes_evil").mkdir()
    (root / f"data/interim/{NOTLD}/keyframes_evil/x.jpg").write_bytes(b"x")
    owned_kf = root / f"data/interim/{OWNED}/keyframes/00000_mid.jpg"
    for public in (True, False):
        assert core.frame_path(root, NOTLD, f"data/interim/{NOTLD}/keyframes/../../{OWNED}/keyframes/00000_mid.jpg", public) is None
        assert core.frame_path(root, NOTLD, str(owned_kf), public) is None  # absolute path
        assert core.frame_path(root, NOTLD, f"data/interim/{NOTLD}/keyframes", public) is None  # the directory itself
        assert core.frame_path(root, NOTLD, f"data/interim/{NOTLD}/keyframes_evil/x.jpg", public) is None  # prefix sibling
        for bad in (None, 3, b"x", ""):
            assert core.frame_path(root, NOTLD, bad, public) is None
    (kd / "link.jpg").symlink_to(owned_kf)
    (kd / "sub").symlink_to(owned_kf.parent, target_is_directory=True)
    for public in (True, False):
        assert core.frame_path(root, NOTLD, f"data/interim/{NOTLD}/keyframes/link.jpg", public) is None
        assert core.frame_path(root, NOTLD, f"data/interim/{NOTLD}/keyframes/sub/00000_mid.jpg", public) is None
    assert core.frame_path(root, NOTLD, f"data/interim/{NOTLD}/keyframes/00001_end.jpg", True) is not None


def test_overlay_public_flag_and_unregistered_are_local_only(root):
    assert core.is_local_only(root, CLIP) and core.is_local_only(root, "stray") and core.is_local_only(root, OWNED)
    assert not core.is_local_only(root, NOTLD)


def test_public_table_named_like_public_but_overlay_listed_nothing(tmp_path):
    # a committed public id that the overlay also lists (clash) hides everything in public mode
    r = make_root(tmp_path, overlay=[{"id": NOTLD, "public_domain": True}], frames=False)
    assert core.public_ids(r) == set() and core.listed_films(r, True) == []


@pytest.mark.parametrize("a,b", [(OWNED, NOTLD), (NOTLD, OWNED), (CLIP, NOTLD)])
def test_local_banner_on_side_by_side_either_slot(monkeypatch, root, a, b):
    at = run_app(monkeypatch, root, False, "Side by side", sb_a=a, sb_b=b)
    assert core.BANNER_LOCAL_ONLY in _texts(at)


@pytest.mark.parametrize("page,key", [("Film overview", "ov_film"), ("Shot explorer", "ex_film")])
def test_local_banner_on_owned_film_every_page(monkeypatch, root, page, key):
    at = run_app(monkeypatch, root, False, page, **{key: OWNED})
    assert core.BANNER_LOCAL_ONLY in _texts(at)
    at = run_app(monkeypatch, root, False, page, **{key: NOTLD})
    assert core.BANNER_LOCAL_ONLY not in _texts(at)


@pytest.mark.parametrize("val", ["false", "no", " 1", "TRUE"])
def test_public_app_with_odd_env_values_hides_owned(monkeypatch, root, val):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("CGT_DEMO_ROOT", str(root))
    monkeypatch.setenv("CGT_DEMO_PUBLIC", val)
    for page in ("Film overview", "Shot explorer", "Side by side"):
        at = testing.AppTest.from_file(str(REPO / "app/demo/app.py"), default_timeout=120)
        at.run()
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception
        text = _texts(at).lower()
        assert OWNED not in text and CLIP not in text and owned_contra_ratio() not in text


def test_side_by_side_chart_data_has_no_levels(root):
    df, _ = core.load_table(root, OWNED, False)
    sh = core.shapes(df, n_boot=10)
    assert set(sh.columns) == {"series", "kind", "frac", "z_within_film", "n_shots", "flag_sparse"}
    for _, d in sh.groupby("series"):
        assert abs(d.z_within_film.mean()) < 1e-9  # standardised within this one film


def test_registry_yaml_untouched_by_reads(root):
    before = {p: p.read_bytes() for p in (root / "configs").iterdir()}
    core.listed_films(root, True), core.listed_films(root, False), core.is_local_only(root, OWNED)
    assert {p: p.read_bytes() for p in (root / "configs").iterdir()} == before
    assert yaml.safe_load((root / "configs/films.yaml").read_text())["films"][0]["id"] == NOTLD
