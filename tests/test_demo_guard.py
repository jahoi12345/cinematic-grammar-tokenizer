"""Phase 6 demo (app/demo, docs/demo_spec.md): public-mode guard, captions, no writes. Synthetic tables under tmp roots only; never touches film data."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

try:
    from _private_values import owned_contra_ratio, owned_contra_tuple
except ImportError:  # the private loader is not part of an exported copy: nothing private to guard against
    def owned_contra_ratio() -> str:
        return "<no private value>"
    def owned_contra_tuple() -> str:
        return "<no private value>"

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app/demo/app.py"
NOTLD, OWNED, CLIP = "night_of_the_living_dead", "owned_film_x", "my_clip"  # OWNED: a synthetic owned film id (tmp registries only)
SECRET = (OWNED, CLIP, "owned_film")


def _core():
    spec = importlib.util.spec_from_file_location("demo_core_under_test", REPO / "app/demo/core.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


core = _core()


def _table(film: str, n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lens = rng.uniform(0.5, 12.0, n)
    start = np.concatenate([[0.0], np.cumsum(lens)[:-1]])
    atoms_pool = [["static"], ["static", "pan-left"], ["pan-left", "pan-right"], ["push-in"], ["tilt-up", "boom-up"]]
    atoms = [np.array(atoms_pool[i % len(atoms_pool)], dtype=object) for i in range(n)]
    d = {"film_id": film, "shot_id": np.arange(n), "start_s": start, "end_s": start + lens, "len_s": lens, "start_min": start / 60.0,
         "kf_start": [f"data/interim/{film}/keyframes/{i:05d}_start.jpg" for i in range(n)],
         "kf_mid": [f"data/interim/{film}/keyframes/{i:05d}_mid.jpg" for i in range(n)],
         "kf_end": [f"data/interim/{film}/keyframes/{i:05d}_end.jpg" for i in range(n)],
         "key": rng.choice(["low-key", "mid-key", "high-key"], n), "dof": rng.choice(["shallow", "medium", "deep"], n), "people": rng.integers(0, 6, n),
         "sound": rng.choice(["score", "dialogue", "ambient", "silence"], n), "len_tok": rng.choice(["1-2s", "2-4s", "4-8s"], n), "palette_tok": rng.choice(["g01", "g02"], n),
         "size": rng.choice(["ECU", "CU", "MCU", "MS", "MWS", "WS", "EWS"], n), "lens": rng.choice(["wide", "medium"], n), "angle": rng.choice(["low", "high", "aerial"], n),
         "framing": rng.choice(["single", "two-shot"], n), "light_type": rng.choice(["daylight", "artificial"], n), "light_cond": rng.choice(["soft", "high-contrast"], n),
         "composition": rng.choice(["balanced", "left-heavy"], n), "move_present": True, "move_atoms": atoms, "n_kf_disagree": rng.integers(0, 3, n)}
    df = pd.DataFrame(d)
    for a in ("size", "lens", "angle", "framing", "light_type", "light_cond", "composition"):
        df[f"{a}_p1"], df[f"{a}_margin"], df[f"{a}_consistent"] = 0.6, 0.2, True
    df["kf_all_agree"] = df["n_kf_disagree"] == 0
    df["move"] = ["+".join(a) for a in atoms]
    df["token_row"] = [f"[size={s} | film={film}]" for s in df["size"]]
    return df


def make_root(tmp: Path, films_yaml: list[dict] | None = None, overlay: list[dict] | None = None, frames: bool = True) -> Path:
    root = tmp / "root"
    (root / "configs").mkdir(parents=True)
    (root / "data/processed").mkdir(parents=True)
    (root / "results/phase6").mkdir(parents=True)
    (root / "results/phase6/keep.md").write_text("x\n")
    if films_yaml is None:
        films_yaml = [{"id": NOTLD, "public_domain": True}, {"id": OWNED, "public_domain": False}]
    (root / "configs/films.yaml").write_text(yaml.safe_dump({"films": films_yaml}))
    if overlay is None:
        overlay = [{"id": CLIP, "public_domain": True}]  # an overlay film is local only whatever its flag says
    if overlay:
        (root / "configs/films.local.yaml").write_text(yaml.safe_dump({"films": overlay}))
    for i, f in enumerate((NOTLD, OWNED, CLIP)):
        _table(f, seed=i).to_parquet(root / f"data/processed/{f}_tokens_v1.parquet")
        if frames:
            kd = root / f"data/interim/{f}/keyframes"
            kd.mkdir(parents=True)
            import cv2
            for j in range(8):
                for pos in ("start", "mid", "end"):
                    cv2.imwrite(str(kd / f"{j:05d}_{pos}.jpg"), np.full((16, 24, 3), 40 * i, np.uint8))
    return root


@pytest.fixture
def root(tmp_path):
    return make_root(tmp_path)


def snapshot(p: Path) -> dict[str, tuple[int, int]]:
    return {str(f.relative_to(p)): (f.stat().st_size, f.stat().st_mtime_ns) for f in p.rglob("*") if f.is_file()} if p.exists() else {}


# ---- pure guard -------------------------------------------------------------------------------------------------------------------------------------

def test_public_mode_lists_only_public_domain(root):
    assert core.listed_films(root, public=True) == [NOTLD]
    assert core.listed_films(root, public=False) == sorted([NOTLD, CLIP, OWNED])


def test_public_mode_refuses_before_reading(root, monkeypatch):
    calls = []
    monkeypatch.setattr(core.pd, "read_parquet", lambda *a, **k: calls.append(a) or pd.DataFrame())
    for f in (OWNED, CLIP, "unregistered_film"):
        with pytest.raises(PermissionError):
            core.load_table(root, f, public=True)
        with pytest.raises(PermissionError):
            core.frame_path(root, f, f"data/interim/{f}/keyframes/00000_mid.jpg", public=True)
    assert calls == []
    core.load_table(root, NOTLD, public=True)
    assert len(calls) == 1


def test_unregistered_table_is_hidden_in_public_and_local_only(root):
    _table("stray").to_parquet(root / "data/processed/stray_tokens_v1.parquet")
    assert "stray" not in core.listed_films(root, True)
    assert core.is_local_only(root, "stray")
    assert core.is_local_only(root, OWNED) and core.is_local_only(root, CLIP) and not core.is_local_only(root, NOTLD)


@pytest.mark.parametrize("films_yaml,overlay", [
    ([{"id": NOTLD, "public_domain": "true"}, {"id": OWNED, "public_domain": False}], []),        # a string is not a boolean
    ([{"id": NOTLD, "public_domain": True}, {"id": OWNED, "public_domain": False}], [{"id": NOTLD}]),  # clash: registry error -> nothing
    ([{"id": NOTLD, "public_domain": True}, {"id": NOTLD, "public_domain": True}], []),  # duplicate id in films.yaml: registry error -> nothing
])
def test_public_mode_fails_closed(tmp_path, films_yaml, overlay):
    r = make_root(tmp_path, films_yaml, overlay, frames=False)
    assert core.listed_films(r, True) == []


def test_public_mode_missing_registry_lists_nothing(root):
    (root / "configs/films.yaml").unlink()
    assert core.listed_films(root, True) == []


@pytest.mark.parametrize("val,expected", [(None, False), ("", False), ("0", False), ("1", True), ("true", True), ("yes", True)])
def test_public_mode_env(monkeypatch, val, expected):
    if val is None:
        monkeypatch.delenv("CGT_DEMO_PUBLIC", raising=False)
    else:
        monkeypatch.setenv("CGT_DEMO_PUBLIC", val)
    assert core.public_mode() is expected


@pytest.mark.parametrize("val", [" 0", "0 ", " ", "  ", "\t", "\n0", "0\n"])
def test_public_mode_whitespace_is_public(monkeypatch, val):
    # A1: only exactly "" or "0" means local; whitespace variants must fail closed (public)
    monkeypatch.setenv("CGT_DEMO_PUBLIC", val)
    assert core.public_mode() is True


def test_frame_path_stays_inside_keyframes(root):
    assert core.frame_path(root, NOTLD, f"data/interim/{NOTLD}/keyframes/00000_mid.jpg", True) is not None
    assert core.frame_path(root, NOTLD, f"data/interim/{OWNED}/keyframes/00000_mid.jpg", False) is None
    assert core.frame_path(root, NOTLD, "configs/films.yaml", False) is None


def test_shapes_have_no_levels(root):
    df, _ = core.load_table(root, NOTLD, True)
    sh = core.shapes(df)
    assert "value" not in sh.columns and "ci_lo" not in sh.columns
    for _, d in sh.groupby("series"):
        assert abs(d.z_within_film.mean()) < 1e-9


def test_movement_caption_cites_committed_share(root):
    df, _ = core.load_table(root, NOTLD, True)
    c = core.movement_caption(NOTLD, df, root)
    assert "INFLATED" in c and "462/1,077 = 42.9%" in c and "UNMEASURED" in c and owned_contra_ratio() not in c


def _real_owned_id() -> str:
    """The owned film of the real registry, read at runtime (this file never names it); skip in a trimmed tree without one."""
    owned = [f["id"] for f in yaml.safe_load((REPO / "configs/films.yaml").read_text())["films"] if f.get("public_domain") is not True]
    if not owned:
        pytest.skip("no owned film in this registry (trimmed release tree)")
    return owned[0]


def test_movement_caption_committed_share_only_for_public_films(tmp_path, root):
    """only public_domain films cite a committed share; owned, overlay and unregistered films show the live share only, and no owned number is in code."""
    df, _ = core.load_table(root, NOTLD, True)
    for other in (OWNED, CLIP, "unregistered_film", NOTLD + " ", NOTLD.upper()):
        c = core.movement_caption(other, df, root)
        assert "no committed share for this film" in c and "share in the loaded table" in c
    r2 = make_root(tmp_path / "r2", [{"id": NOTLD, "public_domain": False}], [], frames=False)  # the same id, not public in this registry -> no committed number
    assert "no committed share for this film" in core.movement_caption(NOTLD, df, r2)
    real = _real_owned_id()
    real_root = make_root(tmp_path / "r3", [{"id": NOTLD, "public_domain": True}, {"id": real, "public_domain": False}], [], frames=False)
    assert "no committed share for this film" in core.movement_caption(real, df, real_root)
    assert set(core.COMMITTED_CONTRADICTORY) <= core.public_ids(REPO)
    assert owned_contra_tuple() not in (REPO / "app/demo/core.py").read_text()


def test_core_writes_nothing(root):
    before = snapshot(root)
    for f in (NOTLD, OWNED):
        df, _ = core.load_table(root, f, False)
        core.arc_series(df, "equal20", n_boot=20)
        core.arc_series(df, "min5", n_boot=20)
        core.movement_series(df, n_boot=20)
        core.shapes(df)
    assert snapshot(root) == before


# ---- the app, run headless with streamlit's AppTest ---------------------------------------------------------------------------------------------------

def _texts(at) -> str:
    out = []

    def walk(node):
        proto = getattr(node, "proto", None)
        if proto is not None:
            out.append(str(proto))  # every field, including chart specs and data (escaped)
        for attr in ("value", "label", "body", "options"):  # the readable values, unescaped
            try:
                v = getattr(node, attr, None)
            except Exception:  # noqa: BLE001 - some element types raise for attributes they do not have
                v = None
            if v is not None:
                out.append(str(v))
        for c in getattr(node, "children", {}).values():
            walk(c)
    walk(at._tree)
    return "\n".join(out)


def run_app(monkeypatch, root: Path, public: bool, page: str, **select):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("CGT_DEMO_ROOT", str(root))
    if public:
        monkeypatch.setenv("CGT_DEMO_PUBLIC", "1")
    else:
        monkeypatch.delenv("CGT_DEMO_PUBLIC", raising=False)
    at = testing.AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.sidebar.radio(key="page").set_value(page).run()
    for key, val in select.items():
        at.sidebar.selectbox(key=key).set_value(val).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


PAGES = ["Film overview", "Shot explorer", "Side by side"]


@pytest.mark.parametrize("page", PAGES)
def test_public_app_never_shows_owned_films(monkeypatch, root, page):
    at = run_app(monkeypatch, root, True, page)
    text = _texts(at).lower()
    for s in SECRET:
        assert s not in text, f"{s} appears on {page} in public mode"
    assert core.BANNER_LOCAL_ONLY.lower() not in text
    for sb in at.sidebar.selectbox:
        assert all(o not in (OWNED, CLIP) for o in sb.options)


def test_public_app_overview_captions(monkeypatch, root):
    text = _texts(run_app(monkeypatch, root, True, "Film overview"))
    for cap in (core.CAPTION_PREDICTIONS, core.CAPTION_INTERVALS, core.CAPTION_SHAPES):
        assert cap.replace("`", "")[:80] in text.replace("`", "")
    assert "462/1,077 = 42.9%" in text and "INFLATED" in text


def test_side_by_side_caption_and_public_has_one_film(monkeypatch, root):
    text = _texts(run_app(monkeypatch, root, True, "Side by side"))
    assert core.CAPTION_SHAPES[:80] in text
    assert "needs two listed films" in text


def test_local_app_banner_on_owned_films(monkeypatch, root):
    at = run_app(monkeypatch, root, False, "Film overview", ov_film=OWNED)
    assert core.BANNER_LOCAL_ONLY in _texts(at)
    at = run_app(monkeypatch, root, False, "Shot explorer", ex_film=CLIP)
    assert core.BANNER_LOCAL_ONLY in _texts(at)
    at = run_app(monkeypatch, root, False, "Film overview", ov_film=NOTLD)
    assert core.BANNER_LOCAL_ONLY not in _texts(at)
    at = run_app(monkeypatch, root, False, "Side by side", sb_a=NOTLD)
    text = _texts(at)
    assert core.CAPTION_SHAPES[:80] in text and core.BANNER_LOCAL_ONLY in text  # film B is owned or overlay


def test_app_writes_nothing(monkeypatch, root):
    before, repo_results = snapshot(root), snapshot(REPO / "results")
    for public in (True, False):
        for page in PAGES:
            run_app(monkeypatch, root, public, page)
    run_app(monkeypatch, root, False, "Film overview", ov_film=OWNED)
    assert snapshot(root) == before
    assert snapshot(REPO / "results") == repo_results

