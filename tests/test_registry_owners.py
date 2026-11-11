"""an earlier change owner changes (docs/v1_entry_point.md "Owner changes"): p1 (and through it p2 and scripts/spotcheck_cuts.py), the p3c runner and the move
stage resolve films through the merged registry.
Synthetic only: tmp_path roots, tiny lavfi clips, stubbed models. No GPU, no real film, no labels."""
import importlib.util
import json
import runpy
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cgt.p1_shots_color import pipeline as p1
from cgt.p3b_vlm import guards
from cgt.p3c_tokens import runner as p3c
from cgt.p4_camera_move import film_move as FM
from cgt.pipeline import adhoc, registry

REPO = Path(__file__).resolve().parents[1]
MAIN = [{"id": "pd_main", "public_domain": True, "path": "data/raw/pd.mp4", "cut_threshold": 12},
        {"id": "owned_main", "public_domain": False, "path": "data/raw/owned.mp4"}]


def make_root(root: Path, overlay: list[dict]) -> Path:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs/films.yaml").write_text(yaml.safe_dump({"films": MAIN}, sort_keys=False))
    if overlay:
        registry.write_overlay(root, overlay)
    return root


def clip(path: Path, seconds: float = 2.0) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg absent")
    path.parent.mkdir(parents=True, exist_ok=True)
    half = seconds / 2  # two colour segments -> one hard cut in the middle
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=red:s=64x48:r=24:d={half}", "-f", "lavfi", "-i",
                    f"color=c=blue:s=64x48:r=24:d={half}", "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture
def root(tmp_path, monkeypatch):
    video = clip(tmp_path / "inputs/my clip.mp4")
    r = make_root(tmp_path / "repo", [{"id": "adhoc_1", "path": str(video), "public_domain": False, "cut_threshold": 20},
                                      {"id": "adhoc_pd", "path": str(video), "public_domain": True}])
    monkeypatch.setattr(p1, "ROOT", r)
    monkeypatch.setattr(p3c, "ROOT", r)
    return r


def _shot_table(root: Path, fid: str) -> None:
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"shot_id": [0], "start_s": [0.], "end_s": [1.], "len_s": [1.]}).to_parquet(root / f"data/processed/{fid}_shots.parquet")


# ---------------------------------------------------------------- A: films.yaml films unchanged
def test_notld_p1_entry_and_threshold_identical_to_films_yaml():
    films = {f["id"]: f for f in yaml.safe_load((REPO / "configs/films.yaml").read_text())["films"]}
    if "night_of_the_living_dead" not in films:
        pytest.skip("NOTLD not registered in this checkout")
    e = p1.film_entry("night_of_the_living_dead")
    assert json.dumps(e, sort_keys=True) == json.dumps(films["night_of_the_living_dead"], sort_keys=True)
    assert p1.resolve_threshold("night_of_the_living_dead") == float(films["night_of_the_living_dead"].get("cut_threshold", p1.DEFAULT_CUT_THRESHOLD))


def test_main_entries_returned_unchanged_everywhere(root):
    assert p1.film_entry("pd_main") == MAIN[0] and p1.resolve_threshold("pd_main") == 12.0
    assert p3c.require_public_domain("pd_main") == MAIN[0]
    with pytest.raises(guards.NotAuthorized):
        p3c.require_public_domain("owned_main")
    assert p3c.require_public_domain("owned_main", allow_owned=True) == MAIN[1]
    assert p1.figures_dir(p1.film_entry("pd_main")) == root / "results/figures"
    assert p1.figures_dir(p1.film_entry("owned_main")) is None


# ---------------------------------------------------------------- B: overlay films resolve; unknown ids refused; allow-owned required
def test_p1_and_p2_resolve_overlay(root):
    e = p1.film_entry("adhoc_1")
    assert registry.is_overlay(e) and e["public_domain"] is False
    assert p1.resolve_threshold("adhoc_1") == 20.0 and p1.resolve_threshold("adhoc_pd") == p1.DEFAULT_CUT_THRESHOLD
    p2 = pytest.importorskip("cgt.p2_pretrained.pipeline")
    assert p2.film_entry("adhoc_1")["id"] == "adhoc_1"  # p2 uses p1's film_entry


def test_unknown_id_refused_everywhere(root):
    with pytest.raises(StopIteration):
        p1.film_entry("ghost")
    with pytest.raises(guards.NotAuthorized, match="not registered"):
        p3c.require_public_domain("ghost", allow_owned=True)
    with pytest.raises(FM.Refused, match="not registered"):
        FM.resolve_film("ghost", allow_owned=True, root=root)


@pytest.mark.parametrize("fid", ["adhoc_1", "adhoc_pd"])
def test_p3c_overlay_needs_allow_owned_even_if_public_domain(root, fid):
    with pytest.raises(guards.NotAuthorized, match="allow-owned"):
        p3c.require_public_domain(fid)
    e = p3c.require_public_domain(fid, allow_owned=True)
    assert e["id"] == fid and registry.is_overlay(e)


@pytest.mark.parametrize("fid", ["adhoc_1", "adhoc_pd"])
def test_film_move_overlay_needs_allow_owned(root, fid):
    _shot_table(root, fid)
    with pytest.raises(FM.Refused, match="allow-owned"):
        FM.resolve_film(fid, root=root)
    f = FM.resolve_film(fid, allow_owned=True, root=root)
    assert f["film_path"].is_file() and f["shots_path"].is_file()


def test_film_move_explicit_films_yaml_is_that_file_only(root):
    _shot_table(root, "adhoc_1")
    with pytest.raises(FM.Refused, match="not registered"):
        FM.resolve_film("adhoc_1", allow_owned=True, root=root, films_yaml=root / "configs/films.yaml")


def test_clash_is_a_refusal_not_a_crash(root):
    registry.write_overlay(root, registry.read_overlay(root) + [{"id": "pd_main", "path": "x.mp4"}])
    with pytest.raises(FM.Refused, match="registered twice"):
        FM.resolve_film("pd_main", root=root)
    with pytest.raises(registry.RegistryClash):
        p1.film_entry("pd_main")


@pytest.fixture
def move_script():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("w1_run_move_film", REPO / "scripts/run_move_film.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("fid", ["adhoc_1", "adhoc_pd"])
def test_run_move_film_resolves_overlay_and_refuses_without_flag(root, move_script, fid):
    _shot_table(root, fid)
    with pytest.raises(FM.Refused, match="allow-owned"):
        move_script.resolve(fid, False, root)
    assert move_script.resolve(fid, True, root)["id"] == fid
    with pytest.raises(FM.Refused, match="not registered"):
        move_script.resolve("ghost", True, root)


def test_spotcheck_cuts_on_overlay_film(root, monkeypatch):
    cuts = root / "data/interim/adhoc_1/cuts_t20.json"
    cuts.parent.mkdir(parents=True, exist_ok=True)
    cuts.write_text(json.dumps([[0.0, 1.0], [1.0, 2.0]]))
    monkeypatch.setattr(sys, "argv", ["spotcheck_cuts.py", "--film", "adhoc_1", "--threshold", "20", "--n", "50"])
    runpy.run_path(str(REPO / "scripts/spotcheck_cuts.py"), run_name="__main__")
    assert (root / "data/interim/adhoc_1/spotcheck_t20/sheet_0.jpg").is_file() and not (root / "results").exists()


# ---------------------------------------------------------------- C: no output of an overlay film under results/
@pytest.mark.parametrize("fid", ["adhoc_1", "adhoc_pd"])
def test_p1_run_film_overlay_figures_stay_local(root, fid):
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    df = p1.run_film(fid, k=2)
    assert len(df) >= 1 and (root / f"data/processed/{fid}_shots.parquet").is_file()
    figs = root / f"data/interim/{fid}/figures"
    assert (figs / f"{fid}_barcode.png").is_file() and (figs / f"{fid}_shot_length.png").is_file()
    assert not (root / "results").exists()


def test_move_report_path(root, move_script):
    for fid in ("adhoc_1", "adhoc_pd"):
        assert move_script.default_report_path(root, p1.film_entry(fid)) == root / f"data/processed/{fid}_move_report.md"
    assert move_script.default_report_path(root, p1.film_entry("pd_main")) == root / "results/phase4/film_move/pd_main_report.md"
    assert move_script.default_report_path(root, p1.film_entry("owned_main")) == root / "data/processed/owned_main_move_report.md"


def test_run_move_film_main_on_public_domain_overlay_film_writes_nothing_under_results(root, move_script, monkeypatch):
    _shot_table(root, "adhoc_pd")
    ctx = types.SimpleNamespace(columns=["n_pairs", "signal"], prior=np.array([.2, .6]), atoms=["static", "push-in"], C=.01,
                                Xtr=np.array([[0., 0.]]), Ytr=np.array([[0, 1]]))
    m = move_script
    monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setattr(m.FR, "geometry_identity", lambda: {"C": .01, "split": {}, "config_hash": "fixture"})
    monkeypatch.setattr(m.FM, "memory_guard", lambda: None)
    monkeypatch.setattr(m.FE, "fit_atom_probs", lambda *a: np.array([[.8, .1]]))
    monkeypatch.setattr(m, "one", lambda args: (0, {"n_pairs": 8, "signal": 0.}, None))
    train = types.ModuleType("p4_final_eval")
    train.TrainContext = lambda *a: ctx

    class SerialPool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def imap_unordered(self, fn, jobs, **k):
            return map(fn, jobs)

    monkeypatch.setitem(sys.modules, "p4_final_eval", train)
    monkeypatch.setattr(m, "Pool", SerialPool)
    monkeypatch.setattr(sys, "argv", ["run_move_film.py", "--film", "adhoc_pd", "--allow-owned"])
    m.main()
    assert (root / "data/processed/adhoc_pd_move.parquet").is_file() and (root / "data/processed/adhoc_pd_move_report.md").is_file()
    assert not (root / "results").exists()


# ---------------------------------------------------------------- D: an earlier change's owner-module probe passes for an overlay film
def test_owner_probes_pass_for_overlay_film(root, monkeypatch):
    _shot_table(root, "adhoc_1")
    monkeypatch.setitem(FM.resolve_film.__kwdefaults__, "root", root)  # the probe calls resolve_film without root
    assert adhoc.owner_blockers("adhoc_1", ["p1", "p2", "p3c", "move"]) == {}
    assert set(adhoc.owner_blockers("ghost", ["p1", "p3c", "move"])) == {"p1", "p3c", "move"}
