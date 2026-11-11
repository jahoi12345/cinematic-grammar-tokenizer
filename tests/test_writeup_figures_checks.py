"""an earlier review (figures verification): close mutation survivors of scripts/writeup_figures.py. Synthetic inputs, plus the committed sample legend."""
import hashlib
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
tspec = importlib.util.spec_from_file_location("test_writeup_figures_base", ROOT / "tests/test_writeup_figures.py")
base = importlib.util.module_from_spec(tspec)
tspec.loader.exec_module(base)
wf = base.wf  # the same module instance base._run calls


def _capture(monkeypatch):
    figs = {}
    real = wf._save

    def keep(fig, path, dpi=wf.DPI):
        figs[Path(path).name] = fig
        real(fig, path, dpi)

    monkeypatch.setattr(wf, "_save", keep)
    return figs


def test_small_windows_hollow_and_caveat_in_arc_title(tmp_path, monkeypatch):
    d = base._make(tmp_path / "in")  # window 5 has 3 shots, the others 12
    figs = _capture(monkeypatch)
    base._run(d, tmp_path / "o")
    fig = figs[f"{base.FILM}_fig2_visual_arc.png"]
    assert wf.ARC_CAPTION in fig._suptitle.get_text()
    for ax in fig.axes:
        hollow = [ln for ln in ax.get_lines() if ln.get_markerfacecolor() == "white"]
        assert len(hollow) == 1 and len(hollow[0].get_xdata()) == 1 and hollow[0].get_xdata()[0] == 5 * 5.0 + 2.5


def test_top1_uses_rank_one_rows(tmp_path, monkeypatch):
    d = base._make(tmp_path / "in")
    _capture(monkeypatch)
    st = base._run(d, tmp_path / "o")["stats"]["gaps"]
    sim = pd.read_csv(d["results"] / f"similar_shots_{base.FILM}.csv")
    r1 = sim.loc[sim["rank"] == 1, "gap_min"].to_numpy(float)
    assert st["n_top1"] == len(r1) and st["median_top1_min"] == pytest.approx(float(np.median(r1)))
    assert st["median_top1_min"] != pytest.approx(float(sim.loc[sim["rank"] == 2, "gap_min"].median()))


def test_refuses_table_of_another_film_even_with_its_own_sha(tmp_path):
    d = base._make(tmp_path / "in")
    df = pd.read_parquet(d["tokens"])
    df["film_id"] = "toy_owned_film"
    other = tmp_path / "other.parquet"
    df.to_parquet(other, index=False)
    sha = hashlib.sha256(other.read_bytes()).hexdigest()
    with pytest.raises(wf.Refused, match="another film"):
        wf.build(base.FILM, other, sha, tmp_path / "o", d["results"], code_root=d["root"], anatomy_shot=5)
    assert not (tmp_path / "o").exists()


def test_model_slots_match_committed_sample_legend():
    md = (ROOT / "results/phase6/notld_token_rows_sample.md").read_text()
    m = re.search(r"Model slots \(([^)]*)\) and heuristic slots \(([^)]*)\)", md)
    assert m, "legend line not found"
    model = {s.strip() for s in m.group(1).split(",")}
    heur = {s.strip() for s in m.group(2).split(",")}
    assert wf.MODEL_SLOTS == model
    assert set(wf.SLOT_TEXT) == model | heur
