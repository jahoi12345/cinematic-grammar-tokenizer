"""scripts/writeup_figures.py on synthetic inputs only (no film data): determinism, data arrays, refusals."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("writeup_figures", ROOT / "scripts/writeup_figures.py")
wf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wf)

FILM = "toy_public_film"
ROW = "[size=MCU* | lens=medium | angle=low | framing=single | light=daylight/soft | comp=center* | key=mid-key | DoF=medium | people=1 | sound=ambient | len=2-4s | palette=g20 | move=static]"


def _make(tmp: Path, n: int = 60) -> dict:
    (tmp / "configs").mkdir(parents=True)
    (tmp / "configs/films.yaml").write_text(
        "films:\n  - id: toy_public_film\n    title: Toy Public Film\n    public_domain: true\n"
        "  - id: toy_owned_film\n    title: Toy Owned Film\n    public_domain: false\n")
    rng = np.random.default_rng(0)
    lens = np.round(np.exp(rng.normal(1.1, 0.8, n)), 3)
    start = np.concatenate([[0.0], np.cumsum(lens)[:-1]])
    rows = [ROW.replace("palette=g20", f"palette=g{i % 32}") for i in range(n)]
    rows[5] = ROW
    tok = tmp / "tokens.parquet"
    pd.DataFrame({"film_id": FILM, "shot_id": np.arange(n), "start_s": start, "len_s": lens, "token_row": rows}).to_parquet(tok, index=False)
    res = tmp / "results"
    res.mkdir()
    recs = []
    for w in range(6):
        for s, _, _, _ in wf.ARC_PANELS:
            v = float(rng.uniform(0.1, 0.9))
            kind = "prediction" if s in ("close_share", "mean_size_ordinal") else "heuristic"
            lo, hi = (None, None) if s == "cuts_per_min" else (v - 0.1, v + 0.1)
            recs.append({"window": w, "window_start_min": 5.0 * w, "n_shots": 3 if w == 5 else 12, "series": s, "kind": kind,
                         "value": v, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(recs).to_csv(res / f"visual_arc_{FILM}.csv", index=False)
    sim = [{"shot_id": i, "rank": r, "other_shot_id": (i + 3 + r) % n, "score": 0.9 - 0.1 * r, "gap_min": abs(start[i] - start[(i + 3 + r) % n]) / 60}
           for i in range(n) for r in (1, 2)]
    pd.DataFrame(sim).to_csv(res / f"similar_shots_{FILM}.csv", index=False)
    (res / "notld_token_rows_sample.md").write_text(f"# sample\n\n```\n00005 000:12.3 {ROW}\n```\n")
    return {"tokens": tok, "results": res, "root": tmp, "sha": hashlib.sha256(tok.read_bytes()).hexdigest()}


def _run(d, out, film=FILM, sha=None):
    return wf.build(film, d["tokens"], sha or d["sha"], out, d["results"], code_root=d["root"], anatomy_shot=5)


def test_deterministic_bytes_and_small(tmp_path):
    d = _make(tmp_path / "in")
    r1 = _run(d, tmp_path / "a")
    _run(d, tmp_path / "b")
    names = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert len(names) == 5 and "figures.md" in names
    for nm in names:
        assert (tmp_path / "a" / nm).read_bytes() == (tmp_path / "b" / nm).read_bytes(), nm
    for p in r1["paths"].values():
        assert p.name.startswith(FILM + "_") and p.stat().st_size < 300_000
    md = (tmp_path / "a/figures.md").read_text()
    assert wf.ARC_CAPTION in md and d["sha"] in md and "Toy Public Film" in md


def test_eligible_gaps_and_bins():
    s = np.array([0.0, 60.0, 180.0, 300.0, 600.0])
    g = wf.eligible_gaps_min(s)
    # ordered pairs with |i-j| > 2: (0,3),(0,4),(1,4) and reverses
    assert sorted(g.tolist()) == sorted([5.0, 10.0, 9.0] * 2)
    assert len(wf.eligible_gaps_min(np.arange(1077.0))) == 1_154_550
    h = wf.gap_hist(np.array([0.5, 1.5, 5.0, 100.0]), np.array([0.0, 1.0, 2.0, 8.0]))
    assert h.tolist() == [0.25, 0.25, 0.5]  # values above the last edge land in the last bin
    b = wf.length_bins(np.array([0.5, 3.0, 40.0]))
    assert b[0] <= 0.5 and b[-1] >= 40.0 and np.all(np.diff(np.log10(b)) > 0)
    assert [k for k, _ in wf.parse_row(ROW)][:3] == ["size", "lens", "angle"] and len(wf.parse_row(ROW)) == 13


def test_refuses_non_public_film(tmp_path):
    d = _make(tmp_path / "in")
    for film in ("toy_owned_film", "unregistered_film"):
        with pytest.raises(wf.Refused, match="public_domain"):
            _run(d, tmp_path / "o", film=film)
    assert not (tmp_path / "o").exists()


def test_refuses_overlay_film(tmp_path):
    d = _make(tmp_path / "in")
    (d["root"] / "configs/films.local.yaml").write_text("films:\n  - id: toy_public_film\n    public_domain: true\n")
    with pytest.raises(wf.Refused):
        _run(d, tmp_path / "o")


def test_refuses_wrong_hash(tmp_path):
    d = _make(tmp_path / "in")
    with pytest.raises(wf.Refused, match="sha256"):
        _run(d, tmp_path / "o", sha="0" * 64)
    assert not (tmp_path / "o").exists()


def test_refuses_mismatched_sample_row(tmp_path):
    d = _make(tmp_path / "in")
    (d["results"] / "notld_token_rows_sample.md").write_text(f"```\n00005 000:12.3 {ROW.replace('low', 'high')}\n```\n")
    with pytest.raises(wf.Refused, match="does not match"):
        _run(d, tmp_path / "o")


def test_cli_refuses_film_absent_from_real_registry(tmp_path, capsys):
    """The CLI reads the committed registry: the toy film is refused on the film check, before any sha check (figures A2)."""
    d = _make(tmp_path / "in")
    rc = wf.main(["--film", FILM, "--tokens", str(d["tokens"]), "--expect-sha", "1" * 64, "--out", str(tmp_path / "o"),
                  "--results", str(d["results"])])
    err = capsys.readouterr().err
    assert rc == 2 and "REFUSED" in err and "not a public_domain" in err and "sha256" not in err


def test_cli_exit_code_on_sha_refusal(tmp_path, capsys):
    """Figures A2: a public film id of the committed registry with a toy table, so the CLI really reaches the sha check; A1: both full hashes."""
    d = _make(tmp_path / "in")
    want = "1" * 64
    rc = wf.main(["--film", "night_of_the_living_dead", "--tokens", str(d["tokens"]), "--expect-sha", want, "--out", str(tmp_path / "o"),
                  "--results", str(d["results"])])
    err = capsys.readouterr().err
    assert rc == 2 and "REFUSED" in err and "sha256" in err and d["sha"] in err and want in err
    assert not (tmp_path / "o").exists()


def test_sha_message_full_hashes_when_prefix_shared(tmp_path):
    """Figures A1: a hash differing only after char 12 must not read as 'x... != expected x...'."""
    d = _make(tmp_path / "in")
    near = d["sha"][:12] + ("0" if d["sha"][12] != "0" else "1") + d["sha"][13:]
    with pytest.raises(wf.Refused) as e:
        _run(d, tmp_path / "o", sha=near)
    assert d["sha"] in str(e.value) and near in str(e.value)


@pytest.mark.parametrize("raw,label", [("051:00.0", "51:00.0"), ("000:12.3", "0:12.3"), ("100:05.5", "100:05.5"), ("007:00.0", "7:00.0")])
def test_start_label(raw, label):
    assert wf.start_label(raw) == label


def test_anatomy_title_uses_start_label(tmp_path, monkeypatch):
    """Figures A5: the toy sample starts at 000:12.3; the title must say 0:12.3, not :12.3."""
    import matplotlib.axes
    seen = []
    real = matplotlib.axes.Axes.text
    monkeypatch.setattr(matplotlib.axes.Axes, "text", lambda self, x, y, s, *a, **k: (seen.append(str(s)), real(self, x, y, s, *a, **k))[1])
    d = _make(tmp_path / "in")
    _run(d, tmp_path / "o")
    title = [t for t in seen if "anatomy of one real token row" in t]
    assert title and "starts 0:12.3)" in title[0]
