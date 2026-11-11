"""the write-up, figures and sample rows refreshed to the 4B film default.

Synthetic tests of the recompute helpers always run. Tests that recompute the committed numbers from the live local token tables skip when the
tables are absent (data/ is gitignored); the owned film is located through the committed registry (public_domain: false), never by name here.
a recorded decision (2026-10-28, a decision record): the official-protocol (matched) ShotBench results are allowed in the public
paper now, so test_public_paper_has_no_internal_or_owned_text no longer bans them (tests/test_public_draft_no_internal.py checks that they come
with their caveats); owned-film and user-clip terms stay banned. In a release tree (public-only registry, no internal draft) the owned-film and
internal-draft tests skip.
"""
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("writeup_figures_w29", ROOT / "scripts/writeup_figures.py")
wf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wf)

SHA_4B = "ee2a20921147fb1bc1ea4fdf2a50d8f2f47fed43a8a4eca9b1ddf3530990c135"
SHA_2B = "0e818c9d48546fc087609c539e9b9dd007b74100fd58b467c42f1350eaee9fb7"
FILM = "night_of_the_living_dead"
NUMBERS = ROOT / "results/phase6/notld_writeup_numbers.json"
SAMPLE = ROOT / "results/phase6/notld_token_rows_sample.md"
DRAFT = ROOT / "docs/writeup/draft.md"
def _main_checkout() -> Path:
    try:
        common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True,
                                text=True, check=True).stdout.strip()
        return Path(common).parent
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ROOT


def _table(film_id: str) -> Path | None:
    for root in (ROOT, _main_checkout()):
        p = root / f"data/processed/{film_id}_tokens_v1.parquet"
        if p.exists():
            return p
    return None


def _synthetic() -> pd.DataFrame:
    rows = []
    for i in range(10):
        rows.append({"size": ["ECU", "CU", "MCU", "MS", "WS"][i % 5], "angle": ["low", "high"][i % 2], "light_type": "daylight" if i < 7 else "artificial",
                     "people": 1 if i < 4 else 2, "framing": "single" if i < 3 else "group",
                     **{f"{a}_consistent": i < (6 if a in wf.SINGLE_LABEL else 2) for a in wf.STATIC},
                     "kf_all_agree": i < 2, "len_s": float(i + 1), "move_atoms": np.array(["static"] if i < 5 else ["static", "pan-left"], dtype=object)})
    return pd.DataFrame(rows)


def test_section6_numbers_synthetic():
    r = wf.section6_numbers(_synthetic())
    assert r["shots"] == 10
    assert r["size_ecu_to_ms_share"] == 0.8
    assert r["angle_share"]["low"] == 0.5 and r["angle_share"]["high"] == 0.5
    assert r["light_type_share"] == {"daylight": 0.7, "artificial": 0.3}
    assert r["four_single_label_agree_share"] == 0.6
    assert r["all_seven_agree"] == {"n": 2, "share": 0.2}
    assert r["one_person_framing"] == {"n_shots": 4, "share_single_insert_ots": 0.75}


def test_contrast_numbers_synthetic_has_no_identity():
    r = wf.contrast_numbers(_synthetic())
    assert r == {"shots": 10, "median_len_s": 5.5, "contradictory_move_sets": {"n": 5, "share": 0.5}, "all_seven_agree": {"n": 2, "share": 0.2}}
    assert set(r) == {"shots", "median_len_s", "contradictory_move_sets", "all_seven_agree"}  # aggregates only


def test_committed_numbers_cite_4b_table():
    doc = json.loads(NUMBERS.read_text())
    assert doc["token_table"]["sha256"] == SHA_4B == wf.NOTLD_SHA
    assert SHA_4B in (ROOT / "docs/writeup/figures/figures.md").read_text()
    assert SHA_2B not in (ROOT / "docs/writeup/figures/figures.md").read_text()
    assert SHA_4B in SAMPLE.read_text()


def test_committed_numbers_recompute_from_live_table():
    p = _table(FILM)
    if p is None or wf.sha256(p) != SHA_4B:
        pytest.skip("live NOTLD 4B token table not present")
    df = pd.read_parquet(p)
    doc = json.loads(NUMBERS.read_text())
    assert doc["section6"] == json.loads(json.dumps(wf.section6_numbers(df)))
    assert doc["contrast"] == json.loads(json.dumps(wf.contrast_numbers(df)))


def test_sample_rows_equal_live_table():
    p = _table(FILM)
    if p is None or wf.sha256(p) != SHA_4B:
        pytest.skip("live NOTLD 4B token table not present")
    df = pd.read_parquet(p, columns=["shot_id", "token_row"]).set_index("shot_id")
    md = SAMPLE.read_text()
    for shot in (269, 538, 807):
        _, row = wf.sample_row(md, shot)
        assert row == df.loc[shot, "token_row"]


def _owned_ids() -> list[str]:
    films = (yaml.safe_load((ROOT / "configs/films.yaml").read_text()) or {}).get("films") or []
    return [f["id"] for f in films if f.get("public_domain") is not True]


def test_owned_id_absent_from_files_this_refresh_writes():
    ids = _owned_ids()
    files = ["results/phase6/notld_writeup_numbers.json", "results/phase6/notld_token_rows_sample.md", "docs/writeup/figures/figures.md",
             "tests/test_writeup_4b.py", "scripts/writeup_figures.py", "docs/writeup/draft.md", "docs/writeup/draft_public.md",
             "docs/writeup/paper_source_public.qmd", "docs/writeup/paper_public.qmd", "docs/writeup/paper_public.html"]
    for f in files:
        if not (ROOT / f).exists():  # the internal draft is not part of a release tree
            continue
        t = (ROOT / f).read_text(errors="replace").lower()
        for i in ids:
            assert i.lower() not in t, f"{f} names an owned film id"


def test_public_paper_has_no_internal_or_owned_text():
    owned_terms = []
    for f in (yaml.safe_load((ROOT / "configs/films.yaml").read_text()) or {}).get("films") or []:
        if f.get("public_domain") is not True:
            owned_terms += [f["id"].lower(), str(f.get("title", "")).lower()]
            owned_terms += [w for w in re.findall(r"[a-z]{6,}", str(f.get("title", "")).lower())]
    for name in ("paper_public.qmd", "paper_public.html", "paper_source_public.qmd"):
        t = (ROOT / "docs/writeup" / name).read_text().lower()
        for term in ("internal until", "user clip", "user's clip", "clip supplied by the user"):  # matched results allowed since a recorded decision
            assert term not in t, f"{name}: {term!r}"
        for term in owned_terms:
            assert term not in t, f"{name}: owned-film term"
