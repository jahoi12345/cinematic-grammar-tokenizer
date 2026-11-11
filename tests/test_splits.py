"""a recorded decision checks; every generated split lives in tmp_path, never the real data."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
FILMS = ("night_of_the_living_dead", "owned_film_x")  # synthetic tables under tmp_path


@pytest.fixture
def split_script(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("synthetic_make_splits", PROJECT / "scripts/make_splits.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    # Long synthetic timelines provide enough eligible shots for default counts.
    # Interleave sub-second shots to ensure the exclusion is exercised.
    for film in FILMS:
        starts = np.arange(0, 7200, 2, dtype=float)
        lengths = np.where(np.arange(len(starts)) % 11 == 0, 0.5, 1.5)
        pd.DataFrame({"shot_id": np.arange(len(starts)), "start_s": starts,
                      "end_s": starts + lengths, "len_s": lengths}).to_parquet(
            processed / f"{film}_shots.parquet", index=False
        )
    return module


def run_split(module, monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["make_splits.py", "--films", *FILMS, *extra])
    module.main()
    # ROOT is tmp_path supplied by split_script; these are scratch outputs only.
    return tuple((module.ROOT / f"data/{role}/shots.json").read_bytes() for role in ("dev", "gold"))


def test_default_counts_and_byte_determinism(split_script, monkeypatch, capsys):
    first = run_split(split_script, monkeypatch)
    second = run_split(split_script, monkeypatch)
    assert first == second
    assert [len(json.loads(value)) for value in first] == [100, 300]
    assert capsys.readouterr().out.splitlines() == ["dev=100 gold=300"] * 2


def test_disjoint_ids_and_30_second_interval_gap(split_script, monkeypatch):
    dev, gold = map(json.loads, run_split(split_script, monkeypatch))
    assert {(row["film"], row["shot_id"]) for row in dev}.isdisjoint(
        {(row["film"], row["shot_id"]) for row in gold}
    )
    for film in FILMS:
        dev_rows = [row for row in dev if row["film"] == film]
        gold_rows = [row for row in gold if row["film"] == film]
        assert dev_rows and gold_rows
        for left in dev_rows:
            for right in gold_rows:
                assert max(left["start_s"] - right["end_s"], right["start_s"] - left["end_s"]) >= 30


def test_subsecond_shots_excluded_and_rows_match_input(split_script, monkeypatch):
    for rows in map(json.loads, run_split(split_script, monkeypatch)):
        for row in rows:
            source = pd.read_parquet(
                split_script.ROOT / f"data/processed/{row['film']}_shots.parquet"
            ).set_index("shot_id").loc[row["shot_id"]]
            assert row["end_s"] - row["start_s"] >= 1
            assert row["start_s"] == source.start_s
            assert row["end_s"] == source.end_s


def test_nondivisible_counts_are_allocated_across_films(split_script, monkeypatch):
    dev, gold = map(json.loads, run_split(split_script, monkeypatch, "--dev", "101", "--gold", "301"))
    assert len(dev) == 101
    assert len(gold) == 301
    assert [sum(row["film"] == film for row in dev) for film in FILMS] == [51, 50]
    assert [sum(row["film"] == film for row in gold) for film in FILMS] == [151, 150]


def test_seed_changes_selection(split_script, monkeypatch):
    first = run_split(split_script, monkeypatch, "--seed", "0")
    second = run_split(split_script, monkeypatch, "--seed", "1")
    assert first != second
