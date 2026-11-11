"""Independent Phase 3C cache/refusal tests; every file/image is synthetic in tmp_path.

Engines are fakes. The real Engine is only constructed with a deliberately wrong
adapter hash, and its loader is replaced by a sentinel before construction.
"""
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cgt.p3b_vlm import guards
from cgt.p3c_tokens import runner, selection

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(f"synthetic_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEngine:
    adapter_sha = "a" * 64

    def __init__(self):
        self.passes = 0
        self.images = []

    def image_state(self, path):
        self.images.append(str(path))
        return path

    def letter_logits(self, state, user, n):
        self.passes += 1
        return np.arange(n, dtype=float)


@pytest.fixture
def film_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(guards, "require_clean_tree", lambda paths: {"commit": "c" * 40})
    monkeypatch.setattr(runner, "shrunk_path", lambda path, root: path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/films.yaml").write_text(
        "films:\n  - id: synthetic\n    public_domain: true\n"
        "  - id: owned\n    public_domain: false\n"
    )
    (tmp_path / "data/processed").mkdir(parents=True)

    def shots(relative):
        image = tmp_path / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (128, 128, 128)).save(image)
        pd.DataFrame({"film_id": ["synthetic"], "shot_id": [7],
                      **{kf: [str(relative)] for kf in runner.KEYFRAMES}}).to_parquet(
            tmp_path / "data/processed/synthetic_shots.parquet", index=False,
        )
        return image

    return tmp_path, shots


def test_run_key_is_idempotent_and_separates_every_registered_component():
    base = ("film", "a" * 64, "c" * 40, "D", 147456)
    key = runner.run_key(*base)
    assert key == runner.run_key(*base) and len(key) == 12
    assert all(ch in "0123456789abcdef" for ch in key)
    for pos, alternative in enumerate(("other", "b" * 64, "d" * 40, "B", 16384)):
        changed = list(base)
        changed[pos] = alternative
        assert runner.run_key(*changed) != key


def test_completed_film_cache_performs_zero_new_image_or_letter_passes(film_environment):
    root, shots = film_environment
    shots(Path("data/interim/synthetic/keyframes/00007.jpg"))
    engine = FakeEngine()
    directory = runner.run_film("synthetic", "D", root / "fake-adapter", engine=engine, log=lambda m: None)
    assert engine.passes == 21 and len(engine.images) == 3
    original = {p.name: p.read_bytes() for p in directory.glob("scores_*.jsonl")}
    resumed = runner.run_film("synthetic", "D", root / "fake-adapter", engine=engine, log=lambda m: None)
    assert resumed == directory and engine.passes == 21 and len(engine.images) == 3
    assert original == {p.name: p.read_bytes() for p in directory.glob("scores_*.jsonl")}


def test_resume_repairs_torn_last_line_before_appending(film_environment):
    root, shots = film_environment
    shots(Path("data/interim/synthetic/keyframes/00007.jpg"))
    directory = root / "data/cache/p3c/runs" / f"synthetic_{runner.run_key('synthetic', 'a' * 64, 'c' * 40, 'D')}"
    directory.mkdir(parents=True)
    (directory / "scores_kf_mid.jsonl").write_text('{"shot_id": 7, "size":')
    runner.run_film("synthetic", "D", root / "fake-adapter", engine=FakeEngine(), log=lambda m: None)
    for kf in runner.KEYFRAMES:
        records = [json.loads(line) for line in (directory / f"scores_{kf}.jsonl").read_text().splitlines()]
        assert [r["shot_id"] for r in records] == [7]


def test_selection_resume_cache_retains_id_attribute_collisions(tmp_path):
    script = _script("p3c_select_method")
    cache = tmp_path / "D.jsonl"
    records = [{"id": "duplicate#0", "attr": "size", "s": [1., 0.]},
               {"id": "duplicate#0", "attr": "light_type", "s": [0., 1.]}]
    cache.write_text("".join(json.dumps(row) + "\n" for row in records))
    done = script.load_done(cache)
    assert len(done) == 2
    assert done[("duplicate#0", "size")]["s"] == [1., 0.]
    assert done[("duplicate#0", "light_type")]["s"] == [0., 1.]


def test_wrong_adapter_hash_is_refused_before_model_loading(tmp_path, monkeypatch):
    from cgt.p3c_tokens import engine

    adapter = tmp_path / "synthetic.safetensors"
    adapter.write_bytes(b"not model weights")
    manifest = tmp_path / "frozen_manifest.json"
    manifest.write_text(json.dumps({"adapter_sha256": hashlib.sha256(b"different").hexdigest()}))
    monkeypatch.setattr(engine, "FROZEN_MANIFEST", manifest)

    def forbidden_load(*args, **kwargs):
        pytest.fail("model loader reached before the adapter hash refusal")

    monkeypatch.setattr(engine, "load", forbidden_load)
    with pytest.raises(guards.AdapterHashError) as err:
        engine.Engine(adapter)
    assert err.value.exit_code == 7


@pytest.mark.parametrize("relative", ["data/external/shotbench/x.jpg", "data/gold/x.jpg", "data/dev/x.jpg"])
def test_film_keyframes_refuse_holdout_roots_before_image_access(film_environment, relative, monkeypatch):
    root, shots = film_environment
    forbidden = shots(Path(relative))
    reads = []
    original = guards.require_images_readable

    def traced(paths):
        reads.extend(paths)
        return original(paths)

    monkeypatch.setattr(guards, "require_images_readable", traced)
    with pytest.raises(guards.DataRootViolation):
        runner.run_film("synthetic", "D", root / "fake-adapter", engine=FakeEngine(), log=lambda m: None)
    assert str(forbidden) not in reads


@pytest.mark.parametrize("relative", ["data/external/shotbench/x.jpg", "data/gold/x.jpg", "data/dev/x.jpg",
                                      "data/interim/synthetic/keyframes/x.jpg", "data/raw/films/x.jpg"])
def test_selection_inputs_refuse_holdouts_and_our_films_before_engine(tmp_path, monkeypatch, relative):
    from cgt.p3c_tokens import engine

    script = _script("p3c_select_method")
    samples = [{"id": "synthetic", "images": [str(tmp_path / relative)]}]
    monkeypatch.setattr(selection, "validation_subset", lambda: samples)
    monkeypatch.setattr(guards, "require_clean_tree", lambda paths: {"commit": "c" * 40})
    enforce = functools.partial(guards.require_samples_under_root, root=tmp_path / "data/external/shotqa")
    monkeypatch.setattr(guards, "require_samples_under_root", enforce)

    def forbidden_engine(*args, **kwargs):
        pytest.fail("engine constructed before the selection data-root refusal")

    monkeypatch.setattr(engine, "Engine", forbidden_engine)
    with pytest.raises(guards.DataRootViolation) as err:
        script.cmd_run(SimpleNamespace(limit=None, methods="D"))
    assert err.value.exit_code == 5


def test_unknown_and_owned_films_are_refused(film_environment):
    for film in ("owned", "unknown"):
        with pytest.raises(guards.NotAuthorized) as err:
            runner.require_public_domain(film)
        assert err.value.exit_code == 13


def test_cli_refuses_unselected_method_before_inference(tmp_path, monkeypatch):
    script = _script("p3c_run_film")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    selection_path = tmp_path / "results/phase3c/method_selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps({"rule": {"selected": "D"}}))
    monkeypatch.setattr("sys.argv", ["p3c_run_film.py", "--film", "synthetic", "--method", "B"])

    def forbidden_run(*args, **kwargs):
        pytest.fail("unselected method reached inference")

    monkeypatch.setattr(runner, "run_film", forbidden_run)
    with pytest.raises(SystemExit, match="not the method chosen"):
        script.main()
