# ruff: noqa: F811
"""F3: film keyframes must come from the authorised film's keyframe root; checked before any image access or model construction."""
import pandas as pd
import pytest
from _p3c_fixtures import KF, FakeEngine, film_env  # noqa: F401
from PIL import Image

from cgt.p3b_vlm import guards
from cgt.p3c_tokens import runner


@pytest.mark.parametrize("rel", ["data/external/shotbench/x.jpg", "data/gold/x.jpg", "data/dev/x.jpg", "data/raw/films/x.jpg",
                                 "data/interim/other_film/keyframes/x.jpg", "data/interim/synthetic/keyframes/../../other/x.jpg",
                                 "data/interim/synthetic/x.jpg"])
def test_film_keyframes_outside_the_authorised_root_fail_before_any_image_or_model_access(film_env, rel, monkeypatch):
    root, shots = film_env
    shots(rel)
    opened = []
    monkeypatch.setattr(guards, "require_images_readable", lambda paths: opened.extend(paths))
    monkeypatch.setattr(Image, "open", lambda *a, **k: opened.append(a))
    built = []
    import cgt.p3c_tokens.engine as eng_mod
    monkeypatch.setattr(eng_mod, "Engine", lambda *a, **k: built.append(a))
    with pytest.raises(guards.DataRootViolation) as e:
        runner.run_film("synthetic", "D", root / "adapter", log=lambda m: None)
    assert e.value.exit_code == 5 and not opened and not built


def test_keyframe_symlink_escape_is_refused(film_env):
    root, shots = film_env
    shots(KF.format(0))
    outside = root / "data/gold/secret.jpg"
    outside.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(outside)
    link = root / KF.format(1)
    link.symlink_to(outside)
    df = pd.read_parquet(root / "data/processed/synthetic_shots.parquet")
    df.loc[0, list(runner.KEYFRAMES)] = KF.format(1)
    df.to_parquet(root / "data/processed/synthetic_shots.parquet", index=False)
    with pytest.raises(guards.DataRootViolation):
        runner.run_film("synthetic", "D", root / "adapter", engine=FakeEngine(), log=lambda m: None)


def test_shot_table_film_id_must_match_authorised_film(film_env):
    root, shots = film_env
    shots(KF.format(0))
    df = pd.read_parquet(root / "data/processed/synthetic_shots.parquet")
    df["film_id"] = "someone_else"
    df.to_parquet(root / "data/processed/synthetic_shots.parquet", index=False)
    with pytest.raises(guards.DataRootViolation):
        runner.run_film("synthetic", "D", root / "adapter", engine=FakeEngine(), log=lambda m: None)


def test_absolute_keyframe_paths_are_accepted_only_inside_the_resolved_keyframe_root(film_env):
    from PIL import Image  # noqa: F401
    root, shots = film_env
    shots("data/interim/synthetic/keyframes/00000.jpg")
    real = (root / "data/interim/synthetic/keyframes").resolve()
    assert runner.keyframe_path("synthetic", str(real / "00000.jpg")) == real / "00000.jpg"
    for bad in (str(root / "data/gold/x.jpg"), str(real.parent / "other.jpg"), str(real / ".." / "x.jpg"), "/etc/passwd"):
        with pytest.raises(guards.DataRootViolation):
            runner.keyframe_path("synthetic", bad)
