"""Verification pass an earlier review checks of the orchestrator movement stage (branch a sibling working copy @b274fe3). Synthetic film, fake runner; no data, no GPU.
Skipped on a tree without the movement stage (main before the merge)."""
import json

import pandas as pd
import pytest
import yaml

stages = pytest.importorskip("cgt.pipeline.stages")
if not hasattr(stages, "move_table_verified"):
    pytest.skip("movement stage not on this tree", allow_module_level=True)
from cgt.pipeline import orchestrator as orch
from cgt.pipeline.orchestrator import Ctx


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path
    (root / "configs").mkdir()
    (root / "configs/films.yaml").write_text(yaml.safe_dump({"films": [{"id": "pd", "public_domain": True, "path": "pd.mp4"}]}))
    (root / "data/processed").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "scripts/run_move_film.py").write_text("# fake runner")
    (root / "data/cache/p4").mkdir(parents=True)
    for rel in stages.MOVE_CODE_FILES + stages.MOVE_CONFIG_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(f"v1 {rel}".encode())
    for rel in stages.MOVE_PREREQ_FILES:  # fake empty ShotQA train label file (move prerequisite since F1, docs/move_prereq_fix_2026-10-13.md; never read)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"")
    (root / "pd.mp4").write_bytes(b"video-pd")
    pd.DataFrame({"shot_id": [0, 1, 2], "start_s": [0.0, 1.0, 2.0], "end_s": [1.0, 2.0, 3.0], "len_s": 1.0}).to_parquet(root / "data/processed/pd_shots.parquet")
    monkeypatch.setattr(stages, "_free_percent", lambda: 80)
    hook = {}

    def fake_script(ctx, cmd):
        pd.DataFrame({"shot_id": [0, 1, 2], "move_atoms": ["static", "push-in + pan-left", "pan-left"], "fallback_top1": [False, False, True], "no_features": False}).to_parquet(stages._move_table(ctx))
        if "during" in hook:
            hook["during"](ctx)
        return 0
    monkeypatch.setattr(stages, "_run_move_script", fake_script)
    monkeypatch.setattr("cgt.p4_camera_move.film_move.describe", lambda sets, fb: {"n_shots": len(sets), "median_set_size": 1.0, "contradictory_share": 0.0})
    return root, hook


def ctx_for(root):
    return Ctx("pd", {"id": "pd", "public_domain": True, "path": "pd.mp4"}, root=root, allow_owned=False, log=lambda m: None)


@pytest.mark.parametrize("rel", ["scripts/run_move_film.py", "src/cgt/p4_camera_move/film_move.py", "src/cgt/p4_camera_move/flow.py", "src/cgt/p4_camera_move/depth_parallax.py",
                                 "src/cgt/p4_camera_move/final_run.py", "src/cgt/p4_camera_move/final_eval.py", "scripts/p4_final_eval.py",
                                 "results/phase4/frozen_config_flow_depth.json", "results/phase4/split_manifest.json",
                                 "data/cache/p4/features_shotqa.parquet", "data/cache/p4/features_depth_shotqa.parquet"])
def test_every_claimed_movement_input_file_invalidates_the_table(env, rel):
    """Criterion 2 claims the identity binds the runner, film_move, flow, depth parallax and evaluation sources and the frozen configuration: EACH file, not only one of them.
   """
    root, _ = env
    ctx = ctx_for(root)
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None) == 0
    assert stages.move_table_verified(ctx)[0]
    (root / rel).write_bytes(b"changed " + rel.encode())
    ok, why = stages.move_table_verified(ctx)
    assert not ok and "code identity" in why


def test_identity_record_binds_the_inputs_seen_before_the_run(env):
    root, hook = env
    ctx = ctx_for(root)
    hook["during"] = lambda c: (root / "src/cgt/p4_camera_move/flow.py").write_bytes(b"edited while the runner ran")
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None) != 0 or not stages.move_table_verified(ctx)[0]


def test_tokens_refuse_an_identity_less_table_even_when_the_runner_is_absent(env):
    root, _ = env
    ctx = ctx_for(root)
    for name in ("p3c", "p2"):
        pd.DataFrame({"shot_id": [0, 1, 2]}).to_parquet(root / f"data/processed/pd_{name}.parquet")
    (root / "data/cache/tokens").mkdir(parents=True)
    (root / "data/cache/tokens/palette_global_k32.npz").write_bytes(b"x")
    pd.DataFrame({"shot_id": [0, 1, 2], "move_atoms": ["static"] * 3, "fallback_top1": False, "no_features": False}).to_parquet(stages._move_table(ctx))
    (root / "scripts/run_move_film.py").unlink()
    assert any("move" in m for _, m in stages.tokens_prereq(ctx))


def test_identity_record_fields(env):
    root, _ = env
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None) == 0
    r = json.loads(stages._move_record_path(ctx_for(root)).read_text())
    assert {"built_by", "video_sha256", "shots_table_sha256", "code_identity", "table_sha256"} <= set(r) and r["built_by"] == "move_stage"
