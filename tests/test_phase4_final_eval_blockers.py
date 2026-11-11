"""Regression tests of the blockers E1-E5 of the independent evaluator verification (docs/phase4_video_plan.md Amendment 12). Synthetic worlds only: no model, no real test data, no real ledger."""
import importlib.util
import itertools
import os
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from cgt.p3b_vlm import guards
from cgt.p4_camera_move import final_eval as fe
from cgt.p4_camera_move import final_run as fr
from cgt.p4_camera_move import final_stores as fs
from cgt.p4_camera_move.camerabench import CameraBenchAlreadyEvaluated

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def world(tmp_path):
    demo = script("p4_final_eval_selfcheck")
    return demo.make_world(tmp_path), demo


def cb_store(w, cls=fs.CameraBenchFinalStore):
    return cls(root=w["cb_root"], ledger=w["ledger"], freeze_path=w["freeze"], repo_root=w["repo"])


def label_args():
    return {"final_evaluation": True, "model_tag": fe.MODEL_TAGS["geometry"], "config_hash": "c" * 16}


def test_camerabench_ledger_is_fsynced_before_label_open(world, monkeypatch):
    w, _ = world
    synced = []
    real_sync, real_read = os.fsync, Path.read_text

    def sync(fd):
        st, lst = os.fstat(fd), w["ledger"].stat()
        if (st.st_dev, st.st_ino) == (lst.st_dev, lst.st_ino):
            synced.append(fd)
        return real_sync(fd)

    def read(path, *a, **k):
        if path == w["cb_root"] / "test.jsonl":
            assert synced, "ledger must be durable BEFORE labels are opened"
            assert fs.ledger_contains(w["ledger"], fs.CAMERABENCH, fe.MODEL_TAGS["geometry"])
        return real_read(path, *a, **k)
    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(Path, "read_text", read)
    cb_store(w).labels(**label_args())


@pytest.mark.parametrize("kind", ["shotbench", "camerabench"])
def test_labels_are_never_opened_after_a_failed_fsync(world, monkeypatch, kind):
    w, _ = world
    opened = []
    real_read = Path.read_text
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **k: (opened.append(p.name) if p.name == "test.jsonl" else None, real_read(p, *a, **k))[1])
    if kind == "camerabench":
        with pytest.raises(OSError):
            cb_store(w).labels(**label_args())
    else:
        import pandas as pd
        real_csv = pd.read_csv
        monkeypatch.setattr(pd, "read_csv", lambda path, *a, **k: (opened.append("answer") if "answer" in (k.get("usecols") or []) else None, real_csv(path, *a, **k))[1])
        store = fs.ShotBenchMovementStore(tsv=w["tsv"], ledger=w["ledger"], freeze_path=w["freeze"], repo_root=w["repo"], expected_n=12)
        with pytest.raises(OSError):
            store.labels(**label_args())
    assert opened == []


def test_duplicate_concurrent_label_reads_reserve_one_dataset_tag(world):
    w, _ = world
    barrier = Barrier(2)

    class ConcurrentStore(fs.CameraBenchFinalStore):
        def _evaluated(self, dataset, tag):
            found = super()._evaluated(dataset, tag)
            if not found:
                time.sleep(0.2)  # widen the legitimate check/append race; a serialized store still passes
            return found

    def read():
        store = cb_store(w, ConcurrentStore)
        barrier.wait(timeout=10)
        try:
            store.labels(**label_args())
            return "read"
        except CameraBenchAlreadyEvaluated:
            return "refused"
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: read(), range(2)))
    assert sorted(outcomes) == ["read", "refused"], outcomes
    assert len(fs.ledger_lines(w["ledger"])) == 1


def test_second_evaluation_process_refuses_while_another_holds_the_reservation(world, tmp_path):
    w, demo = world
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        from cgt.p4_camera_move import final_stores as fs
        with fs.ledger_lock(Path({str(w['ledger'])!r}), '.run.lock', blocking=False):
            print('held', flush=True); time.sleep(30)""")], stdout=subprocess.PIPE, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    try:
        assert holder.stdout.readline().strip() == "held"
        trace = []
        code, _, _ = demo.run_once(w, "camerabench_public_test", "geometry", tmp_path / "run", trace=trace) if (tmp_path / "run").mkdir() is None else (0, 0, 0)
        assert code == 14 and not fs.ledger_lines(w["ledger"]) and trace == []  # refused before any backend work, label read or ledger line
    finally:
        holder.kill()
        holder.wait()
    (tmp_path / "run2").mkdir()
    assert demo.run_once(w, "camerabench_public_test", "geometry", tmp_path / "run2")[0] == 0  # the lock died with its holder


def test_run_reservation_is_global_across_datasets_and_models(world, tmp_path):
    w, demo = world
    with fs.ledger_lock(w["ledger"], ".run.lock", blocking=False):
        # the lock is per ledger file: ANY of the six evaluations refuses, geometry included
        for i, (d, m) in enumerate([("shotbench_movement", "geometry"), ("camerabench_public_test", "combined")]):
            (tmp_path / f"r{i}").mkdir()
            code = _run_in_thread(demo, w, d, m, tmp_path / f"r{i}")
            assert code == 14


def _run_in_thread(demo, w, d, m, d_):
    # flock locks belong to the open file description, so a second open in another thread conflicts like another process
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: demo.run_once(w, d, m, d_)[0]).result()


def test_guards_are_unchanged_by_the_reservation(world, tmp_path):
    w, demo = world
    assert demo.run_once(w, "shotbench_movement", "geometry", tmp_path)[0] == 0
    (tmp_path / "again").mkdir()
    assert demo.run_once(w, "shotbench_movement", "geometry", tmp_path / "again")[0] == 14  # the ledger still refuses a second evaluation
    assert guards is not None and fr is not None


# ------------------------------------------------------------------------------------------------ E2: frozen geometry identity
def _commit(repo, paths, msg="x"):
    for cmd in (["add", *[str(p) for p in paths]], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", msg]):
        subprocess.run(["git", *cmd], cwd=repo, check=True, capture_output=True)


def _geo_repo(w, cfg=None):
    import json
    repo = w["repo"]
    path = repo / fr.GEOMETRY_CONFIG
    path.write_text(json.dumps(cfg or {"chosen_C": 0.01, "config_hash": "c" * 16}))
    (repo / "results/phase4").mkdir(parents=True, exist_ok=True)
    (repo / fr.SPLIT_MANIFEST).write_text(json.dumps({"sha256": "m", "clips": {}}))
    _commit(repo, [path, repo / fr.SPLIT_MANIFEST])
    return repo, path


def test_geometry_edited_parameters_are_refused_for_geometry_and_combined(world):
    import json
    w, _ = world
    repo, path = _geo_repo(w)
    assert fr.geometry_identity(repo)["C"] == 0.01
    path.write_text(json.dumps({"chosen_C": 1000.0, "config_hash": "c" * 16}))
    with pytest.raises(guards.DirtyTree):
        fr.geometry_identity(repo)  # make_spec('geometry') and make_spec('combined') both go through this
    with pytest.raises(guards.DirtyTree):
        fr.make_spec("shotbench_movement", "geometry", root=repo)


def test_geometry_config_must_carry_the_hash_of_its_content(world):
    import json
    w, _ = world
    cfg = {"tag": "t", "c_grid": [0.01], "chosen_C": 0.01, "n_features": 3, "config_hash": "0" * 16}
    repo, _ = _geo_repo(w, cfg)
    with pytest.raises(guards.TagNotBound):
        fr.geometry_identity(repo)
    from cgt.p4_camera_move.baseline import config_hash
    cfg["config_hash"] = config_hash({k: v for k, v in cfg.items() if k != "config_hash"})
    (repo / fr.GEOMETRY_CONFIG).write_text(json.dumps(cfg))
    _commit(repo, [repo / fr.GEOMETRY_CONFIG], "fix")
    assert fr.geometry_identity(repo)["config_hash"] == cfg["config_hash"]


def test_go_file_binds_the_geometry_inputs(world):
    import json
    w, _ = world
    repo, _ = _geo_repo(w)
    freeze = json.loads(w["freeze"].read_text())
    freeze.update({"tag": "synthetic", "adapter": {"file": "a.safetensors", "sha256": "0" * 64}})
    w["freeze"].write_text(json.dumps(freeze))
    _commit(repo, [w["freeze"]], "freeze")
    before, _, _ = fr.make_spec("shotbench_movement", "geometry", root=repo)
    table = repo / fr.TRAIN_TABLES[0]
    table.parent.mkdir(parents=True)
    table.write_bytes(b"features v1")
    changed, _, _ = fr.make_spec("shotbench_movement", "geometry", root=repo)
    assert before.geometry_inputs != changed.geometry_inputs and before.go() != changed.go()
    table.write_bytes(b"features v2")
    again, _, _ = fr.make_spec("shotbench_movement", "geometry", root=repo)
    assert again.geometry_inputs != changed.geometry_inputs
    with pytest.raises(guards.NotAuthorized):  # a go file written for the old inputs authorizes nothing
        go = repo / "GO"
        go.write_text(changed.go())
        fe.require_go(go, again.go())


def test_edited_training_split_is_refused_and_the_committed_split_is_the_one_returned(world):
    import json
    w, _ = world
    repo, _ = _geo_repo(w)
    committed = json.loads((repo / fr.SPLIT_MANIFEST).read_text())
    assert fr.geometry_identity(repo)["split"] == committed
    (repo / fr.SPLIT_MANIFEST).write_text(json.dumps({"sha256": "m", "clips": {"train": ["swapped"], "val": ["x"]}}))  # train and validation lists swapped in the working copy
    with pytest.raises(guards.DirtyTree):
        fr.geometry_identity(repo)


def test_split_manifest_must_be_the_one_the_geometry_config_was_frozen_with(world):
    import json
    w, _ = world
    from cgt.p4_camera_move.baseline import config_hash
    cfg = {"tag": "t", "c_grid": [0.01], "chosen_C": 0.01, "manifest_sha256": "other", "config_hash": ""}
    cfg["config_hash"] = config_hash({k: v for k, v in cfg.items() if k != "config_hash"})
    repo, _ = _geo_repo(w, cfg)
    assert json.loads((repo / fr.SPLIT_MANIFEST).read_text())["sha256"] == "m"
    with pytest.raises(guards.TagNotBound):
        fr.geometry_identity(repo)


# ------------------------------------------------------------------------------------------------ E5: input conformity (fps-conform-v1)
def _avi(path, n, fps):
    import cv2
    import numpy as np
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (64, 48))
    for i in range(n):
        wr.write(np.full((48, 64, 3), (0, 0, i), np.uint8))  # the frame index in the red channel (BGR order)
    wr.release()


def _read_idx(path):
    import cv2
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, fr = cap.read()
        if not ok:
            return out
        out.append(int(fr[0, 0, 2]))


@pytest.mark.parametrize("fps", [29.97, 25.0, 60.0, 14.3])
def test_conformed_video_is_resampled_in_time_and_sampled_at_the_training_rate(tmp_path, fps):
    from cgt.p4_camera_move import conform as C
    from cgt.p4_camera_move import flow as F
    n = 90
    _avi(tmp_path / "src.avi", n, fps)
    k = C.conform_video(tmp_path / "src.avi", tmp_path / "out.avi")
    got = _read_idx(tmp_path / "out.avi")
    want = C.video_source_indices(n, fps)
    assert k == len(got) == len(want) and got == want
    assert all(b >= a for a, b in itertools.pairwise(got))  # time never runs backwards
    frames, fps_used = F.sample_gray(tmp_path / "out.avi")
    assert abs(fps_used - 11.988) < 0.002 and len(frames) >= 3  # exactly the training rate, whatever the source rate


def test_conformed_gif_follows_the_frame_durations(tmp_path):
    from PIL import Image

    from cgt.p4_camera_move import conform as C
    from cgt.p4_camera_move import flow as F
    durs = [100, 40, 40, 200, 100, 100, 60, 60, 300, 100] * 3
    frames = [Image.new("RGB", (64, 48), (10 * (i % 20), 0, 0)) for i in range(len(durs))]
    g = tmp_path / "v.gif"
    frames[0].save(g, save_all=True, append_images=frames[1:], duration=durs, loop=0)
    idx = C.gif_source_indices([float(d) for d in durs])
    starts = [sum(durs[:i]) / 1000 for i in range(len(durs))]
    for j, i in enumerate(idx):  # the frame displayed at j / 23.976 s
        t = j / C.TARGET_FPS
        assert starts[i] <= t + 1e-9 and (i + 1 == len(durs) or starts[i + 1] > t)
    assert C.gif_source_indices([0.0] * 40)[:3] == [0, 0, 0]  # a zero duration is shown for 100 ms
    C.conform_gif(g, tmp_path / "out.avi")
    assert abs(F.sample_gray(tmp_path / "out.avi")[1] - 11.988) < 0.002


def test_conformity_provenance_carries_version_code_hash_and_extractor_keys():
    from cgt.p4_camera_move import conform as C
    p = C.provenance()
    assert p["conform_version"] == "fps-conform-v1" and len(p["conform_code_sha256"]) == 16 and p["flow_key"] and p["depth_key"] and abs(p["target_fps"] - 24000 / 1001) < 1e-12


# ------------------------------------------------------------------------------------------------ E4: prepared-input provenance
@pytest.fixture
def prepared(tmp_path):
    import json

    import pandas as pd

    from cgt.p4_camera_move import final_inputs as fi
    clips = ["a.gif", "b.gif", "c.gif"]
    inputs = tmp_path / "gifs"
    inputs.mkdir()
    for i, c in enumerate(clips):
        (inputs / c).write_bytes(f"gif-bytes-{i}".encode())
    cols = ["tx_mean", "ty_mean"]
    table = tmp_path / "features.parquet"
    pd.DataFrame({"clip": clips, "tx_mean": [0.1, 0.2, 0.3], "ty_mean": [1.0, 2.0, 3.0]}).to_parquet(table, index=False)
    man_path = tmp_path / "prepare_camerabench_public_test.json"
    man = fi.build_manifest("camerabench_public_test", [inputs / c for c in clips], table, cols, root=tmp_path)
    man_path.write_text(json.dumps(man))
    kw = {"dataset": "camerabench_public_test", "clips": clips, "input_dir": inputs, "table": table, "manifest_path": man_path, "train_columns": cols, "root": tmp_path}
    return fi, kw, man, man_path, table, inputs


def test_a_complete_preparation_is_accepted(prepared):
    fi, kw, man, *_ = prepared
    assert fi.validate_prepared(**kw)["labels_read"] is False
    assert fi.validate_prepared(**{**kw, "table": None})["n_files"] == 3  # the video model validates the input bytes only
    assert man["provenance"]["conform_version"] == "fps-conform-v1" and man["provenance"]["conform_code_sha256"]


def test_prepared_feature_table_digest_is_checked_before_use_here(prepared):
    import pandas as pd
    fi, kw, _, _, table, _ = prepared
    pd.DataFrame({"clip": ["a.gif", "b.gif", "c.gif"], "tx_mean": [9.0, 0.2, 0.3], "ty_mean": [1.0, 2.0, 3.0]}).to_parquet(table, index=False)
    with pytest.raises(guards.UnreadableImage):
        fi.validate_prepared(**kw)


def test_input_bytes_are_hashed_every_time(prepared):
    fi, kw, *_, inputs = prepared
    (inputs / "b.gif").write_bytes(b"replaced")
    with pytest.raises(guards.UnreadableImage):
        fi.validate_prepared(**kw)
    with pytest.raises(guards.UnreadableImage):
        fi.validate_prepared(**{**kw, "table": None})


@pytest.mark.parametrize("bad", ["missing_clip", "duplicate_clip", "extra_clip", "limit_manifest"])
def test_coverage_must_be_complete_and_unique(prepared, bad):
    import json

    import pandas as pd
    fi, kw, man, man_path, _table, _ = prepared
    if bad == "missing_clip":
        kw = {**kw, "clips": kw["clips"] + ["d.gif"]}
    elif bad == "duplicate_clip":
        kw = {**kw, "clips": kw["clips"] + ["a.gif"]}
    elif bad == "extra_clip":
        kw = {**kw, "clips": kw["clips"][:2]}
    else:
        man_path.write_text(json.dumps({**man, "n_files": 2}))
    assert pd is not None
    with pytest.raises(guards.UnreadableImage):
        fi.validate_prepared(**kw)


def test_stale_or_missing_provenance_and_conformity_are_refused(prepared):
    import json
    fi, kw, man, man_path, *_ = prepared
    for edit in ({"conform_code_sha256": "0" * 16}, {"conform_version": "fps-conform-v0"}, {"flow_key": "x"}, {"prepare_script_sha256": "x"}):
        man_path.write_text(json.dumps({**man, "provenance": {**man["provenance"], **edit}}))
        with pytest.raises(guards.UnreadableImage):
            fi.validate_prepared(**kw)
    no_conform = {k: v for k, v in man["provenance"].items() if k != "conform_version"}
    man_path.write_text(json.dumps({**man, "provenance": no_conform}))
    with pytest.raises(guards.UnreadableImage):
        fi.validate_prepared(**kw)
    man_path.write_text(json.dumps({k: v for k, v in man.items() if k != "provenance"}))
    with pytest.raises(guards.UnreadableImage):
        fi.validate_prepared(**kw)


def test_feature_columns_must_equal_the_training_columns(prepared):
    fi, kw, *_ = prepared
    with pytest.raises(guards.UnreadableImage):
        fi.validate_prepared(**{**kw, "train_columns": ["ty_mean", "tx_mean"]})


# ------------------------------------------------------------------------------------------------ frame extraction fallback (Amendment 14)
def test_sequential_fallback_gives_the_same_frames_as_the_seeking_extractor(tmp_path, monkeypatch):
    from cgt.p4_camera_move import frames_fallback as ff
    from cgt.p4_camera_move import video_vlm as V
    _avi(tmp_path / "v.avi", 60, 24.0)
    normal = V.extract_frames(tmp_path / "v.avi", 8, 64, tmp_path / "c1")
    monkeypatch.setattr(V, "extract_frames", lambda *a, **k: (_ for _ in ()).throw(ValueError("frame 58 unreadable")))
    fallback = ff.extract_frames_robust(tmp_path / "v.avi", 8, 64, tmp_path / "c2")
    assert [p.name for p in normal] == [p.name for p in fallback]
    assert [p.read_bytes() for p in normal] == [p.read_bytes() for p in fallback]


def test_fallback_refuses_a_video_with_fewer_decodable_frames_than_k(tmp_path, monkeypatch):
    from cgt.p4_camera_move import frames_fallback as ff
    from cgt.p4_camera_move import video_vlm as V
    _avi(tmp_path / "s.avi", 5, 24.0)
    monkeypatch.setattr(V, "extract_frames", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    with pytest.raises(ValueError):
        ff.extract_frames_robust(tmp_path / "s.avi", 8, 64, tmp_path / "c")


def test_a_gif_with_fewer_frames_than_k_is_kept_with_repeated_frames(tmp_path):
    from PIL import Image

    from cgt.p4_camera_move import gif as G
    from cgt.p4_camera_move import video_vlm as V
    frames = [Image.new("RGB", (96, 64), (30 * i, 0, 0)) for i in range(7)]
    g = tmp_path / "seven.gif"
    frames[0].save(g, save_all=True, append_images=frames[1:], duration=70, loop=0)
    out = G.gif_frames_k(g, 8, 64, tmp_path / "fr")
    import numpy as np
    reds = [round(float(np.asarray(Image.open(p)).mean(axis=(0, 1))[0]) / 30) for p in out]
    assert len(out) == 8 and reds == V.frame_indices(7, 8) and len(set(reds)) < 8  # repeats, in time order
