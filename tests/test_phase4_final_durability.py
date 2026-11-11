"""Synthetic ledger durability/reservation checks; no GO or evaluator invocation."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cgt.p4_camera_move import final_eval as fe
from cgt.p4_camera_move import final_stores as fs
from cgt.p4_camera_move.camerabench import CameraBenchAlreadyEvaluated


@pytest.fixture
def sealed_world(tmp_path):
    repo = tmp_path / "repo"
    freeze = repo / "results/phase4/runs/video_v1/FREEZE.json"
    freeze.parent.mkdir(parents=True)
    freeze.write_text(json.dumps({"registered_models": {
        fe.MODEL_TAGS["geometry"]: {"config_hash": "c" * 16},
    }}))
    for args in (("init", "-q"), ("add", str(freeze)),
                 ("-c", "user.name=synthetic", "-c", "user.email=synthetic@invalid",
                  "commit", "-qm", "synthetic freeze")):
        subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True)
    labels = tmp_path / "synthetic_labels/test.jsonl"
    labels.parent.mkdir()
    labels.write_text(json.dumps({"path": "videos_gif/toy.gif", "labels": ["static"]}) + "\n")
    ledger = tmp_path / "synthetic_ledger.jsonl"
    store = fs.CameraBenchFinalStore(root=labels.parent, ledger=ledger,
                                    freeze_path=freeze, repo_root=repo)
    return store, labels, ledger


@pytest.mark.parametrize("fail_directory", [False, True])
def test_failed_ledger_fsync_never_opens_synthetic_labels(sealed_world, monkeypatch, fail_directory):
    store, labels, ledger = sealed_world
    opened = []
    real_read, real_sync = Path.read_text, os.fsync

    def read(path, *args, **kwargs):
        if path == labels:
            opened.append(path)
        return real_read(path, *args, **kwargs)

    def sync(fd):
        actual = os.fstat(fd)
        target = (ledger.parent if fail_directory else ledger).stat()
        if (actual.st_dev, actual.st_ino) == (target.st_dev, target.st_ino):
            raise OSError("synthetic durability failure")
        return real_sync(fd)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(os, "fsync", sync)
    kwargs = {"final_evaluation": True, "model_tag": fe.MODEL_TAGS["geometry"],
              "config_hash": "c" * 16}
    with pytest.raises(OSError, match="synthetic durability failure"):
        store.labels(**kwargs)
    assert opened == []
    # A partially appended line must not authorize a fresh label read on retry.
    monkeypatch.setattr(os, "fsync", real_sync)
    with pytest.raises(CameraBenchAlreadyEvaluated):
        store.labels(**kwargs)
    assert opened == []


def test_ledger_reservation_contends_with_another_process(tmp_path):
    ledger = tmp_path / "synthetic_ledger.jsonl"
    lock_path = Path(str(ledger) + ".run.lock")
    child_code = (
        "import fcntl,sys\n"
        "with open(sys.argv[1], 'a') as handle:\n"
        " fcntl.flock(handle, fcntl.LOCK_EX)\n"
        " print('owned', flush=True)\n"
        " sys.stdin.readline()\n"
    )
    child = subprocess.Popen([sys.executable, "-c", child_code, str(lock_path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "owned"
        with pytest.raises(BlockingIOError), fs.ledger_lock(ledger, ".run.lock", blocking=False):
            pytest.fail("reservation admitted a second process")
    finally:
        child.communicate("release\n", timeout=10)
    assert child.returncode == 0
    with fs.ledger_lock(ledger, ".run.lock", blocking=False):
        pass
