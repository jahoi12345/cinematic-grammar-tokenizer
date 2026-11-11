"""Phase4 metadata/label-lock/freeze/look-order contract, synthetic data only.

The integration checks fit tiny CPU sklearn models. No media, external labels,
GPU, real result path or real look ledger is accessed.
"""
import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from cgt.p4_camera_move import baseline, labels, split


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    with threadpool_limits(limits=1):
        yield


def _entry(clip, answer="A"):
    return {"videos": [f"data/{clip}"],
            "messages": [{"role": "user", "content":
                          "Question: movement?\nOptions:\nA. Static shot\nB. Pan left\n"
                          "C. Tilt up\nD. Push in\nPlease choose.\n"}],
            "solution": f"<answer> {answer} </answer>"}


def test_metadata_loader_ignores_invalid_solutions_options_and_label_parser(tmp_path, monkeypatch):
    entries = [{"videos": ["data/Z.webm_2.mp4"], "solution": {"poison": True}, "messages": None},
               {"images": ["not-a-video.jpg"], "solution": None},
               {"videos": ["data/A.mkv_7.mp4"]}]
    path = tmp_path / "grpo.json"
    path.write_text(json.dumps(entries))

    def forbidden_parse(*args, **kwargs):
        pytest.fail("metadata-only loader reached a label parser")

    monkeypatch.setattr(labels, "_row_from_entry", forbidden_parse)
    monkeypatch.setattr(labels, "parse_move_label", forbidden_parse)
    assert labels.load_shotqa_movement_meta(path) == [
        {"clip": "Z.webm_2.mp4", "source": "Z"}, {"clip": "A.mkv_7.mp4", "source": "A"},
    ]


def test_store_locks_validation_and_mixed_requests_before_any_parse(tmp_path):
    path = tmp_path / "grpo.json"
    path.write_text(json.dumps([_entry("T.webm_1.mp4"), _entry("V.webm_1.mp4", "B")]))
    events = []
    store = labels.MovementLabelStore(["V.webm_1.mp4"], path, events=events)
    for request in (["V.webm_1.mp4"], ["T.webm_1.mp4", "V.webm_1.mp4"]):
        with pytest.raises(labels.ValidationLabelsLocked):
            store.rows(request)
        assert events == []
    with pytest.raises(labels.ValidationLabelsLocked):
        store.validation_rows()
    assert store.rows(["T.webm_1.mp4"])[0]["atoms"] == ("static",)
    assert events == [("parse", "T.webm_1.mp4", "train")]
    assert store.validation_rows(final_evaluation=True)[0]["atoms"] == ("pan-left",)
    assert events[-1] == ("parse", "V.webm_1.mp4", "final")


def test_store_parses_only_requested_train_rows_and_poison_waits_for_final(tmp_path):
    path = tmp_path / "grpo.json"
    poison = _entry("V.webm_1.mp4")
    poison["solution"] = "poisoned validation answer"
    path.write_text(json.dumps([_entry("T.webm_1.mp4"), poison]))
    store = labels.MovementLabelStore(["V.webm_1.mp4"], path)
    assert store.rows(["T.webm_1.mp4"])[0]["gold_letter"] == "A"
    with pytest.raises(ValueError, match="unparseable solution"):
        store.validation_rows(final_evaluation=True)


@pytest.fixture
def synthetic_problem(tmp_path):
    rng = np.random.default_rng(4)
    entries, rows, features = [], [], {}
    for source in range(12):
        for label in range(4):
            clip = f"S{source:02}.webm_{label}.mp4"
            entry = _entry(clip, "ABCD"[label])
            if label == 3:
                entry["messages"][0]["content"] = entry["messages"][0]["content"].replace(
                    "D. Push in", "D. Push in and Pan left",
                )
            entries.append(entry)
            rows.append({"clip": clip, "source": f"S{source:02}"})
            features[clip] = np.eye(4)[label] * 3 + rng.normal(0, .2, 4)
    manifest = split.write_split_manifest(rows, tmp_path / "manifest.json")
    marker_clip = manifest["clips"]["val"][0]
    for entry in entries:
        if entry["videos"][0] == f"data/{marker_clip}":
            entry["messages"][0]["content"] += "PLANTED_VALIDATION_MARKER\n"
    grpo = tmp_path / "grpo.json"
    grpo.write_text(json.dumps(entries))
    feats = pd.DataFrame.from_dict(features, orient="index")
    return tmp_path, manifest, grpo, feats, marker_clip


def _run(problem, out, events, **kwargs):
    _, manifest, grpo, feats, _ = problem
    return baseline.run(feats=feats, manifest=manifest,
                        store=labels.MovementLabelStore(manifest["clips"]["val"], grpo, events=events),
                        feature_sets=["synthetic"], tag="synthetic", out_dir=out,
                        looks_path=out / "looks.jsonl", code_sha="synthetic-code", events=events,
                        cs=[.003, .03], **kwargs)


def test_planted_marker_is_unreachable_until_freeze_and_look_exist(synthetic_problem, monkeypatch):
    root, manifest, _, _, marker_clip = synthetic_problem
    out = root / "result"
    original = labels._row_from_entry
    seen = []

    def spy(entry):
        clip = entry["videos"][0].removeprefix("data/")
        if clip in manifest["clips"]["val"]:
            assert (out / "frozen_config_synthetic.json").is_file()
            assert len((out / "looks.jsonl").read_text().splitlines()) == 1
            seen.append(clip)
        return original(entry)

    monkeypatch.setattr(labels, "_row_from_entry", spy)
    events = []
    _run(synthetic_problem, out, events)
    freeze = events.index(("freeze", "synthetic"))
    assert all(event[2] == "train" for event in events[:freeze])
    assert marker_clip in seen and len(seen) == len(manifest["clips"]["val"])
    assert all(event[2] == "final" for event in events[freeze + 1:])
    cfg = json.loads((out / "frozen_config_synthetic.json").read_text())
    assert {"chosen_C", "cv_macro_ap_by_C", "config_hash", "atoms"} <= cfg.keys()
    assert "validation_per_atom" not in cfg


def test_poisoned_validation_changes_only_final_step_not_train_selection(synthetic_problem):
    root, _, grpo, _, marker_clip = synthetic_problem
    clean_events = []
    _run(synthetic_problem, root / "clean", clean_events)
    clean = json.loads((root / "clean/frozen_config_synthetic.json").read_text())
    entries = json.loads(grpo.read_text())
    for entry in entries:
        if entry["videos"][0] == f"data/{marker_clip}":
            entry["solution"] = "POISONED_VALIDATION_ONLY"
    grpo.write_text(json.dumps(entries))
    poisoned_events = []
    with pytest.raises(ValueError, match="unparseable solution"):
        _run(synthetic_problem, root / "poisoned", poisoned_events)
    poison = json.loads((root / "poisoned/frozen_config_synthetic.json").read_text())
    assert poison == clean
    freeze = poisoned_events.index(("freeze", "synthetic"))
    assert poisoned_events[:freeze + 1] == clean_events[:freeze + 1]
    assert len((root / "poisoned/looks.jsonl").read_text().splitlines()) == 1
    assert not (root / "poisoned/baseline_atoms_synthetic.json").exists()


def test_overwrite_guard_protects_final_files_and_reads_no_validation_label(synthetic_problem):
    root, _, _, _, _ = synthetic_problem
    out = root / "result"
    _run(synthetic_problem, out, [])
    files = [out / "baseline_atoms_synthetic.json", out / "baseline_atoms_synthetic.md",
             out / "val_probs_synthetic.json", out / "looks.jsonl"]
    old = {p.name: p.read_bytes() for p in files}
    events = []
    with pytest.raises(baseline.FinalResultExists):
        _run(synthetic_problem, out, events)
    assert all(p.read_bytes() == old[p.name] for p in files)
    assert not any(e[0] == "parse" and e[2] == "final" for e in events)


def test_explicit_overwrite_is_logged_before_validation_labels_are_read(synthetic_problem, monkeypatch):
    root, manifest, _, _, _ = synthetic_problem
    out = root / "result"
    _run(synthetic_problem, out, [])
    original = labels._row_from_entry

    def spy(entry):
        if entry["videos"][0].removeprefix("data/") in manifest["clips"]["val"]:
            assert len((out / "looks.jsonl").read_text().splitlines()) == 2
        return original(entry)

    monkeypatch.setattr(labels, "_row_from_entry", spy)
    _run(synthetic_problem, out, [], allow_overwrite_final=True)
    looks = [json.loads(line) for line in (out / "looks.jsonl").read_text().splitlines()]
    assert len(looks) == 2 and looks[1]["overwrite_of_prior_final"] is True
    assert looks[0]["overwrite_of_prior_final"] is False
    assert looks[1]["n_logged_looks_so_far"] == looks[0]["n_logged_looks_so_far"] + 1
    assert looks[1]["n_validation_looks_so_far"] == looks[0]["n_validation_looks_so_far"] + 1
    for look in looks:
        assert {"timestamp", "code_sha", "feature_set", "config_hash",
                "historical_looks_declared_unverified"} <= look.keys()


def test_comparison_logs_a_validation_look_before_reading_labels(tmp_path, monkeypatch):
    """Contract section4: every invocation reading validation labels is a look."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "synthetic_p4_comparison", Path(__file__).resolve().parents[1] / "scripts/p4_compare_baselines.py",
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(baseline, "code_sha", lambda: "synthetic-commit")
    folder = tmp_path / "results/phase4"
    folder.mkdir(parents=True)
    clips = ["T.webm_1.mp4", "V.webm_1.mp4"]
    (folder / "split_manifest.json").write_text(json.dumps({"clips": {"val": clips}}))
    (folder / "val_probs_flow.json").write_text(json.dumps({"atoms": ["static"], "clips": clips,
                                                           "probs": [[.9], [.8]]}))
    (folder / "frozen_config_flow.json").write_text(json.dumps({
        "feature_sets": ["flow"], "n_features": 119, "chosen_C": .03, "cv_macro_ap_at_chosen_C": .4,
    }))
    (folder / "baseline_atoms_flow.json").write_text(json.dumps({"confusions_train_oof": []}))
    ledger = folder / "validation_looks.jsonl"
    ledger.write_text(json.dumps({"tag": "previous", "n_logged_looks_so_far": 1}) + "\n")

    class Store:
        def __init__(self, validation_clips):
            assert validation_clips == clips

        def rows(self, requested, *, final_evaluation=False):
            assert final_evaluation is True and requested == clips
            assert len(ledger.read_text().splitlines()) == 2, "comparison reads labels without logging its look"
            return [{"clip": clip, "source": clip[0], "atoms": ("static",)} for clip in clips]

    monkeypatch.setattr(script, "MovementLabelStore", Store)
    # This test checks label-access accounting, not metric arithmetic.
    keys = ["macro_f1", "macro_ap", *script.GROUPS, *script.AUC_GROUPS]
    monkeypatch.setattr(script, "metrics", lambda *args: dict.fromkeys(keys, .5))
    monkeypatch.setattr("sys.argv", ["p4_compare_baselines.py", "--runs", "flow", "--ref", "flow"])
    script.main()


def test_refused_changed_config_cannot_replace_freeze_or_parse_train(synthetic_problem):
    root, manifest, grpo, feats, _ = synthetic_problem
    out = root / "result"
    _run(synthetic_problem, out, [])
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    events = []
    with pytest.raises(baseline.FinalResultExists):
        baseline.run(feats=feats, manifest=manifest,
                     store=labels.MovementLabelStore(manifest["clips"]["val"], grpo, events=events),
                     feature_sets=["synthetic"], tag="synthetic", out_dir=out,
                     looks_path=out / "looks.jsonl", code_sha="changed-code", cs=[.0001], events=events)
    assert events == []
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
