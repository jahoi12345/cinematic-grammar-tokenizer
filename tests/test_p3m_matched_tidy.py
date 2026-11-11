"""an earlier change tidy 2026-10-15: an earlier review advisories A9 (crash check also in the canonical raw dir) and A10 (a refused run never appends to the
committed canonical raw stdout). SYNTHETIC tags only, registered in-process; no GO file is written; ShotBench is never read; the real
results/p3m_matched/raw_eval_logs is never written (the script's canonical dir is patched to a tmp dir)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from cgt.p3b_vlm import guards, sealed
from cgt.p3m_matched import registry, sealed_eval

ROOT = Path(__file__).resolve().parents[1]
TAG = "synthetic_tidy_w1_t"
ENTRY = {"repo": "x/y", "revision": "0" * 40, "adapter_rel": "data/cache/synthetic_does_not_exist.safetensors", "lora": None,
         "identity_sha256": "0" * 64, "identity_kind": "adapter_sha256", "image_spec": {"kind": "processor_default"}, "variant": "reasoning"}


def _clean(_paths):
    return {"commit": "0" * 40, "dirty_files": []}


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setitem(registry.MODELS, TAG, dict(ENTRY))
    raw, canon = tmp_path / "raw", tmp_path / "canon"
    raw.mkdir()
    canon.mkdir()
    return {"raw": raw, "canon": canon, "go": tmp_path / "GO_none", "ledger": tmp_path / "ledger.json", "real_ledger": tmp_path / "real_ledger.json"}


def _preflight(d):
    return sealed_eval.preflight(TAG, d["go"], d["ledger"], d["raw"], clean_tree=_clean, real_ledger=d["real_ledger"], real_raw_dir=d["canon"])


# ---------------- A9 ----------------
@pytest.mark.parametrize("marker", [sealed.crash_path, sealed.in_progress_path, sealed.unknown_path])
def test_crash_marker_in_canonical_raw_dir_refuses_with_scratch_raw_dir(dirs, marker):
    """A crash / killed-run / UNKNOWN-commit marker only in the CANONICAL raw dir refuses (16) although --raw-dir names a clean scratch dir."""
    marker(dirs["canon"], TAG).write_text("{}")
    with pytest.raises(guards.CrashUnacknowledged):
        _preflight(dirs)
    assert marker(dirs["canon"], TAG).exists()  # nothing archived on refusal


def test_crash_marker_in_given_raw_dir_still_refuses(dirs):
    sealed.crash_path(dirs["raw"], TAG).write_text("{}")
    with pytest.raises(guards.CrashUnacknowledged):
        _preflight(dirs)


def test_acknowledged_canonical_crash_reaches_go_check_and_is_archived(dirs):
    sealed.crash_path(dirs["canon"], TAG).write_text("{}")
    sealed.ack_path(dirs["canon"], TAG).write_text("")
    with pytest.raises(guards.NotAuthorized):  # control: acknowledged -> on to the GO check (no GO exists)
        _preflight(dirs)
    assert not sealed.crash_path(dirs["canon"], TAG).exists() and not sealed.ack_path(dirs["canon"], TAG).exists()
    assert list(dirs["canon"].glob(f"{TAG}.CRASH.json.acknowledged.*"))


@pytest.mark.parametrize("marker", [sealed.crash_path, sealed.in_progress_path, sealed.unknown_path])
def test_refusal_in_one_dir_does_not_consume_the_other_dirs_ack(dirs, marker):
    """Given dir: crash + ACK; canonical dir: an unacknowledged marker -> refused, and the given dir's ACK and crash file are left untouched."""
    sealed.crash_path(dirs["raw"], TAG).write_text("{}")
    sealed.ack_path(dirs["raw"], TAG).write_text("")
    marker(dirs["canon"], TAG).write_text("{}")
    with pytest.raises(guards.CrashUnacknowledged):
        _preflight(dirs)
    assert sealed.ack_path(dirs["raw"], TAG).exists() and sealed.crash_path(dirs["raw"], TAG).exists()
    assert not list(dirs["raw"].glob("*.acknowledged.*"))


def test_unknown_commit_ack_does_not_cover_a_crash_in_the_other_dir(dirs):
    sealed.unknown_path(dirs["raw"], TAG).write_text("{}")
    sealed.unknown_ack_path(dirs["raw"], TAG).write_text("")
    sealed.in_progress_path(dirs["canon"], TAG).write_text("{}")
    with pytest.raises(guards.CrashUnacknowledged):
        _preflight(dirs)
    assert sealed.unknown_ack_path(dirs["raw"], TAG).exists() and sealed.unknown_path(dirs["raw"], TAG).exists()


def test_same_dir_given_twice_is_fine(dirs):
    sealed.crash_path(dirs["canon"], TAG).write_text("{}")
    sealed.ack_path(dirs["canon"], TAG).write_text("")
    with pytest.raises(guards.NotAuthorized):
        sealed_eval.preflight(TAG, dirs["go"], dirs["ledger"], dirs["canon"], clean_tree=_clean, real_ledger=dirs["real_ledger"],
                              real_raw_dir=dirs["canon"])


# ---------------- A10 ----------------
def _load_script():
    spec = importlib.util.spec_from_file_location("p3m_matched_eval_tidy", ROOT / "scripts/p3m_matched_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def script(tmp_path, monkeypatch, dirs):
    """The script with its canonical raw dir and refusal-log dir patched to tmp dirs; a pre-existing canonical stdout stands in for the
    committed matched_v1.stdout.txt. Real ledger read only (the synthetic tag is never in it)."""
    mod = _load_script()
    canon, refusals = tmp_path / "canon_script", tmp_path / "refusal_logs"
    canon.mkdir()
    monkeypatch.setattr(mod, "RAW", canon)
    monkeypatch.setattr(mod, "REFUSAL_LOGS", refusals)
    monkeypatch.setattr(sealed, "install_signal_handlers", lambda: None)
    committed = canon / f"{TAG}.stdout.txt"
    committed.write_text("# committed log of a past run\n[t] DONE\n")
    return {"mod": mod, "canon": canon, "refusals": refusals, "committed": committed, "before": committed.read_bytes(), "tmp": tmp_path}


def _argv(s, monkeypatch, raw=None, lock=None):
    led = s["tmp"] / "scratch_ledger.json"
    led.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["p3m_matched_eval.py", "--tag", TAG, "--go-file", str(s["tmp"] / "GO_none"),
                                      "--raw-dir", str(raw or s["canon"]), "--ledger", str(led), "--lock", str(lock or s["tmp"] / "gpu.lock")])


def test_refused_run_with_canonical_raw_dir_does_not_touch_committed_stdout(script, monkeypatch, capsys):
    _argv(script, monkeypatch)
    with pytest.raises(SystemExit) as ex:
        script["mod"].main()
    assert ex.value.code == 13
    assert script["committed"].read_bytes() == script["before"]
    assert sorted(p.name for p in script["canon"].iterdir()) == [f"{TAG}.stdout.txt"]
    ref = (script["refusals"] / f"{TAG}.stdout.txt").read_text()
    assert "FAIL CLOSED" in ref and "NotAuthorized" in ref
    assert "FAIL CLOSED" in capsys.readouterr().out


def test_lock_refusal_with_canonical_raw_dir_does_not_touch_committed_stdout(script, monkeypatch):
    lock = script["tmp"] / "held.lock"
    lock.write_text("other job pid=1 nonce=abc\n")
    _argv(script, monkeypatch, lock=lock)
    with pytest.raises(SystemExit) as ex:
        script["mod"].main()
    assert ex.value.code == 11
    assert script["committed"].read_bytes() == script["before"]
    assert "REFUSED" in (script["refusals"] / f"{TAG}.stdout.txt").read_text()


def test_scratch_raw_dir_keeps_logging_into_it(script, monkeypatch):
    raw = script["tmp"] / "scratch_raw"
    _argv(script, monkeypatch, raw=raw)
    with pytest.raises(SystemExit) as ex:
        script["mod"].main()
    assert ex.value.code == 13
    assert "FAIL CLOSED" in (raw / f"{TAG}.stdout.txt").read_text()
    assert not script["refusals"].exists() and script["committed"].read_bytes() == script["before"]


def test_after_preflight_the_canonical_stdout_gets_the_buffered_and_later_lines(script, monkeypatch):
    """run_sealed replaced by a probe: a line before on_preflight goes to the refusal log and is copied into the canonical stdout once
    on_preflight is called; later lines go to the canonical stdout only."""
    def probe(*_a, log, ev, on_preflight, **_k):
        log("before preflight")
        on_preflight()
        on_preflight()  # idempotent
        log("after preflight")
        raise guards.NotAuthorized("probe")
    monkeypatch.setattr(sealed_eval, "run_sealed", lambda *a, **k: probe(*a, log=a[8], ev=a[9], **k))
    _argv(script, monkeypatch)
    with pytest.raises(SystemExit):
        script["mod"].main()
    canon = script["committed"].read_text()
    assert canon.startswith(script["before"].decode())
    assert canon.count("before preflight") == 1 and "after preflight" in canon and "FAIL CLOSED" in canon
    ref = (script["refusals"] / f"{TAG}.stdout.txt").read_text()
    assert "before preflight" in ref and "after preflight" not in ref


def test_run_sealed_calls_on_preflight_only_after_preflight_passed(dirs, monkeypatch):
    calls = []
    ev = sealed.SealedEvaluation(TAG, dirs["raw"], calls.append)
    with pytest.raises(guards.NotAuthorized):  # real preflight, no GO -> never called
        sealed_eval.run_sealed(TAG, dirs["go"], dirs["ledger"], dirs["raw"], dirs["raw"] / "missing.tsv", dirs["raw"], dirs["raw"] / "img",
                               dirs["raw"] / "out", calls.append, ev, clean_tree=_clean, on_preflight=lambda: calls.append("ON"))
    assert "ON" not in calls

    def fake_pf(*_a, **_k):
        calls.append("PF")
        return {"entry": dict(ENTRY), "git": {"commit": "0" * 40}, "protocol_sha256": "p" * 64, "ledger_sha256_before": None}

    class Stop(Exception):
        pass

    def loader(*_a, **_k):
        calls.append("LOAD")
        raise Stop
    monkeypatch.setattr(sealed_eval, "preflight", fake_pf)
    calls.clear()
    with pytest.raises(Stop):
        sealed_eval.run_sealed(TAG, dirs["go"], dirs["ledger"], dirs["raw"], dirs["raw"] / "missing.tsv", dirs["raw"], dirs["raw"] / "img",
                               dirs["raw"] / "out", calls.append, ev, loader=loader, on_preflight=lambda: calls.append("ON"))
    assert [c for c in calls if c in ("PF", "ON", "LOAD")] == ["PF", "ON", "LOAD"]


def test_refusal_logs_dir_is_gitignored():
    import subprocess
    r = subprocess.run(["git", "check-ignore", "-q", "data/cache/p3m_eval/refusal_logs/x.stdout.txt"], cwd=ROOT, check=False)
    assert r.returncode == 0
    assert json.dumps(str(_load_script().REFUSAL_LOGS)).endswith('data/cache/p3m_eval/refusal_logs"')
