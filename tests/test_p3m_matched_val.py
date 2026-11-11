"""Validation mode refusals (frozen adapter hash, GPU lock, memory guard) and aggregation. Synthetic files only."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cgt.p3b_vlm import guards
from cgt.p3m_matched import registry
from cgt.p3m_matched.evaluate import run_questions, summarize

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p3m_matched_val", ROOT / "scripts/p3m_matched_val.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


class Boom(Exception):
    pass


def _never(*_a, **_k):
    raise Boom("model or data touched")


@pytest.fixture
def droot(tmp_path, monkeypatch):
    rel = registry.MODELS["matched_v1"]["adapter_rel"]
    p = tmp_path / rel
    p.parent.mkdir(parents=True)
    p.write_bytes(b"frozen")
    d = hashlib.sha256(b"frozen").hexdigest()
    monkeypatch.setitem(registry.MODELS, "matched_v1", {**registry.MODELS["matched_v1"], "identity_sha256": d})
    monkeypatch.setattr(registry, "FROZEN_ADAPTER_SHA256", d)
    return tmp_path


def _args(droot, tmp_path, **kw):
    return SimpleNamespace(**{"tag": "matched_v1", "variant": "reasoning", "n": None, "data_root": str(droot), "shotqa_root": str(droot),
                              "cache_dir": str(tmp_path / "cache"), "out_dir": str(tmp_path / "out"), "lock": str(tmp_path / "gpu.lock"), **kw})


def test_val_refuses_wrong_adapter_hash(droot, tmp_path):
    (droot / registry.MODELS["matched_v1"]["adapter_rel"]).write_bytes(b"not the frozen adapter")
    with pytest.raises(guards.AdapterHashError):
        V.run(_args(droot, tmp_path), engine_factory=_never, mem_reader=lambda: 90)
    assert not (tmp_path / "gpu.lock").exists()


def test_val_refuses_other_tags(droot, tmp_path, monkeypatch):
    # a pinned tag with the SAME identity is still refused: validation mode is for matched_v1 only (ShotVL never runs on ShotQA validation)
    monkeypatch.setitem(registry.MODELS, "other_pinned", dict(registry.MODELS["matched_v1"]))
    with pytest.raises(guards.TagNotBound):
        V.run(_args(droot, tmp_path, tag="other_pinned"), engine_factory=_never, mem_reader=lambda: 90)


def test_val_refuses_when_gpu_lock_held(droot, tmp_path):
    (tmp_path / "gpu.lock").write_text("other job pid=1 nonce=abc\n")
    with pytest.raises(V.GpuBusy):
        V.run(_args(droot, tmp_path), engine_factory=_never, mem_reader=lambda: 90)
    assert (tmp_path / "gpu.lock").read_text() == "other job pid=1 nonce=abc\n"  # never stolen


def test_val_memory_guard(droot, tmp_path):
    with pytest.raises(V.MemoryLow):
        V.run(_args(droot, tmp_path), engine_factory=_never, mem_reader=lambda: 24)
    assert not (tmp_path / "gpu.lock").exists()


def test_lock_released_after_failure(droot, tmp_path):
    with pytest.raises(FileNotFoundError):  # no val.jsonl in the synthetic data root: fails after the lock is taken
        V.run(_args(droot, tmp_path), engine_factory=_never, mem_reader=lambda: 90)
    assert not (tmp_path / "gpu.lock").exists()


class FakeEngine:
    def __init__(self, texts):
        self.texts = list(texts)

    def answer(self, image, question, options, variant):
        return {"text": self.texts.pop(0), "prompt_tokens": 0, "generation_tokens": 1}


def test_summary_counts_invalid_as_wrong_and_expected_fallback():
    qs = [{"id": str(i), "attribute": "size", "image": "x", "question": "q", "options": {"A": "Wide", "B": "Close Up", "C": "Medium", "D": "Long"},
           "answer": "A"} for i in range(4)]
    res = run_questions(FakeEngine(["A", "B", "no idea", "<answer>wide</answer>"]), qs, "plain")
    s = summarize(res)["per_attribute"]["size"]
    assert (s["correct"], s["n_invalid"]) == (2, 1)
    assert s["acc"] == 0.5
    assert s["expected_fallback_acc"] == pytest.approx((2 + 1 / 5) / 4)
    assert s["by_method"] == {"option": 2, "invalid": 1, "text": 1}


def test_compare_with_letter_logit():
    res = [{"key": "0:1", "attribute": "size", "letter": "A", "gold": "A"}, {"key": "1:2", "attribute": "size", "letter": None, "gold": "B"}]
    c = V.compare(res, {"0:1": "A", "1:2": "B"})
    assert c["per_attribute"]["size"] == {"n": 2, "letter_logit_acc": 1.0, "agreement": 0.5}


def test_val_questions_rebuilt_from_shotqa_turns(tmp_path):
    turn = "Question: What's the shot size of this shot?\nOptions:\nA. Wide\nB. Close Up\nPlease select the most likely answer from the options above. \n"
    img = tmp_path / "shotqa/x.jpg"
    img.parent.mkdir()
    img.write_bytes(b"")
    (tmp_path / "data/cache/p3b").mkdir(parents=True)
    (tmp_path / "data/cache/p3b/val.jsonl").write_text(json.dumps({"id": "x.jpg#0", "attribute": "size", "images": [str(img)],
                                                                   "messages": [{"role": "user", "content": turn}, {"role": "assistant", "content": "B"}],
                                                                   "options": {"B": "Close Up", "A": "Wide"}, "answer": "B"}) + "\n")
    q = V.load_val_questions(tmp_path, None, tmp_path / "shotqa")[0]
    assert q["question"] == "What's the shot size of this shot?" and list(q["options"]) == ["A", "B"]
    with pytest.raises(guards.DataRootViolation):
        V.load_val_questions(tmp_path, None, tmp_path / "elsewhere")


def test_duplicate_ids_get_distinct_keys(tmp_path):
    """val.jsonl holds 64 ids twice (duplicate sft.json entries): joins must use the line-index key, never the id alone."""
    img = tmp_path / "shotqa/x.jpg"
    img.parent.mkdir()
    img.write_bytes(b"")
    (tmp_path / "data/cache/p3b").mkdir(parents=True)
    rows = []
    for gold, q in (("A", "What's the lens size of this shot?"), ("B", "What's the shot size of this shot?")):
        turn = f"Question: {q}\nOptions:\nA. Wide\nB. Close Up\nPlease select the most likely answer from the options above. \n"
        rows.append(json.dumps({"id": "x.jpg#0", "attribute": "lens", "images": [str(img)], "messages": [{"role": "user", "content": turn}],
                                "options": {"A": "Wide", "B": "Close Up"}, "answer": gold}))
    (tmp_path / "data/cache/p3b/val.jsonl").write_text("\n".join(rows) + "\n")
    qs = V.load_val_questions(tmp_path, None, tmp_path / "shotqa")
    assert [q["key"] for q in qs] == ["0:x.jpg#0", "1:x.jpg#0"]
    assert [q["answer"] for q in qs] == ["A", "B"]
