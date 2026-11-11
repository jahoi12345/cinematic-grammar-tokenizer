"""sound re-runs after the U1 fix, branch a sibling working copy 2995ab3.

Two kinds of checks, no model is run:
1. Model-free property of the re-run: classify_clip is driven with a fake classifier (synthetic top-10 lists over the AST labels) once with
   the v1 grouping and once with the v2 grouping. v2 may only move a shot ambient->score, dialogue->score or ambient->dialogue; the top
   label, its probability and sound_p_speech never change; sound_p_music never decreases. These are the invariants checked on the data.
2. Data checks on a re-run's local, gitignored files: new p2/token tables against their `*_pre_soundgroups` backups. They run only where
   the backups exist. Film ids come from the environment so that no local-only film is named in committed text:
     CGT_VT_RERUN="<processed_dir>:<film_id>,<processed_dir>:<film_id>,..."
   Without it, NOTLD in this checkout's data/processed is checked if its backups exist (otherwise skipped).
See a verification record.
"""

import json
import os
import random
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
AST_CONFIG = (Path.home() / ".cache/huggingface/hub/models--MIT--ast-finetuned-audioset-10-10-0.4593/snapshots"
              / "f826b80d28226b62986cc218e5cec390b1096902/config.json")
SOUND = ["sound", "sound_db", "sound_top", "sound_top_p", "sound_p_speech", "sound_p_music"]
LEGAL = {("ambient", "score"), ("dialogue", "score"), ("ambient", "dialogue")}
# accepted NOTLD artifacts after the re-run
NOTLD_P2_NEW = "98b6eb4ffd99c837944b02eee4a67d8061cf4e775886b1c38a79a3a5fa77392f"
NOTLD_TOKENS_NEW = "0e818c9d48546fc087609c539e9b9dd007b74100fd58b467c42f1350eaee9fb7"
NOTLD_TXT_NEW = "0212863b569417663af0d1e01a80b61b2dca1d76d8f375a8e5ee49796647ed63"
NOTLD_CLASSES = ({"score": 457, "dialogue": 367, "ambient": 253}, {"score": 480, "dialogue": 359, "ambient": 238})


def _audio():
    try:
        from cgt.p2_pretrained import audio
    except ModuleNotFoundError:  # light env: audio.py imports torch/transformers at module level; neither is used here
        saved = {m: sys.modules.get(m) for m in ("torch", "transformers", "cgt.p2_pretrained.audio")}
        stub = types.ModuleType("transformers")
        stub.pipeline = None
        sys.modules["torch"], sys.modules["transformers"] = types.ModuleType("torch"), stub
        try:
            from cgt.p2_pretrained import audio
        finally:  # never leave the stubs (or a module bound to them) behind for other tests
            for m, v in saved.items():
                if v is None:
                    sys.modules.pop(m, None)
                else:
                    sys.modules[m] = v
    return audio


def _v1(label: str) -> str:  # main ee5d1dc _group (pre-U1), verbatim rules
    l = label.lower()
    if "speech" in l or l in {"conversation", "narration, monologue", "shout", "screaming", "whispering"}:
        return "dialogue"
    if "music" in l or l in {"piano", "organ", "violin, fiddle", "orchestra", "synthesizer", "singing"}:
        return "score"
    return "ambient"


def _labels(audio) -> list[str]:
    if AST_CONFIG.exists():
        return list(json.loads(AST_CONFIG.read_text())["id2label"].values())
    extra = ["Speech", "Conversation", "Music", "Wind", "Rain", "Vehicle", "Silence", "Dog", "Gunshot, gunfire", "Footsteps"]
    return sorted(set(audio.MUSIC_SUBTREE_LABELS) | set(audio.SINGING_LABELS) | set(audio.EXCLUDED_LABELS) | set(extra))


def test_vt_rerun_transitions_are_one_directional(monkeypatch):
    audio = _audio()
    labels = _labels(audio)
    rng = random.Random(20261015)
    x = np.full(audio.SR, 0.05, dtype=np.float32)  # about -26 dB: above the silence threshold
    v2 = audio._group
    seen = set()
    for _ in range(3000):
        top = rng.sample(labels, 10)
        p = sorted((rng.random() for _ in top), reverse=True)
        s = sum(p) * rng.uniform(1.0, 2.0)
        preds = [{"label": a, "score": b / s} for a, b in zip(top, p)]
        monkeypatch.setattr(audio, "_get_pipe", lambda preds=preds: (lambda *a, **k: preds))
        monkeypatch.setattr(audio, "_group", _v1)
        old = audio.classify_clip(x)
        monkeypatch.setattr(audio, "_group", v2)
        new = audio.classify_clip(x)
        for k in ("sound_db", "sound_top", "sound_top_p", "sound_p_speech"):
            assert old[k] == new[k], k
        assert new["sound_p_music"] >= old["sound_p_music"]
        if old["sound"] != new["sound"]:
            assert (old["sound"], new["sound"]) in LEGAL
            seen.add((old["sound"], new["sound"]))
    assert ("ambient", "score") in seen  # the property is exercised, not vacuous


def _films():
    spec = os.environ.get("CGT_VT_RERUN")
    if spec:
        return [tuple(item.rsplit(":", 1)) for item in spec.split(",") if item.strip()]
    return [(str(ROOT / "data/processed"), "night_of_the_living_dead")]


@pytest.fixture(params=range(len(_films())), ids=lambda i: f"film{i}")
def rerun(request):
    pd = pytest.importorskip("pandas")
    d, f = _films()[request.param]
    d = Path(d)
    need = [d / f"{f}_p2_pre_soundgroups.parquet", d / f"{f}_tokens_v1_pre_soundgroups.parquet", d / f"{f}_tokens_v1_pre_soundgroups.txt",
            d / f"{f}_p2.parquet", d / f"{f}_tokens_v1.parquet", d / f"{f}_tokens_v1.txt"]
    if not all(p.exists() for p in need):
        pytest.skip("no sound-groups re-run backups here")
    return {"f": f, "d": d, "o": pd.read_parquet(need[0]), "n": pd.read_parquet(need[3]),
            "to": pd.read_parquet(need[1]), "tn": pd.read_parquet(need[4]),
            "lo": need[2].read_text().splitlines(), "ln": need[5].read_text().splitlines()}


def _same(a, b) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
            continue
        if x is None and y is None:
            continue
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            if not np.array_equal(np.asarray(x), np.asarray(y)):
                return False
        elif x != y:
            return False
    return True


def test_vt_only_sound_family_p2_columns_differ(rerun):
    o, n = rerun["o"], rerun["n"]
    assert list(o.columns) == list(n.columns) and len(o) == len(n)
    bad = [c for c in o.columns if c not in SOUND and not _same(list(o[c]), list(n[c]))]
    assert bad == []
    for c in ("sound_top", "sound_top_p", "sound_p_speech"):
        assert _same(list(o[c]), list(n[c])), c


def test_vt_sound_changes_are_legal_and_follow_the_evidence(rerun):
    audio = _audio()
    o, n = rerun["o"], rerun["n"]
    ch = (o["sound"] != n["sound"]).values
    assert all((a, b) in LEGAL for a, b in zip(o["sound"][ch], n["sound"][ch]))
    assert (o["sound"] == "silence").equals(n["sound"] == "silence")
    pm_o, pm_n = o["sound_p_music"].fillna(0).values, n["sound_p_music"].fillna(0).values
    assert (pm_n >= pm_o).all()
    for i in range(len(n)):
        t = n["sound_top"].iat[i]
        if isinstance(t, str) and _v1(t) == "ambient" and audio._group(t) == "score":  # top label newly score: it is the strongest score label
            assert n["sound"].iat[i] == "score" and np.isclose(pm_n[i], n["sound_top_p"].iat[i])


def test_vt_tokens_differ_only_in_the_sound_slot(rerun):
    to, tn, n = rerun["to"], rerun["tn"], rerun["n"]
    assert list(to.columns) == list(tn.columns) and len(to) == len(tn)
    bad = [c for c in to.columns if c not in ("sound", "token_row") and not _same(list(to[c]), list(tn[c]))]
    assert bad == []
    assert list(tn["sound"]) == list(n["sound"])
    mask = lambda s: re.sub(r"sound=[^ |\]]+", "sound=X", s)
    assert [mask(a) for a in to["token_row"]] == [mask(b) for b in tn["token_row"]]
    lo, ln = rerun["lo"], rerun["ln"]
    assert len(lo) == len(ln) and [mask(a) for a in lo] == [mask(b) for b in ln]
    assert [re.search(r"sound=([^ |\]]+)", s).group(1) for s in ln] == list(tn["sound"])
    assert sum(a != b for a, b in zip(lo, ln)) == int((rerun["o"]["sound"] != n["sound"]).sum())


def test_vt_notld_counts_when_the_accepted_bytes_are_present(rerun):
    import hashlib
    if rerun["f"] != "night_of_the_living_dead":
        pytest.skip("NOTLD only")
    d = rerun["d"]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(d / "night_of_the_living_dead_p2.parquet") != NOTLD_P2_NEW:
        pytest.skip("not the accepted post-re-run NOTLD p2 table")
    assert sha(d / "night_of_the_living_dead_tokens_v1.parquet") == NOTLD_TOKENS_NEW
    assert sha(d / "night_of_the_living_dead_tokens_v1.txt") == NOTLD_TXT_NEW
    o, n = rerun["o"], rerun["n"]
    assert int((o["sound"] != n["sound"]).sum()) == 24
    assert o["sound"].value_counts().to_dict() == NOTLD_CLASSES[0]
    assert n["sound"].value_counts().to_dict() == NOTLD_CLASSES[1]
