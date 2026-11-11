"""an earlier review probes (pre-registered, committed BEFORE they are run) for an earlier change's matched-budget base-generation experiment
(docs/plans/base_gen_matched_budget.md; branch a sibling working copy 1d838ea..a432448 + uncommitted run outputs).

All recomputation here is the verification pass's own code (stdlib + numpy); nothing is imported from cgt.* except where a probe explicitly
compares against the worker's function (the reading rule, P-B1d). CPU only, no model is loaded. Never reads data/gold, data/dev,
ShotBench or CameraBench. Skipped (not failed) if the worker's uncommitted outputs are absent (another machine).

Failure rules (each assert is one probe; any failure is a BLOCKER unless the review explains it as a harness bug, disclosed):
B1 pre-registration
  B1a plan commit 1d838ea (contains section 6 reading rule) is earlier than the speed check summary time and the runner start.
  B1b Amendment 1 (bbb45c4) is committed after the speed check finished and before the runner started training.
  B1c the reading-rule text (section 6) and cgt/base_gen/reading.py are unchanged from 1d838ea to a432448.
  B1d applying MY transcription of the section-6 rule to MY recomputed d/CI gives the claimed reading (RECIPE_EXPLAINS) and equals
      the worker's reading.read on the same numbers.
B2 training continuity
  B2a updates.jsonl: updates exactly 1..799, no gap, no duplicate; micro counter = 16 x update; all losses finite.
  B2b LR at each update equals my own formula (linear warm-up 96 from 0 to 1e-4 then cosine to 1e-5 over 3,099 steps, logged at
      step update-1) within 1e-12 relative, AND equals the 2B run's logged LR at the same update for updates 1..799.
  B2c manifest ckpt 25 = update 799, micro 12,784; its sha256 equals adapter_sha256 in val_3b_ckpt025.summary.json.
  B2d 2B summary adapter_sha256 == 5c07252f... == the Phase 3B manifest's ckpt 25 sha.
  B2e the 3B provenance train_sha256 == 09ee851b...; sha256 of the train.jsonl file now == that; the file mtime predates the 2B
      run start (2026-10-02 02:56 CEST), so the 2B run read the same bytes; train.log says samples=51111; no 'shotbench',
      'camerabench', 'test' path, 'gold' or 'dev/' appears in the 3B manifest/provenance/train.log.
  B2f runner.jsonl: one training stage, rc 0, no resume events (resume_update 0).
B3 recompute from predictions
  B3a both prediction files match the sha256 in their summaries (be70b69f..., d2a0ebfd... prefixes as briefed) .
  B3b both cover the identical 5,822 (line, id, attribute) triples, which equal MY dedup_v2 of val.jsonl (key = id, attribute,
      sorted option texts; first occurrence) in order; gold labels in the files equal val.jsonl's answers.
  B3c my per-attribute and primary accuracy (unweighted mean of 7 per-attribute accuracies, points) equal the summaries to 1e-9 and
      the claimed 76.39 / 76.68 / d -0.29 to 0.005.
  B3d my film-clustered paired bootstrap (films from ShotQA meta.jsonl titles, my own film key; 71 films; seed 0, 2,000 draws)
      gives a CI within 0.1 point of the claimed [-1.15, +0.54]. Also reported (not a pass/fail rule): 10,000 draws, seeds 1-3.
B6 the worker's tests and ruff are run separately (commands in the review).
"""
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from cgt.worktrees import main_checkout, sibling

MAIN = main_checkout()
W = sibling("base_gen")
RES = W / "results/base_gen"
RUN3 = RES / "runs/base_gen_3b"
PRED = W / "data/cache/base_gen/predictions"
P3B = sibling("phase2")
RUN2 = P3B / "results/phase3b/runs/qwen3vl2b_lora_filmsplit_run1"
VAL = P3B / "data/cache/p3b/val.jsonl"
TRAIN = P3B / "data/cache/p3b/train.jsonl"
META = MAIN / "data/external/shotqa/meta.jsonl"
ATTRS = ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition"]
SHA_2B = "5c07252f4df6149bd27096ac48eb103a74a7b7dd0dfa3ab8b0ac895785c9c422"
TRAIN_SHA = "09ee851b3f6aa729d47ba4104b33b1275b3c1c26ee19749cffc6f6a8c7cba264"
CLAIM = {"p3": 76.39, "p2": 76.68, "d": -0.29, "lo": -1.15, "hi": 0.54, "films": 71}

pytestmark = pytest.mark.skipif(not (RES / "compare_ckpt025.json").exists(), reason="worker outputs not on this machine")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(W), *a], capture_output=True, text=True, check=True).stdout


def commit_time(c: str) -> datetime:
    return datetime.fromisoformat(git("show", "-s", "--format=%cI", c).strip())


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.split(" (")[0])


# ---------------- own implementations ----------------
def my_dedup(rows):
    seen, keep = set(), []
    for n, r in enumerate(rows):
        k = (r["id"], r["attribute"], tuple(sorted(r["options"].values())))
        if k not in seen:
            seen.add(k)
            keep.append(n)
    return keep


def my_film(title: str) -> str:
    t = title.split("/season")[0].split("/episode")[0]
    return re.sub(r"[^a-z0-9]+", "", t.lower().replace("&", "and"))


def my_primary(hit, attr, w=None):
    w = np.ones(len(hit)) if w is None else w
    return float(np.mean([100.0 * (hit[attr == k] * w[attr == k]).sum() / w[attr == k].sum() for k in range(7)]))


def my_boot(h3, h2, attr, film, n_boot=2000, seed=0):
    films = sorted(set(film))
    fidx = np.array([films.index(f) for f in film])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(films), len(films))
        w = np.bincount(draw, minlength=len(films)).astype(float)[fidx]
        if any(w[attr == k].sum() == 0 for k in range(7)):
            continue
        out.append(my_primary(h3, attr, w) - my_primary(h2, attr, w))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out), len(films)


def my_lr(update: int, total=3195, peak=1e-4, end=1e-5, warm_frac=0.03) -> float:
    warm = round(warm_frac * total)
    step = update - 1
    if step < warm:
        return peak * step / warm
    k = min(step - warm, total - warm)
    return end + 0.5 * (peak - end) * (1 + math.cos(math.pi * k / (total - warm)))


def my_rule(d, lo, hi):
    if hi < 0:
        return "NEWER_GENERATION_MATTERS"
    if lo > 0:
        return "OLDER_3B_BETTER"
    return "RECIPE_EXPLAINS" if abs(d) <= 1.0 else "INCONCLUSIVE"


# ---------------- shared state ----------------
@pytest.fixture(scope="module")
def S():
    s3 = json.loads((RES / "val_3b_ckpt025.summary.json").read_text())
    s2 = json.loads((RES / "val_2b_ckpt025.summary.json").read_text())
    p3, p2 = jl(Path(s3["predictions_file"])), jl(Path(s2["predictions_file"]))
    val = jl(VAL)
    keep = my_dedup(val)
    return {"s3": s3, "s2": s2, "p3": p3, "p2": p2, "val": val, "keep": keep,
            "cmp": json.loads((RES / "compare_ckpt025.json").read_text())}


@pytest.fixture(scope="module")
def R(S):
    rows = [S["val"][i] for i in S["keep"]]
    attr = np.array([ATTRS.index(r["attribute"]) for r in rows])
    h3 = np.array([int(p["pred"] == p["gold"]) for p in S["p3"]], float)
    h2 = np.array([int(p["pred"] == p["gold"]) for p in S["p2"]], float)
    want = {r["img_id"] for r in rows}
    byimg = {}
    for line in META.read_text().splitlines():
        m = json.loads(line)
        t = (m.get("title") or "").strip()
        if t and m["img_id"] in want:
            byimg.setdefault(m["img_id"], set()).add(my_film(t))
    film = []
    for r in rows:
        ks = byimg.get(r["img_id"], set())
        assert len(ks) == 1, f"image {r['images'][0]} has film keys {ks}"
        film.append(next(iter(ks)))
    return {"rows": rows, "attr": attr, "h3": h3, "h2": h2, "film": film}


# ---------------- B1 ----------------
def test_B1a_plan_before_speedcheck_and_run():
    plan = git("show", "1d838ea:docs/plans/base_gen_matched_budget.md")
    assert "## 6. Reading rule" in plan and "RECIPE_EXPLAINS" in plan
    t_plan = commit_time("1d838ea")
    sc = jl(RES / "speedcheck/runner.jsonl")
    t_sc0 = min(ts(e["time"]) for e in sc)
    t_run = ts(jl(RES / "runner.jsonl")[0]["time"])
    assert t_plan < t_sc0 < t_run


def test_B1b_amendment_after_speedcheck_before_launch():
    sc = jl(RES / "speedcheck/runner.jsonl")
    t_sc1 = max(ts(e["time"]) for e in sc)
    t_am = commit_time("bbb45c4")
    t_run = ts(jl(RES / "runner.jsonl")[0]["time"])
    t_upd1 = ts(jl(RUN3 / "updates.jsonl")[0]["time"])
    assert t_sc1 <= t_am < t_run < t_upd1
    assert "### Amendment 1" not in git("show", "1055b78:docs/plans/base_gen_matched_budget.md")
    assert "Target f = 25%" in git("show", "bbb45c4:docs/plans/base_gen_matched_budget.md")


def test_B1c_rule_unchanged_after_registration():
    def sec6(c):
        t = git("show", f"{c}:docs/plans/base_gen_matched_budget.md")
        return t.split("## 6. Reading rule")[1].split("## 7.")[0]
    assert sec6("1d838ea") == sec6("a432448")
    assert git("diff", "1d838ea", "a432448", "--", "src/cgt/base_gen/reading.py", "scripts/base_gen_compare.py") == ""
    # working tree (uncommitted) must not alter the rule or the compare script either
    assert git("diff", "a432448", "--", "src/cgt/base_gen/reading.py", "scripts/base_gen_compare.py",
               "docs/plans/base_gen_matched_budget.md") == ""


def test_B1d_reading_applied_as_written(S, R):
    d = my_primary(R["h3"], R["attr"]) - my_primary(R["h2"], R["attr"])
    lo, hi, _, _ = my_boot(R["h3"], R["h2"], R["attr"], R["film"])
    import sys
    sys.path.insert(0, str(W / "src"))
    from cgt.base_gen.reading import read
    assert my_rule(d, lo, hi) == "RECIPE_EXPLAINS" == read(d, lo, hi) == S["cmp"]["reading"]


# ---------------- B2 ----------------
def test_B2a_updates_contiguous_finite():
    u = jl(RUN3 / "updates.jsonl")
    assert [r["update"] for r in u] == list(range(1, 800))
    assert all(r["micro"] == 16 * r["update"] for r in u)
    assert all(math.isfinite(r["loss"]) and math.isfinite(r["gnorm"]) for r in u)


def test_B2b_lr_schedule_matches_formula_and_2b():
    u3 = {r["update"]: r["lr"] for r in jl(RUN3 / "updates.jsonl")}
    u2 = {r["update"]: r["lr"] for r in jl(RUN2 / "updates.jsonl") if r["update"] <= 799}
    for k in range(1, 800):
        assert u3[k] == pytest.approx(my_lr(k), rel=1e-6, abs=1e-12), k
        assert u3[k] == pytest.approx(u2[k], rel=1e-6, abs=1e-12), k


def test_B2c_evaluated_checkpoint_is_update_799(S):
    m = json.loads((RUN3 / "manifest.json").read_text())["checkpoints"]
    assert set(m) == {"25"} and m["25"]["update"] == 799 and m["25"]["micro"] == 12784
    assert m["25"]["sha256"] == S["s3"]["adapter_sha256"]
    assert S["s3"]["model"].startswith("Qwen/Qwen2.5-VL-3B-Instruct@66285546")


def test_B2d_2b_arm_is_phase3b_ckpt025(S):
    m2 = json.loads((RUN2 / "manifest.json").read_text())["checkpoints"]["25"]
    assert S["s2"]["adapter_sha256"] == SHA_2B == m2["sha256"] and m2["update"] == 799
    assert S["s2"]["model"].startswith("Qwen/Qwen3-VL-2B-Instruct@89644892")


def test_B2e_training_data_same_and_clean():
    prov = jl(RUN3 / "provenance.jsonl")
    assert prov[0]["train_sha256"] == TRAIN_SHA == sha(TRAIN)
    assert datetime.fromtimestamp(TRAIN.stat().st_mtime).astimezone() < datetime.fromisoformat("2026-10-02T02:56:18+02:00")
    log = (RUN3 / "train.log").read_text()
    assert "samples=51111" in log
    blob = ((RUN3 / "manifest.json").read_text() + (RUN3 / "provenance.jsonl").read_text() + log).lower()
    for bad in ("shotbench", "camerabench", "test.jsonl", "data/gold", "data/dev"):
        assert bad not in blob, bad


def test_B2f_single_uninterrupted_stage():
    ev = jl(RES / "runner.jsonl")
    stages = [e for e in ev if e.get("stage") == "train" and "rc" in e]
    assert len(stages) == 1 and stages[0]["rc"] == 0
    assert ev[0]["resume_update"] == 0 and not any("resum" in e.get("msg", "") for e in ev[1:])


# ---------------- B3 ----------------
def test_B3a_prediction_hashes(S):
    for s, pre in ((S["s3"], "be70b69f"), (S["s2"], "d2a0ebfd")):
        h = sha(Path(s["predictions_file"]))
        assert h == s["predictions_sha256"] and h.startswith(pre)


def test_B3b_identical_question_set(S):
    keep, val = S["keep"], S["val"]
    assert len(keep) == 5822
    want = [(n, val[n]["id"], val[n]["attribute"]) for n in keep]
    for p in (S["p3"], S["p2"]):
        assert [(r["line"], r["id"], r["attribute"]) for r in p] == want
    gold_letters = [val[n]["answer"] for n in keep]
    assert [r["gold"] for r in S["p3"]] == [r["gold"] for r in S["p2"]] == gold_letters


def test_B3c_accuracy_recompute(S, R):
    for arm, h in (("s3", R["h3"]), ("s2", R["h2"])):
        q = S[arm]["shotqa_dedup_v2"]
        for k, a in enumerate(ATTRS):
            assert 100.0 * h[R["attr"] == k].mean() == pytest.approx(q["per_attribute"][a], abs=1e-9)
        assert my_primary(h, R["attr"]) == pytest.approx(q["primary"], abs=1e-9)
    p3, p2 = my_primary(R["h3"], R["attr"]), my_primary(R["h2"], R["attr"])
    assert round(p3, 2) == CLAIM["p3"] and round(p2, 2) == CLAIM["p2"] and abs((p3 - p2) - CLAIM["d"]) < 0.005


def test_B3d_bootstrap_ci(R):
    lo, hi, n, nf = my_boot(R["h3"], R["h2"], R["attr"], R["film"])
    print(f"\nV-BG bootstrap seed0 2000: [{lo:+.3f}, {hi:+.3f}] kept {n} films {nf}")
    for seed in (1, 2, 3):
        a, b, m, _ = my_boot(R["h3"], R["h2"], R["attr"], R["film"], n_boot=10000, seed=seed)
        print(f"bootstrap seed{seed} 10000: [{a:+.3f}, {b:+.3f}] kept {m}")
    assert nf == CLAIM["films"]
    assert abs(lo - CLAIM["lo"]) <= 0.1 and abs(hi - CLAIM["hi"]) <= 0.1
