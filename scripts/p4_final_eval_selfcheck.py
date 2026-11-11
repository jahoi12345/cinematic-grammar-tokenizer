"""CPU-only synthetic self-check of the SEALED Phase 4 final evaluator (the Phase 4 analogue of scripts/p3b_sealed_demo.py): synthetic test sets with PLANTED secrets, a fake backend that
knows the planted answers (an "oracle model"), real guards, real stores, real ledger and sealed-state code in temporary directories. NO model, NO real test data, NO real ledger, NO go file in the
repository. Writes results/phase4/final_eval_selfcheck.md.
Usage: uv run python scripts/p4_final_eval_selfcheck.py            (the child modes --child kill|term are internal)"""
import argparse
import builtins
import io
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cgt.p3b_vlm import guards, sealed
from cgt.p4_camera_move import final_eval as FE
from cgt.p4_camera_move import final_run as FR
from cgt.p4_camera_move import final_seal as FSEAL
from cgt.p4_camera_move import final_stores as FS
from cgt.p4_camera_move.baseline import fit_scores
from cgt.p4_camera_move.camerabench import PRIMITIVE_TO_ATOM
from cgt.p4_camera_move.final_seal import FinalSealed
from cgt.p4_camera_move.labels import parse_move_label, schema_moves

ROOT = Path(__file__).resolve().parents[1]
SECRET = "0.4321"
FORBIDDEN = [r"\bacc\w*", r"\bcorrect\b", r"\bpred\w*", r"\bgold\b", r"logits?\s*[=:\[]", r"\bmean\b", SECRET, r"43\.2", r"\bhits?\b", r"ledger recorded"]
results: list = []
SPECS = {"geometry": FR.Spec("", "geometry", None, "c" * 16, None, None, "f" * 64), "video": FR.Spec("", "video", "a" * 64, None, None, None, "f" * 64),
         "combined": FR.Spec("", "combined", "a" * 64, "b" * 16, 0.3, 0.7, "f" * 64)}
REGISTRY = {FE.MODEL_TAGS["geometry"]: {"config_hash": "c" * 16}, FE.MODEL_TAGS["video"]: {"adapter_sha256": "a" * 64},
            FE.MODEL_TAGS["combined"]: {"adapter_sha256": "a" * 64, "config_hash": "b" * 16, "weights": {"w_s": 0.3, "w_g": 0.7}}}
COMMIT = "1234567890abcdef1234567890abcdef12345678"


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(("PASS" if ok else "FAIL") + f": {name}" + (f": {detail}" if detail else ""), flush=True)


def spec_for(dataset: str, model: str) -> FR.Spec:
    s = SPECS[model]
    return FR.Spec(dataset, model, s.adapter_sha256, s.config_hash, s.w_s, s.w_g, s.freeze_sha256)


def leaked(text: str) -> list[str]:
    return [w for w in FORBIDDEN if re.search(w, text, flags=re.IGNORECASE)]


def tree_text(d: Path) -> str:
    return "\n".join(p.read_text() for p in sorted(Path(d).rglob("*")) if p.is_file() and ".git" not in p.parts)


# ---------------------------------------------------------------------------------------------------- synthetic world
TAGS = {"main": ["dolly-in", "pan-left", "pan-right", "tilt-up", "static", "zoom-in", "truck-left", "pedestal-up", "arc-CW", "roll-CW"], "extra": ["no-motion", "side-tracking", "low-speed"]}


def make_world(tmp: Path, n_cb: int = 40, n_sb: int = 12) -> dict:
    """Synthetic ShotBench TSV (n_sb movement rows + image rows), synthetic CameraBench root, a git repo with a committed FREEZE.json, and the planted answers (the oracle)."""
    rng = np.random.default_rng(7)
    atoms = [a for a in schema_moves() if a in FE.OPTION_WORDING]
    sb_rows, oracle = [], {}
    for i in range(n_sb):
        opts = [tuple(sorted(rng.choice(atoms, int(rng.integers(1, 3)), replace=False))) for _ in range(4)]
        while len({o for o in opts}) < 4:
            opts = [tuple(sorted(rng.choice(atoms, int(rng.integers(1, 3)), replace=False))) for _ in range(4)]
        gold = "ABCD"[int(rng.integers(0, 4))]
        name = f"SRC{i % 5}.mp4_{i}.mp4"
        oracle[name] = set(opts["ABCD".index(gold)])
        sb_rows.append({"index": str(3000 + i), "type": '["video"]', "path": f"['video/{name}']", "question": FS.QUESTION_TEXT, "options": json.dumps({L: FE.option_text(o) for L, o in zip("ABCD", opts, strict=True)}),
                        "answer": gold, "category": "camera movement"})
    sb_rows.append({"index": "1", "type": '["image"]', "path": "['image/x.jpg']", "question": "What is the shot size?", "options": '{"A": "Close-up", "B": "Medium", "C": "Long", "D": "Wide"}', "answer": "A", "category": "shot size"})
    tsv = tmp / "shotbench/test.tsv"
    tsv.parent.mkdir(parents=True)
    pd.DataFrame(sb_rows).to_csv(tsv, sep="\t", index=False)
    cb_root = tmp / "camerabench"
    (cb_root / "videos_gif").mkdir(parents=True)
    names, jl = [], []
    for i in range(n_cb):
        tags = [str(t) for t in rng.choice(TAGS["main"], int(rng.integers(1, 3)), replace=False)]
        if i % 9 == 0:
            tags = ["no-motion"]  # only under the sensitivity mapping
        if i % 7 == 0:
            tags.append("side-tracking")
        n = f"V{i:03d}.{i % 5}.{i}.gif"
        (cb_root / "videos_gif" / n).write_bytes(b"GIF89a")
        names.append(n)
        jl.append({"Video": n, "labels": tags, "caption": "", "path": f"videos_gif/{n}"})
        oracle[n] = {PRIMITIVE_TO_ATOM[t] for t in tags if t in PRIMITIVE_TO_ATOM}
    (cb_root / "test.jsonl").write_text("\n".join(json.dumps(r) for r in jl) + "\n")
    repo = tmp / "repo"
    (repo / "results/phase4/runs/video_v1").mkdir(parents=True)
    freeze = repo / "results/phase4/runs/video_v1/FREEZE.json"
    freeze.write_text(json.dumps({"registered_models": REGISTRY}))
    for cmd in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "freeze"]):
        subprocess.run(["git", *cmd], cwd=repo, check=True, capture_output=True)
    return {"tsv": tsv, "cb_root": cb_root, "repo": repo, "freeze": freeze, "oracle": oracle, "ledger": tmp / "test_looks.jsonl", "names": sorted(names), "tmp": tmp}


class FakeBackend:
    """An oracle model: it knows the planted answers of the synthetic world (the real backend never does) and answers from options and clips only."""
    def __init__(self, world: dict, trace: list, model: str, crash_in: str | None = None, hashes: list | None = None, secret_in_error: bool = False):
        self.w, self.trace, self.crash_in, self.hashes, self.secret = world, trace, crash_in, hashes, secret_in_error
        self.atoms = [a for a in schema_moves() if a in FE.OPTION_WORDING]
        self.prior = np.full(len(self.atoms), 0.2)
        self.spec = SPECS[model]

    def _maybe_crash(self, where: str) -> None:
        if self.crash_in == where:
            raise RuntimeError(f"boom in {where}; partial accuracy {SECRET}" if self.secret else f"boom in {where}")
        if self.crash_in == f"kill_{where}":
            os.kill(os.getpid(), signal.SIGKILL)
        if self.crash_in == f"term_{where}":
            os.kill(os.getpid(), signal.SIGTERM)

    def adapter_hash(self):
        if self.hashes:
            return self.hashes.pop(0) if len(self.hashes) > 1 else self.hashes[0]
        return self.spec.adapter_sha256

    def geometry_probs(self, clips, recalibrate):
        self.trace.append(("backend", "geometry", recalibrate))
        self._maybe_crash("geometry")
        P = np.full((len(clips), len(self.atoms)), 0.1)
        for i, c in enumerate(clips):
            for a in self.w["oracle"][c]:
                if a in self.atoms:
                    P[i, self.atoms.index(a)] = 0.7 if not recalibrate else 0.6
        return P

    def preflight(self, store):
        if self.crash_in == "preflight":
            raise guards.UnreadableImage("inputs not prepared")

    def warm_up(self, clips):
        self.trace.append(("backend", "warm_up", len(clips)))
        self._maybe_crash("warm_up")

    def video_logits(self, questions, progress):
        self.trace.append(("backend", "video", len(questions)))
        self._maybe_crash("video")
        out = []
        for i, q in enumerate(questions):
            z = np.zeros(4)
            for j, L in enumerate(FE.LETTERS):
                if set(q["option_atoms"][L] if "option_atoms" in q else parse_move_label(q["options"][L]).atoms) == self.w["oracle"][q["clip"]]:
                    z[j] += 3.0
            out.append(z + 0.01 * np.arange(4))
            progress(i + 1, len(questions))
        return out

    def provenance(self):
        return {"fake": True}


class Cap:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, m):
        self.lines.append(str(m))


def build(world, dataset, model, tmp, trace, **kw):
    spec = spec_for(dataset, model)
    events = trace
    store = FS.ShotBenchMovementStore(tsv=world["tsv"], ledger=world["ledger"], freeze_path=world["freeze"], repo_root=world["repo"], events=events, expected_n=12) if dataset == "shotbench_movement" else \
        FS.CameraBenchFinalStore(root=world["cb_root"], ledger=world["ledger"], freeze_path=world["freeze"], repo_root=world["repo"], events=events)
    backend = FakeBackend(world, trace, model, **{k: v for k, v in kw.items() if k in ("crash_in", "hashes", "secret_in_error")})
    return spec, store, backend


def run_once(world, dataset, model, tmp, *, trace=None, go="good", git_dirty=False, plan_ok=True, registry=None, resume=False, log=None, ref=None, **kw):
    """One in-process evaluation; returns (exit_code or 0, log lines, trace)."""
    trace = trace if trace is not None else []
    spec, store, backend = build(world, dataset, model, tmp, trace, **kw)
    raw, out = tmp / "raw", tmp / "out"
    goff = tmp / f"GO_{spec.key}"
    if go == "good":
        goff.write_text(spec.go())
    elif go == "wrong_tag":
        goff.write_text(FE.go_text(dataset, "other_tag", spec.adapter_sha256, spec.config_hash, spec.freeze_sha256))
    elif go == "wrong_adapter":
        goff.write_text(FE.go_text(dataset, spec.tag, "d" * 64, spec.config_hash, spec.freeze_sha256))
    elif go == "wrong_freeze":
        goff.write_text(FE.go_text(dataset, spec.tag, spec.adapter_sha256, spec.config_hash, "e" * 64))
    elif go == "wrong_dataset":
        goff.write_text(FE.go_text("camerabench_public_test" if dataset == "shotbench_movement" else "shotbench_movement", spec.tag, spec.adapter_sha256, spec.config_hash, spec.freeze_sha256))
    cap = log or Cap()
    ev = FinalSealed(spec.key, raw, cap, code_commit=COMMIT)
    code = 0
    try:
        FR.run_evaluation(spec, store, backend, ev=ev, ledger=world["ledger"], raw=raw, out_dir=out, go_path=goff, registry=REGISTRY if registry is None else registry,
                          git_state=lambda p: {"commit": COMMIT, "dirty_files": ["scripts/x.py"] if git_dirty else []},
                          plan_check=lambda: {"unchanged": plan_ok}, resume=resume, reference_sources=ref, log=cap)
    except guards.FailClosed as e:
        if ev.started and not ev.finished:
            ev.record_crash(e)
        code = sealed.exit_code_for(ev, e)
    except BaseException as e:  # noqa: BLE001
        if ev.started and not ev.finished:
            ev.record_crash(e)
        code = sealed.exit_code_for(ev, e)
    return code, cap.lines, trace


def fresh(tmp_root: Path, name: str) -> tuple[dict, Path]:
    d = tmp_root / name
    d.mkdir()
    return make_world(d), d


# ---------------------------------------------------------------------------------------------------- the checks
def check_pure(tmp_root: Path) -> None:
    atoms = [a for a in schema_moves() if a in FE.OPTION_WORDING]
    bad = [c for r in (1, 2, 3, 4) for c in itertools.combinations(atoms, r) if set(parse_move_label(FE.option_text(c)).atoms) != set(c) or parse_move_label(FE.option_text(c)).unmapped]
    record("option wording: every atom set of up to 4 atoms renders in the ShotQA wording (schema order, ' and ') and parses back to the same set", not bad, f"{sum(len(list(itertools.combinations(atoms, r))) for r in (1, 2, 3, 4))} sets, {len(bad)} failures")
    w, _ = fresh(tmp_root, "gen")
    names = w["names"]
    lbs = FS.labels_by_stem([{**json.loads(x), "stem": Path(json.loads(x)["path"]).stem} for x in (w["cb_root"] / "test.jsonl").read_text().splitlines()])
    q1, d1 = FE.build_camerabench_questions(names, lbs, FE.MAPPINGS["main"])
    q2, _ = FE.build_camerabench_questions(names, lbs, FE.MAPPINGS["main"])
    record("generator: deterministic (two builds give identical questions) and the digest is stable", FE.questions_digest(q1, True) == FE.questions_digest(q2, True), FE.questions_digest(q1)[:16])
    sets = FE.gold_sets(names, lbs, FE.MAPPINGS["main"])
    ok_dist = all(len({q["option_atoms"][L] for L in "ABCD"}) == 4 and q["option_atoms"][q["gold_letter"]] == q["gold_atoms"] for q in q1)
    pool_ok = all(all(q["option_atoms"][L] in [s for j, s in enumerate(sets) if j != q["index"] and s is not None] for L in "ABCD" if L != q["gold_letter"]) for q in q1)
    record("generator: four DISTINCT atom sets per question, the gold set is the clip's own, every distractor is the gold set of ANOTHER clip", ok_dist and pool_ok, f"{len(q1)} questions")
    record("generator: a clip with no mapped atom is excluded and counted (main), and its question exists under the no-motion-as-static mapping", d1["no_mapped_atom"] > 0 and len(FE.build_camerabench_questions(names, lbs, FE.MAPPINGS["no_motion_as_static"])[0]) > len(q1), json.dumps(d1))
    tiny = {"a.gif": ["pan-left"], "b.gif": ["pan-left"], "c.gif": ["tilt-up"], "d.gif": ["pan-left"]}
    qt, dt = FE.build_camerabench_questions(sorted(tiny), {k[:-4]: v for k, v in tiny.items()}, FE.MAPPINGS["main"])
    record("generator: fewer than 3 distinct distractors drops the question and counts it", qt == [] and dt["fewer_than_3_distinct_distractors"] == 4, json.dumps(dt))
    q_static = FE.build_camerabench_questions(["a.gif"] + [f"z{i}.gif" for i in range(5)], {"a": ["no-motion", "pan-left"], **{f"z{i}": [t] for i, t in enumerate(["pan-left", "tilt-up", "dolly-in", "zoom-in", "truck-left"])}}, FE.MAPPINGS["no_motion_as_static"])[0]
    record("mapping variants: 'no-motion' becomes static ONLY in the sensitivity mapping; tracking tags never create an atom", {"static"} <= set(q_static[0]["gold_atoms"]) and "static" not in FE.gold_sets(["a.gif"], {"a": ["no-motion", "pan-left"]}, FE.MAPPINGS["main"])[0]
           and FE.gold_sets(["a.gif"], {"a": ["side-tracking"]}, FE.MAPPINGS["main"])[0] is None)
    record("tie rule: equal scores choose the lowest letter", FE.choose([1, 1, 1, 1]) == "A" and FE.choose([0, 2, 2, 1]) == "B")
    q = {"gold_letter": "B", "gold_atoms": ("pan-left",), "option_atoms": {"A": ("tilt-up",), "B": ("pan-left",), "C": ("pan-left",), "D": ("zoom-in", "pan-left")}, "source": "s"}
    m = FE.option_accuracy([q, {**q, "gold_letter": "C"}, {**q, "gold_letter": "D", "gold_atoms": ("pan-left", "zoom-in")}], ["B", "B", "D"])
    record("metrics: accuracy 2/3, exact-set counts collapsed options (B and C are the same set), strata", m["correct"] == 2 and m["n_questions_with_collapsed_options"] == 3 and m["n_compound"] == 1 and m["exact_set_accuracy"] == 1.0, json.dumps(m)[:160])
    al = FE.atom_level([q], ["D"])
    record("metrics: atom-level precision/recall of the chosen option (gold pan-left; chosen pan-left+zoom-in)", {r["atom"]: (r["precision"], r["recall"]) for r in al["per_atom"]}["pan-left"] == (1.0, 1.0)
           and {r["atom"]: r["fp"] for r in al["per_atom"]}["zoom-in"] == 1)
    rng = np.random.default_rng(1)
    src = np.repeat(np.arange(30), 4)
    a_ = (rng.random(120) < 0.85)
    b_ = (rng.random(120) < 0.55)
    bs = FE.paired_bootstrap(a_, b_, src)
    record("paired bootstrap by source: identical hits give 0 [0, 0]; a clearly better model has an interval above 0", FE.paired_bootstrap(a_, a_, src)["ci95"] == [0.0, 0.0] and bs["ci95"][0] > 0, json.dumps(bs))
    sd = FE.source_disjoint([{"source": "a"}, {"source": "b"}, {"source": "c"}], {"a", "c"})
    record("source-disjoint subset keeps exactly the questions whose source id is absent from the reference pool", [q["source"] for q in sd] == ["b"])
    tr = json.loads((ROOT / "results/phase4/split_manifest.json").read_text())["clips"]["train"][:300]
    feats = pd.concat([pd.read_parquet(ROOT / "data/cache/p4" / f).set_index("clip") for f in ("features_shotqa.parquet", "features_depth_shotqa.parquet")], axis=1)
    from cgt.p4_camera_move.labels import MovementLabelStore
    store = MovementLabelStore(json.loads((ROOT / "results/phase4/split_manifest.json").read_text())["clips"]["val"])
    rows = [r for r in store.rows(tr) if r["atoms"]]
    at = [x for x in schema_moves() if any(x in r["atoms"] for r in rows)]
    X = feats.loc[[r["clip"] for r in rows]].to_numpy(dtype=np.float64)
    Y = np.array([[x in r["atoms"] for x in at] for r in rows], dtype=int)
    base = fit_scores(X[:200], Y[:200], X[200:], 0.01)
    mine = FE.fit_atom_probs(X[:200], Y[:200], X[200:], 0.01)
    record("geometry fit: `final_eval.fit_atom_probs` equals the validated `baseline.fit_scores` (same train rows, C = 0.01, 50 held-out rows from TRAIN clips)", float(np.abs(base - mine).max()) < 1e-9, f"max abs diff {float(np.abs(base - mine).max()):.2e}")
    same = FE.fit_atom_probs(X[:200], Y[:200], X[:200], 0.01, recalibrate=True)
    record("transductive variant: with the test features equal to the train features it equals the primary variant", float(np.abs(same - FE.fit_atom_probs(X[:200], Y[:200], X[:200], 0.01)).max()) < 1e-9)
    shifted = FE.fit_atom_probs(X[:200], Y[:200], X[200:] * 3.0 + 5.0, 0.01, recalibrate=True)
    record("transductive variant: uses only test FEATURES: an affine change of the test features (x3 + 5) leaves its probabilities unchanged, while the primary variant changes", float(np.abs(shifted - FE.fit_atom_probs(X[:200], Y[:200], X[200:], 0.01, recalibrate=True)).max()) < 1e-6
           and float(np.abs(FE.fit_atom_probs(X[:200], Y[:200], X[200:] * 3.0 + 5.0, 0.01) - mine).max()) > 1e-3)


def check_go_and_spec(tmp_root: Path) -> None:
    g = FE.go_text("shotbench_movement", "t", "a" * 64, None, "f" * 64)
    p = tmp_root / "go"
    p.write_text(g)
    ok = True
    for bad in (FE.go_text("camerabench_public_test", "t", "a" * 64, None, "f" * 64), FE.go_text("shotbench_movement", "u", "a" * 64, None, "f" * 64), FE.go_text("shotbench_movement", "t", "b" * 64, None, "f" * 64),
                FE.go_text("shotbench_movement", "t", "a" * 64, "c" * 16, "f" * 64), FE.go_text("shotbench_movement", "t", "a" * 64, None, "0" * 64)):
        try:
            FE.require_go(p, bad)
            ok = False
        except guards.NotAuthorized:
            pass
    FE.require_go(p, g)
    try:
        FE.require_go(tmp_root / "missing", g)
        ok = False
    except guards.NotAuthorized:
        pass
    record("go file: a different dataset, tag, adapter hash, config hash or freeze record, and a missing file, authorize nothing (exit 13); the exact text passes", ok and guards.NotAuthorized.exit_code == 13)
    for model in FE.MODELS:
        spec, reg, _ap = FR.make_spec("shotbench_movement", model)
        try:
            from cgt.p4_camera_move.camerabench import require_registered
            require_registered(spec.tag, reg, adapter_sha256=spec.adapter_sha256, config_hash=spec.config_hash)
            good = True
        except Exception:  # noqa: BLE001
            good = False
        record(f"make_spec ({model}): the hashes recomputed from the adapter file / frozen configuration equal the COMMITTED registry of the real freeze record", good,
               f"tag {spec.tag}, adapter {str(spec.adapter_sha256)[:12]}, config {spec.config_hash}")
    try:
        FR.make_spec("shotbench_movement", "video", git_ref="0000000000000000000000000000000000000000")
        nb = False
    except guards.TagNotBound:
        nb = True
    record("make_spec: a freeze record that is not committed at the git ref authorizes nothing (TagNotBound, exit 18)", nb)


def check_protocol(tmp_root: Path) -> None:
    ref = {"SRC0", "SRC1", "SRC2"}
    # 1. ShotBench movement, combined: label-free scoring BEFORE the ledger line, answers read AFTER it
    w, d = fresh(tmp_root, "sb_ok")
    opened = []
    real_read_csv, real_open = pd.read_csv, builtins.open

    def spy_csv(path, *a, **k):
        if "answer" in (k.get("usecols") or []):
            opened.append(("answers", len(FS.ledger_lines(w["ledger"]))))
        return real_read_csv(path, *a, **k)
    pd.read_csv = spy_csv
    try:
        code, _log, trace = run_once(w, "shotbench_movement", "combined", d, ref=ref)
    finally:
        pd.read_csv = real_read_csv
    kinds = [t[0] if t[0] != "backend" else f"backend:{t[1]}" for t in trace]
    record("ShotBench: the run succeeds and writes the result files", code == 0 and all(p.exists() for p in FR.result_paths(spec_for("shotbench_movement", "combined"), d / "out", d / "raw").values()), f"exit {code}")
    record("ShotBench: questions parsed without answers; geometry, model warm-up and the video scores are computed BEFORE the ledger line; the answers are read AFTER it",
           kinds.index("questions") < kinds.index("backend:video") < kinds.index("ledger") < kinds.index("labels") and opened == [("answers", 1)], f"order {kinds}, answer read saw {opened} ledger line(s)")
    res = json.loads(FR.result_paths(spec_for("shotbench_movement", "combined"), d / "out", d / "raw")["json"].read_text())
    preds = [json.loads(x) for x in FR.result_paths(spec_for("shotbench_movement", "combined"), d / "out", d / "raw")["predictions"].read_text().splitlines()]
    acc = res["variants"]["main"]["option_accuracy"]
    record("ShotBench: result accuracy equals the mean hit of the published predictions; the oracle model is right whenever the options do not collapse; secondary source-disjoint subset reported",
           abs(acc["acc"] - np.mean([p["hit"] for p in preds])) < 1e-4 and acc["n"] == 12 and res["secondary"]["source_disjoint_subset"]["n_questions"] == 12 - sum(1 for p in preds if p["source"] in ref),
           f"acc {acc['acc']}, n {acc['n']}, subset {res['secondary']['source_disjoint_subset']['n_questions']}")
    ledger = FS.ledger_lines(w["ledger"])
    record("ShotBench: exactly one ledger line with dataset, tag, hashes and code commit", len(ledger) == 1 and ledger[0]["dataset"] == "shotbench_movement" and ledger[0]["model_tag"] == FE.MODEL_TAGS["combined"]
           and ledger[0]["adapter_sha256"] == "a" * 64 and ledger[0]["code_sha"] == COMMIT, json.dumps(ledger[0])[:200])
    record("ShotBench: a second evaluation of the same (dataset, tag) is refused (exit 14) and appends nothing", run_once(w, "shotbench_movement", "combined", d)[0] == 14 and len(FS.ledger_lines(w["ledger"])) == 1)
    mani = json.loads(FR.result_paths(spec_for("shotbench_movement", "combined"), d / "out", d / "raw")["hash_manifest"].read_text())
    record("adapter hashes: BEFORE (score-free manifest) and AFTER are recorded and equal; no model hashes for the geometry run", mani["adapter_before"]["sha256"] == mani["adapter_after"]["sha256"] == "a" * 64 and mani["adapter_unchanged"])
    # 2. CameraBench, three models: identical generated questions, variants
    w, d = fresh(tmp_root, "cb_ok")
    digs = {}
    for m in FE.MODELS:
        tr = []
        code, _log, trace = run_once(w, "camerabench_public_test", m, d, trace=tr)
        r = json.loads(FR.result_paths(spec_for("camerabench_public_test", m), d / "out", d / "raw")["json"].read_text()) if code == 0 else {}
        digs[m] = (code, {k: v for k, v in r.get("questions_sha256_no_gold", {}).items() if k in ("main", "no_motion_as_static")}, sorted(r.get("variants", {})))
        if m == "combined":
            kinds = [t[0] if t[0] != "backend" else f"backend:{t[1]}" for t in tr]
    record("CameraBench: all three models run; the generated questions are identical across the models (same digest, paired comparison possible)", all(v[0] == 0 for v in digs.values()) and len({json.dumps(v[1], sort_keys=True) for v in digs.values()}) == 1 and all(v[1] for v in digs.values()),
           f"{ {m: v[0] for m, v in digs.items()} }")
    record("CameraBench variants: main + no_motion_as_static for every model; the transductive recalibration variant for geometry and combined only", digs["video"][2] == ["main", "no_motion_as_static"]
           and digs["geometry"][2] == ["main", "no_motion_as_static", "recalibrated_features"] == digs["combined"][2], json.dumps({m: v[2] for m, v in digs.items()}))
    record("CameraBench: the ledger line precedes the label read, which precedes the label-dependent video scoring (the options depend on the labels)", kinds.index("ledger") < kinds.index("labels") < kinds.index("backend:video"), f"order {kinds}")
    record("CameraBench: three models, three ledger lines", len(FS.ledger_lines(w["ledger"])) == 3)
    # the label file is opened only after the ledger line exists
    w, d = fresh(tmp_root, "cb_order")
    seen = []

    def spy_open(file, *a, **k):
        if str(file).endswith("test.jsonl"):
            seen.append(len(FS.ledger_lines(w["ledger"])))
        return real_open(file, *a, **k)
    builtins.open = io.open = spy_open  # pathlib calls io.open
    try:
        run_once(w, "camerabench_public_test", "geometry", d)
    finally:
        builtins.open = io.open = real_open
    record("CameraBench: test.jsonl is opened for the first time only when the ledger already holds the evaluation's line", seen[:1] == [1], f"ledger lines at first open: {seen}")
    # 3. paired comparison on the published predictions
    sys.path.insert(0, str(ROOT / "scripts"))
    import p4_final_compare as CMP
    w, d = fresh(tmp_root, "cmp")
    for m in FE.MODELS:
        run_once(w, "shotbench_movement", m, d, ref=ref)
    preds = {m: CMP.load(d / "raw", "shotbench_movement", m) for m in FE.MODELS}
    cres = CMP.compare(preds, ref)
    record("paired comparison: runs on the three published prediction files (variant main, source-disjoint subset), identical questions verified", "main" in cres and "source_disjoint_subset" in cres["main"] and set(cres["main"]["paired_bootstrap_by_source_1000"]) == {"combined_minus_geometry", "combined_minus_video", "video_minus_geometry"})
    preds["video"]["main"][0] = {**preds["video"]["main"][0], "options": {**preds["video"]["main"][0]["options"], "A": "Arc"}}
    try:
        CMP.compare(preds, ref)
        refused = False
    except SystemExit:
        refused = True
    record("paired comparison: refuses models that were not scored on identical questions", refused)


def check_failures(tmp_root: Path) -> None:
    # guards
    for label, kw, want in (("no go file", {"go": "none"}, 13), ("go file for another tag", {"go": "wrong_tag"}, 13), ("go file with another adapter hash", {"go": "wrong_adapter"}, 13), ("go file for another freeze record", {"go": "wrong_freeze"}, 13),
                            ("go file for another dataset", {"go": "wrong_dataset"}, 13), ("uncommitted code", {"git_dirty": True}, 10), ("altered original plan text", {"plan_ok": False}, 10),
                            ("tag missing from the committed registry", {"registry": {}}, 18), ("adapter hash differs from the registered hash", {"registry": {FE.MODEL_TAGS["combined"]: {"adapter_sha256": "9" * 64, "config_hash": "b" * 16}}}, 18),
                            ("adapter file hash differs at the start", {"hashes": ["7" * 64]}, 7), ("label-free inputs not prepared (preflight)", {"crash_in": "preflight"}, 6)):
        w, d = fresh(tmp_root, "g_" + label.replace(" ", "_"))
        code, _log, _ = run_once(w, "shotbench_movement", "combined", d, **kw)
        clean = not FS.ledger_lines(w["ledger"]) and not list((d / "raw").glob("*.CRASH.json")) and not list((d / "raw").glob("*.IN_PROGRESS.json")) and not list((d / "out").glob("*.results.*"))
        record(f"guard: {label} -> exit {want}; no ledger line, no state file, no result", code == want and clean, f"exit {code}")
    # crash BEFORE the commit: nothing leaks, ledger untouched, retry needs the ACK
    w, d = fresh(tmp_root, "crash_pre")
    code, log, _ = run_once(w, "shotbench_movement", "combined", d, crash_in="video", secret_in_error=True)
    txt = tree_text(d) + "\n".join(log)
    key = spec_for("shotbench_movement", "combined").key
    record("crash before the commit (secret planted in the error message): exit 15, crash file with class and generic reason only, ledger NOT written, nothing leaks into logs or files", code == 15 and (d / "raw" / f"{key}.CRASH.json").exists()
           and not FS.ledger_lines(w["ledger"]) and not leaked(txt) and "boom" not in txt, f"exit {code}, leaked {leaked(txt)}")
    record("crash before the commit: a retry is refused (exit 16) until the operator creates the CRASH_ACK; with the ACK it completes", run_once(w, "shotbench_movement", "combined", d)[0] == 16
           and (sealed.ack_path(d / "raw", key).write_text("ack") or True) and run_once(w, "shotbench_movement", "combined", d)[0] == 0 and len(FS.ledger_lines(w["ledger"])) == 1)
    # crash in the geometry stage and adapter change before the commit (ShotBench)
    w, d = fresh(tmp_root, "adapter_change_sb")
    code, _, _ = run_once(w, "shotbench_movement", "combined", d, hashes=["a" * 64, "b" * 64])
    record("ShotBench: the adapter changing during the (label-free) evaluation is a pre-commit crash: exit 15, ledger not written", code == 15 and not FS.ledger_lines(w["ledger"]), f"exit {code}")
    # crash AFTER the commit (CameraBench video scoring needs the labels): CONSUMED, no results, retry refused, author-gated resume
    w, d = fresh(tmp_root, "crash_post")
    key = spec_for("camerabench_public_test", "video").key
    code, log, _ = run_once(w, "camerabench_public_test", "video", d, crash_in="video", secret_in_error=True)
    consumed = d / "raw" / f"{key}.CONSUMED.json"
    txt = tree_text(d) + "\n".join(log)
    record("crash AFTER the commit (label-dependent scoring): exit 17, CONSUMED record, ledger line kept, no result file, no secret in any file or log",
           code == 17 and consumed.exists() and len(FS.ledger_lines(w["ledger"])) == 1 and not list((d / "out").glob("*.results.*")) and SECRET not in txt and "boom" not in txt, f"exit {code}, leaked {leaked(txt)}")
    record("CONSUMED: a plain retry is refused (exit 14)", run_once(w, "camerabench_public_test", "video", d)[0] == 14)
    record("CONSUMED: a resume without the operator's RESUME_ACK is refused (exit 16)", run_once(w, "camerabench_public_test", "video", d, resume=True)[0] == 16)
    sealed_ack = FSEAL.resume_ack_path(d / "raw", key)
    sealed_ack.write_text("ack")
    code = run_once(w, "camerabench_public_test", "video", d, resume=True)[0]
    record("CONSUMED: with the RESUME_ACK one resume completes; it appends NOTHING to the ledger, archives the old record, logs the resume; the ACK is consumed", code == 0 and len(FS.ledger_lines(w["ledger"])) == 1
           and not consumed.exists() and list((d / "raw").glob(f"{key}.CONSUMED.json.resumed.*")) and FSEAL.resumes_log_path(d / "raw", key).exists() and not sealed_ack.exists(), f"exit {code}")
    record("CONSUMED: a second resume is refused once results exist (exit 14)", run_once(w, "camerabench_public_test", "video", d, resume=True)[0] == 14)
    w, d = fresh(tmp_root, "resume_other_code")
    run_once(w, "camerabench_public_test", "video", d, crash_in="video")
    FSEAL.resume_ack_path(d / "raw", spec_for("camerabench_public_test", "video").key).write_text("ack")
    ledger_text = w["ledger"].read_text().replace(COMMIT, "f" * 40)
    w["ledger"].write_text(ledger_text)
    record("resume with a ledger line written by another code commit is refused (exit 10)", run_once(w, "camerabench_public_test", "video", d, resume=True)[0] == 10)
    # adapter change AFTER the commit
    w, d = fresh(tmp_root, "adapter_change_cb")
    code, _, _ = run_once(w, "camerabench_public_test", "video", d, hashes=["a" * 64, "b" * 64])
    record("CameraBench: the adapter changing during the label-dependent scoring leaves CONSUMED (exit 17) and publishes nothing", code == 17 and not list((d / "out").glob("*.results.*")), f"exit {code}")
    # UNKNOWN commit: the ledger cannot be written and cannot be read
    w, d = fresh(tmp_root, "unknown")
    spec, _store, _backend = build(w, "shotbench_movement", "geometry", d, [])
    key = spec.key
    (d / f"GO_{key}").write_text(spec.go())
    cap = Cap()
    ev = FinalSealed(key, d / "raw", cap, code_commit=COMMIT)

    def boom_commit():
        raise OSError("ledger unwritable")

    def boom_contains():
        raise OSError("ledger unreadable")
    try:
        ev.start(1)
        ev.seal_after_labels(boom_commit, boom_contains, lambda: ({d / "x.json": "secret 0.4321"}, "echo 0.4321"))
        code = 0
    except guards.CommitUnknown as e:
        code = sealed.exit_code_for(ev, e)
    txt = tree_text(d) + "\n".join(cap.lines)
    record("UNKNOWN commit (ledger neither writable nor readable): exit 19, only UNKNOWN_COMMIT.json is written, nothing published or echoed", code == 19 and (d / "raw" / f"{key}.UNKNOWN_COMMIT.json").exists() and SECRET not in txt and not (d / "x.json").exists())
    # NOT_COMMITTED: commit fails but the ledger is readable and has no line -> crash file, not consumed
    ev2 = FinalSealed(key + "_b", d / "raw", Cap(), code_commit=COMMIT)
    try:
        ev2.start(1)
        ev2.seal_after_labels(boom_commit, lambda: False, lambda: ({}, ""))
        code = 0
    except guards.EvaluationCrashed as e:
        code = sealed.exit_code_for(ev2, e)
    record("commit fails with a readable ledger holding no line: crash file, tag NOT consumed (exit 15)", code == 15 and (d / "raw" / f"{key}_b.CRASH.json").exists() and not (d / "raw" / f"{key}_b.CONSUMED.json").exists())
    # write failure after the commit: results echoed, CONSUMED
    ev3 = FinalSealed(key + "_c", d / "raw", Cap(), code_commit=COMMIT)
    blocked = d / "isadir"
    blocked.mkdir()
    try:
        ev3.start(1)
        ev3.seal_after_labels(lambda: None, lambda: True, lambda: ({blocked: "x", d / "ok.txt": "y"}, "echo"))
        code = 0
    except guards.PostLedgerWriteFailed as e:
        code = sealed.exit_code_for(ev3, e)
    record("write failure after the commit: the other results are still written, CONSUMED (exit 17)", code == 17 and (d / "ok.txt").exists() and (d / "raw" / f"{key}_c.CONSUMED.json").exists())
    # process-level: SIGKILL leaves the IN_PROGRESS marker, SIGTERM leaves a crash file
    for mode, want_marker in (("kill", True), ("term", False)):
        w, d = fresh(tmp_root, "child_" + mode)
        (d / "world.json").write_text(json.dumps({k: str(v) for k, v in w.items() if k in ("tsv", "cb_root", "repo", "freeze", "ledger")} | {"tmp": str(d)}))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", mode, "--tmp", str(d)], capture_output=True, text=True, check=False)
        key = spec_for("shotbench_movement", "combined").key
        inprog, crash = (d / "raw" / f"{key}.IN_PROGRESS.json").exists(), (d / "raw" / f"{key}.CRASH.json").exists()
        retry = run_once(w, "shotbench_movement", "combined", d)[0]
        record("process level: SIGKILL in the label-free stage leaves the IN_PROGRESS marker and blocks a retry (exit 16); SIGTERM writes a crash file" if mode == "kill" else "process level: SIGTERM in the label-free stage writes the crash file (reason terminated_by_signal), ledger untouched, retry refused (16)",
               (inprog and not crash if want_marker else crash and not inprog) and retry == 16 and not FS.ledger_lines(w["ledger"]), f"in_progress {inprog}, crash {crash}, retry exit {retry}")


def check_inputs() -> None:
    """GIF readers on synthetic GIFs: frame selection, lossless parity of the frame pipeline between a GIF and a video of the same frames, depth series parity."""
    import cv2
    from PIL import Image

    from cgt.p4_camera_move import depth_parallax as D
    from cgt.p4_camera_move import gif as G
    from cgt.p4_camera_move import video_vlm as V
    tmp = Path(tempfile.mkdtemp())
    n = 20
    frames = [Image.new("RGB", (160, 96), (10 * i, 255 - 10 * i, 40)) for i in range(n)]
    g = tmp / "t.gif"
    frames[0].save(g, save_all=True, append_images=frames[1:], duration=70, loop=0)
    out = G.gif_frames_k(g, 8, 64, tmp / "fr")
    got = [round(float(np.asarray(Image.open(p)).mean(axis=(0, 1))[0]) / 10) for p in out]
    want = V.frame_indices(n, 8)
    record("GIF frames for the video model: 8 frames at the centres of 8 equal time bins (same rule as videos), cached idempotently, within the 64-token budget", got == want and all(Image.open(p).size[0] * Image.open(p).size[1] <= 64 * 1024 for p in out) and G.gif_frames_k(g, 8, 64, tmp / "fr") == out, f"indices {got}")
    seven = [Image.new("RGB", (96, 64), (30 * i, 0, 0)) for i in range(7)]
    g7 = tmp / "seven.gif"
    seven[0].save(g7, save_all=True, append_images=seven[1:], duration=70, loop=0)
    o7 = G.gif_frames_k(g7, 8, 64, tmp / "fr7")
    reds = [round(float(np.asarray(Image.open(p)).mean(axis=(0, 1))[0]) / 30) for p in o7]
    record("GIF with fewer frames than K (7 < 8) is kept: the bin-centre rule over the true count repeats frames (nearest in time), never dropped", len(o7) == 8 and reds == V.frame_indices(7, 8), f"frames {reds}")
    flat = []
    for i in range(8):
        im = np.zeros((64, 96, 3), np.uint8)
        im[:, :, 0] = 200
        im[16:48, 8 + 6 * i:40 + 6 * i, 1] = 220
        flat.append(im)
    gp, av = tmp / "m.gif", tmp / "m.avi"
    Image.fromarray(flat[0][:, :, ::-1][:, :, ::-1]).save(gp, save_all=True, append_images=[Image.fromarray(x) for x in flat[1:]], duration=70, loop=0)
    wr = cv2.VideoWriter(str(av), cv2.VideoWriter_fourcc(*"FFV1"), 1000 / 70, (96, 64))
    for x in flat:
        wr.write(cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
    wr.release()
    col_g, gray_g, fps_g = G.gif_colour_gray(gp)
    col_v, gray_v, fps_v = D._frames(av)
    same = len(col_g) == len(col_v) and all(np.array_equal(a, b) for a, b in zip(gray_g, gray_v, strict=True)) and abs(fps_g - fps_v) < 0.05
    record("GIF vs video frame pipeline: for the same lossless frames the GIF reader returns the same grey frames and fps as the video reader used for the training features", same, f"{len(col_g)} frames, fps {fps_g:.2f}/{fps_v:.2f}")
    sg, sv = G.gif_depth_series(gp), D.clip_series(av)
    ok = all(np.allclose(sg[k], sv[k], rtol=1e-3, atol=1e-6) for k in D.KEYS) and np.array_equal(sg["depth"], sv["depth"])  # fps: the AVI container stores 14.286, the GIF durations give 14.2857
    record("GIF depth series: the depth-parallax code run on the GIF equals the run on the video of the same frames", ok)
    keys = D.clip_features({k: v for k, v in sg.items() if k != "depth"})
    record("GIF depth features have the training column set", len(keys) == 112, f"{len(keys)} columns")


def check_tripwire_and_dryrun() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import p4_final_eval as EV
    tmp = Path(tempfile.mkdtemp())
    (tmp / "camerabench").mkdir()
    (tmp / "camerabench/test.jsonl").write_text("{}")
    with EV.Tripwire():
        try:
            (tmp / "camerabench/test.jsonl").read_text()
            t1 = False
        except AssertionError:
            t1 = True
        try:
            pd.read_csv(tmp / "test.tsv", sep="\t")
            t2 = False
        except AssertionError:
            t2 = True
    record("dry-run tripwire: opening a CameraBench label file or reading the ShotBench TSV without excluding the answer column raises", t1 and t2)
    p = ROOT / "results/phase4/final_eval_dryrun.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    sb, cb = d.get("shotbench_movement", {}), d.get("camerabench_public_test", {})
    record("dry run on the REAL question files: 464 ShotBench movement questions parsed without answers, 464 videos in the tar, 427 share a ShotQA source, 37 in 18 sources are source-disjoint; 1,071 GIF names, 404 source ids; no label file opened; answer column excluded",
           sb.get("questions_parsed_without_answers") == 464 and sb.get("videos_in_tar") == 464 and sb.get("videos_sharing_a_source_with_shotqa") == 427 and sb.get("source_disjoint_subset") == {"videos": 37, "sources": 18}
           and cb.get("gif_names_listed") == 1071 and cb.get("distinct_source_ids") == 404 and d.get("tripwire", {}).get("label_files_opened") == [] and "answer" not in json.dumps(d.get("tripwire", {}).get("read_csv_calls")),
           json.dumps({"sb": sb.get("source_disjoint_subset"), "gifs": cb.get("gif_names_listed")}))
    record("dry run: every ShotBench prompt and the worst-case CameraBench prompt fit the 2,048-token trainer limit (text + 8 x 64 visual tokens)", sb.get("prompt_tokens_text_plus_8x64_visual", {}).get("max", 9999) < 2048 and cb.get("worst_case_prompt_tokens_four_four_atom_options", 9999) < 2048,
           f"max {sb.get('prompt_tokens_text_plus_8x64_visual', {}).get('max')}, worst CameraBench {cb.get('worst_case_prompt_tokens_four_four_atom_options')}")


def child(mode: str, tmp: Path) -> None:
    wj = json.loads((tmp / "world.json").read_text())
    w = {"tsv": Path(wj["tsv"]), "cb_root": Path(wj["cb_root"]), "repo": Path(wj["repo"]), "freeze": Path(wj["freeze"]), "ledger": Path(wj["ledger"]), "tmp": tmp}
    # the oracle is rebuilt from the synthetic files (names only matter for the crash stage)
    w["oracle"] = {}
    sealed.install_signal_handlers()
    spec, store, backend = build(w, "shotbench_movement", "combined", tmp, [], crash_in=f"{mode}_geometry")
    goff = tmp / f"GO_{spec.key}"
    goff.write_text(spec.go())
    cap = Cap()
    ev = FinalSealed(spec.key, tmp / "raw", cap, code_commit=COMMIT)
    try:
        FR.run_evaluation(spec, store, backend, ev=ev, ledger=w["ledger"], raw=tmp / "raw", out_dir=tmp / "out", go_path=goff, registry=REGISTRY,
                          git_state=lambda p: {"commit": COMMIT, "dirty_files": []}, plan_check=lambda: {"unchanged": True}, log=cap)
    except BaseException as e:  # noqa: BLE001
        if ev.started and not ev.finished:
            ev.record_crash(e)
        sys.exit(sealed.exit_code_for(ev, e))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child")
    ap.add_argument("--tmp")
    a = ap.parse_args()
    if a.child:
        child(a.child, Path(a.tmp))
        sys.exit(0)
    root = Path(tempfile.mkdtemp(prefix="p4final_"))
    check_pure(root)
    check_go_and_spec(root)
    check_protocol(root)
    check_failures(root)
    check_inputs()
    check_tripwire_and_dryrun()
    sentinel = (ROOT / "results/phase4/final_eval").exists() and any((ROOT / "results/phase4/final_eval").glob("GO_*"))
    record("no GO file exists in the repository's results/phase4/final_eval and the real ledger was not written by this self-check", not sentinel)
    ok = all(r[1] for r in results)
    L = ["# Sealed Phase 4 final evaluator: synthetic self-check", "", "CPU only; synthetic test sets with planted answers, an oracle fake backend, temporary directories; no model, no real test data, no real ledger, no GO file.", "",
         "| # | check | result | detail |", "|---:|---|---|---|"]
    L += [f"| {i} | {n} | {'PASS' if o else 'FAIL'} | {d.replace('|', '/')[:300]} |" for i, (n, o, d) in enumerate(results, 1)]
    L += ["", f"All {len(results)} checks passed: {ok}."]
    (ROOT / "results/phase4/final_eval_selfcheck.md").write_text("\n".join(L) + "\n")
    print(f"\nAll {len(results)} checks passed: {ok}")
    sys.exit(0 if ok else 1)
