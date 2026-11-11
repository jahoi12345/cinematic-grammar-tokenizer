"""Orchestration of ONE sealed Phase 4 final evaluation (docs/phase4_interface.md section 10): guards -> label-free `pre` stage -> ledger commit and label read -> `produce` -> result files.
Model, features and frames are behind a `Backend` object (duck-typed; the real one is in scripts/p4_final_eval.py, the synthetic one in scripts/p4_final_eval_selfcheck.py), the labels behind a
store (`final_stores`); the protocol itself (`final_seal.FinalSealed`) is the Phase 3B one. Guard order and exit codes (every guard fails closed, none restarts):
  14 the (dataset, tag) is already in the ledger or CONSUMED  |  16 an earlier crash / resume is unacknowledged  |  13 no go file, or not bound to dataset, tag, hashes and the committed freeze
  18 the tag is not registered in the COMMITTED freeze record, or its hashes differ  |  10 uncommitted code, altered original plan text, or the resume code commit differs
  7 the adapter file hash differs from the registered one (before) or changed during the evaluation (after)  |  15 crash before the commit  |  17 failure after the commit  |  19 UNKNOWN commit."""
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cgt.p3b_vlm import guards, sealed
from cgt.p3b_vlm.runstatus import now_iso
from cgt.p4_camera_move import final_eval as FE
from cgt.p4_camera_move import final_stores as FS
from cgt.p4_camera_move.camerabench import (
    CameraBenchAlreadyEvaluated,
    CameraBenchLocked,
    CameraBenchNotRegistered,
    require_registered,
)
from cgt.p4_camera_move.final_seal import FinalSealed, prepare_resume

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results/phase4/final_eval"
RAW_DIR = OUT_DIR / "raw"
CODE_PATHS = ["scripts", "src", "docs/phase4_video_plan.md", "pyproject.toml", "uv.lock"]
PLAN_COMMIT, PLAN_FILE = "49e248a", "docs/phase4_video_plan.md"


@dataclass(frozen=True)
class Spec:
    """One of the six evaluations. `adapter_sha256` / `config_hash` are the values THIS process recomputed from the files (the registry guard compares them with the committed ones)."""
    dataset: str
    model: str
    adapter_sha256: str | None
    config_hash: str | None
    w_s: float | None
    w_g: float | None
    freeze_sha256: str
    geometry_C: float | None = None  # from the COMMITTED geometry configuration (Amendment 12 item 3); None for the video model
    geometry_inputs: str | None = None

    @property
    def tag(self) -> str:
        return FE.MODEL_TAGS[self.model]

    @property
    def key(self) -> str:
        return FE.key_of(self.dataset, self.tag)

    def go(self) -> str:
        return FE.go_text(self.dataset, self.tag, self.adapter_sha256, self.config_hash, self.freeze_sha256, self.geometry_inputs)


FREEZE_REL = "results/phase4/runs/video_v1/FREEZE.json"


def read_committed(rel: str, root: Path = ROOT, git_ref: str = "HEAD") -> bytes:
    """The bytes of `rel` AS COMMITTED at `git_ref` (never the working copy). CameraBenchNotRegistered if it is not committed there."""
    r = subprocess.run(["git", "show", f"{git_ref}:{rel}"], cwd=root, capture_output=True, check=False)
    if r.returncode != 0:
        raise CameraBenchNotRegistered(f"{rel} is not committed at {git_ref}")
    return r.stdout


GEOMETRY_CONFIG = "results/phase4/frozen_config_flow_depth.json"
SPLIT_MANIFEST = "results/phase4/split_manifest.json"
TRAIN_TABLES = ("data/cache/p4/features_shotqa.parquet", "data/cache/p4/features_depth_shotqa.parquet")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def geometry_identity(root: Path = ROOT, git_ref: str = "HEAD") -> dict:
    """The frozen geometry identity (Amendment 12 item 3): the configuration read from its COMMITTED bytes (an edited working copy is refused, DirtyTree 10); a real baseline configuration must carry the
    hash recomputed from its content (TagNotBound 18); `inputs` = SHA-256 over the committed configuration bytes, the committed split manifest and the two training feature tables as they exist now (a missing
    file is recorded as such, the backend refuses to train without them). Returns {C, config_hash, inputs, split} (`split` = the parsed COMMITTED split manifest, the only split the evaluator trains on; E2b)."""
    from cgt.p4_camera_move.baseline import config_hash as cfg_hash
    try:
        cfg_bytes = read_committed(GEOMETRY_CONFIG, root, git_ref)
        man_bytes = read_committed(SPLIT_MANIFEST, root, git_ref) if (Path(root) / SPLIT_MANIFEST).exists() else b""
        if not man_bytes and (Path(root) / SPLIT_MANIFEST).exists():
            raise CameraBenchNotRegistered("split manifest not committed")
    except CameraBenchNotRegistered as e:
        raise guards.TagNotBound("the frozen geometry configuration is not committed") from e
    working = Path(root) / GEOMETRY_CONFIG
    if not working.exists() or working.read_bytes() != cfg_bytes:
        raise guards.DirtyTree("the geometry configuration differs from its committed bytes")
    split_path = Path(root) / SPLIT_MANIFEST
    if man_bytes and split_path.read_bytes() != man_bytes:  # E2b: the consumed training split must be the COMMITTED one, never an edited working copy
        raise guards.DirtyTree("the split manifest differs from its committed bytes")
    split = json.loads(man_bytes) if man_bytes else None
    cfg = json.loads(cfg_bytes)
    if split is not None and "manifest_sha256" in cfg and split.get("sha256") != cfg["manifest_sha256"]:
        raise guards.TagNotBound("the committed split manifest is not the one the geometry configuration was frozen with")
    if "c_grid" in cfg and cfg_hash({k: v for k, v in cfg.items() if k not in ("code_sha", "config_hash")}) != cfg["config_hash"]:
        raise guards.TagNotBound("the geometry configuration does not carry the hash of its content")
    tables = {t: (_sha(Path(root) / t) if (Path(root) / t).exists() else "missing") for t in TRAIN_TABLES}
    inputs = hashlib.sha256(json.dumps({"config": hashlib.sha256(cfg_bytes).hexdigest(), "split_manifest": hashlib.sha256(man_bytes).hexdigest(), "tables": tables}, sort_keys=True).encode()).hexdigest()
    return {"C": cfg["chosen_C"], "config_hash": cfg["config_hash"], "inputs": inputs, "split": split}


def make_spec(dataset: str, model: str, *, root: Path = ROOT, git_ref: str = "HEAD") -> tuple[Spec, dict, Path | None]:
    """The evaluation spec from the COMMITTED freeze record: its registry, selected epoch and frozen weights; the hashes the caller supplies to the registry guard are RECOMPUTED here from the
    adapter file and the committed geometry configuration (`geometry_identity`, geometry and combined), never copied from the registry or from an editable file. Returns (spec, registry, adapter path or
    None). Raises TagNotBound (18) if the freeze record or the geometry configuration is not committed, AdapterHashError (7) if the adapter file does not match the freeze record, DirtyTree (10) for an
    edited geometry configuration."""
    if dataset not in FE.DATASETS or model not in FE.MODELS:
        raise ValueError(f"unknown dataset or model: {dataset!r}, {model!r}")
    try:
        raw = read_committed(FREEZE_REL, root, git_ref)
        fz = json.loads(raw)
    except (CameraBenchNotRegistered, json.JSONDecodeError) as e:
        raise guards.TagNotBound("the freeze record is not committed (or is not valid JSON): no evaluation can be authorized") from e
    registry = fz.get("registered_models") or {}
    adapter_path, adapter_sha, config_hash, w_s, w_g, geo = None, None, None, None, None, None
    if model in ("video", "combined"):
        adapter_path = Path(root) / "data/cache/p4/video_runs" / fz["tag"] / "final" / fz["adapter"]["file"]
        adapter_sha = guards.sha256_file(adapter_path)
        if adapter_sha != fz["adapter"]["sha256"]:
            raise guards.AdapterHashError("the adapter file does not match the committed freeze record")
    if model in ("geometry", "combined"):
        geo = geometry_identity(root, git_ref)
    if model == "geometry":
        config_hash = geo["config_hash"]
    if model == "combined":
        w_s, w_g = fz["combination_weights"]["w_s"], fz["combination_weights"]["w_g"]
        config_hash = FE.combined_config_hash(fz["selected_epoch_E"], w_s, w_g, adapter_sha)
    return (Spec(dataset, model, adapter_sha, config_hash, w_s, w_g, FE.freeze_digest(raw), geo["C"] if geo else None, geo["inputs"] if geo else None), registry, adapter_path)


def result_paths(spec: Spec, out_dir: Path = OUT_DIR, raw: Path = RAW_DIR) -> dict[str, Path]:
    k = spec.key
    return {"json": Path(out_dir) / f"{k}.results.json", "md": Path(out_dir) / f"{k}.results.md", "predictions": Path(raw) / f"{k}.predictions.jsonl",
            "hash_manifest": Path(raw) / f"{k}.adapter_hash_manifest.json"}


def check_guards(spec: Spec, *, ledger: Path, raw: Path, go_path: Path, registry: dict, git_state, plan_check, adapter_hash, resume: bool, out_dir: Path = OUT_DIR) -> dict:
    """Every pre-evaluation guard, in the documented order. `git_state(paths) -> {commit, dirty_files}`, `plan_check() -> {unchanged, ...}` and `adapter_hash() -> sha256 | None` are injected
    (the real ones in the script). Returns the provenance dict; nothing here reads a label or a score."""
    key, tag = spec.key, spec.tag
    ledger_line = next((r for r in FS.ledger_lines(ledger) if r.get("dataset") == spec.dataset and r.get("model_tag") == tag), None)
    if ledger_line is not None and not resume:
        raise guards.LedgerUsed(f"'{key}' is already in the ledger: each (dataset, model tag) is evaluated exactly once")
    if not resume:
        sealed.require_not_consumed(raw, key)
        sealed.require_no_unacknowledged_crash(raw, key)
    try:
        require_registered(tag, registry, adapter_sha256=spec.adapter_sha256, config_hash=spec.config_hash)
    except CameraBenchNotRegistered as e:
        raise guards.TagNotBound(f"'{tag}' is not registered in the committed freeze record with these hashes") from e
    FE.require_go(go_path, spec.go())
    git = git_state(CODE_PATHS)
    if git["dirty_files"]:
        raise guards.DirtyTree(f"uncommitted changes under the run's code paths: {git['dirty_files'][:6]}")
    plan = plan_check()
    if not plan["unchanged"]:
        raise guards.DirtyTree("the original pre-registered plan text differs from its committed text")
    if resume:
        paths = [v for k, v in result_paths(spec, out_dir, raw).items() if k != "hash_manifest"]  # the BEFORE manifest is written at every start; it is not a result
        prepare_resume(raw, key, paths, ledger_line, git["commit"])
    before = adapter_hash()
    if before != spec.adapter_sha256:
        raise guards.AdapterHashError("the adapter file hash differs from the registered hash")
    return {"code_commit": git["commit"], "plan": plan, "adapter_before": before, "ledger_sha256_before": hashlib.sha256(Path(ledger).read_bytes()).hexdigest() if Path(ledger).exists() else None}


def _clip_rows(questions: list[dict]) -> list[str]:
    return [q["clip"] for q in questions]


def _scores(spec: Spec, qs: list[dict], Z, geo_probs: np.ndarray | None, atoms: list[str]) -> list[np.ndarray]:
    """Option scores of the model for the questions `qs`: Z[i] the four letter logits (video / combined), geo_probs[i] the atom probabilities of question i's clip (geometry / combined)."""
    out = []
    for i, q in enumerate(qs):
        ll = FE.geometry_option_scores(q, geo_probs[i], atoms) if geo_probs is not None else None
        out.append(FE.combine(spec.model, Z[i] if Z is not None else None, ll, spec.w_s or 0.0, spec.w_g or 0.0))
    return out


def _run_evaluation(spec: Spec, store, backend, *, ev: FinalSealed, ledger: Path, raw: Path, out_dir: Path, go_path: Path, registry: dict, git_state, plan_check, resume: bool = False,
                   reference_sources: set[str] | None = None, log=print) -> None:
    """The whole sealed evaluation. `backend` provides: `atoms`, `prior`, `adapter_hash()`, `geometry_probs(clips, recalibrate) -> (n_clips, n_atoms)` (label-free), `warm_up(clips)`
    (label-free: frames of every clip, model load and probes), `video_logits(questions, progress) -> list of 4 logits` (needs the option texts only, never a gold letter), `provenance() -> dict`."""
    raw, out_dir = Path(raw), Path(out_dir)
    prov = check_guards(spec, ledger=ledger, raw=raw, go_path=go_path, registry=registry, git_state=git_state, plan_check=plan_check, adapter_hash=backend.adapter_hash, resume=resume, out_dir=out_dir)
    ev.code_commit = prov["code_commit"]
    if hasattr(backend, "preflight"):  # read-only check that the label-free inputs were prepared (scripts/p4_final_prepare.py): UnreadableImage (6) BEFORE any state file is written
        backend.preflight(store)
    paths = result_paths(spec, out_dir, raw)
    raw.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": spec.dataset, "tag": spec.tag, "model": spec.model, "code_commit": prov["code_commit"], "plan": prov["plan"], "freeze_sha256": spec.freeze_sha256,
                "ledger_sha256_before": prov["ledger_sha256_before"], "adapter_before": {"sha256": prov["adapter_before"], "time": now_iso()}, "resume": resume}
    paths["hash_manifest"].write_text(json.dumps(manifest, indent=1))  # no scores in this file
    log(f"{spec.key}: guards passed; code {prov['code_commit'][:12]}; plan text unchanged: {prov['plan']['unchanged']}; adapter sha256 BEFORE: {prov['adapter_before']}; resume: {resume}")
    shotbench = spec.dataset == "shotbench_movement"
    needs_video, needs_geo = spec.model in ("video", "combined"), spec.model in ("geometry", "combined")
    holder: dict = {}

    def pre(progress):
        if shotbench:
            qs = store.questions()
            holder["clips"], holder["questions"] = _clip_rows(qs), qs
        else:
            holder["clips"] = store.clips()
        total = len(holder["clips"])
        log(f"inputs listed without labels: {total} clips")
        if needs_geo:
            holder["P"] = {"base": backend.geometry_probs(holder["clips"], False)}
            if not shotbench:
                holder["P"]["recal"] = backend.geometry_probs(holder["clips"], True)
            log("geometry atom probabilities computed (features and train labels only)")
        if needs_video:
            backend.warm_up(holder["clips"])  # frames of every clip, model load, dummy-letter independence check: all label-free
        if shotbench and needs_video:
            holder["Z"] = backend.video_logits(holder["questions"], progress)
            check_after_hash()
        progress(total, total)

    def check_after_hash():
        after = backend.adapter_hash()
        if after != prov["adapter_before"]:
            raise guards.AdapterHashError("the adapter changed during the evaluation")
        manifest["adapter_after"] = {"sha256": after, "time": now_iso()}
        manifest["adapter_unchanged"] = True

    def commit():
        kw = {"model_tag": spec.tag, "adapter_sha256": spec.adapter_sha256, "config_hash": spec.config_hash}
        try:
            if resume:
                holder["labels"] = store.labels_after_commit(**kw)
            else:
                holder["labels"] = store.labels(final_evaluation=True, code_sha=prov["code_commit"], **kw)
        except CameraBenchAlreadyEvaluated as e:
            raise guards.LedgerUsed("the ledger already holds this evaluation") from e
        except CameraBenchNotRegistered as e:
            raise guards.TagNotBound("the tag is not registered in the committed freeze record") from e
        except CameraBenchLocked as e:
            raise guards.NotAuthorized("the labels are sealed") from e

    def produce():
        variants, preds, secondary, dropped, qdigests = {}, [], {}, {}, {}
        atoms, prior = backend.atoms, backend.prior
        clips = holder["clips"]
        index_of = {c: i for i, c in enumerate(clips)}
        if shotbench:
            gold = holder["labels"]
            qs = [{**q, "gold_letter": gold[q["id"]], "gold_atoms": FE.sort_atoms(q["option_atoms"][gold[q["id"]]])} for q in holder["questions"]]
            sets = {"main": (qs, "base", holder.get("Z"))}
        else:
            lbs = FS.labels_by_stem(holder["labels"])
            q_main, d_main = FE.build_camerabench_questions(clips, lbs, FE.MAPPINGS["main"])
            q_var, d_var = FE.build_camerabench_questions(clips, lbs, FE.MAPPINGS["no_motion_as_static"])
            dropped = {"main": d_main, "no_motion_as_static": d_var}
            sets = {"main": (q_main, "base", None), "no_motion_as_static": (q_var, "base", None)}
            if needs_geo:
                sets["recalibrated_features"] = (q_main, "recal", None)
            if needs_video:  # label-dependent prompts: scored AFTER the commit; identical prompts are scored once
                uniq: dict = {}
                for q in q_main + q_var:
                    uniq.setdefault((q["clip"], tuple(q["options"][L] for L in FE.LETTERS)), q)
                ulist = list(uniq.values())
                zs = backend.video_logits(ulist, ev.progress)
                zmap = {(q["clip"], tuple(q["options"][L] for L in FE.LETTERS)): z for q, z in zip(ulist, zs, strict=True)}
                check_after_hash()
                sets = {name: (qs_, which, [zmap[(q["clip"], tuple(q["options"][L] for L in FE.LETTERS))] for q in qs_]) for name, (qs_, which, _) in sets.items()}
        for name, (qs_, which, Z) in sets.items():
            P = holder["P"][which][[index_of[q["clip"]] for q in qs_]] if needs_geo else None
            sc = _scores(spec, qs_, Z, P, atoms)
            chosen = [FE.choose(s) for s in sc]
            variants[name] = FE.variant_report(qs_, chosen, geometry_probs=P if spec.model == "geometry" else None, atoms=atoms, prior=prior)
            variants[name]["n_questions"] = len(qs_)
            qdigests[name] = FE.questions_digest(qs_, with_gold=False)
            for q, c, s in zip(qs_, chosen, sc, strict=True):
                preds.append({"variant": name, "id": q["id"], "clip": q["clip"], "source": q["source"], "options": q["options"], "gold_letter": q["gold_letter"], "gold_atoms": list(q["gold_atoms"]),
                              "chosen": c, "hit": c == q["gold_letter"], "scores": [round(float(x), 5) for x in s]})
            if shotbench:
                ref = reference_sources or set()
                sub = FE.source_disjoint(qs_, ref)
                idx = [i for i, q in enumerate(qs_) if q["source"] not in ref]
                secondary["source_disjoint_subset"] = {"declared_secondary": True, "n_questions": len(sub), "n_sources": len({q["source"] for q in sub}),
                                                       "option_accuracy": FE.option_accuracy(sub, [chosen[i] for i in idx]) if sub else {"n": 0},
                                                       "note": "videos whose filename-derived source id has no ShotQA clip; small, report with the Wilson interval; the other videos share a source id with the ShotQA clip pool"}
        pred_text = "\n".join(json.dumps(p) for p in preds) + "\n"
        res = {"dataset": spec.dataset, "model": spec.model, "tag": spec.tag, "key": spec.key, "adapter_sha256": spec.adapter_sha256, "config_hash": spec.config_hash,
               "weights": {"w_s": spec.w_s, "w_g": spec.w_g} if spec.model == "combined" else None, "freeze_sha256": spec.freeze_sha256, "code_commit": prov["code_commit"], "plan": prov["plan"],
               "frozen_at_ledger": ledger_tail(ledger, spec), "resume": resume, "primary_variant": "main", "variants": variants, "secondary": secondary, "dropped": dropped,
               "questions_sha256_no_gold": qdigests, "predictions_sha256": hashlib.sha256(pred_text.encode()).hexdigest(), "backend": backend.provenance(),
               "claims": ("Point estimates with 95% intervals for THIS model on THIS test set; paired statements among the three models only through scripts/p4_final_compare.py (same questions, "
                          "bootstrap by source video). ShotBench movement shares filename-derived source ids with ShotQA (not film-disjoint); the source-disjoint subset is a small secondary report. "
                          "The CameraBench variants (mapping sensitivity, recalibrated features) are declared secondary; nothing was selected on this test set.")}
        manifest["predictions_sha256"] = res["predictions_sha256"]
        outputs = {paths["json"]: json.dumps(res, indent=1), paths["md"]: render_md(res), paths["predictions"]: pred_text, paths["hash_manifest"]: json.dumps(manifest, indent=1)}
        main = variants["main"]["option_accuracy"]
        echo = (f"DONE {spec.key}: option accuracy {main['acc']:.4f} ({main['correct']}/{main['n']}), 95% Wilson {main['ci95_pct']}; variants {sorted(variants)}; "
                f"adapter unchanged; ledger recorded; results {paths['json'].name}")
        return outputs, echo

    ev.start(1)
    ev.run(pre)
    ev.seal_after_labels(commit, lambda: FS.ledger_contains(ledger, spec.dataset, spec.tag), produce)


def run_evaluation(spec: Spec, store, backend, **kw):
    """`_run_evaluation` under the cross-process RESERVATION of Amendment 12 item 2b: an exclusive non-blocking lock on `<ledger>.run.lock` for the whole run (every dataset, every model, resume included).
    A second invocation anywhere refuses with exit 14 BEFORE any state file or label access; the OS releases the lock however the process ends."""
    cm = FS.ledger_lock(kw["ledger"], ".run.lock", blocking=False)
    try:
        cm.__enter__()
    except BlockingIOError as e:
        raise guards.LedgerUsed("another final evaluation holds the reservation of the ledger (run lock): refused") from e
    try:
        return _run_evaluation(spec, store, backend, **kw)
    finally:
        cm.__exit__(None, None, None)


def ledger_tail(ledger: Path, spec: Spec) -> dict | None:
    return next((r for r in reversed(FS.ledger_lines(ledger)) if r.get("dataset") == spec.dataset and r.get("model_tag") == spec.tag), None)


def render_md(res: dict) -> str:
    head = (f"- dataset `{res['dataset']}`, model `{res['model']}` (tag `{res['tag']}`), code commit `{res['code_commit'][:12]}`, adapter `{(res['adapter_sha256'] or '-')[:16]}`, "
            f"config hash `{res['config_hash'] or '-'}`, freeze record sha256 `{res['freeze_sha256'][:16]}`; weights {res['weights']}")
    L = [f"# Final evaluation `{res['key']}`", "", head, "",
         "| variant | n | option accuracy | 95% Wilson (%) | single-label | compound | exact-set | atom macro F1 |", "|---|---:|---:|---|---:|---:|---:|---:|"]
    for name, v in res["variants"].items():
        a = v["option_accuracy"]
        if a["n"] == 0:
            L.append(f"| {name} | 0 | - | - | - | - | - | - |")
            continue
        sg = "-" if a["acc_single"] is None else f"{a['acc_single']:.3f} (n={a['n_single']})"
        cp = "-" if a["acc_compound"] is None else f"{a['acc_compound']:.3f} (n={a['n_compound']})"
        L.append(f"| {name}{' (primary)' if name == res['primary_variant'] else ''} | {a['n']} | {a['acc']:.4f} | {a['ci95_pct']} | {sg} | {cp} | {a['exact_set_accuracy']:.3f} | "
                 f"{v['atom_level_chosen_option']['macro'].get('f1', float('nan')):.3f} |")
    for name, s in res["secondary"].items():
        a = s["option_accuracy"]
        L += ["", f"Secondary `{name}`: n = {s['n_questions']} ({s['n_sources']} sources); " + (f"accuracy {a['acc']:.4f} ({a['correct']}/{a['n']}), 95% Wilson {a['ci95_pct']}" if a["n"] else "empty") + f". {s['note']}"]
    if res["dropped"]:
        L += ["", "Generated-question bookkeeping: " + json.dumps(res["dropped"])]
    L += ["", res["claims"], ""]
    return "\n".join(L)
