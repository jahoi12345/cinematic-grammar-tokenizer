"""SEALED Phase 4 final evaluator: ONE of the six pre-registered evaluations {geometry, video, combined} x {ShotBench movement, CameraBench public test} per invocation
(docs/phase4_video_plan.md Amendments 2 and 9, docs/phase4_interface.md section 10). Reuses the Phase 3B sealed design. NOTHING is trained, tuned or selected here; the frozen adapter and the
frozen geometry configuration come from the COMMITTED freeze record (`results/phase4/runs/video_v1/FREEZE.json`).
Real-mode order: (1) the (dataset, tag) is not in the ledger and not CONSUMED, an earlier crash is acknowledged (14, 16); (2) the tag is registered in the committed freeze record with the
hashes recomputed here (18); (3) the author's go file for exactly this dataset, tag, hashes and committed freeze record exists (13); (4) code committed, original plan text unchanged (10); (5) the
adapter file hash equals the registered one and a score-free BEFORE manifest is written (7); (6) label-free stage: inputs listed without labels, geometry probabilities (features + train labels),
frames and model for the video model, scores that need no label; (7) the LEDGER LINE is written and fsynced, THEN the label file is opened; (8) everything that needs a label: generated CameraBench
questions, label-dependent prompts, metrics; (9) result files, then the echo. Until (7) only progress COUNTS are logged and no exception message is kept (crash file, exit 15, retry needs
`<key>.CRASH_ACK`); a failure after (7) leaves `<key>.CONSUMED.json`; an unreadable ledger leaves
`<key>.UNKNOWN_COMMIT.json` (exit 19). Raw stdout is kept in results/phase4/final_eval/raw/<key>.stdout.txt.
--dry-run only parses the question files (no answer, no label file: tripwire), builds prompts and token lengths and exercises the generator on SYNTHETIC labels; no model, no frames, no ledger.
Usage: uv run python scripts/p4_final_eval.py --dry-run
       uv run python scripts/p4_final_eval.py --dataset shotbench_movement --model combined      """
import argparse
import builtins
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "2")  # scheduling only (Amendment 7): fewer Metal command-buffer operations per submit

import numpy as np
import pandas as pd

from cgt.p3b_vlm import guards, sealed
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p3b_vlm.runstatus import now_iso, plan_original_unchanged
from cgt.p4_camera_move import final_eval as FE
from cgt.p4_camera_move import final_inputs as FI
from cgt.p4_camera_move import final_run as FR
from cgt.p4_camera_move import final_stores as FS
from cgt.p4_camera_move import gif as G
from cgt.p4_camera_move import gpulock
from cgt.p4_camera_move import video_vlm as V
from cgt.p4_camera_move.camerabench import CB_ROOT, PRIMITIVE_TO_ATOM, source_id_of
from cgt.p4_camera_move.final_seal import FinalSealed
from cgt.p4_camera_move.frames_fallback import extract_frames_robust
from cgt.p4_camera_move.labels import (
    MovementLabelStore,
    load_shotqa_movement_meta,
    parse_move_label,
    schema_moves,
)

ROOT = Path(__file__).resolve().parents[1]
REPO, REVISION = "Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"
K, TOKENS = 8, 64  # frames per clip and visual tokens per frame of the frozen video model (Amendment 3)
FRAMES = ROOT / "data/cache/p4/video_frames"
CACHE = ROOT / "data/cache/p4"
EVAL_CACHE = ROOT / "data/cache/p4_eval"
SB_VIDEO_DIR = EVAL_CACHE / "shotbench_movement_video"  # extracted by scripts/p4_final_prepare.py from videos.tar (label-free)
FEATURES = {"shotbench_movement": EVAL_CACHE / "features_shotbench_movement.parquet", "camerabench_public_test": EVAL_CACHE / "features_camerabench_public_test.parquet"}
JOB = "p4-final-eval"
LABEL_FILES = ("test.jsonl", "videos.csv")  # CameraBench files that may hold labels: the dry run must never open them


class TrainContext:
    """TRAIN labels and features only (the validation labels stay locked): the geometry model's training matrix, atom vocabulary and label prior, exactly as in scripts/p4_video_validate.py."""

    def __init__(self, C: float | None, split: dict):
        man = split  # the COMMITTED split manifest (final_run.geometry_identity), never the editable working copy
        store = MovementLabelStore(man["clips"]["val"])
        rows = [r for r in store.rows(man["clips"]["train"]) if r["atoms"]]
        self.atoms = [x for x in schema_moves() if any(x in r["atoms"] for r in rows)]
        feats = pd.concat([pd.read_parquet(CACHE / f).set_index("clip") for f in ("features_shotqa.parquet", "features_depth_shotqa.parquet")], axis=1)
        self.columns = list(feats.columns)
        self.Xtr = feats.loc[[r["clip"] for r in rows]].to_numpy(dtype=np.float64)
        self.Ytr = np.array([[a in r["atoms"] for a in self.atoms] for r in rows], dtype=int)
        self.prior = self.Ytr.mean(0)
        self.C = C  # the C of the COMMITTED geometry configuration (spec.geometry_C); None for the video model, which never fits the geometry head
        self.n_train = len(rows)


class RealBackend:
    """Features, geometry, frames and the frozen video model for the real evaluation (the synthetic one is in scripts/p4_final_eval_selfcheck.py)."""

    def __init__(self, spec: FR.Spec, adapter_path: Path | None, ctx: TrainContext):
        self.spec, self.adapter_path, self.ctx = spec, adapter_path, ctx
        self.atoms, self.prior = ctx.atoms, ctx.prior
        self.model = self.ds = self.ids4 = None
        self.timing: dict = {}
        self._frames_cache: dict = {}
        self.manifest: dict | None = None

    def adapter_hash(self) -> str | None:
        return guards.sha256_file(self.adapter_path) if self.adapter_path else None

    def _manifest_path(self) -> Path:
        return ROOT / "results/phase4/final_eval" / f"prepare_{self.spec.dataset}.json"

    def preflight(self, store) -> None:
        """Read-only, before any state file or label access (Amendment 13 item 2): the preparation manifest, the table digest, unique complete coverage, the BYTES of every input file, the column hash, the code
        and conformity provenance (`final_inputs.validate_prepared`). Nothing is read from a label file."""
        clips = [q["clip"] for q in store.questions()] if self.spec.dataset == "shotbench_movement" else store.clips()
        root = SB_VIDEO_DIR if self.spec.dataset == "shotbench_movement" else CB_ROOT / "videos_gif"
        geo = self.spec.model in ("geometry", "combined")
        self.manifest = FI.validate_prepared(self.spec.dataset, clips, root, FEATURES[self.spec.dataset] if geo else None, self._manifest_path(), train_columns=getattr(self.ctx, "columns", None), root=ROOT)

    def geometry_probs(self, clips: list[str], recalibrate: bool) -> np.ndarray:
        path = FEATURES[self.spec.dataset]
        if not path.exists() or FI.sha256_file(path) != FI.load_manifest(self._manifest_path())["feature_table_sha256"]:  # re-checked at use: the table must still be the prepared one
            raise guards.UnreadableImage("the prepared feature table is missing or no longer matches its manifest")
        feats = pd.read_parquet(path).set_index("clip")
        if list(feats.columns) != self.ctx.columns:
            raise guards.FailClosed("the test feature columns differ from the training feature columns")
        X = feats.loc[clips].to_numpy(dtype=np.float64)
        return FE.fit_atom_probs(self.ctx.Xtr, self.ctx.Ytr, X, self.ctx.C, recalibrate=recalibrate)

    def _frames(self, clip: str) -> list[Path]:
        if self.spec.dataset == "shotbench_movement":
            return extract_frames_robust(SB_VIDEO_DIR / clip, K, TOKENS, FRAMES)
        return G.gif_frames_k(CB_ROOT / "videos_gif" / clip, K, TOKENS, FRAMES)

    def _sample(self, q: dict, letter: str = "A") -> dict:
        return V.build_sample({"clip": q["clip"], "options": q["options"], "gold_letter": letter}, self._frames_cache[q["clip"]])

    def warm_up(self, clips: list[str]) -> None:
        """Label-free: frames of every clip (cached), the pinned base model with the frozen adapter, and the dummy-letter independence check (the logits must not depend on the placeholder
        answer letter, so scoring needs no label)."""
        import mlx.core as mx
        from mlx_vlm.trainer.datasets import VisionDataset
        from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
        from mlx_vlm.utils import load

        from cgt.p3b_vlm.score import answer_token_ids
        from cgt.p4_camera_move.video_score import ListView
        t0 = time.perf_counter()
        self._frames_cache = {c: self._frames(c) for c in clips}
        self.timing["frames_s"] = round(time.perf_counter() - t0, 1)
        mx.set_cache_limit(3 * 1024**3)
        model, processor = load(pinned_model_path(REPO, REVISION))
        V.configure_processor(processor)
        self.ids4 = answer_token_ids(processor)
        model = get_peft_model(model, find_all_linear_names(model.language_model), rank=16, alpha=32.0, dropout=0.05, verbose=False)
        model.load_weights(list(mx.load(str(self.adapter_path)).items()), strict=False)
        model.eval()
        self.model = model
        self.ds = VisionDataset(ListView([]), model.config.__dict__, processor, None, True)
        probe = {"clip": clips[0], "options": {"A": "Pan left", "B": "Pan right", "C": "Tilt up", "D": "Tilt down"}}
        za = self._score([self._sample(probe, "A")], ["A"])[0]
        zb = self._score([self._sample(probe, "B")], ["B"])[0]
        if not np.allclose(za, zb, atol=1e-4):
            raise guards.LeakProbeFailed("the letter logits depend on the placeholder answer letter")
        self.timing["warm_up_s"] = round(time.perf_counter() - t0, 1)

    def _score(self, samples: list[dict], letters: list[str]) -> list[np.ndarray]:
        from cgt.p4_camera_move.video_score import score_samples
        return score_samples(self.model, self.ds, samples, letters, self.ids4)

    def video_logits(self, questions: list[dict], progress) -> list[np.ndarray]:
        import mlx.core as mx
        out, total = [], len(questions)
        for i in range(0, total, 20):
            chunk = questions[i:i + 20]
            out += self._score([self._sample(q) for q in chunk], ["A"] * len(chunk))
            mx.clear_cache()
            progress(len(out), total)
        return out

    def provenance(self) -> dict:
        import mlx.core as mx
        import mlx_vlm
        feat = FEATURES[self.spec.dataset]
        return {"frames_per_clip": K, "tokens_per_frame": TOKENS, "mlx": mx.__version__, "mlx_vlm": getattr(mlx_vlm, "__version__", None), "base_model": f"{REPO}@{REVISION}",
                "prepared_manifest_sha256": FI.sha256_file(self._manifest_path()) if self._manifest_path().exists() else None, "conformity": FI.conform.provenance(),
                "geometry": {"train_clips": self.ctx.n_train, "atoms": self.atoms, "C": self.ctx.C, "features": [str(c) for c in self.ctx.columns[:2]] + ["..."],
                             "test_feature_table_sha256": guards.sha256_file(feat) if feat.exists() else None}, "timing": self.timing,
                "max_ops_per_buffer": os.environ.get("MLX_MAX_OPS_PER_BUFFER")}


# ---------------------------------------------------------------------------------------------------- dry run
class Tripwire:
    """Records every file opened through `open` and every `pandas.read_csv` call during the dry run; the label files must never appear and test.tsv only with the answer column excluded."""

    def __enter__(self):
        self.opened: list[str] = []
        self.csv_calls: list[tuple[str, list | None]] = []
        self._open, self._read_csv = builtins.open, pd.read_csv
        tw = self

        def spy_open(file, *a, **k):
            tw.opened.append(str(file))
            if Path(str(file)).name in LABEL_FILES and "camerabench" in str(file).lower():
                raise AssertionError(f"dry run tried to open a label file: {file}")
            return tw._open(file, *a, **k)

        def spy_read_csv(path, *a, **k):
            tw.csv_calls.append((str(path), k.get("usecols")))
            if str(path).endswith("test.tsv") and (k.get("usecols") is None or "answer" in k["usecols"]):
                raise AssertionError("dry run tried to read the ShotBench answer column")
            return tw._read_csv(path, *a, **k)
        builtins.open = io.open = spy_open  # pathlib's read_text / read_bytes go through io.open
        pd.read_csv = spy_read_csv
        return self

    def __exit__(self, *exc):
        builtins.open = io.open = self._open
        pd.read_csv = self._read_csv

    def label_files_opened(self) -> list[str]:
        return [f for f in self.opened if Path(f).name in LABEL_FILES and "camerabench" in f.lower()]


def _prompt_tokens(processor, row: dict) -> int:
    sample = V.build_sample(row, [Path(f"frame{i}.jpg") for i in range(K)])
    user = {"role": "user", "content": [*([{"type": "image"}] * K), {"type": "text", "text": sample["messages"][0]["content"]}]}
    text = processor.apply_chat_template([user, {"role": "assistant", "content": [{"type": "text", "text": "A"}]}], add_generation_prompt=False)
    return len(processor.tokenizer.encode(text, add_special_tokens=False)) + K * TOKENS


def dry_run() -> dict:
    from transformers import AutoProcessor
    out: dict = {"mode": "DRY RUN: no model, no frame, no ledger entry, no label file opened (tripwire); the ShotBench answer column is excluded at the parser", "time": now_iso()}
    with Tripwire() as tw:
        geo = FR.geometry_identity()
        ctx = TrainContext(geo["C"], geo["split"])
        processor = AutoProcessor.from_pretrained(pinned_model_path(REPO, REVISION))
        events: list = []
        store = FS.ShotBenchMovementStore(events=events)
        qs = store.questions()
        with tarfile.open(FS.SB / "videos.tar") as t:
            names = set(t.getnames())
        ref = {m["source"] for m in load_shotqa_movement_meta()}
        disjoint = FE.source_disjoint(qs, ref)
        opts = [(k, v, q["option_atoms"][k]) for q in qs for k, v in q["options"].items()]
        toks = [_prompt_tokens(processor, {"clip": q["clip"], "options": q["options"], "gold_letter": "A"}) for q in qs]
        out["shotbench_movement"] = {
            "questions_parsed_without_answers": len(qs), "events": events, "question_text_constant": FS.QUESTION_TEXT, "videos_in_tar": sum(q["path"] in names for q in qs),
            "distinct_sources": len({q["source"] for q in qs}), "videos_sharing_a_source_with_shotqa": len(qs) - len(disjoint), "source_disjoint_subset": {"videos": len(disjoint), "sources": len({q["source"] for q in disjoint})},
            "options": {"n": len(opts), "with_unmapped_parts": sum(bool(parse_move_label(v).unmapped) for _, v, _ in opts), "compound": sum(len(a) > 1 for _, _, a in opts), "without_any_schema_atom": sum(not a for _, _, a in opts),
                        "atoms_outside_train_vocabulary": sorted({x for _, _, a in opts for x in a} - set(ctx.atoms)), "options_collapsing_in_a_question": sum(len(set(q["option_atoms"].values())) < 4 for q in qs)},
            "prompt_tokens_text_plus_8x64_visual": {"max": max(toks), "median": int(np.median(toks)), "trainer_limit": 2048}}
        names_gif = [p.name for p in FS.CameraBenchFinalStore(events=events).video_files()]
        rng = np.random.default_rng(123)
        vocab = [*PRIMITIVE_TO_ATOM, "no-motion"]
        fake = {n.rsplit(".", 1)[0]: [vocab[i] for i in rng.choice(len(vocab), int(rng.integers(1, 4)), replace=False)] for n in names_gif}
        demo = {}
        for name, mp in FE.MAPPINGS.items():
            q_, d_ = FE.build_camerabench_questions(names_gif, fake, mp)
            demo[name] = {"n_questions": len(q_), "dropped": d_, "gold_in_each_letter": {L: sum(q["gold_letter"] == L for q in q_) for L in FE.LETTERS},
                          "questions_sha256_no_gold": FE.questions_digest(q_), "repeat_identical": FE.questions_digest(FE.build_camerabench_questions(names_gif, fake, mp)[0]) == FE.questions_digest(q_)}
        longest = [FE.option_text(s) for s in (schema_moves()[:4], schema_moves()[3:7], schema_moves()[6:10], schema_moves()[10:14])]
        worst = _prompt_tokens(processor, {"clip": "x.gif", "options": dict(zip(FE.LETTERS, longest, strict=True)), "gold_letter": "A"})
        out["camerabench_public_test"] = {"gif_names_listed": len(names_gif), "distinct_source_ids": len({source_id_of(n) for n in names_gif}), "labels_opened": False,
                                          "generator_on_SYNTHETIC_labels": demo, "worst_case_prompt_tokens_four_four_atom_options": worst, "trainer_limit": 2048}
        out["train_context"] = {"n_train_clips": ctx.n_train, "atoms": ctx.atoms, "C": ctx.C, "n_features": len(ctx.columns)}
    out["tripwire"] = {"label_files_opened": tw.label_files_opened(), "read_csv_calls": [[Path(p).name, u] for p, u in tw.csv_calls], "n_files_opened": len(tw.opened)}
    if tw.label_files_opened():
        raise AssertionError("a label file was opened during the dry run")
    return out


# ---------------------------------------------------------------------------------------------------- real run
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=FE.DATASETS)
    ap.add_argument("--model", choices=FE.MODELS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="one operator-acknowledged re-run of a CONSUMED evaluation without results (needs <key>.RESUME_ACK)")
    ap.add_argument("--go-file", default=None, help="go-ahead file (default results/phase4/final_eval/GO_<key>)")
    ap.add_argument("--raw-dir", default=str(FR.RAW_DIR))
    ap.add_argument("--out-dir", default=str(FR.OUT_DIR))
    a = ap.parse_args(argv)
    if a.dry_run:
        res = dry_run()
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        (ROOT / "results/phase4/final_eval_dryrun.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    if not (a.dataset and a.model):
        ap.error("--dataset and --model are required (or --dry-run)")
    raw, out_dir = Path(a.raw_dir), Path(a.out_dir)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    ev = None
    try:
        spec, registry, adapter_path = FR.make_spec(a.dataset, a.model)
    except guards.FailClosed as e:
        print(f"FAIL CLOSED ({sealed.safe_text(e)}, exit {sealed.safe_exit_code(e)})")
        sys.exit(sealed.safe_exit_code(e))
    raw.mkdir(parents=True, exist_ok=True)
    rawf = raw / f"{spec.key}.stdout.txt"
    if not rawf.exists():
        rawf.write_text(f"# raw stdout of scripts/p4_final_eval.py --dataset {a.dataset} --model {a.model}\n# started {now_iso()}; code commit {sha}\n# argv: {sys.argv}\n")

    def log(msg: str) -> None:
        line = f"[{now_iso()}] {msg}"
        print(line, flush=True)
        with open(rawf, "a") as f:
            f.write(line + "\n")
    ev = FinalSealed(spec.key, raw, log, code_commit=sha)
    sealed.install_signal_handlers()
    locked = False
    try:
        if spec.model != "geometry":  # one heavy GPU job at a time (the agreed lock protocol); never removes another owner's lock
            if not gpulock.acquire(JOB, spec.key):
                log(f"GPU lock held by: {gpulock.owner()} (exit 11); nothing was started")
                sys.exit(11)
            locked = True
        go_path = Path(a.go_file) if a.go_file else Path(a.out_dir) / f"GO_{spec.key}"
        backend = RealBackend(spec, adapter_path, TrainContext(spec.geometry_C, FR.geometry_identity()["split"]))
        ref = {m["source"] for m in load_shotqa_movement_meta()} if spec.dataset == "shotbench_movement" else None
        store = FS.ShotBenchMovementStore() if spec.dataset == "shotbench_movement" else FS.CameraBenchFinalStore()
        FR.run_evaluation(spec, store, backend, ev=ev, ledger=FS.LEDGER, raw=raw, out_dir=out_dir, go_path=go_path, registry=registry, git_state=guards.git_state,
                          plan_check=lambda: plan_original_unchanged(plan_commit=FR.PLAN_COMMIT, plan_file=FR.PLAN_FILE), resume=a.resume, reference_sources=ref, log=log)
    except guards.FailClosed as e:  # only the exact known guard classes carry their documented exit code; nothing else is reflected
        if ev.started and not ev.finished:
            ev.record_crash(e)
        code = sealed.exit_code_for(ev, e)
        log(f"FAIL CLOSED ({sealed.safe_text(e)}, exit {code})")
        sys.exit(code)
    except BaseException as e:  # noqa: BLE001 - any failure at all must leave a crash file (class + code + count), never a score
        if ev.started and not ev.finished:
            ev.record_crash(e)
        log(f"CRASHED ({sealed.safe_text(e)}); no scores written")
        sys.exit(sealed.exit_code_for(ev, e))
    finally:
        if locked:
            gpulock.release(JOB)


if __name__ == "__main__":
    main()
