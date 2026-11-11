"""Phase 3B ShotBench evaluation of the FROZEN LoRA adapter: exactly one evaluation, under the frozen manifest's ledger tag.
The same image budget (<= 144 tokens), chat template, answer tokens and scorer (option-restricted accuracy from letter logits) as validation.
NEVER trains or tunes. Real mode order: (1) ledger_assert_unused(tag) FIRST (exit 14), (2) tag must equal the frozen manifest's evaluation tag (18),
(3) tag not CONSUMED, (4) an earlier crash must be acknowledged by the author (16), (5) results/phase3b/SHOTBENCH_GO must exist AND be bound to the tag and
the frozen adapter hash (13), (6) code committed and the original plan text unchanged (10), (7) the adapter's SHA-256 is recomputed and must equal
frozen_manifest.json, a score-free BEFORE hash manifest is written, (8) questions parsed, every image path contained in the ShotBench image root (5),
images shrunk into a verified cache (6), (9) inference only, results kept IN MEMORY, (10) the adapter hash is re-verified, (11) AFTER the complete
inference and hash verification the ledger entry is RECORDED FIRST (atomic), (12) then the CSV, predictions and AFTER hash manifest are written, (13) the
results are echoed. Raw stdout is saved with provenance. --dry-run only parses ShotBench questions and builds prompts (no model, no images, no ledger).
SEALED: until the full set is scored and the ledger entry recorded, only progress COUNTS are logged and no exception message is kept; a crash writes
<tag>.CRASH.json (exception class, generic reason code, frame locations, counts) and a retry is refused until the author creates <tag>.CRASH_ACK. A failure
AFTER the ledger commit leaves <tag>.CONSUMED.json (the tag is used, the results are still written/echoed, no new evaluation is allowed).
Exit codes: 5 path outside the ShotBench image root, 6 unreadable image, 7 adapter hash mismatch/missing, 10 uncommitted code or altered plan, 13 no/unbound go-ahead
file, 14 tag already evaluated or consumed, 15 crashed, 16 earlier crash unacknowledged, 17 failure after the ledger commit, 18 tag is not the frozen evaluation tag, 19 ledger commitment UNKNOWN (nothing written but UNKNOWN_COMMIT.json), 20 two distinct images share a stem.
Usage: uv run python scripts/p3b_eval_shotbench.py --dry-run
       uv run python scripts/p3b_eval_shotbench.py --tag qwen3vl2b_lora_filmsplit        (real; needs SHOTBENCH_GO, written by scripts/p3b_write_go.py)"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from cgt.eval.shotbench import LEDGER, SB
from cgt.p3b_vlm import guards
from cgt.p3b_vlm.images import BUDGET_PX, MULTIPLE, prepare_eval_cache
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p3b_vlm.runstatus import now_iso
from cgt.p3b_vlm.shotbench import build_user_text, load_shotbench_samples

ROOT = Path(__file__).resolve().parents[1]
REPO, REVISION = "Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"
RUN_TAG = "qwen3vl2b_lora_filmsplit_run1"
RUN_DIR = ROOT / "data/cache/p3b/runs" / RUN_TAG
FROZEN = ROOT / "results/phase3b/frozen_manifest.json"
GO_FILE = ROOT / "results/phase3b/SHOTBENCH_GO"
IMG_CACHE = ROOT / "data/cache/p3b_eval/shotbench_images_v2_147456_32"  # separate from the training cache; keys carry file name + content hash
RAW = ROOT / "results/phase3b/raw_eval_logs"
CODE_PATHS = ["scripts", "src", "docs/phase3b_plan.md", "pyproject.toml", "uv.lock"]
ATTRS = ["size", "framing", "angle", "lens", "light_type", "light_cond", "composition"]


class ListView:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def dry_run() -> None:
    """CPU only: parse the questions, build the prompts, check template parity with ShotQA training turns and token lengths."""
    from transformers import (
        AutoProcessor,  # tokenizer/processor files only: no model weights, no images
    )

    from cgt.p3_static_attrs.labels import parse_question
    from cgt.p3b_vlm.score import answer_token_ids
    samples, stats = load_shotbench_samples()
    processor = AutoProcessor.from_pretrained(pinned_model_path(REPO, REVISION))
    ids4 = answer_token_ids(processor)
    tl = guards.require_token_lengths(processor, samples, 1024, 144)
    n_stem_checked = guards.require_unique_stems([s["images"][0] for s in samples])  # names only: no image is opened
    # template parity: rebuild the first 300 ShotQA TRAIN prompts from their parsed parts and compare byte-for-byte
    train = [json.loads(x) for x in (ROOT / "data/cache/p3b/train.jsonl").read_text().splitlines()[:300]]
    mismatches = 0
    check_n = len(train)
    for t in train:
        text = t["messages"][0]["content"]
        question = text.split("\n", 1)[0].removeprefix("Question: ")
        _, opts = parse_question(text)
        mismatches += build_user_text(question, opts) != text
    out = {"mode": "DRY RUN: no model loaded, no image read, ledger not touched", "time": now_iso(), "stats": stats,
           "dropped_total": sum(v["dropped_unparsed"] + v["dropped_roundtrip"] for v in stats["per_attribute"].values()),
           "answer_token_ids": ids4, "token_lengths": tl, "distinct_image_files_with_distinct_stems": n_stem_checked, "template_check_n": check_n, "template_mismatches": mismatches,
           "image_budget": {"budget_px": BUDGET_PX, "multiple": MULTIPLE, "max_image_tokens": 144},
           "code_commit": guards.git_state(CODE_PATHS)["commit"]}
    (ROOT / "results/phase3b/shotbench_dryrun.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if out["dropped_total"] or mismatches:
        sys.exit(1)


def real_run(a, log, ev) -> None:
    from cgt.eval import shotbench as sbmod  # the 3A ledger functions and reference numbers
    from cgt.p3b_vlm import sealed
    tag = a.tag
    raw = Path(a.raw_dir)
    # (1) the ledger comes FIRST
    try:
        sbmod.ledger_assert_unused(tag, LEDGER)
    except RuntimeError as e:
        raise guards.LedgerUsed(str(e)) from e
    ledger_sha_before = hashlib.sha256(LEDGER.read_bytes()).hexdigest() if LEDGER.exists() else None
    # (2) the tag is the frozen evaluation tag, (3) not consumed, (4) an earlier crash was acknowledged, (5) a go-ahead bound to tag + adapter hash
    frozen = guards.require_evaluation_tag(tag, FROZEN)
    sealed.require_not_consumed(raw, tag)
    sealed.require_no_unacknowledged_crash(raw, tag)
    guards.require_go_file(Path(a.go_file), tag, frozen["adapter_sha256"])
    # (6) committed code and an unchanged original plan, (7) frozen adapter hash
    git = guards.require_clean_tree(CODE_PATHS)
    ev.code_commit = git["commit"]
    plan = guards.require_plan_unchanged()
    before = guards.require_frozen_adapter(FROZEN, RUN_DIR)
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {"tag": tag, "run_tag": RUN_TAG, "code_commit": git["commit"], "plan": plan, "ledger_sha256_before": ledger_sha_before,
                "adapter_before": {**before, "time": now_iso()}}
    (raw / f"{tag}.adapter_hash_manifest.json").write_text(json.dumps(manifest, indent=1))  # no scores in this file
    log(f"adapter sha256 BEFORE: {before['sha256']} ({before['checkpoint_percent']}% checkpoint); code {git['commit'][:12]}; plan text unchanged: {plan['unchanged']}")

    image_root = SB
    samples, stats = load_shotbench_samples(image_root=image_root)
    if sum(v["dropped_unparsed"] + v["dropped_roundtrip"] for v in stats["per_attribute"].values()):
        raise guards.FailClosed("ShotBench questions were dropped while parsing: refusing to evaluate a partial set")
    for s_ in samples:  # exit 5 BEFORE any image is opened: every path must be inside the ShotBench image root, no symlink
        guards.require_under_image_root(s_["images"][0], image_root)
    guards.require_unique_stems([s_["images"][0] for s_ in samples])  # exit 20 before any cache entry is created
    try:
        cached = prepare_eval_cache([s_["images"][0] for s_ in samples], IMG_CACHE)
    except (OSError, ValueError) as e:
        raise guards.UnreadableImage(f"{type(e).__name__} while preparing the evaluation image cache") from e
    for s_ in samples:
        s_["images"] = [cached[s_["images"][0]]]
    guards.require_images_readable([s["images"][0] for s in samples])

    # from here on a failure is a crash: reason + progress count only, ledger not consumed, retry needs the author (SEALED: no score may appear)
    ev.start(len(samples))
    import mlx.core as mx
    import pandas as pd
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
    from mlx_vlm.utils import load

    from cgt.p3b_vlm.score import answer_token_ids, evaluate_samples, summarize
    model, processor = load(pinned_model_path(REPO, REVISION))
    ids4 = answer_token_ids(processor)
    guards.require_token_lengths(processor, samples, 1024, 144)
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=16, alpha=32.0, dropout=0.05, verbose=False)
    model.load_weights(list(mx.load(before["adapter_path"]).items()), strict=False)
    model.eval()
    ds = VisionDataset(ListView([]), model.config.__dict__, processor, None, True)
    log(f"evaluating {len(samples)} ShotBench questions (inference only; letter-logit scorer; answer tokens {ids4}); progress counts only until sealed")
    t0 = time.perf_counter()
    res = ev.run(lambda cb: evaluate_samples(model, ds, samples, ids4, progress=cb))  # results stay in memory
    secs = time.perf_counter() - t0

    # everything below is computed IN MEMORY; nothing is printed or written until seal() has recorded the ledger entry
    summ = summarize(res)
    rows = []
    for at in ATTRS:
        v = summ["per_attribute"][at]
        lo, hi = wilson(v["correct"], v["n"])
        st = stats["per_attribute"][at]
        rows.append({"model": tag, "attribute": at, "n": v["n"], "option_restricted_acc_pct": round(100 * v["acc"], 1),
                     "full_vocab_hit_at_1_pct": "", "head": f"LoRA Qwen3-VL-2B, {before['checkpoint_percent']}% checkpoint", "seeds": 1,
                     "rows_in_category": st["rows_in_category"], "dropped_unparsed": st["dropped_unparsed"] + st["dropped_roundtrip"],
                     "dropped_no_image_feature": 0, "n_compound_answers": v["n_compound"],
                     "acc_single_answer_pct": "" if v["acc_single"] is None else round(100 * v["acc_single"], 1),
                     "acc_compound_answer_pct": "" if v["acc_compound"] is None else round(100 * v["acc_compound"], 1),
                     "correct": v["correct"], "ci95_low_pct": round(lo, 1), "ci95_high_pct": round(hi, 1),
                     "adapter_sha256": before["sha256"], "scorer": "option-restricted accuracy from letter logits (A-D)"})
    ours = pd.DataFrame(rows)
    ref = [{"model": m, "attribute": at, "n": None, "option_restricted_acc_pct": val, "head": "reference (paper)"}
           for m, d in sbmod.REFERENCE.items() for at, val in d.items()]
    csv_text = pd.concat([ours, pd.DataFrame(ref)]).to_csv(index=False)
    pred_text = "\n".join(json.dumps(r) for r in res) + "\n"
    after = guards.require_frozen_adapter(FROZEN, RUN_DIR)
    if after["sha256"] != before["sha256"]:
        raise guards.AdapterHashError("the adapter changed during the evaluation")
    csv = ROOT / f"results/phase3b/shotbench_{tag}.csv"
    manifest.update({"adapter_after": {**after, "time": now_iso()}, "adapter_unchanged": True, "evaluation_seconds": secs,
                     "predictions_sha256": hashlib.sha256(pred_text.encode()).hexdigest(), "csv": str(csv.relative_to(ROOT)),
                     "mean_acc_pct": round(100 * summ["mean_acc"], 2), "n_questions": summ["n"]})
    echo = (f"DONE: mean option-restricted accuracy {100 * summ['mean_acc']:.2f}% over {len(ATTRS)} attributes; csv {csv.name}; adapter unchanged; ledger recorded\n"
            + ours[["attribute", "n", "option_restricted_acc_pct", "ci95_low_pct", "ci95_high_pct"]].to_string(index=False))
    # SEAL: ledger_record FIRST, then the files, then the echo
    note = f"evaluated once; n={summ['n']}; adapter {before['sha256'][:12]}"
    ev.seal(lambda: sbmod.ledger_record(tag, note, LEDGER), lambda: tag in sbmod.ledger_read(LEDGER),
            {csv: csv_text, raw / f"{tag}.predictions.jsonl": pred_text,
             raw / f"{tag}.adapter_hash_manifest.json": json.dumps(manifest, indent=1)}, echo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen3vl2b_lora_filmsplit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--go-file", default=str(GO_FILE), help="go-ahead file (default results/phase3b/SHOTBENCH_GO)")
    ap.add_argument("--raw-dir", default=str(RAW), help="where the raw stdout and hash manifest are written (demos use a temp dir)")
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    if a.dry_run:
        dry_run()
        return
    raw.mkdir(parents=True, exist_ok=True)
    rawf = raw / f"{a.tag}.stdout.txt"
    if not rawf.exists():
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
        rawf.write_text(f"# raw stdout of scripts/p3b_eval_shotbench.py --tag {a.tag}\n# started {now_iso()}; code commit {sha}\n"
                        f"# argv: {sys.argv}\n")

    def log(msg: str) -> None:
        line = f"[{now_iso()}] {msg}"
        print(line, flush=True)
        with open(rawf, "a") as f:
            f.write(line + "\n")

    from cgt.p3b_vlm import sealed
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    ev = sealed.SealedEvaluation(a.tag, raw, log, code_commit=sha)
    sealed.install_signal_handlers()
    try:
        real_run(a, log, ev)
    except guards.FailClosed as e:  # only the exact known guard classes carry their documented exit code; nothing else is reflected
        if ev.started and not ev.finished:
            ev.record_crash(e)  # a guard failure after the evaluation started counts as a crash: class + reason code + counts only
        code = sealed.exit_code_for(ev, e)
        log(f"FAIL CLOSED ({sealed.safe_text(e)}, exit {code})")
        sys.exit(code)
    except BaseException as e:  # noqa: BLE001 - any failure at all must leave a crash file (class + code + count), never a score
        if ev.started and not ev.finished:
            ev.record_crash(e)
        log(f"CRASHED ({sealed.safe_text(e)}); no scores written")
        sys.exit(sealed.exit_code_for(ev, e))


if __name__ == "__main__":
    main()
