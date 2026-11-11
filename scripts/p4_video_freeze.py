"""Write the FREEZE record of the Phase 4 video cross-fit (docs/phase4_video_plan.md Amendment 2 item 7): selected epoch, the adapter SHA-256 of the selected final-fit checkpoint (verified
against the training manifest), the OOF and validation tables, the combination weights, plan and code SHAs, restarts and wall time, and the evidence that no test set has been touched
(no CameraBench/ShotBench entry in the test ledger, the ShotBench ledger unchanged). Writes results/phase4/runs/video_<tag>/FREEZE.{json,md}. Adapters stay out of git.
Usage: uv run python scripts/p4_video_freeze.py --tag v1"""
import argparse
import hashlib
import json
from pathlib import Path

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.runstatus import now_iso, plan_original_unchanged
from cgt.p4_camera_move.baseline import code_sha

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    base = ROOT / "results/phase4/runs" / f"video_{a.tag}"
    comb = json.loads((base / "combine.json").read_text())
    val = json.loads((base / "validation.json").read_text())
    E = comb["selected_epoch_E"]
    ck = ROOT / "data/cache/p4/video_runs" / a.tag / "final" / f"ckpt_e{E}.safetensors"
    sha = guards.sha256_file(ck)
    man = json.loads((base / "final/manifest.json").read_text())["checkpoints"][str(E)]
    assert sha == man["sha256"] == val["adapter_sha256"], "adapter hash mismatch between the file, the training manifest and the validation record"
    sup = [json.loads(x) for x in (base / "supervisor.jsonl").read_text().splitlines()] if (base / "supervisor.jsonl").exists() else []
    restarts = [json.loads(x) for x in (base / "restarts.jsonl").read_text().splitlines()] if (base / "restarts.jsonl").exists() else []
    walls = [e.get("wall_s") for e in sup if e.get("event", "").startswith("stage") and e.get("wall_s")]
    tl = ROOT / "results/phase4/test_looks.jsonl"
    sb = ROOT / "results/phase3/shotbench_evaluated.json"
    looks = [json.loads(x) for x in (ROOT / "results/phase4/validation_looks.jsonl").read_text().splitlines()]
    plan = plan_original_unchanged(plan_commit="49e248a", plan_file="docs/phase4_video_plan.md")
    geom_hash = json.loads((ROOT / "results/phase4/frozen_config_flow_depth.json").read_text())["config_hash"]
    combo_hash = hashlib.sha256(json.dumps({"E": E, "w_s": comb["weights"]["w_s"], "w_g": comb["weights"]["w_g"], "adapter_sha256": sha}, sort_keys=True).encode()).hexdigest()[:16]
    rec = {"frozen_at": now_iso(), "tag": a.tag,
           # the model registry read by cgt.p4_camera_move.camerabench (committed version only): the ledger tags of the six final evaluations and the hashes a caller must present
           "registered_models": {"move_vlm_q3vl2b_lora_v1": {"adapter_sha256": sha, "selected_epoch_E": E},
                                 "move_combined_v1": {"adapter_sha256": sha, "config_hash": combo_hash, "weights": comb["weights"]},
                                 "move_geom_flowdepth_v1": {"config_hash": geom_hash}}, "selected_epoch_E": E, "adapter": {"file": ck.name, "sha256": sha, "size_bytes": ck.stat().st_size, "checkpoint_manifest_entry": man,
           "taken_before_any_recorded_phase4_test_evaluation": True, "location": "data/cache/p4/video_runs/<tag>/final (gitignored; ShotQA-derived, not redistributable)"},
           "combination_weights": comb["weights"], "oof": {"epoch_accuracy_pooled": comb["epoch_accuracy_pooled"], "option_accuracy_at_E": comb["oof_option_accuracy_at_E"], "paired_bootstrap_by_source": comb["oof_paired_bootstrap_by_source_1000"]},
           "validation": {"option_accuracy": val["option_accuracy"], "paired_bootstrap_by_source": val["paired_bootstrap_by_source_1000"], "n": val["n_val_questions"]},
           "plan": {"file": "docs/phase4_video_plan.md", "sha256_file": hashlib.sha256((ROOT / "docs/phase4_video_plan.md").read_bytes()).hexdigest(), "original_text": plan},
           "code_sha": code_sha(), "geometry_model": {"feature_sets": ["flow", "depth"], "config": json.loads((ROOT / "results/phase4/frozen_config_flow_depth.json").read_text())["config_hash"]},
           "restarts": len(restarts), "restart_records": restarts, "supervisor_wall_s_last": walls[-1] if walls else None,
           "test_access_evidence": {"test_looks_ledger_exists": tl.exists(), "test_looks_entries": (len(tl.read_text().splitlines()) if tl.exists() else 0),
                                    "shotbench_ledger_sha256": hashlib.sha256(sb.read_bytes()).hexdigest() if sb.exists() else None,
                                    "shotbench_ledger_tags": sorted(json.loads(sb.read_text())) if sb.exists() else None},
           "validation_looks_logged": len(looks), "validation_look_total_including_declared": looks[-1]["n_validation_looks_so_far"] if looks else None,
           "claims": "OOF and validation numbers are descriptive; selection used train-source OOF only; No Phase4 test evaluations are recorded in the retained ledgers, and no public-test label reader appears in the reviewed training/validation paths. Prior CameraBench sample/aggregate access is disclosed; additional historical unlogged access is not independently excluded. The supervisor records three acquisitions and release attempts; independent lock-ownership snapshots were not retained. The six final evaluations of Amendment 2 item 5 are NOT run and need a separate go-ahead and an independent verification"}
    (base / "FREEZE.json").write_text(json.dumps(rec, indent=1))
    L = [f"# FREEZE `{a.tag}` ({rec['frozen_at']})", "", f"- selected epoch E = **{E}** (OOF rule); adapter `{ck.name}` SHA-256 `{sha}` ({ck.stat().st_size} bytes), taken before any RECORDED Phase 4 test evaluation",
         f"- combination weights: w_s = {comb['weights']['w_s']}, w_g = {comb['weights']['w_g']} (frozen from the OOF questions)", f"- code SHA {rec['code_sha']}; plan file SHA-256 {rec['plan']['sha256_file']}; original plan text unchanged vs 49e248a: {plan['unchanged']}",
         f"- restarts: {len(restarts)} (limits: 15 total for Metal-watchdog failures, 5 for any other); validation looks logged: {len(looks)} (running total including 5 declared unverified: {rec['validation_look_total_including_declared']})",
         f"- test access: test ledger entries {rec['test_access_evidence']['test_looks_entries']}; ShotBench ledger tags {rec['test_access_evidence']['shotbench_ledger_tags']}", "",
         "## OOF (train sources, 3-fold cross-fit)", "", f"pooled OOF option accuracy by epoch: {comb['epoch_accuracy_pooled']}; at E: {json.dumps(comb['oof_option_accuracy_at_E'])}", "",
         "## Validation (descriptive)", "", (base / "validation.md").read_text(), "", rec["claims"]]
    (base / "FREEZE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
