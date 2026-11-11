"""Step A-4B fixed values (docs/plans/stepa_4b.md, BINDING once committed). Everything not listed here is the Step A static code
(cgt.p3s_stepa.*, unchanged) or the Step C 4B load (cgt.stepc.config, unchanged). Inputs from other worktrees are read only and
sha256-checked; outputs go to this worktree (data/cache/stepa_4b gitignored, results/stepa_4b aggregates only)."""
import os
from pathlib import Path

from cgt.stepc import config as SC
from cgt.worktrees import sibling, worktrees_dir

ROOT = Path(__file__).resolve().parents[3]
WT = worktrees_dir()  # CGT_WORKTREES overrides

# model: the frozen Step C 4B adapter, base Qwen3-VL-4B @ ebb281ec, LoRA config of Step C
ADAPTER = Path(os.environ.get("CGT_STEPA4B_ADAPTER", str(sibling("stepc_4b") / "data/cache/stepc/runs/stepc_4b/ckpt_100.safetensors")))
ADAPTER_SHA = "aba568837c815b16e877bc0f36a4d7435358abe9e1d88c4b7c99eccf0e61ae39"
# Step C full-val per-question predictions of this adapter (C0 parity: base view == these logits)
STEPC_PRED = sibling("stepc_4b") / "data/cache/stepc/predictions/val_ckpt100_aba568837c81.jsonl"
STEPC_PRED_SHA = "8b83965f2d68360edce67db4649c1c1276c32763606ec23ea3f7f1780a83859a"
STEPC_PRIMARY = 82.49184734138178

# views: the Step A view set minus the resolution views (res262k / res590k), Step A run order
VIEWS = ["base", "shift1", "shift2", "shift3", "hflip", "crop80"]
DROPPED_VIEWS = ["res262k", "res590k"]
# the 10 of Step A's 13 configurations that need only VIEWS (cgt.p3s_stepa.combine.CONFIGS order)
RUNNABLE = ["C0", "C1", "C3", "C4a", "C4b", "C5", "C1+C3", "C1+C4*", "C1+C4*+C5", "C1+C3+C4*+C5"]
NOT_RUNNABLE = ["C2a", "C2b", "C1+C2*+C3+C4*+C5"]

# Step A inputs (read only): the prepared hflip / crop80 images of Step A (same recipe; prepared here if missing) and the
# Phase 3A SigLIP option scores (5,762 dedup_first rows + the 60 dedup_v2 added rows, both computed with the Step A head code)
STEPA_IMG = sibling("acc_static") / "data/cache/p3s_stepa/images"
SIGLIP = sibling("acc_static") / "data/cache/p3s_stepa/siglip3a.jsonl"
SIGLIP_SHA = "f16dc5c9f033fe55752d5509f5af0d6c522c685c34fb36a9939056875e9d357f"
SIGLIP_ADDED = sibling("dedup_fix") / "data/cache/val_dedup_v2/siglip3a_added.jsonl"
SIGLIP_ADDED_SHA = "f02e8475e7fcb3d7b4a650804ca562483d52891f45faa7de08e027e717337a62"
# Step A 2B cached views (reproduction check of results/p3s_stepa/static_val.json; dedup_first, 5,762)
STEPA_VIEWS_2B = sibling("acc_static") / "data/cache/p3s_stepa/views"

DATA = ROOT / "data/cache/stepa_4b"          # views/<view>.jsonl ({key, logits}), prepared images (gitignored)
RESULTS = ROOT / "results/stepa_4b"          # aggregates only (committed)
PLAN = "docs/plans/stepa_4b.md"

# budget (plan section 7)
SEC_PER_Q_PLAN = 0.46        # Step C pipecheck 4B s/question (stepc amendment 1); replaced by the measured pipeline-check rate
LOAD_H = 0.03                # model load per view child (~1-2 min)
GPU_CAP_H = 5.5
HARD_STOP = "2026-10-19T09:00:00+02:00"
MAX_CRASH_RESUMES = 20
MEM_FLOOR = 25
MLX_ENV = SC.MLX_ENV         # {"MLX_MAX_OPS_PER_BUFFER": "10", "MLX_MAX_MB_PER_BUFFER": "10"}
