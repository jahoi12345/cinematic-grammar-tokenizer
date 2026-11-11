"""Base-generation matched-budget experiment (docs/plans/base_gen_matched_budget.md, BINDING): the Phase 3B / Step C LoRA recipe
(cgt.stepc.config values, imported, not copied) on Qwen2.5-VL-3B-Instruct (ShotVL-3B's base), compared on dedup_v2 ShotQA val
with the Phase 3B Qwen3-VL-2B run (qwen3vl2b_lora_filmsplit_run1) at the SAME fraction of the epoch.

Only the base model and its image budget change: Qwen2.5-VL uses 28-pixel cells, so the Phase 3B benchmark's equal-token rule gives
112,896 px (336 x 336 area, multiples of 28) = at most 144 image tokens, as 147,456 px / 32 does for Qwen3-VL (results/phase3b/benchmark.md)."""
import os
from pathlib import Path

from cgt.stepc import config as SC
from cgt.worktrees import sibling

ROOT = Path(__file__).resolve().parents[3]

# 3B arm (trained here)
REPO_3B, REVISION_3B = "Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3"
BUDGET_PX_3B, MULTIPLE_3B = 112_896, 28
IMG_CACHE_3B = Path(os.environ.get("CGT_BASE_GEN_IMG", str(ROOT / "data/cache/base_gen/images_112896_28")))

# 2B arm (existing Phase 3B run, read only)
REPO_2B, REVISION_2B = "Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"
RUN_2B_DIR = SC.P3B_DIR / "runs/qwen3vl2b_lora_filmsplit_run1"
MANIFEST_2B = sibling("phase2") / "results/phase3b/runs/qwen3vl2b_lora_filmsplit_run1/manifest.json"
IMG_CACHE_2B = SC.IMG_CACHE                       # images_147456_32 (Phase 3B)

# schedule: the FULL epoch (3,195 updates), so update k has the same LR as update k of the 2B run; checkpoints as Phase 3B
TOTAL_UPDATES = 3_195                              # ceil(51,111 / 16)
MILESTONES = {799: 25, 1598: 50}                   # int(f * 3195 + 0.5), as scripts/p3b_train.py
TARGET_PCT_DEFAULT = 50

RANK, ALPHA, DROPOUT, PEAK_LR, END_LR, WARMUP_FRAC = SC.RANK, SC.ALPHA, SC.DROPOUT, SC.PEAK_LR, SC.END_LR, SC.WARMUP_FRAC
ACCUM, CLIP, MAX_SEQ, IMAGE_TOKENS = SC.ACCUM, SC.CLIP, SC.MAX_SEQ, SC.IMAGE_TOKENS
ORDER_SEED, MX_SEED, RESUME_SEED_BASE = SC.ORDER_SEED, SC.MX_SEED, SC.RESUME_SEED_BASE
STOP_AFTER, STOP_WINDOW, STOP_LOSS = SC.STOP_AFTER, SC.STOP_WINDOW, SC.STOP_LOSS
RESUME_EVERY, MEM_EVERY, MEM_FLOOR = SC.RESUME_EVERY, SC.MEM_EVERY, SC.MEM_FLOOR
MLX_ENV = SC.MLX_ENV
MAX_TRAIN_RESUMES = 20
TRAIN_SHA = SC.TRAIN_SHA                           # the stream itself is read by cgt.stepc.data.train_stream (sha-checked)

PLAN = "docs/plans/base_gen_matched_budget.md"
CODE_PATHS = ["scripts", "src", PLAN, "pyproject.toml", "uv.lock"]
DATA = ROOT / "data/cache/base_gen"               # adapters, resume state, per-question predictions (gitignored, never committed)
RESULTS = ROOT / "results/base_gen"               # aggregates and logs only (committed)

# budget (plan section 4): runner-enforced training deadline and GPU-hour cap
TRAIN_DEADLINE = "2026-10-28T09:30:00+01:00"
GPU_CAP_H = 6.0
