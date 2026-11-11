"""Step C fixed values (docs/plans/stepc_qwen3vl4b.md section 2-6, BINDING). Recipe values are the Phase 3B ones
(scripts/p3b_train.py, docs/phase3b_plan.md section 4); only the base model changes.

Data are read (never written) from the Phase 3B cache, which lives in a sibling working copy worktree; ShotQA originals live in the main
checkout. Both are overridable by environment variables for another machine (CGT_P3B_DIR, CGT_SHOTQA_ROOT); the sha256 checks
below make a wrong file fail closed."""
import os
from pathlib import Path

from cgt.worktrees import main_checkout, sibling

ROOT = Path(__file__).resolve().parents[3]
MAIN = main_checkout()  # the main working copy (found through git); the repo itself in an exported copy
P3B_DIR = Path(os.environ.get("CGT_P3B_DIR", str(sibling("phase2") / "data/cache/p3b")))
SHOTQA_ROOT = Path(os.environ.get("CGT_SHOTQA_ROOT", str(MAIN / "data/external/shotqa")))
IMG_CACHE = P3B_DIR / "images_147456_32"          # budget-shrunk copies made in Phase 3B (147,456 px, multiple 32)
TRAIN_JSONL, VAL_JSONL = P3B_DIR / "train.jsonl", P3B_DIR / "val.jsonl"
TRAIN_SHA = "09ee851b3f6aa729d47ba4104b33b1275b3c1c26ee19749cffc6f6a8c7cba264"
VAL_SHA = "409e87d59faedc4753a27bf0e9151af1e7991cf53549434f53da4ec519c2637e"
N_TRAIN, N_VAL_V2 = 51_111, 5_822
FROZEN_PRED = P3B_DIR / "runs/qwen3vl2b_lora_filmsplit_run1/val_predictions/ckpt_100.predictions.jsonl"
FROZEN_PRED_SHA = "cebea813002eb24971d6ae652b38a90a0f7fc7105ceed92d9a0da3c64770ece6"
FROZEN_SHA = "4c0309679c9cdf6f60e5bde3cf12fd79c409b0fb49d0bbfd84c8b36499299d1f"

REPO, REVISION = "Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"
# Phase 3B recipe, unchanged (scripts/p3b_train.py line 52)
RANK, ALPHA, DROPOUT, PEAK_LR, END_LR, WARMUP_FRAC, ACCUM, CLIP, MAX_SEQ, IMAGE_TOKENS = 16, 32.0, 0.05, 1e-4, 1e-5, 0.03, 16, 1.0, 1024, 144
ORDER_SEED, MX_SEED, RESUME_SEED_BASE = 0, 0, 10_000
STOP_AFTER, STOP_WINDOW, STOP_LOSS = 200, 100, 1.5     # Phase 3B stopping rule (plan section 8)
MILESTONES = (0.5, 1.0)
RESUME_EVERY, MEM_EVERY, MEM_FLOOR = 25, 50, 25     # RESUME_EVERY 25 (was 100): amendment 3

PLAN = "docs/plans/stepc_qwen3vl4b.md"
CODE_PATHS = ["scripts", "src", PLAN, "pyproject.toml", "uv.lock"]
DATA = ROOT / "data/cache/stepc"            # adapters, resume state, per-question predictions (gitignored, never committed)
RESULTS = ROOT / "results/stepc"            # aggregates only (committed)

# budget (plan section 6; cap and validation projection set by Amendment 1, 2026-10-17)
# amendment 3: ONE pair, set by the user: the cap = hours from the first training start
# (2026-10-17 20:01:08) to the hard stop, rounded up, so the cap never binds before the hard stop. Was 18.5 h / 15:00.
RUN_START = "2026-10-17T20:01:08+02:00"
HARD_STOP, GPU_CAP_H = "2026-10-18T21:00:00+02:00", 25.0
TRAIN_PROJ_H = 14.9          # 16.8 s/update x 3,195 updates
VAL_PROJ_H = 0.75            # amendment 1: pipecheck 0.46 s/question x 5,822 = 0.74 h, + model load
MAX_TRAIN_RESUMES = 20       # amendment 3 (was 4, amendment 2): crash resumes in total, across runner launches
# amendment 3: shorter Metal command buffers against the macOS GPU watchdog ("Impacting Interactivity"). mlx 0.32.3 defaults on this
# M2 Max (applegpu_g14s): 50 ops and 50 MB per command buffer (read from the library: mlx::core::metal::Device::get_max_ops_mb_per_buffer).
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "10", "MLX_MAX_MB_PER_BUFFER": "10"}
