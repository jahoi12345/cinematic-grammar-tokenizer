"""Step 3 fixed values (docs/plans/step3_gap.md, BINDING). Continued LoRA from the Step C 4B adapter (aba56883...) on a seeded
ShotQA-train replay subset; the candidates differ ONLY in their variant (augmentation / image budget). The Step C model, LoRA layout,
loss, accumulation, clipping and stream file are reused from cgt.stepc.config (not changed); the continued-LoRA optimiser values
(peak LR 2e-5, cosine to 2e-6, 3% warm-up, Step B stopping rule) are the Step B ones (docs/plans/stepb_cinescale2.md section 4)."""
from pathlib import Path

from cgt.stepc import config as SC
from cgt.worktrees import sibling

ROOT = Path(__file__).resolve().parents[3]
MAIN = SC.MAIN
REPO, REVISION = SC.REPO, SC.REVISION
TRAIN_JSONL, TRAIN_SHA, VAL_JSONL, VAL_SHA, N_TRAIN, N_VAL_V2 = (SC.TRAIN_JSONL, SC.TRAIN_SHA, SC.VAL_JSONL, SC.VAL_SHA, SC.N_TRAIN,
                                                               SC.N_VAL_V2)
SHOTQA_ROOT, IMG_CACHE = SC.SHOTQA_ROOT, SC.IMG_CACHE

# starting point: the Step C 4B adapter at 100% (read-only, in a sibling working copy; hash-checked, exit 7)
STEPC_DIR = sibling("stepc_4b")
INIT_ADAPTER = STEPC_DIR / "data/cache/stepc/runs/stepc_4b/ckpt_100.safetensors"
INIT_SHA = "aba568837c815b16e877bc0f36a4d7435358abe9e1d88c4b7c99eccf0e61ae39"
BASE_PRED = STEPC_DIR / "data/cache/stepc/predictions/val_ckpt100_aba568837c81.jsonl"     # its dedup_v2 predictions
BASE_PRED_SHA = "8b83965f2d68360edce67db4649c1c1276c32763606ec23ea3f7f1780a83859a"
# baseline = the plain 4B on dedup_v2 (results/stepc/selection.json, table pct 100)
BASELINE = {"primary": 82.49184734138178, "n": 5822,
            "per_attribute": {"size": 86.63646659116648, "lens": 67.5609756097561, "angle": 86.41686182669788,
                              "framing": 88.66587957497049, "light_type": 86.15635179153095, "light_cond": 76.72413793103449,
                              "composition": 85.28225806451613}}
TARGET_ATTRS = ("light_cond", "composition", "lens")

# recipe (continued LoRA): LoRA layout and loss from Step C; optimiser as Step B
RANK, ALPHA, DROPOUT, ACCUM, CLIP, MAX_SEQ = SC.RANK, SC.ALPHA, SC.DROPOUT, SC.ACCUM, SC.CLIP, SC.MAX_SEQ
PEAK_LR, END_LR, WARMUP_FRAC = 2e-5, 2e-6, 0.03
STOP_AFTER, STOP_WINDOW, STOP_LOSS = 100, 50, 1.0
MX_SEED, RESUME_SEED_BASE = 0, 10_000
N_UPDATES = 600
N_SAMPLES = N_UPDATES * ACCUM            # 9,600 replay samples, the SAME samples in the same order for every candidate
REPLAY_SEED = 3                          # np.random.default_rng(3).choice(51111, 9600, replace=False) over train.jsonl lines
AUG_SEED = 0
RESUME_EVERY, MEM_EVERY, MEM_FLOOR = 25, 50, 25
MLX_ENV = dict(SC.MLX_ENV)               # 10 ops / 10 MB per Metal command buffer (Step C amendment 3)

BUDGET_1X, BUDGET_2X, MULTIPLE = 147_456, 294_912, 32
CANDIDATES = {                            # run order G0, G1, G2 (fixed)
    "G0": {"augment": False, "budget_px": BUDGET_1X, "image_tokens": 144},
    "G1": {"augment": True, "budget_px": BUDGET_1X, "image_tokens": 144},
    "G2": {"augment": False, "budget_px": BUDGET_2X, "image_tokens": 288},
}
RUN_ORDER = ("G0", "G1", "G2")

# robustness proxy (secondary, reported only): 200 dedup_v2 questions per attribute (default_rng(0 + k) for the k-th attribute in ATTRS order) = 1,400 q,
# images re-sampled (side x0.6 down then back up, bicubic) and JPEG re-compressed at quality 50
PROXY_PER_ATTR, PROXY_SEED, PROXY_SCALE, PROXY_JPEG_Q = 200, 0, 0.6, 50

PLAN = "docs/plans/step3_gap.md"
CODE_PATHS = ["scripts", "src", PLAN, "pyproject.toml", "uv.lock"]
DATA = ROOT / "data/cache/step3"          # adapters, resume state, augmented/2x images, per-question predictions (gitignored)
RESULTS = ROOT / "results/step3"          # aggregates only (committed)

# budget (plan section 6)
# Amendment 2 (2026-10-20, user, after the 2026-10-19 12:14 stop): was 2026-10-19 22:00 / 16.0 h. The cap is set so it cannot bind
# before the hard stop: prior_gpu_hours() counts the killed G2 stage from 09:30:49 to the relaunch (idle time, an accounting
# artefact), so used_h <= 6.94 + (2026-10-20 10:00 - 2026-10-19 09:30:49) = 31.43 h at the hard stop < 34.0.
HARD_STOP, GPU_CAP_H = "2026-10-20T10:00:00+02:00", 34.0
SEC_PER_UPDATE = {"G0": 18.5, "G1": 18.5, "G2": 30.0}     # initial projections; G2 replaced by the pipeline-check measurement
VAL_PROJ_H = {"G0": 0.75, "G1": 0.75, "G2": 1.15}         # full dedup_v2 pass incl. model load
PROXY_PROJ_H = {"base": 0.25, "G0": 0.25, "G1": 0.25, "G2": 0.35}
MAX_TRAIN_RESUMES = 20
