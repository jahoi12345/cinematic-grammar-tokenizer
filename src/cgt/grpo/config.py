"""GRPO pilot fixed values. DESCRIPTIVE ONLY: nothing is
adopted or selected from the pilot. Model, LoRA layout, image budget, data files and the dedup_v2 validation are the Step C / Step 3
ones (imported, not changed); the start adapter is the Step C 4B ckpt_100 (sha256 aba56883..., read-only, hash-checked, exit 7)."""
from pathlib import Path

from cgt.step3 import config as S3
from cgt.stepc import config as SC

ROOT = Path(__file__).resolve().parents[3]
REPO, REVISION = SC.REPO, SC.REVISION
TRAIN_JSONL, TRAIN_SHA, N_TRAIN = SC.TRAIN_JSONL, SC.TRAIN_SHA, SC.N_TRAIN
VAL_SHA, N_VAL_V2 = SC.VAL_SHA, SC.N_VAL_V2
SHOTQA_ROOT = SC.SHOTQA_ROOT
INIT_ADAPTER, INIT_SHA = S3.INIT_ADAPTER, S3.INIT_SHA
BASELINE = S3.BASELINE                    # plain 4B on dedup_v2 (results/stepc/selection.json, pct 100): primary 82.49
RANK, ALPHA, DROPOUT, MAX_SEQ, IMAGE_TOKENS = SC.RANK, SC.ALPHA, SC.DROPOUT, SC.MAX_SEQ, SC.IMAGE_TOKENS
MLX_ENV = dict(SC.MLX_ENV)                # 10 ops / 10 MB per Metal command buffer

# ---- GRPO objective (plan section 2)
ANSWER_MODE = "letter"                    # one generated token (the option letter); no reasoning (plan section 2.2)
G = 8                                     # samples per question (group size)
TEMPERATURE = 1.0                         # sampling temperature = the policy itself (no tempering)
Q_PER_UPDATE = 16                         # questions per policy update (16 x 8 = 128 sampled answers per update)
BETA = 0.04                               # KL penalty to the frozen start adapter (k3 estimator at the sampled tokens)
CLIP_EPS = 0.2                            # PPO clip; mu = 1 optimisation step per sampled batch, so the ratio is exactly 1 (inactive)
ADV_EPS = 1e-4                            # advantage = (r - mean_group) / (std_group + ADV_EPS), population std
FORMAT_REWARD = 0.0                       # none: an answer that is not a parsable option letter gets reward 0
PEAK_LR, WARMUP_FRAC, GRAD_CLIP = 1e-5, 0.05, 1.0   # Adam, linear warm-up then constant; global-norm clip as Step C
MX_SEED = 0
SUBSET_SEED = 26                          # first N_UPDATES * Q_PER_UPDATE of default_rng(26).permutation(51111) over train.jsonl lines
SAMPLE_SEED = 26_000                      # sampling key for question k (stream position) = mx.random.key(SAMPLE_SEED + k)

# ---- pilot size (plan section 3): set from the pipeline check by the plan's formula, filled in before launch
N_UPDATES = 325                           # plan section 9: 25 x floor((2.2 h x 3600) / (1.05 x 22.82 s) / 25) = 325
N_QUESTIONS = N_UPDATES * Q_PER_UPDATE
RESUME_EVERY, MEM_EVERY, MEM_FLOOR = 25, 50, 25
MAX_TRAIN_RESUMES = 20

# ---- budget (plan section 5)
HARD_STOP, GPU_CAP_H = "2026-10-22T10:00:00+02:00", 3.5
SEC_PER_UPDATE = 22.82                    # pipeline check update 2 (results/grpo_pilot/pipecheck/summary.json); was 25.0 before it
VAL_PROJ_H = 0.75                         # full dedup_v2 pass incl. model load (Step C: 0.72 h)

PLAN = "docs/plans/grpo_pilot.md"
CODE_PATHS = ["scripts", "src", PLAN, "pyproject.toml", "uv.lock"]
DATA = ROOT / "data/cache/grpo_pilot"     # adapters, resume state, per-question predictions (gitignored)
RESULTS = ROOT / "results/grpo_pilot"     # aggregates only (committed)
TAG = "grpo_pilot"
