"""Phase 3C model profiles (docs/plans/default_4b.md). The film-pipeline DEFAULT is the Step C 4B single pass (a recorded decision outcome,
pipeline version p3c-v2-4b); the frozen 2B stays an explicit opt-in (`--model 2b`) whose run key, manifest and outputs are exactly the
old default's. Light module: no mlx import (the 4B engine is imported only when a 4B engine is built)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS = ("2b", "4b")
DEFAULT_MODEL = "4b"
PIPELINE_VERSION = {"2b": "p3c-v1", "4b": "p3c-v2-4b"}
ADAPTER_2B_REL = "data/cache/p3c/ckpt_100.safetensors"
SCORING_FILES_2B = ["src/cgt/p3c_tokens/engine.py", "src/cgt/p3c_tokens/methods.py", "src/cgt/p3c_tokens/aggregate.py", "src/cgt/p3c_tokens/vocab.py"]
LOAD_FILES_4B = ["src/cgt/film4b/engine4b.py", "src/cgt/film4b/mode.py", "src/cgt/p3b_vlm/adapter_load.py", "src/cgt/p3m_matched/registry.py"]


def resolve(model: str | None) -> str:
    m = DEFAULT_MODEL if model is None else model
    if m not in MODELS:
        raise ValueError(f"unknown Phase 3C model {model!r}; known: {MODELS}")
    return m


def model_identity(model: str) -> dict | None:
    """None for the 2B (its run key and manifest stay exactly the old default's); the pinned 4B identity otherwise."""
    if resolve(model) == "2b":
        return None
    from cgt.p3m_matched import registry as R
    return {"pipeline_version": PIPELINE_VERSION["4b"], "model": "4b", "repo": R.STEPC_4B_REPO, "revision": R.STEPC_4B_REVISION,
            "adapter_sha256": R.STEPC_4B_ADAPTER_SHA256, "lora": dict(R.STEPC_4B_LORA), "mlx_env": dict(R.STEPC_MLX_ENV)}


def expected_sha(model: str, root: Path = ROOT) -> str:
    if resolve(model) == "2b":
        return json.loads((Path(root) / "results/phase3b/frozen_manifest.json").read_text())["adapter_sha256"]
    from cgt.p3m_matched import registry as R
    return R.STEPC_4B_ADAPTER_SHA256


def adapter_path(model: str, root: Path = ROOT) -> Path:
    if resolve(model) == "2b":
        return Path(root) / ADAPTER_2B_REL
    from cgt.p3m_matched import registry as R
    return Path(R.STEPC_4B_ADAPTER_PATH)


def scoring_files(model: str) -> list[str]:
    """Files whose change since a run's commit invalidates that run's cached scores."""
    return list(SCORING_FILES_2B) + ([] if resolve(model) == "2b" else list(LOAD_FILES_4B))


def stage_key(film: str, method: str, model: str) -> str:
    return f"p3c:{film}:{method}:new" if resolve(model) == "2b" else f"p3c:{film}:{method}:{PIPELINE_VERSION['4b']}:new"


def make_engine(model: str, adapter: Path):
    """Build the scorer. 4B: MLX buffers 10/10 are set BEFORE mlx is imported (refused if mlx is already loaded with other values), then
    the checked 4B load (adapter sha before and after, checked loader)."""
    if resolve(model) == "2b":
        from cgt.p3c_tokens.engine import Engine
        return Engine(adapter)
    from cgt.film4b import mode as M
    M.apply_mlx_env()
    from cgt.film4b.engine4b import Engine4B
    return Engine4B(adapter)
