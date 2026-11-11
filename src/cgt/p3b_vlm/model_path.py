"""Local path of a pinned Hugging Face revision (no network, no completeness check on files we deliberately did not download)."""
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

REPO = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def pinned_model_path(repo: str = REPO, revision: str = REVISION) -> str:
    """`<hf cache>/models--<org>--<name>/snapshots/<revision>`; raises FileNotFoundError unless config.json and safetensors exist there."""
    p = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if not (p / "config.json").exists() or not list(p.glob("*.safetensors")):
        raise FileNotFoundError(f"pinned snapshot missing or incomplete: {p}")
    return str(p)
