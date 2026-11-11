"""Download ShotBench (test-only, 3,572 QA pairs, Apache-2.0) to data/external/shotbench."""
from pathlib import Path

from huggingface_hub import snapshot_download

out = Path(__file__).resolve().parents[1] / "data/external/shotbench"
snapshot_download(repo_id="Vchitect/ShotBench", repo_type="dataset", local_dir=out)
print("saved to", out)
