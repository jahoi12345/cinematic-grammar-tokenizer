"""Writes the author's go-ahead file results/phase3b/SHOTBENCH_GO, bound to the frozen manifest's evaluation tag and adapter SHA-256
(guards.go_text). The ShotBench evaluator refuses any go-ahead file with other content. RUN BY THE AUTHOR ONLY, after the freeze was independently checked.
Usage: uv run python scripts/p3b_write_go.py [--out PATH]"""
import argparse
import json
from pathlib import Path

from cgt.p3b_vlm import guards

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/phase3b/SHOTBENCH_GO"))
    ap.add_argument("--frozen", default=str(ROOT / "results/phase3b/frozen_manifest.json"))
    a = ap.parse_args()
    fm = json.loads(Path(a.frozen).read_text())
    Path(a.out).write_text(guards.go_text(fm["shotbench_ledger_tag_for_the_evaluation"], fm["adapter_sha256"]))
    print(f"wrote {a.out}: tag {fm['shotbench_ledger_tag_for_the_evaluation']}, adapter {fm['adapter_sha256'][:16]}")
