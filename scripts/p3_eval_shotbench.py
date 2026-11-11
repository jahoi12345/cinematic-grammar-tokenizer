"""ShotBench TEST evaluation, once per finished model tag.  --dry-run only checks that option texts parse (no model, no images).
Usage: uv run python scripts/p3_eval_shotbench.py --tag siglip_base [--seeds 0,1,2]"""
import argparse
import json
from pathlib import Path

from cgt.eval import shotbench

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="siglip_base")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print(shotbench.parse_coverage())
        raise SystemExit(0)
    chosen = json.loads((ROOT / f"results/phase3/runs/{a.tag}/chosen_heads_{a.tag}.json").read_text())
    res = shotbench.evaluate(a.tag, chosen, ROOT / "models/p3" / a.tag, [int(s) for s in a.seeds.split(",")],
                             ROOT / "data/cache/p3_eval/shotbench" / a.tag, ROOT / f"results/phase3/shotbench_{a.tag}.csv")
    print(res.to_string(index=False))
    print(f"mean option-restricted acc over the 7 static attributes: {res.option_restricted_acc_pct.mean():.1f}")


if __name__ == "__main__":
    main()
