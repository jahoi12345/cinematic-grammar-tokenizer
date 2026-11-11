"""Usage: uv run python scripts/run_phase2.py --film night_of_the_living_dead [--stages blocking,depth,audio,subtitles] [--limit 40] [--max-seconds 300] [--force]"""
import argparse

from cgt.p2_pretrained.pipeline import STAGES, run_film

ap = argparse.ArgumentParser()
ap.add_argument("--film", required=True)
ap.add_argument("--stages", default=",".join(STAGES))
ap.add_argument("--limit", type=int)
ap.add_argument("--max-seconds", type=float, help="only shots ending before this time (e.g. 300 = first 5 min)")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
df = run_film(a.film, tuple(a.stages.split(",")), a.limit, a.max_seconds, a.force)
print(df.drop(columns=[c for c in ("subtitle_text",) if c in df]).describe(include="all").T.head(30))
