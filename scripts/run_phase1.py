"""Usage: uv run python scripts/run_phase1.py --film night_of_the_living_dead [--threshold 12] [--force] [--dry-run]
Without --threshold the film's cut_threshold from configs/films.yaml is used (decisions/0005); an explicit flag wins.
--dry-run only prints the resolved threshold and whether the cut cache exists (no video is read)."""
import argparse

from cgt.p1_shots_color.pipeline import cuts_cache_path, resolve_threshold, run_film

ap = argparse.ArgumentParser()
ap.add_argument("--film", required=True)
ap.add_argument("--threshold", type=float, default=None)
ap.add_argument("--force", action="store_true")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
thr = resolve_threshold(a.film, a.threshold)
cache = cuts_cache_path(a.film, thr)
print(f"film={a.film} threshold={thr:g} ({'flag' if a.threshold is not None else 'configs/films.yaml'}) "
      f"cuts cache={cache.name} {'HIT' if cache.exists() and not a.force else 'MISS'}")
if a.dry_run:
    raise SystemExit(0)
df = run_film(a.film, thr, force=a.force)
print(df.drop(columns=["palette_lab", "palette_shares"]).describe().T)
print(df["key"].value_counts())
