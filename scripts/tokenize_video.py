"""One command: a movie or clip -> one token row per shot (v1). Registers the input in the gitignored overlay configs/films.local.yaml, runs the film
pipeline (p1, p2, p3c, move, tokens) and prints the token table, the summary and the browse command.
Usage: uv run python scripts/tokenize_video.py <video> [--id ID] [--title T] [--cut-threshold N] [--public-domain] [--stages p1,p2,...] [--dry-run]
An ad-hoc input is local only: it runs with the owned-film permission for THIS film only, its outputs stay under the gitignored data/ tree, and it is never published.
Exit codes: as scripts/run_film.py, plus 10 bad input (missing, unreadable, not a video, unsafe --id) and 11 registry id clash. See docs/v1_entry_point.md."""
import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from cgt.pipeline import adhoc, orchestrator
from cgt.pipeline import registry as films_registry
from cgt.pipeline.orchestrator import EXIT, PipelineError


def outputs(root: Path, film_id: str, local: bool) -> dict[str, Path]:
    return {"token table": root / f"data/processed/{film_id}_tokens_v1.parquet", "token rows (text)": root / f"data/processed/{film_id}_tokens_v1.txt",
            "pipeline summary": root / (f"data/cache/pipeline/{film_id}/summary.md" if local else f"results/films/{film_id}.md")}


def main(argv: list[str] | None = None, root: Path = orchestrator.ROOT, stage_registry: dict | None = None,
         probes: dict[str, Callable[[str], str | None]] | None = None, log: Callable[[str], None] = print, lock_path: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", help="path to a movie or clip (stays where it is; it is not copied)")
    ap.add_argument("--id", default=None, help="film id (default: slug of the file name + 8 hex of its SHA-256)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--cut-threshold", type=float, default=None, help=f"shot-cut threshold (default {adhoc.DEFAULT_CUT_THRESHOLD:g}, UNCHECKED for this input)")
    ap.add_argument("--public-domain", action="store_true", help="assert that the input is public domain (it is still never published from the overlay)")
    ap.add_argument("--stages", default=None, help="comma-separated subset of p1,p2,p3c,move,tokens (default: all available, in order)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; write nothing (not even the overlay)")
    ap.add_argument("--model", default=None, choices=["2b", "4b"], help="Phase 3C model (default 4b: Step C 4B single pass, p3c-v2-4b); 2b = the old default")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return EXIT["usage"] if e.code else 0
    if stage_registry is None:
        from cgt.pipeline.stages import registry
        stage_registry = registry()
    stages = [s.strip() for s in a.stages.split(",") if s.strip()] if a.stages else None
    try:
        res = adhoc.resolve(Path(a.video), root, film_id=a.id, title=a.title, threshold=a.cut_threshold, public_domain=a.public_domain, dry_run=a.dry_run)
    except PipelineError as e:
        log(f"REFUSED [{e.code}]: {e}")
        return EXIT[e.code]
    fid, entry, v = res["film_id"], res["entry"], res["video"]
    log(f"input {a.video}: {v['codec']} {v['width']}x{v['height']}, {v['duration_s']:.1f} s -> film id '{fid}'")
    if res["action"] == "main":
        log(f"this file is the registered film '{fid}' of configs/films.yaml: it is used as registered (no overlay entry)")
    else:
        verb = {"new": "new overlay entry", "unchanged": "already registered in the overlay (unchanged)", "updated": "overlay entry UPDATED"}[res["action"]]
        log(f"{verb} ({films_registry.OVERLAY}, gitignored, local only; public_domain={entry['public_domain']}){' [dry run: not written]' if a.dry_run else ''}")
        for c in res["changes"]:
            log(f"  changed {c}")
        if entry["cut_threshold_source"] != "user":
            log(f"WARNING: unchecked cut threshold {entry['cut_threshold']:g} (library default; no labeled data exists for this input, nothing is tuned automatically). "
                f"Hand-check about 50 cuts after p1: uv run python scripts/spotcheck_cuts.py --film {fid} --threshold {entry['cut_threshold']:g} --n 50 "
                f"(contact sheets under data/interim/{fid}/), then re-run with --cut-threshold N if needed")
    pending = entry if (a.dry_run and res["action"] in ("new", "updated")) else None
    if a.dry_run:
        code = orchestrator.run(fid, stages, stage_registry, p3c_model=a.model, dry_run=True, allow_owned=True, root=root, log=log, pending=pending, lock_path=lock_path)
        if res["action"] == "new":
            log("WARNING: owner probe not run: the overlay entry is not written yet, so the owner modules (p1, p2, p3c, move) cannot see it; "
                "a real run registers it and then probes each stage for overlay support (docs/v1_entry_point.md)")
        elif res["action"] != "main":
            requested = stages if stages is not None else [n for n in orchestrator.ORDER if n in stage_registry]
            blockers = adhoc.owner_blockers(fid, requested, probes)
            if blockers:
                log("WARNING: a real run would STOP [unavailable_or_prerequisite]: these stages are owned by modules that cannot resolve an overlay film yet "
                    "(needs owner change, docs/v1_entry_point.md):")
                for s, why in blockers.items():
                    log(f"  {s}: {why}")
            else:
                log(f"owner probe: every probed stage resolves this overlay film ({', '.join(s for s in requested if s in adhoc.OWNER_STAGES) or 'none requested'})")
        return code
    if res["action"] in ("new", "updated"):
        log(f"registered: {adhoc.register(root, entry)}")
    film = orchestrator.film_entry(fid, root)
    if films_registry.is_overlay(film):
        requested = stages if stages is not None else [n for n in orchestrator.ORDER if n in stage_registry]
        blockers = adhoc.owner_blockers(fid, requested, probes)
        if blockers:
            orchestrator.run(fid, stages, stage_registry, p3c_model=a.model, dry_run=True, allow_owned=True, root=root, log=log, lock_path=lock_path)
            log("STOPPED [unavailable_or_prerequisite]: these stages are owned by modules that cannot resolve an overlay film yet (needs owner change, docs/v1_entry_point.md):")
            for s, why in blockers.items():
                log(f"  {s}: {why}")
            return EXIT["unavailable_or_prerequisite"]
    # owned / ad-hoc input: the owned-film permission applies to THIS film in THIS invocation only; nothing is stored
    code = orchestrator.run(fid, stages, stage_registry, p3c_model=a.model, dry_run=False, allow_owned=True, root=root, log=log, lock_path=lock_path)
    if code == EXIT["ok"]:
        local = films_registry.is_overlay(film) or not film.get("public_domain")
        for k, p in outputs(root, fid, local).items():
            log(f"{k}: {p}{'' if p.exists() else ' (not written by this run)'}")
        log("browse: uv run streamlit run app/browse/app.py  (lists every film with a token table; local only)")
    return code


if __name__ == "__main__":
    sys.exit(main())
