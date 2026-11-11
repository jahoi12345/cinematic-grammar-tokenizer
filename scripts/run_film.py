"""One-command pipeline for a film: p1 (shots, colour) -> p2 (pretrained models) -> p3c (frozen VLM, GPU) -> tokens (v1 rows) -> move (GPU, when available).
Usage: uv run python scripts/run_film.py --film <id> [--stages p1,p2,p3c,tokens,move] [--dry-run] [--allow-owned] [--publish]
Every stage is idempotent (own cache key, own manifest in data/cache/pipeline/<film>/), resumable, and checks its prerequisites first. GPU stages (p3c, move) take
/tmp/cgt_gpu.lock with an owner nonce and refuse to start if it is held. --dry-run prints what would run and writes nothing.
Exit codes: 0 ok; 2 usage; 3 GPU lock held; 4 film not registered or not allowed; 5 stage unavailable / prerequisite missing; 6 stage failed; 7 adapter hash mismatch; 8 uncommitted code."""
import argparse
import sys

from cgt.pipeline import orchestrator
from cgt.pipeline.stages import registry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--film", required=True)
    ap.add_argument("--stages", default=None, help="comma-separated subset of p1,p2,p3c,tokens,move (default: all available, in order)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-owned", action="store_true", help="process a film that is not public domain (never a GPU stage, never publishable); only when explicitly asked")
    ap.add_argument("--no-mps", action="store_true", help="hide the Apple GPU from torch (Phase 2 depth/audio then run on the CPU); MLX stages (p3c) are unaffected")
    ap.add_argument("--provisional", default=None, metavar="REASON", help="mark every output of this run provisional (the reason is written to the stage manifests and the summary)")
    ap.add_argument("--publish", action="store_true", help="assert that outputs may be published: requires public_domain")
    ap.add_argument("--model", default=None, choices=["2b", "4b"], help="Phase 3C model (default 4b: Step C 4B single pass, p3c-v2-4b, the recorded rule); 2b = the old default")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return orchestrator.EXIT["usage"] if e.code else 0
    if a.no_mps:
        import functools

        import torch
        torch.backends.mps.is_available = functools.wraps(torch.backends.mps.is_available)(lambda: False)  # wraps: transformers inspects __wrapped__
    stages = [s.strip() for s in a.stages.split(",") if s.strip()] if a.stages else None
    return orchestrator.run(a.film, stages, registry(), a.dry_run, a.allow_owned, a.publish, provisional=a.provisional, p3c_model=a.model)


if __name__ == "__main__":
    sys.exit(main())
