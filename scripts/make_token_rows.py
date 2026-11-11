"""Build the token-row table and text file of a film from the Phase 1/2/3C tables (CPU only, idempotent, fail closed if Phase 3C is not complete).
Usage: uv run python scripts/make_token_rows.py --film night_of_the_living_dead"""
import argparse

from cgt.tokens import build


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", default="night_of_the_living_dead")
    a = ap.parse_args()
    try:
        out, txt, rebuilt = build.write_tokens(a.film)
    except build.IncompletePhase3C as e:
        raise SystemExit(f"FAIL CLOSED: {e}") from e
    print(f"{'wrote' if rebuilt else 'up to date (inputs unchanged)'}: {out}, {txt}")


if __name__ == "__main__":
    main()
