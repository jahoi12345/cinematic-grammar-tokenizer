"""Build the v1 token-row table (soft vote, length bins, global palette, no style) of a film. CPU only, idempotent, fails closed if Phase 3C is incomplete.
Usage: uv run python scripts/make_token_rows_v1.py --film night_of_the_living_dead"""
import argparse

from cgt.tokens import build, build_v1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", default="night_of_the_living_dead")
    a = ap.parse_args()
    try:
        out, txt, rebuilt = build_v1.write_tokens_v1(a.film)
    except build.IncompletePhase3C as e:
        raise SystemExit(f"FAIL CLOSED: {e}") from e
    print(f"{'wrote' if rebuilt else 'up to date (inputs unchanged)'}: {out}, {txt}")


if __name__ == "__main__":
    main()
