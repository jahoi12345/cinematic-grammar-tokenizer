"""Write results/phase4/split_manifest.json: source-grouped 85/15 split (seed 0) of the 1,058 ShotQA movement clips, from clip and source ids ONLY (no label is loaded or parsed).
Never reads ShotBench or CameraBench. Usage: uv run python scripts/p4_make_split.py"""
from cgt.p4_camera_move.labels import load_shotqa_movement_meta
from cgt.p4_camera_move.split import write_split_manifest


def main() -> None:
    m = write_split_manifest(load_shotqa_movement_meta())
    print({k: m[k] for k in ("seed", "val_frac", "n_sources", "n_val_sources", "n_clips", "n_train_clips", "n_val_clips", "sha256")})


if __name__ == "__main__":
    main()
