"""Gold mini evaluation (docs/gold_mini_eval_plan.md): the 60-shot NOTLD selection, the label vocabularies and the label file.
Shared by the labeling page (app/label/server.py gold-mini) and the sealed scorer (scripts/gold_mini_score.py). Reads no prediction."""
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
FILM = "night_of_the_living_dead"
N_SHOTS = 60
MIN_LEN_S = 1.0
SEED = 0
TOOL_VERSION = "gold-mini/1"
LABELS_SCHEMA = "gold_mini_labels_v1"
LABELS_REL = "data/gold/labels_mini_v1.json"
NOT_PREDICTED = ("dolly-zoom",)  # schema move class without a training example; the Phase 4 geometry model never emits it
FIELDS = ("size", "angle", "movement")
EYE_LEVEL = "eye-level"  # Amendment 1: labeler-only angle value; never in schema.yaml, never predicted by the model

MOVE_HELP = {
    "static": "the camera does not move (alone), or holds still for a clear part of the shot before or after a move (combined)",
    "push-in": "the camera travels forward toward the subject (the space changes, not just the magnification)",
    "pull-out": "the camera travels backward away from the subject",
    "boom-up": "the camera body rises vertically (crane or jib up)",
    "boom-down": "the camera body lowers vertically",
    "pan-left": "the camera turns left on a fixed point (horizontal rotation)",
    "pan-right": "the camera turns right on a fixed point",
    "tilt-up": "the camera pivots upward on a fixed point",
    "tilt-down": "the camera pivots downward on a fixed point",
    "zoom-in": "the lens magnifies (image enlarges, no parallax; the camera stays put)",
    "zoom-out": "the lens widens (image shrinks, no parallax)",
    "move-left": "the camera body travels sideways to the left (truck or track left)",
    "move-right": "the camera body travels sideways to the right",
    "arc": "the camera travels around the subject along a curve",
    "roll": "the camera rotates around the lens axis (the horizon tilts while moving)",
}


def vocab(root: Path = ROOT) -> dict:
    """Label vocabularies, read from configs/schema.yaml: size exactly; angle = the schema values plus the labeler-only EYE_LEVEL (Amendment 1);
    movement = the model's atoms (schema move minus NOT_PREDICTED).
    "Can't tell" is not a vocabulary value: it is stored as null."""
    attrs = yaml.safe_load((Path(root) / "configs/schema.yaml").read_text())["attributes"]
    return {"size": list(attrs["size"]["values"]), "angle": list(attrs["angle"]["values"]) + [EYE_LEVEL],
            "movement": [m for m in attrs["move"]["values"] if m not in NOT_PREDICTED]}


def select_shots(gold: list[dict]) -> list[dict]:
    """The pre-registered rule: NOTLD gold entries sorted by shot_id, drop those shorter than MIN_LEN_S, default_rng(SEED).choice(n, 60, replace=False)."""
    kept = sorted((s for s in gold if s["film"] == FILM and s["end_s"] - s["start_s"] >= MIN_LEN_S), key=lambda s: s["shot_id"])
    if len(kept) < N_SHOTS:
        raise ValueError(f"only {len(kept)} eligible NOTLD gold shots (< {N_SHOTS})")
    idx = np.random.default_rng(SEED).choice(len(kept), size=N_SHOTS, replace=False)
    return [{"film": kept[i]["film"], "shot_id": int(kept[i]["shot_id"]), "start_s": float(kept[i]["start_s"]), "end_s": float(kept[i]["end_s"])} for i in idx]


def load_selection(root: Path = ROOT) -> list[dict]:
    return select_shots(json.loads((Path(root) / "data/gold/shots.json").read_text()))


def validate(rec: dict, voc: dict) -> dict:
    """A checked, normalized label record (movement in vocabulary order); raises ValueError on anything outside the vocabularies."""
    if rec.get("film") != FILM or not isinstance(rec.get("shot_id"), int):
        raise ValueError("film/shot_id")
    for f in ("size", "angle"):
        if rec.get(f) is not None and rec[f] not in voc[f]:
            raise ValueError(f"{f}={rec[f]!r} not in {voc[f]}")
    mv = rec.get("movement")
    if mv is not None:
        if not isinstance(mv, list) or not mv or len(set(mv)) != len(mv) or any(m not in voc["movement"] for m in mv):
            raise ValueError(f"movement={mv!r}: a non-empty list of distinct atoms of {voc['movement']}, or null")
        mv = [m for m in voc["movement"] if m in mv]
    return {"film": FILM, "shot_id": rec["shot_id"], "size": rec.get("size"), "angle": rec.get("angle"), "movement": mv,
            "labeled_at": rec.get("labeled_at"), "tool_version": rec.get("tool_version")}


def new_record(shot_id: int, size, angle, movement, voc: dict) -> dict:
    rec = validate({"film": FILM, "shot_id": shot_id, "size": size, "angle": angle, "movement": movement}, voc)
    rec["labeled_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    rec["tool_version"] = TOOL_VERSION
    return rec


def read_labels(path: Path) -> dict[int, dict]:
    """shot_id -> record (empty when the file does not exist)."""
    path = Path(path)
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    if d.get("schema") != LABELS_SCHEMA:
        raise ValueError(f"{path}: not a {LABELS_SCHEMA} file")
    ids = [int(r["shot_id"]) for r in d["records"]]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:  # Amendment 2 (A4): never collapse duplicates silently
        raise ValueError(f"{path}: duplicate shot_id records {dup}")
    return {int(r["shot_id"]): r for r in d["records"]}


def write_labels(path: Path, labels: dict[int, dict], order: list[int]) -> None:
    """Atomic write; records in presentation order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = [labels[s] for s in order if s in labels]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"schema": LABELS_SCHEMA, "records": recs}, indent=1))
    os.replace(tmp, path)
