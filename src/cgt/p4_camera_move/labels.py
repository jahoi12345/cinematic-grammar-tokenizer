"""Camera-movement label parser and ShotQA movement table (docs/phase4_interface.md section 1).
Pure CPU. Never reads ShotBench or CameraBench data. Atoms are the schema classes of `configs/schema.yaml` (`move`); `tracking` and `rack focus` are auxiliary
flags that are NOT scored in v1 (docs/phase4_plan.md Amendment 1)."""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
GRPO = ROOT / "data/external/shotqa/grpo.json"

# surface form (lower case, stripped) -> schema class
ATOM_ALIASES: dict[str, str] = {
    "static shot": "static", "static": "static", "push in": "push-in", "pull out": "pull-out", "boom up": "boom-up", "boom down": "boom-down",
    "pan left": "pan-left", "pan right": "pan-right", "tilt up": "tilt-up", "tilt down": "tilt-down", "zoom in": "zoom-in", "zoom out": "zoom-out",
    "move to the left": "move-left", "move to the right": "move-right", "trucking left": "move-left", "trucking right": "move-right",  # author: trucking = move
    "arc": "arc", "camera roll": "roll", "roll": "roll", "dolly zoom": "dolly-zoom",
}
FLAG_ALIASES: dict[str, str] = {"tracking": "track", "rack focus": "rack"}  # auxiliary flags, not scored in v1
_SPLIT = re.compile(r",|\band\b|\bthen\b|\bf(?:irst|isrt|rist)ly\b|\bfinally\b")
_SEQ = re.compile(r"\bf(?:irst|isrt|rist)ly\b|\bthen\b|\bfinally\b")


@dataclass(frozen=True)
class MoveLabel:
    kind: str                       # "single" | "simultaneous" | "sequential"
    atoms: tuple[str, ...]          # schema classes, unique, sorted (a SET; order is not predicted in v1)
    order: tuple[str, ...]          # schema atoms in text order (kept for later; empty atoms of other kinds are dropped)
    flags: tuple[str, ...]          # subset of ("rack", "track"), sorted
    unmapped: tuple[str, ...] = field(default=())  # surface forms that are neither atoms nor flags (should be empty on ShotQA)


def schema_moves() -> list[str]:
    """The 16 `move` classes of configs/schema.yaml, in file order (read only)."""
    def find(d):
        if isinstance(d, dict):
            if isinstance(d.get("move"), dict) and "values" in d["move"]:
                return d["move"]["values"]
            for v in d.values():
                r = find(v)
                if r:
                    return r
        return None
    return list(find(yaml.safe_load((ROOT / "configs/schema.yaml").read_text())))


def parse_move_label(text: str) -> MoveLabel:
    """'Push in and Tilt up' -> simultaneous {push-in, tilt-up}; 'Firstly Static shot, then Push in.' -> sequential {static, push-in} with order;
    'Tracking and Pull out' -> atoms {pull-out}, flags {track}; 'Trucking left' -> single {move-left}; a repeated atom collapses to one."""
    t = text.lower().strip()
    parts = [p.strip(" .") for p in _SPLIT.split(t) if p.strip(" .")]
    sequential = bool(_SEQ.search(t))
    order, flags, unmapped = [], [], []
    for p in parts:
        if p in ATOM_ALIASES:
            order.append(ATOM_ALIASES[p])
        elif p in FLAG_ALIASES:
            flags.append(FLAG_ALIASES[p])
        else:
            unmapped.append(p)
    atoms = tuple(sorted(set(order)))
    n_units = len(set(order)) + len(set(flags)) + len(unmapped)
    kind = "sequential" if sequential else ("simultaneous" if n_units > 1 else "single")
    return MoveLabel(kind=kind, atoms=atoms, order=tuple(dict.fromkeys(order)), flags=tuple(sorted(set(flags))), unmapped=tuple(unmapped))


def source_id(clip_name: str) -> str:
    """Source video id of a clip file name: the part before '.webm_' / '.mp4_' / '.mkv_' / '.mov_' ('X.webm_12.mp4' -> 'X')."""
    return re.split(r"\.(?:webm|mp4|mkv|mov)_", clip_name, maxsplit=1)[0]


def _clip_of(entry: dict) -> str:
    return entry["videos"][0].removeprefix("data/")


def _row_from_entry(x: dict) -> dict:
    """Parse the label of ONE grpo video entry (the only place a movement label text is parsed)."""
    text = x["messages"][0]["content"]
    opts = dict(re.findall(r"^([A-D])\. (.*)$", text, re.MULTILINE))
    m = re.search(r"<answer>\s*([A-D])\s*</answer>", x["solution"])
    clip = _clip_of(x)
    if not m or m.group(1) not in opts:
        raise ValueError(f"unparseable solution for {clip}")
    lab = parse_move_label(opts[m.group(1)])
    return {"clip": clip, "source": source_id(clip), "gold_text": opts[m.group(1)], "gold_letter": m.group(1), "options": opts, "kind": lab.kind,
            "atoms": lab.atoms, "order": lab.order, "flags": lab.flags, "unmapped": lab.unmapped,
            "option_atoms": {k: parse_move_label(v).atoms for k, v in opts.items()}}


def load_shotqa_movement_meta(grpo_path: Path = GRPO) -> list[dict]:
    """Metadata only, file order: `{clip, source}` for every video row. Never touches `solution` or the options (use for splitting and feature extraction)."""
    return [{"clip": _clip_of(x), "source": source_id(_clip_of(x))} for x in json.loads(Path(grpo_path).read_text()) if x.get("videos")]


def load_shotqa_movement(grpo_path: Path = GRPO) -> list[dict]:
    """FULL loader: one parsed row per movement clip (file order): {clip, source, gold_text, gold_letter, options{A..D: text}, kind, atoms, order, flags,
    unmapped, option_atoms{letter: atoms tuple}}. Parses the labels of ALL clips, validation included: never call it from selection code (use
    `MovementLabelStore`); raises ValueError for an unparseable solution (fail closed)."""
    return [_row_from_entry(x) for x in json.loads(Path(grpo_path).read_text()) if x.get("videos")]


class ValidationLabelsLocked(RuntimeError):
    """A validation label was requested before the explicit final evaluation (docs/phase4_interface.md section 1b)."""


class MovementLabelStore:
    """Label access with a lock on the validation clips. `rows(clips)` parses and returns the labels of the requested clips; a request that includes any validation clip raises
    `ValidationLabelsLocked` unless `final_evaluation=True`. `events` records ("parse", clip, "train" | "final") in order, so a test can prove when each label was read."""

    def __init__(self, validation_clips, grpo_path: Path = GRPO, events: list | None = None):
        self._raw = {_clip_of(x): x for x in json.loads(Path(grpo_path).read_text()) if x.get("videos")}
        self._val = frozenset(validation_clips)
        self.events: list = events if events is not None else []

    def rows(self, clips, *, final_evaluation: bool = False) -> list[dict]:
        clips = list(clips)
        if not final_evaluation and any(c in self._val for c in clips):
            raise ValidationLabelsLocked("validation labels are locked until the final evaluation (final_evaluation=True)")
        out = []
        for c in clips:
            self.events.append(("parse", c, "final" if c in self._val else "train"))
            out.append(_row_from_entry(self._raw[c]))
        return out

    def validation_rows(self, *, final_evaluation: bool = False) -> list[dict]:
        return self.rows(sorted(self._val), final_evaluation=final_evaluation)
