"""CineScale2 file listing and label mapping to configs/schema.yaml (docs/data/cinescale2.md section 3).

Mapping:
- angle task: overhead -> overhead, high -> high, neutral -> eye-level (NEW in schema v2), low -> low, dutch -> dutch.
  Our schema's `aerial` angle has no CineScale2 angle counterpart, so CineScale2 angle questions never offer it.
- level task (camera height: aerial, eye, shoulder, hip, knee, ground): NOT mapped. Camera height is not an attribute of
  our schema, and the level-labelled frames carry no angle label, so they cannot supervise `angle` either.
- `gen_` files are the dataset's synthetic Dutch frames (random rotations of other raw frames, per the Data in Brief paper);
  they are flagged so Step B can drop them (their source frames are unknown and cannot be hash-checked)."""
from dataclasses import dataclass
from pathlib import Path

TASKS = ("angle", "level")
CLASSES = {
    "angle": ("overhead", "high", "neutral", "low", "dutch"),
    "level": ("aerial", "eye", "shoulder", "hip", "knee", "ground"),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# CineScale2 angle class -> schema v2 angle value
ANGLE_TO_SCHEMA = {"overhead": "overhead", "high": "high", "neutral": "eye-level", "low": "low", "dutch": "dutch"}
# option text shown in Step B questions, in ShotQA's wording (cgt.p3_static_attrs.labels.NORMALIZE['angle']) plus the new class
SCHEMA_ANGLE_TEXT = {"low": "Low angle", "high": "High angle", "overhead": "Overhead", "aerial": "Aerial",
                     "dutch": "Dutch angle", "eye-level": "Eye level"}


@dataclass(frozen=True)
class Item:
    relpath: str        # path relative to the unpacked root, e.g. 'cinescale2/angle/high/high_00648.jpg'
    task: str           # 'angle' or 'level'
    label: str          # CineScale2 class (folder name)
    synthetic: bool     # file name starts with 'gen_' (synthetic Dutch augmentation)


def parse_relpath(relpath: str) -> Item | None:
    """Item for an image path '<...>/<task>/<class>/<file>'; None for non-images (.DS_Store etc.). Raises ValueError on an
    image under an unknown task/class folder (fail closed: the vocabulary is fixed)."""
    p = Path(relpath)
    if p.suffix.lower() not in IMAGE_SUFFIXES or p.name.startswith("."):
        return None
    parts = p.parts
    if len(parts) < 3:
        raise ValueError(f"unexpected layout: {relpath}")
    task, label = parts[-3], parts[-2]
    if task not in CLASSES or label not in CLASSES[task]:
        raise ValueError(f"unknown task/class folder: {relpath}")
    return Item(relpath=str(p.as_posix()), task=task, label=label, synthetic=p.name.lower().startswith("gen_"))


def list_items(root: Path) -> list[Item]:
    """All CineScale2 images under `root` (the unpacked directory), sorted by relpath. Symlinks are refused."""
    root = Path(root)
    out = []
    for f in sorted(root.rglob("*")):
        if f.is_symlink():
            raise ValueError(f"symlink in the CineScale2 tree: {f}")
        if f.is_file():
            it = parse_relpath(f.relative_to(root).as_posix())
            if it is not None:
                out.append(it)
    return out


def to_schema(item: Item) -> tuple[str, str] | None:
    """(schema attribute, schema value) for an item, or None if CineScale2's label has no schema counterpart (level task)."""
    if item.task == "angle":
        return "angle", ANGLE_TO_SCHEMA[item.label]
    return None


def schema_angle_options() -> list[str]:
    """Schema v2 angle values CineScale2 distinguishes (questions on CineScale2 are asked only among these)."""
    return [ANGLE_TO_SCHEMA[c] for c in CLASSES["angle"]]
