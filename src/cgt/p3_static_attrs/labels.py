"""ShotQA question/option parsing into schema vocabularies (docs/phase3_interface.md section 1)."""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]

QUESTION_ATTR = {
    "shot size": "size", "lens size": "lens", "shot framing": "framing", "camera angle": "angle",
    "lighting type": "light_type", "lighting": "light_cond", "composition": "composition",
}
NORMALIZE = {
    "size": {"Extreme Close Up": "ECU", "Close Up": "CU", "Medium Close Up": "MCU", "Medium": "MS",
             "Medium Wide": "MWS", "Wide": "WS", "Extreme Wide": "EWS"},
    "lens": {"Long Lens": "long", "Medium": "medium", "Wide": "wide", "Ultra Wide / Fisheye": "ultra-wide"},
    "angle": {"Low angle": "low", "High angle": "high", "Overhead": "overhead", "Aerial": "aerial",
              "Dutch angle": "dutch"},
    "framing": {"Single": "single", "Insert": "insert", "2 shot": "2-shot", "3 shot": "3-shot",
                "Group shot": "group", "Establishing shot": "establishing", "Over the shoulder": "ots"},
    "light_type": {"Daylight": "daylight", "Artificial light": "artificial", "Mixed light": "mixed",
                   "Firelight": "firelight", "Overcast": "overcast", "Practical light": "practical",
                   "Sunny": "sunny", "Moonlight": "moonlight", "Fluorescent": "fluorescent", "HMI": "hmi",
                   "Tungsten": "tungsten", "LED": "led"},
    "light_cond": {"Side light": "side", "Backlight": "back", "High contrast": "high-contrast",
                   "Silhouette": "silhouette", "Edge light": "edge", "Underlight": "under", "Top light": "top",
                   "Hard light": "hard", "Soft light": "soft", "Low contrast": "low-contrast"},
    "composition": {"Center": "center", "Balanced": "balanced", "Symmetrical": "symmetrical",
                    "Right heavy": "right-heavy", "Left heavy": "left-heavy", "Short side": "short-side"},
}
_Q = re.compile(r"Question: (.*?)\nOptions:\n(.*?)\nPlease", re.DOTALL)
_DIM = re.compile(r"What's the (.*?) of this shot\?")


class UnmappedQuestion(ValueError):
    pass


class UnmappedLabel(ValueError):
    pass


def vocab(attribute: str) -> list[str]:
    schema = yaml.safe_load((ROOT / "configs/schema.yaml").read_text())["attributes"]
    return list(schema[attribute]["values"])


def parse_question(text: str) -> tuple[str, dict[str, str]]:
    m = _Q.search(text)
    if not m:
        raise UnmappedQuestion(text[:80])
    d = _DIM.search(m.group(1))
    attr = QUESTION_ATTR.get(d.group(1)) if d else None
    if attr is None:
        raise UnmappedQuestion(m.group(1))
    opts = {}
    for line in m.group(2).split("\n"):
        if len(line) > 3 and line[1:3] == ". ":
            opts[line[0]] = line[3:].strip()
    return attr, opts


def parse_label_set(attribute: str, text: str) -> frozenset[str]:
    table = NORMALIZE[attribute]
    out = set()
    for piece in text.split(", "):
        if piece not in table:
            raise UnmappedLabel(f"{attribute}: {piece!r}")
        out.add(table[piece])
    return frozenset(out)


@dataclass(frozen=True)
class Example:
    img_id: str
    attribute: str
    options: dict
    answer_letter: str
    target: frozenset


def build_examples(sft_path: str | Path, strict: bool = False) -> tuple[list[Example], dict]:
    """All parseable (image, attribute) QA examples in file order plus drop counters."""
    data = json.loads(Path(sft_path).read_text())
    stats = {"samples": len(data), "turns": 0, "unmapped_question": 0, "unmapped_label": 0}
    out = []
    for s in data:
        img_id = Path(s["images"][0]).name
        msgs = s["messages"]
        for i in range(0, len(msgs) - 1, 2):
            stats["turns"] += 1
            try:
                attr, raw = parse_question(msgs[i]["content"])
                options = {k: parse_label_set(attr, v) for k, v in raw.items()}
                letter = msgs[i + 1]["content"].strip()
                out.append(Example(img_id, attr, options, letter, options[letter]))
            except UnmappedQuestion:
                if strict:
                    raise
                stats["unmapped_question"] += 1
            except UnmappedLabel:
                if strict:
                    raise
                stats["unmapped_label"] += 1
    return out, stats


def check_vocab() -> None:
    """Every normalized value must be a member of configs/schema.yaml."""
    for attr, table in NORMALIZE.items():
        missing = set(table.values()) - set(vocab(attr))
        assert not missing, (attr, missing)
