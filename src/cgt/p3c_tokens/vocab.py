"""Prompt construction for full-vocabulary scoring (docs/phase3c_plan.md section 1). Pure python: the exact ShotQA question template."""
from cgt.p3_static_attrs.labels import NORMALIZE, vocab

ATTRS = ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition"]
SINGLE = ["size", "lens", "angle", "framing"]
MULTI = ["light_type", "light_cond", "composition"]
DIM = {"size": "shot size", "lens": "lens size", "framing": "shot framing", "angle": "camera angle",
       "light_type": "lighting type", "light_cond": "lighting", "composition": "composition"}
LETTERS = "ABCDEFGHIJKL"
TAIL = "Please select the most likely answer from the options above. \n"
DISPLAY = {a: {v: k for k, v in NORMALIZE[a].items()} for a in ATTRS}  # label -> ShotQA display name


def labels(attr: str) -> list[str]:
    """Full vocabulary of `attr` in schema order (configs/schema.yaml)."""
    return vocab(attr)


def option_prompt(attr: str, option_labels: list[str]) -> str:
    """The ShotQA user text for `attr` with the given labels as options A, B, ... (identical to sft.json)."""
    if len(option_labels) > len(LETTERS):
        raise ValueError("too many options")
    opts = "\n".join(f"{LETTERS[i]}. {DISPLAY[attr][lab]}" for i, lab in enumerate(option_labels))
    return f"Question: What's the {DIM[attr]} of this shot?\nOptions:\n{opts}\n{TAIL}"


def free_prompt(attr: str) -> str:
    """Method C: the question without options (plan section 2)."""
    return f"Question: What's the {DIM[attr]} of this shot?\nPlease answer in a few words.\n"
