"""Views = one forward pass per question with a changed input (plan section 4): option order (cyclic shift), image transform, resolution.
Pure helpers here (prompt rebuilding, shift mapping, image preparation, which attributes a view may touch); the GPU loop is
scripts/p3s_stepa_views.py. A view's output is the 4 letter logits in the letter order OF THAT VIEW; `to_original` maps them back."""
import math
from pathlib import Path

import numpy as np
from PIL import Image

from cgt.p3b_vlm.images import BUDGET_PX, MULTIPLE, budget_size

LETTERS = ("A", "B", "C", "D")
ALL_ATTRS = ("size", "lens", "angle", "framing", "light_type", "light_cond", "composition")

# view -> (option shift, image transform, pixel budget, attributes the view may be applied to)
VIEWS: dict[str, dict] = {
    "base": {"shift": 0, "image": "plain", "budget": BUDGET_PX, "attrs": ALL_ATTRS},
    "shift1": {"shift": 1, "image": "plain", "budget": BUDGET_PX, "attrs": ALL_ATTRS},
    "shift2": {"shift": 2, "image": "plain", "budget": BUDGET_PX, "attrs": ALL_ATTRS},
    "shift3": {"shift": 3, "image": "plain", "budget": BUDGET_PX, "attrs": ALL_ATTRS},
    # a flip swaps left-heavy / right-heavy: never for composition
    "hflip": {"shift": 0, "image": "hflip", "budget": BUDGET_PX, "attrs": tuple(a for a in ALL_ATTRS if a != "composition")},
    # a crop changes size / lens impression / framing / angle context / composition: lighting only
    "crop80": {"shift": 0, "image": "crop80", "budget": BUDGET_PX, "attrs": ("light_type", "light_cond")},
    "res262k": {"shift": 0, "image": "plain", "budget": 262_144, "attrs": ALL_ATTRS},
    "res590k": {"shift": 0, "image": "plain", "budget": 589_824, "attrs": ALL_ATTRS},
}
RUN_ORDER = ["base", "shift1", "shift2", "shift3", "hflip", "crop80", "res262k", "res590k"]
CROP_KEEP = 0.8


def applies(view: str, attribute: str) -> bool:
    return attribute in VIEWS[view]["attrs"]


def shifted_options(options: dict[str, str], s: int) -> dict[str, str]:
    """Letter j shows original option (j + s) mod 4."""
    orig = [options[x] for x in LETTERS]
    return {LETTERS[j]: orig[(j + s) % 4] for j in range(4)}


def to_original(view_scores, s: int) -> np.ndarray:
    """Scores in the shifted view's letter order -> scores in the ORIGINAL option order: original option i was shown at letter (i - s) mod 4."""
    v = np.asarray(view_scores, dtype=float)
    return np.array([v[(i - s) % 4] for i in range(4)])


def build_user_text(original_text: str, options: dict[str, str], s: int) -> str:
    """The training prompt with the options block rebuilt for shift s. Raises if rebuilding the unshifted block does not reproduce the
    original text exactly (template drift guard)."""
    def block(opts):
        return "Options:\n" + "".join(f"{x}. {opts[x]}\n" for x in LETTERS)
    orig_block = block(options)
    if orig_block not in original_text or original_text.count("Options:\n") != 1:
        raise ValueError("original prompt does not contain the expected options block")
    return original_text.replace(orig_block, block(shifted_options(options, s)))


def budget_size_no_upscale(w: int, h: int, budget_px: int, multiple: int = MULTIPLE) -> tuple[int, int]:
    """Like `budget_size` but an image smaller than the budget is not enlarged (its own area is the budget)."""
    return budget_size(w, h, min(budget_px, w * h), multiple)


def prepare_image(src: str, dst: Path, transform: str, budget_px: int) -> Path:
    """Transformed copy of the ORIGINAL image `src`, aspect-preserving shrink to `budget_px` (bicubic, RGB JPEG quality 92: the base recipe).
    plain at the base budget = the cached base image recipe (upscaling allowed there, as in Phase 3B); other budgets never upscale."""
    dst = Path(dst)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    if transform == "hflip":
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform == "crop80":
        w, h = im.size
        cw, ch = max(1, math.floor(w * CROP_KEEP)), max(1, math.floor(h * CROP_KEEP))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        im = im.crop((x0, y0, x0 + cw, y0 + ch))
    elif transform != "plain":
        raise ValueError(transform)
    size = budget_size(*im.size, budget_px, MULTIPLE) if budget_px == BUDGET_PX else budget_size_no_upscale(*im.size, budget_px)
    tmp = dst.with_suffix(".tmp.jpg")
    im.resize(size, Image.BICUBIC).save(tmp, quality=92)
    tmp.replace(dst)
    return dst
