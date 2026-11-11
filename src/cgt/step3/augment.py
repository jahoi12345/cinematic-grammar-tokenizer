"""Step 3 G1: label-preserving image augmentation aimed at shortcut cues (docs/plans/step3_gap.md section 3).

Each training sample asks about ONE attribute, so the transform set is chosen per attribute: an operation is allowed only if it
cannot change that attribute's label. Operations (each applied with probability 0.5, fixed order crop -> hflip -> colour ->
brightness/contrast -> rescale -> jpeg):
  crop      random crop keeping 90-100% of each side (aspect may change slightly), random offset
  hflip     horizontal mirror
  colour    saturation x U(0.85, 1.15) (no hue / white-balance shift)
  bc        brightness x U(0.9, 1.1) and contrast x U(0.9, 1.1)
  rescale   resample down to side x U(0.6, 1.0) and back to the original size (bicubic): resolution / sharpness cue
  jpeg      re-encode as JPEG at quality U{50..95}: compression cue
Parameters are drawn from a per-sample RNG seeded by sha256(f"{seed}:{key}") and ALL parameters are always drawn (so the
draws do not depend on the attribute); disallowed operations are then switched off. Deterministic for a given (seed, key)."""
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

OPS = ("crop", "hflip", "colour", "bc", "rescale", "jpeg")
_PHOTOMETRIC = {"colour", "bc"}
_ALWAYS = {"rescale", "jpeg"}      # resolution and compression carry no label for any of the 7 attributes
ALLOWED = {
    # crop changes apparent shot size; flip keeps it; photometric keeps it
    "size": _ALWAYS | _PHOTOMETRIC | {"hflip"},
    # crop can cut a person / shoulder out of frame (single vs 2 shot vs OTS)
    "framing": _ALWAYS | _PHOTOMETRIC | {"hflip"},
    # high/low/overhead/aerial are mirror-invariant; a mirrored dutch tilt is still dutch; a 90% crop keeps the angle
    "angle": _ALWAYS | _PHOTOMETRIC | {"hflip", "crop"},
    # crop narrows the field of view (looks like a longer lens); no geometric warp at all
    "lens": _ALWAYS | _PHOTOMETRIC | {"hflip"},
    # colour temperature and brightness ARE light-type cues (tungsten, daylight, moonlight, fluorescent): no photometric ops
    "light_type": _ALWAYS | {"hflip", "crop"},
    # contrast/brightness define hard/soft, high/low-contrast, silhouette: no photometric ops; side light has no left/right label
    "light_cond": _ALWAYS | {"hflip", "crop"},
    # left/right-heavy and short-side flip under mirroring; crop shifts centre/balance: photometric only
    "composition": _ALWAYS | _PHOTOMETRIC,
}


def _rng(seed: int, key: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16))


def draw_params(seed: int, key: str, attribute: str) -> dict:
    """{op: params or None} for one sample; None = not applied. Raises KeyError for an unknown attribute."""
    allowed = ALLOWED[attribute]
    r = _rng(seed, key)
    raw = {
        "crop": (r.random() < 0.5, {"w": float(r.uniform(0.9, 1.0)), "h": float(r.uniform(0.9, 1.0)),
                                    "x": float(r.random()), "y": float(r.random())}),
        "hflip": (r.random() < 0.5, {}),
        "colour": (r.random() < 0.5, {"sat": float(r.uniform(0.85, 1.15))}),
        "bc": (r.random() < 0.5, {"bright": float(r.uniform(0.9, 1.1)), "contrast": float(r.uniform(0.9, 1.1))}),
        "rescale": (r.random() < 0.5, {"side": float(r.uniform(0.6, 1.0))}),
        "jpeg": (r.random() < 0.5, {"q": int(r.integers(50, 96))}),
    }
    return {op: (p if (on and op in allowed) else None) for op, (on, p) in raw.items()}


def rescale(im: Image.Image, side: float) -> Image.Image:
    w, h = im.size
    small = im.resize((max(1, round(w * side)), max(1, round(h * side))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def jpeg(im: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply(im: Image.Image, params: dict) -> Image.Image:
    im = im.convert("RGB")
    if (p := params.get("crop")) is not None:
        w, h = im.size
        cw, ch = max(1, round(w * p["w"])), max(1, round(h * p["h"]))
        x0, y0 = round((w - cw) * p["x"]), round((h - ch) * p["y"])
        im = im.crop((x0, y0, x0 + cw, y0 + ch))
    if params.get("hflip") is not None:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if (p := params.get("colour")) is not None:
        im = ImageEnhance.Color(im).enhance(p["sat"])
    if (p := params.get("bc")) is not None:
        im = ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(p["bright"])).enhance(p["contrast"])
    if (p := params.get("rescale")) is not None:
        im = rescale(im, p["side"])
    if (p := params.get("jpeg")) is not None:
        im = jpeg(im, p["q"])
    return im


def augment_file(src: str, dst: Path, seed: int, key: str, attribute: str) -> dict:
    """Write the augmented copy of `src` to `dst` (PNG, lossless; rewritten every call so it is always the deterministic output)
    and return the parameters used."""
    params = draw_params(seed, key, attribute)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        out = apply(im, params)
    tmp = dst.with_name(".tmp_" + dst.name)
    out.save(tmp, format="PNG")
    tmp.rename(dst)
    return params


def proxy_transform(im: Image.Image, scale: float, q: int) -> Image.Image:
    """Robustness proxy (fixed, label-preserving for all 7 attributes): resample side x scale and back, then JPEG at quality q."""
    return jpeg(rescale(im.convert("RGB"), scale), q)
