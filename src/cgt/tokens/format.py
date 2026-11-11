"""Token row v0: the display format of one shot (docs/tokens_interface.md). Pure python; provisional until the Phase 5 schema decisions."""
import math

VERSION = "v0"
SLOTS = ["size", "lens", "angle", "framing", "light", "comp", "key", "DoF", "people", "sound", "len", "palette", "move"]
STATIC = ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition"]  # Phase 3C attributes that can carry the * marker
PEOPLE_MAX = 4  # configs/schema.yaml: people is a count with max 4; larger counts display as "4+"
MISSING = "?"
NOT_AVAILABLE = "n/a"


def _none(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _val(rec: dict, key: str, marker: bool = True) -> str:
    """Label of one attribute, `?` when missing, suffixed with `*` when the three keyframes disagree for it (consistent is False)."""
    v = rec.get(key)
    if _none(v):
        return MISSING
    flag = rec.get(f"{key}_consistent")
    return f"{v}*" if marker and flag is False else str(v)


def people_token(n) -> str:
    if _none(n):
        return MISSING
    n = int(n)
    return f"{PEOPLE_MAX}+" if n >= PEOPLE_MAX else str(n)


def palette_token(pid) -> str:
    return MISSING if _none(pid) else f"p{int(pid):02d}"


def length_token(sec) -> str:
    return MISSING if _none(sec) else f"{float(sec):.1f}s"


def slot_values(rec: dict) -> dict[str, str]:
    """Slot name -> display value, in SLOTS order. Movement is n/a until Phase 4; the style (RQ-VAE) slot does not exist yet."""
    return {
        "size": _val(rec, "size"), "lens": _val(rec, "lens"), "angle": _val(rec, "angle"), "framing": _val(rec, "framing"),
        "light": f"{_val(rec, 'light_type')}/{_val(rec, 'light_cond')}",
        "comp": _val(rec, "composition"),
        "key": MISSING if _none(rec.get("key")) else str(rec["key"]),
        "DoF": MISSING if _none(rec.get("dof")) else str(rec["dof"]),
        "people": people_token(rec.get("people")),
        "sound": MISSING if _none(rec.get("sound")) else str(rec["sound"]),
        "len": length_token(rec.get("len_s")),
        "palette": palette_token(rec.get("palette_id")),
        "move": NOT_AVAILABLE,
    }


def format_row(rec: dict) -> str:
    return "[" + " | ".join(f"{k}={v}" for k, v in slot_values(rec).items()) + "]"


def parse_row(text: str) -> dict[str, str]:
    """Inverse of format_row (slot -> display value); raises ValueError unless the slots appear exactly in SLOTS order."""
    t = text.strip()
    if not (t.startswith("[") and t.endswith("]")):
        raise ValueError("a token row is enclosed in [ ]")
    parts = [p.strip() for p in t[1:-1].split(" | ")]
    pairs = [p.split("=", 1) for p in parts]
    if any(len(x) != 2 for x in pairs):
        raise ValueError("every part must be slot=value")
    out = dict(pairs)
    if len(out) != len(pairs):
        raise ValueError("duplicate slot in the row")
    if list(out) != SLOTS:
        raise ValueError(f"slots {list(out)} are not {SLOTS}")
    return out
