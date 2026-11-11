"""Align .srt cues to shot intervals."""
import re
from pathlib import Path

_T = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def _sec(h, m, s, frac) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + float("0." + frac)


def parse_srt(path: str) -> list[tuple[float, float, str]]:
    cues, text = [], Path(path).read_text(encoding="utf-8", errors="replace").replace("\r", "")
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.split("\n")
        for i, ln in enumerate(lines):
            m = _T.search(ln)
            if m:
                g = m.groups()
                cues.append((_sec(*g[:4]), _sec(*g[4:]), " ".join(x.strip() for x in lines[i + 1:] if x.strip())))
                break
    return cues


def text_for_shot(cues, start_s: float, end_s: float, min_overlap: float = 0.3) -> str:
    """Cues overlapping the shot by at least `min_overlap` of the cue length (or 0.5 s)."""
    out = []
    for cs, ce, t in cues:
        ov = min(ce, end_s) - max(cs, start_s)
        if ov > 0 and ov >= min(0.5, min_overlap * (ce - cs)):
            out.append(t)
    return " ".join(out)
