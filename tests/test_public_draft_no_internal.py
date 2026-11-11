"""Guards for the public write-up files (docs/writeup/draft_public.md, paper_source_public.qmd, paper_public.qmd, paper_public.html).

Rule history:
- an earlier review (2026-10-27): the public draft carried no matched-protocol material at all.
- a recorded decision (2026-10-28, a decision record) lifted a recorded decision for publication: the official-protocol (matched)
  ShotBench results ARE allowed in public files now, and must come with their caveats (published numbers, not reruns; repeated looks at
  ShotBench; 202 of 242 films shared with ShotQA; no paired test; supervised fine-tuning only against ShotVL-3B's added GRPO; checkpoint
  selected on ShotQA validation). The wording rule is unchanged: never "beat", "outperform" or "significant(ly)" unless negated.
  Still banned in public files: owned-film and user-clip names, data/gold and data/dev paths, benchmark item text and answers (quoted gold
  keys), and the old "internal until released" marker.
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WRITEUP = ROOT / "docs/writeup"
PUBLIC = ["draft_public.md", "paper_source_public.qmd", "paper_public.qmd", "paper_public.html"]
BANNED = [r"internal until", r"user'?s? clip", r"clip supplied by the user", r"data/gold/", r"data/dev/",
          r"""["'](gold|gold_letter|gold_atoms|gold_answer|correct_answer|gt_answer)["']\s*:"""]
MATCHED = [r"74\.71", r"73\.26", r"73\.17, 76\.25", r"71\.73, 74\.81"]
CAVEATS = [r"not reruns?", r"202 of (the )?242", r"paired", r"GRPO", r"selected on (the )?ShotQA validation", r"look"]
WORDING = re.compile(r"\b(beat|beats|beating|outperform\w*|significant\w*)\b", re.IGNORECASE)
NEGATION = re.compile(r"\b(not|no|never|without|nor|cannot)\b|n't\b", re.IGNORECASE)  # within the 6 words before the term


def owned_terms() -> list[str]:
    films = (yaml.safe_load((ROOT / "configs/films.yaml").read_text()) or {}).get("films") or []
    terms = []
    for f in films:
        if f.get("public_domain") is not True:
            terms += [str(f["id"]).lower(), str(f.get("title", "")).lower()]
    return [t for t in terms if t]


def banned_hits(text: str) -> list[str]:
    low = text.lower()
    return [b for b in BANNED if re.search(b, text, flags=re.IGNORECASE)] + [t for t in owned_terms() if t in low]


def wording_hits(text: str) -> list[str]:
    """Every beat/outperform/significant(ly) not negated within the 6 preceding words (no phrase-level exemption)."""
    out = []
    for m in WORDING.finditer(text):
        if not NEGATION.search(" ".join(text[: m.start()].split()[-6:])):
            out.append(text[max(0, m.start() - 60): m.end() + 20].replace("\n", " "))
    return out


def _read(name: str) -> str:
    return (WRITEUP / name).read_text()


def test_public_files_have_no_banned_material():
    for name in PUBLIC:
        assert banned_hits(_read(name)) == [], name


def test_public_files_report_matched_results_with_caveats():
    """a recorded decision: the official-protocol results are public, never without their caveats."""
    for name in ("draft_public.md", "paper_public.qmd"):
        text = _read(name)
        assert all(re.search(m, text) for m in MATCHED), name
        missing = [c for c in CAVEATS if not re.search(c, text, flags=re.IGNORECASE)]
        assert missing == [], (name, missing)


def test_public_files_keep_the_wording_rule():
    for name in ("draft_public.md", "paper_public.qmd"):
        assert wording_hits(_read(name)) == [], name


def test_guards_catch_inserted_material():
    assert banned_hits('the user clip gave 99 shots; see data/gold/shots.json {"gold_letter": "A"}') != []
    assert wording_hits("Our 4B model outperforms ShotVL-3B.") != []
    assert wording_hits("It is significantly better.") != [] and wording_hits("a significant gain") != []
    assert wording_hits('the hypothesis "combined beats either alone" holds') != []  # no phrase exemption any more
    assert wording_hits("so I do not call the difference significant") == [] and wording_hits("it did not beat the geometry model") == []
