"""Film-title overlap between ShotQA (meta.jsonl `title`) and ShotBench (film list in the paper's Appendix C).
REPORT ONLY: nothing is dropped on it. ShotBench's test.tsv carries no titles, so the paper PDF is the only source.
Usage: uv run --with pypdf python scripts/p3_film_overlap.py --pdf /path/to/2506.21356v2.pdf"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--pdf", required=True)
a = ap.parse_args()

from pypdf import PdfReader

text = "\n".join(p.extract_text() for p in PdfReader(a.pdf).pages)
sb = {}
for m in re.finditer(r"^\s*(\d+)\s+(.+?)\s+(\d{4})\s+tt\d+\s*$", text, re.MULTILINE):
    sb[m.group(2).strip()] = int(m.group(3))


def norm(t: str) -> str:
    t = t.split("/season")[0].split("/episode")[0]
    return re.sub(r"[^a-z0-9]+", "", t.lower().replace("&", "and"))


qa: dict[str, set[str]] = {}
for line in (ROOT / "data/external/shotqa/meta.jsonl").read_text().splitlines():
    r = json.loads(line)
    qa.setdefault(norm(r["title"]), set()).add(str(r.get("time", "")))
sb_norm = {norm(t): (t, y) for t, y in sb.items()}
both = {k: v for k, v in sb_norm.items() if k in qa}
out = ROOT / "results/phase3/film_overlap.md"
out.write_text("\n".join([
    "# Film-title overlap: ShotQA vs ShotBench (report only, nothing dropped)",
    "",
    f"- ShotBench film titles parsed from the paper's Appendix C: {len(sb)} (the paper says 200+; the list in the PDF may be truncated or mis-parsed)",
    f"- ShotQA distinct titles in meta.jsonl (normalised, TV episodes folded into the series): {len(qa)}",
    f"- ShotBench titles that also appear in ShotQA (normalised-title match): {len(both)} of {len(sb_norm)}",
    "- Year check not possible: meta.jsonl `time` is 2021 for every row (scrape year), not the release year.",
    "",
    ("Interpretation: overlap here means the same FILM supplies images to both sets (different frames, since exact ids "
     "and near-duplicate hashes are checked separately). It is a domain-similarity caveat for ShotBench numbers, "
     "not image leakage."),
    "",
    "Matched titles (first 40): " + "; ".join(v[0] for v in list(both.values())[:40]),
]) + "\n")
print(out.read_text())
