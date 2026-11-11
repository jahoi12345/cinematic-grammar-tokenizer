"""Leakage check: ShotQA (training source) vs ShotBench (test). Writes results/phase3/leakage.md + leak_ids.json.
1. exact image-id overlap (sft.json, grpo.json, meta.jsonl vs ShotBench test.tsv image ids)
2. near-duplicate overlap by 64-bit dHash, reported at hamming distances 0..8; exclusion rule is distance <= --max-hamming (4).
3. film-title overlap is reported by scripts/p3_film_overlap.py (report only, never used to drop).
Usage: uv run python scripts/p3_leakage.py [--max-hamming 4]"""
import argparse
import json
from pathlib import Path

from cgt.p3_static_attrs import leakage as L
from cgt.p3_static_attrs.data import assert_shotqa_path, image_path

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data/external/shotqa"
SB = ROOT / "data/external/shotbench"
OUT = ROOT / "results/phase3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hamming", type=int, default=4)
    a = ap.parse_args()
    sft = json.loads((assert_shotqa_path(QA / "sft.json")).read_text())
    grpo = json.loads((assert_shotqa_path(QA / "grpo.json")).read_text())
    meta_ids = {json.loads(line)["img_id"] for line in (assert_shotqa_path(QA / "meta.jsonl")).read_text().splitlines()}
    sft_ids = {Path(i).name for x in sft for i in x["images"]}
    grpo_ids = {Path(i).name for x in grpo for i in x.get("images", [])}
    sb_ids = L.shotbench_image_ids(SB / "test.tsv")
    exact = {"sft": L.exact_overlap(sft_ids, sb_ids), "grpo": L.exact_overlap(grpo_ids, sb_ids),
             "meta": L.exact_overlap(meta_ids, sb_ids)}
    lines = ["# Leakage check: ShotQA vs ShotBench", "",
             (f"ShotBench image ids: {len(sb_ids)}; ShotQA sft images: {len(sft_ids)}, "
              f"grpo images: {len(grpo_ids)}, meta ids: {len(meta_ids)}."), "", "## Exact image-id overlap",
             *[f"- {k}: {len(v)}" for k, v in exact.items()], ""]
    exact_all = set().union(*exact.values())
    near = {}
    used = sft_ids | grpo_ids
    qa_files = [p for p in (image_path(i) for i in sorted(used)) if p.exists()]
    if not qa_files:
        lines += ["## Near-duplicate overlap (dHash)",
                  "NOT RUN: no ShotQA images found on disk. Re-run after extraction."]
    else:
        sb_i, sb_h = L.hash_paths(sorted(p for p in (SB / "image").iterdir() if p.is_file()), workers=8)
        qa_i, qa_h = L.hash_paths(qa_files, workers=8)
        near = L.near_duplicates(qa_i, qa_h, sb_i, sb_h, max_distance=8)
        rows = L.distance_counts(near, 8)
        lines += ["## Near-duplicate overlap (64-bit dHash, 9x8 grey)",
                  (f"- ShotQA images hashed: {len(qa_i)} of {len(used)} referenced (sft + grpo images); "
                   f"ShotBench images hashed: {len(sb_i)}"),
                  "", "| hamming distance | ShotQA images whose nearest ShotBench image is exactly this far | cumulative (<= d) |",
                  "|---|---|---|", *[f"| {r['distance']} | {r['exactly']} | {r['cumulative']} |" for r in rows], "",
                  f"Exclusion rule: distance <= {a.max_hamming} (reported above up to 8, only <= {a.max_hamming} is dropped).",
                  "Excluded images are removed from training and validation (leak_ids.json)."]
        if near:
            lines += ["", f"examples (ShotQA id -> [(ShotBench id, hamming)]): {dict(list(near.items())[:5])}"]
    leak = L.exclusion_ids(exact_all, near, a.max_hamming)
    lines += ["", f"Total excluded ids: {len(leak)}", "",
              ("Caveat: ids/hashes catch identical or lightly re-encoded frames, not different frames of the same "
               "shot or film. Film-title overlap: see results/phase3/film_overlap.md (report only).")]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leakage.md").write_text("\n".join(lines) + "\n")
    (OUT / "leak_ids.json").write_text(json.dumps(sorted(leak)))
    (OUT / "near_pairs.json").write_text(json.dumps(near))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
