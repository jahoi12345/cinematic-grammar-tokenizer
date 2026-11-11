"""Embed ShotQA training images with frozen SigLIP into data/cache/p3/<model>/ (resumable).
Run only after data/external/shotqa/data.tar.gz is fully downloaded AND extracted (it extracts flat into data/external/shotqa/).
Only images referenced by sft.json are embedded (GRPO-only images are deliberately not used in v1).
Usage: uv run python scripts/p3_extract_features.py [--model google/siglip-base-patch16-224] [--limit N]"""
import argparse
from pathlib import Path

from cgt.p3_static_attrs import features
from cgt.p3_static_attrs.data import (
    SHOTQA_ROOT,
    assert_shotqa_path,
    image_dir,
    image_path,
    load_leak_ids,
)
from cgt.p3_static_attrs.labels import build_examples

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=features.MODEL)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cache-dir", default=None)
    a = ap.parse_args()

    img_dir = image_dir()
    leak = load_leak_ids()
    examples, _ = build_examples(assert_shotqa_path(SHOTQA_ROOT / "sft.json"), strict=True)
    ids = sorted({e.img_id for e in examples} - leak)
    if a.limit:
        ids = ids[:a.limit]
    paths = [image_path(i) for i in ids]
    missing = [p for p in paths if not p.exists()]
    print(f"image dir: {img_dir}; images referenced: {len(paths)}, missing on disk: {len(missing)}, excluded as leakage: {len(leak)}")
    paths = [p for p in paths if p.exists()]
    cache = Path(a.cache_dir) if a.cache_dir else ROOT / "data/cache/p3" / features.slug(a.model)
    stats = features.embed_images(paths, cache, a.model, allowed_root=SHOTQA_ROOT)
    print(stats)
    if stats["n_new"] == 0:  # cache hit: keep the report of the run that did the work
        print("cache hit: nothing embedded, features_throughput.md left unchanged")
        return
    ips = round(stats["img_per_s"], 1)
    out = ROOT / "results/phase3/features_throughput.md"
    out.write_text(
        "# Feature extraction throughput\n"
        f"- model: {a.model} (frozen, pooled image embedding, float16 cache at data/cache/p3/{features.slug(a.model)}/)\n"
        f"- images requested: {stats['requested']}, already cached: {stats['cached']}, newly embedded: {stats['n_new']}, "
        f"unreadable: {stats['skipped_unreadable']}; missing on disk before embedding: {len(missing)}\n"
        f"- wall time: {stats['seconds']:.1f} s, throughput: {ips} img/s\n"
    )


if __name__ == "__main__":
    main()
