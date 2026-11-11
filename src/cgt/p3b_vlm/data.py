"""Phase 3B chat-format training data from ShotQA sft.json (docs/phase3b_interface.md section 1).
ShotQA only; the same film-held-out split and leakage exclusions as Phase 3A. Never reads ShotBench, gold, dev or our films."""
import json
from pathlib import Path

import numpy as np

from cgt.p3_static_attrs.data import (
    SHOTQA_ROOT,
    assert_shotqa_path,
    image_films,
    image_path,
    load_leak_ids,
    split_films,
)
from cgt.p3_static_attrs.labels import parse_label_set, parse_question

IMAGE_TAG = "<image>"


def film_split_ids(val_frac: float = 0.1, seed: int = 0) -> tuple[set[str], set[str], dict]:
    """(train_ids, val_ids, info): the Phase 3A film split over ALL sft images (leakage ids and untitled images dropped)."""
    sft = json.loads(assert_shotqa_path(SHOTQA_ROOT / "sft.json").read_text())
    ids = {Path(x["images"][0]).name for x in sft}
    img_film, _ = image_films(assert_shotqa_path(SHOTQA_ROOT / "meta.jsonl"), ids)
    train, val, info = split_films(img_film, val_frac, seed, exclude=load_leak_ids())
    return set(train), set(val), info


def load_chat_samples(split: str = "train") -> list[dict]:
    """One sample per question turn of every sft.json image in the film split (`train` or `val`).
    Sample: {id, img_id, attribute, images: [absolute guarded path], messages: [user, assistant], options, answer}.
    The user text is the original sft.json turn with the leading '<image>' tag removed; the assistant text is the answer letter."""
    train, val, _ = film_split_ids()
    keep = {"train": train, "val": val}[split]
    sft = json.loads(assert_shotqa_path(SHOTQA_ROOT / "sft.json").read_text())
    out = []
    for s in sft:
        img_id = Path(s["images"][0]).name
        if img_id not in keep:
            continue
        msgs = s["messages"]
        for t in range(0, len(msgs) - 1, 2):
            text = msgs[t]["content"].replace(IMAGE_TAG, "", 1)
            attr, options = parse_question(text)
            letter = msgs[t + 1]["content"].strip()
            parse_label_set(attr, options[letter])  # validates the answer maps into the schema vocabulary
            out.append({"id": f"{img_id}#{t // 2}", "img_id": img_id, "attribute": attr,
                        "images": [str(image_path(img_id))],
                        "messages": [{"role": "user", "content": text}, {"role": "assistant", "content": letter}],
                        "options": options, "answer": letter})
    return out


def write_jsonl(samples: list[dict], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(s) for s in samples) + "\n")


def subset(samples: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Seeded random subset of size n (sorted by original order), for benchmarks."""
    idx = np.random.default_rng(seed).permutation(len(samples))[:n]
    return [samples[i] for i in sorted(idx)]
