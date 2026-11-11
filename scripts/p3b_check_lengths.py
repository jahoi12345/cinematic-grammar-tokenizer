"""Token-length check of all training/validation chat samples (CPU, tokenizer only): prompt text + answer via the chat template, plus the
maximum image-token budget (144). Plan section 2: samples over 1,024 tokens would be skipped; expected 0."""
import json
from pathlib import Path

import numpy as np
from transformers import AutoProcessor

from cgt.p3b_vlm.model_path import pinned_model_path

ROOT = Path(__file__).resolve().parents[1]
REPO, REVISION = "Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"
proc = AutoProcessor.from_pretrained(pinned_model_path(REPO, REVISION))
tok = proc.tokenizer
out = {}
for split in ("train", "val"):
    rows = [json.loads(x) for x in (ROOT / f"data/cache/p3b/{split}.jsonl").read_text().splitlines()]
    lens = []
    for r in rows:
        user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["messages"][0]["content"]}]}
        full = proc.apply_chat_template([user, {"role": "assistant", "content": [{"type": "text", "text": r["answer"]}]}], add_generation_prompt=False)
        lens.append(len(tok.encode(full, add_special_tokens=False)) + 144)  # + at most 144 image tokens (placeholder counted once in the text)
    lens = np.array(lens)
    out[split] = {"n": len(rows), "max_tokens_incl_144_image_tokens": int(lens.max()), "mean": float(lens.mean()), "over_1024": int((lens > 1024).sum())}
print(json.dumps(out, indent=1))
Path(ROOT / "results/phase3b/runs/pipecheck_5k/length_check.json").write_text(json.dumps(out, indent=1))
