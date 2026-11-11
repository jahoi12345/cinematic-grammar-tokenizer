"""Letter-logit scoring of video (K-frames-as-images) samples with a Qwen3-VL model loaded through mlx-vlm (shared by the validation step). Needs MLX; pure scoring, no labels
beyond the gold letter that locates the answer position (causal: the answer is never visible to the logits read before it)."""
import numpy as np

from cgt.p3b_vlm.score import letter_logits

LETTERS = ("A", "B", "C", "D")


class ListView:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def score_samples(model, ds, samples: list[dict], gold_letters: list[str], ids4: dict[str, int]) -> list[np.ndarray]:
    """Four letter logits per sample (model in eval mode by the caller). The tokenised gold answer is checked against the template's letter token."""
    out = []
    for s, g in zip(samples, gold_letters, strict=True):
        item = ds.process({"images": s["images"], "messages": s["messages"]})
        four, gold_tok = letter_logits(model, item, ids4)
        if gold_tok != ids4[g]:
            raise ValueError(f"{s['id']}: the tokenised gold answer is not the chat template's token for its letter")
        out.append(np.asarray(four, dtype=np.float64))
    return out
