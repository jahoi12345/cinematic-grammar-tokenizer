"""Frozen Phase 3B model as a scorer (inference only). Loads the pinned base model plus the frozen adapter (SHA-256 checked), and scores
prompts with the image's vision features and the KV cache of the shared prompt prefix (system + image + vision_end) computed once per image.
The scored quantity is identical to the stock full forward pass (verified in scripts/p3c_select_method.py, `equivalence` section)."""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.qwen3_vl.qwen3_vl import masked_scatter
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
from mlx_vlm.utils import load

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.model_path import REPO, REVISION, pinned_model_path
from cgt.p3c_tokens.vocab import LETTERS

ROOT = Path(__file__).resolve().parents[3]
FROZEN_MANIFEST = ROOT / "results/phase3b/frozen_manifest.json"
_SKIP = ("input_ids", "pixel_values", "attention_mask", "completion_mask")


class _Empty(list):
    pass


def frozen_adapter_sha() -> str:
    return json.loads(FROZEN_MANIFEST.read_text())["adapter_sha256"]


def letter_ids(processor, letters: str = LETTERS) -> dict[str, int]:
    """Letter -> the single token the chat template puts first in an assistant answer (as cgt.p3b_vlm.score.answer_token_ids, for A-L)."""
    tok = processor.tokenizer
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q?"}]}
    prompt = tok.encode(processor.apply_chat_template([user], add_generation_prompt=True), add_special_tokens=False)
    out = {}
    for letter in letters:
        full = processor.apply_chat_template([user, {"role": "assistant", "content": [{"type": "text", "text": letter}]}],
                                             add_generation_prompt=False)
        ids = tok.encode(full, add_special_tokens=False)
        rest = ids[len(prompt):]
        if ids[: len(prompt)] != prompt or len(rest) < 2 or tok.decode([rest[0]]).strip() != letter:
            raise ValueError(f"letter {letter} is not a single first token of the answer: {rest[:3]}")
        out[letter] = int(rest[0])
    if len(set(out.values())) != len(letters):
        raise ValueError("letter tokens not distinct")
    return out


class ImageState:
    """Per-image cache: vision features are consumed by the prefix pass; only the prefix KV and positions are kept."""

    def __init__(self, path: str, prefix_ids: np.ndarray, kv: list, next_pos: int, pos_dtype):
        self.path, self.prefix_ids, self.kv, self.next_pos, self.pos_dtype = path, prefix_ids, kv, next_pos, pos_dtype
        self.P = len(prefix_ids)


class Engine:
    def __init__(self, adapter: str | Path, expect_sha: str | None = None):
        sha = guards.sha256_file(Path(adapter))
        want = expect_sha or frozen_adapter_sha()
        if sha != want:
            raise guards.AdapterHashError(f"adapter {adapter} sha256 {sha} != frozen {want}")
        self.adapter_sha = sha
        model, self.processor = load(pinned_model_path(REPO, REVISION))
        self.cfg = model.config.__dict__
        model = get_peft_model(model, find_all_linear_names(model.language_model), rank=16, alpha=32.0, dropout=0.05, verbose=False)
        model.load_weights(list(mx.load(str(adapter)).items()), strict=False)
        model.eval()
        self.model, self.lm = model, model.language_model
        self.ds = VisionDataset(_Empty(), self.cfg, self.processor, None, True)
        self.ids = letter_ids(self.processor)
        self.tok = self.processor.tokenizer
        self.end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.vision_end = getattr(self.model.config, "vision_end_token_id", 151653)
        self.image_token = self.model.config.image_token_index
        self.passes = 0

    # --- tokenisation exactly as in training: the full chat sample through VisionDataset -----------------------------------------
    def _item(self, image_path: str, user: str, answer: str) -> dict:
        return self.ds.process({"images": [image_path], "messages": [{"role": "user", "content": user},
                                                                    {"role": "assistant", "content": answer}]})

    def image_state(self, image_path: str) -> ImageState:
        item = self._item(image_path, "Q?", "A")
        ids = np.array(item["input_ids"]).reshape(-1)
        P = int(np.flatnonzero(ids == self.vision_end)[-1]) + 1
        pix = mx.array(np.array(item["pixel_values"]))
        kw = {k: mx.array(np.array(v)) for k, v in item.items() if k not in _SKIP}
        grid = kw["image_grid_thw"]
        m = self.model
        dtype = m.vision_tower.patch_embed.proj.weight.dtype
        feats, deep = m.vision_tower(pix.astype(dtype), grid)
        pre = mx.array(ids[:P][None])
        emb = self.lm.model.embed_tokens(pre)
        vmask = pre == self.image_token
        emb = masked_scatter(emb, mx.broadcast_to(vmask[..., None], emb.shape), feats)
        deep_m = m._merge_deepstack_features(pre, deep, None)
        pos, _ = self.lm.get_rope_index(pre, grid, None, None)
        cache = [KVCache() for _ in self.lm.model.layers]
        h = self.lm.model(pre, inputs_embeds=emb, mask=None, cache=cache, position_ids=pos,
                          visual_pos_masks=vmask, deepstack_visual_embeds=deep_m)
        mx.eval(h, [c.state for c in cache])
        kv = [c.state for c in cache]
        return ImageState(image_path, ids[:P].copy(), kv, int(pos.max().item()) + 1, pos.dtype)

    def _hidden(self, st: ImageState, ids: np.ndarray) -> mx.array:
        if not np.array_equal(ids[: st.P], st.prefix_ids):
            raise ValueError("prompt does not share the cached image prefix")
        suf = mx.array(ids[st.P:][None])
        n = suf.shape[1]
        pos = mx.broadcast_to((mx.arange(n) + st.next_pos).astype(st.pos_dtype)[None, None, :], (3, 1, n))
        cache = []
        for kv in st.kv:
            c = KVCache()
            c.state = (kv[0], kv[1])
            cache.append(c)
        self.passes += 1
        return self.lm.model(suf, inputs_embeds=self.lm.model.embed_tokens(suf), mask=None, cache=cache, position_ids=pos)

    def _logits(self, h: mx.array) -> mx.array:
        return self.lm.model.embed_tokens.as_linear(h) if self.lm.args.tie_word_embeddings else self.lm.lm_head(h)

    # --- scoring primitives --------------------------------------------------------------------------------------------------------
    def letter_logits(self, st: ImageState, user_text: str, n_letters: int) -> np.ndarray:
        """Logits of the first `n_letters` letter tokens at the position preceding the answer (same read-out as cgt.p3b_vlm.score)."""
        item = self._item(st.path, user_text, "A")
        ids = np.array(item["input_ids"]).reshape(-1)
        p = int(np.flatnonzero(np.array(item["completion_mask"]).reshape(-1))[0])
        h = self._hidden(st, ids[:p])
        lg = self._logits(h[:, -1:, :])[0, 0].astype(mx.float32)
        sel = mx.array([self.ids[LETTERS[i]] for i in range(n_letters)])
        return np.array(lg[sel].astype(mx.float32))

    def completion_loglik(self, st: ImageState, user_text: str, answer: str) -> float:
        """Method C: mean log-probability of the answer tokens up to and including the first <|im_end|> (the trailing newline is excluded)."""
        item = self._item(st.path, user_text, answer)
        ids = np.array(item["input_ids"]).reshape(-1)
        cm = np.flatnonzero(np.array(item["completion_mask"]).reshape(-1))
        p = int(cm[0])
        stop = [int(i) for i in cm if ids[i] == self.end_id]
        q = (stop[0] + 1) if stop else int(cm[-1]) + 1
        h = self._hidden(st, ids[:q])
        lg = self._logits(h[:, p - 1 - st.P: q - 1 - st.P, :])[0].astype(mx.float32)  # logits predicting tokens p .. q-1
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        tgt = mx.array(ids[p:q])
        take = mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
        return float(mx.mean(take).item())

    def stock_letter_logits(self, image_path: str, user_text: str) -> np.ndarray:
        """Reference: the stock full forward pass (cgt.p3b_vlm.score.letter_logits), A-D only. Used for the equivalence check."""
        from cgt.p3b_vlm.score import letter_logits as stock
        item = self._item(image_path, user_text, "A")
        four, _ = stock(self.model, item, {k: self.ids[k] for k in "ABCD"})
        return four
