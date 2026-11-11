"""The Phase 3C prefix-KV scorer (cgt.p3c_tokens.engine.Engine, unchanged) with the Step C 4B model: only the load differs (pinned 4B
snapshot, Step C LoRA config, adapter SHA-256 checked before and after, fail-closed key/shape check). Imports mlx: call
cgt.film4b.mode.apply_mlx_env() first."""
from pathlib import Path

from cgt.film4b import mode as M
from cgt.p3b_vlm import guards
from cgt.p3c_tokens import engine as E


class Engine4B(E.Engine):
    def __init__(self, adapter: str | Path, expect_sha: str = M.ADAPTER_SHA):  # replaces Engine.__init__ (the 2B load)
        from mlx_vlm.trainer.datasets import VisionDataset
        from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
        from mlx_vlm.utils import load

        from cgt.p3b_vlm.adapter_load import load_adapter_checked
        from cgt.p3b_vlm.model_path import pinned_model_path
        adapter = Path(adapter)
        sha = guards.sha256_file(adapter)
        if sha != expect_sha:
            raise guards.AdapterHashError(f"adapter {adapter} sha256 {sha} != Step C 4B {expect_sha}")
        self.adapter_sha = sha
        model, self.processor = load(pinned_model_path(M.REPO, M.REVISION))
        self.cfg = model.config.__dict__
        model = get_peft_model(model, find_all_linear_names(model.language_model), rank=M.LORA["rank"], alpha=M.LORA["alpha"],
                               dropout=M.LORA["dropout"], verbose=False)
        self.n_adapter_tensors = load_adapter_checked(model, adapter)
        if guards.sha256_file(adapter) != sha:
            raise guards.AdapterHashError("adapter changed between the check and the load")
        model.eval()
        self.model, self.lm = model, model.language_model
        self.ds = VisionDataset(E._Empty(), self.cfg, self.processor, None, True)
        self.ids = E.letter_ids(self.processor)
        self.tok = self.processor.tokenizer
        self.end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.vision_end = getattr(self.model.config, "vision_end_token_id", 151653)
        self.image_token = self.model.config.image_token_index
        self.passes = 0
