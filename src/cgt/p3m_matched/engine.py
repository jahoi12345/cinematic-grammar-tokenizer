"""The ONE generation path of the matched evaluator (docs/p3m_matched_protocol.md section 3): any vision-language model that mlx-vlm loads
(Qwen3-VL + our LoRA, Qwen2.5-VL for ShotVL-3B) is driven through `MlxEngine.answer`. Greedy, max 1024 new tokens, special tokens skipped."""
from pathlib import Path

from cgt.p3m_matched.protocol import MAX_NEW_TOKENS, build_messages


def render_prompt(processor, messages: list[dict]) -> str:
    """The model's own chat template, `tokenize=False, add_generation_prompt=True` (as the official script)."""
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class MlxEngine:
    def __init__(self, model, processor, max_new_tokens: int = MAX_NEW_TOKENS):
        self.model, self.processor, self.max_new_tokens = model, processor, max_new_tokens

    def answer(self, image_path: str, question: str, options: dict[str, str], variant: str) -> dict:
        """{text, prompt_tokens, generation_tokens}: the decoded NEW tokens only."""
        return self.generate_messages(image_path, build_messages(question, options, variant))

    def generate_messages(self, image_path: str, messages: list[dict]) -> dict:
        """Greedy generation for already-built messages (used by `answer`, and by the engine parity diagnostic)."""
        from mlx_vlm import generate
        prompt = render_prompt(self.processor, messages)
        r = generate(self.model, self.processor, prompt, image=[str(image_path)], max_tokens=self.max_new_tokens, temperature=0.0,
                     skip_special_tokens=True, verbose=False)
        return {"text": r.text, "prompt_tokens": int(getattr(r, "prompt_tokens", 0) or 0),
                "generation_tokens": int(getattr(r, "generation_tokens", 0) or 0)}


def load_mlx_model(entry: dict, verify_adapter_keys: bool = False):
    """(model, processor) for a registry entry: pinned base snapshot, optional LoRA adapter applied exactly as the Phase 3B evaluator
    (rank 16, alpha 32, all linear layers of the language model). The caller verifies the adapter hash BEFORE calling this.
    verify_adapter_keys=True (all tags except the consumed matched_v1) fails closed unless every adapter tensor matches the model."""
    import mlx.core as mx
    from mlx_vlm.utils import load

    from cgt.p3b_vlm.model_path import pinned_model_path
    model, processor = load(pinned_model_path(entry["repo"], entry["revision"]))
    if entry.get("adapter_file"):
        from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
        lora = entry["lora"]
        model = get_peft_model(model, find_all_linear_names(model.language_model), rank=lora["rank"], alpha=lora["alpha"],
                               dropout=lora["dropout"], verbose=False)
        if verify_adapter_keys:
            from cgt.p3b_vlm.adapter_load import load_adapter_checked
            load_adapter_checked(model, Path(entry["adapter_file"]))
        else:  # matched_v1 (consumed): unchanged
            model.load_weights(list(mx.load(str(Path(entry["adapter_file"]))).items()), strict=False)
    model.eval()
    return model, processor
