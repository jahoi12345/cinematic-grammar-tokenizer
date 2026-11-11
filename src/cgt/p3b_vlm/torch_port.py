"""Port of the MLX (mlx-vlm) Phase 3B LoRA adapter to a peft adapter for transformers' Qwen3-VL.

mlx-vlm LoRALinear: y = W x + scale * (x @ lora_a) @ lora_b, lora_a (in, r), lora_b (r, out), scale = alpha / rank (2.0 for the frozen run).
peft LoraLayer:     y = W x + (lora_alpha / r) * lora_B(lora_A(x)), lora_A.weight (r, in), lora_B.weight (out, r).
So lora_A.weight = lora_a.T, lora_B.weight = lora_b.T and lora_alpha = scale * r. delta W (out, in) = scale * (lora_a @ lora_b).T in both.
Key map: MLX `language_model.model.layers.L.<mod>.<proj>` -> transformers `model.language_model.layers.L.<mod>.<proj>`.
Pure numpy here (no torch import), so the CPU suite can test it; torch/peft are imported only by the scoring script."""
import json
import re
from pathlib import Path

import numpy as np

_MLX_KEY = re.compile(r"^language_model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.(lora_a|lora_b)$")
PEFT_PREFIX = "base_model.model."


def torch_module_name(mlx_module: str) -> str:
    """`language_model.model.layers.L.x.y` -> `model.language_model.layers.L.x.y` (raises on anything else)."""
    m = _MLX_KEY.match(mlx_module + ".lora_a")
    if not m:
        raise ValueError(f"not a Qwen3-VL language-model LoRA module: {mlx_module}")
    return f"model.language_model.layers.{m.group(1)}.{m.group(2)}.{m.group(3)}"


def convert_tensors(mlx: dict[str, np.ndarray], rank: int) -> tuple[dict[str, np.ndarray], list[str]]:
    """(peft state dict as saved by peft: `base_model.model.<module>.lora_A.weight` / `.lora_B.weight`, sorted list of torch module names).
    Raises if a key is unknown, a module lacks its a or b half, or a shape disagrees with `rank`."""
    out, mods = {}, {}
    for k, v in mlx.items():
        m = _MLX_KEY.match(k)
        if not m:
            raise ValueError(f"unexpected adapter key: {k}")
        mod = torch_module_name(k.rsplit(".", 1)[0])
        v = np.asarray(v)
        if v.ndim != 2:
            raise ValueError(f"{k}: expected 2-D, got {v.shape}")
        if m.group(4) == "lora_a":
            if v.shape[1] != rank:
                raise ValueError(f"{k}: lora_a shape {v.shape} does not end in rank {rank}")
            out[f"{PEFT_PREFIX}{mod}.lora_A.weight"] = np.ascontiguousarray(v.T)
        else:
            if v.shape[0] != rank:
                raise ValueError(f"{k}: lora_b shape {v.shape} does not start with rank {rank}")
            out[f"{PEFT_PREFIX}{mod}.lora_B.weight"] = np.ascontiguousarray(v.T)
        mods.setdefault(mod, set()).add(m.group(4))
    half = [m for m, s in mods.items() if s != {"lora_a", "lora_b"}]
    if half:
        raise ValueError(f"modules missing lora_a or lora_b: {half[:3]}")
    return out, sorted(mods)


def peft_config(modules: list[str], rank: int, scale: float) -> dict:
    """adapter_config.json for peft LoraConfig: exact module names, lora_alpha = scale * rank (so peft's alpha / r equals the MLX scale)."""
    return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": float(scale * rank), "lora_dropout": 0.0,
            "target_modules": list(modules), "bias": "none", "fan_in_fan_out": False, "use_rslora": False, "use_dora": False,
            "init_lora_weights": True, "inference_mode": True, "base_model_name_or_path": None, "modules_to_save": None}


def delta_w_mlx(lora_a: np.ndarray, lora_b: np.ndarray, scale: float) -> np.ndarray:
    """delta W (out, in) of an MLX LoRA layer: y - W x = scale * x @ a @ b = (scale * (a @ b)).T @ x."""
    return (scale * (lora_a.astype(np.float64) @ lora_b.astype(np.float64))).T


def delta_w_peft(lora_A: np.ndarray, lora_B: np.ndarray, lora_alpha: float, r: int) -> np.ndarray:
    """delta W (out, in) of a peft LoRA layer: (alpha / r) * B @ A."""
    return (lora_alpha / r) * (lora_B.astype(np.float64) @ lora_A.astype(np.float64))


def write_peft_adapter(mlx_adapter: Path, out_dir: Path, rank: int, scale: float, mlx_adapter_config: Path | None = None) -> dict:
    """Convert an MLX adapter .safetensors into a peft adapter directory (adapter_model.safetensors + adapter_config.json).
    If `mlx_adapter_config` (mlx-vlm adapter_config.json) is given, its rank/scale/keys must agree. Returns {modules, tensors}."""
    from safetensors.numpy import load_file, save_file

    mlx = load_file(str(mlx_adapter))
    sd, mods = convert_tensors(mlx, rank)
    if mlx_adapter_config is not None:
        lp = json.loads(Path(mlx_adapter_config).read_text())["lora_parameters"]
        if lp["rank"] != rank or float(lp["scale"]) != float(scale):
            raise ValueError(f"adapter_config rank/scale {lp['rank']}/{lp['scale']} != requested {rank}/{scale}")
        if sorted(torch_module_name(k) for k in lp["keys"]) != mods:
            raise ValueError("adapter_config keys do not match the tensors in the adapter file")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.astype(np.float32) for k, v in sd.items()}, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "adapter_config.json").write_text(json.dumps(peft_config(mods, rank, scale), indent=1))
    return {"modules": len(mods), "tensors": len(sd)}


def log_softmax4(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def compare(mlx_rows: list[dict], torch_rows: list[dict]) -> dict:
    """Aggregate parity numbers between two lists of per-question results (same order, same ids): argmax agreement,
    letter-logit and 4-option log-prob abs diffs. Aggregates only."""
    if [r["id"] for r in mlx_rows] != [r["id"] for r in torch_rows]:
        raise ValueError("rows are not aligned")
    a = np.array([r["logits"] for r in mlx_rows], dtype=np.float64)
    b = np.array([r["logits"] for r in torch_rows], dtype=np.float64)
    agree = np.array([x["pred"] == y["pred"] for x, y in zip(mlx_rows, torch_rows)])
    lp = np.abs(log_softmax4(a) - log_softmax4(b))
    lg = np.abs(a - b)
    srt = np.sort(a, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    return {"n": len(agree), "argmax_agree": int(agree.sum()), "argmax_agreement": float(agree.mean()),
            "logprob_absdiff_mean": float(lp.mean()), "logprob_absdiff_max": float(lp.max()),
            "logprob_absdiff_p99": float(np.quantile(lp, 0.99)),
            "logit_absdiff_mean": float(lg.mean()), "logit_absdiff_max": float(lg.max()),
            "mlx_top2_margin_of_disagreements": sorted(round(float(x), 4) for x in margin[~agree]),
            "mlx_exact_top_ties": int((margin == 0).sum())}
