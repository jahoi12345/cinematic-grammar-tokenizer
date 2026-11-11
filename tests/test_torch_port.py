"""MLX LoRA -> peft converter on synthetic tensors (no model weights, no ShotQA)."""
import json

import numpy as np
import pytest

from cgt.p3b_vlm import torch_port as tp

R, SCALE = 4, 2.0


def _mlx_adapter(rng, layers=(0, 1)):
    shapes = {"self_attn.q_proj": (8, 8), "self_attn.k_proj": (8, 4), "mlp.down_proj": (12, 8)}
    sd = {}
    for L in layers:
        for mod, (din, dout) in shapes.items():
            sd[f"language_model.model.layers.{L}.{mod}.lora_a"] = rng.normal(size=(din, R)).astype(np.float32)
            sd[f"language_model.model.layers.{L}.{mod}.lora_b"] = rng.normal(size=(R, dout)).astype(np.float32)
    return sd


def test_key_map():
    assert tp.torch_module_name("language_model.model.layers.27.mlp.down_proj") == "model.language_model.layers.27.mlp.down_proj"
    with pytest.raises(ValueError):
        tp.torch_module_name("vision_tower.blocks.0.attn.qkv")


def test_converted_delta_w_equals_mlx_delta_w():
    rng = np.random.default_rng(0)
    mlx = _mlx_adapter(rng)
    sd, mods = tp.convert_tensors(mlx, R)
    cfg = tp.peft_config(mods, R, SCALE)
    assert len(mods) == 6 and len(sd) == 12 and cfg["lora_alpha"] / cfg["r"] == SCALE
    for k in [k for k in mlx if k.endswith("lora_a")]:
        mod = tp.torch_module_name(k[: -len(".lora_a")])
        a, b = mlx[k], mlx[k[:-1] + "b"]
        A, B = sd[f"{tp.PEFT_PREFIX}{mod}.lora_A.weight"], sd[f"{tp.PEFT_PREFIX}{mod}.lora_B.weight"]
        assert A.shape == (R, a.shape[0]) and B.shape == (b.shape[1], R)
        np.testing.assert_allclose(tp.delta_w_peft(A, B, cfg["lora_alpha"], cfg["r"]), tp.delta_w_mlx(a, b, SCALE), rtol=1e-12)
        x = rng.normal(size=(3, a.shape[0]))
        np.testing.assert_allclose(x @ tp.delta_w_mlx(a, b, SCALE).T, SCALE * (x @ a) @ b, rtol=1e-6, atol=1e-6)


def test_wrong_scale_or_transpose_is_detected():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(8, R)), rng.normal(size=(R, 6))
    ref = tp.delta_w_mlx(a, b, SCALE)
    assert not np.allclose(tp.delta_w_peft(a.T, b.T, 32.0, R), ref)  # alpha taken as the MLX alpha (scale 8 instead of 2)
    assert not np.allclose(tp.delta_w_peft(a.T, b.T, SCALE * R, R), tp.delta_w_mlx(a, b, 1.0))


def test_rejects_half_modules_bad_keys_and_rank():
    rng = np.random.default_rng(2)
    mlx = _mlx_adapter(rng, layers=(0,))
    with pytest.raises(ValueError):
        tp.convert_tensors({k: v for k, v in mlx.items() if not k.endswith("q_proj.lora_b")}, R)
    with pytest.raises(ValueError):
        tp.convert_tensors({**mlx, "language_model.lm_head.lora_a": np.zeros((2, R))}, R)
    with pytest.raises(ValueError):
        tp.convert_tensors(mlx, R + 1)


def test_write_adapter_roundtrip_and_config_check(tmp_path):
    from safetensors.numpy import load_file, save_file
    rng = np.random.default_rng(3)
    mlx = _mlx_adapter(rng)
    src = tmp_path / "ckpt.safetensors"
    save_file(mlx, str(src))
    keys = sorted({k.rsplit(".", 1)[0] for k in mlx})
    cfgp = tmp_path / "adapter_config.json"
    cfgp.write_text(json.dumps({"lora_parameters": {"rank": R, "scale": SCALE, "keys": keys}}))
    info = tp.write_peft_adapter(src, tmp_path / "peft", R, SCALE, cfgp)
    assert info == {"modules": 6, "tensors": 12}
    sd = load_file(str(tmp_path / "peft/adapter_model.safetensors"))
    k = "language_model.model.layers.1.mlp.down_proj"
    np.testing.assert_array_equal(sd[f"{tp.PEFT_PREFIX}model.language_model.layers.1.mlp.down_proj.lora_A.weight"], mlx[k + ".lora_a"].T)
    cfgp.write_text(json.dumps({"lora_parameters": {"rank": R, "scale": 8.0, "keys": keys}}))
    with pytest.raises(ValueError):
        tp.write_peft_adapter(src, tmp_path / "peft2", R, SCALE, cfgp)
    cfgp.write_text(json.dumps({"lora_parameters": {"rank": R, "scale": SCALE, "keys": keys[1:]}}))
    with pytest.raises(ValueError):
        tp.write_peft_adapter(src, tmp_path / "peft3", R, SCALE, cfgp)


def test_peft_forward_matches_mlx_formula(tmp_path):
    """A peft-wrapped tiny Linear loaded from the converted state dict reproduces y = W x + scale * (x a) b."""
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    rng = np.random.default_rng(4)
    din, dout = 8, 6
    a, b = rng.normal(size=(din, R)).astype(np.float32), rng.normal(size=(R, dout)).astype(np.float32)
    mlx = {"language_model.model.layers.0.self_attn.q_proj.lora_a": a, "language_model.model.layers.0.self_attn.q_proj.lora_b": b}
    sd, mods = tp.convert_tensors(mlx, R)

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(din, dout, bias=False)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()

    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LM()

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    torch.manual_seed(0)
    base = Outer()
    W = base.model.language_model.layers[0].self_attn.q_proj.weight.detach().numpy().copy()
    cfg = tp.peft_config(mods, R, SCALE)
    lc = peft.LoraConfig(r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0, target_modules=cfg["target_modules"])
    pm = peft.get_peft_model(base, lc)
    res = peft.set_peft_model_state_dict(pm, {k: torch.from_numpy(v) for k, v in sd.items()})
    assert not res.unexpected_keys
    pm.eval()
    x = rng.normal(size=(5, din)).astype(np.float32)
    y = pm.base_model.model.model.language_model.layers[0].self_attn.q_proj(torch.from_numpy(x)).detach().numpy()
    np.testing.assert_allclose(y, x @ W.T + SCALE * (x @ a) @ b, rtol=1e-5, atol=1e-5)
