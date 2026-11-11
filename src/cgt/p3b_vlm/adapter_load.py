"""Fail-closed LoRA adapter loading.

`model.load_weights(..., strict=False)` silently ignores adapter tensors whose names do not exist in the model, so a key-name mismatch
would evaluate or train the base model. `require_adapter_keys_match` asserts that every adapter tensor key (count and names) matches a
model parameter of the same shape; `load_adapter_checked` applies it before loading."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


class AdapterKeyMismatch(RuntimeError):
    """The adapter file does not match the model's parameters (missing names, wrong shapes or no tensors)."""


def require_adapter_keys_match(adapter_shapes: Mapping[str, tuple], model_shapes: Mapping[str, tuple]) -> int:
    """Return the number of adapter tensors if every adapter key is a model parameter with the same shape; raise otherwise."""
    if not adapter_shapes:
        raise AdapterKeyMismatch("the adapter file holds no tensors")
    missing = sorted(k for k in adapter_shapes if k not in model_shapes)
    if missing:
        raise AdapterKeyMismatch(f"{len(missing)} of {len(adapter_shapes)} adapter tensors match no model parameter, e.g. {missing[:3]}")
    bad = sorted(k for k in adapter_shapes if tuple(adapter_shapes[k]) != tuple(model_shapes[k]))
    if bad:
        raise AdapterKeyMismatch(f"{len(bad)} adapter tensors have a shape different from the model, e.g. "
                                 f"{[(k, tuple(adapter_shapes[k]), tuple(model_shapes[k])) for k in bad[:3]]}")
    return len(adapter_shapes)


def load_adapter_checked(model, path: str | Path) -> int:
    """Load a LoRA adapter file into an MLX model after `require_adapter_keys_match`; returns the number of tensors loaded."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    weights = mx.load(str(path))
    n = require_adapter_keys_match({k: tuple(v.shape) for k, v in weights.items()},
                                   {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())})
    model.load_weights(list(weights.items()), strict=False)
    return n
