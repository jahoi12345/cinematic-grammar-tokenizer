"""Model tags of the matched evaluator (docs/p3m_matched_protocol.md section 3). One tag = one model identity = one sealed ShotBench slot.
An entry whose identity is not pinned (identity_sha256 None) is refused by every mode."""
import hashlib
import json
import os
from pathlib import Path

from cgt.p3b_vlm import guards

ROOT = Path(__file__).resolve().parents[3]
FROZEN_ADAPTER_SHA256 = "4c0309679c9cdf6f60e5bde3cf12fd79c409b0fb49d0bbfd84c8b36499299d1f"  # results/phase3b/frozen_manifest.json
WT = Path(os.environ.get("CGT_WORKTREES", "worktrees"))

# ---- Step C 4B ----
STEPC_4B_REPO, STEPC_4B_REVISION = "Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"  # = cgt.stepc.config
STEPC_4B_ADAPTER_SHA256 = "aba568837c815b16e877bc0f36a4d7435358abe9e1d88c4b7c99eccf0e61ae39"  # = cgt.stepa_4b.config.ADAPTER_SHA
# the adapter is gitignored and lives in the Step C worktree (same default and override as cgt.stepa_4b.config.ADAPTER); the sha pins it
STEPC_4B_ADAPTER_PATH = os.environ.get("CGT_STEPA4B_ADAPTER", str(WT / "stepc-4b/data/cache/stepc/runs/stepc_4b/ckpt_100.safetensors"))
STEPC_4B_LORA = {"rank": 16, "alpha": 32.0, "dropout": 0.05}  # = cgt.stepc.config RANK, ALPHA, DROPOUT (Phase 3B recipe)
STEPC_MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "10", "MLX_MAX_MB_PER_BUFFER": "10"}  # = cgt.stepc.config.MLX_ENV
MATCHED_PROTOCOL_SHA256 = "b3e2c2912d12b59dec81c149ae8f883a2d578facea7f2083d6d9900a3e1cd1bb"  # protocol_sha256("reasoning"), as matched_v1
LETTERLOGIT_C5_PROTOCOL_SHA256 = "8681e3bcbc38b688acbfc7e63b928c99823c5691e11476a257ce62a99c4e249d"  # letterlogit_protocol.protocol_sha256()

# ---- the C5 blend ----
C5_ENSEMBLE_W = {"size": 0.3, "lens": 0.2, "angle": 0.25, "framing": 0.2, "light_type": 0.0, "light_cond": 0.25, "composition": 0.15}
C5_BLEND_SHA256 = "6d528fac7ccfcda8d1aa64e856105ce833668f08ac2c7e809961df974f6b054b"  # letterlogit_protocol.blend_sha256(C5_ENSEMBLE_W)
# ---- the blended component: frozen Phase 3A SigLIP heads (tag siglip_base_filmsplit_lr1e-2_ep300, as scripts/p3s_stepa_siglip.py) ----
SIGLIP_TAG = "siglip_base_filmsplit_lr1e-2_ep300"
SIGLIP_HEADS_DIR = os.environ.get("CGT_P3A_HEADS_DIR", str(WT / "phase2/models/p3" / SIGLIP_TAG))  # gitignored; pinned by file sha256
SIGLIP_SPEC = {
    "tag": SIGLIP_TAG, "seeds": [0, 1, 2],
    "chosen": {"size": "softmax_mlp", "lens": "linear", "angle": "mlp", "framing": "mlp", "light_type": "linear", "light_cond": "linear",
               "composition": "mlp"},  # = results/phase3/runs/<tag>/chosen_heads_<tag>.json (the 7 attribute keys)
    "head_files_sha256": {
        "size_softmax_mlp_s0.pt": "6535d1529092f3ef07a5834c8c6db5a07fe0cb67fde4a678012f94bf693cb545",
        "size_softmax_mlp_s1.pt": "055abfd72a692b8fd03e08ddbefbb4c304fe0fbc70801ba18161c2e8ed436998",
        "size_softmax_mlp_s2.pt": "f587d6151b39bd61df34ec01c1ac7813af1018294187f9ef0a8df85989868be8",
        "lens_linear_s0.pt": "a70a47d6e3ae49033588b2bcdd7c9ff945770fda105742ee8db9c0da239d9b5c",
        "lens_linear_s1.pt": "e8735633f2f58de7b8bf88a52bf540e8776e494ca1601509974e723e27c78208",
        "lens_linear_s2.pt": "07e6e2fadb5cacf7dee7ac5adfd42e8aa53c372998e08d922b8ade0ecd7adc2b",
        "angle_mlp_s0.pt": "b96b338e38926cd3d32dab5d5ab8d953f46c14ccaa3d0a9b75981919e57d66c8",
        "angle_mlp_s1.pt": "b3df6f5120dbdd0a5e9a076e55aafda9524aa3154c41535e6641381408ad6a3d",
        "angle_mlp_s2.pt": "6242d1bf0ba2ba194ff0bc07c4ca5da652ee6eec4fd68901f0a8c6a2a66e18f0",
        "framing_mlp_s0.pt": "845d31dd59bb26368b322dea6a1ac5df34e3f349e46b312d2fcbf34e7a5d5177",
        "framing_mlp_s1.pt": "585d631ccb2eb452cfc154b5b42ce6d49d626286859d13f2cada85ff0e84f16a",
        "framing_mlp_s2.pt": "1bf8f197d6d4a583a8da3d7f2ba46f8f751cba6fbe3516fc00f24e5eaf5013c2",
        "light_type_linear_s0.pt": "af5cb1ae38efdb88cb7f7da170f71874199a5b4b43aa388ea16098a200e08365",
        "light_type_linear_s1.pt": "8f9ad0eb3ecb842aa8cc1e8a5a60537ddcd90617bd049a9aa7f0c72a2fae52e3",
        "light_type_linear_s2.pt": "0362a4f600087aa17936e52c1fc8b5a423df337b2ec35ba997020e18657cb8a6",
        "light_cond_linear_s0.pt": "bcbd120232d267b70ce8688e9440c7b7b6f27ac928d73e3faad28edb5e40feb7",
        "light_cond_linear_s1.pt": "672901155c59001bfbdef7b0d23ca00680a572e9d9014274a4989b7d87afc02f",
        "light_cond_linear_s2.pt": "82d45680e086eda4d04cd00f23d93a9c8c38dcd10ad3f4606bbb19e9da2e5663",
        "composition_mlp_s0.pt": "5345361bcf974aaf70e36b320b8827ee00c370b6832e956cc86a98837fcdb505",
        "composition_mlp_s1.pt": "caf7c646b35e8c23dfdef6bdd10d284b6f995029dfc6871ed8c07ff2194606d7",
        "composition_mlp_s2.pt": "548a6b741a929d5dd47449c4e3aaa6d4b864b935de4bb63a83bb959a0aa7b123"},
    "backbone": "google/siglip-base-patch16-224", "backbone_revision": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
    "processor_sha256": "71f49a65be9c90b78ea3dde072a67ff62d14824c40ee72e15742db74ce12faad",  # = the ShotQA 3A feature cache provenance
}


def siglip_identity_sha256(spec: dict) -> str:
    """Identity of the blended SigLIP component: heads (file hashes), chosen head kinds, seeds, backbone revision and processor config hash."""
    keys = ("tag", "seeds", "chosen", "head_files_sha256", "backbone", "backbone_revision", "processor_sha256")
    return hashlib.sha256(json.dumps({k: spec[k] for k in keys}, sort_keys=True).encode()).hexdigest()


SIGLIP_IDENTITY_SHA256 = "4420a8b3ddf6812d69a6ab9cb8745c8a64d32f09164d639a8d6977721985d33d"  # siglip_identity_sha256(SIGLIP_SPEC)

MODELS: dict[str, dict] = {
    "matched_v1": {
        "description": "our frozen Phase 3B model: Qwen3-VL-2B-Instruct + LoRA (film-split, 100% checkpoint)",
        "repo": "Qwen/Qwen3-VL-2B-Instruct", "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "adapter_rel": "data/cache/p3b/runs/qwen3vl2b_lora_filmsplit_run1/ckpt_100.safetensors",
        "lora": {"rank": 16, "alpha": 32.0, "dropout": 0.05},
        "identity_sha256": FROZEN_ADAPTER_SHA256, "identity_kind": "adapter_sha256",
        "image_spec": {"kind": "budget", "budget_px": 147_456, "multiple": 32},
        "variant": "reasoning",  # DECLARED recommendation (docs section 2.2); same for both tags; author to confirm before any GO
    },
    "shotvl3b_matched_v1": {
        "description": "Vchitect/ShotVL-3B (Qwen2.5-VL-3B-Instruct, SFT+GRPO on ShotQA); weights not downloaded yet",
        "repo": "Vchitect/ShotVL-3B", "revision": None, "adapter_rel": None, "lora": None,
        "identity_sha256": None, "identity_kind": "weights_sha256",  # pinned only after the user-approved download (plan Stage 0.2)
        "image_spec": {"kind": "processor_default"},
        "variant": "reasoning",
    },
    # ---- NEW: prepared, NOT run; each needs the user's go and a author GO ----
    "matched_4b_v1": {
        "description": "Step C Qwen3-VL-4B-Instruct + LoRA (stepc_4b ckpt_100), single pass, matched protocol exactly as matched_v1",
        "protocol": "matched", "protocol_sha256": MATCHED_PROTOCOL_SHA256,
        "repo": STEPC_4B_REPO, "revision": STEPC_4B_REVISION, "adapter_rel": None, "adapter_path": STEPC_4B_ADAPTER_PATH, "lora": STEPC_4B_LORA,
        "identity_sha256": STEPC_4B_ADAPTER_SHA256, "identity_kind": "adapter_sha256",
        "image_spec": {"kind": "budget", "budget_px": 147_456, "multiple": 32},
        "variant": "reasoning", "mlx_env": STEPC_MLX_ENV,
    },
    "letterlogit_4b_c5_v1": {
        "description": "a recorded decision candidate recipe: Step C 4B letter logits (single pass) + fixed C5 SigLIP blend; LETTER-LOGIT protocol, "
                       "not the matched generative protocol",
        "protocol": "letterlogit_c5", "protocol_sha256": LETTERLOGIT_C5_PROTOCOL_SHA256,
        "repo": STEPC_4B_REPO, "revision": STEPC_4B_REVISION, "adapter_rel": None, "adapter_path": STEPC_4B_ADAPTER_PATH, "lora": STEPC_4B_LORA,
        "identity_sha256": STEPC_4B_ADAPTER_SHA256, "identity_kind": "adapter_sha256",
        "image_spec": {"kind": "budget", "budget_px": 147_456, "multiple": 32},
        "variant": None, "mlx_env": STEPC_MLX_ENV,
        "blend": {"config": "C5", "source_rel": "results/stepa_4b/val.json", "ensemble_w": C5_ENSEMBLE_W, "blend_sha256": C5_BLEND_SHA256},
        "siglip": {**SIGLIP_SPEC, "heads_dir": SIGLIP_HEADS_DIR, "identity_sha256": SIGLIP_IDENTITY_SHA256},
    },
}


def entry(tag: str) -> dict:
    e = MODELS.get(tag)
    if e is None:
        raise guards.TagNotBound(f"model tag {tag!r} is not registered in cgt.p3m_matched.registry")
    if not e.get("identity_sha256") or not e.get("revision"):
        raise guards.TagNotBound(f"model tag {tag!r} has no pinned identity (revision/weights hash); refused")
    return e


def weights_identity(snapshot_dir: Path) -> str:
    """SHA-256 over the sorted (name, sha256) list of *.safetensors in a snapshot (identity of a full-weights model such as ShotVL-3B)."""
    files = sorted(Path(snapshot_dir).glob("*.safetensors"))
    if not files:
        raise guards.AdapterHashError(f"no safetensors in {snapshot_dir}")
    return hashlib.sha256(json.dumps([[f.name, guards.sha256_file(f)] for f in files]).encode()).hexdigest()


def require_identity(tag: str, data_root: Path = ROOT) -> dict:
    """Recompute the model identity and require it to equal the registry (AdapterHashError, exit 7). Returns the entry with resolved paths."""
    e = dict(entry(tag))
    if e["identity_kind"] == "adapter_sha256":
        adapter = Path(e["adapter_path"]) if e.get("adapter_path") else Path(data_root) / e["adapter_rel"]
        got = guards.sha256_file(adapter)
        e["adapter_file"] = str(adapter)
    else:
        from cgt.p3b_vlm.model_path import pinned_model_path
        got = weights_identity(Path(pinned_model_path(e["repo"], e["revision"])))
        e["adapter_file"] = None
    if got != e["identity_sha256"]:
        raise guards.AdapterHashError(f"model identity of {tag!r} is {got}, the registry pins {e['identity_sha256']}")
    return e


# ---------- protocol dispatch, GO extras and the C5 component checks ----------
def protocol_of(e: dict) -> str:
    """'matched' (default, every pre-existing tag) or 'letterlogit_c5'."""
    return e.get("protocol", "matched")


def protocol_hash(e: dict) -> str:
    """The protocol hash computed from the CURRENT code for the entry's protocol. A pinned `protocol_sha256` must equal it (TagNotBound)."""
    if protocol_of(e) == "letterlogit_c5":
        from cgt.p3m_matched import letterlogit_protocol
        got = letterlogit_protocol.protocol_sha256()
    elif protocol_of(e) == "matched":
        from cgt.p3m_matched.protocol import protocol_sha256
        got = protocol_sha256(e["variant"])
    else:
        raise guards.TagNotBound(f"unknown protocol {protocol_of(e)!r}")
    if e.get("protocol_sha256") and got != e["protocol_sha256"]:
        raise guards.TagNotBound(f"the protocol code hashes to {got[:16]}, the tag pins {e['protocol_sha256'][:16]}: refused")
    return got


def go_extras(e: dict) -> list[tuple[str, str]]:
    """Extra GO lines (fixed order) beyond tag / identity / protocol: the PINNED blend and SigLIP identity of a letterlogit_c5 tag; none otherwise."""
    if protocol_of(e) != "letterlogit_c5":
        return []
    return [("blend_sha256", e["blend"]["blend_sha256"]), ("siglip_identity_sha256", e["siglip"]["identity_sha256"])]


def require_c5_components(e: dict, data_root: Path = ROOT, provenance=None) -> dict:
    """letterlogit_c5 only (AdapterHashError, exit 7, on any mismatch): (a) the C5 weights re-read from the committed Step A-4B fit hash to the
    pinned blend_sha256 and equal the pinned weights; (b) the registry's SigLIP spec hashes to its pinned identity; (c) every head file
    hashes to its pin; (d) chosen_heads_<tag>.json names the pinned head kinds; (e) the SigLIP backbone revision and processor hash
    (`provenance()`, default cgt.p3_static_attrs.features.current_provenance: no weights loaded) equal the pins."""
    from cgt.p3m_matched import letterlogit_protocol as LP
    b, sg = e["blend"], e["siglip"]
    src = Path(data_root) / b["source_rel"]
    if not src.is_file():
        raise guards.AdapterHashError(f"blend source {b['source_rel']} missing")
    w = LP.load_blend(src.read_text())
    if LP.blend_sha256(w) != b["blend_sha256"] or LP.blend_sha256(b["ensemble_w"]) != b["blend_sha256"]:
        raise guards.AdapterHashError("the C5 blend weights do not hash to the pinned blend_sha256")
    if siglip_identity_sha256(sg) != sg["identity_sha256"]:
        raise guards.AdapterHashError("the SigLIP component spec does not hash to its pinned identity")
    hd = Path(sg["heads_dir"])
    need = {f"{a}_{k}_s{s}.pt" for a, k in sg["chosen"].items() for s in sg["seeds"]}
    if need != set(sg["head_files_sha256"]):
        raise guards.AdapterHashError("the pinned head files do not match the chosen heads and seeds")
    for name, sha in sorted(sg["head_files_sha256"].items()):
        if guards.sha256_file(hd / name) != sha:
            raise guards.AdapterHashError(f"SigLIP head {name} does not match its pinned sha256")
    chosen_p = Path(data_root) / f"results/phase3/runs/{sg['tag']}/chosen_heads_{sg['tag']}.json"
    chosen = json.loads(chosen_p.read_text()) if chosen_p.is_file() else {}
    if {a: chosen.get(a) for a in sg["chosen"]} != sg["chosen"]:
        raise guards.AdapterHashError("chosen_heads json does not name the pinned head kinds")
    if provenance is None:
        from cgt.p3_static_attrs.features import current_provenance as provenance
    prov = provenance(sg["backbone"])
    if prov.get("model_revision") != sg["backbone_revision"] or prov.get("processor_sha256") != sg["processor_sha256"]:
        raise guards.AdapterHashError("the SigLIP backbone revision or image-processor config differs from the pins")
    return {"ensemble_w": w, "blend_sha256": b["blend_sha256"], "blend_source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "siglip_identity_sha256": sg["identity_sha256"], "heads_dir": str(hd)}
