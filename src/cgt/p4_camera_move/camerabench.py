"""CameraBench PUBLIC TEST set as sealed, EVALUATION-ONLY data (docs/phase4_interface.md section 7). The GIF files can be listed (names only); the labels (`test.jsonl`) are reachable
ONLY through `CameraBenchTestStore.labels(final_evaluation=True, model_tag=...)`, which (1) raises `CameraBenchLocked` otherwise, (2) refuses a second final evaluation of the same
frozen model tag (`CameraBenchAlreadyEvaluated`) and (3) appends the evaluation to the ledger BEFORE the labels are read. No training, tuning, threshold or feature selection on it.
The ledger (default results/phase4/test_looks.jsonl) is shared with the ShotBench movement final evaluations (field `dataset`)."""
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CB_ROOT = ROOT / "data/cache/p4/camerabench"
LEDGER = ROOT / "results/phase4/test_looks.jsonl"
FREEZE = ROOT / "results/phase4/runs/video_v1/FREEZE.json"  # the committed freeze record whose `registered_models` lists the model tags that may open the labels
DATASET = "camerabench_public_test"

# CameraBench primitive -> our schema atom (docs/phase4_video_plan.md section 6); stability, speed and complexity tags are ignored; tracking-shot tags map to the `track` flag
PRIMITIVE_TO_ATOM = {"dolly-in": "push-in", "dolly-out": "pull-out", "pan-left": "pan-left", "pan-right": "pan-right", "tilt-up": "tilt-up", "tilt-down": "tilt-down",
                     "truck-left": "move-left", "truck-right": "move-right", "pedestal-up": "boom-up", "pedestal-down": "boom-down", "zoom-in": "zoom-in", "zoom-out": "zoom-out",
                     "arc-CW": "arc", "arc-CCW": "arc", "roll-CW": "roll", "roll-CCW": "roll", "static": "static"}
STATIC_VARIANT = {**PRIMITIVE_TO_ATOM, "no-motion": "static"}  # declared sensitivity variant: 'no-motion' also counts as static
TRACKING_TAGS = {"side-tracking", "pan-tracking", "tilt-tracking", "arc-tracking", "lead-tracking", "tail-tracking", "aerial-tracking"}


class CameraBenchLocked(RuntimeError):
    """Labels were requested without the explicit final evaluation."""


class CameraBenchNotRegistered(RuntimeError):
    """The model tag is not registered in a COMMITTED freeze record (or its adapter / configuration hash does not match the registered one)."""


class CameraBenchAlreadyEvaluated(RuntimeError):
    """The ledger already holds a final evaluation of this model tag on this dataset (one final evaluation per frozen model)."""


def committed_registry(freeze_path: Path, *, repo_root: Path = ROOT, git_ref: str = "HEAD") -> dict:
    """The `registered_models` mapping of the freeze record AS COMMITTED at `git_ref` (read with `git show`, never from the working copy: an untracked or uncommitted-edited file
    registers nothing). Raises CameraBenchNotRegistered if the file is not committed there, is not JSON, or has no non-empty `registered_models` mapping."""
    rel = Path(freeze_path).resolve().relative_to(Path(repo_root).resolve())
    r = subprocess.run(["git", "show", f"{git_ref}:{rel.as_posix()}"], cwd=repo_root, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise CameraBenchNotRegistered(f"{rel} is not committed at {git_ref}: no model can open the CameraBench labels")
    try:
        reg = json.loads(r.stdout).get("registered_models")
    except (json.JSONDecodeError, AttributeError) as e:
        raise CameraBenchNotRegistered(f"{rel} at {git_ref} is not a valid freeze record") from e
    if not isinstance(reg, dict) or not reg:
        raise CameraBenchNotRegistered(f"{rel} at {git_ref} registers no model")
    return reg


def require_registered(model_tag: str, registry: dict, *, adapter_sha256: str | None = None, config_hash: str | None = None) -> dict:
    """The guard: `model_tag` must be a key of the committed registry, and every hash the registry entry carries (`adapter_sha256`, `config_hash`) must equal the value the caller supplies
    (a missing value is a mismatch). Returns the registry entry. Pure function over the registry dict (synthetic registries in the tests)."""
    entry = registry.get(model_tag)
    if not isinstance(entry, dict):
        raise CameraBenchNotRegistered(f"model tag '{model_tag}' is not registered in the committed freeze record")
    for key, given in (("adapter_sha256", adapter_sha256), ("config_hash", config_hash)):
        if key in entry and entry[key] != given:
            raise CameraBenchNotRegistered(f"'{model_tag}': {key} does not match the registered value")
    return entry


def source_id_of(name: str) -> str:
    """`-2uIa-XMJC0.5.3.gif` (or .mp4) -> `-2uIa-XMJC0` (YouTube id; filename-derived proxy)."""
    return re.sub(r"\.\d+\.\d+\.(?:gif|mp4)$", "", Path(name).name)


def map_labels(labels: list[str], variant: dict[str, str] | None = None) -> dict:
    """{atoms: sorted unique schema atoms, flags: ['track'] if any tracking tag, ignored: the other tags}; `variant` defaults to PRIMITIVE_TO_ATOM."""
    table = variant or PRIMITIVE_TO_ATOM
    atoms = sorted({table[x] for x in labels if x in table})
    flags = ["track"] if any(x in TRACKING_TAGS for x in labels) else []
    return {"atoms": atoms, "flags": flags, "ignored": sorted(x for x in labels if x not in table and x not in TRACKING_TAGS)}


class CameraBenchTestStore:
    def __init__(self, root: Path = CB_ROOT, ledger: Path = LEDGER, freeze_path: Path = FREEZE, repo_root: Path = ROOT, git_ref: str = "HEAD"):
        self.root, self.ledger = Path(root), Path(ledger)
        self.freeze_path, self.repo_root, self.git_ref = Path(freeze_path), Path(repo_root), git_ref

    def video_files(self) -> list[Path]:
        """The GIF files, sorted by name. Names only: no label is read."""
        return sorted((self.root / "videos_gif").glob("*.gif"))

    def source_ids(self) -> list[str]:
        return [source_id_of(p.name) for p in self.video_files()]

    def _evaluated(self, dataset: str, model_tag: str) -> bool:
        return self.ledger.exists() and any(json.loads(x).get("dataset") == dataset and json.loads(x).get("model_tag") == model_tag for x in self.ledger.read_text().splitlines() if x.strip())

    def labels(self, *, final_evaluation: bool = False, model_tag: str | None = None, adapter_sha256: str | None = None, config_hash: str | None = None, code_sha: str = "") -> list[dict]:
        """Rows of `test.jsonl` ({Video, labels, caption, path} plus `stem`), only for the explicit final evaluation of a model tag REGISTERED in the committed freeze record (adapter and/or
        configuration hash must match the registered ones); the ledger line is written first."""
        if not final_evaluation or not model_tag:
            raise CameraBenchLocked("CameraBench public test labels are sealed: pass final_evaluation=True and the frozen model_tag")
        require_registered(model_tag, committed_registry(self.freeze_path, repo_root=self.repo_root, git_ref=self.git_ref), adapter_sha256=adapter_sha256, config_hash=config_hash)
        if self._evaluated(DATASET, model_tag):
            raise CameraBenchAlreadyEvaluated(f"{DATASET} was already evaluated for model '{model_tag}'")
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger, "a") as fh:
            fh.write(json.dumps({"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), "dataset": DATASET, "model_tag": model_tag, "config_hash": config_hash or "", "adapter_sha256": adapter_sha256 or "",
                                 "code_sha": code_sha, "n_final_evaluations_logged_so_far": 1 + sum(1 for x in (self.ledger.read_text().splitlines() if self.ledger.exists() else []) if x.strip())}) + "\n")
        rows = [json.loads(x) for x in (self.root / "test.jsonl").read_text().splitlines() if x.strip()]
        for r in rows:
            r["stem"] = Path(r["path"]).stem
        return rows
