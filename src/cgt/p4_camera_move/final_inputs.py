"""Provenance of the prepared, label-free inputs of the final evaluations (Amendment 12 item 3b, Amendment 13 item 2). `scripts/p4_final_prepare.py` writes a manifest with `build_manifest`; the evaluator calls
`validate_prepared` BEFORE any state file or label access and again when it uses the table: a stale, partial or hand-edited preparation is refused with `UnreadableImage` (exit 6). Nothing here reads a label."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from cgt.p3b_vlm.guards import UnreadableImage
from cgt.p4_camera_move import conform

ROOT = Path(__file__).resolve().parents[3]
REQUIRED = ("dataset", "n_files", "input_sha256", "feature_table_sha256", "columns_sha256", "provenance", "labels_read")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def columns_sha256(columns: list[str]) -> str:
    return hashlib.sha256(json.dumps(list(columns)).encode()).hexdigest()


def current_provenance(root: Path = ROOT) -> dict:
    """Conformity version, code hash and extractor keys, plus the SHA-256 of the preparation script: what a manifest must carry to be accepted now."""
    script = Path(root) / "scripts/p4_final_prepare.py"
    return {**conform.provenance(), "prepare_script_sha256": sha256_file(script) if script.exists() else "missing"}


def build_manifest(dataset: str, inputs: list[Path], table: Path, columns: list[str], *, root: Path = ROOT, **extra) -> dict:
    """The manifest the preparation writes (input bytes hashed by name, table digest, column hash, code provenance)."""
    return {"dataset": dataset, "labels_read": False, "n_files": len(inputs), "input_sha256": {Path(p).name: sha256_file(Path(p)) for p in inputs}, "feature_table_sha256": sha256_file(table),
            "columns_sha256": columns_sha256(columns), "n_features": len(columns), "provenance": current_provenance(root), **extra}


def load_manifest(manifest_path: Path) -> dict:
    p = Path(manifest_path)
    if not p.is_file():
        raise UnreadableImage("the preparation manifest is missing: run scripts/p4_final_prepare.py first")
    try:
        m = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise UnreadableImage("the preparation manifest is not valid JSON") from e
    missing = [k for k in REQUIRED if k not in m]
    if missing:
        raise UnreadableImage(f"the preparation manifest lacks required fields {missing}")
    return m


def validate_prepared(dataset: str, clips: list[str], input_dir: Path, table: Path | None, manifest_path: Path, *, train_columns: list[str] | None = None, root: Path = ROOT, check_inputs: bool = True) -> dict:
    """Validate the preparation of `dataset` against the clip list of the question set / GIF list. `table=None` validates the inputs only (video model). Returns the manifest."""
    m = load_manifest(manifest_path)
    if m["dataset"] != dataset or m["labels_read"] is not False:
        raise UnreadableImage("the preparation manifest belongs to another dataset or claims label access")
    if table is not None and (not Path(table).is_file() or sha256_file(Path(table)) != m["feature_table_sha256"]):
        raise UnreadableImage("the prepared feature table is missing or does not match its manifest digest")
    cur = current_provenance(root)
    if m["provenance"] != cur or m["provenance"].get("conform_version") != conform.CONFORM_VERSION:
        raise UnreadableImage("the preparation provenance (conformity version/code, extractor keys, preparation script) does not match the current code")
    if len(set(clips)) != len(clips) or set(m["input_sha256"]) != set(clips) or m["n_files"] != len(clips):
        raise UnreadableImage("the manifest does not cover exactly the clips of the question set (unique, complete)")
    if check_inputs:
        for name in clips:
            f = Path(input_dir) / name
            if not f.is_file() or sha256_file(f) != m["input_sha256"][name]:
                raise UnreadableImage("an input file is missing or its bytes differ from the preparation manifest")
    if table is not None:
        tc = pd.read_parquet(table, columns=["clip"])["clip"].tolist()
        if len(set(tc)) != len(tc) or set(tc) != set(clips):
            raise UnreadableImage("the prepared table does not cover exactly the clips once each")
        cols = [c for c in pd.read_parquet(table).columns if c != "clip"]
        if columns_sha256(cols) != m["columns_sha256"] or (train_columns is not None and columns_sha256(train_columns) != m["columns_sha256"]):
            raise UnreadableImage("the prepared feature columns differ from the manifest or the training columns")
    return m
