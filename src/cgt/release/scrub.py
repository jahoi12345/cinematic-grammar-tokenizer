"""Scrubbed copies of the Phase 4 raw prediction dumps (results/phase4/final_eval/raw/*.predictions.jsonl).

Each record keeps its ids and the correctness flag (model outputs only with keep_model_outputs=True, see MODEL_OUTPUTS); the gold answer is replaced by a salted hash and the benchmark
option text is dropped. Every field must be classified below: an unknown field stops the scrub (fail closed). The originals are only
ever opened for reading; outputs go to a separate, self-ignoring directory (a `.gitignore` of `*` is written into it).
"""
import hashlib
import json
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path

SCRUB_VERSION = "cgt-scrub-v1"
SUFFIX = ".scrubbed.jsonl"
KEEP = ("variant", "id", "clip", "source", "chosen", "hit", "scores",  # Phase 4 final_eval/raw: ids, model output, correctness flag
        "attribute", "compound", "pred", "logits",  # Phase 3B raw_eval_logs (same tool, --in results/phase3b/raw_eval_logs)
        "letter", "text", "method")  # p3m matched evaluation (results/p3m_matched/raw_eval_logs)
HASH = ("gold_letter", "gold")  # gold answer -> salted hash
DROP = ("options", "gold_atoms", "question", "answer", "choices", "prompt",  # benchmark item text and other gold forms
        # p3m letter-logit C5 dump: always dropped, also with keep_model_outputs. c0_letter is the VLM-only letter (on a c0 hit it
        # IS the gold letter); vlm_logits / siglip_scores / w allow re-deriving both letters (argmax / blend); letters is the option letter set
        "c0_letter", "letters", "vlm_logits", "siglip_scores", "w")
MODEL_OUTPUTS = ("chosen", "scores", "pred", "logits", "letter", "text")  # dropped by DEFAULT: on a hit row `chosen` equals the gold letter (answer recovery)
ANSWER_RECOVERY_WARNING = ("WARNING: --keep-model-outputs keeps chosen/scores/pred/logits/letter/text; on every row with hit=true the model's letter IS the gold "
                           "letter, so anyone holding the benchmark can recover those gold answers. Do not publish this output without the user's decision.")
SCALAR = (str, int, float, bool, type(None))  # kept non-model values must be scalars: a dict/list could carry nested gold or option text
BOOL_FIELDS = ("hit", "compound")
RENAME_BOOL = {"correct": "hit",  # a correctness flag under another name: kept as the bool `hit`
               "c0_correct": "c0_hit"}  # the descriptive C0 (VLM-only) correctness flag of the C5 dump: kept like `hit`, as the bool `c0_hit`
ENUMS = {"method": ("option",)}  # kept only when the value is a known label (read from the p3m file's distinct values); else refused
DEFAULT_OUT = Path("data/release/phase4_final_eval_raw")
DEFAULT_SALT = Path("data/release/.salt")


class ScrubError(ValueError):
    pass


def _git(cwd: Path, *args: str) -> int | None:
    """Return code of `git -C cwd args`, or None when cwd is not inside a git work tree."""
    try:
        inside = subprocess.run(["git", "-C", str(cwd), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True, check=False)
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return None
        return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False).returncode
    except (FileNotFoundError, NotADirectoryError):
        return None


def _git_tracked(path: Path) -> bool:
    """Tracked or staged (the index holds it)."""
    return _git(path.parent, "ls-files", "--error-unmatch", path.name) == 0


def _git_unignored(path: Path) -> bool:
    """True when `path` lies in a git work tree and `git check-ignore` does NOT ignore it (one `git add -A` from being committed)."""
    rc = _git(path.parent, "check-ignore", "-q", path.name)
    return rc is not None and rc != 0


def load_salt(path: Path, create: bool = False) -> bytes:
    path = Path(path).resolve()  # a symlink to a tracked salt is judged by its target
    if not path.exists():
        if not create:
            raise ScrubError(f"salt file {path} does not exist (pass --create-salt to make one; it is never committed)")
        path.parent.mkdir(parents=True, exist_ok=True)
        _self_ignore(path.parent)
        path.write_text(secrets.token_hex(32) + "\n")
        path.chmod(0o600)
    if _git_tracked(path):
        raise ScrubError(f"salt file {path} is tracked by git: refuse (a committed salt makes the hashes reversible)")
    if _git_unignored(path):
        raise ScrubError(f"salt file {path} is not git-ignored: refuse (add a .gitignore of `*` to its directory, or move it out of the repo)")
    salt = path.read_bytes().strip()
    if len(salt) < 16:
        raise ScrubError(f"salt file {path} is too short (< 16 bytes)")
    return salt


def salt_id(salt: bytes) -> str:
    return hashlib.sha256(b"cgt-salt-id:" + salt).hexdigest()[:8]


def gold_hash(salt: bytes, stem: str, rec_id: str, field: str, value) -> str:
    msg = json.dumps([SCRUB_VERSION, stem, rec_id, field, value], sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(salt + b"\0" + msg).hexdigest()


def scrub_record(rec: dict, salt: bytes, stem: str, keep_model_outputs: bool = False) -> dict:
    unknown = sorted(set(rec) - set(KEEP) - set(HASH) - set(DROP) - set(RENAME_BOOL))
    if unknown:
        raise ScrubError(f"{stem}: unclassified field(s) {unknown}; classify them in cgt.release.scrub before scrubbing")
    if "id" not in rec:
        raise ScrubError(f"{stem}: record without an id")
    out = {"scrub": SCRUB_VERSION}
    for k in KEEP:
        if k in rec and (keep_model_outputs or k not in MODEL_OUTPUTS):
            v = rec[k]
            if k not in MODEL_OUTPUTS and not isinstance(v, SCALAR):
                raise ScrubError(f"{stem}: {k}: non-scalar value ({type(v).__name__}); nested data could carry gold or option text")
            if k in BOOL_FIELDS and not isinstance(v, bool):
                raise ScrubError(f"{stem}: {k}: expected a bool, got {type(v).__name__}")
            if k in ENUMS and v not in ENUMS[k]:
                raise ScrubError(f"{stem}: {k}: value not in the known labels {list(ENUMS[k])}: refuse")
            out[k] = v
    for k, new in RENAME_BOOL.items():
        if k in rec:
            if not isinstance(rec[k], bool):
                raise ScrubError(f"{stem}: {k}: expected a bool, got {type(rec[k]).__name__}")
            if new in rec and rec[new] != rec[k]:
                raise ScrubError(f"{stem}: {k} and {new} disagree: refuse")
            out[new] = rec[k]
    if "hit" not in rec and "pred" in rec and "gold" in rec:
        out["hit"] = rec["pred"] == rec["gold"]  # same rule as cgt.p3b_vlm.score; computed before pred and gold are dropped/hashed
        out["hit_derived"] = True
    for k in HASH:
        if k in rec:
            out[f"{k}_hash"] = gold_hash(salt, stem, str(rec["id"]), k, rec[k])
    return out


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _self_ignore(d: Path) -> None:
    gi = d / ".gitignore"
    if not gi.exists():
        gi.write_text("# release scratch: scrubbed outputs and salts are never committed\n*\n")


def scrub_file(src: Path, out_dir: Path, salt: bytes, keep_model_outputs: bool = False, final_dir: Path | None = None) -> dict:
    """Write the scrubbed copy of `src` into `out_dir` (a staging dir when `final_dir` is the destination it will be moved to)."""
    src, out_dir = Path(src), Path(out_dir)
    name = src.name
    stem = name[: -len(".jsonl")] if name.endswith(".jsonl") else src.stem
    dst = out_dir / (stem + SUFFIX)
    for d in (dst, Path(final_dir) / dst.name if final_dir else dst):
        if d.resolve() == src.resolve():
            raise ScrubError("output would overwrite the input")
    rows = []
    with open(src, encoding="utf-8") as f:  # read-only
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ScrubError(f"{name}:{n}: not JSON ({e.msg})") from None
            if not isinstance(rec, dict):
                raise ScrubError(f"{name}:{n}: not a JSON object")
            rows.append(scrub_record(rec, salt, stem, keep_model_outputs))
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(dst)
    return {"input": name, "input_sha256": _sha256(src), "output": dst.name, "output_sha256": _sha256(dst), "records": len(rows)}


def scrub_dir(in_dir: Path, out_dir: Path, salt: bytes, pattern: str = "*.predictions.jsonl", keep_model_outputs: bool = False) -> dict:
    in_dir, out_dir = Path(in_dir).resolve(), Path(out_dir).resolve()
    if out_dir == in_dir or in_dir in out_dir.parents:
        raise ScrubError("the output directory must not be (inside) the input directory")
    files = sorted(in_dir.glob(pattern))
    if not files:
        raise ScrubError(f"no files match {pattern} in {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _self_ignore(out_dir)
    for probe in ("x" + SUFFIX, "scrub_manifest.json", ".scrub-staging-x"):
        if _git_unignored(out_dir / probe):
            raise ScrubError(f"output directory {out_dir} is in a git work tree and its outputs would not be git-ignored "
                             "(its .gitignore does not ignore them): refuse")
    stage = Path(tempfile.mkdtemp(prefix=".scrub-staging-", dir=out_dir))  # all-or-nothing: moved into place only when every file succeeded
    try:
        items = [scrub_file(p, stage, salt, keep_model_outputs, final_dir=out_dir) for p in files]
        kept = [k for k in KEEP if keep_model_outputs or k not in MODEL_OUTPUTS]
        manifest = {"scrub_version": SCRUB_VERSION, "salt_id": salt_id(salt), "keep_model_outputs": keep_model_outputs, "kept": kept,
                    "hashed": list(HASH), "dropped": list(DROP) + ([] if keep_model_outputs else list(MODEL_OUTPUTS)), "files": items}
        (stage / "scrub_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        for f in sorted(stage.iterdir()):
            f.replace(out_dir / f.name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return manifest
