"""Release checker: scan a tree (a directory, or a `git ls-files` list) and report every file that must not be published.

One hit = (rule, path). Content is read only to match it; it is never printed. Any hit fails the check.
Owned-film terms come from the film registry (configs/films.yaml + the gitignored overlay configs/films.local.yaml) of the code checkout
AND of the scanned tree, so a release tree whose films.yaml was trimmed still cannot hide the owned film.
"""
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from cgt.release.scrub import DROP, HASH, MODEL_OUTPUTS, RENAME_BOOL, SCRUB_VERSION, SUFFIX

CODE_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = "results/phase4/final_eval/raw/"
PRED_RE = re.compile(r"\.predictions(\.scrubbed)?\.jsonl$")  # per-item prediction dumps (and their scrubbed copies) anywhere under results/

RULES = {
    "OWNED_PATH": "path mentions an owned film (non-public-domain or local-overlay registry entry: id, title or title alias)",
    "OWNED_CONTENT": "file content mentions an owned film (id, title or title alias)",
    "MEDIA_UNATTRIBUTED": "image/video/audio file whose path does not name a public-domain film (or also names an owned film)",
    "WEIGHTS": ("model weights or ShotQA-derived artifact: *.safetensors/.pt/.bin/..., adapter_config.json, any .npy/.npz/.pkl/.joblib dump, "
                "adapter/LoRA/heads/codebook/palette/centroid-named .parquet, or such a .json holding numeric arrays"),
    "RAW_PREDICTIONS": ("file under results/phase4/final_eval/raw/, or any *.predictions.jsonl under results/, that is not a valid "
                        "scrubbed copy (*.scrubbed.jsonl, scrub version, no gold/option/model-output field)"),
    "DATA_DIR": "file under data/ or models/ (local data, never published)",
    "BENCH_FILE": "known benchmark/ShotQA data file name or a file inside a shotbench/camerabench/shotqa directory",
    "BENCH_FIELDS": ("file whose records carry benchmark/ShotQA answer fields (gold, gold_*, gt/ground_truth/correct_answer; answer with "
                     "question/options; a *label(s) field when the path names a benchmark; a correctness key (*hit/*correct; *accuracy 0/1 in records) with a model-output key "
                     "(*chosen/*pred/*letter/*text/*output/*response...) in the same record, a direct child dict, or header); .txt/.log dumps: a quoted gold key"),
    "LOCAL_OVERLAY": "the gitignored local film registry overlay (configs/films.local.yaml)",
    "LOCAL_TOOLING": ("file under a root dot-directory other than .github/ and .streamlit/ (local tool settings, worktrees, local absolute "
                      "paths: never published; never opened)"),
    "SYMLINK": "a symbolic link (file or directory); never followed, its target is unknown to the check",
    "REVIEW": ("content the checker cannot read: archives (.zip/.tar/.gz/...), .pdf, .parquet/.arrow, files over 64 MB, binary or "
               "non-UTF-8 files of an unknown type, a nested .git/.venv directory; a human must look before release. Cleared only by an "
               "entry of docs/release_reviewed_files.yaml whose sha256 equals the file's current sha256"),
    "STALE_REVIEW": ("a file flagged REVIEW and listed in docs/release_reviewed_files.yaml whose current sha256 differs from the reviewed one "
                     "(the file changed after the review; the REVIEW hit stays)"),
}
REVIEWED_FILE = "docs/release_reviewed_files.yaml"  # committed allow-list in the scanned tree: clears REVIEW only, by exact sha256
REVIEWED_KEYS = ("path", "sha256", "reviewed_by", "date", "note")
CLEARABLE = {"REVIEW"}  # the only rule an allow-list entry can clear
PDF_SWIFT = Path(__file__).resolve().parents[3] / "scripts/release/pdf_text.swift"
SWIFT = "/usr/bin/swift"
PDF_TIMEOUT = 120
ALIASES_FILE = "configs/release_aliases.yaml"  # optional: aliases: {<film_id>: [<alias>, ...]} (abbreviations, partial/original-language titles)

MEDIA_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic",
             ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".ts", ".y4m",
             ".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg", ".opus", ".aif", ".aiff", ".wma"}
WEIGHT_EXT = {".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".gguf", ".onnx", ".h5", ".mlmodel", ".tflite"}
ARRAY_EXT = {".npy", ".npz", ".pkl", ".pickle", ".joblib"}  # array/pickle dumps can carry weights or centroids: always reviewed
ARTIFACT_EXT = {".parquet", ".arrow"}  # WEIGHTS when the name says adapter/heads/codebook/palette/centroid, else REVIEW
ARCHIVE_EXT = {".zip", ".tar", ".gz", ".tgz", ".7z", ".bz2", ".xz", ".rar", ".zst", ".lz4"}
REVIEW_EXT = ARCHIVE_EXT | {".pdf"} | ARTIFACT_EXT
KNOWN_BINARY_EXT = MEDIA_EXT | WEIGHT_EXT | ARRAY_EXT | REVIEW_EXT  # binary files another rule already decides on
JSON_WEIGHT_MIN = 64  # a keyword-named .json with >= this many numbers inside lists is a weight/centroid dump (hash manifests have none)
ARTIFACT_WORDS = re.compile(r"(adapter|lora|heads?|codebook|palette|centroid)", re.IGNORECASE)
BENCH_DIRS = {"shotbench", "camerabench", "shotqa"}
BENCH_NAMES = {"test.tsv", "test.jsonl", "videos.csv", "grpo.json", "meta.jsonl", "data.tar.gz"}
GOLD_RE = re.compile(r"^(gold(_(letter|atoms|answer|option|options|label|labels))?|gt_answer|ground_truth|correct_answer|gt)$")  # not *_hash
ANSWER_WEAK = {"answer"}
LABEL_RE = re.compile(r"(^|_)labels?$")  # a label field in a file whose path names a benchmark
QUESTION = {"question", "options", "choices"}
# K-A15: correctness and model-output keys by pattern (after norm_key), in any structured file (json/jsonl/yaml/csv/tsv/html/md-table header)
CORRECT_RE = re.compile(r"(^|_)(hit|correct|is_correct)$")
# K-A16: `accuracy` counts as a correctness key only in records whose value is a per-item flag (bool, 0/1, or a column list), see _own_keys;
# header-only formats (csv/tsv/md/html) never pair it, because aggregate tables carry accuracy columns. Output synonyms/plurals added below.
ACCURACY_RE = re.compile(r"(^|_)accuracy$")
OUTPUT_RE = re.compile(r"(^|_)(chosen|pred|preds|predicted|prediction|predictions|letter|letters|choice|text|output|response|answer_letter|model_answer)$")
TEXT_GOLD_RE = re.compile(r"""["']\b(gold|gold_letter|gold_atoms|gold_answer|gold_option|gt_answer|correct_answer)["']\s*:""")
TEXT_DUMP_EXT = {".txt", ".log", ".out"}
YAML_EXT = {".yaml", ".yml"}
EMBED_MAX = 1 << 20  # serialized JSON inside a string value is decoded up to this length (R2)
PUBLIC_DOT_DIRS = {".github", ".streamlit"}  # an earlier change, generalised: every other root dot-directory is local tooling (settings, worktrees)
SKIP_DIRS = {".git", ".venv"}  # only these; caches (__pycache__, .ipynb_checkpoints, ...) are scanned like anything else


def local_tooling(rel: str) -> bool:
    """True for a path inside a root dot-directory that is not public (.github/, .streamlit/) and not a skipped root dir (.git/, .venv/)."""
    parts = rel.lower().split("/")
    return len(parts) > 1 and parts[0].startswith(".") and parts[0] not in PUBLIC_DOT_DIRS | SKIP_DIRS
MAX_READ = 64 << 20
MAX_JSONL_LINES = 20000
COMMON_WORDS = {"living", "nights", "friday", "stands"}  # words too common to act as a title alias


@dataclass
class Registry:
    owned: list[str] = field(default_factory=list)   # lower-case terms
    public: list[str] = field(default_factory=list)


def _slugs(s: str) -> set[str]:
    s = str(s).strip().lower()
    if not s:
        return set()
    out = {s, re.sub(r"\s+", "_", s), re.sub(r"\s+", "-", s), re.sub(r"\s+", "", s)}
    return {x for x in out if len(x) >= 4}


def _terms(entry: dict) -> set[str]:
    t = set()
    for k in ("id", "title"):
        if entry.get(k):
            t |= _slugs(entry[k])
    words = [w for w in re.split(r"[^a-z0-9]+", str(entry.get("title") or "").lower()) if len(w) >= 6 and w not in COMMON_WORDS]
    if words:
        t.add(max(words, key=len))  # title alias: the longest distinctive word of the title
    return t


def _read_yaml_films(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    data = yaml.safe_load(p.read_text()) or {}
    return [e for e in (data.get("films") or []) if isinstance(e, dict)]


def _alias_terms(a) -> set[str]:
    a = str(a).strip().lower()
    if len(a) < 3:
        return set()
    return {a, re.sub(r"\s+", "_", a), re.sub(r"\s+", "-", a), re.sub(r"\s+", "", a)}


def _read_aliases(p: Path) -> dict[str, list]:
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    al = data.get("aliases") or {}
    if not isinstance(al, dict):
        raise TypeError(f"{p}: `aliases` must map film_id -> list of aliases")
    return {str(k): (v if isinstance(v, list) else [v]) for k, v in al.items()}


def load_registry(roots: list[Path]) -> Registry:
    owned, public = set(), set()
    public_ids = set()
    aliases: dict[str, list] = {}
    for r in roots:
        for e in _read_yaml_films(Path(r) / "configs/films.yaml"):
            (public if e.get("public_domain") is True else owned).update(_terms(e))
            if e.get("public_domain") is True:
                public_ids.add(str(e.get("id")))
        for p in sorted((Path(r) / "configs").glob("films.local*.y*ml")) if (Path(r) / "configs").is_dir() else []:
            for e in _read_yaml_films(p):
                owned.update(_terms(e))  # overlay films are local only, whatever their flag says
                public_ids.discard(str(e.get("id")))
        for fid, al in _read_aliases(Path(r) / ALIASES_FILE).items():
            aliases.setdefault(fid, []).extend(al)
    for fid, al in aliases.items():  # aliases of a public film attribute media; any other id (owned or unknown) is treated as owned
        for a in al:
            (public if fid in public_ids else owned).update(_alias_terms(a))
    public -= owned
    return Registry(owned=sorted(owned, key=lambda s: (-len(s), s)), public=sorted(public, key=lambda s: (-len(s), s)))


def list_tree(tree: Path, git: bool = False, files_from: str | None = None) -> list[str]:
    tree = Path(tree)
    if files_from is not None:
        text = Path(files_from).read_text() if files_from != "-" else sys.stdin.read()
        return sorted({ln.strip() for ln in text.splitlines() if ln.strip()})
    if git:
        r = subprocess.run(["git", "-C", str(tree), "ls-files", "-z"], capture_output=True, check=True)
        return sorted(p for p in r.stdout.decode().split("\0") if p)
    out = []
    for d, dirs, files in os.walk(tree):  # followlinks=False: a symlinked dir is listed (SYMLINK hit), never entered
        keep = []
        at_root = Path(d) == tree
        for x in sorted(dirs):
            if Path(d, x).is_symlink():
                out.append(Path(d, x).relative_to(tree).as_posix())
            elif x in SKIP_DIRS:
                if not at_root:  # R3: a nested .git/.venv is listed (REVIEW), not entered and not silently skipped
                    out.append(Path(d, x).relative_to(tree).as_posix())
            else:
                keep.append(x)
        dirs[:] = keep
        for f in files:
            out.append(Path(d, f).relative_to(tree).as_posix())
    return sorted(out)


def _read(p: Path) -> tuple[str | None, str]:
    """(text, status): status is "text", "binary" (NUL bytes or not UTF-8), "large" (> MAX_READ) or "missing"."""
    try:
        if p.is_symlink() or not p.is_file():
            return None, "missing"
        if p.stat().st_size > MAX_READ:
            return None, "large"
        b = p.read_bytes()
    except OSError:
        return None, "missing"
    if b"\0" in b[:8192]:
        return None, "binary"
    try:
        return b.decode("utf-8"), "text"
    except UnicodeDecodeError:
        return None, "binary"


class ReviewListError(ValueError):
    pass


def _git_head_bytes(tree: Path, rel: str) -> bytes | None:
    """The committed bytes of `rel` at HEAD of the checkout `tree` (None if HEAD has no such path); a symlink at HEAD is refused."""
    if subprocess.run(["git", "-C", str(tree), "rev-parse", "--verify", "-q", "HEAD"], capture_output=True, check=False).returncode != 0:
        return None  # no commit yet: nothing is committed at HEAD
    ls = subprocess.run(["git", "-C", str(tree), "ls-tree", "HEAD", "--", rel], capture_output=True, text=True, check=False)
    if ls.returncode != 0:
        raise ReviewListError(f"git ls-tree failed for {rel}")
    if not ls.stdout.strip():
        return None
    if ls.stdout.split()[0] == "120000":
        raise ReviewListError(f"{rel} is a symlink at HEAD")
    r = subprocess.run(["git", "-C", str(tree), "show", f"HEAD:{rel}"], capture_output=True, check=False)
    if r.returncode != 0:
        raise ReviewListError(f"git show HEAD:{rel} failed")
    return r.stdout


def load_reviewed(tree: Path, git: bool = False) -> dict[str, dict]:
    """path -> entry from <tree>/docs/release_reviewed_files.yaml (absent file: {}). A malformed list raises ReviewListError (fail closed).
    git=True (K-A17): the list committed at HEAD (`git show HEAD:...`); a working-tree copy that differs from it (or exists only on one side)
    is refused, so an uncommitted edit can never clear anything."""
    p = Path(tree) / REVIEWED_FILE
    if p.is_symlink():
        raise ReviewListError(f"{REVIEWED_FILE} is a symlink")
    disk = p.read_bytes() if p.is_file() else None
    if git:
        head = _git_head_bytes(Path(tree), REVIEWED_FILE)
        if head != disk:
            raise ReviewListError(f"{REVIEWED_FILE}: the working-tree copy differs from HEAD (commit it or restore it)")
        raw = head
    else:
        raw = disk
    if raw is None:
        return {}
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as e:
        raise ReviewListError(f"{REVIEWED_FILE} unreadable ({type(e).__name__})") from None
    if not isinstance(data, dict) or set(data) != {"reviewed"} or not isinstance(data["reviewed"], list):
        raise ReviewListError(f"{REVIEWED_FILE}: expected exactly one key `reviewed` holding a list")
    out: dict[str, dict] = {}
    for i, e in enumerate(data["reviewed"]):
        if not isinstance(e, dict) or set(e) != set(REVIEWED_KEYS):
            raise ReviewListError(f"{REVIEWED_FILE}: entry {i} must have exactly the keys {list(REVIEWED_KEYS)}")
        e = {k: str(v) for k, v in e.items()}
        if not re.fullmatch(r"[0-9a-f]{64}", e["sha256"]):
            raise ReviewListError(f"{REVIEWED_FILE}: entry {i}: sha256 must be 64 lowercase hex characters")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", e["date"]) or not e["reviewed_by"].strip():
            raise ReviewListError(f"{REVIEWED_FILE}: entry {i}: date must be YYYY-MM-DD and reviewed_by non-empty")
        path = e["path"].strip()
        if not path or path.startswith("/") or ".." in Path(path).parts or path != Path(path).as_posix():
            raise ReviewListError(f"{REVIEWED_FILE}: entry {i}: path must be relative to the tree, without '..'")
        if path in out:
            raise ReviewListError(f"{REVIEWED_FILE}: duplicate path in entry {i}")
        out[path] = e
    return out


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


CURLY = str.maketrans({"\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"', "\u00ab": '"', "\u00bb": '"',
                       "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'"})
PDF_BARE_GOLD_RE = re.compile(r"\b(gold_[a-z0-9_]+|correct_answer|gt_answer)\b", re.IGNORECASE)  # K-A18: unquoted field names in PDF text


def pdf_gold(text: str) -> bool:
    """K-A18: PDF text carries a gold field: a quoted gold key after curly quotes are made straight, or a bare gold_* / correct_answer token."""
    t = text.translate(CURLY)
    return bool(TEXT_GOLD_RE.search(t) or PDF_BARE_GOLD_RE.search(t))


def _fitz():
    try:
        import fitz  # PyMuPDF: optional, not a project dependency
    except ImportError:
        return None
    return fitz


def pdf_text_method() -> str | None:
    """'pymupdf' (in-process, when importable), else 'pdfkit' (macOS /usr/bin/swift + scripts/release/pdf_text.swift), else None."""
    if _fitz() is not None:
        return "pymupdf"
    if Path(SWIFT).is_file() and PDF_SWIFT.is_file():
        return "pdfkit"
    return None


def pdf_text(p: Path) -> str | None:
    """Text layer of a PDF (method: pdf_text_method()); None when no method is available or the file cannot be read as a PDF."""
    method = pdf_text_method()
    if method is None:
        return None
    if method == "pymupdf":
        try:
            with _fitz().open(str(p)) as doc:
                return "\n".join(page.get_text() for page in doc)
        except Exception:  # noqa: BLE001 - any parse failure: no text, the file stays REVIEW
            return None
    try:
        r = subprocess.run([SWIFT, str(PDF_SWIFT), str(p)], capture_output=True, timeout=PDF_TIMEOUT, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    try:
        return r.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def norm_key(k) -> str:
    """goldLetter / Gold-Letter / ' gold letter ' / GOLD_LETTER -> gold_letter."""
    k = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(k).strip())
    return re.sub(r"[\s\-._]+", "_", k).strip("_").lower()  # R4: repeated separators collapse (gold__letter -> gold_letter)


def _keysets(obj, depth: int = 0):
    if depth > 8:
        return
    if isinstance(obj, dict):
        # K-A16: a record's keys are unioned with the keys of its direct child dicts, so {"out": {"letter"}, "eval": {"correct"}} pairs (one level only)
        yield _own_keys(obj).union(*(_own_keys(v) for v in obj.values() if isinstance(v, dict)))
        vals = obj.values()
    elif isinstance(obj, list):
        vals = obj[:5000]
    else:
        return
    for v in vals:
        if isinstance(v, str):
            v = _embedded_json(v)  # R2: a kept string may carry serialized JSON ('{"gold": ...}')
        if isinstance(v, (dict, list)):
            yield from _keysets(v, depth + 1)


def _own_keys(d: dict) -> set[str]:
    # K-A15: a correctness key holding a count or a rate (a number other than 0/1, e.g. "correct": 46) is an aggregate, not a per-item flag
    keys = {nk for k, v in d.items() if not (CORRECT_RE.search(nk := norm_key(k)) and _aggregate_number(v))}
    # K-A16: an `accuracy` key holding a per-item flag (bool, 0/1, or a column list) also counts as correctness: it is added as `<key>_is_correct`
    keys |= {f"{nk}_is_correct" for k, v in d.items() if ACCURACY_RE.search(nk := norm_key(k)) and _per_item_flag(v)}
    return keys


def _per_item_flag(v) -> bool:
    return isinstance(v, (list, bool)) or (isinstance(v, (int, float)) and v in (0, 1))


def _aggregate_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v not in (0, 1)


def _embedded_json(v: str):
    t = v.strip()
    if len(t) < 2 or len(t) > EMBED_MAX or t[0] not in "{[":
        return None
    try:
        return json.loads(t)
    except ValueError:
        return None


def _bench_keys(keys: set[str], bench_path: bool = False) -> bool:
    return (any(GOLD_RE.match(k) for k in keys) or bool(keys & ANSWER_WEAK and keys & QUESTION)
            or (any(CORRECT_RE.search(k) for k in keys) and any(OUTPUT_RE.search(k) for k in keys))  # model output + correctness = gold recoverable
            or (bench_path and any(LABEL_RE.search(k) for k in keys)))


def bench_fields(path: str, text: str) -> bool:
    ext = Path(path).suffix.lower()
    bp = any(b in path.lower() for b in BENCH_DIRS)
    if ext == ".jsonl":
        for i, ln in enumerate(text.splitlines()):
            if i >= MAX_JSONL_LINES:
                break
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if any(_bench_keys(k, bp) for k in _keysets(rec)):
                return True
        return False
    if ext == ".json" or ext in YAML_EXT:
        try:
            obj = json.loads(text) if ext == ".json" else yaml.safe_load(text)
        except (ValueError, yaml.YAMLError):
            return False
        return any(_bench_keys(k, bp) for k in _keysets(obj))
    if ext in (".csv", ".tsv"):
        first = text.splitlines()[0] if text else ""
        cols = {norm_key(c.strip().strip('"')) for c in re.split(r"[,\t]", first)}
        return _bench_keys(cols, bp) or ("answer" in cols and {"a", "b"} <= cols)
    if ext in (".html", ".htm"):  # R4: the first row of each <table> (th or td cells) as column names
        for tbl in re.findall(r"<table\b.*?</table>", text, flags=re.IGNORECASE | re.DOTALL):
            row = re.search(r"<tr\b.*?</tr>", tbl, flags=re.IGNORECASE | re.DOTALL)
            if row:
                cells = {norm_key(re.sub(r"<[^>]+>", "", c)) for c in re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", row.group(0), flags=re.IGNORECASE | re.DOTALL)}
                if _bench_keys(cells - {""}):
                    return True
        return False
    if ext == ".md":  # markdown tables: a header row (the line above |---|) whose cells are gold/answer field names
        lines = text.splitlines()
        for head, sep in itertools.pairwise(lines):
            if head.lstrip().startswith("|") and re.fullmatch(r"\s*\|?[\s:|-]*-[\s:|-]*", sep):
                cells = {norm_key(c.strip().strip("`*")) for c in head.strip().strip("|").split("|")}
                if _bench_keys(cells - {""}):  # no label-column clause: aggregate tables use "single-label" etc.
                    return True
        return False
    if ext in TEXT_DUMP_EXT:
        return bool(TEXT_GOLD_RE.search(text))
    return False


def _list_numbers(x, depth: int = 0) -> int:
    if depth > 8:
        return 0
    if isinstance(x, dict):
        return sum(_list_numbers(v, depth + 1) for v in x.values())
    if isinstance(x, list):
        return sum(1 if isinstance(v, (int, float)) and not isinstance(v, bool) else _list_numbers(v, depth + 1) for v in x)
    return 0


def json_weight_dump(path: str, text: str) -> bool:
    """A .json whose name says adapter/heads/codebook/palette/centroid and which holds numeric arrays (not a hash manifest or metrics file)."""
    if Path(path).suffix.lower() != ".json" or not ARTIFACT_WORDS.search(Path(path).name):
        return False
    try:
        return _list_numbers(json.loads(text)) >= JSON_WEIGHT_MIN
    except ValueError:
        return True  # unreadable keyword-named json: review it


def valid_scrubbed(text: str) -> bool:
    forbidden = set(HASH) | set(DROP) | set(MODEL_OUTPUTS) | set(RENAME_BOOL)  # a --keep-model-outputs copy is not certified (answer recovery)
    n = 0
    for ln in text.splitlines():
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
        except ValueError:
            return False
        if not isinstance(rec, dict) or rec.get("scrub") != SCRUB_VERSION or set(rec) & forbidden:
            return False
        n += 1
    return n > 0


def _mentions(s: str, terms: list[str]) -> bool:
    s = s.lower()
    return any(t in s for t in terms)


def check_path(rel: str, reg: Registry) -> list[str]:
    hits = []
    low = rel.lower()
    parts = low.split("/")
    name, ext = parts[-1], Path(low).suffix
    owned = _mentions(low, reg.owned)
    if owned:
        hits.append("OWNED_PATH")
    if ext in MEDIA_EXT and (owned or not _mentions(low, reg.public)):
        hits.append("MEDIA_UNATTRIBUTED")
    if ext in WEIGHT_EXT or ext in ARRAY_EXT or name == "adapter_config.json" or (ext in ARTIFACT_EXT and ARTIFACT_WORDS.search(name)):
        hits.append("WEIGHTS")
    if parts[0] in ("data", "models") and name != ".gitkeep":
        hits.append("DATA_DIR")
    if any(p in BENCH_DIRS for p in parts[:-1]) or (name in BENCH_NAMES and any(b in low for b in BENCH_DIRS)):
        hits.append("BENCH_FILE")
    if re.fullmatch(r"films\.local.*\.ya?ml", name):
        hits.append("LOCAL_OVERLAY")
    if local_tooling(low):
        hits.append("LOCAL_TOOLING")
    return hits


def check_tree(tree: Path, reg: Registry, git: bool = False, files_from: str | None = None, open_local: bool = False,
               files: list[str] | None = None, reviewed: dict[str, dict] | None = None,
               cleared: list[str] | None = None) -> list[tuple[str, str]]:
    """open_local=True also reads data/ and models/ files (for a bundle whose data/ is meant to ship; DATA_DIR is still reported).
    files: check exactly these paths relative to <tree> instead of listing it.
    reviewed: the allow-list (default: load_reviewed(tree, git)); it clears a REVIEW hit only when the file's current sha256 matches.
    cleared: if a list is given, the paths whose REVIEW hit the allow-list cleared are appended to it."""
    tree = Path(tree)
    reviewed = load_reviewed(tree, git=git) if reviewed is None else reviewed
    hits: list[tuple[str, str]] = []
    for rel in (sorted(set(files)) if files is not None else list_tree(tree, git, files_from)):
        rh = check_path(rel, reg)
        in_raw = rel.startswith(RAW_DIR) or (rel.startswith("results/") and PRED_RE.search(rel) is not None)  # any dump under results/
        local = (rel.split("/")[0] in ("data", "models") and not open_local) or local_tooling(rel)  # local worktrees hold data/
        if (tree / rel).is_symlink():
            rh.append("SYMLINK")
        elif (tree / rel).is_dir():  # only a nested .git/.venv is listed as a directory (R3)
            hits.append(("REVIEW", rel))
            continue
        # never open local data/models (benchmark data lives there); the path rule already blocks them
        text, status = (None, "skipped") if local else _read(tree / rel)
        ext = Path(rel.lower()).suffix
        if not local and "WEIGHTS" not in rh and (ext in REVIEW_EXT or status == "large" or (status == "binary" and ext not in KNOWN_BINARY_EXT)):
            rh.append("REVIEW")
        if in_raw and not (rel.endswith(SUFFIX) and text is not None and valid_scrubbed(text)) and not rel.endswith("/scrub_manifest.json"):
            rh.append("RAW_PREDICTIONS")
        if not local and ext == ".pdf" and status in ("text", "binary"):
            pdf = pdf_text(tree / rel)  # PDF text layer: the same owned-term and quoted-gold-key rules as a .txt dump; REVIEW stays
            if pdf:
                if _mentions(pdf, reg.owned):
                    rh += ["OWNED_CONTENT"]
                if pdf_gold(pdf):
                    rh += ["BENCH_FIELDS"]
        if text is not None:
            if _mentions(text, reg.owned):
                rh.append("OWNED_CONTENT")
            if bench_fields(rel, text):
                rh.append("BENCH_FIELDS")
            if "WEIGHTS" not in rh and json_weight_dump(rel, text):
                rh.append("WEIGHTS")
        if "REVIEW" in rh and rel in reviewed:
            if not (tree / rel).is_symlink() and (tree / rel).is_file() and file_sha256(tree / rel) == reviewed[rel]["sha256"]:
                rh = [r for r in rh if r not in CLEARABLE]
                if cleared is not None:
                    cleared.append(rel)
            else:
                rh.append("STALE_REVIEW")
        hits += [(r, rel) for r in rh]
    return hits


def redact(s: str, reg: Registry) -> str:
    for t in reg.owned:
        s = re.sub(re.escape(t), "<OWNED>", s, flags=re.IGNORECASE)
    return s
