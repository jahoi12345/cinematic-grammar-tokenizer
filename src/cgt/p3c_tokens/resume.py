"""Resume bookkeeping of the method-selection scoring (scripts/p3c_select_method.py). Pure python.
Full-vocabulary method records depend only on (image, attribute) and are keyed (sample id, attribute): sft.json asks some images twice under
the same id. Native 4-option records depend on the question's options, so they are keyed by the stable position in the registered subset
(`pos`) and carry a hash of the question text (`q`)."""
import hashlib
import json
from pathlib import Path


def question_hash(user_text: str) -> str:
    return hashlib.sha1(user_text.encode()).hexdigest()[:12]


def repair_jsonl(path: Path) -> list[dict]:
    """Truncate `path` after its last complete, parseable record (a kill can leave one torn line) and return the kept records."""
    path = Path(path)
    if not path.exists():
        return []
    raw, keep, out = path.read_bytes(), 0, []
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        try:
            out.append(json.loads(line))
        except ValueError:
            break
        keep += len(line)
    if keep != len(raw):
        with open(path, "r+b") as f:
            f.truncate(keep)
    return out


def load_done(path: Path) -> dict[tuple[str, str], dict]:
    """(id, attribute) -> record, for the full-vocabulary methods."""
    return {(r["id"], r["attr"]): r for r in repair_jsonl(path)}


def native_done_positions(path: Path, samples: list[dict]) -> set[int]:
    """Subset positions whose native record exists. Records with `pos` count for that position; legacy records (no `pos`) count only for an
    (id, attribute) pair that occurs once in the subset, because a repeated pair has several questions with different option orders."""
    from collections import Counter
    pair_count = Counter((s["id"], s["attribute"]) for s in samples)
    where = {(s["id"], s["attribute"]): j for j, s in enumerate(samples) if pair_count[(s["id"], s["attribute"])] == 1}
    done = set()
    for r in repair_jsonl(path):
        if "pos" in r:
            done.add(int(r["pos"]))
        elif (r["id"], r["attr"]) in where:
            done.add(where[(r["id"], r["attr"])])
    return done
