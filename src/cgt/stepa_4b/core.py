"""Pure helpers for Step A-4B (docs/plans/stepa_4b.md): the dedup_v2 questions with Step A's film halves, the view work list,
budget projections and the distillation gate. No model, no GPU."""
import json
from collections.abc import Callable
from pathlib import Path

from cgt.eval.val_dedup import dedup_v2_indices
from cgt.p3s_stepa.split import ATTRS, SPLIT_SEED, dedup_first, split_films_half
from cgt.p3s_stepa.views import applies
from cgt.stepa_4b import config as K

# exit codes of the views child that are NOT crashes (no resume): 0 ok, 3-20 fail-closed guards (cgt.p3b_vlm.guards)
FAIL_CLOSED = tuple(range(3, 21))
TERMINATED = -999


def v2_split(rows: list[dict], film_of: Callable[[set[str]], dict[str, str]], seed: int = SPLIT_SEED) -> dict:
    """dedup_v2 questions (line order) with `line`, `key` = "<line>:<id>" (Step A's key format), `film`, `half`.
    The halves are Step A's: split_films_half over the films of the dedup_first rows with the same seed; every dedup_v2 row takes
    the half of its film (the 60 rows dedup_v2 adds share images, hence films, with dedup_first rows). Raises if a film is missing."""
    old = dedup_first(rows)
    keep = dedup_v2_indices(rows)
    fmap = film_of({rows[n]["img_id"] for n in keep} | {q["img_id"] for q in old})
    if any(not fmap.get(r["img_id"]) for r in rows):
        raise ValueError("an image has no film key")
    cal, sel = split_films_half([fmap[q["img_id"]] for q in old], seed)
    qs = []
    for n in keep:
        f = fmap[rows[n]["img_id"]]
        if f not in cal | sel:
            raise ValueError(f"line {n}: film {f!r} not in the Step A film split")
        qs.append({**rows[n], "line": n, "key": f"{n}:{rows[n]['id']}", "film": f, "half": "cal" if f in cal else "sel"})
    old_lines = {q["line"] for q in old}
    info = {"val_lines": len(rows), "dedup_first": len(old), "dedup_v2": len(qs), "films": len(cal | sel), "seed": seed}
    for h, fs in (("cal", cal), ("sel", sel)):
        hq = [q for q in qs if q["half"] == h]
        info[h] = {"films": len(fs), "questions": len(hq), "added_v2_rows": sum(q["line"] not in old_lines for q in hq),
                   "per_attribute": {a: sum(q["attribute"] == a for q in hq) for a in ATTRS}}
    return {"questions": qs, "info": info}


def todo(qs: list[dict], view: str, done: set[str]) -> list[dict]:
    """Questions the view applies to and that have no cached output yet (resume unit = one question, flushed per line)."""
    return [q for q in qs if applies(view, q["attribute"]) and q["key"] not in done]


def done_keys(path: Path) -> set[str]:
    """Keys already written to a view file; a torn last line (crash mid-write) is ignored and later rewritten."""
    if not Path(path).exists():
        return set()
    out = set()
    for line in Path(path).read_text().splitlines():
        try:
            out.add(json.loads(line)["key"])
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def projection_h(n_todo: int, sec_per_q: float, load_h: float = K.LOAD_H) -> float:
    return n_todo * sec_per_q / 3600 + load_h


def rerun_after(rc: int, resumes_used: int, max_resumes: int = K.MAX_CRASH_RESUMES) -> bool:
    """A crash (any exit other than 0, a fail-closed guard, a termination at the stop time or a lock timeout) is resumed while
    fewer than `max_resumes` crash resumes have been used in total."""
    return rc not in (0, TERMINATED, -998, *FAIL_CLOSED) and resumes_used < max_resumes


def distill_gate(decision: dict) -> dict:
    """a recorded decision / plan section 6: distillation is PROPOSED (its own plan, later) only if the rule adopts a winner other than C0
    (gain >= 0.5 points on the selection half AND the film-clustered 95% lower bound > 0)."""
    ok = decision["winner"] != "C0"
    return {"propose_distillation": ok,
            "reason": f"{decision['winner']} adopted by the pre-registered rule" if ok else "rule keeps C0: no distillation proposed"}


def shotqa_film_of(img_ids: set[str]) -> dict[str, str]:
    """{img_id: film key} from ShotQA meta.jsonl titles (the Step A join, cgt.p3_static_attrs.data.image_films), with the ShotQA root
    of cgt.stepc.config (main checkout) instead of the worktree-relative one."""
    from cgt.p3_static_attrs.data import image_films
    from cgt.stepc import config as SC
    return image_films(SC.SHOTQA_ROOT / "meta.jsonl", img_ids)[0]


def load_val_rows() -> list[dict]:
    """val.jsonl (sha256-checked, Step C path)."""
    from cgt.p3b_vlm import guards
    from cgt.stepc import config as SC
    if guards.sha256_file(SC.VAL_JSONL) != SC.VAL_SHA:
        raise guards.DataRootViolation("val.jsonl does not match its recorded sha256")
    return [json.loads(x) for x in SC.VAL_JSONL.read_text().splitlines()]
