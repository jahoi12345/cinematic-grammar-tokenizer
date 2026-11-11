"""ShotQA validation dedup v2 (docs/plans/val_dedup_v2.md). For FUTURE experiments; a recorded decision rule (`dedup_first` in
cgt.p3s_stepa.split and cgt.stepb.data) stays unchanged so past results reproduce.

val.jsonl repeats 64 question ids, but only 4 of those pairs are the same question; the other 60 differ in attribute or option
set. dedup_v2 keys a question by (id, attribute, sorted option texts) and keeps the first occurrence of each key in line order:
5,826 -> 5,822 on the real file. Option ORDER is ignored, so a question asked twice with reordered letters counts once."""
from collections.abc import Sequence

V2_COUNT_REAL_VAL = 5822   # dedup_v2 size of data/cache/p3b/val.jsonl (sha256 409e87d5...)


def dedup_key(row: dict) -> tuple:
    return (row["id"], row["attribute"], tuple(sorted(row["options"].values())))


def dedup_v2_indices(rows: Sequence[dict]) -> list[int]:
    """Line indices (in order) of the rows dedup_v2 keeps."""
    seen, out = set(), []
    for n, r in enumerate(rows):
        k = dedup_key(r)
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out


def dedup_v2(rows: Sequence[dict]) -> list[dict]:
    """The rows (unchanged objects, line order) whose (id, attribute, sorted option set) key occurs for the first time."""
    return [rows[i] for i in dedup_v2_indices(rows)]
