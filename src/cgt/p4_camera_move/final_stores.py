"""Sealed, EVALUATION-ONLY stores of the two Phase 4 test sets for the final evaluations (docs/phase4_interface.md section 10). Both follow the CameraBench store contract
(`camerabench.CameraBenchTestStore`): the labels are reachable ONLY through `labels(final_evaluation=True, model_tag=..., adapter_sha256=..., config_hash=..., code_sha=...)`, which
(1) refuses unless the model tag is registered in the COMMITTED freeze record with matching hashes, (2) refuses a second final evaluation of the same (dataset, tag), (3) appends the
evaluation to the shared ledger `results/phase4/test_looks.jsonl` (flushed and fsynced) BEFORE the label file is opened. Everything that does not need a label is available without one:
- `ShotBenchMovementStore.questions()` parses the 464 camera-movement questions of `test.tsv` with the answer column EXCLUDED at the parser (`usecols`): the question texts, the options,
  the video paths and the filename-derived source ids;
- `CameraBenchFinalStore.clips()` lists the GIF names (names only).
`labels_after_commit` serves a author-acknowledged RESUME of an evaluation whose ledger line is already committed (docs/phase4_interface.md section 10.5): it appends nothing and
requires the ledger line to exist. `events` records ("questions", n) / ("ledger", dataset, tag) / ("labels", dataset, tag) in order so that a test can prove the order."""
import ast
import contextlib
import fcntl
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from cgt.p4_camera_move.camerabench import (
    DATASET as CAMERABENCH,
)
from cgt.p4_camera_move.camerabench import (
    FREEZE,
    LEDGER,
    ROOT,
    CameraBenchAlreadyEvaluated,
    CameraBenchLocked,
    CameraBenchTestStore,
    committed_registry,
    require_registered,
)
from cgt.p4_camera_move.labels import parse_move_label, source_id

SHOTBENCH = "shotbench_movement"
SB = ROOT / "data/external/shotbench"
QUESTION_COLUMNS = ["index", "type", "path", "question", "options", "category"]  # NEVER `answer`
CATEGORY = "camera movement"
QUESTION_TEXT = "What's the camera movement in this movie shot?"
EXPECTED_N = 464


def ledger_lines(path: Path) -> list[dict]:
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def ledger_contains(path: Path, dataset: str, tag: str) -> bool:
    """True iff the ledger holds a final evaluation of `tag` on `dataset`; a read failure raises (the caller maps that to UNKNOWN, never to 'present')."""
    return any(r.get("dataset") == dataset and r.get("model_tag") == tag for r in ledger_lines(path))


def ledger_append(path: Path, dataset: str, tag: str, *, adapter_sha256: str | None, config_hash: str | None, code_sha: str) -> None:
    """Append one final-evaluation line (same fields as the CameraBench store): write, flush, `os.fsync` (and an fsync of the directory when the file is new) BEFORE the call returns; any failure raises,
    so no label file is ever opened after a failed flush or fsync. The ONE shared durable append of both stores (Amendment 12 item 1)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    rec = {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), "dataset": dataset, "model_tag": tag, "config_hash": config_hash or "", "adapter_sha256": adapter_sha256 or "",
           "code_sha": code_sha, "n_final_evaluations_logged_so_far": 1 + len(ledger_lines(p))}
    with open(p, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    if new:
        dfd = os.open(p.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


@contextlib.contextmanager
def ledger_lock(path: Path, suffix: str = ".append.lock", blocking: bool = True):
    """Exclusive `flock` on `<ledger><suffix>` (released by the OS when the process ends, however it ends). `blocking=False` raises BlockingIOError when another process holds it."""
    lock = Path(str(path) + suffix)
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield
    finally:
        os.close(fd)


def reserve_once(path: Path, dataset: str, tag: str, check, *, adapter_sha256: str | None, config_hash: str | None, code_sha: str) -> None:
    """The check-then-append of the once-only ledger as ONE critical section across processes and both datasets (Amendment 12 item 2a): `check()` is evaluated under the lock and must return True if
    the evaluation is already recorded (the caller's already-evaluated exception is raised by `check` itself)."""
    with ledger_lock(path):
        check()
        ledger_append(path, dataset, tag, adapter_sha256=adapter_sha256, config_hash=config_hash, code_sha=code_sha)


class ShotBenchMovementStore:
    def __init__(self, tsv: Path = SB / "test.tsv", ledger: Path = LEDGER, freeze_path: Path = FREEZE, repo_root: Path = ROOT, git_ref: str = "HEAD", events: list | None = None,
                 expected_n: int = EXPECTED_N):
        self.tsv, self.ledger, self.freeze_path, self.repo_root, self.git_ref = Path(tsv), Path(ledger), Path(freeze_path), Path(repo_root), git_ref
        self.expected_n = expected_n  # 464 for the real file; a synthetic file in the self-checks
        self.events: list = events if events is not None else []

    def questions(self) -> list[dict]:
        """The camera-movement questions in file order, WITHOUT the answers: {id: 'sb<index>', clip: file name, path: 'video/<name>', source, options{A..D}, option_atoms{letter: schema atoms}}.
        Fails closed (ValueError) on any row that is not a four-option question with the registered question text, on a duplicate id or file, and unless exactly EXPECTED_N rows are found."""
        df = pd.read_csv(self.tsv, sep="\t", usecols=QUESTION_COLUMNS, dtype=str)
        df = df[(df.category == CATEGORY) & df.type.str.contains("video")]
        rows, seen_ids, seen_files = [], set(), set()
        for r in df.itertuples():
            options = json.loads(r.options)
            path = ast.literal_eval(r.path)[0]
            if r.question != QUESTION_TEXT or sorted(options) != list("ABCD") or not path.startswith("video/") or len(ast.literal_eval(r.path)) != 1:
                raise ValueError(f"row {r.index}: not a standard four-option camera-movement question")
            name = path.removeprefix("video/")
            if r.index in seen_ids or name in seen_files:
                raise ValueError(f"row {r.index}: duplicate id or video file")
            seen_ids.add(r.index)
            seen_files.add(name)
            rows.append({"id": f"sb{r.index}", "clip": name, "path": path, "source": source_id(name), "options": options,
                         "option_atoms": {k: parse_move_label(v).atoms for k, v in options.items()}})
        if len(rows) != self.expected_n:
            raise ValueError(f"expected {self.expected_n} camera-movement questions, found {len(rows)}")
        self.events.append(("questions", len(rows)))
        return rows

    def _check(self, final_evaluation, model_tag, adapter_sha256, config_hash) -> None:
        if not final_evaluation or not model_tag:
            raise CameraBenchLocked("ShotBench movement answers are sealed: pass final_evaluation=True and the frozen model_tag")
        require_registered(model_tag, committed_registry(self.freeze_path, repo_root=self.repo_root, git_ref=self.git_ref), adapter_sha256=adapter_sha256, config_hash=config_hash)

    def _answers(self) -> dict[str, str]:
        df = pd.read_csv(self.tsv, sep="\t", usecols=["index", "category", "type", "answer"], dtype=str)
        df = df[(df.category == CATEGORY) & df.type.str.contains("video")]
        out = {f"sb{r.index}": r.answer for r in df.itertuples()}
        if any(a not in list("ABCD") for a in out.values()):
            raise ValueError("an answer is not a letter A-D")
        self.events.append(("labels", SHOTBENCH, ""))
        return out

    def labels(self, *, final_evaluation: bool = False, model_tag: str | None = None, adapter_sha256: str | None = None, config_hash: str | None = None, code_sha: str = "") -> dict[str, str]:
        """{question id: gold letter}; the ledger line is written (and fsynced) FIRST."""
        self._check(final_evaluation, model_tag, adapter_sha256, config_hash)
        def check():
            if ledger_contains(self.ledger, SHOTBENCH, model_tag):
                raise CameraBenchAlreadyEvaluated(f"{SHOTBENCH} was already evaluated for model '{model_tag}'")
        reserve_once(self.ledger, SHOTBENCH, model_tag, check, adapter_sha256=adapter_sha256, config_hash=config_hash, code_sha=code_sha)
        self.events.append(("ledger", SHOTBENCH, model_tag))
        return self._answers()

    def labels_after_commit(self, *, model_tag: str, adapter_sha256: str | None = None, config_hash: str | None = None) -> dict[str, str]:
        """RESUME of a author-acknowledged evaluation: the ledger line must already exist (nothing is appended)."""
        self._check(True, model_tag, adapter_sha256, config_hash)
        if not ledger_contains(self.ledger, SHOTBENCH, model_tag):
            raise CameraBenchLocked("resume refused: there is no committed ledger line for this evaluation")
        return self._answers()


class CameraBenchFinalStore(CameraBenchTestStore):
    """The sealed CameraBench public test store plus the label-free GIF list and the resume path; `labels()` is the inherited ledger-first reader (guarded by the committed registry)."""

    def __init__(self, *args, events: list | None = None, **kw):
        super().__init__(*args, **kw)
        self.events: list = events if events is not None else []

    def clips(self) -> list[str]:
        """GIF file names, sorted (names only: no label is read)."""
        names = [p.name for p in self.video_files()]
        self.events.append(("clips", len(names)))
        return names

    def labels(self, *, final_evaluation: bool = False, model_tag: str | None = None, adapter_sha256: str | None = None, config_hash: str | None = None, code_sha: str = "") -> list[dict]:
        """The inherited contract with the DURABLE, once-only ledger of Amendment 12: the registry check, then check + append (flushed and fsynced) under one lock, and only THEN the label file."""
        if not final_evaluation or not model_tag:
            raise CameraBenchLocked("CameraBench public test labels are sealed: pass final_evaluation=True and the frozen model_tag")
        require_registered(model_tag, committed_registry(self.freeze_path, repo_root=self.repo_root, git_ref=self.git_ref), adapter_sha256=adapter_sha256, config_hash=config_hash)

        def check():
            if self._evaluated(CAMERABENCH, model_tag):
                raise CameraBenchAlreadyEvaluated(f"{CAMERABENCH} was already evaluated for model '{model_tag}'")
        reserve_once(self.ledger, CAMERABENCH, model_tag, check, adapter_sha256=adapter_sha256, config_hash=config_hash, code_sha=code_sha)
        self.events.append(("ledger", CAMERABENCH, model_tag))
        rows = [json.loads(x) for x in (self.root / "test.jsonl").read_text().splitlines() if x.strip()]
        for r in rows:
            r["stem"] = Path(r["path"]).stem
        self.events.append(("labels", CAMERABENCH, model_tag))
        return rows

    def labels_after_commit(self, *, model_tag: str, adapter_sha256: str | None = None, config_hash: str | None = None) -> list[dict]:
        require_registered(model_tag, committed_registry(self.freeze_path, repo_root=self.repo_root, git_ref=self.git_ref), adapter_sha256=adapter_sha256, config_hash=config_hash)
        if not self._evaluated(CAMERABENCH, model_tag):
            raise CameraBenchLocked("resume refused: there is no committed ledger line for this evaluation")
        rows = [json.loads(x) for x in (self.root / "test.jsonl").read_text().splitlines() if x.strip()]
        for r in rows:
            r["stem"] = Path(r["path"]).stem
        self.events.append(("labels", CAMERABENCH, model_tag))
        return rows


def labels_by_stem(rows: list[dict]) -> dict[str, list[str]]:
    """{GIF stem: CameraBench tags}; fails closed on a duplicate stem."""
    out: dict[str, list[str]] = {}
    for r in rows:
        if r["stem"] in out:
            raise ValueError(f"duplicate label row for {r['stem']}")
        out[r["stem"]] = list(r["labels"])
    return out
