"""The sealed protocol of the Phase 4 final evaluations: the Phase 3B `SealedEvaluation` (crash file, IN_PROGRESS marker, CONSUMED and UNKNOWN_COMMIT states, generic reason codes, exit codes) with
one extension, because here the ledger entry is written BEFORE the labels are read and the label-dependent work happens after it:
    pre  (label-free: inputs, model, scores that need no label)  -> any failure: crash file, ledger NOT consumed, retry needs the operator's CRASH_ACK
    commit  (ledger line, fsynced, then the label file is opened) -> signals blocked; a failed commit is settled by reading the ledger (NOT_COMMITTED / CONFIRMED / UNKNOWN)
    produce (label-dependent: question generation, scoring of label-dependent prompts, metrics, result texts) and the writes of the result files -> after the commit, a failure leaves
            `<key>.CONSUMED.json` (exit 17) and publishes nothing that was not written; a retry is refused (exit 14).
RESUME: a CONSUMED evaluation in which no result file exists may be re-run once per `<key>.RESUME_ACK` that the operator creates after reading the CONSUMED record; the
re-run appends nothing to the ledger (the line must exist and carry the same code commit), reads the labels again through the store's `labels_after_commit` and archives the old record.
Nothing score-bearing is printed or written before the commit; progress logs carry counts only."""
import json
from pathlib import Path

from cgt.p3b_vlm import guards, sealed
from cgt.p3b_vlm.runstatus import now_iso


def resume_ack_path(raw_dir: Path, key: str) -> Path:
    return Path(raw_dir) / f"{key}.RESUME_ACK"


def resumes_log_path(raw_dir: Path, key: str) -> Path:
    return Path(raw_dir) / f"{key}.resumes.jsonl"


def prepare_resume(raw_dir: Path, key: str, result_paths: list[Path], ledger_line: dict | None, code_commit: str) -> dict:
    """Authorise ONE resume of a consumed evaluation. Refuses (a) when nothing is consumed (LedgerUsed 14: use a normal run), (b) while a result file of this evaluation exists (LedgerUsed 14),
    (c) when the ledger line is missing or was written by another code commit (DirtyTree 10), (d) without the operator's RESUME_ACK (CrashUnacknowledged 16). On success the CONSUMED record is
    archived, the ACK is consumed and the resume is logged; returns the archived record."""
    raw = Path(raw_dir)
    consumed = sealed.consumed_path(raw, key)
    if not consumed.exists():
        raise guards.LedgerUsed(f"nothing to resume for '{key}': there is no {consumed.name}")
    if any(Path(p).exists() for p in result_paths):
        raise guards.LedgerUsed(f"'{key}' already has result files: it is complete and cannot be resumed")
    if ledger_line is None or ledger_line.get("code_sha") != code_commit:
        raise guards.DirtyTree("resume refused: the ledger line is missing or was written by another code commit")
    ack = resume_ack_path(raw, key)
    if not ack.exists():
        raise guards.CrashUnacknowledged(f"'{key}' is CONSUMED without results; tell the operator, who must read {consumed.name} and create {ack.name} to allow one resume")
    rec = json.loads(consumed.read_text())
    stamp = now_iso().split(" ")[0].replace(":", "-")
    consumed.rename(consumed.with_name(f"{consumed.name}.resumed.{stamp}"))
    ack.unlink()
    with open(resumes_log_path(raw, key), "a") as f:
        f.write(json.dumps({"time": now_iso(), "code_commit": code_commit, "archived_consumed_record": {k: rec.get(k) for k in ("exception_class", "reason_code", "questions_scored", "total_questions")}}) + "\n")
    return rec


class FinalSealed(sealed.SealedEvaluation):
    def seal_after_labels(self, commit, ledger_contains, produce) -> None:
        """`commit()` records the ledger entry and opens the labels (stashing them for `produce`); `produce() -> (outputs {path: text}, echo)` computes everything that needs a label. Order: commit
        (SIGTERM/SIGINT blocked) -> produce -> atomic writes of the outputs -> echo. The three ledger states of the Phase 3B protocol apply: CONFIRMED (the commit returned or a successful read of
        the ledger shows the tag), NOT_COMMITTED (crash file, tag not consumed, exit 15) and UNKNOWN (the read failed: nothing is published, `<key>.UNKNOWN_COMMIT.json`, exit 19)."""
        written, cause, outputs, produced = [], None, {}, False
        try:
            with sealed._signals_blocked():
                commit()
                self.committed = True
            outputs, echo = produce()
            produced = True
            for path, text in outputs.items():
                sealed._atomic_write(Path(path), text)
                written.append(str(path))
        except BaseException as e:  # noqa: BLE001 - any failure is classified, never propagated raw
            cause = e
        if cause is None:
            self.finished = True
            sealed.in_progress_path(self.raw_dir, self.tag).unlink(missing_ok=True)
            self.log(echo)
            return
        state = "CONFIRMED" if self.committed else self._read_commitment(ledger_contains)
        if state == "NOT_COMMITTED":
            self.record_crash(cause)
            raise guards.EvaluationCrashed("could not record the ledger entry; no labels were read, no scores written") from cause
        if state == "UNKNOWN":
            self._write_unknown(cause)
            raise guards.CommitUnknown("could not establish whether the ledger entry was recorded; nothing was published") from cause
        self.committed = True
        failed: list[str] = []
        if produced:
            with sealed._signals_blocked():
                for path, text in outputs.items():
                    if str(path) not in written:
                        try:
                            sealed._atomic_write(Path(path), text)
                            written.append(str(path))
                        except BaseException:  # noqa: BLE001 - reported as not written
                            failed.append(str(path))
        else:
            failed = ["<results were not computed>"]
        self.finished = True
        self._write_consumed(cause, [Path(w).name for w in written], [Path(f).name for f in failed])
        if produced:
            self.log(echo)  # allowed now: the ledger entry is CONFIRMED recorded and the results exist
            by_name = {str(k): v for k, v in outputs.items()}
            for f in failed:
                print(f"----- {f} -----\n{by_name[f]}")
        raise guards.PostLedgerWriteFailed(f"failure after the ledger commit ({sealed.safe_text(cause)}); evaluation CONSUMED; {len(failed)} result files not written; "
                                           f"see {sealed.consumed_path(self.raw_dir, self.tag).name}") from cause
