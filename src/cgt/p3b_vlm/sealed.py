"""Sealed ShotBench evaluation.
Until the FULL question set is scored AND the ledger entry is recorded, nothing may print or write an accuracy, a per-question outcome or a
partial score: only progress COUNTS are logged. Exception MESSAGES are never kept (they can carry answers, logits, counts): a crash writes
`<tag>.CRASH.json` with the exception class, an allowlisted generic reason code, source frame locations, progress counts and provenance, and does
not consume the ledger tag; a retry is refused (exit 16) until the operator creates `<tag>.CRASH_ACK`.
Order at the end: the ledger entry is recorded FIRST (atomically, signals blocked), then the result files, then the echo. Any failure after the
ledger commit leaves a CONSUMED state (`<tag>.CONSUMED.json`, exit 17): the completed results are still written, a new evaluation is refused."""
import contextlib
import json
import os
import signal
import traceback
from pathlib import Path

from cgt.p3b_vlm.guards import (
    AdapterHashError,
    CommitUnknown,
    CrashUnacknowledged,
    DataRootViolation,
    DirtyTree,
    EvaluationCrashed,
    FailClosed,
    LeakProbeFailed,
    LedgerUsed,
    NotAuthorized,
    PostLedgerWriteFailed,
    SameStemImages,
    TagNotBound,
    TokenLengthViolation,
    UnreadableImage,
)
from cgt.p3b_vlm.runstatus import now_iso


class SignalTermination(RuntimeError):
    """Raised by the SIGTERM handler. Its message is never recorded."""


def _known_table() -> dict[type, tuple[str, str]]:
    """Exact class -> (class name to record, generic reason code), all FIXED strings. Lookup is by `type(exc)`, never by ancestry or by any
    attribute of the exception (so a subclass, its name and its `exit_code` are never reflected into a record)."""
    internal = (RuntimeError, ValueError, KeyError, IndexError, TypeError, AssertionError, AttributeError, ImportError, ZeroDivisionError,
                OverflowError, EOFError, StopIteration, SystemExit, UnicodeDecodeError, json.JSONDecodeError)
    io = (OSError, FileNotFoundError, PermissionError, BrokenPipeError, TimeoutError)
    t: dict[type, tuple[str, str]] = {c: (c.__name__, "internal_error") for c in internal}
    t.update({c: (c.__name__, "io_error") for c in io})
    t[MemoryError] = ("MemoryError", "out_of_memory")
    t[KeyboardInterrupt] = ("KeyboardInterrupt", "interrupted")
    t[SignalTermination] = ("SignalTermination", "terminated_by_signal")
    t.update({FailClosed: ("FailClosed", "guard_failed_exit_1"), LeakProbeFailed: ("LeakProbeFailed", "guard_failed_exit_4"),
              DataRootViolation: ("DataRootViolation", "guard_failed_exit_5"), UnreadableImage: ("UnreadableImage", "guard_failed_exit_6"),
              AdapterHashError: ("AdapterHashError", "guard_failed_exit_7"), TokenLengthViolation: ("TokenLengthViolation", "guard_failed_exit_8"),
              DirtyTree: ("DirtyTree", "guard_failed_exit_10"), NotAuthorized: ("NotAuthorized", "guard_failed_exit_13"),
              LedgerUsed: ("LedgerUsed", "guard_failed_exit_14"), EvaluationCrashed: ("EvaluationCrashed", "guard_failed_exit_15"),
              CrashUnacknowledged: ("CrashUnacknowledged", "guard_failed_exit_16"), PostLedgerWriteFailed: ("PostLedgerWriteFailed", "guard_failed_exit_17"),
              TagNotBound: ("TagNotBound", "guard_failed_exit_18"), CommitUnknown: ("CommitUnknown", "guard_failed_exit_19"),
              SameStemImages: ("SameStemImages", "guard_failed_exit_20")})
    return t


_KNOWN = _known_table()
_FIXED_EXIT = {"guard_failed_exit_" + str(n): n for n in (1, 4, 5, 6, 7, 8, 10, 13, 14, 15, 16, 17, 18, 19, 20)}


def safe_reason(exc: BaseException) -> dict:
    """The only representation of a failure that may be logged or written before the seal: {exception_class, reason_code}. The message is dropped.
    Only the exact known classes above are named; ANY other class, including a subclass of a known one, is `OtherException` /
    `unclassified_exception`."""
    name, code = _KNOWN.get(type(exc), ("OtherException", "unclassified_exception"))
    return {"exception_class": name, "reason_code": code}


def safe_exit_code(exc: BaseException) -> int:
    """Process exit code for a pre-seal failure: the documented code of an exact known guard class, else 15. Never `exc.exit_code`."""
    return _FIXED_EXIT.get(safe_reason(exc)["reason_code"], 15)


def safe_frames(exc: BaseException, n: int = 8) -> list[str]:
    """Source locations only (file name, line, function): no variable values, no message."""
    return [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in traceback.extract_tb(exc.__traceback__)][-n:]


def safe_text(exc: BaseException) -> str:
    r = safe_reason(exc)
    return f"{r['exception_class']} ({r['reason_code']})"


def crash_path(raw_dir: Path, tag: str) -> Path:
    return Path(raw_dir) / f"{tag}.CRASH.json"


def in_progress_path(raw_dir: Path, tag: str) -> Path:
    return Path(raw_dir) / f"{tag}.IN_PROGRESS.json"


def ack_path(raw_dir: Path, tag: str) -> Path:
    return Path(raw_dir) / f"{tag}.CRASH_ACK"


def consumed_path(raw_dir: Path, tag: str) -> Path:
    return Path(raw_dir) / f"{tag}.CONSUMED.json"


def unknown_path(raw_dir: Path, tag: str) -> Path:
    return Path(raw_dir) / f"{tag}.UNKNOWN_COMMIT.json"


def unknown_ack_path(raw_dir: Path, tag: str) -> Path:
    return Path(raw_dir) / f"{tag}.UNKNOWN_COMMIT_ACK"


def _atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".tmp_" + path.name)
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextlib.contextmanager
def _signals_blocked():
    """SIGTERM/SIGINT stay pending (and are delivered after the block) so the ledger commit and the consumed-state write cannot be interrupted."""
    sigs = {signal.SIGTERM, signal.SIGINT}
    old = signal.pthread_sigmask(signal.SIG_BLOCK, sigs)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


def require_not_consumed(raw_dir: Path, tag: str) -> None:
    """A tag whose ledger entry was committed (CONSUMED.json exists) can never be evaluated again, even if the ledger file were lost (exit 14)."""
    if consumed_path(raw_dir, tag).exists():
        raise LedgerUsed(f"tag '{tag}' is CONSUMED ({consumed_path(raw_dir, tag).name}): its ledger entry was committed; a new evaluation is not allowed")


def require_no_unacknowledged_crash(raw_dir: Path, tag: str) -> None:
    """Refuse (CrashUnacknowledged, exit 16) if a crash file or a leftover IN_PROGRESS marker (a hard-killed run) exists without a
    `<tag>.CRASH_ACK` file. With the ACK present, the old files are archived (renamed `*.acknowledged.<time>`) and the ACK is consumed."""
    raw_dir = Path(raw_dir)
    unk = [p for p in (unknown_path(raw_dir, tag),) if p.exists()]
    if unk:  # commitment was UNKNOWN: only the operator's UNKNOWN_COMMIT_ACK (after reconciling the ledger by hand) allows a retry
        uack = unknown_ack_path(raw_dir, tag)
        if not uack.exists():
            raise CrashUnacknowledged(f"a previous attempt for tag '{tag}' ended with an UNKNOWN ledger commitment ({unk[0].name}); tell the operator, who must "
                                      f"reconcile the ledger by hand and create {uack.name} before any retry. Nothing was published.")
        stamp = now_iso().split(" ")[0].replace(":", "-")
        for p in unk:
            p.rename(p.with_name(f"{p.name}.acknowledged.{stamp}"))
        uack.unlink()
    old = [p for p in (crash_path(raw_dir, tag), in_progress_path(raw_dir, tag)) if p.exists()]
    if not old:
        return
    ack = ack_path(raw_dir, tag)
    if not ack.exists():
        raise CrashUnacknowledged(f"a previous attempt for tag '{tag}' crashed or was killed ({', '.join(p.name for p in old)}); "
                                  f"tell the operator, who must create {ack.name} before any retry. The ledger tag is not consumed.")
    stamp = now_iso().split(" ")[0].replace(":", "-")
    for p in old:
        p.rename(p.with_name(f"{p.name}.acknowledged.{stamp}"))
    ack.unlink()


class SealedEvaluation:
    """Wraps one evaluation. `progress` logs counts only; `run` converts any failure into a crash file; `seal` records the ledger first."""

    def __init__(self, tag: str, raw_dir: Path, log, code_commit: str = "?", progress_every: int = 100):
        self.tag, self.raw_dir, self.log, self.code_commit = tag, Path(raw_dir), log, code_commit
        self.every, self.done, self.total, self.started, self.finished = progress_every, 0, None, False, False
        self.unknown = False  # True once the commitment is UNKNOWN (strict: nothing score-bearing is written)
        self.committed = False  # True once the ledger entry is known to be recorded

    def start(self, total: int) -> None:
        """Mark the evaluation as in progress (a marker with no scores: pid, time, commit, question count)."""
        self.total, self.started = total, True
        _atomic_write(in_progress_path(self.raw_dir, self.tag), json.dumps(
            {"tag": self.tag, "pid": os.getpid(), "started": now_iso(), "code_commit": self.code_commit, "total_questions": total}, indent=1))
        self.log(f"progress 0/{total}")

    def progress(self, done: int, total: int) -> None:
        self.done, self.total = done, total
        if done % self.every == 0 or done == total:
            self.log(f"progress {done}/{total}")

    def record_crash(self, exc: BaseException) -> None:
        """Write `<tag>.CRASH.json`: exception class, generic reason code, frame locations, progress counts, provenance. NEVER the message.
        Idempotent. If the ledger entry is already committed this is not a crash: the CONSUMED state is written instead."""
        if self.finished or self.unknown or crash_path(self.raw_dir, self.tag).exists():
            return
        if self.committed:
            self._write_consumed(exc, [], [])
            return
        rec = {"tag": self.tag, "time": now_iso(), **safe_reason(exc), "where": safe_frames(exc),
               "questions_scored_before_crash": self.done, "total_questions": self.total, "ledger_consumed": False,
               "code_commit": self.code_commit,
               "note": "No score was printed or written. The ledger tag is NOT consumed, but the operator must be told before any retry: "
                       "a retry is refused until the operator creates the matching CRASH_ACK file."}
        _atomic_write(crash_path(self.raw_dir, self.tag), json.dumps(rec, indent=1))
        in_progress_path(self.raw_dir, self.tag).unlink(missing_ok=True)
        self.log(f"CRASH {safe_text(exc)} after {self.done} of {self.total} questions; no scores written; ledger not consumed; crash file written")

    def run(self, evaluator):
        """`evaluator(progress_callback) -> results` (kept in memory). Any BaseException (including SIGTERM/SIGINT turned into one) is recorded."""
        try:
            return evaluator(self.progress)
        except BaseException as e:
            self.record_crash(e)
            raise EvaluationCrashed(f"evaluation crashed after {self.done} of {self.total} questions") from e

    def _write_consumed(self, cause: BaseException, written: list[str], failed: list[str]) -> None:
        rec = {"tag": self.tag, "time": now_iso(), "state": "LEDGER_CONSUMED", "ledger_consumed": True, **safe_reason(cause), "where": safe_frames(cause),
               "questions_scored": self.done, "total_questions": self.total, "result_files_written": written, "result_files_not_written": failed,
               "code_commit": self.code_commit,
               "recovery": ["The ledger entry for this tag is recorded: the tag is CONSUMED and a new evaluation is refused (exit 14).",
                            "Do NOT retry the evaluation. The completed results were kept in memory and were written/echoed after the commit.",
                            "Read result_files_written / result_files_not_written above; any file not written was echoed to stdout and to the raw stdout log.",
                            "Tell the operator; the operator decides how to restore a missing file from the stdout log."]}
        with _signals_blocked():
            _atomic_write(consumed_path(self.raw_dir, self.tag), json.dumps(rec, indent=1))
            in_progress_path(self.raw_dir, self.tag).unlink(missing_ok=True)

    def seal(self, ledger_record, ledger_contains, outputs: dict[Path, str], echo: str) -> None:
        """1. `ledger_record()` FIRST (atomic, SIGTERM/SIGINT blocked until the commit is marked). 2. write every output atomically. 3. echo.
        Ledger commitment has three states: CONFIRMED (the callback returned, or a SUCCESSFUL read of the ledger shows the tag), NOT_COMMITTED (a
        successful read shows no tag) and UNKNOWN (the read itself failed). A read failure is never taken as "tag present".
        NOT_COMMITTED: crash file, tag not consumed (exit 15). CONFIRMED but a later failure (exception, SIGTERM, write error): the remaining results
        are still written and echoed, `<tag>.CONSUMED.json` is written, exit 17. UNKNOWN: NO result is published or echoed; the completed results
        stay in memory only; `<tag>.UNKNOWN_COMMIT.json` (class, code, counts) is the only file written, it blocks any retry until the
        author acknowledges, exit 19. Only CONFIRMED may publish results."""
        written, cause = [], None
        try:
            with _signals_blocked():
                ledger_record()  # a failure here is settled below by reading the actual ledger
                self.committed = True
            for path, text in outputs.items():
                _atomic_write(Path(path), text)
                written.append(str(path))
        except BaseException as e:  # noqa: BLE001 - any failure is classified, never propagated raw
            cause = e
        if cause is None:
            self.finished = True
            in_progress_path(self.raw_dir, self.tag).unlink(missing_ok=True)
            self.log(echo)
            return
        state = "CONFIRMED" if self.committed else self._read_commitment(ledger_contains)
        if state == "NOT_COMMITTED":
            self.record_crash(cause)
            raise EvaluationCrashed("could not record the ledger entry; no scores written") from cause
        if state == "UNKNOWN":
            self._write_unknown(cause)
            raise CommitUnknown("could not establish whether the ledger entry was recorded; nothing was published") from cause
        self.committed = True
        failed = []
        with _signals_blocked():
            for path, text in outputs.items():
                if str(path) not in written:
                    try:
                        _atomic_write(Path(path), text)
                        written.append(str(path))
                    except BaseException:  # noqa: BLE001 - reported as not written
                        failed.append(str(path))
        self.finished = True
        self._write_consumed(cause, [Path(w).name for w in written], [Path(f).name for f in failed])
        self.log(echo)  # allowed now: the ledger entry is CONFIRMED recorded
        by_name = {str(k): v for k, v in outputs.items()}
        for f in failed:
            print(f"----- {f} -----\n{by_name[f]}")
        raise PostLedgerWriteFailed(f"failure after the ledger commit ({safe_text(cause)}); tag CONSUMED; {len(failed)} result files not written; see {consumed_path(self.raw_dir, self.tag).name}") from cause

    @staticmethod
    def _read_commitment(ledger_contains) -> str:
        """CONFIRMED / NOT_COMMITTED from a successful read; UNKNOWN if the read fails (never mapped to 'present')."""
        try:
            return "CONFIRMED" if ledger_contains() else "NOT_COMMITTED"
        except BaseException:  # noqa: BLE001 - an unreadable ledger is UNKNOWN
            return "UNKNOWN"

    def _write_unknown(self, cause: BaseException) -> None:
        """STRICT: the only file written is `<tag>.UNKNOWN_COMMIT.json` (class, reason code, counts, provenance, recovery text). The completed results
        stay in process memory and are dropped when the process exits. Nothing score-bearing is written, printed or logged, and the ledger is NOT
        stated to be recorded."""
        self.unknown = self.finished = True  # set first: a signal delivered inside this method must not turn into a crash record
        with _signals_blocked():
            _atomic_write(unknown_path(self.raw_dir, self.tag), json.dumps(
                {"tag": self.tag, "time": now_iso(), "state": "UNKNOWN_COMMIT", "ledger_state": "unknown", **safe_reason(cause), "where": safe_frames(cause),
                 "questions_scored": self.done, "total_questions": self.total, "code_commit": self.code_commit,
                 "recovery": ["Whether the ledger entry was recorded could not be established. No result was written, printed or echoed anywhere.",
                              "Do NOT retry. Tell the operator, who reconciles the ledger by hand (docs/phase3b_interface.md 7.1).",
                              "The operator creates <tag>.UNKNOWN_COMMIT_ACK only after that reconciliation."]}, indent=1))
            in_progress_path(self.raw_dir, self.tag).unlink(missing_ok=True)
        self.log(f"COMMIT UNKNOWN ({safe_text(cause)}) after {self.done} of {self.total} questions; nothing published; tell the operator; see {unknown_path(self.raw_dir, self.tag).name}")


def exit_code_for(ev: "SealedEvaluation", exc: BaseException) -> int:
    """Process exit code after `exc` escaped the evaluation: 19 once the commitment is UNKNOWN (whatever exception surfaced), else the known guard
    code or 15."""
    return 19 if ev.unknown else safe_exit_code(exc)


def install_signal_handlers() -> None:
    """Turn SIGTERM into an exception so a handled termination writes the crash (or consumed) file (SIGKILL cannot be handled: the IN_PROGRESS
    marker covers it)."""
    def handler(signum, frame):
        raise SignalTermination("signal")
    signal.signal(signal.SIGTERM, handler)
