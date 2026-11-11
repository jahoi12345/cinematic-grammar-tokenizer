"""CPU-only demonstration that the ShotBench evaluation is SEALED: no accuracy, per-question outcome or partial score is printed or written until the
full set is scored and the ledger entry is recorded; a simulated crash leaks no score; a crashed or killed run blocks a retry until acknowledged.
Uses synthetic questions with a planted SECRET score (so any leak is detectable); no model, no ShotBench data, no real ledger.
Usage: uv run python scripts/p3b_sealed_demo.py -> results/phase3b/shotbench_sealed_demo.md"""
import contextlib
import io
import json
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from cgt.p3b_vlm import guards, sealed
from cgt.p3b_vlm.score import evaluate_samples

ROOT = Path(__file__).resolve().parents[1]
SECRET_ACC = "0.4321"  # a planted partial/final score: it must never appear anywhere before the seal
FORBIDDEN = ["acc", "correct", "pred", "gold", "logit", "mean", SECRET_ACC, "43.2", "[\"A\"", "ledger recorded"]
results = []


def record(name, ok, detail):
    results.append((name, ok, detail))
    print(("PASS" if ok else "FAIL") + f": {name}: {detail}", flush=True)


def tree_text(d: Path) -> str:
    return "\n".join(p.read_text() for p in sorted(Path(d).rglob("*")) if p.is_file())


def leaked(text: str) -> list[str]:
    low = text.lower()
    return [w for w in FORBIDDEN if w.lower() in low]


class Cap:
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(msg)


def fake_evaluator(n, crash_at=None, secret_msg=False):
    """Scores n synthetic questions in memory; optionally raises at question `crash_at` (message may even contain a planted score)."""
    def ev(progress):
        out = []
        for i in range(n):
            if crash_at is not None and i == crash_at:
                raise RuntimeError(f"simulated crash; partial accuracy {SECRET_ACC} ({SECRET_ACC.replace('0.', '')}%)" if secret_msg else "simulated crash")
            out.append({"id": f"q{i}", "pred": "A", "gold": "A" if i % 2 else "B"})
            progress(len(out), n)
        return out
    return ev


# ---------- 1. crash midway: only a reason and a count are left; the ledger is untouched; no result file exists
d = Path(tempfile.mkdtemp())
ledger = d / "ledger.json"
ledger.write_text("{}")
cap = Cap()
e1 = sealed.SealedEvaluation("t1", d / "raw", cap, "deadbeef", progress_every=100)
e1.start(1000)
try:
    e1.run(fake_evaluator(1000, crash_at=250, secret_msg=True))
except guards.EvaluationCrashed:
    pass
crash = json.loads(sealed.crash_path(d / "raw", "t1").read_text())
# the scan covers EVERYTHING written or logged: the crash file in full (no field is exempt) and every log line
all_text = tree_text(d / "raw") + "\n" + "\n".join(cap.lines)
record("crash midway writes a crash file with exception class, generic reason code and progress counts", crash["questions_scored_before_crash"] == 250 and crash["total_questions"] == 1000
       and crash["ledger_consumed"] is False and crash["exception_class"] == "RuntimeError" and crash["reason_code"] == "internal_error" and "reason" not in crash,
       f"class={crash['exception_class']}, code={crash['reason_code']}, scored={crash['questions_scored_before_crash']}/{crash['total_questions']}, ledger_consumed={crash['ledger_consumed']}")
record("a planted score in the exception message appears nowhere (the message is not retained at all)", "0.4321" not in all_text and "4321" not in all_text and "simulated" not in all_text.lower(), f"crash file keys: {sorted(crash)}")
record("crash leaks nothing: no forbidden word anywhere in the crash file (reason field NOT exempt), the raw dir or the log lines", not leaked(all_text), f"leaks found: {leaked(all_text) or 'none'}")
# F1: synthetic disclosure cases (answer letters, integer logits and correct counts, scientific notation, structured outcomes, a data-bearing class name)
class Score0p4321Error(Exception):
    pass


F1_CASES = {
    "answer letters": RuntimeError("pred=B gold=A"),
    "correct count": ValueError("correct=2/3 of 3"),
    "integer logits": KeyError("logits=[1,2,3,4]"),
    "scientific notation": RuntimeError("accuracy=1e-1 and 4.3e-01"),
    "structured outcome": TypeError("{'id': 'sb7', 'pred': 'B', 'gold': 'A', 'correct': True}"),
    "JSON outcome": OSError('{"correct": 1, "n": 3, "acc": 0.33}'),
    "data-bearing class name": Score0p4321Error("0.4321"),
}
f1_bad = []
for label, exc in F1_CASES.items():
    df = Path(tempfile.mkdtemp())
    cp = Cap()
    ef = sealed.SealedEvaluation("f1", df, cp, "abc")
    ef.start(10)
    ef.done = 4
    try:
        ef.run(lambda progress, exc=exc: (_ for _ in ()).throw(exc))
    except guards.EvaluationCrashed:
        pass
    txt = tree_text(df) + "\n" + "\n".join(cp.lines)
    frag = [f for f in ("pred=B", "gold=A", "2/3", "[1,2,3,4]", "1e-1", "4.3e-01", "'correct': True", "0.33", "0.4321", "4321", "sb7") if f in txt]
    if frag or leaked(txt):
        f1_bad.append((label, frag, leaked(txt)))
record("F1: crash files for exceptions carrying letters, integer logits, correct counts, scientific notation and structured outcomes leak nothing (reason not exempt)",
       not f1_bad, f"{len(F1_CASES)} cases; leaks: {f1_bad or 'none'}")
guard_log = Cap()
guard_log(f"FAIL CLOSED: {sealed.safe_text(guards.UnreadableImage('pred=B gold=A correct=2/3 acc 0.4321'))}")
record("F1: pre-seal error logging uses the same safe representation (class + reason code only)", not leaked(guard_log.lines[0]) and "0.4321" not in guard_log.lines[0], guard_log.lines[0])
record("crash leaves the ledger untouched and writes no result file", ledger.read_text() == "{}" and not list((d / "raw").glob("*.csv")) and not list((d / "raw").glob("*predictions*")),
       f"files in raw dir: {sorted(p.name for p in (d / 'raw').iterdir())}")
record("only progress counts were logged during the run", all(l.startswith(("progress ", "CRASH")) for l in cap.lines), "; ".join(cap.lines))
try:
    sealed.require_no_unacknowledged_crash(d / "raw", "t1")
    record("a retry after a crash is refused until acknowledged", False, "retry was allowed")
except guards.CrashUnacknowledged as ex:
    record("a retry after a crash is refused until acknowledged", ex.exit_code == 16, f"exit {ex.exit_code}")
sealed.ack_path(d / "raw", "t1").write_text("operator acknowledged\n")
sealed.require_no_unacknowledged_crash(d / "raw", "t1")
record("after the operator's ACK the retry is allowed and the old files are archived", not sealed.crash_path(d / "raw", "t1").exists() and any("acknowledged" in p.name for p in (d / "raw").iterdir()), "")

# ---------- 2. a handled SIGTERM mid-run writes the crash file; a hard SIGKILL leaves the IN_PROGRESS marker which also blocks a retry
child = f"""
import time, sys
sys.path.insert(0, {str(ROOT / 'src')!r})
from pathlib import Path
from cgt.p3b_vlm import sealed
sealed.install_signal_handlers()
ev = sealed.SealedEvaluation('t2', Path({str(d / 'raw2')!r}), lambda m: print(m, flush=True), 'cafe', progress_every=10)
ev.start(1000)
def slow(progress):
    for i in range(1000):
        time.sleep(0.05); progress(i + 1, 1000)
ev.run(slow)
"""
for sig, label in ((signal.SIGTERM, "t2"), (signal.SIGKILL, "t3")):
    cpath = d / f"child_{label}.py"
    cpath.write_text(child.replace("'t2'", f"'{label}'"))
    pr = subprocess.Popen([sys.executable, str(cpath)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(2.5)
    pr.send_signal(sig)
    out, err = pr.communicate(timeout=20)
    files = sorted(p.name for p in (d / "raw2").iterdir()) if (d / "raw2").exists() else []
    if sig == signal.SIGTERM:
        cj = json.loads(sealed.crash_path(d / "raw2", label).read_text())
        record("SIGTERM mid-run: crash file with reason and progress count, nothing else", cj["questions_scored_before_crash"] > 0 and cj["reason_code"] == "terminated_by_signal" and not leaked(out + json.dumps(cj)),
               f"scored {cj['questions_scored_before_crash']}/1000, code {cj['reason_code']}, files {files}")
    else:
        try:
            sealed.require_no_unacknowledged_crash(d / "raw2", label)
            record("SIGKILL mid-run: the IN_PROGRESS marker blocks a retry", False, "retry allowed")
        except guards.CrashUnacknowledged:
            record("SIGKILL mid-run (cannot be handled): the IN_PROGRESS marker blocks a retry until acknowledged", True, f"files {files}; stdout had only progress lines: {all(l.startswith('progress') for l in out.splitlines())}")

# ---------- 3. normal completion: progress only during the run; the ledger is recorded BEFORE any result file exists; results appear only after
d3 = Path(tempfile.mkdtemp())
cap3 = Cap()
state = {}
e3 = sealed.SealedEvaluation("t4", d3 / "raw", cap3, "feed", progress_every=100)
e3.start(500)
res = e3.run(fake_evaluator(500))
during = list(cap3.lines)


def ledger_record():
    state["files_at_ledger_time"] = sorted(p.name for p in (d3 / "raw").iterdir())
    state["log_leaks_at_ledger_time"] = leaked("\n".join(cap3.lines))
    (d3 / "ledger.json").write_text(json.dumps({"t4": "evaluated once"}))


e3.seal(ledger_record, lambda: (d3 / "ledger.json").exists(), {d3 / "raw" / "t4.csv": "attribute,acc\nsize,0.4321\n", d3 / "raw" / "t4.predictions.jsonl": "{}\n"}, "DONE: mean option-restricted accuracy 43.21%")
record("during the run only 'progress k/n' lines are logged", all(l.startswith("progress ") for l in during), f"{len(during)} lines, e.g. {during[:2]}")
record("the ledger is recorded BEFORE any result file exists and before any score is echoed",
       state["files_at_ledger_time"] == ["t4.IN_PROGRESS.json"] and not state["log_leaks_at_ledger_time"], f"files at ledger time: {state['files_at_ledger_time']}")
record("after the seal the result files exist, the echo is logged and the IN_PROGRESS marker is gone",
       (d3 / "raw" / "t4.csv").exists() and "DONE" in cap3.lines[-1] and not sealed.in_progress_path(d3 / "raw", "t4").exists(), "")

# ---------- 4. a write failure AFTER the ledger entry: results are echoed to stdout, the failure file holds reasons only
d4 = Path(tempfile.mkdtemp())
(d4 / "raw").mkdir()
blocked = d4 / "blocked"
blocked.write_text("a file, not a directory")  # writing blocked/x fails
e4 = sealed.SealedEvaluation("t5", d4 / "raw", Cap(), "f00d")
e4.start(10)
e4.run(fake_evaluator(10))
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        e4.seal(lambda: (d4 / "ledger.json").write_text("recorded"), lambda: (d4 / "ledger.json").exists(), {blocked / "t5.csv": "secret,results\n"}, "DONE: results")
    record("write failure after the ledger", False, "no error")
except guards.PostLedgerWriteFailed as ex:
    pf = json.loads((d4 / "raw" / "t5.CONSUMED.json").read_text())
    record("a write failure after the ledger is recorded echoes the results to stdout and writes a CONSUMED file without results",
           ex.exit_code == 17 and "secret,results" in buf.getvalue() and pf["ledger_consumed"] is True and "secret" not in json.dumps(pf) and pf["result_files_not_written"] == ["t5.csv"], f"exit {ex.exit_code}, state {pf['state']}")

# ---------- 5. the real scorer: progress callback gets counts only; its error message never contains the gold letter
class FakeDS:
    def process(self, s):
        return {}


import cgt.p3b_vlm.score as score_mod

orig = score_mod.letter_logits
calls = []
score_mod.letter_logits = lambda model, item, ids4: (np.array([0.1, 0.9, 0.2, 0.3]), 33)
samples = [{"id": f"s{i}", "attribute": "size", "images": ["x"], "messages": [], "answer": "B", "options": {"B": "Medium"}} for i in range(5)]
r = evaluate_samples(None, FakeDS(), samples, {"A": 32, "B": 33, "C": 34, "D": 35}, progress=lambda d, t: calls.append((d, t)))
record("the real scorer's progress callback receives counts only", calls == [(i, 5) for i in range(1, 6)], f"calls {calls}")
score_mod.letter_logits = lambda model, item, ids4: (np.array([0.1, 0.9, 0.2, 0.3]), 99)
try:
    evaluate_samples(None, FakeDS(), samples[:1], {"A": 32, "B": 33, "C": 34, "D": 35})
except ValueError as ex:
    record("the scorer's error message carries no gold letter, prediction or token id", "B" not in str(ex).replace("Because", "") and "99" not in str(ex) and "33" not in str(ex), f"message: {ex}")
score_mod.letter_logits = orig
# ---------- 6. F2: ledger commitment is explicit; any failure AFTER the commit leaves a CONSUMED tag, keeps the results and forbids a new evaluation
CHILD = """
import os, signal, sys, json
sys.path.insert(0, {src!r})
from pathlib import Path
from cgt.p3b_vlm import guards, sealed
from cgt.eval import shotbench as sbmod
mode, d = sys.argv[1], Path(sys.argv[2])
ledger = d / "ledger.json"
sealed.install_signal_handlers()
ev = sealed.SealedEvaluation("tf2", d / "raw", lambda m: print(m, flush=True), "beef")
ev.start(3)
ev.run(lambda cb: [cb(i, 3) for i in (1, 2, 3)])
outputs = {{d / "raw" / "tf2.csv": "RESULT-CSV\\n", d / "raw" / "tf2.predictions.jsonl": "RESULT-PRED\\n", d / "raw" / "tf2.manifest.json": "RESULT-MANIFEST\\n"}}
def record():
    if mode in ("unknown_before_write", "sigterm_unknown_window"):
        raise RuntimeError("SECRET_OUTCOME_73 failed before any write")
    if mode == "sigterm_in_callback":
        os.kill(os.getpid(), signal.SIGTERM)  # pending while the ledger commit window is blocked
        raise RuntimeError("SECRET_OUTCOME_73 callback failed")
    sbmod.ledger_record("tf2", "synthetic", ledger)
    if mode == "term_after_commit":
        os.kill(os.getpid(), signal.SIGTERM)  # pending until the commit is marked, then delivered
    if mode in ("raise_after_commit", "unreadable_ledger"):
        raise RuntimeError("callback outcome uncertain: pred=B")

if mode == "sigterm_unknown_window":
    orig_w = sealed._atomic_write
    def patched_w(path, text):
        orig_w(path, text)
        if str(path).endswith("UNKNOWN_COMMIT.json"):
            os.kill(os.getpid(), signal.SIGTERM)  # delivered when the signals are unblocked after the marker is written
    sealed._atomic_write = patched_w
if mode == "term_between_writes":
    orig = sealed._atomic_write
    state = {{"n": 0}}
    def patched(path, text):
        orig(path, text)
        if str(path).endswith("tf2.csv") and state["n"] == 0:
            state["n"] = 1
            os.kill(os.getpid(), signal.SIGTERM)  # handled right after the first result file
    sealed._atomic_write = patched
contains = (lambda: (_ for _ in ()).throw(OSError("unreadable"))) if mode in ("unreadable_ledger", "unknown_before_write", "sigterm_unknown_window", "sigterm_in_callback") else (lambda: "tf2" in sbmod.ledger_read(ledger))
if mode == "pre_commit_failure":
    def record():
        raise RuntimeError("write failed before the ledger changed")
try:
    ev.seal(record, contains, outputs, "ECHO-RESULTS mean 43.21%")
except BaseException as e:
    if ev.started and not ev.finished:
        ev.record_crash(e)
    sys.exit(sealed.exit_code_for(ev, e))
"""


def run_child(mode):
    dd = Path(tempfile.mkdtemp())
    (dd / "raw").mkdir()
    cp = dd / "child.py"
    cp.write_text(CHILD.format(src=str(ROOT / "src")))
    pr = subprocess.run([sys.executable, str(cp), mode, str(dd)], capture_output=True, text=True, timeout=120, check=False)
    return dd, pr


from cgt.eval import (
    shotbench as sbmod,
)

for mode, label in (("term_after_commit", "SIGTERM immediately after the ledger commit"), ("term_between_writes", "SIGTERM between result writes"),
                    ("raise_after_commit", "exception after the ledger was written (callback outcome uncertain, reconciled by a SUCCESSFUL positive read)")):
    dd, pr = run_child(mode)
    raw = dd / "raw"
    cj = sealed.consumed_path(raw, "tf2")
    consumed = json.loads(cj.read_text()) if cj.exists() else {}
    files_ok = all((raw / n).read_text() == f"RESULT-{k}\n" for n, k in (("tf2.csv", "CSV"), ("tf2.predictions.jsonl", "PRED"), ("tf2.manifest.json", "MANIFEST")))
    try:
        sbmod.ledger_assert_unused("tf2", dd / "ledger.json")
        refused = False
    except RuntimeError:
        refused = True
    (dd / "ledger.json").unlink()  # even a lost ledger must not allow a second evaluation
    try:
        sealed.require_not_consumed(raw, "tf2")
        refused2 = False
    except guards.LedgerUsed:
        refused2 = True
    ok = (pr.returncode == 17 and consumed.get("state") == "LEDGER_CONSUMED" and consumed.get("ledger_consumed") is True and not sealed.crash_path(raw, "tf2").exists()
          and files_ok and "ECHO-RESULTS" in pr.stdout and refused and refused2 and "pred=B" not in cj.read_text() and bool(consumed.get("recovery")))
    record(f"F2: {label}: tag CONSUMED (never 'unconsumed'), results written and echoed, new evaluation refused", ok,
           f"exit {pr.returncode}, state {consumed.get('state')}, cause {consumed.get('reason_code')}, written {consumed.get('result_files_written')}, "
           f"ledger refuses {refused}, consumed marker refuses {refused2}")
# R1 (STRICT): UNKNOWN commitment writes NO score-bearing file anywhere. Every file under the scratch raw directory is scanned, no exemption by name.
PLANTED = ("RESULT-", "ECHO", "SECRET", "43.21", "0.4321", "mean")
for mode, label, ledger_has in (("unknown_before_write", "callback failed before any write AND the ledger is unreadable", False),
                                ("unreadable_ledger", "ledger really written, then a callback exception AND the ledger is unreadable", True),
                                ("sigterm_in_callback", "SIGTERM during the commit window, callback failed, ledger unreadable", False),
                                ("sigterm_unknown_window", "SIGTERM while the UNKNOWN marker is being written", False)):
    dd, pr = run_child(mode)
    raw = dd / "raw"
    files = sorted(f.name for f in raw.rglob("*") if f.is_file())
    all_raw = "\n".join(f.read_text() for f in raw.rglob("*") if f.is_file()) + pr.stdout + pr.stderr
    hits = [w for w in PLANTED if w in all_raw]
    uj = json.loads(sealed.unknown_path(raw, "tf2").read_text()) if sealed.unknown_path(raw, "tf2").exists() else {}
    marker_text = sealed.unknown_path(raw, "tf2").read_text() if sealed.unknown_path(raw, "tf2").exists() else ""
    try:
        sealed.require_no_unacknowledged_crash(raw, "tf2")
        blocked = False
    except guards.CrashUnacknowledged:
        blocked = True
    sealed.unknown_ack_path(raw, "tf2").write_text("operator reconciled the ledger by hand\n")
    sealed.require_no_unacknowledged_crash(raw, "tf2")
    ok = (pr.returncode == 19 and uj.get("state") == "UNKNOWN_COMMIT" and uj.get("ledger_state") == "unknown" and files == ["tf2.UNKNOWN_COMMIT.json"] and not hits
          and not sealed.consumed_path(raw, "tf2").exists() and not sealed.crash_path(raw, "tf2").exists() and "ledger_consumed" not in marker_text
          and "recorded" not in marker_text.split("recovery")[0] and blocked and (dd / "ledger.json").exists() == ledger_has)
    record(f"R1 strict: UNKNOWN commitment, {label}: the only file is UNKNOWN_COMMIT.json, no planted score in ANY file, stdout or stderr, retry blocked until acknowledged", ok,
           f"exit {pr.returncode}, files {files}, planted hits {hits or 'none'}, retry blocked {blocked}")
dd, pr = run_child("pre_commit_failure")
cj = json.loads(sealed.crash_path(dd / "raw", "tf2").read_text()) if sealed.crash_path(dd / "raw", "tf2").exists() else {}
record("F2: a failure BEFORE the ledger commit is an unconsumed crash (control): exit 15, no result file, tag not in the ledger",
       pr.returncode == 15 and cj.get("ledger_consumed") is False and not (dd / "ledger.json").exists() and not (dd / "raw" / "tf2.csv").exists() and not sealed.consumed_path(dd / "raw", "tf2").exists(),
       f"exit {pr.returncode}, crash file ledger_consumed={cj.get('ledger_consumed')}")
# atomic ledger: an interrupted replace leaves the old ledger intact and valid
import os as _os

dl = Path(tempfile.mkdtemp()) / "ledger.json"
dl.write_text(json.dumps({"old": "x"}, indent=1))
real_replace = _os.replace
_os.replace = lambda *a_, **k_: (_ for _ in ()).throw(OSError("interrupted"))
try:
    sbmod.ledger_record("new", "n", dl)
except OSError:
    pass
_os.replace = real_replace
record("F2: the ledger write is temp file + atomic replace: an interruption leaves the old ledger intact and parseable",
       json.loads(dl.read_text()) == {"old": "x"}, f"ledger after interrupted write: {dl.read_text().strip()!r}")

# ---------- 7. R2: only FIXED known classes/codes are named; unknown subclasses of anything map to OtherException; exit_code attributes are never reflected
class predB_goldA_accuracy_0_75(guards.FailClosed):
    exit_code = 75


class predB_signal_0_75(sealed.SignalTermination):
    pass


class gold_A_ValueError(ValueError):
    pass


class HostileStr(Exception):
    def __str__(self):
        raise AssertionError("__str__ must never be called")

    def __repr__(self):
        raise AssertionError("__repr__ must never be called")


r2_bad = []
for label, exc in {"subclass of a project guard (name carries an outcome, exit code 75)": predB_goldA_accuracy_0_75("pred=B"),
                   "subclass of SignalTermination": predB_signal_0_75("x"), "subclass of ValueError": gold_A_ValueError("x"),
                   "ordinary unknown exception": ArithmeticError("correct=2/3"), "exception with hostile __str__/__repr__": HostileStr()}.items():
    df = Path(tempfile.mkdtemp())
    cp = Cap()
    ef = sealed.SealedEvaluation("r2", df, cp, "abc")
    ef.start(10)
    try:
        ef.run(lambda progress, exc=exc: (_ for _ in ()).throw(exc))
    except guards.EvaluationCrashed:
        pass
    cj_ = json.loads(sealed.crash_path(df, "r2").read_text())
    txt = tree_text(df) + "\n" + "\n".join(cp.lines) + sealed.safe_text(exc)
    if (cj_["exception_class"] != "OtherException" or cj_["reason_code"] != "unclassified_exception" or "predB" in txt or "gold_A" in txt or "75" in txt
            or sealed.safe_exit_code(exc) != 15 or leaked(txt)):
        r2_bad.append(label)
record("R2: data-bearing subclasses of a guard, of SignalTermination and of ValueError, an unknown exception and a hostile __str__ all map to OtherException; "
       "no name or exit code reflected", not r2_bad, f"5 cases; failures: {r2_bad or 'none'}")
known_ok = all(sealed.safe_exit_code(c("m")) == c.exit_code and sealed.safe_reason(c("m"))["exception_class"] == c.__name__
               for c in (guards.LedgerUsed, guards.NotAuthorized, guards.TagNotBound, guards.DataRootViolation, guards.UnreadableImage,
                         guards.AdapterHashError, guards.DirtyTree, guards.CrashUnacknowledged, guards.PostLedgerWriteFailed, guards.CommitUnknown))
record("R2: the real project guards keep their documented exit codes and names", known_ok, "exit codes 5,6,7,10,13,14,16,17,18,19 verified from the fixed table")

# ---------- 8. adversarial pass over every place a score-bearing value could be written, logged or echoed (docs/phase3b_interface.md 7.2)
class Chained(Exception):
    pass


def chained():
    try:
        raise ValueError("accuracy=0.4321 pred=B gold=A")
    except ValueError as inner:
        raise Chained("wrapper") from inner


def exit_with_secret():
    raise SystemExit("correct=2/3 acc 0.4321")


def secret_in_local():
    secret_correct = "SECRET_LOCAL_0.4321"  # noqa: F841 - a local variable must never reach the crash file via frames
    raise RuntimeError("x")


adv_bad = []
for label, fn in (("chained exception (__cause__ carries the score)", chained), ("SystemExit with a score as its code", exit_with_secret),
                  ("score in a local variable of a failing frame", secret_in_local)):
    df = Path(tempfile.mkdtemp())
    cp = Cap()
    ef = sealed.SealedEvaluation("adv", df, cp, "abc")
    ef.start(10)
    try:
        ef.run(lambda progress, fn=fn: fn())
    except guards.EvaluationCrashed:
        pass
    txt = tree_text(df) + "\n" + "\n".join(cp.lines)
    if leaked(txt) or "0.4321" in txt or "SECRET_LOCAL" in txt or "pred=B" in txt:
        adv_bad.append(label)
record("adversarial: chained exception, SystemExit(score) and a score in a local variable leave nothing in the crash file or the log", not adv_bad, f"3 cases; failures: {adv_bad or 'none'}")
# a ledger callback failing with a score-bearing message BEFORE the commit (read succeeds: NOT_COMMITTED) leaves only a crash file; dotfiles/tmp files included in the scan
df = Path(tempfile.mkdtemp())
cp = Cap()
ef = sealed.SealedEvaluation("adv2", df, cp, "abc")
ef.start(3)
ef.run(lambda cb: [cb(i, 3) for i in (1, 2, 3)])
try:
    ef.seal(lambda: (_ for _ in ()).throw(RuntimeError("ledger said accuracy=0.4321 pred=B")), lambda: False, {df / "adv2.csv": "mean,0.4321\n"}, "DONE 43.21%")
except guards.EvaluationCrashed:
    pass
txt = tree_text(df) + "\n" + "\n".join(cp.lines)
record("adversarial: a ledger failure with a score in its message (negative read) is an unconsumed crash: no result file, temp file or log line carries a score",
       not leaked(txt) and "0.4321" not in txt and "43.21" not in txt and sealed.crash_path(df, "adv2").exists() and not (df / "adv2.csv").exists(),
       f"files: {sorted(p.name for p in df.rglob('*') if p.is_file())}")
# under UNKNOWN every file in the raw dir is score-free (no exemption), including temp files
df = Path(tempfile.mkdtemp())
cp = Cap()
ef = sealed.SealedEvaluation("adv3", df, cp, "abc")
ef.start(3)
ef.run(lambda cb: [cb(i, 3) for i in (1, 2, 3)])
try:
    ef.seal(lambda: (_ for _ in ()).throw(RuntimeError("x")), lambda: (_ for _ in ()).throw(OSError("unreadable")), {df / "adv3.csv": "mean,0.4321\n"}, "DONE 43.21%")
except guards.CommitUnknown:
    pass
others = "\n".join(p.read_text() for p in df.rglob("*") if p.is_file()) + "\n".join(cp.lines)
record("adversarial: under UNKNOWN commitment EVERY file in the raw dir and every log line is score-free (no exemption)",
       "0.4321" not in others and "43.21" not in others and not leaked(others) and sorted(p.name for p in df.rglob("*") if p.is_file()) == ["adv3.UNKNOWN_COMMIT.json"],
       f"files: {sorted(p.name for p in df.rglob('*') if p.is_file())}")

lines = ["# ShotBench evaluation SEALED: CPU-only demonstration (synthetic questions with a planted secret score; no model, no ShotBench data, real ledger untouched)", "",
         "Rule: no accuracy, per-question correctness or partial score is printed or written until the full set is scored AND the ledger entry is recorded; only progress counts are logged; a crash leaves only a reason and a count and does not consume the tag; a retry is refused until the operator acknowledges.", "",
         "| check | result | detail |", "|---|---|---|"]
for name, ok, detail in results:
    lines.append(f"| {name} | **{'PASS' if ok else 'FAIL'}** | {detail.replace('|', '/')[:300]} |")
lines += ["", f"All {len(results)} checks passed: {all(r[1] for r in results)}."]
(ROOT / "results/phase3b/shotbench_sealed_demo.md").write_text("\n".join(lines) + "\n")
raise SystemExit(0 if all(r[1] for r in results) else 1)
