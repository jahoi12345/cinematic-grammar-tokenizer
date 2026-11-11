"""CPU self-check of the sealed CameraBench test store on a SYNTHETIC copy with a planted label marker and a SYNTHETIC git repository holding the freeze record: labels are unreachable
without the final flag; a model tag must be registered in the COMMITTED freeze record (an unregistered tag, a wrong or missing adapter / configuration hash, an untracked or only
locally edited freeze record all refuse before any label is read); listing and source ids never read a label; the ledger line is written BEFORE the labels are read; a second final evaluation
of the same tag is refused; the label mapping behaves as documented; the REAL store lists the 1,071 downloaded GIFs, refuses every tag while no freeze record is committed and reads no label.
Writes results/phase4/camerabench_lock_selfcheck.md. Usage: uv run python scripts/p4_camerabench_lock_selfcheck.py"""
import json
import subprocess
import tempfile
from pathlib import Path

from cgt.p4_camera_move import camerabench as C

ROOT = Path(__file__).resolve().parents[1]
rows = []


def check(name, ok, detail=""):
    rows.append((name, ok, detail))
    print(("PASS" if ok else "FAIL"), name, detail, flush=True)


def git(repo, *a):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo, capture_output=True, text=True, check=True).stdout


tmp = Path(tempfile.mkdtemp())
cb = tmp / "cb"
(cb / "videos_gif").mkdir(parents=True)
for n in ("AAA.0.0.gif", "AAA.1.2.gif", "BBB.3.4.gif"):
    (cb / "videos_gif" / n).write_bytes(b"GIF89a")
MARK = "PLANTED_LABEL_MARKER"
(cb / "test.jsonl").write_text("\n".join(json.dumps({"Video": f"x/{s}.gif", "labels": ["pan-left", MARK], "caption": MARK, "path": f"videos/{s}.mp4"}) for s in ("AAA.0.0", "AAA.1.2", "BBB.3.4")))
repo = tmp / "repo"
(repo / "results/phase4/runs/video_v1").mkdir(parents=True)
freeze = repo / "results/phase4/runs/video_v1/FREEZE.json"
git(repo, "init", "-q")
led = tmp / "ledger.jsonl"


def store(**kw):
    return C.CameraBenchTestStore(cb, led, freeze, repo, **kw)


# --- no committed freeze record at all
freeze.write_text(json.dumps({"registered_models": {"m_adapter": {"adapter_sha256": "AAA"}}}))  # present but UNTRACKED and uncommitted
git(repo, "add", "-N", ".")
try:
    store().labels(final_evaluation=True, model_tag="m_adapter", adapter_sha256="AAA")
    check("an untracked / uncommitted freeze record registers nothing", False)
except C.CameraBenchNotRegistered:
    check("an untracked / uncommitted freeze record registers nothing", not led.exists(), "refused, ledger untouched")
(repo / ".gitignore").write_text("")
reg = {"m_adapter": {"adapter_sha256": "AAA"}, "m_geom": {"config_hash": "CCC"}, "m_plain": {}}
freeze.write_text(json.dumps({"registered_models": reg}))
git(repo, "add", "-A")
git(repo, "commit", "-q", "-m", "freeze")
check("listing and source ids never read a label", [p.name for p in store().video_files()] == ["AAA.0.0.gif", "AAA.1.2.gif", "BBB.3.4.gif"] and store().source_ids() == ["AAA", "AAA", "BBB"] and not led.exists())
for label, kw in (("no flag", {}), ("flag without a model tag", {"final_evaluation": True}), ("model tag without the flag", {"model_tag": "m_plain"})):
    try:
        store().labels(**kw)
        check(f"labels refused: {label}", False)
    except C.CameraBenchLocked:
        check(f"labels refused: {label}", not led.exists(), "ledger untouched")
real_read = Path.read_text
seen = {"label_reads": 0}


def spy(self, *a, **k):
    if self.name == "test.jsonl":
        seen["label_reads"] += 1
        seen["ledger_lines_when_labels_read"] = len(led.read_text().splitlines()) if led.exists() else 0
    return real_read(self, *a, **k)


Path.read_text = spy
try:
    for label, kw in (("an unregistered tag", {"model_tag": "evil"}), ("a registered adapter tag with a WRONG adapter hash", {"model_tag": "m_adapter", "adapter_sha256": "BBB"}),
                      ("a registered adapter tag with NO adapter hash", {"model_tag": "m_adapter"}), ("a registered geometry tag with a wrong configuration hash", {"model_tag": "m_geom", "config_hash": "XXX"}),
                      ("a registered geometry tag with no configuration hash", {"model_tag": "m_geom"})):
        try:
            store().labels(final_evaluation=True, **kw)
            check(f"refused before any label is read: {label}", False)
        except C.CameraBenchNotRegistered:
            check(f"refused before any label is read: {label}", seen["label_reads"] == 0 and not led.exists(), "label file not read, ledger untouched")
    # a locally edited (uncommitted) freeze record cannot register a new tag
    freeze.write_text(json.dumps({"registered_models": {**reg, "evil": {}}}))
    try:
        store().labels(final_evaluation=True, model_tag="evil")
        check("an uncommitted local edit of the freeze record cannot register a tag", False)
    except C.CameraBenchNotRegistered:
        check("an uncommitted local edit of the freeze record cannot register a tag", seen["label_reads"] == 0 and not led.exists(), "the committed version is used")
    git(repo, "checkout", "--", ".")
    r = store().labels(final_evaluation=True, model_tag="m_adapter", adapter_sha256="AAA", code_sha="s")
    check("a registered tag with the right adapter hash opens the labels, ledger line written BEFORE the label file is read", len(r) == 3 and seen["ledger_lines_when_labels_read"] == 1 and r[0]["stem"] == "AAA.0.0", str(seen))
finally:
    Path.read_text = real_read
try:
    store().labels(final_evaluation=True, model_tag="m_adapter", adapter_sha256="AAA")
    check("a second final evaluation of the same model tag is refused", False)
except C.CameraBenchAlreadyEvaluated:
    check("a second final evaluation of the same model tag is refused", len(led.read_text().splitlines()) == 1, "ledger still has 1 line")
store().labels(final_evaluation=True, model_tag="m_geom", config_hash="CCC")
store().labels(final_evaluation=True, model_tag="m_plain")
check("other registered tags (matching configuration hash; a tag without hashes) each get their own ledger line", len(led.read_text().splitlines()) == 3)
m = C.map_labels(["dolly-in", "pedestal-down", "truck-left", "arc-CCW", "roll-CW", "side-tracking", "unsteady", "regular-speed", "zoom-out", "no-motion"])
check("label mapping: dolly-in/pedestal/truck/arc/roll map to atoms, tracking is a flag, stability/speed tags are ignored, no-motion is NOT static in the main mapping",
      m == {"atoms": ["arc", "boom-down", "move-left", "push-in", "roll", "zoom-out"], "flags": ["track"], "ignored": ["no-motion", "regular-speed", "unsteady"]}, str(m))
check("the declared sensitivity variant also maps no-motion to static", C.map_labels(["no-motion"], C.STATIC_VARIANT)["atoms"] == ["static"])
guard_ok = C.require_registered("m_geom", reg, config_hash="CCC") == {"config_hash": "CCC"}
for t, k in (("nope", {}), ("m_adapter", {"adapter_sha256": "no"}), ("m_geom", {})):
    try:
        C.require_registered(t, reg, **k)
        guard_ok = False
    except C.CameraBenchNotRegistered:
        pass
check("require_registered is a pure guard over a registry dict (unknown tag, wrong adapter hash, missing configuration hash all raise)", guard_ok)
real = C.CameraBenchTestStore()
n = len(real.video_files())
real_ledger_before = C.LEDGER.read_text() if C.LEDGER.exists() else ""
real_reads = {"n": 0}
Path.read_text = lambda self, *a, **k: (real_reads.__setitem__("n", real_reads["n"] + (self.name == "test.jsonl")) or real_read(self, *a, **k))
try:
    try:
        real.labels(final_evaluation=True, model_tag="move_vlm_q3vl2b_lora_v1", adapter_sha256="0" * 64)
        refused = False
    except (C.CameraBenchNotRegistered, C.CameraBenchAlreadyEvaluated):
        refused = True
finally:
    Path.read_text = real_read
check("the REAL store lists the downloaded GIFs by name; with no committed freeze record it refuses every tag, reads no label and leaves the ledger untouched",
      n == 1071 and len(set(real.source_ids())) == 404 and refused and real_reads["n"] == 0 and (C.LEDGER.read_text() if C.LEDGER.exists() else "") == real_ledger_before
      and not any("camerabench" in x for x in real_ledger_before.splitlines()), f"{n} GIFs, {len(set(real.source_ids()))} source ids, label reads {real_reads['n']}")
(ROOT / "results/phase4/camerabench_lock_selfcheck.md").write_text("# CameraBench sealed-store self-check (synthetic copy with a planted label marker and a synthetic git repository for the freeze record; CPU)\n\n| check | result | detail |\n|---|---|---|\n" + "\n".join(
    f"| {n_} | **{'PASS' if ok else 'FAIL'}** | {d.replace('|', '/')} |" for n_, ok, d in rows) + f"\n\nAll {len(rows)} checks passed: {all(r_[1] for r_ in rows)}.\n")
raise SystemExit(0 if all(r_[1] for r_ in rows) else 1)
