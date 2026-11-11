"""CPU self-check of the label parser, the source-grouped split and the flow cache idempotence (docs/phase4_interface.md sections 1-3). No ShotBench or CameraBench data.
Writes results/phase4/labels_split_selfcheck.md. Usage: uv run python scripts/p4_labels_split_selfcheck.py"""
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from cgt.p4_camera_move import flow as F
from cgt.p4_camera_move.labels import (
    load_shotqa_movement,
    parse_move_label,
    schema_moves,
    source_id,
)
from cgt.p4_camera_move.split import source_grouped_split, write_split_manifest

ROOT = Path(__file__).resolve().parents[1]
rows = []


def check(name, ok, detail=""):
    rows.append((name, ok, detail))
    print(("PASS" if ok else "FAIL"), name, detail, flush=True)


P = parse_move_label
check("simultaneous: Push in and Tilt up", (P("Push in and Tilt up").kind, P("Push in and Tilt up").atoms) == ("simultaneous", ("push-in", "tilt-up")))
s = P("Firstly Static shot, then Push in.")
check("sequential keeps the set and the order", (s.kind, s.atoms, s.order) == ("sequential", ("push-in", "static"), ("static", "push-in")))
t = P("Tracking and Pull out")
check("tracking is a flag, not an atom", (t.atoms, t.flags) == (("pull-out",), ("track",)))
check("rack focus is a flag", P("Pull out and Rack focus").flags == ("rack",) and P("Pull out and Rack focus").atoms == ("pull-out",))
check("trucking = move-left/right", P("Trucking left").atoms == ("move-left",) and P("Trucking right").atoms == ("move-right",))
check("trailing comma and dots are stripped", P("Static shot,").atoms == ("static",) and P("Static shot,").kind == "single")
check("a repeated atom collapses", P("Pan left, then Pan left").atoms == ("pan-left",))
check("typo markers (fisrtly, fristly, finally) are sequential", all(P(x).kind == "sequential" for x in ("Fisrtly Push in, then Pan left", "Fristly Pan left, then Push in", "Finally pan left")))
check("unknown text is reported as unmapped, never guessed", P("Long").atoms == () and P("Long").unmapped == ("long",))
check("camera roll -> roll, dolly zoom -> dolly-zoom", P("Camera roll").atoms == ("roll",) and P("Dolly zoom").atoms == ("dolly-zoom",))
check("source id", source_id("-4dDC0lPRB0.webm_14.mp4") == "-4dDC0lPRB0" and source_id("videoplayback (27).mp4_scene_7_1.mp4") == "videoplayback (27)")
classes = schema_moves()
data = load_shotqa_movement()
check("all parsed atoms are schema classes; 1058 rows; no unmapped part", len(data) == 1058 and all(a in classes for r in data for a in r["atoms"]) and not any(r["unmapped"] for r in data),
      f"{len(data)} rows; flags in {sum(bool(r['flags']) for r in data)}; no schema atom in {sum(not r['atoms'] for r in data)}")
srcs = [r["source"] for r in data]
a, b = source_grouped_split(srcs, 0.15, 0), source_grouped_split(srcs, 0.15, 0)
check("split is deterministic for a fixed seed", a == b)
check("a different seed gives a different split", a != source_grouped_split(srcs, 0.15, 1))
check("every source gets exactly one side; 19 of 129 are validation", set(a) == set(srcs) and sum(v == "val" for v in a.values()) == 19 and len(a) == 129, f"{sum(v == 'val' for v in a.values())} val of {len(a)}")
tmp = Path(tempfile.mkdtemp()) / "m.json"
m = write_split_manifest(data, tmp)
trs, vas = set(m["train_sources"]), set(m["val_sources"])
check("manifest: no source on both sides, every clip on exactly one side", not trs & vas and set(m["clips"]["train"]) | set(m["clips"]["val"]) == {r["clip"] for r in data} and not set(m["clips"]["train"]) & set(m["clips"]["val"]))
committed = json.loads((ROOT / "results/phase4/split_manifest.json").read_text())
check("the committed manifest equals a fresh recomputation", committed == m, f"sha256 {m['sha256'][:16]}")
try:
    source_grouped_split(["only"], 0.15, 0)
    check("a single source cannot be split", False)
except ValueError:
    check("a single source cannot be split", True)
clip = ROOT / "data/external/shotqa" / data[0]["clip"]
cache = Path(tempfile.mkdtemp())
p1 = F.extract_clip(clip, cache)
t0 = p1.stat().st_mtime_ns
time.sleep(0.05)
p2 = F.extract_clip(clip, cache)
check("flow cache is idempotent (second call reuses the file)", p1 == p2 and p2.stat().st_mtime_ns == t0, p1.name)
check("flow cache file name carries the clip name, the code key and the content hash", p1.name == f"{clip.name}.{F.clip_key(clip)}.npz")
keys = ["tx", "ty", "div", "rot", "shear1", "shear2", "r2", "resid_rms", "flow_rms", "res_top_bottom", "res_left_right", "div_spread", "tx_spread"]
doc = [f"{k}_{s_}" for k in keys for s_ in ["mean", "median", "std", "p10", "p90", "sum", "maxabs", "h1_minus_h2"]] + [f"{k}_{s_}" for k in ["tx", "ty", "rot"] for s_ in ["jitter_std", "jitter_hf_share", "jitter_zcr", "diff_std"]] + ["n_pairs", "fps_used", "duration_s"]
ser = {k: np.linspace(0, 1, 30) for k in F.SERIES_KEYS} | {"fps_used": np.array(12.0), "duration_s": np.array(2.5)}
check("the documented 119-column feature schema equals the implementation (names and order)", list(F.clip_features(ser)) == doc and len(doc) == 119, f"{len(doc)} columns")
(ROOT / "results/phase4/labels_split_selfcheck.md").write_text("# Phase 4 labels / split / cache self-check (CPU)\n\n| check | result | detail |\n|---|---|---|\n" + "\n".join(
    f"| {n} | **{'PASS' if ok else 'FAIL'}** | {d} |" for n, ok, d in rows) + f"\n\nAll {len(rows)} checks passed: {all(r[1] for r in rows)}.\n")
raise SystemExit(0 if all(r[1] for r in rows) else 1)
