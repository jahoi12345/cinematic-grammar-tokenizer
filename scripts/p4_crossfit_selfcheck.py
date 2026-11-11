"""CPU self-check of the cross-fitting bookkeeping fixed by Amendment 2 (no labels beyond source ids of TRAIN clips, no model): fold assignment, stage split, sample order, learning-rate
schedule, epoch rule, the stage-wise disjointness of train and held-out sources. Writes results/phase4/crossfit_selfcheck.md. Usage: uv run python scripts/p4_crossfit_selfcheck.py"""
import json
import math
from pathlib import Path

from cgt.p4_camera_move import crossfit as X
from cgt.p4_camera_move.labels import load_shotqa_movement_meta

ROOT = Path(__file__).resolve().parents[1]
rows = []


def check(name, ok, detail=""):
    rows.append((name, ok, detail))
    print(("PASS" if ok else "FAIL"), name, detail, flush=True)


man = json.loads((ROOT / "results/phase4/split_manifest.json").read_text())
meta = {m["clip"]: m["source"] for m in load_shotqa_movement_meta()}
train = [{"clip": c, "source": meta[c]} for c in man["clips"]["train"]]
fa = X.fold_assignment([r["source"] for r in train])
check("fold assignment is deterministic and covers every train source once", fa == X.fold_assignment(sorted([r["source"] for r in train], reverse=True)) and len(fa) == len({r["source"] for r in train}), f"{len(fa)} sources")
sizes = [sum(v == k for v in fa.values()) for k in range(3)]
check("folds hold 37 / 37 / 36 sources (rank r goes to fold r mod 3)", sizes == [37, 37, 36] or sorted(sizes) == [36, 37, 37], str(sizes))
ok = True
for k in range(3):
    tr, ho = X.stage_split(train, f"fold{k}")
    ok &= not ({r["source"] for r in tr} & {r["source"] for r in ho}) and len(tr) + len(ho) == len(train) and {r["source"] for r in ho} == {s for s, f in fa.items() if f == k}
check("every fold: train and held-out sources are disjoint, together all train clips, held-out = the fold's sources", ok)
held = [r["clip"] for k in range(3) for r in X.stage_split(train, f"fold{k}")[1]]
check("every train clip is held out in exactly one fold", sorted(held) == sorted(r["clip"] for r in train) and len(held) == len(set(held)))
tr, ho = X.stage_split(train, "final")
check("the final fit trains on all clips and holds out none", len(tr) == len(train) and not ho)
o1, o2, o3 = X.epoch_order(100, "fold0", 1), X.epoch_order(100, "fold0", 2), X.epoch_order(100, "fold1", 1)
check("sample order: a permutation, reproducible, different for another epoch and another stage", sorted(o1) == list(range(100)) and (o1 == X.epoch_order(100, "fold0", 1)).all() and not (o1 == o2).all() and not (o1 == o3).all())
check("stage seeds: 0 for the final fit, k + 1 for fold k", X.stage_seed("final") == 0 and [X.stage_seed(f"fold{k}") for k in range(3)] == [1, 2, 3])
check("updates per epoch / total follow ceil(n / 8) x 3", X.updates_per_epoch(566) == 71 and X.total_updates(566) == 213 and X.total_updates(903) == 3 * 113, f"566 -> {X.total_updates(566)}, 903 -> {X.total_updates(903)}")
T = X.total_updates(903)
check("learning rate: warm-up to the peak in 10 updates, cosine to the end value at the last update", math.isclose(X.lr_at(1, T), 1e-5) and math.isclose(X.lr_at(10, T), 1e-4) and math.isclose(X.lr_at(T, T), 1e-5, rel_tol=1e-9)
      and X.lr_at(11, T) <= 1e-4 and all(X.lr_at(u + 1, T) <= X.lr_at(u, T) + 1e-15 for u in range(10, T)), f"lr(1)={X.lr_at(1, T):.1e} lr(10)={X.lr_at(10, T):.1e} lr(last)={X.lr_at(T, T):.1e}")
E, acc = X.choose_epoch({1: [True] * 6 + [False] * 4, 2: [True] * 7 + [False] * 3, 3: [True] * 7 + [False] * 3})
check("epoch rule: highest pooled OOF accuracy, ties to the smaller epoch", E == 2 and acc[2] == acc[3], f"E={E}, acc={acc}")
E, _ = X.choose_epoch({1: [True] * 5 + [False] * 5, 2: [True] * 5 + [False] * 5, 3: [True] * 5 + [False] * 5})
check("epoch rule: all equal gives epoch 1", E == 1)
E, _ = X.choose_epoch({1: [True] * 5 + [False] * 5, 2: [True] * 3 + [False] * 7, 3: [True] * 8 + [False] * 2})
check("epoch rule: a clear winner is chosen", E == 3)
check("the real cross-fit sizes (903 clips with an atom are not needed here): the train clips of the manifest are 914 in 110 sources", len(train) == 914 and len(fa) == 110, f"{len(train)} clips, {len(fa)} sources")
# ---- Amendment 8: V1 (every overlapping update, relative 1e-3) and V2 (stopping-rule grace across restarts): the REAL trainer fragments are executed on synthetic state
import ast
import math
from types import SimpleNamespace

import numpy as np

tree = ast.parse((ROOT / "scripts/p4_video_train.py").read_text())
run_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
resume_node = next(n for n in ast.walk(run_fn) if isinstance(n, ast.If) and ast.unparse(n.test).startswith("prev is not None"))
stop_node = next(n for n in ast.walk(run_fn) if isinstance(n, ast.If) and ast.unparse(n.test).startswith("not math.isfinite(ul)"))
init_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "n_micro_total" for t in n.targets))


def run_node(node, **state):
    def exit_code(code):
        raise SystemExit(code)
    ns = {"sys": SimpleNamespace(exit=exit_code), "np": np, "math": math, "append_jsonl": lambda *a, **k: None, "write_status": lambda *a, **k: None, "log": lambda *a: None,
          "logd": Path("synthetic"), "a": SimpleNamespace(tag="synthetic"), "stage": "fold2", "total": 200, "micro": 120, "last_ck": "synthetic", **state}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "fragment", "exec"), ns)  # noqa: S102
    return ns


def exit_of(node, **state):
    try:
        run_node(node, **state)
        return None
    except SystemExit as e:
        return e.code


check("V1: the trainer's resume check accepts a repeated update within 1e-3 and ignores an update with no earlier record", exit_of(resume_node, resumed_at=100, update=101, prev=1.0, ul=1.0005) is None and exit_of(resume_node, resumed_at=100, update=101, prev=None, ul=5.0) is None)
check("V1: it REJECTS 1.0 -> 1.002 (0.2%) with exit 13 at update +1, +4 and +8 (no limit to the first updates after a resume)", all(exit_of(resume_node, resumed_at=100, update=100 + k, prev=1.0, ul=1.002) == 13 for k in (1, 4, 8)))
check("V2: the stage-total micro-step counter is initialised from the resume state, (epoch - 1) * n + micro", run_node(init_node, st={"epoch": 2, "micro": 100}, n=566)["n_micro_total"] == 666 and run_node(init_node, st={"epoch": 1, "micro": 0}, n=566)["n_micro_total"] == 0)
n_total = run_node(init_node, st={"epoch": 2, "micro": 100}, n=566)["n_micro_total"]
check("V2: a process resumed at epoch 2 / micro 100 stops (exit 3) once its refilled window of 50 losses of 6.0 is complete", exit_of(stop_node, update=107, ul=6.0, n_micro_total=n_total + 50, micro_win=[6.0] * 50) == 3)
check("V2: the old per-process counter (0 + 50) would NOT have stopped it, and nothing stops inside the first 100 micro-steps, with a short window or a low mean; a non-finite loss always stops (exit 3)",
      exit_of(stop_node, update=107, ul=6.0, n_micro_total=50, micro_win=[6.0] * 50) is None and exit_of(stop_node, update=1, ul=6.0, n_micro_total=99, micro_win=[6.0] * 50) is None
      and exit_of(stop_node, update=1, ul=6.0, n_micro_total=500, micro_win=[6.0] * 49) is None and exit_of(stop_node, update=1, ul=1.0, n_micro_total=500, micro_win=[1.0] * 50) is None
      and exit_of(stop_node, update=1, ul=float("nan"), n_micro_total=1, micro_win=[1.0]) == 3)
(ROOT / "results/phase4/crossfit_selfcheck.md").write_text("# Cross-fitting bookkeeping self-check (CPU; Amendment 2 rules)\n\n| check | result | detail |\n|---|---|---|\n" + "\n".join(
    f"| {n} | **{'PASS' if ok_ else 'FAIL'}** | {d.replace('|', '/')} |" for n, ok_, d in rows) + f"\n\nAll {len(rows)} checks passed: {all(r[1] for r in rows)}.\n")
raise SystemExit(0 if all(r[1] for r in rows) else 1)
