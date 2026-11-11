"""Execute isolated CPU guard AST from the trainer with synthetic state only.

No trainer imports/model construction, real frames, caches or GPU calls occur.
The fragment extraction is harness coupling; behavioral expectations are Amendment7.
"""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def guard_nodes():
    tree = ast.parse((ROOT / "scripts/p4_video_train.py").read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    resume = next(n for n in ast.walk(run) if isinstance(n, ast.If) and ast.unparse(n.test).startswith("prev is not None"))
    stopping = next(n for n in ast.walk(run) if isinstance(n, ast.If) and ast.unparse(n.test).startswith("not math.isfinite(ul)"))
    return resume, stopping


def execute(node, **state):
    def exit_code(code):
        raise SystemExit(code)
    ns = {"sys": SimpleNamespace(exit=exit_code), "np": np, "math": math,
          "append_jsonl": lambda *args, **kw: None, "write_status": lambda *args, **kw: None,
          "log": lambda *args: None, "logd": Path("synthetic"), "a": SimpleNamespace(tag="synthetic"),
          "stage": "fold2", "total": 200, "micro": 120, "last_ck": "synthetic", **state}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "synthetic_guard", "exec"), ns)  # noqa: S102
    return ns


@pytest.mark.parametrize("distance", [1, 4])
def test_resume_relative_tolerance_is_one_per_thousand_for_every_repeated_update(distance):
    resume, _ = guard_nodes()
    with pytest.raises(SystemExit) as exc:
        execute(resume, resumed_at=100, update=100 + distance, prev=1.0, ul=1.002)
    assert exc.value.code == 13


def test_resume_matching_and_within_tolerance_succeed():
    resume, _ = guard_nodes()
    execute(resume, resumed_at=100, update=101, prev=1.0, ul=1.0005)
    execute(resume, resumed_at=100, update=101, prev=None, ul=1.0)


def test_nonfinite_loss_always_stops():
    _, stopping = guard_nodes()
    with pytest.raises(SystemExit) as exc:
        execute(stopping, update=1, ul=float("inf"), n_micro_total=1, micro_win=[1.0])
    assert exc.value.code == 3


def test_resume_does_not_reset_the_first_hundred_microstep_grace_period():
    tree = ast.parse((ROOT / "scripts/p4_video_train.py").read_text())
    initialization = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "n_micro_total" for t in n.targets))
    state = execute(initialization, st={"epoch": 2, "micro": 100}, n=566)
    _, stopping = guard_nodes()
    # The resumed stage already passed 100 microsteps. Fifty new high losses fill
    # the specified window; a restart must not provide another 100-step grace.
    with pytest.raises(SystemExit) as exc:
        execute(stopping, update=107, ul=6.0, n_micro_total=state["n_micro_total"] + 50, micro_win=[6.0] * 50)
    assert exc.value.code == 3


def test_video_validation_records_a_look_before_parsing_validation_labels(tmp_path):
    tree = ast.parse((ROOT / "scripts/p4_video_validate.py").read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                          and x.func.id == "record_look" for x in ast.walk(n)))
    ledger = tmp_path / "look.jsonl"
    events = []

    def record_look(_path, **kwargs):
        ledger.write_text(json.dumps(kwargs) + "\n")
        events.append("look")

    def rows(_clips, *, final_evaluation):
        assert final_evaluation and ledger.exists()
        assert json.loads(ledger.read_text())["kind"] == "video_validation"
        events.append("parse")
        return [{"atoms": ["static"]}]

    state = {"a": SimpleNamespace(test_aid=False, tag="synthetic"), "ROOT": tmp_path,
             "E": 3, "ws": 0.3, "wg": 0.7, "adapter_sha": "0" * 64,
             "store": SimpleNamespace(rows=rows), "man": {"clips": {"val": ["synthetic.mp4"]}},
             "record_look": record_look}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])), "synthetic_validation_look", "exec"), state)  # noqa: S102
    assert events == ["look", "parse"]
