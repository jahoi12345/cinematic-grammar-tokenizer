"""F1: selection resume keys ((id, attribute) for method scores, subset position for the native reference)."""
import importlib.util
import json
from pathlib import Path

from cgt.p3c_tokens import resume

REPO = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(f"p3c_{name}", REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_selection_resume_keys_are_id_and_attribute(tmp_path):
    p = tmp_path / "D.jsonl"
    p.write_text(json.dumps({"id": "x#0", "attr": "size", "s": [1.0]}) + "\n" + json.dumps({"id": "x#0", "attr": "light_type", "s": [2.0]}) + "\n")
    done = _script("p3c_select_method").load_done(p)
    assert set(done) == {("x#0", "size"), ("x#0", "light_type")}


def test_native_resume_uses_subset_position_not_id(tmp_path):
    samples = [{"id": "a#0", "attribute": "size"}, {"id": "b#0", "attribute": "lens"}, {"id": "b#0", "attribute": "lens"}, {"id": "c#0", "attribute": "angle"}]
    p = tmp_path / "native.jsonl"
    rows = [{"id": "a#0", "attr": "size", "logits": [0] * 4},            # legacy record, unique pair -> position 0
            {"id": "b#0", "attr": "lens", "logits": [0] * 4},            # legacy record of a repeated pair -> ambiguous, not counted
            {"id": "b#0", "attr": "lens", "pos": 2, "q": "h", "logits": [0] * 4}]  # new record carries its position
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert resume.native_done_positions(p, samples) == {0, 2}
    assert resume.question_hash("a") != resume.question_hash("b") and len(resume.question_hash("a")) == 12


def test_resume_loader_repairs_a_torn_last_line_before_append(tmp_path):
    p = tmp_path / "A2m.jsonl"
    good = json.dumps({"id": "x#0", "attr": "size", "s": [1.0]}) + "\n"
    p.write_text(good + '{"id": "y#0", "att')
    assert set(resume.load_done(p)) == {("x#0", "size")} and p.read_text() == good
