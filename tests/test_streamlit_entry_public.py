"""The hosted demo's entry point forces public mode and the bundled public data root before the app
runs. In the public tree the file is at the root; in the development repository it is the release overlay."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = next((p for p in (ROOT / "streamlit_app.py", ROOT / "scripts/release/public_overlay/streamlit_app.py") if p.is_file()), None)


def _top_level():
    if ENTRY is None:
        pytest.skip("no streamlit_app.py in this checkout")
    return ast.parse(ENTRY.read_text()).body


def test_entry_forces_public_mode_and_demo_root_before_running_the_app():
    body = _top_level()
    sets, run_at = {}, None
    for i, node in enumerate(body):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
            sets[ast.unparse(node.targets[0])] = (i, ast.unparse(node.value))
        if isinstance(node, ast.Expr) and "run_path" in ast.unparse(node):
            run_at = i
    pub, root = sets.get("os.environ['CGT_DEMO_PUBLIC']"), sets.get("os.environ['CGT_DEMO_ROOT']")
    assert pub == (pub[0], "'1'") if pub else False, "CGT_DEMO_PUBLIC is not set to '1' unconditionally"
    assert root and "demo_data" in root[1]
    assert run_at is not None and pub[0] < run_at and root[0] < run_at


def test_entry_sets_are_unconditional():
    """No if/try around the forcing: both assignments are top-level statements."""
    body = _top_level()
    nested = [n for n in ast.walk(ast.Module(body=[b for b in body if not isinstance(b, ast.Assign)], type_ignores=[]))
              if isinstance(n, ast.Assign) and "CGT_DEMO" in ast.unparse(n)]
    assert nested == []
