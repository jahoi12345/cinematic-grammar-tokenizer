"""an earlier review adversarial probes of the demo app v1.0 surfaces (About, Add-clip, CSV export; docs/demo_spec.md v1.0, review a verification record).
Synthetic tmp roots only, except one public-mode About render against the real registry (CGT_DEMO_ROOT = this checkout's main root when present). Skipped without app/demo v1.0."""
import contextlib
import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if not (REPO / "app/demo/core.py").exists() or "csv_bytes" not in (REPO / "app/demo/core.py").read_text():
    pytest.skip("demo app v1.0 not on this branch", allow_module_level=True)

from test_demo_guard import CLIP, NOTLD, OWNED, _texts, core, make_root, run_app

from cgt.worktrees import main_checkout

OWN_TITLE, CLIP_TITLE, ADD = "Zz Secret Owned Title", "Qq Private Clip Title", "Add your own clip"


@pytest.fixture
def root(tmp_path):
    return make_root(tmp_path, [{"id": NOTLD, "public_domain": True, "title": "Night of the Living Dead", "year": 1968},
                                {"id": OWNED, "public_domain": False, "title": OWN_TITLE, "year": 1999}],
                     [{"id": CLIP, "public_domain": True, "title": CLIP_TITLE, "year": 2026}])


def _secret_hits(text: str) -> list[str]:
    low = text.lower()
    return [s for s in (OWNED, CLIP, OWN_TITLE.lower(), CLIP_TITLE.lower(), "owned_film") if s in low]


@pytest.mark.parametrize("page", ["About", "Film overview", "Shot explorer", "Side by side"])
def test_public_pages_never_name_owned_or_overlay_titles(monkeypatch, root, page):
    assert _secret_hits(_texts(run_app(monkeypatch, root, True, page))) == []


def test_local_titles_are_used_so_the_public_probe_is_meaningful(monkeypatch, root):
    text = _texts(run_app(monkeypatch, root, False, "About"))
    assert OWN_TITLE in text and CLIP_TITLE in text  # the same render path does show titles when allowed


def _fresh(monkeypatch, root, public=True):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("CGT_DEMO_ROOT", str(root))
    if public:
        monkeypatch.setenv("CGT_DEMO_PUBLIC", "1")
    else:
        monkeypatch.delenv("CGT_DEMO_PUBLIC", raising=False)
    return testing.AppTest.from_file(str(REPO / "app/demo/app.py"), default_timeout=120)


@pytest.mark.parametrize("val", [ADD, "add your own clip", "Add your own clip ", 4, -1])
def test_add_clip_unreachable_via_session_state(monkeypatch, root, val):
    at = _fresh(monkeypatch, root)
    at.session_state["page"] = val
    with contextlib.suppress(Exception):  # refusing (raising) is closed too
        at.run()
    text = _texts(at)
    assert "tokenize_video.py" not in text and "films.local.yaml" not in text and _secret_hits(text) == []


@pytest.mark.parametrize("qp", [{"page": ADD}, {"page": "4"}, {"embed_options": "show_toolbar"}])
def test_add_clip_unreachable_via_query_params(monkeypatch, root, qp):
    at = _fresh(monkeypatch, root)
    for k, v in qp.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception
    assert at.sidebar.radio(key="page").value in ("About", "Home") and ADD not in at.sidebar.radio(key="page").options
    assert "tokenize_video.py" not in _texts(at)


def test_add_clip_state_carried_from_local_session_does_not_render_in_public(monkeypatch, root):
    at = _fresh(monkeypatch, root, public=False)
    at.run()
    at.sidebar.radio(key="page").set_value(ADD).run()
    assert "tokenize_video.py" in _texts(at)
    monkeypatch.setenv("CGT_DEMO_PUBLIC", "1")  # same session, env flipped to public before the next rerun
    with contextlib.suppress(Exception):  # refusing (raising) is closed too
        at.run()
    assert "tokenize_video.py" not in _texts(at)


def test_no_uploader_no_exec_in_app_sources():
    src = "\n".join((REPO / "app/demo" / f).read_text() for f in os.listdir(REPO / "app/demo") if f.endswith(".py"))
    for bad in ("file_uploader", "subprocess", "os.system", "os.popen", "Popen", "exec(", "eval(", "camera_input", "chat_input", "to_csv(\"", ".write_bytes", ".write_text"):
        assert bad not in src, bad


@pytest.mark.parametrize("fid", [OWNED, CLIP, "unregistered", f"{NOTLD}/../{OWNED}", f"../{OWNED}", NOTLD.upper(), NOTLD + " ", "/" + NOTLD, ""])
def test_csv_bytes_refuses_non_public_ids_before_touching_rows(root, fid):
    class Boom(pd.DataFrame):
        def __getitem__(self, k):
            raise AssertionError("rows touched before the guard")
    with pytest.raises(PermissionError):
        core.csv_bytes(root, fid, True, Boom({"film_id": [fid]}))


@pytest.mark.parametrize("bad", [[OWNED], [NOTLD, OWNED], [NOTLD, None], [NOTLD, np.nan], [NOTLD, NOTLD + " "], [NOTLD, NOTLD.upper()]])
def test_csv_bytes_refuses_foreign_or_unlabelled_rows(root, bad):
    df, _ = core.load_table(root, NOTLD, True)
    df = df.iloc[: len(bad)].copy()
    df["film_id"] = bad
    with pytest.raises(ValueError):
        core.csv_bytes(root, NOTLD, True, df)


def test_csv_public_export_of_every_row_has_no_path_and_no_secret(root):
    df, _ = core.load_table(root, NOTLD, True)
    df["extra_obj"] = [np.array([f"data/interim/{NOTLD}/keyframes/x.jpg"])] * len(df)  # a future array column holding paths would leak: documents the rule
    raw = core.csv_bytes(root, NOTLD, True, df.drop(columns=["extra_obj"])).decode()
    assert "keyframes" not in raw and ".jpg" not in raw and "data/" not in raw and _secret_hits(raw) == []
    assert set(pd.read_csv(io.StringIO(raw)).columns) == set(df.columns) - {"kf_start", "kf_mid", "kf_end", "extra_obj"}


def test_csv_in_public_app_after_switching_from_local_state(monkeypatch, root):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.runtime.media_file_manager import MediaFileManager
    got, orig = [], MediaFileManager.add

    def add(self, data, mimetype, *a, **k):
        if mimetype == "text/csv":
            got.append(data)
        return orig(self, data, mimetype, *a, **k)
    monkeypatch.setattr(MediaFileManager, "add", add)
    at = run_app(monkeypatch, root, False, "Shot explorer", ex_film=OWNED)
    monkeypatch.setenv("CGT_DEMO_PUBLIC", "1")  # stale selection of the owned film in session state, then public
    got.clear()
    with contextlib.suppress(Exception):  # refusing (raising) is closed too
        at.run()
    for data in got:
        assert set(pd.read_csv(io.BytesIO(data))["film_id"]) <= {NOTLD} and _secret_hits(data.decode()) == []
    assert _secret_hits(_texts(at)) == []


def test_about_public_against_real_registry(monkeypatch):
    real = Path(os.environ.get("CGT_VH_REAL_ROOT") or main_checkout())
    if not (real / "data/processed").exists() or not (real / "configs/films.yaml").exists():
        pytest.skip("real data root not present")
    from test_deown_checks import _owned
    _, terms = _owned()
    at = _fresh(monkeypatch, real)
    at.run()
    assert not at.exception and at.sidebar.radio(key="page").value in ("About", "Home")  # default page: About (v1.0), Home (v2.0)
    at.sidebar.radio(key="page").set_value("About").run()  # this is the About probe, whatever the default page is
    assert not at.exception
    low = re.sub(r"\$\$id-[0-9a-f]+-", "", _texts(at).lower())  # widget id hashes are not page text (a hash can contain "607")
    assert [t for t in terms if t.lower() in low] == []
    assert "410" not in low and "67.5" not in low and "607" not in low
    assert "night of the living dead (1968)" in low
