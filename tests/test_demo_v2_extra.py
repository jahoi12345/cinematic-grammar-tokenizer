"""an earlier review adversarial probes of the demo v2.0 front end (a recorded decision; docs/demo_spec.md v2.0; review a verification record).
Public-mode reachability of local papers, the internal matched result, overlay reports/essays and data/ files. Synthetic tmp roots and synthetic film ids, except the
real-root public probes (skipped when the main checkout's data root is absent). Skipped on a branch without the v2.0 Reports code."""
import contextlib
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if not (REPO / "app/demo/core.py").exists() or "def release_clean" not in (REPO / "app/demo/core.py").read_text():
    pytest.skip("demo v2.0 not on this branch", allow_module_level=True)

from test_demo_guard import CLIP, NOTLD, OWNED, _texts, core, make_root, run_app
from test_demo_v2_frontend import MARK, _w, add_essay, add_reports

from cgt.worktrees import main_checkout

ESSAY, ADD = "Tokens vs my essay", "Add your own clip"
LOCAL_ONLY_MARKS = ("paper_local", "draft_local", "matched", "clip_report")
REAL = Path(os.environ.get("CGT_VH_REAL_ROOT") or main_checkout())


@pytest.fixture
def root(tmp_path):
    r = make_root(tmp_path, [{"id": NOTLD, "public_domain": True, "title": "Night of the Living Dead", "year": 1968},
                             {"id": OWNED, "public_domain": False, "title": "Zz Owned Title", "year": 2000}],
                  [{"id": CLIP, "public_domain": False, "title": "Qq Clip Title", "year": 2019}])
    return add_essay(add_reports(r))


def _leaks(text: str) -> list[str]:
    low = text.lower()
    hits = [s for s in (OWNED, CLIP, "zz owned title", "qq clip title", "essayclaimone", core.CAPTION_ESSAY.lower()) if s in low]
    return hits + [k for k in LOCAL_ONLY_MARKS if MARK[k] in text]


def _rendered(at) -> str:
    """Text of every rendered element's proto only (AppTest echoes a test-injected session_state value through widget `.value`, which is not what the app shows)."""
    out = []

    def walk(node):
        proto = getattr(node, "proto", None)
        if proto is not None:
            out.append(str(proto))
        for c in getattr(node, "children", {}).values():
            walk(c)
    walk(at._tree)
    return "\n".join(out)


def _fresh(monkeypatch, root, public=True):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("CGT_DEMO_ROOT", str(root))
    monkeypatch.setenv("CGT_DEMO_PUBLIC", "1" if public else "0")
    return testing.AppTest.from_file(str(REPO / "app/demo/app.py"), default_timeout=120)


# ---- read_file / read paths ------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rel", ["data/reports/paper_full.html", "data/reports/paper_full.pdf", "docs/writeup/draft.md", "results/p3m_matched/shotbench_matched_v1.md",
                                 f"data/processed/{CLIP}_report.md", f"data/processed/{CLIP}_essay_claims.json", f"data/processed/{NOTLD}_tokens_v1.parquet",
                                 "configs/films.local.yaml", "models/x.bin"])
def test_public_allowed_lists_never_contain_local_files(root, rel):
    clean = core.release_clean(root, core.public_candidates(root))
    allowed = [x for x in core.paper_files(root, True, clean).values() if x] + core.report_list(root, True, clean)
    assert rel not in allowed
    with pytest.raises(PermissionError):
        core.read_file(root, rel, True, allowed)


@pytest.mark.parametrize("trick", ["./docs/phase6_report.md", "docs//phase6_report.md", "docs/./phase6_report.md", "Docs/phase6_report.md", "docs/phase6_report.md ",
                                   "docs/../data/reports/paper_full.html", "../root/docs/phase6_report.md"])
def test_read_file_path_tricks_not_in_allowed(root, trick):
    allowed = core.report_list(root, True)
    with pytest.raises(PermissionError):
        core.read_file(root, trick, True, allowed)


def test_read_file_absolute_path_refused(root, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.md").write_text("OTHERMARK")
    rel = str(other / "x.md")
    with pytest.raises(PermissionError):
        core.read_file(root, rel, True, [rel])  # an absolute path outside ROOT, even if someone put it in the allowed list


@pytest.mark.parametrize("target", ["data/reports/paper_full.html", "results/p3m_matched/shotbench_matched_v1.md", f"data/processed/{CLIP}_report.md"])
def test_file_symlink_at_report_path_is_dropped_by_checker(root, target):
    p = root / "docs/phase6_report.md"
    p.unlink()
    p.symlink_to(root / target)
    assert "docs/phase6_report.md" not in core.report_list(root, True)


def run_app_public_report(root, rel):
    testing = pytest.importorskip("streamlit.testing.v1")
    os.environ["CGT_DEMO_ROOT"], os.environ["CGT_DEMO_PUBLIC"] = str(root), "1"
    try:
        at = testing.AppTest.from_file(str(REPO / "app/demo/app.py"), default_timeout=120)
        at.run()
        at.sidebar.radio(key="page").set_value("Reports").run()
        with contextlib.suppress(Exception):
            at.selectbox(key="rep_sel").set_value(rel).run()
        return at
    finally:
        os.environ.pop("CGT_DEMO_ROOT", None)
        os.environ.pop("CGT_DEMO_PUBLIC", None)


def test_directory_symlink_into_data_is_never_shown_in_public(root):
    """A phase-report directory that is a symlink into data/ (path component, not the file itself): the file passes the checker by content, so read_file must refuse it."""
    secret = root / "data/secret"
    secret.mkdir(parents=True)
    (secret / "similar_shots_night_of_the_living_dead.md").write_text("# DATASECRETMARK benign words only\n")
    (root / "results/phase6").rename(root / "results/phase6_real")
    (root / "results/phase6").symlink_to(secret)
    rel = "results/phase6/similar_shots_night_of_the_living_dead.md"
    allowed = core.report_list(root, True)
    shown = _rendered(run_app_public_report(root, rel))
    assert "DATASECRETMARK" not in shown
    with contextlib.suppress(PermissionError):
        data = core.read_file(root, rel, True, allowed)
        assert b"DATASECRETMARK" not in data, "a data/ file was read in public mode through a symlinked directory"


# ---- checker filter -------------------------------------------------------------------------------------------------------------------------------------

def test_checker_error_shows_no_paper_and_no_report(monkeypatch, root):
    from cgt.release import check
    monkeypatch.setattr(check, "check_tree", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    at = run_app(monkeypatch, root, True, "Reports")
    text = _texts(at)
    assert MARK["draft_public"] not in text and MARK["phase6"] not in text and _leaks(text) == []
    assert len(at.get("download_button")) == 0 and "not available in this build" in text


def test_checker_registry_missing_shows_nothing(monkeypatch, root):
    from cgt.release import check
    monkeypatch.setattr(check, "load_registry", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no registry")))
    assert core.release_clean(root, core.public_candidates(root)) == set()


def test_matched_never_public_even_if_checker_passes_everything(monkeypatch, root):
    monkeypatch.setattr(core, "release_clean", lambda rt, rels: set(rels) | {"results/p3m_matched/shotbench_matched_v1.md", "data/reports/paper_full.html"})
    assert "results/p3m_matched/shotbench_matched_v1.md" not in core.report_list(root, True)
    assert core.paper_files(root, True)["main"] in (None, "docs/writeup/paper_public.html", "docs/writeup/draft_public.md")  # v2.0 fix: no draft fallback
    text = _texts(run_app(monkeypatch, root, True, "Reports"))
    assert _leaks(text) == []


def test_public_paper_never_carries_the_internal_matched_result(monkeypatch, root):
    """A recorded decision: the public paper has no internal result. A draft_public.md that still carries the decision-0018 internal block must not be shown publicly."""
    _w(root, "docs/writeup/draft_public.md", "# Paper\nPUBLICDRAFTMARK\n**Matched-protocol check.** *Internal until the user releases it* "
       "our mean is 99.9% [results/p3m_matched/shotbench_matched_v1.md]\n")
    text = _texts(run_app(monkeypatch, root, True, "Reports"))
    assert "p3m_matched" not in text and "99.9%" not in text


# ---- page reachability ----------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key,val", [("page", ESSAY), ("page", ADD), ("rep_sel", "results/p3m_matched/shotbench_matched_v1.md"),
                                     ("rep_sel", f"data/processed/{CLIP}_report.md"), ("rep_sel", "data/reports/paper_full.html"), ("es_film", CLIP),
                                     ("ov_film", OWNED), ("ex_film", CLIP)])
def test_public_session_state_tricks(monkeypatch, root, key, val):
    at = _fresh(monkeypatch, root)
    at.session_state[key] = val
    with contextlib.suppress(Exception):
        at.run()
    for p in ("Home", "Reports", "Film overview", "Shot explorer"):
        with contextlib.suppress(Exception):
            at.session_state[key] = val
            at.sidebar.radio(key="page").set_value(p).run()
        assert _leaks(_rendered(at)) == [], (key, val, p)


@pytest.mark.parametrize("qp", [{"page": ESSAY}, {"page": "Reports", "rep_sel": "results/p3m_matched/shotbench_matched_v1.md"}, {"film": CLIP}])
def test_public_query_params(monkeypatch, root, qp):
    at = _fresh(monkeypatch, root)
    for k, v in qp.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception and ESSAY not in at.sidebar.radio(key="page").options and _leaks(_texts(at)) == []


@pytest.mark.parametrize("page,extra", [(ESSAY, {}), ("Reports", {"rep_sel": "results/p3m_matched/shotbench_matched_v1.md"}), ("Shot explorer", {"ex_film": CLIP})])
def test_local_session_flipped_to_public(monkeypatch, root, page, extra):
    at = _fresh(monkeypatch, root, public=False)
    at.run()
    at.sidebar.radio(key="page").set_value(page).run()
    for k, v in extra.items():
        with contextlib.suppress(Exception):
            (at.selectbox(key=k) if k == "rep_sel" else at.sidebar.selectbox(key=k)).set_value(v).run()
    assert _leaks(_texts(at))  # positive control: local mode does show local content
    monkeypatch.setenv("CGT_DEMO_PUBLIC", "1")
    with contextlib.suppress(Exception):
        at.run()
    assert _leaks(_texts(at)) == []


def test_films_hub_public_cards_and_buttons(monkeypatch, root):
    at = run_app(monkeypatch, root, True, "Films")
    keys = [b.key for b in at.get("button")]
    assert keys and all(NOTLD in k for k in keys if k.startswith("films_")) and not [k for k in keys if OWNED in k or CLIP in k]


# ---- local mode (AppTest only): overlay banner and badges --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("page,sel", [("Film overview", {"ov_film": CLIP}), ("Shot explorer", {"ex_film": CLIP}), ("Side by side", {"sb_a": CLIP}),
                                      ("Side by side", {"sb_a": NOTLD, "sb_b": CLIP}), (ESSAY, {})])
def test_local_overlay_banner(monkeypatch, root, page, sel):
    assert core.BANNER_LOCAL_ONLY in _texts(run_app(monkeypatch, root, False, page, **sel))


def test_local_films_badges_exactly_non_public(monkeypatch, root):
    at = run_app(monkeypatch, root, False, "Films")
    badges = [e.value for e in at.error if "LOCAL ONLY" in e.value]
    assert len(badges) == 2  # owned + overlay, not NOTLD


# ---- real root, public mode only --------------------------------------------------------------------------------------------------------------------------

def _real_or_skip():
    if not (REAL / "data/processed").exists() or not (REAL / "configs/films.yaml").exists():
        pytest.skip("real data root not present")


@pytest.mark.parametrize("page", ["Home", "Films", "Reports", "About"])
def test_real_root_public_pages(monkeypatch, page):
    _real_or_skip()
    from test_deown_checks import _owned
    _, terms = _owned()
    text = re.sub(r"\$\$id-[0-9a-f]+-", "", _texts(run_app(monkeypatch, REAL, True, page)).lower())  # widget id hashes are not page text
    assert [t for t in terms if t.lower() in text] == []
    # a recorded decision lifted a recorded decision for publication: the public paper (shown on Reports) carries the official-protocol (matched) ShotBench
    # results with their caveats, so their numbers and the source folder name may appear THERE. Everywhere else they stay banned, and on every
    # page the internal 2B report file (shotbench_matched_v1) and the internal banner never appear; owned/clip terms are checked above.
    published = ("p3m_matched", "73.26", "71.73", "74.81", "73.3% [71.7") if page == "Reports" else ()
    for bad in ("p3m_matched", "matched_v1", "73.26", "71.73", "74.81", "73.3% [71.7", "internal until the user releases it"):
        if bad in published:
            continue
        assert bad not in text, (page, bad)
    if page == "Reports":
        at = run_app(monkeypatch, REAL, True, page)
        opts = list(at.selectbox(key="rep_sel").options) if [w for w in at.selectbox if w.key == "rep_sel"] else []
        assert not any(o.startswith(core.PUBLIC_DENY) or o in core.MATCHED_REPORTS for o in opts)


@pytest.mark.parametrize("rel", ["DATA/reports/paper_full.html", "Data/reports/paper_full.html", "RESULTS/p3m_matched/shotbench_matched_v1.md"])
def test_read_file_case_variant_of_denied_path(root, rel):
    """On a case-insensitive file system a case variant names the same file; read_file must refuse it in public mode even if it were in the allowed list."""
    if not (root / rel).exists():
        pytest.skip("case-sensitive file system")
    with pytest.raises(PermissionError):
        core.read_file(root, rel, True, [rel])


def test_unreadable_candidate_is_dropped(root):
    p = root / "docs/phase6_report.md"
    p.chmod(0)
    try:
        assert "docs/phase6_report.md" not in core.report_list(root, True)
    finally:
        p.chmod(0o644)


def test_html_paper_with_internal_marker_is_not_public(root):
    # a recorded decision lifted a recorded decision for publication: a matched-result source name is no longer an internal marker, the banner still is
    _w(root, "docs/writeup/paper_public.html", "<html><body>PUBLICHTMLMARK see results/p3m_matched/val_stage1.md</body></html>")
    assert core.paper_files(root, True)["main"] == "docs/writeup/paper_public.html"
    _w(root, "docs/writeup/paper_public.html", "<html><body>PUBLICHTMLMARK Internal until the user releases it</body></html>")
    assert core.paper_files(root, True)["main"] is None
    _w(root, "docs/writeup/paper_public.html", "<html><body>PUBLICHTMLMARK clean</body></html>")
    assert core.paper_files(root, True)["main"] == "docs/writeup/paper_public.html"


@pytest.mark.parametrize("rel", ["DATA/reports/paper_full.html", "RESULTS/p3m_matched/shotbench_matched_v1.md", "Docs/writeup/draft.md"])
def test_case_variants_never_in_public_lists(root, rel):
    clean = core.release_clean(root, core.public_candidates(root))
    allowed = [x for x in core.paper_files(root, True, clean).values() if x] + core.report_list(root, True, clean)
    assert all(a == a.lower() or a.startswith(("docs/", "results/")) for a in allowed) and rel not in allowed
