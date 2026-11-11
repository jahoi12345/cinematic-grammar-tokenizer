"""Demo v2.0: Home, Films hub, Compare label, Reports (paper + phase reports, release-checker
filter in public mode), Tokens vs my essay (local only), overlay films. Synthetic tmp roots and synthetic film ids only; never touches film data."""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if not (REPO / "app/demo/core.py").exists():
    pytest.skip("app/demo not on this branch", allow_module_level=True)

from test_demo_guard import CLIP, NOTLD, OWNED, SECRET, _texts, core, make_root, run_app, snapshot

MARK = {"draft_public": "PUBLICDRAFTMARK", "paper_public": "PUBLICHTMLMARK", "draft_local": "LOCALDRAFTMARK", "paper_local": "LOCALPAPERMARK",
        "matched": "MATCHEDRESULTMARK", "clip_report": "CLIPREPORTMARK", "phase6": "PHASESIXMARK", "status": "STATUSMARK"}
ESSAY_REPORT = f"""# {MARK['clip_report']} local report

## Descriptive comparison with the essay notes (not accuracy)

| essay claim | candidate shots | what the tokens say | verdict |
|---|---|---|---|
| chase: handheld tracking | 1-4 (7-30 s) | moves vary | partly representable; mixed movement |
| low-angle shot of the hero | 5 (face in the door), also 9 | 5: angle=low | agree (low angle) |
| with shallow depth of field | same | DoF medium | mostly disagree: rarely shallow |
| slow motion | around 2-3 | nothing | not representable |
| close-up vs wide | whole clip | 7 switches | partly visible |

Summary line.
"""
CLAIMS = {"claims": ["ESSAYCLAIMONE handheld chase", "ESSAYCLAIMTWO low angle"]}


def _w(root: Path, rel: str, text: str) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text)


def add_reports(root: Path, owned_in_status: bool = True) -> Path:
    _w(root, "docs/writeup/draft_public.md", f"# Paper\n{MARK['draft_public']}\n")
    _w(root, "docs/writeup/draft.md", f"# Full paper\n{MARK['draft_local']} {OWNED}\n")
    _w(root, "data/reports/paper_full.html", f"<html><body><h1>{MARK['paper_local']}</h1> {OWNED} {CLIP}</body></html>")
    _w(root, "data/reports/paper_full.pdf", "%PDF-1.4 local\n")
    _w(root, "results/p3m_matched/shotbench_matched_v1.md", f"# {MARK['matched']}\n")
    _w(root, "docs/phase6_report.md", f"# Phase 6\n{MARK['phase6']}\n")
    _w(root, "docs/project_log.md", f"# Status\n{MARK['status']} {OWNED if owned_in_status else 'nothing owned'}\n")  # fails the checker when it names the owned film
    _w(root, f"data/processed/{CLIP}_report.md", ESSAY_REPORT)
    return root


def add_essay(root: Path) -> Path:
    _w(root, f"data/processed/{CLIP}_report.md", ESSAY_REPORT)
    _w(root, f"data/processed/{CLIP}_essay_claims.json", json.dumps(CLAIMS))
    return root


@pytest.fixture
def root(tmp_path):
    r = make_root(tmp_path, [{"id": NOTLD, "public_domain": True, "title": "Night of the Living Dead", "year": 1968},
                             {"id": OWNED, "public_domain": False, "title": "Owned Film X", "year": 2000}],
                  [{"id": CLIP, "public_domain": False, "title": "My Clip Title", "year": 2019}])
    return add_essay(add_reports(r))


def _no_secret(text: str) -> list[str]:
    low = text.lower()
    return [s for s in (*SECRET, "owned film x", "my clip title", core.BANNER_LOCAL_ONLY.lower(), "local only") if s in low]


# ---- C1 Home ---------------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("public", [True, False])
def test_home_first_and_default(monkeypatch, root, public):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("CGT_DEMO_ROOT", str(root))
    monkeypatch.setenv("CGT_DEMO_PUBLIC", "1" if public else "0")
    at = testing.AppTest.from_file(str(REPO / "app/demo/app.py"), default_timeout=120)
    at.run()
    radio = at.sidebar.radio(key="page")
    assert not at.exception and radio.value == "Home" and radio.options[0] == "Home"
    text = _texts(at)
    assert "Night of the Living Dead (1968)" in text
    if public:
        assert _no_secret(text) == []
        assert set(radio.options) == {"Home", "Films", "Film overview", "Shot explorer", "Compare (side by side)", "Reports", "About"}  # displayed labels
    else:
        assert "My Clip Title (2019) (LOCAL ONLY)" in text and "Owned Film X (2000) (LOCAL ONLY)" in text
        assert "Tokens vs my essay" in radio.options and "Add your own clip" in radio.options


def test_compare_label(monkeypatch, root):
    at = run_app(monkeypatch, root, True, "Side by side")
    assert "Compare" in _texts(at) and "Compare (side by side)" in str(at.sidebar.radio(key="page").proto)


def test_home_button_switches_page(monkeypatch, root):
    at = run_app(monkeypatch, root, True, "Home")
    at.button(key="home_Reports").click().run()
    assert at.sidebar.radio(key="page").value == "Reports"
    assert "home_Tokens vs my essay" not in [b.key for b in at.button]


# ---- C2 Films hub ----------------------------------------------------------------------------------------------------------------------------------------

def test_films_public_only_public_domain(monkeypatch, root):
    at = run_app(monkeypatch, root, True, "Films")
    text = _texts(at)
    assert _no_secret(text) == []
    keys = [b.key for b in at.button]
    assert keys == [f"films_ov_{NOTLD}", f"films_ex_{NOTLD}"]
    df, _ = core.load_table(root, NOTLD, True)
    s = core.length_stats(df)
    assert f"{s['shots']:,} shots · {s['running_min']:.1f} min" in text


def test_films_local_cards_and_badges(monkeypatch, root):
    at = run_app(monkeypatch, root, False, "Films")
    assert sorted(b.key for b in at.button if b.key.startswith("films_ov_")) == sorted(f"films_ov_{f}" for f in (NOTLD, OWNED, CLIP))
    assert [e.value for e in at.error].count("LOCAL ONLY") == 2  # owned film and overlay clip, not the public-domain film
    assert "My Clip Title (2019)" in _texts(at)


@pytest.mark.parametrize("film", [CLIP, OWNED, NOTLD])
@pytest.mark.parametrize("which,page,key", [("ov", "Film overview", "ov_film"), ("ex", "Shot explorer", "ex_film")])
def test_films_card_opens_page_with_film(monkeypatch, root, which, page, key, film):
    at = run_app(monkeypatch, root, False, "Films")
    at.button(key=f"films_{which}_{film}").click().run()
    assert not at.exception
    assert at.sidebar.radio(key="page").value == page and at.sidebar.selectbox(key=key).value == film
    assert (core.BANNER_LOCAL_ONLY in _texts(at)) is (film != NOTLD)


# ---- C3 overlay film ---------------------------------------------------------------------------------------------------------------------------------------

def test_overlay_listed_local_never_public(root):
    assert CLIP in core.listed_films(root, False) and CLIP not in core.listed_films(root, True)
    assert core.is_local_only(root, CLIP)
    with pytest.raises(PermissionError):
        core.load_table(root, CLIP, True)


@pytest.mark.parametrize("page,sel", [("Film overview", {"ov_film": CLIP}), ("Shot explorer", {"ex_film": CLIP}), ("Side by side", {"sb_a": CLIP}),
                                      ("Tokens vs my essay", {})])
def test_overlay_banner_on_every_page(monkeypatch, root, page, sel):
    at = run_app(monkeypatch, root, False, page, **sel)
    assert core.BANNER_LOCAL_ONLY in _texts(at)


@pytest.mark.parametrize("page", ["Home", "Films", "Film overview", "Shot explorer", "Side by side", "Reports", "About"])
def test_public_pages_never_show_owned_overlay_or_local_reports(monkeypatch, root, page):
    text = _texts(run_app(monkeypatch, root, True, page))
    assert _no_secret(text) == []
    for k in ("paper_local", "draft_local", "matched", "clip_report", "status"):
        assert MARK[k] not in text, (page, k)
    assert "ESSAYCLAIMONE" not in text and core.CAPTION_ESSAY not in text


# ---- C4/C5 Reports ---------------------------------------------------------------------------------------------------------------------------------------

def test_reports_public_paper_and_checker_filter(monkeypatch, root):
    at = run_app(monkeypatch, root, True, "Reports")
    text = _texts(at)
    assert MARK["draft_public"] not in text and MARK["phase6"] in text  # no draft fallback in public mode
    assert "The paper is not available in this build." in text
    sel = at.selectbox(key="rep_sel")
    assert sel.options == ["docs/phase6_report.md"]  # STATUS names the owned film: dropped by the checker; matched and clip reports never listed
    assert [b.label for b in at.get("download_button")] == []


def test_reports_public_prefers_html_paper_that_passes(monkeypatch, root):
    _w(root, "docs/writeup/paper_public.html", f"<html><body>{MARK['paper_public']}</body></html>")
    text = _texts(run_app(monkeypatch, root, True, "Reports"))
    assert MARK["paper_public"] in text and MARK["draft_public"] not in text
    _w(root, "docs/writeup/paper_public.html", f"<html><body>{MARK['paper_public']} {OWNED}</body></html>")  # now fails the checker
    assert core.paper_files(root, True)["main"] is None  # never the draft
    assert "The paper is not available in this build." in _texts(run_app(monkeypatch, root, True, "Reports"))


def test_reports_public_status_listed_when_clean(tmp_path):
    r = add_reports(make_root(tmp_path, [{"id": NOTLD, "public_domain": True}, {"id": OWNED, "public_domain": False}], []), owned_in_status=False)
    assert core.report_list(r, True) == ["docs/phase6_report.md", "docs/project_log.md"]


@pytest.mark.parametrize("marker", core.INTERNAL_MARKERS)
def test_public_never_shows_internal_matched_result(monkeypatch, root, marker):
    """a public candidate carrying a decision-0018 marker is excluded even though the checker passes it."""
    _w(root, "docs/writeup/paper_public.html", f"<html><body>{MARK['paper_public']} see {marker} here</body></html>")
    _w(root, "docs/phase6_report.md", f"# Phase 6\n{MARK['phase6']} {marker}\n")
    from cgt.release import check
    assert check.check_tree(root, check.load_registry([root]), files=["docs/phase6_report.md", "docs/writeup/paper_public.html"]) == []  # the checker alone passes them
    assert core.paper_files(root, True)["main"] is None and core.report_list(root, True) == []
    text = _texts(run_app(monkeypatch, root, True, "Reports"))
    assert MARK["paper_public"] not in text and MARK["phase6"] not in text and marker not in text


def test_internal_marker_unreadable_is_excluded(root, monkeypatch):
    monkeypatch.setattr(core, "safe_path", lambda r, rel: None)
    assert core._carries_internal(root, "docs/phase6_report.md")


def test_directory_symlink_never_read_or_listed(monkeypatch, tmp_path):
    """a report path reached through a symlinked directory (here results/phase6 -> data/secret) is never listed or read, in either mode."""
    r = make_root(tmp_path, overlay=[], frames=False)
    import shutil
    shutil.rmtree(r / "results/phase6")
    _w(r, "data/secret/visual_arc_night_of_the_living_dead.md", "SECRETDATAMARK benign words\n")
    (r / "results/phase6").symlink_to(r / "data/secret", target_is_directory=True)
    rel = "results/phase6/visual_arc_night_of_the_living_dead.md"
    for public in (True, False):
        assert rel not in core.report_list(r, public)
        with pytest.raises(PermissionError):
            core.read_file(r, rel, public, [rel])
    assert "SECRETDATAMARK" not in _texts(run_app(monkeypatch, r, True, "Reports"))


def test_file_symlink_and_absolute_paths_refused(tmp_path):
    r = make_root(tmp_path, overlay=[], frames=False)
    _w(r, "data/secret.md", "SECRET\n")
    (r / "docs").mkdir(exist_ok=True)
    (r / "docs/phase6_report.md").symlink_to(r / "data/secret.md")
    for rel in ("docs/phase6_report.md", str(r / "data/secret.md"), "docs/../data/secret.md"):
        assert core.safe_path(r, rel) is None
        with pytest.raises(PermissionError):
            core.read_file(r, rel, False, [rel])
    assert core.report_list(r, False) == []


def test_read_file_applies_deny_list_to_resolved_path(root, monkeypatch):
    """even if safe_path returned a data/ file for an allowed public path, the resolved path is denied."""
    monkeypatch.setattr(core, "safe_path", lambda rt, rel: Path(rt).resolve() / "data/reports/paper_full.html")
    with pytest.raises(PermissionError):
        core.read_file(root, "docs/phase6_report.md", True, ["docs/phase6_report.md"])


def test_checker_cache_key_includes_size(monkeypatch, root):
    """a content change that keeps the modification time is re-checked because the size changes."""
    pytest.importorskip("streamlit")
    import importlib.util
    import os
    monkeypatch.setenv("CGT_DEMO_ROOT", str(root))
    spec = importlib.util.spec_from_file_location("demo_app_cache_unit", REPO / "app/demo/app.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    f = root / "docs/phase6_report.md"
    assert "docs/phase6_report.md" in m.clean_set(root)
    st = f.stat()
    f.write_text(f.read_text() + f"{OWNED}\n")
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert "docs/phase6_report.md" not in m.clean_set(root)


def _render_essay(root_s: str, app_s: str, public: bool, leak: str):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("demo_app_essay_unit", app_s)  # not __main__: importing renders nothing
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    orig = m.core.essay_films, m.core.load_essay
    m.core.essay_films = lambda rt, pub: [leak]  # even if the essay list (and loader) leaked a film, the page's own public check renders nothing
    m.core.load_essay = lambda rt, f, pub: {"claims": ["ESSAYCLAIMONE"], "rows": []}
    try:
        m.page_essay(Path(root_s), public, [])
    finally:
        m.core.essay_films, m.core.load_essay = orig


@pytest.mark.parametrize("public", [True, False])
def test_page_essay_renders_nothing_in_public_mode(monkeypatch, root, public):
    """page_essay's own public check, independent of the page list."""
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("CGT_DEMO_ROOT", str(root))
    at = testing.AppTest.from_function(_render_essay, args=(str(root), str(REPO / "app/demo/app.py"), public, CLIP), default_timeout=60)
    at.run()
    text = _texts(at)
    if public:
        assert "ESSAYCLAIMONE" not in text and core.CAPTION_ESSAY not in text and CLIP not in text
    else:
        assert core.CAPTION_ESSAY in text


def test_reports_public_list_comes_from_checker(monkeypatch, root):
    calls = []
    monkeypatch.setattr(core, "release_clean", lambda rt, rels: calls.append(sorted(rels)) or set())
    assert core.report_list(root, True) == [] and core.paper_files(root, True) == {"main": None, "pdf": None}
    assert calls and all(not r.startswith(("data/", "results/p3m_matched/")) for c in calls for r in c)


def test_release_clean_fails_closed(monkeypatch, root):
    from cgt.release import check
    monkeypatch.setattr(check, "check_tree", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert core.release_clean(root, ["docs/phase6_report.md"]) == set()


def test_check_tree_files_param(root):
    from cgt.release import check
    reg = check.load_registry([root])
    hits = check.check_tree(root, reg, files=["docs/project_log.md", "docs/phase6_report.md"])
    assert {p for _, p in hits} == {"docs/project_log.md"}


@pytest.mark.parametrize("rel", ["data/reports/paper_full.html", "docs/writeup/draft.md", "results/p3m_matched/shotbench_matched_v1.md",
                                 f"data/processed/{CLIP}_report.md", "data/reports/paper_full.pdf", "configs/films.local.yaml"])
def test_read_file_refuses_local_files_in_public_even_if_allowed(root, rel):
    with pytest.raises(PermissionError):
        core.read_file(root, rel, True, [rel])


def test_read_file_refuses_unlisted_and_escapes(root):
    with pytest.raises(PermissionError):
        core.read_file(root, "docs/project_log.md", False, ["docs/phase6_report.md"])
    with pytest.raises(PermissionError):
        core.read_file(root, "../outside.md", False, ["../outside.md"])
    assert MARK["phase6"].encode() in core.read_file(root, "docs/phase6_report.md", True, ["docs/phase6_report.md"])


def test_reports_local_paper_downloads_and_selector(monkeypatch, root):
    at = run_app(monkeypatch, root, False, "Reports")
    text = _texts(at)
    assert MARK["paper_local"] in text and core.BANNER_LOCAL_REPORT in text
    assert [b.label for b in at.get("download_button")] == ["Download the paper (HTML)", "Download the paper (PDF)"]
    opts = at.selectbox(key="rep_sel").options
    assert "results/p3m_matched/shotbench_matched_v1.md" in opts and f"data/processed/{CLIP}_report.md" in opts and "docs/project_log.md" in opts
    banners = lambda: sum(core.BANNER_LOCAL_REPORT in e.value for e in at.error)
    at.selectbox(key="rep_sel").set_value("docs/phase6_report.md").run()
    assert banners() == 1  # the local paper only
    for rel in ("results/p3m_matched/shotbench_matched_v1.md", f"data/processed/{CLIP}_report.md"):
        at.selectbox(key="rep_sel").set_value(rel).run()
        assert banners() == 2, rel  # the paper and the selected local report
    assert MARK["clip_report"] in _texts(at)


def test_reports_local_falls_back_to_full_draft(monkeypatch, root):
    (root / "data/reports/paper_full.html").unlink()
    text = _texts(run_app(monkeypatch, root, False, "Reports"))
    assert MARK["draft_local"] in text and "showing the draft" in text


# ---- C6 Tokens vs my essay -------------------------------------------------------------------------------------------------------------------------------

def test_parse_shot_cell():
    assert core.parse_shot_cell("1-4 (7-30 s)", None, 10) == ([1, 2, 3, 4], ["7-30 s"])
    assert core.parse_shot_cell("5 (face 2-3 in the door), also 9", None, 10)[0] == [5, 9]
    assert core.parse_shot_cell("same", [5, 9], 10)[0] == [5, 9]
    assert core.parse_shot_cell("whole clip", None, 4)[0] == [0, 1, 2, 3]
    assert core.parse_shot_cell("stabbing of A-b at 46-47; of C-d 97-98", None, 200)[0] == [46, 47, 97, 98]
    assert core.parse_shot_cell("no shots here", None, 10)[0] == []


def test_parse_essay_table_and_verdicts():
    rows = core.parse_essay_table(ESSAY_REPORT, 10)
    assert [r["shot_ids"] for r in rows] == [[1, 2, 3, 4], [5, 9], [5, 9], [2, 3], list(range(10))]
    assert [r["verdict_class"] for r in rows] == ["partly", "agree", "disagree", "not representable", "partly"]
    assert core.verdict_class("agree on pans; speed is not representable") == "agree"
    assert core.verdict_class("agree on low-key; the tint would disagree") == "agree"
    assert core.verdict_class("alternation visible; POV is not a slot") == "see text"


def test_essay_page_absent_in_public(monkeypatch, root):
    at = run_app(monkeypatch, root, True, "Home")
    assert "Tokens vs my essay" not in at.sidebar.radio(key="page").options
    with pytest.raises(ValueError):
        at.sidebar.radio(key="page").set_value("Tokens vs my essay").run()
    assert core.essay_films(root, True) == []
    with pytest.raises(PermissionError):
        core.load_essay(root, CLIP, True)


def test_essay_page_needs_claims_file(monkeypatch, root):
    (root / f"data/processed/{CLIP}_essay_claims.json").unlink()
    assert core.essay_films(root, False) == []
    at = run_app(monkeypatch, root, False, "Home")
    assert "Tokens vs my essay" not in at.sidebar.radio(key="page").options


def test_essay_page_never_for_registered_films(tmp_path):
    r = make_root(tmp_path, overlay=[])
    for f in (NOTLD, OWNED):
        _w(r, f"data/processed/{f}_report.md", ESSAY_REPORT)
        _w(r, f"data/processed/{f}_essay_claims.json", json.dumps(CLAIMS))
    assert core.essay_films(r, False) == []


def test_essay_page_local(monkeypatch, root):
    at = run_app(monkeypatch, root, False, "Tokens vs my essay")
    text = _texts(at)
    assert core.CAPTION_ESSAY in text and core.BANNER_LOCAL_ONLY in text
    assert "descriptive comparison, not accuracy; the essay is reference, not ground truth" in text.lower()
    assert "ESSAYCLAIMONE handheld chase" in text and "chase: handheld tracking" in text and "mostly disagree: rarely shallow" in text
    assert f"film={CLIP}" in text  # token rows of the clip's table
    assert "Candidate shots (report): 1-4 (7-30 s)" in text
    assert "shot 1 · " in text and "shot 1: mid frame not available" not in text  # keyframe image through the guard


# ---- C7 writes nothing -----------------------------------------------------------------------------------------------------------------------------------

def test_v2_pages_write_nothing(monkeypatch, root):
    before, repo_results = snapshot(root), snapshot(REPO / "results")
    for public in (True, False):
        for page in ("Home", "Films", "Reports"):
            run_app(monkeypatch, root, public, page)
    run_app(monkeypatch, root, False, "Tokens vs my essay")
    assert snapshot(root) == before and snapshot(REPO / "results") == repo_results


def test_data_reports_gitignored():
    import subprocess
    r = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", "data/reports/paper_full.html"], check=False)
    assert r.returncode == 0


def test_read_file_applies_deny_list_to_requested_path(root, monkeypatch):
    """The lexical deny list holds on its own: a data/ path is refused in public mode even if it resolved to a harmless file."""
    monkeypatch.setattr(core, "safe_path", lambda rt, rel: Path(rt).resolve() / "docs/phase6_report.md")
    with pytest.raises(PermissionError):
        core.read_file(root, f"data/processed/{CLIP}_report.md", True, [f"data/processed/{CLIP}_report.md"])


# ---- reviewed-files allow-list in the app's release filter ----
def _fake_pdf(root: Path) -> bytes:
    body = b"%PDF-1.4\n\x00\xff synthetic paper\n%%EOF\n"
    (root / "docs/writeup").mkdir(parents=True, exist_ok=True)
    (root / "docs/writeup/paper_public.pdf").write_bytes(body)
    return body


def _entry(sha: str) -> dict:
    return {"docs/writeup/paper_public.pdf": {"path": "docs/writeup/paper_public.pdf", "sha256": sha, "reviewed_by": "m", "date": "2026-10-17",
                                              "note": "x"}}


def test_release_clean_applies_code_checkout_allow_list(monkeypatch, root):
    import hashlib

    from cgt.release import check
    body = _fake_pdf(root)
    monkeypatch.setattr(check, "pdf_text", lambda p: None)
    rel = "docs/writeup/paper_public.pdf"
    seen = []
    monkeypatch.setattr(check, "load_reviewed", lambda tree, git=False: seen.append(Path(tree)) or {})
    assert core.release_clean(root, [rel]) == set()  # REVIEW, not listed
    assert seen == [check.CODE_ROOT]  # read from the code checkout, not from the demo root
    monkeypatch.setattr(check, "load_reviewed", lambda tree, git=False: _entry(hashlib.sha256(body).hexdigest()))
    assert core.release_clean(root, [rel]) == {rel}
    monkeypatch.setattr(check, "load_reviewed", lambda tree, git=False: _entry("0" * 64))  # stale
    assert core.release_clean(root, [rel]) == set()


def test_release_clean_malformed_allow_list_fails_closed(monkeypatch, root):
    from cgt.release import check
    monkeypatch.setattr(check, "load_reviewed", lambda tree, git=False: (_ for _ in ()).throw(check.ReviewListError("bad")))
    assert core.release_clean(root, ["docs/phase6_report.md"]) == set()
