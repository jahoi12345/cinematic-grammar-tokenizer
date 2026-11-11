"""an earlier review checks for the de-owning of code and tests (review a verification record).
The tmp registry registers the REAL owned id (read at runtime from configs/films.yaml; this file never names it), so a public-mode leak keyed to the real id is caught,
which the synthetic-id demo tests cannot do. Contract after A1 (a sibling working copy ecbe06f): a committed contradictory share is cited only for public-domain films; the owned
film shows only the live share of its loaded table; public mode never shows the owned id, title or aliases. Synthetic tables under tmp roots only; never touches film data."""
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
if not (REPO / "app/demo/core.py").exists():
    pytest.skip("app/demo not on this branch", allow_module_level=True)

try:
    from _private_values import owned_contra_ratio
except ImportError:  # the private loader is not part of an exported copy: nothing private to guard against
    def owned_contra_ratio() -> str:
        return "<no private value>"
from test_demo_guard import NOTLD, _table, _texts, core, make_root


def _owned() -> tuple[str, list[str]]:
    films = yaml.safe_load((REPO / "configs/films.yaml").read_text())["films"]
    owned = [f for f in films if f.get("public_domain") is not True]
    if not owned:
        pytest.skip("no owned film in this registry (trimmed release tree)")
    fid = owned[0]["id"]
    al_path = REPO / "configs/release_aliases.yaml"
    aliases = (yaml.safe_load(al_path.read_text()) or {}).get("aliases", {}).get(fid, []) if al_path.exists() else []
    terms = [fid, fid.replace("_", " "), str(owned[0].get("title", fid)), *aliases]
    return fid, [t.lower() for t in terms if t]


@pytest.fixture
def real_root(tmp_path):
    fid, _ = _owned()
    root = make_root(tmp_path, [{"id": NOTLD, "public_domain": True}, {"id": fid, "public_domain": False}], [], frames=False)
    _table(fid, seed=7).to_parquet(root / f"data/processed/{fid}_tokens_v1.parquet")
    return root


def _run(monkeypatch, root, public, page, **sel):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("CGT_DEMO_ROOT", str(root))
    monkeypatch.setenv("CGT_DEMO_PUBLIC", "1" if public else "")
    at = testing.AppTest.from_file(str(REPO / "app/demo/app.py"), default_timeout=120)
    at.run()
    at.sidebar.radio(key="page").set_value(page).run()
    for k, v in sel.items():
        at.sidebar.selectbox(key=k).set_value(v).run()
    assert not at.exception
    return _texts(at).lower()


def test_local_owned_overview_shows_live_share_only(monkeypatch, real_root):
    """Contract after A1 (a sibling working copy ecbe06f): in LOCAL mode the real owned id's overview shows the banner and the live share of its loaded table,
    and no committed number (no owned share is kept in code)."""
    fid, _ = _owned()
    df, _ = core.load_table(real_root, fid, False)
    live = f"share in the loaded table: {100 * core.arc.contradictory_share(df['move_atoms']):.1f}%"
    t = _run(monkeypatch, real_root, False, "Film overview", ov_film=fid)
    assert core.BANNER_LOCAL_ONLY.lower() in t and live.lower() in t and "no committed share for this film" in t
    assert owned_contra_ratio() not in t and "committed share for this film:" not in t


def test_positive_control_public_notld_overview_shows_committed_share(monkeypatch, real_root):
    """The public text checks below can fire: the committed-share caption is rendered in public mode (for the public-domain film)."""
    t = _run(monkeypatch, real_root, True, "Film overview", ov_film=NOTLD)
    assert "committed share for this film: 462/1,077" in t


@pytest.mark.parametrize("page", ["Film overview", "Shot explorer", "Side by side"])
def test_public_mode_never_shows_owned_share_or_name(monkeypatch, real_root, page):
    _, terms = _owned()
    t = _run(monkeypatch, real_root, True, page)
    assert owned_contra_ratio() not in t
    assert [x for x in terms if x in t] == []


def test_committed_share_only_for_public_domain_films(tmp_path, real_root):
    fid, _ = _owned()
    df, _ = core.load_table(real_root, NOTLD, False)
    assert "committed share for this film: 462/1,077" in core.movement_caption(NOTLD, df, real_root)
    for other in (fid, fid.upper(), fid + "_x", "", "None", NOTLD.upper()):
        c = core.movement_caption(other, df, real_root)
        assert "no committed share for this film" in c and owned_contra_ratio() not in c
    # the same public id is not cited when the registry under the root does not mark it public_domain (True only), or the registry is unreadable
    for films in ([{"id": NOTLD, "public_domain": False}, {"id": fid, "public_domain": False}], [{"id": NOTLD, "public_domain": "true"}]):
        r = make_root(tmp_path / str(len(films)) / str(films[0]["public_domain"]), films, [], frames=False)
        assert "no committed share for this film" in core.movement_caption(NOTLD, df, r)
    (real_root / "configs/films.yaml").write_text("films: [\n  - {id: x")
    assert "no committed share for this film" in core.movement_caption(NOTLD, df, real_root)


REGISTRY_FILES = {"configs/films.yaml", "configs/release_aliases.yaml"}


def test_code_dirs_never_name_the_owned_film(tmp_path):
    """Regression gate for the de-owning (docs/deown_2026-10-14.md): no tracked file under src/ app/ scripts/ tests/ configs/ except the registry
    names the owned film (its id, title or any release alias), as judged by the release checker itself."""
    import subprocess
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from cgt.release import check
    r = subprocess.run(["git", "-C", str(REPO), "ls-files", "src", "app", "scripts", "tests", "configs"], capture_output=True, text=True, check=False)
    if r.returncode != 0 or not r.stdout.strip():
        pytest.skip("not a git checkout")
    files = [f for f in r.stdout.splitlines() if f not in REGISTRY_FILES]
    lst = tmp_path / "files.txt"
    lst.write_text("\n".join(files) + "\n")
    reg = check.load_registry([REPO])
    hits = check.check_tree(REPO, reg, files_from=str(lst))
    assert [(rule, check.redact(p, reg)) for rule, p in hits] == []
