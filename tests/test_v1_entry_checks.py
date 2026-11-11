"""an earlier review verification pass checks for the v1 entry point and the registry-aware owners.
Written by the verification pass, independently of the authors' tests. Synthetic inputs only (tiny lavfi clips in tmp_path, tmp registry roots, stub stages).
Skips on a tree without cgt.pipeline.registry (main before the merge)."""
import hashlib
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

reg = pytest.importorskip("cgt.pipeline.registry")
from cgt.pipeline import adhoc, orchestrator
from cgt.pipeline.orchestrator import EXIT, Ctx, PipelineError, Stage, Status

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg/ffprobe")


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TV = _script("tokenize_video")


def ff(path: Path, *args) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, str(path)], check=True)
    return path


def video(path: Path, color="red") -> Path:
    return ff(path, "-f", "lavfi", "-i", f"color=c={color}:s=32x24:r=10:d=1", "-pix_fmt", "yuv420p")


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "repo"
    (r / "configs").mkdir(parents=True)
    (r / "configs/films.yaml").write_text(yaml.safe_dump({"films": [
        {"id": "pd_main", "public_domain": True, "path": "data/raw/pd.mp4", "cut_threshold": 12},
        {"id": "owned_main", "public_domain": False, "path": "data/raw/owned.mp4"}]}))
    return r


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        h.update(str(p.relative_to(root)).encode())
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def stubs(calls):
    """A stage is a hit iff its marker under <root>/data/processed exists (written by its run)."""
    def st(name):
        def marker(ctx):
            return ctx.root / "data/processed" / f"{ctx.film_id}_{name}.stub"

        def run(ctx):
            calls.append((name, ctx.film_id, ctx.allow_owned))
            marker(ctx).parent.mkdir(parents=True, exist_ok=True)
            marker(ctx).write_text("x")
            return {}
        return Stage(name, False, lambda ctx: Status(marker(ctx).exists(), "k", "stub", [marker(ctx)] if marker(ctx).exists() else []), run)
    return {n: st(n) for n in orchestrator.ORDER}


OK = {s: (lambda fid: None) for s in adhoc.OWNER_STAGES}


def go(root, *argv, calls=None):
    logs = []
    code = TV.main([str(a) for a in argv], root=root, stage_registry=stubs([] if calls is None else calls), probes=OK, log=logs.append,
                   lock_path=root.parent / "gpu.lock")
    return code, "\n".join(logs)


# C2 overlay merge and clash
def test_merge_and_clashes(root):
    reg.write_overlay(root, [{"id": "a1", "path": "/a.mp4"}])
    films = reg.load_films(root)
    assert set(films) == {"pd_main", "owned_main", "a1"} and films["a1"]["registry"] == "overlay" and films["a1"]["public_domain"] is False
    reg.write_overlay(root, [{"id": "pd_main", "path": "/a.mp4"}])
    with pytest.raises(reg.RegistryClash):
        reg.load_films(root)
    reg.write_overlay(root, [{"id": "a1", "path": "/a.mp4"}, {"id": "a1", "path": "/b.mp4"}])
    with pytest.raises(reg.RegistryClash):
        reg.load_films(root)


def test_clash_exits_11_from_the_script(root, tmp_path):
    reg.write_overlay(root, [{"id": "pd_main", "path": "/a.mp4"}])
    assert go(root, video(tmp_path / "c.mp4"))[0] == EXIT["registry_clash"] == 11


# C3 never publishable, also when the overlay entry claims public_domain true
def test_overlay_claiming_public_domain_is_never_publishable(root):
    reg.write_overlay(root, [{"id": "pdclaim", "path": "/a.mp4", "public_domain": True}])
    with pytest.raises(PipelineError) as e:
        orchestrator.check_film("pdclaim", [], allow_owned=True, root=root, publish=True)
    assert e.value.code == "film_not_allowed"
    assert "pdclaim" in reg.local_only_ids(root) and "pd_main" not in reg.local_only_ids(root)


# C5 no results/ output for an overlay film claiming public_domain true (summary, p1 figures, move report)
def test_no_results_paths_for_overlay_film(root, monkeypatch):
    reg.write_overlay(root, [{"id": "pdclaim", "path": "/a.mp4", "public_domain": True}])
    film = orchestrator.film_entry("pdclaim", root)
    p = orchestrator.write_summary(Ctx("pdclaim", film, root=root), [], [], 0.0, 0)
    assert "results" not in p.relative_to(root).parts and p.read_text().count("LOCAL ONLY") == 1
    from cgt.p1_shots_color import pipeline as p1
    monkeypatch.setattr(p1, "ROOT", root)
    assert "results" not in p1.figures_dir(p1.film_entry("pdclaim")).relative_to(root).parts
    rmf = _script("run_move_film")
    assert "results" not in rmf.default_report_path(root, film).relative_to(root).parts
    assert rmf.default_report_path(root, {"id": "pd_main", "public_domain": True}).relative_to(root).parts[0] == "results"


# C4 allow-owned: owners refuse an overlay film without it, even with public_domain true
def test_owners_refuse_overlay_without_allow_owned(root, tmp_path, monkeypatch):
    import pandas as pd

    from cgt.p3b_vlm import guards
    from cgt.p3c_tokens import runner
    from cgt.p4_camera_move import film_move as FM
    reg.write_overlay(root, [{"id": "pdclaim", "path": str(video(tmp_path / "pd.mp4")), "public_domain": True}])
    (root / "data/processed").mkdir(parents=True)
    pd.DataFrame({"shot_id": [0], "start_s": [0.0], "end_s": [1.0]}).to_parquet(root / "data/processed/pdclaim_shots.parquet")
    monkeypatch.setattr(runner, "ROOT", root)
    with pytest.raises(guards.NotAuthorized):
        runner.require_public_domain("pdclaim")
    assert runner.require_public_domain("pdclaim", allow_owned=True)["id"] == "pdclaim"
    with pytest.raises(FM.Refused, match="overlay"):
        FM.resolve_film("pdclaim", root=root)
    assert FM.resolve_film("pdclaim", root=root, allow_owned=True)["id"] == "pdclaim"  # the overlay film IS found (not just "missing file")


def test_permission_is_this_film_this_invocation_only(root, tmp_path):
    calls = []
    assert go(root, video(tmp_path / "c.mp4"), "--stages", "p1", calls=calls)[0] == 0
    assert calls and all(c[2] is True for c in calls)
    e = reg.read_overlay(root)[0]
    assert not any("allow" in k or "owned" in k or "permission" in k for k in e)
    # a later plain orchestrator run of the same overlay film (or the owned main film) is still refused without allow_owned
    assert orchestrator.run(e["id"], ["p1"], stubs([]), dry_run=True, root=root, log=lambda s: None) == EXIT["film_not_allowed"]
    assert orchestrator.run("owned_main", ["p1"], stubs([]), dry_run=True, root=root, log=lambda s: None) == EXIT["film_not_allowed"]


# C6 dry run writes nothing (content hash of the whole root, plus the clip directory)
def test_dry_run_writes_nothing_by_content(root, tmp_path):
    clip = video(tmp_path / "in/c.mp4")
    before = (tree_hash(root), tree_hash(clip.parent))
    code = go(root, clip, "--dry-run")[0]
    assert code == 0 and (tree_hash(root), tree_hash(clip.parent)) == before
    assert go(root, clip, "--stages", "p1")[0] == 0
    before = tree_hash(root)
    assert go(root, clip, "--cut-threshold", "5", "--title", "T", "--public-domain", "--dry-run")[0] == 0
    assert tree_hash(root) == before


# C7 input checks
@pytest.mark.parametrize("kind", ["missing", "dir", "text", "audio", "empty_mp4", "image"])
def test_bad_inputs_exit_10_and_write_nothing(root, tmp_path, kind):
    p = tmp_path / "in" / {"missing": "nope.mp4", "dir": "d.mp4", "text": "t.mp4", "audio": "a.m4a", "empty_mp4": "e.mp4", "image": "i.png"}[kind]
    p.parent.mkdir(parents=True, exist_ok=True)
    if kind == "dir":
        p.mkdir()
    elif kind == "text":
        p.write_text("not a video\n" * 50)
    elif kind == "audio":
        ff(p, "-f", "lavfi", "-i", "sine=frequency=440:duration=1")
    elif kind == "empty_mp4":
        p.write_bytes(b"")
    elif kind == "image":
        ff(p, "-f", "lavfi", "-i", "color=c=red:s=32x24", "-frames:v", "1")
    before = tree_hash(root)
    code, log = go(root, p)
    assert code == EXIT["bad_input"] == 10 and "REFUSED [bad_input]" in log
    assert tree_hash(root) == before


# C8 content-derived stable id; an id never switches files
def test_id_stable_and_content_derived(root, tmp_path):
    a = video(tmp_path / "x/Clip One.mp4")
    b = tmp_path / "y/Clip One.mp4"
    b.parent.mkdir()
    shutil.copy(a, b)
    os.utime(b, (1, 1))
    ida = adhoc.resolve(a, root)["film_id"]
    assert ida == adhoc.resolve(a, root, dry_run=True)["film_id"] == adhoc.resolve(b, root)["film_id"]
    assert ida == f"clip_one_{hashlib.sha256(a.read_bytes()).hexdigest()[:8]}"
    c = video(tmp_path / "z/Clip One.mp4", "blue")
    assert adhoc.resolve(c, root)["film_id"] != ida


def test_id_never_switches_files(root, tmp_path):
    a, c = video(tmp_path / "a.mp4"), video(tmp_path / "c.mp4", "blue")
    assert go(root, a, "--id", "mine", "--stages", "p1")[0] == 0
    sha = reg.read_overlay(root)[0]["sha256"]
    assert go(root, c, "--id", "mine", "--stages", "p1")[0] == EXIT["registry_clash"]
    assert go(root, c, "--id", "mine", "--dry-run")[0] == EXIT["registry_clash"]
    assert [e["sha256"] for e in reg.read_overlay(root)] == [sha]


# C10 threshold warning and stored threshold
def test_threshold_warning_and_storage(root, tmp_path):
    a = video(tmp_path / "a.mp4")
    code, log = go(root, a, "--stages", "p1")
    fid = reg.read_overlay(root)[0]["id"]
    assert code == 0 and "WARNING: unchecked cut threshold 27" in log and f"scripts/spotcheck_cuts.py --film {fid} --threshold 27 --n 50" in log
    code, log = go(root, a, "--stages", "p1", "--cut-threshold", "9")
    assert code == 0 and "WARNING: unchecked" not in log and reg.read_overlay(root)[0]["cut_threshold"] == 9.0
    code, log = go(root, a, "--stages", "p1")
    assert "WARNING: unchecked" not in log and reg.read_overlay(root)[0]["cut_threshold"] == 9.0


# F-C1: the movement stage passes --allow-owned only for a non-public-domain film; an overlay film registered with --public-domain is then
# refused by scripts/run_move_film.py (overlay films always need --allow-owned), so `tokenize_video.py clip --public-domain` fails at the move stage.
def test_move_stage_passes_allow_owned_for_public_domain_overlay_film(root, monkeypatch):
    from cgt.pipeline import stages
    reg.write_overlay(root, [{"id": "pdclaim", "path": "/a.mp4", "public_domain": True}])
    seen = []
    monkeypatch.setattr(stages, "_free_percent", lambda: 99)
    monkeypatch.setattr(stages, "_run_move_script", lambda ctx, cmd: seen.append(cmd) or 1)
    film = orchestrator.film_entry("pdclaim", root)
    with pytest.raises(PipelineError):
        stages.move_run(Ctx("pdclaim", film, root=root, allow_owned=True))
    assert "--allow-owned" in seen[0]

