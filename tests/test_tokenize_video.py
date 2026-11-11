"""v1 entry point (scripts/tokenize_video.py, the recorded rule) on synthetic inputs only: tiny ffmpeg clips in tmp_path, a temporary registry root, stub stages
(no model, no GPU, no real film). docs/v1_entry_point.md records the mutation check of each listed fault."""
import hashlib
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from cgt.pipeline import adhoc, orchestrator
from cgt.pipeline import registry as reg
from cgt.pipeline.orchestrator import EXIT, Stage, Status

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg/ffprobe")


def _load_script():
    spec = importlib.util.spec_from_file_location("tokenize_video", REPO / "scripts/tokenize_video.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TV = _load_script()


def make_clip(path: Path, color: str = "red", seconds: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=64x48:r=10:d={seconds}", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    return {"a": make_clip(d / "My Clip (2024).mp4"), "b": make_clip(d / "other.mp4", "blue")}


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "repo"
    (r / "configs").mkdir(parents=True)
    make_clip(r / "data/raw/films/pd_film/film.mp4", "green")
    (r / "configs/films.yaml").write_text(yaml.safe_dump({"films": [
        {"id": "pd_film", "title": "PD", "public_domain": True, "path": "data/raw/films/pd_film/film.mp4", "cut_threshold": 12},
        {"id": "owned_film", "title": "Owned", "public_domain": False, "path": "data/raw/films/owned_film/film.mp4"}]}))
    return r


def stub_registry(root: Path, calls: list):
    """p1 -> p2 -> p3c -> move -> tokens; a stage is a hit iff its marker under data/ exists (per film)."""
    def stage(name):
        def marker(ctx):
            return root / "data/processed" / (f"{ctx.film_id}_tokens_v1.parquet" if name == "tokens" else f"{ctx.film_id}_{name}.stub")

        def status(ctx):
            m = marker(ctx)
            return Status(m.exists(), f"key-{name}", "stub", [m] if m.exists() else [])

        def run(ctx):
            calls.append((name, ctx.film_id, ctx.allow_owned))
            marker(ctx).parent.mkdir(parents=True, exist_ok=True)
            marker(ctx).write_text("x")
            if name == "tokens":
                (root / f"data/processed/{ctx.film_id}_tokens_v1.txt").write_text("row\n")
            return {"n": 1}
        return Stage(name, False, status, run)
    return {n: stage(n) for n in ("p1", "p2", "p3c", "move", "tokens")}


OK_PROBES = {s: (lambda fid: None) for s in adhoc.OWNER_STAGES}


def go(root, *argv, probes=OK_PROBES, calls=None):
    logs = []
    calls = [] if calls is None else calls
    code = TV.main([str(a) for a in argv], root=root, stage_registry=stub_registry(root, calls), probes=probes, log=logs.append, lock_path=root / "gpu.lock")
    return code, "\n".join(logs), calls


def overlay(root):
    return {e["id"]: e for e in reg.read_overlay(root)}


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*")}


# ---------------------------------------------------------------- 1. overlay registry
def test_overlay_merged_with_defaults(root):
    reg.write_overlay(root, [{"id": "clip_1", "path": "/x.mp4"}, {"id": "clip_2", "path": "/y.mp4", "public_domain": True}])
    films = reg.load_films(root)
    assert set(films) == {"pd_film", "owned_film", "clip_1", "clip_2"}
    assert films["pd_film"]["registry"] == "main" and films["clip_1"]["registry"] == "overlay"
    assert films["clip_1"]["public_domain"] is False and films["clip_2"]["public_domain"] is True
    assert reg.local_only_ids(root) == {"owned_film", "clip_1", "clip_2"}  # an overlay film is local only even when public domain
    assert "LOCAL ONLY" in (root / reg.OVERLAY).read_text()


def test_clash_between_films_yaml_and_overlay_is_an_error(root, clips):
    reg.write_overlay(root, [{"id": "pd_film", "path": str(clips["a"])}])
    with pytest.raises(reg.RegistryClash, match="pd_film"):
        reg.load_films(root)
    logs = []
    assert orchestrator.run("pd_film", ["p1"], stub_registry(root, []), dry_run=True, root=root, log=logs.append) == EXIT["registry_clash"]
    code, log, _ = go(root, clips["a"], "--dry-run")
    assert code == EXIT["registry_clash"] and "clash" in log


def test_clash_within_overlay_and_pending_is_an_error(root):
    reg.write_overlay(root, [{"id": "c", "path": "/1.mp4", "sha256": "aa"}, {"id": "c", "path": "/2.mp4", "sha256": "bb"}])
    with pytest.raises(reg.RegistryClash):
        reg.load_films(root)
    reg.write_overlay(root, [{"id": "c", "path": "/1.mp4", "sha256": "aa"}])
    with pytest.raises(reg.RegistryClash):
        reg.load_films(root, pending={"id": "c", "path": "/2.mp4", "sha256": "bb"})
    with pytest.raises(reg.RegistryClash):
        reg.load_films(root, pending={"id": "pd_film", "path": "/2.mp4", "sha256": "bb"})


def test_explicit_id_of_a_main_film_with_another_file_is_a_clash(root, clips):
    code, _, calls = go(root, clips["a"], "--id", "pd_film")
    assert code == EXIT["registry_clash"] and not calls and not (root / reg.OVERLAY).exists()


def test_id_bound_to_another_file_is_a_clash(root, clips):
    assert go(root, clips["a"], "--id", "mine")[0] == 0
    code, log, _ = go(root, clips["b"], "--id", "mine")
    assert code == EXIT["registry_clash"] and "another file" in log
    assert overlay(root)["mine"]["path"] == str(clips["a"].resolve())


def test_publish_refused_for_overlay_film_even_if_public_domain(root, clips):
    assert go(root, clips["a"], "--id", "pdclip", "--public-domain", "--stages", "p1")[0] == 0
    assert overlay(root)["pdclip"]["public_domain"] is True
    logs = []
    code = orchestrator.run("pdclip", ["p1"], stub_registry(root, []), publish=True, root=root, log=logs.append, lock_path=root / "gpu.lock")
    assert code == EXIT["film_not_allowed"] and "overlay" in "\n".join(logs)
    # a public-domain film of films.yaml may still be published (unchanged behaviour)
    assert orchestrator.run("pd_film", ["p1"], stub_registry(root, []), dry_run=True, publish=True, root=root, log=logs.append) == 0


def test_overlay_summary_stays_local_even_if_public_domain(root, clips):
    assert go(root, clips["a"], "--id", "pdclip", "--public-domain")[0] == 0
    assert (root / "data/cache/pipeline/pdclip/summary.md").exists()
    assert not (root / "results").exists()


# ---------------------------------------------------------------- 2. input handling
def test_missing_file_refused(root, tmp_path):
    code, log, calls = go(root, tmp_path / "nope.mp4")
    assert code == EXIT["bad_input"] == 10 and "not found" in log and not calls


def test_unreadable_file_refused(root, clips, tmp_path):
    p = tmp_path / "locked.mp4"
    shutil.copy(clips["a"], p)
    p.chmod(0)
    try:
        if os.access(p, os.R_OK):
            pytest.skip("running as a user that can read mode-0 files")
        code, log, _ = go(root, p)
        assert code == EXIT["bad_input"] and "not readable" in log
    finally:
        p.chmod(0o644)


@pytest.mark.parametrize("kind", ["text", "audio", "directory"])
def test_non_video_refused(root, tmp_path, kind):
    p = tmp_path / "fake.mp4"
    if kind == "text":
        p.write_text("this is not a video\n" * 50)
    elif kind == "audio":
        p = tmp_path / "tone.m4a"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=d=1", str(p)], check=True)
    else:
        p.mkdir()
    code, log, calls = go(root, p)
    assert code == EXIT["bad_input"] and not calls and not (root / reg.OVERLAY).exists()
    assert ("not a video" in log) or ("not a regular file" in log)


def test_unsafe_explicit_id_refused(root, clips):
    for bad in ("../x", "Upper", "a/b", "_lead", "x" * 65):
        code, log, _ = go(root, clips["a"], "--id", bad)
        assert code == EXIT["bad_input"] and "unsafe --id" in log


def test_id_is_slug_plus_content_hash_and_stable(root, clips, tmp_path):
    r1 = adhoc.resolve(clips["a"], root, dry_run=True)
    r2 = adhoc.resolve(clips["a"], root, dry_run=True)
    sha = hashlib.sha256(clips["a"].read_bytes()).hexdigest()
    assert r1["film_id"] == r2["film_id"] == f"my_clip_2024_{sha[:8]}"
    assert adhoc.ID_RE.match(r1["film_id"])
    copy = tmp_path / "elsewhere" / clips["a"].name
    copy.parent.mkdir()
    shutil.copy(clips["a"], copy)
    os.utime(copy, (1, 1))  # a different mtime and directory: the id depends on the name stem and bytes only
    assert adhoc.resolve(copy, root, dry_run=True)["film_id"] == r1["film_id"]
    other = make_clip(tmp_path / "x" / clips["a"].name, "yellow")
    assert adhoc.resolve(other, root, dry_run=True)["film_id"] != r1["film_id"]
    assert adhoc.slug("!!!") == "video" and adhoc.slug("Ünïcode Film") == "n_code_film"


def test_same_file_twice_same_id_and_idempotent_rerun(root, clips):
    calls = []
    code, _, _ = go(root, clips["a"], calls=calls)
    assert code == 0
    fid = next(iter(overlay(root)))
    ran = [c[0] for c in calls]
    assert ran == ["p1", "p2", "p3c", "move", "tokens"]
    before = (root / reg.OVERLAY).stat().st_mtime_ns, (root / reg.OVERLAY).read_bytes()
    code, log2, _ = go(root, clips["a"], calls=calls)
    assert code == 0 and len(calls) == 5  # every stage a cache hit the second time
    assert "unchanged" in log2 and f"'{fid}'" in log2 and log2.count("cache hit") == 5
    assert ((root / reg.OVERLAY).stat().st_mtime_ns, (root / reg.OVERLAY).read_bytes()) == before  # overlay not rewritten
    assert list(overlay(root)) == [fid]


def test_registered_film_file_uses_its_registered_id(root):
    code, log, calls = go(root, root / "data/raw/films/pd_film/film.mp4", "--stages", "p1")
    assert code == 0 and "configs/films.yaml" in log and calls == [("p1", "pd_film", True)]
    assert not (root / reg.OVERLAY).exists()


# ---------------------------------------------------------------- 3. threshold
def test_default_threshold_warns_with_spotcheck_command(root, clips):
    code, log, _ = go(root, clips["a"], "--dry-run")
    fid = adhoc.resolve(clips["a"], root, dry_run=True)["film_id"]
    assert code == 0 and "WARNING: unchecked cut threshold 27" in log
    assert f"scripts/spotcheck_cuts.py --film {fid} --threshold 27 --n 50" in log


def test_explicit_threshold_stored_kept_and_updated(root, clips):
    code, log, _ = go(root, clips["a"], "--cut-threshold", "12", "--stages", "p1")
    e = next(iter(overlay(root).values()))
    assert code == 0 and e["cut_threshold"] == 12 and e["cut_threshold_source"] == "user" and "unchecked" not in log
    code, log, _ = go(root, clips["a"], "--stages", "p1")  # no flag: the stored threshold is kept
    assert overlay(root)[e["id"]]["cut_threshold"] == 12 and "unchanged" in log
    code, log, _ = go(root, clips["a"], "--cut-threshold", "16", "--stages", "p1")
    assert overlay(root)[e["id"]]["cut_threshold"] == 16 and "UPDATED" in log and "cut_threshold: 12.0 -> 16.0" in log


# ---------------------------------------------------------------- 4. owned inputs
def test_owned_permission_for_this_film_only_and_outputs_under_data(root, clips):
    code, log, calls = go(root, clips["a"])
    assert code == 0 and calls and all(c[2] is True for c in calls)  # allow_owned passed to this run
    e = next(iter(overlay(root).values()))
    assert e["public_domain"] is False and not any("allow" in k for k in e)  # no permission is stored
    # another owned film is still refused by the plain orchestrator without the explicit flag
    assert orchestrator.run("owned_film", ["p1"], stub_registry(root, []), dry_run=True, root=root, log=lambda s: None) == EXIT["film_not_allowed"]
    assert orchestrator.run(e["id"], ["p1"], stub_registry(root, []), dry_run=True, root=root, log=lambda s: None) == EXIT["film_not_allowed"]
    new = [p for p in root.rglob("*") if p.is_file() and not str(p.relative_to(root)).startswith(("data/", "configs/films.yaml"))]
    assert [str(p.relative_to(root)) for p in new] == [reg.OVERLAY]  # besides the gitignored overlay, everything is under data/
    assert f"token table: {root}/data/processed/{e['id']}_tokens_v1.parquet" in log and "streamlit run app/browse/app.py" in log


def test_overlay_is_gitignored_on_main_or_reported(root):
    """The overlay path is ignored on main; this branch predates that line, so only report it."""
    r = subprocess.run(["git", "check-ignore", "-q", reg.OVERLAY], cwd=REPO, check=False)
    if r.returncode != 0:
        pytest.skip(f"{reg.OVERLAY} not yet ignored on this branch (added on main by a recorded decision)")


# ---------------------------------------------------------------- 5. dry run
def test_dry_run_writes_nothing(root, clips):
    before = snapshot(root)
    code, log, calls = go(root, clips["a"], "--dry-run")
    assert code == 0 and not calls and "would RUN" in log and "dry run" in log
    assert snapshot(root) == before and not (root / reg.OVERLAY).exists() and not adhoc.hash_cache(root).exists()


def test_dry_run_of_a_changed_entry_writes_nothing(root, clips):
    assert go(root, clips["a"], "--stages", "p1")[0] == 0
    before = snapshot(root)
    code, log, calls = go(root, clips["a"], "--cut-threshold", "9", "--dry-run")
    assert code == 0 and "UPDATED" in log and "not written" in log and not calls and snapshot(root) == before


def test_pending_entry_refused_outside_a_dry_run(root):
    logs = []
    code = orchestrator.run("x", ["p1"], stub_registry(root, []), root=root, log=logs.append, pending={"id": "x", "path": "/x.mp4"})
    assert code == EXIT["usage"]


# ---------------------------------------------------------------- owner-module support
def test_owner_blocker_stops_before_any_stage_runs(root, clips):
    probes = {**OK_PROBES, "p1": lambda fid: "p1 reads configs/films.yaml only"}
    code, log, calls = go(root, clips["a"], probes=probes)
    assert code == EXIT["unavailable_or_prerequisite"] and not calls and "needs owner change" in log and "p1 reads" in log
    code, log, calls = go(root, clips["a"], "--stages", "tokens", probes=probes)  # tokens resolves nothing itself
    assert code == 0 and [c[0] for c in calls] == ["tokens"]


def test_real_p1_probe_reports_an_unregistered_overlay_film():
    pytest.importorskip("cv2")
    assert adhoc._probe_p1("no_such_film_ffffffff") is not None
