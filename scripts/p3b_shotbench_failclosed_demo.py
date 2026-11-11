"""CPU-only demonstration that the ShotBench evaluator fails closed. Every scenario stops BEFORE any model load, image read or ledger write
(demo tags, a temporary raw-log directory, and no scenario that could fall through into a real evaluation).
Usage: uv run python scripts/p3b_shotbench_failclosed_demo.py   -> results/phase3b/shotbench_failclosed_demo.md"""
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.images import BUDGET_PX, MULTIPLE, prepare_eval_cache, shrink_to_budget
from cgt.p3b_vlm.runstatus import plan_original_unchanged
from cgt.p3b_vlm.shotbench import load_shotbench_samples

ROOT = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp())
rows = []


def script(args, label, expect, raw="raw"):
    r = subprocess.run(["uv", "run", "python", "scripts/p3b_eval_shotbench.py", *args, "--raw-dir", str(tmp / raw)], cwd=ROOT,
                       capture_output=True, text=True, check=False)
    line = next((x for x in reversed((r.stdout + r.stderr).splitlines()) if "FAIL CLOSED" in x), "")
    rows.append((label, "scripts/p3b_eval_shotbench.py " + " ".join(args), r.returncode, expect, line[:200]))
    print(f"{label}: exit {r.returncode} (expected {expect})", flush=True)


def lib(label, fn, exc):
    try:
        fn()
        rows.append((label, "library call", "none", exc.exit_code, "NO ERROR RAISED"))
    except exc as e:
        rows.append((label, "library call", e.exit_code, exc.exit_code, str(e)[:200]))
        print(f"{label}: {type(e).__name__} exit_code {e.exit_code}", flush=True)


# 1. ledger comes first: a tag that is already in the ledger stops the run before anything else (even with a go file present)
go = tmp / "go"
go.write_text("test only\n")
script(["--tag", "siglip_base_lr1e-2_ep300", "--go-file", str(go)], "tag already in the ledger", 14)
# 2. the tag must be the frozen manifest's evaluation tag (F4): another string cannot obtain its own ledger slot (stops before any go-ahead check)
TAG = "qwen3vl2b_lora_filmsplit"
script(["--tag", "demo_unused_tag", "--go-file", str(go)], "tag is not the frozen evaluation tag (F4)", 18)
# 3. no go-ahead file for the real tag (explicit missing path: never the real default, which the operator may create later)
script(["--tag", TAG, "--go-file", str(tmp / "no_such_go_file")], "no SHOTBENCH_GO file", 13, raw="raw_nogo")
# 3b. a go-ahead file with the wrong content (not bound to the tag and adapter hash) authorizes nothing (F4)
wrong = tmp / "wrong_go"
wrong.write_text("SHOTBENCH_GO\ntag=some_other_tag\nadapter_sha256=" + "0" * 64 + "\n")
script(["--tag", TAG, "--go-file", str(wrong)], "go-ahead file bound to another tag/hash (F4)", 13, raw="raw_wronggo")
# 3c. an earlier crash blocks a retry (exit 16) before the go-ahead check; the ACK file is what the operator creates
(tmp / "raw_crash").mkdir(exist_ok=True)
(tmp / "raw_crash" / f"{TAG}.CRASH.json").write_text(json.dumps({"tag": TAG, "exception_class": "RuntimeError", "reason_code": "internal_error", "questions_scored_before_crash": 7}))
script(["--tag", TAG, "--go-file", str(go)], "earlier crash not acknowledged by the operator", 16, raw="raw_crash")
# 3d. a consumed tag (CONSUMED.json) is refused even when the ledger lacks it (exit 14)
(tmp / "raw_consumed").mkdir(exist_ok=True)
(tmp / "raw_consumed" / f"{TAG}.CONSUMED.json").write_text(json.dumps({"state": "LEDGER_CONSUMED"}))
script(["--tag", TAG, "--go-file", str(go)], "consumed tag (CONSUMED.json) refused even if the ledger lost it", 14, raw="raw_consumed")
# 3. adapter hash checks (library level: the evaluator calls exactly this function before loading anything)
run_dir = ROOT / "data/cache/p3b/runs/qwen3vl2b_lora_filmsplit_run1"
real = guards.require_frozen_adapter(ROOT / "results/phase3b/frozen_manifest.json", run_dir)
print("frozen adapter verified:", real["sha256"][:16], flush=True)
bad = json.loads((ROOT / "results/phase3b/frozen_manifest.json").read_text())
bad["adapter_sha256"] = "0" * 64
(tmp / "bad_manifest.json").write_text(json.dumps(bad))
lib("freeze manifest hash differs from the adapter file", lambda: guards.require_frozen_adapter(tmp / "bad_manifest.json", run_dir), guards.AdapterHashError)
bad2 = json.loads((ROOT / "results/phase3b/frozen_manifest.json").read_text())
bad2["adapter_file"] = "ckpt_075.safetensors"  # a different checkpoint than the one the freeze names
(tmp / "bad2.json").write_text(json.dumps(bad2))
lib("manifest names a different checkpoint than its recorded hash", lambda: guards.require_frozen_adapter(tmp / "bad2.json", run_dir), guards.AdapterHashError)
lib("go file missing (library)", lambda: guards.require_go_file(tmp / "nope"), guards.NotAuthorized)
fm_real = ROOT / "results/phase3b/frozen_manifest.json"
fmj = json.loads(fm_real.read_text())
guards.require_evaluation_tag(TAG, fm_real)
lib("F4: library: tag other than the frozen evaluation tag", lambda: guards.require_evaluation_tag("another_tag", fm_real), guards.TagNotBound)
good_go = tmp / "good_go"
good_go.write_text(guards.go_text(TAG, fmj["adapter_sha256"]))
guards.require_go_file(good_go, TAG, fmj["adapter_sha256"])  # a correctly bound file (synthetic path only) is accepted
lib("F4: go file for the right tag but another adapter hash", lambda: guards.require_go_file(good_go, TAG, "1" * 64), guards.NotAuthorized)
lib("F4: go file for another tag", lambda: guards.require_go_file(good_go, "other_tag", fmj["adapter_sha256"]), guards.NotAuthorized)

# F3: a COMMITTED alteration of the original plan text -> exit 10; an amendments-only change stays allowed (synthetic git repository)
def git(*a):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


repo = tmp / "planrepo"
repo.mkdir()
git("init", "-q")
(repo / "plan.md").write_text("# Plan\noriginal text\n\n## Amendments\n1. first\n")
git("add", "plan.md")
git("commit", "-q", "-m", "plan")
c0 = git("rev-parse", "HEAD")
(repo / "plan.md").write_text("# Plan\noriginal text\n\n## Amendments\n1. first\n2. second amendment\n")
git("commit", "-q", "-am", "amend")
okp = guards.require_plan_unchanged(repo, c0, "plan.md")
rows.append(("F3: committed amendment only is allowed", "library call", 0 if okp["unchanged"] else "changed", 0, "plan unchanged: " + str(okp["unchanged"])))
(repo / "plan.md").write_text("# Plan\nALTERED text\n\n## Amendments\n1. first\n2. second amendment\n")
git("commit", "-q", "-am", "alter the original")
lib("F3: committed alteration of the original plan text", lambda: guards.require_plan_unchanged(repo, c0, "plan.md"), guards.DirtyTree)
real_plan = plan_original_unchanged()
rows.append(("F3: the real plan is unchanged", "library call", 0 if real_plan["unchanged"] else "CHANGED", 0, real_plan["current_text_sha256"][:16]))

# F5 / F6: synthetic image root and a synthetic TSV (real question text, synthetic paths; no ShotBench image is opened)
root_dir = tmp / "sbroot"
(root_dir / "image").mkdir(parents=True)
outside = tmp / "outside.jpg"
Image.new("RGB", (64, 64), (200, 10, 10)).save(outside)
Image.new("RGB", (64, 64), (10, 200, 10)).save(root_dir / "image" / "ok.jpg")
(root_dir / "image" / "link.jpg").symlink_to(outside)
real_tsv = pd.read_csv(ROOT / "data/external/shotbench/test.tsv", sep="\t")
tmpl = real_tsv[real_tsv.type.str.contains("image")].iloc[0]


def synth_tsv(rows_):
    df = pd.DataFrame([{"index": 9000 + i, "type": t, "path": f"[{p!r}]", "question": tmpl.question, "options": tmpl.options, "answer": tmpl.answer, "category": tmpl.category}
                       for i, (t, p) in enumerate(rows_)])
    out = tmp / f"synth_{abs(hash(tuple(rows_)))}.tsv"
    df.to_csv(out, sep="\t", index=False)
    return out


for label, path in (("relative escape (../)", "../outside.jpg"), ("absolute path outside the root", str(outside)), ("symlink inside the root pointing outside", "image/link.jpg")):
    lib(f"F5: image path rejected before any image is opened: {label}", lambda path=path: guards.require_under_image_root(path, root_dir), guards.DataRootViolation)
    lib(f"F5: the question loader rejects it too: {label}", lambda path=path: load_shotbench_samples(synth_tsv([('["image"]', path)]), root_dir), guards.DataRootViolation)
guards.require_under_image_root("image/ok.jpg", root_dir)
# F6: the image/video row is excluded and never scored; counts come from the same masks
smp, st = load_shotbench_samples(synth_tsv([('["image"]', "image/ok.jpg"), ('["image", "video"]', "image/ok.jpg"), ('["video"]', "image/ok.jpg")]), root_dir)
ids = [x["id"] for x in smp]
rows.append(("F6: image + image/video + video rows: only the pure image row is scored", "library call", len(ids), 1,
             f"scored ids {ids}; rows_video_excluded={st['rows_video_excluded']} rows_static_image={st['rows_static_image']} (expected 2 and 1); scored+excluded=rows_total: {len(ids) + st['rows_video_excluded'] == st['rows_total']}"))
rows.append(("F6: excluded count equals the mask-derived count", "library call", st["rows_video_excluded"], 2, "image/video and video rows both counted as excluded"))

# cache: collision-resistant keys, fail closed on collisions, stale/foreign/tampered caches, provenance manifest, unchanged resize semantics
cdir = tmp / "cache_src"
(cdir / "a").mkdir(parents=True)
(cdir / "b").mkdir()
Image.new("RGB", (640, 480), (255, 0, 0)).save(cdir / "same.png")
Image.new("RGB", (640, 480), (0, 0, 255)).save(cdir / "same.jpg")
Image.new("RGB", (640, 480), (0, 0, 200)).save(cdir / "other.jpg")
Image.new("RGB", (640, 480), (0, 255, 0)).save(cdir / "a" / "dup.png")
Image.new("RGB", (640, 480), (0, 255, 0)).save(cdir / "b" / "dup.png")  # identical name and content in another directory
cc = tmp / "evalcache"
m = prepare_eval_cache([str(cdir / "same.png"), str(cdir / "other.jpg")], cc)
rows.append(("cache: different-stem images get distinct keys (name with extension + content hash)", "library call", len(set(m.values())), 2, "; ".join(sorted(Path(v).name for v in m.values()))))
direct = tmp / "direct.jpg"
shrink_to_budget(str(cdir / "same.png"), direct, BUDGET_PX, MULTIPLE)
rows.append(("cache: resize semantics unchanged (bytes equal shrink_to_budget)", "library call", int(Path(m[str(cdir / 'same.png')]).read_bytes() == direct.read_bytes()), 1, ""))
m2 = prepare_eval_cache([str(cdir / "same.png"), str(cdir / "other.jpg")], cc)
rows.append(("cache: a second call verifies and reuses the cache", "library call", int(m2 == m), 1, "provenance manifest present: " + str((cc / "provenance.json").exists())))


def lib_value(label, fn):
    try:
        fn()
        rows.append((label, "library call", "none", "ValueError", "NO ERROR RAISED"))
    except ValueError as e:
        rows.append((label, "library call", "ValueError", "ValueError", type(e).__name__))


lib_value("cache: two source images that map to the same key fail closed", lambda: prepare_eval_cache([str(cdir / "a" / "dup.png"), str(cdir / "b" / "dup.png")], tmp / "cache_dup"))
(cc / "stray.jpg").write_bytes(b"not from this run")
lib_value("cache: a cache holding a file absent from its provenance manifest is refused", lambda: prepare_eval_cache([str(cdir / "same.png")], cc))
(cc / "stray.jpg").unlink()
first = next(iter(m.values()))
Path(first).write_bytes(b"tampered")
lib_value("cache: a tampered cached file is refused", lambda: prepare_eval_cache([str(cdir / "same.png")], cc))
legacy = tmp / "legacy_cache"
legacy.mkdir()
(legacy / "same.jpg").write_bytes(b"stem-keyed legacy entry")
lib_value("cache: a legacy stem-keyed cache without provenance is refused", lambda: prepare_eval_cache([str(cdir / "same.png")], legacy))


# same-stem policy (REFUSE): two distinct source files sharing a stem fail closed (exit 20) before any cache entry exists; the same file repeated is fine
def lib_exc(label, fn, exc):
    try:
        fn()
        rows.append((label, "library call", "none", exc.exit_code, "NO ERROR RAISED"))
    except exc as e:
        rows.append((label, "library call", e.exit_code, exc.exit_code, str(e)[:120]))


lib_exc("same stem: red same.png + blue same.jpg refused (guard, exit 20)", lambda: guards.require_unique_stems([str(cdir / "same.png"), str(cdir / "same.jpg")]), guards.SameStemImages)
stemcache = tmp / "cache_stem"
lib_value("same stem: the cache refuses the red/blue pair BEFORE creating any entry", lambda: prepare_eval_cache([str(cdir / "same.png"), str(cdir / "same.jpg")], stemcache))
rows.append(("same stem: no cache directory or entry was created by the refused call", "library call", int(not stemcache.exists()), 1, f"exists={stemcache.exists()}"))
lib_exc("same stem: names differing only in extension case (a.JPG vs a.jpg) refused", lambda: guards.require_unique_stems(["x/a.JPG", "x/a.jpg"]), guards.SameStemImages)
lib_exc("same stem: same stem in different directories refused", lambda: guards.require_unique_stems(["x/b.png", "y/b.jpeg"]), guards.SameStemImages)
rep = prepare_eval_cache([str(cdir / "same.png"), str(cdir / "same.png"), str(cdir / "other.jpg")], tmp / "cache_rep")
rows.append(("same stem: a repeated reference to the SAME source file is allowed", "library call", int(guards.require_unique_stems([str(cdir / "same.png")] * 3) == 1 and len(rep) == 2), 1,
             f"{len(rep)} cache entries for 3 references"))
_, st_all = load_shotbench_samples()
real_files = {s_["images"][0] for s_ in load_shotbench_samples()[0]}
rows.append(("same stem: the REAL ShotBench file list passes the check (names only)", "real question file", guards.require_unique_stems(real_files), st_all["image_files"],
             f"{st_all['image_files']} files, {st_all['unique_image_stems']} stems, {st_all['image_names_sharing_a_stem_with_another_extension']} shared"))

# R3: cache configuration provenance: budget, multiple and recipe of an existing manifest must equal the CURRENT request (no relabelling of old outputs)
cfg = tmp / "cache_cfg"
small = prepare_eval_cache([str(cdir / "same.png")], cfg, 4096, 32)
lib_value("R3: cache built at budget 4096 is refused for the evaluator budget 147456", lambda: prepare_eval_cache([str(cdir / "same.png")], cfg, BUDGET_PX, MULTIPLE))
lib_value("R3: cache built at multiple 32 is refused for multiple 16", lambda: prepare_eval_cache([str(cdir / "same.png")], cfg, 4096, 16))
prov_file = cfg / "provenance.json"
pj = json.loads(prov_file.read_text())
rows.append(("R3: the refused requests did not relabel the manifest", "library call", int(pj["budget_px"] == 4096 and pj["multiple"] == 32), 1, f"manifest budget {pj['budget_px']}, multiple {pj['multiple']}"))
pj["resize"] = "some other recipe"
prov_file.write_text(json.dumps(pj))
lib_value("R3: a manifest with another preprocessing recipe is refused", lambda: prepare_eval_cache([str(cdir / "same.png")], cfg, 4096, 32))
# matching inputs still reproduce the frozen preprocessing (the same function the validation and training caches use) byte for byte
from cgt.p3b_vlm.images import shrunk_path

frozen_like = shrunk_path(str(cdir / "same.png"), tmp / "validation_style_cache")
fresh = prepare_eval_cache([str(cdir / "same.png")], tmp / "cache_match")  # fresh cache, default (evaluator) configuration
rows.append(("R3: matching inputs reproduce the frozen preprocessing bytes and hash", "library call",
             int(Path(fresh[str(cdir / "same.png")]).read_bytes() == Path(frozen_like).read_bytes()), 1,
             "evaluator cache entry equals shrunk_path output; sha256 " + json.loads((tmp / "cache_match" / "provenance.json").read_text())["entries"][Path(fresh[str(cdir / "same.png")]).name]["output_sha256"][:16]))
# same-stem fact from the real question file names only (no image read)
_, st_real = load_shotbench_samples()
rows.append(("same-stem fact (file names only): image names sharing a stem with another extension", "real question file", st_real["image_names_sharing_a_stem_with_another_extension"], 0,
             f"{st_real['image_files']} image files, {st_real['unique_image_stems']} distinct stems"))

# the operator's go-ahead writer produces exactly the bound content (synthetic output path)
subprocess.run(["uv", "run", "python", "scripts/p3b_write_go.py", "--out", str(tmp / "written_go")], cwd=ROOT, capture_output=True, text=True, check=True)
rows.append(("scripts/p3b_write_go.py writes the content the evaluator requires (to a synthetic path)", "script", int((tmp / "written_go").read_text() == guards.go_text(TAG, fmj["adapter_sha256"])), 1, ""))

# the dry run reports the actual check size
dr = json.loads((ROOT / "results/phase3b/shotbench_dryrun.json").read_text())
rows.append(("dry run: the template check reports its actual size", "file", int("template_check_n" in dr and "template_mismatches" in dr and not any("of_300" in k for k in dr)), 1, f"n={dr.get('template_check_n')}"))
lines = ["# ShotBench evaluator: fail-closed demonstration (CPU only; stops before any model load, image read or ledger write)", "",
         f"The real adapter verifies against the freeze manifest: sha256 `{real['sha256']}` ({real['checkpoint_percent']}% checkpoint, {real['size_bytes']} bytes).", "",
         "| scenario | command | exit code | expected | message |", "|---|---|---|---|---|"]
for label, cmd, code, exp, msg in rows:
    ok = "PASS" if code == exp else ("SKIPPED" if exp == "skipped" else "FAIL")
    lines.append(f"| {label} | `{cmd[:120]}` | **{code}** | {exp} | {ok}: {msg.replace('|', '/')} |")
(ROOT / "results/phase3b/shotbench_failclosed_demo.md").write_text("\n".join(lines) + "\n")
raise SystemExit(0 if all(r[2] == r[3] or r[3] == "skipped" for r in rows) else 1)
