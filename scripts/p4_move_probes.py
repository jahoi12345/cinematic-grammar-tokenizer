"""Label-free direction probes for the film movement slot (docs/phase4_move_probes.md): Check A (time reversal, horizontal and vertical mirror of 400 random shots per film, unchanged flow/depth/feature code
and frozen geometry model) and Check B (motion-size consistency on all shots). They measure CONSISTENCY OF MODEL BEHAVIOUR, never accuracy, and change nothing: no retraining, no threshold, no rule change.
Controls first: identity through the same clip-writing path and a fresh forward re-run must reproduce the existing table on 50 shots (otherwise the run stops with exit 3). A tripwire makes every label path unopenable.
CPU only, lowest priority, memory guard, at most 3 workers. Owned films (--allow-owned) write their results LOCALLY under data/processed/move_probes/ (gitignored) plus a manifest of hashes.
Usage: uv run python scripts/p4_move_probes.py --film night_of_the_living_dead
       uv run python scripts/p4_move_probes.py --film <owned_film_id> --allow-owned"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cgt.p4_camera_move import depth_parallax as D
from cgt.p4_camera_move import film_move as FM
from cgt.p4_camera_move import final_eval as FE
from cgt.p4_camera_move import final_run as FR
from cgt.p4_camera_move import flow as F
from cgt.p4_camera_move import probes as PR

ROOT = Path(__file__).resolve().parents[1]
THREADS = 2
N_SUBSET, N_CONTROL = 400, 50


def _init() -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(THREADS)


def _save(path: Path, series: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".tmp_" + path.name)
    with open(tmp, "wb") as fh:
        np.savez(fh, **series)
    tmp.replace(path)


def _row(sf: dict, sd: dict) -> dict:
    return F.clip_features(sf) | D.clip_features({k: v for k, v in sd.items() if k != "depth"})


def wait_memory(max_wait: int = 3600, step: int = 120) -> None:
    """Free memory >= 25% before every feature batch (P-M1): wait and retry, fail closed after `max_wait` seconds."""
    waited = 0
    while True:
        try:
            FM.memory_guard()
            return
        except FM.Refused:
            if waited >= max_wait:
                raise
            time.sleep(step)
            waited += step


def _names(probe_dir: Path, sid: int, t: str, key: str, impl: str) -> tuple[Path, Path]:
    return Path(probe_dir) / "flow" / f"shot{sid}.{t}.{key}.{impl[:16]}.npz", Path(probe_dir) / "depth" / f"shot{sid}.{t}.{key}.{impl[:16]}.npz"


def _load(path: Path, impl: str) -> dict | None:
    """A cached series, only if it carries the producing identity `impl` (a file without or with another identity is not reused: P-C1)."""
    if not path.exists():
        return None
    z = np.load(path)
    if "impl_identity" not in z.files or str(z["impl_identity"]) != impl:
        return None
    return {k: z[k] for k in z.files if k != "impl_identity"}


def probe_shot(args) -> tuple[int, dict | None, str | None]:
    """One shot: the conformed forward clip once, then (control shots) the identity and a fresh forward re-run compared with the accepted cache, then T, H, V (cached under the producing identity)."""
    sid, film, start_s, end_s, fkey, acc_dir, probe_dir, control = args
    try:
        key = f"{F.cache_key()}.{D.cache_key()}.{fkey}"
        impl = PR.impl_identity()
        cached = {t: (_load(_names(probe_dir, sid, t, key, impl)[0], impl), _load(_names(probe_dir, sid, t, key, impl)[1], impl)) for t in PR.TRANSFORMS}
        todo = [t for t in PR.TRANSFORMS if cached[t][0] is None or cached[t][1] is None]
        out: dict = {"rows": {}, "control": {}}
        if todo or control:
            with tempfile.TemporaryDirectory() as td:
                fwd = Path(td) / "fwd.avi"
                if FM.cut_clip(Path(film), start_s, end_s, fwd) < 3:
                    return sid, None, "fewer than 3 frames"
                if control:  # fresh forward re-run and identity through the same writing path vs the accepted cache (no cache is read for the re-run)
                    fa, da = FM.cache_names(Path(acc_dir), sid, fkey)
                    acc_f, acc_d = np.load(fa), np.load(da)
                    ident = Path(td) / "ident.avi"
                    PR.write_transformed_clip(fwd, ident, None)
                    for name, clip in (("forward_rerun", fwd), ("identity", ident)):
                        wait_memory()
                        sf, sd = F.clip_series(clip), D.clip_series(clip)
                        diff = max([float(np.abs(sf[k] - acc_f[k]).max()) if sf[k].shape == acc_f[k].shape else float("inf") for k in acc_f.files if k in sf]
                                   + [float(np.abs(sd[k].astype(float) - acc_d[k].astype(float)).max()) if sd[k].shape == acc_d[k].shape else float("inf") for k in acc_d.files if k in sd])
                        out["control"][name] = {"series_max_abs_diff": diff, "row": _row(sf, sd)}
                for t in todo:
                    clip = Path(td) / f"{t}.avi"
                    PR.write_transformed_clip(fwd, clip, t)
                    wait_memory()
                    sf = F.clip_series(clip)
                    sd = D.clip_series(clip)
                    fpath, dpath = _names(probe_dir, sid, t, key, impl)
                    _save(fpath, {**sf, "impl_identity": np.array(impl)})
                    _save(dpath, {**sd, "impl_identity": np.array(impl)})
                    cached[t] = (_load(fpath, impl), _load(dpath, impl))
        for t in PR.TRANSFORMS:
            out["rows"][t] = _row(*cached[t])
        return sid, out, None
    except Exception as e:  # noqa: BLE001 - reported, the run stops
        return sid, None, f"{type(e).__name__}: {e}"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def table_text(res_a: dict, film: str) -> list[str]:
    L = []
    names = {"T": "time reversal", "H": "horizontal mirror", "V": "vertical mirror"}
    for t, e in res_a["transforms"].items():
        L += [f"### {t}: {names[t]}", "", "| pair | n | rho primary [95% CI] | band (primary) | confident n | rho confident [95% CI] | band (confident) | null rho (permuted) | forward a-not-a': flipped / unchanged / other (n) | forward a'-not-a: flipped / unchanged / other (n) |",
              "|---|---:|---|---|---:|---|---|---:|---|---|"]
        for p in e["pairs"]:
            s1, s2 = list(p["sets"].values())
            L.append(f"| {p['pair'][0]} / {p['pair'][1]} | {p['n']} | {p['rho_primary']} {p['ci_primary']} | **{p['band_primary']}** | {p['n_confident']} | {p['rho_confident']} {p['ci_confident']} | {p['band_confident']} | {p['null_rho_permuted']} | "
                     f"{s1['flipped']} / {s1['unchanged']} / {s1['other']} ({s1['n']}) | {s2['flipped']} / {s2['unchanged']} / {s2['other']} ({s2['n']}) |")
        L += ["", "Invariant atoms (Spearman rho of the forward and the transformed probability):", "", "| atom | rho [95% CI] | band | null rho (permuted) |", "|---|---|---|---:|"]
        L += [f"| {x['atom']} | {x['rho']} {x['ci']} | {x['band']} | {x['null_rho_permuted']} |" for x in e["invariant"]]
        L.append("")
    return L


def render(film: str, subset_n: int, ctrl: dict, res_a: dict, res_b: dict) -> str:
    L = [f"# Direction probes of the frozen movement model on `{film}`: consistency of model behaviour (NOT accuracy; nothing here is a statement about the film)", "",
         "Docs/phase4_move_probes.md (committed before the run). Check A: time reversal (T), horizontal (H) and vertical (V) mirror of a fixed random subset of "  # noqa: ISC004 - adjacent string literals are intended
         f"{subset_n} shots (seed 0) through the unchanged flow, depth and feature code and the frozen geometry model; shots flagged `short_lt_1s` or `few_pairs_lt_8` are excluded from the primary analysis ({res_a['n_flagged_excluded']} of the subset). "
         "Check B: motion-size consistency on all shots. Bands (fixed in advance): swapping pair rho <= -0.5 'reads direction', -0.5 < rho < -0.2 'weak', rho >= -0.2 'does not read direction'; invariant atom rho >= 0.8 'stable'; AUC >= 0.80 'consistent', 0.60-0.80 'weak', < 0.60 'inconsistent'. "
         "Verdicts are per pair, per transform, per film; there is no overall pass.", "",
         f"**Controls** ({ctrl['n']} shots): identity through the same clip-writing path: series max abs difference {ctrl['identity_series_max']:.2e}, probabilities max abs difference {ctrl['identity_prob_max']:.2e}; fresh forward re-run: series {ctrl['forward_series_max']:.2e}, probabilities {ctrl['forward_prob_max']:.2e} (accepted: series exactly equal, probabilities within 1e-12).",
         f"**Excluded atoms** (no clear partner): {', '.join(res_a['excluded_atoms'])}.", "", "## Check A: reversal and mirror probes", ""]
    L += table_text(res_a, film)
    L += ["## Check B: motion-size consistency", "", "Primary summary `motion_translation` = median over frame pairs of sqrt(tx^2 + ty^2); secondary `motion_affine` = median of sqrt(div^2 + rot^2) (flow series, half-image-widths per second). Groups from the Amendment 10 sets: S = exactly {static}; M = set excludes static; X = static with other atoms.", "",
          f"Shots {res_b['n_all']}, flagged excluded from the primary analysis {res_b['n_flagged_excluded']} (by group S/M/X: {res_b['n_flagged_by_group']}).", ""]
    for name in ("primary", "secondary"):
        e = res_b[name]
        g = e["groups"]
        L += [f"### {name} summary", "", "| group | n | median | IQR |", "|---|---:|---:|---|"] + [f"| {k} | {g[k]['n']} | {g[k]['median']} | {g[k]['iqr']} |" for k in "SMX"]
        L += ["", f"- AUC (M larger than S): **{e['auc_M_vs_S']['auc']}** [95% CI {e['auc_M_vs_S']['ci']}] -> **{e['auc_M_vs_S']['band']}**; X versus S {e['auc_X_vs_S']}; M versus X {e['auc_M_vs_X']}",
              f"- Spearman rho(motion size, p_static) = {e['rho_motion_vs_p_static']} [95% CI {e['rho_ci']}]; permuted null {e['null_rho_permuted']}", ""]
        if name == "primary":
            L += ["| decile | n | motion range | share of shots whose set contains static |", "|---:|---:|---|---:|"] + [f"| {d['decile']} | {d['n']} | {d['motion_range']} | {d['share_static_in_set']} |" for d in e["deciles"]] + [""]
    L += ["## Label access", "", "The only labels opened are the ShotQA TRAIN labels, solely to rebuild the frozen geometry predictor; no label of any film, no test, validation, gold or dev label. The first NOTLD run (`35aada6`) happened under this plan's own statement of that exception BEFORE a recorded decision, which is not retroactive evidence; the absence of any other read is not established. The tripwire of this version denies every other label path.", "", "## Caveats", "", "- The same flow feeds the model, so Check B tests internal consistency and not truth.",
          "- Large flow can come from subject motion rather than camera motion; the check cannot separate the two.",
          "- Check A after time reversal samples the reversed clip from the other end (a one-frame parity difference, not corrected).",
          "- Everything is a statement about the BEHAVIOUR of the frozen model, never about accuracy or about what the film contains."]
    return "\n".join(L) + "\n"


def main() -> None:
    global THREADS
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", required=True)
    ap.add_argument("--allow-owned", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--subset", type=int, default=N_SUBSET)
    ap.add_argument("--probe-cache-dir", help="cache of the transformed series (default data/cache/p4/move_probes/<film>); a new directory = a cold rebuild")
    a = ap.parse_args()
    THREADS = min(a.threads, 2)
    try:
        film = FM.resolve_film(a.film, allow_owned=a.allow_owned, root=ROOT)
        FM.memory_guard()
    except FM.Refused as e:
        print(f"REFUSED (exit {e.code}): {e}")
        sys.exit(e.code)
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 19)
    except OSError:
        pass
    owned = not film.get("public_domain")
    out_dir = ROOT / "data/processed/move_probes" if owned else ROOT / "results/films/move_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = ROOT / f"data/processed/{a.film}_move.parquet"
    acc_dir = ROOT / ("data/cache/p4/notld" if a.film == "night_of_the_living_dead" else f"data/cache/p4/film_move/{a.film}")
    probe_dir = Path(a.probe_cache_dir) if a.probe_cache_dir else ROOT / f"data/cache/p4/move_probes/{a.film}"
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    from p4_final_eval import TrainContext
    with PR.tripwire():
        t0 = time.time()
        PR.check_maps()
        tab = pd.read_parquet(table_path).sort_values("shot_id").reset_index(drop=True)
        shots = pd.read_parquet(film["shots_path"])[["shot_id", "start_s", "end_s", "len_s"]].sort_values("shot_id").reset_index(drop=True)
        assert (tab.shot_id.to_numpy() == shots.shot_id.to_numpy()).all()
        geo = FR.geometry_identity()
        with PR.train_labels_allowed():  # a recorded decision: the ShotQA TRAIN labels, only to rebuild the frozen predictor
            ctx = TrainContext(geo["C"], geo["split"])
        atoms = ctx.atoms
        pcols = [f"p_{x}" for x in atoms]
        film_sha = FM.film_sha256(film["film_path"])
        fk = {int(r.shot_id): FM.film_key(film_sha, r.start_s, r.end_s) for r in shots.itertuples()}
        rng = np.random.default_rng(0)
        sub = np.sort(rng.choice(len(shots), min(a.subset, len(shots)), replace=False))
        ctrl_idx = np.sort(np.random.default_rng([0, 1]).choice(sub, min(N_CONTROL, len(sub)), replace=False))
        jobs = {int(i): (int(shots.shot_id[i]), str(film["film_path"]), float(shots.start_s[i]), float(shots.end_s[i]), fk[int(shots.shot_id[i])], str(acc_dir), str(probe_dir), int(i) in set(ctrl_idx.tolist())) for i in sub}
        results = {}
        with Pool(a.workers, initializer=_init) as pool:
            # phase 1: the control shots first; stop if the controls fail
            first = [jobs[int(i)] for i in ctrl_idx]
            for sid, out, err in pool.imap_unordered(probe_shot, first):
                if err:
                    print(f"control shot {sid} failed: {err}")
                    sys.exit(3)
                results[sid] = out
            cdiff = {"identity_series_max": 0.0, "forward_series_max": 0.0}
            rows_i, rows_f, idx = [], [], []
            for i in ctrl_idx:
                sid = int(shots.shot_id[i])
                cdiff["identity_series_max"] = max(cdiff["identity_series_max"], results[sid]["control"]["identity"]["series_max_abs_diff"])
                cdiff["forward_series_max"] = max(cdiff["forward_series_max"], results[sid]["control"]["forward_rerun"]["series_max_abs_diff"])
                rows_i.append([results[sid]["control"]["identity"]["row"][c] for c in ctx.columns])
                rows_f.append([results[sid]["control"]["forward_rerun"]["row"][c] for c in ctx.columns])
                idx.append(int(i))
            Pi, Pf = (FE.fit_atom_probs(ctx.Xtr, ctx.Ytr, np.array(r), ctx.C) for r in (rows_i, rows_f))
            tabP = tab.loc[idx, pcols].to_numpy()
            cdiff.update({"identity_prob_max": float(np.abs(Pi - tabP).max()), "forward_prob_max": float(np.abs(Pf - tabP).max())})
            print("controls:", cdiff, flush=True)
            if cdiff["identity_series_max"] != 0.0 or cdiff["forward_series_max"] != 0.0 or cdiff["identity_prob_max"] > 1e-12 or cdiff["forward_prob_max"] > 1e-12:
                (out_dir / f"{a.film}_CONTROL_FAILED.json").write_text(json.dumps(cdiff, indent=1))
                print("CONTROL FAILED: stop and report")
                sys.exit(3)
            rest = [jobs[int(i)] for i in sub if int(shots.shot_id[i]) not in results]
            for n, (sid, out, err) in enumerate(pool.imap_unordered(probe_shot, rest, chunksize=2), 1):
                if err:
                    print(f"shot {sid} failed: {err}")
                    sys.exit(4)
                results[sid] = out
                if n % 25 == 0:
                    print(f"  {n}/{len(rest)} shots, {time.time() - t0:.0f}s", flush=True)
        sids = [int(shots.shot_id[i]) for i in sub]
        P_f = tab.loc[sub, pcols].to_numpy()
        P_by_t = {t: FE.fit_atom_probs(ctx.Xtr, ctx.Ytr, np.array([[results[s]["rows"][t][c] for c in ctx.columns] for s in sids]), ctx.C) for t in PR.TRANSFORMS}
        flagged_sub = (tab.short_lt_1s | tab.few_pairs_lt_8).to_numpy()[sub]
        res_a = PR.evaluate_check_a(P_f, P_by_t, atoms, flagged_sub)
        # Check B on all shots from the cached accepted flow series
        prim, sec = [], []
        for r in shots.itertuples():
            fa, _ = FM.cache_names(acc_dir, int(r.shot_id), fk[int(r.shot_id)])
            z = np.load(fa)
            p, s = PR.motion_summaries({k: z[k] for k in ("tx", "ty", "div", "rot")})
            prim.append(p)
            sec.append(s)
        sets = [x.split(" + ") for x in tab.move_atoms]
        res_b = PR.evaluate_check_b(sets, tab["p_static"].to_numpy(), {"primary": np.array(prim), "secondary": np.array(sec)}, (tab.short_lt_1s | tab.few_pairs_lt_8).to_numpy())
        ctrl = {**cdiff, "n": len(ctrl_idx)}
        md = render(a.film, len(sub), ctrl, res_a, res_b)
        base = out_dir / f"{a.film}_probes"
        base.with_suffix(".md").write_text(md)
        base.with_suffix(".json").write_text(json.dumps({"film": a.film, "subset_n": len(sub), "controls": ctrl, "check_a": res_a, "check_b": res_b, "wording": "consistency of model behaviour, never accuracy, never a claim about the film"}, indent=1))
        np.savez_compressed(out_dir / f"{a.film}_probes_vectors.npz", shot_ids=np.array(sids), forward=P_f, flagged=flagged_sub, atoms=np.array(atoms), **{f"transformed_{t}": P_by_t[t] for t in PR.TRANSFORMS})
        def git(*c):
            return subprocess.run(["git", *c], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
        dirty = [x for x in git("status", "--porcelain", "--", "src", "scripts", "docs/phase4_move_probes.md").splitlines() if x.strip()]
        manifest = {"film": a.film, "owned_local_only": owned, "started": started, "ended": datetime.now().astimezone().isoformat(timespec="seconds"), "code_commit": git("rev-parse", "HEAD"), "clean_tree": not dirty, "dirty_files": dirty[:10],
                    "frozen_predictor": {"C": geo["C"], "config_hash": geo["config_hash"], "geometry_inputs_sha256": geo["inputs"], "split_manifest_sha256": hashlib.sha256(json.dumps(geo["split"], sort_keys=True).encode()).hexdigest()},
                    "train_labels": "only the ShotQA TRAIN labels, to rebuild the frozen predictor; absence of any other read is not established",
                    "cache_identity": PR.impl_identity(), "probe_cache_dir": str(probe_dir),
                    "inputs": {"film_sha256": film_sha, "shots_table_sha256": sha(film["shots_path"]), "forward_table_sha256": sha(table_path), "doc": sha(ROOT / "docs/phase4_move_probes.md"),
                               "probes_py_sha256": sha(ROOT / "src/cgt/p4_camera_move/probes.py"), "script_sha256": sha(Path(__file__))},
                    "outputs": {f"{a.film}_probes.md": sha(base.with_suffix(".md")), f"{a.film}_probes.json": sha(base.with_suffix(".json")), f"{a.film}_probes_vectors.npz": sha(out_dir / f"{a.film}_probes_vectors.npz")}, "seconds": round(time.time() - t0, 1)}
        mp = ROOT / "results/films/move_probes" / f"{a.film}_manifest.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(manifest, indent=1))
        print(md[:3500])


if __name__ == "__main__":
    main()
