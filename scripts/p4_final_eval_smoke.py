"""GPU smoke test of the REAL backend of the sealed evaluator (scripts/p4_final_eval.py `RealBackend`), WITHOUT any test data or label: six ShotQA TRAIN clips and two public-test sample GIFs (names and frames
only, as in the earlier feasibility sample; their labels are never read). It proves that (1) the frozen adapter loads and the dummy-letter independence probe passes, (2) the letter logits scored with the
placeholder answer letter "A" equal the logits scored with the TRUE gold letter through the validated `score_samples` path (so scoring needs no label), (3) GIF frames flow through the same scorer, (4) the
geometry path (train fit, primary and transductive variants) runs on features prepared from GIFs. No ledger, no go file, no ShotBench movement video, no label file. One heavy GPU job under the lock.
Usage: uv run python scripts/p4_final_eval_smoke.py     -> results/phase4/final_eval_smoke.md"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_final_eval as EV
import p4_final_prepare as PREP

from cgt.p4_camera_move import final_run as FR
from cgt.p4_camera_move import gif as G
from cgt.p4_camera_move import gpulock
from cgt.p4_camera_move import video_vlm as V
from cgt.p4_camera_move.labels import MovementLabelStore

ROOT = EV.ROOT


def main() -> None:
    if not gpulock.acquire("p4-final-eval-smoke", "smoke"):
        print(f"GPU lock held by: {gpulock.owner()}; nothing was started")
        sys.exit(11)
    try:
        t0 = time.perf_counter()
        spec, _registry, adapter = FR.make_spec("shotbench_movement", "combined")
        ctx = EV.TrainContext(spec.geometry_C, FR.geometry_identity()["split"])
        man = FR.geometry_identity()["split"]
        store = MovementLabelStore(man["clips"]["val"])
        rows = [r for r in store.rows(man["clips"]["train"][:40]) if r["atoms"]][:6]  # TRAIN clips only
        qs = [{"clip": r["clip"], "options": r["options"], "option_atoms": r["option_atoms"]} for r in rows]
        be = EV.RealBackend(spec, adapter, ctx)
        be._frames = lambda clip: V.extract_frames(ROOT / "data/external/shotqa" / clip, EV.K, EV.TOKENS, EV.FRAMES)
        be.warm_up([q["clip"] for q in qs])
        done = []
        z = be.video_logits(qs, lambda d, t: done.append((d, t)))
        ref = be._score([V.build_sample(r, be._frames_cache[r["clip"]]) for r in rows], [r["gold_letter"] for r in rows])
        diff = max(float(np.abs(a - b).max()) for a, b in zip(z, ref, strict=True))
        gifs = sorted((ROOT / "data/cache/p4/camerabench_sample").glob("*.gif"))[:2]
        gspec = FR.Spec("camerabench_public_test", "combined", spec.adapter_sha256, spec.config_hash, spec.w_s, spec.w_g, spec.freeze_sha256)
        bg = EV.RealBackend(gspec, adapter, ctx)
        bg.model, bg.ds, bg.ids4 = be.model, be.ds, be.ids4
        bg._frames = lambda clip: G.gif_frames_k(ROOT / "data/cache/p4/camerabench_sample" / clip, EV.K, EV.TOKENS, EV.FRAMES)
        bg._frames_cache = {g.name: bg._frames(g.name) for g in gifs}
        gq = [{"clip": g.name, "options": {"A": "Pan left", "B": "Pan right", "C": "Tilt up", "D": "Push in"}} for g in gifs]
        gz = bg.video_logits(gq, lambda d, t: None)
        tmp = Path(tempfile.mkdtemp())
        PREP.prepare_paths(gifs, "gif", tmp / "f.parquet", cache=tmp, workers=2)
        EV.FEATURES["camerabench_public_test"] = tmp / "f.parquet"
        names = [g.name for g in gifs]
        p0, p1 = bg.geometry_probs(names, False), bg.geometry_probs(names, True)
        ok = diff < 1e-4 and done[-1] == (6, 6) and all(np.isfinite(x).all() and x.shape == (4,) for x in gz) and p0.shape == p1.shape == (2, len(ctx.atoms)) and np.isfinite(p0).all() and np.isfinite(p1).all()
        L = ["# Smoke test of the real backend of the sealed evaluator (GPU; TRAIN clips and two sample GIFs; no test label, no ledger, no go file)", "",
             f"- adapter `{spec.adapter_sha256[:16]}` loaded with the pinned base model (frames, load and dummy-letter probe {be.timing}); `MLX_MAX_OPS_PER_BUFFER={EV.os.environ.get('MLX_MAX_OPS_PER_BUFFER')}`",
             f"- 6 ShotQA TRAIN clips: logits with the placeholder letter vs logits with the true gold letter (validated `score_samples` path): max abs difference **{diff:.2e}** (tolerance 1e-4): scoring needs no label",
             f"- 2 sample GIFs through `gif_frames_k` and the same scorer: logits finite, shape (4,); first GIF logits {np.round(gz[0], 3).tolist()}",
             f"- geometry on GIF features: primary and transductive variants {p0.shape}, finite; max abs difference between variants {float(np.abs(p0 - p1).max()):.3f} (domain shift of the 2 GIFs)",
             f"- total {time.perf_counter() - t0:.0f} s; result: **{'PASS' if ok else 'FAIL'}**"]
        (ROOT / "results/phase4/final_eval_smoke.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        sys.exit(0 if ok else 1)
    finally:
        gpulock.release("p4-final-eval-smoke")


if __name__ == "__main__":
    main()
