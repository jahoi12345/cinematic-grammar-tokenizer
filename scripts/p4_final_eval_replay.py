"""Fresh 20-clip equivalence replay of the FIXED sealed evaluator's real video path (`p4_final_eval.RealBackend`: warm_up, frame extraction, sample building, letter-logit scoring with the placeholder answer
letter) on 20 ShotQA VALIDATION clips (seed 0 among the 140 of results/phase4/runs/video_v1/validation_logits.json): letters and option scores must equal the retained validation record. Only the QUESTION
text of each clip is used (its options, parsed from the prompt); no gold letter or `solution` is read (the retained gold letters are not used). One heavy GPU job under /tmp/cgt_gpu.lock (owner nonce; waits and
retries every 2 minutes up to 60 minutes); no ShotBench movement, no CameraBench. Writes results/phase4/final_eval_replay.md.
Usage: uv run python scripts/p4_final_eval_replay.py"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_final_eval as EV

from cgt.p4_camera_move import final_run as FR
from cgt.p4_camera_move import gpulock
from cgt.p4_camera_move import video_vlm as V

ROOT = EV.ROOT
BASE = ROOT / "results/phase4/runs/video_v1"


def free_percentage() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=False).stdout
    return int(re.search(r"free percentage:\s*(\d+)%", out).group(1))


def main() -> None:
    if free_percentage() < 25:
        print("memory guard: free percentage below 25%; nothing started")
        sys.exit(12)
    for _ in range(31):
        if gpulock.acquire("p4-final-eval-replay", "replay"):
            break
        print(f"GPU lock held by: {gpulock.owner()}; retry in 2 minutes", flush=True)
        time.sleep(120)
    else:
        print("GPU lock not obtained within 60 minutes; nothing started")
        sys.exit(11)
    try:
        t0 = time.perf_counter()
        rec = json.loads((BASE / "validation_logits.json").read_text())
        pick = sorted(np.random.default_rng(0).choice(len(rec["clips"]), 20, replace=False).tolist())
        clips = [rec["clips"][i] for i in pick]
        raw = {Path(x["videos"][0]).name: x for x in json.loads((ROOT / "data/external/shotqa/grpo.json").read_text()) if x.get("videos")}
        qs = []
        for c in clips:
            text = raw[c]["messages"][0]["content"]  # the QUESTION only
            qs.append({"clip": c, "options": dict(re.findall(r"^([A-D])\. (.*)$", text, re.MULTILINE))})
        spec, _registry, adapter = FR.make_spec("shotbench_movement", "video")
        be = EV.RealBackend(spec, adapter, SimpleNamespace(atoms=[], prior=[], columns=None))
        be._frames = lambda clip: V.extract_frames(ROOT / "data/external/shotqa" / clip, EV.K, EV.TOKENS, EV.FRAMES)
        be.warm_up(clips)
        z = be.video_logits(qs, lambda d, t: None)
        letters_ok = diff = 0
        diff = 0.0
        for i, zi in zip(pick, z, strict=True):
            old = np.array(rec["logits"]["normal"][i])
            letters_ok += int(np.argmax(zi) == np.argmax(old))
            diff = max(diff, float(np.abs(zi - old).max()))
        from cgt.p4_camera_move.final_eval import lsm
        sdiff = max(float(np.abs(lsm(zi) - lsm(np.array(rec["logits"]["normal"][i]))).max()) for i, zi in zip(pick, z, strict=True))
        ok = letters_ok == 20 and diff < 1e-4 and sdiff < 1e-4 and be.adapter_hash() == spec.adapter_sha256
        L = ["# Fresh 20-clip equivalence replay of the fixed sealed evaluator (GPU, ShotQA validation clips; question text only, no gold letter read)", "",
             f"- evaluator code `{subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()[:12]}`; adapter `{spec.adapter_sha256[:16]}` (hash re-verified after scoring); {EV.os.environ.get('MLX_MAX_OPS_PER_BUFFER')} ops per buffer; actual `RealBackend.warm_up/_frames/_sample/video_logits` (placeholder answer letter)",
             f"- 20 clips (seed 0 of the 140 in `validation_logits.json`): letters identical **{letters_ok}/20**; maximum raw-logit difference **{diff:.2e}**, maximum normalized option-score difference **{sdiff:.2e}** (tolerance 1e-4; the record holds 4 decimals)",
             f"- timing {be.timing}; total {time.perf_counter() - t0:.0f} s; result **{'PASS' if ok else 'FAIL'}**"]
        (ROOT / "results/phase4/final_eval_replay.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        sys.exit(0 if ok else 1)
    finally:
        gpulock.release("p4-final-eval-replay")


if __name__ == "__main__":
    main()
