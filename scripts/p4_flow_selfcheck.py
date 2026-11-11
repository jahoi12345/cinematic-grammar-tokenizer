"""CPU self-check of the flow-geometry conventions on SYNTHETIC data (no clip, no ShotBench/CameraBench): fit_affine on analytic flow fields and the full
DIS path on a textured image with known global motion. Writes results/phase4/flow_selfcheck.md. Usage: uv run python scripts/p4_flow_selfcheck.py"""
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from cgt.p4_camera_move import flow as F

ROOT = Path(__file__).resolve().parents[1]
rows = []


def check(name, ok, detail):
    rows.append((name, ok, detail))
    print(("PASS" if ok else "FAIL"), name, detail, flush=True)


h, w = 160, 384
ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
half = w / 2
xn, yn = (xs - w / 2) / half, (ys - h / 2) / half


def field(u, v):
    return np.stack([u * half, v * half], axis=-1).astype(np.float32)  # normalised -> pixels


r = F.fit_affine(field(0.05 + 0 * xn, -0.02 + 0 * xn))
check("pure translation recovered", abs(r["tx"] - 0.05) < 1e-3 and abs(r["ty"] + 0.02) < 1e-3 and abs(r["div"]) < 1e-3 and abs(r["rot"]) < 1e-3, f"tx={r['tx']:.4f} ty={r['ty']:.4f}")
r = F.fit_affine(field(0.03 * xn, 0.03 * yn))
check("zoom-in (outward radial flow): divergence > 0, no translation", abs(r["div"] - 0.03) < 1e-3 and abs(r["tx"]) < 1e-3 and r["r2"] > 0.99, f"div={r['div']:.4f} r2={r['r2']:.3f}")
r = F.fit_affine(field(-0.04 * yn, 0.04 * xn))
check("rotation: curl recovered, divergence 0", abs(r["rot"] - 0.04) < 1e-3 and abs(r["div"]) < 1e-3, f"rot={r['rot']:.4f}")
# exact analytic reference of the review (F1): constant flow u = 2 px, v = -1 px at w = 384 -> HALF-image-width units
r = F.fit_affine(np.dstack([np.full((160, 384), 2.0), np.full((160, 384), -1.0)]).astype(np.float32))
check("F1 exact analytic field (u=2 px, v=-1 px, w=384): tx = 2/192, ty = -1/192 in half-width units", abs(r["tx"] - 0.010416667) < 1e-7 and abs(r["ty"] + 0.005208333) < 1e-7 and abs(r["div"]) < 1e-9,
      f"tx={r['tx']:.9f} (0.010416667) ty={r['ty']:.9f} (-0.005208333); in full-width fractions: {r['tx'] / 2:.9f}, {r['ty'] / 2:.9f}")
cx, cy = (w - 1) / 2, (h - 1) / 2
r = F.fit_affine(np.stack([0.01 * (xs - cx), 0.01 * (ys - cy)], axis=-1).astype(np.float32))
check("F1 divergence is the exact relative scale rate per pair (1% growth -> div 0.01)", abs(r["div"] - 0.01) < 1e-4, f"div={r['div']:.5f}")
import tempfile

vp = Path(tempfile.mkdtemp()) / "pan.avi"
wr = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (384, 160))
tex0 = cv2.GaussianBlur((np.random.default_rng(1).random((420, 900)) * 255).astype(np.uint8), (0, 0), 3)
for t in range(24):
    fr = cv2.warpAffine(tex0, np.float32([[1, 0, 2.0 * t], [0, 1, 0]]), (900, 420), borderMode=cv2.BORDER_REFLECT)[130:290, 100:484]
    wr.write(cv2.cvtColor(np.ascontiguousarray(fr), cv2.COLOR_GRAY2BGR))
wr.release()
sv = F.clip_series(vp)
check("F1 per-second scaling: 2 px/frame image-right motion at 12 fps gives tx about 0.125 per second (half-width units)", 0.115 < float(np.median(sv["tx"])) < 0.135 and float(sv["fps_used"]) == 12.0,
      f"median tx/s={float(np.median(sv['tx'])):.4f} (analytic 0.1250; 0.0625 in full-width fractions)")
# parallax: a translation that is NOT affine over the frame (near top half moves 8x more than the far bottom half) leaves a residual and lowers r2; uniform translation does not
r_par = F.fit_affine(field(np.where(yn < 0, 0.08, 0.01), 0 * xn))
r_uni = F.fit_affine(field(0.05 + 0 * xn, 0 * xn))
check("depth-dependent (non-affine) translation raises the residual proxies over uniform translation",
      r_par["resid_rms"] > 10 * r_uni["resid_rms"] and r_par["r2"] < r_uni["r2"] and r_par["res_top_bottom"] != 0,
      f"resid_rms {r_par['resid_rms']:.4f} vs {r_uni['resid_rms']:.4f}; r2 {r_par['r2']:.3f} vs {r_uni['r2']:.3f}")
# robustness: a moving object patch must not drag the global fit
fl = field(0.05 + 0 * xn, 0 * xn)
fl[40:100, 150:260] += np.array([60.0, 30.0], dtype=np.float32)
r = F.fit_affine(fl)
check("robust to a moving object covering ~19% of the frame", abs(r["tx"] - 0.05) < 0.01, f"tx={r['tx']:.4f} (true 0.05)")
# full path: textured image with known global motion through sample/DIS/fit
rng = np.random.default_rng(0)
tex = cv2.GaussianBlur((rng.random((420, 900)) * 255).astype(np.uint8), (0, 0), 3)
frames = []
for t in range(12):
    M = np.float32([[1, 0, 2.0 * t], [0, 1, 0]])  # content moves right 2 px per frame
    frames.append(cv2.resize(cv2.warpAffine(tex, M, (900, 420), borderMode=cv2.BORDER_REFLECT)[130:290, 100:484], (384, 160)))
rr = [F.fit_affine(F.dis_flow(a, b)) for a, b in pairwise(frames)]
tx = float(np.mean([x["tx"] for x in rr]))
check("DIS path: content moving right gives tx > 0 (a pan to the LEFT)", 0.008 < tx < 0.014, f"tx={tx:.4f} per pair (2 px / 192 = 0.0104)")
zz = []
for t in range(12):
    s = 1.0 + 0.01 * t
    M = cv2.getRotationMatrix2D((450, 210), 0, s)
    zz.append(cv2.resize(cv2.warpAffine(tex, M, (900, 420), borderMode=cv2.BORDER_REFLECT)[130:290, 100:484], (384, 160)))
rz = [F.fit_affine(F.dis_flow(a, b)) for a, b in pairwise(zz)]
dv = float(np.mean([x["div"] for x in rz]))
check("DIS path: scale growth 1% per frame gives divergence > 0", 0.004 < dv < 0.02, f"div={dv:.4f} per pair")
ser = {k: np.linspace(0, 1, 30) for k in F.SERIES_KEYS} | {"fps_used": np.array(12.0), "duration_s": np.array(2.5)}
ft = F.clip_features(ser)
check("clip_features produces finite values for a smooth series", all(np.isfinite(v) for k, v in ft.items() if "hf_share" not in k), f"{len(ft)} features")
(ROOT / "results/phase4/flow_selfcheck.md").write_text("# Flow geometry self-check (synthetic, CPU)\n\n| check | result | detail |\n|---|---|---|\n" + "\n".join(
    f"| {n} | **{'PASS' if ok else 'FAIL'}** | {d} |" for n, ok, d in rows) + f"\n\nAll {len(rows)} checks passed: {all(r[1] for r in rows)}.\n")
raise SystemExit(0 if all(r[1] for r in rows) else 1)
