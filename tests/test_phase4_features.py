"""Synthetic checks for the corrected flow units, schema and cache provenance."""
import numpy as np

from cgt.p4_camera_move import flow


def test_analytic_flow_uses_half_width_and_divergence_is_relative_scale():
    field = np.broadcast_to(np.array([2., -1.], dtype=np.float32), (160, 384, 2)).copy()
    fit = flow.fit_affine(field)
    np.testing.assert_allclose([fit["tx"], fit["ty"]], [2 / 192, -1 / 192], atol=1e-9)
    y, x = np.mgrid[:160, :384]
    expansion = np.stack([.01 * (x - 192), .01 * (y - 80)], axis=-1).astype(np.float32)
    assert abs(flow.fit_affine(expansion)["div"] - .01) < 1e-8


def test_complete_flow_schema_has_119_ordered_features():
    names = ["tx", "ty", "div", "rot", "shear1", "shear2", "r2", "resid_rms", "flow_rms",
             "res_top_bottom", "res_left_right", "div_spread", "tx_spread"]
    summary_names = ["mean", "median", "std", "p10", "p90", "sum", "maxabs", "h1_minus_h2"]
    expected = [f"{name}_{stat}" for name in names for stat in summary_names]
    expected += [f"{name}_{stat}" for name in ("tx", "ty", "rot")
                 for stat in ("jitter_std", "jitter_hf_share", "jitter_zcr", "diff_std")]
    expected += ["n_pairs", "fps_used", "duration_s"]
    series = {name: np.linspace(.1, .9, 30) for name in names}
    series.update(fps_used=np.array(12.), duration_s=np.array(2.5))
    features = flow.clip_features(series)
    assert len(features) == 119 and list(features) == expected
    assert np.isfinite(list(features.values())).all()
    assert abs(features["tx_sum"] - series["tx"].sum() / 12) < 1e-12


def test_flow_cache_key_is_stable_but_changes_for_replaced_same_name_content(tmp_path):
    path = tmp_path / "same.mp4"
    path.write_bytes(b"synthetic clip version one")
    original = flow.clip_key(path)
    assert flow.clip_key(path) == original
    path.write_bytes(b"synthetic clip version two")
    assert flow.clip_key(path) != original
    other = tmp_path / "different.mp4"
    other.write_bytes(path.read_bytes())
    assert flow.clip_key(other) == flow.clip_key(path)
