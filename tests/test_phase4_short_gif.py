"""Amendment15 retains short GIFs with the registered frame-index bin rule."""
import numpy as np
import pytest
from PIL import Image

from cgt.p4_camera_move import gif as gif_inputs


@pytest.mark.parametrize(
    ("n_frames", "expected"),
    [(1, [0] * 8), (3, [0, 0, 0, 1, 1, 2, 2, 2]), (7, [0, 1, 2, 3, 3, 4, 5, 6])],
)
def test_short_gif_keeps_eight_ordered_bin_centre_frames(tmp_path, n_frames, expected):
    frames = [Image.new("RGB", (96, 64), (30 * i, 0, 0)) for i in range(n_frames)]
    source = tmp_path / "synthetic.gif"
    frames[0].save(source, save_all=True, append_images=frames[1:], duration=70, loop=0)
    paths = gif_inputs.gif_frames_k(source, 8, 64, tmp_path / "cache")
    assert len(paths) == 8
    got = []
    for path in paths:
        with Image.open(path) as image:
            assert image.mode == "RGB"
            assert image.width * image.height <= 64 * 1024
            got.append(round(float(np.asarray(image).mean(axis=(0, 1))[0]) / 30))
    assert got == expected
    mtimes = [path.stat().st_mtime_ns for path in paths]
    assert gif_inputs.gif_frames_k(source, 8, 64, tmp_path / "cache") == paths
    assert [path.stat().st_mtime_ns for path in paths] == mtimes
