"""Color barcode and shot-length plots."""
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .rhythm import rolling_asl


def _lab_to_rgb(lab) -> np.ndarray:
    px = np.asarray(lab, dtype=np.float32).reshape(1, -1, 3)
    return cv2.cvtColor(px, cv2.COLOR_LAB2RGB).reshape(-1, 3)


def color_barcode(df, path: str) -> None:
    """One stripe per shot, coloured by the shot's dominant palette colour."""
    dom = _lab_to_rgb(np.stack([np.asarray(p).reshape(-1, 3)[0] for p in df["palette_lab"]]))
    img = np.repeat(np.clip(dom, 0, 1)[None, :, :], 100, axis=0)
    fig, ax = plt.subplots(figsize=(14, 2.5))
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def shot_length_plot(df, path: str) -> None:
    shots = list(zip(df["start_s"], df["end_s"]))
    x, y = rolling_asl(shots)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(df["start_s"] / 60, df["len_s"], s=6, alpha=0.3, label="shot")
    ax.plot(x / 60, y, color="crimson", lw=2, label="rolling ASL (5 min)")
    ax.set_yscale("log")
    ax.set_xlabel("runtime (min)")
    ax.set_ylabel("shot length (s)")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
