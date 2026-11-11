"""8-frames-as-images adapter for training/scoring a VLM (Qwen3-VL via mlx-vlm 0.7.4) on camera-movement clips (docs/phase4_interface.md section 6).
mlx-vlm's trainer has no video path, so a clip becomes K uniformly sampled frames passed as K images of ONE sample: the chat template puts K image placeholders into the first user
turn and the processor makes one `image_grid_thw` row per frame. Frames are shrunk to a per-frame token budget (32-pixel cells, as in Phase 3B). Pure CPU helpers here; the model
is only touched by the scripts. Never reads ShotBench movement or CameraBench: it only sees the clip paths and rows it is given."""
import re
from pathlib import Path

import cv2
from PIL import Image

from cgt.p3b_vlm.images import MULTIPLE, budget_size
from cgt.p4_camera_move.flow import file_digest

VIDEO_TAG = "<video>"
PREFIX = "These {k} frames are sampled in time order from one movie shot. "


MIN_PIXELS = 4096  # the video processor's own lower bound; the image processor's default (65,536 px = 64 tokens) would UPSCALE small frames above their token budget


def configure_processor(processor) -> None:
    """Lower the Qwen3-VL image processor's minimum pixel count so frames keep the size they were shrunk to (a 384 x 160 frame stays 60 tokens, not 78). Must be applied
    identically at training and at scoring time."""
    ip = processor.image_processor
    if not hasattr(ip, "min_pixels"):
        raise AttributeError(f"{type(ip).__name__} has no min_pixels: the processor changed, re-check how small frames are resized")
    ip.min_pixels = MIN_PIXELS


def frame_indices(n_total: int, k: int) -> list[int]:
    """Frame indices at the centres of k equal time bins of a clip with n_total frames (strictly increasing for n_total >= k)."""
    return [min(n_total - 1, int((i + 0.5) * n_total / k)) for i in range(k)]


def budget_px(tokens_per_frame: int) -> int:
    """Pixel budget of one frame: tokens x (32 x 32) (a Qwen3-VL image token covers a 32 x 32 pixel cell after the 2 x 2 merge)."""
    return tokens_per_frame * MULTIPLE * MULTIPLE


def extract_frames(clip_path: str | Path, k: int, tokens_per_frame: int, cache_dir: Path) -> list[Path]:
    """K frames of the clip, RGB, resized (aspect kept, sides multiples of 32, area <= budget) and saved as JPEG q92 under
    `<cache_dir>/<tokens>t/<clip name>.<content hash 12>.<k>f.<i>.jpg`. Idempotent; raises ValueError if a frame cannot be decoded."""
    clip_path = Path(clip_path)
    key = file_digest(clip_path)[:12]
    out = [Path(cache_dir) / f"{tokens_per_frame}t" / f"{clip_path.name}.{key}.{k}f.{i}.jpg" for i in range(k)]
    if all(p.exists() for p in out):
        return out
    cap = cv2.VideoCapture(str(clip_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < k:
        raise ValueError(f"{clip_path}: {n} frames < {k}")
    for i, fi in enumerate(frame_indices(n, k)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            raise ValueError(f"{clip_path}: frame {fi} unreadable")
        im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        im = im.resize(budget_size(*im.size, budget_px(tokens_per_frame), MULTIPLE), Image.BICUBIC)
        out[i].parent.mkdir(parents=True, exist_ok=True)
        im.save(out[i], quality=92)
    cap.release()
    return out


def user_text(question_content: str, k: int) -> str:
    """The ShotQA movement question with its `<video>` tag removed and a one-line frame description in front (the options and the closing line are unchanged)."""
    body = question_content.replace(VIDEO_TAG, "", 1)
    if not body.startswith("Question:"):
        raise ValueError("unexpected movement question format")
    return PREFIX.format(k=k) + body


def build_sample(row: dict, frames: list[Path]) -> dict:
    """Chat sample for mlx-vlm's VisionDataset: {id, images: [K paths], messages: [user, assistant(letter)]}; `row` is a parsed label row (from MovementLabelStore)."""
    q = "Question: What's the camera movement in this movie shot?\nOptions:\n" + "\n".join(f"{k}. {v}" for k, v in sorted(row["options"].items())) + "\nPlease select the most likely answer from the options above. \n"
    return {"id": row["clip"], "images": [str(p) for p in frames],
            "messages": [{"role": "user", "content": user_text("<video>" + q, len(frames))}, {"role": "assistant", "content": row["gold_letter"]}]}


def image_token_count(image_grid_thw) -> int:
    """Visual tokens of a processed sample = sum over frames of t*h*w / 4 (2 x 2 spatial merge)."""
    return int(sum(int(t) * int(h) * int(w) // 4 for t, h, w in image_grid_thw))


def check_prompt_format(text: str) -> bool:
    """A sample prompt starts with the frame line and keeps the four lettered options."""
    return bool(re.match(r"^These \d+ frames are sampled in time order from one movie shot\. Question: ", text)) and all(f"\n{c}. " in text for c in "ABCD")
