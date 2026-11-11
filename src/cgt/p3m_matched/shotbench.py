"""ShotBench questions for the SEALED matched evaluation (docs/p3m_matched_protocol.md section 2.8). Pure CPU, never opens an image.
Called only by `cgt.p3m_matched.sealed_eval.run_sealed` AFTER every preflight refusal has passed. Unit-tested on synthetic fixtures only.
All static image rows of the 7 categories are kept (nothing is dropped); options keep their stored order (json first, ast fallback, as the
official script)."""
import ast
import json
from pathlib import Path

from cgt.p3b_vlm.guards import FailClosed, require_under_image_root

CATEGORY_ATTR = {"shot size": "size", "lens size": "lens", "shot framing": "framing", "camera angle": "angle",
                 "lighting type": "light_type", "lighting": "light_cond", "composition": "composition"}  # = cgt.eval.shotbench.CATEGORY_ATTR


def safe_load(x):
    try:
        return json.loads(x)
    except (TypeError, ValueError):
        return ast.literal_eval(x)


def load_matched_questions(test_tsv: Path, image_root: Path) -> tuple[list[dict], dict]:
    import pandas as pd
    sb = pd.read_csv(test_tsv, sep="\t")
    is_video = sb["type"].astype(str).str.contains("video")
    static = sb[sb["type"].astype(str).str.contains("image") & ~is_video & sb["category"].isin(CATEGORY_ATTR)]
    qs, per = [], {}
    for r in static.itertuples():
        attr = CATEGORY_ATTR[r.category]
        options = safe_load(r.options)
        if not isinstance(options, dict) or r.answer not in options:
            raise FailClosed("a ShotBench row has malformed options or a gold letter outside its options: refusing a partial set")
        img = require_under_image_root(ast.literal_eval(r.path)[0], image_root)
        qs.append({"id": f"sb{r.index}", "attribute": attr, "image": str(img), "question": r.question, "options": options, "answer": r.answer})
        per[attr] = per.get(attr, 0) + 1
    return qs, {"rows_total": len(sb), "rows_video_excluded": int(is_video.sum()), "rows_static_image": len(static),
                "per_attribute_rows": per}
