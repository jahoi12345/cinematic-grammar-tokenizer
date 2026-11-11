"""ShotBench question parsing and prompt building for the VLM evaluator (docs/phase3b_interface.md section 7).
Pure CPU: never reads an image, loads a model or touches the ledger. The user text is built exactly like the ShotQA training turns."""
import ast
import json
from pathlib import Path

import pandas as pd

from cgt.eval.shotbench import CATEGORY_ATTR, SB
from cgt.p3_static_attrs.labels import (
    UnmappedLabel,
    UnmappedQuestion,
    parse_label_set,
    parse_question,
)
from cgt.p3b_vlm.guards import require_under_image_root

PROMPT_TAIL = "Please select the most likely answer from the options above. \n"


def build_user_text(question: str, options: dict[str, str]) -> str:
    """`Question: <q>\\nOptions:\\nA. ..\\nB. ..\\nC. ..\\nD. ..\\nPlease select the most likely answer from the options above. \\n`,
    byte-identical to the user turns of ShotQA `sft.json` (without the leading image tag); options are listed in letter order."""
    return f"Question: {question}\nOptions:\n" + "\n".join(f"{k}. {options[k]}" for k in sorted(options)) + "\n" + PROMPT_TAIL


def load_shotbench_samples(test_tsv: Path = SB / "test.tsv", image_root: Path = SB) -> tuple[list[dict], dict]:
    """Chat samples for the image questions of the 7 static ShotBench categories, in file order, plus coverage statistics.
    Sample: {id: "sb<index>", attribute, images: [original image path (not read here)], messages: [user, assistant(gold letter)],
    options: {letter: text}, answer: letter}. The gold letter is only used to locate the answer token and to score; it is never fed to the model.
    A question is DROPPED (and counted) if an option text does not map into the schema vocabulary or if its prompt does not parse back
    to the same attribute and options. Every image path is resolved and must lie inside `image_root` with no symlink (DataRootViolation, exit 5);
    the static-image mask excludes every row whose type mentions video (image/video rows are never scored). Stats keys: rows_total, rows_video_excluded, rows_static_image, per_attribute{rows_in_category,
    parsed, dropped_unparsed, dropped_roundtrip}, n_samples, unique_image_stems, image_files."""
    sb = pd.read_csv(test_tsv, sep="\t")
    is_video = sb.type.str.contains("video")
    is_static = sb.type.str.contains("image") & ~is_video & sb.category.isin(CATEGORY_ATTR)  # disjoint from the video mask by construction
    stats = {"rows_total": len(sb), "rows_video_excluded": int(is_video.sum()), "per_attribute": {}}
    samples, stems, files = [], set(), set()
    static = sb[is_static]
    stats["rows_static_image"] = len(static)
    for r in static.itertuples():
        attr = CATEGORY_ATTR[r.category]
        st = stats["per_attribute"].setdefault(attr, {"rows_in_category": 0, "parsed": 0, "dropped_unparsed": 0, "dropped_roundtrip": 0})
        st["rows_in_category"] += 1
        options = json.loads(r.options)
        try:
            for text in options.values():
                parse_label_set(attr, text)
        except UnmappedLabel:
            st["dropped_unparsed"] += 1
            continue
        user = build_user_text(r.question, options)
        try:
            a2, o2 = parse_question(user)
        except UnmappedQuestion:
            st["dropped_roundtrip"] += 1
            continue
        if a2 != attr or o2 != options or r.answer not in options:
            st["dropped_roundtrip"] += 1
            continue
        img = require_under_image_root(ast.literal_eval(r.path)[0], image_root)
        stems.add(img.stem)
        files.add(str(img))
        samples.append({"id": f"sb{r.index}", "attribute": attr, "images": [str(img)],
                        "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": r.answer}],
                        "options": options, "answer": r.answer})
        st["parsed"] += 1
    excluded_ids = {f"sb{i}" for i in sb[is_video]["index"]}
    if excluded_ids & {x["id"] for x in samples}:  # contract check: every counted-excluded row is absent from the scored samples
        raise RuntimeError("a row counted as excluded video was selected for scoring")
    names = {Path(f).name for f in files}  # file names only: no image is read
    by_stem: dict[str, set[str]] = {}
    for n in names:
        by_stem.setdefault(Path(n).stem, set()).add(n)
    shared = sum(len(v) for v in by_stem.values() if len(v) > 1)
    stats.update({"n_samples": len(samples), "unique_image_stems": len(stems), "image_files": len(files), "image_names_sharing_a_stem_with_another_extension": shared})
    return samples, stats
