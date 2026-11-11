"""Source-id overlap of the CameraBench public test set (GIF FILE NAMES only, via the sealed store's name listing: no label file is opened) with the ShotQA movement clips and the ShotBench
movement videos (names only, from the question file). Writes results/phase4/camerabench_overlap.json. Usage: uv run python scripts/p4_camerabench_overlap.py"""
import ast
import json
from pathlib import Path

import pandas as pd

from cgt.p4_camera_move.camerabench import CameraBenchTestStore
from cgt.p4_camera_move.labels import load_shotqa_movement_meta, source_id

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cb = CameraBenchTestStore().source_ids()
    qa = {m["source"] for m in load_shotqa_movement_meta()}
    tsv = pd.read_csv(ROOT / "data/external/shotbench/test.tsv", usecols=["type", "path"], sep="\t")
    sb = {source_id(ast.literal_eval(p)[0].removeprefix("video/")) for p in tsv[tsv.type.str.contains("video")].path}
    cbs = set(cb)
    out = {"camerabench_gifs": len(cb), "camerabench_distinct_source_ids": len(cbs), "shotqa_sources": len(qa), "shotbench_movement_sources": len(sb),
           "overlap_with_shotqa_sources": len(cbs & qa), "overlap_with_shotbench_movement_sources": len(cbs & sb), "clips_whose_source_is_in_shotqa": sum(c in qa for c in cb),
           "clips_whose_source_is_in_shotbench_movement": sum(c in sb for c in cb), "example_ids": sorted(cbs)[:3],
           "note": "ids are filename-derived proxies (YouTube ids), not verified titles; no video content and no label file was read"}
    (ROOT / "results/phase4/camerabench_overlap.json").write_text(json.dumps(out, indent=1))
    # names-only metadata so that the overlap can be recomputed by anyone (file-name derived ids, no label, no video content)
    (ROOT / "results/phase4/camerabench_source_ids.json").write_text(json.dumps({
        "note": "filename-derived source ids (YouTube ids) of the CameraBench public test GIFs and their intersections; file-name metadata only, not verified film identities",
        "camerabench_source_ids": sorted(cbs), "n_camerabench_source_ids": len(cbs), "intersection_with_shotqa_movement_sources": sorted(cbs & qa),
        "intersection_with_shotbench_movement_sources": sorted(cbs & sb)}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
