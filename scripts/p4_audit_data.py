"""Phase 4 data audit (CPU only, no training): ShotQA movement clips, ShotBench movement question format, and a ShotQA-vs-ShotBench clip leakage check.
ShotBench videos are read ONLY for the leakage check (sha256 + three-frame dHash); nothing is trained, tuned or selected on them, and the correct answers of the
ShotBench movement questions are never read into any statistic. Writes results/phase4/data_audit.json. Usage: uv run python scripts/p4_audit_data.py"""
import ast
import collections
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data/external/shotqa"
SB = ROOT / "data/external/shotbench"
SB_VIDEOS = ROOT / "data/cache/p4_leak/video"  # extracted from videos.tar for the leakage check only (gitignored)
OUT = ROOT / "results/phase4/data_audit.json"

ATOM_TO_CLASS = {"static shot": "static", "push in": "push-in", "pull out": "pull-out", "boom up": "boom-up", "boom down": "boom-down",
                 "pan left": "pan-left", "pan right": "pan-right", "tilt up": "tilt-up", "tilt down": "tilt-down", "zoom in": "zoom-in",
                 "zoom out": "zoom-out", "move to the left": "move-left", "move to the right": "move-right", "arc": "arc", "camera roll": "roll"}
ALIAS = {"trucking left": "move-left", "trucking right": "move-right"}  # same lateral translation under another name
NOT_IN_SCHEMA = {"tracking", "rack focus", "dolly zoom"}  # tracking: follow shot (no direction); rack focus: focus change, not camera motion; dolly-zoom is a schema class


def schema_moves() -> list[str]:
    s = yaml.safe_load((ROOT / "configs/schema.yaml").read_text())

    def find(d):
        if isinstance(d, dict):
            if "move" in d and isinstance(d["move"], dict) and "values" in d["move"]:
                return d["move"]["values"]
            for v in d.values():
                r = find(v)
                if r:
                    return r
        return None
    return find(s)


def parse_answer(label: str) -> tuple[str, list[str]]:
    """(kind, atoms): kind in single | simultaneous ('A and B', 'A, B') | sequential ('Firstly A, then B ...'); atoms lower-cased, stripped."""
    t = label.lower().strip()
    kind_seq = t.startswith("firstly") or " then " in t
    atoms = [a.strip(" .") for a in re.split(r",|\band\b|\bthen\b|\bfirstly\b", t) if a.strip(" .")]
    return ("sequential" if kind_seq else "simultaneous" if len(atoms) > 1 else "single"), atoms


def atom_class(a: str) -> str:
    if a in ATOM_TO_CLASS:
        return ATOM_TO_CLASS[a]
    if a in ALIAS:
        return ALIAS[a]
    return "UNMAPPED:" + a


def ffprobe(p: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames,duration:format=duration,size",
                        "-of", "json", str(p)], capture_output=True, text=True, check=False)
    j = json.loads(r.stdout or "{}")
    s = (j.get("streams") or [{}])[0]
    fr = s.get("r_frame_rate", "0/1").split("/")
    fps = float(fr[0]) / float(fr[1]) if len(fr) == 2 and float(fr[1]) else 0.0
    dur = float(s.get("duration") or j.get("format", {}).get("duration") or 0)
    return {"codec": s.get("codec_name"), "w": s.get("width"), "h": s.get("height"), "fps": fps, "dur": dur, "size": int(j.get("format", {}).get("size", 0)),
            "nb_frames": int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None}


def source_id(name: str) -> str:
    return re.split(r"\.(?:webm|mp4|mkv|mov)_", name, maxsplit=1)[0]


def dhash(gray: np.ndarray) -> int:
    sm = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (sm[:, 1:] > sm[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def frame_hashes(p: Path) -> list[int] | None:
    cap = cv2.VideoCapture(str(p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for f in (0.25, 0.5, 0.75):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(n * f)))
        ok, fr = cap.read()
        if not ok:
            cap.release()
            return None
        out.append(dhash(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)))
    cap.release()
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    classes = schema_moves()
    res: dict = {"schema_move_classes": classes}
    # ---------------- (a) ShotQA movement clips
    grpo = json.loads((QA / "grpo.json").read_text())
    sft = json.loads((QA / "sft.json").read_text())
    vrows = [x for x in grpo if x.get("videos")]
    res["shotqa"] = {"sft_rows": len(sft), "sft_rows_with_video": sum(1 for x in sft if x.get("videos")), "grpo_rows": len(grpo), "grpo_video_rows": len(vrows),
                     "mp4_files_on_disk": len(list(QA.glob("*.mp4"))), "mp4_bytes": sum(p.stat().st_size for p in QA.glob("*.mp4"))}
    referenced = [x["videos"][0].removeprefix("data/") for x in vrows]
    on_disk = {p.name for p in QA.glob("*.mp4")}
    res["shotqa"].update({"referenced_clips": len(referenced), "unique_referenced": len(set(referenced)), "referenced_missing_on_disk": sorted(set(referenced) - on_disk)[:5],
                          "n_referenced_missing_on_disk": len(set(referenced) - on_disk), "on_disk_not_referenced": len(on_disk - set(referenced))})
    rows = []
    for x in vrows:
        t = x["messages"][0]["content"]
        opts = dict(re.findall(r"^([A-D])\. (.*)$", t, re.MULTILINE))
        m = re.search(r"<answer>\s*([A-D])\s*</answer>", x["solution"])
        gold = opts[m.group(1)] if m and m.group(1) in opts else None
        kind, atoms = parse_answer(gold) if gold else ("none", [])
        rows.append({"clip": x["videos"][0].removeprefix("data/"), "question": t.split("\n")[0], "n_options": len(opts), "gold": gold, "kind": kind, "atoms": atoms,
                     "gold_letter": m.group(1) if m else None, "options": opts})
    df = pd.DataFrame(rows)
    res["shotqa"]["question_texts"] = df.question.value_counts().to_dict()
    res["shotqa"]["options_per_question"] = df.n_options.value_counts().to_dict()
    res["shotqa"]["clips_without_label"] = int(df.gold.isna().sum())
    res["shotqa"]["answer_kind"] = df.kind.value_counts().to_dict()
    res["shotqa"]["gold_letter_distribution"] = df.gold_letter.value_counts().sort_index().to_dict()
    res["shotqa"]["distinct_gold_strings"] = int(df.gold.nunique())
    all_opts = collections.Counter(v for o in df.options for v in o.values())
    res["shotqa"]["distinct_option_strings"] = len(all_opts)
    atoms_all = collections.Counter(a for at in df.atoms for a in at)
    res["shotqa"]["atom_counts_in_gold"] = dict(atoms_all.most_common())
    res["shotqa"]["atom_to_class"] = {a: atom_class(a) for a in atoms_all}
    # class balance on the 16 schema classes
    single = df[df.kind == "single"]
    cls_single = collections.Counter(atom_class(at[0]) for at in single.atoms)
    cls_atoms = collections.Counter(atom_class(a) for at in df.atoms for a in at)
    res["shotqa"]["class_balance_single_label_clips"] = {c: cls_single.get(c, 0) for c in classes} | {k: v for k, v in cls_single.items() if k not in classes}
    res["shotqa"]["class_balance_all_atoms"] = {c: cls_atoms.get(c, 0) for c in classes} | {k: v for k, v in cls_atoms.items() if k not in classes}
    fully_mapped = df.atoms.map(lambda at: all(not atom_class(a).startswith("UNMAPPED") for a in at))
    res["shotqa"]["clips_all_atoms_mapped_to_schema"] = int(fully_mapped.sum())
    res["shotqa"]["clips_with_unmapped_atom"] = int((~fully_mapped).sum())
    res["shotqa"]["clips_with_trucking_alias"] = int(df.atoms.map(lambda at: any(a in ALIAS for a in at)).sum())
    res["shotqa"]["clips_with_tracking_or_rack_focus"] = int(df.atoms.map(lambda at: any(a in NOT_IN_SCHEMA for a in at)).sum())
    res["shotqa"]["clips_atoms_count"] = df.atoms.map(len).value_counts().sort_index().to_dict()
    # distribution of the 4-way MC option sets: how many options per question are compound
    comp_opts = sum(1 for o in df.options for v in o.values() if parse_answer(v)[0] != "single")
    res["shotqa"]["compound_options_share"] = round(comp_opts / (4 * len(df)), 4)
    # clip metadata
    meta = [ffprobe(QA / c) for c in df["clip"]]
    m = pd.DataFrame(meta)
    q = lambda s: {k: round(float(v), 3) for k, v in s.quantile([0, .05, .25, .5, .75, .95, 1]).items()}
    res["shotqa"]["clip_duration_s"] = q(m.dur)
    res["shotqa"]["clip_fps"] = m.fps.round(2).value_counts().head(8).to_dict()
    res["shotqa"]["clip_resolution"] = (m.w.astype(str) + "x" + m.h.astype(str)).value_counts().head(10).to_dict()
    res["shotqa"]["clip_codec"] = m.codec.value_counts().to_dict()
    res["shotqa"]["clip_frames"] = q(m.nb_frames.dropna()) if m.nb_frames.notna().any() else None
    res["shotqa"]["clips_unreadable"] = int((m.dur == 0).sum())
    res["shotqa"]["total_clip_seconds"] = round(float(m.dur.sum()), 1)
    srcs = df["clip"].map(source_id)
    res["shotqa"]["distinct_source_videos"] = int(srcs.nunique())
    res["shotqa"]["clips_per_source"] = srcs.value_counts().describe().round(2).to_dict()
    res["shotqa"]["max_clips_one_source"] = int(srcs.value_counts().max())
    # ---------------- (c) ShotBench movement questions: format only (question file), no answers
    tsv = pd.read_csv(SB / "test.tsv", sep="\t")
    vq = tsv[tsv.type.str.contains("video")]
    sbo = collections.Counter(v for o in vq.options for v in json.loads(o).values())
    sb_atoms = {a for s_ in sbo for a in parse_answer(s_)[1]}
    res["shotbench_movement"] = {"rows": len(vq), "categories": vq.category.value_counts().to_dict(), "question_texts": vq.question.value_counts().to_dict(),
                                 "options_per_question": vq.options.map(lambda o: len(json.loads(o))).value_counts().to_dict(),
                                 "distinct_option_strings": len(sbo), "option_atoms_not_mappable": sorted(a for a in sb_atoms if atom_class(a).startswith("UNMAPPED")),
                                 "compound_option_share": round(sum(c for s_, c in sbo.items() if parse_answer(s_)[0] != "single") / sum(sbo.values()), 4),
                                 "paths_unique": int(vq.path.nunique()), "path_example": vq.path.iloc[0],
                                 "note": "correct answers of these questions were NOT read into any statistic"}
    # ---------------- leakage ShotQA clips vs ShotBench videos (video id, name, sha256, three-frame dHash)
    sbv = [ast.literal_eval(p)[0].removeprefix("video/") for p in vq.path]
    res["leakage"] = {"shotqa_clips": len(df), "shotbench_videos": len(sbv), "shotbench_videos_present": sum((SB_VIDEOS / v).exists() for v in sbv)}
    qa_src, sb_src = set(df["clip"].map(source_id)), {source_id(v) for v in sbv}
    res["leakage"].update({"exact_clip_name_overlap": len(set(df["clip"]) & set(sbv)), "source_video_id_overlap": len(qa_src & sb_src),
                           "shotqa_distinct_sources": len(qa_src), "shotbench_distinct_sources": len(sb_src),
                           "shotbench_videos_whose_source_is_in_shotqa": sum(source_id(v) in qa_src for v in sbv),
                           "shotqa_clips_whose_source_is_in_shotbench": int(sum(source_id(c) in sb_src for c in df["clip"])),
                           "overlapping_source_ids_sample": sorted(qa_src & sb_src)[:5]})
    qa_sha = {c: sha256(QA / c) for c in df["clip"]}
    sb_sha = {v: sha256(SB_VIDEOS / v) for v in sbv if (SB_VIDEOS / v).exists()}
    inv = collections.defaultdict(list)
    for c, h in qa_sha.items():
        inv[h].append(c)
    sha_pairs = [(v, inv[h]) for v, h in sb_sha.items() if h in inv]
    res["leakage"]["sha256_identical_files"] = len(sha_pairs)
    qa_dups = [v for v in inv.values() if len(v) > 1]
    res["leakage"]["shotqa_internal_sha256_duplicate_groups"] = len(qa_dups)
    qa_h = {c: frame_hashes(QA / c) for c in df["clip"]}
    sb_h = {v: frame_hashes(SB_VIDEOS / v) for v in sb_sha}
    qa_ok = {c: h for c, h in qa_h.items() if h}
    sb_ok = {v: h for v, h in sb_h.items() if h}
    res["leakage"]["frame_hash_failures"] = {"shotqa": len(qa_h) - len(qa_ok), "shotbench": len(sb_h) - len(sb_ok)}
    qa_names = list(qa_ok)
    A = np.array([[h for h in qa_ok[c]] for c in qa_names], dtype=np.uint64)  # (n,3)
    pop = lambda x: np.array([int(i).bit_count() for i in x.flatten()]).reshape(x.shape)
    hist = {"le4_frames>=2of3": 0, "le8_frames>=2of3": 0, "le4_any_frame": 0, "le0_all3": 0}
    flagged = []
    for v, h in sb_ok.items():
        ham = np.stack([pop(A[:, k] ^ np.uint64(h[k])) for k in range(3)], axis=1)  # (n,3), same frame position
        n4, n8 = (ham <= 4).sum(1), (ham <= 8).sum(1)
        hist["le4_frames>=2of3"] += int((n4 >= 2).any())
        hist["le8_frames>=2of3"] += int((n8 >= 2).any())
        hist["le4_any_frame"] += int((n4 >= 1).any())
        hist["le0_all3"] += int(((ham == 0).sum(1) == 3).any())
        if (n4 >= 2).any():
            flagged.extend((v, qa_names[i]) for i in np.where(n4 >= 2)[0])
    res["leakage"]["frame_dhash_shotbench_videos_with_a_shotqa_match"] = hist
    # pixel verification of every flagged pair: mean absolute difference of 64x64 grey mid frames, and the contrast of each frame (flat frames hash alike)
    def mid_gray(p: Path) -> np.ndarray:
        cap = cv2.VideoCapture(str(p))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.5)))
        _, fr = cap.read()
        cap.release()
        return cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    detail = []
    for v, c in flagged:
        a, b = mid_gray(SB_VIDEOS / v), mid_gray(QA / c)
        detail.append({"shotbench": v, "shotqa": c, "same_source": source_id(v) == source_id(c), "mean_abs_diff_of_255": round(float(np.abs(a - b).mean()), 2),
                       "frame_std_shotbench": round(float(a.std()), 1), "frame_std_shotqa": round(float(b.std()), 1)})
    res["leakage"]["flagged_pairs"] = len(flagged)
    res["leakage"]["flagged_pairs_detail"] = detail
    res["leakage"]["flagged_pairs_pixel_verified_same_frame(mad<=6)"] = sum(d["mean_abs_diff_of_255"] <= 6 for d in detail)
    res["leakage"]["frame_hash_failed_shotbench_videos"] = sorted(set(sbv) - set(sb_ok))
    # scene-index neighbours: ShotQA clips of the same source video within +-2 shot indices of a ShotBench video (same scene, residual same-shot risk)
    def idx(name: str) -> int | None:
        m_ = re.search(r"_(\d+)\.mp4$", name)
        return int(m_.group(1)) if m_ else None
    by_src = collections.defaultdict(list)
    for c in df["clip"]:
        by_src[source_id(c)].append(idx(c))
    near = {1: 0, 2: 0, 5: 0}
    for v in sbv:
        iv = idx(v)
        ds = [abs(iv - j) for j in by_src.get(source_id(v), []) if iv is not None and j is not None]
        for k in near:
            near[k] += int(bool(ds) and min(ds) <= k)
    res["leakage"]["shotbench_videos_with_a_shotqa_clip_within_n_shot_indices_of_same_source"] = near
    res["leakage"]["method"] = ("video id = name before '.webm_'/'.mp4_'; exact name; sha256 of file bytes; dHash (9x8) of frames at 25/50/75% of each clip, a pair "
                                "matches if >=2 of 3 same-position frames are within Hamming 4. Titles are not available for video clips (names are source ids), so "
                                "film overlap is measured by shared source video id.")
    res["elapsed_s"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
