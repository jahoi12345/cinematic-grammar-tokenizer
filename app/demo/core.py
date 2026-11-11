"""Pure logic of the Phase 6 demo (no streamlit): data root, public-mode film guard, table loading, captions, arc series. Spec: docs/demo_spec.md.
Everything the demo shows about shots is a MODEL PREDICTION or an unverified heuristic; nothing here is accuracy. Nothing in this module writes to disk."""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "src"))

from cgt.pipeline import registry
from cgt.tokens import arc, similar
from cgt.tokens import palette_global as pg

N_WIN, MIN_SHOTS = 20, 15  # plan section 1.2: 20 equal windows of running time, windows below 15 shots flagged (not merged)

BANNER_LOCAL_ONLY = "LOCAL ONLY - owned film, never publish"
CAPTION_PREDICTIONS = ("Every series and token shown is a MODEL PREDICTION (size, lens, angle, framing, light, composition, movement) or an unverified Phase 1/2 heuristic "
                       "(shot starts, length, key, DoF, people, sound, palette) whose accuracy on this film is UNMEASURED: no labels exist. "
                       "Read the curves as what the model or heuristic tends to say, not as measurements of the film. Consistency is not correctness.")
# The static model of the film pipeline. Numbers: results/stepc/selection.md (ShotQA film-held-out
# validation, dedup_v2). ShotBench results of the 4B are not shown here.
STATIC_MODEL = "Qwen3-VL-4B with a LoRA adapter fine-tuned on ShotQA (Step C)"
CAPTION_MODEL = ("Static slots (size, lens, angle, framing, light, comp) are predicted by " + STATIC_MODEL + ", the film pipeline's default since 2026-10-22 "
                 "(pipeline version p3c-v2-4b); it replaced an earlier Qwen3-VL-2B adapter. On film-held-out ShotQA validation it scores 82.49% against "
                 "81.16% for the 2B (results/stepc/selection.md), but it was selected on that same set, and benchmark accuracy does not transfer to these films. "
                 "Switching models changed many tokens (for example the lighting-condition slot of 574 of the 1,077 Night of the Living Dead shots): a change, not a "
                 "proven improvement. Movement comes from the Phase 4 geometry model, which the switch did not change.")
CAPTION_INTERVALS = ("Mean-based series show 95% bootstrap intervals from resampling each window's shots (seed 0); median shot length shows its within-window interquartile range; "
                     "the shot-start rate has no interval. Shots are not independent, so the intervals are likely too narrow, and none of them measures label accuracy. "
                     "Shot starts per minute counts shot STARTS (including the first shot). Windows with fewer than 15 shots are flagged.")
CAPTION_SHAPES = ("Within-film shapes only: each series is z-scored WITHIN its own film, so only the shape over each film's running time can be read. "
                  "Levels are never compared across films (plan section 1.4); no similarity statistic, no pooled curve, no ranking of films.")
# committed contradictory-set shares of the predicted movement sets.
# Public-domain films only (shown only when the registry says public_domain); owned films get no committed number in code, only the live share of the loaded table
#.
COMMITTED_CONTRADICTORY = {"night_of_the_living_dead": (462, 1077)}
COMMITTED_SOURCE = "experiment log, 2026-10-13"
CAPTION_SIMILAR = ("Shots like this: the other shots of THIS film whose token rows are most similar (docs/similar_shots_spec.md: per-slot matches, equal weights, `*` values count half). "
                   "This is similarity of PREDICTED tokens whose accuracy is UNMEASURED, not ground truth: it does not mean the shots look alike or that the film-makers intended a callback. "
                   "The shot itself and its two neighbours on each side are never listed.")
CENTROIDS_REL = "data/cache/tokens/palette_global_k32.npz"


def data_root() -> Path:
    return Path(os.environ.get("CGT_DEMO_ROOT") or CODE_ROOT).resolve()


def public_mode() -> bool:
    """Fail closed: any value other than exactly empty or exactly '0' turns public mode on (documented value: CGT_DEMO_PUBLIC=1).
    No whitespace stripping: ' 0', '0 ' and whitespace-only values are public."""
    return os.environ.get("CGT_DEMO_PUBLIC", "") not in ("", "0")


def films_with_tables(root: Path) -> list[str]:
    d = Path(root) / "data/processed"
    names = {p.name.removesuffix("_tokens_v1.parquet") for p in d.glob("*_tokens_v1.parquet")}
    names |= {p.name.removesuffix("_tokens.parquet") for p in d.glob("*_tokens.parquet")}
    return sorted(names)


def public_ids(root: Path) -> set[str]:
    """Films allowed in public mode: public_domain is the boolean True in the committed configs/films.yaml and the id is not in the overlay.
    Any error reading or merging the registry -> empty set (fail closed)."""
    try:
        main = yaml.safe_load((Path(root) / registry.MAIN).read_text()) or {}
        overlay = {e.get("id") for e in registry.read_overlay(root)}
        registry.load_films(root)  # raises on a clash
    except Exception:  # noqa: BLE001 - any registry problem hides every film in public mode
        return set()
    return {f["id"] for f in (main.get("films") or []) if f.get("public_domain") is True and f.get("id") not in overlay}


def listed_films(root: Path, public: bool) -> list[str]:
    films = films_with_tables(root)
    if public:
        allowed = public_ids(root)
        return [f for f in films if f in allowed]
    return films


def is_local_only(root: Path, film: str) -> bool:
    """Owned, overlay or unregistered films are local only (fail closed if the registry is unreadable)."""
    try:
        reg = registry.load_films(root)
    except Exception:  # noqa: BLE001
        return True
    return film not in reg or film in registry.local_only_ids(root)


def _check(root: Path, film: str, public: bool) -> None:
    if public and film not in public_ids(root):
        raise PermissionError(f"public mode: '{film}' is not a public-domain film of configs/films.yaml")


def table_path(root: Path, film: str) -> tuple[Path, str]:
    d = Path(root) / "data/processed"
    for v, suffix in (("v1", "_tokens_v1.parquet"), ("v0", "_tokens.parquet")):
        if (d / f"{film}{suffix}").exists():
            return d / f"{film}{suffix}", v
    raise FileNotFoundError(f"no token table for {film}")


def load_table(root: Path, film: str, public: bool) -> tuple[pd.DataFrame, str]:
    _check(root, film, public)
    p, v = table_path(root, film)
    return pd.read_parquet(p), v


def frame_path(root: Path, film: str, rel: str, public: bool) -> Path | None:
    """Keyframes only from ROOT/data/interim/<film>/keyframes (resolved); anything else is not displayed."""
    _check(root, film, public)
    if not isinstance(rel, str) or not rel:
        return None
    kroot = (Path(root) / "data/interim" / film / "keyframes").resolve()
    p = (Path(root) / rel).resolve()
    return p if kroot in p.parents and p.is_file() else None


def length_stats(df: pd.DataFrame) -> dict[str, float]:
    s = df["len_s"]
    return {"shots": len(df), "running_min": float(df["end_s"].max() / 60.0), "mean_s": float(s.mean()), "median_s": float(s.median()),
            "q25_s": float(s.quantile(0.25)), "q75_s": float(s.quantile(0.75)), "min_s": float(s.min()), "max_s": float(s.max())}


def has_arc_columns(df: pd.DataFrame) -> bool:
    need = {"start_min", "end_s", "len_s", "size", "key", "people", "sound", "light_cond", "light_type", "angle", "kf_all_agree"}
    return need <= set(df.columns)


def has_movement(df: pd.DataFrame) -> bool:
    return "move_atoms" in df.columns and "move_present" in df.columns and bool(df["move_present"].iloc[0]) and not df["move_atoms"].isna().all()


def window_minutes(df: pd.DataFrame, mode: str) -> float:
    """'equal20': 20 equal windows of the running time (primary); 'min5': 5-minute windows (secondary)."""
    return float(df["end_s"].max() / 60.0) / N_WIN if mode == "equal20" else 5.0


def arc_series(df: pd.DataFrame, mode: str = "equal20", n_boot: int = 1000) -> pd.DataFrame:
    m = window_minutes(df, mode)
    t = arc.series_table(df, m, n_boot=n_boot)
    t["window_mid_min"] = t["window_start_min"] + m / 2
    t["flag_sparse"] = t["n_shots"] < MIN_SHOTS
    return t


def movement_series(df: pd.DataFrame, mode: str = "equal20", n_boot: int = 500) -> pd.DataFrame:
    m = window_minutes(df, mode)
    t = pd.concat([arc.movement_series_table(df, m, n_boot=n_boot), arc.contradictory_series(df, m, n_boot=n_boot)], ignore_index=True)
    t["window_mid_min"] = t["window_start_min"] + m / 2
    t["flag_sparse"] = t["n_shots"] < MIN_SHOTS
    return t


def shapes(df: pd.DataFrame, n_boot: int = 50) -> pd.DataFrame:
    """Within-film z-scores of every arc series on 20 equal windows (x = fraction of running time). Returns NO level column: shape only."""
    t = arc_series(df, "equal20", n_boot=n_boot)
    out = []
    for name, d in t.groupby("series"):
        v = d["value"].to_numpy(dtype=float)
        sd = v.std()
        z = (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)
        out.append(pd.DataFrame({"series": name, "kind": d["kind"].to_numpy(), "frac": (d["window"].to_numpy() + 0.5) / N_WIN, "z_within_film": z,
                                 "n_shots": d["n_shots"].to_numpy(), "flag_sparse": d["flag_sparse"].to_numpy()}))
    return pd.concat(out, ignore_index=True)


def movement_caption(film: str, df: pd.DataFrame, root: Path | None = None) -> str:
    """The committed share is cited only for a film that is public_domain in the registry under ROOT (default: data_root()); every other film shows the live share only."""
    live = arc.contradictory_share(df["move_atoms"])
    c = COMMITTED_CONTRADICTORY.get(film) if film in public_ids(data_root() if root is None else root) else None
    committed = (f"committed share for this film: {c[0]}/{c[1]:,} = {100 * c[0] / c[1]:.1f}% ({COMMITTED_SOURCE}); " if c else "no committed share for this film; ")
    return ("Movement sets are INFLATED: a contradictory set (an opposite pair such as pan-left with pan-right, or `static` together with another atom) is something no real camera "
            f"move produces; {committed}share in the loaded table: {100 * live:.1f}%. These are predictions of the Phase 4 geometry model whose accuracy on this film is UNMEASURED; "
            "read them as the model's tendencies, never as camera work, and never compare movement levels across films.")


def slot_options(df: pd.DataFrame) -> dict[str, str]:
    """Token slot -> table column used for filtering (every slot of the row that has a column)."""
    cols = {"size": "size", "lens": "lens", "angle": "angle", "framing": "framing", "light type": "light_type", "light condition": "light_cond", "comp": "composition",
            "key": "key", "DoF": "dof", "sound": "sound", "len": "len_tok", "palette": "palette_tok"}
    return {k: v for k, v in cols.items() if v in df.columns}


def palette_centroids(root: Path):
    """The frozen global palette centroids under ROOT (digest-checked), or None (the similarity palette slot then falls back to exact id match)."""
    p = Path(root) / CENTROIDS_REL
    try:
        return pg.load(p) if p.is_file() else None
    except Exception:  # noqa: BLE001 - a replaced or unreadable artifact is not used
        return None


def similar_table(root: Path, film: str, public: bool, df: pd.DataFrame, k: int = similar.K_DEFAULT) -> pd.DataFrame:
    """Top-k similar shots of every shot of ONE film's already loaded table (same public-mode check as load_table). Writes nothing."""
    _check(root, film, public)
    return similar.top_k_table(df, k=k, centroids=palette_centroids(root))


# ---- v1.0 readability ---------------------------------------------------------------------------

SERIES_LABELS = {
    "cuts_per_min": "shot starts per minute", "median_len_s": "median shot length (s)",
    "close_share": "share of close shots (ECU/CU/MCU)", "mean_size_ordinal": "mean shot size (0 = ECU ... 6 = EWS)",
    "low_key_share": "share of low-key shots", "high_contrast_share": "share of high-contrast light", "daylight_share": "share of daylight shots",
    "people_mean": "mean people per shot (4+ counted as 4)", "multi_person_share": "share of shots with 2+ people", "dialogue_share": "share of dialogue shots",
    "low_angle_share": "share of low-angle shots", "keyframes_all_agree_share": "share of shots whose 3 keyframes agree",
    "move_static_share": "share of sets containing static", "move_static_alone_share": "share of sets that are static alone",
    "move_mean_set_size": "mean number of atoms per set", "move_contradictory_share": "share of contradictory sets",
    **{f"move_{g}_share": f"share of sets with any {g} atom" for g in arc.MOVE_GROUPS},
}


def series_label(name: str) -> str:
    return SERIES_LABELS.get(name, name.replace("_", " "))


def film_label(root: Path, film: str) -> str:
    """'Title (year)' from the registry under ROOT; the id itself when the registry has no title for it or cannot be read. Only called for films already listed."""
    try:
        e = registry.load_films(root).get(film) or {}
    except Exception:  # noqa: BLE001 - an unreadable registry shows the id
        e = {}
    title, year = e.get("title"), e.get("year")
    if not title:
        return film
    return f"{title} ({year})" if year else str(title)


# slot -> one-line meaning (v1 row of configs/tokens.yaml, vocabularies of configs/schema.yaml)
SLOT_HELP = [
    ("size", "model prediction (4B)", "shot size, how big the subject is in the frame: ECU, CU, MCU, MS, MWS, WS, EWS (extreme close-up to extreme wide)"),
    ("lens", "model prediction (4B)", "lens class: long, medium, wide, ultra-wide"),
    ("angle", "model prediction (4B)", "camera angle: low, high, overhead, aerial, dutch (there is no eye-level class)"),
    ("framing", "model prediction (4B)", "what is framed: single, insert, 2-shot, 3-shot, group, establishing, ots (over the shoulder)"),
    ("light", "model prediction (4B)", "light type / light condition, e.g. daylight/soft or artificial/high-contrast"),
    ("comp", "model prediction (4B)", "composition: center, balanced, symmetrical, right-heavy, left-heavy, short-side"),
    ("key", "heuristic", "brightness key from the luminance histogram: low-key, mid-key, high-key"),
    ("DoF", "heuristic", "depth of field: shallow, medium, deep"),
    ("people", "detector + rule", "number of people detected (4+ for four or more)"),
    ("sound", "heuristic (provisional)", "dominant sound: score, dialogue, ambient, silence"),
    ("len", "heuristic cut table", "shot length bin in seconds: <1, 1-2, 2-4, 4-8, 8-16, 16-32, 32+"),
    ("palette", "heuristic", "colour palette cluster id g00-g31 (32 global clusters)"),
    ("move", "model prediction (geometry)", "set of predicted camera-movement atoms joined with + (e.g. pan-left+push-in); static alone = no movement"),
]
MARKER_HELP = [("*", "low confidence: the start, mid and end keyframes of the shot give different predictions for this slot"),
               ("?", "missing: no value for this shot (e.g. a shot absent from the movement table)"),
               ("n/a", "not available: the film has no movement table")]
CAPTION_MOVEMENT_GENERIC = ("Movement sets are INFLATED: many predicted sets are contradictory (an opposite pair such as pan-left with pan-right, or `static` together with another atom), "
                            "which no real camera move produces. Read them as the model's tendencies, never as camera work, and never compare movement levels across films.")

CSV_DROP = ("kf_start", "kf_mid", "kf_end")  # keyframe file paths are never exported


def csv_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in CSV_DROP]


def _cell(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return ";".join(str(x) for x in v)
    return v


def csv_bytes(root: Path, film: str, public: bool, df: pd.DataFrame) -> bytes:
    """CSV of the given (filtered) rows of ONE film's loaded table: every column except the keyframe paths. Same public-mode check as load_table;
    refuses rows that belong to another film. Writes nothing."""
    _check(root, film, public)
    if "film_id" in df.columns and not (df["film_id"] == film).all():
        raise ValueError("rows of another film in the export")
    out = df[csv_columns(df)].copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(_cell)
    return out.to_csv(index=False).encode("utf-8")


ADD_CLIP_COMMAND = "uv run python scripts/tokenize_video.py <file> [--cut-threshold N]"


# ---- v2.0 unified front end ----------------------------------------------------------------------------

BANNER_LOCAL_REPORT = "LOCAL ONLY - internal results or owned/user-clip content, never publish"
CAPTION_ESSAY = "Descriptive comparison, not accuracy; the essay is reference, not ground truth."
PAPER_PUBLIC = ("docs/writeup/paper_public.html",)  # only the HTML paper, if it passes the checker; no draft fallback
PAPER_PUBLIC_PDF = "docs/writeup/paper_public.pdf"
PAPER_LOCAL = ("data/reports/paper_full.html", "docs/writeup/draft.md")  # gitignored local paper, else the full draft
PAPER_LOCAL_PDF = "data/reports/paper_full.pdf"
PHASE_REPORTS = ("docs/phase6_report.md", "results/phase4/PHASE4_FINAL_REPORT.md", "results/phase3b/shotbench_summary.md",
                 "results/phase6/visual_arc_night_of_the_living_dead.md", "results/phase6/similar_shots_night_of_the_living_dead.md", "docs/project_log.md")
MATCHED_REPORTS = ("results/p3m_matched/shotbench_matched_v1.md",)  # internal result: local mode only
PUBLIC_DENY = ("data/", "models/", "results/p3m_matched/", "configs/films.local")  # never readable in public mode, whatever the checker says
# A public candidate whose text carries any of these is excluded whatever the checker says. a recorded decision (a decision record)
# lifted a recorded decision for publication: the matched (official-protocol) ShotBench results may appear in public files, so their source names
# (p3m_matched, matched_v1, matched_4b_v1) are no longer markers; the explicit "internal" banner still is.
INTERNAL_MARKERS = ("Internal until the user releases it",)
ESSAY_SUFFIX, REPORT_SUFFIX = "_essay_claims.json", "_report.md"
ESSAY_HEADING = "## Descriptive comparison"
ESSAY_MAX_SHOTS = 12


def overlay_ids(root: Path) -> list[str]:
    """Ids of the gitignored overlay registry (user clips); empty on any error."""
    try:
        return sorted(str(e["id"]) for e in registry.read_overlay(root) if e.get("id"))
    except Exception:  # noqa: BLE001
        return []


def release_clean(root: Path, rels) -> set[str]:
    """The subset of `rels` (paths relative to ROOT) on which the release checker (cgt.release.check) reports no hit. Owned-film terms come from the registry
    of the code checkout and of ROOT; the code checkout's docs/release_reviewed_files.yaml clears REVIEW by exact sha256. Any checker, registry or
    allow-list error, or no registry at all, returns the empty set (fail closed). Reads, never writes."""
    rels = sorted(set(rels))
    if not rels:
        return set()
    try:
        from cgt.release import check
        reg = check.load_registry([check.CODE_ROOT, Path(root)])
        if not reg.owned and not reg.public:
            return set()
        reviewed = check.load_reviewed(check.CODE_ROOT)  # the code checkout's reviewed-files allow-list (clears REVIEW by exact sha256 only)
        bad = {p for _, p in check.check_tree(Path(root), reg, files=rels, reviewed=reviewed)}
    except Exception:  # noqa: BLE001 - the checker could not decide: publish nothing
        return set()
    return {r for r in rels if r not in bad and not _carries_internal(root, r)}


def _carries_internal(root: Path, rel: str) -> bool:
    """True if the file's text carries a decision-0018 internal marker, or cannot be read (unreadable = excluded)."""
    p = safe_path(root, rel)
    if p is None:
        return True
    try:
        text = p.read_bytes().decode("utf-8", "replace")
    except OSError:
        return True
    return any(m in text for m in INTERNAL_MARKERS)


def safe_path(root: Path, rel: str) -> Path | None:
    """ROOT/rel as an existing regular file reached without any symlink on the way (the resolved path must equal the lexical one), else None."""
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
        return None
    rootr = Path(root).resolve()
    lex = rootr / rel
    try:
        p = lex.resolve()
    except (OSError, RuntimeError):
        return None
    return p if p == lex and rootr in p.parents and p.is_file() else None


def _exists(root: Path, rel: str) -> bool:
    return safe_path(root, rel) is not None


def public_candidates(root: Path) -> list[str]:
    """Every file the public build could show (before the checker): the public paper files and the phase reports, minus the hard deny list."""
    rels = [*PAPER_PUBLIC, PAPER_PUBLIC_PDF, *PHASE_REPORTS]
    return [r for r in rels if _exists(root, r) and not r.startswith(PUBLIC_DENY)]


def report_list(root: Path, public: bool, clean: set[str] | None = None) -> list[str]:
    """Phase reports selectable in this mode. Public: PHASE_REPORTS that exist and pass the checker (`clean`, else computed now), never the deny list.
    Local: also the matched ShotBench result and every overlay film's local report."""
    if public:
        ok = release_clean(root, public_candidates(root)) if clean is None else clean
        return [r for r in PHASE_REPORTS if _exists(root, r) and not r.startswith(PUBLIC_DENY) and r in ok]
    rels = [*PHASE_REPORTS, *MATCHED_REPORTS, *(f"data/processed/{f}{REPORT_SUFFIX}" for f in overlay_ids(root))]
    return [r for r in rels if _exists(root, r)]


def paper_files(root: Path, public: bool, clean: set[str] | None = None) -> dict:
    """{'main': rel or None, 'pdf': rel or None} of the paper of this mode. Public: the first of PAPER_PUBLIC that exists and passes the checker; the PDF
    only if it passes (the checker flags every PDF for review, so in practice none). No draft fallback in public mode. Local: the local paper, else the full draft."""
    if public:
        ok = release_clean(root, public_candidates(root)) if clean is None else clean
        main = next((r for r in PAPER_PUBLIC if _exists(root, r) and r in ok), None)
        return {"main": main, "pdf": PAPER_PUBLIC_PDF if _exists(root, PAPER_PUBLIC_PDF) and PAPER_PUBLIC_PDF in ok else None}
    return {"main": next((r for r in PAPER_LOCAL if _exists(root, r)), None), "pdf": PAPER_LOCAL_PDF if _exists(root, PAPER_LOCAL_PDF) else None}


def is_local_report(rel: str) -> bool:
    return rel.startswith(("data/", "results/p3m_matched/")) or rel in (PAPER_LOCAL[1],)


def read_file(root: Path, rel: str, public: bool, allowed) -> bytes:
    """Bytes of ROOT/rel only if rel is in `allowed` (the lists above for this mode) and, in public mode, not on the deny list or the local paper; resolved inside ROOT.
    Refuses (PermissionError) before reading."""
    def denied(r: str) -> bool:  # case-insensitive: on a case-insensitive file system DATA/... names the same file as data/...
        r = r.lower()
        return public and (r.startswith(PUBLIC_DENY) or r in PAPER_LOCAL or r == PAPER_LOCAL_PDF)
    if rel not in set(allowed) or denied(rel):
        raise PermissionError(f"'{rel}' is not readable in {'public' if public else 'local'} mode")
    p = safe_path(root, rel)
    if p is None or denied(p.relative_to(Path(root).resolve()).as_posix()):  # the deny list also applies to the resolved path
        raise PermissionError(f"'{rel}' is not a plain file inside the data root")
    return p.read_bytes()


# essay comparison (local only) ----------------------------------------------------------------------------------------------------------------------------

def essay_films(root: Path, public: bool) -> list[str]:
    """Overlay films with a token table, a local report and an essay-claims file. Always empty in public mode."""
    if public:
        return []
    d = Path(root) / "data/processed"
    have = set(films_with_tables(root))
    return [f for f in overlay_ids(root) if f in have and (d / f"{f}{ESSAY_SUFFIX}").is_file() and (d / f"{f}{REPORT_SUFFIX}").is_file()]


_PAREN = re.compile(r"\([^)]*\)")
_TIME = re.compile(r"\((\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*s\)")
_IDS = re.compile(r"(?<![\w.-])(\d+)(?:\s*-\s*(\d+))?(?![\w.])")


def parse_shot_cell(cell: str, prev: list[int] | None, n_shots: int) -> tuple[list[int], list[str]]:
    """Candidate shot ids of a report cell: 'same' repeats the previous row, 'whole clip' is every shot, otherwise every N or N-M outside parentheses.
    Also returns the time ranges written as '(A-B s)'. Nothing is invented: an unparsable cell gives no shot."""
    low = cell.strip().lower()
    times = [f"{a}-{b} s" for a, b in _TIME.findall(cell)]
    if low == "same":
        return list(prev or []), times
    if "whole clip" in low:
        return list(range(n_shots)), times
    ids: list[int] = []
    for a, b in _IDS.findall(_PAREN.sub(" ", cell)):
        lo, hi = int(a), int(b) if b else int(a)
        if hi >= lo and hi - lo <= 500:
            ids += [i for i in range(lo, hi + 1) if i not in ids]
    return ids, times


def verdict_class(v: str) -> str:
    """Class of the verdict's first clause (before ';'): the full verdict is always shown verbatim next to it."""
    low = v.lower().split(";")[0].strip()
    if "disagree" in low:
        return "disagree"
    for key, cls in (("not representable", "not representable"), ("partly", "partly"), ("agree", "agree"), ("consistent", "agree")):
        if low.startswith(key):
            return cls
    return "see text"


def parse_essay_table(md: str, n_shots: int) -> list[dict]:
    """Rows of the report table under '## Descriptive comparison' (essay claim | candidate shots | what the tokens say | verdict)."""
    rows: list[dict] = []
    on, prev = False, None
    for line in md.splitlines():
        if line.startswith("## "):
            on = line.startswith(ESSAY_HEADING)
            continue
        if not on or not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or cells[0].lower() == "essay claim" or set(cells[0]) <= set("-: "):
            continue
        ids, times = parse_shot_cell(cells[1], prev, n_shots)
        prev = ids
        rows.append({"claim": cells[0], "candidates": cells[1], "shot_ids": ids, "times": times, "tokens": cells[2], "verdict": cells[3],
                     "verdict_class": verdict_class(cells[3])})
    return rows


def load_essay(root: Path, film: str, public: bool) -> dict:
    """The claims file and the parsed report table of ONE overlay film (local mode only; PermissionError otherwise). Reads, never writes."""
    if film not in essay_films(root, public):
        raise PermissionError("the essay comparison is local only and needs an overlay film with a claims file and a report")
    d = Path(root) / "data/processed"
    claims = json.loads((d / f"{film}{ESSAY_SUFFIX}").read_text())
    df, _ = load_table(root, film, public)
    rows = parse_essay_table((d / f"{film}{REPORT_SUFFIX}").read_text(), len(df))
    return {"claims": [str(c) for c in claims.get("claims", [])], "rows": rows}
