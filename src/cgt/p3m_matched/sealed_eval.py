"""Sealed, single-use matched ShotBench evaluation per model tag (docs/p3m_matched_protocol.md section 3).
`preflight` runs EVERY refusal before any ShotBench path is touched, in this order:
(1) tag registered with a pinned identity (TagNotBound, exit 18); (2) ledger tag unused in the given ledger AND ALWAYS in the real ledger
results/phase3/shotbench_evaluated.json (LedgerUsed, 14; a scratch --ledger can never bypass single use); (3) tag not CONSUMED: no
<tag>.CONSUMED.json and no <tag>.predictions.jsonl in the given raw dir or in the canonical results/p3m_matched/raw_eval_logs (14);
(4) no unacknowledged crash in the given raw dir AND in the canonical raw dir (CrashUnacknowledged, 16; every dir is checked before any ACK is
consumed); (5) the operator's GO file exists and equals `go_text(tag, identity, protocol)` (NotAuthorized, 13);
(6) code committed (DirtyTree, 10); (7) model identity recomputed and equal to the registry (AdapterHashError, 7).
`run_sealed` then loads the questions, scores them inside the existing `cgt.p3b_vlm.sealed.SealedEvaluation` (counts only until sealed; the
ledger entry is recorded FIRST, then the files, then the echo) and never prints a score before the seal."""
import hashlib
import json
import math
import time
from pathlib import Path

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.runstatus import now_iso
from cgt.p3m_matched import registry
from cgt.p3m_matched.protocol import protocol_spec

ROOT = Path(__file__).resolve().parents[3]
CODE_PATHS = ["scripts", "src", "docs/p3m_matched_protocol.md", "pyproject.toml", "uv.lock"]
GO_DIR = ROOT / "results/p3m_matched"
REAL_RAW_DIR = ROOT / "results/p3m_matched/raw_eval_logs"


def require_not_consumed_anywhere(tag: str, raw_dir: Path, real_raw_dir: Path = REAL_RAW_DIR) -> None:
    """CONSUMED markers (exit 14): the sealed evaluator's CONSUMED file or its predictions file for this tag, in the given raw dir and ALWAYS in
    the canonical raw dir (a scratch --raw-dir can never bypass single use)."""
    from cgt.p3b_vlm import sealed
    for d in dict.fromkeys([Path(raw_dir), Path(real_raw_dir)]):
        sealed.require_not_consumed(d, tag)
        if (d / f"{tag}.predictions.jsonl").exists():
            raise guards.LedgerUsed(f"tag '{tag}' is CONSUMED ({tag}.predictions.jsonl exists in {d.name}); a new evaluation is not allowed")


def require_no_unacknowledged_crash_anywhere(tag: str, raw_dir: Path, real_raw_dir: Path = REAL_RAW_DIR) -> None:
    """an unacknowledged crash, killed run or UNKNOWN commitment (exit 16) in the given raw dir OR the canonical raw dir refuses, so a
    scratch --raw-dir can never sidestep a crash recorded in the canonical one. Check-only pass over EVERY dir first; only when nothing refuses
    are the acknowledged files archived and the ACKs consumed (`sealed.require_no_unacknowledged_crash`), so a refusal never half-consumes an ACK."""
    from cgt.p3b_vlm import sealed
    dirs = list(dict.fromkeys(Path(d).resolve() for d in (raw_dir, real_raw_dir)))
    for d in dirs:
        if sealed.unknown_path(d, tag).exists() and not sealed.unknown_ack_path(d, tag).exists():
            raise guards.CrashUnacknowledged(f"a previous attempt for tag '{tag}' ended with an UNKNOWN ledger commitment ({sealed.unknown_path(d, tag).name} "
                                             f"in {d.name}); tell the operator, who must reconcile the ledger by hand and create "
                                             f"{sealed.unknown_ack_path(d, tag).name} there before any retry. Nothing was published.")
        old = [p.name for p in (sealed.crash_path(d, tag), sealed.in_progress_path(d, tag)) if p.exists()]
        if old and not sealed.ack_path(d, tag).exists():
            raise guards.CrashUnacknowledged(f"a previous attempt for tag '{tag}' crashed or was killed ({', '.join(old)} in {d.name}); tell the operator, "
                                             f"who must create {sealed.ack_path(d, tag).name} there before any retry. The ledger tag is not consumed.")
    for d in dirs:
        sealed.require_no_unacknowledged_crash(d, tag)


def adapter_keys_checked(tag: str) -> bool:
    """every tag except the consumed matched_v1 asserts the adapter tensor keys match the model before evaluating."""
    return tag != "matched_v1"


def go_path(tag: str, go_dir: Path = GO_DIR) -> Path:
    return Path(go_dir) / f"GO_{tag}"


def go_text(tag: str, identity_sha256: str, protocol_hash: str, extras: list[tuple[str, str]] | None = None) -> str:
    """The GO text. `extras` (letterlogit_c5 tags only: blend_sha256, siglip_identity_sha256) are appended in order; without extras the text
    is byte-identical to the original matched GO."""
    return (f"P3M_MATCHED_GO\ntag={tag}\nidentity_sha256={identity_sha256}\nprotocol_sha256={protocol_hash}\n"
            + "".join(f"{k}={v}\n" for k, v in (extras or [])))


def expected_go_text(tag: str) -> str:
    """The GO text for a registered, pinned tag (used by scripts/p3m_matched_write_go.py): pinned identity, the protocol hash of the current
    code (must equal a pinned protocol_sha256) and the pinned extras."""
    e = registry.entry(tag)
    return go_text(tag, e["identity_sha256"], registry.protocol_hash(e), registry.go_extras(e))


def require_go(path: Path, tag: str, identity_sha256: str, protocol_hash: str, extras: list[tuple[str, str]] | None = None) -> None:
    p = Path(path)
    if not p.is_file():
        raise guards.NotAuthorized(f"operator GO file missing: {p}")
    if p.read_text() != go_text(tag, identity_sha256, protocol_hash, extras):
        raise guards.NotAuthorized(f"GO file {p} is not bound to tag {tag!r}, its model identity, the protocol hash"
                                   + (" and the blend / SigLIP identity" if extras else ""))


def preflight(tag: str, go_file: Path, ledger: Path, raw_dir: Path, data_root: Path = ROOT, code_paths=CODE_PATHS,
              clean_tree=guards.require_clean_tree, real_ledger: Path | None = None, real_raw_dir: Path = REAL_RAW_DIR,
              provenance=None) -> dict:
    """(1) also refuses a pinned protocol_sha256 that the current code does not reproduce (TagNotBound, 18); (5) the GO of a letterlogit_c5
    tag must also carry the pinned blend and SigLIP identity hashes; (7) a letterlogit_c5 tag also re-derives the C5 weights from the committed
    fit and re-hashes the SigLIP heads (AdapterHashError, 7). Matched tags without a pinned protocol hash behave exactly as before."""
    from cgt.eval import shotbench as sbmod
    e = registry.entry(tag)  # (1)
    phash = registry.protocol_hash(e)  # (1) pinned protocol hash (if any) reproduced by the current code
    try:  # (2) the given ledger and ALWAYS the real one
        for led in dict.fromkeys([Path(ledger), Path(real_ledger or sbmod.LEDGER)]):
            sbmod.ledger_assert_unused(tag, led)
    except RuntimeError as exc:
        raise guards.LedgerUsed(str(exc)) from exc
    require_not_consumed_anywhere(tag, raw_dir, real_raw_dir)  # (3)
    require_no_unacknowledged_crash_anywhere(tag, raw_dir, real_raw_dir)  # (4)
    require_go(go_file, tag, e["identity_sha256"], phash, registry.go_extras(e))  # (5)
    git = clean_tree(code_paths)  # (6)
    resolved = registry.require_identity(tag, data_root)  # (7)
    c5 = registry.require_c5_components(e, data_root, provenance) if registry.protocol_of(e) == "letterlogit_c5" else None
    return {"entry": resolved, "git": git, "protocol_sha256": phash, "c5": c5,
            "ledger_sha256_before": hashlib.sha256(Path(ledger).read_bytes()).hexdigest() if Path(ledger).exists() else None}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def run_sealed(tag: str, go_file: Path, ledger: Path, raw_dir: Path, test_tsv: Path, image_root: Path, img_cache: Path, out_dir: Path,
               log, ev, data_root: Path = ROOT, loader=None, engine_factory=None, clean_tree=guards.require_clean_tree,
               on_preflight=None) -> None:
    """`on_preflight()` (optional) is called once, right after the preflight passed and before anything is written (the script opens
    the committed raw stdout log only then)."""
    from cgt.eval import shotbench as sbmod
    from cgt.p3m_matched.evaluate import ATTRS, run_questions, summarize
    if registry.protocol_of(registry.entry(tag)) != "matched":
        raise guards.TagNotBound(f"tag {tag!r} is not a matched-protocol tag (use cgt.p3m_matched.letterlogit_c5.run_sealed)")
    pf = preflight(tag, go_file, ledger, raw_dir, data_root, clean_tree=clean_tree)
    if on_preflight is not None:
        on_preflight()
    e = pf["entry"]
    ev.code_commit = pf["git"]["commit"]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"tag": tag, "time": now_iso(), "code_commit": pf["git"]["commit"], "protocol": protocol_spec(e["variant"]),
                "protocol_sha256": pf["protocol_sha256"], "identity_sha256_before": e["identity_sha256"], "image_spec": e["image_spec"],
                "ledger_sha256_before": pf["ledger_sha256_before"]}
    if e.get("protocol_sha256"):  # a pinned tag also records its base model, as the C5 manifest; matched_v1 unchanged
        manifest.update({"repo": e["repo"], "revision": e["revision"]})
    (raw_dir / f"{tag}.matched_manifest.json").write_text(json.dumps(manifest, indent=1))  # no scores
    log(f"preflight passed: tag {tag}, identity {e['identity_sha256'][:16]}, protocol {pf['protocol_sha256'][:16]}, code {pf['git']['commit'][:12]}")
    if loader is None:
        from cgt.p3m_matched.shotbench import load_matched_questions as loader
    qs, stats = loader(test_tsv, image_root)
    if e["image_spec"]["kind"] == "budget":
        from cgt.p3b_vlm.images import prepare_eval_cache
        guards.require_unique_stems([q["image"] for q in qs])
        try:
            cached = prepare_eval_cache([q["image"] for q in qs], img_cache, e["image_spec"]["budget_px"], e["image_spec"]["multiple"])
        except (OSError, ValueError) as exc:
            raise guards.UnreadableImage(f"{type(exc).__name__} while preparing the image cache") from exc
        for q in qs:
            q["image"] = cached[q["image"]]
    guards.require_images_readable([q["image"] for q in qs])
    ev.start(len(qs))
    if engine_factory is None:
        from cgt.p3m_matched.engine import MlxEngine, load_mlx_model

        def engine_factory(entry):
            return MlxEngine(*load_mlx_model(entry, verify_adapter_keys=adapter_keys_checked(tag)))
    engine = engine_factory(e)
    log(f"evaluating {len(qs)} questions under the matched protocol ({e['variant']}); progress counts only until sealed")
    t0 = time.perf_counter()
    res = ev.run(lambda cb: run_questions(engine, qs, e["variant"], progress=cb))
    secs = time.perf_counter() - t0
    after = registry.require_identity(tag, data_root)
    if after["identity_sha256"] != e["identity_sha256"]:
        raise guards.AdapterHashError("the model identity changed during the evaluation")
    summ = summarize(res)
    rows = []
    for at in ATTRS:
        v = summ["per_attribute"].get(at)
        if v is None:
            continue
        lo, hi = wilson(v["correct"], v["n"])
        rows.append({"attribute": at, "n": v["n"], "correct": v["correct"], "acc_pct": round(100 * v["acc"], 2), "ci95": [round(lo, 1), round(hi, 1)],
                     "invalid_rate_pct": round(100 * v["invalid_rate"], 2), "expected_fallback_acc_pct": round(100 * v["expected_fallback_acc"], 2)})
    summary = {"tag": tag, "protocol_sha256": pf["protocol_sha256"], "variant": e["variant"], "identity_sha256": e["identity_sha256"], "stats": stats,
               "rows": rows, "mean_acc_pct": round(100 * summ["mean_acc"], 2), "micro_acc_pct": round(100 * summ["micro_acc"], 2),
               "invalid_rate_pct": round(100 * summ["invalid_rate"], 2), "mean_expected_fallback_acc_pct": round(100 * summ["mean_expected_fallback_acc"], 2),
               "seconds": secs, "n": summ["n"]}
    if e.get("protocol_sha256"):  # tags pinned by an earlier change (matched_4b_v1): an earlier review count, computed at run time; matched_v1's summary is unchanged
        summary["n_fewer_than_4_options"] = sum(r["n_choices"] < 4 for r in res)
    pred_text = "\n".join(json.dumps({k: r[k] for k in ("id", "attribute", "text", "letter", "method", "correct")}) for r in res) + "\n"
    manifest.update({"identity_sha256_after": after["identity_sha256"], "predictions_sha256": hashlib.sha256(pred_text.encode()).hexdigest()})
    echo = f"DONE: matched mean accuracy {summary['mean_acc_pct']}% over {len(rows)} attributes, invalid {summary['invalid_rate_pct']}%; ledger recorded"
    note = f"matched protocol {pf['protocol_sha256'][:12]}; n={summ['n']}; identity {e['identity_sha256'][:12]}"
    ev.seal(lambda: sbmod.ledger_record(tag, note, ledger), lambda: tag in sbmod.ledger_read(ledger),
            {Path(out_dir) / f"shotbench_{tag}.json": json.dumps(summary, indent=1), raw_dir / f"{tag}.predictions.jsonl": pred_text,
             raw_dir / f"{tag}.matched_manifest.json": json.dumps(manifest, indent=1)}, echo)
