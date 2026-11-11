"""Sealed, single-use LETTER-LOGIT + C5 ShotBench evaluation (protocol cgt.p3m_matched.letterlogit_protocol; docs/plans/shotbench_4b_request.md).
Same sealing as the matched evaluator (`cgt.p3m_matched.sealed_eval.preflight`: tag pinned, real ledger + canonical raw dir unused, no
unacknowledged crash, GO bound to tag + adapter identity + protocol hash + blend hash + SigLIP identity, clean code, adapter / blend / SigLIP
heads re-verified) and the same `cgt.p3b_vlm.sealed.SealedEvaluation` (counts only until sealed; ledger entry recorded FIRST, then the files,
then the echo). Nothing is selected on ShotBench: the blend weights are the committed Step A-4B fit."""
import hashlib
import json
import time
from pathlib import Path

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.runstatus import now_iso
from cgt.p3m_matched import letterlogit_protocol as LP
from cgt.p3m_matched import registry, sealed_eval

ATTRS = ["size", "framing", "angle", "lens", "light_type", "light_cond", "composition"]  # = cgt.p3m_matched.evaluate.ATTRS (report order)


def vlm_sample(s: dict) -> dict:
    """The chat sample fed to the letter-logit scorer: the ShotQA-template user turn and a FIXED assistant 'A' (the gold letter is never an input;
    only positions before the answer token are run)."""
    return {"images": s["images"], "messages": [s["messages"][0], {"role": "assistant", "content": "A"}]}


def score_questions(samples: list[dict], vlm, siglip: dict[str, list[float]], w: dict[str, float], progress=None) -> list[dict]:
    """`vlm(sample) -> 4 letter logits (A-D)`; `siglip[id]` = SigLIP option scores in the question's sorted letter order."""
    out, total = [], len(samples)
    for s in samples:
        letters = LP.option_letters(s["options"])
        logits = [float(x) for x in vlm(vlm_sample(s))]
        sig = siglip[s["id"]]
        c5, c0 = LP.predict(logits, sig, letters, w[s["attribute"]])
        out.append({"id": s["id"], "attribute": s["attribute"], "letters": letters, "vlm_logits": [round(x, 4) for x in logits],
                    "siglip_scores": [round(float(x), 5) for x in sig], "w": w[s["attribute"]], "letter": c5, "correct": c5 == s["answer"],
                    "c0_letter": c0, "c0_correct": c0 == s["answer"]})
        if progress is not None:
            progress(len(out), total)
    return out


def summarize(res: list[dict]) -> dict:
    per, per0 = {}, {}
    for a in ATTRS:
        r = [x for x in res if x["attribute"] == a]
        if r:
            per[a] = {"n": len(r), "correct": sum(bool(x["correct"]) for x in r)}
            per0[a] = {"n": len(r), "correct": sum(bool(x["c0_correct"]) for x in r)}
    mean = sum(v["correct"] / v["n"] for v in per.values()) / len(per)
    mean0 = sum(v["correct"] / v["n"] for v in per0.values()) / len(per0)
    return {"per_attribute": per, "c0_per_attribute": per0, "mean_acc": mean, "c0_mean_acc": mean0, "n": len(res)}


def siglip_option_scores(samples: list[dict], feats: dict[str, "object"], spec: dict, heads_dir: Path) -> dict[str, list[float]]:
    """Phase 3A option scores exactly as scripts/p3s_stepa_siglip.py (chosen head kind per attribute, mean head output over the seeds,
    multihot set log-likelihood or log mean class probability), for the question's sorted option letters. `feats[id]` = raw SigLIP feature."""
    import numpy as np
    import torch

    from cgt.eval import metrics as M
    from cgt.p3_static_attrs import heads
    from cgt.p3_static_attrs.labels import parse_label_set, vocab
    out = {}
    for attr in LP.ATTRS:
        aq = [s for s in samples if s["attribute"] == attr]
        if not aq:
            continue
        v = vocab(attr)
        vi = {c: i for i, c in enumerate(v)}
        x = torch.from_numpy(heads.l2norm(np.stack([np.asarray(feats[s["id"]]) for s in aq])))
        kind_name = spec["chosen"][attr]
        preds = []
        for seed in spec["seeds"]:
            if attr == "size" and kind_name in {"softmax", "coral", "softmax_mlp", "coral_mlp"}:
                kind, hidden = kind_name.split("_")[0], 512 if kind_name.endswith("_mlp") else 0
            else:
                kind, hidden = "multihot", (512 if kind_name == "mlp" else 0)
            h = heads.Head(x.shape[1], len(v), kind, hidden)
            h.load_state_dict(torch.load(Path(heads_dir) / f"{attr}_{kind_name}_s{seed}.pt"))
            h.eval()
            preds.append(h.predict(x))
        p = np.mean(preds, axis=0)
        cat = kind != "multihot"
        for s, row in zip(aq, p, strict=True):
            opt = [[vi[c] for c in parse_label_set(attr, s["options"][x])] for x in LP.option_letters(s["options"])]
            sc = M.categorical_option_scores(row, opt) if cat else M.multihot_option_scores(row, opt)
            out[s["id"]] = [float(z) for z in sc]
    return out


def _default_vlm_factory(entry: dict):
    """(callable sample -> 4 letter logits) for the pinned base + LoRA (as scripts/stepa_4b_views.py load_model)."""
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
    from mlx_vlm.utils import load

    from cgt.p3b_vlm.adapter_load import load_adapter_checked
    from cgt.p3b_vlm.model_path import pinned_model_path
    from cgt.p3b_vlm.score import answer_token_ids, letter_logits
    model, processor = load(pinned_model_path(entry["repo"], entry["revision"]))
    cfg = model.config.__dict__
    ids4 = answer_token_ids(processor)
    lora = entry["lora"]
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=lora["rank"], alpha=lora["alpha"], dropout=lora["dropout"],
                           verbose=False)
    load_adapter_checked(model, entry["adapter_file"])  # fails closed on any adapter/model key mismatch
    model.eval()

    class _Empty:
        def __len__(self):
            return 0
    ds = VisionDataset(_Empty(), cfg, processor, None, True)

    def vlm(sample: dict):
        four, _ = letter_logits(model, ds.process(sample), ids4)
        return four
    vlm.processor = processor
    return vlm


def _default_siglip_factory(samples: list[dict], originals: dict[str, str], image_root: Path, feat_cache: Path, entry: dict, c5: dict):
    """SigLIP features of the ORIGINAL ShotBench images (fresh tag-specific cache, provenance-checked) -> option scores."""
    from cgt.p3_static_attrs import features
    paths = sorted(set(originals.values()))
    features.embed_images([Path(p) for p in paths], feat_cache, allowed_root=image_root)
    ids, feats = features.load_features(feat_cache)
    fidx = {i: n for n, i in enumerate(ids.tolist())}
    missing = [s["id"] for s in samples if Path(originals[s["id"]]).name not in fidx]
    if missing:
        raise guards.UnreadableImage(f"{len(missing)} ShotBench images have no SigLIP feature")
    by_id = {s["id"]: feats[fidx[Path(originals[s["id"]]).name]] for s in samples}
    return siglip_option_scores(samples, by_id, entry["siglip"], Path(c5["heads_dir"]))


def run_sealed(tag: str, go_file: Path, ledger: Path, raw_dir: Path, test_tsv: Path, image_root: Path, img_cache: Path, feat_cache: Path,
               out_dir: Path, log, ev, data_root: Path = registry.ROOT, loader=None, vlm_factory=None, siglip_factory=None,
               clean_tree=guards.require_clean_tree, on_preflight=None, provenance=None) -> None:
    from cgt.eval import shotbench as sbmod
    if registry.protocol_of(registry.entry(tag)) != "letterlogit_c5":
        raise guards.TagNotBound(f"tag {tag!r} is not a letterlogit_c5 tag")
    pf = sealed_eval.preflight(tag, go_file, ledger, raw_dir, data_root, clean_tree=clean_tree, provenance=provenance)
    if on_preflight is not None:
        on_preflight()
    e, c5 = pf["entry"], pf["c5"]
    ev.code_commit = pf["git"]["commit"]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"tag": tag, "time": now_iso(), "code_commit": pf["git"]["commit"], "protocol": LP.protocol_spec(),
                "protocol_sha256": pf["protocol_sha256"], "identity_sha256_before": e["identity_sha256"], "repo": e["repo"], "revision": e["revision"],
                "blend": {"ensemble_w": c5["ensemble_w"], "blend_sha256": c5["blend_sha256"], "source": e["blend"]["source_rel"],
                          "source_sha256": c5["blend_source_sha256"]},
                "siglip_identity_sha256": c5["siglip_identity_sha256"], "image_spec": e["image_spec"],
                "ledger_sha256_before": pf["ledger_sha256_before"]}
    (raw_dir / f"{tag}.letterlogit_manifest.json").write_text(json.dumps(manifest, indent=1))  # no scores
    log(f"preflight passed: tag {tag}, identity {e['identity_sha256'][:16]}, protocol {pf['protocol_sha256'][:16]}, blend {c5['blend_sha256'][:16]}, "
        f"siglip {c5['siglip_identity_sha256'][:16]}, code {pf['git']['commit'][:12]}")
    if loader is None:
        from cgt.p3b_vlm.shotbench import load_shotbench_samples as loader
    samples, stats = loader(test_tsv, image_root)
    if sum(v["dropped_unparsed"] + v["dropped_roundtrip"] for v in stats["per_attribute"].values()):
        raise guards.FailClosed("ShotBench questions were dropped while parsing: refusing to evaluate a partial set")
    for s in samples:
        guards.require_under_image_root(s["images"][0], image_root)
        LP.option_letters(s["options"])
    originals = {s["id"]: s["images"][0] for s in samples}
    guards.require_unique_stems(list(originals.values()))
    from cgt.p3b_vlm.images import prepare_eval_cache
    try:
        cached = prepare_eval_cache(list(originals.values()), img_cache, e["image_spec"]["budget_px"], e["image_spec"]["multiple"])
    except (OSError, ValueError) as exc:
        raise guards.UnreadableImage(f"{type(exc).__name__} while preparing the image cache") from exc
    for s in samples:
        s["images"] = [cached[originals[s["id"]]]]
    guards.require_images_readable([s["images"][0] for s in samples])
    ev.start(len(samples))
    vlm = (vlm_factory or _default_vlm_factory)(e)
    if getattr(vlm, "processor", None) is not None:
        guards.require_token_lengths(vlm.processor, samples, 1024, 144)
    log(f"evaluating {len(samples)} questions under the letter-logit + C5 protocol; progress counts only until sealed")
    t0 = time.perf_counter()

    def _evaluate(cb):
        sig = (siglip_factory or _default_siglip_factory)(samples, originals, image_root, feat_cache, e, c5)
        return score_questions(samples, vlm, sig, c5["ensemble_w"], progress=cb)
    res = ev.run(_evaluate)
    secs = time.perf_counter() - t0
    after = registry.require_identity(tag, data_root)
    if after["identity_sha256"] != e["identity_sha256"]:
        raise guards.AdapterHashError("the model identity changed during the evaluation")
    summ = summarize(res)
    rows, rows0 = [], []
    for at in ATTRS:
        v, v0 = summ["per_attribute"].get(at), summ["c0_per_attribute"].get(at)
        if v is None:
            continue
        lo, hi = sealed_eval.wilson(v["correct"], v["n"])
        acc = round(100 * v["correct"] / v["n"], 2)
        rows.append({"attribute": at, "n": v["n"], "correct": v["correct"], "acc_pct": acc, "ci95": [round(lo, 1), round(hi, 1)],
                     "invalid_rate_pct": 0.0, "expected_fallback_acc_pct": acc})  # same row keys as the matched summary (compare script)
        rows0.append({"attribute": at, "n": v0["n"], "correct": v0["correct"], "acc_pct": round(100 * v0["correct"] / v0["n"], 2)})
    summary = {"tag": tag, "protocol": LP.PROTOCOL_VERSION, "protocol_sha256": pf["protocol_sha256"], "identity_sha256": e["identity_sha256"],
               "blend_sha256": c5["blend_sha256"], "siglip_identity_sha256": c5["siglip_identity_sha256"], "stats": stats, "rows": rows,
               "mean_acc_pct": round(100 * summ["mean_acc"], 2), "invalid_rate_pct": 0.0, "mean_expected_fallback_acc_pct": round(100 * summ["mean_acc"], 2),
               "descriptive_c0_vlm_only": {"rows": rows0, "mean_acc_pct": round(100 * summ["c0_mean_acc"], 2),
                                           "note": "same pass, VLM letter logits alone; descriptive only, never selectable"},
               "n_fewer_than_4_options": sum(len(r["letters"]) < 4 for r in res),  # where this rule differs from the 2B argmax over A-D
               "seconds": secs, "n": summ["n"]}
    pred_text = "\n".join(json.dumps(r) for r in res) + "\n"
    manifest.update({"identity_sha256_after": after["identity_sha256"], "predictions_sha256": hashlib.sha256(pred_text.encode()).hexdigest()})
    echo = (f"DONE: letter-logit + C5 mean accuracy {summary['mean_acc_pct']}% over {len(rows)} attributes "
            f"(descriptive VLM-only {summary['descriptive_c0_vlm_only']['mean_acc_pct']}%); ledger recorded")
    note = (f"letterlogit+C5 protocol {pf['protocol_sha256'][:12]}; n={summ['n']}; identity {e['identity_sha256'][:12]}; "
            f"blend {c5['blend_sha256'][:12]}; siglip {c5['siglip_identity_sha256'][:12]}")
    ev.seal(lambda: sbmod.ledger_record(tag, note, ledger), lambda: tag in sbmod.ledger_read(ledger),
            {Path(out_dir) / f"shotbench_{tag}.json": json.dumps(summary, indent=1), raw_dir / f"{tag}.predictions.jsonl": pred_text,
             raw_dir / f"{tag}.letterlogit_manifest.json": json.dumps(manifest, indent=1)}, echo)
