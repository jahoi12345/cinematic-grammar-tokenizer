"""Descriptive re-score of past ShotQA validation results under dedup v2 (docs/plans/val_dedup_v2.md). CPU only, no inference;
reads existing per-question prediction files and writes AGGREGATES only to results/val_dedup_v2/rescore.{json,md}.

- Frozen baseline (= Step A C0): Phase 3B ckpt_100.predictions.jsonl (sha256 cebea813...) covers all 5,826 lines -> exact v2 numbers.
- Step A winner and Step B checkpoints have predictions on the 5,762 dedup_first rows only. v2 adds 60 rows, so for them we give the
  exact old number, the worst/best-case bounds (added rows all wrong / all right) and a labelled imputation (same-attribute pair ->
  the model's correctness on the first occurrence; different attribute -> the model's old accuracy on that attribute). The
  imputation is checked on the baseline, where the truth is known. Nothing is re-decided."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.eval.metrics import pick_option
from cgt.eval.val_dedup import dedup_v2_indices
from cgt.p3b_vlm.guards import sha256_file
from cgt.p3s_stepa import combine as C
from cgt.p3s_stepa.split import ATTRS, load_split, shotqa_film_of
from cgt.stepb.select import select

VAL = ROOT / "data/cache/p3b/val.jsonl"
VAL_SHA = "409e87d59faedc4753a27bf0e9151af1e7991cf53549434f53da4ec519c2637e"
FROZEN_PRED = ROOT / "data/cache/p3b/runs/qwen3vl2b_lora_filmsplit_run1/val_predictions/ckpt_100.predictions.jsonl"
FROZEN_PRED_SHA = "cebea813002eb24971d6ae652b38a90a0f7fc7105ceed92d9a0da3c64770ece6"
STEPA_WINNER = "C1+C3+C4*+C5"
STEPB_PRED = ROOT / "data/cache/stepb/predictions"
STEPB_RUNS = ROOT / "results/stepb/runs"
OUT = ROOT / "results/val_dedup_v2"
ADDED = ROOT / "data/cache/val_dedup_v2"   # per-question outputs of scripts/val_dedup_v2_{infer,siglip}.py (gitignored)
STEPA_RES = ROOT / "results/p3s_stepa/static_val.json"
NOT_RUN_ON_ADDED = ("C2a", "C2b", "C1+C2*+C3+C4*+C5")   # need the res262k / res590k views, not run on the added rows


def scores_from_hits(lines: list[int], hit: dict[int, float], rows: list[dict]) -> dict:
    """{'per_attribute': points, 'primary': points, 'n': int} from per-line hit values (0/1 or an expected value)."""
    per = {}
    for a in ATTRS:
        h = [hit[n] for n in lines if rows[n]["attribute"] == a]
        per[a] = 100.0 * float(np.mean(h))
    return {"primary": float(np.mean(list(per.values()))), "per_attribute": per, "n": len(lines)}


class Ctx:
    def __init__(self) -> None:
        if sha256_file(VAL) != VAL_SHA:
            raise SystemExit("val.jsonl does not match its recorded sha256")
        self.rows = [json.loads(x) for x in VAL.read_text().splitlines()]
        sp = load_split(VAL, shotqa_film_of)
        self.qs = sp["questions"]                                  # dedup_first rows with line, key, film, half
        self.old = [q["line"] for q in self.qs]
        self.new = dedup_v2_indices(self.rows)
        assert set(self.old) <= set(self.new) and len(self.old) == 5762 and len(self.new) == 5822
        self.added = sorted(set(self.new) - set(self.old))
        first_line = {q["id"]: q["line"] for q in self.qs}
        self.partner = {n: first_line[self.rows[n]["id"]] for n in self.added}
        half_of_img = {q["img_id"]: q["half"] for q in self.qs}
        self.half = {n: half_of_img[self.rows[n]["img_id"]] for n in range(len(self.rows))}

    def part(self, lines: list[int], half: str | None) -> list[int]:
        return [n for n in lines if half is None or self.half[n] == half]

    def added_counts(self, half: str | None) -> dict:
        add = self.part(self.added, half)
        same = [n for n in add if self.rows[n]["attribute"] == self.rows[self.partner[n]]["attribute"]]
        return {"added": len(add), "same_attribute_as_first": len(same), "different_attribute": len(add) - len(same),
                "per_attribute": {a: sum(self.rows[n]["attribute"] == a for n in add) for a in ATTRS}}

    def impute(self, hit_old: dict[int, float], half: str | None) -> dict[int, float]:
        old = self.part(self.old, half)
        acc = {a: float(np.mean([hit_old[n] for n in old if self.rows[n]["attribute"] == a])) for a in ATTRS}
        out = {}
        for n in self.part(self.added, half):
            p = self.partner[n]
            out[n] = hit_old[p] if self.rows[p]["attribute"] == self.rows[n]["attribute"] else acc[self.rows[n]["attribute"]]
        return out

    def report_partial(self, hit_old: dict[int, float], half: str | None) -> dict:
        """Old exact, v2 bounds (added rows all 0 / all 1) and v2 imputed, for a model scored on the old rows only."""
        old, new = self.part(self.old, half), self.part(self.new, half)
        lo = scores_from_hits(new, {**hit_old, **{n: 0.0 for n in self.added}}, self.rows)
        hi = scores_from_hits(new, {**hit_old, **{n: 1.0 for n in self.added}}, self.rows)
        imp = scores_from_hits(new, {**hit_old, **self.impute(hit_old, half)}, self.rows)
        return {"old": scores_from_hits(old, hit_old, self.rows), "v2_worst": lo, "v2_best": hi, "v2_imputed": imp}


def frozen_hits(ctx: Ctx) -> dict[int, float]:
    if sha256_file(FROZEN_PRED) != FROZEN_PRED_SHA:
        raise SystemExit("frozen predictions do not match their recorded sha256")
    P = [json.loads(x) for x in FROZEN_PRED.read_text().splitlines()]
    assert [p["id"] for p in P] == [r["id"] for r in ctx.rows]
    return {n: float(p["pred"] == p["gold"]) for n, p in enumerate(P)}


def stepa_hits(ctx: Ctx, name: str) -> dict[int, float]:
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("p3s_stepa_report", ROOT / "scripts/p3s_stepa_report.py")
    rep = module_from_spec(spec)
    spec.loader.exec_module(rep)
    views, complete = rep.load_views(ctx.qs, ROOT / "data/cache/p3s_stepa/views")
    assert all(c["done"] == c["needed"] for c in complete.values())
    sig = {r["key"]: r["scores"] for r in map(json.loads, (ROOT / "data/cache/p3s_stepa/siglip3a.jsonl").read_text().splitlines())}
    sc = C.run_config(name, ctx.qs, views, sig)["scores"]
    return {q["line"]: float(pick_option(np.asarray(sc[q["key"]]), list(C.LETTERS)) == q["answer"]) for q in ctx.qs}


def stepb_hits(ctx: Ctx, cand: str, pct: int) -> tuple[dict[int, float], str]:
    s = json.loads((STEPB_RUNS / f"stepb_{cand}" / f"val_ckpt{pct:03d}.summary.json").read_text())
    f = STEPB_PRED / f"val_ckpt{pct:03d}_{s['adapter_sha256'][:12]}.shotqa.jsonl"
    P = [json.loads(x) for x in f.read_text().splitlines()]
    assert [p["id"] for p in P] == [ctx.rows[n]["id"] for n in ctx.old]
    hit = {n: float(p["pred"] == p["gold"]) for n, p in zip(ctx.old, P, strict=True)}
    got = scores_from_hits(ctx.old, hit, ctx.rows)["primary"]
    assert abs(got - s["shotqa"]["primary"]) < 1e-9, (cand, pct, got, s["shotqa"]["primary"])
    return hit, f.name


def main() -> None:
    ctx = Ctx()
    out: dict = {"plan": "docs/plans/val_dedup_v2.md", "key": "(id, attribute, sorted option texts), first occurrence in line order",
                 "val_sha256": VAL_SHA, "lines": len(ctx.rows), "dedup_first": len(ctx.old), "dedup_v2": len(ctx.new),
                 "added_rows": {"full": ctx.added_counts(None), "selection_half": ctx.added_counts("sel")}}
    fb = frozen_hits(ctx)
    base = {}
    for h, label in ((None, "full"), ("sel", "selection_half")):
        base[label] = {"old": scores_from_hits(ctx.part(ctx.old, h), fb, ctx.rows), "v2": scores_from_hits(ctx.part(ctx.new, h), fb, ctx.rows),
                       "imputation_check": ctx.report_partial({n: fb[n] for n in ctx.old}, h)}
    base["full"]["all_5826_lines"] = scores_from_hits(list(range(len(ctx.rows))), fb, ctx.rows)
    base["full"]["added_rows_correct"] = int(sum(fb[n] for n in ctx.added))
    base["full"]["added_rows_correct_imputed"] = float(sum(ctx.impute({n: fb[n] for n in ctx.old}, None).values()))
    out["frozen_baseline_eq_stepA_C0"] = base

    # Step A: C0 recomputed from views (must equal the baseline on the old rows) and the winner
    a_c0 = stepa_hits(ctx, "C0")
    assert all(a_c0[n] == fb[n] for n in ctx.old)
    a_w = stepa_hits(ctx, STEPA_WINNER)
    sa = {lab: ctx.report_partial(a_w, h) for h, lab in ((None, "full"), ("sel", "selection_half"))}
    sel_c0_old, sel_c0_new = base["selection_half"]["old"]["primary"], base["selection_half"]["v2"]["primary"]
    w = sa["selection_half"]
    sa["gain_vs_C0_selection_half"] = {"old": w["old"]["primary"] - sel_c0_old, "v2_worst": w["v2_worst"]["primary"] - sel_c0_new,
                                       "v2_best": w["v2_best"]["primary"] - sel_c0_new, "v2_imputed": w["v2_imputed"]["primary"] - sel_c0_new}
    out["stepA_winner"] = {"config": STEPA_WINNER, **sa}

    # Step B
    sb, ck_old, ck_lo, ck_hi, ck_imp = {}, [], [], [], []
    for cand in ("B0", "B1"):
        for pct in (50, 100):
            hit, fname = stepb_hits(ctx, cand, pct)
            r = ctx.report_partial(hit, None)
            sb[f"{cand} {pct}%"] = {"predictions_file": fname, **r}
            for lst, k in ((ck_old, "old"), (ck_lo, "v2_worst"), (ck_hi, "v2_best"), (ck_imp, "v2_imputed")):
                lst.append({"candidate": cand, "pct": pct, "secondary": 0.0, **r[k]})
    b_old = {**base["full"]["old"]}
    b_new = {**base["full"]["v2"]}

    def elig(cks: list[dict], b: dict) -> dict:
        res = select(cks, b)
        return {f"{t['candidate']} {t['pct']}%": {"eligible": t["eligible"], "reasons": t["reasons"]} for t in res["table"]}

    # worst case for eligibility: the checkpoint at its v2 lower bound; best case: at its upper bound (baseline v2 is exact)
    out["stepB"] = {"checkpoints": sb, "eligibility": {"old (as recorded)": elig(ck_old, b_old), "v2 imputed": elig(ck_imp, b_new),
                                                       "v2 worst case": elig(ck_lo, b_new), "v2 best case": elig(ck_hi, b_new)}}
    if (ADDED / "run.json").exists():
        out["actual_v2"] = actual(ctx, fb, out)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rescore.json").write_text(json.dumps(out, indent=1) + "\n")
    print(tables(out))


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines()]


def actual(ctx: Ctx, fb: dict[int, float], out: dict) -> dict:
    """Actual v2 numbers from the inference on the 60 added rows (Amendment 1)."""
    from cgt.p3s_stepa.views import RUN_ORDER
    res: dict = {"run": json.loads((ADDED / "run.json").read_text())}
    # ---- Step B
    sb = {}
    b_new = out["frozen_baseline_eq_stepA_C0"]["full"]["v2"]
    cks = []
    for cand in ("B0", "B1"):
        hit_old, fname = stepb_hits(ctx, cand, 100)
        add = jl(ADDED / f"stepb_{cand}.jsonl")
        assert sorted(r["line"] for r in add) == ctx.added
        for r in add:
            assert ctx.rows[r["line"]]["id"] == r["id"] and r["gold"] == ctx.rows[r["line"]]["answer"]
        hit = {**hit_old, **{r["line"]: float(r["pred"] == r["gold"]) for r in add}}
        old_pred = {p["id"]: p for p in jl(STEPB_PRED / fname)}
        par = jl(ADDED / f"stepb_{cand}_parity.jsonl")
        pdiff = max(float(np.abs(np.array(r["logits"]) - np.array(old_pred[r["id"]]["logits"])).max()) for r in par)
        same = sum(r["pred"] == old_pred[r["id"]]["pred"] for r in par)
        v2 = scores_from_hits(ctx.new, hit, ctx.rows)
        sb[f"{cand} 100%"] = {"v2_actual": v2, "added_rows_correct": int(sum(hit[n] for n in ctx.added)),
                              "parity_5_rows": {"max_abs_logit_diff": pdiff, "same_pred": same}}
        cks.append({"candidate": cand, "pct": 100, "secondary": 0.0, **v2})
    sel = select(cks, b_new)
    res["stepB"] = {"checkpoints": sb, "eligibility_v2_actual": {f"{t['candidate']} {t['pct']}%": {"eligible": t["eligible"], "reasons": t["reasons"]}
                                                                 for t in sel["table"]},
                    "note": "B0 50% and B1 50% were not run on the added rows (their v2 best case is already ineligible)"}
    # ---- Step A
    recorded = json.loads(STEPA_RES.read_text())
    qs_old = ctx.qs
    film_of_img = {q["img_id"]: q["film"] for q in qs_old}
    qadd = []
    for n in ctx.added:
        r = ctx.rows[n]
        h = ctx.half[n]
        qadd.append({**r, "line": n, "key": f"{n}:{r['id']}", "film": film_of_img[r["img_id"]],
                     "half": "sel" if h == "sel" else "cal_added_v2"})   # never 'cal': excluded from every fit
    qs_v2 = sorted(qs_old + qadd, key=lambda q: q["line"])
    sel_old = [q for q in qs_old if q["half"] == "sel"]
    sel_v2 = [q for q in qs_v2 if q["half"] == "sel"]
    views = {}
    for v in RUN_ORDER:
        d = {}
        f = ROOT / "data/cache/p3s_stepa/views" / f"{v}.jsonl"
        if f.exists():
            d = {r["key"]: r["logits"] for r in jl(f)}
        fa = ADDED / f"views_{v}.jsonl"
        if fa.exists():
            d.update({r["key"]: r["logits"] for r in jl(fa)})
        views[v] = d
    P = FROZEN_PRED.read_text().splitlines()
    base_diff = max(float(np.abs(np.array(views["base"][q["key"]]) - np.array(json.loads(P[q["line"]])["logits"])).max()) for q in qadd)
    sig = {r["key"]: r["scores"] for r in jl(ROOT / "data/cache/p3s_stepa/siglip3a.jsonl")}
    sig.update({r["key"]: r["scores"] for r in jl(ADDED / "siglip3a_added.jsonl")})
    configs = [n for n in C.ORDER if n not in NOT_RUN_ON_ADDED]
    tab, hits_sel = {}, {}
    for name in configs:
        r = C.run_config(name, qs_v2, views, sig)
        fit_rec = recorded["results"][name]["fit"]
        fit_new = json.loads(json.dumps(r["info"]))
        assert fit_new == fit_rec, (name, fit_new, fit_rec)                       # nothing refitted
        old_sel = C.accuracy(sel_old, r["scores"])["mean"]
        assert abs(old_sel - recorded["results"][name]["selection"]["mean"]) < 1e-12, name
        tab[name] = {"selection_v2": 100 * C.accuracy(sel_v2, r["scores"])["mean"], "full_v2": 100 * C.accuracy(qs_v2, r["scores"])["mean"],
                     "selection_old": 100 * old_sel, "full_old": 100 * recorded["results"][name]["full"]["mean"]}
        hits_sel[name] = C.hits(sel_v2, r["scores"])
        added_sel = [q for q in qadd if q["half"] == "sel"]
        tab[name]["added_selection_rows_correct"] = f"{int(C.hits(added_sel, r['scores']).sum())}/{len(added_sel)}"
        tab[name]["added_rows_correct"] = f"{int(C.hits(qadd, r['scores']).sum())}/{len(qadd)}"
        if name in ("C0", STEPA_WINNER):
            hits_sel[name + "@old"] = C.hits(sel_old, r["scores"])
    chk = C.paired_bootstrap(sel_old, hits_sel["C0@old"], hits_sel[STEPA_WINNER + "@old"], cluster="film", n_boot=10_000, seed=0)
    assert chk["ci95"] == recorded["bootstrap_vs_C0"][STEPA_WINNER]["film"]["ci95"], chk   # the recorded interval reproduces
    not_run = {}
    for name in NOT_RUN_ON_ADDED:   # bounds only: scored on the old rows, the added selection rows all wrong / all right
        r = C.run_config(name, qs_old, views, sig)
        assert json.loads(json.dumps(r["info"])) == recorded["results"][name]["fit"], name
        hit = {q["line"]: float(h) for q, h in zip(qs_old, C.hits(qs_old, r["scores"]), strict=True)}
        sl_old, sl_new = ctx.part(ctx.old, "sel"), ctx.part(ctx.new, "sel")
        not_run[name] = {"selection_old": scores_from_hits(sl_old, hit, ctx.rows)["primary"],
                         "selection_v2_worst": scores_from_hits(sl_new, {**hit, **{n: 0.0 for n in ctx.added}}, ctx.rows)["primary"],
                         "selection_v2_best": scores_from_hits(sl_new, {**hit, **{n: 1.0 for n in ctx.added}}, ctx.rows)["primary"]}
    boot = {cl: C.paired_bootstrap(sel_v2, hits_sel["C0"], hits_sel[STEPA_WINNER], cluster=cl, n_boot=10_000, seed=0) for cl in ("film", "question")}
    ranked = sorted(configs, key=lambda n: (-tab[n]["selection_v2"], C.ORDER.index(n)))
    res["stepA"] = {"parity_base_view_vs_frozen_logits_max_abs_diff": base_diff, "fits_equal_recorded": True,
                    "selection_v2_n": len(sel_v2), "full_v2_n": len(qs_v2), "configs": tab,
                    "not_run_on_added_rows": not_run,
                    "winner_vs_C0_selection_v2_bootstrap": boot, "top_on_selection_v2_among_computed": ranked[0]}
    return res


def tables(o: dict) -> str:
    """Markdown tables of the aggregate numbers (pasted into rescore.md)."""
    f2 = "{:.2f}".format
    b = o["frozen_baseline_eq_stepA_C0"]
    L = ["### Frozen baseline (= Step A C0), exact", "", "| set | old (dedup_first) | v2 | diff | " + " | ".join(f"{a} old / v2" for a in ATTRS) + " |",
         "|---|---|---|---|" + "---|" * len(ATTRS)]
    for h in ("full", "selection_half"):
        x = b[h]
        L.append(f"| {h} ({x['old']['n']} -> {x['v2']['n']}) | {f2(x['old']['primary'])} | {f2(x['v2']['primary'])} | "
                 f"{x['v2']['primary'] - x['old']['primary']:+.2f} | "
                 + " | ".join(f"{f2(x['old']['per_attribute'][a])} / {f2(x['v2']['per_attribute'][a])}" for a in ATTRS) + " |")
    L += ["", "Imputation check on the baseline (where the truth is known):", "", "| set | v2 true | v2 imputed | imputation error | v2 worst | v2 best |",
          "|---|---|---|---|---|---|"]
    for h in ("full", "selection_half"):
        x, c = b[h], b[h]["imputation_check"]
        L.append(f"| {h} | {f2(x['v2']['primary'])} | {f2(c['v2_imputed']['primary'])} | {c['v2_imputed']['primary'] - x['v2']['primary']:+.3f} | "
                 f"{f2(c['v2_worst']['primary'])} | {f2(c['v2_best']['primary'])} |")
    L += ["", "### Models scored on the 5,762 old rows only (v2 adds 60 rows without predictions)", "",
          "| model | set | old (exact) | v2 worst (added all wrong) | v2 best (added all right) | v2 imputed | angle old | angle v2 imputed [worst, best] |",
          "|---|---|---|---|---|---|---|---|"]
    a = o["stepA_winner"]
    rows = [(f"Step A winner {a['config']}", h, a[h]) for h in ("selection_half", "full")]
    rows += [(f"Step B {k}", "full", v) for k, v in o["stepB"]["checkpoints"].items()]
    for name, h, r in rows:
        L.append(f"| {name} | {h} | {f2(r['old']['primary'])} | {f2(r['v2_worst']['primary'])} | {f2(r['v2_best']['primary'])} | "
                 f"{f2(r['v2_imputed']['primary'])} | {f2(r['old']['per_attribute']['angle'])} | {f2(r['v2_imputed']['per_attribute']['angle'])} "
                 f"[{f2(r['v2_worst']['per_attribute']['angle'])}, {f2(r['v2_best']['per_attribute']['angle'])}] |")
    g = a["gain_vs_C0_selection_half"]
    L += ["", (f"Step A gain of the winner over C0 on the selection half: old {g['old']:+.2f}; v2 worst {g['v2_worst']:+.2f}, "
               f"best {g['v2_best']:+.2f}, imputed {g['v2_imputed']:+.2f} (C0 v2 is exact)."), "",
          "### Step B eligibility (rule of cgt.stepb.select; baseline old 81.23 / v2 exact " + f2(b["full"]["v2"]["primary"]) + ")", "",
          "| checkpoint | " + " | ".join(o["stepB"]["eligibility"]) + " |", "|---|" + "---|" * len(o["stepB"]["eligibility"])]
    for ck in o["stepB"]["checkpoints"]:
        L.append(f"| {ck} | " + " | ".join("eligible" if e[ck]["eligible"] else "no" for e in o["stepB"]["eligibility"].values()) + " |")
    if "actual_v2" in o:
        L += actual_tables(o)
    return "\n".join(L) + "\n"


def actual_tables(o: dict) -> list[str]:
    f2 = "{:.2f}".format
    act, b = o["actual_v2"], o["frozen_baseline_eq_stepA_C0"]["full"]["v2"]
    L = ["", "### ACTUAL v2 numbers (inference on the 60 added rows, Amendment 1)", "", "Step B (full val, 5,822; baseline v2 exact "
         + f2(b["primary"]) + ", bar " + f2(b["primary"] + 0.5) + "):", "",
         "| checkpoint | old | v2 imputed | v2 actual | angle v2 actual | added rows right | eligible (v2 actual) |", "|---|---|---|---|---|---|---|"]
    for ck, v in act["stepB"]["checkpoints"].items():
        prev = o["stepB"]["checkpoints"][ck]
        e = act["stepB"]["eligibility_v2_actual"][ck]
        L.append(f"| {ck} | {f2(prev['old']['primary'])} | {f2(prev['v2_imputed']['primary'])} | **{f2(v['v2_actual']['primary'])}** | "
                 f"{f2(v['v2_actual']['per_attribute']['angle'])} | {v['added_rows_correct']}/60 | {'yes' if e['eligible'] else 'no: ' + '; '.join(e['reasons'])} |")
    a = act["stepA"]
    L += ["", f"Step A (selection half {a['selection_v2_n']}, full {a['full_v2_n']}; fits identical to the recorded ones):", "",
          "| config | selection old | selection v2 | full old | full v2 | added selection rows right |", "|---|---|---|---|---|---|"]
    for n, v in a["configs"].items():
        L.append(f"| {n} | {f2(v['selection_old'])} | {f2(v['selection_v2'])} | {f2(v['full_old'])} | {f2(v['full_v2'])} | {v['added_selection_rows_correct']} |")
    for n, v in a["not_run_on_added_rows"].items():
        L.append(f"| {n} (res views not run on added rows) | {f2(v['selection_old'])} | [{f2(v['selection_v2_worst'])}, {f2(v['selection_v2_best'])}] | | | - |")
    bf, bq = a["winner_vs_C0_selection_v2_bootstrap"]["film"], a["winner_vs_C0_selection_v2_bootstrap"]["question"]
    L += ["", (f"Winner vs C0 on the v2 selection half: {bf['diff_points']:+.2f} points, film-clustered 95% interval "
               f"[{bf['ci95'][0]:+.2f}, {bf['ci95'][1]:+.2f}], question-level [{bq['ci95'][0]:+.2f}, {bq['ci95'][1]:+.2f}] (10,000 resamples, seed 0).")]
    return L


if __name__ == "__main__":
    main()
