"""Step A static track (docs/acc_step_a_static_plan.md): synthetic fixtures only (no model, no real labels, no ShotBench)."""
import copy
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.special import log_softmax

from cgt.p3_static_attrs.data import SHOTQA_ROOT
from cgt.p3s_stepa import combine as C
from cgt.p3s_stepa.split import dedup_first, split_films_half
from cgt.p3s_stepa.views import (
    LETTERS,
    VIEWS,
    applies,
    budget_size_no_upscale,
    build_user_text,
    prepare_image,
    shifted_options,
    to_original,
)

ROOT = Path(__file__).resolve().parents[1]
SIZE_LABELS = ["Extreme Close Up", "Close Up", "Medium Close Up", "Medium", "Medium Wide", "Wide", "Extreme Wide"]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def prompt(opts):
    return "Question: What's the shot size of this shot?\nOptions:\n" + "".join(f"{x}. {opts[x]}\n" for x in LETTERS) + \
        "Please select the most likely answer from the options above. \n"


# ---------- option permutations ----------
def test_shift_mapping_roundtrip():
    opts = dict(zip(LETTERS, ["p", "q", "r", "t"], strict=True))
    for s in range(4):
        sh = shifted_options(opts, s)
        content_scores = {"p": 1.0, "q": 2.0, "r": 3.0, "t": 4.0}
        view = [content_scores[sh[x]] for x in LETTERS]  # model output in the view's letter order
        assert list(to_original(view, s)) == [1.0, 2.0, 3.0, 4.0]


def test_permutation_averaging_cancels_letter_bias():
    """A synthetic model = content score + a strong letter-position bias. C0 follows the bias; C1 (K=4 cyclic) recovers the content argmax."""
    rng = np.random.default_rng(0)
    bias = np.array([3.0, 0.0, 0.0, 0.0])  # loves letter A
    qs, views = [], {v: {} for v in ("base", "shift1", "shift2", "shift3")}
    for n in range(50):
        opts = dict(zip(LETTERS, rng.permutation(SIZE_LABELS)[:4], strict=True))
        content = dict(zip(opts.values(), rng.normal(size=4), strict=True))
        best = max(content, key=content.get)
        q = {"key": f"{n}:x", "attribute": "size", "half": "sel", "film": "f", "options": opts,
             "answer": next(x for x in LETTERS if opts[x] == best)}
        qs.append(q)
        for v, out in views.items():
            sh = shifted_options(opts, VIEWS[v]["shift"])
            out[q["key"]] = [content[sh[x]] + bias[i] for i, x in enumerate(LETTERS)]
    c1 = C.averaged(qs, views, ["base", "shift1", "shift2", "shift3"])
    assert C.accuracy(qs, c1)["mean"] == 1.0
    assert C.accuracy(qs, C.averaged(qs, views, ["base"]))["mean"] < 0.6
    q0 = qs[0]
    manual = np.mean([to_original(log_softmax(np.asarray(views[v][q0["key"]])), VIEWS[v]["shift"]) for v in views], 0)
    assert np.allclose(c1[q0["key"]], manual)


def test_build_user_text():
    opts = {"A": "Wide", "B": "Medium", "C": "Close Up", "D": "Extreme Wide"}
    t = prompt(opts)
    assert build_user_text(t, opts, 0) == t
    assert "A. Medium\nB. Close Up\nC. Extreme Wide\nD. Wide\n" in build_user_text(t, opts, 1)
    with pytest.raises(ValueError):
        build_user_text(t.replace("A. Wide", "A. Wode"), opts, 1)


def test_view_attribute_rules():
    assert not applies("hflip", "composition") and all(applies("hflip", a) for a in ("size", "lens", "angle", "framing", "light_type", "light_cond"))
    assert {a for a in C.ATTRS if applies("crop80", a)} == {"light_type", "light_cond"}
    assert all(applies(v, a) for v in ("base", "shift1", "shift2", "shift3", "res262k", "res590k") for a in C.ATTRS)


def test_resolution_never_upscales(tmp_path):
    assert budget_size_no_upscale(320, 240, 589_824) == (320, 224)
    src = tmp_path / "s.png"
    Image.new("RGB", (400, 300), (10, 20, 30)).save(src)
    out = prepare_image(str(src), tmp_path / "o.jpg", "plain", 589_824)
    assert Image.open(out).size[0] * Image.open(out).size[1] <= 400 * 300
    big = tmp_path / "b.png"
    Image.new("RGB", (2000, 1000)).save(big)
    w, h = Image.open(prepare_image(str(big), tmp_path / "b.jpg", "crop80", 147_456)).size
    assert w * h <= 147_456 and w % 32 == 0 and h % 32 == 0


def test_hflip_mirrors(tmp_path):
    src = tmp_path / "s.png"
    im = Image.new("RGB", (640, 480), (0, 0, 0))
    im.paste((255, 255, 255), (0, 0, 100, 480))  # white stripe on the LEFT
    im.save(src)
    out = Image.open(prepare_image(str(src), tmp_path / "f.jpg", "hflip", 147_456)).convert("L")
    a = np.asarray(out)
    assert a[:, -5:].mean() > 200 and a[:, :5].mean() < 50


# ---------- dedup and split ----------
def test_dedup_keeps_first_occurrence():
    rows = [{"id": "a#0", "v": 1}, {"id": "b#0", "v": 2}, {"id": "a#0", "v": 3}]
    d = dedup_first(rows)
    assert [r["v"] for r in d] == [1, 2] and [r["key"] for r in d] == ["0:a#0", "1:b#0"]


def test_film_split_deterministic_disjoint():
    films = [f"f{i}" for i in range(71)]
    c1, s1 = split_films_half(films, 0)
    c2, s2 = split_films_half(list(reversed(films)), 0)
    assert c1 == c2 and s1 == s2 and not (c1 & s1) and c1 | s1 == set(films) and len(c1) == 36
    assert split_films_half(films, 1)[0] != c1


# ---------- calibration / ensemble: fitted on the calibration half only ----------
def synthetic(n_per=60, seed=0):
    """Two attributes x two halves; VLM scores carry a letter bias toward A; 3A scores are informative."""
    rng = np.random.default_rng(seed)
    qs, views, sig = [], {"base": {}}, {}
    k = 0
    for attr, labs in (("size", SIZE_LABELS), ("lens", ["Wide", "Medium", "Long Lens", "Ultra Wide / Fisheye"])):
        for half in ("cal", "sel"):
            for _ in range(n_per):
                opts = dict(zip(LETTERS, rng.permutation(labs)[:4], strict=True))
                ans = LETTERS[rng.integers(4)]
                key = f"{k}:x{k}"
                k += 1
                z = rng.normal(size=4)
                z[LETTERS.index(ans)] += 1.0
                z[0] += 1.5
                s = rng.normal(size=4)
                s[LETTERS.index(ans)] += 1.5
                qs.append({"key": key, "attribute": attr, "half": half, "film": f"{half}{k % 6}", "options": opts, "answer": ans})
                views["base"][key] = list(z)
                sig[key] = list(s)
    return qs, views, sig


def _scramble_selection(qs, views, sig):
    q2, v2, s2 = copy.deepcopy(qs), copy.deepcopy(views), copy.deepcopy(sig)
    rng = np.random.default_rng(9)
    for q in q2:
        if q["half"] == "sel":
            q["answer"] = LETTERS[rng.integers(4)]
            v2["base"][q["key"]] = list(rng.normal(size=4) * 5)
            s2[q["key"]] = list(rng.normal(size=4) * 5)
    return q2, v2, s2


@pytest.mark.parametrize("name", ["C4a", "C4b", "C5", "C1+C4*+C5"])
def test_fit_uses_calibration_half_only(name):
    qs, views, sig = synthetic()
    if name.startswith("C1"):
        for v in ("shift1", "shift2", "shift3"):
            views[v] = {k: list(np.roll(x, -VIEWS[v]["shift"])) for k, x in views["base"].items()}
    a = C.run_config(name, qs, views, sig)
    q2, v2, s2 = _scramble_selection(qs, views, sig)
    if name.startswith("C1"):
        for v in ("shift1", "shift2", "shift3"):
            v2[v] = {k: list(np.roll(x, -VIEWS[v]["shift"])) for k, x in v2["base"].items()}
    b = C.run_config(name, q2, v2, s2)
    assert json.dumps(a["info"], sort_keys=True) == json.dumps(b["info"], sort_keys=True)  # fitted parameters unchanged
    for q in qs:
        if q["half"] == "cal":
            assert np.allclose(a["scores"][q["key"]], b["scores"][q["key"]])


def test_calibration_changes_with_calibration_labels():
    qs, views, sig = synthetic()
    a = C.run_config("C4a", qs, views, sig)
    q2 = copy.deepcopy(qs)
    for q in q2:
        if q["half"] == "cal":
            q["answer"] = "A"
    b = C.run_config("C4a", q2, views, sig)
    assert a["info"]["calibration_params"] != b["info"]["calibration_params"]


def test_letter_bias_calibration_removes_bias():
    qs, views, sig = synthetic(n_per=300)
    sel = [q for q in qs if q["half"] == "sel"]
    r0 = C.accuracy(sel, C.run_config("C0", qs, views, sig)["scores"])["mean"]
    r4 = C.accuracy(sel, C.run_config("C4a", qs, views, sig)["scores"])["mean"]
    assert r4 > r0 + 0.03
    p = C.run_config("C4a", qs, views, sig)["info"]["calibration_params"]["size"]
    assert all(x > 0 for x in p[1:])  # b_B..b_D positive relative to b_A = 0: the A bias is undone


def test_ensemble_weight_tracks_informativeness():
    qs, views, sig = synthetic(n_per=200)
    w_good = C.fit_ensemble_w([q for q in qs if q["half"] == "cal"], C.averaged(qs, views, ["base"]), sig)
    nr = np.random.default_rng(1)
    noise = {k: list(nr.normal(size=4)) for k in sig}
    w_bad = C.fit_ensemble_w([q for q in qs if q["half"] == "cal"], C.averaged(qs, views, ["base"]), noise)
    # an uninformative 3A score still gets a small w: (1 - w) shrinks over-confident VLM log-probs (NLL fit; plan section 4)
    assert all(w_good[a] >= 0.5 for a in ("size", "lens")) and all(w_bad[a] <= 0.25 and w_bad[a] < w_good[a] - 0.3 for a in ("size", "lens"))
    assert set(w_good.values()) <= {float(x) for x in C.W_GRID}


def test_c2_star_picked_on_calibration_only():
    qs, views, sig = synthetic()
    rng = np.random.default_rng(3)
    for v in ("res262k", "res590k", "shift1", "shift2", "shift3", "hflip", "crop80"):
        views[v] = {k: list(np.array(x) + rng.normal(size=4) * 0.1) for k, x in views["base"].items()}
    a = C.run_config("C1+C2*+C3+C4*+C5", qs, views, sig)
    q2, v2, s2 = _scramble_selection(qs, views, sig)
    for v in ("res262k", "res590k", "shift1", "shift2", "shift3", "hflip", "crop80"):
        for q in q2:
            if q["half"] == "sel":
                v2[v][q["key"]] = list(rng.normal(size=4) * 5)
    b = C.run_config("C1+C2*+C3+C4*+C5", q2, v2, s2)
    assert a["info"]["C2*"] == b["info"]["C2*"] and a["info"]["C4*"] == b["info"]["C4*"]


def test_missing_view_output_raises():
    qs, views, _sig = synthetic()
    views["shift1"] = {}
    with pytest.raises(KeyError):
        C.averaged(qs, views, ["base", "shift1"])


# ---------- bootstrap and decision ----------
def test_bootstrap():
    qs, _, _ = synthetic()
    sel = [q for q in qs if q["half"] == "sel"]
    h = np.array([i % 2 == 0 for i in range(len(sel))])
    for cl in ("film", "question"):
        r = C.paired_bootstrap(sel, h, h, cluster=cl, n_boot=500)
        assert r["diff_points"] == 0 and r["ci95"] == [0.0, 0.0]
        r = C.paired_bootstrap(sel, np.zeros(len(sel), bool), np.ones(len(sel), bool), cluster=cl, n_boot=500)
        assert r["diff_points"] == 100 and r["ci95"] == [100.0, 100.0]
    hb = h.copy()
    hb[:5] = True
    r = C.paired_bootstrap(sel, h, hb, cluster="film", n_boot=2000)
    assert r["ci95"][0] <= r["diff_points"] <= r["ci95"][1] and r["diff_points"] > 0


def test_decision_rule():
    m = {"C0": 0.80, "C1": 0.81, "C3": 0.802}
    ok = {"C1": {"ci95": [0.1, 1.9]}, "C3": {"ci95": [0.1, 0.3]}}
    assert C.decide(m, ok, ["C0", "C1", "C3"])["winner"] == "C1"
    assert C.decide(m, {"C1": {"ci95": [-0.1, 1.9]}, "C3": ok["C3"]}, ["C0", "C1", "C3"])["winner"] == "C0"
    assert C.decide({"C0": 0.80, "C1": 0.804}, {"C1": {"ci95": [0.1, 0.7]}}, ["C0", "C1"])["winner"] == "C0"  # gain < 0.5
    # only the top one is tested: C3 passes nothing here, and C1 failing does not fall through
    assert C.decide({"C0": 0.8, "C1": 0.82, "C3": 0.81}, {"C1": {"ci95": [-1, 3]}, "C3": {"ci95": [0.5, 1.5]}}, ["C0", "C1", "C3"])["winner"] == "C0"
    assert C.decide({"C0": 0.8, "C1": 0.82}, {"C1": {"ci95": [0.5, 3]}}, ["C0"])["winner"] == "C0"  # unavailable configs never win


# ---------- GPU view script refusals (no model loaded) ----------
def _fake_root(tmp_path, adapter_bytes=b"x"):
    root = tmp_path / "root"
    (root / "data/cache/p3b/runs/qwen3vl2b_lora_filmsplit_run1").mkdir(parents=True)
    (root / "data/cache/p3b/runs/qwen3vl2b_lora_filmsplit_run1/ckpt_100.safetensors").write_bytes(adapter_bytes)
    opts = {"A": "Wide", "B": "Medium", "C": "Close Up", "D": "Extreme Wide"}
    rows = []
    for n, i in enumerate(["a#0", "b#0", "a#0"]):
        rows.append({"id": i, "img_id": i[0] + ".jpg", "attribute": "size", "images": [str(SHOTQA_ROOT / (i[0] + ".jpg"))],
                     "messages": [{"role": "user", "content": prompt(opts)}, {"role": "assistant", "content": "C"}], "options": opts,
                     "answer": "C" if n else "B"})
    (root / "data/cache/p3b/val.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return root


def test_views_refuse_wrong_adapter_before_anything(tmp_path):
    m = _load_script("p3s_stepa_views")
    root = _fake_root(tmp_path)
    called = []
    with pytest.raises(m.guards.AdapterHashError):
        m.run(Namespace(views="base", stop_at=None, n=None, out_dir=str(tmp_path / "o"), lock=str(tmp_path / "lock")),
              mem_reader=lambda: called.append("mem") or 90, model_loader=lambda *a: called.append("model"), root=root)
    assert called == [] and not (tmp_path / "lock").exists()


def test_views_memory_guard_and_lock(tmp_path, monkeypatch):
    m = _load_script("p3s_stepa_views")
    root = _fake_root(tmp_path)
    monkeypatch.setattr(m, "require_frozen_adapter", lambda p: "f" * 64)
    a = Namespace(views="base", stop_at=None, n=None, out_dir=str(tmp_path / "o"), lock=str(tmp_path / "lock"))
    with pytest.raises(m.MemoryLow):
        m.run(a, mem_reader=lambda: 10, model_loader=lambda *x: pytest.fail("model loaded"), root=root)
    (tmp_path / "lock").write_text("other job nonce=zz\n")
    with pytest.raises(m.GpuBusy):
        m.run(a, mem_reader=lambda: 90, model_loader=lambda *x: pytest.fail("model loaded"), root=root)
    assert (tmp_path / "lock").read_text() == "other job nonce=zz\n"  # never stolen


def test_views_never_see_labels(tmp_path, monkeypatch):
    m = _load_script("p3s_stepa_views")
    root = _fake_root(tmp_path)
    qs = m.load_questions(root, None)
    assert len(qs) == 2 and [q["key"] for q in qs] == ["0:a#0", "1:b#0"]
    assert all("answer" not in q and len(q["messages"]) == 1 and q["messages"][0]["role"] == "user" for q in qs)
    seen = []
    monkeypatch.setattr(m, "require_frozen_adapter", lambda p: "f" * 64)
    monkeypatch.setattr(m, "view_image", lambda r, q, v: "img.jpg")

    class DS:
        def process(self, s):
            seen.append(s)
            return s

    def loader(r, sha):
        return None, DS(), {}, lambda model, item, ids4: (np.arange(4.0), 0)

    st = m.run(Namespace(views="shift1,base", stop_at=None, n=None, out_dir=str(tmp_path / "o"), lock=str(tmp_path / "lock")),
               mem_reader=lambda: 90, model_loader=loader, root=root)
    assert set(st) == {"base", "shift1"} and not (tmp_path / "lock").exists()
    assert all(s["messages"][1]["content"] == "A" for s in seen)  # fixed placeholder, never the gold letter
    assert "A. Medium\n" in seen[-1]["messages"][0]["content"]  # shift1 rebuilt prompt (base ran first)
    out = [json.loads(x) for x in (tmp_path / "o/base.jsonl").read_text().splitlines()]
    assert set(out[0]) == {"key", "logits"}
    # resume: nothing re-run
    seen.clear()
    m.run(Namespace(views="base", stop_at=None, n=None, out_dir=str(tmp_path / "o"), lock=str(tmp_path / "lock")),
          mem_reader=lambda: 90, model_loader=loader, root=root)
    assert seen == []
