"""Generation + parsing per question and the matched aggregate (docs/p3m_matched_protocol.md section 2). Results stay in memory; the caller
decides what may be written (the sealed mode writes nothing before the ledger entry is recorded)."""
from cgt.p3m_matched.parser import build_choices, parse_answer

ATTRS = ["size", "framing", "angle", "lens", "light_type", "light_cond", "composition"]


def run_questions(engine, questions: list[dict], variant: str, progress=None) -> list[dict]:
    """`questions`: {id, attribute, image, question, options (stored order), answer}. `progress(done, total)` gets COUNTS ONLY."""
    out, total = [], len(questions)
    for q in questions:
        g = engine.answer(q["image"], q["question"], q["options"], variant)
        letter, method = parse_answer(g["text"], q["options"])
        out.append({"id": q["id"], "attribute": q["attribute"], "text": g["text"], "letter": letter, "method": method,
                    "n_choices": len(build_choices(q["options"])), "generation_tokens": g["generation_tokens"],
                    "gold": q["answer"], "correct": letter is not None and letter == q["answer"]})
        if progress is not None:
            progress(len(out), total)
    return out


def summarize(results: list[dict]) -> dict:
    """Per attribute: n, correct, acc (invalid = wrong), n_invalid, invalid_rate, n_z, by_method, expected_fallback_acc (adds the official
    random fallback's expected credit 1/(n_choices+1) per invalid answer). Means: unweighted over the attributes present, and micro."""
    per = {}
    for a in ATTRS:
        r = [x for x in results if x["attribute"] == a]
        if not r:
            continue
        n = len(r)
        correct = sum(bool(x["correct"]) for x in r)
        inv = [x for x in r if x["letter"] is None]
        exp = correct + sum(1.0 / (x["n_choices"] + 1) for x in inv)
        methods: dict[str, int] = {}
        for x in r:
            methods[x["method"]] = methods.get(x["method"], 0) + 1
        per[a] = {"n": n, "correct": correct, "acc": correct / n, "n_invalid": len(inv), "invalid_rate": len(inv) / n,
                  "n_z": sum(x["letter"] == "Z" for x in r), "by_method": methods, "expected_fallback_acc": exp / n}
    vals = list(per.values())
    n_all = sum(v["n"] for v in vals)
    return {"per_attribute": per, "n": n_all,
            "mean_acc": sum(v["acc"] for v in vals) / len(vals) if vals else None,
            "micro_acc": sum(v["correct"] for v in vals) / n_all if n_all else None,
            "mean_expected_fallback_acc": sum(v["expected_fallback_acc"] for v in vals) / len(vals) if vals else None,
            "invalid_rate": sum(v["n_invalid"] for v in vals) / n_all if n_all else None}
