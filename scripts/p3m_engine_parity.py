"""Diagnostic (validation only, never ShotBench): does the matched evaluator's generation path reproduce the Phase 3B letter-logit predictions
when it is fed the Phase 3B TRAINING prompt (no system message, ShotQA user turn unchanged)? Greedy generation's first token is the argmax over
the whole vocabulary, the letter-logit scorer's argmax is over A-D only, so agreement should be ~100%. A deterministic subset (--n, same seeded
permutation as scripts/p3b_eval_val.py) of the 5,826 film-held-out validation questions. Same refusals as scripts/p3m_matched_val.py
(frozen adapter hash, memory >= 25%, GPU lock). Writes results/p3m_matched/engine_parity_n<N>.json (counts only).
Usage: uv run python scripts/p3m_engine_parity.py --n 500 --data-root PATH --shotqa-root PATH"""
import argparse
import importlib.util
import json
from pathlib import Path

from cgt.p3b_vlm.images import shrunk_path
from cgt.p3b_vlm.runstatus import now_iso
from cgt.p3m_matched.parser import parse_answer

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p3m_matched_val", ROOT / "scripts/p3m_matched_val.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--data-root", default=str(ROOT))
    ap.add_argument("--shotqa-root", default=str(ROOT / "data/external/shotqa"))
    a = ap.parse_args()
    data_root = Path(a.data_root)
    e = V.require_val_tag("matched_v1", data_root)
    V.memory_guard()
    from cgt.pipeline.gpulock import GpuLock
    with GpuLock("p3m_engine_parity"):
        val = [json.loads(x) for x in (data_root / "data/cache/p3b/val.jsonl").read_text().splitlines()]
        qs = V.load_val_questions(data_root, a.n, Path(a.shotqa_root))
        ll = V.letter_logit_preds(data_root)
        from cgt.p3m_matched.engine import MlxEngine, load_mlx_model
        eng = MlxEngine(*load_mlx_model(e))
        agree, invalid, n = 0, 0, 0
        for q in qs:
            s = val[int(q["key"].split(":")[0])]
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": s["messages"][0]["content"]}]}]
            g = eng.generate_messages(shrunk_path(q["image"], data_root / "data/cache/p3b/images_147456_32"), msgs)
            letter, _ = parse_answer(g["text"], q["options"])
            n += 1
            invalid += letter is None
            agree += letter == ll[q["key"]]
        out = {"time": now_iso(), "n": n, "agreement_with_letter_logit": agree / n, "invalid": invalid,
               "prompt": "Phase 3B training prompt (no system message, ShotQA user turn incl. trailing ' \\n')", "adapter_sha256": e["identity_sha256"]}
        (ROOT / f"results/p3m_matched/engine_parity_n{a.n}.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out))


if __name__ == "__main__":
    main()
