"""AUTHOR ONLY, after the user's go for ONE sealed matched ShotBench evaluation: writes results/p3m_matched/GO_<tag>, bound to the tag, the
registry's pinned model identity and the protocol hash (cgt.p3m_matched.sealed_eval.expected_go_text; a letterlogit_c5 tag also binds the
pinned blend hash and SigLIP identity). For matched_v1 the text is byte-identical to before.
Usage: uv run python scripts/p3m_matched_write_go.py --tag matched_v1"""
import argparse

from cgt.p3m_matched import sealed_eval

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    text = sealed_eval.expected_go_text(a.tag)  # refuses an unregistered / unpinned tag or a protocol hash the code does not reproduce
    p = sealed_eval.go_path(a.tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(f"wrote {p}")
