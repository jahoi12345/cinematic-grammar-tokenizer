"""Determinism check between two film runs of the same method/adapter (different code SHA): per attribute and keyframe, primary-label agreement and the maximum
absolute score difference. Usage: p3c_compare_runs.py <run_dir_old> <run_dir_new> [out.md]. CPU only."""
import json
import sys
from pathlib import Path

import numpy as np

from cgt.p3c_tokens.vocab import ATTRS

KFS = ("kf_mid", "kf_start", "kf_end")


def load(d: Path, kf: str) -> dict[int, dict]:
    return {r["shot_id"]: r for r in map(json.loads, (d / f"scores_{kf}.jsonl").read_text().splitlines())}


def main() -> None:
    old, new = Path(sys.argv[1]), Path(sys.argv[2])
    mo, mn = json.loads((old / "manifest.json").read_text()), json.loads((new / "manifest.json").read_text())
    L = ["# Determinism check: first film run vs re-run on the final code", "",
         f"- old run `{mo['key']}`: code `{mo['code_sha'][:10]}`; new run `{mn['key']}`: code `{mn['code_sha'][:10]}`; same method `{mo['method']}`/`{mn['method']}`, adapter `{mo['adapter_sha256'][:10]}`/`{mn['adapter_sha256'][:10]}`.",
         f"- new run state: `{mn.get('state')}`, coverage {mn.get('coverage')}.", "",
         "| attribute | keyframe | shots | primary label agrees | disagreeing shots | max abs score diff | mean abs score diff |", "|---|---|---|---|---|---|---|"]
    tot_dis, gmax = 0, 0.0
    for a in ATTRS:
        for kf in KFS:
            o, n = load(old, kf), load(new, kf)
            assert set(o) == set(n)
            ids = sorted(o)
            diffs = [np.abs(np.array(o[i][a]) - np.array(n[i][a])) for i in ids]
            dis = sum(int(np.argmax(o[i][a]) != np.argmax(n[i][a])) for i in ids)
            tot_dis += dis
            mx = float(max(d.max() for d in diffs))
            gmax = max(gmax, mx)
            L.append(f"| {a} | {kf} | {len(ids)} | {100 * (1 - dis / len(ids)):.2f}% | {dis} | {mx:.6g} | {float(np.mean([d.mean() for d in diffs])):.6g} |")
    L += ["", f"Total primary-label disagreements: {tot_dis} of {len(ATTRS) * len(KFS) * len(o)}; global max absolute score difference {gmax:.6g}."]
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    text = "\n".join(L) + "\n"
    if out:
        out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
