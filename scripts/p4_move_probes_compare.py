"""Comparison of a rebuilt probe result with the committed one under the tolerance declared in docs/phase4_move_probes.md (addendum 2026-10-05): every numeric statistic within 0.001 (the rounding unit of the report),
every band, group size, count and string identical. Writes a hash-bound evidence note. Usage: uv run python scripts/p4_move_probes_compare.py <committed.json> <rebuilt.json> <out.md>"""
import hashlib
import json
import math
import sys
from pathlib import Path

TOL = 0.001


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out["mismatch"].append((path, f"keys differ: {sorted(set(a) ^ set(b))}"))
        for k in a.keys() & b.keys():
            walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["mismatch"].append((path, f"length {len(a)} vs {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        out["exact"] += 1
        if a != b:
            out["mismatch"].append((path, f"{a!r} vs {b!r}"))
    elif isinstance(a, int) and isinstance(b, int):
        out["exact"] += 1  # counts and sizes are identical
        if a != b:
            out["mismatch"].append((path, f"{a} vs {b}"))
    else:
        out["numeric"] += 1
        if math.isnan(a) and math.isnan(b):
            return
        d = abs(a - b)
        out["max_abs"] = max(out["max_abs"], d)
        if not d <= TOL + 1e-12:
            out["mismatch"].append((path, f"{a} vs {b} (|diff| {d:.4f})"))


def main() -> None:
    old_p, new_p, out_p = (Path(x) for x in sys.argv[1:4])
    a, b = json.loads(old_p.read_text()), json.loads(new_p.read_text())
    out = {"mismatch": [], "exact": 0, "numeric": 0, "max_abs": 0.0}
    walk(a, b, "", out)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    ok = not out["mismatch"]
    L = ["# Cold-cache rebuild of the NOTLD probes vs the committed result (`35aada6`)", "",
         f"Tolerance (declared before the rebuild): every numeric statistic within {TOL}, every band, count, group size and string identical.",
         f"- committed `{old_p.name}` sha256 `{sha(old_p)[:16]}`; rebuilt sha256 `{sha(new_p)[:16]}`",
         f"- compared: {out['numeric']} numeric values (maximum absolute difference **{out['max_abs']:.6f}**) and {out['exact']} exact values (bands, counts, sizes, strings, flags)",
         f"- mismatches: **{len(out['mismatch'])}**; result: **{'REPRODUCED' if ok else 'NOT REPRODUCED'}**"]
    L += [f"  - `{p}`: {m}" for p, m in out["mismatch"][:40]]
    out_p.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
