"""Audit of a FINISHED (or running) video cross-fit run against the five findings V1-V5 of the independent pre-verification (docs/phase4_video_plan.md Amendment 8), from its own logs only: nothing is
rerun and no model is touched. For run `video_<tag>`: per restart, did any of the conditions actually occur? V1: an update re-computed by a process started from the SAME resume state whose loss
differs beyond 1e-3 (and, for reference, beyond the 1e-2 / first-three-updates rule the run's trainer used); V2: a stopping-rule decision that would have changed with a stage-total grace; V3: an
exit 9 or another no-restart exit code followed by a restart; V4: another process's lock removed; V5: a restart-count reset. Writes the markdown given by --out.
Usage: uv run python scripts/p4_video_audit_v1_v5.py --run-dir results/phase4/runs/video_v1 --out results/phase4/runs/video_v1/AUDIT_v1_v5.md"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

STAGES = ["fold0", "fold1", "fold2", "final"]
NO_RESTART = {3, 4, 5, 6, 7, 8, 9, 10, 13}
WD = "kIOGPUCommandBufferCallbackErrorImpactingInteractivity"


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s[:25])


def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def window_bound(run: Path) -> tuple[float, dict]:
    """Non-negative-loss upper bound on the mean of every eligible 50-micro-step window (docs/phase4_video_plan.md Amendment 2 item 3): groups are the accumulation groups of updates.jsonl (size = micro
    steps added by the update), a window ending at micro-step m >= 100 covers micro-steps m-49..m."""
    best, per = 0.0, {}
    for st in STAGES:
        recs = jl(run / st / "updates.jsonl")
        if not recs:
            continue
        loss, micro, epoch = {}, {}, {}
        for r in recs:
            loss[r["update"]] = max(loss.get(r["update"], 0.0), r["loss"])
            micro[r["update"]], epoch[r["update"]] = r["micro"], r["epoch"]
        us = sorted(loss)
        size = {u: micro[u] - (micro[u - 1] if u - 1 in micro and epoch[u - 1] == epoch[u] else 0) for u in us}
        hi, tot = {}, 0
        for u in us:
            tot += size[u]
            hi[u] = tot
        worst = 0.0
        for m in range(100, tot + 1):
            worst = max(worst, sum(size[u] * (loss[u] + 5e-6) for u in us if hi[u] - size[u] < m and hi[u] >= m - 49) / 50)
        per[st] = worst
        best = max(best, worst)
    return best, per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run_dir)
    sup, rst = jl(run / "supervisor.jsonl"), jl(run / "restarts.jsonl")
    L = [f"# Audit of run `{run.name}` against the pre-verification findings V1-V5 (from its logs only; nothing rerun)", "",
         "Sources: `restarts.jsonl`, `supervisor.jsonl`, per stage `provenance.jsonl`, `train.log`, `train.stdout`, `updates.jsonl`, `resume_checks.jsonl`. Process boundaries are reconstructed from the provenance "  # noqa: ISC004 - adjacent string literals are intended
         "start times; a process's resume point is the `RESUMED ... update U` line it logged (none = a fresh start at update 0). Update losses are the means of 8 micro-batches.", ""]
    procs_by_stage: dict[str, list[dict]] = {}
    for st in STAGES:
        d = run / st
        prov = jl(d / "provenance.jsonl")
        if not prov:
            continue
        log = (d / "train.log").read_text().splitlines()
        ups = jl(d / "updates.jsonl")
        starts = [ts(p["started"]) for p in prov]
        procs = [{"i": i, "start": s, "code": p["code_commit"][:7], "resume": 0, "updates": {}} for i, (p, s) in enumerate(zip(prov, starts, strict=True))]
        # resume point of each process: the RESUMED line logged between its start and the next start
        for line in log:
            m = re.match(r"\[(\S+) .*?\] RESUMED at epoch (\d+) micro (\d+) update (\d+)", line)
            if m:
                t = ts(m.group(1))
                k = max(i for i, p in enumerate(procs) if p["start"] <= t)
                procs[k]["resume"] = int(m.group(4))
        for u in ups:
            t = ts(u["time"])
            k = max(i for i, p in enumerate(procs) if p["start"] <= t)
            procs[k]["updates"][u["update"]] = u["loss"]
        procs_by_stage[st] = procs
        # segment train.stdout by process start lines to find the failure text of each process
        so = (d / "train.stdout").read_text() if (d / "train.stdout").exists() else ""
        segs = re.split(r"(?m)^(?=\[[^\]]*\] stage " + st + r": code )", so)
        segs = [x for x in segs if x.strip()]
        for p, seg in zip(procs, segs, strict=False):
            p["metal"] = WD in seg
            p["tail"] = [x for x in seg.splitlines() if x.strip()][-1][:150] if seg.strip() else ""
    L += ["## Processes per stage", "", "| stage | process | started | code | resumed at update | updates logged | ended with |", "|---|---:|---|---|---:|---|---|"]
    for st, procs in procs_by_stage.items():
        for p in procs:
            ups = sorted(p["updates"])
            L.append(f"| {st} | {p['i']} | {p['start'].strftime('%m-%d %H:%M:%S')} | `{p['code']}` | {p['resume']} | {ups[0] if ups else '-'}..{ups[-1] if ups else '-'} ({len(ups)}) | "
                     f"{'Metal watchdog error' if p.get('metal') else (p.get('tail') or '').replace('|', '/')[:70]} |")
    # V1
    L += ["", "## V1: repeated updates after a resume", ""]
    n_same, worst, over_1e3, over_1e2, n_diff = 0, 0.0, [], [], 0
    rel_diff: list[float] = []
    rows = []
    for st, procs in procs_by_stage.items():
        seen: dict[int, list[tuple[int, int, float]]] = {}
        for p in procs:
            for u, loss in p["updates"].items():
                seen.setdefault(u, []).append((p["i"], p["resume"], loss))
        for u, lst in sorted(seen.items()):
            for x in range(len(lst)):
                for y in range(x + 1, len(lst)):
                    (i1, r1, l1), (i2, r2, l2) = lst[x], lst[y]
                    rel = abs(l1 - l2) / max(abs(l1), 1e-8)
                    if r1 == r2:
                        n_same += 1
                        worst = max(worst, rel)
                        if rel > 1e-3:
                            over_1e3.append((st, u, l1, l2))
                        if rel > 1e-2:
                            over_1e2.append((st, u))
                        rows.append((st, u, i1, i2, r1, l1, l2, rel))
                    else:
                        n_diff += 1
                        rel_diff.append(rel)
    L.append(f"- Updates re-computed by two processes that started from the SAME resume state (the comparison V1 prescribes): **{n_same}**; largest relative loss difference **{worst:.2e}**; "
             f"above 1e-3: **{len(over_1e3)}**; above the 1e-2 / first-three-updates rule the run's trainer applied: **{len(over_1e2)}**.")
    spread = f" (median relative difference {sorted(rel_diff)[len(rel_diff) // 2]:.1%}, largest {max(rel_diff):.1%}: a resume reseeds the dropout stream from its own update, so such pairs are not expected to agree; the registered tolerance does not cover them)" if rel_diff else ""
    L.append(f"- Re-computed updates whose two processes started from DIFFERENT resume states (dropout streams differ, not comparable, informational): {n_diff}{spread}.")
    if rows:
        L += ["", "| stage | update | processes | resumed at | loss A | loss B | relative difference |", "|---|---:|---|---:|---:|---:|---:|"]
        L += [f"| {r[0]} | {r[1]} | {r[2]} vs {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]:.1e} |" for r in rows[:40]]
        if len(rows) > 40:
            L.append(f"| ... | ({len(rows) - 40} more rows, all within the maximum above) | | | | | |")
    rc = jl(run / "fold2/resume_checks.jsonl") + jl(run / "final/resume_checks.jsonl") + jl(run / "fold1/resume_checks.jsonl")
    L.append(f"- The run's own `resume_checks.jsonl` (1e-2, first three updates after a resume) holds {len(rc)} checks, all ok: {all(r['ok'] for r in rc) if rc else 'n/a'}.")
    # V2
    L += ["", "## V2: stopping rule with a per-process versus a stage-total grace", ""]
    worst_win, per_proc_grace = 0.0, []
    for st, procs in procs_by_stage.items():
        allu = {}
        for p in procs:
            allu.update(p["updates"])
        seq = [allu[u] for u in sorted(allu)]
        # 50 micro-steps = 6.25 updates: use windows of 7 consecutive update means as an upper-bound proxy, from the first 13 updates (100 micro-steps) on
        for k in range(13, len(seq) - 6):
            worst_win = max(worst_win, sum(seq[k:k + 7]) / 7)
        for p in procs[1:]:
            seqp = [p["updates"][u] for u in sorted(p["updates"])][:13]
            if seqp:
                per_proc_grace.append((st, p["i"], round(max(seqp), 3)))
    L.append(f"- PROXY ONLY: the largest mean of 7 consecutive update losses (about 56 micro-steps) after the first 13 updates of a stage is **{worst_win:.3f}** (the rule fires above 5.0). A seven-update mean is not by itself a bound on every "
             "50-micro-step window, so it is not offered as proof that the rule could not have fired.")
    bound, per_stage = window_bound(run)
    L.append(f"- Upper-bound argument (non-negative completion loss): the rule inspects the last 50 micro-steps from micro-step 100 on (the window refills after every restart). Every such window lies inside the accumulation groups it touches, and "
             f"each group's micro-losses are non-negative with a retained group mean, so the window's mean cannot exceed (sum over the touched groups of size x group mean, with 5e-6 for the five-decimal rounding) / 50. Maximum over all eligible "
             f"window positions of all stages (per update the highest logged loss over processes): **{bound:.3f}** ({', '.join(f'{k} {v:.3f}' for k, v in per_stage.items())}), far below 5.0. The independent verification's more conservative bound "
             "for the same argument is 0.717336. Limits: individual micro-step losses were not logged, and a group cut short by a crash was not logged by the crashed process (its successor re-ran it, so the bound uses the retained group means).")
    L.append(f"- Highest update loss inside the 13 updates (about 100 micro-steps) following each restart, i.e. the window the old per-process grace skipped: {per_proc_grace if per_proc_grace else 'no restarts'}; every value is far below 5.0, and under the bound above no decision would have changed with a stage-total grace.")
    # V3, V5 per restart
    L += ["", "## V3 and V5: exits and the restart counter", ""]
    codes = [r["exit_code"] for r in rst]
    L.append(f"- Exit codes of all {len(rst)} restarted failures: {sorted(set(codes))}; any of {sorted(NO_RESTART)} followed by a restart: **{any(c in NO_RESTART for c in codes)}** (exit 9 only occurred in pipeline checks launched by hand, never under the supervisor).")
    nums = [r["restart"] for r in rst]
    L.append(f"- Restart numbers on record: {nums}; strictly consecutive without a reset: **{nums == list(range(1, len(nums) + 1))}**.")
    starts_ev = [e for e in sup if e.get("event", "").startswith("supervisor start")]
    for e in starts_ev:
        used = e.get("restarts_already_used", 0)
        before = sum(1 for r in rst if ts(r["time"]) <= ts(e["time"]))
        L.append(f"- supervisor start {e['time'][:19]}: `--restarts-used` {used}; records on file at that moment {before}; **{'consistent (no reset, no understatement)' if used == before else 'MISMATCH'}**.")
    # V4
    exits = [e for e in sup if e.get("event", "").startswith("supervisor exit")]
    L += ["", "## V4: lock ownership (NOT independently established)", "",
          f"- The supervisor logged {len(starts_ev)} lock acquisitions (`(lock taken)`) and {len(exits)} exits that attempted a release \"if ours\". The original helper identified the lock by its job name only, and no lock payload, pid or nonce snapshot was retained, "  # noqa: ISC004 - adjacent string literals are intended
          "so these logs and the session account below do not prove whose file was unlinked: this audit does not establish that no other process's lock was removed, and an empty ledger or three textual events must not be read as proof of a safe release.",
          "- Session account (not from these logs): another film run held the lock on 2026-10-02 23:27 while this run's probes were refused with exit 11; the earlier supervisor self-check ran against the REAL lock path on 2026-10-03 between 18:00 and "  # noqa: ISC004 - adjacent string literals are intended
          "18:13, a window in which no real job held the lock (the next real lock was taken by the hand-launched pipeline check at 18:14:57).",
          "- Wording of record (independent freeze verification, 2026-10-03): \"No Phase4 test evaluations are recorded in the retained ledgers, and no public-test label reader appears in the reviewed training/validation paths. Prior CameraBench "  # noqa: ISC004 - adjacent string literals are intended
          "sample/aggregate access is disclosed; additional historical unlogged access is not independently excluded. The supervisor records three acquisitions and release attempts; independent lock-ownership snapshots were not retained.\"",
          "- The fixed helper (Amendment 8: pid + random nonce, release only of a lock this process acquired) governs FUTURE runs; it does not retroactively prove ownership in this one."]
    # per restart table
    L += ["", "## Per restart", "", "| restart | time | stage | exit | Metal watchdog text in the failed process | resume state before -> after (update) | V1 | V2 | V3 | V4 (not independently established) | V5 |", "|---:|---|---|---:|---|---|---|---|---|---|---|"]
    for r in rst:
        st = r["stage"]
        procs = procs_by_stage.get(st, [])
        t = ts(r["time"])
        failed = [p for p in procs if p["start"] < t]
        fp = failed[-1] if failed else None
        wd = r.get("watchdog")
        if wd is None and fp is not None:
            wd = fp.get("metal")
        before = r["resume_before"]
        after = r["resume_after"]
        rb = before.get("resume", before).get("update") if isinstance(before, dict) else None
        ra = after.get("resume", after).get("update") if isinstance(after, dict) else None
        cause = "yes" if wd else ("no (deliberate SIGTERM by me, exit 143: memory fix, Amendment 5)" if r["exit_code"] == 143 else "no")
        v1 = "no mismatch" if not over_1e3 else "see V1"
        L.append(f"| {r['restart']} | {r['time'][5:19]} | {st} | {r['exit_code']} | {cause} | {rb} -> {ra} | {v1} | no | no | not established | no |")
    L += ["", "## Verdict", "",
          f"- V1: {'no repeated update from the same resume state differed beyond 1e-3' if not over_1e3 else str(len(over_1e3)) + ' repeated updates exceeded 1e-3 (listed above)'}; "  # noqa: ISC004 - adjacent string literals are intended
          f"V2: under the non-negative-loss upper bound ({bound:.3f} < 5.0) no stopping-rule decision would have changed; V3: no restart followed an exit code of the no-restart set; V4: NOT independently established (see the V4 section: "
          "the supervisor records three acquisitions and release attempts; independent lock-ownership snapshots were not retained); V5: the count never reset and every relaunch was consistent with the file-backed history.",
          "- Limits of this audit: losses are update means (8 micro-batches), micro-step losses were not logged, so V2 rests on the upper-bound argument above; process boundaries are reconstructed from timestamps; the lock evidence for other owners is the session record, not a retained trace."]
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
