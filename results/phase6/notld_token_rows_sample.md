# NOTLD token rows: three real samples from the 4B table (refreshed 2026-10-22)

Three rows from the live NOTLD v1 token table, whose static slots now come from the Step C Qwen3-VL-4B single pass (pipeline version p3c-v2-4b; swapped in 2026-10-22).
Table `data/processed/night_of_the_living_dead_tokens_v1.parquet` sha256 `ee2a20921147fb1bc1ea4fdf2a50d8f2f47fed43a8a4eca9b1ddf3530990c135`;
text file `night_of_the_living_dead_tokens_v1.txt` sha256 `a41ea0355d6378e857d50064dceddc27c719e2f6ea332437d85699921dc7e585`. Night of the Living Dead (1968) is public domain.

Previous version of this file (2026-10-14/15, T12-A): the same three shots from the accepted 2B table (parquet `0e818c9d48546fc087609c539e9b9dd007b74100fd58b467c42f1350eaee9fb7`, now kept as `*_pre_4b`; the sealed gold scorer stays pinned to it). The switch changed only model slots of these rows (cuts, heuristic slots and movement are unchanged). Value changes: shot 269 light type daylight -> artificial; shot 538 light daylight/soft -> artificial/hard; shot 807 angle high -> low and framing single -> insert. Several `*` (keyframe-disagreement) marks also moved. The 2B rows are in the git history of this file. A change is not an improvement: accuracy on this film is unmeasured for both models.

**These rows are predictions of unmeasured accuracy.** Model slots (size, lens, angle, framing, light, comp, move) and heuristic slots (key, DoF, people, sound, len, palette) were not checked against human labels on this film. A `*` marks a value where the three keyframes disagree (a near-tie).

Selection is deterministic: the rows at positions `int(q * 1077)` for q = 0.25, 0.5, 0.75 of the 1077-shot table (in shot order), not chosen by content.

| q | shot id | start (s) | start (mm:ss) |
|---|---|---|---|
| 0.25 | 269 | 1334.47 | 22:14.5 |
| 0.50 | 538 | 3059.99 | 51:00.0 |
| 0.75 | 807 | 4482.28 | 74:42.3 |

Rows exactly as in the text file (`<shot id> <start mmm:ss.s> <token row>`):

```
00269 022:14.5 [size=MCU | lens=ultra-wide | angle=low | framing=single | light=artificial*/hard* | comp=right-heavy | key=mid-key | DoF=deep | people=1 | sound=score | len=4-8s | palette=g21 | move=static]
00538 051:00.0 [size=MCU | lens=medium | angle=low | framing=single | light=artificial*/hard* | comp=center* | key=mid-key | DoF=medium | people=1 | sound=ambient | len=2-4s | palette=g20 | move=push-in+boom-up+tilt-down+move-left]
00807 074:42.3 [size=CU* | lens=long* | angle=low* | framing=insert* | light=artificial/high-contrast | comp=right-heavy* | key=low-key | DoF=shallow | people=0 | sound=ambient | len=2-4s | palette=g21 | move=pull-out+boom-up+tilt-up]
```

Timestamp note: shot 538 starts at 3059.993 s and prints as `051:00.0` since the fix in `cgt.tokens.build.text_lines` (commit `6f2ea99`), which rounds the total seconds before the minute split (it once printed `050:60.0`).

The rows are checked against the table by `tests/test_writeup_4b.py` (when the local table is present) and by `scripts/writeup_figures.py` (shot 538).
