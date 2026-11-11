# Step C selection (rule fixed in docs/plans/stepc_qwen3vl4b.md section 5)

Primary: unweighted 7-attribute mean on the dedup_v2 ShotQA film-held-out validation set (5822 questions).

| model | primary | size | lens | angle | framing | light_type | light_cond | composition | eligible | reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| frozen 2B 4c030967 (baseline) | 81.16 | 85.62 | 67.44 | 84.54 | 87.25 | 84.85 | 74.14 | 84.27 | - | - |
| 4B LoRA 50% | 80.57 | 84.48 | 64.39 | 84.07 | 86.89 | 85.99 | 73.65 | 84.48 | no | primary 80.57 < baseline 81.16 + 0.5; lens 64.39 dropped more than 2.0 below 67.44 |
| 4B LoRA 100% | 82.49 | 86.64 | 67.56 | 86.42 | 88.67 | 86.16 | 76.72 | 85.28 | yes | - |

Missing checkpoints (dropped or not produced): none

**Winner: 4B LoRA 100%**, adapter sha256 `aba568837c815b16e877bc0f36a4d7435358abe9e1d88c4b7c99eccf0e61ae39` (tie set [100]).

Gain over the frozen 2B: +1.33 points; film-clustered bootstrap 95% interval [+0.46, +2.18] (71 films, 2000 draws, seed 0; descriptive, not a test).
