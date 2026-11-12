# Similar shots: definition

This definition was fixed before any similarity code was written and before any similarity was computed on a film. Implementation: `src/cgt/tokens/similar.py`; tests: `tests/test_similar_core.py` and `tests/test_similar_demo.py`.

## What this is, and what it is not
For one shot, rank the OTHER shots of the SAME film by how similar their v1 token rows are (configs/tokens.yaml, 13 slots). Every slot value is a MODEL PREDICTION
(size, lens, angle, framing, light, comp, move) or an unverified Phase 1/2 heuristic (key, DoF, people, sound, len, palette) whose accuracy on the films is UNMEASURED (no labels).
"Similar" therefore means "the pipeline wrote similar tokens", not "the shots look alike" and not "the film-makers made a callback". Nothing here is ground truth; no accuracy,
no significance and no intent are claimed. Only within-film comparisons (no cross-film similarity).

## Inputs
- One film's v1 token table (read only). Columns used: `shot_id`, `start_s`, `size`, `lens`, `angle`, `framing`, `light_type`, `light_cond`,
  `composition`, `<attr>_consistent` for those 7, `key`, `dof`, `people`, `sound`, `len_bin` (or `len_s`), `palette_global`, `move_present`, `move_atoms`, `token_row`.
- The frozen global palette centroids, loaded read only and digest-checked. They are derived from ShotQA and are not published; only distances between them are used.
  Without them (as in the public demo) the palette slot falls back to exact id match and every output says `palette_mode=exact_id`.

## Slot similarity s_k(a, b) in [0, 1] (13 slots, equal weights)
| slot | kind | s_k |
|---|---|---|
| size | ordinal, schema order ECU..EWS (7) | 1 - abs(i - j) / 6 |
| lens | ordinal, schema order long, medium, wide, ultra-wide (4) | 1 - abs(i - j) / 3 |
| angle | categorical | 1 if equal else 0 |
| framing | categorical | 1 if equal else 0 |
| light | two halves: light_type and light_cond, each categorical | 0.5 * [type equal] + 0.5 * [cond equal] (the `*` factor applies per half) |
| comp | categorical (primary composition label) | 1 if equal else 0 |
| key | ordinal low-key, mid-key, high-key (3) | 1 - abs(i - j) / 2 |
| DoF | ordinal shallow, medium, deep (3) | 1 - abs(i - j) / 2 |
| people | count, displayed 0..3, 4+ (5 ordered levels; raw counts >= 4 cap at 4) | 1 - abs(i - j) / 4 |
| sound | categorical (score, dialogue, ambient, silence) | 1 if equal else 0 |
| len | ordinal over the 7 fixed log bins | 1 - abs(i - j) / 6 |
| palette | global palette id g00..g31 | 1 - D[g_a, g_b] / max(D), D = Euclidean distance between the two frozen 15-d LAB centroids, max over all centroid pairs |
| move | unordered atom set | Jaccard: size of intersection / size of union |

Size, lens, key, DoF, len and people have a natural order (schema or count), so a one-step difference is graded; palette ids are arbitrary labels of points in LAB space,
so their centroid distance is used. Weights are equal (1 per slot); no slot is up- or down-weighted because there are no labels to tune on.

- Low-confidence values: a `*` in the row (the three keyframes disagree on that attribute, `<attr>_consistent` is False) applies to size, lens, angle, framing, light_type,
  light_cond and composition. The slot similarity is multiplied by c = c_a * c_b with c = 0.5 for a starred value and 1 otherwise (one side starred: half;
  both starred: a quarter). The weight in the denominator is NOT reduced: a starred match is half the evidence of a clean one.
- Missing values: `?` on either side gives s_k = 0 for that slot (counted in the denominator).
- Movement absent for the film (`move_present` False, slot `n/a`): the move slot is dropped for every pair of that film (12 slots in the denominator); outputs say so.

Score: S(a, b) = sum_k s_k(a, b) / K with K = 13 (or 12 without movement); S is in [0, 1] and symmetric. S is rounded to 12 decimals before ranking, so mathematically
equal scores reached through different slot contributions tie exactly. The top-k relation is NOT symmetric (b in a's top-5 does not imply a in b's top-5).

## Candidates, exclusions, ranking, ties
- Shots are ordered by `shot_id` (= film order); position p(a) is the index in that order.
- Candidates for shot a: every shot b of the same film with abs(p(b) - p(a)) > 2. The shot itself and its two neighbours on each side are excluded, so a "callback" is never an adjacent shot.
- Ranking: S descending; ties broken by the smaller `shot_id` (deterministic; this favours earlier shots among exact ties, so the tie count is reported).
- k = 5 in the demo; the descriptive analysis uses top-1.
- Determinism: same table + same centroids -> identical output (no randomness anywhere).

## Descriptive analysis (Night of the Living Dead, public domain)
`scripts/similar_shots.py` reports, for each shot, its top-1 similar shot and the time gap between them; the gap distribution against all eligible candidate pairs
(a descriptive reference, no test, no p-value); the 10 strongest long-range pairs (gap > 10 minutes); the number of tied top-1s and the share of mutual top-1 pairs.
It claims nothing beyond description: these are similarities between predicted tokens of unmeasured accuracy, a long-range pair is not evidence of a deliberate
visual callback, and equal weights and the slot rules are choices, not findings.

## Demo ("Shots like this")
Each shot in the explorer has a "Shots like this" panel with the top-5 of the same film: rank, shot id, start time, S, the mid keyframe and the token row, under a caption
saying the tokens are predicted, their accuracy is unmeasured, the result is not ground truth and neighbours are excluded. The panel only uses the already loaded table of the
selected film.
