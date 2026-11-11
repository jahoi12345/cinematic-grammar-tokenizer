# Write-up figures: Night of the Living Dead

Produced by `scripts/writeup_figures.py` (plan: `docs/writeup/figures_plan.md`). Night of the Living Dead is public domain
(`configs/films.yaml`). No figure contains a film frame or keyframe; no owned film and no benchmark result is drawn. Regenerate with
`PYTHONPATH=src python scripts/writeup_figures.py --film night_of_the_living_dead --tokens data/processed/night_of_the_living_dead_tokens_v1.parquet --expect-sha ee2a20921147fb1bc1ea4fdf2a50d8f2f47fed43a8a4eca9b1ddf3530990c135`.

## Inputs
- token table: `night_of_the_living_dead_tokens_v1.parquet` sha256 `ee2a20921147fb1bc1ea4fdf2a50d8f2f47fed43a8a4eca9b1ddf3530990c135`
- visual arc CSV: `visual_arc_night_of_the_living_dead.csv` sha256 `46422a7a5d8419cd88a30cd107caf20d906196e10f709ebfa287037c69291401`
- similar shots CSV: `similar_shots_night_of_the_living_dead.csv` sha256 `7c47088e7c096230158c2c3add05edc81a2274944c97754f7fee4563e9ba76bc`
- token-row sample: `notld_token_rows_sample.md` sha256 `74d7dc89569b7eac04f3eb34ba5ae283c7bcec280bb33103e32926657b2534cf`

General caveat for every figure: shot boundaries are automatic (heuristic cut detector); model slots are predictions and heuristic slots are
unverified, and their accuracy on this film is unmeasured. Everything is descriptive and within this one film.

## Figure 1. Shot-length distribution
File: `night_of_the_living_dead_fig1_shot_lengths.png` (28 KB). Inputs: token table (`len_s`).

Caption: Distribution of the 1,077 automatically detected shot lengths of Night of the Living Dead (log-spaced bins, log seconds axis). Dashed line:
median 3.0 s (interquartile range 1.7-6.1 s).

Caveats: lengths depend on the cut detector (threshold fixed in `configs/films.yaml`, hand-checked on 50 cuts); missed or false cuts move
shots between bins.

## Figure 2. Visual arc
File: `night_of_the_living_dead_fig2_visual_arc.png` (215 KB). Inputs: visual arc CSV.

Caption: Night of the Living Dead in 20 five-minute windows: shot starts per minute, median shot length (band: within-window IQR),
close-shot share, mean shot size (0 = ECU ... 6 = EWS), low-key share, mean people count (capped at 4), multi-person share and dialogue share
(bands: 95% bootstrap intervals over the window's shots). Blue: model predictions; orange: heuristics. Hollow markers: windows with fewer
than 10 shots (the last window is partial).
All series: "model predictions / heuristics of unmeasured accuracy; within-film shapes only".

Caveats: intervals describe resampling of the window's shots, not label accuracy; shot starts per minute has no interval and counts the
first shot of the film; dialogue share is from the audio-corrected table (). No narrative, causal or cross-film claim.

## Figure 3. "Shots like this": time gap to the top-1 similar shot
File: `night_of_the_living_dead_fig3_similar_gap.png` (53 KB). Inputs: similar shots CSV (rank-1 rows), token table (`start_s`, for the reference).

Caption: For each of the 1,077 shots, the time gap to its most similar other shot by token similarity (solid vermilion line), against the gap of
all 1,154,550 eligible candidate pairs (ordered pairs more than two shots apart; dashed grey line), as the share per
4-minute bin. Medians 17.3 min (top-1) and 28.6 min (all pairs). Gaps of at most 1 min:
17.5% of top-1 partners vs 2.2% of all pairs.

Caveats: similarity is computed on predicted / heuristic tokens with equal slot weights (a choice, docs/similar_shots_spec.md); a high
score means the pipeline wrote similar tokens, not that the shots look alike, and a long-range pair is not evidence of a deliberate
callback. The reference is a scale reference only; no test.

## Figure 4. Token-row anatomy
File: `night_of_the_living_dead_fig4_token_row_anatomy.png` (120 KB). Inputs: token-row sample (shot 538), checked equal to the token table's row.

Caption: One real token row of Night of the Living Dead (shot 538, the median-position row of the deterministic sample), with each of its
13 slots labelled as a model prediction or a heuristic. `*` marks a value on which the three keyframes disagree; `?` a
missing value.

Caveats: the row was selected by position, not by content; its values are not checked against human labels. Text-only: no image of the shot.
