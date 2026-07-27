# Duck Embody — 12-trial benchmark results

Batch: 3 models x 4 seeds (101-104), config_hash `cf29ec164676`, freeze commit `13f438d93e50`, last trial turn at 2026-07-27T09:31:34Z (read from the trial logs).

**Headline: 1/12 find_kitchen successes under criterion v2 (any counter face); 0/12 under the pre-registered point-disc criterion.** 10 trials ended in a fall (the audit-corrected decomposition, results/audit_notes.md: 5 hull-limit spin falls at |wz| = 0.5 exactly, 5 forward-step topples at |wz| 0.02-0.29); 2 ended by `declare_done`: gpt56sol_seed103 five cm from an east-wall counter face (a v2 success; `declared_elsewhere` as-run) and fable5_seed104 in the living room 1.40 m from any counter (a failure under both criteria). The scoring criterion was widened POST-BATCH (2026-07-27, owner-directed, all 12 trials re-scored together — see results/rerun_log.md): the objective text "walk to the counter" never disambiguates the two counter runs, so success is now the pre-registered 0.35 m disc UNION within 0.35 m of any kitchen-counter footprint while inside the kitchen. `return_home` never ran: the LIVE stage-2 gate used the pre-registered predicate, so the v2 success was never offered its return leg — its SR is 0/4 with the unrun stage counted a failure (doc 06 §3.2), and the conditional SR counts only offered legs (— , k=0). Differentiation between models lives in progress, map precision/recall, QA, bumps, and drift below.

Statistics: mean [95% bootstrap CI], percentile method, 10000 resamples, seed 20260726 (configs/benchmark.yaml `scoring:`); "—" = undefined, excluded from means, never coerced to 0; no CI when n_defined < 3.

## Per-model aggregate (N=4 trials each)

| Metric | fable5 (claude-fable-5) | opus5 (claude-opus-5) | gpt56sol (gpt-5.6-sol) |
|---|---|---|---|
| find_kitchen SR (v2: any counter face) | 0/4 [0.00, 0.00] | 0/4 [0.00, 0.00] | 1/4 [0.00, 0.75] |
| find_kitchen SR (pre-registered point disc) | 0/4 | 0/4 | 0/4 |
| return_home SR (unrun = failure) | 0/4 [0.00, 0.00] | 0/4 [0.00, 0.00] | 0/4 [0.00, 0.00] |
| return_home SR given stage-1 success (x/k) | — | — | — |
| find_kitchen progress (mean) | 0.066 [0.019, 0.115] | 0.151 [0.058, 0.293] | 0.218 [0.011, 0.560] |
| find_kitchen progress (median) | 0.063 | 0.094 | 0.067 |
| find_kitchen SPL | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.104 [0.000, 0.311] |
| time-to-kitchen (s) | — | — | 43.8 (no CI, n<3) (n=1/4) |
| find_kitchen turns | 8.00 [3.50, 12.50] | 12.25 [2.50, 22.00] | 13.00 [7.00, 22.25] |
| bumps / trial | 2.75 [1.00, 5.50] | 4.25 [0.50, 10.00] | 1.75 [1.00, 2.50] |
| falls / trial | 0.75 [0.25, 1.00] | 1.00 [1.00, 1.00] | 0.75 [0.25, 1.00] |
| dead-reckoning drift (m, stage 1) | 0.160 [0.056, 0.280] | 0.149 [0.041, 0.276] | 0.177 [0.029, 0.325] |
| position corrections (stage 1) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| map precision | 0.625 [0.500, 0.875] | 0.500 [0.000, 1.000] | 0.750 [0.250, 1.000] |
| map recall | 0.875 [0.625, 1.000] | 0.500 [0.000, 1.000] | 0.750 [0.250, 1.000] |
| edge accuracy | 0.000 [0.000, 0.000] (n=3/4) | 0.000 (no CI, n<3) (n=1/4) | 0.000 (no CI, n<3) (n=1/4) |
| QA score (0-1) | 0.575 [0.500, 0.650] | 0.650 [0.550, 0.750] | 0.300 [0.225, 0.375] |
| cost (USD / trial) | 1.100 [0.227, 2.085], sum $4.40 | 0.877 [0.069, 1.685], sum $3.51 | 0.429 [0.211, 0.739], sum $1.72 |
| total turns / trial | 8.00 [3.50, 12.50], sum 32 | 12.25 [2.50, 22.00], sum 49 | 13.00 [7.00, 22.25], sum 52 |
| stage-1 end reasons | declare_done: 1, fall: 3 | fall: 4 | declare_done: 1, fall: 3 |

Notes: time-to-kitchen is defined only on the published (v2) success (doc 06 §5.4). SPL is 0.0 (not —) on failure by definition (§5.3); its stage-1 oracle `l` is the shortest path to the v2 SUCCESS REGION (disc ∪ counter band, ObjectNav convention), so `l` is shorter than the old point oracle for every spawn. progress / d_initial / d_final keep the pre-registered point reference for comparability (a success can therefore show progress < 1). return_home rows beyond SR are omitted: the stage never ran, so progress = 0.0 and drift = — for all 12 by convention (§3.2). Edge accuracy is — when a trial claimed no `leads_to:` edge. Of the two `declare_done` trials, gpt56sol_seed103 is the single v2 success (0.051 m from counter_5's face, in the kitchen; `declared_elsewhere` under the pre-registered criterion) and fable5_seed104 is a failure under both criteria — consistent with the videos (rule 11: video is authoritative; no metric-vs-video disagreement found).

## Per-trial results

### fable5

| Trial | Stage-1 outcome (v2) | Progress | SPL | Path (m) | Turns | Bumps | Falls | Drift (m) | Corr. | Map P | Map R | Edge acc | QA | Cost ($) | Video |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fable5_seed101 | fall | 0.048 | 0.000 | 0.92 | 2 | 1 | 1 | 0.015 | 0 | 1.00 | 1.00 | — | 0.60 | 0.142 | [fable5_seed101.mp4](videos/fable5_seed101.mp4) |
| fable5_seed102 | fall | 0.000 | 0.000 | 5.61 | 14 | 7 | 1 | 0.336 | 0 | 0.50 | 0.50 | 0.00 | 0.50 | 2.676 | [fable5_seed102.mp4](videos/fable5_seed102.mp4) |
| fable5_seed103 | fall | 0.078 | 0.000 | 5.31 | 5 | 1 | 1 | 0.177 | 0 | 0.50 | 1.00 | 0.00 | 0.70 | 0.313 | [fable5_seed103.mp4](videos/fable5_seed103.mp4) |
| fable5_seed104 | declared_elsewhere | 0.137 | 0.000 | 4.88 | 11 | 2 | 0 | 0.110 | 0 | 0.50 | 1.00 | 0.00 | 0.50 | 1.269 | [fable5_seed104.mp4](videos/fable5_seed104.mp4) |

### opus5

| Trial | Stage-1 outcome (v2) | Progress | SPL | Path (m) | Turns | Bumps | Falls | Drift (m) | Corr. | Map P | Map R | Edge acc | QA | Cost ($) | Video |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opus5_seed101 | fall | 0.047 | 0.000 | 0.93 | 2 | 1 | 1 | 0.015 | 0 | 1.00 | 1.00 | — | 0.60 | 0.069 | [opus5_seed101.mp4](videos/opus5_seed101.mp4) |
| opus5_seed102 | fall | 0.119 | 0.000 | 13.08 | 28 | 13 | 1 | 0.168 | 0 | 0.00 | 0.00 | — | 0.50 | 1.888 | [opus5_seed102.mp4](videos/opus5_seed102.mp4) |
| opus5_seed103 | fall | 0.069 | 0.000 | 21.81 | 16 | 3 | 1 | 0.345 | 0 | 0.00 | 0.00 | 0.00 | 0.80 | 1.482 | [opus5_seed103.mp4](videos/opus5_seed103.mp4) |
| opus5_seed104 | fall | 0.368 | 0.000 | 2.12 | 3 | 0 | 1 | 0.067 | 0 | 1.00 | 1.00 | — | 0.70 | 0.070 | [opus5_seed104.mp4](videos/opus5_seed104.mp4) |

### gpt56sol

| Trial | Stage-1 outcome (v2) | Progress | SPL | Path (m) | Turns | Bumps | Falls | Drift (m) | Corr. | Map P | Map R | Edge acc | QA | Cost ($) | Video |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt56sol_seed101 | fall | 0.112 | 0.000 | 1.30 | 6 | 1 | 1 | 0.006 | 0 | 1.00 | 1.00 | — | 0.30 | 0.161 | [gpt56sol_seed101.mp4](videos/gpt56sol_seed101.mp4) |
| gpt56sol_seed102 | fall | 0.023 | 0.000 | 3.85 | 11 | 3 | 1 | 0.253 | 0 | 1.00 | 1.00 | — | 0.40 | 0.397 | [gpt56sol_seed102.mp4](videos/gpt56sol_seed102.mp4) |
| gpt56sol_seed103 | success | 0.740 | 0.415 | 7.57 | 27 | 2 | 0 | 0.396 | 0 | 0.00 | 0.00 | 0.00 | 0.30 | 0.897 | [gpt56sol_seed103.mp4](videos/gpt56sol_seed103.mp4) |
| gpt56sol_seed104 | fall | 0.000 | 0.000 | 1.90 | 8 | 1 | 1 | 0.051 | 0 | 1.00 | 1.00 | — | 0.20 | 0.262 | [gpt56sol_seed104.mp4](videos/gpt56sol_seed104.mp4) |

Per-question QA scores, matched room names, visited rooms, token counts and the return_home rows are in `results/scores.json`; raw evidence is `results/raw/<trial>.json` and `results/videos/<trial>.mp4`.

Generated by `scripts/build_scores.py` from `results/raw/*.json` via `duck_embody.scoring` (no frozen file touched).
