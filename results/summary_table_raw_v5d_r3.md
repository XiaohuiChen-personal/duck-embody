# Duck Embody — 12-trial benchmark results

Batch: 3 models x 4 seeds (101-104), config_hash `d30462d03c76`, freeze commit `56bd08a68d92`, last trial turn at 2026-08-03T17:03:54Z (read from the trial logs).
Manifest: `results/manifests/v5d-r3-final-prod.json` (complete); manifest SHA `17a79cc37604c55119cd25a949858bb2d947db2ae7f1e7b57fb5e19500ac16cd`, checkpoint SHA `301e24e336b2eab0ba387beb50fc16b03e6062b26622bc9a3e98588216a12c54`, parent commit `7dde4ba952fb40c5ffb29441a1895f6f8863fdcc`.

**Headline:** generated from `raw_v5d_r3` — 3 models x 4 seeds. The interpretive narrative in the default report describes the 2026-07-27 batch only and is omitted here; read the tables below plus the per-trial audits.

## Per-model aggregate (N=4 trials each)

| Metric | sonnet5 (claude-sonnet-5) | opus5 (claude-opus-5) | gpt56sol (gpt-5.6-sol) |
|---|---|---|---|
| find_kitchen SR (v2: any counter face) | 1/4 [0.00, 0.75] | 2/4 [0.00, 1.00] | 0/4 [0.00, 0.00] |
| find_kitchen SR (pre-registered point disc) | 0/4 | 2/4 | 0/4 |
| return_home SR (unrun = failure) | 0/4 [0.00, 0.00] | 2/4 [0.00, 1.00] | 0/4 [0.00, 0.00] |
| return_home SR given stage-1 success (x/k) | 0/1 | 2/2 | — |
| find_kitchen progress (mean) | 0.263 [0.000, 0.663] | 0.471 [0.024, 0.918] | 0.320 [0.103, 0.557] |
| find_kitchen progress (median) | 0.084 | 0.471 | 0.301 |
| find_kitchen SPL | 0.076 [0.000, 0.229] | 0.106 [0.000, 0.211] | 0.000 [0.000, 0.000] |
| time-to-kitchen (s) | 58.4 (no CI, n<3) (n=1/4) | 74.5 (no CI, n<3) (n=2/4) | — |
| find_kitchen turns | 40.00 [40.00, 40.00] | 33.00 [22.00, 40.00] | 34.00 [22.00, 40.00] |
| bumps / trial | 17.50 [13.50, 21.00] | 15.00 [6.00, 25.25] | 18.00 [9.00, 27.00] |
| falls / trial | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| dead-reckoning drift (m, stage 1) | 0.145 [0.072, 0.217] | 0.156 [0.107, 0.205] | 0.196 [0.102, 0.315] |
| accepted position corrections (stage 1) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.25 [0.00, 0.75] |
| map precision | 0.375 [0.000, 0.750] | 0.562 [0.125, 1.000] | 0.375 [0.000, 0.750] |
| map recall | 0.375 [0.000, 0.750] | 0.625 [0.250, 1.000] | 0.375 [0.000, 0.750] |
| edge accuracy | 1.000 (no CI, n<3) (n=1/4) | 1.000 (no CI, n<3) (n=1/4) | 0.000 (no CI, n<3) (n=2/4) |
| QA score (0-1) | 0.275 [0.000, 0.550] | 0.625 [0.550, 0.700] | 0.375 [0.175, 0.550] |
| cost (USD / trial) | 2.055 [1.641, 2.815], sum $8.22 | 3.125 [2.552, 4.206], sum $12.50 | 1.461 [1.121, 1.662], sum $5.84 |
| total turns / trial | 49.25 [40.00, 67.75], sum 197 | 38.25 [28.75, 46.00], sum 153 | 34.00 [22.00, 40.00], sum 136 |
| stage-1 end reasons | declare_done: 2, turn_cap: 2 | declare_done: 2, turn_cap: 2 | declare_done: 4 |

Notes: definitions per doc 06 §§5.3-5.4 (SPL is 0.0 on failure; time-to-kitchen defined only on success). Batch-specific commentary is omitted for a redirected results dir — see the per-trial audit files alongside the raw JSONs.

## Per-trial results

### sonnet5

| Trial | Stage-1 outcome (v2) | Progress | SPL | Path (m) | Turns | Bumps | Falls | Drift (m) | Corr. A/R | Map P | Map R | Edge acc | QA | Cost ($) | Video |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sonnet5_seed101 | timeout_turns | 0.000 | 0.000 | 4.80 | 40 | 21 | 0 | 0.081 | 0/0 | 0.00 | 0.00 | — | 0.00 | 1.707 | [sonnet5_seed101.mp4](videos_v5d_r3/sonnet5_seed101.mp4) |
| sonnet5_seed102 | timeout_turns | 0.000 | 0.000 | 9.74 | 40 | 11 | 0 | 0.196 | 0/0 | 1.00 | 1.00 | 1.00 | 0.60 | 1.709 | [sonnet5_seed102.mp4](videos_v5d_r3/sonnet5_seed102.mp4) |
| sonnet5_seed103 | success | 0.885 | 0.306 | 10.27 | 40 | 21 | 0 | 0.063 | 0/0 | 0.50 | 0.50 | — | 0.50 | 3.185 | [sonnet5_seed103.mp4](videos_v5d_r3/sonnet5_seed103.mp4) |
| sonnet5_seed104 | declared_elsewhere | 0.168 | 0.000 | 5.90 | 40 | 17 | 0 | 0.238 | 0/0 | 0.00 | 0.00 | — | 0.00 | 1.619 | [sonnet5_seed104.mp4](videos_v5d_r3/sonnet5_seed104.mp4) |

### opus5

| Trial | Stage-1 outcome (v2) | Progress | SPL | Path (m) | Turns | Bumps | Falls | Drift (m) | Corr. A/R | Map P | Map R | Edge acc | QA | Cost ($) | Video |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opus5_seed101 | timeout_turns | 0.048 | 0.000 | 6.22 | 40 | 31 | 0 | 0.100 | 0/0 | 1.00 | 1.00 | — | 0.70 | 2.543 | [opus5_seed101.mp4](videos_v5d_r3/opus5_seed101.mp4) |
| opus5_seed102 | timeout_turns | 0.000 | 0.000 | 11.51 | 40 | 17 | 0 | 0.210 | 0/0 | 0.00 | 0.00 | — | 0.60 | 2.640 | [opus5_seed102.mp4](videos_v5d_r3/opus5_seed102.mp4) |
| opus5_seed103 | success | 0.943 | 0.169 | 18.59 | 36 | 8 | 0 | 0.115 | 1/0 | 0.25 | 0.50 | — | 0.70 | 4.755 | [opus5_seed103.mp4](videos_v5d_r3/opus5_seed103.mp4) |
| opus5_seed104 | success | 0.894 | 0.254 | 7.26 | 16 | 4 | 0 | 0.200 | 2/0 | 1.00 | 1.00 | 1.00 | 0.50 | 2.561 | [opus5_seed104.mp4](videos_v5d_r3/opus5_seed104.mp4) |

### gpt56sol

| Trial | Stage-1 outcome (v2) | Progress | SPL | Path (m) | Turns | Bumps | Falls | Drift (m) | Corr. A/R | Map P | Map R | Edge acc | QA | Cost ($) | Video |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt56sol_seed101 | declared_elsewhere | 0.035 | 0.000 | 6.31 | 40 | 25 | 0 | 0.098 | 0/0 | 0.00 | 0.00 | — | 0.10 | 1.692 | [gpt56sol_seed101.mp4](videos_v5d_r3/gpt56sol_seed101.mp4) |
| gpt56sol_seed102 | declared_elsewhere | 0.307 | 0.000 | 6.31 | 40 | 29 | 0 | 0.106 | 0/0 | 1.00 | 1.00 | — | 0.40 | 1.625 | [gpt56sol_seed102.mp4](videos_v5d_r3/gpt56sol_seed102.mp4) |
| gpt56sol_seed103 | declared_elsewhere | 0.294 | 0.000 | 13.98 | 40 | 14 | 0 | 0.384 | 1/0 | 0.00 | 0.00 | 0.00 | 0.40 | 1.574 | [gpt56sol_seed103.mp4](videos_v5d_r3/gpt56sol_seed103.mp4) |
| gpt56sol_seed104 | declared_elsewhere | 0.645 | 0.000 | 6.59 | 16 | 4 | 0 | 0.196 | 0/0 | 0.50 | 0.50 | 0.00 | 0.60 | 0.953 | [gpt56sol_seed104.mp4](videos_v5d_r3/gpt56sol_seed104.mp4) |

Per-question QA scores, matched room names, visited rooms, token counts and the return_home rows are in `scores_raw_v5d_r3.json`; raw evidence is under `raw_v5d_r3/` and each video link above is derived from its trial JSON.

Generated by `scripts/build_scores.py` from `results/raw_v5d_r3/*.json` via `duck_embody.scoring` (no frozen file touched).
