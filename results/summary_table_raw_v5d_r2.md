# Duck Embody — 12-trial benchmark results

Batch: 3 models x 4 seeds (101-104), config_hash `0e9017a84c06`, freeze commit `84af3f8089a8`, last trial turn at 2026-07-30T11:27:07Z (read from the trial logs).

**Headline:** generated from `raw_v5d_r2` — 3 models x 4 seeds. The interpretive narrative in the default report describes the 2026-07-27 batch only and is omitted here; read the tables below plus the per-trial audits.

## Per-model aggregate (N=4 trials each)

| Metric | sonnet5 (claude-sonnet-5) | opus5 (claude-opus-5) | gpt56sol (gpt-5.6-sol) |
|---|---|---|---|
| find_kitchen SR (v2: any counter face) | 0/4 [0.00, 0.00] | 2/4 [0.00, 1.00] | 0/4 [0.00, 0.00] |
| find_kitchen SR (pre-registered point disc) | 0/4 | 1/4 | 0/4 |
| return_home SR (unrun = failure) | 0/4 [0.00, 0.00] | 1/4 [0.00, 0.75] | 0/4 [0.00, 0.00] |
| return_home SR given stage-1 success (x/k) | — | 1/1 | — |
| find_kitchen progress (mean) | 0.353 [0.089, 0.618] | 0.517 [0.199, 0.834] | 0.221 [0.000, 0.457] |
| find_kitchen progress (median) | 0.316 | 0.548 | 0.137 |
| find_kitchen SPL | 0.000 [0.000, 0.000] | 0.121 [0.000, 0.243] | 0.000 [0.000, 0.000] |
| time-to-kitchen (s) | — | 46.1 (no CI, n<3) (n=2/4) | — |
| find_kitchen turns | 39.75 [39.25, 40.00] | 30.50 [22.50, 38.50] | 36.00 [28.00, 40.00] |
| bumps / trial | 7.00 [5.00, 8.50] | 13.75 [6.50, 21.25] | 10.75 [6.50, 15.00] |
| falls / trial | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| dead-reckoning drift (m, stage 1) | 0.576 [0.061, 1.143] | 0.147 [0.069, 0.230] | 0.337 [0.102, 0.611] |
| position corrections (stage 1) | 1.75 [0.50, 3.25] | 0.25 [0.00, 0.75] | 1.00 [0.25, 1.75] |
| map precision | 0.208 [0.000, 0.417] | 0.375 [0.000, 0.750] | 0.625 [0.250, 1.000] |
| map recall | 0.250 [0.000, 0.500] | 0.375 [0.000, 0.750] | 0.500 [0.125, 0.875] |
| edge accuracy | 0.000 [0.000, 0.000] (n=3/4) | 0.333 [0.000, 1.000] (n=3/4) | 0.000 [0.000, 0.000] (n=3/4) |
| QA score (0-1) | 0.350 [0.300, 0.450] | 0.600 [0.475, 0.700] | 0.275 [0.100, 0.450] |
| cost (USD / trial); GPT lower bound | 1.321 [1.184, 1.464], sum $5.28 | 2.913 [1.787, 4.038], sum $11.65 | 0.870 [0.667, 1.034], sum $3.48 |
| total turns / trial | 39.75 [39.25, 40.00], sum 159 | 32.75 [26.50, 38.50], sum 131 | 36.00 [28.00, 40.00], sum 144 |
| stage-1 end reasons | declare_done: 3, turn_cap: 1 | declare_done: 4 | declare_done: 3, turn_cap: 1 |

**v5d_r2 GPT cost correction.** Raw trial JSON is unchanged. The GPT cost cells above are lower bounds computed from total input, recoverable cache reads, and output at the 2026-08-02 GPT-5.6 Sol rates. Legacy logs omitted `cache_write_tokens`, so the exact charge cannot be recovered; each hidden write would add the 25% write premium. Original reported → corrected lower bound: `gpt56sol_seed101` $0.869054 → ≥$0.576264, `gpt56sol_seed102` $1.373387 → ≥$0.876917, `gpt56sol_seed103` $1.582117 → ≥$1.085647, `gpt56sol_seed104` $1.450030 → ≥$0.940830.

Notes: definitions per doc 06 §§5.3-5.4 (SPL is 0.0 on failure; time-to-kitchen defined only on success). Batch-specific commentary is omitted for a redirected results dir — see the per-trial audit files alongside the raw JSONs.

## Per-trial results

### sonnet5

| Trial | Stage-1 outcome (v2) | Progress | SPL | Path (m) | Turns | Bumps | Falls | Drift (m) | Corr. | Map P | Map R | Edge acc | QA | Cost ($) | Video |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sonnet5_seed101 | declared_elsewhere | 0.541 | 0.000 | 7.29 | 39 | 8 | 0 | 1.504 | 1 | 0.50 | 0.50 | 0.00 | 0.50 | 1.204 | [sonnet5_seed101.mp4](videos/sonnet5_seed101.mp4) |
| sonnet5_seed102 | declared_elsewhere | 0.091 | 0.000 | 7.19 | 40 | 7 | 0 | 0.061 | 0 | 0.00 | 0.00 | — | 0.30 | 1.365 | [sonnet5_seed102.mp4](videos/sonnet5_seed102.mp4) |
| sonnet5_seed103 | declared_elsewhere | 0.087 | 0.000 | 19.12 | 40 | 9 | 0 | 0.060 | 2 | 0.00 | 0.00 | 0.00 | 0.30 | 1.551 | [sonnet5_seed103.mp4](videos/sonnet5_seed103.mp4) |
| sonnet5_seed104 | timeout_turns | 0.695 | 0.000 | 9.80 | 40 | 4 | 0 | 0.678 | 4 | 0.33 | 0.50 | 0.00 | 0.30 | 1.163 | [sonnet5_seed104.mp4](videos/sonnet5_seed104.mp4) |

### opus5

| Trial | Stage-1 outcome (v2) | Progress | SPL | Path (m) | Turns | Bumps | Falls | Drift (m) | Corr. | Map P | Map R | Edge acc | QA | Cost ($) | Video |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opus5_seed101 | success | 0.825 | 0.256 | 8.02 | 23 | 7 | 0 | 0.042 | 0 | 1.00 | 1.00 | 1.00 | 0.70 | 1.245 | [opus5_seed101.mp4](videos/opus5_seed101.mp4) |
| opus5_seed102 | declared_elsewhere | 0.128 | 0.000 | 32.00 | 40 | 26 | 0 | 0.264 | 1 | 0.00 | 0.00 | 0.00 | 0.70 | 4.357 | [opus5_seed102.mp4](videos/opus5_seed102.mp4) |
| opus5_seed103 | declared_elsewhere | 0.270 | 0.000 | 33.19 | 37 | 16 | 0 | 0.131 | 0 | 0.00 | 0.00 | — | 0.60 | 3.718 | [opus5_seed103.mp4](videos/opus5_seed103.mp4) |
| opus5_seed104 | success | 0.843 | 0.230 | 7.99 | 22 | 6 | 0 | 0.150 | 0 | 0.50 | 0.50 | 0.00 | 0.40 | 2.330 | [opus5_seed104.mp4](videos/opus5_seed104.mp4) |

### gpt56sol

| Trial | Stage-1 outcome (v2) | Progress | SPL | Path (m) | Turns | Bumps | Falls | Drift (m) | Corr. | Map P | Map R | Edge acc | QA | Cost ($) | Video |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt56sol_seed101 | declared_elsewhere | 0.609 | 0.000 | 4.90 | 24 | 4 | 0 | 0.766 | 1 | 0.50 | 0.50 | 0.00 | 0.50 | 0.576 | [gpt56sol_seed101.mp4](videos/gpt56sol_seed101.mp4) |
| gpt56sol_seed102 | declared_elsewhere | 0.000 | 0.000 | 16.05 | 40 | 16 | 0 | 0.057 | 0 | 1.00 | 0.50 | — | 0.20 | 0.877 | [gpt56sol_seed102.mp4](videos/gpt56sol_seed102.mp4) |
| gpt56sol_seed103 | declared_elsewhere | 0.274 | 0.000 | 10.40 | 40 | 14 | 0 | 0.148 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.086 | [gpt56sol_seed103.mp4](videos/gpt56sol_seed103.mp4) |
| gpt56sol_seed104 | timeout_turns | 0.000 | 0.000 | 13.65 | 40 | 9 | 0 | 0.377 | 2 | 1.00 | 1.00 | 0.00 | 0.40 | 0.941 | [gpt56sol_seed104.mp4](videos/gpt56sol_seed104.mp4) |

Per-question QA scores, matched room names, visited rooms, token counts and the return_home rows are in `results/scores.json`; raw evidence is `results/raw/<trial>.json` and `results/videos/<trial>.mp4`.

Generated by `scripts/build_scores.py` from `results/raw_v5d_r2/*.json` via `duck_embody.scoring` (no frozen file touched).
