# Rerun log — doc 06 §7

Every resume move and infra rerun in the batch, appended by
`duck_embody/runner.py`. It ships with the results: reruns are visible, not
silent. Model failures (cap / fall / wrong `declare_done`) are final results
and never appear here — the only legitimate rerun is a logged infra failure.
T4.3's restart branch, when taken, is also recorded here
(a: fix touches non-frozen code -> keep the freeze commit, resume;
b: fix touches any frozen file -> new freeze commit, new batch directory,
restart from zero).

| trial id | timestamp (UTC) | cause | evidence |
|---|---|---|---|
| opus5_seed101 | 2026-07-27T08:51:25Z | infra failure (attempt 1): anthropic.OverloadedError: Error code: 529 - {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}, 'request_id': 'req_011CdSH1VvSXFUyxu4NQrHDb'} | results/incomplete/opus5_seed101.20260727-085125.json |

## Post-batch scoring changes — `duck_embody/scoring.py`'s own protocol

`scoring.py` is deliberately outside the freeze hash ("re-scoring is free,
re-running a paid batch is not"); its header requires any post-batch change to
re-score **all** models together and be logged here.

### 2026-07-27 — success criterion v2: "any counter face" (owner-directed)

- **What changed.** The published `find_kitchen` success predicate widened from
  the pre-registered 0.35 m disc around the target point `(2.55, 0.75)` to the
  UNION of that disc and "within the same 0.35 m of any of the five kitchen
  counters' footprint rectangles while standing inside the kitchen".
  `declare_done` is still required; `return_home`'s predicate is unchanged.
  Stage-1 SPL's oracle `l` became the shortest free-space path to the success
  REGION (ObjectNav convention); `progress`/`d_initial`/`d_final` keep the
  point reference. Implemented in `duck_embody/scoring.py`
  (`stage_success` / `position_success_v2` / `region_oracle_length_m`);
  the as-run verdicts remain published per trial as `success_preregistered` /
  `outcome_preregistered`, and every as-run log-consistency check still runs.
- **Why.** The frozen objective — "Find the kitchen and walk to the counter" —
  never disambiguates the two counter runs (south, where the scored point sat,
  and east-wall), and `gpt56sol_seed103` declared done 0.051 m from the
  east-wall run's `counter_5` and scored `declared_elsewhere`. The owner
  directed adopting the any-counter reading as the published criterion. The
  union (not the counter band alone) because the regions are not nested: the
  pinned target point is 0.397 m from the nearest counter footprint, so a pure
  any-counter test would fail a robot standing exactly on the old goal
  (adversarial review, workflow `wf_fc153dc0`, three CONFIRMED verdicts).
- **Decided post hoc**, after all 12 results were visible — a sensitivity
  analysis (`scripts/rescore_any_counter.py`, committed with this change)
  preceded and motivated the adoption. All 12 trials of all 3 models were
  re-scored together; no raw trial JSON, no frozen file, and no as-run verdict
  was modified.
- **Effect.** `find_kitchen` SR 0/12 → **1/12** (`gpt56sol_seed103` flips,
  `declared_elsewhere` → `success`; every other verdict unchanged).
  `fable5_seed104`, the only other declare, is a failure under both criteria
  (living room, 1.396 m from any counter). Stage-1 oracle lengths shrink for
  all 12 (region vs point), which lowers the SPL of a hypothetical
  point-reaching success; all 11 remaining SPLs stay 0.0 (failures).
- **What cannot be recovered.** The LIVE stage-2 gate consulted the
  pre-registered predicate, so `gpt56sol_seed103` was never offered its
  `return_home` leg; that data does not exist without a rerun. The conditional
  return-home rate therefore counts only offered legs (k=0 → "—"), with the
  exclusion published as `stage1_successes_never_offered_return`.
| gpt56sol_seed102 | 2026-08-03T07:46:08Z | infra failure (attempt 1): openai.RateLimitError: Error code: 429 - {'error': {'message': 'You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/ | results/incomplete/gpt56sol_seed102.20260803-074608.json |

### 2026-08-03 — v5d-r3-final-prod L8 pause: OpenAI credit exhaustion

- **Batch.** Write-once manifest `results/manifests/v5d-r3-final-prod.json`
  (`manifest_sha256`
  `17a79cc37604c55119cd25a949858bb2d947db2ae7f1e7b57fb5e19500ac16cd`), freeze
  `56bd08a68d922d205992679a403bfb577b5e2194`. Out dir `results/raw_v5d_r3/`.
- **Progress.** 9/12 cells wrote `final` before stop (sonnet5 101–104, opus5
  101–104, gpt56sol_seed101). Missing: gpt56sol seeds 102–104.
- **Infra cause.** First missing cell quarantined as
  `results/incomplete/gpt56sol_seed102.20260803-074608.json` with
  `infra_failure` → `openai.RateLimitError` 429,
  `code: credit_balance_exhausted`, `type: insufficient_quota`. Live probe
  later the same day still returned 429 insufficient_quota. Not a model
  failure; not a Kit/GPU failure (GPU idle).
- **Resume.** After credits are restored, same `--batch-id v5d-r3-final-prod`
  into `results/raw_v5d_r3` / `results/videos_v5d_r3` (exact command in
  `docs/PLAN.md` TR.9 L8 block). Do not invent a new manifest unless a frozen
  file changes.

### 2026-08-03/04 — v5d-r3-final-prod L8 resume complete (gpt56sol 102–104)

- **Same manifest.** No new freeze; no `--force`; successful cells not re-run.
- **Cells added.** `gpt56sol_seed{102,103,104}.json` with `final` under
  `results/raw_v5d_r3/` + videos/filmstrips under `results/videos_v5d_r3/`.
  Outcomes (v2): all three `declared_elsewhere` / `return_home=not_run`
  (costs $1.625 / $1.574 / $0.953). Seed 102 video/filmstrip replaced the
  aborted credit-block attempt's partial media.
- **Closure.** Scores 12/12 (`results/scores_raw_v5d_r3.json`): headline
  **3/12** find_kitchen (v2), **2/12** preregistered, **2/12** return_home,
  **0 falls**, **$26.56**. Machine+visual audits 12/12 PASS; batch audit
  `results/raw_v5d_r3_batch_audit.json` = PASS. Report:
  `docs/research/V5D_R3_PERFORMANCE_REPORT.md`. Credit quarantine retained.

### 2026-08-04 — companion batch prep: `v5d-r3-fable5` (Fable 5 × seeds 101–104)

- **Why.** Owner-directed L8-equivalent companion to measure Fable 5 on the
  same v5d-r3 harness discipline. Certifying L8
  (`results/manifests/v5d-r3-final-prod.json`, out-dir `results/raw_v5d_r3/`,
  matrix sonnet5/opus5/gpt56sol) stays untouched.
- **Why a new batch-id, not extending `v5d-r3-final-prod`.** Write-once
  manifests refuse overwrite; the live matrix also differs (fable5-only vs
  the three L8 contestants). Extending in place would either mutate an
  immutable manifest or pool incomparable matrices under one id.
- **Live matrix.** `configs/benchmark.yaml` → `models: [fable5]`, seeds
  `[101, 102, 103, 104]`.
- **Fairness set.** `configs/models/fable5.yaml` re-added to
  `FROZEN_FILES` (sonnet5/opus5/gpt56sol remain hashed for continuity).
  Provenance is the existing file: `model_id: claude-fable-5`, pricing
  $10/$50 per MTok — no invented ids or rates.
- **Out dirs (planned).** `results/raw_v5d_r3_fable5` +
  `results/videos_v5d_r3_fable5`. Write-once manifest created on first
  real launch as `results/manifests/v5d-r3-fable5.json`.
- **Scope of this entry.** Prep only (matrix/freeze/probe/dry-run). Paid
  4-trial launch is a later step.

### 2026-08-04 — companion batch complete: `v5d-r3-fable5`

- **Batch.** 4/4 `final` under write-once manifest
  `results/manifests/v5d-r3-fable5.json`
  (`manifest_sha256=1d4249e73fce0fc3b7f0104af64034828cfcf9244f2e8275b6d12983cf53b416`),
  config_hash `7fbce2573c184a10…`, checkpoint SHA identical to L8
  (`301e24e336b2…`). Raw `results/raw_v5d_r3_fable5/`, videos
  `results/videos_v5d_r3_fable5/`.
- **Headline.** find_kitchen v2 **1/4**, prereg **0/4**, return_home **1/4**
  (seed104 only; 1/1 given stage-1 success), **0 falls**, cost **$27.37**.
  Scores: `results/scores_raw_v5d_r3_fable5.json`.
- **Audits.** Machine 4/4 PASS (`results/raw_v5d_r3_fable5_audits/`);
  visual 4/4 PASS (`results/raw_v5d_r3_fable5_visual_audits/`); batch audit
  `results/raw_v5d_r3_fable5_batch_audit.json` = PASS.
- **Report.** `docs/research/V5D_R3_FABLE5_PERFORMANCE_REPORT.md`.
  Certifying L8 (`v5d-r3-final-prod` / `results/raw_v5d_r3/`) untouched.
