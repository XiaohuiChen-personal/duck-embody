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
