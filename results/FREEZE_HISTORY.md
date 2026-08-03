# Freeze history

`results/freeze.json` always holds the CURRENT frozen configuration. Superseded
manifests are kept beside it, because the trials they certify are still
published and a batch whose manifest is gone cannot be verified.

| Manifest | config_hash | policy under test | batch |
|---|---|---|---|
| `freeze_v4_baseline.json` | `cf29ec164676…` | v4_robust (`policy/model_2999.pt`) | `results/raw/` — the 2026-07-27 batch, 10 falls in 12 trials |
| `freeze.json` | `6a65f33582eb…` | v5d_contact_wrench (`--checkpoint`, sha `301e24e336b2eab0`) | `results/raw_v5d/` |

The v5d hash differs because two frozen files changed: `configs/benchmark.yaml`
(locomotion constants re-measured for v5d) and `duck_embody/sim/policy_wrapper.py`
(`K_VELOCITY_REALISATION` 1.004 → 0.9617, the only runtime consumer). Both are
properties of the shipped policy, so a new policy necessarily means a new hash —
the two batches are not the same configuration and must not be pooled.

Note the standing caveat: no policy artifact is in `FROZEN_FILES`, so the hash
does not cover the checkpoint bytes. `scripts/auto_pipeline.sh` therefore writes
`results/logs/provenance_<label>.json` with the checkpoint sha256 before spending,
and refuses if the candidate's sha matches the baseline's.

## 2026-07-30 — odometry redesign + matrix swap

Superseded manifest archived as `freeze_pre_odometry_20260730.json`
(`config_hash 6a65f335…`). It certifies the two orphaned fable5 trials in
`results/raw_v5d/`, which are evidence only and were never a scored batch.

The new freeze covers a different contestant set (`sonnet5` replaces `fable5`)
and a changed motion contract (dead reckoning consumes simulated leg odometry,
not commanded velocity — AGENTS.md rule 5). Because both the models and the
mechanism changed, results across this boundary are **not** comparable; see
`docs/METRICS.md` §2.8 for the drift caveat specifically.

`freeze_v4_baseline.json` remains the manifest for the published v4 batch and
is untouched.

## 2026-08-02 — write-once batch manifests

TR.6 supersedes `freeze.json` as the complete provenance contract for every new
batch. `results/manifests/<batch_id>.json` is exclusive-create and self-hashed;
every trial points back to that SHA. The legacy freeze files above remain
readable evidence for v4/v5d and are never rewritten or upgraded.

The new manifest additionally binds the runner, pyproject, exact checkpoint,
checkpoint-keyed timeout calibration, parent commit/tree and robot USD, asset
verification, runtime/SDK versions, criterion, model configs, ordered slots,
and invocation environment-variable names. Benchmark launch refuses all drift
before Kit; explicit smoke output can downgrade provenance checks only outside
the benchmark result directories and is marked `config.smoke=true`.
