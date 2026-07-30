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
