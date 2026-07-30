# ABORTED BATCH — NOT A RESULT

Two partial trials from the v5d batch launched 2026-07-30 under `config_hash
6a65f335…` and stopped by hand after ~$4.87. **This is not a scored batch and
must never be pooled or published as one.** `scripts/auto_pipeline.sh` refuses
to score anything that is not the full 3x4 matrix with a `final` block per
trial, and these have neither.

They are kept because they are the primary evidence for two findings:

**1. The locomotion retrain works.** `fable5_seed101` ran 34 turns with **35
bump events and zero falls**. In the frozen v4 batch the same seed and spawn
fell on turn 2, 3.74 policy-seconds into its first `move`, torso on the sofa.
Frame-by-frame audit of the 543 recorded frames: trunk upright, legs alternating
with real ground clearance, no drag, no crumple.

**2. Dead reckoning was crediting motion to a wedged robot** — the bug fixed in
the same commit as this file. Measured here: 49 `send_velocity` calls reported
27.09 m travelled against 1.99 m of true displacement, worst case 0.60 m
credited for 0.01 m of real motion while `bumped=True` for a full 3 s. That one
tool produced 25.10 m of a 26.65 m position error, i.e. ~95% of what looked like
"drift" was an accounting bug. Genuine policy-tracking error over the same
trial's clean moves was 0.13 m.

Because of (2), **the position estimates in these files are corrupted** and the
trials cannot be used to say anything about localisation or about the models'
map quality. They are only usable as locomotion evidence (falls, bumps, turns
survived) and as the record of the bug.

They are also not reproducible: the fix changes `policy_wrapper.py`,
`memory.py` and `tools.py`, all frozen files, so any re-run happens under a
different `config_hash`. See `results/FREEZE_HISTORY.md`.
