# sonnet5_seed101 — frame-by-frame audit

**Locomotion: HEALTHY.** 12 frames sampled across the trial (every 60th frame,
evenly spanned). Trunk upright in every frame; legs alternating with visible
ground clearance; no drag, no crumple, no stumble recovery. 39 turns, 8 bump
events, zero falls. Path: living room (sofa/armchair) → past the coffee table →
through the doorway → into the appliance zone.

**Odometry (the redesign under live conditions): CONFIRMED.**
10 motion calls, believed 2.65 m against 2.71 m of true displacement —
**-0.06 m**, i.e. the estimate slightly UNDER-reports, which is the honest
noise model working. The same seed under commanded-velocity reckoning
(fable5_seed101, v4 batch) believed 30.75 m against 4.41 m true: **+26.34 m**.
The accounting artifact does not appear in a paid trial.

**Outcome:** `declared_elsewhere` on find_kitchen; `return_home` not run.
The robot reached the appliance zone (visible in the last four frames) and
declared done without satisfying the counter-proximity criterion — a
navigation/judgement outcome, not a locomotion failure.

**Loop closure: 0 `correct_position` calls.** The anchors render and resolve
(smoke-verified end-to-end), but this model did not use them in this trial.
One trial is not evidence either way; tracking across the batch.

Cost $1.20 (vs $6.57 for the fable5 trial it replaces).
