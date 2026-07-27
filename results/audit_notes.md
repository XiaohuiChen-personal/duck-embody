# Audit notes — Rule-11 video audits and figure spot-check

Batch: 12-trial benchmark, config_hash `cf29ec164676`, 12/12 AUDIT PASS.
Date: 2026-07-26.

One trial per model was selected for a full Rule-11 video audit (frozen
`results/raw/*.json` vs `results/videos/*.mp4`, filmstrips in
`results/figures/`). All three audits returned **CONSISTENT**. A separate
figure spot-check independently recomputed all 5 figures in
`results/figures/` from the raw JSON and also returned **CONSISTENT**.

Audit notes below are quoted verbatim from the audit outputs.

---

## fable5 — fable5_seed102

**Verdict: CONSISTENT**

> Rule-11 video audit of fable5_seed102 (filmstrip: results/figures/audit_fable5_seed102.png, 24 tiles at 1.52 s spacing; video 36.24 s maps ~1:1 onto 36.02 policy-seconds). Turn-by-turn check against results/raw/fable5_seed102.json: (1) t=0-5.6 in-place turn near bed matches turn_to_heading 90; (2) northward move ends t=7.9 with the robot's head visibly pressed into the white divider wall - matches logged head bump, move auto-stop at 0.42 m; (3) westward move ends t=13.9 wedged against the divider wall end - matches right_leg bump; (4) send_velocity backing t=14-17 visible; (5) t=17-27 repeated wall-gap struggle matches the bump cluster (turns 9-12); (6) sidestep + long corridor walk past the red bucket t=27-35.4 matches turn 13 (0.88 m move, stop on left_leg+torso bump at policy 35.44); (7) fall: at t=35.8, 0.36 s into the final turn_to_heading 180 (commanded wz=+0.5, the hull limit), the robot is visibly mid-topple leaning into the corridor wall; final frame shows ~60 deg tilt, matching fall diagnostics (tilt 59.99 vs 60.0 threshold, 0.58 s into call, height 0.15 m). Video ends pre-ground-impact by design - confirmed. All 7 counted bumps correspond to visible wall contact; no unlogged contact observed, no logged bump contradicted. Final true_pose (3.58, 3.41) mid-corridor beside the south wall matches the last frame. Video adds diagnostic context: the terminal tip-over occurs while in contact with the corridor wall immediately after a bump, during the max-rate in-place turn - consistent with the batch-wide wz-at-hull-limit fall diagnosis. No metric-vs-video disagreement; Rule 11 resolution not required.

Disagreements recorded (verbatim; both are map-content notes, not metric-vs-video conflicts):

> Map content (not a metric): the 'large dark grey angular object ... overhangs at head height' landmark in Hall (~x4.0,y2.6) has no corresponding object in the overhead video; egocentric frame t005_0.jpg shows a shadowed close-range surface (likely the divider wall corner or the robot's own head shell after the head bump) that the model interpreted as furniture. The associated bumps were actually against the white divider wall. Bump metrics match the video; only the model's semantic map entry is wrong - a map-precision item, already the model's error, not a harness metric error.

> Map content (not a metric): the Hall description claims 'two red buckets'; only one red bucket is visible in both the overhead video and the sampled egocentric frames (t013_0.jpg). Second bucket unconfirmed (could be outside camera coverage). Minor map-precision note.

---

## opus5 — opus5_seed102

**Verdict: CONSISTENT**

> Rule-11 video audit, opus5_seed102. Filmstrip: results/figures/audit_opus5_seed102.png (24 tiles, 3.52 s spacing, timestamps burned in); supplementary dense strips of the fall window (75.0-81.3 s) and the eastward excursion (39-47 s) were also reviewed. Timeline alignment: 81.04 cumulative policy-seconds vs 81.36 s video at 25 fps - turn-by-turn cumulative mapping holds. (1) Tool calls vs behavior: every turn_to_heading window shows in-place rotation (e.g., the final 215-deg turn at 76.7-79.9 s), every move/send_velocity window shows matching translation, and the vx=-0.148 reverse escapes visibly back the robot out of wall wedges at 24-27 s and 45-47 s. (2) Bumps: all 13 scored bumps fall in windows where the robot is visibly against the bed (t~4 s, torso) or the wall nook/panels (t~18-64 s, head/torso/right_leg); no bump is logged while the robot is visibly in free space, and no unlogged hard contact was seen at the sampling density used. (3) Eastward excursion: T17-T19 (39-47 s) shows the robot walking east through the wall gap to true x~4.72, wedging between two panels (head bumps at 44.1 and 47.1 s) with the out-of-bounds black grid void visible - matches the model's own 'stepped outside the building' plan note. (4) Fall: occurs where/when diagnosed - 1.18 s into the final move (video t~81.1 s) at true (3.846, 2.274), still inside the start bedroom (bed visible top-left of final frames, corroborating scoring's true_rooms_visited=[bedroom] and map precision/recall 0.0). Final full-res frame shows the robot mid-topple at a strong tilt matching fall_diagnostics tilt_deg 57.14 (values_pre_step=true; fell_over term, tilt-60 threshold), and the video ends pre-topple by design. Torso contact is recorded at the fall step; the topple begins on open floor beside the partition, plausibly torso-to-wall or torso-to-floor as it tips - nothing contradicts. Verdict: video and the trial's frozen log/metrics agree on every checked point. One flag, recorded under disagreements: it is a batch-summary-vs-trial-diagnostics wording issue, not a metric-vs-video conflict.

Disagreement recorded (verbatim; a batch-summary wording issue, not a metric-vs-video conflict — the video sides with the frozen JSON):

> Batch headline wording vs this trial's frozen diagnostics (video sides with the JSON): the claim '10 falls, all diagnosed wz at/near the +/-0.5 hull limit at the tilt-60 termination' does not describe opus5_seed102. Its fall_diagnostics.commanded is [vx=0.2, vy=0.0, wz=0.0219] - a forward step with wz~0 and a torso bump at the fall step; the preceding +/-0.5-wz turn_to_heading completed cleanly (stop_reason 'reached') 1.18 s earlier. Cross-check of all 12 raw files: only 5 of the 10 falls are at |wz|=0.5 exactly (fable5_seed102, fable5_seed103, gpt56sol_seed101, gpt56sol_seed102, gpt56sol_seed104); the other 5 (fable5_seed101 and all four opus5 falls: seeds 101-104) fell during a forward 'move' with |wz| between 0.02 and 0.29, four of them with torso contact at the fall step. The batch-level fall diagnosis should be corrected to distinguish hull-limit spin falls (5) from forward-step topples, mostly with contact (5).

**Reporting action:** the batch-level fall diagnosis must be worded as 5
hull-limit spin falls (|wz| = 0.5 exactly) plus 5 forward-step topples
(|wz| 0.02–0.29, four with torso contact at the fall step), not "all 10 at
the wz hull limit."

---

## gpt56sol — gpt56sol_seed103

**Verdict: CONSISTENT**

> gpt56sol_seed103 Rule-11 video audit: CONSISTENT. Filmstrip at results/figures/audit_gpt56sol_seed103.png (24 tiles, 1.8 s apart). Cumulative policy-seconds (43.76) match video length (43.80 s) and every commanded motion is visible at its predicted time: T03/T05 in-place turns (0-6.4 s), T06+T10 corridor moves east past the red floor box (6.4-22.2 s), T11/T14 turn-south + 1.0 m into the white room with red bar chair and two dark-blue counter/fridge blocks (22.2-30.2 s), T16/T17 turn-east + approach (30.2-35.2 s). Bump 1 (head, T17 auto-stop at 35.2 s): frame at 35.0 s shows the robot flush against the far side of the dark-blue counter, antennae visible at its top edge - timing and geometry consistent; the contact itself is occluded by the counter, so the video is consistent-but-not-independently-confirming here. The robot is fully occluded behind that counter 35-38.5 s (T19 turn + start of T21), which explains the two robot-free tiles in the main strip; it re-emerges at 38.6 s exactly where the 0.3 m southward move predicts. Bump 2 (torso, T25 auto-stop at 43.8 s): final frame shows the robot upright directly against a low gray floor-level fixture - consistent with a torso-height contact on a 0.08 m move. bumps=2 tally is coherent (two distinct contact episodes: head T17-T21 flags, torso T26-T27 flags). fell=false throughout matches the video: the robot never topples; the video ends with it standing at declare_done. One visually alarming frame (41.4 s, strong lean) resolves in a 0.4 s-step tail strip as normal turning-gait dynamics, upright again by 42.6 s - not a fall. Outcome nuance for reporting (not a disagreement): declared_elsewhere is a goal-radius miss (0.83 m from goal (2.55,0.75), radius 0.35 m), and the video shows the robot genuinely inside the kitchen-looking room next to counter fixtures when it declared - the room identification was right, the declare position was short. Dead-reckoning drift at end: estimate (2.60,1.72) vs true (2.93,1.49), ~0.40 m - metric-side note only, video cannot arbitrate it.

Disagreements recorded: none.

---

## Figure spot-check (results/figures/ vs results/raw/)

**Verdict: CONSISTENT**

> All 5 figures in results/figures/ verified against results/raw/*.json by independent recomputation (system python3, hard-coded layout constants, no scores.json dependence; verification scripts in scratchpad: verify_figures.py, verify_map.py). per_metric_bars: bumps means 2.75/4.25/1.75 from raw final.bumps sums 11/17/7 over 4 seeds (cross-checked vs per-call counted_as_bump flags); progress means 0.0658/0.1507/0.2185 -> labels 0.07/0.15/0.22 via clip(1-d_f/d_i) with target (2.55,0.75) (e.g. gpt56sol_seed103: d_i=3.2022, d_f=0.8342 -> 0.7395); drift means 0.1598/0.1491/0.1767 -> 0.16/0.15/0.18 via obs.position_estimate vs prev-turn true_pose; map precision/recall means 0.625/0.50/0.75 and 0.875/0.50/0.75 -> 0.62/0.50/0.75, 0.88/0.50/0.75 with full doc-06 s5.7 matching re-implemented from scratch; QA means 0.575/0.65/0.30 -> 0.57/0.65/0.30; success 0/4 x3 and SPL 0.00 honest (10 falls, 2 declares at d_f 1.66 m and 0.83 m > 0.35 m radius). Independent 10k percentile bootstrap reproduces every plotted CI whisker; widths plausible for n=4 (opus bumps [0.5,10.0] from {0,1,3,13}). Axes zero-based, no swapped models (config.model asserted per raw file; legend/colors consistent). turns_survived: all 12 heights equal raw find_kitchen turn counts (2/14/5/11, 2/28/16/3, 6/11/27/8); hatched bars exactly the 2 declare_done trials; cap line 40 = memory.TURN_CAP. Trajectory figures: all subtitle stats recomputed and matching; all 5 Hz true-path points inside the 4.8x3.6 floor plan (0 out-of-bounds points across all 12 trials); claimed-room labels exactly match final memory_snapshot rooms and diamonds sit at first-claim true poses. Notably, fable5_seed102's "Hall" (no match) was verified correct by rule, not just consistent: "hall" IS a hallway synonym but only 2/5 evidence points lie in the hallway, failing the evidence-majority half of s5.7 - worth a footnote in the report since readers may assume name-match suffices. Cosmetic-only observations: belief traces honestly drawn outside walls where raw estimates go there (fable5_102 belief-at-end (3.65,3.71) above the north wall; opus5_102 belief x to 4.83 past the east wall) - faithful to raw, arguably the figure's point; opus5_102's true-end X marker is visually buried in the dense path scribble. No metric-vs-video disagreements arose from this check (videos not re-reviewed; figure-vs-raw only, per the spot-check scope).

---

## Rule-11 summary

No metric-vs-video disagreement was found in any of the three audited
trials; Rule-11 resolution (video overrides metric, with a note) was not
required anywhere. The items recorded under "disagreements" are, in every
case, either model map-content errors (fable5_seed102 — already scored
against the model as map precision) or a batch-summary wording issue
(opus5_seed102 — the raw JSON and the video agree with each other and
against the earlier headline wording; see the reporting action above).
The figure spot-check likewise surfaced no metric-vs-video conflicts.

## Addendum 2026-07-27 — scoring criterion v2 (post-dates every audit above)

Every audit note above was written under the pre-registered point-target
criterion and is preserved verbatim: each remains a true record of what it
audited **at the time**. After they were written, the published success
criterion was widened to v2 ("any counter face" — `results/rerun_log.md`,
`docs/METRICS.md` §2.1). Two things follow, stated completely:

**Superseded verdict wordings.** Reading the notes above under v2:
- gpt56sol_seed103's "declared_elsewhere is a goal-radius miss (0.83 m …)"
  describes the pre-registered verdict; under v2 that same declare — which the
  video audit itself describes as "genuinely inside the kitchen-looking room
  next to counter fixtures" — is the batch's single success (0.051 m from
  counter_5's face). The video observation and the v2 verdict agree.
- The figure spot-check's recorded aggregates — "success 0/4 x3 and SPL 0.00
  honest", "2 declares at d_f 1.66 m and 0.83 m > 0.35 m radius" — are the
  pre-registered numbers. Under v2: success 0/4, 0/4, 1/4; gpt56sol's SPL
  0.4153; the seed-103 declare is inside the v2 region.

**The five figures were REGENERATED after that spot-check.** The spot-check's
CONSISTENT verdict applies to the pre-v2 figures (recoverable from git
history), not to the files now in `results/figures/`, which carry the v2
numbers and draw the v2 success region. The regenerated figures were verified
separately (2026-07-27, adoption-verification workflow `wf_3a009fd5`): an
independent re-derivation — own point-to-rect, point-in-room and region-Dijkstra
code, no `duck_embody` imports — reproduced every per-trial v2 verdict, all
four region-oracle lengths (2.0521/3.1071/3.1420/1.8399 m), seed103's SPL
0.4153 = 3.142/7.5661, and every summary_table cell; a second reviewer
visually inspected all five PNGs (1/12 labeling, counter bands clipped to the
kitchen, verdict-neutral declare_done legend). `layout_plan.png` and the
`audit_*.png` filmstrips were deliberately NOT regenerated: the filmstrips are
as-run rule-11 evidence for the notes above, and `layout_plan.png` is
pre-batch scene QA that makes no success-region claim.
