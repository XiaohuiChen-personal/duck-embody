# V5D R3 — Why Opus 5 outperforms (success vs fail traces)

**Status:** FORENSIC NOTE (read-only on trial JSON)  
**Date:** 2026-08-04  
**Batches:** certifying L8 `results/raw_v5d_r3/` + companion `results/raw_v5d_r3_fable5/`  
**Scores:** `results/scores_raw_v5d_r3.json`, `results/scores_raw_v5d_r3_fable5.json`  
**Parent reports:** [`V5D_R3_PERFORMANCE_REPORT.md`](V5D_R3_PERFORMANCE_REPORT.md), [`V5D_R3_FABLE5_PERFORMANCE_REPORT.md`](V5D_R3_FABLE5_PERFORMANCE_REPORT.md)

This note compares **primary trial traces** (turn tool calls, memory snapshots, declare geometry), not narrative summaries alone. Claims cite `results/raw_*/<trial>.json` turn ids / score fields. Frozen trial JSON was not modified.

---

## Verdict

Opus 5’s L8 headline (**2/4** find_kitchen v2, **2/4** return_home, **0** falls) is **partly seed-structure, partly capability**:

1. **Seeds 101–102 are furniture traps.** All four models fail find_kitchen on both seeds (timeout or declared_elsewhere). Opus’s two failures match everyone else’s failure mode: early wedge → `send_velocity` grind → never reach kitchen. That is **not** an Opus-specific weakness; it is a **shared hard-start effect**.
2. **On seeds where the kitchen is reachable (103, 104), Opus is the only L8 model that both reaches the counter face and returns home.** Sonnet 5 gets kitchen on 103 but fails return_home; GPT 5.6 sol reaches the kitchen room on 104 but declares too far from the counter face; Fable 5 (companion) matches Opus’s full success on 104 only.
3. **Reusable differentiators that show up in traces** (not harness bugs): (a) refuse `send_velocity` death spirals after sustained bumps; (b) declare only when true pose is inside the counter-face success region; (c) on return_home, use recorded anchors via `correct_to_anchor` and stop in the home disc; (d) prefer `turn_and_move` + observe cycles over raw velocity spam.

**Capability vs luck vs harness:** capability on declare calibration + RH closure; luck/seed on whether spawn avoids the living-room pinch (101) or bedroom/hallway west dead-end (102); **harness not implicated** — zero multi-motion violations across 16 cells, no tool-error/`nudged` storms in the success cells, same frozen scaffold for all contestants (AGENTS rule 4: leave model bad choices).

N=4 per model. Do not overclaim statistical superiority.

---

## 1. Label table (v2 find_kitchen + return_home)

Sources: `results/scores_raw_v5d_r3.json`, `results/scores_raw_v5d_r3_fable5.json` (`stages.*.success`, `outcome`, `end_reason`, `d_nearest_counter_face_m`).

| Trial | FK success | FK outcome | d_face (m) | RH success | RH outcome | bumps |
|---|---|---|---|---|---|---|
| opus5_seed101 | F | timeout_turns | 1.915 | F | not_run | 31 |
| opus5_seed102 | F | timeout_turns | 2.048 | F | not_run | 17 |
| opus5_seed103 | **T** | success | 0.221 | **T** | success | 8 |
| opus5_seed104 | **T** | success | 0.211 | **T** | success | 4 |
| sonnet5_seed101 | F | timeout_turns | 1.947 | F | not_run | 21 |
| sonnet5_seed102 | F | timeout_turns | 1.443 | F | not_run | 11 |
| sonnet5_seed103 | **T** | success | 0.200 | F | declared_elsewhere | 21 |
| sonnet5_seed104 | F | declared_elsewhere | 1.661 | F | not_run | 17 |
| gpt56sol_seed101 | F | declared_elsewhere | 1.940 | F | not_run | 25 |
| gpt56sol_seed102 | F | declared_elsewhere | 0.546 | F | not_run | 29 |
| gpt56sol_seed103 | F | declared_elsewhere | 2.040 | F | not_run | 14 |
| gpt56sol_seed104 | F | declared_elsewhere | 0.798 | F | not_run | 4 |
| fable5_seed101 | F | timeout_turns | 1.920 | F | not_run | 34 |
| fable5_seed102 | F | declared_elsewhere | 2.238 | F | not_run | 17 |
| fable5_seed103 | F | timeout_turns | 1.633 | F | not_run | 27 |
| fable5_seed104 | **T** | success | 0.298 | **T** | success | 4 |

Headline check: Opus FK 2/4 + RH 2/4 (103, 104); Sonnet/Fable FK 1/4; GPT 0/4 — **verified**.

Identical start poses per seed (first-turn `true_pose` in every model’s JSON):  
101 `(0.49,0.50,h≈89)`, 102 `(4.31,2.20,h≈269)`, 103 `(0.43,3.16,h≈359)`, 104 `(1.37,2.26,h≈179)`.

---

## 2. Within Opus: seeds 103/104 (success) vs 101/102 (fail)

### 2.1 Failures — wedge + `send_velocity` grind

**opus5_seed101** (`results/raw_v5d_r3/opus5_seed101.json`):

- t01 `look_around`; t02 maps `living_room` + plan; t03 `turn_and_move` already `bumped=true` with `distance_moved_m≈1.22` into the sofa aisle (`true_pose≈(0.58,1.14)`).
- From ~t08 onward, odometry reports **≤0.03 m** per motion while `status.bumped` stays true; tool mix is dominated by `send_velocity` (**25** calls in FK) vs **0** in either success trial.
- Never authors a kitchen room; end plan (`memory_snapshot.plan` at t40) is `Mode: STUCK-RECOVERY… wedged in the narrow N-S aisle`. Outcome `timeout_turns` / `d_face=1.915` (`final.stages.find_kitchen`).

**opus5_seed102** (`results/raw_v5d_r3/opus5_seed102.json`):

- Early phase is competent: bedroom → hallway (`rooms` gain `hallway_west` by t18; `turn_and_move` westward to `true≈(1.44,3.41)` by t21).
- Then west-end pinch: t23–24 bumps; t26–39 mostly `send_velocity` (**11** FK calls) with cm-scale motion; t40 `add_landmark` documents “permanently wedged… No kitchen evidence”.
- Visited true rooms `bedroom`,`hallway` only (`scores_raw_v5d_r3.json`). Timeout, `d_face=2.048`.

### 2.2 Successes — macros, verify, declare close, RH with anchors

**opus5_seed103** (`results/raw_v5d_r3/opus5_seed103.json`):

- Corridor spawn. Plan by t10 explicitly marks east void as dead end and prioritizes south doorway (plan text in memory block).
- Kitchen hypothesis in map: `kitchen_area` present by **t18**; confirmed `kitchen` room authored at **t28** (`update_room` description: “CONFIRMED KITCHEN… Black double-oven range…”).
- **0** `send_velocity` in FK; **9** `turn_and_move`, **8** `look_around`, **10** `get_observation`.
- `declare_done` at FK **t36**: `true_pose≈(2.50,0.57)`, score `distance_to_counter_m=0.2208`, `distance_to_success_region_m=0.0`, `in_goal_room=true`.
- RH: `correct_to_anchor` at RH t11 (`corridor_west_end`); `declare_done` RH t12 with `distance_to_success_region_m=0.0` (`final.stages.return_home.score`).

**opus5_seed104** (`results/raw_v5d_r3/opus5_seed104.json`):

- Living-room spawn. Kitchen room on map by **t07**; counter approach t11–15; declare FK **t16** at `true≈(2.63,0.56)`, `d_face=0.2105`, region `0.0`.
- Again **0** `send_velocity`; RH uses `correct_to_anchor` twice (RH t03 `kitchen_entry`, t09 `start`) and succeeds in 9 turns.

### 2.3 Within-Opus contrast (one line)

| Signal | 101/102 fail | 103/104 success |
|---|---|---|
| FK `send_velocity` count | 25 / 11 | **0 / 0** |
| Kitchen on map before end | no | t18 / t07 |
| Declare | never / never | t36 @0.22 m / t16 @0.21 m |
| Bumps | 31 / 17 | 8 / 4 |
| RH | not_run | success + `correct_to_anchor` |

The fail traces are not “Opus cannot map”; they are “Opus (like peers) cannot free itself from a furniture latch and then burns the turn budget on raw velocity.”

---

## 3. Same-seed cross-model (Opus success seeds 103 & 104)

### 3.1 Seed 103 — corridor west start

| Model | FK | Key trace fact | RH |
|---|---|---|---|
| **opus5** | success @ t36, d_face 0.22 | South doorway after east dead-end; kitchen room t28; 0 send_vel | **success** + `correct_to_anchor` |
| **sonnet5** | success @ t40, d_face 0.20 | Kitchen room by t13; then **t14–t39** oscillate near counter (`true` y flips 0.41↔1.04) before late declare | **declared_elsewhere** — had `HallwayStart` anchor, never `correct_to_anchor`; final `distance_m=2.376` from home (`return_home.score`) |
| **gpt56sol** | declared_elsewhere @ t40, d_face 2.04 | Rooms stay `Entry Hall`/`Living Room`; 12× `send_velocity`; never enters kitchen | not_run |
| **fable5** | timeout, d_face 1.63 | Reaches bedroom doorway and latches (`true≈(4.64,2.49)`); 24× `send_velocity`; kitchen never observed | not_run |

**What Opus did differently on 103:** exit-selection (south into kitchen rather than bedroom trap or living loop), verify-then-declare without burning 25 turns at the counter, then close RH with the start anchor. Sonnet’s FK success shows the seed is solvable for non-Opus models; Opus’s edge is **RH closure + less thrash after arrival**.

### 3.2 Seed 104 — living/dining start (easiest kitchen approach)

| Model | FK | Declare geometry | RH |
|---|---|---|---|
| **opus5** | success t16 | d_face **0.211**, region 0.0, `true≈(2.63,0.56)` | **success** |
| **fable5** | success t11 | d_face **0.298**, region 0.0, `true≈(2.03,0.43)` | **success** + `correct_to_anchor` |
| **gpt56sol** | declared_elsewhere t16 | **In kitchen room** (`in_goal_room=true`) but d_face **0.798**, region **0.333**; `true≈(1.94,1.06)` — plan claims “immediate proximity… Objective reached” | not_run |
| **sonnet5** | declared_elsewhere t40 | Walks **west** from spawn (`true` → `(0.55,2.29)` by t03); single room `StartRoom`; d_face 1.66 | not_run |

**What Opus did differently on 104:** vs GPT — same early kitchen discovery (~t06–07) but **refuses premature declare** until on the counter face; vs Sonnet — takes the east/south kitchen opening instead of exploring away from it; vs Fable — **tied** on full success (Fable is actually slightly faster: 11 vs 16 FK turns). Seed 104 alone does **not** make Opus unique; it separates Opus+Fable from GPT’s declare error and Sonnet’s wrong frontier.

---

## 4. Other successes vs failures (Sonnet 103 FK; Fable 104 FK+RH)

- **Sonnet 103 FK success** is real (v2; preregistered point criterion fails — see scores `success_preregistered=false`). Trace weakness vs Opus: no RH anchor correction; RH declare text at t37 still narrates kitchen approach (“identified the counter…”) while `true≈(1.90,1.28)` is ~2.4 m from home (`goal_xy=[0.43,3.15]`).
- **Fable 104** is the cleanest non-Opus full success: kitchen map t07, declare t11 @0.30 m face, RH with `correct_to_anchor` — same playbook as Opus 104. Fable does **not** generalize that playbook to 101–103 (wedge/timeout/wrong declare).

---

## 5. Reusable differentiators (with evidence)

1. **Zero `send_velocity` on success paths.** Opus 103/104 and Fable 104: `send_velocity=0`. Opus 101/102: 25/11. Fable 101/103: 30/24. GPT 101–103: 20/25/12. Cited: per-trial `model_output.tool_calls` counts in §2–3 extracts.
2. **Declare only inside the counter-face region.** Success declares: Opus 103/104 and Fable 104 have `distance_to_success_region_m=0.0`. GPT 104 declares with region **0.333 m** and face **0.798 m** while claiming verify-complete (`plan` at t16).
3. **Return_home uses `correct_to_anchor`.** All three RH successes (opus103, opus104, fable104) call it; Sonnet’s only RH attempt never does (`correction_calls.calls=0` in scores). Tool name in logs is `correct_to_anchor` (not `correct_position`).
4. **Early kitchen authorship predicts only if followed by approach.** GPT 104 maps `Kitchen` by t06 yet fails declare geometry; Opus 103 maps kitchen later (t28) but ends on the face.
5. **Bump recovery = observe/replan, not velocity spam.** Success trials after a bump typically `get_observation` / `look_around` / `turn_and_move`. Failures chain `send_velocity` for 10–20 turns with `distance_moved_m` in the 0–0.05 m band (opus101 t08–t40; opus102 t26–t39).
6. **Motion discipline is harness-enforced, not a model edge.** Multi-motion turns: **0** across all 16 trials (one motion tool per turn). Fairness: same constraint for all.
7. **Plan mode language tracks outcome.** Success end plans: `verify -> DONE` / counter proximity. Fail end plans: `STUCK-RECOVERY` / `EPISODE END… wedged` (opus101, fable101, fable103 memory `plan` at last FK turn).
8. **Map precision is a weak predictor.** Opus 103 map precision **0.25** (4 claimed rooms, over-segmentation) yet full success; opus101 map precision **1.0** (only living_room) yet fail. Navigation + declare beat room-name F1 here (`scores_raw_v5d_r3.json` `map_accuracy`).
9. **QA is secondary.** Opus mean QA 0.625 vs Sonnet 0.275 (L8 report), but opus101 QA 0.7 with FK fail — layout quiz ≠ task success.
10. **Harness-visible errors vs model choice.** No systematic `parse_errors` / `nudged` / tool `is_error` pattern separating Opus wins. Premature declare (GPT), wrong frontier (Sonnet 104), and wedge persistence (all on 101) are **model decisions** under AGENTS rule 4 — leave them as measured capability, not harness bugs.

---

## 6. Capability vs luck/seed vs harness (explicit)

| Factor | Role in Opus’s 2/4 + 2/2 RH |
|---|---|
| **Seed / layout luck** | Large. Seeds 101–102 fail for **all** models (shared pinch geometry). Opus’s two wins are exactly the two seeds where *someone* also found the kitchen (103 Sonnet; 104 Fable/GPT-near). |
| **Capability** | Real on (i) declare threshold, (ii) RH with anchors after FK, (iii) avoiding bedroom trap on 103 vs Fable, (iv) not declaring mid-kitchen on 104 vs GPT. |
| **Harness** | Not explanatory. Shared tools, caps, one-motion rule, identical starts per seed. No evidence Opus received different observations or leaked GT. |

Honest one-liner: **Opus wins the reachable seeds by finishing the job (counter-face declare + home anchor); it does not uniquely escape the unreachable seeds.**

---

## 7. Caveats

- N=4 seeds; two are near-automatic failures for every contestant — effective comparison density is ~2 hard/medium seeds.
- Fable is a companion batch (different freeze/config_hash; same checkpoint) — cross-batch comparisons are directional, not a locked ablation.
- Criterion is v2 any-counter-face; Opus also uniquely clears **preregistered** point success 2/4 on L8 (Sonnet/Fable prereg 0/4).
- Visual audits were not re-litigated here; used only as secondary confirmation that locomotion stayed upright (batch audits PASS).
- `progress` / path efficiency differ (opus103 true_path 18.6 m vs opus104 7.3 m) — longer path still succeeds if it reaches the face; efficiency is not the headline mechanism.

---

## Pointers

- L8 performance report: [`V5D_R3_PERFORMANCE_REPORT.md`](V5D_R3_PERFORMANCE_REPORT.md)  
- Fable companion: [`V5D_R3_FABLE5_PERFORMANCE_REPORT.md`](V5D_R3_FABLE5_PERFORMANCE_REPORT.md)  
- Raw trials: `results/raw_v5d_r3/opus5_seed{101–104}.json` and peers.
