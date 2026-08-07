# V5D R3 — Furniture-wedge harness improvements

**Status:** RESEARCH NOTE (trial JSON untouched) — B3+A1 harness fix landed 2026-08-05 (`policy_wrapper.execute` rising-edge/reconfirm + macro cross-chunk sustained accumulation; `tools.status.progress`). Adversarial review same day fixed two confirmed defects: (1) rising-edge must be free/candidate_contact→sustained only — candidate_release→sustained is same-event hysteresis and was re-aborting pre-latched reverse on gait-cycle force troughs; (2) A1 streak updates only on `counts_bump` tools so sticky-latch `turn_to_heading` cannot inflate `no_progress`. Note: `MACRO_CHUNK_S` (0.2 s) < `CONTACT_SUSTAINED_STEPS` (0.4 s), so execute-only reconfirm is insufficient for macros — they must accumulate sustained steps across chunks or a single chunk reintroduces early abort. Kit smoke 2026-08-05 found a third defect: still-touching reverse reconfirm at 0.4 s caps backup at ~0.04 m (`REVERSE_MOVE_SPEED_MPS`); fixed with `CONTACT_REVERSE_GRACE_S=1.5` for `vx<0` only (forward reconfirm unchanged). Smoke script: `scripts/smoke_wedge_reverse.py`.

**§7#4 kit smoke verdict (2026-08-05):** B3 stop-predicate **PASS** (pre-latched reverse runs full grace, steps=90 ≠ step-0 abort). Physical clearance **FAIL** — **plant-limited**, not the old mm no-op. Definitive reverse-phase artifact `results/logs/wedge_reverse_20260805-013345/`: latch (0.31, 0.88, h≈86°), `hold_heading_deg=90`, true_disp=0.012 m, axis_progress=−0.010 m, contact stays `sustained_contact` (head+torso), measured=0.060 m (phantom odom). Layout south free ≥0.78 m exists; filmstrip shows west-wall+sofa-arm corner involvement. Follow-ups: outer-west wall / sofa@x=0.50 bump but never latch sustained; do **not** extend `CONTACT_REVERSE_GRACE_S` past 2.0 s. Clearance fail root cause = **plant/gait under sustained furniture press** (+ corner geometry), not the old step-0 macro no-op.  
**Measured companion (2026-08-06):** `v5d-r3-opus5-b3a1` COMPLETE — see §11. Headline **0/4** find_kitchen vs L8 Opus **2/4**; no Tier C / furniture-gap change (LEAVE).  
**Date:** 2026-08-06 (measured §11); smoke 2026-08-05  
**Scope:** The six `timeout_turns` / turn-cap find_kitchen failures attributed to furniture (or wall-adjacent) wedges  
**Batches:** L8 `results/raw_v5d_r3/` + companion `results/raw_v5d_r3_fable5/` + companion `results/raw_v5d_r3_opus5_b3a1/`  
**Constraints:** AGENTS.md §3 rules 4–5 (FIX reporting/harness defects; LEAVE bad model choices; no decision-making pathfinding tools)  
**Related:** [`V5D_R3_OPUS_SUCCESS_VS_FAIL_TRACE_COMPARISON.md`](V5D_R3_OPUS_SUCCESS_VS_FAIL_TRACE_COMPARISON.md), [`V5D_R2_HARNESS_FORENSICS.md`](V5D_R2_HARNESS_FORENSICS.md) F-04, design doc `docs/designs/05-agent-harness.html` §4.2, prompt `duck_embody/agent/prompts.py`, report [`V5D_R3_OPUS5_B3A1_PERFORMANCE_REPORT.md`](V5D_R3_OPUS5_B3A1_PERFORMANCE_REPORT.md)

Frozen trial JSON was not modified. Every quantitative claim below names a trial path + turn / field.

---

## Verdict (adversarial)

**The harness did not silently hide the wedge.** On every timeout cell, macros and status already report `bumped=true`, multi-region `contact`, `status.current_contact.state=sustained_contact`, and centimetre-scale `distance_moved_m` / measured distance. The frozen prompt already says a shortfall vs requested distance means obstruction, and that `move(distance_m)` accepts a **negative** backup (`prompts.py` ~L300–L344).

**Models usually know they are stuck** (thoughts early; end-of-episode plans often name STUCK / wedged). They still burn the remaining turn budget on `send_velocity` thrashing.

**There is still a real harness defect in recovery mechanics:** once contact is already latched, signed `move(-x)` aborts on the first control step while `_contact_state == sustained_contact`, so the advertised closed-loop backup is a near no-op (measured ≈ 0.001–0.004 m). Models escalate to `send_velocity` (no auto-stop) — which is honest about ~0 progress but has empty `stop_reason` on 100% of bumped calls in these six trials, and invites open-loop grinding.

| Diagnosis | Verdict |
|---|---|
| “Harness never said stuck” | **Mostly false.** Signals are present; an explicit `stuck`/`blocked` aggregate is missing. |
| “Model ignored clear stuck signal” | **Mostly true** for the long `send_velocity` tails after awareness. |
| “Macro recovery is broken when pre-latched” | **True** — rule-4 FIX (macro no-op vs advertised backup). |
| “Need auto path-around furniture” | **Out of scope** for an honest scaffold (rule 5 boundary). |

**Recommended first fix:** make pre-latched signed `move` able to run long enough to clear contact (rising-edge / this-command sustained abort), *and* add a model-facing `progress` / no-progress streak summary derived only from already-reported fields. Ranked options below.

---

## 1. The six timeout cells

Sources: `results/scores_raw_v5d_r3.json`, `results/scores_raw_v5d_r3_fable5.json`; trial JSON under paths in the table.

| Trial | Path | FK end | bumps | End true_pose (approx) | Geometry class |
|---|---|---|---|---|---|
| opus5_seed101 | `results/raw_v5d_r3/opus5_seed101.json` | turn_cap | 31 | (0.67, 1.34, h≈74) | Living-room sofa–coffee_table N–S aisle |
| sonnet5_seed101 | `results/raw_v5d_r3/sonnet5_seed101.json` | turn_cap | 21 | (0.48, 1.00, h≈109) | Same living-room pinch (slightly south) |
| fable5_seed101 | `results/raw_v5d_r3_fable5/fable5_seed101.json` | turn_cap | 34 | (0.67, 1.34, h≈74) | Same living-room pinch |
| opus5_seed102 | `results/raw_v5d_r3/opus5_seed102.json` | turn_cap | 17 | (1.40, 2.89, h≈221) | Hallway west after bedroom→west push; south hallway wall / doorway frame |
| sonnet5_seed102 | `results/raw_v5d_r3/sonnet5_seed102.json` | turn_cap | 11 | (2.81, 3.02, h≈277) | Hallway mid/west latch (less absolute pin than 101) |
| fable5_seed103 | `results/raw_v5d_r3_fable5/fable5_seed103.json` | turn_cap | 27 | (4.64, 2.49, h≈358) | Bedroom doorway / east-bedroom latch (Opus/Sonnet *succeed* on seed 103) |

Identical starts per seed (first-turn `true_pose`): 101 `(0.49,0.50,h≈89)`, 102 `(4.31,2.20,h≈269)`, 103 `(0.43,3.16,h≈359)`.

Layout evidence (`duck_embody/env/apartment_layout.py`):

- Sofa AABB ≈ `[0.10,0.50]×[1.11,2.09]`, coffee_table ≈ `[0.73,1.03]×[1.34,1.86]` → N–S slot near x≈0.6, y≈1.1–1.4. `clearance(0.67,1.34)=0.060 m` vs ~0.16 m body → true pinch.
- Seed 102 late poses sit in the hallway band y∈(2.7,3.6) against wall A2 / west exploration; furniture clearance to `planter_w` is larger (~0.2–0.4 m) — this is often **wall/doorframe latch**, not sofa pinch. Models still narrate “wedged”.
- Seed 103 fable end `(4.64,2.49)` is south of the bedroom doorway center `(4.05,2.7)`; `clearance≈0.145 m`.

Visual audit (`results/raw_v5d_r3_visual_audits/opus5_seed101.md`): locomotion PASS; “late frames are sofa-wedged under sustained_contact”.

---

## 2. What the model is told when wedged

### 2.1 Macro / status fields (present)

Example — `opus5_seed101` t08 `move` result (`tool_results[].json_text` → `status`):

- `status.bumped = true`
- `status.contact = ["left_leg","right_leg","head","torso"]`
- `status.current_contact.state = "sustained_contact"` (same regions)
- `status.distance_moved_m = 0.004` (requested −0.5 m)
- top-level `stop_reason = "sustained_contact"`
- `target_reached = false` (macros)

After latch, nearly every subsequent motion keeps full-body or head+torso contact lists. Across these six trials, **`current_contact.state=sustained_contact` is the dominant latched report** (dozens of motion results per trial); a minority of bumped results briefly show `current_contact.state=free` while legacy `status.contact` still lists regions — legacy vs structured fields can disagree for one call, but the structured latch is usually correct.

### 2.2 `send_velocity` reporting gap (confirmed)

Across the six trials: **111 / 111** bumped `send_velocity` results have **empty** top-level `stop_reason` and empty `status.last_motion.stop_reason`, while `status.bumped=true`, `distance_moved_m` is centimetres, and `current_contact.state` is usually `sustained_contact`.

Cause (code, not guess): `_send_velocity` calls `execute(..., stop_on_bump=False)` (`duck_embody/agent/tools.py`); `execute` only sets `stop_reason="sustained_contact"` when `stop_on_bump` triggers the early break (`policy_wrapper.py` ~L1138–1141). Full-duration collision therefore ends with `stop_reason=""`.

This is not “never told about contact” — bumped + odom + current_contact remain — but it is a **reporting asymmetry**: macros name the stop; the escape hatch does not.

### 2.3 Prompt already documents backup + obstruction

`duck_embody/agent/prompts.py`:

- Signed `move`: negative backs up more slowly; macros auto-stop on persistent contact (~L300–305).
- `send_velocity` is the raw escape hatch and does **not** auto-stop (~L315–317).
- “If a move reports far less than you asked for, you are obstructed” (~L341–344).
- Contact regions describe *which part of the body* feels force, not object identity (~L307–312).

There is **no** explicit `stuck` / `blocked` / consecutive-no-progress field, and **no** stuck-recovery mode in the CogNav state machine (~L346–351: broad / contextual / verify only).

### 2.4 Memory block

`memory.py` does not re-inject bump streaks or last-contact summaries into the map block. Awareness is turn-local (tool result + whatever the model wrote into `update_plan`).

---

## 3. Do models know? Evidence

Stuck language was searched in `model_output.thought` / `text` and in `update_plan` / landmark args.

| Trial | Early awareness | End-state self-report |
|---|---|---|
| opus5_seed101 | thought@t06 notes odometry vs estimate discrepancy after bumper travel | Final memory plan: `Mode: STUCK-RECOVERY… wedged in the narrow N-S aisle between the red sofa…` |
| opus5_seed102 | thought@t13 “north… blocked”; late plan STUCK/recovery at west hallway | Final plan: wedged at west end; “every command yields only 2–8 cm” |
| sonnet5_seed101 | thought@t06 “contact… back up and reassess” | Continues SV; no kitchen |
| sonnet5_seed102 | thought@t28 “hit a wall… reorient” | Late plan still “contextual search” toward presumed kitchen while motions are centimetre-scale |
| fable5_seed101 | thought@t06 narrow corridor sofa/table; stuck terms @t07–08 | Final plan: `EPISODE END… irrecoverably wedged` |
| fable5_seed103 | stuck terms @t14,16,21 | Final plan: `MODE: last-ditch escape… macros abort (sustained_contact latched)` |

**Adversarial read:** for 101 and fable103, “ignored clear signal” is fair for the *late* grind. For all six, the *first* recovery attempt is usually correct in intent (backup). The failure mode is **recovery does not free the body**, then **open-loop thrash consumes the budget**.

---

## 4. Mechanical timeline (shared pattern)

### 4.1 Seed 101 (all three models) — sofa aisle

1. Spawn living room → north into sofa / coffee_table slot (`true` → ~`(0.6–0.7, 1.1)` by t03–t06).
2. First `stop_reason=sustained_contact` with meaningful prior travel (e.g. opus t07 `move` measured 0.752 m then latch; fable t04 `turn_and_move` stop=sustained_contact).
3. Immediate signed backup: **opus t08 `move(-0.5)` → measured 0.004 m**; **sonnet t07 `move(-0.3)` → 0.001 m**; **fable t06 `move(-0.25)` → 0.002 m** — all `stop_reason=sustained_contact`, still multi-region contact.
4. Escalate to `send_velocity` (often `vx<0` first, then lateral/spin/forward mixes). Counts: opus 25, sonnet 16, fable 30 `send_velocity` in FK.
5. Near-zero bump streaks of length **11–33** motion events; net spawn→end displacement only ~0.5–0.9 m; path after first zero-bump ≈ 0.5–0.7 m over ~30 turns (creep, not escape).
6. Timeout at turn 40; kitchen never mapped.

### 4.2 Seed 102 — hallway west latch

Competent early mapping (bedroom → hallway) then west drive. Latch ~t23–t28 with head+torso contact. Same backup→SV pattern (opus 11 SV; sonnet only 5 SV but still times out). Opus end plan correctly reports cm-scale motion.

### 4.3 Fable seed 103 — bedroom doorway (seed is solvable)

Opus/Sonnet reach kitchen on 103. Fable walks east hallway, turns into bedroom doorway (~t09), latches, then 24× `send_velocity`. This cell shows the wedge is **not only “hard seeds”** — it is also a **recovery / anti-thrash** failure on a seed peers solve.

### 4.4 `send_velocity` as anti-pattern

Success cells in the companion note used **0** `send_velocity` (opus103/104, fable104). All six timeouts are SV-heavy. Models *do* try negative `vx` and signed `move`; SV is not always “forward into wall” — it is “anything, full duration, while latched,” which burns `policy_seconds` and turns for centimetres.

---

## 5. Harness code paths that produce this

| Mechanism | Where | Effect on wedge |
|---|---|---|
| Contact machine | `policy_wrapper._update_contact_state` | Debounce → `sustained_contact`; release needs clear run ≥ `CONTACT_SUSTAINED_STEPS`. |
| `move` / `turn_and_move` auto-stop | `execute(..., stop_on_bump=True)` break when state is already `sustained_contact` | **Pre-latched backup aborts after ~1 step** → measured mm. |
| `send_velocity` | `stop_on_bump=False` | Runs full clamped duration through contact; odom ≈ true ≈ cm; **`stop_reason` stays empty**. |
| Status assembly | `tools.status_payload` / `_record_motion` | Exposes `last_motion`, `current_contact`, legacy `bumped`/`contact`/`distance_moved_m`. No streak / blocked aggregate. |
| Prompt | `prompts.py` | Documents backup + obstruction; labels SV as escape hatch; no recovery doctrine / thrash warning. |
| One-motion-per-turn | agent loop (post F-04) | Already enforced in these batches (0 multi-motion violations in the Opus comparison note) — not the wedge cause. |

**Pre-latch abort (load-bearing):** after a prior command leaves `_contact_state == "sustained_contact"`, the next `move(-0.5)` still steps once, updates contact (still touching), hits `if stop_on_bump and self._contact_state == "sustained_contact": break` (`policy_wrapper.py` ~L1138–1141). Reverse velocity never gets a sustained window to clear the event. That contradicts the prompt’s “negative backs up” affordance whenever the robot is already latched — the common case at the start of recovery.

Design doc `05-agent-harness.html` §4.2 is **stale** relative to the live signed-`move` surface (it still narrates `move(distance≤0) → invalid_args` from T3.2). Live code + prompt are the contract agents actually used in R3.

---

## 6. Ranked improvement plan

Fairness axis: rule 4–5. SR axis: hypothesized effect on these six cells only (not a promise).

### Tier A — reporting-only (fair)

| ID | Change | Hypothesis on 6 timeouts | Fairness | Files | Acceptance idea | Freeze risk |
|---|---|---|---|---|---|---|
| **A1** | Add `status.progress` (or sibling): `no_progress`, `consecutive_no_progress`, `last_measured_m`, optional `hint` built only from distance_moved + bumped + current_contact — e.g. streak≥3 ⇒ `no_progress=true` | Low–medium alone (models often already know). High for *forensics* and for models that thrash before naming stuck. May cut a few late-SV turns if the streak is salient. | **Fair FIX** — formats what already happened; no path choice. | `tools.py` (`_record_motion` / `status_payload`), tests, design 05 | Unit: after 3 synthetic blocked motions, streak==3 and flag true; structural leak test still passes. | New freeze required (model-facing). |
| **A2** | On `send_velocity`, set `stop_reason` when call ends in `sustained_contact` / bumped (e.g. `completed_in_contact` or `sustained_contact`) so macros and SV share vocabulary | Low SR; removes asymmetry that makes SV look “cleaner” than a stopped macro. | **Fair FIX** — honest report of contact outcome. | `tools.py` and/or `policy_wrapper.execute` | 111/111 empty bumped-SV stop_reasons in these logs would become non-empty under replay of the rule. | New freeze. |
| **A3** | Perception-only turns: do not imply fresh motion progress; keep last_motion frozen (already mostly true) + surface streak in memory footer | Low | Fair | `memory.py` render | Snapshot string contains streak when ≥N | New freeze if memory text changes. |

### Tier B — mild scaffold (borderline fair)

| ID | Change | Hypothesis | Fairness | Files | Acceptance | Freeze risk |
|---|---|---|---|---|---|---|
| **B1** | Model-blind prompt note: if measured ≪ requested **and** contact persists across consecutive motions, prefer observe / replan / backed-out heading over repeating raw velocity; do **not** name models or seeds | Medium if combined with A1; alone, weak (101 models already plan STUCK-RECOVERY and keep SV). | Borderline — still “LEAVE navigating is hard” if it becomes a script. Keep to one short doctrine line, recorded in PLAN/AGENTS. | `prompts.py`, design 05 | Diff review: no model names; applies to all contestants | New freeze; comparison invalid vs R3. |
| **B2** | Tool-result tip **only when** streak≥N: e.g. `notes: ["no progress over N motions while in sustained contact"]` | Medium — same info as A1, more imperative | Borderline (formatting vs coaching) | `tools.py` | Tip absent when streak<N; present when ≥N | New freeze. |
| **B3** | Rising-edge / this-command sustained-contact abort for `move`/`turn_and_move`: do not abort solely because state was already latched *before* the command; require sustained contact (re)confirmed over the threshold **during** this command’s steps, *or* grant a short reverse grace window | **Highest fair SR impact.** Directly addresses mm-scale `move(-x)` no-ops. May free mild latches; may still fail 6 cm sofa pinch. | **FIX under rule 4** (macro advertised backup must be able to run). Not pathfinding. | `policy_wrapper.py` `execute`/`move`, tests, smoke | Unit/smoke: start latched against a face with free space behind; `move(-0.4)` yields measured ≫ 0.05 m and can reach `current_contact.state=free`. Forward still auto-stops on *new* sustained contact. | New freeze; must video-smoke (rule 11). |

B3 is listed under Tier B only because it changes motion behavior; fairness-wise it is closer to Tier A “make the tool honest.”

### Tier C — active assist (higher SR, weaker claim)

| ID | Change | Hypothesis | Fairness | Flag |
|---|---|---|---|---|
| **C1** | Auto-backup on bump (harness issues reverse without model) | High on mild contact; medium on true pinch | **Weaker claim** — harness decides recovery action | Optional / out-of-honest-scaffold |
| **C2** | Random escape / spiral / wall-follow | High variance SR | Measures our controller | Out of scope for portfolio “LLM-as-SLAM” |
| **C3** | `navigate_around` / geometric detour | High | Rule 5 violation | Reject for honest batch |
| **C4** | Scene edit: widen sofa–table gap | Would delete seed-101 shared trap | Changes task, not agent | Separate scene decision, not harness |

F-04 already asked for signed/dedicated backup and sustained-contact abort discipline (`V5D_R2_HARNESS_FORENSICS.md`). R3 shipped signed `move` + `turn_and_move` + one-motion-per-turn; **pre-latch interaction was left open** — that is the remaining F-04-shaped gap these traces expose.

---

## 7. Recommended first fix (one-by-one)

**Do B3 + A1 together as one freeze commit** (or B3 first if the owner insists on a single lever):

1. **B3 (primary):** Change auto-stop so a command that begins already in `sustained_contact` is allowed to run until *this command* re-establishes sustained contact after a documented policy (preferred: treat stop as rising-edge of sustained contact *during this call*, after at least `CONTACT_SUSTAINED_STEPS` of contact under the new command — equivalently, clear the “already latched ⇒ immediate break” path). Keep forward safety: driving deeper into a face must still stop.
2. **A1 (companion reporting):** Add `status.progress` with `consecutive_no_progress` using threshold ε≈0.05 m and `bumped`/`sustained_contact`, so even if backup fails physically, the streak is explicit and model-blind.

**Why not A-only first?** Traces show models often *already* know; A-only likely saves few of the six. Why not C? Violates the portfolio fairness claim. Why not prompt-only B1? Weak against demonstrated STUCK-RECOVERY + SV grind.

### Acceptance criteria (B3 + A1)

1. **Unit (no kit):** simulate contact state machine + `execute` stop predicate — pre-latched + reverse command does not break on step 0/1 solely due to prior latch; forward into sustained contact still stops within documented steps.
2. **Unit:** `status.progress.consecutive_no_progress` increments on blocked motions and resets after measured > ε without block.
3. **Regression:** existing tool payload leak tests + bump scoring tests green via `scripts/run_tests.sh`.
4. **Smoke (kit, rule 11):** scripted wedge: drive into a known furniture face, then `move(-0.5)` — filmstrip shows visible reverse clearance when space exists; mp4 + stills under `results/`.
5. **Replay note:** do **not** expect post-hoc JSON rewrite of R3; next batch under a new freeze/manifest only.
6. **Success metric for the *next* mini-cert:** on seed 101 (shared trap), at least one model escapes the aisle and reaches a non-living-room room *or* demonstrates ≥1.5 m post-latch true displacement without SV spam — directional only; N=4 remains fragile.

### Explicit non-goals for this first fix

- No auto path around furniture.
- No per-model prompt text.
- No claiming R3 scores would have changed retrospectively.

---

## 8. Expected effect matrix (hypothesis only)

| Cell | A1 alone | A2 alone | B3 alone | B3+A1 | C1 auto-backup |
|---|---|---|---|---|---|
| opus/sonnet/fable 101 sofa pinch | low | low | low–med (6 cm clearance; may still fail) | low–med | med if reverse opens slot |
| opus/sonnet 102 hallway latch | low | low | **med–high** (more free space if reverse works) | med–high | med–high |
| fable 103 bedroom doorway | low | low | **med–high** (peers solve seed) | med–high | med |

Honest bottom line: **B3 repairs a broken advertised recovery tool; it does not guarantee sofa-pinch solves.** Scene pinch (seed 101) may remain a layout hardness issue (C4 / seed design), separate from harness honesty.

---

## 9. Rule 4 table mapping

| Observation | Diagnosis | Action |
|---|---|---|
| `move(-0.5)` while latched returns ~0.00 m without a real reverse attempt | Harness macro no-op vs prompt | **FIX (B3)** |
| SV bumped with empty `stop_reason` | Reporting asymmetry | **FIX (A2)** |
| No streak / `blocked` aggregate | Reporting gap | **FIX (A1)** |
| Model keeps SV for 20 turns after STUCK plan | Model chose badly | **LEAVE** (measure); optional mild B1/B2 |
| Auto detour around sofa | Harness pathfinding | **Out of scope (C)** |

---

## 10. Pointers

- Traces: `results/raw_v5d_r3/{opus,sonnet}5_seed{101,102}.json`, `results/raw_v5d_r3_fable5/fable5_seed{101,103}.json`
- Scores: `results/scores_raw_v5d_r3.json`, `results/scores_raw_v5d_r3_fable5.json`
- Visual: `results/raw_v5d_r3_visual_audits/opus5_seed101.md`
- Prior forensics: `docs/research/V5D_R2_HARNESS_FORENSICS.md` §F-04
- Cross-model narrative: `docs/research/V5D_R3_OPUS_SUCCESS_VS_FAIL_TRACE_COMPARISON.md` §2.1 / §5

---

## 11. Measured results — companion `v5d-r3-opus5-b3a1` (2026-08-06)

**Batch:** Opus 5 × seeds 101–104 under B3+A1 freeze  
(`manifest_sha256=0d3dd82e81ff9b798704a3915e9ab4ca6ee22fd72f894d5a0bcaa547ca473738`,
config_hash `7260ee9a7889…`). Raw `results/raw_v5d_r3_opus5_b3a1/`; scores
`results/scores_raw_v5d_r3_opus5_b3a1.json`; report
[`V5D_R3_OPUS5_B3A1_PERFORMANCE_REPORT.md`](V5D_R3_OPUS5_B3A1_PERFORMANCE_REPORT.md).
Certifying L8 `results/raw_v5d_r3/` mtimes match
`results/incomplete/l8_opus5_mtime_pre_b3a1.txt` (untouched).

| Metric | L8 Opus (`raw_v5d_r3`) | B3+A1 companion |
|---|---|---|
| find_kitchen SR (v2) | **2 / 4** | **0 / 4** |
| return_home SR | **2 / 4** | **0 / 4** |
| falls | 0 | 0 |
| cost sum | $12.50 | $13.14 |

Per-seed (`distance_to_success_region_m` from each trial's
`final.stages.find_kitchen.score`):

| Seed | L8 | B3+A1 | Reading |
|---|---|---|---|
| 101 | timeout_turns, dist_region=1.616, bumps=31 | timeout_turns, dist_region=1.644, bumps=31 | **No rescue** of sofa-pinch turn-cap |
| 102 | timeout_turns, dist_region=2.079, bumps=17 | timeout_turns, dist_region=1.812, bumps=25 | **No rescue** of hallway latch turn-cap |
| 103 | **success** + return_home success, dist_region=0 | declared_elsewhere, dist_region=1.761, bumps=10 | **Regression** |
| 104 | **success** + return_home success, dist_region=0 | timeout_turns, dist_region=1.586, bumps=22 | **Regression** |

**Verdict (measured, not hypothesized):** B3+A1 did **not** mitigate the
furniture-wedge failure mode on the target timeout seeds, and it **lost** the
two L8 successes. Consistent with §7#4 smoke: stop-predicate honesty can PASS
while plant clearance under sustained press remains FAIL. Do **not** claim
wedge success. Explicit non-actions retained: no Tier C auto-backup / path
assist; no furniture-gap widening (C4 / scene LEAVE).
