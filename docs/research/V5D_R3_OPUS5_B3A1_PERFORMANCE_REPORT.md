# V5D R3 Opus 5 B3+A1 Companion — Performance Report

**Status:** COMPLETE — **4/4** cells, machine + visual audits PASS  
**Date:** 2026-08-06  
**Batch:** `v5d-r3-opus5-b3a1` (**companion**; not the certifying L8 matrix)  
**Manifest:** [`results/manifests/v5d-r3-opus5-b3a1.json`](../../results/manifests/v5d-r3-opus5-b3a1.json)  
**Manifest SHA-256:** `0d3dd82e81ff9b798704a3915e9ab4ca6ee22fd72f894d5a0bcaa547ca473738`  
**Freeze commit:** `fb134f29dafd5e5424a2fc770ec5a054d2dfdbe9`  
**Config hash:** `7260ee9a788968fb2f85c622a88114c34cd89cc480cb5bae20b6afb609d3b02b`  
**Checkpoint SHA:** `301e24e336b2eab0ba387beb50fc16b03e6062b26622bc9a3e98588216a12c54` (same as certifying L8)  
**Parent (Open_Duck_Mini_Jetson):** `7dde4ba952fb40c5ffb29441a1895f6f8863fdcc`  
**Raw dir:** `results/raw_v5d_r3_opus5_b3a1/`  
**Criterion:** `v2_any_counter` (live)

Certifying L8 remains `v5d-r3-final-prod` (`results/raw_v5d_r3/`, report
[`V5D_R3_PERFORMANCE_REPORT.md`](V5D_R3_PERFORMANCE_REPORT.md)). This companion
measures the B3+A1 furniture-wedge harness (rising-edge / reverse grace +
`status.progress`) against historical L8 Opus outcomes on the same seeds.
Config hash differs from L8 (`d30462d0…`) because the live freeze includes the
B3+A1 harness files and an opus5-only matrix — expected drift.

Smoke pre-batch (2026-08-05): B3 stop-predicate **PASS**; physical sofa
clearance **FAIL** (plant-limited) — see
[`V5D_R3_FURNITURE_WEDGE_HARNESS_IMPROVEMENTS.md`](V5D_R3_FURNITURE_WEDGE_HARNESS_IMPROVEMENTS.md).
Clearance FAIL was recorded as LEAVE (no scene / Tier C change).

---

## Headline

| Metric | Value | Source |
|---|---|---|
| Cells with `final` | **4 / 4** | `results/raw_v5d_r3_opus5_b3a1/*.json` |
| `find_kitchen` SR (v2 any counter face) | **0 / 4** | `results/scores_raw_v5d_r3_opus5_b3a1.json` |
| `find_kitchen` SR (pre-registered point disc) | **0 / 4** | same |
| `return_home` SR (unrun = failure) | **0 / 4** | same |
| `return_home` given stage-1 success | **—** (0 stage-1 successes) | same |
| Falls | **0 / 4** | same |
| Batch cost | **$13.14** | sum of `cost_usd` |
| Batch audit | **PASS** (4/4 machine + 4/4 visual) | `results/raw_v5d_r3_opus5_b3a1_batch_audit.json` |

---

## Per-trial (stage-1 outcome = v2)

| Trial | find_kitchen | return_home | bumps | dist_region_m | cost USD |
|---|---|---|---|---|---|
| opus5_seed101 | timeout_turns | not_run | 31 | 1.644 | 2.865 |
| opus5_seed102 | timeout_turns | not_run | 25 | 1.812 | 3.762 |
| opus5_seed103 | declared_elsewhere | not_run | 10 | 1.761 | 3.562 |
| opus5_seed104 | timeout_turns | not_run | 22 | 1.586 | 2.952 |

Source: `results/summary_table_raw_v5d_r3_opus5_b3a1.md` /
`results/scores_raw_v5d_r3_opus5_b3a1.json`;
`distance_to_success_region_m` from each trial's
`final.stages.find_kitchen.score`.

---

## Before / after wedge mitigation vs certifying L8 Opus

Same seeds; L8 paths `results/raw_v5d_r3/opus5_seed{101..104}.json`
(mtime baseline `results/incomplete/l8_opus5_mtime_pre_b3a1.txt` — **unchanged**).
L8 distances from the same `score.distance_to_success_region_m` field;
L8 outcomes from trial `final.stages.find_kitchen`.

| Seed | L8 find_kitchen | L8 dist_region_m | B3+A1 find_kitchen | B3+A1 dist_region_m | Verdict |
|---|---|---|---|---|---|
| 101 | timeout_turns (bumps=31) | 1.616 | timeout_turns (bumps=31) | 1.644 | **No mitigation** of target failure mode (still turn-cap wedge) |
| 102 | timeout_turns (bumps=17) | 2.079 | timeout_turns (bumps=25) | 1.812 | **No mitigation** (still turn-cap; slightly closer region dist, more bumps) |
| 103 | **success** + return_home success | 0.0 | declared_elsewhere | 1.761 | **Regression** vs L8 success |
| 104 | **success** + return_home success | 0.0 | timeout_turns | 1.586 | **Regression** vs L8 success |

### Honest wedge verdict

**B3+A1 did not improve Opus find_kitchen SR on this N=4 companion**
(L8 Opus was **2/4**; companion is **0/4**).

1. **Seeds 101–102 (the timeout / wedge targets):** both remain `timeout_turns`.
   End poses stay in the same living-room sofa pinch (101) and hallway latch
   (102) classes. The harness fix does not convert these turn-cap failures into
   kitchen reaches. This matches the pre-batch smoke: stop-predicate honesty
   improved; plant clearance under sustained furniture press remained FAIL
   (LEAVE — no Tier C / furniture-gap edit).
2. **Seeds 103–104 (L8 successes):** both **regressed**. Seed 103 declared at
   the kitchen/living doorway threshold (`true_xy≈(2.55, 2.86)`,
   `dist_region≈1.76 m`) instead of at a counter face. Seed 104 timed out
   wedged in the living-room sofa/table gap (`true_xy≈(1.00, 1.91)`).

Do **not** claim wedge success. The measured story is: B3+A1 repairs the
advertised reverse / progress reporting contract, but does not unlock the
plant-limited clearance failure, and on this draw it also lost the two L8
successes.

---

## Audit package

| Artifact | Result |
|---|---|
| Machine audits `results/raw_v5d_r3_opus5_b3a1_audits/*.json` | **4/4 PASS** |
| Visual audits `results/raw_v5d_r3_opus5_b3a1_visual_audits/*.md` | **4/4 PASS** |
| Batch audit `results/raw_v5d_r3_opus5_b3a1_batch_audit.json` | **PASS**, `publication_gate` 4/4 |
| Scores / summary | `results/scores_raw_v5d_r3_opus5_b3a1.json`, `results/summary_table_raw_v5d_r3_opus5_b3a1.md` |

---

## Reproduce scores (offline; no Isaac launch)

```bash
cd /home/xiaohui_chen/Projects/duck-embody
PYTHONPATH=. DUCK_EMBODY_RAW_DIR=results/raw_v5d_r3_opus5_b3a1 \
DUCK_EMBODY_MANIFEST=results/manifests/v5d-r3-opus5-b3a1.json \
  python3 scripts/build_scores.py
```

Machine + publication audit:

```bash
PYTHONPATH=. python3 scripts/audit_batch.py \
  --batch-dir results/raw_v5d_r3_opus5_b3a1 \
  --manifest results/manifests/v5d-r3-opus5-b3a1.json \
  --audit-dir results/raw_v5d_r3_opus5_b3a1_visual_audits \
  --out results/raw_v5d_r3_opus5_b3a1_batch_audit.json
```
