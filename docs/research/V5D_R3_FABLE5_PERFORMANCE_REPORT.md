# V5D R3 Fable 5 Companion — Performance Report

**Status:** COMPLETE — **4/4** cells, machine + visual audits PASS  
**Date:** 2026-08-04  
**Batch:** `v5d-r3-fable5` (**companion**; not the certifying L8 matrix)  
**Manifest:** [`results/manifests/v5d-r3-fable5.json`](../../results/manifests/v5d-r3-fable5.json)  
**Manifest SHA-256:** `1d4249e73fce0fc3b7f0104af64034828cfcf9244f2e8275b6d12983cf53b416`  
**Freeze commit:** `29e433f8ef141b7ce1e2988513c59eb93c323cf7`  
**Config hash:** `7fbce2573c184a10b42359bafc44cc084da94044d20b6e034b7d4b082fd67d10`  
**Checkpoint SHA:** `301e24e336b2eab0ba387beb50fc16b03e6062b26622bc9a3e98588216a12c54` (same as certifying L8)  
**Parent (Open_Duck_Mini_Jetson):** `7dde4ba952fb40c5ffb29441a1895f6f8863fdcc`  
**Raw dir:** `results/raw_v5d_r3_fable5/`  
**Criterion:** `v2_any_counter` (live)

Certifying L8 remains `v5d-r3-final-prod` (`results/raw_v5d_r3/`, report
[`V5D_R3_PERFORMANCE_REPORT.md`](V5D_R3_PERFORMANCE_REPORT.md)). This companion
uses the same harness code + checkpoint, but a **different** freeze set /
config_hash because the live matrix is `[fable5]` only (expected drift vs L8
`d30462d0…`).

---

## Headline

| Metric | Value | Source |
|---|---|---|
| Cells with `final` | **4 / 4** | `results/raw_v5d_r3_fable5/*.json` |
| `find_kitchen` SR (v2 any counter face) | **1 / 4** | `results/scores_raw_v5d_r3_fable5.json` |
| `find_kitchen` SR (pre-registered point disc) | **0 / 4** | same |
| `return_home` SR (unrun = failure) | **1 / 4** | same |
| `return_home` given stage-1 success | **1 / 1** | fable5_seed104 |
| Falls | **0 / 4** | same |
| Batch cost | **$27.37** | sum of `cost_usd` |
| Batch audit | **PASS** (4/4 machine + 4/4 visual) | `results/raw_v5d_r3_fable5_batch_audit.json` |

---

## Per-trial (stage-1 outcome = v2)

| Trial | find_kitchen | return_home | bumps | cost USD |
|---|---|---|---|---|
| fable5_seed101 | timeout_turns | not_run | 34 | 8.751 |
| fable5_seed102 | declared_elsewhere | not_run | 17 | 7.678 |
| fable5_seed103 | timeout_turns | not_run | 27 | 8.345 |
| fable5_seed104 | **success** (prereg: declared_elsewhere) | **success** | 4 | 2.596 |

Source: `results/summary_table_raw_v5d_r3_fable5.md` / `results/scores_raw_v5d_r3_fable5.json`.

---

## Comparison vs certifying L8 (same seeds; companion label)

L8 per-model (N=4 each) from `docs/research/V5D_R3_PERFORMANCE_REPORT.md` /
`results/scores_raw_v5d_r3.json`:

| Metric | fable5 (companion) | sonnet5 (L8) | opus5 (L8) | gpt56sol (L8) |
|---|---|---|---|---|
| find_kitchen SR (v2) | **1/4** | 1/4 | **2/4** | 0/4 |
| find_kitchen SR (prereg) | 0/4 | 0/4 | 2/4 | 0/4 |
| return_home SR | **1/4** | 0/4 | **2/4** | 0/4 |
| return_home \| stage-1 success | **1/1** | 0/1 | **2/2** | — |
| falls | 0 | 0 | 0 | 0 |
| cost sum | $27.37 | $8.22 | $12.50 | $5.84 |

Notes:

- Fable 5 pricing is $10/$50 per MTok vs Sonnet 5's cheaper rates — cost is
  **not** a capability comparison.
- Config hashes differ (`7fbce257…` vs L8 `d30462d0…`) because the freeze matrix
  includes `fable5.yaml` and the live model list is fable5-only. Checkpoint SHA
  and parent commit match L8.
- On the same seeds, Fable 5 matches Sonnet 5's v2 kitchen rate (1/4) and is the
  only non-Opus cell here that also completed return_home (seed 104). Opus 5
  remains the strongest L8 contestant (2/4 kitchen + 2/2 return).

---

## Audit package

| Artifact | Result |
|---|---|
| Machine audits `results/raw_v5d_r3_fable5_audits/*.json` | **4/4 PASS** |
| Visual audits `results/raw_v5d_r3_fable5_visual_audits/*.md` | **4/4 PASS** |
| Batch audit `results/raw_v5d_r3_fable5_batch_audit.json` | **PASS**, `publication_gate` 4/4 |
| Scores / summary | `results/scores_raw_v5d_r3_fable5.json`, `results/summary_table_raw_v5d_r3_fable5.md` |

---

## Reproduce scores (offline; no Isaac launch)

```bash
cd /home/xiaohui_chen/Projects/duck-embody
PYTHONPATH=. DUCK_EMBODY_RAW_DIR=results/raw_v5d_r3_fable5 \
DUCK_EMBODY_MANIFEST=results/manifests/v5d-r3-fable5.json \
  python3 scripts/build_scores.py
```
