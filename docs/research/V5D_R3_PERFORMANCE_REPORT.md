# V5D R3 Performance Report — L8 complete

**Status:** COMPLETE — **12/12** matrix cells, machine + visual audits PASS  
**Date:** 2026-08-04  
**Batch:** `v5d-r3-final-prod`  
**Manifest:** [`results/manifests/v5d-r3-final-prod.json`](../../results/manifests/v5d-r3-final-prod.json)  
**Manifest SHA-256:** `17a79cc37604c55119cd25a949858bb2d947db2ae7f1e7b57fb5e19500ac16cd`  
**Freeze commit:** `56bd08a68d922d205992679a403bfb577b5e2194`  
**Config hash:** `d30462d03c764c7c8520cae5910ea704eedbe94f96dfbf965d551ba6859ea46b`  
**Checkpoint SHA:** `301e24e336b2eab0ba387beb50fc16b03e6062b26622bc9a3e98588216a12c54`  
**Parent (Open_Duck_Mini_Jetson):** `7dde4ba952fb40c5ffb29441a1895f6f8863fdcc`  
**Raw dir:** `results/raw_v5d_r3/`  
**Criterion:** `v2_any_counter` (live and published; not post-batch widened)

---

## Headline

| Metric | Value | Source |
|---|---|---|
| Cells with `final` | **12 / 12** | `results/raw_v5d_r3/*.json` |
| `find_kitchen` SR (v2 any counter face) | **3 / 12** | `results/scores_raw_v5d_r3.json` |
| `find_kitchen` SR (pre-registered point disc) | **2 / 12** | same |
| `return_home` SR (unrun = failure) | **2 / 12** | same |
| `return_home` given stage-1 success | **2 / 3** | opus5 seeds 103–104; sonnet5_seed103 offered but declared elsewhere |
| Falls | **0 / 12** | same |
| Batch cost | **$26.56** | sum of `cost_usd` |
| Batch audit | **PASS** (12/12 machine + 12/12 visual) | `results/raw_v5d_r3_batch_audit.json` |

**Compared with the frozen v4 batch (policy `v4_robust`):** v4 scored **1/12** find_kitchen under the same v2 criterion and **10/12 falls**. This v5d-r3 batch uses the remediated harness + `v5d_contact_wrench` checkpoint; falls dropped to zero and kitchen success rose to 3/12. Batches are **not** a controlled ablation of a single lever.

---

## Per-model (N=4)

| Metric | sonnet5 | opus5 | gpt56sol |
|---|---|---|---|
| find_kitchen SR (v2) | 1/4 | **2/4** | 0/4 |
| find_kitchen SR (prereg) | 0/4 | 2/4 | 0/4 |
| return_home SR | 0/4 | **2/4** | 0/4 |
| return_home \| stage-1 success | 0/1 | **2/2** | — |
| mean progress (stage 1) | 0.263 | 0.471 | 0.320 |
| mean bumps | 17.5 | 15.0 | 18.0 |
| falls | 0 | 0 | 0 |
| mean QA | 0.275 | 0.625 | 0.375 |
| cost sum | $8.22 | $12.50 | $5.84 |

Source: `results/summary_table_raw_v5d_r3.md` / `results/scores_raw_v5d_r3.json`.

---

## Per-trial (stage-1 outcome = v2)

| Trial | find_kitchen | return_home | bumps | cost USD |
|---|---|---|---|---|
| sonnet5_seed101 | timeout_turns | not_run | 21 | 1.707 |
| sonnet5_seed102 | timeout_turns | not_run | 11 | 1.709 |
| sonnet5_seed103 | **success** (prereg: declared_elsewhere) | declared_elsewhere | 21 | 3.185 |
| sonnet5_seed104 | declared_elsewhere | not_run | 17 | 1.619 |
| opus5_seed101 | timeout_turns | not_run | 31 | 2.543 |
| opus5_seed102 | timeout_turns | not_run | 17 | 2.640 |
| opus5_seed103 | **success** | **success** | 8 | 4.755 |
| opus5_seed104 | **success** | **success** | 4 | 2.561 |
| gpt56sol_seed101 | declared_elsewhere | not_run | 25 | 1.692 |
| gpt56sol_seed102 | declared_elsewhere | not_run | 29 | 1.625 |
| gpt56sol_seed103 | declared_elsewhere | not_run | 14 | 1.574 |
| gpt56sol_seed104 | declared_elsewhere | not_run | 4 | 0.953 |

Opus 5 is the only model that both reached the kitchen and returned home (2/2 of its stage-1 successes). GPT 5.6 sol declared done in every cell without ever meeting the counter-face radius. Sonnet 5's single kitchen success (seed 103) is v2-only; its return-home declare was outside the home disc.

Forensic success-vs-fail trace comparison (within-Opus + same-seed cross-model, primary JSON): [`V5D_R3_OPUS_SUCCESS_VS_FAIL_TRACE_COMPARISON.md`](V5D_R3_OPUS_SUCCESS_VS_FAIL_TRACE_COMPARISON.md).

---

## Audit package

| Artifact | Result |
|---|---|
| Machine audits `results/raw_v5d_r3_audits/*.json` | **12/12 PASS** |
| Visual audits `results/raw_v5d_r3_visual_audits/*.md` | **12/12 PASS** (structured verdict fields) |
| Batch audit `results/raw_v5d_r3_batch_audit.json` | **PASS**, `publication_gate` 12/12 |
| Scores / summary | `results/scores_raw_v5d_r3.json`, `results/summary_table_raw_v5d_r3.md` |
| Videos | `results/videos_v5d_r3/<trial>.mp4` + `_filmstrip.png` |

---

## Resume history (credit block → completion)

1. **2026-08-03** — L8 paused at 9/12 after `gpt56sol_seed102` hit OpenAI 429 `credit_balance_exhausted` (quarantine `results/incomplete/gpt56sol_seed102.20260803-074608.json`; `results/rerun_log.md`).
2. **2026-08-03 later** — credits restored; same write-once manifest resumed into `results/raw_v5d_r3`. Cells 10–12 (`gpt56sol` seeds 102–104) wrote `final` (runner log: `results/incomplete/l8_resume_runner.log`, last turn ~2026-08-03T17:04:06Z).
3. **2026-08-04** — scores rebuilt for 12/12; remaining machine/visual audits completed; batch audit PASS; this report.

L7 certifying mini (`results/mini_v5d_r3/`, scores `results/scores_mini_v5d_r3.json`) remains the gate evidence for the same manifest; it is not part of the L8 matrix.

---

## Reproduction (audit / scores only — do not re-run successful trials)

```bash
DUCK_EMBODY_RAW_DIR=results/raw_v5d_r3 \
DUCK_EMBODY_MANIFEST=results/manifests/v5d-r3-final-prod.json \
~/IsaacLab/_isaac_sim/python.sh scripts/build_scores.py

python3 scripts/audit_batch.py \
  --batch-dir results/raw_v5d_r3 \
  --manifest results/manifests/v5d-r3-final-prod.json \
  --audit-dir results/raw_v5d_r3_visual_audits \
  --out results/raw_v5d_r3_batch_audit.json
```
