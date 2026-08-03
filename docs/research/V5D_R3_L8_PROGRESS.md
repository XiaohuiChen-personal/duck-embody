# V5D R3 L8 Progress — Partial Evidence Package

**Status:** PARTIAL — **9/12** matrix cells complete. **Not** L8 complete.  
**Date:** 2026-08-03  
**Batch:** `v5d-r3-final-prod`  
**Manifest:** `results/manifests/v5d-r3-final-prod.json`  
**Manifest SHA-256:** `17a79cc37604c55119cd25a949858bb2d947db2ae7f1e7b57fb5e19500ac16cd`  
**Freeze commit:** `56bd08a68d922d205992679a403bfb577b5e2194`  
**Config hash:** `d30462d03c764c7c8520cae5910ea704eedbe94f96dfbf965d551ba6859ea46b`  
**Raw dir:** `results/raw_v5d_r3/`  
**Criterion:** `v2_any_counter` (from `duck_embody.scoring.SUCCESS_CRITERION`)

---

## Headline (complete cells only)

| Metric | Value | Source |
|---|---|---|
| Cells with `final` | 9 / 12 | `results/raw_v5d_r3/*.json` |
| Manifest bind (9/9) | PASS — all `config.batch_manifest_sha256` == expected | trial JSON `config` |
| `find_kitchen` success (v2) | **3 / 9** | `results/scores_raw_v5d_r3_partial.json` |
| `find_kitchen` success (preregistered) | **2 / 9** | same (`success_preregistered`) |
| `return_home` success | **2 / 9** (only offered after stage-1 success) | same |
| Partial cost sum | **$22.410516** | `final.tokens.cost_usd_estimate` via scorer |
| Batch audit | **INCOMPLETE** (expected) | `results/audits_v5d_r3/batch_audit.json` |
| Machine audit (9 present) | **9/9 PASS** | same |
| Visual publication gate | **0/12 written** (all present cells visual INCOMPLETE) | same `publication_gate` |

---

## Complete trials (9)

Outcomes from canonical `duck_embody.scoring.score_trial` +
`duck_embody.forensics.correction_summary` (artifact:
`results/scores_raw_v5d_r3_partial.json`). Drift = stage-1 `drift_m`.
Corrections = forensic call counts (accepted / rejected).

| Trial | find_kitchen outcome | FK success (v2) | return_home | RH success | end_reason (FK) | cost USD | corrections calls (acc/rej) | drift_m (FK) | machine audit |
|---|---|---|---|---|---|---|---|---|---|
| sonnet5_seed101 | timeout_turns | false | not_run | false | turn_cap | 1.7068 | 0 (0/0) | 0.0811 | PASS |
| sonnet5_seed102 | timeout_turns | false | not_run | false | turn_cap | 1.7093 | 0 (0/0) | 0.1961 | PASS |
| sonnet5_seed103 | success | **true** | declared_elsewhere | false | declare_done | 3.1845 | 0 (0/0) | 0.0634 | PASS |
| sonnet5_seed104 | declared_elsewhere | false | not_run | false | declare_done | 1.6186 | 0 (0/0) | 0.2381 | PASS |
| opus5_seed101 | timeout_turns | false | not_run | false | turn_cap | 2.5434 | 0 (0/0) | 0.0997 | PASS |
| opus5_seed102 | timeout_turns | false | not_run | false | turn_cap | 2.6404 | 0 (0/0) | 0.2104 | PASS |
| opus5_seed103 | success | **true** | success | **true** | declare_done | 4.7549 | 1 (1/0) | 0.1149 | PASS |
| opus5_seed104 | success | **true** | success | **true** | declare_done | 2.5606 | 2 (2/0) | 0.1997 | PASS |
| gpt56sol_seed101 | declared_elsewhere | false | not_run | false | declare_done | 1.6920 | 0 (0/0) | 0.0982 | PASS |

Notes:

- sonnet5_seed103 stage-1 is v2 success but **preregistered failure** (counts in the 3 vs 2 split above).
- opus5 is the only model with stage-2 successes so far (seeds 103 and 104).
- Correction call counts are forensic (`correction_summary`); stage metric `corrections` on RH for opus5_seed103/104 is 1 and 2 respectively (accepted loop closures).

---

## Missing trials (3) — credit block

| Trial | Status | Evidence |
|---|---|---|
| gpt56sol_seed102 | Infra quarantine — no `final` in raw | `results/incomplete/gpt56sol_seed102.20260803-074608.json` (`infra_failure`: OpenAI 429 `credit_balance_exhausted` / `insufficient_quota`); logged in `results/rerun_log.md` |
| gpt56sol_seed103 | Never started | absent from `results/raw_v5d_r3/` |
| gpt56sol_seed104 | Never started | absent from `results/raw_v5d_r3/` |

Live credit probe (still blocked): `results/incomplete/l8_resume_watch.json`
(`probe_ok: false`, `http: 429`, `code: credit_balance_exhausted`).

Do **not** delete the incomplete quarantine. Do **not** invent a new manifest
unless a frozen file changes.

---

## Videos under `results/videos_v5d_r3/`

| Asset | Present |
|---|---|
| sonnet5_seed101.mp4 + `_filmstrip.png` | yes |
| sonnet5_seed102.mp4 + `_filmstrip.png` | yes |
| sonnet5_seed103.mp4 + `_filmstrip.png` | yes |
| sonnet5_seed104.mp4 + `_filmstrip.png` | yes |
| opus5_seed101.mp4 + `_filmstrip.png` | yes |
| opus5_seed102.mp4 + `_filmstrip.png` | yes |
| opus5_seed103.mp4 + `_filmstrip.png` | yes |
| opus5_seed104.mp4 + `_filmstrip.png` | yes |
| gpt56sol_seed101.mp4 + `_filmstrip.png` | yes |
| gpt56sol_seed102.mp4 + `_filmstrip.png` | yes (from aborted/quarantined run — **no** complete JSON in raw) |
| gpt56sol_seed103 / 104 video | **no** |

---

## Tooling disposition

| Tool | Result |
|---|---|
| `scripts/build_scores.py` with `DUCK_EMBODY_RAW_DIR=results/raw_v5d_r3` | **Refused** — `FileNotFoundError` on missing `gpt56sol_seed102.json`. Did **not** write `results/scores.json` or `results/scores_raw_v5d_r3.json`. |
| One-shot via `duck_embody.scoring.score_trial` / `summarise` | Wrote `results/scores_raw_v5d_r3_partial.json` (`schema: duck-embody-scores-v2-partial`, disposition PARTIAL). Named sources inside `disposition.sources`. |
| `scripts/audit_batch.py --batch-dir results/raw_v5d_r3 --manifest results/manifests/v5d-r3-final-prod.json --audit-dir results/audits_v5d_r3 --out results/audits_v5d_r3/batch_audit.json` | Batch **INCOMPLETE**; 9 present trials machine **PASS**; visual gate 0/12. |

`results/scores.json` (v4 publication) left untouched (mtime 2026-07-29).

---

## Resume command (after OpenAI credits restored)

Same write-once batch id — runner skips the 9 finished cells:

```bash
PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p duck_embody/runner.py \
  --batch-id v5d-r3-final-prod \
  --checkpoint /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson/exported_policies/v5d_contact_wrench_ppo/model_5998.pt \
  --calibration /home/xiaohui_chen/Projects/duck-embody/results/calibrations/v5d_contact_wrench.json \
  --out-dir results/raw_v5d_r3 \
  --video-dir results/videos_v5d_r3
```

Source of command: `docs/PLAN.md` TR.9 L8 block; pause rationale in
`results/rerun_log.md` (§ 2026-08-03 credit exhaustion).

Hard L8 completion still requires 12/12 JSON + machine audits + visual audits
with zero pending fields — this note is evidence for the **finished subset only**.
