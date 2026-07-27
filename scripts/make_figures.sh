#!/usr/bin/env bash
# Regenerate EVERY report figure from results/raw/*.json alone.
#
#   bash scripts/make_figures.sh              # all figures into results/figures/
#   bash scripts/make_figures.sh --out /tmp/x # elsewhere (extra args pass through)
#
# What it does (one command, reproducible):
#   1. Scoring sweep: `duck_embody.charts` main() loads every complete trial
#      JSON under results/raw/ and re-scores it with duck_embody.scoring
#      (score_trial + metric_estimates — the very functions summarise() and the
#      published tables are built from, so figures and tables cannot disagree).
#      Nothing is read from scores.json or any other derived artefact.
#   2. Figures:
#        results/figures/per_metric_bars.png        (doc 06 §10.2 grid)
#        results/figures/turns_survived.png         (per-trial end reasons)
#        results/figures/trajectory_vs_belief_<trial>.png
#           one per model, "richest" = most stage-1 turns (auto rule; on the
#           frozen batch: fable5_seed102, opus5_seed102, gpt56sol_seed103).
#           Override: --trajectories a,b,c   or   --trajectories none
#
# Interpreter: the KIT python (AGENTS.md §4 — the one interpreter policy).
# Its bundled matplotlib 3.10.3 renders headless (Agg). This does NOT launch
# Isaac Sim / touch the GPU — it is a plain python process over JSON files —
# but keep the one-kit-at-a-time habit (AGENTS.md rule 1) if a sim job runs.
#
# Determinism: bootstrap CIs come from configs/benchmark.yaml scoring:
# (fixed seed + resample count), so byte-stable numbers; PNG bytes may differ
# across matplotlib versions, the drawn values do not.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

KIT_PYTHON="${KIT_PYTHON:-$HOME/IsaacLab/_isaac_sim/python.sh}"

exec "$KIT_PYTHON" -m duck_embody.charts results/raw "$@"
