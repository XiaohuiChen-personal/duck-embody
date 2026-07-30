#!/usr/bin/env bash
# Re-measure the T1.3 locomotion constants for a candidate policy, with a v4
# control run to validate the measurement itself.
#
# WHY (gap-hunt S5, 2026-07-29)
# -----------------------------
# configs/benchmark.yaml's locomotion block is not harness configuration, it is
# a MEASUREMENT OF v4_robust: k_velocity_realisation 1.004,
# turn_rate_realisation 0.982, open_loop_yaw_drift_deg_per_s 1.83 — the last of
# which the file itself justifies with v4's wz tracking error of 0.067 rad/s.
# v5d's is 0.098 rad/s, 46% higher.
#
# The harness dead-reckons the model's believed position from COMMANDED
# velocity, so belief-vs-truth drift is set by the policy. Benchmark success is
# `declare_done` within 0.35 m of a counter, judged on TRUE position while the
# model reasons on the belief. Ship a new policy against the old constants and
# the batch measures the mismatch, not the policy. gap-hunt S5 already
# demonstrated the failure end to end: its scripted navigator declared on a
# drifted belief and scored `declared_elsewhere`.
#
# The v4 control matters as much as the v5d run: if this script cannot
# reproduce the frozen 1.004 / 0.982 / 1.83 for v4, then any v5d number it
# produces is unusable and the differences mean nothing.
#
# Usage: scripts/calibrate.sh <candidate_ckpt> <label>

set -uo pipefail

CAND="${1:?usage: calibrate.sh <candidate_ckpt> <label>}"
LABEL="${2:?}"

EMBODY=/home/xiaohui_chen/Projects/duck-embody
ISAACLAB=/home/xiaohui_chen/IsaacLab
BASE=$EMBODY/policy/model_2999.pt
LOGS=$EMBODY/results/logs

exec > >(tee -a "$LOGS/calibrate_${LABEL}.log") 2>&1
echo "######## calibrate $LABEL  $(date '+%F %T') ########"

wait_for_gpu () {
    while pgrep -f "replay_falls\.py" >/dev/null 2>&1 \
       || pgrep -f "smoke_gap_hunt\.py" >/dev/null 2>&1 \
       || pgrep -f "smoke_physics_pass\.py" >/dev/null 2>&1 \
       || pgrep -f "train""_ppo.py" >/dev/null 2>&1; do sleep 60; done
    sleep 15
}

measure () {  # label, checkpoint
    local lbl="$1"; local ckpt="$2"
    echo "---- measuring $lbl ----"
    wait_for_gpu
    ( cd "$EMBODY" && PYTHONUNBUFFERED=1 "$ISAACLAB/isaaclab.sh" -p \
        scripts/smoke_displacement.py --checkpoint "$ckpt" \
        --out-json "$LOGS/calibration_${lbl}.json" ) \
        >> "$LOGS/calibrate_${lbl}_run.log" 2>&1
    echo "$lbl exit=$?"
    sleep 15
}

# Control first: if v4 does not reproduce the frozen numbers, stop.
measure baseline "$BASE"
measure "$LABEL" "$CAND"

echo
echo "######## comparison vs the FROZEN constants ########"
# Unit conversion and the control check live in derive_calibration.py. The
# earlier inline version compared deg/20s against deg/s and a raw heading
# change against a ratio, and so reported a false "CONTROL FAILED" — the exact
# failure mode that stops a good unattended run for no reason.
python3 scripts/derive_calibration.py \
    "$LOGS/calibration_baseline.json" "$LOGS/calibration_${LABEL}.json"
echo "derive_calibration exit=$?"
echo "######## done $(date '+%F %T') ########"
