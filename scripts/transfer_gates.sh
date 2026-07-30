#!/usr/bin/env bash
# Run the duck-embody transfer gates for a candidate policy, unattended.
#
# Order matters and is not arbitrary:
#
#   1. replay the 10 frozen falls with the BASELINE (v4_robust, the policy the
#      frozen batch actually ran). This establishes which scenarios reproduce.
#      Any scenario the baseline survives is not a valid gate and the candidate
#      gets no credit for it.
#   2. replay them with the CANDIDATE, cross-checked against (1).
#   3. the T2.4 physics pass with the candidate.
#   4. the gap-hunt suite with the candidate. NOTE on scoring: S2 is a
#      forced-fall HARNESS diagnostic that passes only if the robot DOES fall,
#      so for a press-surviving candidate "S2 inconclusive/no fall" is the
#      desired policy outcome, not a regression. Score S0/S1/S3/S4/S5.
#
# One kit job at a time (AGENTS.md rule 1), hence strict serialisation.
#
# Usage: scripts/transfer_gates.sh <candidate_ckpt> <label> [reps]

set -uo pipefail

CAND="${1:?usage: transfer_gates.sh <candidate_ckpt> <label> [reps]}"
LABEL="${2:?}"
REPS="${3:-3}"

EMBODY=/home/xiaohui_chen/Projects/duck-embody
ISAACLAB=/home/xiaohui_chen/IsaacLab
BASE=$EMBODY/policy/model_2999.pt
LOGS=$EMBODY/results/logs
STAMP=$(date '+%Y%m%d-%H%M%S')

exec > >(tee -a "$LOGS/transfer_gates_${LABEL}_${STAMP}.log") 2>&1
echo "######## transfer gates: $LABEL ($(date '+%F %T')) ########"
echo "baseline : $BASE"
echo "candidate: $CAND"
echo "reps     : $REPS"

wait_for_gpu () {
    while pgrep -f "train""_ppo.py" >/dev/null 2>&1 \
       || pgrep -f "evaluate""_policies" >/dev/null 2>&1; do sleep 60; done
    sleep 15
}

run_kit () {  # tag, then command args
    local tag="$1"; shift
    echo "---- $tag ----"
    wait_for_gpu
    ( cd "$EMBODY" && PYTHONUNBUFFERED=1 "$ISAACLAB/isaaclab.sh" -p "$@" ) \
        >> "$LOGS/gate_${LABEL}_${tag}.log" 2>&1
    echo "$tag exit=$?"
    sleep 15
}

BASE_OUT=$LOGS/replay_baseline_${STAMP}
CAND_OUT=$LOGS/replay_${LABEL}_${STAMP}

# 1 + 2: the fall regression suite, baseline first.
run_kit replay_baseline scripts/replay_falls.py \
    --checkpoint "$BASE" --reps "$REPS" --out-dir "$BASE_OUT"

run_kit "replay_${LABEL}" scripts/replay_falls.py \
    --checkpoint "$CAND" --reps "$REPS" --out-dir "$CAND_OUT" \
    --baseline-report "$BASE_OUT/replay_falls_report.json"

# 3 + 4: the scripted physics gates with the candidate.
run_kit physics_pass scripts/smoke_physics_pass.py --checkpoint "$CAND"

BUDGET=$( cd "$EMBODY" && "$ISAACLAB/isaaclab.sh" -p scripts/smoke_gap_hunt.py --print-budget 2>/dev/null | tail -n1 )
echo "gap-hunt budget: ${BUDGET:-unknown}s"
run_kit gap_hunt scripts/smoke_gap_hunt.py --checkpoint "$CAND"

echo
echo "######## summary ########"
for f in "$BASE_OUT/replay_falls_report.json" "$CAND_OUT/replay_falls_report.json"; do
    [ -f "$f" ] || { echo "MISSING $f"; continue; }
    python3 - "$f" <<'PYS'
import json, sys
d = json.load(open(sys.argv[1]))
a = d.get("aggregate", {})
# `scenarios` is a LIST of dicts (replay_falls.py), and the aggregate exposes
# n_scored / scenarios_survived — not scenarios_run / label, which an earlier
# version guessed and which crashed this block with AttributeError while the
# surrounding shell reported success.
print(f"\n{d.get('checkpoint', '?')}")
print(f"  survived {a.get('scenarios_survived','?')}/{a.get('n_scored','?')}  "
      f"mean survival {a.get('mean_survival_fraction', float('nan')):.2f}")
for sc in d.get("scenarios", []):
    print(f"   {sc.get('trial_id','?'):20s} {sc.get('fall_mechanism',''):24s} "
          f"survival {sc.get('survival_fraction', 0):.2f}")
PYS
done
echo
echo "######## done $(date '+%F %T') ########"
