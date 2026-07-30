#!/usr/bin/env bash
# Resume the v5d transfer gates after the 2026-07-29 fixes.
#
# WHAT WENT WRONG AND WHAT THIS RE-RUNS
# -------------------------------------
# The first chain produced a good baseline replay (9 of 10 scenarios reproduce
# their original fall; only opus5_seed104 does not) and then wasted its
# candidate step: my wrapper passed --baseline-report with the wrong filename
# (`replay_report.json`; the script writes `replay_falls_report.json`), so
# replay_falls.py exited 1 before running a single scenario. The baseline
# replay is unaffected and is REUSED here rather than repeated.
#
# It also surfaced a real defect in the repo's own physics gate: `wall_bump`
# started at x=2.55, which is the exact centre of the hallway<->kitchen doorway
# (width 0.35), while the test called itself "off-doorway". A policy that walks
# straight therefore transits the gap and "fails" a wall-collision check. Fixed
# to x=1.70 (inside wall segment A2). Because that changes a pass criterion,
# the corrected test is run for BOTH policies so the comparison is fair and the
# baseline's number is re-earned rather than inherited.
#
# Usage: scripts/resume_gates.sh <candidate_ckpt> <label> <baseline_report_json> [reps]

set -uo pipefail

CAND="${1:?usage: resume_gates.sh <cand_ckpt> <label> <baseline_report> [reps]}"
LABEL="${2:?}"
BASE_REPORT="${3:?}"
REPS="${4:-3}"

EMBODY=/home/xiaohui_chen/Projects/duck-embody
ISAACLAB=/home/xiaohui_chen/IsaacLab
BASE=$EMBODY/policy/model_2999.pt
LOGS=$EMBODY/results/logs
STAMP=$(date '+%Y%m%d-%H%M%S')
GATES_LOG=$LOGS/transfer_gates_${LABEL}_${STAMP}.log

exec > >(tee -a "$GATES_LOG") 2>&1
echo "######## resume gates: $LABEL ($(date '+%F %T')) ########"
echo "reusing baseline replay: $BASE_REPORT"

wait_for_gpu () {
    while pgrep -f "train""_ppo.py" >/dev/null 2>&1 \
       || pgrep -f "evaluate""_policies" >/dev/null 2>&1; do sleep 60; done
    sleep 15
}

run_kit () {
    local tag="$1"; shift
    echo "---- $tag ----"
    wait_for_gpu
    ( cd "$EMBODY" && PYTHONUNBUFFERED=1 "$ISAACLAB/isaaclab.sh" -p "$@" ) \
        >> "$LOGS/gate_${LABEL}_${tag}.log" 2>&1
    echo "$tag exit=$?"
    sleep 15
}

CAND_OUT=$LOGS/replay_${LABEL}_${STAMP}

# 1. The candidate's fall-replay run — the step that never happened.
run_kit "replay_${LABEL}" scripts/replay_falls.py \
    --checkpoint "$CAND" --reps "$REPS" --out-dir "$CAND_OUT" \
    --baseline-report "$BASE_REPORT"

# 2. Corrected physics gate, candidate then baseline. The baseline run is the
#    control that makes the corrected wall_bump check interpretable at all.
run_kit physics_pass_cand scripts/smoke_physics_pass.py --checkpoint "$CAND"
cp -f "$EMBODY/results/figures/smoke/physics_pass_report.json" \
      "$LOGS/physics_pass_${LABEL}_${STAMP}.json" 2>/dev/null

run_kit physics_pass_base scripts/smoke_physics_pass.py --checkpoint "$BASE"
cp -f "$EMBODY/results/figures/smoke/physics_pass_report.json" \
      "$LOGS/physics_pass_baseline_${STAMP}.json" 2>/dev/null

# Restore the candidate's report as the live one, since auto_pipeline reads it.
cp -f "$LOGS/physics_pass_${LABEL}_${STAMP}.json" \
      "$EMBODY/results/figures/smoke/physics_pass_report.json" 2>/dev/null

echo
echo "######## summary ########"
python3 - "$BASE_REPORT" "$CAND_OUT/replay_falls_report.json" \
         "$LOGS/physics_pass_baseline_${STAMP}.json" \
         "$LOGS/physics_pass_${LABEL}_${STAMP}.json" <<'PY'
import json, sys

def load(p):
    try: return json.load(open(p))
    except Exception: return None

base, cand, ppb, ppc = (load(p) for p in sys.argv[1:5])

if base and cand:
    ba, ca = base["aggregate"], cand["aggregate"]
    ids = [s["trial_id"] for s in base["scenarios"]]
    nr = set(ba.get("scenarios_not_reproducing_ids") or [])
    valid = [i for i in ids if i not in nr]
    surv = set(ca.get("scenarios_survived_ids") or [])
    credited = [i for i in valid if i in surv]
    print(f"\nFALL REGRESSION SUITE")
    print(f"  valid gates (baseline still falls): {len(valid)}/10   "
          f"excluded: {sorted(nr) or 'none'}")
    print(f"  candidate survived: {len(credited)}/{len(valid)} "
          f"({len(credited)/max(1,len(valid)):.0%})")
    print(f"  survived ids: {sorted(credited) or 'none'}")
    cmap = {s['trial_id']: s for s in cand['scenarios']}
    for i in valid:
        s = cmap.get(i, {})
        print(f"    {i:20s} {s.get('fall_mechanism','?'):24s} "
              f"survivals={s.get('survivals','?')}")

for tag, pp in (("baseline", ppb), ("candidate", ppc)):
    if pp:
        print(f"\nPHYSICS PASS ({tag}, corrected wall_bump): {pp.get('acceptance')}")
PY

echo
echo "######## done $(date '+%F %T') ########"
