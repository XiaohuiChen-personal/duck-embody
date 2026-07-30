#!/usr/bin/env bash
# Fully autonomous remainder of the v5 evaluation: wait for the transfer gates,
# decide from their reports, run the paid benchmark if they pass, score it.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# Every earlier stall in this project came from the same shape: a long GPU job
# finished, and the next step waited for a human to notice. The gates, the
# benchmark and the scoring are all mechanical once the acceptance rule is
# written down, so they are written down here instead of being re-decided each
# time.
#
# TWO REPO FACTS THIS SCRIPT IS BUILT AROUND (both verified, both would silently
# ruin an unattended run):
#
#   1. `duck_embody.runner` skips a trial iff its JSON already exists in the
#      output dir. The 12 baseline trials live in results/raw, so pointing a new
#      batch there would run ZERO trials and look like instant success.
#      => the candidate batch goes to results/raw_<label>/.
#   2. `scripts/build_scores.py` hardcodes RAW = results/raw, so it cannot score
#      a batch in another directory. => the headline metric is computed here,
#      directly from the trial JSONs, and full build_scores parity is left as a
#      follow-up rather than faked.
#
# The benchmark SPENDS REAL MONEY (~$10 of LLM API calls). It is therefore gated
# on the transfer gates passing, and the gate decision is logged before the
# first paid call.
#
# Usage: scripts/auto_pipeline.sh <candidate_ckpt> <label> <gates_task_logfile>

set -uo pipefail

CAND="${1:?usage: auto_pipeline.sh <candidate_ckpt> <label> <gates_logfile>}"
LABEL="${2:?}"
GATES_LOG="${3:?}"

EMBODY=/home/xiaohui_chen/Projects/duck-embody
ISAACLAB=/home/xiaohui_chen/IsaacLab
LOGS=$EMBODY/results/logs
RAW_OUT=$EMBODY/results/raw_${LABEL}
VID_OUT=$EMBODY/results/videos_${LABEL}
SUMMARY=$EMBODY/results/${LABEL}_benchmark_summary.md

# v4_robust's frozen headline, the number to beat (results/scores.json).
BASELINE_FALLS_PER_MIN=1.58
TARGET_FALLS_PER_MIN=0.30
# The frozen matrix: 3 models x 4 seeds. Read from the config rather than
# assumed, so a matrix change cannot silently shrink the completeness check.
EXPECTED_TRIALS=$(python3 -c "
import re
src = open('$EMBODY/configs/benchmark.yaml').read()
m = re.search(r'^models:\s*\[([^]]*)\]', src, re.M)
s = re.search(r'^seeds:\s*\[([^]]*)\]', src, re.M)
print(len(m.group(1).split(',')) * len(s.group(1).split(','))) if m and s else print(12)
" 2>/dev/null || echo 12)

exec > >(tee -a "$LOGS/auto_pipeline_${LABEL}.log") 2>&1
echo "######## auto_pipeline $LABEL  $(date '+%F %T') ########"

# ----------------------------------------------------------------------
# 0. Wait for the transfer-gate chain to finish.
# ----------------------------------------------------------------------
echo "[auto] waiting for transfer gates to finish ..."
while ! grep -q "######## done" "$GATES_LOG" 2>/dev/null; do
    if ! pgrep -f "gates\.sh" >/dev/null 2>&1 \
       && ! grep -q "######## done" "$GATES_LOG" 2>/dev/null; then
        echo "[auto] transfer_gates.sh is gone without a done marker."
        echo "[auto] treating that as FAILED — refusing to spend money on an unverified candidate."
        exit 1
    fi
    sleep 60
done
echo "[auto] transfer gates finished"

# ----------------------------------------------------------------------
# 1. Read the two replay reports and apply the acceptance rule.
# ----------------------------------------------------------------------
BASE_REPORT=$(ls -t "$LOGS"/replay_baseline_*/replay_falls_report.json 2>/dev/null | head -1)
CAND_REPORT=$(ls -t "$LOGS"/replay_${LABEL}_*/replay_falls_report.json 2>/dev/null | head -1)
echo "[auto] baseline report : ${BASE_REPORT:-MISSING}"
echo "[auto] candidate report: ${CAND_REPORT:-MISSING}"

DECISION=$(python3 - "$BASE_REPORT" "$CAND_REPORT" <<'PY'
import json, sys

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

base, cand = load(sys.argv[1] if len(sys.argv) > 1 else ""), load(sys.argv[2] if len(sys.argv) > 2 else "")
if not base or not cand:
    print("ABORT missing_reports 0 0 0"); raise SystemExit

# The report computes these itself; use its own sets rather than re-deriving.
# scenarios is a LIST of dicts keyed by trial_id, and the aggregate already
# exposes the two ID sets we need.
ba, ca = base.get("aggregate", {}), cand.get("aggregate", {})
all_ids = [s["trial_id"] for s in base.get("scenarios", [])]

# A scenario is a VALID gate only if the BASELINE still falls there. One the
# baseline survives proves nothing about the candidate.
not_repro = set(ba.get("scenarios_not_reproducing_ids") or ba.get("scenarios_survived_ids") or [])
valid = [i for i in all_ids if i not in not_repro]

cand_survived = set(ca.get("scenarios_survived_ids") or [])
survived = [i for i in valid if i in cand_survived]

n_valid, n_surv = len(valid), len(survived)
frac = (n_surv / n_valid) if n_valid else 0.0

# Acceptance: the suite must be meaningful (>=8 of 10 reproduce) and the
# candidate must survive most of it. 0.70 rather than the plan's 9/10 because
# 3 reps quantise survival coarsely; per-scenario detail is in the report.
ok = n_valid >= 8 and frac >= 0.70
print(f"{'PASS' if ok else 'FAIL'} valid={n_valid} survived={n_surv} frac={frac:.2f} "
      f"not_reproducing={','.join(sorted(not_repro)) or 'none'}")
PY
)
echo "[auto] REPLAY DECISION: $DECISION"

# Physics pass verdict (the gap-hunt S2 caveat is scored by hand, see
# transfer_gates.sh header, so it is reported but not auto-gated).
PP=$EMBODY/results/figures/smoke/physics_pass_report.json
PP_OK=$(python3 -c "
import json
try: print(json.load(open('$PP')).get('acceptance','UNKNOWN'))
except Exception: print('UNKNOWN')")
echo "[auto] physics pass: $PP_OK"

case "$DECISION" in
  PASS*) ;;
  *) echo "[auto] STOP: transfer gates did not pass. No benchmark, no spend."
     echo "[auto] The candidate needs work; the reports above say where."
     exit 0 ;;
esac
if [ "$PP_OK" != "PASS" ]; then
    echo "[auto] STOP: physics pass = $PP_OK (need PASS). No benchmark, no spend."
    exit 0
fi

# ----------------------------------------------------------------------
# 1b. CALIBRATION GATE — added 2026-07-29 after gap-hunt S5 failed on v5d.
#
# configs/benchmark.yaml carries locomotion constants MEASURED ON v4_robust:
# k_velocity_realisation 1.004, turn_rate_realisation 0.982, and — decisively —
# open_loop_yaw_drift_deg_per_s 1.83, whose in-file justification is v4's own
# wz tracking error of 0.067 rad/s. v5d's is 0.098 rad/s: 46 % higher.
#
# The harness dead-reckons from COMMANDED velocity, so the gap between the
# model's believed position and the truth grows at a rate set by the POLICY.
# Benchmark success is `declare_done` within 0.35 m judged on TRUE position
# while the model reasons on that belief. Run with another policy's constants
# and the batch measures a mis-calibrated harness rather than the policy — which
# is exactly what gap-hunt S5 caught (return_home -> declared_elsewhere) and
# exactly the way to waste $10.
#
# Refuse to spend until a calibration measured for THIS checkpoint exists.
# No silent fallback to v4's numbers.
# ----------------------------------------------------------------------
CALIB=$LOGS/calibration_${LABEL}.json
DERIVED=$LOGS/derived_calibration.json
if ! python3 - "$CALIB" "$DERIVED" "$EMBODY/duck_embody/sim/policy_wrapper.py" "$LABEL" <<'PYGATE'
import json, re, sys
calib, derived, wrapper, label = sys.argv[1:5]
try:
    d = json.load(open(derived))
except Exception as e:
    print(f"  no derived calibration ({e}) — run scripts/calibrate.sh first"); sys.exit(1)
if not d.get("control_ok"):
    print("  derived calibration control_ok is FALSE — the measurement itself is"
          " untrustworthy, so its constants must not be shipped"); sys.exit(1)
want = (d.get("derived") or {}).get(label)
if not want:
    print(f"  derived calibration has no entry for '{label}'"); sys.exit(1)
# The ONLY runtime consumer is the python constant; the yaml locomotion block is
# documentation (verified: zero python readers). So gate on the python.
src = open(wrapper).read()
m = re.search(r"^K_VELOCITY_REALISATION\s*=\s*([0-9.]+)", src, re.M)
if not m:
    print("  cannot find K_VELOCITY_REALISATION in policy_wrapper.py"); sys.exit(1)
live, target = float(m.group(1)), float(want["k_velocity_realisation"])
if abs(live - target) > 1e-4:
    print(f"  policy_wrapper.K_VELOCITY_REALISATION = {live} but {label} measures"
          f" {target}. The harness would servo move() for a different policy —"
          f" which is the mis-calibration this gate exists to stop."); sys.exit(1)
print(f"  calibration OK: K={live} matches measured {target} for {label}")
PYGATE
then
    echo "[auto] STOP: calibration gate failed (see above). No benchmark, no spend."
    exit 0
fi

# ----------------------------------------------------------------------
# 2. Dry-run the benchmark. Abort on anything the guard dislikes BEFORE paying.
# ----------------------------------------------------------------------
echo "[auto] === benchmark dry-run ==="
mkdir -p "$RAW_OUT" "$VID_OUT"
( cd "$EMBODY" && "$ISAACLAB/isaaclab.sh" -p -m duck_embody.runner --dry-run \
    --out-dir "$RAW_OUT" --video-dir "$VID_OUT" --checkpoint "$CAND" ) \
    > "$LOGS/bench_dryrun_${LABEL}.log" 2>&1
DRY_RC=$?
PENDING=$(grep -ciE "pending|will run|queued" "$LOGS/bench_dryrun_${LABEL}.log" 2>/dev/null)
echo "[auto] dry-run exit=$DRY_RC, pending-ish lines=$PENDING"
tail -20 "$LOGS/bench_dryrun_${LABEL}.log"
if [ "$DRY_RC" -ne 0 ]; then
    echo "[auto] STOP: dry-run failed. Not spending money against a refused guard."
    exit 1
fi

# ----------------------------------------------------------------------
# 3. The paid batch: 12 trials, one persistent sim process.
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 2b. PROVENANCE MANIFEST — written BEFORE the paid run.
#
# Nothing in the trial JSONs records which policy produced them: the `config`
# block is {freeze_commit, config_hash, model, model_config, seed, spawn}, and
# no policy artifact is in FROZEN_FILES, so a v5d batch carries the SAME
# config_hash as the v4 baseline. Without this file the headline claim
# ("v5d cut falls/policy-min from 1.58 to X") is unfalsifiable from the
# evidence. Written outside RAW_OUT so the scorer's glob cannot pick it up.
# ----------------------------------------------------------------------
python3 - "$CAND" "$LABEL" "$EMBODY" "$RAW_OUT" > "$LOGS/provenance_${LABEL}.json" <<'PYPROV'
import hashlib, json, subprocess, sys
from pathlib import Path
cand, label, embody, raw_out = sys.argv[1:5]
def sha(p):
    try: return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception: return None
def git(args, cwd):
    try: return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                               text=True, timeout=20).stdout.strip()
    except Exception: return None
base = f"{embody}/policy/model_2999.pt"
print(json.dumps({
    "label": label,
    "candidate_checkpoint": cand,
    "candidate_sha256": sha(cand),
    "baseline_checkpoint": base,
    "baseline_sha256": sha(base),
    "is_baseline": sha(cand) == sha(base),
    "raw_out": raw_out,
    "duck_embody_commit": git(["rev-parse", "HEAD"], embody),
    "duck_embody_dirty": bool(git(["status", "--porcelain"], embody)),
    "parent_commit": git(["rev-parse", "HEAD"], "/home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson"),
}, indent=2))
PYPROV
echo "[auto] provenance -> $LOGS/provenance_${LABEL}.json"
python3 -c "
import json,sys
d=json.load(open('$LOGS/provenance_${LABEL}.json'))
if d['candidate_sha256'] is None:
    print('[auto] STOP: candidate checkpoint unreadable'); sys.exit(1)
if d['is_baseline']:
    print('[auto] STOP: the candidate IS the v4 baseline (sha match). Refusing to'); 
    print('[auto] spend \$10 re-measuring the baseline under v5d constants.'); sys.exit(1)
print('[auto] provenance OK: candidate sha', d['candidate_sha256'][:16])
" || exit 1

echo "[auto] === RUNNING PAID BENCHMARK (~\$10, 12 trials) $(date '+%F %T') ==="
( cd "$EMBODY" && "$ISAACLAB/isaaclab.sh" -p -m duck_embody.runner \
    --out-dir "$RAW_OUT" --video-dir "$VID_OUT" --checkpoint "$CAND" ) \
    > "$LOGS/bench_run_${LABEL}.log" 2>&1
BENCH_RC=$?
echo "[auto] benchmark exit=$BENCH_RC $(date '+%F %T')"
N_TRIALS=$(ls "$RAW_OUT"/*.json 2>/dev/null | wc -l)
echo "[auto] trials written: $N_TRIALS / $EXPECTED_TRIALS"
if [ "$BENCH_RC" -ne 0 ]; then
    echo "[auto] STOP: the batch exited non-zero. NOT scoring a failed batch —"
    echo "[auto] a partial batch scores as improbably clean (the cheap early"
    echo "[auto] trials are the ones that do not fall)."
    exit 1
fi
if [ "$N_TRIALS" -ne "$EXPECTED_TRIALS" ]; then
    echo "[auto] STOP: incomplete batch ($N_TRIALS/$EXPECTED_TRIALS). NOT scoring."
    exit 1
fi

# ----------------------------------------------------------------------
# 4. Conformance audit per trial, then the headline metric.
# ----------------------------------------------------------------------
echo "[auto] === audit_trial per trial ==="
AUDIT_FAIL=0
for f in "$RAW_OUT"/*.json; do
    [ -f "$f" ] || continue
    ( cd "$EMBODY" && "$ISAACLAB/_isaac_sim/python.sh" scripts/audit_trial.py "$f" ) \
        >> "$LOGS/bench_audit_${LABEL}.log" 2>&1 || AUDIT_FAIL=$((AUDIT_FAIL+1))
done
echo "[auto] audit failures: $AUDIT_FAIL"

echo "[auto] === scoring ==="
python3 - "$RAW_OUT" "$LABEL" "$BASELINE_FALLS_PER_MIN" "$TARGET_FALLS_PER_MIN" "$SUMMARY" <<'PY'
import json, sys
from pathlib import Path

raw, label, base_fpm, target_fpm, out = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), Path(sys.argv[5])

rows, falls_total, policy_s_total = [], 0, 0.0
files = sorted(raw.glob("*.json"))

# The repo's own completeness predicate (duck_embody/scoring.py: is_complete):
# a trial counts only with a `final` block and no `infra_failure`. An
# infra-failed trial DOES leave a JSON, and it contributes policy-seconds with
# no fall — which pushes falls-per-minute DOWN, i.e. flatters the candidate.
incomplete = []
for f in files:
    doc = json.loads(f.read_text())
    if "final" not in doc or "infra_failure" in doc:
        incomplete.append(f.name)
if incomplete:
    out.write_text(f"# {label} benchmark: NOT SCORED\n\n"
                   f"{len(incomplete)} incomplete/infra-failed trial(s): "
                   f"{', '.join(incomplete)}\n")
    print(f"REFUSING TO SCORE: incomplete trials {incomplete}")
    raise SystemExit(1)
if len(files) != 12:
    out.write_text(f"# {label} benchmark: NOT SCORED\n\n"
                   f"found {len(files)} trials, expected 12 (3 models x 4 seeds)\n")
    print(f"REFUSING TO SCORE: {len(files)} trials, expected 12")
    raise SystemExit(1)

for f in files:
    d = json.loads(f.read_text())
    fin = d.get("final", {})
    # Count falls and policy seconds from the per-call record: `fell` on a
    # motion call is the same signal the frozen batch was scored on.
    falls = 0
    psec = 0.0
    for turn in d.get("turns", []):
        ex = turn.get("execution") or {}
        for c in ex.get("calls", []) or []:
            psec += float(c.get("policy_seconds_used") or 0.0)
            if c.get("fell"):
                falls += 1
    stages = fin.get("stages", {}) or {}
    outcome = "; ".join(f"{k}={v.get('outcome')}" for k, v in stages.items())
    rows.append(dict(trial=f.stem, falls=falls, policy_s=round(psec, 1),
                     bumps=fin.get("bumps"), outcome=outcome,
                     cost=(fin.get("tokens", {}) or {}).get("cost_usd_estimate")))
    falls_total += falls
    policy_s_total += psec

fpm = (falls_total / (policy_s_total / 60.0)) if policy_s_total else float("nan")
verdict = "MEETS TARGET" if fpm < target_fpm else ("IMPROVED" if fpm < base_fpm else "NO IMPROVEMENT")

lines = [f"# {label} benchmark summary", "",
         f"- trials: {len(rows)}",
         f"- total falls: {falls_total}",
         f"- total policy-seconds: {policy_s_total:.1f}",
         f"- **falls per policy-minute: {fpm:.3f}**  (v4_robust baseline {base_fpm}, target < {target_fpm})",
         f"- verdict: **{verdict}**",
         f"- total cost: ${sum(r['cost'] or 0 for r in rows):.2f}", "",
         "| trial | falls | policy-s | bumps | outcome |", "|---|---|---|---|---|"]
lines += [f"| {r['trial']} | {r['falls']} | {r['policy_s']} | {r['bumps']} | {r['outcome']} |" for r in rows]
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines[:8]))
print(f"\n[auto] summary written: {out}")
PY

echo "######## auto_pipeline done $(date '+%F %T') ########"
