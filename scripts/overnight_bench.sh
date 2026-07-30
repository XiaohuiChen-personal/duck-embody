#!/usr/bin/env bash
# Overnight v5d benchmark: preflight -> refreeze -> paid batch -> audit -> score.
#
# OWNER DIRECTIVES ENCODED HERE (2026-07-30, user asleep, full autonomy):
#   * matrix amended fable5 -> sonnet5 (cost); opus5 approved for smoke use
#   * dead-reckoning redesigned to leg odometry; anchors + correct_position(place)
#   * refreeze after the fix, rerun the benchmark, same evaluation rules,
#     frame-by-frame video analysis per trial (the agent watcher does that part)
#
# HARD RULES
#   * absolutely no paid trial before: unit suite green, physics smoke PASS,
#     probe green (pennies), freeze verified, provenance verified
#   * spend watchdog: abort the batch if summed cost_usd exceeds $80 — the
#     expected batch is ~$25-40 with sonnet5; $80 means something is wrong
#   * a partial batch is never scored (auto refusal in build_scores/completeness)
set -uo pipefail
REPO=/home/xiaohui_chen/Projects/duck-embody
ISAACLAB=/home/xiaohui_chen/IsaacLab
V5D=/home/xiaohui_chen/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-29_08-59-25/model_5998.pt
BASELINE=$REPO/policy/model_2999.pt
RAW=$REPO/results/raw_v5d_r2
VIDEOS=$REPO/results/videos_v5d_r2
LOG=$REPO/results/logs/overnight_bench.log
STATUSF=$REPO/results/logs/overnight_status.txt
mkdir -p "$RAW" "$VIDEOS" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
say() { echo "[overnight $(date '+%F %T')] $*"; echo "$*" > "$STATUSF"; }

say "=== PREFLIGHT ==="
# 1. unit suite
if ! bash "$REPO/scripts/run_tests.sh" tests/ -q > /tmp/overnight_tests.log 2>&1; then
    say "ABORT: unit suite red"; tail -5 /tmp/overnight_tests.log; exit 1
fi
say "unit suite green"
# 2. physics smoke verdict (must have been produced by scripts/smoke_odometry.py)
ACC=$(python3 -c "import json;print(json.load(open('$REPO/results/logs/smoke_odometry.json'))['acceptance'])" 2>/dev/null)
if [ "$ACC" != "PASS" ]; then say "ABORT: odometry smoke acceptance='$ACC' (need PASS)"; exit 1; fi
say "odometry physics smoke PASS"
# 3. provenance: candidate must not be the baseline
CSHA=$(sha256sum "$V5D" | cut -d' ' -f1); BSHA=$(sha256sum "$BASELINE" | cut -d' ' -f1)
if [ "$CSHA" = "$BSHA" ]; then say "ABORT: candidate == baseline checkpoint"; exit 1; fi
printf '{"candidate":"%s","candidate_sha256":"%s","baseline_sha256":"%s"}\n' "$V5D" "$CSHA" "$BSHA" > "$RAW/provenance.json"
say "provenance recorded (candidate sha ${CSHA:0:12})"
# 5. REFREEZE, then dry-run. The first version had no --freeze call at all while
# its own header promised one, and results/freeze.json still hashed
# configs/models/fable5.yaml — so the dry-run would have refused ("frozen file
# absent from freeze.json") and the whole unattended run would have died at the
# gate. --freeze additionally requires a fully clean tracked tree, so the
# caller MUST have committed first; we assert that here rather than discovering
# it after the abort.
cd "$REPO"
if [ -n "$(git status --porcelain -uno)" ]; then
    say "ABORT: tracked tree is dirty — --freeze requires a clean commit"; git status --porcelain -uno | head -5; exit 1
fi
if [ ! -f "$REPO/results/freeze_v4_baseline.json" ]; then
    say "ABORT: freeze_v4_baseline.json missing — refusing to overwrite the manifest that certifies the published v4 batch"; exit 1
fi
cp "$REPO/results/freeze.json" "$REPO/results/freeze_pre_odometry_$(date +%Y%m%d).json" 2>/dev/null || true
if ! PYTHONUNBUFFERED=1 "$ISAACLAB/isaaclab.sh" -p duck_embody/runner.py --freeze > /tmp/overnight_freeze.log 2>&1; then
    say "ABORT: --freeze refused"; grep -iE "fatal|refus|dirty" /tmp/overnight_freeze.log | head -5; exit 1
fi
NEWHASH=$(python3 -c "import json;print(json.load(open('$REPO/results/freeze.json'))['config_hash'][:12])")
say "refrozen: config_hash ${NEWHASH}"

# NOTE ON ORDER: the freeze happens BEFORE the probe. The probe rewrites the
# tracked results/figures/smoke/provider_probe.json, which would dirty the tree
# and make --freeze refuse. Freezing first is also the correct semantics: the
# manifest certifies the committed tree the batch runs against, and the probe
# is evidence gathered against that tree.
# 4. provider probe — pennies; catches sonnet5 config/auth before dollars
# PYTHONPATH: probe_providers imports duck_embody, and the script is invoked
# by absolute path from an arbitrary cwd, so the repo is not on sys.path.
if ! (cd "$REPO" && PYTHONPATH="$REPO" python3 scripts/probe_providers.py) > /tmp/overnight_probe.log 2>&1; then
    say "ABORT: provider probe failed"; tail -10 /tmp/overnight_probe.log; exit 1
fi
say "provider probe green (sonnet5/opus5/gpt56sol reachable)"
# 4b. WATCHDOG SELF-TEST: prove the cost extractor returns non-zero on a real
# trial JSON before trusting it to cap spend on an unattended run.
WT=$(python3 - <<'PYT'
import json, pathlib
f = pathlib.Path("/home/xiaohui_chen/Projects/duck-embody/results/raw_v5d/fable5_seed101.json")
print(float(json.load(open(f)).get("final", {}).get("tokens", {}).get("cost_usd_estimate") or 0))
PYT
)
if python3 -c "import sys; sys.exit(0 if float('$WT') > 0.5 else 1)"; then
    say "watchdog extractor self-test OK (reads \$$WT from a known trial)"
else
    say "ABORT: watchdog cost extractor returned '$WT' on a known-nonzero trial — spend cap would fail open"; exit 1
fi
if ! PYTHONUNBUFFERED=1 "$ISAACLAB/isaaclab.sh" -p duck_embody/runner.py --dry-run --out-dir "$RAW" > /tmp/overnight_dry.log 2>&1; then
    say "ABORT: dry-run refused"; grep -E "FATAL|refus" /tmp/overnight_dry.log | head -5; exit 1
fi
PENDING=$(grep -c "pending" /tmp/overnight_dry.log || true)
say "dry-run OK ($PENDING pending trials)"

say "=== BATCH (spend watchdog at \$80) ==="
(
  while true; do
    sleep 300
    SPENT=$(python3 - "$RAW" <<'PYW'
import json, pathlib, sys
total = 0.0
for f in pathlib.Path(sys.argv[1]).glob("*_seed*.json"):
    # final.tokens.cost_usd_estimate — NOT final.cost_usd, which does not exist.
    # The first version read the missing key, so the watchdog summed 0.00 and the
    # spend cap could never fire: a guard that fails OPEN is worse than none,
    # because it is believed. Preflight now self-tests this extraction.
    try: total += float(json.load(open(f)).get("final", {}).get("tokens", {}).get("cost_usd_estimate") or 0)
    except Exception: pass
print(f"{total:.2f}")
PYW
)
    echo "[watchdog $(date '+%T')] spent \$$SPENT"
    if python3 -c "import sys; sys.exit(0 if float('$SPENT') > 80.0 else 1)"; then
        echo "[watchdog] SPEND CAP EXCEEDED (\$$SPENT > \$80) — killing the batch"
        P="duck_embody/run""ner.py"; for K in $(pgrep -f "$P"); do kill "$K" 2>/dev/null; done
        exit 0
    fi
  done
) &
WATCHDOG=$!
PYTHONUNBUFFERED=1 "$ISAACLAB/isaaclab.sh" -p duck_embody/runner.py \
    --out-dir "$RAW" --video-dir "$VIDEOS" --checkpoint "$V5D"
BENCH_RC=$?
kill "$WATCHDOG" 2>/dev/null
say "batch finished rc=$BENCH_RC"

say "=== POST: completeness -> audit -> score ==="
COMPLETE=$(python3 - "$RAW" <<'PYC'
import json, pathlib, sys, yaml
repo = pathlib.Path("/home/xiaohui_chen/Projects/duck-embody")
cfg = yaml.safe_load((repo / "configs" / "benchmark.yaml").read_text())
missing, bad = [], []
for m in cfg["models"]:
    for s in (101, 102, 103, 104):
        f = pathlib.Path(sys.argv[1]) / f"{m}_seed{s}.json"
        if not f.exists(): missing.append(f.name); continue
        try:
            d = json.load(open(f))
            if "final" not in d: bad.append(f.name + ":no-final")
            elif d["final"].get("infra_failure"): bad.append(f.name + ":infra")
        except Exception as e: bad.append(f.name + f":{e}")
print("OK" if not missing and not bad else f"INCOMPLETE missing={missing} bad={bad}")
PYC
)
if [ "$COMPLETE" != "OK" ] || [ "$BENCH_RC" -ne 0 ]; then
    say "REFUSING TO SCORE: rc=$BENCH_RC completeness=$COMPLETE"
    say "DONE-INCOMPLETE"; exit 1
fi
for f in "$RAW"/*_seed*.json; do
    python3 "$REPO/scripts/audit_trial.py" "$f" > "${f%.json}_audit.txt" 2>&1 || echo "audit failed: $f"
done
DUCK_EMBODY_RAW_DIR="$RAW" python3 "$REPO/scripts/build_scores.py" > /tmp/overnight_score.log 2>&1
SC_RC=$?
say "scoring rc=$SC_RC -> results/scores_raw_v5d_r2.json"
say "DONE-COMPLETE"
