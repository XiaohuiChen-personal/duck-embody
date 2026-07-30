#!/usr/bin/env bash
# Block until the batch reaches N completed trials, then print a per-model
# report and exit. Run with run_in_background so the agent is re-invoked at
# each model boundary instead of polling until a tool timeout.
#
# The runner is model-major (all 4 seeds of one model, then the next), so
# N=4/8/12 are the model boundaries.
#
# Usage: scripts/watch_batch.sh <target_trials> <raw_dir>
set -uo pipefail
TARGET="${1:?}"; RAW="${2:?}"
EMBODY=/home/xiaohui_chen/Projects/duck-embody
while :; do
    n=$(ls "$RAW"/*.json 2>/dev/null | wc -l)
    [ "$n" -ge "$TARGET" ] && break
    # If the pipeline died, stop waiting rather than hanging forever.
    pgrep -f "auto_pipeline.sh" >/dev/null 2>&1 || {
        echo "PIPELINE GONE at $n/$TARGET trials"; break; }
    sleep 60
done
echo "=== batch progress: $(ls "$RAW"/*.json 2>/dev/null | wc -l) trials ==="
cd "$EMBODY"
python3 - "$RAW" <<'PY'
import json, sys
from pathlib import Path
from collections import defaultdict
raw = Path(sys.argv[1])
per = defaultdict(lambda: {"trials": 0, "falls": 0, "psec": 0.0, "bumps": 0,
                           "turns": 0, "outcomes": [], "cost": 0.0, "incomplete": 0})
for f in sorted(raw.glob("*.json")):
    d = json.loads(f.read_text())
    model = d.get("config", {}).get("model", f.stem.split("_seed")[0])
    p = per[model]
    if "final" not in d or "infra_failure" in d:
        p["incomplete"] += 1
        continue
    p["trials"] += 1
    for t in d.get("turns", []):
        for c in (t.get("execution") or {}).get("calls", []) or []:
            p["psec"] += float(c.get("policy_seconds_used") or 0.0)
            if c.get("fell"):
                p["falls"] += 1
    fin = d["final"]
    p["turns"] += len(d.get("turns", []))
    p["bumps"] += int(fin.get("bumps") or 0)
    p["cost"] += float((fin.get("tokens") or {}).get("cost_usd_estimate") or 0.0)
    st = fin.get("stages") or {}
    p["outcomes"].append(",".join(str(v.get("outcome")) for v in st.values()))

print(f"{'model':10s} {'trials':>6s} {'falls':>5s} {'policy-s':>9s} "
      f"{'falls/min':>9s} {'bumps':>5s} {'turns':>5s} {'cost':>7s}")
tot_f = tot_p = 0.0
for m, p in per.items():
    fpm = p["falls"] / (p["psec"] / 60.0) if p["psec"] else float("nan")
    tot_f += p["falls"]; tot_p += p["psec"]
    print(f"{m:10s} {p['trials']:>6d} {p['falls']:>5d} {p['psec']:>9.1f} "
          f"{fpm:>9.3f} {p['bumps']:>5d} {p['turns']:>5d} ${p['cost']:>6.2f}"
          + (f"  ({p['incomplete']} incomplete)" if p["incomplete"] else ""))
    for o in p["outcomes"]:
        print(f"{'':10s}   outcome: {o}")
if tot_p:
    print(f"\nRUNNING TOTAL falls/policy-minute: {tot_f/(tot_p/60.0):.3f}"
          f"   (v4_robust baseline 1.58, target < 0.30)")
PY
