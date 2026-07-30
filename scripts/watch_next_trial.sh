#!/usr/bin/env bash
# Block until the number of COMPLETE trials exceeds `since`, then print the live
# status and exit — so the agent is re-invoked per trial instead of polling.
# "Complete" means a `final` block exists: the runner flushes the JSON as it
# goes, so an in-progress file otherwise reads as a very short finished trial
# (which is exactly how a partial file got misreported as a result once).
set -uo pipefail
SINCE="${1:?}"; RAW="${2:?}"
EMBODY=/home/xiaohui_chen/Projects/duck-embody
count_complete () {
    python3 - "$RAW" <<'PY'
import json, sys
from pathlib import Path
n = 0
for f in Path(sys.argv[1]).glob("*.json"):
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    if "final" in d or "infra_failure" in d:
        n += 1
print(n)
PY
}
while :; do
    n=$(count_complete)
    [ "$n" -gt "$SINCE" ] && break
    [ "$n" -ge 12 ] && break
    pgrep -f "auto_pipeline.sh" >/dev/null 2>&1 || { echo "PIPELINE EXITED at $n complete"; break; }
    sleep 45
done
"$EMBODY/scripts/live_status.sh" "$RAW"
