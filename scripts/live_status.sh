#!/usr/bin/env bash
# Live batch status: one line per completed trial, plus the running headline.
# Distinguishes COMPLETE trials (have a `final` block) from the one currently
# being written — the JSON is flushed incrementally, so an in-progress file
# looks like a very short trial if you read it as a result.
set -uo pipefail
RAW="${1:?}"
cd /home/xiaohui_chen/Projects/duck-embody
python3 - "$RAW" <<'PY'
import json, sys
from pathlib import Path
raw = Path(sys.argv[1])
done, live = [], None
for f in sorted(raw.glob("*.json"), key=lambda p: p.stat().st_mtime):
    d = json.loads(f.read_text())
    falls = psec = 0.0
    falls = 0
    for t in d.get("turns", []):
        for c in (t.get("execution") or {}).get("calls", []) or []:
            psec += float(c.get("policy_seconds_used") or 0.0)
            if c.get("fell"):
                falls += 1
    row = dict(id=f.stem, turns=len(d.get("turns", [])), falls=falls, psec=psec,
               complete=("final" in d and "infra_failure" not in d),
               infra="infra_failure" in d)
    if row["complete"] or row["infra"]:
        fin = d.get("final") or {}
        st = fin.get("stages") or {}
        row["outcome"] = "/".join(str(v.get("outcome")) for v in st.values()) or "-"
        row["bumps"] = fin.get("bumps")
        row["cost"] = (fin.get("tokens") or {}).get("cost_usd_estimate")
        done.append(row)
    else:
        live = row

print(f"{'trial':20s} {'turns':>5s} {'falls':>5s} {'policy-s':>8s} {'bumps':>5s} {'cost':>6s}  outcome")
tf = tp = 0.0; nf = 0
for r in done:
    nf += r["falls"]; tp += r["psec"]
    c = f"${r['cost']:.2f}" if r.get("cost") else "-"
    print(f"{r['id']:20s} {r['turns']:>5d} {r['falls']:>5d} {r['psec']:>8.1f} "
          f"{str(r.get('bumps')):>5s} {c:>6s}  {r.get('outcome','-')}"
          + ("   INFRA-FAIL" if r["infra"] else ""))
if live:
    print(f"{live['id']:20s} {live['turns']:>5d} {live['falls']:>5d} {live['psec']:>8.1f} "
          f"{'-':>5s} {'-':>6s}  << IN PROGRESS (partial file)")
print()
print(f"complete: {len(done)}/12   falls: {nf}   policy-s: {tp:.1f}")
if tp:
    print(f"falls/policy-minute so far: {nf/(tp/60.0):.3f}   "
          f"(v4_robust 1.58 | target < 0.30)")
PY
