#!/usr/bin/env bash
# Watch the batch and, for each trial that completes, produce (a) the numeric
# audit and (b) a 12-frame contact sheet ready for the mandatory visual pass.
# The visual judgement itself is not automatable — this only removes the
# mechanical work around it.
set -uo pipefail
REPO=/home/xiaohui_chen/Projects/duck-embody
RAW=$REPO/results/raw_v5d_r2
VID=$REPO/results/videos_v5d_r2
OUT=$REPO/results/audits_v5d_r2
mkdir -p "$OUT" "$OUT/sheets"
while true; do
  for f in "$RAW"/*_seed*.json; do
    [ -e "$f" ] || continue
    tid=$(basename "$f" .json)
    [ -f "$OUT/$tid.md" ] && continue
    python3 -c "import json,sys; d=json.load(open('$f')); sys.exit(0 if 'final' in d else 1)" 2>/dev/null || continue
    python3 - "$f" > "$OUT/$tid.md" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); fin = d["final"]
calls = [c for t in d["turns"] for c in (t.get("execution") or {}).get("calls", []) or []
         if c.get("tool") in ("move", "send_velocity")]
bel = sum(float(c.get("distance_moved_m") or 0) for c in calls)
tru = sum(float(c.get("true_displacement_m") or 0) for c in calls)
wedged = [c for c in calls if c.get("bumped")]
wb = sum(float(c.get("distance_moved_m") or 0) for c in wedged)
wt = sum(float(c.get("true_displacement_m") or 0) for c in wedged)
print(f"# {d['trial_id']} — audit\n")
print(f"turns {len(d['turns'])} | bumps {fin.get('bumps')} | "
      f"cost ${fin.get('tokens',{}).get('cost_usd_estimate',0):.2f}")
print(f"outcome {fin.get('outcome')}")
print(f"end_reason {fin.get('end_reason')}\n")
print("## Odometry (leg-odometry redesign, 2026-07-30)\n")
print(f"- motion calls {len(calls)}: believed {bel:.2f} m vs true {tru:.2f} m "
      f"-> **{bel-tru:+.2f} m**")
if wedged:
    print(f"- of which BUMPED {len(wedged)}: believed {wb:.2f} m vs true {wt:.2f} m "
          f"-> **{wb-wt:+.2f} m**  (the class that produced +25.10 m in the v4 batch)")
print(f"- correct_position calls: {len(d.get('corrections') or [])}")
for st, sv in (fin.get("stages") or {}).items():
    print(f"- {st}: success={sv.get('success')} drift={sv.get('drift_m')}")
print("\n## Frame audit\n\n_pending visual pass — sheet in sheets/_")
PY
    if [ -f "$VID/$tid.mp4" ]; then
      rm -rf /tmp/aa; mkdir -p /tmp/aa
      ffmpeg -loglevel error -i "$VID/$tid.mp4" \
        -vf "select='not(mod(n\,60))',scale=420:-1" -vsync 0 /tmp/aa/f_%03d.png 2>/dev/null
      python3 - "$tid" <<'PY'
from PIL import Image
import glob, sys, pathlib
fs = sorted(glob.glob('/tmp/aa/f_*.png'))
if fs:
    sel = fs[::max(1, len(fs)//12)][:12]
    ims = [Image.open(f) for f in sel]
    w, h = ims[0].size; cols = 4; rows = (len(ims)+cols-1)//cols
    sheet = Image.new('RGB', (w*cols, h*rows), 'black')
    for i, im in enumerate(ims): sheet.paste(im, ((i%cols)*w, (i//cols)*h))
    out = pathlib.Path('/home/xiaohui_chen/Projects/duck-embody/results/audits_v5d_r2/sheets')
    sheet.save(out / f"{sys.argv[1]}.png")
PY
    fi
    echo "[auto_audit $(date '+%H:%M')] audited $tid"
  done
  N=$(ls "$OUT"/*_seed*.md 2>/dev/null | wc -l)
  [ "$N" -ge 12 ] && { echo "[auto_audit] all 12 audited"; break; }
  sleep 60
done
