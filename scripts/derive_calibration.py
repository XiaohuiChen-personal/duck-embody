"""Derive the benchmark.yaml locomotion constants from a T1.3 displacement run.

WHY THIS EXISTS
---------------
`scripts/calibrate.sh` originally compared constants by grepping the report for
the first numeric field whose key matched a substring. That is wrong in a way
that matters: the report stores `open_loop_yaw_drift_deg_per_20s` (36.63 for
v4) while benchmark.yaml stores `open_loop_yaw_drift_deg_per_s` (1.83), and it
stores raw `heading_change_deg` where the config stores a *ratio*. The naive
comparison therefore reported "CONTROL FAILED: measured v4 36.63 != frozen
1.83" when the two numbers are the same measurement in different units — a
false alarm that, inside an unattended gate, halts a good run for no reason.

Units are converted explicitly here, and every derived value states the
arithmetic that produced it so a reader can check it without rerunning the sim.

The T1.3 runs this reads (see smoke_displacement.py):
  a_straight_20s   commanded (0.2, 0, 0) for 20 s  -> distance + yaw drift
  b_turn_10s       commanded (0, 0, 0.3) for 10 s  -> turn-rate realisation
  f_heading_hold   same as (a) with the heading servo closed -> residual drift

Usage:
    python3 scripts/derive_calibration.py <report.json> [<report.json> ...]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Commanded set-points of the T1.3 runs, from smoke_displacement.py.
STRAIGHT_VX = 0.2
STRAIGHT_S = 20.0
TURN_WZ = 0.3
TURN_S = 10.0

# The frozen values, measured on v4_robust (configs/benchmark.yaml).
FROZEN = {
    "k_velocity_realisation": 1.004,
    "measured_speed_mps": 0.201,
    "turn_rate_realisation": 0.982,
    "open_loop_yaw_drift_deg_per_s": 1.83,
    "heading_hold_residual_drift_deg_per_20s": 0.39,
}


def derive(report: dict) -> dict:
    """Convert a T1.3 report into benchmark.yaml's constant set, with workings."""
    runs = report["runs"]
    a, b, f = runs["a_straight_20s"], runs["b_turn_10s"], runs["f_heading_hold_20s"]

    commanded_distance = STRAIGHT_VX * STRAIGHT_S            # 4.0 m
    commanded_turn_deg = math.degrees(TURN_WZ * TURN_S)      # 171.887 deg

    k = a["net_displacement_m"] / commanded_distance
    speed = a["net_displacement_m"] / STRAIGHT_S
    turn_ratio = abs(b["heading_change_deg"]) / commanded_turn_deg
    drift_per_s = abs(a["heading_change_deg"]) / STRAIGHT_S
    hold_drift = abs(f["heading_change_deg"])

    return {
        "checkpoint": report.get("checkpoint"),
        "k_velocity_realisation": round(k, 4),
        "measured_speed_mps": round(speed, 4),
        "turn_rate_realisation": round(turn_ratio, 4),
        "open_loop_yaw_drift_deg_per_s": round(drift_per_s, 4),
        "heading_hold_residual_drift_deg_per_20s": round(hold_drift, 4),
        "_workings": {
            "k_velocity_realisation":
                f"{a['net_displacement_m']:.4f} m achieved / {commanded_distance:.1f} m commanded",
            "measured_speed_mps":
                f"{a['net_displacement_m']:.4f} m / {STRAIGHT_S:.0f} s",
            "turn_rate_realisation":
                f"|{b['heading_change_deg']:.2f}| deg achieved / {commanded_turn_deg:.3f} deg "
                f"commanded ({TURN_WZ} rad/s x {TURN_S:.0f} s)",
            "open_loop_yaw_drift_deg_per_s":
                f"|{a['heading_change_deg']:.2f}| deg / {STRAIGHT_S:.0f} s",
            "heading_hold_residual_drift_deg_per_20s":
                f"|{f['heading_change_deg']:.2f}| deg over {STRAIGHT_S:.0f} s with the servo closed",
        },
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    derived = {}
    for p in sys.argv[1:]:
        rep = json.loads(Path(p).read_text())
        label = Path(p).stem.replace("calibration_", "")
        derived[label] = derive(rep)

    keys = [k for k in FROZEN]
    width = max(len(k) for k in keys) + 2
    labels = list(derived)

    print(f"{'constant':{width}s} {'frozen(v4)':>12s}" + "".join(f"{l:>14s}" for l in labels))
    control_ok = True
    for k in keys:
        row = f"{k:{width}s} {FROZEN[k]:12.4f}"
        for l in labels:
            row += f"{derived[l][k]:14.4f}"
        print(row)
        # The control: whichever label is the baseline must match the frozen value.
        for l in labels:
            if "baseline" in l:
                got, want = derived[l][k], FROZEN[k]
                tol = 0.02 * max(abs(want), 1e-9)
                if abs(got - want) > tol:
                    control_ok = False
                    print(f"{'':{width}s}   !! CONTROL MISMATCH on {k}: {got} vs frozen {want}")

    print()
    print("CONTROL:", "PASS — the tool reproduces the frozen v4 constants, so the other "
                      "columns are trustworthy" if control_ok else
                      "FAIL — do not trust any column")

    # Report material differences against the baseline column.
    base = next((l for l in labels if "baseline" in l), None)
    if base:
        print()
        for l in labels:
            if l == base:
                continue
            diffs = []
            for k in keys:
                bv, cv = derived[base][k], derived[l][k]
                if abs(bv) > 1e-9 and abs(cv - bv) / abs(bv) > 0.02:
                    diffs.append(f"{k}: {bv:.4f} -> {cv:.4f} ({100*(cv-bv)/abs(bv):+.1f}%)")
            print(f"{l} vs {base}: " + ("; ".join(diffs) if diffs else "no material difference"))

    print()
    print("--- benchmark.yaml values for each policy ---")
    for l in labels:
        print(f"\n# {l}  (checkpoint: {derived[l]['checkpoint']})")
        for k in keys:
            print(f"{k}: {derived[l][k]}    # {derived[l]['_workings'][k]}")

    out = Path("results/logs/derived_calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"control_ok": control_ok, "frozen": FROZEN,
                               "derived": derived}, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0 if control_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
