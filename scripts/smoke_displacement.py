"""T1.3 — measure what the duck ACTUALLY does when told to walk (VIDEO smoke).

The parent project never measured net displacement: its velocity metric is an
*instantaneous* L2 error (v4_robust: 0.153 m/s mean against commands of ~0.2 m/s
magnitude), which cannot distinguish "tracks with small symmetric noise" from
"tracks with a consistent shortfall". Dead reckoning built on commanded velocity
inherits that unknown bias, and the 240-policy-second cap is arithmetic on top of
it. This script measures the real thing before either is frozen.

Five runs, all on the empty plane at seed 42 (doc 02 §7):

  a. vx=0.2 held 20 s          -> the velocity realisation factor k
  b. wz=0.3 held 10 s          -> realised turn rate
  c. vx=0.2 re-issued every 2 s -> macro-style stop/start, the real usage pattern
  d. vx=0.2 held 120 s         -> long-hold yaw creep (doc 02 §7 mitigation 1)
  e. turn -> drive -> turn      -> step command changes (doc 02 §7 mitigation 2)

Runs (d) and (e) exist specifically because doc 02 §7 promised them: the
evaluation record covers fixed 30 s windows only, while a Duck Embody stage holds
commands for up to 240 policy-seconds and switches them every 0.2 s.

**k policy (PLAN T1.3, and AGENTS.md rule 5 wins over doc 02 §6.2's pseudocode):**
the dead-reckoning integrator uses **commanded** velocity with **no k**, so the
drift the model must notice and correct is honest and measurable. k is consumed
ONLY by (i) time-cap / wall-clock forecasting and (ii) the `move()` servo target
(`dist / k`) and its timeout margin.

Run:
    PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/smoke_displacement.py
"""

from __future__ import annotations

import json
import pathlib
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "figures" / "smoke"

# Acceptance band for k, recorded as a PLAN decision (not a doc claim).
K_MIN, K_MAX = 0.6, 1.1
# Heading drift budget over 20 s / ~4 m of travel, applied to the closed-loop
# heading-hold run (f) — that is what the motion macros actually deliver.
HEADING_DRIFT_LIMIT_DEG = 10.0
# P gain for the heading hold, on heading error in radians -> wz. Saturates the
# +/-0.5 rad/s hull at ~19 deg of error.
KP_HEADING = 1.5

SEED = 42


def summarize(name: str, results, t0_heading: float, playback) -> dict:
    """Collapse a run's ExecResults into the numbers we actually care about."""
    trace: list[tuple[float, float]] = []
    policy_s = 0.0
    bumped = False
    fell = False
    for r in results:
        trace.extend(r.pose_trace)
        policy_s += r.policy_seconds
        bumped = bumped or r.bumped
        fell = fell or r.fell

    net = math.dist(trace[0], trace[-1]) if len(trace) >= 2 else 0.0
    path = sum(math.dist(trace[i], trace[i + 1]) for i in range(len(trace) - 1))
    final_heading = playback.compass_deg()
    return {
        "name": name,
        "policy_seconds": round(policy_s, 3),
        "net_displacement_m": round(net, 4),
        "path_length_m": round(path, 4),
        "start_xy": [round(v, 4) for v in trace[0]],
        "end_xy": [round(v, 4) for v in trace[-1]],
        "start_heading_deg": round(t0_heading, 2),
        "final_heading_deg": round(final_heading, 2),
        "heading_change_deg": round((final_heading - t0_heading + 180) % 360 - 180, 2),
        "bumped": bumped,
        "fell": fell,
        "n_trace_points": len(trace),
    }


def build_parser():
    """Parser built outside main so --help needs no kit."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "T1.3 displacement")
    ap.add_argument(
        "--checkpoint", default=None,
        help="policy .pt to measure. Default: session.py's DEFAULT_CHECKPOINT "
             "(v4_robust). REQUIRED when calibrating a retrained policy: the "
             "constants this script measures — k_velocity_realisation, "
             "turn_rate_realisation, open_loop_yaw_drift_deg_per_s — are "
             "properties of the POLICY, not of the harness, and "
             "configs/benchmark.yaml's values were measured on v4_robust.",
    )
    ap.add_argument(
        "--out-json", default=None,
        help="also write the report here (e.g. results/logs/calibration_<label>.json)",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()

    from duck_embody.sim.policy_wrapper import shortest_angle_diff_deg
    from duck_embody.sim.recorder import Recorder
    from duck_embody.sim.session import SimSession, SpawnPose

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = SimSession.launch(
        task_id="DuckEmbody-v0", headless=True, checkpoint=args.checkpoint
    )
    playback = session.playback
    print("== session up ==")

    report: dict = {"seed": SEED, "checkpoint": str(args.checkpoint or "DEFAULT(v4_robust)"), "runs": {}}
    failures: list[str] = []

    def run(name: str, script, every_n: int = 2) -> dict:
        print(f"\n== run {name} ==")
        session.reset(seed=SEED, spawn=SpawnPose(0.0, 0.0, 0.0))
        rec = Recorder(OUT_DIR / f"displacement_{name}", fps=25, every_n=every_n)
        h0 = playback.compass_deg()
        results = session.scripted_drive(script, recorder=rec)
        mp4 = rec.encode()
        rec.filmstrip(mp4, fps=1.0)
        summary = summarize(name, results, h0, playback)
        summary["video"] = str(mp4.relative_to(REPO_ROOT)) if mp4 else None
        summary["height_at_end_m"] = round(playback.true_height(), 4)
        report["runs"][name] = summary
        for key in ("policy_seconds", "net_displacement_m", "heading_change_deg", "fell"):
            print(f"    {key}: {summary[key]}")
        return summary

    # (a) straight-line hold — the k measurement
    a = run("a_straight_20s", [(0.2, 0.0, 0.0, 20.0)])
    k = a["net_displacement_m"] / (0.2 * 20.0)
    report["k_velocity_realisation"] = round(k, 4)
    report["measured_speed_mps"] = round(a["net_displacement_m"] / 20.0, 4)
    print(f"\n  k = {k:.3f}  (net {a['net_displacement_m']} m / commanded 4.0 m)")

    # (b) yaw hold — realised turn rate
    b = run("b_turn_10s", [(0.0, 0.0, 0.3, 10.0)])
    # Unwrapped total rotation: heading_change_deg alone cannot distinguish one
    # turn from several, so use the commanded direction and 10 s of integration.
    turn_rate = math.radians(abs(b["heading_change_deg"])) / 10.0
    report["measured_turn_rate_radps_wrapped"] = round(turn_rate, 4)
    print(f"  wrapped turn rate >= {turn_rate:.3f} rad/s (commanded 0.3)")

    # (c) macro-style stop/start: 10 x (2 s drive, then actually STOP).
    # The explicit zero-command segment is the point of this run. Ten
    # back-to-back drive commands with no stop between them produce a byte
    # -identical trajectory to run (a) — the first version of this script did
    # exactly that and "passed" while testing nothing.
    run("c_stopstart", [(0.2, 0.0, 0.0, 2.0), (0.0, 0.0, 0.0, 0.4)] * 10)

    # (d) long hold — the regime the 30 s eval windows never covered.
    # every_n=10 keeps the video ~5 fps: 120 s at 25 fps would be 3,000 frames
    # of a duck walking in a straight line.
    d = run("d_longhold_120s", [(0.2, 0.0, 0.0, 120.0)], every_n=10)
    if d["policy_seconds"] > 0:
        report["yaw_creep_deg_per_100s"] = round(
            d["heading_change_deg"] / d["policy_seconds"] * 100.0, 3
        )

    # (e) turn -> drive -> turn, watching for stumbles at the switches
    run(
        "e_turn_drive_turn",
        [
            (0.0, 0.0, 0.5, 3.0),
            (0.2, 0.0, 0.0, 5.0),
            (0.0, 0.0, -0.5, 3.0),
            (0.2, 0.0, 0.0, 5.0),
        ],
    )

    # (f) THE MITIGATION: the same 20 s straight drive, but with wz closed on
    # the compass to hold the initial heading. Run (a) measures the policy's
    # open-loop yaw creep; this measures what the `move()` macro will actually
    # deliver, since AGENTS.md rule 5 explicitly permits closed-loop macros
    # servoing on compass + dead reckoning as a sensor-realistic exception.
    print("\n== run f_heading_hold_20s ==")
    session.reset(seed=SEED, spawn=SpawnPose(0.0, 0.0, 0.0))
    rec_f = Recorder(OUT_DIR / "displacement_f_heading_hold_20s", fps=25, every_n=2)
    target_heading = playback.compass_deg()
    f_results = []
    CHUNK_S = 0.2
    for _ in range(int(20.0 / CHUNK_S)):
        err_deg = shortest_angle_diff_deg(target_heading, playback.compass_deg())
        wz = max(-0.5, min(0.5, KP_HEADING * math.radians(err_deg)))
        f_results.append(session._execute_recording(0.2, 0.0, wz, CHUNK_S, rec_f))
        if f_results[-1].fell:
            break
    mp4_f = rec_f.encode()
    rec_f.filmstrip(mp4_f, fps=1.0)
    f = summarize("f_heading_hold_20s", f_results, target_heading, playback)
    f["video"] = str(mp4_f.relative_to(REPO_ROOT)) if mp4_f else None
    f["height_at_end_m"] = round(playback.true_height(), 4)
    report["runs"]["f_heading_hold_20s"] = f
    report["k_with_heading_hold"] = round(f["net_displacement_m"] / (0.2 * 20.0), 4)
    for key in ("policy_seconds", "net_displacement_m", "heading_change_deg", "fell"):
        print(f"    {key}: {f[key]}")

    # --- acceptance -------------------------------------------------------
    print("\n== acceptance ==")
    if not (K_MIN <= k <= K_MAX):
        failures.append(f"k={k:.3f} outside [{K_MIN}, {K_MAX}] — STOP and re-plan caps/macros")
    else:
        print(f"  PASS  k = {k:.3f} in [{K_MIN}, {K_MAX}]")

    # The plan's drift budget applies to what the MACROS deliver, which is run
    # (f) — the closed-loop heading hold. Run (a)'s open-loop drift is reported
    # as the measured property of the bare policy that motivates the hold.
    open_loop_drift = abs(a["heading_change_deg"])
    held_drift = abs(f["heading_change_deg"])
    report["open_loop_yaw_drift_deg_per_20s"] = round(open_loop_drift, 2)
    report["heading_hold_drift_deg_per_20s"] = round(held_drift, 2)
    print(f"  INFO  open-loop yaw drift over 20 s / 4 m: {open_loop_drift:.1f} deg")
    if held_drift > HEADING_DRIFT_LIMIT_DEG:
        failures.append(
            f"heading-hold drift {held_drift:.1f} deg > {HEADING_DRIFT_LIMIT_DEG} "
            "— closed-loop macro does not control the policy's yaw creep"
        )
    else:
        print(f"  PASS  heading-hold drift {held_drift:.1f} deg <= {HEADING_DRIFT_LIMIT_DEG}")

    for name, summary in report["runs"].items():
        if summary["fell"]:
            failures.append(f"{name}: robot FELL")

    report["failures"] = failures
    report["acceptance"] = "PASS" if not failures else "FAIL"

    out_json = OUT_DIR / "displacement_report.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n  wrote {out_json.relative_to(REPO_ROOT)}")
    # A second, explicitly-named copy so a per-policy calibration can live
    # outside the single-slot smoke directory (which the next run overwrites).
    if args.out_json:
        extra = pathlib.Path(args.out_json)
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  wrote {extra}")

    print("\n== summary ==")
    print(json.dumps({k2: v for k2, v in report.items() if k2 != "runs"}, indent=2))
    for name, summary in report["runs"].items():
        print(
            f"  {name:<18} net={summary['net_displacement_m']:>7.3f} m  "
            f"path={summary['path_length_m']:>7.3f} m  "
            f"dheading={summary['heading_change_deg']:>7.2f} deg  "
            f"fell={summary['fell']}"
        )

    if failures:
        print("\n  FAILURES:")
        for f in failures:
            print(f"    {f}")
    else:
        print("\n  OK - all runs completed, k in band, no falls")

    # Everything is written and printed BEFORE close(): the app terminates the
    # process and nothing after it executes (AGENTS.md §5).
    print("  closing app (nothing after this line runs)")
    session.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
