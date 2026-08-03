"""TR.3 gate: recording must not change what the robot did (forensics F-03).

Before TR.3, ``attach_recorder`` replaced ``playback.execute`` with a wrapper
that re-entered it in 0.04 s pieces so a viewport frame could be grabbed between
them. That moved the command boundary, and the command boundary carries the bump
debounce window, the pose-trace phase, the clamp-note list, the fall-diagnostics
stamp and the odometry noise draw. Each of those had already been found and
fixed as its own bug; F-03's point is that the seam itself was the defect — a
recorded run and an unrecorded run were different experiments, and the paid
batch only ever ran the recorded one.

This script is the falsifier F-03 asks for. Each scripted sequence runs THREE
times from a full reset at one seed and spawn:

  A   recording OFF
  A2  recording OFF   <- control: the machine's own run-to-run repeatability
  B   recording ON

Semantic fields (steps, stop reason, bump/fall, contact groups) must match
EXACTLY. Numeric fields (odometry, true pose, pose trace) must match A to within
the tolerance the A-vs-A2 control derives, so rendering-induced GPU
nondeterminism is measured rather than assumed away. The only thing B may add is
frames.

Sequences: clean straight walk, curved ``send_velocity``, wall/furniture bump,
turn beside an obstacle.

Run (ONE GPU job — AGENTS.md rule 1; the script preflights):

    PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/smoke_record_invariance.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "logs" / "smoke_record_invariance.json"
VIDEO_DIR = REPO_ROOT / "results" / "logs" / "record_invariance"

#: Fields that must be BYTE-identical between a recorded and an unrecorded run.
#: These are semantics, not measurements: no amount of GPU nondeterminism turns
#: a "reached" into a "bump" or invents a contact region.
EXACT_FIELDS = (
    "steps",
    "policy_seconds",
    "stop_reason",
    "stopped_early",
    "bumped",
    "fell",
    "contact_groups",
    "clamp_notes",
    "n_pose_trace",
    "n_sampled_xy",
)

#: Numeric fields compared against the control-derived tolerance.
NUMERIC_FIELDS = (
    "odom_dx",
    "odom_dy",
    "odom_distance_m",
    "true_x",
    "true_y",
    "true_heading_deg",
    "true_displacement_m",
    "contact_steps",
)

#: Floor on the tolerance, so an exactly-repeatable control (the expected case
#: on this machine) does not demand bit-equality of float arithmetic that the
#: renderer could legitimately reorder.
TOLERANCE_FLOOR = 1e-9


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=None,
                    help="policy .pt (default: the vendored baseline)")
    ap.add_argument("--seed", type=int, default=101)
    return ap


def flatten(result) -> dict:
    """The comparable projection of one ``ExecResult``.

    Scoring-only fields ARE included: this is an audit script, and the whole
    question is whether ground truth moved when the recorder attached.
    """
    return {
        "steps": result.steps,
        "policy_seconds": round(result.policy_seconds, 9),
        "stop_reason": result.stop_reason,
        "stopped_early": bool(result.stopped_early),
        "bumped": bool(result.bumped),
        "fell": bool(result.fell),
        "contact_groups": list(result.contact_groups),
        "clamp_notes": list(result.clamp_notes),
        "n_pose_trace": len(result.pose_trace),
        "n_sampled_xy": len(result.sampled_xy),
        "odom_dx": result.odom_dxy[0],
        "odom_dy": result.odom_dxy[1],
        "odom_distance_m": result.odom_distance_m,
        "true_x": result.true_pose[0],
        "true_y": result.true_pose[1],
        "true_heading_deg": result.true_pose[2],
        "true_displacement_m": result.true_displacement_m,
        "contact_steps": result.contact_steps,
    }


def sequences():
    """Four scripted sequences with spawns proven usable by earlier gates.

    The clean-walk spawn is computed by the SAME machinery the T2.4 physics gate
    used (``smoke_physics_pass.approach_point`` on a doorway normal). An earlier
    smoke hand-picked a point that passed a point-in-rect check and then clipped
    the coffee table: a clear POINT is not a clear PATH.
    """
    from duck_embody.env.apartment_layout import LAYOUT, grid

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from smoke_physics_pass import approach_point, doorway_runs

    g = grid()
    run = next(r for r in doorway_runs(LAYOUT) if r["name"] == "door_hallway_to_kitchen")
    ax, ay = approach_point(g, *run["center"], *run["normal"])

    return [
        {
            "name": "clean_walk",
            "spawn": (ax, ay, run["heading"]),
            # One long command: the case where the retired chunker cut a single
            # semantic call into ~150 pieces.
            "commands": [(0.2, 0.0, 0.0, 3.0, False)],
        },
        {
            "name": "curved_send_velocity",
            "spawn": (ax, ay, run["heading"]),
            "commands": [(0.2, 0.0, 0.12, 4.0, False)],
        },
        {
            "name": "wall_bump",
            # Facing the sofa from the living room: the T2.4 / smoke_odometry
            # wedge spawn, which reliably reaches sustained contact.
            "spawn": (0.30, 0.75, 90.0),
            "commands": [(0.2, 0.0, 0.0, 4.0, False)],
        },
        {
            "name": "turn_near_wall",
            "spawn": (0.30, 0.75, 90.0),
            # Drive into contact, then rotate while touching: the combination
            # that made the debounce window depend on the chunk length.
            "commands": [(0.2, 0.0, 0.0, 2.0, False), (0.0, 0.0, 0.4, 3.0, False)],
        },
    ]


def run_sequence(session, seq, seed: int, recorder=None):
    """Full reset, then execute the sequence once. Returns flattened results.

    The recorder is attached AFTER ``session.reset`` on purpose: reset settles
    the gait for 0.5 s, and including those steps in one run but not the other
    would make the two runs differ in step count before the first command.
    """
    from duck_embody.sim.recorder import attach_recorder
    from duck_embody.sim.session import SpawnPose

    session.reset(seed=seed, spawn=SpawnPose(*seq["spawn"]))
    detach = None
    if recorder is not None:
        detach = attach_recorder(session.playback, session.env.unwrapped, recorder)
    try:
        out = []
        for vx, vy, wz, duration, stop_on_bump in seq["commands"]:
            result = session.playback.execute(
                vx, vy, wz, duration, stop_on_bump=stop_on_bump
            )
            out.append(flatten(result))
            if result.fell:
                break
        return out
    finally:
        if detach is not None:
            detach()


def compare(a, a2, b) -> dict:
    """Per-command diff report for one sequence."""
    report = {"n_commands": {"off": len(a), "control": len(a2), "on": len(b)},
              "commands": [], "failures": []}
    if not (len(a) == len(a2) == len(b)):
        report["failures"].append(
            f"command count differs: off={len(a)} control={len(a2)} on={len(b)}"
        )
        return report

    for idx, (ra, ra2, rb) in enumerate(zip(a, a2, b)):
        entry = {"index": idx, "exact": {}, "numeric": {}}
        for field in EXACT_FIELDS:
            same = rb[field] == ra[field]
            control_same = ra2[field] == ra[field]
            entry["exact"][field] = {
                "off": ra[field], "on": rb[field], "control": ra2[field],
                "match": bool(same),
            }
            if not same:
                report["failures"].append(
                    f"cmd{idx} {field}: off={ra[field]!r} on={rb[field]!r} "
                    f"(control={ra2[field]!r}) — semantic field, must be exact"
                )
            elif not control_same:
                # Recorded matched but the control did not: the sequence is not
                # deterministic enough to prove anything about recording.
                report["failures"].append(
                    f"cmd{idx} {field}: unrecorded control disagrees with itself "
                    f"({ra[field]!r} vs {ra2[field]!r}) — this case cannot gate"
                )
        for field in NUMERIC_FIELDS:
            control_diff = abs(ra2[field] - ra[field])
            rec_diff = abs(rb[field] - ra[field])
            tol = max(control_diff, TOLERANCE_FLOOR)
            entry["numeric"][field] = {
                "off": ra[field], "on": rb[field], "control": ra2[field],
                "recorded_diff": rec_diff, "control_diff": control_diff,
                "tolerance": tol, "within": bool(rec_diff <= tol),
            }
            if rec_diff > tol:
                report["failures"].append(
                    f"cmd{idx} {field}: recorded differs by {rec_diff:.6g} > "
                    f"tolerance {tol:.6g} (control repeatability {control_diff:.6g})"
                )
        report["commands"].append(entry)
    return report


def main() -> int:
    args, kit_argv = build_parser().parse_known_args()
    sys.argv = [sys.argv[0], *kit_argv]

    # Rule 1, automated: refuse to be the second kit process.
    from duck_embody.sim.preflight import format_refusal, rule1_violations

    violations = rule1_violations()
    if violations:
        print(format_refusal(violations))
        return 2

    from duck_embody.sim.recorder import Recorder
    from duck_embody.sim.session import SimSession

    session = SimSession.launch(
        task_id="DuckEmbody-Apartment-v0", checkpoint=args.checkpoint, headless=True
    )

    report = {
        "checkpoint": str(args.checkpoint or "DEFAULT"),
        "seed": args.seed,
        "tolerance_floor": TOLERANCE_FLOOR,
        "sequences": {},
    }
    failures: list[str] = []
    videos: list[dict] = []

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    for seq in sequences():
        name = seq["name"]
        print(f"\n=== {name} :: spawn {seq['spawn']} ===")
        off = run_sequence(session, seq, args.seed)
        print(f"  off      {[r['steps'] for r in off]} "
              f"odom={[round(r['odom_distance_m'], 4) for r in off]}")
        control = run_sequence(session, seq, args.seed)
        print(f"  control  {[r['steps'] for r in control]} "
              f"odom={[round(r['odom_distance_m'], 4) for r in control]}")

        recorder = Recorder(VIDEO_DIR / name, fps=25, every_n=1, hide_ceiling=True)
        on = run_sequence(session, seq, args.seed, recorder=recorder)
        print(f"  on       {[r['steps'] for r in on]} "
              f"odom={[round(r['odom_distance_m'], 4) for r in on]} "
              f"frames={recorder.n_frames}")

        seq_report = compare(off, control, on)
        seq_report["frames"] = recorder.n_frames
        total_steps = sum(r["steps"] for r in on)
        # 25 fps over 50 Hz stepping: one frame per 2 control steps, +/- the
        # boundary. A frozen or empty video is the rule-11 failure mode.
        seq_report["frames_expected_approx"] = total_steps // 2
        if recorder.n_frames == 0:
            seq_report["failures"].append("no frames captured — no rule-11 evidence")
        elif abs(recorder.n_frames - total_steps // 2) > 2:
            seq_report["failures"].append(
                f"frame count {recorder.n_frames} is off the 25 fps grid "
                f"(expected ~{total_steps // 2} for {total_steps} control steps)"
            )

        mp4 = recorder.encode()
        strip = recorder.filmstrip(mp4) if mp4 else None
        videos.append({
            "sequence": name,
            "mp4": str(mp4.relative_to(REPO_ROOT)) if mp4 else None,
            "filmstrip": str(strip.relative_to(REPO_ROOT)) if strip else None,
            "frames": recorder.n_frames,
        })

        for line in seq_report["failures"]:
            print(f"   [FAIL] {name}: {line}")
        if not seq_report["failures"]:
            print(f"   [ok] {name}: recorded == unrecorded on every field")
        failures.extend(f"{name}: {line}" for line in seq_report["failures"])
        report["sequences"][name] = seq_report

    report["videos"] = videos
    report["failures"] = failures
    report["acceptance"] = "PASS" if not failures else "FAIL"

    # Artifacts BEFORE close(): SimulationApp.close() terminates the process, so
    # anything written after it is silently lost (AGENTS.md §5).
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nacceptance: {report['acceptance']}  ({len(failures)} failures)")
    print(f"wrote {OUT}")
    for video in videos:
        print(f"  video {video['sequence']}: {video['mp4']} / {video['filmstrip']}")

    session.close()
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
