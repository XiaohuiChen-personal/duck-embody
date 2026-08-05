"""B3 kit smoke — pre-latched signed reverse clearance (rule 11).

Acceptance (docs/research/V5D_R3_FURNITURE_WEDGE_HARNESS_IMPROVEMENTS.md §7 #4):
drive into a known face until ``sustained_contact`` latches, then ``move(-0.5)``
while still latched. Filmstrip must show visible reverse clearance when free
space exists behind the robot.

Geometry (forced 2026-08-05 RCA):
  Sofa **south** face, approach heading **90°** (north). Reverse holds the
  *approach* heading via ``move(..., hold_heading_deg=90)`` so the reverse
  axis is south into ≥0.5 m free floor — not the bump-yawed compass (~112°)
  that previously reversed into the sofa arm. Outer-west wall was rejected:
  bumps fire but never latch ``sustained_contact`` (intermittent foot force).

PASS (research §7#4 + P9): ``true_displacement_m ≥ 0.05`` OR
``contact_state → free``; steps > 2; upright; filmstrip-visible reverse.
Measured odometry alone is insufficient (can phantom-progress while wedged).

ORCHESTRATOR-RUN ONLY (AGENTS.md rule 1)::

    BUDGET=$(TERM=xterm-256color ~/IsaacLab/isaaclab.sh -p \\
        scripts/smoke_wedge_reverse.py --print-budget | tail -n1)
    PYTHONUNBUFFERED=1 TERM=xterm-256color timeout --kill-after=60 "$BUDGET" \\
        ~/IsaacLab/isaaclab.sh -p scripts/smoke_wedge_reverse.py \\
        --checkpoint /path/to/v5d_contact_wrench_ppo/model_5998.pt
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from duck_embody.sim.policy_wrapper import (  # noqa: E402
    MOVE_SPEED_MPS,
    REVERSE_MOVE_SPEED_MPS,
)

# Sofa south face, EAST of the west-wall corner (x=0.30 wedged wall+sofa arm —
# filmstrip wedge_reverse_20260805-013345). Spawn at x=0.50 clears the outer
# west wall so reverse-south is into open floor, not a corner pinch.
APPROACH_SPAWN = (0.50, 0.65)
APPROACH_HEADING_DEG = 90.0
APPROACH_DISTANCE_M = 0.90
REVERSE_DISTANCE_M = 0.5
MIN_REVERSE_FREE_M = 0.5
TARGET_FACE = (
    "sofa south face @ x=0.50 (not west-wall corner); "
    "approach 90°; reverse holds 90° (south free ≥0.5 m)"
)

MIN_TRUE_CLEARANCE_M = 0.05
MIN_REVERSE_STEPS = 3

V5D_CHECKPOINT = (
    Path.home()
    / "Projects"
    / "Open_Duck_Mini_Jetson"
    / "exported_policies"
    / "v5d_contact_wrench_ppo"
    / "model_5998.pt"
)


def estimated_policy_seconds() -> float:
    approach_s = (APPROACH_DISTANCE_M / MOVE_SPEED_MPS) * 2.5
    reverse_s = (REVERSE_DISTANCE_M / REVERSE_MOVE_SPEED_MPS) * 2.5
    return approach_s + reverse_s + 4.0


def wallclock_budget_s() -> int:
    return int(900 + estimated_policy_seconds() * 30.0)


def _reach_along(g, x: float, y: float, heading_deg: float, cap_m: float = 2.0) -> float:
    dx = math.cos(math.radians(heading_deg))
    dy = math.sin(math.radians(heading_deg))
    reach = 0.0
    while reach < cap_m and g.is_free(x + dx * (reach + 0.02), y + dy * (reach + 0.02)):
        reach += 0.02
    return reach


def _reverse_progress_m(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    approach_heading_deg: float,
) -> float:
    """Displacement projected onto the reverse axis (opposite approach heading)."""
    rdx = -math.cos(math.radians(approach_heading_deg))
    rdy = -math.sin(math.radians(approach_heading_deg))
    return (end_xy[0] - start_xy[0]) * rdx + (end_xy[1] - start_xy[1]) * rdy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smoke_wedge_reverse.py",
        description="B3 pre-latched reverse clearance smoke (no LLM).",
    )
    p.add_argument(
        "--print-budget",
        action="store_true",
        help="print kill-switch wallclock budget (seconds) as LAST stdout line",
    )
    p.add_argument("--checkpoint", default=None, help="policy .pt")
    return p


def _still_sheet(frame_paths: list[Path], out_path: Path, cols: int = 4) -> None:
    from PIL import Image

    imgs = [Image.open(p) for p in frame_paths]
    w, h = imgs[0].size
    scale = min(1.0, 480 / w)
    w, h = int(w * scale), int(h * scale)
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, im in enumerate(imgs):
        sheet.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    sheet.save(out_path)


def _video_checklist(
    *,
    fell: bool,
    reverse_steps: int,
    true_disp_m: float,
    reverse_progress_m: float,
    contact_after: str,
) -> dict:
    upright = not fell
    cleared = (
        true_disp_m >= MIN_TRUE_CLEARANCE_M
        or reverse_progress_m >= MIN_TRUE_CLEARANCE_M
        or contact_after == "free"
    )
    reverse_visible = reverse_steps >= MIN_REVERSE_STEPS and cleared
    not_immediate_abort = reverse_steps >= MIN_REVERSE_STEPS
    return {
        "upright": upright,
        "reverse_motion_visible": reverse_visible,
        "not_immediate_abort": not_immediate_abort,
        "pass": upright and reverse_visible and not_immediate_abort,
        "notes": (
            f"true_disp={true_disp_m:.3f} reverse_progress={reverse_progress_m:.3f} "
            f"contact_after={contact_after!r} steps={reverse_steps}"
        ),
    }


def main() -> int:
    args, kit_argv = build_parser().parse_known_args()
    sys.argv = [sys.argv[0], *kit_argv]

    if args.print_budget:
        print(wallclock_budget_s())
        return 0

    from duck_embody.sim.preflight import format_refusal, rule1_violations

    violations = rule1_violations()
    if violations:
        print(format_refusal(violations))
        return 2

    checkpoint = Path(args.checkpoint) if args.checkpoint else V5D_CHECKPOINT
    if not checkpoint.is_file():
        print(f"FAIL: checkpoint missing: {checkpoint}")
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / "results" / "logs" / f"wedge_reverse_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from duck_embody.env.apartment_layout import grid
    from duck_embody.sim.recorder import Recorder, attach_recorder
    from duck_embody.sim.session import SimSession, SpawnPose

    g = grid()
    sx, sy = APPROACH_SPAWN
    if not g.is_free(sx, sy):
        cell = g.nearest_free(sx, sy)
        if cell is None:
            print("FAIL: no free spawn for west-wall approach")
            return 1
        sx, sy = g.center(*cell)

    approach_reach = _reach_along(g, sx, sy, APPROACH_HEADING_DEG)
    # Expected latch just south of sofa; reverse free-space along south (270°).
    latch_x, latch_y = sx, sy + 0.20
    reverse_heading = (APPROACH_HEADING_DEG + 180.0) % 360.0
    reverse_free = _reach_along(g, latch_x, latch_y, reverse_heading)
    print(f"== smoke_wedge_reverse {stamp} ==")
    print(f"  artifacts : {out_dir}")
    print(f"  policy    : {checkpoint}")
    print(f"  target    : {TARGET_FACE}")
    print(
        f"  geometry  : approach_reach={approach_reach:.3f} m "
        f"reverse_free={reverse_free:.3f} m (need ≥{MIN_REVERSE_FREE_M})"
    )
    if reverse_free < MIN_REVERSE_FREE_M:
        print("FAIL: reverse free-space precondition unmet (layout)")
        return 1

    session = SimSession.launch(
        task_id="DuckEmbody-Apartment-v0",
        headless=True,
        checkpoint=str(checkpoint),
    )
    report: dict = {
        "smoke": "wedge_reverse",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "out_dir": str(out_dir.relative_to(REPO_ROOT)),
        "scenario": {
            "spawn": [sx, sy],
            "heading_deg": APPROACH_HEADING_DEG,
            "approach_distance_m": APPROACH_DISTANCE_M,
            "reverse_distance_m": REVERSE_DISTANCE_M,
            "target": TARGET_FACE,
            "approach_reach_m": round(approach_reach, 3),
            "reverse_free_m": round(reverse_free, 3),
        },
    }
    exit_code = 1
    detach_rec = None
    rec = None
    try:
        pb = session.playback
        session.reset(seed=101, spawn=SpawnPose(sx, sy, APPROACH_HEADING_DEG))

        rec = Recorder(
            out_dir / "wedge_reverse", fps=25, every_n=1, hide_ceiling=True
        )
        detach_rec = attach_recorder(pb, session.env.unwrapped, rec)

        print("\n== phase 1: approach latch ==")
        approach = pb.move(
            APPROACH_DISTANCE_M,
            hold_heading=True,
            stop_on_bump=True,
        )
        latch_pose = list(approach.true_pose)
        latch_state = pb.contact_state
        print(
            f"  approach: true_disp={approach.true_displacement_m:.3f} m "
            f"stop={approach.stop_reason!r} bumped={approach.bumped} "
            f"contact={latch_state!r} pose={latch_pose}"
        )
        report["approach"] = {
            "true_displacement_m": round(approach.true_displacement_m, 4),
            "measured_distance_m": round(approach.measured_distance_m, 4),
            "stop_reason": approach.stop_reason,
            "bumped": approach.bumped,
            "fell": approach.fell,
            "steps": approach.steps,
            "contact_state_live": latch_state,
            "last_contact_event": pb.last_contact_event,
            "latch_pose": [round(v, 4) for v in latch_pose],
        }

        # Press into the wall until sustained (move can "reach" after a graze).
        presses = []
        for i in range(4):
            if pb.contact_state == "sustained_contact":
                break
            print(f"  press[{i}] into wall (state={pb.contact_state!r})")
            press = pb.execute(
                MOVE_SPEED_MPS, 0.0, 0.0, 0.8, stop_on_bump=True
            )
            entry = {
                "stop_reason": press.stop_reason,
                "contact_state": pb.contact_state,
                "pose": [round(v, 4) for v in press.true_pose],
                "steps": press.steps,
                "bumped": press.bumped,
            }
            presses.append(entry)
            print(
                f"    stop={press.stop_reason!r} state={pb.contact_state!r} "
                f"pose={entry['pose']}"
            )
            latch_pose = list(press.true_pose)
        if presses:
            report["approach_presses"] = presses

        latch_state = pb.contact_state
        if latch_state != "sustained_contact":
            report["verdict"] = "FAIL"
            report["fail_reason"] = "never reached sustained_contact before reverse"
            report["video_checklist"] = {
                "pass": False,
                "notes": "latch phase failed — reverse not exercised",
            }
            _write_report(out_dir, report, rec)
            print("VERDICT: FAIL (no sustained_contact latch)")
            return 1

        print("\n== phase 2: move(-0.5) while latched ==")
        print(
            f"  pre-reverse contact={pb.contact_state!r} "
            f"heading={pb.compass_deg():.1f}° "
            f"(approach held {APPROACH_HEADING_DEG}°)"
        )
        reverse_start_xy = pb.true_xy()
        # Hold the APPROACH heading so reverse travels into free space south,
        # not along the bump-yawed compass (sofa RCA: h≈112° scraped the arm).
        reverse = pb.move(
            -REVERSE_DISTANCE_M,
            hold_heading=True,
            stop_on_bump=True,
            hold_heading_deg=APPROACH_HEADING_DEG,
        )
        reverse_end = list(reverse.true_pose)
        post_state = pb.contact_state
        progress = _reverse_progress_m(
            reverse_start_xy,
            (reverse_end[0], reverse_end[1]),
            APPROACH_HEADING_DEG,
        )
        print(
            f"  reverse: measured={reverse.measured_distance_m:.4f} m "
            f"true_disp={reverse.true_displacement_m:.4f} m "
            f"axis_progress={progress:.4f} m steps={reverse.steps} "
            f"stop={reverse.stop_reason!r} contact_after={post_state!r} "
            f"pose={reverse_end}"
        )

        report["reverse"] = {
            "requested_m": -REVERSE_DISTANCE_M,
            "measured_distance_m": round(reverse.measured_distance_m, 4),
            "true_displacement_m": round(reverse.true_displacement_m, 4),
            "reverse_axis_progress_m": round(progress, 4),
            "steps": reverse.steps,
            "stop_reason": reverse.stop_reason,
            "bumped": reverse.bumped,
            "fell": reverse.fell,
            "contact_state_result": reverse.contact_state,
            "contact_state_after": post_state,
            "last_contact_event_after": pb.last_contact_event,
            "start_xy": [round(reverse_start_xy[0], 4), round(reverse_start_xy[1], 4)],
            "end_pose": [round(v, 4) for v in reverse_end],
            "heading_at_latch_deg": round(latch_pose[2], 2),
        }

        # §7#4: true clearance OR free — measured alone is NOT enough.
        true_ok = (
            reverse.true_displacement_m >= MIN_TRUE_CLEARANCE_M
            or progress >= MIN_TRUE_CLEARANCE_M
        )
        contact_free = post_state == "free"
        checks = {
            "latched_before_reverse": True,
            "steps_gt_2": reverse.steps >= MIN_REVERSE_STEPS,
            "not_fell": not reverse.fell,
            "true_disp_or_free": true_ok or contact_free,
        }
        # Informational only (not a PASS gate).
        report["measured_info"] = {
            "measured_distance_m": round(reverse.measured_distance_m, 4),
            "measured_ge_0p05": reverse.measured_distance_m >= 0.05,
            "note": "measured alone must not PASS; odom can phantom while wedged",
        }

        checklist = _video_checklist(
            fell=bool(reverse.fell or approach.fell),
            reverse_steps=reverse.steps,
            true_disp_m=reverse.true_displacement_m,
            reverse_progress_m=progress,
            contact_after=post_state,
        )
        report["acceptance_checks"] = checks
        report["video_checklist"] = checklist

        all_ok = all(checks.values()) and checklist["pass"]
        report["verdict"] = "PASS" if all_ok else "FAIL"
        if not all_ok:
            report["fail_reason"] = [k for k, v in checks.items() if not v] + (
                [] if checklist["pass"] else ["video_checklist"]
            )

        artifacts = _write_report(out_dir, report, rec)
        report["artifacts"] = artifacts
        (out_dir / "wedge_reverse_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

        print("\n== acceptance checks ==")
        for k, v in checks.items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        print(
            f"  measured (info only): {reverse.measured_distance_m:.4f} m "
            f"(not a PASS gate)"
        )
        print(
            f"  video_checklist: "
            f"{'PASS' if checklist['pass'] else 'FAIL'} "
            f"({checklist['notes']})"
        )
        for key, path in artifacts.items():
            print(f"  {key}: {path}")
        print(f"VERDICT: {report['verdict']}")
        exit_code = 0 if all_ok else 1
        return exit_code
    except Exception as exc:  # noqa: BLE001
        import traceback

        detail = traceback.format_exc()
        print(detail)
        report["verdict"] = "FAIL"
        report["fail_reason"] = f"raised: {exc!r}"
        report["traceback"] = detail
        if rec is not None:
            try:
                _write_report(out_dir, report, rec)
            except Exception:  # noqa: BLE001
                pass
        (out_dir / "wedge_reverse_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print("VERDICT: FAIL (exception)")
        return 1
    finally:
        if detach_rec is not None:
            try:
                detach_rec()
            except Exception:  # noqa: BLE001
                pass
        print("  closing app (nothing after this line runs)")
        sys.stdout.flush()
        time.sleep(1.0)
        session.close()


def _write_report(out_dir: Path, report: dict, rec) -> dict:
    artifacts: dict = {}
    mp4 = rec.encode(keep_frames=True)
    strip = rec.filmstrip(mp4, fps=1.0) if mp4 else None
    if mp4:
        artifacts["mp4"] = str(mp4.relative_to(REPO_ROOT))
    if strip:
        artifacts["filmstrip"] = str(strip.relative_to(REPO_ROOT))

    frames = sorted(rec.frames_dir.glob("f*.png"))
    if frames:
        stills_dir = out_dir / "stills"
        stills_dir.mkdir(exist_ok=True)
        if len(frames) <= 16:
            pick = frames
        else:
            step = max(1, len(frames) // 12)
            pick = list(dict.fromkeys(frames[::step][:12] + frames[-8:]))
        for i, src in enumerate(pick):
            shutil.copy2(src, stills_dir / f"still_{i:02d}_{src.name}")
        sheet = out_dir / "stills_sheet.png"
        _still_sheet(pick, sheet, cols=4)
        artifacts["stills_dir"] = str(stills_dir.relative_to(REPO_ROOT))
        artifacts["stills_sheet"] = str(sheet.relative_to(REPO_ROOT))
        artifacts["n_frames"] = len(frames)

    report["artifacts"] = artifacts
    (out_dir / "wedge_reverse_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    artifacts["report"] = str(
        (out_dir / "wedge_reverse_report.json").relative_to(REPO_ROOT)
    )
    return artifacts


if __name__ == "__main__":
    raise SystemExit(main())
