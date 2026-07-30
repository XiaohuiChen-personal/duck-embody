"""Validate the 2026-07-30 leg-odometry redesign against REAL physics.

Exists because the previous fix (contact-time discounting) passed 1573 unit
tests and then FAILED this exact step: real contact force is an impulse train
for v4's bouncing gait (above 1 N on 6.9% of wedged steps), so the discount was
inert precisely where it mattered. Mocks encode the author's assumptions; the
physics does not. Nothing about the odometry path is considered fixed until
this passes on the GPU.

Five cases, each asserting the numbers the MODEL would receive:

  wedge         pressed into the sofa: the estimate must track the ~0.16 m of
                true motion, not the 1.2 m commanded (the 26 m-drift bug class)
  clean_walk    open floor: odometry within a few % of true distance
  curved_drive  translate+rotate: the estimate lands near the true endpoint —
                arc fidelity now comes from measurement, not arc arithmetic
  drift_persists the estimate is NOT truth (research question intact)
  anchor_loop   full tool-stack round trip: update_room stamps an anchor,
                correct_position(place=...) snaps the estimate back to it

Run: PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/smoke_odometry.py \
        --checkpoint <policy.pt>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "logs" / "smoke_odometry.json"


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=None,
                    help="policy .pt (default: the vendored baseline)")
    return ap


def main() -> int:
    args, kit_argv = build_parser().parse_known_args()
    sys.argv = [sys.argv[0], *kit_argv]

    from duck_embody.agent.memory import Counters, Memory, PositionIntegrator
    from duck_embody.agent.tools import ToolContext, dispatch
    from duck_embody.agent.providers.base import ToolCall
    from duck_embody.sim.session import SimSession, SpawnPose

    session = SimSession.launch(
        task_id="DuckEmbody-Apartment-v0", checkpoint=args.checkpoint, headless=True
    )
    pb = session.playback
    report: dict = {"checkpoint": str(args.checkpoint or "DEFAULT"), "cases": {}}
    failures: list[str] = []

    def check(name, cond, detail):
        if not cond:
            failures.append(f"{name}: {detail}")
        print(f"   [{'ok' if cond else 'FAIL'}] {name}: {detail}")

    def fresh(spawn, heading):
        session.reset(seed=101, spawn=SpawnPose(spawn[0], spawn[1], heading))
        return pb.true_xy()

    # ---- 1. wedge ---------------------------------------------------------
    start = fresh((0.30, 0.75), 90.0)
    integ = PositionIntegrator(*start)
    r = pb.execute(0.2, 0.0, 0.0, 6.0, stop_on_bump=False)
    integ.apply_delta(*r.odom_dxy)
    true_end = pb.true_xy()
    true_d = math.dist(start, true_end)
    est_err = math.dist(integ.xy, true_end)
    report["cases"]["wedge"] = dict(
        commanded_m=round(0.2 * r.policy_seconds, 3), true_disp_m=round(true_d, 4),
        odom_dist_m=round(r.odom_distance_m, 4), est_err_m=round(est_err, 4),
        contact_steps=r.contact_steps, fell=bool(r.fell),
    )
    print(f"\n-- wedge -- {report['cases']['wedge']}")
    check("wedge_estimate_tracks_truth", est_err < 0.10,
          f"estimate ended {est_err:.3f} m from truth (true motion {true_d:.3f} m, "
          f"commanded {0.2*r.policy_seconds:.2f} m)")
    check("wedge_no_command_credit", r.odom_distance_m < 0.5 * 0.2 * r.policy_seconds,
          f"odometry {r.odom_distance_m:.3f} m vs commanded {0.2*r.policy_seconds:.2f} m")

    # ---- 2. clean walk ----------------------------------------------------
    # A straight run PROVEN clear by the T2.4 physics gate (8/8 doorway
    # transits, no spurious bumps): approach the hallway<->kitchen doorway
    # along its normal, exactly the way smoke_physics_pass computes it. The
    # first version of this case hand-picked (1.20, 0.40) heading north after
    # a point-in-rect check — and clipped the coffee table, because a clear
    # POINT is not a clear PATH. Reuse the machinery that already solved this.
    from duck_embody.env.apartment_layout import LAYOUT, grid
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from smoke_physics_pass import approach_point, doorway_runs
    g = grid()  # no-arg: reads module-level LAYOUT (same as smoke_physics_pass:183)
    run = next(r for r in doorway_runs(LAYOUT) if r["name"] == "door_hallway_to_kitchen")
    ax, ay = approach_point(g, *run["center"], *run["normal"])
    start = fresh((ax, ay), run["heading"])
    integ = PositionIntegrator(*start)
    r = pb.execute(0.2, 0.0, 0.0, 6.0, stop_on_bump=False)
    integ.apply_delta(*r.odom_dxy)
    true_end = pb.true_xy()
    true_d = math.dist(start, true_end)
    est_err = math.dist(integ.xy, true_end)
    report["cases"]["clean_walk"] = dict(
        true_disp_m=round(true_d, 4), odom_dist_m=round(r.odom_distance_m, 4),
        est_err_m=round(est_err, 4), bumped=bool(r.bumped), fell=bool(r.fell),
    )
    print(f"\n-- clean_walk -- {report['cases']['clean_walk']}")
    check("clean_walk_moved", true_d > 0.7, f"true displacement {true_d:.3f} m — "
          "if ~0 the spawn is blocked and the case is invalid")
    if true_d > 0.7:
        check("clean_odom_tracks", abs(r.odom_distance_m - true_d) / true_d < 0.12,
              f"odom {r.odom_distance_m:.3f} vs true {true_d:.3f} "
              f"({100*abs(r.odom_distance_m-true_d)/true_d:.1f}% off)")
        check("clean_estimate_near_truth", est_err / true_d < 0.12,
              f"estimate error {est_err:.3f} m over {true_d:.3f} m walked")

    # ---- 3. curved drive --------------------------------------------------
    start = fresh((ax, ay), run["heading"])
    integ = PositionIntegrator(*start)
    r = pb.execute(0.2, 0.0, 0.12, 8.0, stop_on_bump=False)
    integ.apply_delta(*r.odom_dxy)
    true_end = pb.true_xy()
    est_err = math.dist(integ.xy, true_end)
    swept = math.dist(start, true_end)
    report["cases"]["curved_drive"] = dict(
        net_disp_m=round(swept, 4), est_err_m=round(est_err, 4), fell=bool(r.fell),
    )
    print(f"\n-- curved_drive -- {report['cases']['curved_drive']}")
    # The first version used vx=0.15,wz=0.4 and the policy barely translated
    # (0.089 m net against a 0.63 m commanded chord), so "arc fidelity comes
    # from measurement" passed by NOT MOVING. Require a real sweep first.
    check("curve_actually_swept", swept > 0.25,
          f"net sweep {swept:.3f} m — below 0.25 m the arc case proves nothing")
    check("curve_estimate_near_truth", est_err < max(0.15, 0.15 * max(swept, 0.4)),
          f"estimate {est_err:.3f} m from the true endpoint (net sweep {swept:.3f} m). "
          "Under commanded-arc reckoning this case was the ~0.45 m one-heading trap; "
          "under odometry the measurement carries the curve")

    # ---- 4. drift persists -------------------------------------------------
    start = fresh((1.20, 0.40), 90.0)
    integ = PositionIntegrator(*start)
    path = 0.0
    for _ in range(4):
        r = pb.execute(0.2, 0.0, 0.0, 2.0, stop_on_bump=False)
        integ.apply_delta(*r.odom_dxy)
        path += r.odom_distance_m
        pb.execute(0.0, 0.0, 0.5, 2.0)  # turn ~57 deg, then keep walking
    true_end = pb.true_xy()
    err = math.dist(integ.xy, true_end)
    report["cases"]["drift_persists"] = dict(path_m=round(path, 3), final_err_m=round(err, 4))
    print(f"\n-- drift_persists -- {report['cases']['drift_persists']}")
    check("estimate_is_not_ground_truth", err > 0.005,
          f"final error {err:.4f} m after {path:.2f} m — must be NONZERO or the "
          "research question (can an LLM close loops against real drift?) is gone")
    check("drift_is_sane", err < 1.0,
          f"final error {err:.4f} m — if huge, the odometry model is broken")

    # NOTE deliberately NOT asserted: turns 2-4 above run pb.execute directly
    # without feeding the integrator (only translation legs are applied), so
    # `err` also contains real turn-wander. The tool stack DOES feed turn
    # odometry (turn_to_heading applies odom_dxy); the anchor_loop case below
    # covers that path.

    # ---- 5. anchor round-trip through the real tool stack ------------------
    start = fresh((1.20, 0.40), 90.0)
    ctx = ToolContext(
        playback=pb, camera=None, memory=Memory(),
        integrator=PositionIntegrator(*start), counters=Counters(),
    )
    out = dispatch(ToolCall(id="c1", name="update_room",
                            args={"name": "start_room", "description": "smoke"}), ctx)
    anchor = ctx.memory.rooms["start_room"].anchor_xy
    check("anchor_stamped", anchor is not None and anchor == (round(start[0], 2), round(start[1], 2)),
          f"anchor {anchor} vs integrator-at-mapping {tuple(round(v,2) for v in start)}")
    dispatch(ToolCall(id="c2", name="move", args={"distance_m": 1.0}), ctx)
    moved_est = ctx.integrator.xy
    out = dispatch(ToolCall(id="c3", name="correct_position",
                            args={"place": "start_room", "reason": "smoke round trip"}), ctx)
    check("place_correction_applied", not out.is_error
          and ctx.integrator.xy == (anchor[0], anchor[1]),
          f"estimate {tuple(round(v,3) for v in ctx.integrator.xy)} vs anchor {anchor} "
          f"(was {tuple(round(v,3) for v in moved_est)} after the move); "
          f"error={out.is_error}")

    # ---- 6. RECORDED wedge: the path the batch actually runs -------------
    # attach_recorder patches execute() into 0.04 s slices whenever video is on
    # — which is every batch trial. Cases 1-5 above run UNRECORDED, so the
    # first version of this smoke passed on a code path the benchmark never
    # takes, and missed a per-call noise floor that accrued ~25x/s under
    # recording (0.094 m reported for a wedged 3 s command instead of ~0).
    from duck_embody.sim.recorder import Recorder, attach_recorder

    rec_dir = REPO_ROOT / "results" / "logs" / "smoke_rec_tmp"
    recorder = Recorder(rec_dir, fps=25, every_n=4, hide_ceiling=True)
    detach = attach_recorder(pb, session.env.unwrapped, recorder)
    try:
        start = fresh((0.30, 0.75), 90.0)
        integ = PositionIntegrator(*start)
        r = pb.execute(0.2, 0.0, 0.0, 3.0, stop_on_bump=False)
        integ.apply_delta(*r.odom_dxy)
        true_end = pb.true_xy()
        rec = dict(
            reported_m=round(r.odom_distance_m, 4),
            true_disp_m=round(math.dist(start, true_end), 4),
            est_err_m=round(math.dist(integ.xy, true_end), 4),
            policy_seconds=round(r.policy_seconds, 2),
        )
    finally:
        detach()
    report["cases"]["wedge_recorded"] = rec
    print(f"\n-- wedge_recorded -- {rec}")
    # ASSERTION CORRECTED (this check was wrong, not the code). It originally
    # demanded reported < 0.02 m on the reasoning "a wedged duck moves ~0".
    # The physics says otherwise: pressed against the sofa the duck still
    # shifted 0.094 m in 3 s. A sensor reporting ~0 there would be BROKEN.
    # The property that actually matters — and the one the whole redesign is
    # about — is that odometry reports what HAPPENED, not what was COMMANDED.
    # So: agreement with truth (tight), and far below the commanded arc.
    _commanded = 0.2 * rec["policy_seconds"]
    check("recorded_wedge_tracks_truth_not_command",
          abs(rec["reported_m"] - rec["true_disp_m"]) < 0.03,
          f"reported {rec['reported_m']} m vs true {rec['true_disp_m']} m "
          f"(commanded {_commanded:.2f} m) — odometry must follow truth")
    check("recorded_wedge_credits_little_of_the_command",
          rec["reported_m"] < 0.5 * _commanded,
          f"reported {rec['reported_m']} m is {rec['reported_m']/_commanded:.0%} of "
          f"the {_commanded:.2f} m commanded")
    check("recorded_wedge_estimate_tracks_truth", rec["est_err_m"] < 0.10,
          f"estimate {rec['est_err_m']} m from truth on the recorded path")

    report["cases"]["anchor_loop"] = dict(
        anchor=list(anchor) if anchor else None,
        est_after_move=[round(v, 3) for v in moved_est],
        est_after_correction=[round(v, 3) for v in ctx.integrator.xy],
        correction_was_error=bool(out.is_error),
    )

    report["failures"] = failures
    report["acceptance"] = "PASS" if not failures else "FAIL"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nacceptance: {report['acceptance']}  ({len(failures)} failures)")
    print(f"wrote {OUT}")
    session.close()
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
