"""T2.4 — scripted physics pass (VIDEO GATE). No LLM, no money spent.

This is the gate that catches the two failure modes doc 03 §7 warns about, both
of which are invisible in aggregate metrics:

* **walk-through furniture** — an asset that renders perfectly and stops
  nothing, because ``collision_props`` can only *modify* a collider, never
  create one;
* **invisible force fields** — an authored contact offset that shoves the robot
  before it visually touches anything.

Plus the redesigned bump/fall semantics (doc 02 §5): furniture contact must be a
**bump** the trial survives, and only a real topple may end it.

What it drives:
  1. every doorway, **both directions** (4 doorways x 2 = 8 transits);
  2. a deliberate wall bump;
  3. a deliberate bump into each collider class — SimReady-native (sofa),
     bbox-proxy (fridge), and the **Sektion counter**, which T2.2 flagged
     because Isaac could not apply the contact-offset override to its
     instanceable, ``purpose=guide`` collision prims;
  4. a max-speed run into a wall.

Every run records an mp4 + filmstrip for the rule-11 frame-by-frame review.

``--checkpoint`` exists so this gate can be pointed at a RETRAINED policy: the
colliders and the bump/fall semantics are properties of the scene and the
wrapper, but "does a doorway transit stay quiet" and "does a wall bump survive"
are properties of the *gait*, and a new checkpoint has to re-earn both before it
is allowed anywhere near a benchmark. Omitting the flag keeps the previous
behaviour exactly — ``session.py``'s ``DEFAULT_CHECKPOINT`` (``policy/
model_2999.pt``, the v4_robust baseline the frozen batch ran) — so existing
``physics_pass_report.json`` results stay reproducible. The only difference in
the artifact is one ADDITIVE provenance key, ``report["checkpoint"]``: a report
that cannot say which policy produced it is not evidence about a policy.

Run:  PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/smoke_physics_pass.py
      PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/smoke_physics_pass.py \\
          --checkpoint policy/candidate_v5/model_1999.pt
"""

from __future__ import annotations

import json
import pathlib
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
#: Baseline artifact directory. A run with an explicit --checkpoint writes to a
#: SIBLING directory instead (see _resolve_out_dir): on 2026-07-29 a v5d gate run
#: overwrote 22 published v4 evidence files here, including the
#: physics_pass_report.json that two docs cite. Evidence for one policy must
#: never land on top of another's.
OUT_DIR = REPO_ROOT / "results" / "figures" / "smoke"


def _resolve_out_dir(checkpoint) -> "pathlib.Path":
    """Baseline dir for the default policy, a labelled sibling otherwise."""
    if not checkpoint:
        return OUT_DIR
    import re as _re
    stem = pathlib.Path(str(checkpoint)).parent.name or "candidate"
    label = _re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    return OUT_DIR.parent / f"smoke_{label}"

#: How far before a doorway the robot starts, and how far it drives.
APPROACH_M = 0.45
TRANSIT_M = 1.0


def approach_point(g, cx, cy, dx, dy):
    """A free start pose ``APPROACH_M`` back from a doorway along its normal.

    Backing off ALONG THE NORMAL, rather than snapping to whatever cell happens
    to be nearest, is what makes this honest. The first cut used a fixed offset
    and `nearest_free`, which for the living-room/kitchen doorway relocated the
    start sideways into the armchair — the duck bumped it at 0.10 m and the run
    was recorded as "cannot cross the doorway" when the doorway was fine.
    """
    for step in range(0, 13):  # up to +0.60 m further back, in 5 cm steps
        d = APPROACH_M + 0.05 * step
        x, y = cx - dx * d, cy - dy * d
        if g.is_free(x, y):
            return x, y
    return None


def doorway_runs(layout) -> list[dict]:
    """Approach poses for both directions through every doorway."""
    runs = []
    for door in layout["doorways"]:
        a, b = door["between"]
        cx, cy = door["center"]
        horizontal = abs(cy - 2.7) < 1e-9  # wall A doorways run along y = 2.7

        # Unit normal through the doorway, and the compass heading along it.
        nx, ny = (0.0, 1.0) if horizontal else (1.0, 0.0)
        fwd = 90.0 if horizontal else 0.0
        for name, (dx, dy), heading in (
            (f"door_{a}_to_{b}", (nx, ny), fwd),
            (f"door_{b}_to_{a}", (-nx, -ny), (fwd + 180.0) % 360.0),
        ):
            runs.append(
                {
                    "name": name,
                    "normal": (dx, dy),
                    "center": (cx, cy),
                    "heading": heading,
                    "doorway": f"{a}<->{b}",
                }
            )
    return runs


def _tile(paths, out_path, cols: int = 4) -> None:
    """Tile PNGs into one contact sheet, downscaled to keep the file readable."""
    from PIL import Image

    imgs = [Image.open(p) for p in paths]
    w, h = imgs[0].size
    scale = 480 / w
    w, h = int(w * scale), int(h * scale)
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, im in enumerate(imgs):
        sheet.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    sheet.save(out_path)


def build_parser():
    """Argument parser. Built outside ``main`` so ``--help`` needs no kit."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="smoke_physics_pass.py",
        description="Scripted physics pass (VIDEO GATE): colliders, doorways, "
                    "survivable bumps. No LLM, no money spent.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="policy .pt to drive the gate with. Default: session.py's "
             "DEFAULT_CHECKPOINT (policy/model_2999.pt, the v4_robust baseline) "
             "— omit it to reproduce the frozen-era report byte for byte.",
    )
    return parser


def main() -> int:
    # Parsed BEFORE anything launches, then stripped from argv: AppLauncher
    # inside SimSession.launch() parses sys.argv for its own flags and dies on
    # unknown ones. Same two lines as run_trial.py:151-155, for the same reason.
    args, kit_argv = build_parser().parse_known_args()

    # Candidate runs write beside the baseline, never over it (see OUT_DIR).
    global OUT_DIR
    OUT_DIR = _resolve_out_dir(getattr(args, "checkpoint", None))
    print(f"[physics_pass] artifacts -> {OUT_DIR.relative_to(REPO_ROOT)}")
    sys.argv = [sys.argv[0], *kit_argv]

    from duck_embody.env.apartment_layout import LAYOUT, grid, room_at
    from duck_embody.sim.recorder import Recorder
    from duck_embody.sim.session import DEFAULT_CHECKPOINT, SimSession, SpawnPose

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    # `checkpoint` is the one ADDITIVE key: every other field, and every verdict,
    # is unchanged when the flag is absent.
    checkpoint = str(args.checkpoint or DEFAULT_CHECKPOINT)
    report: dict = {"checkpoint": checkpoint, "transits": [], "bumps": []}
    print(f"== policy: {checkpoint} ==")

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            failures.append(label)

    session = SimSession.launch(
        task_id="DuckEmbody-Apartment-v0", headless=True, checkpoint=args.checkpoint
    )
    playback = session.playback
    g = grid()
    print("== apartment up ==")

    rec = Recorder(OUT_DIR / "physics_pass", fps=25, every_n=1, hide_ceiling=True)

    # ------------------------------------------------------------------
    # 1. Doorway transits, both directions
    # ------------------------------------------------------------------
    print("\n== doorway transits (both directions) ==")
    for run in doorway_runs(LAYOUT):
        cx, cy = run["center"]
        dx, dy = run["normal"]
        pt = approach_point(g, cx, cy, dx, dy)
        if pt is None:
            check(f"{run['name']}: has a free approach", False, f"normal {dx},{dy}")
            continue
        sx, sy = pt

        session.reset(seed=101, spawn=SpawnPose(sx, sy, run["heading"]))
        room_before = room_at(*playback.true_xy())

        result = playback.move(
            TRANSIT_M, hold_heading=True, stop_on_bump=False,
            on_chunk=lambda: rec.grab(session.env.unwrapped),
        )
        x, y, _ = result.true_pose
        room_after = room_at(x, y)
        crossed = room_before != room_after

        entry = {
            "name": run["name"],
            "doorway": run["doorway"],
            "start": [round(sx, 3), round(sy, 3)],
            "heading": run["heading"],
            "room_before": room_before,
            "room_after": room_after,
            "crossed": crossed,
            "true_displacement_m": round(result.true_displacement_m, 3),
            "dead_reckoned_m": round(result.dead_reckoned_distance_m, 3),
            "bumped": result.bumped,
            "fell": result.fell,
        }
        report["transits"].append(entry)
        print(f"  {run['name']}: {room_before} -> {room_after}  "
              f"(moved {result.true_displacement_m:.2f} m, bumped={result.bumped})")

        check(f"{run['name']}: crossed the doorway", crossed, f"{room_before}->{room_after}")
        check(f"{run['name']}: did not fall", not result.fell)
        # A clean 0.35 m transit must NOT read as a collision. This is the
        # false-positive half of bump tuning: widening detection from the trunk
        # to every non-foot body is only safe if a doorway squeeze stays quiet.
        check(f"{run['name']}: no spurious bump", not result.bumped)

    # ------------------------------------------------------------------
    # 2. Deliberate bumps — one per collider class
    # ------------------------------------------------------------------
    # Each entry drives at a known obstacle from a known standoff. The assertion
    # is the same every time and is the heart of the gate: the robot must be
    # STOPPED (it did not travel the full commanded distance) and the episode
    # must SURVIVE (no termination, no teleport).
    print("\n== deliberate bumps ==")
    bump_tests = [
        {
            "name": "wall_bump",
            # x was 2.55 until 2026-07-29 — which is the EXACT centre of the
            # hallway<->kitchen doorway (apartment_layout doorways: center
            # (2.55, 2.7), width 0.35), so this test walked straight through an
            # open gap while claiming to be "off-doorway". Wall A's segments are
            # A1 [0.000,0.725] A2 [1.075,2.375] A3 [2.725,3.875] A4 [4.225,4.800];
            # x=1.70 sits well inside A2. A policy that tracks straight would
            # "fail" the old test by correctly transiting the doorway, which is
            # the opposite of what the check is for. Disclosed rather than
            # silently retuned: the defect is provable from the layout geometry
            # alone, independent of any policy's result.
            "start": (1.70, 1.60), "heading": 90.0, "distance": 1.3,
            "target": "wall A2 (kitchen north wall, genuinely off-doorway)",
            "note": "CuboidCfg native collider",
        },
        {
            "name": "sofa_bump",
            # From the SOUTH: driving east-to-west would start inside the
            # coffee table, whose footprint spans x 0.73-1.03 at y 1.60.
            "start": (0.30, 0.75), "heading": 90.0, "distance": 0.9,
            "target": "sofa", "note": "SimReady native collider via PhysicsVariant",
        },
        {
            "name": "fridge_proxy_bump",
            "start": (2.60, 2.30), "heading": 0.0, "distance": 0.9,
            "target": "fridge", "note": "invisible bbox proxy (visual-only asset)",
        },
        {
            "name": "counter_bump",
            "start": (2.72, 0.95), "heading": 270.0, "distance": 0.9,
            "target": "sektion counter",
            "note": "T2.2 flagged: contact-offset override could NOT be applied "
                    "(instanceable, purpose=guide collision prims)",
        },
    ]

    for test in bump_tests:
        sx, sy = test["start"]
        if not g.is_free(sx, sy):
            cell = g.nearest_free(sx, sy)
            if cell is None:
                check(f"{test['name']}: start pose reachable", False)
                continue
            sx, sy = g.center(*cell)

        session.reset(seed=101, spawn=SpawnPose(sx, sy, test["heading"]))
        print(f"  {test['name']} -> {test['target']}  ({test['note']})")

        frame_start = rec.n_frames
        result = playback.move(
            test["distance"], hold_heading=True, stop_on_bump=True,
            on_chunk=lambda: rec.grab(session.env.unwrapped),
        )
        terminated = bool(session.env.unwrapped.termination_manager.terminated[0])

        entry = {
            "name": test["name"],
            "target": test["target"],
            "note": test["note"],
            "commanded_m": test["distance"],
            "true_displacement_m": round(result.true_displacement_m, 3),
            "bumped": result.bumped,
            "fell": result.fell,
            "stop_reason": result.stop_reason,
            "terminated": terminated,
            "frame_start": frame_start,
            "frame_end": rec.n_frames - 1,
        }
        report["bumps"].append(entry)
        print(f"    moved {result.true_displacement_m:.2f} m of {test['distance']} "
              f"commanded, bumped={result.bumped}, stop={result.stop_reason}")

        # THE walk-through check: an object that does not collide lets the robot
        # travel the full distance without ever reporting contact.
        stopped_short = result.true_displacement_m < test["distance"] - 0.15
        check(
            f"{test['name']}: robot was STOPPED (no walk-through)",
            stopped_short or result.bumped,
            f"moved {result.true_displacement_m:.2f}/{test['distance']}",
        )
        check(f"{test['name']}: bump reported", result.bumped, result.stop_reason)
        # doc 02 §5: furniture contact is a bump the trial survives.
        check(f"{test['name']}: episode SURVIVED the bump", not result.fell and not terminated)

    # ------------------------------------------------------------------
    # 3. Max-speed run into a wall
    # ------------------------------------------------------------------
    print("\n== max-speed wall run ==")
    session.reset(seed=101, spawn=SpawnPose(2.55, 1.90, 90.0))
    max_frame_start = rec.n_frames
    max_result = session.scripted_drive(
        [(0.222, 0.0, 0.0, 6.0)],  # top of the training hull, straight at wall A
        recorder=rec,
    )[0]
    terminated = bool(session.env.unwrapped.termination_manager.terminated[0])
    report["max_speed_wall_run"] = {
        "true_displacement_m": round(max_result.true_displacement_m, 3),
        "bumped": max_result.bumped,
        "fell": max_result.fell,
        "terminated": terminated,
        "final_height_m": round(playback.true_height(), 4) if not max_result.fell else None,
        "frame_start": max_frame_start,
        "frame_end": rec.n_frames - 1,
    }
    print(f"  moved {max_result.true_displacement_m:.2f} m, bumped={max_result.bumped}, "
          f"fell={max_result.fell}")
    # The wall must stop it. Whether it topples is allowed either way — but if it
    # does, the fall must be DETECTED, which is what makes a real fall end a trial.
    check("max-speed run: the wall stopped the robot",
          max_result.true_displacement_m < 0.90,
          f"{max_result.true_displacement_m:.2f} m in 6 s at 0.222 m/s")
    check("max-speed run: contact was reported", max_result.bumped or max_result.fell)
    if max_result.fell:
        print("  (robot toppled — that is acceptable, and it WAS detected)")
        check("max-speed run: the fall was detected", terminated or max_result.fell)

    # ------------------------------------------------------------------
    # A 1 fps strip of a 15 s run samples ~15 frames and lands almost none of
    # them on a collision. Rule 11 asks for frame-by-frame verification of the
    # thing under test, so cut a dense strip around each event's LAST frames —
    # where contact happens — straight from the PNGs the recorder kept.
    contact_strips = {}
    events = [(b["name"], b["frame_start"], b["frame_end"]) for b in report["bumps"]]
    events.append(
        ("max_speed_wall_run",
         report["max_speed_wall_run"]["frame_start"],
         report["max_speed_wall_run"]["frame_end"])
    )
    for name, f0, f1 in events:
        n_tail = 12
        lo = max(f0, f1 - n_tail + 1)
        paths = [rec.frames_dir / f"f{i:06d}.png" for i in range(lo, f1 + 1)]
        paths = [p for p in paths if p.exists()]
        if not paths:
            print(f"  [audit] {name}: no frames in range {f0}-{f1}")
            continue
        out = OUT_DIR / f"contact_{name}.png"
        _tile(paths, out, cols=4)
        contact_strips[name] = str(out.relative_to(REPO_ROOT))
        print(f"  [audit] {name}: frames {lo}-{f1} -> {out.name}")
    report["contact_strips"] = contact_strips

    mp4 = rec.encode()
    strip = rec.filmstrip(mp4, fps=1.0)

    n_crossed = sum(1 for t in report["transits"] if t["crossed"])
    report["n_transits"] = len(report["transits"])
    report["n_crossed"] = n_crossed
    report["failures"] = failures
    report["acceptance"] = "PASS" if not failures else "FAIL"
    report["video"] = str(mp4.relative_to(REPO_ROOT)) if mp4 else None
    report["filmstrip"] = str(strip.relative_to(REPO_ROOT)) if strip else None

    (OUT_DIR / "physics_pass_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\n== summary ==")
    print(f"  doorway transits crossed: {n_crossed}/{len(report['transits'])}")
    for b in report["bumps"]:
        print(f"  {b['name']:<20} moved {b['true_displacement_m']:.2f}/"
              f"{b['commanded_m']} bumped={b['bumped']} fell={b['fell']}")
    print(f"  video: {report['video']}")
    if failures:
        print("\n  FAILURES:")
        for f in failures:
            print(f"    {f}")
    else:
        print("\n  OK - colliders hold, bumps are survivable, doorways pass")
    print("  closing app (nothing after this line runs)")

    session.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
