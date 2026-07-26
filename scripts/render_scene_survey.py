"""T2.3 part 1 (ONE kit launch): render the scene survey.

Captures, into ``results/figures/survey/``:

* a **top-down** view of the whole apartment, directly comparable to
  ``layout_plan.png`` and doc 03's floor plan;
* a **duck-height sweep**: several poses per room x four bearings each, from
  0.36 m with the frozen camera config — exactly what ``look_around()`` will
  produce during a trial.

It also re-measures the **MDL warmup**, which doc 04 §5.2 left open and T1.4
could only answer for the empty plane. This is the scene that actually has
materials to stream, so this is the number that gets frozen.

Part 2 (``scripts/judge_scene_survey.py``) runs **after this process exits** —
`SimulationApp.close()` terminates the process, and the judge must not share a
process with a kit app anyway.

Run:  PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/render_scene_survey.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "figures" / "survey"

#: Poses per room for the duck-height sweep. Chosen to spread across each room
#: rather than clustering, since a gate passed from one lucky vantage proves
#: nothing about what the model will see mid-trial. Snapped to free space at
#: runtime if any lands inside the inflation margin.
#: FIVE poses per room, not three. With a judge that cannot be pinned to a
#: temperature, three samples gave a majority that flipped between runs on
#: identical frames (measured: hallway scored 3/3, 2/3 and a 1/1/1 tie across
#: three runs). Raising the sample count is a fix to the INSTRUMENT's power, not
#: a relaxation of the bar — the acceptance criterion is still all four rooms.
SWEEP_POSES = {
    "living_room": [(0.90, 0.60), (1.35, 1.85), (0.62, 2.35), (1.40, 0.50), (0.75, 1.10)],
    "kitchen": [(2.55, 0.78), (2.25, 1.95), (2.85, 1.35), (2.10, 0.90), (2.60, 2.30)],
    "bedroom": [(3.70, 0.60), (4.40, 1.90), (3.68, 2.40), (4.62, 0.45), (3.95, 1.55)],
    "hallway": [(0.70, 3.15), (2.40, 3.15), (4.00, 3.15), (1.50, 3.08), (3.40, 3.20)],
}

BEARINGS = (0, 90, 180, 270)


def main() -> int:
    from PIL import Image

    from duck_embody.env.apartment_layout import LAYOUT, grid, room_at
    from duck_embody.env.camera import HeadCamera
    from duck_embody.sim.session import SimSession, SpawnPose

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"poses": {}, "frames": []}

    # Building the apartment happens inside this call — T2.2's construction
    # errors surface here, attributably, before anything is rendered.
    session = SimSession.launch(task_id="DuckEmbody-Apartment-v0", headless=True)
    print("== apartment scene built ==")
    session.reset(seed=101, spawn=SpawnPose(*LAYOUT["spawn_points"][101]["pos"],
                                            LAYOUT["spawn_points"][101]["heading_deg"]))
    cam = HeadCamera(session.env)

    # -- warmup, re-measured against the FURNISHED scene -------------------
    print("\n== warmup (furnished scene — the number doc 04 §5.2 actually wants) ==")
    measured_warmup = None
    for i in range(1, 21):
        arr = cam.capture_rgb()
        spread = int(arr.max()) - int(arr.min())
        std = float(arr.std())
        if i <= 10:
            print(f"    render {i:>2}: spread {spread:>3}  std {std:6.2f}")
        if measured_warmup is None and not cam.is_gray(arr):
            measured_warmup = i
    report["measured_warmup_renders_furnished"] = measured_warmup
    print(f"  first non-gray frame at render {measured_warmup}")

    cam.warmup(8)

    # -- top-down -----------------------------------------------------------
    # The ceiling is hidden for this shot only. doc 03 §3.1 wants walls "low
    # enough for top-down debug renders"; the roof T2.3 added to make the rooms
    # read as interiors would otherwise end that, so we toggle it rather than
    # choosing between the two.
    print("\n== top-down (ceiling hidden for this shot) ==")
    import torch
    from pxr import UsdGeom

    from duck_embody.env.scene_builder import CEILING_PRIM_PATH

    stage = session.env.unwrapped.sim.stage
    ceiling_prim = stage.GetPrimAtPath(CEILING_PRIM_PATH)
    if ceiling_prim and ceiling_prim.IsValid():
        UsdGeom.Imageable(ceiling_prim).MakeInvisible()
        print("  ceiling hidden")
    else:
        print(f"  WARNING: no ceiling prim at {CEILING_PRIM_PATH}")

    w, h = LAYOUT["extents"]
    eye = torch.tensor([[w / 2, h / 2, 6.0]], device=cam.sensor.device, dtype=torch.float32)
    # Look straight down. The target is offset a hair in +y so the up-vector is
    # well defined (a perfectly vertical view has a degenerate roll).
    tgt = torch.tensor([[w / 2, h / 2 + 1e-3, 0.0]], device=cam.sensor.device, dtype=torch.float32)
    cam.sensor.set_world_poses_from_view(eye, tgt)
    for _ in range(4):
        cam._render_once()
    arr = cam.sensor.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
    Image.fromarray(arr).save(OUT_DIR / "topdown.png")
    print(f"  wrote topdown.png  (mean {arr.mean():.1f}, std {arr.std():.1f})")
    report["topdown"] = {"mean": round(float(arr.mean()), 2), "std": round(float(arr.std()), 2)}

    if ceiling_prim and ceiling_prim.IsValid():
        UsdGeom.Imageable(ceiling_prim).MakeVisible()
        for _ in range(3):
            cam._render_once()
        print("  ceiling restored for the sweep")

    # -- duck-height sweep --------------------------------------------------
    print("\n== duck-height sweep (0.36 m, frozen camera config) ==")
    g = grid()
    eye_z = 0.36

    for room, poses in SWEEP_POSES.items():
        report["poses"][room] = []
        for pi, (px, py) in enumerate(poses):
            # Snap into free space if the chosen pose sits inside the inflation
            # margin — a camera embedded in a wall proves nothing.
            if not g.is_free(px, py):
                cell = g.nearest_free(px, py)
                if cell is None:
                    print(f"  SKIP {room} pose {pi}: no free space near {(px, py)}")
                    continue
                px, py = g.center(*cell)
                print(f"  snapped {room} pose {pi} to {(round(px, 2), round(py, 2))}")

            actual_room = room_at(px, py)
            report["poses"][room].append(
                {"index": pi, "xy": [round(px, 3), round(py, 3)], "resolved_room": actual_room}
            )
            if actual_room != room:
                print(f"  WARNING {room} pose {pi} resolved to {actual_room}")

            for bearing in BEARINGS:
                fx, fy = math.cos(math.radians(bearing)), math.sin(math.radians(bearing))
                eye = torch.tensor(
                    [[px, py, eye_z]], device=cam.sensor.device, dtype=torch.float32
                )
                tgt = torch.tensor(
                    [[px + 5.0 * fx, py + 5.0 * fy, eye_z]],
                    device=cam.sensor.device,
                    dtype=torch.float32,
                )
                cam.sensor.set_world_poses_from_view(eye, tgt)
                cam._render_once()
                frame = cam.sensor.data.output["rgb"][0].detach().cpu().numpy()[..., :3]

                name = f"{room}_p{pi}_b{bearing:03d}.png"
                Image.fromarray(frame).save(OUT_DIR / name)
                report["frames"].append(
                    {
                        "file": name,
                        "room": room,
                        "pose_index": pi,
                        "xy": [round(px, 3), round(py, 3)],
                        "bearing_deg": bearing,
                        "mean": round(float(frame.mean()), 2),
                        "std": round(float(frame.std()), 2),
                        "is_gray": bool(cam.is_gray(frame)),
                    }
                )
        print(f"  {room}: {len(report['poses'][room])} poses x {len(BEARINGS)} bearings")

    gray = [f for f in report["frames"] if f["is_gray"]]
    report["n_frames"] = len(report["frames"])
    report["n_gray"] = len(gray)

    print("\n== summary ==")
    print(f"  frames: {report['n_frames']}   gray: {report['n_gray']}")
    print(f"  warmup (furnished): {measured_warmup}")
    if gray:
        print("  GRAY FRAMES (would poison the model's room guess):")
        for f in gray[:10]:
            print(f"    {f['file']}")

    (OUT_DIR / "survey_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"  wrote {(OUT_DIR / 'survey_manifest.json').relative_to(REPO_ROOT)}")
    print("\n  next: ~/IsaacLab/isaaclab.sh -p scripts/judge_scene_survey.py")
    print("  closing app (nothing after this line runs)")

    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
