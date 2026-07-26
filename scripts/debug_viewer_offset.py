"""Pick a chase-camera offset that can actually AUDIT the apartment (T2.4).

The first T2.4 video was unusable: the ViewerCfg eye offset (1.2, 1.2, 0.6) is a
1.7 m diagonal, but the rooms are 1.5-1.8 m across and now have a ceiling (added
by the T2.3 gate), so the chase camera spent most of the run *inside* a wall
slab, filming featureless white. Rule 11 wants frame-by-frame verification; you
cannot verify what you cannot see.

Rather than guess a new offset, this sweeps candidates from the same robot pose
in the WORST case — the duck tucked into a room corner, where the camera is most
likely to end up buried — and writes one frame per candidate to compare.

Run:  PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/debug_viewer_offset.py
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "figures" / "smoke" / "viewer_offsets"

# All of these are run with the CEILING HIDDEN. The in-room candidates from the
# first sweep are gone: at 1.5-1.8 m room width every one of them was either
# buried in a wall or jammed against furniture, and the best std score of the lot
# (75.4, "e") turned out to be a close-up of the stove with no duck in frame —
# std measures variance, not usefulness, the same trap as the in-head camera in
# T1.4. These look DOWN from above the 0.7 m walls instead.
CANDIDATES = [
    # Round 3. Round 2's overhead offsets cleared the walls but put the duck
    # 2.3 m away: in the T2.4 audit strip it was a ~30 px speck, and behind the
    # kitchen counter it vanished entirely. An audit frame has to show whether
    # the duck is TOUCHING the obstacle or INSIDE it, which needs it large.
    # So: close AND steep — near enough to fill the frame, high enough that the
    # 0.7 m walls stay below the sightline.
    ("n_close_0.4_0.4_1.2", (0.4, 0.4, 1.20), (0.0, 0.0, 0.15)),
    ("o_close_0.5_0.5_1.0", (0.5, 0.5, 1.00), (0.0, 0.0, 0.15)),
    ("p_close_0.3_0.3_1.0", (0.3, 0.3, 1.00), (0.0, 0.0, 0.15)),
    ("q_close_0.6_0.6_1.4", (0.6, 0.6, 1.40), (0.0, 0.0, 0.15)),
    ("r_topdown_0.0_0.0_1.2", (0.0, 0.0, 1.20), (0.0, 0.0, 0.15)),
    ("s_close_0.7_0.7_1.2", (0.7, 0.7, 1.20), (0.0, 0.0, 0.15)),
]

#: Worst case for a chase camera: the duck in a corner of the smallest room.
POSES = [
    ("kitchen_corner", (2.05, 0.30, 90.0)),
    ("living_open", (0.90, 1.20, 0.0)),
    # The actual T2.4 contact poses — where the audit has to work or it is
    # decorative. Standoffs are ~0.3 m short of the obstacle, i.e. mid-bump.
    ("at_fridge", (2.88, 2.30, 0.0)),
    ("at_wall", (2.55, 2.55, 90.0)),
]


def main() -> int:
    import numpy as np
    from PIL import Image

    from duck_embody.sim.session import SimSession, SpawnPose

    from duck_embody.env.scene_builder import ceiling_hidden

    OUT.mkdir(parents=True, exist_ok=True)
    session = SimSession.launch(task_id="DuckEmbody-Apartment-v0", headless=True)
    env = session.env.unwrapped
    vcc = env.viewport_camera_controller

    with ceiling_hidden(env.sim.stage):
     for pose_name, (x, y, h) in POSES:
        session.reset(seed=101, spawn=SpawnPose(x, y, h))
        session.playback.settle(0.4)
        for label, eye, lookat in CANDIDATES:
            vcc.update_view_location(eye=eye, lookat=lookat)
            # STEP after setting the view. update_view_location() composes the
            # offset with the controller's CACHED tracking origin, which only
            # refreshes on a sim step — without this the camera aims at wherever
            # the duck used to be, and round 2/3 of this sweep measured frames
            # whose subject was half out of shot.
            session.playback.settle(0.12)
            env.sim.render()
            frame = env.render(recompute=True)
            if frame is None:
                print(f"  {pose_name}/{label}: NO FRAME")
                continue
            arr = np.asarray(frame)
            # A buried camera films one flat surface: near-zero spatial variance.
            std = float(arr.std())
            path = OUT / f"{pose_name}__{label}.png"
            Image.fromarray(arr).save(path)
            print(f"  {pose_name:<16} {label:<24} std={std:6.2f}  {path.name}")

    print(f"\nwrote frames to {OUT.relative_to(REPO_ROOT)}")
    print("closing app (nothing after this line runs)")
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
