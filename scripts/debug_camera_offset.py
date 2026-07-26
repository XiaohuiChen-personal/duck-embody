"""T1.4 diagnostic: find a lens offset that is OUTSIDE the duck's own head.

The first slaved-mount run rendered a near-uniform light-gray field with a small
blue disc and a dark wedge at one corner. That is not sky — it is the inside of
the robot's own head shell (the duck is white), seen from a lens placed 0.02 m
forward of the root at head height. The disc and wedge are gaps in the geometry.

This sweeps the forward offset and saves one frame each, plus a top-down
reference frame that proves the aiming path itself is sound.

Run:  PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/debug_camera_offset.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "figures" / "smoke" / "camera_offset_sweep"

FORWARD_OFFSETS = [0.02, 0.06, 0.10, 0.14, 0.18]


def main() -> int:
    import torch
    from PIL import Image

    from duck_embody.env import camera as camera_mod
    from duck_embody.env.camera import HeadCamera
    from duck_embody.sim.session import SimSession, SpawnPose

    OUT.mkdir(parents=True, exist_ok=True)
    session = SimSession.launch(task_id="DuckEmbody-v0", headless=True)
    session.reset(seed=42, spawn=SpawnPose(0.0, 0.0, 0.0))
    cam = HeadCamera(session.env)
    cam.warmup()

    stats = {}

    # Reference: look at the duck from 2 m away and above. If THIS shows a duck
    # on a grid floor, the eye/target aiming path is correct and any weirdness
    # in the head shots is about placement, not orientation.
    eyes = torch.tensor([[1.5, 1.5, 1.2]], device=cam.sensor.device, dtype=torch.float32)
    targets = torch.tensor([[0.0, 0.0, 0.15]], device=cam.sensor.device, dtype=torch.float32)
    cam.sensor.set_world_poses_from_view(eyes, targets)
    cam._render_once()
    arr = cam.sensor.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
    Image.fromarray(arr).save(OUT / "ref_thirdperson.png")
    stats["ref_thirdperson"] = {
        "mean": round(float(arr.mean()), 2),
        "std": round(float(arr.std()), 2),
    }
    print(f"ref_thirdperson: mean={arr.mean():.1f} std={arr.std():.1f}")

    for fwd in FORWARD_OFFSETS:
        camera_mod.MOUNT_FORWARD_M = fwd
        cam.aim()
        cam._render_once()
        arr = cam.sensor.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
        name = f"fwd_{fwd:.2f}"
        Image.fromarray(arr).save(OUT / f"{name}.png")
        top = float(arr[:256].mean())
        bottom = float(arr[256:].mean())
        stats[name] = {
            "mean": round(float(arr.mean()), 2),
            "std": round(float(arr.std()), 2),
            "top_half": round(top, 2),
            "bottom_half": round(bottom, 2),
            "horizon_delta": round(abs(top - bottom), 2),
        }
        print(
            f"fwd={fwd:.2f}: mean={arr.mean():6.1f} std={arr.std():6.2f} "
            f"top={top:6.1f} bottom={bottom:6.1f} delta={abs(top - bottom):6.2f}"
        )

    (OUT / "sweep.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    print("  closing app (nothing after this line runs)")
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
