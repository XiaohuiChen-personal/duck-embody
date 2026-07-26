"""The duck's eye: head-mounted egocentric camera + on-demand capture.

This is the model's ONLY view of the world (plus a compass and a drifting
position estimate). Everything here is frozen before the batch and identical for
all three models — the camera is part of the benchmark definition, the way the
ruler is part of the race.

Three design points worth understanding before editing:

**Mount — rung 1b, "slaved", chosen on measured evidence (doc 04 §3 ladder).**
The camera prim is a *sibling* of the robot, not a child of it, and its world
pose is written from the robot's true pose before every capture using
``set_world_poses_from_view(eye, target)``.

Rung 1 as designed — parent the camera under ``/Robot/base`` with an
``OffsetCfg`` — was implemented and **measured to fail** (T1.4, first run):

* with ``rot`` identity and ``convention="world"`` (documented as forward +X,
  up +Z, which is exactly the base frame) the camera **filmed the sky**; and
* the pose readback lied about it. ``camera.data.quat_w_world`` reported a
  level, forward-facing camera while the pixels showed the sky dome, and during
  ``look_around`` it reported an unchanged orientation at all four bearings even
  though the four frames plainly differed. ``Camera._update_poses()`` re-reads
  the pose from the prim view, and for a prim parented under a physics-driven
  articulation that readback does not survive the render.

A mount whose orientation cannot be trusted *or verified* is not a mount we can
freeze a benchmark on, so the camera moved out from under the robot.
``set_world_poses_from_view`` takes an eye point and a look-at target, so there
is **no orientation convention and no corrective quaternion anywhere** — the
class of bug doc 04 §2.1 warns about cannot occur. It also retires the
instanceable-USD risk entirely, since nothing is parented into the robot.

The cost is that the pose must be written before each capture. That is free
here: rendering is already on demand, one frame per model turn.

**The frame trap (doc 04 §2.1)** is therefore avoided rather than solved: the
MJCF head link is rotated −90° about Y so robot-forward is head-local −Z, and an
identity mount there films the sky. Aiming by eye/target sidesteps it.

**Rendering is on demand.** ``sim.render_interval`` is raised to 10,000 in the
env config, so the 50 Hz step loop never ray-traces. Frames are produced only
when the model asks, while the sim is paused, which is also why every frame shows
a settled robot rather than a mid-stride blur.
"""

from __future__ import annotations

import base64
import io
import math

#: Frozen capture parameters (doc 04 §4, AGENTS.md §2).
RESOLUTION = (512, 512)
HFOV_DEG = 90.0
JPEG_QUALITY = 85

#: Isaac's pinhole model derives HFOV from the aperture/focal ratio:
#: HFOV = 2 * atan(aperture / (2 * focal)). aperture = 2 * focal gives exactly 90°.
FOCAL_LENGTH = 12.0
HORIZONTAL_APERTURE = 2.0 * FOCAL_LENGTH * math.tan(math.radians(HFOV_DEG) / 2.0)

#: Lens offset from the robot's root frame, in the ROBOT's frame (metres):
#: forward, left, up. The root frame sits at the trunk origin, measured at
#: ~0.174 m world height while standing (T1.3/T1.4), and the head rides ~0.19 m
#: above it — putting the lens at ~0.36 m, doc 04 §4's design height.
#: smoke_camera.py prints the MEASURED world height, so this is evidence.
#:
#: MOUNT_FORWARD_M is 0.12, NOT the ~0.02 the design implied, and the difference
#: is not cosmetic: the duck's head shell surrounds the lens at small offsets and
#: the camera films *the inside of its own head*. That failure is deceptive —
#: the frame is a plausible-looking uniform light gray (the duck is white) with a
#: patch of sky showing through a gap, so it passes a naive "not black, not flat"
#: check while showing the model nothing. Measured sweep at head height
#: (scripts/debug_camera_offset.py):
#:     forward 0.02 -> mean 153.5, std 13.7, horizon delta  2.2   (inside head)
#:     forward 0.06 -> mean 152.7, std 12.7, horizon delta  3.7   (inside head)
#:     forward 0.10 -> mean  78.0, std 60.1, horizon delta 95.2   (clear)
#:     forward 0.14 -> mean  77.7, std 60.2, horizon delta 96.0   (clear)
#: 0.12 sits comfortably past the shell, roughly at the head's front face —
#: which is also where the real IMX219 module would sit.
MOUNT_FORWARD_M = 0.12
MOUNT_LEFT_M = 0.0
MOUNT_UP_M = 0.19

#: How far ahead the look-at target is placed. Only the direction matters.
LOOK_AT_DISTANCE_M = 5.0

#: The camera is a SIBLING of the robot, not a child (see the module docstring).
CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/head_cam"

#: Number of throwaway renders after a scene load or reset before a frame is fit
#: to send to a model. MDL materials stream in asynchronously and early frames
#: come back gray/black; a gray first observation would poison the model's very
#: first room guess. Isaac Lab's own built-in warmup is only 2 renders, which
#: doc 04 §5.2 judged insufficient for a full apartment. Starting value; the
#: smoke test measures the real number and writes it to configs/benchmark.yaml.
WARMUP_RENDERS = 5


def head_camera_cfg():
    """Build the frozen ``CameraCfg``. Imported lazily — needs a running kit app."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    return CameraCfg(
        prim_path=CAMERA_PRIM_PATH,
        update_period=0.0,  # we drive updates explicitly; no periodic cost
        height=RESOLUTION[1],
        width=RESOLUTION[0],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=FOCAL_LENGTH,
            horizontal_aperture=HORIZONTAL_APERTURE,
            clipping_range=(0.01, 20.0),
        ),
        # No OffsetCfg: the pose is written per capture from the robot's true
        # pose (see the module docstring). An offset here would be overwritten
        # anyway, and would reintroduce the convention trap.
    )


class HeadCamera:
    """On-demand capture from the head camera, plus the ``look_around`` gimbal."""

    def __init__(self, env, warmup_renders: int = WARMUP_RENDERS):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.sensor = self.env.scene.sensors["head_cam"]
        self.robot = self.env.scene["robot"]
        self.warmup_renders = warmup_renders

    # -- aiming -------------------------------------------------------------

    def _robot_pose(self):
        """True base position and yaw. Used to place the lens, never shown."""
        pos = self.robot.data.root_pos_w[0]
        quat = self.robot.data.root_quat_w[0]
        w, x, y, z = (float(v) for v in quat)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return (float(pos[0]), float(pos[1]), float(pos[2])), yaw

    def eye_position(self):
        """Where the lens sits in world coordinates, right now."""
        (rx, ry, rz), yaw = self._robot_pose()
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        # Rotate the body-frame mount offset into world coordinates.
        ex = rx + MOUNT_FORWARD_M * cos_y - MOUNT_LEFT_M * sin_y
        ey = ry + MOUNT_FORWARD_M * sin_y + MOUNT_LEFT_M * cos_y
        return (ex, ey, rz + MOUNT_UP_M), yaw

    def aim(self, bearing_deg: float | None = None) -> tuple:
        """Place and aim the camera. Returns (eye, forward_unit_vector).

        ``bearing_deg`` overrides the robot's own heading (used by
        ``look_around``); ``None`` means "look where the duck is facing".
        Aiming is by eye/target, so there is no orientation convention to get
        wrong — the failure mode that sank the parented mount.
        """
        import torch

        (ex, ey, ez), yaw = self.eye_position()
        heading = yaw if bearing_deg is None else math.radians(bearing_deg)
        fx, fy = math.cos(heading), math.sin(heading)

        eyes = torch.tensor(
            [[ex, ey, ez]], device=self.sensor.device, dtype=torch.float32
        )
        targets = torch.tensor(
            [[ex + LOOK_AT_DISTANCE_M * fx, ey + LOOK_AT_DISTANCE_M * fy, ez]],
            device=self.sensor.device,
            dtype=torch.float32,
        )
        self.sensor.set_world_poses_from_view(eyes, targets)
        return (ex, ey, ez), (fx, fy, 0.0)

    # -- capture ------------------------------------------------------------

    def _render_once(self) -> None:
        self.env.sim.render()
        self.sensor.update(dt=0.0, force_recompute=True)

    def warmup(self, n: int | None = None) -> None:
        """Burn N renders so MDL materials finish streaming. Frames discarded."""
        for _ in range(self.warmup_renders if n is None else n):
            self.aim()
            self._render_once()

    def capture_rgb(self, bearing_deg: float | None = None):
        """Aim from the robot's current pose, render on demand, return HxWx3 RGB."""
        self.aim(bearing_deg)
        self._render_once()
        rgb = self.sensor.data.output["rgb"][0]
        arr = rgb.detach().cpu().numpy()
        # Isaac may hand back RGBA depending on the annotator; keep 3 channels.
        return arr[..., :3]

    def capture_jpeg(self, quality: int = JPEG_QUALITY) -> bytes:
        from PIL import Image

        img = Image.fromarray(self.capture_rgb())
        if img.size != RESOLUTION:
            img = img.resize(RESOLUTION)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def capture_b64(self, quality: int = JPEG_QUALITY) -> str:
        return base64.b64encode(self.capture_jpeg(quality)).decode("ascii")

    # -- diagnostics --------------------------------------------------------

    def world_pose(self):
        """Camera (position, quaternion wxyz) as the SENSOR reports it.

        Kept for diagnostics only. Do not build behaviour on the orientation:
        T1.4 measured it disagreeing with the pixels for a robot-parented mount.
        ``aim()`` returns the authoritative forward vector.
        """
        pos = self.sensor.data.pos_w[0].detach().cpu().numpy()
        quat = self.sensor.data.quat_w_world[0].detach().cpu().numpy()
        return pos, quat

    def forward_vector(self):
        """Unit vector the lens points along, from the pose we commanded.

        Derived from the robot's true yaw rather than read back from the sensor,
        because we place the camera ourselves and that placement is the ground
        truth. The smoke test verifies ``look_around`` against this: on a
        featureless empty plane four bearings look nearly identical to a human,
        so the check must be geometric.
        """
        _, yaw = self._robot_pose()
        return (math.cos(yaw), math.sin(yaw), 0.0)

    def is_gray(self, arr, tol: int = 2) -> bool:
        """True if the frame is flat/unshaded — the MDL warmup failure mode."""
        return int(arr.max()) - int(arr.min()) <= tol

    # -- look_around --------------------------------------------------------

    def look_around(self, bearings_deg=(0, 90, 180, 270)):
        """Four captures at absolute compass bearings, sim paused, robot still.

        A *virtual gimbal*: the camera is re-aimed rather than the robot turned,
        because rotating the whole body 4x90° would burn policy-seconds and
        inject dead-reckoning error into what is a purely perceptual act.
        Declared in the methods write-up as a sensor-realistic exception (it
        approximates a pan the real head_yaw could perform).

        No pose save/restore is needed: every capture re-aims from the robot's
        live pose, so the next ``get_observation`` corrects itself by
        construction — there is no lingering state to leak.

        Returns ``[(bearing_deg, rgb_array, forward_vector), ...]``.
        """
        out = []
        for bearing in bearings_deg:
            _, forward = self.aim(bearing)
            self._render_once()
            rgb = self.sensor.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
            out.append((bearing, rgb, forward))
        # Leave the camera looking where the duck is actually facing.
        self.aim()
        self._render_once()
        return out
