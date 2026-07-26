"""Locomotion-policy playback and velocity-command injection.

Wraps the frozen ``v4_robust`` PPO checkpoint so the rest of the harness can say
"walk at 0.2 m/s for 3 seconds" and get back what actually happened. The loading
path is byte-for-byte the pattern the parent repo already validated over 3,200
evaluation episodes (``scripts/evaluate_policies.py:1205-1255``); novelty here
would buy only risk.

Two subtleties that are not obvious from the code alone:

* **``torch.no_grad()``, never ``torch.inference_mode()``.** Stepping the env
  inside ``inference_mode`` marks lazily-created sim-state tensors as inference
  tensors, and the next out-of-scope ``env.reset()`` — which we do between
  trials — dies with *"Inplace update to inference tensor outside InferenceMode"*.

* **Observations are fed raw.** Normalisation is baked into the checkpoint and
  applied inside ``get_inference_policy()``. Normalising upstream would
  double-normalise and produce actions that look almost plausible.

The pure math here (clamping, duration→steps, heading wrap) lives in module-level
functions with no Isaac dependency so ``tests/test_wrapper_math.py`` can exercise
it without a kit process.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- Constants -------------------------------------------------------------

#: Training command hull, read from policy/params/env.yaml (T0.1 verified).
#: Commands outside it are not just unwise, they are outside the reference-motion
#: library the gait was trained against.
VX_RANGE = (-0.148, 0.222)
VY_RANGE = (-0.111, 0.111)
WZ_RANGE = (-0.5, 0.5)

#: 50 Hz control = sim dt 0.005 s x decimation 4 (parent env_cfg.py:189-190).
CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ

#: True base XY is sampled into `pose_trace` every this many control steps.
#: 10 steps = 5 Hz. Scoring integrates this for SPL path length; sampling only
#: once per turn would miss within-turn curvature and inflate SPL (doc 06 §5.3).
POSE_TRACE_EVERY = 10

#: Contact force (N) on the trunk that counts as a bump, and how many
#: consecutive control steps must exceed it. Debouncing matters: a single
#: grazing spike while squeezing through a 0.35 m doorway must not read as a
#: collision (doc 02 §6.2). Tuned in T2.4 against the real apartment.
BUMP_FORCE_N = 1.0
BUMP_DEBOUNCE_STEPS = 3


def clamp_command(
    vx: float, vy: float, wz: float
) -> tuple[tuple[float, float, float], list[str]]:
    """Clamp to the training hull. Returns the clamped triple and any notes.

    The notes are echoed back to the model so clamping is visible rather than
    silent — a model that asks for 0.5 m/s should learn it did not get it.
    """
    notes: list[str] = []
    out = []
    for name, value, (lo, hi) in (
        ("vx", vx, VX_RANGE),
        ("vy", vy, VY_RANGE),
        ("wz", wz, WZ_RANGE),
    ):
        clamped = min(max(value, lo), hi)
        if clamped != value:
            notes.append(f"{name} {value:+.3f} clamped to {clamped:+.3f} (hull [{lo}, {hi}])")
        out.append(clamped)
    return (out[0], out[1], out[2]), notes


def duration_to_steps(duration_s: float) -> int:
    """``duration_s`` -> control steps at 50 Hz, minimum 1."""
    return max(1, round(duration_s * CONTROL_HZ))


def wrap_deg(angle_deg: float) -> float:
    """Wrap to [0, 360)."""
    return angle_deg % 360.0


def shortest_angle_diff_deg(target_deg: float, current_deg: float) -> float:
    """Signed smallest rotation from ``current`` to ``target``, in **[-180, 180)**.

    Note the half-open interval: an exact 180° error returns ``-180.0``, not
    ``+180.0``. Either direction is equally short at half a turn, so the choice
    is arbitrary — but it is deterministic, which is what the ``turn_to_heading``
    P-loop needs (an implementation that flipped sign at the boundary could
    dither there forever).
    """
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


@dataclass
class ExecResult:
    """What one command execution actually did."""

    commanded: tuple[float, float, float]
    duration_s: float
    steps: int
    policy_seconds: float
    bumped: bool
    fell: bool
    #: True base XY sampled at 5 Hz during the motion, bracketed by the exact
    #: start and end poses. SCORING ONLY — never shown to the model (doc 06 §4).
    pose_trace: list[tuple[float, float]] = field(default_factory=list)
    #: Just the periodic 5 Hz samples, without the start/end bookends. Callers
    #: that stitch several executions together (session._execute_recording)
    #: must merge THIS and add bookends once, or every chunk boundary
    #: contributes two extra near-duplicate points at the full 50 Hz step rate.
    sampled_xy: list[tuple[float, float]] = field(default_factory=list)
    #: True pose at the end, scoring only.
    true_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Straight-line true displacement over this execution, scoring only.
    true_displacement_m: float = 0.0
    clamp_notes: list[str] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str = ""


class PolicyPlayback:
    """Loads ``model_2999.pt`` and drives the env under injected commands."""

    def __init__(self, gym_env, task_id: str, checkpoint_path: str, device: str | None = None):
        # Imported here, not at module scope: these require a running kit app,
        # while the pure functions above must stay importable for unit tests.
        import importlib.metadata as metadata

        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner

        self._torch = torch
        self.gym_env = gym_env
        self.base_env = gym_env.unwrapped

        agent_cfg = load_cfg_from_registry(task_id.split(":")[-1], "rsl_rl_cfg_entry_point")
        # REQUIRED with rsl-rl-lib 5.x: the config keys were renamed, and
        # OnPolicyRunner otherwise dies with KeyError: 'class_name'. T0.1
        # confirmed the vendored agent.yaml uses the new schema.
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        if device is not None:
            agent_cfg.device = device

        self.env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        self.runner = OnPolicyRunner(
            self.env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        self.runner.load(checkpoint_path)
        # Deterministic mean actions, with the baked-in normaliser applied.
        self.policy = self.runner.get_inference_policy(device=self.base_env.device)

        self._obs = None
        self._fell = False
        # Bump debounce state lives on the INSTANCE, not inside execute().
        # session._execute_recording() chunks a long command into 0.04 s (2
        # control step) pieces so it can grab a video frame between them; a
        # per-call counter could never reach BUMP_DEBOUNCE_STEPS=3 inside a
        # 2-step chunk, so bumps would have been undetectable in exactly the
        # runs that record video — including T2.4's physics gate.
        self._bump_run = 0
        # Likewise a persistent control-step counter for pose_trace sampling.
        # doc 06 §5.3 pins that trace to 5 Hz; a per-call index would restart at
        # 0 in every 2-step recording chunk, fire `step % 10 == 0` on the first
        # step of each, and sample at ~50 Hz instead. The extra points are pure
        # per-step gait sway, which inflates the SPL path integral and would
        # have quietly depressed every recorded trial's SPL.
        self._step_counter = 0

        self.command_term = self.base_env.command_manager.get_term("base_velocity")
        self._defuse_command_term()

        self._contact_sensor = self.base_env.scene.sensors["contact_forces"]
        trunk_ids, trunk_names = self._contact_sensor.find_bodies("trunk_assembly")
        if not trunk_ids:
            raise RuntimeError(
                "No 'trunk_assembly' body in the contact sensor; bump detection "
                f"would silently never fire. Sensor bodies: {self._contact_sensor.body_names}"
            )
        self._trunk_body_ids = trunk_ids
        self._trunk_body_names = trunk_names

        self._robot = self.base_env.scene["robot"]

    # -- command channel ----------------------------------------------------

    def _defuse_command_term(self) -> None:
        """Re-assert the cfg-level defusal on the LIVE term.

        ``DuckEmbodyEnvCfg`` already sets these, but the term reads its cfg
        dynamically on every resample, so re-asserting on the constructed term
        costs nothing and closes the gap if the env was ever built from a
        different cfg (doc 02 §4).
        """
        cfg = self.command_term.cfg
        cfg.heading_command = False
        cfg.rel_standing_envs = 0.0
        cfg.resampling_time_range = (1.0e9, 1.0e9)

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        """Pin the ranges (belt) and write the buffer directly (suspenders).

        The direct write takes effect on the very next control step instead of
        waiting for a resample; the degenerate ranges mean that if anything ever
        *does* resample — notably ``env.reset()`` — it redraws the same value.
        """
        cfg = self.command_term.cfg
        cfg.ranges.lin_vel_x = (vx, vx)
        cfg.ranges.lin_vel_y = (vy, vy)
        cfg.ranges.ang_vel_z = (wz, wz)
        self.command_term.vel_command_b[:, 0] = vx
        self.command_term.vel_command_b[:, 1] = vy
        self.command_term.vel_command_b[:, 2] = wz

    # -- true state (SCORING ONLY — never shown to the model) ---------------

    def true_xy(self) -> tuple[float, float]:
        pos = self._robot.data.root_pos_w[0]
        return (float(pos[0]), float(pos[1]))

    def true_height(self) -> float:
        return float(self._robot.data.root_pos_w[0, 2])

    def compass_deg(self) -> float:
        """Absolute heading, degrees CCW from +x (doc 03 §3 convention).

        This IS given to the model — declared sensor-realistic exception (a):
        the physical duck's BNO055 IMU provides absolute yaw.
        """
        quat = self._robot.data.root_quat_w[0]
        w, x, y, z = (float(v) for v in quat)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return wrap_deg(math.degrees(yaw))

    def trunk_contact_force(self) -> float:
        forces = self._contact_sensor.data.net_forces_w[0, self._trunk_body_ids]
        return float(forces.norm(dim=-1).max())

    # -- execution ----------------------------------------------------------

    def reset(self):
        self._obs, _ = self.env.reset()
        self._fell = False
        self._bump_run = 0
        self._step_counter = 0
        return self._obs

    @property
    def fell(self) -> bool:
        return self._fell

    def execute(
        self,
        vx: float,
        vy: float,
        wz: float,
        duration_s: float,
        stop_on_bump: bool = False,
        stop_predicate=None,
    ) -> ExecResult:
        """Run one velocity command for ``duration_s`` and report what happened.

        ``stop_on_bump`` is what separates ``move`` (auto-stops on collision)
        from ``send_velocity`` (runs its full duration; doc 05 §4.2).
        ``stop_predicate(step_idx)`` lets the distance servo in ``tools.move``
        end the command as soon as it has covered the requested distance.
        """
        torch = self._torch
        (cvx, cvy, cwz), notes = clamp_command(vx, vy, wz)
        n_steps = duration_to_steps(duration_s)

        if self._obs is None:
            self.reset()

        start_xy = self.true_xy()
        sampled_xy: list[tuple[float, float]] = []
        bumped = False
        stopped_early = False
        stop_reason = ""
        steps_done = 0

        # Last pose observed while the episode was still live. On a fall this is
        # the only trustworthy final pose — see the termination branch below.
        last_live_xy = start_xy
        last_live_heading = self.compass_deg()
        terminated_this_call = False

        for step in range(n_steps):
            # Re-write every step: cheap, and it makes the command immune to
            # anything that might touch the buffer between steps.
            self.set_command(cvx, cvy, cwz)

            # Snapshot BEFORE stepping. If this step turns out to be the one
            # that terminated the episode, the post-step scene has already been
            # teleported (see below) and this snapshot — one control step, 20 ms,
            # earlier — is the closest true pose we can honestly report.
            pre_step_xy = self.true_xy()
            pre_step_heading = self.compass_deg()

            with torch.no_grad():
                actions = self.policy(self._obs)
                self._obs, _, _, _ = self.env.step(actions)

            steps_done = step + 1
            self._step_counter += 1

            # A fall is a real termination (tilt/height, per T1.1).
            # CRITICAL: Isaac Lab auto-resets a terminated env INSIDE step()
            # (manager_based_rl_env.py:216-221) and returns the post-reset
            # observation — so by the time we get here the robot has already
            # been teleported back to spawn. Reading true_xy() now would record
            # the spawn point as the fall location, quietly corrupting the SPL
            # path, the drift metric and the trajectory figure. Use the
            # pre-step snapshot and stop touching live state.
            if bool(self.base_env.termination_manager.terminated[0]):
                self._fell = True
                stopped_early = True
                stop_reason = "fell"
                terminated_this_call = True
                last_live_xy = pre_step_xy
                last_live_heading = pre_step_heading
                break

            last_live_xy = self.true_xy()
            last_live_heading = self.compass_deg()

            if self.trunk_contact_force() > BUMP_FORCE_N:
                self._bump_run += 1
                if self._bump_run >= BUMP_DEBOUNCE_STEPS:
                    bumped = True
                    if stop_on_bump:
                        stopped_early = True
                        stop_reason = "bump"
                        break
            else:
                self._bump_run = 0

            if self._step_counter % POSE_TRACE_EVERY == 0:
                sampled_xy.append(last_live_xy)

            if stop_predicate is not None and stop_predicate(step):
                stopped_early = True
                stop_reason = "target_reached"
                break

        # Never leave a command armed: the sim pauses between LLM turns, and a
        # live command must not be waiting when the next macro starts.
        # (Safe even after a termination — it only writes command buffers.)
        self.set_command(0.0, 0.0, 0.0)

        end_xy = last_live_xy
        end_heading = last_live_heading
        if not terminated_this_call:
            end_xy = self.true_xy()
            end_heading = self.compass_deg()
        pose_trace = [start_xy, *sampled_xy, end_xy]

        return ExecResult(
            commanded=(cvx, cvy, cwz),
            duration_s=duration_s,
            steps=steps_done,
            policy_seconds=steps_done * CONTROL_DT,
            bumped=bumped,
            fell=self._fell,
            pose_trace=pose_trace,
            sampled_xy=sampled_xy,
            true_pose=(end_xy[0], end_xy[1], end_heading),
            true_displacement_m=math.dist(start_xy, end_xy),
            clamp_notes=notes,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )

    def settle(self, duration_s: float = 0.4) -> None:
        """Step with a zero command so the gait comes to rest before a capture."""
        self.execute(0.0, 0.0, 0.0, duration_s)
