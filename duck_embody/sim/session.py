"""The persistent Isaac Sim session — one kit process for the whole batch.

Kit cold start costs minutes and this machine allows exactly one GPU job
(AGENTS.md rule 1), so the entire 12-trial batch runs inside a single process
with an env *reset* between trials rather than a relaunch.

Import-order rules this module enforces so callers do not have to remember them:

* ``AppLauncher`` must be constructed **before** ``torch`` or any ``isaaclab``
  module is imported. ``SimSession.launch()`` is therefore a classmethod that
  does the launching, and every heavyweight import inside this file is deferred.
* ``SimulationApp.close()`` **terminates the process** — nothing after it runs.
  Callers must write artifacts and print verdicts before ``close()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "policy" / "model_2999.pt"


@dataclass
class SpawnPose:
    x: float
    y: float
    heading_deg: float


class SimSession:
    """Owns the kit app, the gym env, and the policy playback wrapper."""

    def __init__(self, app, env, playback, task_id: str):
        self.app = app
        self.env = env
        self.playback = playback
        self.task_id = task_id

    # -- construction -------------------------------------------------------

    @classmethod
    def launch(
        cls,
        task_id: str = "DuckEmbody-v0",
        checkpoint: Path | str | None = None,
        headless: bool = True,
        enable_cameras: bool = True,
        device: str | None = None,
    ) -> "SimSession":
        """Start kit, build the env, load the policy. Call this exactly once."""
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=headless, enable_cameras=enable_cameras)
        app = launcher.app

        # Everything below must come AFTER AppLauncher.
        import gymnasium as gym
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        # Registers DuckEmbody-v0 / DuckEmbody-Apartment-v0 as an import side
        # effect (and bootstraps the pinned parent repo).
        from duck_embody.env import embody_env_cfg  # noqa: F401
        from duck_embody.sim.policy_wrapper import PolicyPlayback

        # Pass `device` only when the caller actually chose one: parse_env_cfg
        # defaults to "cuda:0", and forwarding None overrides that default with
        # None, which blows up much later inside
        # SimulationManager.set_physics_sim_device ('in' on a NoneType).
        cfg_kwargs = {"num_envs": 1}
        if device is not None:
            cfg_kwargs["device"] = device
        env_cfg = parse_env_cfg(task_id, **cfg_kwargs)

        # render_mode="rgb_array" is NOT optional: without it
        # ManagerBasedRLEnv.render() returns None, every Recorder.grab() is a
        # silent no-op, and rule-11 smoke tests produce no video at all
        # (verified against Isaac Lab 2.3.2).
        env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array")

        ckpt = str(checkpoint or DEFAULT_CHECKPOINT)
        playback = PolicyPlayback(env, task_id=task_id, checkpoint_path=ckpt, device=device)

        return cls(app=app, env=env, playback=playback, task_id=task_id)

    # -- episode control ----------------------------------------------------

    def reset(self, seed: int | None = None, spawn: SpawnPose | None = None):
        """Reset to a pinned spawn pose.

        The spawn is applied by rewriting the ``reset_base`` event's degenerate
        ``pose_range`` before resetting — the same dynamic-cfg trick the command
        term uses, and the reason ``DuckEmbodyEnvCfg`` pins those ranges to
        ``(0, 0)`` instead of leaving the inherited ±0.5 m / ±π randomisation.
        """
        base_env = self.env.unwrapped
        if spawn is not None:
            yaw = math.radians(spawn.heading_deg)
            base_env.cfg.events.reset_base.params["pose_range"] = {
                "x": (spawn.x, spawn.x),
                "y": (spawn.y, spawn.y),
                "yaw": (yaw, yaw),
            }
            # The event term caches its params dict at construction, so the live
            # term must be updated too — updating only cfg silently resets to
            # the previous seed's spawn.
            term_cfg = base_env.event_manager.get_term_cfg("reset_base")
            term_cfg.params["pose_range"] = base_env.cfg.events.reset_base.params["pose_range"]

        if seed is not None:
            base_env.seed(seed)

        obs = self.playback.reset()

        # Settle briefly so the first observation shows a standing duck rather
        # than one mid-drop from the spawn pose.
        self.playback.settle(0.5)
        return obs

    def scripted_drive(self, script, recorder=None, log=None):
        """Run a list of ``(vx, vy, wz, duration_s)`` commands with no LLM.

        Used by the T1.3 displacement smoke and the T2.4 physics pass. Returns
        the per-segment ``ExecResult``s.
        """
        results = []
        for segment in script:
            vx, vy, wz, duration = segment[:4]
            opts = segment[4] if len(segment) > 4 else {}
            result = self._execute_recording(vx, vy, wz, duration, recorder, **opts)
            results.append(result)
            if log is not None:
                log(result)
            if result.fell:
                break
        return results

    def _execute_recording(self, vx, vy, wz, duration_s, recorder, **kwargs):
        """Execute one command, grabbing a viewport frame per control step.

        Frame grabbing has to interleave with stepping, so this re-implements
        the loop rather than calling ``PolicyPlayback.execute`` — done by
        chunking the command into short executes and grabbing between them,
        which keeps a single definition of the stepping semantics.
        """
        from duck_embody.sim.policy_wrapper import CONTROL_DT, ExecResult, duration_to_steps

        if recorder is None:
            return self.playback.execute(vx, vy, wz, duration_s, **kwargs)

        if kwargs.get("stop_predicate") is not None:
            # A stop_predicate is called with the step index *within one
            # execute()*. Chunking restarts that index every 2 steps, so a
            # predicate like "stop after 40 steps" would never fire. Closed-loop
            # macros (tools.move) run their own loop and record per iteration.
            raise ValueError(
                "stop_predicate cannot be combined with a recorder: chunked "
                "recording restarts the per-call step index. Drive the loop from "
                "the caller instead."
            )

        # 0.04 s chunks = 2 control steps: fine enough for 25 fps video while
        # keeping per-chunk overhead negligible.
        chunk_s = 0.04
        total_steps = duration_to_steps(duration_s)
        done_steps = 0
        merged: ExecResult | None = None
        start_xy: tuple[float, float] | None = None
        sampled: list[tuple[float, float]] = []

        while done_steps < total_steps:
            remaining = (total_steps - done_steps) * CONTROL_DT
            part = self.playback.execute(vx, vy, wz, min(chunk_s, remaining), **kwargs)
            recorder.grab(self.env.unwrapped)
            done_steps += part.steps

            if start_xy is None:
                start_xy = part.pose_trace[0]
            # Merge the 5 Hz SAMPLES only. Concatenating each chunk's full
            # pose_trace would add its start/end bookends — two extra points per
            # 2-step chunk, i.e. a ~50 Hz trace of per-step gait sway, which
            # inflates the SPL path integral (doc 06 §5.3 pins it to 5 Hz).
            sampled.extend(part.sampled_xy)

            if merged is None:
                merged = part
            else:
                merged.steps += part.steps
                merged.policy_seconds += part.policy_seconds
                merged.bumped = merged.bumped or part.bumped
                merged.fell = part.fell
                merged.true_pose = part.true_pose
                merged.stopped_early = part.stopped_early
                merged.stop_reason = part.stop_reason or merged.stop_reason

            if part.fell or part.stopped_early:
                break

        end_xy = (merged.true_pose[0], merged.true_pose[1])
        merged.duration_s = duration_s
        merged.sampled_xy = sampled
        merged.pose_trace = [start_xy, *sampled, end_xy]
        merged.true_displacement_m = math.dist(start_xy, end_xy)
        return merged

    # -- teardown -----------------------------------------------------------

    def close(self) -> None:
        """Shut down. NOTHING AFTER THIS RUNS — write artifacts first."""
        try:
            self.env.close()
        finally:
            self.app.close()
