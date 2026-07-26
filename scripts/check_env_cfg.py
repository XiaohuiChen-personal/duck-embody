"""T1.1 check: construct ``DuckEmbodyEnvCfg`` and assert every doc 02 §5 delta.

Config-only — it does NOT build the scene or step physics (that is T1.3's job).
It exists so a broken config fails here, attributably, instead of surfacing as a
confusing error inside the first real Isaac launch.

Run:  PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/check_env_cfg.py
"""

from __future__ import annotations

import math
import sys

from isaaclab.app import AppLauncher

# AppLauncher must run BEFORE importing isaaclab/torch/duck_embody.env: the
# config module transitively imports isaaclab_tasks, which imports pxr at module
# scope, and pxr only exists inside a running kit app.
_launcher = AppLauncher(headless=True, enable_cameras=True)
_app = _launcher.app

import gymnasium as gym  # noqa: E402

from duck_embody.env import embody_env_cfg as eec  # noqa: E402

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(label)


print("== constructing DuckEmbodyEnvCfg ==")
cfg = eec.DuckEmbodyEnvCfg()
print("  constructed OK")

print("\n== doc 02 §5 deltas ==")
check("scene.num_envs == 1", cfg.scene.num_envs == 1, str(cfg.scene.num_envs))
check("terminations.time_out is None", cfg.terminations.time_out is None)
check("terminations.base_contact is None", cfg.terminations.base_contact is None)
check(
    "fall: tilt termination added at 60 deg",
    cfg.terminations.fell_over is not None
    and math.isclose(cfg.terminations.fell_over.params["limit_angle"], math.radians(60.0)),
)
check(
    "fall: height termination added at 0.09 m",
    cfg.terminations.fell_low is not None
    and math.isclose(cfg.terminations.fell_low.params["minimum_height"], 0.09),
)

cmd = cfg.commands.base_velocity
check("commands.heading_command is False", cmd.heading_command is False)
check("commands.rel_standing_envs == 0.0", cmd.rel_standing_envs == 0.0)
check("commands.rel_heading_envs == 0.0", cmd.rel_heading_envs == 0.0)
check(
    "commands.resampling_time_range pinned to ~never",
    min(cmd.resampling_time_range) >= 1e8,
    str(cmd.resampling_time_range),
)
check(
    "commands.ranges are degenerate (0,0)",
    cmd.ranges.lin_vel_x == (0.0, 0.0)
    and cmd.ranges.lin_vel_y == (0.0, 0.0)
    and cmd.ranges.ang_vel_z == (0.0, 0.0),
)
check("commands.debug_vis is False (no velocity arrows in frame)", cmd.debug_vis is False)

pose = cfg.events.reset_base.params["pose_range"]
check(
    "reset_base pose_range degenerate (deterministic spawn)",
    all(pose[k] == (0.0, 0.0) for k in ("x", "y", "yaw")),
    str(pose),
)

print("\n== things that must SURVIVE ==")
check("reward manager kept (gait_phase obs depends on it)", cfg.rewards is not None)
check(
    "gait_phase observation term still present",
    getattr(cfg.observations.policy, "gait_phase", None) is not None,
)
check("sim.dt == 0.005", cfg.sim.dt == 0.005, str(cfg.sim.dt))
check("decimation == 4 (50 Hz control)", cfg.decimation == 4, str(cfg.decimation))
check(
    "contact_forces sensor present (bump detection)",
    getattr(cfg.scene, "contact_forces", None) is not None,
)

print("\n== video + rendering prerequisites ==")
check("viewer.origin_type == 'asset_body'", cfg.viewer.origin_type == "asset_body")
check("viewer.asset_name == 'robot'", cfg.viewer.asset_name == "robot")
check("viewer.body_name == 'trunk_assembly'", cfg.viewer.body_name == "trunk_assembly")
check(
    "sim.render_interval raised (no 50 Hz RTX renders)",
    cfg.sim.render_interval >= 1000,
    str(cfg.sim.render_interval),
)

print("\n== gym registration ==")
for task_id in (eec.TASK_ID, eec.TASK_ID_APARTMENT):
    spec = gym.registry.get(task_id)
    check(f"{task_id} registered", spec is not None)
    if spec is not None:
        check(
            f"{task_id} has env_cfg_entry_point",
            "env_cfg_entry_point" in spec.kwargs,
        )
        check(
            f"{task_id} has rsl_rl_cfg_entry_point (doc 02 §3 loader needs it)",
            "rsl_rl_cfg_entry_point" in spec.kwargs,
        )

# The loader path doc 02 §3 actually uses — prove it resolves now rather than
# discovering it inside T1.2's first OnPolicyRunner construction.
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

agent_cfg = load_cfg_from_registry(eec.TASK_ID, "rsl_rl_cfg_entry_point")
check(
    "load_cfg_from_registry(rsl_rl_cfg_entry_point) resolves",
    agent_cfg is not None,
    type(agent_cfg).__name__,
)

print("\n== result ==")
if failures:
    for f in failures:
        print(f"  FAILED: {f}")
else:
    print("  OK - every doc 02 §5 delta present and both tasks registered")
print("  closing app (nothing after this line runs)")

_app.close()
sys.exit(1 if failures else 0)
