"""``DuckEmbodyEnvCfg`` — the Isaac Lab environment the LLM drives.

**This is the ONLY file in Duck Embody permitted to import the parent robot
repo** (AGENTS.md rule 8). Everything else talks to the sim through
``duck_embody.sim``.

It subclasses the parent's playback config and applies every delta in design
doc 02 §5. Four of those deltas disarm inherited behaviours that would silently
sabotage an LLM-driven episode — each is annotated with the failure it prevents,
because none of them produce an error message when left at their defaults:

* ``time_out = None`` — otherwise the env **teleports the robot** mid-episode.
* ``base_contact`` removed — otherwise brushing a chair ends the trial.
* tilt/height fall termination added — so a *real* fall still ends it.
* command term defused — otherwise the LLM's velocity commands are overwritten.

Import order matters: ``AppLauncher`` must already have started the kit app
before this module is imported, because ``isaaclab_tasks`` (pulled in
transitively) imports ``pxr`` at module scope.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Parent-repo bootstrap (rule 8: read-only, pinned, imported from here only)
# ---------------------------------------------------------------------------


def _project_table() -> dict:
    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return cfg["tool"]["duck-embody"]


def _parent_repo_path() -> Path:
    env_override = os.environ.get("DUCK_EMBODY_PARENT_REPO")
    if env_override:
        return Path(env_override).expanduser()
    return Path(_project_table()["parent_repo_path"]).expanduser()


def bootstrap_parent_repo() -> Path:
    """Put the parent repo on ``sys.path`` and check its commit against the pin.

    A **warning, not an error**, on mismatch: the pin records what the vendored
    policy was produced against, and a divergent parent is worth flagging loudly,
    but refusing to run would make the harness unusable during parent-side work.
    A missing parent repo *is* fatal — nothing downstream can function.
    """
    parent = _parent_repo_path()
    if not parent.is_dir():
        raise RuntimeError(
            f"Parent robot repo not found at {parent}. Set DUCK_EMBODY_PARENT_REPO "
            "or fix [tool.duck-embody].parent_repo_path in pyproject.toml."
        )

    pinned = _project_table()["parent_repo_commit"]
    try:
        actual = subprocess.run(
            ["git", "-C", str(parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - a missing git is not fatal here
        warnings.warn(f"Could not read parent repo commit ({exc}); pin unverified.")
    else:
        if actual != pinned:
            warnings.warn(
                "Parent repo commit MISMATCH.\n"
                f"  pinned (pyproject.toml): {pinned}\n"
                f"  actual ({parent}): {actual}\n"
                "The vendored policy in policy/ was produced at the pinned commit. "
                "Results from a divergent parent are not comparable to earlier runs.",
                stacklevel=2,
            )

    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    return parent


bootstrap_parent_repo()

# Importing the parent package registers its gym tasks as a side effect, and
# gives us the PLAY config we subclass.
from isaac_lab_env.open_duck_mini_v2.env_cfg import (  # noqa: E402
    OpenDuckRobustEnvCfg_PLAY,
)

import gymnasium as gym  # noqa: E402
import isaaclab.envs.mdp as mdp  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (doc 02 §5, doc 03 §3)
# ---------------------------------------------------------------------------

#: Training command hull, verified in policy/params/env.yaml. Everything the LLM
#: sends is clamped to this before injection.
COMMAND_HULL = {
    "vx": (-0.148, 0.222),
    "vy": (-0.111, 0.111),
    "wz": (-0.5, 0.5),
}

#: Fall thresholds (doc 02 §5). Nominal trunk height is ~0.17 m, so 0.09 m is
#: unambiguous; 60 deg of tilt is far beyond normal gait oscillation.
FALL_TILT_LIMIT_RAD = math.radians(60.0)
FALL_MIN_HEIGHT_M = 0.09

#: Effectively-never resample interval (~31 years of sim time). See below.
NEVER_RESAMPLE_S = 1.0e9

#: Body whose contact forces are read as the `bumped` status flag, and which the
#: tracking viewport follows.
TRUNK_BODY = "trunk_assembly"

TASK_ID = "DuckEmbody-v0"
TASK_ID_APARTMENT = "DuckEmbody-Apartment-v0"


@configclass
class DuckEmbodyEnvCfg(OpenDuckRobustEnvCfg_PLAY):
    """One duck, one world, no auto-reset, no command hijacking. Empty plane.

    The apartment subclass below adds the scene; this base is what the Phase-1
    smoke tests (T1.3 displacement, T1.4 camera) run against.
    """

    # Rule-11 tracking video: the recorder grabs viewport frames via
    # env.render(), so the viewport must follow the robot or every mp4 is a
    # fixed shot of the duck walking out of frame. "asset_body" tracks a
    # specific body rather than the articulation root.
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.2, 1.2, 0.6),
        lookat=(0.0, 0.0, 0.15),
        origin_type="asset_body",
        asset_name="robot",
        body_name=TRUNK_BODY,
        env_index=0,
        resolution=(1280, 720),
    )

    def __post_init__(self):
        super().__post_init__()

        # --- Scene: a single agent, not a training farm -----------------------
        self.scene.num_envs = 1

        # --- Terminations ----------------------------------------------------
        # time_out: Isaac Lab auto-resets a done env INSIDE env.step() and
        # returns the post-reset observation. With the inherited 40 s episode the
        # duck would silently teleport to its spawn mid-trial, corrupting dead
        # reckoning, the LLM's map, and scoring — with no exception raised.
        # The harness enforces caps instead (40 turns / 240 policy-seconds).
        self.terminations.time_out = None
        # Belt-and-suspenders: nothing should consult the episode clock now, but
        # a large value means any code path that does cannot trip within a trial.
        self.episode_length_s = 1.0e6

        # base_contact: >1 N on the trunk ends the episode. In a furnished
        # apartment the trunk WILL brush walls and furniture; that is normal
        # exploration, not a failure. The same contact sensor is instead read as
        # the `bumped` status flag (doc 02 §5, doc 01 §8).
        self.terminations.base_contact = None

        # ...but a genuine fall must still end the trial: the policy is 59-dim
        # proprioceptive with no get-up skill, so continuing would be theatre.
        # Tilt via projected gravity, plus a height floor.
        self.terminations.fell_over = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": FALL_TILT_LIMIT_RAD},
        )
        self.terminations.fell_low = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": FALL_MIN_HEIGHT_M},
        )

        # --- Commands: the LLM owns this channel -----------------------------
        # Three inherited behaviours fight an external commander, all silently:
        #   heading_command=True  -> a P-controller rewrites wz EVERY step
        #   rel_standing_envs>0   -> the env may be flagged "standing" and zeroed
        #   resampling 10 s       -> a fresh random command replaces ours
        # Defused here at cfg level; duck_embody.sim.policy_wrapper re-asserts the
        # same values on the LIVE term (the term reads cfg dynamically, so both
        # layers are cheap and the pair is genuinely belt-and-suspenders).
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_standing_envs = 0.0
        cmd.rel_heading_envs = 0.0
        cmd.resampling_time_range = (NEVER_RESAMPLE_S, NEVER_RESAMPLE_S)

        # Degenerate ranges: if anything ever DOES resample (notably env.reset(),
        # which resamples all envs), it can only redraw a standstill rather than
        # dealing the duck a random walk before the LLM's first turn.
        cmd.ranges.lin_vel_x = (0.0, 0.0)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)

        # Command arrows would render into the head camera and hand the model a
        # picture of its own commanded velocity (doc 04 §7 info-leak hygiene).
        cmd.debug_vis = False

        # --- Spawn: pinned per seed, never random ----------------------------
        # The inherited reset randomises xy by +/-0.5 m and yaw over the full
        # circle. Degenerate ranges make reset deterministic; the session
        # rewrites these three entries per seed before calling env.reset().
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        # --- Rewards: KEPT, deliberately -------------------------------------
        # We are not training and the reward values are discarded. But
        # ImitationReward's step counter feeds observation dims 58-59
        # (gait_phase) through a module-global registry, and the observation
        # function returns ZEROS when no reward instance is registered. Removing
        # the reward manager would silently corrupt 2 of 59 policy inputs and
        # degrade gait with no error. See doc 02 §2 and policy/README.md.

        # --- Rendering: on demand, not every control step --------------------
        # Adding an RTX camera makes the step loop render every
        # sim.render_interval physics steps (= 50 Hz here), i.e. ~12,000 wasted
        # ray-traced frames per stage when the model looks at one per turn.
        # duck_embody.env.camera renders explicitly while the sim is paused.
        self.sim.render_interval = 10_000


@configclass
class DuckEmbodyApartmentEnvCfg(DuckEmbodyEnvCfg):
    """The benchmark world: the same duck, inside the apartment of doc 03.

    Split into its own class (rather than a mutable ``apartment=None|LAYOUT``
    field) so both worlds are ordinary registered tasks. A configclass field
    holding the layout dict would be a mutable default shared across instances,
    and the scene has to be built in ``__post_init__`` regardless.
    """

    def __post_init__(self):
        super().__post_init__()
        # Imported lazily: scene_builder pulls in isaaclab spawner modules, and
        # the empty-plane config above must stay importable without them.
        from duck_embody.env.apartment_layout import LAYOUT
        from duck_embody.env.scene_builder import add_apartment_to_scene

        add_apartment_to_scene(self.scene, LAYOUT)


# ---------------------------------------------------------------------------
# Gym registration
# ---------------------------------------------------------------------------
# BOTH entry points are required. doc 02 §3's policy loader calls
# `load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")`; registering only
# the env cfg raises at the first OnPolicyRunner construction in T1.2.
_RSL_RL_ENTRY = (
    "isaac_lab_env.open_duck_mini_v2.agents.rsl_rl_ppo_cfg:OpenDuckRobustPPORunnerCfg"
)

for _task_id, _cfg_cls in (
    (TASK_ID, "DuckEmbodyEnvCfg"),
    (TASK_ID_APARTMENT, "DuckEmbodyApartmentEnvCfg"),
):
    if _task_id not in gym.registry:
        gym.register(
            id=_task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{_cfg_cls}",
                "rsl_rl_cfg_entry_point": _RSL_RL_ENTRY,
            },
        )
