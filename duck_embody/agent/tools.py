"""Tool schemas + dispatch: get_observation, look_around, turn_to_heading, move,
send_velocity, memory tools, declare_done.

This module is **the wire** (PLAN T3.2): the macros live in
:mod:`duck_embody.sim.policy_wrapper`, the map lives in
:mod:`duck_embody.agent.memory`, and nothing here re-implements either. What it
*does* own is the boundary the rest of the harness cannot see:

**1. No exception may escape :func:`dispatch` for a model-supplied argument.**
Doc 05 §8 classifies an escaping harness exception as an *infra* fault and
reruns the trial WHOLE — so a malformed argument would buy the model a free
retry, which is precisely the selection bias §8 exists to prevent, while §8's
own first row says a bad argument comes back as ``invalid_args`` and the turn
still counts. Everything a model can put in a tool call is therefore validated
before it reaches the sim: :func:`memory.number_arg` rejects ``None``, ``bool``,
``nan`` and ``inf`` (``clamp_command("0.2", 0, 0)`` raises ``TypeError``, and
``clamp_command(nan, 0, 0)`` silently returns ``nan``, which would then poison
the command buffer *and* the position estimate). Render and physics faults are
deliberately NOT caught: doc 05 §4.1 routes those to the infra path, and
swallowing them would launder a broken GPU into a model failure — the same line
drawn from the other side.

**2. No scoring-only field may reach the model — but none may be DESTROYED
either.** ``ExecResult`` carries FOUR ground-truth fields (``pose_trace``,
``sampled_xy``, ``true_pose``, ``true_displacement_m``). Every *payload* below is
assembled key by key from the safe set; a ``dataclasses.asdict(result)`` anywhere
in this file would leak the robot's true trajectory into the transcript and
silently invalidate the whole benchmark (doc 06 §4). ``tests/test_tools.py``
asserts it numerically, because nothing crashes when a benchmark leaks its answer
key. The mirror-image failure is just as silent: ``dispatch`` is the only code
that ever sees an ``ExecResult``, so dropping it would make doc 06 §4's
``turns[].execution.pose_trace`` — the 5 Hz true trajectory §5.3 pins SPL to —
unproducible for T3.4, and T4.1's scorer is specified to RAISE on a missing
``pose_trace`` rather than fall back to per-turn chords. Hence
:attr:`ToolOutcome.execution`: a scoring-side channel that :meth:`to_block` never
touches. Two directions, one boundary.

**3. The dead-reckoning feed.** ``move()``'s merged ``policy_seconds`` includes
a trailing 0.2 s *zero-command* settle chunk that its ``dead_reckoned_distance_m``
excludes, so integrating 0.2 m/s over the merged figure would fabricate 0.04 m
of forward motion per call — up to 1.6 m per stage of harness-manufactured
"drift", inverting the failure PLAN T3.2 (b) was written to prevent. See
:func:`_move` for the correction (PLAN T3.2 (b) is amended in the same commit).

**4. Clamping is echoed, never silent** (doc 05 §4.2 / §6). ``move`` clamps
distance silently inside the wrapper and adds no note, and the
``duration_s`` ∈ ``[0.2, 3.0]`` clamp exists in no code at all — this module
owns both notes, in the same ``notes`` list the hull clamps use, so the model
reads one uniform echo.

**5. A fall has already teleported the robot.** Isaac Lab auto-resets a
terminated env *inside* ``env.step()`` and returns the post-reset observation, so
every live sensor read after a fall reports the SPAWN pose —
``policy_wrapper.execute`` guards its own ``true_pose`` against exactly this and
this module inherits none of that care for free. ``compass_deg()`` is therefore
latched at the fall (:func:`observed_compass_deg`) rather than re-read, and the motion
tools refuse to run at all once ``playback.fell`` is set (:func:`dispatch`), so
no further physics can step against a respawned robot. Both recorded in doc 05
§4.1/§4.2/§8.

The 12 schemas are doc 05 §4's canonical block **verbatim**. Doc 05 §6 records
why that is load-bearing rather than tidy: the prompt carries a paraphrase, so
§4's numeric bounds — the ``send_velocity`` hull, ``turn_to_heading``'s
``[0,360)`` — reach the model through *no other* model-facing text.
``tests/test_tools.py`` extracts the block from the HTML doc and compares it, so
the two cannot drift.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from duck_embody.agent.memory import (
    STAGE_FIND_KITCHEN,
    Counters,
    Memory,
    PositionIntegrator,
    correct_position,
    invalid_args,
    number_arg,
)
from duck_embody.agent.prompts import STAGE2_OBJECTIVE_TOOL_RESULT
from duck_embody.agent.providers.base import ImageBlock, ToolCall, ToolResultBlock
from duck_embody.env.camera import encode_b64
from duck_embody.sim.policy_wrapper import (
    MOVE_MAX_DISTANCE_M,
    MOVE_SPEED_MPS,
    shortest_angle_diff_deg,
    wrap_deg,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Absolute compass bearings ``look_around`` captures (doc 04 §5.3, doc 05 §4.1;
#: mirrored in ``configs/benchmark.yaml`` as ``camera.look_around_bearings_deg``,
#: which the tests below assert still agrees).
LOOK_AROUND_BEARINGS_DEG = (0.0, 90.0, 180.0, 270.0)

#: ``send_velocity``'s duration clamp. DESIGN doc 05 §4 (the schema description
#: the model reads) and doc 02 §6.3 (``d = clip(duration_s, 0.2, 3.0)``); until
#: T3.2 this bound existed in no code anywhere. The floor is not cosmetic: at
#: 50 Hz a 0.02 s command is a single control step, which cannot express a gait
#: cycle and mostly measures where in the stride the robot happened to be. The
#: ceiling keeps the raw escape hatch from crossing a room blind — that is what
#: ``move``'s bump auto-stop is for, and ``send_velocity`` deliberately has none.
#: Mirrored into ``configs/benchmark.yaml`` as
#: ``locomotion.send_velocity_duration_s``, with an agreement test, following the
#: precedent set by the caps.
DURATION_RANGE_S = (0.2, 3.0)

#: The stage signal. ``declare_done`` is not dispatched like the other tools:
#: doc 05 §3.1's loop branches on this name *before* :func:`dispatch` is reached,
#: because the result is the stage OUTCOME, not something a tool can compute.
#: Exported so T3.4's loop branches on a constant rather than a spelled string.
DECLARE_DONE = "declare_done"

#: Model-facing label for the drifting estimate. Verbatim from doc 04 §6's
#: frozen payload example, where it rides in every ``position_estimate``.
POSITION_ESTIMATE_NOTE = (
    "dead-reckoned estimate, may drift — correct via correct_position"
)


# ---------------------------------------------------------------------------
# Canonical tool schemas — doc 05 §4, VERBATIM (doc 05 §6, PLAN T3.2 (a))
# ---------------------------------------------------------------------------
#
# Do not "improve" these. There are deliberately no `default`s, no `enum`s, no
# `minimum`/`maximum` and no `additionalProperties`: doc 05 §4's preamble pins
# the division of labour — "the schema is documentation for the model, the
# harness is the enforcer" — and every bound stated in English below is enforced
# by the handlers further down. Editing a description here changes what every
# model in the batch reads, which is a doc 06 §2 freeze violation, so the test
# compares these strings against the design doc itself rather than against a
# copy pasted into the test file (a copy drifts in lockstep and still passes).

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "get_observation",
        "description": (
            "One egocentric camera frame (512x512 RGB, 90 deg HFOV, head-mounted "
            "at ~0.36 m) plus compass heading, dead-reckoned position estimate, "
            "and status flags."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "look_around",
        "description": (
            "Four frames at compass headings 0/90/180/270 deg via virtual "
            "camera-mount rotation while the sim is paused. The robot does NOT "
            "move."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "turn_to_heading",
        "description": (
            "Rotate in place to an absolute compass heading. Closed-loop on the "
            "compass, tolerance +/-5 deg; yaw rate clamped to the trained hull; "
            "times out rather than spinning forever."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "heading_deg": {
                    "type": "number",
                    "description": "Target absolute heading, degrees [0,360)",
                }
            },
            "required": ["heading_deg"],
        },
    },
    {
        "name": "move",
        "description": (
            "Walk forward at vx=0.2 m/s, closed-loop on dead-reckoned distance. "
            "Max 1.5 m per call. Auto-stops on collision and reports the bump."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "distance_m": {
                    "type": "number",
                    "description": "Forward distance in meters, (0, 1.5]",
                }
            },
            "required": ["distance_m"],
        },
    },
    {
        "name": "send_velocity",
        "description": (
            "Raw escape hatch: command (vx, vy, wz) for duration_s. Clamped to "
            "the policy's training hull: vx in (-0.148, 0.222) m/s, vy in "
            "(-0.111, 0.111) m/s, wz in (-0.5, 0.5) rad/s. duration_s in "
            "[0.2, 3.0]. Note vy is near-useless for navigation (~9 s/m) - "
            "prefer turn-then-drive."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vx": {"type": "number"},
                "vy": {"type": "number"},
                "wz": {"type": "number"},
                "duration_s": {"type": "number"},
            },
            "required": ["vx", "vy", "wz", "duration_s"],
        },
    },
    {
        "name": "update_room",
        "description": (
            "Create or update a room node in YOUR map with a visual description "
            "you observed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name", "description"],
        },
    },
    {
        "name": "add_landmark",
        "description": "Attach a landmark description to a room in YOUR map.",
        "input_schema": {
            "type": "object",
            "properties": {
                "room": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["room", "description"],
        },
    },
    {
        "name": "mark_exit",
        "description": (
            "Record an exit/doorway from a room at an absolute compass "
            "direction. status is 'unexplored' or 'leads_to:<room>'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "room": {"type": "string"},
                "direction_deg": {"type": "number"},
                "status": {
                    "type": "string",
                    "description": "'unexplored' or 'leads_to:<room>'",
                },
            },
            "required": ["room", "direction_deg", "status"],
        },
    },
    {
        "name": "set_current_room",
        "description": "Assert which of YOUR rooms you are currently in.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "correct_position",
        "description": (
            "Reset the dead-reckoning integrator to (x, y) because you "
            "re-recognized a landmark (cognitive loop closure). Give the reason; "
            "every correction is logged."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["x", "y", "reason"],
        },
    },
    {
        "name": "update_plan",
        "description": (
            "Replace your standing plan. It is carried forward and re-injected "
            "every turn until you change it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": DECLARE_DONE,
        "description": (
            "End the current stage and trigger scoring. Call ONLY when you "
            "believe you are at the objective."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

#: Declaration order is doc 05 §4's array order and is what both adapters ship.
TOOL_NAMES: tuple[str, ...] = tuple(schema["name"] for schema in TOOL_SCHEMAS)

_SCHEMA_BY_NAME: dict[str, dict] = {s["name"]: s for s in TOOL_SCHEMAS}


# ---------------------------------------------------------------------------
# Dispatch state and results
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Everything :func:`dispatch` may touch. Owned by T3.4's loop.

    The camera is **injected**, not constructed: ``SimSession`` owns no camera
    (every existing call site builds ``HeadCamera(session.env)`` itself), and
    injection is what keeps this module unit-testable without a kit process —
    which PLAN T3.2's "sim mocked" requirement demands anyway.

    ``turn`` is the **stage-local** model-turn index. It is carried here rather
    than derived because :func:`memory.correct_position` stamps it into the
    correction log and doc 05 §3.3 resets it at the stage boundary; a global
    turn index would make doc 06 §5.8's per-stage drift series unreadable.
    """

    playback: object
    camera: object
    memory: Memory
    integrator: PositionIntegrator
    counters: Counters
    #: Stage-local model-turn index (doc 05 §3.3).
    turn: int = 0
    #: doc 06 §5.6's bump counter: `move` auto-stops plus `send_velocity`
    #: collision reports, one per command. Lives here rather than on `Counters`
    #: because `Counters` is rendered into the model's budget line and bumps are
    #: a scoring quantity, not a budget.
    #:
    #: **TRIAL-scoped — never reset at the stage boundary.** doc 06 §5.6 counts
    #: collisions "over the trial", while doc 05 §3.3 resets only turns and
    #: policy-seconds. Zeroing this alongside them would silently drop every
    #: stage-1 collision from a published headline metric, and nothing would
    #: fail. :meth:`reset_for_stage` is the only sanctioned reset and
    #: deliberately leaves this field alone.
    bumps: int = 0
    #: doc 05 §4.1: `get_observation`'s status fields "describe the *last*
    #: motion command". Before any motion has run they are the zero state below
    #: — a fresh stage has no last command, and either inventing one or omitting
    #: the keys would change the payload's shape between turn 1 and turn 2.
    #: STAGE-scoped: cleared by :meth:`reset_for_stage`.
    last_bumped: bool = False
    #: Regions in contact at the last bump; carried so the next
    #: `get_observation` can report it alongside `bumped`.
    last_contact_groups: list = field(default_factory=list)
    last_distance_moved_m: float = 0.0
    #: Compass heading latched at the moment of the fall, or ``None`` while the
    #: robot is upright. TRIAL-scoped and write-once: a fall ends the trial, so
    #: there is no boundary at which un-latching it would be correct. See
    #: :func:`observed_compass_deg` for why the live sensor cannot be trusted afterwards.
    compass_at_fall: float | None = None

    def reset_for_stage(self) -> None:
        """Zero exactly the STAGE-scoped fields (doc 05 §3.3, §4.1).

        Exists so T3.4's loop has one call to make at the ``find_kitchen`` →
        ``return_home`` boundary instead of a field list to get right. Rebuilding
        the whole :class:`ToolContext` instead — the obvious reading of doc 05
        §4.1's "which T3.4's loop owns and resets with the stage" — would also
        zero :attr:`bumps`, halving a doc 06 §5.6 headline metric with no test
        and no traceback. ``compass_at_fall`` is likewise left alone: a fall ends
        the trial, so stage 2 never starts after one.
        """
        self.turn = 0
        self.last_bumped = False
        self.last_distance_moved_m = 0.0
        self.counters.turns = 0
        self.counters.policy_seconds = 0.0


@dataclass
class ToolOutcome:
    """One tool's result, before it is wrapped for a provider.

    ``payload`` is the JSON status object; ``images`` are camera frames. Kept
    separate from :class:`ToolResultBlock` so the dispatcher can be tested
    without the provider layer, and so serialisation happens in exactly one
    place (:meth:`to_block`).

    ``execution`` is the other half of the boundary: the SCORING-side record of
    what the sim actually did, which the model must never see. It is a plain
    attribute rather than a payload key precisely so that :meth:`to_block` — the
    only path to a provider — cannot carry it by accident.
    """

    payload: dict
    images: list[ImageBlock] = field(default_factory=list)
    is_error: bool = False
    #: doc 06 §4's ``turns[].execution`` block + its sibling ``true_pose``, for
    #: the tools that stepped physics; ``None`` for everything else. GROUND
    #: TRUTH: ``pose_trace`` is the 5 Hz true trajectory doc 06 §5.3 pins the SPL
    #: path integral ``p`` to, and §5.3 warns in as many words that chord-summing
    #: the once-per-turn ``true_pose`` entries instead "would under-measure any
    #: within-turn curved motion ... shrinking p and inflating SPL". T3.4 logs
    #: this; T4.1's scorer raises if it is missing. Never serialised — see
    #: :meth:`to_block`, and the test that asserts the two cannot meet.
    execution: dict | None = None

    def to_block(self, tool_use_id: str, tool_name: str) -> ToolResultBlock:
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            # One JSON-escaped single-line string, per doc 04 §6's formatting
            # note. Key order is insertion order, which is the doc's field
            # order — stable across turns, so the model is not re-parsing a
            # payload whose shape moves under it.
            #
            # `allow_nan=False` is doc 05 §8, not tidiness. Python's default
            # emits the bare tokens `NaN`/`Infinity`, which are not JSON, are
            # rejected by neither provider, and would be shown to the model for
            # the rest of the episode. `memory.number_arg` already rejects
            # non-finite values coming FROM the model; this closes the other
            # direction, where a physics NaN reaches `compass_deg()` or the
            # integrator. §8's table names "physics NaN" as the canonical
            # sim-side fault whose policy is an infra rerun of the whole trial,
            # so it has to raise here rather than serialise.
            text=json.dumps(self.payload, allow_nan=False),
            images=list(self.images),
            is_error=self.is_error,
        )


# ---------------------------------------------------------------------------
# Structured errors and control shapes (doc 05 §8, §3.1)
# ---------------------------------------------------------------------------


def unknown_tool(name: object) -> dict:
    """Doc 05 §8's other error kind. Nothing else in the repo authors it."""
    return {
        "error": "unknown_tool",
        "detail": f"no tool named {name!r}",
        "hint": "available tools: " + ", ".join(TOOL_NAMES),
    }


def not_executed(tool_name: str) -> dict:
    """For calls listed *after* ``declare_done`` in the same turn (doc 05 §3.1).

    §3.1 requires "a structured ``not_executed`` result" but §8's table lists
    only ``unknown_tool``/``invalid_args``; the third kind is recorded into §8
    in the same commit (AGENTS.md rule 5). It must still be a *result*, because
    every ``tool_use`` block in the echoed assistant turn has to be answered
    (§7.2) and an unanswered one is an API error — i.e. an infra rerun of a
    trial the model actually finished.
    """
    return {
        "error": "not_executed",
        "detail": f"{tool_name} was listed after declare_done and was not executed",
        "hint": (
            "declare_done ends the stage immediately — list memory writes "
            "BEFORE it in the same turn"
        ),
    }


def trial_over(tool_name: str) -> dict:
    """A motion tool called after the robot has fallen (doc 05 §4.2, §8).

    doc 05 §4.2 says "falls end the trial" for all three motion tools but names
    no mechanism, and until T3.2 there was none: ``dispatch`` had no fall guard
    and ``loop.py`` does not exist yet, so a model that reasonably tried to
    recover would have kept commanding a robot Isaac had already teleported back
    to spawn — walking it out of the spawn point while ``pose_trace`` (doc 06
    §5.3) and the drift metric (§5.8) accumulated across the teleport, with
    nothing raising and nothing logged as anomalous. The guard lives here rather
    than in the loop so it holds regardless of what T3.4 does.

    Perception and memory tools are deliberately still answered: doc 05 §4.1
    pins that ``fell`` is read live so "the final observation must report it
    however it is reached", and they step no physics.
    """
    return {
        "error": "stage_ended",
        "detail": f"{tool_name} was not executed: the robot fell and the trial is over",
        "hint": (
            "a fall ends the trial — there is no recovery. The remaining tools "
            "still answer, but no further motion will be executed."
        ),
    }


#: The outcome-neutral stage-end text. Used for stage 2 (the trial really is
#: over) AND for a stage-1 ``declare_done`` that failed the distance test — see
#: :func:`stage_end_result`. Byte-identical in both cases on purpose: the model
#: learns that the run ended, never that it was wrong.
TRIAL_OVER_DETAIL = "The trial is over; no further tool calls will be executed."


def stage_end_result(stage: str, *, continue_to_return_home: bool = True) -> dict:
    """``declare_done``'s own tool_result: the stage outcome (doc 05 §3.3).

    On ``find_kitchen`` it carries the ``return_home`` objective **verbatim**
    from the frozen prompt module. That string is deliberately absent from
    ``SYSTEM_PROMPT`` (``tests/test_memory.py`` asserts it): a model that knew
    about the return leg in advance would map differently from the design the
    benchmark describes, so it exists only here, delivered mid-episode.

    ``continue_to_return_home`` is T3.4's resolution of doc 05 §12 / doc 06 §12
    (see :data:`duck_embody.tasks.find_kitchen.STAGE2_REQUIRES_STAGE1_SUCCESS`
    for the full reasoning and its recorded cost): **stage 2 runs iff stage 1
    succeeded**, so a ``declare_done`` in the wrong place must NOT hand the model
    the return-leg objective. Doc 05 §3.1's normative pseudocode always
    anticipated an episode-state-dependent result here — it calls
    ``stage_end_result(stage, state)`` — and T3.2 narrowed the signature to one
    argument; this keyword is that second argument, restored in the only shape
    the shipped API needs.

    It defaults to ``True`` so the doc-05-faithful success path stays a
    one-argument call and :func:`_declare_done`'s fallback keeps working, but
    **the loop always passes it explicitly**. A default that silently offered
    the return leg is precisely the failure this parameter exists to prevent, so
    it must never be reached by the code that actually decides.
    """
    if stage == STAGE_FIND_KITCHEN and continue_to_return_home:
        return {
            "ok": True,
            "stage_ended": STAGE_FIND_KITCHEN,
            "detail": STAGE2_OBJECTIVE_TOOL_RESULT,
        }
    return {
        "ok": True,
        "stage_ended": stage,
        "detail": TRIAL_OVER_DETAIL,
    }


def _arg_list(properties: dict) -> str:
    if not properties:
        return "no arguments"
    return ", ".join(f"{key} ({spec['type']})" for key, spec in properties.items())


def _arg_error(name: str, args: object) -> dict | None:
    """Arity/shape check — the gap ``memory.py`` structurally cannot close.

    The memory tools are ordinary Python methods, so a call missing a required
    key or carrying a surplus one dies on the ``**`` splat with a ``TypeError``
    *before* their own type validation ever runs — and doc 05 §8 would then
    rerun the whole trial for a mistake the model made.

    Surplus keys are **rejected rather than dropped**. Silently ignoring
    ``turn_to_heading(heading_deg=90, distance_m=1.0)`` is the harness guessing
    that the second argument was surplus rather than a conflation of two tools
    (§8: "the harness never guesses intent"), and the model would never learn
    that half of what it asked for was discarded.
    """
    if not isinstance(args, dict):
        return invalid_args(
            f"{name} arguments must be a JSON object, got {type(args).__name__}",
            "pass the arguments as a JSON object of named parameters",
        )
    schema = _SCHEMA_BY_NAME[name]["input_schema"]
    accepted = schema["properties"]
    missing = [key for key in schema["required"] if key not in args]
    if missing:
        return invalid_args(
            f"{name} is missing required argument(s): {', '.join(missing)}",
            f"{name} takes: {_arg_list(accepted)}",
        )
    surplus = [key for key in args if key not in accepted]
    if surplus:
        return invalid_args(
            f"{name} got unexpected argument(s): {', '.join(sorted(surplus))}",
            f"{name} takes exactly: {_arg_list(accepted)}",
        )
    return None


# ---------------------------------------------------------------------------
# Shared payload assembly
# ---------------------------------------------------------------------------
#
# doc 04 §6 freezes these field names and doc 04 §6.1 freezes the *logical*
# payload for all three models. Every motion tool returns this block plus its
# own keys, so the model reads the same three facts — where it thinks it is,
# which way it faces, what the last command did — off every result without
# special-casing per tool.


def observed_compass_deg(context: ToolContext) -> float:
    """The heading to report: the live compass, EXCEPT after a fall.

    PUBLIC because T3.4's loop needs the identical rule in two places this
    module cannot reach: the memory block rendered into every request (doc 05
    §3.1 passes ``sim.compass_deg()`` there) and the block rendered for the
    post-episode QA exchange. A loop that called ``playback.compass_deg()``
    directly would show the model the SPAWN heading in the QA prompt of every
    trial that ended in a fall — the exact failure the latch below exists to
    prevent, reintroduced one layer up. One implementation, one rule.

    ``policy_wrapper.execute`` documents the trap it guards its own ``true_pose``
    against: Isaac Lab auto-resets a terminated env *inside* ``env.step()`` and
    teleports the robot back to spawn, so every sensor read afterwards describes
    the spawn point. ``compass_deg()`` is one of those reads. Left live, a duck
    that toppled at 120° would report the seed's spawn heading in the very
    payload that carries ``fell: true`` — an unexplained discontinuity of tens of
    degrees, written *permanently* into the breadcrumb trail that is re-injected
    every turn (doc 05 §5.2) and logged as doc 06 §4's ``obs.compass_deg``.

    The latched value is the heading read BEFORE the command that fell — the
    closest honest number available, and deliberately not ``result.true_pose[2]``,
    which is scoring-only (doc 06 §4) and may not cross into a payload.
    """
    if context.compass_at_fall is not None:
        return context.compass_at_fall
    return context.playback.compass_deg()


def _state_payload(context: ToolContext) -> dict:
    """The frozen ``compass_deg`` / ``position_estimate`` / ``status`` block.

    Assembled key by key. The temptation this exists to remove is serialising an
    ``ExecResult``, four of whose fields are ground truth (doc 06 §4).
    """
    x, y = context.integrator.xy
    return {
        "compass_deg": round(observed_compass_deg(context), 1),
        "position_estimate": {
            "x": round(x, 2),
            "y": round(y, 2),
            "note": POSITION_ESTIMATE_NOTE,
        },
        "status": {
            "bumped": context.last_bumped,
            # WHERE the last collision was felt: head / torso / left_leg /
            # right_leg. Without it `bumped` is a bare boolean and the model can
            # only guess which way is blocked — T3.5 measured 6 of 13 moves
            # stopping under 0.11 m while it pinballed. Proprioception, not
            # ground truth: it says what the ROBOT felt, never what was hit or
            # where that thing is, so it stays on the sensor side of doc 05 §1.
            "contact": list(context.last_contact_groups),
            # Read LIVE, not carried: `fell` is sticky and ends the trial, so
            # the final observation must report it however it is reached
            # (doc 04 §6.2).
            "fell": bool(context.playback.fell),
            "distance_moved_m": round(context.last_distance_moved_m, 3),
        },
    }


def _record_motion(
    context: ToolContext,
    result,
    distance_moved_m: float,
    *,
    counts_bump: bool,
    heading_before: float,
) -> dict:
    """Budget, bump counter, carry-over status, the breadcrumb — and doc 06 §4's
    scoring record, which is this function's return value.

    ``policy_seconds`` here is the MERGED figure — a macro's trailing settle
    chunk really did step physics and really does spend the 240 s cap, even
    though it moves the robot nowhere. It is the *integrator* feed that must
    exclude it (see :func:`_move`); the two are different numbers by design and
    conflating them in either direction fails silently.

    ``counts_bump`` is passed explicitly by all three motion tools rather than
    defaulted, because the correct answer is not the same for all three and a
    default would decide it silently. doc 06 §5.6 enumerates exactly two sources
    for the published ``bumps`` metric — ``move`` auto-stops and ``send_velocity``
    collision reports — and ``turn_to_heading`` is not one of them. Counting it
    would be behaviour-dependent inflation, not a stricter measurement:
    ``PolicyPlayback._bump_run`` is instance state that survives across calls, so
    after a bump-stopped ``move`` the debounce counter is already at its
    threshold and the first control step of the recovery turn re-flags
    ``bumped`` — bump-then-turn-away, the canonical recovery pattern, would score
    one real collision as three. The flag is still reported to the model in
    ``status.bumped`` (doc 04 §6.2, whose "auto-stopped" wording was already too
    narrow for ``send_velocity`` and is corrected in this commit); only the
    scored counter is restricted.

    ``heading_before`` is the compass read before the command ran, latched here
    if the command was the one that fell — see :func:`observed_compass_deg`.
    """
    context.counters.policy_seconds += result.policy_seconds
    if result.bumped and counts_bump:
        context.bumps += 1
    context.last_bumped = bool(result.bumped)
    context.last_contact_groups = list(result.contact_groups)
    context.last_distance_moved_m = distance_moved_m
    if result.fell and context.compass_at_fall is None:
        # BEFORE the breadcrumb below, which would otherwise be the first thing
        # written with the post-teleport spawn heading.
        context.compass_at_fall = heading_before
    x, y = context.integrator.xy
    # The harness's ONE autonomous write into memory (doc 05 §5.1): the
    # integrator's estimate plus the compass — never the true pose.
    context.memory.add_breadcrumb(x, y, observed_compass_deg(context))
    return {
        # doc 06 §4's `turns[].execution` block. `pose_trace` is the field the
        # whole channel exists for: §5.3 pins the SPL path integral to these
        # 5 Hz sub-turn samples and warns that chord-summing the once-per-turn
        # `true_pose` entries instead shrinks p and inflates SPL.
        "policy_seconds_used": result.policy_seconds,
        "pose_trace": [list(point) for point in result.pose_trace],
        "sampled_xy": [list(point) for point in result.sampled_xy],
        # doc 06 §4 logs this as a sibling of `execution`, once per turn.
        "true_pose": list(result.true_pose),
        "true_displacement_m": result.true_displacement_m,
        # The model-facing facts, repeated here so the trial log's
        # `execution.result` line can be written without re-parsing the payload.
        "distance_moved_m": distance_moved_m,
        "bumped": bool(result.bumped),
        "contact_groups": list(result.contact_groups),
        "fell": bool(result.fell),
        # WHY it ended — height, tilt, which term fired, and the command in
        # flight. A fall ends the whole TRIAL (doc 01 §8), so it is the most
        # consequential event in a run; T3.5 produced one that could not be
        # audited afterwards because the log recorded only the boolean. None
        # on every non-terminating call.
        # Falls back to the playback's own copy. The ExecResult carries it on
        # every path I could exercise, but a T3.5 trial recorded `fell: true`
        # with None and I could not reproduce that specific case — so rather
        # than trust a negative, read the instance state too. Safe: a fall ends
        # the trial, so there is at most one per run, and `reset()` clears it.
        "fall_diagnostics": (
            result.fall_diagnostics
            or (context.playback.fall_diagnostics if result.fell else None)
        ),
        "stop_reason": result.stop_reason,
        "counted_as_bump": bool(result.bumped and counts_bump),
    }


def _clamp_duration(duration_s: float) -> tuple[float, list[str]]:
    """Doc 05 §4.2's ``[0.2, 3.0]`` clamp, echoed like every other clamp.

    A sub-0.2 s request clamps UP, spending marginally more motion than asked
    for. That is what "duration clamped to [0.2, 3.0] s" says, and the echo
    makes it visible; rejecting instead would burn the model's scarcest budget
    (a turn) over a command the hull can perfectly well run.
    """
    low, high = DURATION_RANGE_S
    clamped = min(max(duration_s, low), high)
    if clamped == duration_s:
        return clamped, []
    return clamped, [
        f"duration_s {duration_s:+.3f} clamped to {clamped:+.3f} "
        f"(range [{low}, {high}] s)"
    ]


def _numbers(args: dict, fields: tuple[str, ...], tool: str):
    """Type-check several number arguments, returning the first error.

    All-or-nothing, mirroring ``correct_position``'s both-coordinates-before-
    either rule: a ``send_velocity`` whose ``vy`` is malformed must not have
    already stepped physics with its ``vx``.
    """
    values: list[float] = []
    for name in fields:
        value, error = number_arg(args[name], name, tool)
        if error is not None:
            return None, error
        values.append(value)
    return values, None


# ---------------------------------------------------------------------------
# Perception tools (doc 05 §4.1, doc 04 §5.3/§6)
# ---------------------------------------------------------------------------


def _get_observation(context: ToolContext, args: dict) -> ToolOutcome:
    """One frame + the state block. No physics steps, no policy-seconds.

    A render failure is deliberately allowed to propagate: doc 05 §4.1 routes it
    to the infra path, and catching it here would report a broken GPU to the
    model as if it were the model's own mistake.
    """
    frame = context.camera.capture_b64()
    return ToolOutcome(payload=_state_payload(context), images=[ImageBlock(frame)])


def _look_around(context: ToolContext, args: dict) -> ToolOutcome:
    """Four bearings via the virtual gimbal — the robot does not move.

    Charges zero policy-seconds and appends no breadcrumb, because nothing
    stepped: ``HeadCamera.look_around`` re-aims the camera between renders while
    the sim stays paused (doc 04 §5.3). The frozen prompt promises the model
    exactly that ("no fall risk and no motion budget spent"), so charging for it
    would make the frozen prose false.
    """
    images = [
        ImageBlock(
            data_b64=encode_b64(rgb),
            # Doc 04 §6.2 pins the caption text, in absolute bearings.
            label=f"view at compass {bearing:g}°",
        )
        # The third element of each tuple is the forward vector, derived from
        # the robot's TRUE yaw. Discarded deliberately: it is ground truth
        # (doc 06 §4), and it is exactly the kind of field that leaks by being
        # convenient.
        for bearing, rgb, _forward in context.camera.look_around(
            LOOK_AROUND_BEARINGS_DEG
        )
    ]
    return ToolOutcome(payload=_state_payload(context), images=images)


# ---------------------------------------------------------------------------
# Motion tools (doc 05 §4.2, doc 02 §6)
# ---------------------------------------------------------------------------


def _turn_to_heading(context: ToolContext, args: dict) -> ToolOutcome:
    """Closed-loop yaw to an absolute compass heading.

    Out-of-domain headings are **wrapped and echoed**, not rejected. §4's schema
    documents ``[0,360)`` but §4.2 states no failure mode for leaving it, and an
    angle modulo 360 has exactly one meaning — wrapping is arithmetic, not
    intent-guessing, which is where doc 05 §8 draws its line. The echo follows
    ``mark_exit``'s precedent (§5.1: "the ack echoes the raw value, so the snap
    is visible rather than silent"): a model that asked for 725 deg and was
    silently sent to 5 deg would have no way to notice its own arithmetic
    slipped. Recorded in doc 05 §4.2.
    """
    heading, error = number_arg(args["heading_deg"], "heading_deg", "turn_to_heading")
    if error is not None:
        return ToolOutcome(payload=error, is_error=True)

    target = wrap_deg(heading)
    notes: list[str] = []
    if target != heading:
        notes.append(
            f"heading_deg {heading:g} wrapped to {target:g} "
            "(compass domain [0, 360))"
        )

    heading_before = context.playback.compass_deg()
    result = context.playback.turn_to_heading(target)
    # No integrator feed: the macro commands vx = vy = 0, so a rotation in place
    # displaces the estimate by exactly zero. Real slip during the turn is drift
    # the integrator is *supposed* not to see (doc 05 §5.1).
    #
    # `counts_bump=False`: doc 06 §5.6 counts `move` and `send_velocity` only.
    # See `_record_motion` for why counting rotations inflates the metric.
    execution = _record_motion(
        context,
        result,
        distance_moved_m=0.0,
        counts_bump=False,
        heading_before=heading_before,
    )

    compass = observed_compass_deg(context)
    payload = {
        # The RAW argument, not `target`. `mark_exit`'s 15° snap set the
        # precedent doc 05 §4.2 cites — "the ack echoes the raw value, so the
        # snap is visible rather than silent" — and a key named `requested_` that
        # answers 5.0 to a model that requested 725.0 is the opposite of that.
        # The wrap note carries the target; `compass_deg` carries where the robot
        # ended up. Both wrapped and raw name the same angle, so
        # `heading_error_deg` below is unaffected.
        "requested_heading_deg": round(heading, 1),
        # doc 05 §4.2's "achieved error", signed as the rotation the robot STILL
        # needs to reach the target (+ = further counter-clockwise). Note this
        # is the opposite sign to `policy_wrapper`'s internal `residual`, which
        # is only ever used inside an `abs()`.
        "heading_error_deg": round(shortest_angle_diff_deg(target, compass), 1),
        # doc 05 §4.2's named field. `policy_wrapper` spells the same state
        # "timeout" in `stop_reason`, which is internal vocabulary — the frozen
        # prompt promises the model `timed_out`.
        "timed_out": result.stop_reason == "timeout",
        "policy_seconds": round(result.policy_seconds, 2),
        **_state_payload(context),
    }
    if notes:
        payload["notes"] = notes
    return ToolOutcome(payload=payload, execution=execution)


def _move(context: ToolContext, args: dict) -> ToolOutcome:
    """Walk forward with heading hold; auto-stops on collision.

    ``distance_m <= 0`` is an ``invalid_args``, which §4 does not settle either
    way (its domain is ``(0, 1.5]``, open at zero; §4.2's only stated failure
    mode is the ``> 1.5`` clamp). The wrapper would clamp it to 0.0 and still
    run one chunk — a silent no-op that burns a turn and tells the model
    nothing. Nor is there an honest clamp available: raising 0 to some positive
    distance invents a command the model did not give, and clamping -1 to +1
    would drive the robot the opposite way from what a model that typed -1
    meant. Backing out of a corner is ``send_velocity``'s job and the hint says
    so. Recorded in doc 05 §4.2.
    """
    requested, error = number_arg(args["distance_m"], "distance_m", "move")
    if error is not None:
        return ToolOutcome(payload=error, is_error=True)
    if requested <= 0.0:
        return ToolOutcome(
            payload=invalid_args(
                f"distance_m must be greater than 0 in move, got {requested:g}",
                "move only walks forward; the schema domain is (0, 1.5]. To "
                "back out of a corner use send_velocity with a negative vx.",
            ),
            is_error=True,
        )

    distance = min(requested, MOVE_MAX_DISTANCE_M)
    notes: list[str] = []
    if distance != requested:
        # `policy_wrapper.move` performs the same clamp but records NO note
        # (`clamp_notes` only ever carries velocity-hull notes), while doc 05
        # §4.2 requires "Argument > 1.5 clamped with a note in the result".
        notes.append(
            f"distance_m {requested:+.3f} clamped to {distance:+.3f} "
            f"(max {MOVE_MAX_DISTANCE_M:g} m per move call)"
        )

    # The heading the macro holds for the whole drive, read before it starts —
    # which is exactly the value `move()` latches internally.
    held_heading = context.playback.compass_deg()
    # Both flags passed EXPLICITLY even though they are the wrapper's defaults.
    # `stop_on_bump=True` is the entire difference between this tool and
    # `send_velocity` (doc 05 §4.2, asserted twice there and again in PLAN
    # T3.2's Context line), and `hold_heading=True` is T1.3's measured
    # correction for a policy that yaws ~1.8 deg/s when commanded straight. A
    # default flipped in `policy_wrapper` would otherwise silently change what
    # the tool surface means, and every test that pins the distinction would
    # still pass.
    result = context.playback.move(distance, hold_heading=True, stop_on_bump=True)
    # NOTE (recorded in doc 05 §4.2): `travelled` is quantised UP to a multiple
    # of MOVE_SPEED_MPS * MACRO_CHUNK_S = 0.04 m. `policy_wrapper.move` servos in
    # whole 0.2 s chunks and breaks on `travelled >= distance / k`, so a move is
    # never SHORT of what was asked and may exceed it by up to 0.04 m —
    # `move(1.5)` covers 1.52 m, `move(0.05)` covers 0.08 m. Deliberately NOT
    # echoed as a clamp note: it fires on essentially every call, which would
    # turn `notes` from a signal into decoration (the one thing that key must not
    # become), and `status.distance_moved_m` below already reports the real
    # figure on every single move. Not "fixed" here either: shortening the final
    # chunk is a change to a doc 02 §6.2 macro validated by T2.4's physics pass.
    travelled = result.dead_reckoned_distance_m

    # PLAN T3.2 (b), AMENDED IN THIS COMMIT. Its instruction — feed
    # `integrate()` the `ExecResult.policy_seconds` actually run — is right for
    # `send_velocity` and WRONG here: `move()` merges a trailing 0.2 s
    # zero-command settle chunk into `policy_seconds` that `travelled` (which
    # accumulates driving chunks only) excludes. Integrating 0.2 m/s over the
    # merged figure fabricates 0.04 m of forward motion per call — up to 1.6 m
    # per 40-turn stage of drift that is ours, not the robot's: exactly the
    # failure (b) exists to prevent, inverted. Dividing the dead-reckoned
    # distance by the commanded speed recovers the DRIVING seconds exactly,
    # because every chunk is a whole number of 50 Hz steps and
    # `duration_to_steps` round-trips it.
    if travelled > 0.0:
        # Zero only if not one driving step ran, in which case
        # `duration_to_steps`' floor of 1 step would integrate motion that never
        # happened.
        context.integrator.integrate(
            MOVE_SPEED_MPS, 0.0, held_heading, travelled / MOVE_SPEED_MPS
        )
    execution = _record_motion(
        context,
        result,
        distance_moved_m=travelled,
        # doc 06 §5.6's first source: "auto-stops reported by `move`".
        counts_bump=True,
        heading_before=held_heading,
    )

    payload = {
        # The RAW argument, as `turn_to_heading` above and `mark_exit` before it:
        # a key named `requested_` must answer what the model requested. The
        # clamp note carries both numbers whenever they differ, and
        # `status.distance_moved_m` always carries what was actually covered.
        "requested_distance_m": round(requested, 3),
        "policy_seconds": round(result.policy_seconds, 2),
        **_state_payload(context),
    }
    if notes:
        payload["notes"] = notes
    return ToolOutcome(payload=payload, execution=execution)


def _send_velocity(context: ToolContext, args: dict) -> ToolOutcome:
    """Raw command, clamped to the hull. **No auto-stop** (doc 05 §4.2).

    ``stop_on_bump`` is left at ``PolicyPlayback.execute``'s default of ``False``
    on purpose: the command runs its full clamped duration even through contact,
    and the collision is reported afterwards. That is the entire difference
    between this tool and ``move``, and doc 06 §5.6 counts the bump either way.
    """
    values, error = _numbers(args, ("vx", "vy", "wz", "duration_s"), "send_velocity")
    if error is not None:
        return ToolOutcome(payload=error, is_error=True)
    vx, vy, wz, requested_duration = values

    duration, notes = _clamp_duration(requested_duration)
    heading = context.playback.compass_deg()
    # Raw velocities in: `execute()` clamps to the hull and reports the notes,
    # so there is exactly one clamp site and the echo cannot disagree with what
    # actually ran. `stop_on_bump=False` is passed EXPLICITLY, though it is also
    # the default: it is the load-bearing half of doc 05 §4.2's move-vs-
    # send_velocity distinction, and a flipped default in `policy_wrapper` must
    # not be able to turn the raw escape hatch into a second `move` silently.
    result = context.playback.execute(vx, vy, wz, duration, stop_on_bump=False)
    cvx, cvy, cwz = result.commanded

    # The one tool that can translate and rotate at once, so its dead reckoning
    # advances the heading per control step from the COMMANDED wz rather than
    # integrating a whole arc at one stale heading (see
    # `PositionIntegrator.integrate_arc`). `policy_seconds`, never the requested
    # duration: a fall cuts the command short and the estimate must not walk on
    # without the robot (PLAN T3.2 (b) — correct as written for this tool).
    context.integrator.integrate_arc(cvx, cvy, cwz, heading, result.policy_seconds)
    # Arc length of the commanded motion: speed x time, the same rule `move`
    # reports (doc 04 §6.2 — "dead-reckoned distance actually covered by the
    # most recent motion command"). Pinned into doc 05 §4.2, which named the
    # field but not its formula.
    travelled = math.hypot(cvx, cvy) * result.policy_seconds
    execution = _record_motion(
        context,
        result,
        distance_moved_m=travelled,
        # doc 06 §5.6's second source: "`bumped = true` collision reports
        # surfaced in `status` for `send_velocity` commands", one per command.
        counts_bump=True,
        heading_before=heading,
    )

    payload = {
        # doc 05 §4.2's "executed (clamped) command echo" — the COMMAND after
        # clamping, so a model whose command came back changed can see the
        # change. Deliberately not "what ran": `duration_s` here is the clamped
        # request, and the seconds that actually elapsed are `policy_seconds`
        # below. The two differ only when a fall cut the command short, which
        # ends the trial anyway.
        "executed": {
            "vx": round(cvx, 3),
            "vy": round(cvy, 3),
            "wz": round(cwz, 3),
            "duration_s": round(duration, 2),
        },
        "policy_seconds": round(result.policy_seconds, 2),
        **_state_payload(context),
    }
    clamp_notes = [*result.clamp_notes, *notes]
    if clamp_notes:
        payload["notes"] = clamp_notes
    return ToolOutcome(payload=payload, execution=execution)


# ---------------------------------------------------------------------------
# Memory tools (doc 05 §4.3) — the wire, not a second implementation
# ---------------------------------------------------------------------------
#
# Every one of these already returns doc 05 §8's structured shapes WITH type
# validation (T3.1's review pass, doc 05 §5.1). Re-validating here would create
# a second set of rules to keep in sync and a second place for them to diverge;
# the only thing this layer adds is the arity check above, which the methods
# structurally cannot do for themselves.


def _memory_outcome(ack: dict) -> ToolOutcome:
    return ToolOutcome(payload=ack, is_error="error" in ack)


def _update_room(context: ToolContext, args: dict) -> ToolOutcome:
    return _memory_outcome(context.memory.update_room(args["name"], args["description"]))


def _add_landmark(context: ToolContext, args: dict) -> ToolOutcome:
    return _memory_outcome(context.memory.add_landmark(args["room"], args["description"]))


def _mark_exit(context: ToolContext, args: dict) -> ToolOutcome:
    return _memory_outcome(
        context.memory.mark_exit(args["room"], args["direction_deg"], args["status"])
    )


def _set_current_room(context: ToolContext, args: dict) -> ToolOutcome:
    return _memory_outcome(context.memory.set_current_room(args["name"]))


def _correct_position(context: ToolContext, args: dict) -> ToolOutcome:
    # A module-level function, not a `Memory` method: it needs the integrator
    # and the stage-local turn index, both of which live in loop state. They are
    # threaded from the context rather than synthesised here.
    return _memory_outcome(
        correct_position(
            context.memory,
            context.integrator,
            context.turn,
            args["x"],
            args["y"],
            args["reason"],
        )
    )


def _update_plan(context: ToolContext, args: dict) -> ToolOutcome:
    return _memory_outcome(context.memory.update_plan(args["text"]))


def _declare_done(context: ToolContext, args: dict) -> ToolOutcome:
    """Reached only if a caller dispatches it; doc 05 §3.1's loop branches first.

    It still routes to the stage signal rather than erroring, so the tool is
    exercisable end to end (PLAN T3.2's acceptance: "T3.5 exercises every tool
    at least once") and so a loop that lost the branch fails in scoring rather
    than by telling the model its ``declare_done`` was an unknown tool.
    """
    return ToolOutcome(payload=stage_end_result(context.memory.stage))


#: The three tools that step physics. Named once, because the fall guard in
#: :func:`dispatch` and doc 06 §5.6's accounting both key on the same set.
MOTION_TOOLS: tuple[str, ...] = ("turn_to_heading", "move", "send_velocity")


_HANDLERS = {
    "get_observation": _get_observation,
    "look_around": _look_around,
    "turn_to_heading": _turn_to_heading,
    "move": _move,
    "send_velocity": _send_velocity,
    "update_room": _update_room,
    "add_landmark": _add_landmark,
    "mark_exit": _mark_exit,
    "set_current_room": _set_current_room,
    "correct_position": _correct_position,
    "update_plan": _update_plan,
    DECLARE_DONE: _declare_done,
}


# ---------------------------------------------------------------------------
# The dispatcher (doc 05 §3.1, §8)
# ---------------------------------------------------------------------------


def dispatch(call: ToolCall, context: ToolContext) -> ToolOutcome:
    """Route one model tool call. Returns a result for EVERY input.

    The order of the three guards is doc 05 §8's: an unparseable ``arguments``
    JSON, an unknown name, and a failed argument check are all *model* failures
    whose turn still counts, so each returns a structured result rather than
    raising. Only faults with no model agency behind them — a render error, a
    physics NaN — are allowed to escape to the infra path, where the trial
    reruns whole. "The line between model failure and infra failure is drawn at
    agency" (§8).
    """
    if call.parse_error:
        # OpenAI delivers `arguments` as a JSON string and the adapter records a
        # parse failure instead of raising. It looks like a wire fault but is
        # model-attributable — the model emitted the malformed JSON — so §8's
        # first row applies and the turn counts.
        return ToolOutcome(
            payload=invalid_args(
                f"could not parse the arguments of {call.name}: {call.parse_error}",
                "emit the arguments as a valid JSON object",
            ),
            is_error=True,
        )
    if call.name not in _HANDLERS:
        return ToolOutcome(payload=unknown_tool(call.name), is_error=True)
    error = _arg_error(call.name, call.args)
    if error is not None:
        return ToolOutcome(payload=error, is_error=True)
    if call.name in MOTION_TOOLS and context.playback.fell:
        # The fourth guard, and the only one that is about SIM state rather than
        # the model's arguments (hence its position: a malformed call is still
        # told it was malformed). Once the robot has fallen, Isaac has already
        # teleported it back to spawn, so any further command would walk a
        # respawned duck while `pose_trace` and the drift metric accumulated
        # across the teleport. Recorded in doc 05 §4.2/§8.
        return ToolOutcome(payload=trial_over(call.name), is_error=True)
    return _HANDLERS[call.name](context, call.args)
