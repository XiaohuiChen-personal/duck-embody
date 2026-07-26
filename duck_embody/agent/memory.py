"""LLM-authored spatial memory: room/exit/landmark graph, breadcrumb trail,
dead-reckoned position, ``correct_position`` (cognitive loop closure).

**The boundary principle governs every line in this file** (doc 05 §1, AGENTS.md
§3 rule 5): *the harness stores and formats; the LLM perceives, estimates, and
decides.* No geometric fact enters memory unless the model asserted it from its
own observations. The moment the harness starts injecting ground truth — real
positions, real room labels, a geometry-built graph — the benchmark stops
measuring the model's spatial cognition and starts measuring our scaffolding.

Concretely, this module must never:

* create, rename, merge, or de-duplicate a room, exit or landmark on the model's
  behalf, or infer ``leads_to`` from geometry (repairing the model's graph is
  exactly the scaffolding doc 05 §9 says we are here to *not* provide);
* import :mod:`duck_embody.env.apartment_layout` — the ground truth — for any
  purpose, including "just validating" a model assertion;
* reject or sanity-check a ``correct_position`` anchor (doc 05 §4.3: "None
  rejected — a bad anchor is the model's error to make; the log makes it
  measurable");
* emit a covariance, confidence or score (doc 05 §2 drops uncertainty
  propagation: the model is *told* the estimate drifts, and given no number);
* **raise** on anything a model can put in a tool call — doc 05 §5.1: the
  memory tools return structured dicts, never exceptions. An escaping exception
  is classified by doc 05 §8 as an infra fault and reruns the whole trial,
  turning a malformed model argument into a free retry.

Three declared, sensor-realistic exceptions mirror what the physical duck's
hardware provides for free (doc 05 §1): (a) compass heading — the BNO055 IMU
gives absolute yaw; (b) the dead-reckoned position in
:class:`PositionIntegrator`; (c) the closed-loop motion macros that servo on
those two signals. :meth:`Memory.add_breadcrumb` is the harness's *only*
autonomous write into memory, and it stores exactly (b) + (a).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from duck_embody.sim.policy_wrapper import CONTROL_DT, duration_to_steps, wrap_deg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Exits are keyed on (room, direction rounded to the nearest 15°) — doc 05 §4.3.
#: Re-marking the same rounded direction UPDATES the exit rather than growing a
#: second near-duplicate every time the model re-sights the same doorway from a
#: slightly different pose.
EXIT_DIRECTION_QUANTUM_DEG = 15.0

#: The two legal ``Exit.status`` forms (doc 05 §5.1). There is deliberately no
#: ``explored``/``blocked``/``dead_end`` state and no delete tool: every write is
#: an upsert, so all four transitions between the two forms are reachable —
#: including the downgrade ``leads_to:X -> unexplored``, which is how a model
#: retracts a wrong guess.
STATUS_UNEXPLORED = "unexplored"
STATUS_LEADS_TO_PREFIX = "leads_to:"

#: Per-stage caps, rendered into the budget line the model sees.
#: DESIGN doc 06 §3.2 / doc 05 §3.2; mirrored in ``configs/benchmark.yaml``
#: (``caps.turns`` / ``caps.policy_seconds``), which ``tests/test_memory.py``
#: asserts still agrees with these — a cap that drifted between the config the
#: runner enforces and the number the model budgets against would make every
#: "ran out of turns" trial unexplainable.
TURN_CAP = 40
POLICY_SECONDS_CAP = 240.0

#: Maximum length of the standing plan, in characters (doc 05 §4.3, recorded by
#: T3.1's review pass). The plan is re-injected into every request of both
#: stages plus the QA exchange, so it is the one model-authored field whose cost
#: is paid ~85 times over. Uncapped, a single runaway `update_plan` could push a
#: trial into a hard context-window failure, which doc 05 §8 would then classify
#: as an infra fault and rerun WHOLE — laundering a model-caused failure into a
#: free retry. Over-long plans are rejected with the §8 error shape rather than
#: truncated: `update_plan` replaces the plan *verbatim* (§4.3), so silently
#: keeping a prefix would show the model a plan it never wrote. 2000 chars is
#: ~10x doc 05 §5.2's worked example.
PLAN_MAX_CHARS = 2000

#: The stage names of doc 05 §3.3's stage machine. ``Memory.stage`` carries the
#: current one so every ``Correction`` is self-describing.
STAGE_FIND_KITCHEN = "find_kitchen"
STAGE_RETURN_HOME = "return_home"


# ---------------------------------------------------------------------------
# Argument validation (doc 05 §5.1: "structured dicts, never exceptions")
# ---------------------------------------------------------------------------
#
# Every public entry point below is reachable from a model-authored tool call,
# and models routinely emit `"direction_deg": "270"` or `null` despite a
# `{"type": "number"}` schema. Doc 05 §8 puts that case in its FIRST row —
# "args fail validation" -> `{error: "invalid_args", detail, hint}`, the turn
# still counts — and doc 05 §5.1 pins the layer: *the memory tools* return
# structured dicts, never exceptions (PLAN T3.2 relies on it, calling tools.py
# "the wire, not a second implementation"). An escaping TypeError would instead
# land on §8's LAST row ("harness exception outside the model's control"), whose
# policy is to rerun the whole trial — inverting §8's own agency rule and
# handing a model that emitted a malformed argument a free retry.
#
# Validation is deliberately type-only. It never inspects *meaning* (§8: "the
# harness never guesses intent"; §1: no repairing the model's graph).
#
# `text_error` and `number_arg` are PUBLIC because `agent/tools.py` type-checks
# the MOTION arguments (`heading_deg`, `distance_m`, `vx/vy/wz/duration_s`) with
# exactly these functions (T3.2; recorded in doc 05 §5.1, whose rules were
# previously scoped to the memory tools alone). They need the same semantics for
# the same reason — `clamp_command("0.2", 0, 0)` raises `TypeError` and
# `clamp_command(nan, 0, 0)` silently returns `nan`, and either outcome hands a
# malformed model argument the free trial rerun §8 exists to prevent — and one
# implementation is the only way the two layers cannot drift into disagreeing
# about whether `"270"` is a number.


def invalid_args(detail: str, hint: str) -> dict:
    """The doc 05 §8 error shape, returned — never raised."""
    return {"error": "invalid_args", "detail": detail, "hint": hint}


def text_error(value: object, field: str, tool: str) -> dict | None:
    """``None`` if ``value`` is a string, else the §8 error shape.

    Strings are required rather than coerced: ``str(None)`` would create a room
    literally named ``None``, and the model would then have to address it by
    that name for the rest of the episode.
    """
    if isinstance(value, str):
        return None
    return invalid_args(
        f"{field} must be a string in {tool}, got {type(value).__name__}",
        f"pass {field} as a JSON string",
    )


def number_arg(value: object, field: str, tool: str) -> tuple[float | None, dict | None]:
    """``(number, None)`` or ``(None, error)`` — a finite float, or §8's shape.

    ``float()`` accepts the numeric strings models emit for number-typed fields
    (``"270"``), which is parsing, not intent-guessing: the result is exactly
    the value the schema asked for. ``bool`` is excluded even though it is an
    ``int`` subclass — ``True`` is not a bearing. Non-finite values are rejected
    because ``inf``/``nan`` are not quantities: ``json.loads`` accepts the
    ``Infinity``/``NaN`` literals, ``inf % 360`` is ``nan``, and ``nan`` would
    poison the map or the position estimate irrecoverably while rendering as the
    bare token ``nan``.
    """
    if isinstance(value, bool):
        number = None
    else:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            number = None
    if number is None:
        return None, invalid_args(
            f"{field} must be a number in {tool}, got {value!r}",
            f"pass {field} as a JSON number",
        )
    if not math.isfinite(number):
        return None, invalid_args(
            f"{field} must be a finite number in {tool}, got {number!r}",
            f"pass {field} as an ordinary JSON number, not infinity or NaN",
        )
    return number, None


# ---------------------------------------------------------------------------
# Data structures (doc 05 §5.1 — field names and order are the doc's)
# ---------------------------------------------------------------------------


@dataclass
class Room:
    """A place the model says it saw. Created only by ``update_room``."""

    name: str
    description: str
    landmarks: list[str] = field(default_factory=list)


@dataclass
class Exit:
    """A doorway the model says it saw. Created only by ``mark_exit``.

    ``direction_deg`` is stored **already quantised** to the 15° grid the
    harness keys exits on (doc 05 §4.3). Storing the raw assertion instead would
    let the rendered map show ``exit at 272 deg`` for a record that a later
    ``mark_exit(268)`` silently overwrites — the model would see one number and
    be unable to predict which of its exits it was addressing. The ack echoes
    the raw value so the snap is visible rather than silent, the way
    ``clamp_command`` echoes clamps.
    """

    room: str
    direction_deg: float  # absolute compass direction of the doorway
    status: str  # "unexplored" | "leads_to:<room>"


@dataclass
class Crumb:
    """One breadcrumb: where the *integrator* thinks the robot has been.

    Never where it truly is (doc 05 §5.1). ``x``/``y`` come from
    :class:`PositionIntegrator`, ``heading_deg`` from the compass.
    """

    x: float
    y: float
    heading_deg: float


@dataclass
class Correction:
    """One ``correct_position`` call, logged for the post-hoc drift audit.

    ``turn`` is the **stage-local** model-turn index (doc 05 §3.3 resets the
    counters at the ``find_kitchen`` → ``return_home`` transition while keeping
    this memory object), so two corrections in one trial can share a ``turn``
    value. List order gives the sequence but NOT the stage: a stage-2 correction
    on turn 12 is otherwise byte-identical to a stage-1 one, and doc 06 §5.8
    reports drift per stage. ``stage`` is therefore stamped from
    :attr:`Memory.stage` at call time (T3.1 review addition, recorded in doc 05
    §5.1) so the series can be split after the batch, when nothing else can
    recover the boundary.
    """

    turn: int
    old_xy: tuple[float, float]
    new_xy: tuple[float, float]
    reason: str
    stage: str = STAGE_FIND_KITCHEN


@dataclass
class Memory:
    """Everything the model has asserted, plus the breadcrumb trail.

    DEVIATION from doc 05 §5.1 (recorded in that section in the same commit):
    ``room_sequence`` is added. §5.2's worked example renders
    ``Trajectory: living_room -> hallway``, but the §5.1 dataclass has no
    temporal field at all, and §1 forbids the harness deriving room identity
    itself — so the sequence has to come from the model's own
    ``set_current_room`` assertions, and it has to be stored somewhere.
    """

    rooms: dict[str, Room] = field(default_factory=dict)
    exits: list[Exit] = field(default_factory=list)
    breadcrumbs: list[Crumb] = field(default_factory=list)
    current_room: str | None = None
    plan: str = ""
    corrections: list[Correction] = field(default_factory=list)
    #: Rooms in the order the model asserted standing in them, consecutive
    #: repeats collapsed. A genuine revisit (a -> b -> a) keeps all three
    #: entries — that IS the trajectory; only re-asserting the room you are
    #: already in is dropped, or a model that re-confirms its room every turn
    #: would render `Trajectory: hallway -> hallway -> hallway -> ...`.
    room_sequence: list[str] = field(default_factory=list)
    #: Which of doc 05 §3.3's two stages is running. Never rendered and never
    #: model-writable — the loop (T3.4) sets it to ``STAGE_RETURN_HOME`` at the
    #: transition, and it exists only so each `Correction` records its stage.
    stage: str = STAGE_FIND_KITCHEN

    # -- model-asserted writes (doc 05 §4.3) --------------------------------

    def update_room(self, name: str, description: str) -> dict:
        """Upsert a room node. Overwriting a description is legal — the model
        may revise what it thinks a place looks like (doc 05 §4.3)."""
        for value, field_name in ((name, "name"), (description, "description")):
            error = text_error(value, field_name, "update_room")
            if error is not None:
                return error
        if not name.strip():
            # The ONE naming rule. A blank name is not a name the model can
            # address later (`set_current_room("")`), and it renders as
            # `Place 1:  -- ...` and a blank `Trajectory:` entry — the harness
            # showing the model a place it cannot refer to. Nothing else about
            # the name is touched: lookups stay exact, no trimming, no case
            # folding (doc 05 §5.1).
            return invalid_args(
                "room name is empty in update_room",
                "give the place a short name you can refer to later, e.g. "
                "'room_with_tiled_floor'",
            )
        room = self.rooms.get(name)
        if room is None:
            self.rooms[name] = Room(name=name, description=description)
            action = "created"
        else:
            # The upsert keeps the room's original dict slot, so its `Place N`
            # number stays stable for the whole episode. A model that revised a
            # description and saw its rooms renumber would lose the only stable
            # handle it has on its own map.
            room.description = description
            action = "updated"
        return {
            "ok": True,
            "detail": f"room {name!r} {action}",
            "rooms": len(self.rooms),
        }

    def add_landmark(self, room: str, description: str) -> dict:
        """Append a landmark string to a room the model already created."""
        for value, field_name in ((room, "room"), (description, "description")):
            error = text_error(value, field_name, "add_landmark")
            if error is not None:
                return error
        target = self.rooms.get(room)
        if target is None:
            return self._unknown_room_error(room, "add_landmark")
        target.landmarks.append(description)
        return {
            "ok": True,
            "detail": f"landmark added to {room!r}",
            "landmarks": len(target.landmarks),
        }

    def mark_exit(self, room: str, direction_deg: float, status: str) -> dict:
        """Record or update an exit, keyed on (room, direction snapped to 15°)."""
        error = text_error(room, "room", "mark_exit")
        if error is not None:
            return error
        if room not in self.rooms:
            return self._unknown_room_error(room, "mark_exit")
        bearing, error = number_arg(direction_deg, "direction_deg", "mark_exit")
        if error is not None:
            return {
                **error,
                "hint": "direction_deg must be a number in degrees, "
                "counter-clockwise from east (0 = east, 90 = north)",
            }
        clean = normalise_status(status)
        if clean is None:
            return {
                "error": "invalid_args",
                "detail": f"status {status!r} is not a legal exit status",
                "hint": (
                    "status must be exactly 'unexplored' or 'leads_to:<room>' "
                    "— those are the two legal forms"
                ),
            }
        snapped = quantise_direction(bearing)
        existing = self._find_exit(room, snapped)
        if existing is None:
            self.exits.append(Exit(room=room, direction_deg=snapped, status=clean))
            action = "recorded"
        else:
            existing.status = clean
            action = "updated"
        detail = f"exit {action}: {room} at {snapped:g} deg -> {clean}"
        if abs(wrap_deg(bearing) - snapped) > 1e-9:
            # Echo the snap. Silent quantisation would leave a model wondering
            # why the 272° exit it recorded comes back as 270° in the map block.
            detail += f" (snapped from {wrap_deg(bearing):g} deg)"
        return {"ok": True, "detail": detail, "exits": len(self.exits)}

    def set_current_room(self, name: str) -> dict:
        """Assert which of the model's OWN rooms it is standing in."""
        error = text_error(name, "name", "set_current_room")
        if error is not None:
            return error
        if name not in self.rooms:
            return {
                "error": "invalid_args",
                "detail": f"unknown room {name!r} in set_current_room",
                "hint": (
                    "call update_room first to create it; known rooms: "
                    f"{self._known_rooms_text()}"
                ),
            }
        self.current_room = name
        if not self.room_sequence or self.room_sequence[-1] != name:
            self.room_sequence.append(name)
        return {"ok": True, "detail": f"current room set to {name!r}"}

    def update_plan(self, text: str) -> dict:
        """Replace the standing plan verbatim — no re-wrapping, no editing.

        The one bound on the block's size: see :data:`PLAN_MAX_CHARS`. Rejected,
        never truncated, so what the model reads back is always what it wrote.
        """
        error = text_error(text, "text", "update_plan")
        if error is not None:
            return error
        if len(text) > PLAN_MAX_CHARS:
            return invalid_args(
                f"plan is {len(text)} characters; the limit is {PLAN_MAX_CHARS}",
                f"the plan is re-shown every turn, so it must stay under "
                f"{PLAN_MAX_CHARS} characters — keep the standing plan short "
                "and put detail in landmarks and room descriptions "
                "(the previous plan is unchanged)",
            )
        self.plan = text
        return {"ok": True, "detail": "plan replaced"}

    # -- the harness's ONE autonomous write (doc 05 §5.1) -------------------

    def add_breadcrumb(self, x: float, y: float, heading_deg: float) -> Crumb:
        """Append the integrator's estimate + the compass after a motion command.

        This is the only place the harness writes into memory without the model
        asking, and it is pure sensor-realistic state (declared exception (b)
        plus (a) of doc 05 §1). It records where the integrator *thinks* the
        robot has been — never where it truly is.
        """
        crumb = Crumb(x=x, y=y, heading_deg=heading_deg)
        self.breadcrumbs.append(crumb)
        return crumb

    # -- read helpers -------------------------------------------------------

    def unexplored_exits(self) -> list[Exit]:
        return [e for e in self.exits if e.status == STATUS_UNEXPLORED]

    def claimed_edges(self) -> list[tuple[str, str]]:
        """Undirected edges the model claimed via ``leads_to:`` exits.

        Ordering is **exit-creation order**: edges follow the position of their
        source exit in :attr:`exits`, which is where that exit was first
        ``mark_exit``'d — not where its status later became ``leads_to:``. So
        upgrading a long-standing ``unexplored`` frontier renders its edge
        *above* an edge asserted earlier on a newer exit. Doc 05 §5.2 originally
        said "the order the model first asserted it"; T3.1's review pass amended
        it to exit-creation order in the same commit (AGENTS.md rule 5) rather
        than adding an assertion counter to :class:`Exit`, because both rules
        are stable and deterministic and neither is visible to the model as
        anything but a fixed line order.

        Reciprocal assertions (a→b and b→a) collapse to one edge, keeping the
        earlier exit's (room, target) direction; that is the pair the
        ``Connections:`` line renders. An edge whose target room was never
        ``update_room``'d is still returned: it is the model's assertion, and
        dropping it would be the harness quietly repairing the model's graph
        (doc 05 §1).
        """
        edges: list[tuple[str, str]] = []
        seen: set[frozenset[str]] = set()
        for exit_ in self.exits:
            target = exit_status_target(exit_.status)
            if target is None:
                continue
            key = frozenset((exit_.room, target))
            if key in seen:
                continue
            seen.add(key)
            edges.append((exit_.room, target))
        return edges

    # -- internals ----------------------------------------------------------

    def _find_exit(self, room: str, snapped_deg: float) -> Exit | None:
        for exit_ in self.exits:
            if exit_.room == room and exit_.direction_deg == snapped_deg:
                return exit_
        return None

    def _known_rooms_text(self) -> str:
        # Creation order, not alphabetical: it is the order the model sees its
        # own rooms numbered in the map block.
        return ", ".join(self.rooms) if self.rooms else "(none yet)"

    def _unknown_room_error(self, room: str, tool: str) -> dict:
        return {
            "error": "invalid_args",
            "detail": f"unknown room {room!r} in {tool}",
            "hint": f"known rooms: {self._known_rooms_text()}",
        }


@dataclass
class Counters:
    """Per-stage budget counters, rendered into the block's ``Budget:`` line.

    Reset at the ``find_kitchen`` → ``return_home`` transition (doc 05 §3.3);
    the :class:`Memory` object is not.
    """

    turns: int = 0
    policy_seconds: float = 0.0
    turn_cap: int = TURN_CAP
    policy_seconds_cap: float = POLICY_SECONDS_CAP


# ---------------------------------------------------------------------------
# Status / direction helpers
# ---------------------------------------------------------------------------


def quantise_direction(direction_deg: float) -> float:
    """Wrap to [0, 360) and snap to the nearest 15° (doc 05 §4.3 keying).

    Half-up on purpose. ``round()`` is banker's rounding, which would snap 7.5°
    down to 0° but 22.5° up to 30° — an inconsistency a model probing the grid
    would see and no doc explains.
    """
    wrapped = wrap_deg(direction_deg)
    steps = math.floor(wrapped / EXIT_DIRECTION_QUANTUM_DEG + 0.5)
    return (steps * EXIT_DIRECTION_QUANTUM_DEG) % 360.0


def normalise_status(status: str) -> str | None:
    """Return the canonical status string, or ``None`` if malformed.

    Surrounding whitespace is stripped (formatting, not intent-guessing); a
    ``leads_to:`` with an empty target is malformed, because an edge to nothing
    would render a ``Connections:`` line with a blank endpoint.
    """
    if not isinstance(status, str):
        return None
    text = status.strip()
    if text == STATUS_UNEXPLORED:
        return STATUS_UNEXPLORED
    if text.startswith(STATUS_LEADS_TO_PREFIX):
        target = text[len(STATUS_LEADS_TO_PREFIX) :].strip()
        if target:
            return f"{STATUS_LEADS_TO_PREFIX}{target}"
    return None


def exit_status_target(status: str) -> str | None:
    """The room named by a ``leads_to:<room>`` status, else ``None``."""
    if status.startswith(STATUS_LEADS_TO_PREFIX):
        target = status[len(STATUS_LEADS_TO_PREFIX) :].strip()
        return target or None
    return None


# ---------------------------------------------------------------------------
# Dead reckoning (doc 05 §5.1, declared exception (b))
# ---------------------------------------------------------------------------


class PositionIntegrator:
    """Dead reckoning: per 50 Hz control step, rotate the COMMANDED body-frame
    velocity (vx, vy) into the world frame using the compass heading and
    accumulate ``x += v_world_x * dt``, ``y += v_world_y * dt``. Commanded !=
    actual (slip, gait dynamics, bumps), so the estimate drifts honestly.
    ``correct_position(x, y)`` overwrites (x, y); heading is never reset.

    **Commanded, never measured, and never scaled by k.** Two independent
    reasons, both load-bearing:

    1. The deployed observation stack has no linear-velocity sensor at all
       [source: parent ``.claude/rules/rl-training.md`` — the v4_robust actor
       obs is 59-dim with NO ``base_lin_vel``; the BNO055 IMU cannot measure
       it]. An integrator fed measured velocity could not be built on the real
       duck, so it would not be sensor-realistic.
    2. T1.3 measured a velocity-realisation factor k = 1.004
       (``results/figures/smoke/displacement_report.json``: 4.018 m achieved vs
       4.0 m commanded). Applying it here would launder away the very
       phenomenon under study: **the gap between this estimate and the true
       pose IS the drift being measured** (doc 06 §5.8 — |estimate − true| at
       ``declare_done``). A "helpfully" k-corrected integrator would report
       smaller drift for reasons that have nothing to do with the model's
       spatial cognition, and the benchmark would quietly stop measuring what
       it claims to. k is consumed ONLY by the ``move()`` servo target and
       wall-clock forecasting (PLAN T1.3's pinned policy, which AGENTS.md rule 5
       gives precedence over doc 02 §6.2's pseudocode).

    The integrator is initialised at the seed's spawn coordinates — the "known
    start" every dead-reckoning scheme needs — so the estimate and the scorer's
    true poses share one world frame (doc 05 §5.2). That single t=0 anchor is
    the ONLY thing the estimate ever takes from ground truth.
    """

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    def step(
        self, vx: float, vy: float, heading_deg: float, dt: float = CONTROL_DT
    ) -> None:
        """Integrate one control step of a commanded body-frame velocity.

        Body frame: +x forward, +y left. ``heading_deg`` is the compass reading,
        degrees counter-clockwise from world +x (doc 03 §3), so this is the
        standard 2-D rotation — flipping its sign is the classic way to produce
        an estimate that drifts *systematically* rather than honestly.
        """
        rad = math.radians(heading_deg)
        cos_h, sin_h = math.cos(rad), math.sin(rad)
        self.x += (vx * cos_h - vy * sin_h) * dt
        self.y += (vx * sin_h + vy * cos_h) * dt

    def integrate(
        self, vx: float, vy: float, heading_deg: float, duration_s: float
    ) -> None:
        """Integrate a commanded velocity held for ``duration_s``.

        The step count comes from :func:`duration_to_steps`, the same function
        the sim uses to turn a duration into control steps. Computing it any
        other way here would make the estimate disagree with the commanded
        motion for reasons unrelated to drift — and drift is the measurement.

        The caller supplies the compass heading valid for this interval; the
        motion macros call this once per 0.2 s servo chunk, so a turning robot
        is integrated piecewise rather than at one stale heading.

        **The duration must be the one the sim actually ran**, i.e.
        ``ExecResult.policy_seconds``, never the requested duration: a command
        cut short by a bump or a fall would otherwise be integrated in full and
        the estimate would drift for a reason that is ours, not the robot's.
        ``duration_to_steps`` floors at 1 step, so a zero, negative or sub-step
        duration integrates exactly one step — which is what
        ``policy_wrapper.execute()`` also runs, and the two must agree step for
        step (``tests/test_memory.py`` pins them to the same function).
        """
        for _ in range(duration_to_steps(duration_s)):
            self.step(vx, vy, heading_deg)

    def integrate_arc(
        self, vx: float, vy: float, wz: float, heading_deg: float, duration_s: float
    ) -> float:
        """Integrate a commanded velocity whose ``wz`` is turning the robot.

        Returns the commanded heading at the end. Identical to
        :meth:`integrate` when ``wz == 0``, step for step.

        Added by T3.2 for ``send_velocity``, which is the only tool that can
        command translation and rotation at once (``move`` holds its heading and
        ``turn_to_heading`` does not translate). Doc 02 §6.3's pseudocode
        integrates such a command in ONE call at ONE heading, and this method is
        the recorded deviation from it (AGENTS.md rule 5, doc 02 §6.3 / doc 05
        §4.2) because that arithmetic is wrong by an amount larger than the
        success radius: ``send_velocity(0.222, 0, 0.5, 3.0)`` sweeps 86° while
        travelling 0.67 m, so integrating the whole command along the *start*
        heading misplaces the estimate by ~0.45 m — against a
        ``find_kitchen`` success radius of 0.35 m (doc 06 §5.3). That error is
        the harness's arithmetic, not the robot's slip, and PLAN T3.2 (b) exists
        precisely to keep those two apart.

        Only *commanded* values are used, so nothing here reads a sensor: the
        heading advances by ``degrees(wz) * CONTROL_DT`` per control step, the
        same 50 Hz grid :func:`duration_to_steps` puts the sim on. The compass is
        re-read by the caller before the *next* command, so a commanded-vs-realised
        yaw gap (T1.3 measured 0.982) never accumulates across calls — it shows
        up as honest within-call drift, which is the measurement.
        """
        heading = heading_deg
        per_step_deg = math.degrees(wz) * CONTROL_DT
        for _ in range(duration_to_steps(duration_s)):
            # Step at the heading valid at the START of the control step, then
            # advance — the same convention `execute()` uses, which writes the
            # command and then steps physics.
            self.step(vx, vy, heading)
            heading += per_step_deg
        return heading

    def correct(self, x: float, y: float) -> tuple[float, float]:
        """Overwrite (x, y); return the old estimate. Heading is never reset —
        the compass is absolute, so there is nothing about it to correct."""
        old = self.xy
        self.x = float(x)
        self.y = float(y)
        return old


def correct_position(
    memory: Memory,
    integrator: PositionIntegrator,
    turn: int,
    x: float,
    y: float,
    reason: str,
) -> dict:
    """Cognitive loop closure: re-anchor the estimate and log the correction.

    Obeyed **unconditionally** (doc 05 §4.3, §5.3). The harness checks the
    anchor against nothing — a bad anchor is the model's error to make, and the
    log is what makes it measurable (doc 06 §5.8). Checking it would require the
    ground truth the model is never given, and would turn a cognition
    measurement into a scaffolding measurement. The type check below is not that
    check: it rejects arguments that are not numbers at all, and every finite
    (x, y) — including wildly wrong ones — is obeyed.

    **Both coordinates are validated before either is written.** Coercing
    inside :meth:`PositionIntegrator.correct` instead would let a malformed
    ``y`` abort *after* ``x`` had been re-anchored, leaving the estimate in a
    coordinate frame that never existed and — because the exception escaped
    before the append below — no ``Correction`` in the log to explain it.
    """
    new_x, error = number_arg(x, "x", "correct_position")
    if error is not None:
        return {**error, "hint": "x and y must be numbers, in metres"}
    new_y, error = number_arg(y, "y", "correct_position")
    if error is not None:
        return {**error, "hint": "x and y must be numbers, in metres"}
    reason_error = text_error(reason, "reason", "correct_position")
    if reason_error is not None:
        return reason_error
    old = integrator.correct(new_x, new_y)
    new = (new_x, new_y)
    memory.corrections.append(
        Correction(
            turn=turn, old_xy=old, new_xy=new, reason=reason, stage=memory.stage
        )
    )
    return {
        "ok": True,
        "detail": (
            f"position estimate re-anchored from ({old[0]:.2f}, {old[1]:.2f}) "
            f"to ({new[0]:.2f}, {new[1]:.2f}); heading unchanged"
        ),
        "delta_m": round(math.dist(old, new), 3),
    }
