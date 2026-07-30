"""Tool-surface unit tests: schemas, dispatch routing, clamping + echo,
structured errors, and the two things that fail SILENTLY (doc 05 §4/§8).

The sim is mocked throughout — no kit process, no GPU, no Isaac import. The
fakes below re-implement only the *shapes* ``policy_wrapper`` returns, and they
call the wrapper's own pure functions (``clamp_command``, ``duration_to_steps``)
so a clamp rule cannot be tested against a second definition of itself.

Seven guards here protect failures that produce no traceback and no red test:

* **The schema is compared against the design doc itself**, extracted from
  ``docs/designs/05-agent-harness.html`` §4 and parsed as JSON — not against a
  copy pasted into this file, which would drift in lockstep with ``tools.py``
  and still pass. Doc 05 §6 records why this is load-bearing: the frozen prompt
  carries only a paraphrase, so §4's numeric bounds (the ``send_velocity`` hull,
  ``turn_to_heading``'s ``[0,360)``) reach the model through *no other*
  model-facing text. A thinner description ships a quietly harder benchmark.

* **The ground-truth leak test.** ``ExecResult`` carries four scoring-only
  fields. Nothing crashes when a benchmark hands the model its own answer key;
  the numbers just stop meaning what they say (doc 06 §4).

* **The dead-reckoning feed.** ``move()``'s merged ``policy_seconds`` includes a
  trailing zero-command settle chunk its ``dead_reckoned_distance_m`` excludes.
  Integrating over the merged figure fabricates 0.04 m per call — 0.4 % of a
  1.5 m move, invisible to the eye, and it accumulates into exactly the drift
  metric doc 06 §5.8 exists to measure. PLAN T3.2 (b) originally *instructed*
  that arithmetic; it is amended in this commit.

* **Every malformed argument returns rather than raises.** An escaping exception
  is classified by doc 05 §8 as an infra fault and reruns the trial WHOLE,
  turning a bad tool call into a free retry — the selection bias §8 exists to
  prevent.

* **The scoring-side channel survives.** ``dispatch`` is the only code that ever
  sees an ``ExecResult``. Dropping it is exactly as silent as leaking it and
  strictly more expensive: doc 06 §4 requires ``turns[].execution.pose_trace`` in
  every trial JSON and T4.1's scorer raises without it, so the failure surfaces
  only after 12 paid trials have run.

* **``to_block`` is inspected, not just its ``.text``.** It is the only path from
  the dispatcher to a provider, and its other two channels fail invisibly:
  blanking ``images`` makes every model drive blind for the whole batch, and
  hard-coding ``is_error`` delivers every rejected call as a success. Both leave
  a perfectly well-formed transcript.

* **A fall has already teleported the robot** back to spawn, inside
  ``env.step()``. Every live sensor read afterwards describes the spawn point, so
  the fakes below model the teleport — a fake that left the compass alone would
  let ``tools.py`` re-read it and write a spawn heading permanently into the
  breadcrumb trail, with nothing to see.
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
import re
from pathlib import Path

import pytest

from duck_embody.agent.memory import (
    Counters,
    Memory,
    PositionIntegrator,
    STAGE_FIND_KITCHEN,
    STAGE_RETURN_HOME,
)
from duck_embody.agent.prompts import STAGE2_OBJECTIVE_TOOL_RESULT, SYSTEM_PROMPT
from duck_embody.agent.providers.base import ToolCall
from duck_embody.agent.tools import (
    DECLARE_DONE,
    DURATION_RANGE_S,
    LOOK_AROUND_BEARINGS_DEG,
    MOTION_TOOLS,
    POSITION_ESTIMATE_NOTE,
    TOOL_NAMES,
    TOOL_SCHEMAS,
    TRIAL_OVER_DETAIL,
    ToolContext,
    ToolOutcome,
    dispatch,
    not_executed,
    stage_end_result,
    trial_over,
    unknown_tool,
)
from duck_embody.env.camera import RESOLUTION, encode_b64
from duck_embody.sim.policy_wrapper import (
    CONTROL_DT,
    K_VELOCITY_REALISATION,
    MACRO_CHUNK_S,
    MOVE_MAX_DISTANCE_M,
    MOVE_SPEED_MPS,
    VX_RANGE,
    VY_RANGE,
    WZ_RANGE,
    ExecResult,
    clamp_command,
    duration_to_steps,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_04 = REPO_ROOT / "docs" / "designs" / "04-camera-observation.html"
DOC_05 = REPO_ROOT / "docs" / "designs" / "05-agent-harness.html"
BENCHMARK_YAML = REPO_ROOT / "configs" / "benchmark.yaml"

EPS = 1e-9

#: Every JSON-reachable value a model can put in an argument slot. Same list as
#: `tests/test_memory.py`, deliberately: the memory tools and the motion tools
#: must survive the identical set, and `json.loads` really does accept the
#: `NaN`/`Infinity` literals.
JSON_VALUES = [
    None, True, False, 0, -1, 2.5, "", "text", "270", [], [1, 2], {}, {"a": 1},
    float("nan"), float("inf"), float("-inf"),
]

#: Ground truth planted in every fake ``ExecResult``. Chosen so no legitimate
#: model-facing number can collide with them by accident.
TRUE_XY = (7.77, -8.88)
TRUE_POSE = (7.77, -8.88, 123.45)
TRUE_DISPLACEMENT_M = 9.99
SCORING_ONLY_FIELDS = (
    "pose_trace",
    "sampled_xy",
    "true_pose",
    "true_displacement_m",
)

#: The integrator's starting estimate for the fixtures below — doc 04 §6's own
#: worked example, so the payload can be eyeballed against the doc.
START_XY = (1.42, -0.31)
START_COMPASS_DEG = 87.4


# ---------------------------------------------------------------------------
# Helpers: pull the doc's own schema out of the HTML, so the test cites the source
# ---------------------------------------------------------------------------


def doc_tool_schemas() -> list[dict]:
    """Doc 05 §4's canonical 12-tool block, parsed out of the design doc.

    Extracted rather than transcribed for the reason `tests/test_memory.py`
    extracts §5.2's golden memory block: a transcribed copy can be edited in the
    same commit as the code it is supposed to police, and nothing fails.
    """
    source = DOC_05.read_text(encoding="utf-8")
    match = re.search(
        r'<pre><code>(\[\s*\{"name": "get_observation".*?\])\s*</code></pre>',
        source,
        flags=re.DOTALL,
    )
    assert match, "doc 05 §4's canonical tool schema block is no longer in the HTML"
    # `&lt;`/`&gt;` are the only escaping in the block; they decode to the `<>`
    # of `leads_to:<room>`.
    return json.loads(html.unescape(match.group(1)))


def doc_04_frozen_payload() -> dict:
    """Doc 04 §6's frozen ``get_observation`` payload, parsed out of the doc.

    Extracted, never transcribed — the same rule the schema block above follows.
    Two things this pins that substring greps could not: the exact wording of the
    ``position_estimate.note`` (a grep for "drift" and "correct_position" passes
    happily on a note that says drift is negligible and ``correct_position`` is
    rarely needed — the inversion of the one behaviour the memory scaffolding
    exists to exercise), and the FIELD ORDER, which ``to_block``'s own docstring
    calls load-bearing.

    The block is HTML-escaped JSON whose ``"text"`` value is itself a
    JSON-escaped object printed with real newlines "for readability only" (the
    doc says so). Unescaping the inner quotes yields an object whose newlines are
    ordinary JSON whitespace, so it parses.
    """
    source = DOC_04.read_text(encoding="utf-8")
    match = re.search(
        r'"type": "text",\s*\n\s*"text": "(\{.*?\})"\s*\n\s*\}', source, flags=re.DOTALL
    )
    assert match, "doc 04 §6's frozen observation payload is no longer in the HTML"
    return json.loads(html.unescape(match.group(1)).replace('\\"', '"'))


def prompt_section_2() -> str:
    """The frozen SYSTEM_PROMPT's tool-documentation section."""
    start = SYSTEM_PROMPT.index("2. **Tool documentation.**")
    end = SYSTEM_PROMPT.index("3. **Navigation doctrine")
    return SYSTEM_PROMPT[start:end]


def backticked_identifiers(text: str) -> set[str]:
    """Leading identifier of every ``\u0060...\u0060`` span in ``text``.

    ``turn_to_heading(heading_deg)`` and ``bumped: true`` both reduce to their
    leading identifier, which is what makes the prompt's prose comparable with a
    tool-name list at all.
    """
    found = set()
    for span in re.findall(r"`([^`]+)`", text):
        match = re.match(r"[a-z_][a-z0-9_]*", span.strip())
        if match:
            found.add(match.group(0))
    return found


#: Backticked in the frozen prompt's §2 but NOT tools: they are result-field
#: names. Curated rather than inferred — a naive "every backticked identifier is
#: a tool" assertion fails on these three, and weakening it to a substring check
#: would stop catching a renamed tool, which is the whole point (PLAN T3.2).
#: The test below also asserts `tools.py` really does emit all three.
#: Backticked names in the prompt that are RESULT FIELDS or field VALUES, not
#: tools. Excusing them from the tool-name check is safe only because
#: `test_the_result_fields_the_prompt_promises_are_really_emitted` asserts each
#: one is genuinely produced — the allowlist redirects the assertion, it does
#: not remove it.
RESULT_FIELDS_NAMED_IN_THE_PROMPT = {
    "timed_out", "bumped", "hint",
    # `status.contact` and its four possible values (T3.5).
    "status", "contact", "head", "torso", "left_leg", "right_leg",
}

#: The exact regions `status.contact` may report. The prompt names all four, so
#: the code must be able to produce all four and no others.
CONTACT_REGIONS = {"head", "torso", "left_leg", "right_leg"}


# ---------------------------------------------------------------------------
# Fakes (the "sim mocked" of PLAN T3.2) — shapes only, wrapper math reused
# ---------------------------------------------------------------------------


def _exec_result(**kwargs) -> ExecResult:
    """An ``ExecResult`` with the SCORING-ONLY fields loaded with sentinels.

    Every fake result carries them, so any test that inspects a payload is
    simultaneously a leak test.
    """
    base = dict(
        commanded=(0.0, 0.0, 0.0),
        duration_s=0.0,
        steps=0,
        policy_seconds=0.0,
        bumped=False,
        fell=False,
        pose_trace=[TRUE_XY, TRUE_XY],
        sampled_xy=[TRUE_XY],
        true_pose=TRUE_POSE,
        true_displacement_m=TRUE_DISPLACEMENT_M,
    )
    base.update(kwargs)
    return ExecResult(**base)


class FakePlayback:
    """Stands in for ``PolicyPlayback``: same signatures, same result shapes.

    The macro fakes reproduce the one structural detail that matters to
    ``tools.py`` — a macro's ``policy_seconds`` includes a trailing 0.2 s
    zero-command settle chunk that its ``dead_reckoned_distance_m`` excludes
    (``policy_wrapper.move`` lines 592-617). A fake that merged them would make
    the dead-reckoning test below pass against the exact bug it exists to catch.
    """

    def __init__(self, compass_deg: float = START_COMPASS_DEG, fell: bool = False):
        self._compass = compass_deg
        self._fell = fell
        self.calls: list[tuple[str, dict]] = []
        #: Set to force a macro to stop early (bump / fall) after N drive chunks.
        self.stop_after_chunks: int | None = None
        self.bumped = False
        #: What `contact_groups()` sampled at the bump. The real `execute()`
        #: only populates `ExecResult.contact_groups` when `bumped` went true
        #: (`policy_wrapper.py:518-526`), so the fakes mirror that: non-empty
        #: iff the command bumped. A fake that always returned [] would leave
        #: the carry-and-reset path of `status.contact` untestable.
        self.bump_contact_groups: list[str] = ["torso"]
        self.turn_stop_reason = "reached"
        #: Compass reading the turn macro lands on; None = perfect turn.
        self.turn_lands_on: float | None = None
        #: Force `execute()` to terminate after N control steps, as a fall does.
        #: WITHOUT THIS THE FAKE CANNOT TRUNCATE: the real `execute()` breaks out
        #: of its step loop on termination (`policy_wrapper.py:390-397) and
        #: returns fewer `policy_seconds` than were asked for, but a fake that
        #: always returns `duration_to_steps(duration_s)` makes
        #: `policy_seconds == duration_s` an invariant — and PLAN T3.2 (b)
        #: ("feed the integrator the duration actually run, never the requested
        #: duration") then has ZERO coverage for `send_velocity`, the one tool
        #: the corrected clause was written for. `move`'s side is covered by
        #: `stop_after_chunks`; this is the `execute()` side.
        self.execute_stop_after_steps: int | None = None
        #: Mirrors PolicyPlayback._fall_diagnostics: stamped on the
        #: ExecResult only by the call that actually terminated.
        self.fall_diagnostics: dict | None = None
        #: Compass reading `move()` ends on; None = the heading never changes.
        #: The real robot's does: T1.3 measured ~1.8 deg/s of yaw under a
        #: straight command, and a bump-stopped move can end rotated. A fake that
        #: never moves the compass cannot tell a heading latched BEFORE the macro
        #: from one read after it.
        self.move_ends_on_compass: float | None = None
        #: Set to make `move()` terminate in a fall.
        self.move_falls = False
        #: What `compass_deg()` reads AFTER a fall. Isaac Lab auto-resets a
        #: terminated env INSIDE `env.step()` and teleports the robot back to
        #: spawn before the call returns (`policy_wrapper.execute` lines
        #: 382-397), so every live sensor read afterwards describes the spawn
        #: point rather than where the duck toppled. `policy_wrapper` guards its
        #: own `true_pose` against this; a fake that left the compass alone would
        #: let `tools.py` re-read it and never notice.
        self.spawn_compass_deg = 45.0

    # -- sensors ------------------------------------------------------------

    def compass_deg(self) -> float:
        return self._compass

    @property
    def fell(self) -> bool:
        return self._fell

    def _teleport_to_spawn(self) -> None:
        """What Isaac's auto-reset does to every live sensor after a fall."""
        self._compass = self.spawn_compass_deg

    # -- execution ----------------------------------------------------------

    def execute(self, vx, vy, wz, duration_s, stop_on_bump=False, stop_predicate=None):
        self.calls.append(
            (
                "execute",
                dict(
                    vx=vx, vy=vy, wz=wz, duration_s=duration_s,
                    stop_on_bump=stop_on_bump,
                ),
            )
        )
        # The real `execute()` clamps here; reuse the real function so the echo
        # under test is not compared against a second implementation of it.
        commanded, notes = clamp_command(vx, vy, wz)
        steps = duration_to_steps(duration_s)
        stop_reason = ""
        if (
            self.execute_stop_after_steps is not None
            and self.execute_stop_after_steps < steps
        ):
            steps = self.execute_stop_after_steps
            # `_fell` is real instance state on `PolicyPlayback` (it is what the
            # `fell` property reads), so the fake has to set it, not just stamp
            # the result — otherwise `status.fell` and `result.fell` disagree.
            self._fell = True
            stop_reason = "fell"
        result = _exec_result(
            commanded=commanded,
            duration_s=duration_s,
            steps=steps,
            policy_seconds=steps * CONTROL_DT,
            bumped=self.bumped,
            contact_groups=list(self.bump_contact_groups) if self.bumped else [],
            fell=self._fell,
            fall_diagnostics=self.fall_diagnostics if stop_reason == "fell" else None,
            clamp_notes=notes,
            stop_reason=stop_reason,
        )
        if stop_reason == "fell":
            self._teleport_to_spawn()
        return result

    def turn_to_heading(self, heading_deg, **kwargs):
        self.calls.append(("turn_to_heading", dict(heading_deg=heading_deg, **kwargs)))
        chunks = self.stop_after_chunks if self.stop_after_chunks is not None else 3
        drive_s = chunks * MACRO_CHUNK_S
        fell = self.turn_stop_reason == "fell"
        settle_s = 0.0 if fell else MACRO_CHUNK_S
        self._compass = (
            heading_deg if self.turn_lands_on is None else self.turn_lands_on
        )
        if fell:
            self._fell = True
        result = _exec_result(
            commanded=(0.0, 0.0, 0.3),
            duration_s=MACRO_CHUNK_S,  # stale on macro results — never reported
            steps=duration_to_steps(drive_s + settle_s),
            policy_seconds=drive_s + settle_s,
            bumped=self.bumped,
            contact_groups=list(self.bump_contact_groups) if self.bumped else [],
            fell=self._fell,
            stop_reason=self.turn_stop_reason,
        )
        if fell:
            self._teleport_to_spawn()
        return result

    def move(self, distance_m, hold_heading=True, stop_on_bump=True, on_chunk=None):
        self.calls.append(
            (
                "move",
                dict(
                    distance_m=distance_m,
                    hold_heading=hold_heading,
                    stop_on_bump=stop_on_bump,
                ),
            )
        )
        # The real servo target is `distance / k` and the loop breaks on
        # `travelled >= target`, so the distance served is quantised UP to a
        # whole number of 0.2 s chunks (0.04 m each). Reproduced faithfully — a
        # fake that served the requested distance exactly would hide the
        # granularity `move` really has, which is the finest final-positioning
        # step a model gets against doc 06 §5.3's 0.35 m success radius.
        ideal = max(
            1,
            math.ceil(
                distance_m
                / K_VELOCITY_REALISATION
                / (MOVE_SPEED_MPS * MACRO_CHUNK_S)
            ),
        )
        chunks = ideal if self.stop_after_chunks is None else self.stop_after_chunks
        drive_s = chunks * MACRO_CHUNK_S
        travelled = MOVE_SPEED_MPS * drive_s
        fell = self._fell or self.move_falls
        self._fell = fell
        if self.move_ends_on_compass is not None:
            self._compass = self.move_ends_on_compass
        # THE STRUCTURAL DETAIL: the settle chunk is merged into policy_seconds
        # but contributes nothing to `travelled` (and is skipped after a fall).
        settle_s = 0.0 if fell else MACRO_CHUNK_S
        reason = "fell" if self.move_falls else ("bump" if self.bumped else "reached")
        result = _exec_result(
            commanded=(MOVE_SPEED_MPS, 0.0, 0.0),
            duration_s=MACRO_CHUNK_S,  # stale on macro results — never reported
            steps=duration_to_steps(drive_s + settle_s),
            policy_seconds=drive_s + settle_s,
            bumped=self.bumped,
            contact_groups=list(self.bump_contact_groups) if self.bumped else [],
            fell=fell,
            stop_reason=reason,
            dead_reckoned_distance_m=travelled,
        )
        if self.move_falls:
            self._teleport_to_spawn()
        return result

    # -- assertions helpers -------------------------------------------------

    def names_called(self) -> list[str]:
        return [name for name, _ in self.calls]


class FakeCamera:
    """Stands in for ``HeadCamera``. Returns real arrays so the encoder runs."""

    def __init__(self):
        self.captures = 0
        self.look_around_bearings = None

    @staticmethod
    def _frame(seed: int):
        import numpy as np

        # Deliberately NOT 512x512: `encode_jpeg` must resize to the frozen
        # resolution, and a same-size fixture would never exercise that.
        return np.full((8, 8, 3), seed % 256, dtype=np.uint8)

    def capture_b64(self, quality: int = 85) -> str:
        self.captures += 1
        return encode_b64(self._frame(1), quality)

    def look_around(self, bearings_deg=(0, 90, 180, 270)):
        self.look_around_bearings = tuple(bearings_deg)
        return [
            # (bearing, rgb, forward_vector) — the third element is ground truth
            # derived from the robot's TRUE yaw and must never reach a payload.
            (bearing, self._frame(i + 2), (TRUE_XY[0], TRUE_XY[1], 0.0))
            for i, bearing in enumerate(bearings_deg)
        ]


def make_context(**kwargs) -> ToolContext:
    """A dispatch context built through the public API, like the loop builds it."""
    playback = kwargs.pop("playback", None) or FakePlayback()
    return ToolContext(
        playback=playback,
        camera=kwargs.pop("camera", None) or FakeCamera(),
        memory=kwargs.pop("memory", None) or Memory(),
        integrator=kwargs.pop("integrator", None) or PositionIntegrator(*START_XY),
        counters=kwargs.pop("counters", None) or Counters(),
        **kwargs,
    )


def call(tool, /, **args) -> ToolCall:
    """Build a ``ToolCall``. Positional-only, because ``update_room`` and
    ``set_current_room`` both take an argument literally called ``name``."""
    return ToolCall(id=f"call_{tool}", name=tool, args=dict(args))


#: One legal call per tool, for the routing and never-raise sweeps.
VALID_ARGS: dict[str, dict] = {
    "get_observation": {},
    "look_around": {},
    "turn_to_heading": {"heading_deg": 90.0},
    "move": {"distance_m": 0.6},
    "send_velocity": {"vx": 0.1, "vy": 0.0, "wz": 0.2, "duration_s": 1.0},
    "update_room": {"name": "living_room", "description": "sofa, blue rug"},
    "add_landmark": {"room": "living_room", "description": "coffee table"},
    "mark_exit": {"room": "living_room", "direction_deg": 90, "status": "unexplored"},
    "set_current_room": {"name": "living_room"},
    "correct_position": {"x": 0.6, "y": 0.6, "reason": "recognized the blue rug"},
    "update_plan": {"text": "follow the hallway east"},
    DECLARE_DONE: {},
}


def seeded_context(**kwargs) -> ToolContext:
    """A context whose memory already holds ``living_room`` (so the memory tools
    that require an existing room have one to address)."""
    memory = Memory()
    memory.update_room("living_room", "sofa, blue rug")
    return make_context(memory=memory, **kwargs)


# ---------------------------------------------------------------------------
# The schema IS doc 05 §4 (doc 05 §6, PLAN T3.2 (a))
# ---------------------------------------------------------------------------


class TestSchemaMatchesTheDesignDoc:
    def test_the_tool_set_is_the_docs_twelve_in_the_docs_order(self):
        assert [s["name"] for s in TOOL_SCHEMAS] == [
            s["name"] for s in doc_tool_schemas()
        ]
        assert len(TOOL_SCHEMAS) == 12

    @pytest.mark.parametrize("index", range(12))
    def test_every_description_string_is_the_docs_verbatim(self, index):
        """PLAN T3.2 (a): compare the STRINGS, not just the names.

        Doc 05 §6 records the deviation that makes this the only guard: the
        frozen prompt carries a paraphrase and says so, so the `send_velocity`
        hull and `turn_to_heading`'s [0,360) domain reach the model through
        these strings and nothing else. A description trimmed here ships a
        benchmark where every model is quietly told less, and the whole batch
        freezes that way.
        """
        doc = doc_tool_schemas()[index]
        mine = TOOL_SCHEMAS[index]
        assert mine["name"] == doc["name"]
        assert mine["description"] == doc["description"], (
            f"{doc['name']}'s description drifted from doc 05 §4"
        )

    def test_every_input_schema_is_the_docs_object_for_object(self):
        for mine, doc in zip(TOOL_SCHEMAS, doc_tool_schemas()):
            assert mine["input_schema"] == doc["input_schema"], mine["name"]

    def test_the_whole_block_round_trips_through_json_unchanged(self):
        """Both adapters serialise this straight onto the wire; a tuple or a
        NaN sneaked in here would 400 the request for every model at once."""
        assert json.loads(json.dumps(TOOL_SCHEMAS)) == TOOL_SCHEMAS

    @pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["name"])
    def test_every_schema_is_a_well_formed_json_schema_object(self, schema):
        body = schema["input_schema"]
        assert body["type"] == "object"
        assert isinstance(body["properties"], dict)
        assert isinstance(body["required"], list)
        # Doc 05 §4 has ZERO optional parameters: every declared property is
        # required. A property that slipped out of `required` would be a
        # silently optional argument the handlers still index into.
        assert sorted(body["required"]) == sorted(body["properties"])
        for name, spec in body["properties"].items():
            assert spec["type"] in ("number", "string"), name

    @pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["name"])
    def test_no_machine_readable_bounds_were_added_to_the_schema(self, schema):
        """Doc 05 §4's preamble: "the schema is documentation for the model, the
        harness is the enforcer".

        Adding `minimum`/`enum`/`additionalProperties` would look like a
        tightening but is a silent change to the frozen prompt (doc 06 §2) — and
        worse, it moves enforcement to the provider, whose rejection arrives as
        an API error rather than as the `invalid_args` doc 05 §8 requires, so
        the turn would stop counting.
        """
        blob = json.dumps(schema)
        for banned in ("minimum", "maximum", "enum", "default", "additionalProperties"):
            assert banned not in blob, f"{schema['name']} grew a `{banned}`"


class TestToolNamesMatchTheFrozenPrompt:
    """PLAN T3.2 (added by T3.1): the prompt is frozen and describes the tool
    surface in prose. A tool renamed in `tools.py` alone would leave every model
    reading instructions for a tool that no longer exists, and nothing else in
    the suite would fail."""

    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_every_tool_is_documented_in_system_prompt_section_2(self, name):
        assert f"`{name}" in prompt_section_2(), (
            f"{name} is in the schema but not in the frozen prompt's §2"
        )

    def test_the_prompt_documents_no_tool_that_does_not_exist(self):
        named = backticked_identifiers(prompt_section_2())
        unknown = named - set(TOOL_NAMES) - RESULT_FIELDS_NAMED_IN_THE_PROMPT
        assert not unknown, (
            f"the frozen prompt promises tools that do not exist: {sorted(unknown)}"
        )

    def test_the_result_fields_the_prompt_promises_are_really_emitted(self):
        """The other half of the allowlist above: `timed_out`, `bumped` and
        `hint` are excused from the tool-name check because they are result
        keys — so they had better BE result keys."""
        context = make_context()
        turn = dispatch(call("turn_to_heading", heading_deg=180.0), context)
        assert "timed_out" in turn.payload
        assert "bumped" in turn.payload["status"]
        bad = dispatch(call("move", distance_m="north"), context)
        assert "hint" in bad.payload

        # `status.contact` and the four region names the prompt promises.
        obs = dispatch(call("get_observation"), context)
        assert "contact" in obs.payload["status"], (
            "the prompt documents status.contact but no observation emits it"
        )
        assert isinstance(obs.payload["status"]["contact"], list)

    def test_the_contact_regions_the_prompt_names_are_the_ones_the_code_groups(self):
        """The prompt tells the model to expect head / torso / left_leg /
        right_leg. If `policy_wrapper` ever groups the articulation tree into a
        different set, the prompt would be describing regions the model can
        never be shown — silent, and frozen into the batch."""
        import inspect

        from duck_embody.sim import policy_wrapper

        src = inspect.getsource(policy_wrapper.PolicyPlayback.__init__)
        for region in CONTACT_REGIONS:
            assert f'"{region}"' in src, f"{region} is promised but never grouped"


# ---------------------------------------------------------------------------
# Dispatch routing (doc 05 §3.1)
# ---------------------------------------------------------------------------


class TestDispatchRouting:
    def test_the_handler_table_and_the_schema_name_the_same_tools(self):
        """`_arg_error` indexes the schema by the dispatched name. A handler
        registered under a name the schema does not declare would KeyError
        there — an escaping exception, which doc 05 §8 sends to the infra path
        and reruns the whole trial. A schema with no handler is the reverse:
        the model reads about a tool that answers `unknown_tool`."""
        from duck_embody.agent.tools import _HANDLERS

        assert set(_HANDLERS) == set(TOOL_NAMES)

    def test_motion_tools_names_exactly_the_tools_that_step_physics(self):
        """`MOTION_TOOLS` drives the post-fall guard and doc 06 §5.6's
        accounting. Derived here from behaviour rather than trusted: a fourth
        motion tool added to the schema and forgotten in this tuple would keep
        commanding a robot that has already been teleported back to spawn."""
        from duck_embody.agent.tools import _HANDLERS

        stepped = set()
        for name in TOOL_NAMES:
            context = seeded_context()
            dispatch(call(name, **VALID_ARGS[name]), context)
            if context.playback.calls:
                stepped.add(name)
        assert stepped == set(MOTION_TOOLS)
        assert set(MOTION_TOOLS) <= set(_HANDLERS)

    def test_the_fixture_covers_every_tool(self):
        """`VALID_ARGS` drives the routing and never-raise sweeps below. A tool
        added to the schema and forgotten here would be swept by nothing while
        the suite stayed green."""
        assert set(VALID_ARGS) == set(TOOL_NAMES)

    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_every_tool_dispatches_to_a_structured_result(self, name):
        context = seeded_context()
        outcome = dispatch(call(name, **VALID_ARGS[name]), context)
        assert isinstance(outcome.payload, dict)
        assert not outcome.is_error, outcome.payload
        # Serialisable, because `to_block` will do exactly this on the wire.
        json.loads(outcome.to_block("id", name).text)

    def test_perception_tools_step_no_physics(self):
        """doc 05 §4.1: `get_observation` does no physics steps, and
        `look_around` is a virtual gimbal. The frozen prompt promises the model
        `look_around` costs "no motion budget spent" — charging for it would
        make the frozen prose false."""
        context = make_context()
        dispatch(call("get_observation"), context)
        dispatch(call("look_around"), context)
        assert context.playback.calls == []
        assert context.counters.policy_seconds == 0.0
        assert context.memory.breadcrumbs == []

    def test_move_routes_to_the_macro_that_auto_stops(self):
        """doc 05 §4.2: `move` auto-stops on collision. That behaviour lives in
        `PolicyPlayback.move(stop_on_bump=True)`; reaching for `execute()` here
        would silently produce a `move` that drives on through walls."""
        context = make_context()
        dispatch(call("move", distance_m=0.6), context)
        assert context.playback.names_called() == ["move"]
        assert context.playback.calls[0][1]["stop_on_bump"] is True
        assert context.playback.calls[0][1]["hold_heading"] is True

    def test_send_velocity_routes_to_execute_and_never_auto_stops(self):
        """The distinction doc 05 §4.2 asserts twice and PLAN T3.2 puts in its
        Context line: "send_velocity has NO auto-stop; move does". Blurring it
        turns the raw escape hatch into a second `move` and makes doc 06 §5.6's
        bump counter mean two different things across tools."""
        context = make_context()
        dispatch(
            call("send_velocity", vx=0.1, vy=0.0, wz=0.0, duration_s=1.0), context
        )
        assert context.playback.names_called() == ["execute"]
        assert context.playback.calls[0][1]["stop_on_bump"] is False

    def test_memory_tools_mutate_memory_and_advance_no_physics(self):
        context = seeded_context()
        dispatch(call("add_landmark", room="living_room", description="rug"), context)
        dispatch(
            call("mark_exit", room="living_room", direction_deg=272, status="unexplored"),
            context,
        )
        dispatch(call("set_current_room", name="living_room"), context)
        dispatch(call("update_plan", text="head east"), context)
        assert context.memory.rooms["living_room"].landmarks == ["rug"]
        assert [(e.room, e.direction_deg) for e in context.memory.exits] == [
            ("living_room", 270.0)
        ]
        assert context.memory.current_room == "living_room"
        assert context.memory.plan == "head east"
        assert context.playback.calls == []
        assert context.counters.policy_seconds == 0.0

    def test_correct_position_reanchors_and_logs_with_the_stage_local_turn(self):
        """`correct_position` is the one memory tool that is NOT a `Memory`
        method: it needs the integrator and the stage-local turn index, both of
        which live in loop state. A dispatcher that synthesised the turn number
        would make doc 06 §5.8's per-stage correction series unreadable."""
        context = seeded_context(turn=7)
        outcome = dispatch(
            call("correct_position", x=0.6, y=0.6, reason="blue rug again"), context
        )
        assert context.integrator.xy == (0.6, 0.6)
        assert outcome.payload["ok"] is True
        assert [(c.turn, c.stage) for c in context.memory.corrections] == [
            (7, STAGE_FIND_KITCHEN)
        ]

    def test_a_rejected_memory_write_is_an_error_result_not_an_ack(self):
        """doc 05 §4.3: an unknown room is a structured error naming the known
        rooms. `is_error` has to follow, or the adapters mark a rejection as a
        success and the model has one fewer signal that it must self-correct."""
        context = seeded_context()
        outcome = dispatch(
            call("add_landmark", room="kitchen", description="fridge"), context
        )
        assert outcome.is_error is True
        assert outcome.payload["error"] == "invalid_args"
        assert "living_room" in outcome.payload["hint"]


# ---------------------------------------------------------------------------
# Structured errors (doc 05 §8) — a raise here would buy a free trial rerun
# ---------------------------------------------------------------------------


class TestStructuredErrorsNeverExceptions:
    def test_an_unknown_tool_name_is_the_documented_error_shape(self):
        outcome = dispatch(call("teleport", x=1), make_context())
        assert outcome.is_error is True
        assert outcome.payload["error"] == "unknown_tool"
        assert set(outcome.payload) == {"error", "detail", "hint"}
        # The hint has to be actionable: the model's only route back is the
        # list of names that do exist.
        assert "get_observation" in outcome.payload["hint"]

    def test_the_unknown_tool_helper_lists_every_real_tool(self):
        assert all(name in unknown_tool("nope")["hint"] for name in TOOL_NAMES)

    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_a_missing_required_argument_is_invalid_args(self, name):
        """The single biggest gap `memory.py` cannot close: these are ordinary
        Python methods, so a missing key dies on the `**` splat with a TypeError
        — which doc 05 §8 routes to the INFRA path and reruns the whole trial,
        handing the model a free retry for its own malformed call."""
        required = TOOL_SCHEMAS[TOOL_NAMES.index(name)]["input_schema"]["required"]
        if not required:
            pytest.skip(f"{name} takes no arguments")
        args = dict(VALID_ARGS[name])
        args.pop(required[0])
        outcome = dispatch(ToolCall(id="x", name=name, args=args), seeded_context())
        assert outcome.is_error is True
        assert outcome.payload["error"] == "invalid_args"
        assert required[0] in outcome.payload["detail"]

    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_a_surplus_argument_is_rejected_rather_than_silently_dropped(self, name):
        """Dropping it is the harness guessing that the extra argument was
        surplus rather than a conflation of two tools (doc 05 §8: "the harness
        never guesses intent"), and the model would never learn that half of
        what it asked for was discarded."""
        args = dict(VALID_ARGS[name], distance_m=1.0) if name != "move" else dict(
            VALID_ARGS[name], heading_deg=90.0
        )
        outcome = dispatch(ToolCall(id="x", name=name, args=args), seeded_context())
        assert outcome.is_error is True
        assert outcome.payload["error"] == "invalid_args"

    def test_arguments_that_are_not_an_object_are_invalid_args(self):
        outcome = dispatch(ToolCall(id="x", name="move", args=[1.0]), make_context())
        assert outcome.payload["error"] == "invalid_args"

    def test_unparseable_arguments_json_is_a_model_failure_not_an_infra_one(self):
        """doc 05 §8 puts "unparseable `arguments` JSON (OpenAI)" in its FIRST
        row, with `unknown name` and `args fail validation`. It looks like a wire
        fault, but the model emitted the malformed JSON — so the turn counts and
        the trial does not rerun."""
        outcome = dispatch(
            ToolCall(id="x", name="move", args={}, parse_error="Expecting value"),
            make_context(),
        )
        assert outcome.is_error is True
        assert outcome.payload["error"] == "invalid_args"
        assert "Expecting value" in outcome.payload["detail"]

    @pytest.mark.parametrize("bad", JSON_VALUES, ids=repr)
    @pytest.mark.parametrize(
        "name,field",
        [
            (tool, field)
            # `.get`, so a tool renamed in `tools.py` fails as a red test in
            # TestToolNamesMatchTheFrozenPrompt rather than as a collection
            # error here, which reports the symptom instead of the cause.
            for tool in TOOL_NAMES
            for field in VALID_ARGS.get(tool, {})
        ],
    )
    def test_no_argument_value_can_make_a_tool_raise(self, name, field, bad):
        """Every JSON-reachable value in every argument slot of every tool.

        An escaping exception is doc 05 §8's LAST row — "harness exception
        outside the model's control" — whose policy is to rerun the trial WHOLE.
        A malformed argument would therefore buy the model a free retry, the
        selection bias §8 exists to prevent, while §8's own first row says the
        turn still counts.
        """
        args = dict(VALID_ARGS[name])
        args[field] = bad
        outcome = dispatch(ToolCall(id="x", name=name, args=args), seeded_context())
        assert isinstance(outcome.payload, dict)
        # Either it was accepted, or it came back as doc 05 §8's shape — never
        # as an exception, and never as a bare `is_error` with nothing to read.
        if outcome.is_error:
            assert set(outcome.payload) >= {"error", "detail", "hint"}
        # Whatever it is, it has to survive the wire.
        json.loads(outcome.to_block("x", name).text)

    @pytest.mark.parametrize("bad", [None, "north", [], {}, True, float("nan"),
                                     float("inf")], ids=repr)
    @pytest.mark.parametrize(
        "name,field",
        [
            ("turn_to_heading", "heading_deg"),
            ("move", "distance_m"),
            ("send_velocity", "vx"),
            ("send_velocity", "vy"),
            ("send_velocity", "wz"),
            ("send_velocity", "duration_s"),
        ],
    )
    def test_a_malformed_motion_argument_steps_no_physics(self, name, field, bad):
        """Doc 05 §5.1 scopes its type checks to the MEMORY tools; nothing
        covered the motion arguments until T3.2. Two concrete failures this
        pins: `clamp_command("0.2", 0, 0)` raises `TypeError` (str < float), and
        `clamp_command(nan, 0, 0)` silently returns `nan` — because every
        min/max comparison against NaN is False — which then poisons the command
        buffer AND the position estimate, unrecoverably and without a traceback.
        """
        context = seeded_context()
        before = context.integrator.xy
        args = dict(VALID_ARGS[name])
        args[field] = bad
        outcome = dispatch(ToolCall(id="x", name=name, args=args), context)
        assert outcome.is_error is True
        assert outcome.payload["error"] == "invalid_args"
        assert field in outcome.payload["detail"]
        assert context.playback.calls == [], "physics ran on a rejected argument"
        assert context.integrator.xy == before
        assert context.counters.policy_seconds == 0.0

    @pytest.mark.parametrize("bad_distance", [0, 0.0, -0.5, -2.0])
    def test_a_non_positive_move_distance_is_rejected_not_silently_no_opped(
        self, bad_distance
    ):
        """§4's domain is `(0, 1.5]` — OPEN at zero — but §4.2 states a failure
        mode only for the upper bound. `policy_wrapper.move` does
        `max(0.0, min(distance_m, 1.5))`, so a zero or negative argument becomes
        a no-op that still runs a chunk, burns a turn and explains nothing.
        There is also no honest clamp to apply: raising 0 to a positive distance
        invents a command, and clamping -1 to +1 would walk the robot the
        opposite way from what the model meant. T3.2 chose `invalid_args`;
        recorded in doc 05 §4.2 in this commit.
        """
        context = make_context()
        outcome = dispatch(call("move", distance_m=bad_distance), context)
        assert outcome.is_error is True
        assert outcome.payload["error"] == "invalid_args"
        # The hint must route the model to the tool that CAN reverse.
        assert "send_velocity" in outcome.payload["hint"]
        assert context.playback.calls == []

    def test_a_numeric_string_parses_because_the_schema_asked_for_a_number(self):
        """`memory.number_arg`'s rule, shared with the motion tools: `"270"` is
        parsing, not intent-guessing — the result is exactly what the schema
        asked for. Models emit it routinely for number-typed fields."""
        context = make_context()
        outcome = dispatch(call("move", distance_m="0.6"), context)
        assert not outcome.is_error
        assert context.playback.calls[0][1]["distance_m"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Clamping and echo (doc 05 §4.2, §6)
# ---------------------------------------------------------------------------


class TestClampingIsEchoedNotSilent:
    def test_an_out_of_hull_velocity_is_clamped_and_the_change_is_reported(self):
        context = make_context()
        outcome = dispatch(
            call("send_velocity", vx=0.9, vy=-0.5, wz=2.0, duration_s=1.0), context
        )
        executed = outcome.payload["executed"]
        assert executed["vx"] == pytest.approx(VX_RANGE[1])
        assert executed["vy"] == pytest.approx(VY_RANGE[0])
        assert executed["wz"] == pytest.approx(WZ_RANGE[1])
        notes = " ".join(outcome.payload["notes"])
        assert "vx" in notes and "vy" in notes and "wz" in notes
        # The frozen prompt's promise: "if a command returns changed, the
        # changed one is what ran." So the echo must be what was EXECUTED.
        assert context.playback.calls[0][1]["vx"] == 0.9

    def test_a_command_inside_the_hull_carries_no_notes_at_all(self):
        """Notes are a signal, not decoration: a `notes` key on every result
        would train the model to stop reading it."""
        outcome = dispatch(
            call("send_velocity", vx=0.1, vy=0.0, wz=0.1, duration_s=1.0),
            make_context(),
        )
        assert "notes" not in outcome.payload

    @pytest.mark.parametrize(
        "requested,expected", [(5.0, 3.0), (0.05, 0.2), (0.0, 0.2), (-1.0, 0.2)]
    )
    def test_the_duration_is_clamped_to_the_documented_range_with_a_note(
        self, requested, expected
    ):
        """`duration_s in [0.2, 3.0]` is stated in doc 05 §4's schema description
        and doc 02 §6.3 and was implemented NOWHERE — `execute()` accepts any
        duration. A bound the model is told about but the harness does not
        enforce is worse than no bound: a `duration_s: 60` would run 60 s of
        blind motion out of a 240 s stage budget in one call."""
        context = make_context()
        outcome = dispatch(
            call("send_velocity", vx=0.1, vy=0.0, wz=0.0, duration_s=requested),
            context,
        )
        assert outcome.payload["executed"]["duration_s"] == pytest.approx(expected)
        assert context.playback.calls[0][1]["duration_s"] == pytest.approx(expected)
        assert any("duration_s" in note for note in outcome.payload["notes"])

    def test_an_over_long_move_is_clamped_with_a_note_the_wrapper_never_writes(self):
        """doc 05 §4.2: "Argument > 1.5 clamped with a note in the result".
        `policy_wrapper.move` performs the clamp but adds NOTHING to
        `clamp_notes` (which only ever carries velocity-hull notes), so this
        note has to be synthesised here or the doc's promise is unmet."""
        context = make_context()
        outcome = dispatch(call("move", distance_m=3.0), context)
        assert context.playback.calls[0][1]["distance_m"] == MOVE_MAX_DISTANCE_M
        assert any("clamped" in note for note in outcome.payload["notes"])

    def test_the_requested_fields_echo_what_the_model_asked_for_not_the_clamp(self):
        """`requested_distance_m` / `requested_heading_deg` carry the model's RAW
        argument. They used to carry the clamped/wrapped value, so
        `move(distance_m=5.0)` answered `requested_distance_m: 1.5` — a key named
        "requested" telling the model something it never requested, and
        recoverable only by cross-referencing `notes`, the one key doc 05 §4.2
        deliberately makes ABSENT when nothing changed. The repo's own precedent
        is the other way: doc 05 §4.2 cites `mark_exit`'s 15-degree snap, whose
        ack "echoes the raw value, so the snap is visible rather than silent".
        Corrected in doc 05 §4.2 in this commit; what actually ran stays
        readable in `notes` (both numbers) and `status.distance_moved_m`.
        """
        context = make_context()
        move = dispatch(call("move", distance_m=5.0), context)
        assert move.payload["requested_distance_m"] == 5.0
        # The clamp note names BOTH numbers, so nothing is lost.
        note = " ".join(move.payload["notes"])
        assert "5.000" in note and "1.500" in note

        turn = dispatch(call("turn_to_heading", heading_deg=725.0), make_context())
        assert turn.payload["requested_heading_deg"] == 725.0
        assert "725" in " ".join(turn.payload["notes"])

    def test_a_move_within_the_cap_reports_no_clamp(self):
        outcome = dispatch(call("move", distance_m=1.0), make_context())
        assert "notes" not in outcome.payload
        assert outcome.payload["requested_distance_m"] == 1.0

    def test_move_serves_distance_in_004_m_increments_and_rounds_up(self):
        """The granularity NOTHING documented until T3.2 (doc 05 §4.2, this
        commit). `policy_wrapper.move` servos in whole 0.2 s chunks worth 0.04 m
        each and breaks on `travelled >= distance / k`, so a move is never short
        of the request and can exceed it by up to 0.04 m — `move(1.5)` covers
        1.52 m, past the schema's own `(0, 1.5]` domain, and `move(0.05)` covers
        0.08 m (+60 %). Pinned here so the figure cannot change silently: it is
        the finest final-positioning step a model has against doc 06 §5.3's
        0.35 m success radius.
        """
        quantum = MOVE_SPEED_MPS * MACRO_CHUNK_S
        assert quantum == pytest.approx(0.04)
        for requested in (1.5, 0.5, 0.1, 0.05):
            outcome = dispatch(call("move", distance_m=requested), make_context())
            covered = outcome.payload["status"]["distance_moved_m"]
            expected = quantum * math.ceil(
                requested / K_VELOCITY_REALISATION / quantum
            )
            assert covered == pytest.approx(expected, abs=1e-9)
            assert covered >= requested - 1e-9, "a move must never fall short"
            # Overshoot is bounded by ONE quantum above the k-adjusted servo
            # target, not above the raw request. With k < 1 the servo
            # deliberately commands further than requested because the robot
            # under-travels, so `covered - requested` legitimately exceeds one
            # quantum: at k=0.9617, move(1.5) targets 1.5598 m and rounds to
            # 1.56 m, i.e. 0.06 m over the request but 0.0002 m over the target.
            # (This bound read `covered - requested` while k was 1.004, where
            # the two are indistinguishable — the 2026-07-29 v5d recalibration
            # is what separated them.)
            target = requested / K_VELOCITY_REALISATION
            assert covered - target < quantum + 1e-9

    @pytest.mark.parametrize(
        "requested,wrapped", [(725.0, 5.0), (-30.0, 330.0), (360.0, 0.0)]
    )
    def test_an_out_of_domain_heading_is_wrapped_and_echoed(self, requested, wrapped):
        """§4 documents `[0,360)` but §4.2 names no failure mode outside it, and
        `wrap_deg` would wrap SILENTLY. An angle modulo 360 has exactly one
        meaning, so wrapping is arithmetic rather than intent-guessing — but the
        echo follows `mark_exit`'s 15-degree snap precedent (§5.1: "the ack
        echoes the raw value, so the snap is visible rather than silent"), or a
        model that asked for 725 and was sent to 5 could never see its own
        arithmetic slip. Recorded in doc 05 §4.2 in this commit."""
        context = make_context()
        outcome = dispatch(call("turn_to_heading", heading_deg=requested), context)
        assert context.playback.calls[0][1]["heading_deg"] == pytest.approx(wrapped)
        # The RAW value comes back (see the `requested_*` test above); the note
        # is what makes the wrap visible, and it names the wrapped target.
        assert outcome.payload["requested_heading_deg"] == pytest.approx(requested)
        note = next(n for n in outcome.payload["notes"] if "wrapped" in n)
        assert f"{wrapped:g}" in note

    def test_an_in_domain_heading_reports_no_wrap(self):
        outcome = dispatch(call("turn_to_heading", heading_deg=270.0), make_context())
        assert "notes" not in outcome.payload


# ---------------------------------------------------------------------------
# Ground truth (doc 06 §4) — nothing crashes when a benchmark leaks its answers
# ---------------------------------------------------------------------------


def _walk(value):
    """Every leaf in a nested payload."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


#: Every ground-truth number planted in the fakes, in every rendering a leak
#: could plausibly take: raw, and rounded to 1 or 2 decimals. The exact-value
#: walk below cannot see a leak that was rounded on its way out
#: (`round(true_x, 1)` = 7.8, which is 0.03 away from the sentinel and therefore
#: `math.isclose(..., abs_tol=1e-9)`-clean), nor one formatted into prose
#: ("you are at 7.77, -8.88") — the string never contains a banned FIELD name and
#: the number never appears as a JSON leaf at all.
TRUTH_VALUES = (*TRUE_XY, TRUE_POSE[2], TRUE_DISPLACEMENT_M)
LEAKED_TEXT_FORMS = tuple(
    sorted({f"{t:.2f}" for t in TRUTH_VALUES} | {f"{t:.1f}" for t in TRUTH_VALUES})
)

#: The COMPLETE payload key set of every tool: (always present, sometimes
#: present). Pinned per tool rather than for `get_observation` alone, because an
#: unpinned key set is how a leak with a NEW name walks straight through: a
#: `"recent_track": [[7.8, -8.9], ...]` added to `move`'s payload is not in
#: `SCORING_ONLY_FIELDS`, is not exactly equal to any sentinel, and no assertion
#: anywhere else counts `move`'s keys. This table is also doc 05 §4.2's
#: "model-facing key names" freeze, expressed as an assertion.
_STATE_KEYS = {"compass_deg", "position_estimate", "status"}
PAYLOAD_KEYS: dict[str, tuple[set[str], set[str]]] = {
    "get_observation": (_STATE_KEYS, set()),
    "look_around": (_STATE_KEYS, set()),
    "turn_to_heading": (
        _STATE_KEYS
        | {"requested_heading_deg", "heading_error_deg", "timed_out", "policy_seconds"},
        {"notes"},
    ),
    "move": (_STATE_KEYS | {"requested_distance_m", "policy_seconds"}, {"notes"}),
    "send_velocity": (_STATE_KEYS | {"executed", "policy_seconds"}, {"notes"}),
    "update_room": ({"ok", "detail", "rooms"}, set()),
    "add_landmark": ({"ok", "detail", "landmarks"}, set()),
    "mark_exit": ({"ok", "detail", "exits"}, set()),
    "set_current_room": ({"ok", "detail"}, set()),
    "correct_position": ({"ok", "detail", "delta_m"}, set()),
    "update_plan": ({"ok", "detail"}, set()),
    DECLARE_DONE: ({"ok", "stage_ended", "detail"}, set()),
}


class TestNoGroundTruthLeak:
    def test_the_key_table_covers_every_tool(self):
        assert set(PAYLOAD_KEYS) == set(TOOL_NAMES)

    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_the_payload_carries_exactly_the_keys_doc_05_freezes(self, name):
        """doc 05 §4.2 froze the model-facing key names; this is that freeze as
        an assertion, for all 12 tools rather than for `get_observation` alone.
        Its real job is the leak below: a brand-new key is invisible to every
        name-based and value-based check in this class."""
        required, optional = PAYLOAD_KEYS[name]
        outcome = dispatch(call(name, **VALID_ARGS[name]), seeded_context())
        keys = set(outcome.payload)
        assert required <= keys, f"{name} lost {sorted(required - keys)}"
        assert keys <= required | optional, f"{name} grew {sorted(keys - required)}"

    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_no_tool_result_carries_a_scoring_only_field(self, name):
        """`ExecResult` has FOUR ground-truth fields, not the three usually
        listed: `sampled_xy` is built from `true_xy()` at 5 Hz and is the raw
        trajectory. Any `dataclasses.asdict(result)` in `tools.py` leaks all
        four and invalidates the benchmark — silently, because the transcript
        still looks fine and every metric still computes.

        Three nets, because each has a hole the others cover: the field names
        (catches a wholesale `asdict`), the numeric leaves at 0.06 tolerance
        (catches a value copied under a new name, INCLUDING one rounded to 1 dp
        on the way out), and a substring scan of the serialised blob (catches a
        number formatted into prose, where there is no numeric leaf to walk).
        """
        context = seeded_context()
        outcome = dispatch(call(name, **VALID_ARGS[name]), context)
        blob = outcome.to_block("x", name).text
        for field in SCORING_ONLY_FIELDS:
            assert field not in blob, f"{name} leaked the field name {field}"
        leaked = [
            leaf
            for leaf in _walk(outcome.payload)
            if isinstance(leaf, (int, float))
            and not isinstance(leaf, bool)
            # 0.06, not 1e-9: `round(7.77, 1)` is 7.8, a leak of the true
            # position to 10 cm — finer than doc 06 §5.3's 0.35 m success
            # radius, and exact-match blind.
            and any(abs(float(leaf) - truth) < 0.06 for truth in TRUTH_VALUES)
        ]
        assert not leaked, f"{name} leaked a ground-truth value: {leaked}"
        for form in LEAKED_TEXT_FORMS:
            assert form not in blob, f"{name} leaked the formatted value {form!r}"

    def test_the_position_the_model_is_shown_is_the_estimate_not_the_truth(self):
        """The estimate and the truth are deliberately different numbers; the
        gap between them IS the drift doc 06 §5.8 measures. Reporting the true
        pose would make every model look like a perfect navigator."""
        context = make_context()
        outcome = dispatch(call("get_observation"), context)
        estimate = outcome.payload["position_estimate"]
        assert (estimate["x"], estimate["y"]) == START_XY
        assert (estimate["x"], estimate["y"]) != TRUE_XY

    def test_look_around_discards_the_true_forward_vector(self):
        """`HeadCamera.look_around` returns `(bearing, rgb, forward)` where the
        forward vector is derived from the robot's TRUE yaw — a field that leaks
        by being convenient to pass along."""
        outcome = dispatch(call("look_around"), make_context())
        assert "forward" not in outcome.to_block("x", "look_around").text
        for leaf in _walk(outcome.payload):
            assert leaf not in TRUE_XY


# ---------------------------------------------------------------------------
# The wire: ToolOutcome -> ToolResultBlock (doc 05 §7.2, doc 04 §6)
# ---------------------------------------------------------------------------


class TestToolResultBlock:
    """`to_block` is the ONLY path from the dispatcher to a provider, and until
    now nothing inspected anything but its `.text`. Two whole channels — every
    camera frame and the error flag — could be dropped on the floor with the
    suite fully green and the transcript still well-formed."""

    @pytest.mark.parametrize("name", ("get_observation", "look_around"))
    def test_the_frames_survive_the_trip_to_the_block(self, name):
        """Blanking `images` here makes every model in the batch drive blind for
        the whole run: no observation and no panorama would carry a single
        pixel, and the JSON status object would still look perfect."""
        outcome = dispatch(call(name, **VALID_ARGS[name]), make_context())
        block = outcome.to_block("id-1", name)
        assert block.images == outcome.images
        assert len(block.images) == len(outcome.images) > 0
        assert all(image.data_b64 for image in block.images)

    def test_the_error_flag_survives_the_trip_to_the_block(self):
        """`providers/anthropic.py` reads exactly this flag to set the API's
        `tool_result.is_error`. Hard-coding it False delivers every
        invalid_args / unknown_tool result to the model as a SUCCESS, removing
        the one protocol-level signal it has that it must self-correct."""
        context = seeded_context()
        ack = dispatch(call("update_plan", text="go east"), context)
        assert ack.to_block("a", "update_plan").is_error is False

        rejected = dispatch(
            call("add_landmark", room="kitchen", description="fridge"), context
        )
        assert rejected.is_error is True
        assert rejected.to_block("b", "add_landmark").is_error is True

        unknown = dispatch(call("teleport", x=1), context)
        assert unknown.to_block("c", "teleport").is_error is True

    def test_the_id_and_the_name_land_in_their_own_fields(self):
        """Both are plain strings and therefore positionally interchangeable; a
        swap would answer every tool_use block with the wrong `tool_use_id`,
        which is an API error — i.e. doc 05 §8's infra rerun of a trial the model
        actually ran."""
        block = dispatch(call("get_observation"), make_context()).to_block(
            "toolu_abc123", "get_observation"
        )
        assert block.tool_use_id == "toolu_abc123"
        assert block.tool_name == "get_observation"

    def test_a_non_finite_harness_value_raises_instead_of_shipping_bad_json(self):
        """doc 05 §8's table names "physics NaN" as the canonical sim-side fault
        whose policy is an INFRA rerun of the whole trial, and `tools.py`'s
        docstring says physics faults are deliberately not caught. But a NaN does
        not raise on its own: `json.dumps` happily emits the bare token `NaN`,
        which is not JSON, is rejected by neither provider, and would then be
        shown to the model for the rest of the episode. `memory.number_arg`
        guards the values coming FROM the model; this guards the ones the harness
        produces, which is the direction §8 actually assigns to infra."""
        outcome = ToolOutcome(payload={"compass_deg": float("nan")})
        with pytest.raises(ValueError):
            outcome.to_block("x", "get_observation")
        with pytest.raises(ValueError):
            ToolOutcome(payload={"x": float("inf")}).to_block("x", "move")


# ---------------------------------------------------------------------------
# The scoring-side channel (doc 06 §4 / §5.3) — the leak test's mirror image
# ---------------------------------------------------------------------------


class TestScoringSideExecutionRecord:
    """`dispatch` is the only code that ever sees an `ExecResult`. Dropping it
    is as silent as leaking it, and strictly more expensive: doc 06 §4 requires
    `turns[].execution.pose_trace` in every trial JSON, §5.3 pins the SPL path
    integral `p` to those 5 Hz samples, and PLAN T4.1's scorer is specified to
    RAISE on a missing `pose_trace` rather than fall back to per-turn chords.
    Discovered after T4.3 launches, that is 12 paid trials rerun."""

    @pytest.mark.parametrize("name", MOTION_TOOLS)
    def test_every_motion_tool_hands_scoring_the_true_trajectory(self, name):
        outcome = dispatch(call(name, **VALID_ARGS[name]), make_context())
        assert outcome.execution is not None, f"{name} destroyed its ExecResult"
        record = outcome.execution
        assert record["pose_trace"] == [list(TRUE_XY), list(TRUE_XY)]
        assert record["sampled_xy"] == [list(TRUE_XY)]
        assert record["true_pose"] == list(TRUE_POSE)
        assert record["true_displacement_m"] == TRUE_DISPLACEMENT_M
        assert record["policy_seconds_used"] > 0.0
        assert set(record) >= {
            "policy_seconds_used",
            "pose_trace",
            "sampled_xy",
            "true_pose",
            "true_displacement_m",
            "stop_reason",
        }
        # It has to survive being written to the trial JSON (doc 06 §4).
        assert json.loads(json.dumps(record)) == record

    @pytest.mark.parametrize(
        "name", [n for n in TOOL_NAMES if n not in MOTION_TOOLS]
    )
    def test_tools_that_step_no_physics_carry_no_execution_record(self, name):
        outcome = dispatch(call(name, **VALID_ARGS[name]), seeded_context())
        assert outcome.execution is None

    @pytest.mark.parametrize("name", MOTION_TOOLS)
    def test_the_execution_record_is_unreachable_from_the_model_facing_block(
        self, name
    ):
        """The two channels must not meet. `to_block` is the only path to a
        provider; if the record ever rode along, the leak test above would be
        asserting the safety of a payload the model no longer receives alone."""
        outcome = dispatch(call(name, **VALID_ARGS[name]), make_context())
        block = outcome.to_block("x", name)
        assert not hasattr(block, "execution")
        for field in ("pose_trace", "sampled_xy", "true_pose", "true_displacement_m"):
            assert field not in block.text

    def test_the_record_reports_what_the_bump_counter_did(self):
        """`counted_as_bump` is how T3.4's log can reconcile a `bumped: true` the
        model was shown against doc 06 §5.6's counter, which deliberately does
        not count `turn_to_heading`."""
        playback = FakePlayback()
        playback.bumped = True
        context = make_context(playback=playback)
        turn = dispatch(call("turn_to_heading", heading_deg=90.0), context)
        move = dispatch(call("move", distance_m=0.4), context)
        assert turn.execution["bumped"] is True
        assert turn.execution["counted_as_bump"] is False
        assert move.execution["counted_as_bump"] is True
        assert context.bumps == 1


# ---------------------------------------------------------------------------
# After a fall (doc 05 §4.2 / §8) — the robot is no longer where it fell
# ---------------------------------------------------------------------------


class TestAfterAFall:
    """Isaac Lab auto-resets a terminated env INSIDE `env.step()` and teleports
    the robot back to spawn before the call returns.
    `policy_wrapper.execute` guards its own `true_pose` against exactly that;
    `tools.py` inherits none of it for free, and every field it reads live —
    `compass_deg()`, and the camera — comes from the respawned robot."""

    def _fallen(self, *, heading: float = 120.0, spawn: float = 45.0):
        playback = FakePlayback(compass_deg=heading)
        playback.spawn_compass_deg = spawn
        playback.move_falls = True
        return playback

    def test_the_compass_reported_with_the_fall_is_not_the_spawn_heading(self):
        """Without the latch, the payload that reports `fell: true` also reports
        an unexplained 75-degree compass jump — the seed's spawn heading."""
        playback = self._fallen()
        context = make_context(playback=playback)
        outcome = dispatch(call("move", distance_m=0.4), context)
        assert outcome.payload["status"]["fell"] is True
        assert playback.compass_deg() == 45.0, "the fake must model the teleport"
        assert outcome.payload["compass_deg"] == 120.0

    def test_the_breadcrumb_trail_is_not_poisoned_by_the_teleport(self):
        """The breadcrumb is written into memory PERMANENTLY and re-injected in
        every subsequent prompt (doc 05 §5.2), and logged as doc 06 §4's
        `obs.compass_deg`. A spawn heading written there is not recoverable."""
        context = make_context(playback=self._fallen())
        dispatch(call("move", distance_m=0.4), context)
        assert [crumb.heading_deg for crumb in context.memory.breadcrumbs] == [120.0]

    def test_a_later_observation_still_reports_the_latched_heading(self):
        """`fell` is sticky, so the latch has to be too — otherwise the very next
        `get_observation` (which doc 05 §4.1 keeps answerable on purpose) shows
        the spawn heading instead."""
        context = make_context(playback=self._fallen())
        dispatch(call("move", distance_m=0.4), context)
        outcome = dispatch(call("get_observation"), context)
        assert outcome.payload["compass_deg"] == 120.0
        assert outcome.payload["status"]["fell"] is True

    @pytest.mark.parametrize("name", MOTION_TOOLS)
    def test_no_motion_tool_runs_once_the_robot_has_fallen(self, name):
        """doc 05 §4.2 says falls end the trial for all three motion tools but
        named no mechanism, and `loop.py` does not exist yet. Without this guard
        a model that reasonably tries to recover keeps commanding a respawned
        duck, while `pose_trace` (doc 06 §5.3) and the drift metric (§5.8)
        accumulate across a teleport — nothing raising, nothing logged."""
        context = make_context(playback=FakePlayback(fell=True))
        outcome = dispatch(call(name, **VALID_ARGS[name]), context)
        assert outcome.is_error is True
        assert outcome.payload["error"] == "stage_ended"
        assert set(outcome.payload) == {"error", "detail", "hint"}
        assert context.playback.calls == [], "physics ran after the trial ended"
        assert context.counters.policy_seconds == 0.0
        assert outcome.execution is None

    def test_the_perception_and_memory_tools_still_answer(self):
        """doc 05 §4.1 pins that `fell` is read live so "the final observation
        must report it however it is reached, including on a get_observation that
        ran no motion of its own". They step no physics, so the guard does not
        apply to them."""
        context = seeded_context(playback=FakePlayback(fell=True))
        for name in ("get_observation", "look_around", "update_room", DECLARE_DONE):
            outcome = dispatch(call(name, **VALID_ARGS[name]), context)
            assert not outcome.is_error, name
        assert context.playback.calls == []

    def test_the_guard_helper_is_the_documented_shape(self):
        result = trial_over("move")
        assert set(result) == {"error", "detail", "hint"}
        assert result["error"] == "stage_ended"
        assert "move" in result["detail"]


# ---------------------------------------------------------------------------
# Dead reckoning feed (PLAN T3.2 (b), AMENDED — doc 05 §4.2, doc 02 §6.2/§6.3)
# ---------------------------------------------------------------------------


class TestDeadReckoningFeed:
    def test_move_integrates_the_driving_seconds_not_the_settled_ones(self):
        """PLAN T3.2 (b) says to feed `integrate()` the `policy_seconds` ACTUALLY
        RUN. That is right for `send_velocity` and WRONG for `move`:
        `PolicyPlayback.move` appends a trailing 0.2 s ZERO-command settle chunk
        and merges it into `policy_seconds`, while `travelled` (→
        `dead_reckoned_distance_m`) accumulates driving chunks only. Integrating
        0.2 m/s over the merged figure fabricates 0.2 x 0.2 = 0.04 m of forward
        motion per call — up to 1.6 m over a 40-turn stage, injected straight
        into doc 06 §5.8's drift metric: the exact failure (b) exists to
        prevent, inverted. PLAN is corrected in this commit.
        """
        context = make_context(playback=FakePlayback(compass_deg=0.0))
        context.integrator = PositionIntegrator(0.0, 0.0)
        outcome = dispatch(call("move", distance_m=1.0), context)

        # The fake reproduces the real macro: 5 driving chunks (1.0 m at
        # 0.2 m/s = 5.0 s... 25 chunks) plus one settle chunk.
        reported = outcome.payload["status"]["distance_moved_m"]
        assert context.integrator.x == pytest.approx(reported, abs=1e-6)
        # And the settle chunk IS still charged to the motion budget, because it
        # really did step physics. The two numbers differ by design.
        assert context.counters.policy_seconds == pytest.approx(
            reported / MOVE_SPEED_MPS + MACRO_CHUNK_S
        )

    def test_a_bump_shortened_move_integrates_only_what_it_covered(self):
        """The failure (b) was written for: a command cut short by a bump must
        not integrate in full, or the estimate drifts for our reasons rather
        than the robot's."""
        playback = FakePlayback(compass_deg=0.0)
        playback.stop_after_chunks = 2  # bumped after 0.4 s of driving
        playback.bumped = True
        context = make_context(playback=playback)
        context.integrator = PositionIntegrator(0.0, 0.0)

        outcome = dispatch(call("move", distance_m=1.5), context)
        assert outcome.payload["status"]["bumped"] is True
        assert context.integrator.x == pytest.approx(
            MOVE_SPEED_MPS * 2 * MACRO_CHUNK_S, abs=1e-6
        )
        assert context.bumps == 1

    def test_move_integrates_along_the_heading_the_macro_holds(self):
        """`move` holds the heading it started at (T1.3's measured correction:
        the bare policy yaws ~1.8 deg/s). Integrating along anything else would
        put the estimate in a direction the robot was never commanded.

        The fake ENDS ON A DIFFERENT HEADING than it started, which is the only
        thing that makes this test able to fail: with a fake whose compass never
        moves, a `held_heading` re-read after the macro integrates identically to
        one latched before it, and the property the test is named for is untested.
        The real robot's heading at the end of a move is not the heading at the
        start — residual yaw under heading hold, and a bump-stopped move can end
        rotated.
        """
        playback = FakePlayback(compass_deg=90.0)
        playback.move_ends_on_compass = 0.0
        context = make_context(playback=playback)
        context.integrator = PositionIntegrator(0.0, 0.0)
        dispatch(call("move", distance_m=0.4), context)
        # Along 90 deg (+y), the heading held — NOT along the 0 deg it ended on.
        # The integrator feeds on COMMANDED velocity with no k, so the value is
        # the chunk-quantised servo target, not the request:
        #   ceil(0.4 / 0.9617 / 0.04) = 11 chunks x 0.04 m = 0.44 m
        # It read 0.4 while k was 1.004 (ceil(9.96) = 10 chunks); the 2026-07-29
        # v5d recalibration moved it. Derived, not relaxed — the tolerance is
        # still 1e-6, and the axis assertion is untouched.
        expected_y = MOVE_SPEED_MPS * MACRO_CHUNK_S * math.ceil(
            0.4 / K_VELOCITY_REALISATION / (MOVE_SPEED_MPS * MACRO_CHUNK_S)
        )
        assert context.integrator.x == pytest.approx(0.0, abs=1e-9)
        assert context.integrator.y == pytest.approx(expected_y, abs=1e-6)

    def test_a_send_velocity_cut_short_integrates_only_the_seconds_that_ran(self):
        """PLAN T3.2 (b) as CORRECTED, for the tool it is right for. `execute()`
        breaks out of its step loop on termination, so `policy_seconds` can be a
        fraction of the requested duration; integrating the request instead walks
        the estimate on without the robot. Quantified: a 3.0 s command at
        vx=0.222 that falls at 0.5 s integrates 0.111 m correctly and 0.666 m
        wrongly — a 0.555 m harness-manufactured error against doc 06 §5.3's
        0.35 m success radius, injected straight into §5.8's drift metric.
        """
        ran_s = 0.5
        playback = FakePlayback(compass_deg=0.0)
        playback.execute_stop_after_steps = duration_to_steps(ran_s)
        context = make_context(playback=playback)
        context.integrator = PositionIntegrator(0.0, 0.0)

        outcome = dispatch(
            call("send_velocity", vx=0.222, vy=0.0, wz=0.0, duration_s=3.0), context
        )
        assert context.playback.calls[0][1]["duration_s"] == pytest.approx(3.0)
        assert outcome.payload["policy_seconds"] == pytest.approx(ran_s)
        assert context.integrator.x == pytest.approx(0.222 * ran_s, abs=1e-9)
        assert outcome.payload["status"]["distance_moved_m"] == pytest.approx(
            0.222 * ran_s, abs=1e-3
        )

    def test_a_command_that_ends_in_a_fall_still_charges_the_budget(self):
        """The budget, the breadcrumb and the carry-over status are all recorded
        for the command that fell — it really did step physics for those seconds.
        Skipping the recording on a fall would under-report doc 06 §5.4's
        time-to-kitchen and leave the trajectory line one point short of where
        the trial actually ended."""
        playback = FakePlayback(compass_deg=0.0)
        playback.execute_stop_after_steps = duration_to_steps(0.5)
        context = make_context(playback=playback)

        outcome = dispatch(
            call("send_velocity", vx=0.2, vy=0.0, wz=0.0, duration_s=3.0), context
        )
        assert outcome.payload["status"]["fell"] is True
        assert context.counters.policy_seconds == pytest.approx(0.5)
        assert len(context.memory.breadcrumbs) == 1
        assert context.last_distance_moved_m == pytest.approx(0.2 * 0.5, abs=1e-3)

    def test_turning_in_place_moves_the_position_estimate_by_exactly_zero(self):
        """`turn_to_heading` commands vx = vy = 0. Real slip during the turn is
        drift the integrator is SUPPOSED not to see (doc 05 §5.1) — an
        integrator that "helpfully" accounted for it would be measuring our
        model of the robot instead of the model's cognition."""
        context = make_context()
        before = context.integrator.xy
        dispatch(call("turn_to_heading", heading_deg=180.0), context)
        assert context.integrator.xy == before

    def test_send_velocity_integrates_the_arc_not_a_straight_line(self):
        """`send_velocity` is the only tool that can translate and rotate at
        once. Doc 02 §6.3's pseudocode integrates such a command in ONE call at
        ONE heading; for `send_velocity(0.222, 0, 0.5, 3.0)` — 86 deg of sweep
        over 0.67 m — that misplaces the estimate by ~0.45 m, against a 0.35 m
        `find_kitchen` success radius. That error is the harness's arithmetic,
        not the robot's slip. Recorded in doc 02 §6.3 / doc 05 §4.2."""
        vx, wz, duration = 0.222, 0.5, 3.0
        context = make_context(playback=FakePlayback(compass_deg=0.0))
        context.integrator = PositionIntegrator(0.0, 0.0)
        dispatch(
            call("send_velocity", vx=vx, vy=0.0, wz=wz, duration_s=duration), context
        )

        # Independently re-derived here rather than asserted as a magic pair:
        # the commanded heading advances by degrees(wz) * dt each control step.
        expected_x = expected_y = 0.0
        heading = 0.0
        for _ in range(duration_to_steps(duration)):
            expected_x += vx * math.cos(math.radians(heading)) * CONTROL_DT
            expected_y += vx * math.sin(math.radians(heading)) * CONTROL_DT
            heading += math.degrees(wz) * CONTROL_DT
        assert context.integrator.x == pytest.approx(expected_x, abs=1e-12)
        assert context.integrator.y == pytest.approx(expected_y, abs=1e-12)

        # And the point of the exercise: a straight-line integration would land
        # at (0.666, 0.0), which is further from the truth than the whole
        # `find_kitchen` success radius (0.35 m, doc 06 §5.3) — a harness-made
        # error big enough to swamp what doc 06 §5.8 is trying to measure.
        straight_line_error = math.dist(
            (expected_x, expected_y), (vx * duration, 0.0)
        )
        assert straight_line_error > 0.35

    def test_a_pure_translation_integrates_identically_to_the_plain_integrator(self):
        """With wz = 0 the arc feed must reduce, step for step, to
        `PositionIntegrator.integrate` — or `send_velocity` and `move` would
        disagree about the same commanded motion."""
        context = make_context(playback=FakePlayback(compass_deg=30.0))
        context.integrator = PositionIntegrator(0.0, 0.0)
        dispatch(
            call("send_velocity", vx=0.2, vy=0.05, wz=0.0, duration_s=1.0), context
        )
        reference = PositionIntegrator(0.0, 0.0)
        reference.integrate(0.2, 0.05, 30.0, 1.0)
        assert context.integrator.xy == pytest.approx(reference.xy, abs=1e-12)

    def test_the_integrator_runs_the_step_count_the_sim_ran(self):
        """`duration_to_steps` floors at 1 and rounds; a tools layer that
        clamped for `execute()` but forwarded the raw duration to the integrator
        would drift by a step per call, silently (`tests/test_memory.py` records
        this hazard against T3.2 by name)."""
        context = make_context(playback=FakePlayback(compass_deg=0.0))
        context.integrator = PositionIntegrator(0.0, 0.0)
        # 0.05 s is clamped UP to the 0.2 s floor, so 10 steps must run.
        dispatch(
            call("send_velocity", vx=0.1, vy=0.0, wz=0.0, duration_s=0.05), context
        )
        assert context.integrator.x == pytest.approx(
            0.1 * duration_to_steps(0.2) * CONTROL_DT, abs=1e-12
        )

    def test_the_distance_reported_for_send_velocity_is_speed_times_time(self):
        """doc 04 §6.2 defines `distance_moved_m` as the dead-reckoned distance
        actually covered, but the wrapper sets `dead_reckoned_distance_m` only
        in `move` — it is 0.0 on every other result. Reporting that raw would
        tell the model it had not moved."""
        context = make_context()
        outcome = dispatch(
            call("send_velocity", vx=0.1, vy=0.1, wz=0.0, duration_s=1.0), context
        )
        executed = outcome.payload["executed"]
        expected = math.hypot(executed["vx"], executed["vy"]) * 1.0
        assert outcome.payload["status"]["distance_moved_m"] == pytest.approx(
            expected, abs=1e-3
        )


# ---------------------------------------------------------------------------
# Payload shape and budget accounting (doc 04 §6, doc 05 §4.1, doc 06 §5.6)
# ---------------------------------------------------------------------------


class TestObservationPayload:
    def test_get_observation_returns_one_frame_and_the_frozen_field_names(self):
        outcome = dispatch(call("get_observation"), make_context())
        assert len(outcome.images) == 1
        assert outcome.images[0].label is None
        assert set(outcome.payload) == {"compass_deg", "position_estimate", "status"}
        assert set(outcome.payload["position_estimate"]) == {"x", "y", "note"}
        # Derived from doc 04 §6, not transcribed: a hardcoded set here means
        # the doc and the payload can drift apart silently, and adding
        # `status.contact` in T3.5 is exactly the change that would have done it.
        assert set(outcome.payload["status"]) == set(
            doc_04_frozen_payload()["status"]
        )

    def test_the_estimate_note_is_doc_04s_frozen_string_verbatim(self):
        """doc 04 §6's frozen payload carries the note in every observation, and
        the whole batch freezes with whatever it says.

        Compared against the DOC, not grepped for keywords. "drift" +
        "correct_position" both survive a note rewritten to *"drift is
        negligible, correct_position is rarely needed"* — which would spend the
        entire batch actively discouraging the one loop-closure tool the memory
        scaffolding exists to exercise, while the suite stayed green. `tools.py`
        already calls the string "verbatim from doc 04 §6", so the doc is the
        source and this is the assertion that makes the claim true.
        """
        doc = doc_04_frozen_payload()
        assert POSITION_ESTIMATE_NOTE == doc["position_estimate"]["note"]
        outcome = dispatch(call("get_observation"), make_context())
        assert outcome.payload["position_estimate"]["note"] == doc["position_estimate"]["note"]

    def test_the_payload_field_order_is_the_docs(self):
        """`to_block`'s own docstring calls the insertion order load-bearing —
        "stable across turns, so the model is not re-parsing a payload whose
        shape moves under it". Every other assertion in this file compares
        `set(...)`, under which `json.dumps(..., sort_keys=True)` is invisible."""
        doc = doc_04_frozen_payload()
        outcome = dispatch(call("get_observation"), make_context())
        assert list(outcome.payload) == list(doc)
        assert list(outcome.payload["position_estimate"]) == list(doc["position_estimate"])
        assert list(outcome.payload["status"]) == list(doc["status"])
        # And the SERIALISED form keeps it — that string is what reaches the
        # model, and `json.dumps(..., sort_keys=True)` reorders only there. The
        # nested block is the one that can tell: the doc's top level happens to
        # be alphabetical already (compass < position < status), while
        # `position_estimate` is `x, y, note` and sorts to `note, x, y`.
        wire = json.loads(outcome.to_block("x", "get_observation").text)
        assert list(wire) == list(doc)
        assert list(wire["position_estimate"]) == list(doc["position_estimate"]) != sorted(doc["position_estimate"])
        assert list(wire["status"]) == list(doc["status"])

    def test_the_compass_keeps_the_docs_one_decimal_of_precision(self):
        """doc 04 §6's frozen payload shows `"compass_deg": 87.4` — one decimal.
        It is the only heading sensor the model has, and the only assertion on
        the value anywhere else is a whole number (150.0), which survives being
        coarsened to `round(..., -1)`: 10-degree granularity on the compass,
        green suite."""
        doc = doc_04_frozen_payload()
        outcome = dispatch(
            call("get_observation"),
            make_context(playback=FakePlayback(compass_deg=doc["compass_deg"])),
        )
        assert outcome.payload["compass_deg"] == doc["compass_deg"] == 87.4
        # And a finer reading rounds to 1 dp rather than to something coarser.
        finer = dispatch(
            call("get_observation"), make_context(playback=FakePlayback(compass_deg=87.44))
        )
        assert finer.payload["compass_deg"] == 87.4

    def test_look_around_returns_four_labelled_frames_at_the_frozen_bearings(self):
        """doc 04 §6.2 pins the shape: four image blocks, each preceded by a
        one-line label using ABSOLUTE compass bearings, plus the state block
        once. The labels are the only thing telling the model which frame is
        which."""
        context = make_context()
        outcome = dispatch(call("look_around"), context)
        assert len(outcome.images) == 4
        assert [img.label for img in outcome.images] == [
            "view at compass 0°",
            "view at compass 90°",
            "view at compass 180°",
            "view at compass 270°",
        ]
        assert context.camera.look_around_bearings == LOOK_AROUND_BEARINGS_DEG

    def test_every_frame_is_encoded_at_the_frozen_resolution(self):
        """AGENTS.md rule 4: every model sees the same 512x512."""
        from PIL import Image

        outcome = dispatch(call("look_around"), make_context())
        for image in outcome.images:
            decoded = Image.open(io.BytesIO(base64.b64decode(image.data_b64)))
            assert decoded.size == RESOLUTION

    def test_the_panorama_is_four_different_views_not_one_repeated(self):
        """The hazard PLAN T3.2 decision 8 records, asserted rather than
        described. `HeadCamera.look_around` hands back RAW arrays and
        `capture_jpeg`/`capture_b64` cannot encode them: both re-`capture_rgb()`,
        and by the time `tools.py`'s comprehension runs, `look_around` has
        already `aim()`ed the camera back to the robot's own heading — so routing
        the frames through the single-frame path yields FOUR COPIES of the
        forward view, labelled 0/90/180/270. A resolution check cannot see that:
        the wrong frames are perfectly valid 512x512 JPEGs. Every model in the
        batch would get a mislabelled panorama and nothing would look wrong.
        """
        context = make_context()
        outcome = dispatch(call("look_around"), context)
        assert len({image.data_b64 for image in outcome.images}) == 4
        # And it must not have gone through the single-frame path AT ALL.
        assert context.camera.captures == 0

    def test_the_first_observation_of_a_stage_reports_a_zeroed_last_command(self):
        """doc 05 §4.1 says the status fields describe the LAST motion command
        and never says what they hold before one has run. Omitting the keys
        would change the payload's shape between turn 1 and turn 2, which is a
        worse answer than a documented zero."""
        outcome = dispatch(call("get_observation"), make_context())
        assert outcome.payload["status"] == {
            "bumped": False,
            # Empty, not absent: `contact` refines `bumped`, so it has to keep
            # the same shape on turn 1 as on every turn after it.
            "contact": [],
            "fell": False,
            "distance_moved_m": 0.0,
        }

    def test_the_status_block_describes_the_previous_motion_command(self):
        context = make_context()
        dispatch(call("move", distance_m=0.4), context)
        outcome = dispatch(call("get_observation"), context)
        assert outcome.payload["status"]["distance_moved_m"] == pytest.approx(
            0.4, abs=0.05
        )

    def test_contact_is_carried_to_the_next_observation_and_cleared_by_clean_motion(self):
        """The end-to-end trace `status.contact` was added for (doc 04 §6.2):
        `playback` samples the groups AT the bump → `ExecResult.contact_groups`
        → `_record_motion` carries them on the context → the NEXT
        `get_observation` reports them beside `bumped`. And like `bumped`, the
        reading describes the LAST motion command — a clean command clears it,
        so a stale list can never outlive the collision it described."""
        playback = FakePlayback()
        playback.bumped = True
        context = make_context(playback=playback)
        dispatch(call("move", distance_m=0.5), context)
        obs = dispatch(call("get_observation"), context)
        assert obs.payload["status"]["bumped"] is True
        assert obs.payload["status"]["contact"] == ["torso"]

        playback.bumped = False
        dispatch(call("move", distance_m=0.5), context)
        obs = dispatch(call("get_observation"), context)
        assert obs.payload["status"]["bumped"] is False
        assert obs.payload["status"]["contact"] == []

    def test_a_fall_is_reported_live_rather_than_carried(self):
        """`fell` is sticky and ends the TRIAL, so the final observation must
        report it however it is reached — including on a `get_observation` that
        ran no motion of its own (doc 04 §6.2)."""
        context = make_context(playback=FakePlayback(fell=True))
        outcome = dispatch(call("get_observation"), context)
        assert outcome.payload["status"]["fell"] is True

    def test_the_turn_result_names_the_fields_doc_05_promises(self):
        """doc 05 §4.2: "final compass_deg, achieved error, policy-seconds
        used", plus `timed_out: true` on timeout. `policy_wrapper` spells that
        state "timeout" in its internal `stop_reason`; the frozen prompt
        promises the model `timed_out`."""
        playback = FakePlayback(compass_deg=0.0)
        playback.turn_stop_reason = "timeout"
        playback.turn_lands_on = 150.0
        context = make_context(playback=playback)
        outcome = dispatch(call("turn_to_heading", heading_deg=180.0), context)
        assert outcome.payload["timed_out"] is True
        assert outcome.payload["compass_deg"] == pytest.approx(150.0)
        assert outcome.payload["heading_error_deg"] == pytest.approx(30.0)
        assert outcome.payload["policy_seconds"] > 0.0

    def test_the_stale_macro_duration_is_never_reported_as_the_time_spent(self):
        """`_merge()` returns the FIRST chunk's `ExecResult` mutated in place: it
        accumulates `steps` and `policy_seconds` but never touches `duration_s`
        or `commanded`, so a macro result claims `duration_s == 0.2` however
        long it ran. Reporting that would tell the model every move took a fifth
        of a second."""
        context = make_context()
        outcome = dispatch(call("move", distance_m=1.0), context)
        assert outcome.payload["policy_seconds"] > MACRO_CHUNK_S


class TestBudgetAndCounters:
    def test_motion_charges_policy_seconds_to_the_stage_budget(self):
        context = make_context()
        dispatch(call("move", distance_m=0.6), context)
        first = context.counters.policy_seconds
        assert first > 0.0
        dispatch(
            call("send_velocity", vx=0.1, vy=0.0, wz=0.0, duration_s=2.0), context
        )
        assert context.counters.policy_seconds == pytest.approx(first + 2.0)

    def test_every_motion_command_leaves_one_breadcrumb(self):
        """`add_breadcrumb` is the harness's ONE autonomous write into memory
        (doc 05 §5.1) and it records the ESTIMATE plus the compass — never the
        true pose."""
        context = make_context()
        dispatch(call("move", distance_m=0.4), context)
        dispatch(call("turn_to_heading", heading_deg=180.0), context)
        dispatch(
            call("send_velocity", vx=0.1, vy=0.0, wz=0.0, duration_s=0.5), context
        )
        assert len(context.memory.breadcrumbs) == 3
        for crumb in context.memory.breadcrumbs:
            assert (crumb.x, crumb.y) != TRUE_XY

    def test_a_bumped_send_velocity_counts_even_though_it_did_not_stop(self):
        """doc 06 §5.6 counts `move` auto-stops AND `send_velocity` collision
        reports, one per command. Counting only the auto-stops would make the
        raw escape hatch a free way to bounce off walls."""
        playback = FakePlayback()
        playback.bumped = True
        context = make_context(playback=playback)
        dispatch(
            call("send_velocity", vx=0.2, vy=0.0, wz=0.0, duration_s=1.0), context
        )
        assert context.bumps == 1
        assert context.playback.calls[0][1]["stop_on_bump"] is False

    def test_a_bumped_turn_reports_the_contact_but_does_not_score_a_bump(self):
        """doc 06 §5.6 enumerates TWO sources for the published `bumps` metric —
        `move` auto-stops and `send_velocity` collision reports — and
        `turn_to_heading` is not one of them.

        Counting rotations is not a stricter measurement, it is
        behaviour-dependent inflation: `PolicyPlayback._bump_run` is instance
        state that is NOT reset between calls, so after a bump-stopped `move` the
        debounce counter is already at its threshold and the FIRST control step
        of the recovery turn re-flags `bumped`. Bump-then-turn-away is the
        canonical recovery pattern, so a model that turns while still in contact
        would score worse on a README headline metric than one that reverses with
        `send_velocity`, for the same number of real collisions. The flag is
        still reported to the model — it is real information — but doc 06 §5.6's
        counter stays what §5.6 says it is.
        """
        playback = FakePlayback()
        playback.bumped = True
        context = make_context(playback=playback)
        outcome = dispatch(call("turn_to_heading", heading_deg=90.0), context)
        assert outcome.payload["status"]["bumped"] is True
        assert context.bumps == 0

    def test_one_collision_survived_by_turning_away_is_still_one_bump(self):
        """The whole point of the test above, end to end: bump, then two recovery
        turns while still in contact. Counted per doc 06 §5.6 that is ONE."""
        playback = FakePlayback(compass_deg=0.0)
        playback.bumped = True
        playback.stop_after_chunks = 2
        context = make_context(playback=playback)
        dispatch(call("move", distance_m=1.0), context)
        dispatch(call("turn_to_heading", heading_deg=90.0), context)
        dispatch(call("turn_to_heading", heading_deg=180.0), context)
        assert context.bumps == 1

    def test_perception_and_memory_tools_cost_no_motion_budget(self):
        context = seeded_context()
        for name in ("get_observation", "look_around", "update_room", "update_plan"):
            dispatch(call(name, **VALID_ARGS[name]), context)
        assert context.counters.policy_seconds == 0.0
        assert context.bumps == 0

    def test_the_stage_reset_clears_stage_state_and_keeps_the_trial_bump_count(self):
        """doc 05 §3.3 resets turns and policy-seconds at the find_kitchen →
        return_home boundary; doc 06 §5.6 counts bumps "over the trial".

        `reset_for_stage()` exists so T3.4 has one call to make instead of a
        field list to get right: doc 05 §4.1 tells the loop it "owns and resets
        [the ToolContext] with the stage", and the natural reading — rebuild or
        zero the object — silently drops every stage-1 collision from a published
        headline metric with no failure and no traceback.
        """
        playback = FakePlayback()
        playback.bumped = True
        context = make_context(playback=playback, turn=9)
        dispatch(call("move", distance_m=0.4), context)
        assert (context.bumps, context.last_bumped) == (1, True)
        assert context.last_contact_groups == ["torso"]
        assert context.counters.policy_seconds > 0.0

        context.reset_for_stage()
        assert context.bumps == 1, "bumps is trial-scoped (doc 06 §5.6)"
        assert context.turn == 0
        assert context.last_bumped is False
        # T3.5 added `last_contact_groups` AFTER reset_for_stage was written
        # (commit dbb9ff5 vs 87150fe), and the shipped reset left it carried —
        # so stage 2's first observation opened with the contradictory status
        # `bumped: false, contact: ["torso"]`, a contact list from a stage-1
        # bump the same payload no longer reported. It resets with the
        # `bumped` flag it refines (doc 05 §4.1, amended in the same commit).
        assert context.last_contact_groups == []
        assert context.last_distance_moved_m == 0.0
        assert context.counters.turns == 0
        assert context.counters.policy_seconds == 0.0


# ---------------------------------------------------------------------------
# declare_done: the stage signal (doc 05 §3.1, §3.3, §4.4)
# ---------------------------------------------------------------------------


class TestDeclareDoneStageSignal:
    def test_stage_one_carries_the_return_home_objective_verbatim(self):
        """doc 05 §3.3 (4): the new objective is delivered AS the declare_done
        call's tool_result. It is deliberately absent from the system prompt
        (`tests/test_memory.py` asserts that) — a model that knew about the
        return leg in advance would map differently from the design the
        benchmark describes."""
        result = stage_end_result(STAGE_FIND_KITCHEN)
        assert result["detail"] == STAGE2_OBJECTIVE_TOOL_RESULT
        # `ok` was asserted nowhere: a `declare_done` answering `{"ok": false}`
        # alongside the stage-2 objective is a contradictory signal delivered at
        # the single most important moment of the episode.
        assert result["ok"] is True
        assert set(result) == {"ok", "stage_ended", "detail"}

    def test_stage_two_ends_the_trial_without_a_new_objective(self):
        result = stage_end_result(STAGE_RETURN_HOME)
        assert result["detail"] != STAGE2_OBJECTIVE_TOOL_RESULT
        assert result["stage_ended"] == STAGE_RETURN_HOME
        assert result["ok"] is True
        assert set(result) == {"ok", "stage_ended", "detail"}

    def test_a_failed_stage_one_is_not_offered_the_return_leg(self):
        """T3.4's resolution of doc 05 §12 / doc 06 §12: stage 2 runs iff stage
        1 SUCCEEDED, so a `declare_done` outside the 0.35 m target region must
        not receive the return-home objective — otherwise a wrong declare that
        happens to land inside the 0.5 m home disc scores `return_home` a
        success with zero motion, i.e. 25 percentage points of an N=4 SR."""
        result = stage_end_result(STAGE_FIND_KITCHEN, continue_to_return_home=False)
        assert result["detail"] != STAGE2_OBJECTIVE_TOOL_RESULT
        assert result["stage_ended"] == STAGE_FIND_KITCHEN
        assert result["ok"] is True
        assert set(result) == {"ok", "stage_ended", "detail"}

    def test_the_failure_branch_is_byte_identical_to_the_trial_over_text(self):
        """The recorded mitigation for what that resolution costs: the model can
        infer pass/fail from whether a return leg is offered, so the text it does
        get must be outcome-NEUTRAL — it says the trial ended, never that the
        model was wrong. Byte-identical to stage 2's, which carries no verdict at
        all."""
        failed = stage_end_result(STAGE_FIND_KITCHEN, continue_to_return_home=False)
        stage_two = stage_end_result(STAGE_RETURN_HOME)
        assert failed["detail"] == stage_two["detail"] == TRIAL_OVER_DETAIL
        for word in ("fail", "wrong", "missed", "incorrect", "score"):
            assert word not in TRIAL_OVER_DETAIL.lower()

    def test_the_success_branch_is_unchanged_by_the_new_keyword(self):
        assert stage_end_result(STAGE_FIND_KITCHEN) == stage_end_result(
            STAGE_FIND_KITCHEN, continue_to_return_home=True
        )

    def test_declare_done_advances_no_physics_and_spends_no_budget(self):
        context = make_context()
        outcome = dispatch(call(DECLARE_DONE), context)
        assert outcome.payload["stage_ended"] == STAGE_FIND_KITCHEN
        assert context.playback.calls == []
        assert context.counters.policy_seconds == 0.0

    def test_declare_done_follows_the_stage_the_memory_object_carries(self):
        context = make_context()
        context.memory.stage = STAGE_RETURN_HOME
        outcome = dispatch(call(DECLARE_DONE), context)
        assert outcome.payload["stage_ended"] == STAGE_RETURN_HOME

    def test_calls_listed_after_declare_done_get_a_structured_result(self):
        """doc 05 §3.1: they are not executed, but every `tool_use` block in the
        echoed assistant turn must still be ANSWERED (§7.2). An unanswered one
        is an API error — i.e. an infra rerun of a trial the model finished."""
        result = not_executed("move")
        assert set(result) == {"error", "detail", "hint"}
        assert result["error"] == "not_executed"
        assert "declare_done" in result["hint"]


# ---------------------------------------------------------------------------
# Config agreement — the caps precedent (no module reads the YAML at runtime)
# ---------------------------------------------------------------------------


class TestConstantsAgreeWithTheFrozenConfig:
    def test_the_duration_clamp_agrees_with_benchmark_yaml(self):
        """The repo convention is: import the Python constant, and let a test
        pin the YAML agreement (`tests/test_memory.py::TestCapsAgreeWithTheFrozenConfig`).
        The regex is `[0-9.]+`, not `\\d+`, because `0.2` would otherwise
        capture `0` and pass."""
        text = BENCHMARK_YAML.read_text(encoding="utf-8")
        matches = re.findall(
            r"send_velocity_duration_s:\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", text
        )
        assert len(matches) == 1
        assert tuple(float(v) for v in matches[0]) == DURATION_RANGE_S

    def test_the_look_around_bearings_agree_with_benchmark_yaml(self):
        text = BENCHMARK_YAML.read_text(encoding="utf-8")
        matches = re.findall(r"look_around_bearings_deg:\s*\[([0-9,\s.]+)\]", text)
        assert len(matches) == 1
        bearings = tuple(float(v) for v in matches[0].split(","))
        assert bearings == LOOK_AROUND_BEARINGS_DEG


class TestFallDiagnosticsNeverReachTheModel:
    """A fall's diagnostics (height, tilt, which term fired) are ground truth
    the model has no sensor for — they ride the SCORING channel only.

    Added after T3.5 logged a fall that could not be audited: the record held
    only the boolean. Fixing that put real ground truth into the pipeline, so
    the leak guard has to cover it too.
    """

    def test_no_tool_payload_may_carry_them(self):
        for name, (required, optional) in PAYLOAD_KEYS.items():
            allowed = required | optional
            for banned in ("fall_diagnostics", "tilt_deg", "height_m", "terms"):
                assert banned not in allowed, f"{name} would expose {banned}"

    def test_a_falling_move_puts_them_on_the_scoring_channel_only(self):
        playback = FakePlayback(compass_deg=0.0)
        playback.execute_stop_after_steps = duration_to_steps(0.2)
        playback.fall_diagnostics = {
            "height_m": 0.061, "tilt_deg": 74.3,
            "terms": {"fell_over": True, "fell_low": False},
        }
        context = make_context(playback=playback)

        outcome = dispatch(call("send_velocity", vx=0.2, vy=0.0, wz=0.0,
                                duration_s=3.0), context)

        assert outcome.execution is not None
        assert outcome.execution.get("fall_diagnostics") == playback.fall_diagnostics, (
            "the audit record must carry WHY the trial ended"
        )
        blob = json.dumps(outcome.payload)
        for banned in ("fall_diagnostics", "tilt_deg", "height_m", "74.3", "0.061"):
            assert banned not in blob, f"{banned} leaked into the model payload"


class TestMoveServoPlanIsGuarded:
    """Pin ``policy_wrapper.move_servo_plan`` — the REAL servo arithmetic.

    Added 2026-07-29. Every pre-existing test of move() distances drives
    ``FakePlayback``, which re-implements the arithmetic, so the real function
    was unguarded: mutating it (ceil -> floor, and removing the ``/ k``) left
    all 547 tests in this file green. These tests assert on the real function,
    so such a mutation now fails.
    """

    def test_target_divides_by_k_so_achieved_distance_matches_the_request(self):
        """The servo must aim FURTHER than the request when k < 1."""
        from duck_embody.sim.policy_wrapper import (
            K_VELOCITY_REALISATION,
            move_servo_plan,
        )

        for requested in (0.1, 0.4, 1.0, 1.5):
            _, target, _ = move_servo_plan(requested)
            assert target == pytest.approx(requested / K_VELOCITY_REALISATION, rel=1e-12)
            # Guards the mutation "drop / k": with k != 1 the target must differ
            # from the request, in the direction that compensates for the policy.
            if K_VELOCITY_REALISATION < 1.0:
                assert target > requested
            elif K_VELOCITY_REALISATION > 1.0:
                assert target < requested

    def test_chunks_round_up_never_down(self):
        """Guards the mutation ceil -> floor: served time never falls short."""
        from duck_embody.sim.policy_wrapper import (
            MACRO_CHUNK_S,
            MACRO_TIME_MARGIN,
            MOVE_SPEED_MPS,
            move_servo_plan,
        )

        for requested in (0.05, 0.1, 0.37, 0.4, 0.83, 1.0, 1.5):
            _, target, n_chunks = move_servo_plan(requested)
            ideal_s = target / MOVE_SPEED_MPS
            needed = ideal_s * MACRO_TIME_MARGIN / MACRO_CHUNK_S
            assert n_chunks >= needed - 1e-12, "a move must never be served short"
            assert n_chunks - needed < 1.0, "and never over-served by a whole chunk"
            assert n_chunks >= 1

    def test_distance_is_clamped_to_the_schema_domain(self):
        from duck_embody.sim.policy_wrapper import MOVE_MAX_DISTANCE_M, move_servo_plan

        assert move_servo_plan(99.0)[0] == pytest.approx(MOVE_MAX_DISTANCE_M)
        assert move_servo_plan(-5.0)[0] == 0.0

    def test_fake_playback_agrees_with_the_real_servo_plan(self):
        """The fake re-implements this arithmetic; pin the two together.

        Without this, ``FakePlayback`` and ``move_servo_plan`` can silently
        diverge and every distance assertion in this file becomes a statement
        about the fake alone.
        """
        from duck_embody.sim.policy_wrapper import (
            MACRO_CHUNK_S,
            MOVE_SPEED_MPS,
            move_servo_plan,
        )

        quantum = MOVE_SPEED_MPS * MACRO_CHUNK_S
        for requested in (1.5, 0.5, 0.1, 0.05):
            covered = dispatch(
                call("move", distance_m=requested), make_context()
            ).payload["status"]["distance_moved_m"]
            _, target, _ = move_servo_plan(requested)
            # The fake quantises the same target up to the same 0.04 m grid.
            assert covered == pytest.approx(
                quantum * math.ceil(target / quantum), abs=1e-9
            )
