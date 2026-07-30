"""Memory unit tests: graph ops, dead-reckoning integration, MapGPT-grammar
rendering, ``correct_position``.

Three of these guard things that fail *silently* rather than loudly:

* **The golden render** is compared against the block extracted from
  ``docs/designs/05-agent-harness.html`` §5.2 itself, not a copy pasted in here.
  A copy would let the doc and the renderer drift apart without either looking
  wrong — and doc 05 §5.2 is the frozen prompt format (doc 06 §2), so drift
  means two batches that are not comparable. This is stricter than PLAN T3.1's
  "whitespace-normalized" requirement: it is byte-exact.
* **The no-k tests** would pass just as happily against an integrator that
  multiplied by ``K_VELOCITY_REALISATION = 1.004``, if they only checked the
  sign and direction of the motion. They check the exact magnitude, and a
  separate test greps the module source, because a 0.4 % correction is invisible
  to the eye and would quietly shrink the drift metric doc 06 §5.8 exists to
  measure.
* **The leak tests** assert the model-facing strings contain no ground truth.
  Nothing crashes when a benchmark leaks its answer key; the numbers just stop
  meaning what they say (doc 05 §1, doc 06 §4).
"""

from __future__ import annotations

import ast
import html
import math
import re
from pathlib import Path

import pytest

from duck_embody.agent.memory import (
    EXIT_DIRECTION_QUANTUM_DEG,
    PLAN_MAX_CHARS,
    POLICY_SECONDS_CAP,
    STAGE_FIND_KITCHEN,
    STAGE_RETURN_HOME,
    STATUS_UNEXPLORED,
    TURN_CAP,
    Correction,
    Counters,
    Crumb,
    Memory,
    PositionIntegrator,
    correct_position,
    exit_status_target,
    quantise_direction,
)
from duck_embody.agent.prompts import (
    BREADCRUMB_WINDOW,
    DERAILMENT_NUDGE,
    EMPTY_SLOT,
    LAYOUT_QA_PREAMBLE,
    LAYOUT_QA_QUESTIONS,
    ROOM_SYNONYMS,
    STAGE1_OBJECTIVE,
    STAGE2_OBJECTIVE_TOOL_RESULT,
    SYSTEM_PROMPT,
    extract_room_mention,
    normalize_room_name,
    render_memory_block,
    render_qa_prompt,
)
from duck_embody.env.apartment_layout import LAYOUT
from duck_embody.sim.policy_wrapper import (
    CONTROL_DT,
    MOVE_SPEED_MPS,
    duration_to_steps,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_05 = REPO_ROOT / "docs" / "designs" / "05-agent-harness.html"
DOC_06 = REPO_ROOT / "docs" / "designs" / "06-benchmark-evaluation.html"

#: MEASURED (T1.3, results/figures/smoke/displacement_report.json): the factor
#: that must NOT appear in dead reckoning. Kept here as a literal so the test
#: still fails if someone deletes the constant and inlines 1.004.
K_VELOCITY_REALISATION = 1.004

#: Modules on the MODEL-FACING dead-reckoning path, where k is banned outright.
#: `policy_wrapper.py` is deliberately absent — it owns the one legitimate use
#: (the `move()` servo target) and gets the targeted test below instead.
NO_K_MODULES = (
    "duck_embody/agent/memory.py",
    "duck_embody/agent/prompts.py",
    "duck_embody/agent/tools.py",  # T3.2; guarded from the day it is written
)

EPS = 1e-9


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Identities of the string nodes that are docstrings.

    They are excluded from the guard below because the modules on this path
    *explain in prose* why k is absent, quoting both its name and its value — a
    guard that could not tell prose from code would have to be weakened until it
    caught nothing, which is how the original memory.py-only version was worded.
    """
    ids: set[int] = set()
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            ids.add(id(body[0].value))
    return ids


def _k_references(tree: ast.AST) -> list[str]:
    """Every way the velocity-realisation factor can reach a subtree.

    Names, attributes, imports, the bare 1.004 literal, and the constant's name
    as a non-docstring *string* — ``getattr(pw, "K_VELOCITY_REALISATION")``
    would otherwise walk straight past a guard that only looked for identifiers.
    """
    doc_ids = _docstring_ids(tree)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "K_VELOCITY_REALISATION":
            hits.append("name")
        elif isinstance(node, ast.Attribute) and node.attr == "K_VELOCITY_REALISATION":
            hits.append("attribute")
        elif isinstance(node, ast.alias) and node.name == "K_VELOCITY_REALISATION":
            hits.append("import")
        elif isinstance(node, ast.Constant) and id(node) not in doc_ids:
            if isinstance(node.value, float) and node.value == pytest.approx(
                K_VELOCITY_REALISATION, abs=1e-9
            ):
                hits.append("inlined literal")
            elif isinstance(node.value, str) and "K_VELOCITY_REALISATION" in node.value:
                hits.append("name as string")
    return hits


# ---------------------------------------------------------------------------
# Helpers: pull the doc's own text out of the HTML, so tests cite the source
# ---------------------------------------------------------------------------


def _doc_text(path: Path) -> str:
    """Tags stripped, entities decoded, whitespace collapsed — for containment
    checks against prose the doc wraps across lines."""
    raw = re.sub(r"<[^>]+>", " ", path.read_text(encoding="utf-8"))
    return " ".join(html.unescape(raw).split())


def golden_memory_block() -> str:
    """The doc 05 §5.2 worked example, extracted from the design doc itself."""
    source = DOC_05.read_text(encoding="utf-8")
    match = re.search(
        r"<pre><code>(== YOUR MAP.*?)</code></pre>", source, flags=re.DOTALL
    )
    assert match, "doc 05 §5.2's worked memory block is no longer in the HTML"
    return html.unescape(match.group(1)).rstrip("\n")


def seed_101_fixture() -> tuple[Memory, Counters]:
    """Rebuild doc 05 §5.2's mid-trial state from model-asserted tool calls only.

    Everything here goes in through the same methods the tools call, so the
    golden test exercises the write path rather than hand-built dataclasses.
    Seed 101 spawns at (0.5, 0.5) heading 90° (apartment_layout spawn_points),
    which is the block's first breadcrumb.
    """
    memory = Memory()
    # anchor_xy mirrors what the tools stamp since 2026-07-30: the integrator
    # estimate at the moment of first assertion. Values follow the breadcrumb
    # story below (mapped the living room at spawn, the hallway on arrival).
    memory.update_room(
        "living_room",
        "sofa along the west wall, armchair opposite, blue rug in center",
        anchor_xy=(0.50, 0.50),
    )
    memory.set_current_room("living_room")
    memory.add_landmark("living_room", "coffee table")
    memory.add_landmark("living_room", "armchair by the south wall")
    memory.mark_exit("living_room", 0, STATUS_UNEXPLORED, anchor_xy=(0.50, 0.50))
    memory.mark_exit("living_room", 90, "leads_to:hallway", anchor_xy=(0.53, 1.11))

    memory.update_room(
        "hallway", "narrow, wooden floor, doorways along the south side",
        anchor_xy=(0.88, 2.56),
    )
    memory.set_current_room("hallway")
    memory.mark_exit("hallway", 0, STATUS_UNEXPLORED, anchor_xy=(0.90, 2.75))
    memory.mark_exit("hallway", 270, STATUS_UNEXPLORED, anchor_xy=(0.90, 2.75))

    for x, y, heading in (
        (0.50, 0.50, 90),
        (0.53, 1.11, 88),
        (0.85, 1.95, 68),
        (0.88, 2.56, 87),
        (0.90, 2.75, 88),
    ):
        memory.add_breadcrumb(x, y, heading)

    memory.update_plan(
        "Kitchen unlikely behind me. Follow the hallway east (exit at 0 deg) and "
        "check the\nnext doorway at 270 deg; if tiled floor or counters appear, "
        "switch to contextual\nsearch for the counter."
    )
    return memory, Counters(turns=12, policy_seconds=63.4)


# ---------------------------------------------------------------------------
# Dead reckoning
# ---------------------------------------------------------------------------


class TestPositionIntegrator:
    def test_starts_at_the_spawn_anchor(self):
        """The only thing the estimate ever takes from truth (doc 05 §5.2)."""
        spawn = LAYOUT["spawn_points"][101]["pos"]
        integrator = PositionIntegrator(*spawn)
        assert integrator.xy == (0.5, 0.5)

    def test_one_step_moves_exactly_v_times_dt_along_the_heading(self):
        integrator = PositionIntegrator(0.0, 0.0)
        integrator.step(MOVE_SPEED_MPS, 0.0, 0.0)
        assert integrator.x == pytest.approx(MOVE_SPEED_MPS * CONTROL_DT, abs=EPS)
        assert integrator.y == pytest.approx(0.0, abs=EPS)

    def test_heading_90_drives_north_not_east(self):
        """Sign of the rotation. Getting it backwards produces an estimate that
        drifts systematically, which would look like model error, not ours."""
        integrator = PositionIntegrator(0.0, 0.0)
        integrator.integrate(MOVE_SPEED_MPS, 0.0, 90.0, 1.0)
        assert integrator.x == pytest.approx(0.0, abs=1e-12)
        assert integrator.y == pytest.approx(MOVE_SPEED_MPS, abs=EPS)

    def test_body_frame_vy_is_left_of_heading(self):
        """+vy at heading 0 (east) must go north, i.e. to the robot's left."""
        integrator = PositionIntegrator(0.0, 0.0)
        integrator.integrate(0.0, 0.1, 0.0, 1.0)
        assert integrator.y == pytest.approx(0.1, abs=EPS)
        assert integrator.x == pytest.approx(0.0, abs=1e-12)

    def test_integrates_commanded_velocity_with_no_k_factor(self):
        """T1.3's pinned policy: commanded velocity, NO k.

        20 s at the commanded 0.2 m/s must read exactly 4.000 m — not the
        4.016 m a k-corrected integrator would report, and not the 4.018 m the
        robot actually achieved. The gap between this number and the truth IS
        the drift metric (doc 06 §5.8); correcting it would launder the
        phenomenon under study away.
        """
        integrator = PositionIntegrator(0.0, 0.0)
        integrator.integrate(MOVE_SPEED_MPS, 0.0, 0.0, 20.0)
        assert integrator.x == pytest.approx(4.0, abs=1e-9)
        assert integrator.x != pytest.approx(4.0 * K_VELOCITY_REALISATION, abs=1e-4)

    @pytest.mark.parametrize("relative_path", NO_K_MODULES)
    def test_no_velocity_realisation_factor_on_the_dead_reckoning_path(
        self, relative_path
    ):
        """A source guard, because 0.4 % is invisible to the eye and survives
        any assertion written with a loose tolerance.

        Parsed rather than grepped: ``memory.py``'s docstring *explains* why k
        is absent and quotes the value, so a text search would either fail on
        the explanation or have to be weakened until it caught nothing. The AST
        sees code only — a reference to the constant, the literal 1.004 inlined
        to dodge the constant, or the constant's name as a string handed to
        ``getattr`` all fail here.

        Scoped to a LIST of files, not one: the spec violation T3.1 actually
        fixed lived in ``policy_wrapper.move()``, which a memory.py-only guard
        cannot see (see the targeted test below), and the next module on this
        path — ``agent/tools.py`` — is the one that will drive the integrator.
        """
        path = REPO_ROOT / relative_path
        if not path.exists():
            pytest.skip(f"{relative_path} not written yet (T3.2)")
        assert not _k_references(ast.parse(path.read_text())), (
            f"{relative_path} touches the velocity-realisation factor; the "
            "number the model is shown must be commanded velocity, uncorrected"
        )

    def test_the_wrappers_reported_distance_is_not_k_corrected_either(self):
        """The OTHER half of the same pin — and the half that was actually
        violated (PLAN T3.1: "A LIVE SPEC VIOLATION FIXED IN THIS COMMIT").

        `policy_wrapper.move()` used to set
        `dead_reckoned_distance_m = travelled * K_VELOCITY_REALISATION`, so the
        one motion number the model is told was quietly moved 0.4 % toward the
        true displacement, shrinking the drift doc 06 §5.8 exists to measure and
        putting the two dead-reckoning paths (this and `PositionIntegrator`)
        into silent disagreement. A blanket ban is impossible here — the `move`
        servo target legitimately divides by k — so the rule is targeted:
        **every assignment to `dead_reckoned_distance_m` must be k-free, and the
        servo target must still use k.** Both directions, so neither the
        violation nor a "fix" that deletes the legitimate use passes.
        """
        module = ast.parse(
            (REPO_ROOT / "duck_embody" / "sim" / "policy_wrapper.py").read_text()
        )

        reported = [
            node
            for node in ast.walk(module)
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, (ast.Attribute, ast.Name))
            and (
                target.attr if isinstance(target, ast.Attribute) else target.id
            ) == "dead_reckoned_distance_m"
        ]
        assert reported, "dead_reckoned_distance_m is never assigned — guard is blind"
        for node in reported:
            assert node.value is None or not _k_references(node.value), (
                "dead_reckoned_distance_m is k-corrected; the model must be told "
                "the commanded-velocity integral, uncorrected (T1.3's pin)"
            )

        # The servo target moved out of `move` into the pure helper
        # `move_servo_plan` on 2026-07-29, so this guard follows it there. The
        # invariant is unchanged and still enforced: k is divided in exactly
        # once, in exactly one place. (The extraction happened because the
        # arithmetic was duplicated in the test suite's FakePlayback and the
        # real function was consequently unguarded — mutating it left every
        # test green. See TestMoveServoPlanIsGuarded in tests/test_tools.py.)
        servo_fn = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef) and node.name == "move_servo_plan"
        )
        servo = [
            node
            for node in ast.walk(servo_fn)
            if isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and _k_references(node.right)
        ]
        assert len(servo) == 1, (
            "move_servo_plan() must divide by k exactly once (the servo target)"
        )
        assert len(_k_references(servo_fn)) == 1, (
            "k appears in move_servo_plan() somewhere other than the servo "
            "target — the only two legitimate consumers are that target and "
            "wall-clock forecasting (PLAN T1.3)"
        )

        # `move` itself must no longer touch k: it delegates. If a future edit
        # reintroduces a k reference there, the constant has two consumers again.
        move = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef) and node.name == "move"
        )
        assert not _k_references(move), (
            "move() references k directly again — it must delegate to "
            "move_servo_plan so the arithmetic stays in one testable place"
        )

    def test_step_count_matches_the_sim_step_count(self):
        """Same duration -> steps rule as the wrapper, or the estimate would
        disagree with the commanded motion for reasons unrelated to drift."""
        integrator = PositionIntegrator(0.0, 0.0)
        integrator.integrate(1.0, 0.0, 0.0, 0.2)  # 0.2 s at 50 Hz = 10 steps
        assert integrator.x == pytest.approx(10 * CONTROL_DT, abs=EPS)

    def test_a_sub_step_duration_still_integrates_one_step(self):
        """duration_to_steps has a floor of 1; the integrator must not silently
        drop a commanded motion and then wonder where the drift came from."""
        integrator = PositionIntegrator(0.0, 0.0)
        integrator.integrate(1.0, 0.0, 0.0, 0.001)
        assert integrator.x == pytest.approx(CONTROL_DT, abs=EPS)

    @pytest.mark.parametrize(
        "duration", [-5.0, -0.001, 0.0, 0.001, 0.019, 0.02, 0.2, 1.0, 3.0]
    )
    def test_the_estimate_runs_exactly_the_step_count_the_sim_runs(self, duration):
        """Pinned to `duration_to_steps` itself, for every degenerate duration.

        A zero or negative duration integrates ONE step (the function floors at
        1) and so displaces the estimate slightly forward — which is correct
        only because `policy_wrapper.execute()` also runs one step for it, so
        the estimate agrees with what the robot actually did. Nothing tested
        that agreement, and doc 05 §4.2's duration clamp lives in `tools.py`,
        which does not exist yet: a T3.2 that clamps for `execute()` but
        forwards the raw duration to `integrate()` would make the two diverge by
        a step per call, silently.
        """
        integrator = PositionIntegrator(0.0, 0.0)
        integrator.integrate(1.0, 0.0, 0.0, duration)
        assert integrator.x == pytest.approx(
            duration_to_steps(duration) * CONTROL_DT, abs=EPS
        )


class TestCorrectPosition:
    def test_reanchors_and_logs(self):
        memory = Memory()
        integrator = PositionIntegrator(1.9, 0.7)
        ack = correct_position(
            memory, integrator, turn=7, x=0.6, y=0.6,
            reason="recognized blue rug in living_room",
        )
        assert integrator.xy == (0.6, 0.6)
        assert ack["ok"] is True
        assert memory.corrections == [
            Correction(
                turn=7,
                old_xy=(1.9, 0.7),
                new_xy=(0.6, 0.6),
                reason="recognized blue rug in living_room",
                stage=STAGE_FIND_KITCHEN,
            )
        ]

    def test_each_correction_records_the_stage_it_happened_in(self):
        """doc 06 §5.8 reports drift per stage. `Correction.turn` is
        stage-LOCAL (doc 05 §3.3 resets the counters while keeping this Memory),
        so without the stage field a stage-2 correction on turn 12 is
        byte-identical to a stage-1 one and the series cannot be split after the
        batch — when nothing else can recover the boundary."""
        memory = Memory()
        integrator = PositionIntegrator(0.0, 0.0)
        correct_position(memory, integrator, turn=7, x=1.0, y=1.0, reason="out")
        memory.stage = STAGE_RETURN_HOME  # the loop's one line at the transition
        correct_position(memory, integrator, turn=7, x=2.0, y=2.0, reason="back")
        assert [c.stage for c in memory.corrections] == [
            STAGE_FIND_KITCHEN,
            STAGE_RETURN_HOME,
        ]
        assert [c.turn for c in memory.corrections] == [7, 7]

    @pytest.mark.parametrize("bad", [None, "abc", [0.6], {"x": 1}, float("nan"),
                                     float("inf"), True])
    @pytest.mark.parametrize("field", ["x", "y"])
    def test_a_malformed_anchor_is_a_structured_error_and_moves_nothing(
        self, field, bad
    ):
        """Both coordinates are validated BEFORE either is written.

        The bug this pins: `integrator.correct()` set `self.x = float(x)` and
        only then `self.y = float(y)`, so a malformed `y` left the estimate
        half re-anchored — x from the new anchor, y still drifted, a coordinate
        frame that never existed — and the exception escaped BEFORE the
        `Correction` was appended, so the drift audit lost the event entirely.
        Worse, doc 05 §8 routes an uncaught harness exception to the infra path,
        which reruns the WHOLE trial: a malformed argument would have bought the
        model a free retry, the exact selection bias §8 exists to prevent.
        """
        memory = Memory()
        integrator = PositionIntegrator(1.9, 0.7)
        args = {"x": 0.6, "y": 0.6, field: bad}
        result = correct_position(memory, integrator, turn=7, reason="r", **args)
        assert result["error"] == "invalid_args"
        assert "metres" in result["hint"]
        assert integrator.xy == (1.9, 0.7), "the estimate moved on a rejected call"
        assert memory.corrections == []

    def test_a_malformed_reason_is_rejected_before_the_estimate_moves(self):
        memory = Memory()
        integrator = PositionIntegrator(1.9, 0.7)
        result = correct_position(memory, integrator, turn=7, x=0.6, y=0.6, reason=None)
        assert result["error"] == "invalid_args"
        assert integrator.xy == (1.9, 0.7)
        assert memory.corrections == []

    def test_ack_reports_the_old_to_new_delta(self):
        memory = Memory()
        integrator = PositionIntegrator(0.0, 0.0)
        ack = correct_position(memory, integrator, turn=1, x=3.0, y=4.0, reason="r")
        assert ack["delta_m"] == pytest.approx(5.0, abs=1e-6)

    def test_a_wrong_anchor_is_obeyed_unconditionally(self):
        """doc 05 §4.3: 'None rejected — a bad anchor is the model's error to
        make; the log makes it measurable.' Sanity-checking it would need the
        ground truth the model is never given."""
        memory = Memory()
        integrator = PositionIntegrator(0.5, 0.5)
        ack = correct_position(
            memory, integrator, turn=3, x=-99.0, y=1e6, reason="confidently lost"
        )
        assert "error" not in ack
        assert integrator.xy == (-99.0, 1e6)
        assert len(memory.corrections) == 1

    def test_every_call_appends_even_when_it_changes_nothing(self):
        """A no-op correction is still a cognitive act doc 06 §5.8 counts."""
        memory = Memory()
        integrator = PositionIntegrator(1.0, 1.0)
        correct_position(memory, integrator, turn=2, x=1.0, y=1.0, reason="confirm")
        correct_position(memory, integrator, turn=4, x=1.0, y=1.0, reason="confirm")
        assert [c.turn for c in memory.corrections] == [2, 4]

    def test_heading_is_never_reset(self):
        """The compass is absolute, so there is nothing about it to correct
        (doc 05 §4.3). The integrator must not own or touch a heading."""
        integrator = PositionIntegrator(0.0, 0.0)
        assert not hasattr(integrator, "heading_deg")
        integrator.correct(1.0, 1.0)
        integrator.step(0.2, 0.0, 45.0)
        assert integrator.x > 1.0 and integrator.y > 1.0


# ---------------------------------------------------------------------------
# Model-asserted graph writes (doc 05 §4.3)
# ---------------------------------------------------------------------------


class TestRoomsAndLandmarks:
    def test_update_room_upserts_and_keeps_the_place_number(self):
        memory = Memory()
        memory.update_room("a", "first")
        memory.update_room("b", "second")
        ack = memory.update_room("a", "revised")
        assert ack["ok"] is True and ack["rooms"] == 2
        assert list(memory.rooms) == ["a", "b"]
        assert memory.rooms["a"].description == "revised"

    def test_add_landmark_to_unknown_room_is_a_structured_error(self):
        memory = Memory()
        memory.update_room("a", "first")
        result = memory.add_landmark("nope", "a chair")
        assert result["error"] == "invalid_args"
        assert "a" in result["hint"]
        assert "error" not in memory.rooms

    def test_landmarks_append_in_order(self):
        memory = Memory()
        memory.update_room("a", "first")
        memory.add_landmark("a", "coffee table")
        memory.add_landmark("a", "armchair by the south wall")
        assert memory.rooms["a"].landmarks == [
            "coffee table",
            "armchair by the south wall",
        ]

    def test_set_current_room_requires_the_model_to_have_created_it(self):
        memory = Memory()
        result = memory.set_current_room("ghost")
        assert result["error"] == "invalid_args"
        assert "update_room" in result["hint"]
        assert memory.current_room is None

    def test_trajectory_records_assertions_and_collapses_repeats(self):
        memory = Memory()
        for name in ("a", "b"):
            memory.update_room(name, name)
        memory.set_current_room("a")
        memory.set_current_room("a")
        memory.set_current_room("b")
        memory.set_current_room("a")
        assert memory.room_sequence == ["a", "b", "a"]

    def test_update_plan_replaces_verbatim(self):
        memory = Memory()
        memory.update_plan("  first plan\nsecond line  ")
        assert memory.plan == "  first plan\nsecond line  "
        memory.update_plan("replaced")
        assert memory.plan == "replaced"


class TestExitStatus:
    def test_records_and_keys_on_the_room_and_snapped_direction(self):
        memory = Memory()
        memory.update_room("a", "a")
        memory.mark_exit("a", 272, STATUS_UNEXPLORED)
        assert len(memory.exits) == 1
        assert memory.exits[0].direction_deg == 270.0

    def test_the_ack_echoes_the_snap_so_it_is_never_silent(self):
        """doc 05 §5.1's deviation note: the exit stores the SNAPPED direction
        and "the ack echoes the raw value, so the snap is visible rather than
        silent — the way `clamp_command` echoes clamps". Without this assertion
        the echo can be deleted and every test still passes, leaving a model
        that marked 272 deg to find 270 deg in its map with no explanation and
        no way to predict which exit a later `mark_exit(268)` addresses."""
        memory = Memory()
        memory.update_room("a", "a")
        ack = memory.mark_exit("a", 272, STATUS_UNEXPLORED)
        assert "snapped from 272 deg" in ack["detail"]
        assert "270 deg" in ack["detail"]
        on_grid = memory.mark_exit("a", 90, STATUS_UNEXPLORED)
        assert "snapped" not in on_grid["detail"]

    @pytest.mark.parametrize(
        "raw", ["90", " 90 ", 90, 90.0, -270.0], ids=repr
    )
    def test_a_numeric_direction_is_parsed_however_the_model_typed_it(self, raw):
        """`"90"` against a `{"type": "number"}` schema is a routine LLM tool-call
        malformation. Parsing it is not guessing intent (doc 05 §8) — the result
        is exactly the value the schema asked for — and it must never raise."""
        memory = Memory()
        memory.update_room("a", "a")
        assert memory.mark_exit("a", raw, STATUS_UNEXPLORED)["ok"] is True
        assert memory.exits[0].direction_deg == 90.0

    @pytest.mark.parametrize(
        "raw", [None, "north", "", [90], {"deg": 90}, float("nan"),
                float("inf"), True],
        ids=repr,
    )
    def test_a_non_numeric_direction_is_a_structured_error_not_an_exception(
        self, raw
    ):
        """The asymmetry that gave this away: `mark_exit("a", "90", "bogus")`
        already returned a clean `invalid_args` for the status, while
        `mark_exit("a", "90", "unexplored")` blew up inside `wrap_deg`'s
        `"90" % 360.0`. `inf % 360` is `nan`, which then failed in
        `math.floor`."""
        memory = Memory()
        memory.update_room("a", "a")
        result = memory.mark_exit("a", raw, STATUS_UNEXPLORED)
        assert result["error"] == "invalid_args"
        assert "direction_deg" in result["hint"]
        assert memory.exits == []

    def test_remarking_the_same_snapped_direction_updates_not_duplicates(self):
        memory = Memory()
        memory.update_room("a", "a")
        memory.mark_exit("a", 272, STATUS_UNEXPLORED)
        ack = memory.mark_exit("a", 268, "leads_to:b")
        assert ack["exits"] == 1
        assert len(memory.exits) == 1
        assert memory.exits[0].status == "leads_to:b"

    def test_different_rooms_keep_separate_exits_at_the_same_bearing(self):
        memory = Memory()
        memory.update_room("a", "a")
        memory.update_room("b", "b")
        memory.mark_exit("a", 90, STATUS_UNEXPLORED)
        memory.mark_exit("b", 90, STATUS_UNEXPLORED)
        assert len(memory.exits) == 2

    def test_all_four_status_transitions_are_reachable(self):
        """There is no terminal state and no delete tool: the downgrade
        leads_to:X -> unexplored is how a model retracts a wrong guess."""
        memory = Memory()
        memory.update_room("a", "a")
        memory.mark_exit("a", 0, STATUS_UNEXPLORED)
        assert memory.exits[0].status == STATUS_UNEXPLORED
        memory.mark_exit("a", 0, "leads_to:x")
        assert memory.exits[0].status == "leads_to:x"
        memory.mark_exit("a", 0, "leads_to:y")
        assert memory.exits[0].status == "leads_to:y"
        memory.mark_exit("a", 0, STATUS_UNEXPLORED)
        assert memory.exits[0].status == STATUS_UNEXPLORED
        memory.mark_exit("a", 15, "leads_to:z")
        assert memory.exits[1].status == "leads_to:z"

    @pytest.mark.parametrize(
        "status", ["explored", "leads to: b", "leads_to:", "leads_to:   ", "", "LEADS_TO:b"]
    )
    def test_malformed_status_is_a_structured_error_naming_both_forms(self, status):
        memory = Memory()
        memory.update_room("a", "a")
        result = memory.mark_exit("a", 0, status)
        assert result["error"] == "invalid_args"
        assert "unexplored" in result["hint"] and "leads_to:<room>" in result["hint"]
        assert memory.exits == []

    def test_unknown_room_is_a_structured_error_naming_the_known_rooms(self):
        memory = Memory()
        memory.update_room("kitchen_i_found", "tiled")
        result = memory.mark_exit("elsewhere", 0, STATUS_UNEXPLORED)
        assert result["error"] == "invalid_args"
        assert "kitchen_i_found" in result["hint"]

    def test_leads_to_target_is_not_validated_against_the_room_set(self):
        """doc 05 §4.3 validates the `room` argument only. A leads_to naming a
        room the model has not created yet is a legitimate forward reference —
        rejecting it would be the harness repairing the model's graph."""
        memory = Memory()
        memory.update_room("a", "a")
        ack = memory.mark_exit("a", 0, "leads_to:not_yet_visited")
        assert ack["ok"] is True
        assert memory.claimed_edges() == [("a", "not_yet_visited")]

    def test_edges_render_in_exit_creation_order_including_upgrades(self):
        """The ordering rule, pinned on the case that discriminates it.

        `b <-> c` is asserted BEFORE `a <-> c`, but a@0 was recorded earlier as
        an unexplored frontier, so the a-edge renders first. doc 05 §5.2 said
        "in the order the model first asserted it"; T3.1's review pass amended
        the doc to exit-creation order (AGENTS.md rule 5) rather than adding an
        assertion counter to `Exit` — both rules are stable and deterministic,
        and the §5.2 example (where no exit is ever upgraded) cannot tell them
        apart, which is why it needs its own test.
        """
        memory = Memory()
        for name in ("a", "b"):
            memory.update_room(name, name)
        memory.mark_exit("a", 0, STATUS_UNEXPLORED)
        memory.mark_exit("b", 90, "leads_to:c")
        memory.mark_exit("a", 0, "leads_to:c")
        assert memory.claimed_edges() == [("a", "c"), ("b", "c")]
        block = render_memory_block(memory, Counters(), (0.0, 0.0), 0.0)
        connections = [ln for ln in block.splitlines() if ln.startswith("Connections")]
        assert connections == ["Connections: a <-> c", "Connections: b <-> c"]

    def test_reciprocal_assertions_collapse_to_one_edge(self):
        memory = Memory()
        memory.update_room("a", "a")
        memory.update_room("b", "b")
        memory.mark_exit("a", 90, "leads_to:b")
        memory.mark_exit("b", 270, "leads_to:a")
        assert memory.claimed_edges() == [("a", "b")]

    @pytest.mark.parametrize(
        "raw,snapped",
        [(0, 0.0), (7.4, 0.0), (7.5, 15.0), (22.5, 30.0), (272, 270.0),
         (359, 0.0), (-15, 345.0), (360, 0.0), (370, 15.0)],
    )
    def test_direction_snapping_is_half_up_and_wraps(self, raw, snapped):
        assert quantise_direction(raw) == snapped

    def test_snapped_directions_are_always_on_the_grid(self):
        for raw in range(-360, 721):
            snapped = quantise_direction(raw)
            assert 0.0 <= snapped < 360.0
            assert math.isclose(snapped % EXIT_DIRECTION_QUANTUM_DEG, 0.0, abs_tol=EPS)

    def test_exit_status_target_only_reads_leads_to(self):
        assert exit_status_target("leads_to:kitchen") == "kitchen"
        assert exit_status_target(STATUS_UNEXPLORED) is None


class TestMalformedArgumentsNeverRaise:
    """doc 05 §5.1: "the memory tools return **structured dicts, never
    exceptions**"; doc 05 §8's first row: args that fail validation come back as
    `{error: "invalid_args", detail, hint}` and the turn still counts.

    Why this is worth a class of its own: an escaping `TypeError` lands instead
    on §8's LAST row — "harness exception outside the model's control" — whose
    policy is to rerun the whole trial. That inverts §8's own agency rule ("if
    the model could have acted differently ... it is a scored model failure")
    and launders a model failure into a free retry, which is precisely the
    selection bias the section exists to prevent. Models routinely emit
    `"direction_deg": "270"` or `null` against a `{"type": "number"}` schema,
    and `json.loads` accepts the `NaN`/`Infinity` literals, so every value below
    is reachable from a real tool call. PLAN T3.2 makes this layer's job
    explicit: `tools.py` is "the wire, not a second implementation".
    """

    #: Everything `json.loads` can hand a tool argument, plus the two
    #: non-finite floats its non-strict mode accepts.
    JSON_VALUES = [
        None, True, False, 0, -1, 2.5, "", "text", "270", [], [1, 2],
        {}, {"a": 1}, float("nan"), float("inf"), float("-inf"),
    ]

    @pytest.mark.parametrize("value", JSON_VALUES, ids=repr)
    def test_no_memory_tool_raises_for_any_json_argument(self, value):
        for call in (
            lambda m: m.update_room(value, "d"),
            lambda m: m.update_room("a", value),
            lambda m: m.add_landmark(value, "d"),
            lambda m: m.add_landmark("a", value),
            lambda m: m.mark_exit(value, 0, STATUS_UNEXPLORED),
            lambda m: m.mark_exit("a", value, STATUS_UNEXPLORED),
            lambda m: m.mark_exit("a", 0, value),
            lambda m: m.set_current_room(value),
            lambda m: m.update_plan(value),
        ):
            memory = Memory()
            memory.update_room("a", "a room the model created")
            result = call(memory)
            assert isinstance(result, dict)
            assert result.get("ok") is True or result["error"] == "invalid_args"
            assert "detail" in result

    @pytest.mark.parametrize("value", JSON_VALUES, ids=repr)
    def test_correct_position_never_raises_either(self, value):
        for args in ({"x": value, "y": 0.6}, {"x": 0.6, "y": value},
                     {"x": 0.6, "y": 0.6, "reason": value}):
            memory = Memory()
            integrator = PositionIntegrator(1.9, 0.7)
            call = {"x": 0.6, "y": 0.6, "reason": "r", **args}
            result = correct_position(memory, integrator, turn=1, **call)
            assert isinstance(result, dict)
            assert result.get("ok") is True or result["error"] == "invalid_args"

    @pytest.mark.parametrize("bad", [None, 1, 1.5, True, ["a"], {"a": 1}], ids=repr)
    def test_a_non_string_room_name_is_rejected_not_coerced(self, bad):
        """`str(None)` would create a room literally named `None`, which the
        model would then have to address by that name for the rest of the
        episode — the harness inventing a map node (doc 05 §1)."""
        memory = Memory()
        assert memory.update_room(bad, "d")["error"] == "invalid_args"
        assert memory.rooms == {}

    def test_an_empty_room_name_is_rejected(self):
        """A name the model cannot refer to later, and one that renders as
        `Place 1:  -- ...` plus a blank `Trajectory:` entry. The only rule
        applied to a room name — no trimming, no case folding (doc 05 §5.1)."""
        memory = Memory()
        result = memory.update_room("   ", "an unnamed place")
        assert result["error"] == "invalid_args"
        assert "name" in result["detail"]
        assert memory.rooms == {}
        assert memory.update_room(" kitchen_ish ", "d")["ok"] is True, (
            "only wholly blank names are rejected; the name is not trimmed"
        )
        assert " kitchen_ish " in memory.rooms

    def test_an_over_long_plan_is_rejected_and_the_old_plan_survives(self):
        """The block is re-injected into every turn of both stages plus the QA
        exchange, so the plan's cost is paid ~85 times. Rejected rather than
        truncated: `update_plan` replaces the plan verbatim (doc 05 §4.3), so
        keeping a prefix would show the model a plan it never wrote."""
        memory = Memory()
        memory.update_plan("the good plan")
        result = memory.update_plan("x" * (PLAN_MAX_CHARS + 1))
        assert result["error"] == "invalid_args"
        assert str(PLAN_MAX_CHARS) in result["hint"]
        assert memory.plan == "the good plan"
        assert memory.update_plan("y" * PLAN_MAX_CHARS)["ok"] is True

    def test_the_rendered_block_stays_bounded_by_the_plan_cap(self):
        """The renderer's docstring claims the block costs nothing per *turn of
        history*; that is true, but it is not constant absolutely. The plan is
        the one unbounded free-text field, and it is the one that is capped."""
        memory = Memory()
        memory.update_room("r", "d")
        memory.update_plan("z" * PLAN_MAX_CHARS)
        block = render_memory_block(memory, Counters(), (0.0, 0.0), 0.0)
        assert len(block) < PLAN_MAX_CHARS + 1000


class TestBreadcrumbs:
    def test_append_is_the_only_autonomous_write(self):
        memory = Memory()
        memory.add_breadcrumb(0.5, 0.5, 90.0)
        memory.add_breadcrumb(0.53, 1.11, 88.0)
        assert memory.breadcrumbs == [Crumb(0.5, 0.5, 90.0), Crumb(0.53, 1.11, 88.0)]
        # ...and it wrote nothing else. No room, no exit, no current_room.
        assert memory.rooms == {} and memory.exits == []
        assert memory.current_room is None and memory.room_sequence == []


# ---------------------------------------------------------------------------
# The rendered block (doc 05 §5.2) — the frozen prompt format
# ---------------------------------------------------------------------------


class TestRenderMemoryBlock:
    def test_the_block_is_never_empty_or_whitespace_only(self):
        """A vacuous memory block is a 400 on turn 1 of EVERY trial.

        The block is sent as a top-level user text block, and the Messages API
        applies a non-empty rule to those. Measured 2026-07-26 against the live
        API (claude-opus-5):

            user content=""                -> 400 "user messages must have
                                              non-empty content"
            user content="   "             -> 400 "text content blocks must
                                              contain non-whitespace text"
            user [{"type":"text","text":""}] -> 400 "text content blocks must
                                                be non-empty"

        (The same empty block nested inside a `tool_result` IS accepted — the
        check does not recurse there — so tool payloads are not at risk. This
        one is, because it rides at the top level.)

        Nothing currently makes the block empty: the headers are unconditional.
        This pins that, because the failure mode is silent until it is total —
        it would 400 the FIRST request of every trial in the batch, after the
        freeze commit, with no partial results to diagnose from.
        """
        cases = {
            "turn 1, nothing known": (Memory(), Counters()),
            "populated mid-trial": seed_101_fixture(),
        }
        for label, (memory, counters) in cases.items():
            block = render_memory_block(memory, counters, (0.0, 0.0), 0.0)
            assert block.strip(), f"{label}: memory block is whitespace-only"
            # Not merely non-empty — it must carry the headers the model is
            # told to rely on, or it is vacuous in substance if not in bytes.
            assert "==" in block, f"{label}: block lost its section headers"

    def test_reproduces_the_design_docs_own_worked_example(self):
        """Byte-exact against doc 05 §5.2, extracted from the HTML at test time.

        Stricter than PLAN T3.1's whitespace-normalized requirement on purpose:
        a normalized comparison would hide the double space before
        '(dead-reckoned', the two-space indents, and the tuple spacing — all of
        which are part of the frozen prompt format (doc 06 §2).
        """
        memory, counters = seed_101_fixture()
        rendered = render_memory_block(
            memory, counters, position_estimate=(0.90, 2.75), compass_deg=88.0
        )
        assert rendered == golden_memory_block()

    def test_matches_the_example_under_whitespace_normalization_too(self):
        """PLAN T3.1's stated acceptance criterion, kept as its own assertion so
        the plan's wording maps onto a named test."""
        memory, counters = seed_101_fixture()
        rendered = render_memory_block(
            memory, counters, position_estimate=(0.90, 2.75), compass_deg=88.0
        )
        assert " ".join(rendered.split()) == " ".join(golden_memory_block().split())

    def test_place_numbering_follows_creation_order_not_the_alphabet(self):
        memory, counters = seed_101_fixture()
        block = render_memory_block(memory, counters, (0.9, 2.75), 88.0)
        assert "Place 1: living_room" in block
        assert "Place 2: hallway" in block

    def test_a_revised_description_does_not_renumber_the_map(self):
        memory, counters = seed_101_fixture()
        memory.update_room("living_room", "revised description")
        block = render_memory_block(memory, counters, (0.9, 2.75), 88.0)
        assert "Place 1: living_room -- revised description" in block

    def test_explored_exits_render_as_connections_not_as_frontiers(self):
        """doc 06 §5.7: unexplored exits are not adjacency assertions, so the
        two lists must not overlap."""
        memory, counters = seed_101_fixture()
        block = render_memory_block(memory, counters, (0.9, 2.75), 88.0)
        frontier_lines = block.split("Unexplored exits:")[1]
        assert "leads_to" not in block
        assert "living_room: exit at 90 deg" not in frontier_lines

    def test_unexplored_exits_sort_by_room_then_bearing(self):
        memory = Memory()
        for name in ("zed", "alpha"):
            memory.update_room(name, name)
        memory.mark_exit("zed", 180, STATUS_UNEXPLORED)
        memory.mark_exit("alpha", 270, STATUS_UNEXPLORED)
        memory.mark_exit("alpha", 90, STATUS_UNEXPLORED)
        block = render_memory_block(memory, Counters(), (0.0, 0.0), 0.0)
        bullets = [ln for ln in block.splitlines() if ln.startswith("  - ")]
        assert bullets == [
            "  - alpha: exit at 90 deg (unexplored)",
            "  - alpha: exit at 270 deg (unexplored)",
            "  - zed: exit at 180 deg (unexplored)",
        ]

    def test_a_room_without_landmarks_omits_the_landmarks_line(self):
        """The one empty case doc 05 §5.2's example pins (Place 2 / hallway)."""
        memory, counters = seed_101_fixture()
        block = render_memory_block(memory, counters, (0.9, 2.75), 88.0)
        hallway_block = block.split("Place 2: hallway")[1].split("\n")[1]
        assert not hallway_block.startswith("  landmarks:")

    def test_turn_one_block_has_every_section_and_no_dangling_lines(self):
        """doc 05 §5.2's example is mid-trial; this is the shape T3.1 chose for
        an empty memory, recorded in §5.2 in the same commit."""
        block = render_memory_block(Memory(), Counters(), (0.5, 0.5), 90.0)
        lines = block.splitlines()
        assert lines[0].startswith("== YOUR MAP")
        assert lines[1] == EMPTY_SLOT
        assert "Connections:" not in block
        assert "Trajectory:" not in block
        assert "Unexplored exits:" not in block
        assert f"Current room (your assertion): {EMPTY_SLOT}" in block
        assert f"Breadcrumbs (last {BREADCRUMB_WINDOW}): {EMPTY_SLOT}" in block
        assert lines[-1] == EMPTY_SLOT  # the plan
        assert "Position estimate: x=0.50, y=0.50" in block

    def test_only_the_last_five_breadcrumbs_render(self):
        memory = Memory()
        for i in range(9):
            memory.add_breadcrumb(float(i), 0.0, 0.0)
        block = render_memory_block(memory, Counters(), (8.0, 0.0), 0.0)
        line = [ln for ln in block.splitlines() if ln.startswith("Breadcrumbs")][0]
        assert line == (
            "Breadcrumbs (last 5): (4.00,0.00,0) (5.00,0.00,0) (6.00,0.00,0) "
            "(7.00,0.00,0) (8.00,0.00,0)"
        )

    def test_the_plan_is_rendered_verbatim_including_its_line_breaks(self):
        memory = Memory()
        memory.update_plan("line one\n  line two indented")
        block = render_memory_block(memory, Counters(), (0.0, 0.0), 0.0)
        assert block.endswith("line one\n  line two indented")

    def test_budget_line_uses_the_frozen_caps(self):
        block = render_memory_block(
            Memory(), Counters(turns=12, policy_seconds=63.44), (0.0, 0.0), 0.0
        )
        assert "Budget: turns 12/40, policy-seconds 63.4/240" in block

    @pytest.mark.parametrize(
        "heading,expected",
        [(88.0, "88"), (359.7, "0"), (360.0, "0"), (-1.0, "359"), (88.5, "89"),
         (0.0, "0")],
    )
    def test_headings_render_as_wrapped_integers(self, heading, expected):
        block = render_memory_block(Memory(), Counters(), (0.0, 0.0), heading)
        assert f"Compass heading: {expected} deg" in block

    def test_negative_zero_never_reaches_the_model(self):
        block = render_memory_block(Memory(), Counters(), (-0.0001, -0.0001), 0.0)
        assert "Position estimate: x=0.00, y=0.00" in block

    def test_the_live_sensors_are_rendered_not_the_last_breadcrumb(self):
        """The whole point of doc 05 §5.2's signature deviation, and until now
        untested: every other renderer test passes a position identical to its
        last crumb, so a renderer that read the crumb instead of its arguments
        passed the entire suite.

        `correct_position` re-anchors the integrator WITHOUT appending a crumb
        (§5.1: crumbs follow motion commands), so this is exactly the state in
        which the two disagree — and showing the pre-correction number here
        would show the model a value it had just overwritten. T3.4 wires the
        call site `render_memory_block(memory, counters, integrator.xy,
        sim.compass_deg())`, where a swapped argument is equally invisible.
        """
        memory = Memory()
        integrator = PositionIntegrator(1.90, 0.70)
        memory.add_breadcrumb(1.85, 0.68, 88.0)
        memory.add_breadcrumb(1.90, 0.70, 88.0)
        correct_position(
            memory, integrator, turn=7, x=0.60, y=0.60, reason="recognized the rug"
        )
        block = render_memory_block(memory, Counters(), integrator.xy, 315.0)
        assert "Position estimate: x=0.60, y=0.60" in block
        assert "Compass heading: 315 deg" in block
        assert "(1.85,0.68,88) (1.90,0.70,88)" in block, (
            "the breadcrumbs are the honest record and must NOT be rewritten"
        )

    def test_a_correction_is_visible_in_the_block_that_outlives_its_ack(self):
        """Why the line exists: the crumb series legitimately contains a jump
        that no motion command explains (here 1.3 m backwards inside a 0.5 m
        move). The tool_result that explained it is dropped once its turn ages
        past the K=10 window, after which the discontinuity is indistinguishable
        from a bad integration — noise in the exact metric (doc 06 §5.8) that
        `correct_position` exists to serve. Added by T3.1's review pass and
        recorded in doc 05 §5.2; conditional, so the golden block is unchanged.
        """
        memory = Memory()
        integrator = PositionIntegrator(1.90, 0.70)
        memory.add_breadcrumb(1.85, 0.68, 88.0)
        memory.add_breadcrumb(1.90, 0.70, 88.0)
        correct_position(memory, integrator, turn=7, x=0.60, y=0.60, reason="rug")
        integrator.integrate(MOVE_SPEED_MPS, 0.0, 90.0, 2.5)
        memory.add_breadcrumb(*integrator.xy, 90.0)

        block = render_memory_block(memory, Counters(), integrator.xy, 90.0)
        line = [ln for ln in block.splitlines() if ln.startswith("Re-anchored")]
        assert line == [
            "Re-anchored: 1 time (latest moved the estimate 1.30 m; "
            "breadcrumbs before it are in the old frame)"
        ]
        correct_position(memory, integrator, turn=9, x=0.60, y=1.10, reason="again")
        block = render_memory_block(memory, Counters(), integrator.xy, 90.0)
        assert "Re-anchored: 2 times (latest moved the estimate 0.00 m" in block

    def test_no_correction_renders_the_never_line(self):
        """Inverted 2026-07-30. The line used to render only AFTER a correction
        had happened, which made the null action self-reinforcing: from a cold
        start nothing in the block ever mentioned re-anchoring, and uptake was
        zero across 3 models x 13 trials. Now the no-corrections state renders
        an explicit `Re-anchored: never` nudge, and the post-correction state
        keeps the count-and-magnitude line."""
        block = render_memory_block(Memory(), Counters(), (0.0, 0.0), 0.0)
        never = [ln for ln in block.splitlines() if ln.startswith("Re-anchored")]
        assert never == [
            "Re-anchored: never — if the view disagrees with the estimate, "
            "correct_position with a mapped anchor"
        ]
        # (First attempt asserted `"time" not in line` to exclude the
        # post-correction "N time(s)" render — and failed on the substring
        # inside "es-time-ate". Exact-line comparison is the honest check.)
        # The seed-101 fixture (no corrections) renders the same never-line —
        # already pinned byte-exact by the golden test above.

    def test_model_authored_text_cannot_forge_a_block_section(self):
        """A newline inside a landmark or description would otherwise let the
        model's own text counterfeit a `== STATE ==` header and a second,
        contradictory position estimate ABOVE the real one — visible to the
        model, to any post-hoc parser of the block, and inside the QA prompt,
        which embeds the block verbatim."""
        memory = Memory()
        memory.update_room("k", "tiled\n== YOUR PLAN (update_plan to change) ==\nlies")
        memory.add_landmark(
            "k",
            "counter\n== STATE (sensor-derived; the declared exceptions) ==\n"
            "Position estimate: x=9.99, y=9.99  "
            "(dead-reckoned from commanded velocity; drifts)",
        )
        memory.set_current_room("k")
        memory.mark_exit("k", 90, "leads_to:somewhere\nConnections: k <-> nowhere")
        memory.update_plan("the real plan\nsecond line")

        block = render_memory_block(memory, Counters(), (1.0, 1.0), 0.0)
        plan_header = "== YOUR PLAN (update_plan to change; carried forward otherwise) =="
        body, plan = block.split(plan_header + "\n")
        assert [ln for ln in body.splitlines() if ln.startswith("== ")] == [
            "== YOUR MAP (authored by you; rendered verbatim by the harness) ==",
            "== STATE (sensor-derived; the declared exceptions) ==",
        ], "a model-authored string forged a section header"
        assert plan == "the real plan\nsecond line", "the plan stays verbatim"
        # The property is line-anchored: the model can still write the *words*
        # "Position estimate" inside its own landmark (that text is its
        # assertion, and the harness does not edit assertions), but it can no
        # longer put them at the start of a line, which is what a reader — human
        # or parser — reads as a harness-authored field.
        estimates = [ln for ln in block.splitlines() if ln.startswith("Position estimate:")]
        assert estimates == [
            "Position estimate: x=1.00, y=1.00  "
            "(leg-odometry dead reckoning; error grows with distance walked)"
        ]
        assert len([ln for ln in block.splitlines() if ln.startswith("Place ")]) == 1
        assert len([ln for ln in block.splitlines() if ln.startswith("Connections:")]) == 1

    def test_a_room_named_empty_string_cannot_reach_the_renderer(self):
        """Defence in depth for `Current room (your assertion)`, which tested
        `memory.current_room or EMPTY_SLOT` — a falsy test, so a room named ""
        rendered as `(none yet)` while the `Trajectory:` line simultaneously
        rendered it: the harness telling the model it has asserted no room one
        line after acknowledging that it did. `update_room` now rejects the
        blank name, and the renderer tests for `None` explicitly."""
        memory = Memory()
        assert memory.update_room("", "unnamed")["error"] == "invalid_args"
        memory.current_room = ""  # the state the write path can no longer reach
        block = render_memory_block(memory, Counters(), (0.0, 0.0), 0.0)
        assert "Current room (your assertion): " in block
        assert f"Current room (your assertion): {EMPTY_SLOT}" not in block


# ---------------------------------------------------------------------------
# The frozen prompt + QA artifacts (doc 05 §6, doc 06 §2, §5.9)
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    #: doc 05 §6's six sections, in order, with their bolded titles.
    SECTIONS = [
        "**Embodiment & physics.**",
        "**Tool documentation.**",
        "**Navigation doctrine — a CogNav-style state machine.**",
        "**Frontier scoring.**",
        "**Plan carry-forward.**",
        "**Honesty rules.**",
    ]

    def test_all_six_sections_are_present_in_order(self):
        positions = [SYSTEM_PROMPT.find(title) for title in self.SECTIONS]
        assert all(p >= 0 for p in positions), dict(zip(self.SECTIONS, positions))
        assert positions == sorted(positions)

    @pytest.mark.parametrize(
        "fragment",
        [
            # doc 05 §6.1 — embodiment numbers, all cited in AGENTS.md §5 / doc 03
            "42 cm bipedal robot",
            "camera at ~0.36 m",
            "90° HFOV",
            "doorways are ~0.35 m wide vs your ~0.16 m body",
            "max 0.222 m/s forward",
            "A fall ends the trial",
            # doc 05 §6.2 — the one usage note the doc gives verbatim
            "bundle memory writes with your motion command",
            "turns are your scarcest budget",
            # doc 05 §6.3 — the three CogNav mode names are normative (doc 05 §9)
            "*broad search*",
            "*contextual search*",
            "*verify*",
            # doc 05 §6.4 — the frontier definition
            "boundary between explored and unexplored space",
            "unexplored-exits list",
            # doc 05 §6.5 / §6.6
            "either act consistently with it or call `update_plan`",
            "never invent a room you haven't entered",
            "mark uncertain exits `unexplored` rather than guessing",
        ],
    )
    def test_carries_the_normative_wording_from_doc_05_section_6(self, fragment):
        # Whitespace-collapsed: the prompt is hard-wrapped prose, so a phrase
        # can straddle a newline. The wording is normative; the wrap column is
        # not.
        assert " ".join(fragment.split()) in " ".join(SYSTEM_PROMPT.split())

    def test_states_the_stage_one_objective(self):
        assert STAGE1_OBJECTIVE in SYSTEM_PROMPT
        assert STAGE1_OBJECTIVE == "Find the kitchen and walk to the counter"

    def test_does_not_pre_announce_the_return_home_stage(self):
        """doc 05 §3.3 delivers stage 2 as declare_done's tool_result mid-run. A
        model told in advance would map for a round trip it is not supposed to
        know about, which is a different experiment."""
        assert "return_home" not in SYSTEM_PROMPT
        assert "return to your starting position" not in SYSTEM_PROMPT

    def test_states_the_heading_convention_the_compass_uses(self):
        flat = " ".join(SYSTEM_PROMPT.split())
        assert "counter-clockwise from east" in flat
        assert "0 = east, 90 = north, 180 = west, 270 = south" in flat

    def test_frozen_harness_strings_match_the_docs_verbatim(self):
        doc = _doc_text(DOC_05)
        assert DERAILMENT_NUDGE in doc, "doc 05 §8's fixed nudge text changed"
        assert STAGE2_OBJECTIVE_TOOL_RESULT in doc, "doc 05 §3.3's objective changed"


class TestNoGroundTruthLeak:
    """doc 05 §1: 'Ground-truth position is never given to the model.' doc 06 §4
    requires a unit test that the prompt path carries no true_pose-derived
    value. Nothing crashes when this breaks — the benchmark just stops
    measuring the model's spatial cognition."""

    MODEL_FACING = (SYSTEM_PROMPT, LAYOUT_QA_PREAMBLE)

    def test_the_system_prompt_never_enumerates_the_true_rooms(self):
        """The model must coin its own room names (doc 05 §1, 'real room
        labels'). ROOM_SYNONYMS exists for the scorer and the out-of-benchmark
        survey judge, and must never be shown to the driving model."""
        lowered = SYSTEM_PROMPT.lower()
        for room in LAYOUT["rooms"]:
            if room == "kitchen":
                continue  # the objective names the goal; that is the task
            assert room.replace("_", " ") not in lowered, room
            assert room not in lowered, room

    def test_the_system_prompt_leaks_no_room_count_or_layout_geometry(self):
        for banned in ("4.8", "3.6", "four rooms", "apartment is", "2.55"):
            assert banned not in SYSTEM_PROMPT, banned

    def test_the_rendered_block_contains_only_model_assertions_and_sensors(self):
        """Ground truth for seed 101 is spawn (0.5, 0.5); the block's numbers
        must all trace to the fixture's own assertions, not to LAYOUT."""
        memory, counters = seed_101_fixture()
        block = render_memory_block(memory, counters, (0.90, 2.75), 88.0)
        target_x, target_y = LAYOUT["target"]["point"]
        assert f"{target_x:.2f}" not in block
        assert f"{target_y:.2f}" not in block
        for room in LAYOUT["rooms"]:
            if room in ("living_room", "hallway"):
                continue  # the fixture's model happens to have coined these
            assert room not in block, room
        for banned in ("true_pose", "ground truth", "score", "confidence", "covariance"):
            assert banned not in block.lower(), banned

    def test_no_model_facing_text_mentions_scores_or_distance_to_goal(self):
        for text in self.MODEL_FACING:
            lowered = text.lower()
            for banned in ("your score", "distance to the counter", "you are close"):
                assert banned not in lowered, banned


class TestLayoutQAArtifacts:
    def test_there_are_exactly_five_questions_numbered_one_to_five(self):
        assert [q.number for q in LAYOUT_QA_QUESTIONS] == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize("question", LAYOUT_QA_QUESTIONS, ids=lambda q: str(q.number))
    def test_question_and_rubric_text_are_verbatim_from_doc_06_section_5_9(self, question):
        doc = _doc_text(DOC_06)
        for field_text in (
            question.text,
            question.rubric_1,
            question.rubric_half,
            question.rubric_0,
        ):
            assert " ".join(field_text.split()) in doc, field_text

    def test_the_frozen_questions_are_answerable_from_the_committed_layout(self):
        """doc 06 §9.2's uniqueness precondition, re-asserted here because Q1's
        rubric ('the unique connector room') is meaningless without it."""
        from duck_embody.env.apartment_layout import connecting_rooms

        assert connecting_rooms("bedroom", "kitchen") == ["hallway"]

    def test_the_qa_prompt_shows_the_block_and_all_five_questions(self):
        memory, counters = seed_101_fixture()
        block = render_memory_block(memory, counters, (0.90, 2.75), 88.0)
        prompt = render_qa_prompt(block)
        assert block in prompt
        for question in LAYOUT_QA_QUESTIONS:
            assert f"{question.number}. {question.text}" in prompt

    def test_the_qa_exchange_offers_no_new_perception(self):
        """doc 06 §5.9: the model 'sees only its own final map/memory block — no
        new camera frames, no sim access'."""
        lowered = LAYOUT_QA_PREAMBLE.lower()
        assert "no camera" in lowered and "no tools" in lowered


class TestFrozenSynonymTable:
    def test_every_canonical_value_is_a_real_room(self):
        for synonym, canonical in ROOM_SYNONYMS.items():
            assert canonical in LAYOUT["rooms"], synonym

    def test_every_true_room_name_maps_to_itself(self):
        for room in LAYOUT["rooms"]:
            assert normalize_room_name(room) == room

    def test_kitchenette_is_not_a_synonym(self):
        """doc 06 §9.1 names it as *the* non-synonym near-string that must not
        match. A fixed table that accepted it would make the map-accuracy metric
        stop meaning what it says — which is the exact rationale the table's own
        comment cites."""
        assert normalize_room_name("kitchenette") is None
        assert "kitchenette" not in ROOM_SYNONYMS

    def test_normalization_is_exact_after_synonyms_never_fuzzy(self):
        """doc 06 §5.7: 'no fuzzy embedding matching'."""
        assert normalize_room_name("lounge") == "living_room"
        assert normalize_room_name("Living Room") == "living_room"
        assert normalize_room_name("kitchen area") is None
        assert normalize_room_name("bedrooms") is None

    def test_every_key_is_its_own_cleaned_form_so_none_is_unreachable(self):
        """Both matchers clean their input (lowercase, punctuation → space,
        whitespace collapsed) BEFORE looking it up, so a key that is not already
        in that form matches nothing. A `"living_room"` key sat here doing
        exactly that until T3.1's review pass — the cleaner turns `_` into a
        space, so `"living room"` is the key that was really doing the work.
        The table is a frozen scoring artifact whose comment presents itself as
        the exact matching contract; an entry that matches nothing makes that
        comment false, and invites more of them."""
        for synonym in ROOM_SYNONYMS:
            cleaned = "".join(
                c if c.isalnum() or c.isspace() else " " for c in synonym.lower()
            )
            assert " ".join(cleaned.split()) == synonym, (
                f"{synonym!r} is unreachable: it cleans to "
                f"{' '.join(cleaned.split())!r}"
            )


class TestExtractRoomMention:
    """The frozen scorer that decided T2.3's room-recognition gate (its only
    caller outside this file is `scripts/judge_scene_survey.py`), and which had
    no test anywhere in the repo — flipping first-mention to last-mention left
    the whole suite green."""

    def test_the_first_room_mentioned_wins(self):
        """A judge that answers "This is a kitchen, not a bedroom" is answering
        `kitchen`; scoring the last mention would flip a gate result that
        AGENTS.md treats as evidence."""
        assert extract_room_mention("This is a kitchen, not a bedroom") == "kitchen"
        assert extract_room_mention("A bedroom — definitely not a kitchen") == "bedroom"

    def test_matching_is_whole_word_so_hallway_is_not_hall(self):
        assert extract_room_mention("hallway") == "hallway"
        assert extract_room_mention("It looks like a hallway to me") == "hallway"
        assert extract_room_mention("the hall") == "hallway"
        assert extract_room_mention("a hallwayish space") is None

    def test_the_kitchenette_near_string_still_does_not_match(self):
        """doc 06 §9.1's canonical non-synonym, checked through THIS matcher
        too — `normalize_room_name` rejecting it says nothing about the
        substring search."""
        assert extract_room_mention("there is a kitchenette in the corner") is None

    def test_punctuation_and_case_do_not_matter_but_nothing_is_guessed(self):
        assert extract_room_mention("A LOUNGE.") == "living_room"
        assert extract_room_mention("living-room") == "living_room"
        assert extract_room_mention("some sort of utility space") is None
        assert extract_room_mention("") is None

    def test_a_longer_synonym_beats_the_shorter_one_inside_it(self):
        """Position decides the winner, so this passes for the right reason
        only when the longer match starts no later than the shorter one."""
        assert extract_room_mention("the sitting room") == "living_room"
        assert extract_room_mention("a bed room") == "bedroom"


class TestCapsAgreeWithTheFrozenConfig:
    def test_rendered_caps_match_configs_benchmark_yaml(self):
        """The number the model budgets against must be the number the runner
        enforces (doc 06 §2/§3.2), or a capped trial is unexplainable.

        The numeric pattern is `[0-9.]+`, not `\\d+`: with `\\d+` a config
        reading `policy_seconds: 240.5` captured "240", which compares equal to
        the constant and passes — so the runner would enforce 240.5 while the
        model budgeted against 240, the exact mismatch this test exists to
        catch. `re.search` also takes the FIRST match, so a second `caps:`
        mapping added later would go uncompared; hence the count assertion.
        """
        text = (REPO_ROOT / "configs" / "benchmark.yaml").read_text(encoding="utf-8")
        assert len(re.findall(r"^caps:\s*$", text, flags=re.MULTILINE)) == 1
        turns = re.search(r"^\s+turns:\s*([0-9.]+)", text, flags=re.MULTILINE)
        seconds = re.search(r"^\s+policy_seconds:\s*([0-9.]+)", text, flags=re.MULTILINE)
        assert turns and seconds
        assert float(turns.group(1)) == float(TURN_CAP)
        assert float(seconds.group(1)) == POLICY_SECONDS_CAP


class TestOdometryIsWhatTheEstimateConsumes:
    """The 2026-07-30 redesign: the estimate advances by MEASURED leg odometry.

    Replaces TestWedgedRobotEarnsNoDistance, whose subject —
    ``credited_distance_m``, the contact-time discount — was retired after
    real-physics measurement showed it inert for v4's bouncing contact (force
    above 1 N on only 6.9% of wedged steps; scripts/smoke_odometry.py now owns
    the physics-level assertions). The property these tests keep alive is the
    same one: a robot that did not move must not believe it moved. Under
    odometry that is structural — a wedged call MEASURES ~zero — so the unit
    tests pin the plumbing: merge arithmetic and the integrator feed.
    """

    def _result(self, odom_dxy, policy_seconds=3.0, bumped=False):
        from duck_embody.sim.policy_wrapper import ExecResult
        import math as _math

        return ExecResult(
            commanded=(0.2, 0.0, 0.0), duration_s=policy_seconds, steps=0,
            policy_seconds=policy_seconds, bumped=bumped, fell=False,
            odom_dxy=odom_dxy,
            odom_distance_m=_math.hypot(*odom_dxy),
        )

    def test_apply_delta_moves_the_estimate_by_exactly_the_measurement(self):
        from duck_embody.agent.memory import PositionIntegrator

        integ = PositionIntegrator(1.0, 2.0)
        integ.apply_delta(0.35, -0.1)
        assert integ.xy == pytest.approx((1.35, 1.9))

    def test_a_wedged_call_measures_nothing_so_the_estimate_stays_put(self):
        """The exact call that used to credit 0.60 m for 0.01 m of motion.

        Batch trial fable5_seed101: send_velocity(0.2, 0, 0) held 3.0 s,
        bumped throughout, true displacement 0.01 m. Under commanded-velocity
        reckoning the estimate advanced 0.60 m; 49 such calls put the belief
        26 m outside the apartment. Odometry reports the ~0.01 m that happened.
        """
        from duck_embody.agent.memory import PositionIntegrator

        wedged = self._result((0.0, 0.01), bumped=True)
        old_formula = 0.2 * wedged.policy_seconds
        assert old_formula == pytest.approx(0.60)
        integ = PositionIntegrator(0.0, 0.0)
        integ.apply_delta(*wedged.odom_dxy)
        assert math.dist((0, 0), integ.xy) == pytest.approx(0.01, abs=1e-9)
        assert wedged.odom_distance_m < 0.02, "the wedge must not earn half a metre"

    def test_merge_sums_odometry_vectors_and_path_length(self):
        """Chunk boundaries (macro 0.2 s and recorder 0.04 s) must not drop or
        double the measurement — the same single-merge rule as policy_seconds."""
        from duck_embody.sim.policy_wrapper import merge_exec_results

        total = merge_exec_results(None, self._result((0.1, 0.0), policy_seconds=0.2))
        total = merge_exec_results(total, self._result((0.0, 0.2), policy_seconds=0.2))
        assert total.odom_dxy == pytest.approx((0.1, 0.2))
        # NET displacement, not a sum of magnitudes. The first version summed
        # magnitudes on the reasoning that "a there-and-back drive covered
        # distance even if it ended where it started" — true of a path length,
        # but it made the number depend on how finely the command was sliced.
        # Real physics settled it: under the recorder's 0.04 s slicing a duck
        # vibrating against a sofa reported 0.72 m of travel for 0.09 m of net
        # motion, because random-direction jitter accumulates in a magnitude
        # sum and cancels in a vector sum.
        assert total.odom_distance_m == pytest.approx(math.hypot(0.1, 0.2))
        assert total.policy_seconds == pytest.approx(0.4)

    def test_contact_steps_still_sum_as_diagnostics(self):
        from duck_embody.sim.policy_wrapper import ExecResult, merge_exec_results

        a = ExecResult(commanded=(0.2, 0, 0), duration_s=0.2, steps=10,
                       policy_seconds=0.2, bumped=True, fell=False, contact_steps=5)
        b = ExecResult(commanded=(0.2, 0, 0), duration_s=0.2, steps=10,
                       policy_seconds=0.2, bumped=True, fell=False, contact_steps=7)
        assert merge_exec_results(merge_exec_results(None, a), b).contact_steps == 12




class TestOdometryNoiseIsChunkingInvariant:
    """The reported distance must not depend on whether video is recording.

    `attach_recorder` patches `execute()` so every command is sliced into
    0.04 s pieces (recorder.RECORD_CHUNK_S), and every batch trial records.
    Two separate defects were found here, the second only by real physics:

    1. A per-CALL noise floor accrued ~25x/s under slicing (0.094 m reported
       for a wedged 3 s command vs 0.0013 m unrecorded). Fixed by making the
       floor a RATE.
    2. `merge_exec_results` summed per-piece MAGNITUDES, so a duck vibrating
       against furniture accumulated its jitter as travel: 0.72 m reported for
       0.09 m of net motion on the GPU. Fixed by reporting the NET displacement
       of the summed vector, which is exactly slicing-invariant.

    These tests drive the real `merge_exec_results`, not a model of it.
    """

    @staticmethod
    def _merged(total_dxy, n_slices):
        """Merge `n_slices` equal parts summing to `total_dxy`."""
        from duck_embody.sim.policy_wrapper import ExecResult, merge_exec_results

        acc = None
        for _ in range(n_slices):
            part = ExecResult(
                commanded=(0.2, 0.0, 0.0), duration_s=3.0 / n_slices, steps=2,
                policy_seconds=3.0 / n_slices, bumped=False, fell=False,
                odom_dxy=(total_dxy[0] / n_slices, total_dxy[1] / n_slices),
                odom_distance_m=math.hypot(
                    total_dxy[0] / n_slices, total_dxy[1] / n_slices
                ),
            )
            acc = merge_exec_results(acc, part)
        return acc

    def test_the_report_is_identical_at_any_slicing(self):
        one = self._merged((0.6, 0.2), 1)
        seventyfive = self._merged((0.6, 0.2), 75)
        assert seventyfive.odom_distance_m == pytest.approx(one.odom_distance_m)
        assert seventyfive.odom_dxy == pytest.approx(one.odom_dxy)

    def test_random_direction_jitter_does_not_accumulate_as_travel(self):
        """The measured failure: a wedged duck vibrating in place.

        75 slices of 9.6 mm in alternating directions is 0.72 m of summed
        magnitude and ~0 m of net displacement. The model must be told ~0.
        """
        from duck_embody.sim.policy_wrapper import ExecResult, merge_exec_results

        acc = None
        for i in range(75):
            sign = 1.0 if i % 2 == 0 else -1.0
            part = ExecResult(
                commanded=(0.2, 0.0, 0.0), duration_s=0.04, steps=2,
                policy_seconds=0.04, bumped=True, fell=False,
                odom_dxy=(sign * 0.0096, 0.0), odom_distance_m=0.0096,
            )
            acc = merge_exec_results(acc, part)
        assert acc.odom_distance_m < 0.02, (
            f"jitter accumulated as {acc.odom_distance_m:.3f} m of phantom travel"
        )

    def test_the_floor_is_a_rate_not_a_per_call_constant(self):
        import duck_embody.sim.policy_wrapper as pw

        assert hasattr(pw, "ODOM_NOISE_FLOOR_RATE_MPS")
        assert not hasattr(pw, "ODOM_NOISE_FLOOR_M"), (
            "per-call noise floor reintroduced — it is not chunking-invariant"
        )


class TestMemoryBlockStructureIndependentOfTheGolden:
    """Non-circular checks on the block.

    The doc 05 §5.2 golden is regenerated FROM this renderer whenever the
    renderer legitimately changes, so on its own it cannot catch a bug
    introduced in the same edit — it only pins the block against LATER drift.
    These assertions are written against the intended contract instead, so a
    renderer change has to satisfy something that was not derived from it.
    """

    def _block(self):
        memory, counters = seed_101_fixture()
        return render_memory_block(memory, counters, (0.90, 2.75), 88.0)

    def test_a_used_doorway_exposes_its_anchor_handle(self):
        """The prompt tells the model to re-anchor on passing a doorway; before
        2026-07-30 the anchor vanished at exactly that moment (exits rendered
        anchors only while `unexplored`)."""
        lines = self._block().splitlines()
        doorway = [ln for ln in lines if ln.startswith("  - living_room@90")]
        assert doorway, "the used doorway lost its anchor line"
        assert "[anchor x=0.53, y=1.11]" in doorway[0]

    def test_adjacency_is_asserted_exactly_once(self):
        """doc 06 §5.7 keeps frontier and adjacency separate; the doorway list
        must not restate `leads_to`."""
        block = self._block()
        assert block.count("leads_to") == 0
        assert block.count("Connections: living_room <-> hallway") == 1

    def test_the_estimate_line_claims_no_unmeasured_magnitude(self):
        """The block is frozen into the batch, so any number here is a claim the
        harness makes to the model about its own sensor accuracy."""
        line = next(
            ln for ln in self._block().splitlines()
            if ln.startswith("Position estimate:")
        )
        assert "leg-odometry" in line
        assert "%" not in line, f"unmeasured accuracy claim in: {line}"

    def test_every_rendered_anchor_is_two_decimal_places(self):
        import re as _re

        for value in _re.findall(r"anchor x=([\d.-]+), y=([\d.-]+)", self._block()):
            for v in value:
                assert len(v.split(".")[1]) == 2, f"anchor not 2dp: {v}"
