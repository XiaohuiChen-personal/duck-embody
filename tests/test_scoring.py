"""Scoring unit tests — MUST pass before any batch launch (non-negotiable).

doc 06 §9's locked gate, and AGENTS.md rule 2. Every case enumerated in §9.1 is
implemented here, plus PLAN T4.1's ``pose_trace``-missing case. A scoring bug
found after the batch invalidates paid, hours-long runs, so this file is the
thing that has to be right first.

Three conventions this file keeps deliberately:

* **Fixtures are built from the WRITER's own shapes.** ``merge_executions``,
  ``memory_snapshot``, ``StageResult`` and ``StageScore`` come from
  ``duck_embody.agent.loop`` / ``duck_embody.tasks.find_kitchen``, so a fixture
  cannot describe a log the harness would never write. (T3.5's real trial JSON
  does not exist yet — ``results/raw/`` holds only ``.gitkeep`` — so every
  fixture here is authored; PLAN T4.1's smoke step should be re-run against the
  real JSON once T3.5 lands.)
* **The schema check reads doc 06 §4 out of the HTML at test time**, rather than
  keeping a copy here. A golden test carrying its own copy of the contract is a
  tautology; this one tracks the doc even if §4 is amended concurrently.
* **QA gold answers are checked against the COMMITTED layout**, not §11's
  "representative" one. §9.1 says "a fixture layout", but the layout dict *is*
  the answer key (AGENTS.md §2) and §11's representative dict is exactly what
  produced §11's wrong Q2 route. Synthetic points are used only for the §5.7
  matching cases, which need contrived evidence.
"""

from __future__ import annotations

import ast
import html
import json
import math
import re
from pathlib import Path

import pytest

from duck_embody import scoring
from duck_embody.agent.loop import StageResult, memory_snapshot, merge_executions
from duck_embody.agent.memory import (
    STAGE_FIND_KITCHEN,
    STAGE_RETURN_HOME,
    Counters,
    Memory,
    PositionIntegrator,
    correct_position,
)
from duck_embody.agent.prompts import (
    LAYOUT_QA_QUESTIONS,
    ROOM_SYNONYMS,
    render_memory_block,
    render_qa_prompt,
)
from duck_embody.agent.providers.base import Usage
from duck_embody.agent.tools import MOTION_TOOLS
from duck_embody.env.apartment_layout import (
    LAYOUT,
    bearing_deg,
    compass_8,
    oracle_length,
    room_at,
    room_centroid,
    spawn_pose,
    target_point,
)
from duck_embody.scoring import NA
from duck_embody.tasks.find_kitchen import (
    REASON_DECLARE_DONE,
    REASON_TURN_CAP,
    StageSpec,
    find_kitchen_spec,
    outcome_for,
    return_home_spec,
    score_stage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_06 = REPO_ROOT / "docs" / "designs" / "06-benchmark-evaluation.html"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN_TRIAL = FIXTURES / "trial_seed101_success.json"


# ---------------------------------------------------------------------------
# Fixture construction — the writer's shapes, never hand-typed JSON
# ---------------------------------------------------------------------------


def motion_execution(
    trace,
    *,
    policy_seconds: float,
    heading_deg: float = 90.0,
    bumped: bool = False,
    fell: bool = False,
    stop_reason: str = "",
    counted_as_bump: bool = False,
    distance_moved_m: float | None = None,
) -> dict:
    """The ten-key dict ``tools._record_motion`` returns (doc 06 §4).

    ``pose_trace`` is ``[start_xy, *sampled_xy, end_xy]``, the shape
    ``sim/policy_wrapper.py`` actually produces, so the concatenation the SPL
    path integral relies on is exercised with its real duplicate boundary points.
    """
    points = [tuple(p) for p in trace]
    end = points[-1] if points else (0.0, 0.0)
    straight = math.dist(points[0], points[-1]) if len(points) >= 2 else 0.0
    return {
        "policy_seconds_used": policy_seconds,
        "pose_trace": [list(p) for p in points],
        "sampled_xy": [list(p) for p in points[1:-1]],
        "true_pose": [end[0], end[1], heading_deg],
        "true_displacement_m": straight,
        "distance_moved_m": straight if distance_moved_m is None else distance_moved_m,
        "bumped": bumped,
        "fell": fell,
        "stop_reason": stop_reason,
        "counted_as_bump": counted_as_bump,
    }


class TrialBuilder:
    """Builds a doc 06 §4 trial JSON the way ``EpisodeRunner`` would.

    Memory writes go through :class:`Memory`'s own methods and the snapshot
    through ``loop.memory_snapshot``, so a fixture can only contain state the
    harness could actually have produced.
    """

    def __init__(self, seed: int = 101, model: str = "claude-fable-5") -> None:
        self.seed = seed
        self.spawn, self.spawn_heading = spawn_pose(seed)
        self.memory = Memory()
        self.integrator = PositionIntegrator(*self.spawn)
        self.counters = Counters()
        self.stage = STAGE_FIND_KITCHEN
        self.stage_turns = {STAGE_FIND_KITCHEN: 0, STAGE_RETURN_HOME: 0}
        self.stage_seconds = {STAGE_FIND_KITCHEN: 0.0, STAGE_RETURN_HOME: 0.0}
        self.global_turn = 0
        self.bumps = 0
        self.true_pose = (self.spawn[0], self.spawn[1], float(self.spawn_heading))
        self.turns: list[dict] = []
        self.document = {
            "trial_id": f"{model}_seed{seed}",
            "config": {
                "freeze_commit": "0" * 40,
                "config_hash": "0" * 64,
                "model": model,
                "model_config": "fable5",
                "seed": seed,
                "spawn": {
                    "xy": [self.spawn[0], self.spawn[1]],
                    "heading_deg": float(self.spawn_heading),
                },
            },
            "turns": self.turns,
            "video_path": None,
        }

    # -- memory writes, through the real methods ---------------------------

    def _apply(self, name: str, args: dict) -> None:
        memory = self.memory
        if name == "update_room":
            memory.update_room(args["name"], args.get("description", "seen"))
        elif name == "set_current_room":
            memory.set_current_room(args["name"])
        elif name == "add_landmark":
            memory.add_landmark(args["room"], args.get("description", "thing"))
        elif name == "mark_exit":
            memory.mark_exit(args["room"], args["direction_deg"], args["status"])
        elif name == "correct_position":
            correct_position(
                memory,
                self.integrator,
                self.stage_turns[self.stage],
                args["x"],
                args["y"],
                args.get("reason", "recognised the sofa"),
            )

    # -- one turn ----------------------------------------------------------

    def turn(
        self,
        calls=(),
        *,
        estimate: tuple[float, float] | None = None,
        true_pose: tuple[float, float, float] | None = None,
        end_reason: str | None = None,
        compass_deg: float = 90.0,
        execution: dict | None = None,
    ) -> dict:
        """Append one doc 06 §4 turn record.

        ``calls`` is a list of ``{"name", "args", ["execution"]}``; a motion tool
        must carry an ``execution`` dict, exactly as ``dispatch`` would produce.
        """
        self.global_turn += 1
        self.stage_turns[self.stage] += 1
        self.counters.turns = self.stage_turns[self.stage]

        if estimate is None:
            estimate = self.integrator.xy
        else:
            # The real loop reads `obs.position_estimate` straight off the
            # integrator at the top of the turn, so a fixture that states an
            # estimate must re-anchor the integrator too — otherwise a
            # `correct_position` in this turn would log an `old_xy` the model
            # was never shown, and the §5.8 magnitudes would be fiction.
            self.integrator.x, self.integrator.y = float(estimate[0]), float(estimate[1])
        # Rendered BEFORE the calls are applied — doc 06 §4's `block` is the
        # pre-dispatch bytes, while the structured snapshot fields are read
        # after. The two vintages disagree by one turn's writes by construction.
        block = render_memory_block(self.memory, self.counters, estimate, compass_deg)

        motion: list[tuple[str, dict, dict]] = []
        non_motion: list[str] = []
        tool_calls: list[dict] = []
        # `loop._run_turn` BREAKS at `declare_done` and answers every remaining
        # tool_use block with `not_executed`, so neither `declare_done` itself
        # nor anything after it is counted in `dispatched`. Mirrored here because
        # the scorer reads `dispatched` to decide which calls actually happened.
        dispatched = 0
        ended = False
        for call in calls:
            name = call["name"]
            args = call.get("args", {})
            tool_calls.append({"name": name, "args": args})
            if name == "declare_done":
                ended = True
            elif not ended:
                dispatched += 1
            if ended and name != "declare_done":
                continue
            self._apply(name, args)
            if "execution" in call:
                assert name in MOTION_TOOLS, f"{name} is not a motion tool"
                motion.append((name, args, call["execution"]))
                if call["execution"].get("counted_as_bump"):
                    self.bumps += 1
                self.stage_seconds[self.stage] += call["execution"][
                    "policy_seconds_used"
                ]
            elif name not in MOTION_TOOLS:
                non_motion.append(name)

        if true_pose is not None:
            self.true_pose = true_pose
        elif motion:
            pose = motion[-1][2]["true_pose"]
            self.true_pose = (pose[0], pose[1], pose[2])

        self.counters.policy_seconds = self.stage_seconds[self.stage]
        record = {
            "stage": self.stage,
            "turn_idx": self.stage_turns[self.stage],
            "global_turn_idx": self.global_turn,
            "timestamp": "2026-07-27T04:12:31Z",
            "obs": {
                "frame_paths": [],
                "compass_deg": round(compass_deg, 1),
                "position_estimate": {
                    "x": round(estimate[0], 2),
                    "y": round(estimate[1], 2),
                },
                "status": {"bumped": False, "fell": False, "distance_moved_m": 0.0},
            },
            "model_output": {
                "thought": "",
                "text": "",
                "stop_reason": "tool_use",
                "refusal": None,
                "tool_calls": tool_calls,
                "dispatched": dispatched,
                "parse_errors": [],
                "nudged": not tool_calls,
            },
            "execution": (
                merge_executions(motion, non_motion) if execution is None else execution
            ),
            "true_pose": {
                "x": round(self.true_pose[0], 4),
                "y": round(self.true_pose[1], 4),
                "heading_deg": round(self.true_pose[2], 2),
            },
            "memory_snapshot": memory_snapshot(self.memory, block),
            "budget": {
                "stage_turns_used": self.counters.turns,
                "stage_turn_cap": self.counters.turn_cap,
                "stage_policy_seconds_used": round(self.counters.policy_seconds, 4),
                "stage_policy_seconds_cap": self.counters.policy_seconds_cap,
            },
            "usage": Usage().as_dict(),
            "end_reason": end_reason,
        }
        self.turns.append(record)
        return record

    def start_return_home(self) -> None:
        """The stage boundary: the counters reset, memory and bumps do not."""
        self.stage = STAGE_RETURN_HOME
        self.memory.stage = STAGE_RETURN_HOME
        self.counters = Counters()

    # -- the final block ---------------------------------------------------

    def _spec(self, stage: str):
        if stage == STAGE_FIND_KITCHEN:
            return find_kitchen_spec()
        return return_home_spec(self.spawn)

    def _stage_result(self, stage: str, end_reason: str) -> StageResult:
        spec = self._spec(stage)
        if end_reason == "not_run":
            return StageResult.not_run(spec)
        score = score_stage(spec, (self.true_pose[0], self.true_pose[1]))
        success = bool(score.success) and end_reason == REASON_DECLARE_DONE
        return StageResult(
            stage=stage,
            end_reason=end_reason,
            outcome=outcome_for(end_reason, success),
            success=success,
            turns_used=self.stage_turns[stage],
            policy_seconds_used=self.stage_seconds[stage],
            score=score,
            true_pose=self.true_pose,
        )

    def finish(
        self,
        *,
        stage1_end: str = REASON_DECLARE_DONE,
        stage2_end: str = "not_run",
        stage1_pose: tuple[float, float, float] | None = None,
        qa_answers=None,
    ) -> dict:
        if stage1_pose is not None:
            saved, self.true_pose = self.true_pose, stage1_pose
            stage1 = self._stage_result(STAGE_FIND_KITCHEN, stage1_end)
            self.true_pose = saved
        else:
            stage1 = self._stage_result(STAGE_FIND_KITCHEN, stage1_end)
        stage2 = self._stage_result(STAGE_RETURN_HOME, stage2_end)

        answers = list(qa_answers if qa_answers is not None else [""] * 5)
        qa = [
            {
                "number": question.number,
                "question": question.text,
                "answer": answers[index],
                "score": None,
            }
            for index, question in enumerate(LAYOUT_QA_QUESTIONS)
        ]
        self.document["final"] = {
            "outcome": {stage1.stage: stage1.outcome, stage2.stage: stage2.outcome},
            "end_reason": {
                stage1.stage: stage1.end_reason,
                stage2.stage: stage2.end_reason,
            },
            "stages": {
                stage1.stage: stage1.as_dict(),
                stage2.stage: stage2.as_dict(),
            },
            "bumps": self.bumps,
            "metrics": {},
            "tokens": Usage().as_dict(),
            "tokens_breakdown": {
                "episode": Usage().as_dict(),
                "qa": Usage().as_dict(),
            },
            "qa": qa,
            "qa_raw": "\n".join(f"{i + 1}. {a}" for i, a in enumerate(answers)),
            "qa_parse_failed": any(not a for a in answers),
        }
        return self.document


def straight_trace(start, end, steps: int = 4):
    """A pose_trace sampled along a straight segment, bookends included."""
    return [
        (
            start[0] + (end[0] - start[0]) * i / steps,
            start[1] + (end[1] - start[1]) * i / steps,
        )
        for i in range(steps + 1)
    ]


GOLDEN_QA_ANSWERS = [
    "The hallway connects the bedroom to the kitchen.",
    "From the front of the sofa, head east through the doorway into the kitchen; "
    "the fridge is on the north wall.",
    "I visited two rooms: the living room and the kitchen.",
    "The kitchen is to the northeast of my spawn point.",
    "Living room: the blue rug. Kitchen: the fridge.",
]


def successful_trial(seed: int = 101, qa_answers=None) -> dict:
    """A complete stage-1 + stage-2 success, seed 101 — the workhorse fixture.

    Route: spawn (0.5, 0.5) → living room → the living_room↔kitchen doorway at
    (1.8, 1.2) → the counter target (2.55, 0.75), then home again. It exercises a
    no-motion turn, a bumped ``move``, a ``correct_position``, a ``leads_to:``
    exit and both stages.
    """
    builder = TrialBuilder(seed=seed)
    builder.turn([{"name": "look_around", "args": {}}])
    builder.turn(
        [
            {
                "name": "update_room",
                "args": {"name": "living room", "description": "sofa, blue rug"},
            },
            {"name": "set_current_room", "args": {"name": "living room"}},
            {
                "name": "move",
                "args": {"distance_m": 0.8},
                "execution": motion_execution(
                    straight_trace((0.5, 0.5), (1.2, 0.9)), policy_seconds=4.0
                ),
            },
        ],
        estimate=(0.5, 0.5),
    )
    builder.turn(
        [
            {
                "name": "add_landmark",
                "args": {"room": "living room", "description": "blue rug"},
            },
            {
                "name": "move",
                "args": {"distance_m": 1.2},
                "execution": motion_execution(
                    straight_trace((1.2, 0.9), (2.2, 1.05)),
                    policy_seconds=6.0,
                    bumped=True,
                    stop_reason="bump",
                    counted_as_bump=True,
                ),
            },
            {"name": "update_room", "args": {"name": "kitchen", "description": "counter"}},
            {"name": "set_current_room", "args": {"name": "kitchen"}},
            {
                "name": "mark_exit",
                "args": {
                    "room": "living room",
                    "direction_deg": 0,
                    "status": "leads_to:kitchen",
                },
            },
        ],
        estimate=(1.18, 0.92),
    )
    builder.turn(
        [
            {
                "name": "correct_position",
                "args": {"x": 2.2, "y": 1.05, "reason": "counter in view"},
            },
            {
                "name": "move",
                "args": {"distance_m": 0.45},
                "execution": motion_execution(
                    straight_trace((2.2, 1.05), (2.55, 0.72)), policy_seconds=2.5
                ),
            },
            {"name": "declare_done", "args": {}},
        ],
        estimate=(2.24, 1.01),
        end_reason=REASON_DECLARE_DONE,
    )

    builder.start_return_home()
    builder.turn(
        [
            {
                "name": "move",
                "args": {"distance_m": 1.4},
                "execution": motion_execution(
                    straight_trace((2.55, 0.72), (1.3, 0.95)), policy_seconds=7.0
                ),
            }
        ],
        estimate=(2.5, 0.75),
    )
    builder.turn(
        [
            {
                "name": "move",
                "args": {"distance_m": 0.9},
                "execution": motion_execution(
                    straight_trace((1.3, 0.95), (0.62, 0.58)), policy_seconds=4.5
                ),
            },
            {"name": "declare_done", "args": {}},
        ],
        estimate=(1.28, 0.9),
        end_reason=REASON_DECLARE_DONE,
    )
    return builder.finish(
        stage1_end=REASON_DECLARE_DONE,
        stage2_end=REASON_DECLARE_DONE,
        stage1_pose=(2.55, 0.72, 90.0),
        qa_answers=GOLDEN_QA_ANSWERS if qa_answers is None else qa_answers,
    )


def wandering_success_trial() -> dict:
    """A SUCCESS whose walked path far exceeds the oracle, so SPL is strictly
    below the ``max(p, l)`` clamp.

    ``successful_trial`` walks straight lines while the oracle threads a 5 cm
    grid, so BOTH its stages have ``p < l`` and its SPL is pinned at the clamp —
    which makes SPL insensitive to ``l``, to ``p``, and to the wiring between
    them. Route here: spawn → up the living room → hallway → across → down into
    the kitchen → the counter, then ``declare_done``.
    """
    legs = [
        (0.5, 0.5),
        (0.9, 2.5),   # living room, below the hallway doorway
        (0.9, 2.9),   # hallway
        (2.55, 2.9),  # hallway, above the kitchen doorway
        (2.55, 2.4),  # kitchen
        (2.55, 0.75),  # the counter target
    ]
    builder = TrialBuilder()
    for index in range(len(legs) - 1):
        calls = [
            {
                "name": "move",
                "args": {},
                "execution": motion_execution(
                    straight_trace(legs[index], legs[index + 1]), policy_seconds=5.0
                ),
            }
        ]
        last = index == len(legs) - 2
        if last:
            calls.append({"name": "declare_done", "args": {}})
        builder.turn(
            calls,
            estimate=legs[index],
            end_reason=REASON_DECLARE_DONE if last else None,
        )
    return builder.finish(stage1_end=REASON_DECLARE_DONE)


def returned_via_hallway_trial() -> dict:
    """Stage 2 walks home through a room stage 1 never entered.

    ``visited_rooms`` is deliberately TRIAL-scoped ("because memory is"), and no
    other fixture can tell trial- from stage-scoping apart: the workhorse trial
    walks the same two rooms out and back. Scoping it to stage 1 would shrink
    §5.7's recall denominator and Q3/Q5's gold room set for every trial that
    takes a different route home — the common case.
    """
    builder = TrialBuilder()
    builder.turn(
        [
            {
                "name": "move",
                "args": {},
                "execution": motion_execution(
                    straight_trace((0.5, 0.5), (2.55, 0.75)), policy_seconds=10.0
                ),
            },
            {"name": "declare_done", "args": {}},
        ],
        estimate=(0.5, 0.5),
        end_reason=REASON_DECLARE_DONE,
    )
    builder.start_return_home()
    for start, end in (
        ((2.55, 0.75), (2.55, 2.9)),   # kitchen -> hallway
        ((2.55, 2.9), (0.9, 2.9)),     # across the hallway
        ((0.9, 2.9), (0.6, 0.6)),      # back down into the living room, home
    ):
        builder.turn(
            [
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        straight_trace(start, end), policy_seconds=6.0
                    ),
                }
            ]
            + ([{"name": "declare_done", "args": {}}] if end == (0.6, 0.6) else []),
            estimate=start,
            end_reason=REASON_DECLARE_DONE if end == (0.6, 0.6) else None,
        )
    return builder.finish(
        stage1_end=REASON_DECLARE_DONE,
        stage2_end=REASON_DECLARE_DONE,
        stage1_pose=(2.55, 0.75, 90.0),
    )


# ---------------------------------------------------------------------------
# §9.1 row 1 — Progress clipping (doc 06 §5.2)
# ---------------------------------------------------------------------------


class TestProgressClipping:
    def test_moving_away_from_the_goal_never_goes_negative(self):
        assert scoring.progress(1.0, 2.5) == 0.0

    def test_success_uses_the_canonical_formula_with_no_override(self):
        """doc 06 §9.1: d_final = 0.34 m, d_initial = 1.0 m ⇒ 0.66 — the same
        value a failure would report at the same distances."""
        assert scoring.progress(1.0, 0.34) == pytest.approx(0.66)

    def test_standing_still_scores_zero(self):
        assert scoring.progress(2.4, 2.4) == 0.0

    def test_reaching_the_goal_exactly_is_the_upper_clip_boundary(self):
        """d_final = 0 on a FAILURE still scores 1.0 — that is intended."""
        assert scoring.progress(2.4, 0.0) == 1.0

    def test_a_zero_length_task_is_zero_progress_not_a_division_by_zero(self):
        """Unreachable for stage 1; representable for an unrun stage 2, where
        doc 06 §3.2 requires 0.0 because d_final == d_initial."""
        assert scoring.progress(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# §9.1 row 2 — SPL edge cases (doc 06 §5.3)
# ---------------------------------------------------------------------------


class TestSPLEdgeCases:
    def test_a_path_shorter_than_the_oracle_is_capped_at_one(self):
        assert scoring.spl(True, 2.4, 2.0) == 1.0

    def test_failure_scores_zero_regardless_of_path(self):
        assert scoring.spl(False, 2.4, 2.4) == 0.0
        assert scoring.spl(False, 2.4, 99.0) == 0.0

    def test_walking_exactly_the_oracle_scores_the_success_indicator(self):
        assert scoring.spl(True, 2.4, 2.4) == 1.0
        assert scoring.spl(False, 2.4, 2.4) == 0.0

    def test_a_wandering_success_scores_below_one(self):
        assert scoring.spl(True, 2.4, 4.8) == pytest.approx(0.5)

    def test_zero_length_path_and_oracle_do_not_blow_up(self):
        assert scoring.spl(True, 0.0, 0.0) == 1.0

    def test_failure_short_circuits_before_the_division(self):
        """§9.1's stage-2 case, in both shapes it can arrive in: after a stage-1
        failure the end pose is arbitrary, so ``l`` can be ~0 and ``max(p, l)``
        can be 0/0, or the oracle can be unreachable and ``l`` undefined. An
        implementation that computes the ratio first raises on either."""
        assert scoring.spl(False, 0.0, 0.0) == 0.0
        assert scoring.spl(False, None, 0.0) == 0.0
        assert scoring.spl(False, None, 12.3) == 0.0

    def test_a_success_with_no_oracle_path_raises_rather_than_inventing_one(self):
        with pytest.raises(scoring.ScoringError, match="oracle"):
            scoring.spl(True, None, 1.0)


# ---------------------------------------------------------------------------
# PLAN T4.1 — the pose_trace-missing case (doc 06 §4 widening #3)
# ---------------------------------------------------------------------------


class TestPathIntegral:
    def test_the_path_integral_sums_the_five_hz_samples(self):
        builder = TrialBuilder()
        builder.turn(
            [
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        straight_trace((0.0, 0.0), (1.0, 0.0)), policy_seconds=5.0
                    ),
                }
            ]
        )
        assert scoring.path_length_m(
            builder.turns, STAGE_FIND_KITCHEN
        ) == pytest.approx(1.0)

    def test_a_curved_within_turn_path_is_longer_than_its_chord(self):
        """The whole reason §5.3 pins p to the sub-turn samples: chord-summing
        the once-per-turn true_pose would shrink p and INFLATE SPL."""
        builder = TrialBuilder()
        arc = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)]
        builder.turn(
            [
                {
                    "name": "send_velocity",
                    "args": {},
                    "execution": motion_execution(arc, policy_seconds=3.0),
                }
            ]
        )
        walked = scoring.path_length_m(builder.turns, STAGE_FIND_KITCHEN)
        assert walked == pytest.approx(2 * math.hypot(0.5, 0.5))
        assert walked > math.dist(arc[0], arc[-1])

    def test_an_empty_pose_trace_is_legal_and_contributes_nothing(self):
        """doc 06 §4 widening #3: a turn that stepped no physics has an EMPTY
        trace. It must NOT raise — only a MISSING one does."""
        builder = TrialBuilder()
        builder.turn([{"name": "look_around", "args": {}}])
        execution = builder.turns[0]["execution"]
        assert execution["pose_trace"] == []
        assert execution["policy_seconds_used"] == 0.0
        assert execution["motion_calls"] == 0
        assert scoring.path_length_m(builder.turns, STAGE_FIND_KITCHEN) == 0.0

    def test_a_missing_pose_trace_raises_loudly(self):
        """PLAN T4.1: "raises loudly, never silently falls back" to per-turn
        chords, which would inflate SPL in the headline metric."""
        builder = TrialBuilder()
        builder.turn([{"name": "look_around", "args": {}}])
        del builder.turns[0]["execution"]["pose_trace"]
        with pytest.raises(scoring.MissingPoseTraceError, match="pose_trace"):
            scoring.path_length_m(builder.turns, STAGE_FIND_KITCHEN)

    def test_a_missing_execution_block_raises_loudly(self):
        builder = TrialBuilder()
        builder.turn([{"name": "look_around", "args": {}}])
        builder.turns[0]["execution"] = None
        with pytest.raises(scoring.MissingPoseTraceError, match="execution"):
            scoring.path_length_m(builder.turns, STAGE_FIND_KITCHEN)

    def test_a_missing_trace_is_never_silently_replaced_by_a_chord(self):
        """The failure this raise prevents, stated as a measurement: the chord
        of a curved turn is strictly shorter than its trace, so a fallback would
        shrink p and raise SPL."""
        builder = TrialBuilder()
        arc = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)]
        builder.turn(
            [
                {
                    "name": "send_velocity",
                    "args": {},
                    "execution": motion_execution(arc, policy_seconds=3.0),
                }
            ]
        )
        walked = scoring.path_length_m(builder.turns, STAGE_FIND_KITCHEN)
        chord = math.dist((0.0, 0.0), (1.0, 0.0))
        assert scoring.spl(True, 1.0, chord) > scoring.spl(True, 1.0, walked)

    def test_the_integral_is_segmented_per_stage(self):
        """Summing across the boundary would charge stage 2 with the arbitrary
        jump from wherever stage 1 ended."""
        document = successful_trial()
        stage1 = scoring.path_length_m(document["turns"], STAGE_FIND_KITCHEN)
        stage2 = scoring.path_length_m(document["turns"], STAGE_RETURN_HOME)
        assert stage1 > 0.0 and stage2 > 0.0
        merged = scoring.path_length_m(
            [dict(turn, stage=STAGE_FIND_KITCHEN) for turn in document["turns"]],
            STAGE_FIND_KITCHEN,
        )
        # Here the stages meet at the same point, so merging adds a zero-length
        # segment; the assertion is that nothing else is double counted.
        assert merged == pytest.approx(stage1 + stage2)

    def test_duplicate_boundary_points_contribute_nothing(self):
        """Each call's trace is [start, *samples, end], so consecutive calls
        share a point. A duplicate contributes exactly 0 to the sum."""
        builder = TrialBuilder()
        builder.turn(
            [
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)], policy_seconds=5.0
                    ),
                },
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        [(1.0, 0.0), (1.5, 0.0), (2.0, 0.0)], policy_seconds=5.0
                    ),
                },
            ]
        )
        assert scoring.path_length_m(
            builder.turns, STAGE_FIND_KITCHEN
        ) == pytest.approx(2.0)

    def test_a_null_pose_trace_is_the_same_fault_as_a_missing_one(self):
        """`null` reached the path integral as a raw TypeError with no trial
        named — the class docstring is entirely about keeping "no motion" and
        "trace dropped" distinguishable, and `null` is the latter."""
        document = successful_trial()
        document["turns"][1]["execution"]["pose_trace"] = None
        with pytest.raises(scoring.MissingPoseTraceError, match="pose_trace"):
            scoring.path_length_m(document["turns"], STAGE_FIND_KITCHEN)


class TestChordFloor:
    """An EMPTY pose_trace is legal, so losing every sample is silent — and
    every such loss pushes SPL UP, because ``max(p, l)`` caps it at 1.0."""

    def test_the_floor_is_tight_on_the_golden_trial(self):
        """Straight-line traces make the bound exact, which is why the tolerance
        has to stay small enough to catch a real loss."""
        document = successful_trial()
        for stage in (STAGE_FIND_KITCHEN, STAGE_RETURN_HOME):
            walked = scoring.path_length_m(document["turns"], stage)
            floor, chords = scoring.chord_floor_m(document, stage)
            assert walked == pytest.approx(floor, abs=1e-9)
            assert chords == len(
                [t for t in document["turns"] if t["stage"] == stage]
            )

    def test_a_wholly_dropped_trace_is_caught_instead_of_scoring_a_perfect_spl(self):
        """Measured before the floor existed: blanking every ``pose_trace`` to
        ``[]`` gave ``p = 0`` and ``spl = 1.0`` for BOTH stages — a wholly broken
        trace pipeline was indistinguishable from a perfect run."""
        document = successful_trial()
        for turn in document["turns"]:
            turn["execution"]["pose_trace"] = []
        with pytest.raises(scoring.ScoringError, match="chord floor"):
            scoring.score_trial(document)

    def test_a_partially_dropped_trace_is_caught_too(self):
        """Keeping only each trace's last sample gave p = 1.4922 against a true
        2.2985 (-35 %) and still spl = 1.0."""
        document = successful_trial()
        for turn in document["turns"]:
            if turn["execution"]["pose_trace"]:
                turn["execution"]["pose_trace"] = turn["execution"]["pose_trace"][-1:]
        with pytest.raises(scoring.ScoringError, match="chord floor"):
            scoring.score_trial(document)

    def test_a_curved_path_is_comfortably_above_its_chord_floor(self):
        """The floor is a LOWER bound by the triangle inequality, so real
        within-turn curvature must never trip it."""
        builder = TrialBuilder()
        builder.turn(
            [
                {
                    "name": "send_velocity",
                    "args": {},
                    "execution": motion_execution(
                        [(0.5, 0.5), (1.0, 1.2), (1.5, 0.5)], policy_seconds=3.0
                    ),
                }
            ],
            end_reason=REASON_TURN_CAP,
        )
        document = builder.finish(stage1_end=REASON_TURN_CAP)
        walked = scoring.path_length_m(document["turns"], STAGE_FIND_KITCHEN)
        floor, _ = scoring.chord_floor_m(document, STAGE_FIND_KITCHEN)
        assert walked > floor
        assert scoring.score_trial(document).stages[STAGE_FIND_KITCHEN].true_path_m > 0


class TestLogConsistencyGuards:
    """Fields the log states about itself, cross-checked the way §5.6 already
    cross-checks bumps. Each of these otherwise flatters a model silently."""

    def test_a_turns_used_that_contradicts_the_turns_is_rejected(self):
        document = successful_trial()
        document["final"]["stages"][STAGE_FIND_KITCHEN]["turns_used"] = 1
        with pytest.raises(scoring.ScoringError, match="turns_used"):
            scoring.score_trial(document)

    def test_a_turn_stamped_with_an_unknown_stage_is_rejected(self):
        """Measured: relabelling the last stage-1 turn dropped p from 2.2985 to
        1.8174 (-21 %) and raised SPL, with nothing raising. A dropped tail is
        pure SPL inflation."""
        document = successful_trial()
        document["turns"][3]["stage"] = "qa"
        with pytest.raises(scoring.ScoringError, match="not one of"):
            scoring.score_trial(document)

    def test_a_declare_done_stage_with_no_turns_is_rejected(self):
        """A trial with ``turns: []`` and a ``final`` claiming a successful
        declare at the counter scored spl 1.0, progress 0.985, time_s 0.0 — a
        teleport scoring perfectly."""
        document = successful_trial()
        document["turns"].clear()
        for stage in (STAGE_FIND_KITCHEN, STAGE_RETURN_HOME):
            document["final"]["stages"][stage]["turns_used"] = 0
        document["final"]["bumps"] = 0
        with pytest.raises(scoring.ScoringError, match="truncated"):
            scoring.score_trial(document)

    def test_a_tampered_spawn_is_rejected(self):
        """Everything downstream is derived from the spawn — stage-1 d_initial,
        the return_home goal and radius, the SPL oracle, Q4's gold bearing — so a
        harness that spawned the robot elsewhere would produce a full set of
        plausible-looking but wrong numbers."""
        document = successful_trial()
        document["config"]["spawn"]["xy"] = [9.99, 9.99]
        with pytest.raises(scoring.ScoringError, match="spawn_points"):
            scoring.score_trial(document)


class TestNonFiniteValues:
    """NaN was the one corrupt value that made every number BETTER."""

    def test_progress_rejects_a_non_finite_distance(self):
        """``min(1.0, nan)`` is 1.0 and ``max(0.0, 1.0)`` is 1.0, so a garbage
        distance reported the maximum possible progress."""
        with pytest.raises(scoring.ScoringError, match="finite"):
            scoring.progress(2.0, float("nan"))
        with pytest.raises(scoring.ScoringError, match="finite"):
            scoring.progress(float("nan"), 1.0)

    def test_spl_rejects_non_finite_lengths(self):
        with pytest.raises(scoring.ScoringError, match="finite"):
            scoring.spl(True, 2.4, float("inf"))
        with pytest.raises(scoring.ScoringError, match="finite"):
            scoring.spl(True, float("nan"), 2.4)

    def test_one_corrupt_trial_cannot_nan_out_a_whole_model_column(self):
        """``estimate([1.0, nan, 1.0])`` returned mean nan, ci95 [nan, nan] and
        claimed n_defined = 3."""
        with pytest.raises(scoring.ScoringError, match="finite"):
            scoring.estimate([1.0, float("nan"), 1.0])

    def test_a_nan_in_a_pose_trace_is_caught_where_it_enters(self):
        document = successful_trial()
        document["turns"][1]["execution"]["pose_trace"][1][0] = float("nan")
        with pytest.raises(scoring.ScoringError, match="finite"):
            scoring.score_trial(document)

    def test_a_nan_trace_on_a_FAILED_stage_is_caught_too(self):
        """Validating at the point of READING is what makes this safe. Downstream
        guards are not enough: ``spl`` short-circuits on ``S = 0`` before its own
        check, so a failed stage would publish ``true_path_m = nan`` — and the
        chord-floor comparison cannot catch it either, because every comparison
        against NaN is False."""
        builder = TrialBuilder()
        builder.turn(
            [
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        straight_trace((0.5, 0.5), (1.0, 1.0)), policy_seconds=4.0
                    ),
                }
            ],
            end_reason=REASON_TURN_CAP,
        )
        document = builder.finish(stage1_end=REASON_TURN_CAP)
        document["turns"][0]["execution"]["pose_trace"][2][1] = float("nan")
        with pytest.raises(scoring.ScoringError, match="finite"):
            scoring.path_length_m(document["turns"], STAGE_FIND_KITCHEN)
        with pytest.raises(scoring.ScoringError, match="finite"):
            scoring.score_trial(document)

    def test_a_nan_true_pose_is_caught(self):
        document = successful_trial()
        document["turns"][2]["true_pose"]["y"] = float("nan")
        with pytest.raises(scoring.ScoringError, match="finite"):
            scoring.score_trial(document)


# ---------------------------------------------------------------------------
# §9.1 row 3 — Map matching tie-breaks (doc 06 §5.7)
# ---------------------------------------------------------------------------

LIVING_ROOM_POINT = (0.6, 1.0)
KITCHEN_POINT = (2.4, 1.0)
BEDROOM_POINT = (4.0, 1.0)


class TestMapMatching:
    def test_a_synonym_table_hit_matches(self):
        matches = scoring.match_rooms(
            ["lounge"], {"lounge": [LIVING_ROOM_POINT]}, {"lounge": 1}
        )
        assert matches == [("lounge", "living_room")]

    def test_a_non_synonym_near_string_does_not_match(self):
        """doc 06 §9.1's canonical case: "kitchenette" claimed in the bedroom
        polygon. It is deliberately absent from the FROZEN table, so the NAME
        half of the rule is what rejects it."""
        assert "kitchenette" not in ROOM_SYNONYMS
        matches = scoring.match_rooms(
            ["kitchenette"], {"kitchenette": [BEDROOM_POINT]}, {"kitchenette": 1}
        )
        assert matches == []

    def test_two_name_similar_claims_on_one_true_room_give_exactly_one_match(self):
        matches = scoring.match_rooms(
            ["living room", "lounge"],
            {"living room": [LIVING_ROOM_POINT], "lounge": [LIVING_ROOM_POINT] * 3},
            {"living room": 1, "lounge": 2},
        )
        assert matches == [("lounge", "living_room")], "evidence count wins first"

    def test_a_tie_on_evidence_is_broken_by_name_similarity(self):
        """With a fixed synonym table the name test is boolean, so the only two
        grades are "is the canonical name" and "reached it via a synonym"."""
        matches = scoring.match_rooms(
            ["lounge", "living room"],
            {"lounge": [LIVING_ROOM_POINT], "living room": [LIVING_ROOM_POINT]},
            {"lounge": 1, "living room": 2},
        )
        assert matches == [("living room", "living_room")]

    def test_a_tie_on_evidence_and_name_grade_is_broken_by_earliest_claim(self):
        matches = scoring.match_rooms(
            ["den", "lounge"],
            {"den": [LIVING_ROOM_POINT], "lounge": [LIVING_ROOM_POINT]},
            {"den": 7, "lounge": 2},
        )
        assert matches == [("lounge", "living_room")]

    def test_evidence_straddling_two_polygons_follows_the_majority(self):
        matches = scoring.match_rooms(
            ["kitchen"],
            {"kitchen": [KITCHEN_POINT, KITCHEN_POINT, LIVING_ROOM_POINT]},
            {"kitchen": 1},
        )
        assert matches == [("kitchen", "kitchen")]

    def test_a_claim_whose_majority_of_evidence_is_outside_does_not_match(self):
        matches = scoring.match_rooms(
            ["kitchen"],
            {"kitchen": [KITCHEN_POINT, LIVING_ROOM_POINT, LIVING_ROOM_POINT]},
            {"kitchen": 1},
        )
        assert matches == []

    def test_an_exact_half_split_is_not_a_majority(self):
        matches = scoring.match_rooms(
            ["kitchen"],
            {"kitchen": [KITCHEN_POINT, LIVING_ROOM_POINT]},
            {"kitchen": 1},
        )
        assert matches == []

    def test_claiming_nothing_leaves_precision_undefined_and_recall_zero(self):
        """doc 06 §5.7's edge convention: precision is "—", excluded from the
        aggregate, never coerced to a number — and no division by zero."""
        builder = TrialBuilder()
        builder.turn(
            [
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        straight_trace((0.5, 0.5), (1.2, 1.0)), policy_seconds=4.0
                    ),
                }
            ]
        )
        document = builder.finish(stage1_end=REASON_TURN_CAP)
        accuracy = scoring.map_accuracy(document)
        assert accuracy.claimed == 0
        assert accuracy.precision == NA
        assert accuracy.recall == 0.0
        assert accuracy.true_rooms_visited >= 1

    def test_an_undefined_precision_is_excluded_from_the_aggregate(self):
        assert scoring.defined([0.5, NA, 1.0]) == [0.5, 1.0]
        assert scoring.estimate([NA, NA]).mean == NA

    def test_evidence_comes_from_the_three_room_naming_tools(self):
        assert set(scoring.ROOM_CLAIM_TOOLS) == {
            "update_room",
            "set_current_room",
            "add_landmark",
        }
        evidence = scoring.room_evidence(successful_trial())
        assert evidence["living room"]
        assert evidence["kitchen"]

    def test_the_end_to_end_map_accuracy_of_the_golden_trial(self):
        accuracy = scoring.map_accuracy(successful_trial())
        assert accuracy.claimed == 2
        assert accuracy.matched == 2
        assert accuracy.precision == 1.0
        assert accuracy.recall == 1.0
        assert accuracy.edges_claimed == 1
        assert accuracy.edges_correct == 1
        assert accuracy.edge_accuracy == 1.0

    def test_an_edge_to_an_unmatched_room_is_wrong_not_excluded(self):
        """Excluding it would make an unmatched claim FREE after §5.7 already
        counted it against precision. (Reported as an open §5.7 wording point.)"""
        builder = TrialBuilder()
        builder.turn(
            [
                {"name": "update_room", "args": {"name": "kitchen", "description": "x"}},
                {
                    "name": "mark_exit",
                    "args": {
                        "room": "kitchen",
                        "direction_deg": 90,
                        "status": "leads_to:cupboard",
                    },
                },
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        straight_trace((0.5, 0.5), (2.4, 1.0)), policy_seconds=8.0
                    ),
                },
            ]
        )
        document = builder.finish(stage1_end=REASON_TURN_CAP)
        accuracy = scoring.map_accuracy(document)
        assert accuracy.edges_claimed == 1
        assert accuracy.edges_correct == 0
        assert accuracy.edge_accuracy == 0.0
        # Published so the OTHER reading of §5.7 ("edges between matched rooms"
        # = exclude the unresolved one) stays recomputable from the numbers.
        assert accuracy.edges_unresolved == 1
        # …and the asymmetric precision/recall case the file otherwise lacks:
        # one claim, matched, against two visited rooms.
        assert accuracy.claimed == 1 and accuracy.matched == 1
        assert accuracy.precision == 1.0
        assert accuracy.recall == 0.5

    def test_two_claims_for_one_visited_room_invert_precision_and_recall(self):
        """The mirror of the case above. Without a fixture where the two numbers
        differ in BOTH directions, swapping precision and recall — or defining
        either over the wrong denominator — changes no assertion in this file."""
        builder = TrialBuilder()
        builder.turn(
            [
                {"name": "update_room", "args": {"name": "lounge", "description": "x"}},
                {"name": "update_room", "args": {"name": "galley", "description": "y"}},
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        straight_trace((0.5, 0.5), (0.6, 1.0)), policy_seconds=2.0
                    ),
                },
            ]
        )
        document = builder.finish(stage1_end=REASON_TURN_CAP)
        accuracy = scoring.map_accuracy(document)
        # "galley" -> kitchen fails the POLYGON half (the robot never left the
        # living room), so it is a claim that matched nothing.
        assert accuracy.claimed == 2 and accuracy.matched == 1
        assert accuracy.true_rooms_visited == 1
        assert accuracy.precision == 0.5
        assert accuracy.recall == 1.0

    def test_a_call_the_harness_never_executed_is_not_evidence(self):
        """`loop._run_turn` breaks at ``declare_done`` and answers every later
        tool_use with ``not_executed``; ``dispatched`` records how many actually
        ran. Counting the rest let a rejected claim tip the majority rule."""
        builder = TrialBuilder()
        builder.turn(
            [
                {"name": "update_room", "args": {"name": "kitchen", "description": "x"}},
                {"name": "declare_done", "args": {}},
                {"name": "set_current_room", "args": {"name": "kitchen"}},
            ],
            true_pose=(2.4, 1.0, 90.0),
            end_reason=REASON_DECLARE_DONE,
        )
        turn = builder.turns[0]
        assert [call["name"] for call in turn["model_output"]["tool_calls"]] == [
            "update_room",
            "declare_done",
            "set_current_room",
        ]
        assert turn["model_output"]["dispatched"] == 1
        assert scoring.room_evidence(builder.document) == {"kitchen": [(2.4, 1.0)]}

    def test_a_tool_call_with_null_args_does_not_kill_the_trial(self):
        """The OpenAI adapter parses tool arguments out of a JSON string, so a
        provider-side artefact can land ``null`` there. It used to raise
        ``AttributeError`` and lose the whole trial's score."""
        document = successful_trial()
        document["turns"][1]["model_output"]["tool_calls"].append(
            {"name": "update_room", "args": None}
        )
        document["turns"][1]["model_output"]["dispatched"] += 1
        assert scoring.room_evidence(document)["living room"]
        assert scoring.claim_order(document)["kitchen"] == 3

    def test_unexplored_exits_are_not_adjacency_assertions(self):
        builder = TrialBuilder()
        builder.turn(
            [
                {"name": "update_room", "args": {"name": "kitchen", "description": "x"}},
                {
                    "name": "mark_exit",
                    "args": {
                        "room": "kitchen",
                        "direction_deg": 90,
                        "status": "unexplored",
                    },
                },
            ]
        )
        document = builder.finish(stage1_end=REASON_TURN_CAP)
        accuracy = scoring.map_accuracy(document)
        assert accuracy.edges_claimed == 0
        assert accuracy.edge_accuracy == NA


# ---------------------------------------------------------------------------
# §9.1 row 4 — QA rubric fixtures (doc 06 §5.9)
# ---------------------------------------------------------------------------


def _layout_furniture(name: str) -> dict:
    """The layout's own entry for a furniture item — the scene spec IS the key."""
    for item in LAYOUT["furniture"]:
        if item["name"] == name:
            return item
    raise AssertionError(f"the layout has no furniture item named {name!r}")


def qa_context(visited=("living_room", "kitchen"), seed: int = 101) -> scoring.QAContext:
    spawn, _heading = spawn_pose(seed)
    return scoring.QAContext(spawn_xy=spawn, visited=tuple(visited))


class TestQ1:
    def test_naming_the_unique_connector_scores_one(self):
        assert scoring.score_q1("The hallway connects them.") == 1.0

    def test_the_questions_own_rooms_are_not_read_as_the_answer(self):
        """Models restate the question ("the room that connects the BEDROOM to
        the KITCHEN is the hallway"), so a naive first-mention parse would score
        a correct answer 0."""
        assert (
            scoring.score_q1(
                "The room that connects the bedroom to the kitchen is the hallway."
            )
            == 1.0
        )

    def test_a_room_adjacent_to_exactly_one_of_the_two_scores_a_half(self):
        assert scoring.score_q1("I think it is the living room.") == 0.5

    def test_naming_one_of_the_two_rooms_scores_zero(self):
        assert scoring.score_q1("The kitchen.") == 0.0

    def test_an_unparseable_answer_scores_zero(self):
        assert scoring.score_answer(1, "", qa_context()) == 0.0
        assert scoring.score_q1("I do not remember.") == 0.0


class TestQ2ParseRules:
    """doc 06 §12's open question, RESOLVED by T4.1 and fixtured before freeze."""

    CORPUS = json.loads((FIXTURES / "qa_q2_answers.json").read_text(encoding="utf-8"))

    def test_the_gold_route_is_computed_from_the_committed_layout(self):
        """NEVER transcribed from doc 06 §11, whose row says
        living_room → hallway → kitchen for its own representative layout."""
        assert scoring.q2_start_room() == "living_room"
        assert scoring.q2_goal_room() == "kitchen"
        assert scoring.q2_oracle_route() == ["living_room", "kitchen"]

    def test_the_gold_start_point_and_bearings_are_pinned(self):
        assert scoring.q2_start_point() == pytest.approx((0.4955, 1.60))
        assert scoring.q2_gold_bearing(["living_room", "kitchen"]) == pytest.approx(
            342.9528, abs=1e-3
        )
        assert scoring.q2_gold_bearing(
            ["living_room", "hallway", "kitchen"]
        ) == pytest.approx(69.8101, abs=1e-3)

    def test_the_hallway_detour_is_a_valid_route_that_is_not_the_oracle(self):
        """doc 06 §5.9's "decide before freeze": 3.152 m direct vs 3.611 m via
        the hallway, only +14.55 % — plausibly the route the robot walked."""
        # Derived from LAYOUT, never hand-copied: every neighbouring test reads
        # the layout through its helpers, and hardcoding these would let the
        # sofa or the fridge move while this test kept asserting 3.1521 /
        # 3.6107 m about furniture that is no longer there — silently turning
        # the +14.55 % justification for MAX_EXTRA_ROOMS into fiction.
        sofa = tuple(_layout_furniture("sofa")["pos"])
        fridge = tuple(_layout_furniture("fridge")["pos"])
        assert sofa == (0.30, 1.60) and fridge == (3.10, 2.30)
        direct = oracle_length(sofa, fridge)
        waypoints = [
            sofa,
            scoring.doorway_center("living_room", "hallway"),
            scoring.doorway_center("hallway", "kitchen"),
            fridge,
        ]
        assert waypoints[1] == (0.9, 2.7) and waypoints[2] == (2.55, 2.7)
        via = sum(
            oracle_length(a, b) for a, b in zip(waypoints, waypoints[1:])
        )
        assert direct == pytest.approx(3.1521, abs=1e-3)
        assert via == pytest.approx(3.6107, abs=1e-3)
        assert via / direct == pytest.approx(1.1455, abs=1e-3)

    @pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda c: c["id"])
    def test_fixture_corpus(self, case):
        assert scoring.score_q2(case["answer"]) == case["expected"], (
            f"{case['id']}: {case['why']}\n{scoring.parse_q2(case['answer'])}"
        )

    def test_the_corpus_covers_all_three_rubric_anchors(self):
        expected = [case["expected"] for case in self.CORPUS["cases"]]
        assert len(expected) == 35
        assert expected.count(1.0) >= 1
        assert expected.count(0.5) >= 1
        assert expected.count(0.0) >= 1

    def test_the_prose_ambiguous_list_is_pinned_on_BOTH_sides(self):
        """Skipping a synonym in free prose costs a model that legitimately used
        it, so the list is a decision with a cost in each direction and both
        sides need a fixture. Skipped: ordinary English words for a DOORWAY.
        Kept: words that are genuine room names here far more often than not —
        the residual risk recorded in ``docs/METRICS.md`` §5.

        ``normalize_claim`` — whole-string room NAMES, doc 06 §5.7 — keeps the
        full frozen table either way, so a model that names a room "Entry" is
        still matched.
        """
        assert scoring.PROSE_AMBIGUOUS_SYNONYMS == {"entry", "entryway", "landing"}
        for phrase in ("the entry to the kitchen", "head east through the entryway"):
            assert "hallway" not in scoring.extract_room_mentions(phrase), phrase
        assert scoring.extract_room_mentions("the landing gear") == []
        # …and the words that stay in the prose vocabulary:
        assert scoring.extract_room_mentions("through the hall into the kitchen") == [
            "hallway",
            "kitchen",
        ]
        assert scoring.extract_room_mentions("the den has a sofa") == ["living_room"]
        assert scoring.extract_room_mentions("along the corridor") == ["hallway"]
        # The claim-normalisation path is untouched by the prose list.
        for name in ("Entry", "entryway", "Landing"):
            assert scoring.normalize_claim(name) == "hallway", name

    def test_an_explicit_degrees_phrase_does_not_swallow_the_next_word(self):
        """``consume`` marked one token PAST the matched span, so a direction
        word immediately after an explicit-degrees phrase vanished. Invisible in
        the scored bit (the swallowed token is never first) but a silently
        missing leg in ``direction_sequence``, which doc 06 §10.4's figures and
        any per-answer audit consume — and the ``Q2Parse`` record exists so a
        disputed score is auditable."""
        tokens = scoring.direction_tokens("turn 90 degrees left east then stop")
        assert [token.token for token in tokens] == ["90 degrees left", "east"]
        assert [t.token for t in scoring.direction_tokens("walk 90 degrees left north")] == [
            "90 degrees left",
            "north",
        ]
        # …and the leg really does reach the published sequence.
        assert len(scoring.direction_sequence("turn 90 degrees left east then stop")) == 2

    def test_only_the_first_cue_anchored_direction_is_scored(self):
        parse = scoring.parse_q2(
            "The sofa is against the west wall. Walk east into the kitchen; "
            "the fridge is there."
        )
        assert parse.direction == "east"
        assert parse.direction_ok


class TestQ3:
    def test_matching_names_and_count_scores_one(self):
        assert (
            scoring.score_q3(
                "I visited two rooms: the living room and the kitchen.", qa_context()
            )
            == 1.0
        )

    def test_a_stated_count_off_by_one_scores_a_half(self):
        assert (
            scoring.score_q3(
                "I visited three rooms: the living room and the kitchen.", qa_context()
            )
            == 0.5
        )

    def test_one_extra_room_scores_a_half(self):
        assert (
            scoring.score_q3(
                "Three rooms: the living room, the kitchen and the hallway.",
                qa_context(),
            )
            == 0.5
        )

    def test_two_wrong_rooms_score_zero(self):
        assert (
            scoring.score_q3("Two rooms: the bedroom and the hallway.", qa_context())
            == 0.0
        )

    def test_a_count_the_answer_never_states_is_taken_as_implied(self):
        assert scoring.score_q3("The living room and the kitchen.", qa_context()) == 1.0

    def test_an_ordinal_inside_prose_is_not_read_as_the_count(self):
        answer = "Room 1 was the living room and then I entered the kitchen."
        assert scoring.stated_count(answer) is None
        assert scoring.score_q3(answer, qa_context()) == 1.0

    def test_two_invented_rooms_without_a_stated_count_score_zero(self):
        """Pins the SYMMETRIC difference. ``test_one_extra_room_scores_a_half``
        cannot: its answer states a count, so the stated-count branch returns 0.5
        independently. With no count stated, an answer that names both true rooms
        plus two invented ones must still score 0."""
        assert scoring.stated_count(
            "The living room, the kitchen, the hallway and the bedroom."
        ) is None
        assert (
            scoring.score_q3(
                "The living room, the kitchen, the hallway and the bedroom.",
                qa_context(),
            )
            == 0.0
        )

    def test_a_symmetric_difference_of_two_in_the_other_direction_scores_zero(self):
        """One true room named, one invented — difference 2, not 1."""
        assert (
            scoring.score_q3("Two rooms: the living room and the bedroom.", qa_context())
            == 0.0
        )

    def test_rooms_the_answer_says_it_did_not_visit_are_not_counted(self):
        """Enumerating what you did NOT visit is ordinary LLM answer style. The
        mention-set reading scored this fully correct answer 0.0."""
        assert (
            scoring.score_q3(
                "I visited two rooms, the living room and the kitchen. "
                "I did not see the bedroom or the hallway.",
                qa_context(),
            )
            == 1.0
        )
        assert (
            scoring.score_q3(
                "I visited 2 rooms: the living room and the kitchen. "
                "I never entered the bedroom.",
                qa_context(),
            )
            == 1.0
        )

    def test_a_negation_does_not_run_past_its_own_clause(self):
        """The scope has to reach six tokens ("not see the bedroom or the
        hallway") without swallowing the next clause — otherwise the fix trades
        one wrong answer for another."""
        assert (
            scoring.score_q3(
                "It is not clear, but I visited the living room and the kitchen.",
                qa_context(),
            )
            == 1.0
        )
        assert (
            scoring.score_q3("No, I visited the living room and the kitchen.", qa_context())
            == 1.0
        )

    def test_an_unparseable_answer_scores_zero(self):
        assert scoring.score_q3("", qa_context()) == 0.0
        assert scoring.score_answer(3, "", qa_context()) == 0.0


class TestQ4:
    def test_the_gold_bearings_reuse_compass_8_and_are_pinned_per_seed(self):
        """doc 06 §5.9: T4.1 must reuse ``compass_8``, not author a second
        bucketer. Seed 101 is NE by 0.021°; §11's E/E/SE/SE does not transfer."""
        for seed, gold in {101: "NE", 102: "SW", 103: "SE", 104: "SE"}.items():
            assert scoring.q4_gold(qa_context(seed=seed)) == gold
        spawn, _ = spawn_pose(101)
        bearing = bearing_deg(spawn, room_centroid("kitchen"))
        assert bearing == pytest.approx(22.521, abs=1e-3)
        assert compass_8(bearing) == "NE"

    def test_the_bucketer_is_compass_8_itself_not_a_second_one(self):
        """doc 06 §5.9: "T4.1 must reuse ``compass_8`` rather than author a
        second bucketer." The convention is half-open with 22.5 rounding UP —
        exactly where the obvious hand-rolled ``round(b / 45)`` diverges, because
        Python's ``round`` is banker's rounding (22.5° → E, 112.5° → N). A
        cardinals-only test would not notice the difference."""
        assert compass_8(22.5) == "NE" and compass_8(112.5) == "NW"
        assert scoring.compass_tokens("bearing 22.5") == ["NE"]
        assert scoring.compass_tokens("bearing 112.5") == ["NW"]
        assert (
            scoring.score_q4("The kitchen is at 22.5 degrees.", qa_context(seed=101))
            == 1.0
        )

    def test_the_true_bucket_scores_one(self):
        assert scoring.score_q4("Northeast.", qa_context(seed=101)) == 1.0

    def test_an_adjacent_bucket_scores_a_half(self):
        """§9.1 names this case explicitly. Seed 101's bearing is 0.021° past the
        E/NE boundary, so "east" is 0.021° from correct and still 0.5 — the
        rubric working as intended, and worth stating in the write-up."""
        assert scoring.score_q4("Due east of the spawn.", qa_context(seed=101)) == 0.5
        assert scoring.score_q4("North.", qa_context(seed=101)) == 0.5

    def test_a_non_adjacent_bucket_scores_zero(self):
        assert scoring.score_q4("South-west.", qa_context(seed=101)) == 0.0

    def test_an_answer_with_no_compass_token_scores_zero(self):
        assert scoring.score_q4("Somewhere over there.", qa_context()) == 0.0
        assert scoring.score_answer(4, "", qa_context()) == 0.0

    def test_uppercase_abbreviations_are_read_but_lowercase_prose_is_not(self):
        assert scoring.score_q4("NE", qa_context(seed=101)) == 1.0
        assert scoring.score_q4("i.e. somewhere ahead", qa_context(seed=101)) == 0.0

    def test_an_abbreviation_must_stand_alone_to_be_a_compass_claim(self):
        """The uppercase rule was written for "i.e."; the uppercase "E.g." and
        "N/A" defeated it. Measured before the guard: ``score_q4('N/A')`` was
        **0.5** — half a point for declining to answer — and the "E.g." answer
        was scored on the abbreviation of *exempli gratia* rather than on the
        model's actual "southwest"."""
        assert scoring.compass_tokens("N/A") == []
        assert scoring.compass_tokens("E.g. somewhere") == []
        assert scoring.score_q4("N/A", qa_context(seed=101)) == 0.0
        assert scoring.score_q4("n/a", qa_context(seed=101)) == 0.0
        assert (
            scoring.score_q4(
                "E.g. somewhere to the southwest, but I am not sure.",
                qa_context(seed=101),
            )
            == 0.0
        ), "scored on the model's real answer (southwest), which is wrong here"
        # A trailing full stop at a sentence end must NOT disqualify it.
        assert scoring.score_q4("The kitchen is NE.", qa_context(seed=101)) == 1.0
        assert scoring.score_q4("Maybe S.", qa_context(seed=101)) == 0.0

    def test_a_direction_the_answer_rejects_is_not_the_answer(self):
        """Both phrasings state NE unambiguously and used to score 0.5, because
        the rejected direction is the one that got scored."""
        assert (
            scoring.score_q4(
                "Not north — the kitchen is northeast of the spawn.",
                qa_context(seed=101),
            )
            == 1.0
        )
        assert (
            scoring.score_q4("Northeast, definitely not southwest.", qa_context(seed=101))
            == 1.0
        )

    def test_the_first_surviving_token_is_the_one_scored(self):
        """Pinned in BOTH directions: every other Q4 fixture holds exactly one
        compass token, so reading the last would score identically."""
        assert (
            scoring.score_q4(
                "The kitchen is northeast; the bedroom lies south.", qa_context(seed=101)
            )
            == 1.0
        )
        assert (
            scoring.score_q4(
                "The bedroom lies south; the kitchen is northeast.", qa_context(seed=101)
            )
            == 0.0
        )


class TestQ5:
    def test_a_true_landmark_for_every_visited_room_scores_one(self):
        assert (
            scoring.score_q5("Living room: the blue rug. Kitchen: the fridge.", qa_context())
            == 1.0
        )

    def test_a_head_noun_is_accepted_when_it_is_unambiguous(self):
        assert (
            scoring.score_q5(
                "In the living room there is a rug; in the kitchen a stove.",
                qa_context(),
            )
            == 1.0
        )

    def test_every_layout_landmark_head_noun_is_unique_across_rooms(self):
        """The head-noun shortcut is only sound while this holds; a layout edit
        that breaks it must fail this gate, not silently credit a room."""
        heads: dict[str, set[str]] = {}
        for room, spec in LAYOUT["rooms"].items():
            for landmark in spec["landmarks"]:
                heads.setdefault(landmark.split()[-1].lower(), set()).add(room)
        assert {head: rooms for head, rooms in heads.items() if len(rooms) > 1} == {}

    def test_one_wrong_room_scores_a_half(self):
        assert (
            scoring.score_q5("Living room: the blue rug. Kitchen: a wardrobe.", qa_context())
            == 0.5
        )

    def test_two_wrong_rooms_score_zero(self):
        assert (
            scoring.score_q5("Living room: a wardrobe. Kitchen: a bathtub.", qa_context())
            == 0.0
        )

    def test_a_landmark_is_credited_to_the_room_it_was_attached_to(self):
        """"one landmark in EACH room" — a fridge named under the living room is
        not a living-room landmark."""
        assert (
            scoring.score_q5("Living room: the fridge. Kitchen: the blue rug.", qa_context())
            == 0.0
        )

    @pytest.mark.parametrize(
        "answer",
        [
            "The sofa is in the living room and the fridge is in the kitchen.",
            "A sofa in the living room, a fridge in the kitchen.",
            "Sofa - living room. Fridge - kitchen.",
            "I saw a blue rug (living room) and a fridge (kitchen).",
        ],
    )
    def test_a_landmark_named_before_its_room_still_counts(self, answer):
        """Each of these names a true layout landmark for every visited room —
        Q5's verbatim 1 anchor — and every one of them scored **0.0** under
        forward-only segmentation, for a phrasing reason unrelated to map
        quality. Answer order is a per-model habit, so this shifted a published
        comparison."""
        assert scoring.score_q5(answer, qa_context()) == 1.0

    def test_backward_attachment_never_reaches_across_a_sentence(self):
        """The clamp that keeps the swap case honest: without it, "Kitchen:"
        reaches back over the full stop and claims the living room's landmark."""
        assert (
            scoring.score_q5("Living room: the fridge. Kitchen: the blue rug.", qa_context())
            == 0.0
        )

    def test_a_visited_room_the_answer_never_names_is_not_credited(self):
        """Q5's denominator is every visited room, not just the ones named."""
        assert scoring.score_q5("Living room: the blue rug.", qa_context()) == 0.5
        assert (
            scoring.score_q5(
                "Nothing in particular.",
                qa_context(visited=("living_room", "kitchen", "hallway")),
            )
            == 0.0
        )

    def test_a_half_needs_at_least_one_correct_room(self):
        """With a single visited room "correct for all but one" is vacuous:
        ``score_q5('', QAContext(visited=('living_room',)))`` returned 0.5 —
        half credit for a blank answer — and only ``score_answer``'s guard, which
        no fixture stressed, turned it into 0."""
        one_room = qa_context(visited=("living_room",))
        assert scoring.score_q5("", one_room) == 0.0
        assert scoring.score_q5("A wardrobe, I think.", one_room) == 0.0
        assert scoring.score_q5("The blue rug in the living room.", one_room) == 1.0

    def test_an_unparseable_answer_scores_zero(self):
        assert scoring.score_answer(5, "", qa_context()) == 0.0


class TestQABlock:
    def test_the_five_questions_are_the_frozen_ones(self):
        assert [q.number for q in LAYOUT_QA_QUESTIONS] == [1, 2, 3, 4, 5]

    def test_scoring_the_golden_trials_answers(self):
        result = scoring.score_qa(successful_trial())
        assert result.scores == (1.0, 1.0, 1.0, 1.0, 1.0)
        assert result.score == 1.0

    def test_an_all_empty_qa_block_scores_zero_not_none(self):
        """doc 06 §5.9: an unparseable answer is logged "" and T4.1 scores it 0
        rather than the harness inventing text."""
        result = scoring.score_qa(successful_trial(qa_answers=[""] * 5))
        assert result.scores == (0.0,) * 5
        assert result.score == 0.0

    def test_a_malformed_qa_block_is_rejected(self):
        document = successful_trial()
        document["final"]["qa"] = document["final"]["qa"][:3]
        with pytest.raises(scoring.ScoringError, match="five frozen questions"):
            scoring.score_qa(document)


# ---------------------------------------------------------------------------
# §9.1 row 5 — Drift with/without corrections (doc 06 §5.8)
# ---------------------------------------------------------------------------


class TestDrift:
    def _declaring_trial(self, *, corrections=(), extra_correction=None) -> dict:
        builder = TrialBuilder()
        builder.turn(
            [
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        straight_trace((0.5, 0.5), (1.0, 1.0)), policy_seconds=4.0
                    ),
                }
            ],
            estimate=(0.5, 0.5),
        )
        builder.turn(
            list(corrections)
            + [
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        straight_trace((1.0, 1.0), (2.55, 0.75)), policy_seconds=8.0
                    ),
                },
                {"name": "declare_done", "args": {}},
            ],
            estimate=(1.2, 1.1),
            end_reason=REASON_DECLARE_DONE,
        )
        document = builder.finish(stage1_end=REASON_DECLARE_DONE)
        if extra_correction is not None:
            document["turns"][-1]["memory_snapshot"]["corrections"].append(
                extra_correction
            )
        return document

    def test_without_corrections_drift_is_the_raw_integrator_error(self):
        """The two halves are sampled at the SAME instant.

        Turn 2's ``obs.position_estimate`` was captured before the turn's move,
        so the true pose it must be compared against is the one that held at
        that moment — turn 1's logged ``true_pose`` (1.0, 1.0), NOT turn 2's
        post-move (2.55, 0.75). Pairing across that gap reported 1.3946 m where
        the integrator's real error was 0.2236 m, a 6.2x inflation that is just
        the length of the final move.
        """
        drift = scoring.stage_drift(self._declaring_trial(), STAGE_FIND_KITCHEN)
        assert drift.count == 0
        assert drift.magnitudes_m == ()
        assert drift.paired_at == "pre_dispatch"
        assert drift.true_xy == pytest.approx((1.0, 1.0))
        assert drift.drift_m == pytest.approx(0.223607, abs=1e-6)
        assert drift.drift_m == pytest.approx(math.dist((1.2, 1.1), (1.0, 1.0)))

    def test_a_perfect_dead_reckoner_scores_zero_drift(self):
        """The measurement that condemned the old pairing.

        Every turn's ``position_estimate`` equals ground truth at the instant
        ``obs`` was captured — a model whose dead reckoning is never wrong by a
        millimetre — and the declaration is bundled with the last move, which
        doc 05 §3.3 explicitly allows. The old pairing charged it 1.3583 m,
        exactly ``math.dist((1.2, 0.9), (2.55, 0.75))``: the final move.
        """
        builder = TrialBuilder()
        builder.turn(
            [
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        straight_trace((0.5, 0.5), (1.2, 0.9)), policy_seconds=4.0
                    ),
                }
            ],
            estimate=(0.5, 0.5),
        )
        builder.turn(
            [
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        straight_trace((1.2, 0.9), (2.55, 0.75)), policy_seconds=8.0
                    ),
                },
                {"name": "declare_done", "args": {}},
            ],
            estimate=(1.2, 0.9),
            end_reason=REASON_DECLARE_DONE,
        )
        document = builder.finish(stage1_end=REASON_DECLARE_DONE)
        assert scoring.stage_drift(document, STAGE_FIND_KITCHEN).drift_m == 0.0

    def test_turn_packing_does_not_change_the_drift_number(self):
        """The same navigation with `declare_done` bundled into the last move's
        turn, and split into a turn of its own, must report the same drift —
        otherwise the metric measures a phrasing habit, not localization."""
        packed = scoring.stage_drift(
            self._declaring_trial(), STAGE_FIND_KITCHEN
        ).drift_m

        builder = TrialBuilder()
        builder.turn(
            [
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        straight_trace((0.5, 0.5), (1.0, 1.0)), policy_seconds=4.0
                    ),
                }
            ],
            estimate=(0.5, 0.5),
        )
        builder.turn(
            [
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        straight_trace((1.0, 1.0), (2.55, 0.75)), policy_seconds=8.0
                    ),
                }
            ],
            estimate=(1.2, 1.1),
        )
        # The belief the SAME dead reckoner holds after that move: the packed
        # turn's pre-move estimate (1.2, 1.1) plus the move's own displacement
        # (1.55, -0.25). Its error against the true (2.55, 0.75) is unchanged —
        # which is the whole point: only the turn boundary moved.
        builder.turn(
            [{"name": "declare_done", "args": {}}],
            estimate=(2.75, 0.85),
            end_reason=REASON_DECLARE_DONE,
        )
        split = scoring.stage_drift(
            builder.finish(stage1_end=REASON_DECLARE_DONE), STAGE_FIND_KITCHEN
        ).drift_m
        assert split == pytest.approx(packed)

    def test_a_logged_end_of_turn_estimate_is_preferred_when_present(self):
        """``turns[].position_estimate_end`` is the post-dispatch integrator —
        §5.8's "at the moment of declare_done" verbatim, with no ambiguity about
        where in the turn a correction landed. Writing it is a `loop.py` change
        T4.1 reports rather than makes, so the key is optional; this fixture
        injects the exact shape the proposed edit would write."""
        document = self._declaring_trial()
        assert "position_estimate_end" not in document["turns"][-1], (
            "loop.py has gained the key: TrialBuilder must emit it too"
        )
        document["turns"][-1][scoring.POSITION_ESTIMATE_END] = {"x": 2.4, "y": 0.8}
        drift = scoring.stage_drift(document, STAGE_FIND_KITCHEN)
        assert drift.paired_at == "post_dispatch"
        assert drift.true_xy == pytest.approx((2.55, 0.75))
        assert drift.drift_m == pytest.approx(math.dist((2.4, 0.8), (2.55, 0.75)))

    def test_a_correction_on_the_declaring_turn_supersedes_the_estimate(self):
        document = self._declaring_trial(
            corrections=[
                {
                    "name": "correct_position",
                    "args": {"x": 1.9, "y": 0.9, "reason": "saw the counter"},
                }
            ]
        )
        drift = scoring.stage_drift(document, STAGE_FIND_KITCHEN)
        assert drift.used_correction
        assert drift.count == 1
        assert drift.magnitudes_m[0] == pytest.approx(math.dist((1.2, 1.1), (1.9, 0.9)))
        # The correction is the model's belief about the pose it was LOOKING at,
        # so it pairs with the same instant the superseded estimate did.
        assert drift.drift_m == pytest.approx(math.dist((1.9, 0.9), (1.0, 1.0)))

    def test_the_last_of_several_corrections_on_one_turn_wins(self):
        """§5.8 takes the LAST same-turn correction — "that call is what the
        model believed when it declared". With one correction per fixture the
        choice was unpinned; taking the first instead moves this stage's
        published drift by 4.2x."""
        document = self._declaring_trial(
            corrections=[
                {
                    "name": "correct_position",
                    "args": {"x": 1.9, "y": 0.9, "reason": "saw the counter"},
                },
                {
                    "name": "correct_position",
                    "args": {"x": 1.05, "y": 1.02, "reason": "no, the doorway"},
                },
            ]
        )
        drift = scoring.stage_drift(document, STAGE_FIND_KITCHEN)
        assert drift.count == 2
        assert drift.estimate_xy == pytest.approx((1.05, 1.02))
        assert drift.drift_m == pytest.approx(0.053852, abs=1e-6)

    def test_a_correction_after_declare_done_is_ignored(self):
        """§9.1. Filtering is by stage first — ``Correction.turn`` is stage-local,
        so only the stamped stage can split the series."""
        clean = scoring.stage_drift(self._declaring_trial(), STAGE_FIND_KITCHEN)
        later = scoring.stage_drift(
            self._declaring_trial(
                extra_correction={
                    "turn": 99,
                    "old_xy": [1.2, 1.1],
                    "new_xy": [2.55, 0.75],
                    "reason": "too late",
                    "stage": STAGE_FIND_KITCHEN,
                }
            ),
            STAGE_FIND_KITCHEN,
        )
        assert later.count == 0
        assert later.drift_m == pytest.approx(clean.drift_m)

    def test_a_stage_two_correction_never_lands_in_stage_ones_series(self):
        document = successful_trial()
        assert scoring.stage_drift(document, STAGE_FIND_KITCHEN).count == 1
        assert scoring.stage_drift(document, STAGE_RETURN_HOME).count == 0

    def test_an_unrun_stage_has_no_drift_number(self):
        drift = scoring.stage_drift(declared_elsewhere_trial(), STAGE_RETURN_HOME)
        assert drift.drift_m == NA
        assert drift.count == 0

    def test_drift_is_still_reported_for_a_capped_stage(self):
        """§5.8's rationale — "how honest the estimate ended up" — applies just
        as much to a cap-out, and dropping it would delete the metric for exactly
        the trials most worth explaining. Only an unrun stage is "—"."""
        builder = TrialBuilder()
        builder.turn(
            [
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        straight_trace((0.5, 0.5), (1.0, 1.0)), policy_seconds=4.0
                    ),
                }
            ],
            estimate=(0.7, 0.7),
            end_reason=REASON_TURN_CAP,
        )
        document = builder.finish(stage1_end=REASON_TURN_CAP)
        drift = scoring.stage_drift(document, STAGE_FIND_KITCHEN)
        # The stage's FIRST turn has no predecessor, so the true pose at the
        # instant `obs` was captured is the stage's own start — the spawn here.
        assert drift.true_xy == pytest.approx(spawn_pose(101)[0])
        assert drift.drift_m == pytest.approx(math.dist((0.7, 0.7), (0.5, 0.5)))

    def test_the_golden_trials_drift_is_the_belief_error_not_the_last_move(self):
        """End to end on the workhorse fixture. Stage 1 declares after a
        ``correct_position`` onto (2.2, 1.05) — which was the true pose at that
        instant — so the belief error is 0; the old pairing published 0.4810 m,
        exactly the length of the move that followed."""
        document = successful_trial()
        stage1 = scoring.stage_drift(document, STAGE_FIND_KITCHEN)
        assert stage1.used_correction and stage1.drift_m == pytest.approx(0.0)
        stage2 = scoring.stage_drift(document, STAGE_RETURN_HOME)
        assert stage2.drift_m == pytest.approx(0.053852, abs=1e-6)


# ---------------------------------------------------------------------------
# §9.1 row 6 — Schema / leak guard
# ---------------------------------------------------------------------------


def doc_schema_block() -> str:
    """doc 06 §4's annotated schema, pulled out of the HTML at test time."""
    source = DOC_06.read_text(encoding="utf-8")
    match = re.search(
        r'<h2 id="logs">.*?<pre><code>(.*?)</code></pre>', source, flags=re.DOTALL
    )
    assert match, "doc 06 §4's annotated schema block is no longer in the HTML"
    return html.unescape(match.group(1))


def schema_paths(block: str) -> set[str]:
    """Dotted key paths of doc 06 §4's schema.

    Arrays are transparent (``turns[]`` contributes nothing of its own), so
    ``turns.execution.calls.tool`` is the path of a key inside an array of
    objects inside an array of objects.
    """
    text = re.sub(r"//[^\n]*", "", block)
    token = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:|[{}\[\],]')
    paths: set[str] = set()
    stack: list[tuple[str, str | None]] = []
    pending: str | None = None
    for match in token.finditer(text):
        piece = match.group(0)
        if piece.endswith(":"):
            prefix = [name for kind, name in stack if kind == "obj" and name]
            paths.add(".".join(prefix + [match.group(1)]))
            pending = match.group(1)
        elif piece == "{":
            stack.append(("obj", pending))
            pending = None
        elif piece == "[":
            stack.append(("arr", None))  # transparent
        elif piece in "}]":
            if stack:
                stack.pop()
            pending = None
        elif piece == ",":
            pending = None
    return paths


def resolve_path(document, path: str) -> bool:
    """Does ``path`` exist in ``document``? Lists are searched element-wise."""
    if isinstance(document, list):
        return any(resolve_path(item, path) for item in document)
    head, _, tail = path.partition(".")
    if not isinstance(document, dict) or head not in document:
        return False
    return True if not tail else resolve_path(document[head], tail)


#: The one doc path that is an EXAMPLE rather than a schema key: §4 illustrates
#: ``tool_calls`` with ``{"name": "move", "args": {"distance_m": 1.0}}``, and
#: ``distance_m`` is ``move``'s argument, not a field every tool call carries.
SCHEMA_EXAMPLE_ONLY = {"turns.model_output.tool_calls.args.distance_m"}
DOC_SCHEMA_PATHS = sorted(schema_paths(doc_schema_block()) - SCHEMA_EXAMPLE_ONLY)


class TestSchemaConformance:
    def test_the_schema_block_is_still_extractable(self):
        """The tripwire is set just under the MEASURED path count, not at a
        third of it. doc 06's HTML is edited concurrently; if the ``<h2
        id="logs">…<pre><code>`` regex ever latched onto a smaller block in the
        same section, a threshold of 40 would stay green while two thirds of §4's
        contract silently stopped being checked. [measured 2026-07-26: the
        extractor yields 107 paths after the one deliberate exclusion.]"""
        assert '"trial_id"' in doc_schema_block()
        assert len(DOC_SCHEMA_PATHS) >= 100

    @pytest.mark.parametrize("path", DOC_SCHEMA_PATHS)
    def test_every_doc_06_section_4_key_exists_in_the_fixture(self, path):
        assert resolve_path(successful_trial(), path), (
            f"doc 06 §4 documents {path!r} but the fixture has no such key"
        )

    def test_the_committed_golden_trial_also_conforms(self):
        document = json.loads(GOLDEN_TRIAL.read_text(encoding="utf-8"))
        missing = [p for p in DOC_SCHEMA_PATHS if not resolve_path(document, p)]
        assert missing == [], f"{GOLDEN_TRIAL.name} is stale: {missing}"

    def test_execution_is_always_an_object_never_null(self):
        for turn in successful_trial()["turns"]:
            assert isinstance(turn["execution"], dict)
            assert "pose_trace" in turn["execution"]


class TestLeakGuard:
    def test_the_rendered_memory_block_contains_no_true_pose_values(self):
        """doc 06 §4's gotcha: ``true_pose`` lives in the file the scorer reads
        but must never reach the prompt path.

        A true-pose coordinate that happens to equal the *estimate* is exempt:
        the integrator is anchored at the spawn (doc 05 §5.2's declared t=0
        exception), so before the first motion the two agree and the block is
        showing the estimate, not ground truth. Every other coordinate must be
        absent, at either rendering precision.
        """
        leaked = []
        for turn in successful_trial()["turns"]:
            block = turn["memory_snapshot"]["block"]
            pose = turn["true_pose"]
            shown = turn["obs"]["position_estimate"]
            for value in (pose["x"], pose["y"]):
                if any(
                    math.isclose(value, seen, abs_tol=5e-3)
                    for seen in (shown["x"], shown["y"])
                ):
                    continue
                for rendered in (f"{value:.2f}", f"{value:.4f}"):
                    if rendered in block:
                        leaked.append((turn["global_turn_idx"], rendered))
        assert leaked == []

    def test_the_qa_prompt_contains_no_true_pose_values(self):
        last = successful_trial()["turns"][-1]
        prompt = render_qa_prompt(last["memory_snapshot"]["block"])
        for value in (last["true_pose"]["x"], last["true_pose"]["y"]):
            assert f"{value:.2f}" not in prompt

    def test_the_pose_trace_is_never_shown_to_the_model(self):
        for turn in successful_trial()["turns"]:
            block = turn["memory_snapshot"]["block"]
            for point in turn["execution"]["pose_trace"]:
                assert f"{point[0]:.2f}, {point[1]:.2f}" not in block


class TestResumeCheck:
    def test_a_trial_without_final_is_rejected(self, tmp_path):
        builder = TrialBuilder()
        builder.turn([{"name": "look_around", "args": {}}])
        path = tmp_path / "incomplete.json"
        path.write_text(json.dumps(builder.document), encoding="utf-8")
        assert not scoring.is_complete(builder.document)
        with pytest.raises(scoring.IncompleteTrialError):
            scoring.load_trial(path)

    def test_an_infra_failed_trial_is_rejected_even_with_a_final(self, tmp_path):
        """``note_infra_failure`` deliberately writes no ``final`` — but if one
        ever appeared, the trial still reruns whole (doc 05 §8)."""
        document = successful_trial()
        document["infra_failure"] = "APIConnectionError"
        path = tmp_path / "infra.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(scoring.IncompleteTrialError):
            scoring.load_trial(path)

    def test_a_complete_trial_loads(self, tmp_path):
        path = tmp_path / "ok.json"
        path.write_text(json.dumps(successful_trial()), encoding="utf-8")
        assert scoring.load_trial(path)["trial_id"]

    def test_score_trial_refuses_an_incomplete_document(self):
        builder = TrialBuilder()
        builder.turn([{"name": "look_around", "args": {}}])
        with pytest.raises(scoring.IncompleteTrialError):
            scoring.score_trial(builder.document)


# ---------------------------------------------------------------------------
# §9.1 row 7 — Stage-2 gate (added by T3.4)
# ---------------------------------------------------------------------------


def declared_elsewhere_trial() -> dict:
    """``declare_done`` in the hallway: stage 1 declared_elsewhere, stage 2 not_run."""
    builder = TrialBuilder()
    builder.turn(
        [
            {
                "name": "move",
                "args": {},
                "execution": motion_execution(
                    straight_trace((0.5, 0.5), (0.9, 2.9)), policy_seconds=12.0
                ),
            },
            {"name": "declare_done", "args": {}},
        ],
        estimate=(0.6, 2.8),
        end_reason=REASON_DECLARE_DONE,
    )
    return builder.finish(stage1_end=REASON_DECLARE_DONE, stage2_end="not_run")


class TestStage2Gate:
    def test_declaring_outside_the_radius_stops_the_trial(self):
        document = declared_elsewhere_trial()
        assert document["final"]["outcome"][STAGE_FIND_KITCHEN] == "declared_elsewhere"
        assert document["final"]["outcome"][STAGE_RETURN_HOME] == "not_run"
        assert scoring.score_trial(document).stages[STAGE_FIND_KITCHEN].success is False

    def test_an_unrun_stage_two_keeps_the_denominator_and_scores_zero(self):
        metrics = scoring.score_trial(declared_elsewhere_trial())
        stage2 = metrics.stages[STAGE_RETURN_HOME]
        assert stage2.outcome == "not_run"
        assert stage2.progress == 0.0
        assert stage2.spl == 0.0
        assert stage2.time_s == NA
        assert stage2.drift_m == NA
        assert stage2.turns_used == 0

        summary = scoring.summarise("fable5", [metrics], resamples=200, seed=1)
        assert summary[STAGE_RETURN_HOME]["success_rate"]["n"] == 1
        assert summary[STAGE_RETURN_HOME]["success_rate"]["printed"] == "0/1"

    def test_the_na_cells_are_excluded_from_the_means(self):
        metrics = scoring.score_trial(declared_elsewhere_trial())
        summary = scoring.summarise("fable5", [metrics], resamples=200, seed=1)
        assert summary[STAGE_RETURN_HOME]["time_s"]["mean"] == NA
        assert summary[STAGE_RETURN_HOME]["time_s"]["n_defined"] == 0
        assert summary[STAGE_RETURN_HOME]["drift_m"]["mean"] == NA

    def test_the_conditional_return_home_rate_is_reported_with_its_k(self):
        """doc 06 §3.2: x/k over the stage-1 successes, k printed, "—" when k = 0."""
        failed = scoring.score_trial(declared_elsewhere_trial())
        summary = scoring.summarise("fable5", [failed], resamples=200, seed=1)
        conditional = summary[STAGE_RETURN_HOME]["success_rate_given_stage1"]
        assert conditional["n"] == 0
        assert conditional["rate"] == NA
        assert conditional["printed"] == NA

        ok = scoring.score_trial(successful_trial())
        summary = scoring.summarise("fable5", [ok, failed], resamples=200, seed=1)
        assert summary[STAGE_RETURN_HOME]["success_rate"]["printed"] == "1/2"
        assert (
            summary[STAGE_RETURN_HOME]["success_rate_given_stage1"]["printed"] == "1/1"
        )

    def test_exactly_the_radius_is_a_success_because_within_is_inclusive(self):
        """§9.1(iii): "the boundary at exactly 0.35 m is a success".

        Probed at the origin rather than at the real goal, because
        ``2.55 + 0.35`` is ``2.9000000000000004`` in binary floating point and
        the distance back comes out at 0.3500000000000001 — that would test IEEE
        754, not the predicate. The radius under test is still the layout's.
        """
        radius = find_kitchen_spec().success_radius_m
        assert radius == LAYOUT["target"]["radius"] == 0.35
        probe = StageSpec(
            name="boundary_probe",
            objective="",
            goal_xy=(0.0, 0.0),
            success_radius_m=radius,
            goal_label="probe",
        )
        assert math.dist((radius, 0.0), (0.0, 0.0)) == radius
        assert score_stage(probe, (radius, 0.0)).success is True
        assert score_stage(probe, (math.nextafter(radius, 1.0), 0.0)).success is False
        assert score_stage(find_kitchen_spec(), target_point()).success is True

    def test_the_scorer_consults_the_same_predicate_the_live_loop_used(self):
        """doc 06 §9.1(iii): the scorer and the gate cannot disagree. A log whose
        success flag contradicts ``score_stage`` is rejected, not published."""
        document = successful_trial()
        document["final"]["stages"][STAGE_FIND_KITCHEN]["success"] = False
        with pytest.raises(scoring.ScoringError, match="declare_done"):
            scoring.score_trial(document)

    def test_a_tampered_distance_is_rejected(self):
        document = successful_trial()
        document["final"]["stages"][STAGE_FIND_KITCHEN]["score"]["distance_m"] = 0.01
        with pytest.raises(scoring.ScoringError, match="disagrees"):
            scoring.score_trial(document)

    def test_arriving_without_declaring_is_not_a_success(self):
        """§5.1: the model must KNOW it arrived. A cap-out inside the radius logs
        ``score.success = true`` and ``stages.success = false``; SR reads the
        latter, and reading the former would inflate the headline number."""
        builder = TrialBuilder()
        builder.turn(
            [
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        straight_trace((0.5, 0.5), (2.55, 0.75)), policy_seconds=20.0
                    ),
                }
            ],
            estimate=(2.4, 0.8),
            end_reason=REASON_TURN_CAP,
        )
        document = builder.finish(stage1_end=REASON_TURN_CAP)
        stage = document["final"]["stages"][STAGE_FIND_KITCHEN]
        assert stage["score"]["success"] is True
        assert stage["success"] is False
        metrics = scoring.score_trial(document)
        assert metrics.stages[STAGE_FIND_KITCHEN].success is False
        assert metrics.stages[STAGE_FIND_KITCHEN].spl == 0.0
        assert metrics.stages[STAGE_FIND_KITCHEN].time_s == NA

    def test_the_per_stage_start_pose_is_not_the_spawn_for_return_home(self):
        """Read literally, §5.2/§5.3's "from spawn to target" makes stage 2's
        d_initial and oracle length zero — contradicting §9.1's own stage-2 case
        and doc 05's measured 1.574 m floor."""
        stage2 = scoring.score_trial(successful_trial()).stages[STAGE_RETURN_HOME]
        assert stage2.d_initial_m > 1.0
        assert stage2.oracle_path_m != NA and stage2.oracle_path_m > 1.0


# ---------------------------------------------------------------------------
# 5.6 Bumps and falls
# ---------------------------------------------------------------------------


class TestBumpsAndFalls:
    def test_bumps_are_trial_scoped_and_agree_with_the_per_call_flag(self):
        document = successful_trial()
        assert document["final"]["bumps"] == 1
        assert scoring.bumps(document) == 1

    def test_a_bump_total_that_contradicts_the_turns_is_rejected(self):
        document = successful_trial()
        document["final"]["bumps"] = 5
        with pytest.raises(scoring.ScoringError, match="counted_as_bump"):
            scoring.bumps(document)

    def test_turn_to_heading_collisions_are_deliberately_not_counted(self):
        """doc 06 §5.6's T3.2 pin: ``PolicyPlayback._bump_run`` is not reset
        between calls, so counting rotations would score the canonical
        bump-then-turn-away recovery as three collisions."""
        builder = TrialBuilder()
        builder.turn(
            [
                {
                    "name": "turn_to_heading",
                    "args": {"heading_deg": 0},
                    "execution": motion_execution(
                        [(0.5, 0.5), (0.5, 0.5)],
                        policy_seconds=1.5,
                        bumped=True,
                        counted_as_bump=False,
                    ),
                }
            ]
        )
        document = builder.finish(stage1_end=REASON_TURN_CAP)
        assert scoring.bumps(document) == 0

    def test_falls_are_zero_or_one(self):
        assert scoring.falls(successful_trial()) == 0
        document = successful_trial()
        document["final"]["end_reason"][STAGE_FIND_KITCHEN] = "fall"
        assert scoring.falls(document) == 1


# ---------------------------------------------------------------------------
# 6. Statistics — mean ± bootstrap 95% CI (doc 06 §6)
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_the_bootstrap_constants_come_from_the_frozen_config(self):
        config = scoring.scoring_config()
        assert config["bootstrap_resamples"] == 10000
        assert config["bootstrap_seed"] == 20260726
        assert config["find_kitchen_success_radius_m"] == LAYOUT["target"]["radius"]
        assert config["return_home_success_radius_m"] == LAYOUT["return_home_radius"]

    def test_the_frozen_config_is_read_only(self):
        """``lru_cache`` hands every caller the SAME dict, so one accidental
        write would re-seed every subsequent bootstrap in the run and the
        published intervals would stop being reproducible from the committed
        YAML. Measured before the proxy: ``c['bootstrap_seed'] = 999999``
        changed what every later call returned."""
        config = scoring.scoring_config()
        assert scoring.scoring_config() is config
        with pytest.raises(TypeError):
            config["bootstrap_seed"] = 999999
        assert scoring.scoring_config()["bootstrap_seed"] == 20260726

    def test_the_interval_is_reproducible_for_a_fixed_seed(self):
        values = [0.2, 0.6, 0.9, 1.0]
        first = scoring.bootstrap_ci(values, resamples=2000, seed=20260726)
        assert first == scoring.bootstrap_ci(values, resamples=2000, seed=20260726)

    def test_the_interval_is_pinned_to_a_value_not_merely_to_a_property(self):
        """Every other assertion here is a PROPERTY (brackets the mean, constant
        sample is degenerate, seeds agree at 10k) — and a bootstrap that
        resampled WITHOUT replacement satisfies all of them while collapsing the
        interval to zero width. Measured: swapping ``rng.choice`` for
        ``rng.sample`` turned (0.375, 0.95) into (0.675, 0.675) and the whole
        suite stayed green."""
        assert scoring.bootstrap_ci(
            [0.2, 0.6, 0.9, 1.0], resamples=10000, seed=20260726
        ) == pytest.approx((0.375, 0.95))

    def test_the_resampling_is_with_replacement(self):
        """A permutation resample makes every resample mean the SAMPLE mean, so
        the interval collapses. Two independent witnesses, neither of which a
        without-replacement bootstrap can satisfy."""
        values = [0.2, 0.6, 0.9, 1.0]
        low, high = scoring.bootstrap_ci(values, resamples=2000, seed=20260726)
        assert high - low > 0.3
        assert low < sum(values) / len(values) < high

    def test_the_seed_really_drives_the_resampling(self):
        """Determinism alone would also hold for a scorer that ignored the seed,
        so prove it is used: at a low resample count the intervals differ — by a
        real margin, not by the 1-ULP summation-order noise that a broken
        bootstrap also produces."""
        values = [0.2, 0.6, 0.9, 1.0]
        first = scoring.bootstrap_ci(values, resamples=50, seed=1)
        second = scoring.bootstrap_ci(values, resamples=50, seed=7)
        assert abs(first[0] - second[0]) > 1e-6
        assert abs(first[1] - second[1]) > 1e-6

    def test_the_defaults_are_the_frozen_config_values(self):
        """The path the real 12-trial batch runs through had ZERO coverage:
        every CI test passed ``seed=``/``resamples=`` explicitly, and the
        constants test only asserted what the YAML holds, never that
        ``bootstrap_ci`` uses it. Measured: defaulting to ``seed = 424242`` or
        ``resamples = 200`` instead survived the whole suite, and both move the
        published interval at N=12."""
        config = scoring.scoring_config()
        values = [0.0, 0.13, 0.25, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        assert scoring.bootstrap_ci(values) == scoring.bootstrap_ci(
            values,
            resamples=config["bootstrap_resamples"],
            seed=config["bootstrap_seed"],
        )
        assert scoring.bootstrap_ci(values) != scoring.bootstrap_ci(
            values, seed=config["bootstrap_seed"] + 1
        )
        assert scoring.bootstrap_ci(values) != scoring.bootstrap_ci(
            values, resamples=200
        )
        assert scoring.estimate(values).ci == scoring.bootstrap_ci(values)

    def test_at_the_locked_resample_count_the_interval_is_not_a_seed_artifact(self):
        """[measured 2026-07-26] With N=4 and 10,000 resamples the 2.5/97.5
        percentiles land on the same values for every seed tried — the locked
        seed buys exact reproducibility, and the reported width is a property of
        the data rather than of the RNG. Worth stating in the write-up."""
        values = [0.2, 0.6, 0.9, 1.0]
        intervals = {
            scoring.bootstrap_ci(values, resamples=10000, seed=seed)
            for seed in (1, 7, 20260726)
        }
        assert len(intervals) == 1

    def test_the_interval_brackets_the_mean(self):
        values = [0.2, 0.6, 0.9, 1.0]
        low, high = scoring.bootstrap_ci(values, resamples=2000, seed=20260726)
        assert low <= sum(values) / len(values) <= high

    def test_a_constant_sample_has_a_degenerate_interval(self):
        low, high = scoring.bootstrap_ci([0.5] * 4, resamples=500, seed=1)
        assert low == pytest.approx(0.5) and high == pytest.approx(0.5)

    def test_no_interval_is_reported_below_three_defined_values(self):
        """doc 06 §3.2: "a bootstrap over one value is theatre". §3.2 states it
        for the conditional return-home SR; it applies to every metric."""
        assert scoring.bootstrap_ci([0.5], resamples=500, seed=1) is None
        assert scoring.bootstrap_ci([0.5, 0.7], resamples=500, seed=1) is None
        assert scoring.bootstrap_ci([0.5, 0.7, 0.9], resamples=500, seed=1) is not None

    def test_percentiles_are_linearly_interpolated(self):
        assert scoring.percentile([0.0, 1.0, 2.0, 3.0], 50.0) == pytest.approx(1.5)
        assert scoring.percentile([0.0, 1.0], 2.5) == pytest.approx(0.025)

    def test_undefined_cells_are_dropped_before_the_mean(self):
        result = scoring.estimate([1.0, NA, 3.0, 5.0], resamples=500, seed=1)
        assert result.mean == pytest.approx(3.0)
        assert result.as_dict()["n_defined"] == 3
        assert result.as_dict()["n_total"] == 4


# ---------------------------------------------------------------------------
# End to end: every published number for one trial
# ---------------------------------------------------------------------------


class TestGoldenTrialEndToEnd:
    def test_every_metric_of_the_successful_trial(self):
        metrics = scoring.score_trial(successful_trial())
        stage1 = metrics.stages[STAGE_FIND_KITCHEN]
        spawn, _ = spawn_pose(101)

        assert stage1.success is True
        assert stage1.d_initial_m == pytest.approx(math.dist(spawn, target_point()))
        assert stage1.d_final_m == pytest.approx(0.03, abs=1e-6)
        assert stage1.progress == pytest.approx(
            scoring.progress(stage1.d_initial_m, stage1.d_final_m)
        )
        assert stage1.progress > 0.98
        assert stage1.time_s == pytest.approx(12.5)
        assert stage1.turns_used == 4
        # §9.1's "p < l ⇒ ratio capped at 1.0", exercised end to end: the
        # fixture walks straight lines while the oracle threads a 5 cm grid, so
        # the true path really is shorter than the oracle here. Pinned to
        # LITERALS, not restated from the implementation's own outputs: the old
        # `spl == oracle / max(p, oracle)` assertion was true by construction for
        # ANY values of l and p, so corrupting either survived it.
        assert stage1.oracle_path_m == pytest.approx(2.3935, abs=1e-3)
        assert stage1.true_path_m == pytest.approx(2.2985, abs=1e-3)
        assert stage1.true_path_m < stage1.oracle_path_m
        assert stage1.spl == 1.0
        assert stage1.corrections == 1

        stage2 = metrics.stages[STAGE_RETURN_HOME]
        assert stage2.success is True
        assert stage2.turns_used == 2
        assert stage2.time_s == pytest.approx(11.5)

        assert metrics.bumps == 1
        assert metrics.falls == 0
        assert metrics.visited_rooms == ("living_room", "kitchen")
        assert metrics.qa.score == 1.0

    def test_a_wandering_success_scores_strictly_below_the_clamp(self):
        """The fixture that makes SPL sensitive at all.

        ``successful_trial`` has ``p < l`` in BOTH stages, so its SPL sits on the
        ``max(p, l)`` clamp and is insensitive to the oracle, to the walked path,
        and to the wiring between them: inflating ``l`` by 1.5x, halving ``p``,
        or never passing ``p`` to ``spl`` at all each survived the whole suite.
        Here ``p > l``, and all three constants are pinned to independently
        computed values.

        ``p`` is the sum of five straight legs:
        2.0396 + 0.4 + 1.65 + 0.5 + 1.65 = 6.2396 m.
        """
        metrics = scoring.score_trial(wandering_success_trial())
        stage1 = metrics.stages[STAGE_FIND_KITCHEN]
        assert stage1.success is True
        legs = [(0.5, 0.5), (0.9, 2.5), (0.9, 2.9), (2.55, 2.9), (2.55, 2.4), (2.55, 0.75)]
        by_hand = sum(math.dist(a, b) for a, b in zip(legs, legs[1:]))
        assert by_hand == pytest.approx(6.2396, abs=1e-3)
        assert stage1.true_path_m == pytest.approx(by_hand, abs=1e-9)
        assert stage1.true_path_m == pytest.approx(6.2396, abs=1e-3)
        assert stage1.oracle_path_m == pytest.approx(2.3935, abs=1e-3)
        assert stage1.oracle_path_m == pytest.approx(
            oracle_length(spawn_pose(101)[0], target_point()), abs=1e-9
        )
        assert stage1.spl == pytest.approx(0.3836, abs=1e-3)
        assert stage1.spl < 1.0, "this fixture exists to be off the clamp"
        assert metrics.visited_rooms == ("living_room", "hallway", "kitchen")
        # The PUBLISHED precision, which only an off-the-clamp SPL can pin:
        # rounding to 1 dp is invisible on a trial whose SPL is exactly 1.0.
        assert metrics.as_dict()["stages"][STAGE_FIND_KITCHEN]["spl"] == 0.3836

    def test_the_metrics_dict_is_json_serialisable_for_final_metrics(self):
        payload = scoring.score_trial(successful_trial()).as_dict()
        round_tripped = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        assert round_tripped["stages"][STAGE_RETURN_HOME]["stage"] == STAGE_RETURN_HOME
        # The audit trail T4.4/T4.5 and a human reviewer read to adjudicate a
        # disputed score. Emptying either of these is not a wrong number, but it
        # is a lost explanation — and both survived the suite unasserted.
        stage1 = round_tripped["stages"][STAGE_FIND_KITCHEN]
        assert stage1["correction_magnitudes_m"] == [0.0566]
        assert round_tripped["map_accuracy"]["matches"] == [
            ["living room", "living_room"],
            ["kitchen", "kitchen"],
        ]
        assert round_tripped["map_accuracy"]["edges_unresolved"] == 0
        # …and the rounding precision itself: 4 decimals, not fewer.
        assert stage1["d_initial_m"] == 2.0652

    def test_the_committed_golden_trial_scores_identically(self):
        """The committed artifact exists so T4.4/T4.5 and a human reviewer have
        one concrete file to read; it must agree with the builder."""
        committed = json.loads(GOLDEN_TRIAL.read_text(encoding="utf-8"))
        assert (
            scoring.score_trial(committed).as_dict()
            == scoring.score_trial(successful_trial()).as_dict()
        )

    def test_a_summary_over_four_trials_carries_its_ns_and_intervals(self):
        trials = [scoring.score_trial(successful_trial()) for _ in range(4)]
        summary = scoring.summarise("fable5", trials, resamples=500, seed=1)
        assert summary["n_trials"] == 4
        assert summary[STAGE_FIND_KITCHEN]["success_rate"]["printed"] == "4/4"
        assert summary["qa"]["mean"] == pytest.approx(1.0)
        assert summary[STAGE_FIND_KITCHEN]["spl"]["ci95"] is not None

    def test_the_success_rate_carries_an_interval_like_every_other_column(self):
        """doc 06 §10's README table asks for "SR (both stages), progress, SPL,
        … each as mean ± 95 % CI". SR shipped as a bare ``x/N`` ratio with no
        interval and no per-trial column, so T4.4 had no sanctioned way to draw
        its whisker."""
        good = scoring.score_trial(successful_trial())
        bad = scoring.score_trial(declared_elsewhere_trial())
        summary = scoring.summarise("fable5", [good, bad, good, bad], resamples=500, seed=1)
        rate = summary[STAGE_FIND_KITCHEN]["success_rate"]
        assert rate["printed"] == "2/4" and rate["rate"] == 0.5
        assert rate["mean"] == pytest.approx(0.5)
        assert rate["ci95"] is not None and rate["n_defined"] == 4
        # …and §3.2's k < 3 rule needs no special case: it falls out of the same
        # MIN_CI_VALUES gate every other column uses.
        small = scoring.summarise("fable5", [good, bad], resamples=500, seed=1)
        assert small[STAGE_FIND_KITCHEN]["success_rate"]["ci95"] is None
        assert small[STAGE_FIND_KITCHEN]["success_rate"]["printed"] == "1/2"
        conditional = small[STAGE_RETURN_HOME]["success_rate_given_stage1"]
        assert conditional["printed"] == "1/1" and conditional["ci95"] is None

    def test_the_figures_and_the_table_are_fed_by_one_function(self):
        """``charts.bar_with_ci`` is annotated ``dict[str, Estimate]`` while
        ``summarise`` returns plain dicts, so T4.4 could only satisfy the stub by
        re-deriving the columns itself — the exact duplication the stub exists to
        prevent. ``metric_estimates`` is the shared accessor; this asserts the
        two agree cell for cell."""
        trials = [scoring.score_trial(successful_trial()) for _ in range(4)]
        estimates = scoring.metric_estimates(trials, resamples=500, seed=1)
        summary = scoring.summarise("fable5", trials, resamples=500, seed=1)
        assert isinstance(estimates["find_kitchen.spl"], scoring.Estimate)
        assert estimates["find_kitchen.spl"].as_dict() == summary[STAGE_FIND_KITCHEN]["spl"]
        assert estimates["qa"].as_dict() == summary["qa"]
        assert estimates["return_home.drift_m"].as_dict() == (
            summary[STAGE_RETURN_HOME]["drift_m"]
        )
        for stage in (STAGE_FIND_KITCHEN, STAGE_RETURN_HOME):
            for metric, cell in summary[stage].items():
                if metric == "success_rate_given_stage1":
                    continue
                key = f"{stage}.{metric}"
                assert key in estimates, f"{key} is published but not accessible"
                assert cell.items() >= estimates[key].as_dict().items()


# ---------------------------------------------------------------------------
# Package separation: the agent side must never reach the scorer
# ---------------------------------------------------------------------------


class TestVisitedRooms:
    """§2.10's true trace: the union of the 5 Hz samples, the per-turn poses and
    the spawn — trial-scoped, with no dwell threshold. Both are conventions the
    doc leaves open, so both are pinned here rather than left to drift."""

    def test_the_visited_set_spans_the_whole_trial_not_just_stage_one(self):
        """A stage-scoped reading would shrink §5.7's recall denominator and
        Q3/Q5's gold room set for every trial that goes home a different way."""
        document = returned_via_hallway_trial()
        assert scoring.visited_rooms(document) == (
            "living_room",
            "kitchen",
            "hallway",
        )
        stage1_only = scoring.true_trace(document, STAGE_FIND_KITCHEN)
        assert "hallway" not in {
            room_at(x, y) for x, y in stage1_only if room_at(x, y)
        }, "the hallway must be reachable ONLY through the stage-2 leg"

    def test_the_gold_room_set_follows_the_whole_trial(self):
        context = scoring.qa_context(returned_via_hallway_trial())
        assert context.visited == ("living_room", "kitchen", "hallway")
        assert (
            scoring.score_q3("I visited two rooms: the living room and the kitchen.", context)
            == 0.5
        )
        accuracy = scoring.map_accuracy(returned_via_hallway_trial())
        assert accuracy.true_rooms_visited == 3

    def test_a_single_doorway_sample_counts_as_a_visit(self):
        """DECIDED, not discovered later: there is no dwell threshold, so one
        0.2 s sample at a doorway centre adds a room. ``room_at``'s bounds are
        half-open, so the doorway centre (2.55, 2.7) belongs to the hallway —
        4 cm of trajectory decides a published denominator. Recorded in
        ``docs/METRICS.md`` §2.10; this fixture is what stops the convention
        changing by accident."""
        assert room_at(2.55, 2.7) == "hallway"
        builder = TrialBuilder()
        builder.turn(
            [
                {
                    "name": "move",
                    "args": {},
                    "execution": motion_execution(
                        [(2.4, 2.5), (2.55, 2.7), (2.7, 2.5)], policy_seconds=2.0
                    ),
                }
            ],
            end_reason=REASON_TURN_CAP,
        )
        document = builder.finish(stage1_end=REASON_TURN_CAP)
        trace = scoring.true_trace(document)
        assert sum(1 for x, y in trace if room_at(x, y) == "hallway") == 1
        assert "hallway" in scoring.visited_rooms(document)


class TestChartsSkeleton:
    """PLAN T4.1 shipped a ``charts.py`` skeleton; T4.4 filled it in.

    The point of testing the skeleton was the contract: the signatures are what
    stop T4.4 from re-deriving a metric that would then be free to disagree with
    the table beside it. With T4.4 landed, the surviving contract is those
    pinned signatures plus the matplotlib-free import; the stub-honesty
    assertion (``raises NotImplementedError``) retired with the stubs it
    described. ``tests/test_charts.py`` covers the implementation's pure
    helpers.
    """

    def test_importing_charts_does_not_pull_in_a_plotting_stack(self):
        import sys

        sys.modules.pop("matplotlib", None)
        from duck_embody import charts

        assert "matplotlib" not in sys.modules
        assert charts.MODEL_ORDER == ("fable5", "opus5", "gpt56sol")

    def test_the_figure_entry_points_keep_the_pinned_signatures(self):
        import inspect

        from duck_embody import charts

        assert list(inspect.signature(charts.bar_with_ci).parameters) == [
            "metric", "estimates", "out_path", "ylabel",
        ]
        assert list(inspect.signature(charts.trajectory_vs_belief).parameters) == [
            "trial", "document", "out_path",
        ]
        assert list(inspect.signature(charts.per_trial_table).parameters) == [
            "trials",
        ]
        # per_trial_table is pure: callable with no plotting stack at all.
        header = charts.per_trial_table([]).splitlines()[0]
        assert header.startswith("| Trial | Model |")


def imported_module_names(source: str, package: str) -> set[str]:
    """Every module name ``source`` imports, with relative imports resolved.

    ``package`` is the dotted package the source file lives in, so
    ``from .. import scoring`` inside ``duck_embody.agent.providers`` resolves to
    ``duck_embody.scoring``. Both halves of an ``ImportFrom`` count: the module
    path AND each imported name, because ``from duck_embody import scoring``
    carries the module in the *alias*, not in ``node.module``.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parts = package.split(".")
            base = (
                ".".join(parts[: len(parts) - node.level + 1])
                if node.level
                else ""
            )
            module = ".".join(filter(None, [base, node.module or ""]))
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            target = node.func
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                names.add(node.args[0].value)
    return names


class TestPackageSeparation:
    #: Every syntactic form that actually imports the scorer at runtime. The
    #: detector is what is under test here, not the current source tree — which
    #: is clean today, so a detector that caught nothing would also pass.
    LEAK_FORMS = [
        "import duck_embody.scoring",
        "import duck_embody.scoring as s",
        "from duck_embody.scoring import spl",
        "from duck_embody import scoring",
        "from duck_embody import scoring as sc",
        "from .. import scoring",
        "from ..scoring import spl",
        "import importlib\nimportlib.import_module('duck_embody.scoring')",
    ]

    @pytest.mark.parametrize("form", LEAK_FORMS)
    def test_the_detector_catches_every_way_of_reaching_the_scorer(self, form):
        """Four of these were MISSED by inspecting ``node.module`` alone —
        including ``from duck_embody import scoring``, the form a person writes
        by hand. A guard that only stops the forms nobody uses is not a guard."""
        names = imported_module_names(form, "duck_embody.agent")
        assert any("duck_embody.scoring" in name for name in names), form

    def test_the_detector_does_not_flag_the_packages_own_imports(self):
        clean = "from duck_embody.agent.memory import Memory\nfrom . import tools\n"
        names = imported_module_names(clean, "duck_embody.agent")
        assert not any("duck_embody.scoring" in name for name in names)

    def test_the_agent_package_does_not_import_scoring(self):
        """``scoring`` reads ground truth by design. A path from the model-facing
        package to it is one refactor away from leaking the answer key into a
        prompt."""
        root = REPO_ROOT / "duck_embody" / "agent"
        offenders: list[str] = []
        scanned = 0
        for path in sorted(root.rglob("*.py")):
            scanned += 1
            package = ".".join(
                path.relative_to(REPO_ROOT).with_suffix("").parts[:-1]
            )
            names = imported_module_names(path.read_text(encoding="utf-8"), package)
            if any("duck_embody.scoring" in name for name in names):
                offenders.append(str(path.relative_to(REPO_ROOT)))
        assert scanned > 5, "the agent package moved; this guard scanned nothing"
        assert offenders == []

    def test_the_scorer_reuses_the_frozen_tables_rather_than_copying_them(self):
        """doc 06 §5.7/§5.9: import the frozen synonym table and ``compass_8``,
        do not author a second one."""
        source = (REPO_ROOT / "duck_embody" / "scoring.py").read_text(encoding="utf-8")
        assert "from duck_embody.agent.prompts import" in source
        assert "compass_8" in source
        assert scoring.normalize_claim("lounge") == "living_room"
        assert scoring.normalize_claim("kitchenette") is None
