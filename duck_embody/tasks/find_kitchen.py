"""The two stages of a trial: spawn, goals, and the success predicates.

``find_kitchen`` ("Find the kitchen and walk to the counter") and its
continuation ``return_home`` are defined here — not in :mod:`loop`, and not in
:mod:`duck_embody.scoring` — because **one predicate has two consumers**. The
live loop consults it at ``declare_done`` to decide whether the trial gets a
return leg (see :data:`STAGE2_REQUIRES_STAGE1_SUCCESS`), and T4.1's post-hoc
scorer recomputes the same verdict from the trial JSON. If those two ever
computed the distance differently, a trial could be logged
``find_kitchen: success`` (and run a stage 2) while the published table said
failure, with nothing raising. Both call :func:`score_stage`.

**That is exactly what happened between 2026-07-27 and TR.2, in the other
direction.** Criterion v2 was adopted in ``scoring.py`` alone while this
module — the live gate — stayed on the pre-registered point disc, so the
published benchmark and the task the robot was actually running were two
different tasks (forensics F-02). ``opus5_seed101`` is the receipt: a published
success that the live loop scored ``declared_elsewhere`` and never offered its
return leg. TR.2 moved the criterion here, versioned it
(:data:`SUCCESS_CRITERION`), and made the scorer import it. Add a new criterion
by adding a VERSION, never by editing a predicate in one consumer.

**Radii come from the layout dict, never from a literal here.** AGENTS.md §2:
``duck_embody/env/apartment_layout.py`` is simultaneously the scene spec and the
scoring ground truth. ``configs/benchmark.yaml`` mirrors both radii for the
freeze manifest, and ``tests/test_loop.py`` asserts the two agree — the caps
precedent (``memory.TURN_CAP`` ↔ ``caps.turns``). Before T3.4 the two radii were
duplicated with **no** agreement test, which is exactly the shape of a silent
divergence between the live gate and the scorer.

Nothing here is ever shown to the model: the target point, the spawn coordinates
as a *goal*, and every distance computed below are ground truth (doc 06 §4). The
spawn coordinates do reach :class:`PositionIntegrator` as its t=0 anchor, which
doc 05 §5.1 declares as the one thing the dead-reckoned estimate takes from
ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from duck_embody.agent.memory import STAGE_FIND_KITCHEN, STAGE_RETURN_HOME
from duck_embody.agent.prompts import STAGE1_OBJECTIVE, STAGE2_OBJECTIVE_TOOL_RESULT
from duck_embody.env.apartment_layout import (
    LAYOUT,
    nearest_counter_face,
    room_at,
    spawn_pose,
    target_point,
)

# ---------------------------------------------------------------------------
# The success criterion — ONE versioned predicate, two consumers (TR.2)
# ---------------------------------------------------------------------------

#: The pre-registered criterion the v4 and v5d_r2 batches RAN under: Euclidean
#: distance to the pinned target point, inside ``success_radius_m``. Kept as a
#: named, callable version rather than deleted, because every legacy trial JSON
#: must still be validated against the predicate that actually decided it —
#: see :func:`score_stage`'s ``criterion`` argument.
CRITERION_PREREGISTERED = "v1_point_disc"

#: Criterion v2, "any counter face" (adopted post-batch 2026-07-27 at owner
#: direction; definition in ``docs/METRICS.md`` §2.1, change log in
#: ``results/rerun_log.md``). ``find_kitchen`` succeeds inside the pre-registered
#: point disc **OR** within the same radius of any of the five kitchen counter
#: footprints *while standing in the kitchen*. ``return_home`` is unchanged.
CRITERION_V2_ANY_COUNTER = "v2_any_counter"

#: The criterion the LIVE loop and the published scorer both use, and the value
#: every new trial JSON stamps into ``config.success_criterion``.
#:
#: **This constant is the F-02 fix.** Until TR.2 criterion v2 existed only in
#: post-hoc ``scoring.py`` while this module — the live stage gate — was still
#: on the point disc, so the two were different tasks. Measured consequence:
#: ``opus5_seed101`` declared 0.3607 m from the point (live failure) but
#: 0.0577 m from a counter face inside the kitchen (published success), and the
#: return leg it had earned was never offered. That trial is unrecoverable; the
#: split is not. Widening v2 further needs owner approval, a NEW version string
#: here, and a sensitivity report (remediation plan T2).
SUCCESS_CRITERION = CRITERION_V2_ANY_COUNTER

CRITERIA: tuple[str, ...] = (CRITERION_PREREGISTERED, CRITERION_V2_ANY_COUNTER)

# ---------------------------------------------------------------------------
# The stage-2 gate — RESOLVED BY T3.4 (doc 05 §12, doc 06 §12)
# ---------------------------------------------------------------------------

#: Does ``return_home`` run after a stage-1 cap-out? **No** — and, resolved with
#: it, nor after a stage-1 ``declare_done`` in the wrong place: stage 2 runs
#: **iff stage 1 SUCCEEDED**.
#:
#: The literal open question (cap-out) was never actually contested — doc 05
#: §3.3, doc 06 §3.2 and doc 01 §8 all already said no. Reading them together to
#: settle it surfaced a real contradiction they *did* have, on a case far more
#: likely than a cap-out: doc 05 §3.1/§3.3/§4.4 trigger stage 2 on "stage 1
#: ended via ``declare_done``" (score not consulted) while doc 01 §8 and doc 06
#: §3.2 trigger it on "stage 1 succeeded". They differ exactly when the model
#: declares done outside the 0.35 m target region — which both of those docs
#: separately call a failure.
#:
#: Resolved toward "succeeded", i.e. doc 05 is the one amended (same commit,
#: AGENTS.md rule 5). The decisive reason is a measured one. Under "any
#: declare_done", a wrong declare that happens to land inside the 0.5 m home
#: disc scores ``return_home`` a **success with zero motion**, and one such
#: trial is 25 percentage points of a success rate at N=4 (doc 06 §6's honesty
#: clause). Under "succeeded", that is geometrically impossible: stage 2 always
#: starts within 0.35 m of the counter, whose worst-case distance to a spawn
#: point is 1.574 m (seed 104) against a 0.5 m return radius — every seed clears
#: it, and the guarantee is already load-bearing in ``tests/test_layout.py``
#: (spawn > 3 × 0.35 m from target, doc 06 §9.2). The alternative has no floor
#: at all and would need an unprincipled extra guard no doc specifies.
#: [measured: spawn/target geometry recomputed from LAYOUT — see
#: ``tests/test_loop.py::TestStage2GateGeometry``]
#:
#: Three supporting reasons: comparability (every stage 2 starts from the same
#: 0.35 m disc, so ``d_initial``, the SPL oracle length and time-to-home are
#: homogeneous across trials and models); doc 06 §3.1's own statement of what
#: stage 2 is *for* ("the model must navigate back using only what it wrote
#: down" — a return leg from a spot the model never navigated *to* is not that
#: test); and cost (up to 40 turns + 240 policy-seconds of paid API on a trial
#: already scored a failure, on a stage AGENTS.md §8 lists first in the cut
#: order).
#:
#: **The honest cost, recorded rather than buried:** ``declare_done``'s
#: tool_result now differs by outcome, so the model can infer pass/fail at the
#: transition. That narrows doc 05 §3.3 item (1) "the model never sees the
#: score" to "the model never sees the numeric score or the distance; the only
#: signal is whether the return leg is offered". The blast radius is the
#: post-episode QA exchange alone, which runs after the episode either way and
#: is scored against ground truth; the mitigation is that the failure branch's
#: text is **outcome-neutral and byte-identical** to the stage-2 trial-over text
#: (:func:`duck_embody.agent.tools.stage_end_result`), so it says *the trial
#: ended*, never *you failed*.
#:
#: Mirrored in ``configs/benchmark.yaml`` as
#: ``protocol.stage2_requires_stage1_success`` so the rule sits inside the
#: hashed fairness contract (doc 06 §2/§7); ``tests/test_loop.py`` asserts they
#: agree. It previously lived in no config at all.
STAGE2_REQUIRES_STAGE1_SUCCESS = True


# ---------------------------------------------------------------------------
# Stage end reasons (doc 05 §3.2) and scored outcomes (doc 06 §4)
# ---------------------------------------------------------------------------
#
# These are two different things and conflating them loses the case the section
# above exists to name. `end_reason` is HOW the stage stopped — doc 05 §3.2's
# four stop conditions, exactly. `outcome` is reason + score, which is the only
# way a wrong-place `declare_done` is distinguishable from a right-place one in
# the log. doc 06 §4 shows only "success" / "timeout_turns" by example and never
# enumerates the vocabulary; the rest is authored here and recorded in §4.

REASON_DECLARE_DONE = "declare_done"
REASON_FALL = "fall"
REASON_TURN_CAP = "turn_cap"
REASON_MOTION_CAP = "motion_cap"
#: Not a doc 05 §3.2 stop condition: the stage never started (see
#: :data:`STAGE2_REQUIRES_STAGE1_SUCCESS`).
REASON_NOT_RUN = "not_run"

END_REASONS: tuple[str, ...] = (
    REASON_DECLARE_DONE,
    REASON_FALL,
    REASON_TURN_CAP,
    REASON_MOTION_CAP,
    REASON_NOT_RUN,
)

OUTCOME_SUCCESS = "success"
#: ``declare_done`` outside the target radius. doc 06 §3.1: "declaring elsewhere
#: is a failure". Kept distinct from a timeout because the model *believed* it
#: had arrived — a different cognitive failure, and the one that decides whether
#: stage 2 runs.
OUTCOME_DECLARED_ELSEWHERE = "declared_elsewhere"
OUTCOME_TIMEOUT_TURNS = "timeout_turns"
OUTCOME_TIMEOUT_MOTION = "timeout_motion"
OUTCOME_FALL = "fall"
OUTCOME_NOT_RUN = "not_run"

OUTCOMES: tuple[str, ...] = (
    OUTCOME_SUCCESS,
    OUTCOME_DECLARED_ELSEWHERE,
    OUTCOME_TIMEOUT_TURNS,
    OUTCOME_TIMEOUT_MOTION,
    OUTCOME_FALL,
    OUTCOME_NOT_RUN,
)

_OUTCOME_BY_REASON = {
    REASON_TURN_CAP: OUTCOME_TIMEOUT_TURNS,
    REASON_MOTION_CAP: OUTCOME_TIMEOUT_MOTION,
    REASON_FALL: OUTCOME_FALL,
    REASON_NOT_RUN: OUTCOME_NOT_RUN,
}


def runs_return_home(stage1_end_reason: str, stage1_success: bool) -> bool:
    """Does stage 2 run? THE gate, and the loop's only copy of it.

    Called from two places that must never disagree: choosing whether
    ``declare_done``'s tool_result carries the return-home objective, and
    choosing whether to actually run the stage. If those two ever diverged, a
    model would be handed the objective for a leg that never runs (or run a leg
    it was never told about) — and the trial JSON would still look fine.

    Written against :data:`STAGE2_REQUIRES_STAGE1_SUCCESS` rather than inlining
    ``and``: flipping that flag must revert to doc 05 §3.1's original rule
    ("stage 1 ended via ``declare_done``, score not consulted"), not disable the
    stage entirely, which is what the obvious one-line conjunction would do.
    """
    if stage1_end_reason != REASON_DECLARE_DONE:
        # A cap-out or a fall never continues, under either rule (doc 05 §3.3,
        # doc 06 §3.2, doc 01 §8 — all three already agreed on this).
        return False
    return stage1_success or not STAGE2_REQUIRES_STAGE1_SUCCESS


def outcome_for(end_reason: str, success: bool) -> str:
    """Scored outcome from doc 05 §3.2's stop reason plus the distance verdict.

    Only ``declare_done`` can produce a success; every other reason is a scored
    failure with partial progress (doc 06 §3.2, "Locked"). A ``declare_done``
    that fails the distance test is ``declared_elsewhere``, never a timeout —
    the model stopped on purpose. Raises rather than coercing if a caller claims
    a success for a capped or fallen stage: doc 06 §3.1 requires BOTH the
    distance and the declaration, so "arrived but never declared" is a timeout
    and quietly promoting it would inflate the headline SR.
    """
    if end_reason == REASON_DECLARE_DONE:
        return OUTCOME_SUCCESS if success else OUTCOME_DECLARED_ELSEWHERE
    if success:
        raise ValueError(
            f"end_reason {end_reason!r} cannot be a success: doc 06 §3.1 requires "
            "declare_done as well as the distance (arriving without declaring is "
            "a timeout)"
        )
    return _OUTCOME_BY_REASON[end_reason]


# ---------------------------------------------------------------------------
# Stage specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    """One stage's goal. Frozen: it is protocol, not runtime state.

    ``goal_xy`` and ``success_radius_m`` are GROUND TRUTH and never rendered
    into anything the model reads. ``objective`` is the model-facing text, and
    it is deliberately the frozen string from :mod:`prompts` in both cases —
    stage 1's lives inside ``SYSTEM_PROMPT``, stage 2's is delivered mid-episode
    as ``declare_done``'s tool_result (doc 05 §3.3) — so no third copy of either
    objective exists to drift.
    """

    name: str
    objective: str
    goal_xy: tuple[float, float]
    success_radius_m: float
    #: What the goal *is*, for log readability. Never model-facing.
    goal_label: str


@dataclass(frozen=True)
class StageScore:
    """The position verdict at ``declare_done`` (or at a stage's end).

    Every input to the decision is carried so the trial log can record them and
    T4.1's scorer can **recompute** the verdict rather than trust it — doc 06 §4:
    "The log is the single source for scoring — the scorer never touches the
    simulator."

    TR.2 stopped overloading one distance. Under criterion v2 the goal is a
    REGION (disc ∪ counter band), so a single ``distance_m`` could not answer
    "how did this pose pass?" or "how far outside was it?" without the reader
    re-deriving the geometry — which is how two readings of one batch appear.
    The point distance keeps its own field (and its legacy ``distance_m`` name
    in the JSON) because doc 06 §5.2's continuous progress metric is defined
    against the point and stays comparable across criterion versions.

    Scoring/audit channels ONLY. Nothing here is ever rendered to a model
    (doc 06 §4) — it is all ground truth.
    """

    criterion_version: str
    success: bool
    #: Euclidean distance to the pinned target point. Under
    #: ``return_home`` this is the distance to the spawn point.
    distance_to_point_m: float
    #: How much closer the robot would have had to be, in a straight line, for
    #: the pose to satisfy SOME branch of this criterion. ``0.0`` on a success.
    #: A slack, deliberately not a walking distance: through a wall the nearest
    #: counter is centimetres away and metres of walking away, and only the
    #: boolean verdict is allowed to depend on the in-room condition.
    distance_to_success_region_m: float
    radius_m: float
    goal_xy: tuple[float, float]
    true_xy: tuple[float, float]
    #: Stage-1 only (``None`` for ``return_home``, whose goal has no counter
    #: semantics): the nearest kitchen counter and the distance to its
    #: footprint rectangle, reported whether or not the counter branch fired.
    nearest_counter_name: str | None = None
    distance_to_counter_m: float | None = None
    #: Was the robot inside the kitchen? The load-bearing half of the counter
    #: branch — counter_4/5 back onto the bedroom partition, so a bedroom pose
    #: 4 cm through that wall is within 0.35 m of their rectangles.
    in_goal_room: bool | None = None

    @property
    def distance_m(self) -> float:
        """The pre-registered point distance, under its historical name.

        Kept so every existing reader (``scoring.stage_success_preregistered``'s
        log cross-check, the audit scripts, the committed batches' JSON) keeps
        meaning exactly what it meant before TR.2.
        """
        return self.distance_to_point_m

    def as_dict(self) -> dict:
        return {
            "criterion_version": self.criterion_version,
            "success": self.success,
            # Legacy name, unchanged semantics: distance to the pinned point.
            # Emitted alongside the explicit name so old and new readers agree.
            "distance_m": round(self.distance_to_point_m, 4),
            "distance_to_point_m": round(self.distance_to_point_m, 4),
            "distance_to_success_region_m": round(
                self.distance_to_success_region_m, 4
            ),
            "radius_m": self.radius_m,
            "goal_xy": [self.goal_xy[0], self.goal_xy[1]],
            "true_xy": [round(self.true_xy[0], 4), round(self.true_xy[1], 4)],
            "nearest_counter_name": self.nearest_counter_name,
            "distance_to_counter_m": (
                None
                if self.distance_to_counter_m is None
                else round(self.distance_to_counter_m, 4)
            ),
            "in_goal_room": self.in_goal_room,
        }


def spawn_for_seed(seed: int) -> tuple[tuple[float, float], float]:
    """``((x, y), heading_deg)`` for a trial seed — the layout's own table."""
    return spawn_pose(seed)


def find_kitchen_spec() -> StageSpec:
    """Stage 1. Success radius and target point are the layout's, not literals."""
    target = LAYOUT["target"]
    return StageSpec(
        name=STAGE_FIND_KITCHEN,
        objective=STAGE1_OBJECTIVE,
        goal_xy=target_point(),
        success_radius_m=float(target["radius"]),
        goal_label=target["name"],
    )


def return_home_spec(spawn_xy: tuple[float, float]) -> StageSpec:
    """Stage 2. The goal is *this seed's* spawn point, so it is a parameter."""
    return StageSpec(
        name=STAGE_RETURN_HOME,
        objective=STAGE2_OBJECTIVE_TOOL_RESULT,
        goal_xy=(float(spawn_xy[0]), float(spawn_xy[1])),
        success_radius_m=float(LAYOUT["return_home_radius"]),
        goal_label="spawn_point",
    )


def stage_specs(seed: int) -> tuple[StageSpec, StageSpec]:
    """Both stages for one seed, in protocol order (doc 06 §3.1)."""
    spawn_xy, _heading = spawn_for_seed(seed)
    return find_kitchen_spec(), return_home_spec(spawn_xy)


def _counter_branch_applies(stage: str, criterion: str) -> bool:
    """Does the counter half of v2 exist for this stage under this criterion?"""
    return criterion == CRITERION_V2_ANY_COUNTER and stage == STAGE_FIND_KITCHEN


def position_success(
    stage: str,
    true_xy: tuple[float, float],
    spec: StageSpec,
    *,
    criterion: str = SUCCESS_CRITERION,
) -> bool:
    """THE position predicate, for one stage under one named criterion.

    ``find_kitchen`` under v2: the pre-registered point disc **OR** within the
    same radius of any kitchen counter footprint *while standing in the
    kitchen*. The UNION, not the counter branch alone, because the two regions
    are NOT nested — the pinned target point is 0.397 m from the nearest counter
    footprint, so a pure any-counter test would fail a robot standing exactly on
    the pre-registered goal. The in-kitchen condition is the other load-bearing
    half: counter_4/5 back onto the bedroom partition.

    ``return_home`` is the pre-registered disc under BOTH criteria. Its goal has
    no counter semantics, and a pose near a counter is not "home".

    The boundary is inclusive — doc 06 §3.1 says "within" — on both branches.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"unknown success criterion {criterion!r}; expected {CRITERIA}")
    if math.dist(true_xy, spec.goal_xy) <= spec.success_radius_m:
        return True
    if not _counter_branch_applies(stage, criterion):
        return False
    if room_at(true_xy[0], true_xy[1]) != LAYOUT["target"]["room"]:
        return False
    return nearest_counter_face(true_xy)[1] <= spec.success_radius_m


def score_stage(
    spec: StageSpec,
    true_xy: tuple[float, float],
    *,
    criterion: str = SUCCESS_CRITERION,
) -> StageScore:
    """THE verdict, with every input that produced it.

    Takes the **true** base XY (``PolicyPlayback.true_xy()``, scoring-only), not
    the dead-reckoned estimate: doc 06 §5.1 scores the distance condition from
    ground truth, and a model whose estimate drifted onto the counter while the
    robot stood in the hallway must fail. That asymmetry is the benchmark.

    ``criterion`` defaults to :data:`SUCCESS_CRITERION`, which is what the live
    loop and the published scorer both get. It is an argument at all so a
    LEGACY trial (logged before the live gate was unified) can still be
    validated against the predicate that actually decided it — never
    reinterpreted as though v2 had run live.
    """
    point_distance = math.dist(true_xy, spec.goal_xy)
    xy = (float(true_xy[0]), float(true_xy[1]))
    success = position_success(spec.name, xy, spec, criterion=criterion)
    slack = max(0.0, point_distance - spec.success_radius_m)

    counter_name: str | None = None
    counter_distance: float | None = None
    in_goal_room: bool | None = None
    if _counter_branch_applies(spec.name, criterion):
        counter_name, counter_distance = nearest_counter_face(xy)
        in_goal_room = room_at(xy[0], xy[1]) == LAYOUT["target"]["room"]
        if in_goal_room:
            slack = min(slack, max(0.0, counter_distance - spec.success_radius_m))
    return StageScore(
        criterion_version=criterion,
        success=success,
        distance_to_point_m=point_distance,
        distance_to_success_region_m=0.0 if success else slack,
        radius_m=spec.success_radius_m,
        goal_xy=spec.goal_xy,
        true_xy=xy,
        nearest_counter_name=counter_name,
        distance_to_counter_m=counter_distance,
        in_goal_room=in_goal_room,
    )
