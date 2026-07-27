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
from duck_embody.env.apartment_layout import LAYOUT, spawn_pose, target_point

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
    """The distance verdict at ``declare_done`` (or at a stage's end).

    Every input to the decision is carried so the trial log can record them and
    T4.1's scorer can **recompute** the verdict rather than trust it — doc 06 §4:
    "The log is the single source for scoring — the scorer never touches the
    simulator."
    """

    success: bool
    distance_m: float
    radius_m: float
    goal_xy: tuple[float, float]
    true_xy: tuple[float, float]

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "distance_m": round(self.distance_m, 4),
            "radius_m": self.radius_m,
            "goal_xy": [self.goal_xy[0], self.goal_xy[1]],
            "true_xy": [round(self.true_xy[0], 4), round(self.true_xy[1], 4)],
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


def score_stage(spec: StageSpec, true_xy: tuple[float, float]) -> StageScore:
    """THE predicate. The boundary is inclusive — doc 06 §3.1 says "within".

    Takes the **true** base XY (``PolicyPlayback.true_xy()``, scoring-only), not
    the dead-reckoned estimate: doc 06 §5.1 scores the distance condition from
    ground truth, and a model whose estimate drifted onto the counter while the
    robot stood in the hallway must fail. That asymmetry is the benchmark.
    """
    distance = math.dist(true_xy, spec.goal_xy)
    return StageScore(
        success=distance <= spec.success_radius_m,
        distance_m=distance,
        radius_m=spec.success_radius_m,
        goal_xy=spec.goal_xy,
        true_xy=(float(true_xy[0]), float(true_xy[1])),
    )
