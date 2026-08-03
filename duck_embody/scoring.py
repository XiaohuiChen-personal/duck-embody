"""Post-hoc metrics: doc 06 §5, computed from the trial JSON + ground truth.

**This module reads ground truth on purpose.** ``true_pose``, ``pose_trace``,
oracle paths and room polygons are all scoring-only channels (doc 06 §4), and
this file is the one place they may be touched. The separation that matters runs
the other way: nothing under ``duck_embody/agent/`` may import this module —
that package is the model-facing side, and a scorer reachable from it is one
refactor away from leaking the answer key into a prompt.
``tests/test_scoring.py::TestPackageSeparation`` asserts the direction.

**What the scorer reads.** Doc 06 §5's preamble says "the trial JSON plus the
layout ground truth — nothing else". As implemented that is three frozen inputs,
and naming them honestly is better than repeating "nothing else" (AGENTS.md
rule 3):

1. the per-trial JSON (doc 06 §4);
2. ``duck_embody/env/apartment_layout.py`` — scene spec *and* answer key
   (AGENTS.md §2), read through its own helpers so the geometry cannot drift;
3. two frozen contract modules whose logic doc 06 explicitly forbids
   re-authoring here: ``tasks/find_kitchen.py::score_stage`` (§9.1(iii): "the
   same predicate the live loop consulted, so the scorer and the gate cannot
   disagree") and ``agent/prompts.py``'s ``ROOM_SYNONYMS`` /
   ``LAYOUT_QA_QUESTIONS`` (§5.7, §5.9).

``configs/benchmark.yaml``'s ``scoring:`` block supplies §6's bootstrap
constants; it is the only place the RNG seed exists.

**"—" is a value, never a number.** :data:`NA` marks a metric that is genuinely
undefined for a trial (time-to-kitchen on a failure, precision with zero claimed
rooms, drift on an unrun stage). It is excluded from means and confidence
intervals and is never coerced to 0.0 — doc 06 §5.4/§5.7/§3.2 all require that,
and a 0.0 in a time column would read as "arrived instantly".

**Q2's parse rules live here, not in ``configs/``.** They are post-hoc scorer
logic: they never touch what a model sees, so they are not a doc 06 §2 fairness
item, and §7's config-hash guard deliberately does not hash this file — re-
scoring is free, re-running a paid batch is not. What keeps them honest is PLAN
T4.1's ordering: they are committed, with fixtures, *before* the batch, so the
vocabulary cannot be tuned after model answers are visible. Any post-batch change
re-scores **all** models together and is logged in ``results/rerun_log.md``.

**Success criterion v2 (post-batch change, 2026-07-27, owner-directed).** The
published stage-1 success predicate is no longer the pre-registered point-disc
test alone: it is the UNION of that disc and "within the same 0.35 m of any of
the five kitchen counters' footprints, standing inside the kitchen". Decided
AFTER the batch was visible (the trigger: ``gpt56sol_seed103`` declared done
0.05 m from an east-wall counter face and scored ``declared_elsewhere`` against
the south-run target point, while the frozen objective — "walk to the counter"
— never disambiguates the runs). Recorded per this module's own protocol: all
12 trials of all three models re-scored together, logged in
``results/rerun_log.md``, and the pre-registered verdict is still published
per-trial (``success_preregistered`` / ``outcome_preregistered``) so both
readings stay reproducible. The live stage-2 gate ran under the pre-registered
predicate, so a trial that succeeds only under v2 was never offered its return
leg — the conditional return-home rate therefore counts only trials whose
return leg actually ran. See :func:`stage_success` / :func:`stage_success_preregistered`.
"""

from __future__ import annotations

import bisect
import heapq
import json
import math
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from duck_embody.agent.memory import (
    STAGE_FIND_KITCHEN,
    STAGE_RETURN_HOME,
    exit_status_target,
)
from duck_embody.agent.prompts import LAYOUT_QA_QUESTIONS, ROOM_SYNONYMS
from duck_embody.env.apartment_layout import (
    COMPASS_8,
    LAYOUT,
    adjacency,
    bearing_deg,
    compass_8,
    connecting_rooms,
    grid,
    kitchen_counter_rects,
    nearest_counter_face,
    oracle_length,
    room_at,
    room_centroid,
    room_path,
    spawn_pose,
)
from duck_embody.tasks.find_kitchen import (
    CRITERION_PREREGISTERED,
    CRITERION_V2_ANY_COUNTER,
    REASON_DECLARE_DONE,
    REASON_FALL,
    REASON_NOT_RUN,
    SUCCESS_CRITERION,
    find_kitchen_spec,
    outcome_for,
    position_success,
    return_home_spec,
    score_stage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CONFIG = REPO_ROOT / "configs" / "benchmark.yaml"

#: The em dash doc 06 §5.4/§5.7/§3.2 all print for an undefined metric. A
#: string, deliberately: any arithmetic on it raises instead of silently
#: producing a number that means "zero" when it means "no answer".
NA = "—"

#: doc 06 §3.2's "no CI when k < 3 — a bootstrap over one value is theatre".
#: §3.2 states it for the conditional return-home SR; it applies to every metric
#: for the same reason, and a metric with 2 defined values out of 4 is exactly
#: that situation (T4.1 reports it as a proposed §6 wording edit).
MIN_CI_VALUES = 3

STAGES: tuple[str, str] = (STAGE_FIND_KITCHEN, STAGE_RETURN_HOME)

#: doc 06 §5.7's evidence sources: the tool calls that name a room, mapped to the
#: argument carrying the name. See :func:`room_evidence` for why
#: ``add_landmark`` contributes the pose at the call, not a landmark position.
ROOM_CLAIM_TOOLS: dict[str, str] = {
    "update_room": "name",
    "set_current_room": "name",
    "add_landmark": "room",
}


#: Doc 06 §5.8's pairing needs the belief AFTER the turn's tool calls ran; the
#: turn record's ``obs.position_estimate`` is captured BEFORE them. This is the
#: optional post-dispatch key (T4.1's proposed ``loop.py`` edit). Absent logs
#: fall back to the instant-consistent pairing in :func:`stage_drift`.
POSITION_ESTIMATE_END = "position_estimate_end"


class ScoringError(ValueError):
    """The trial JSON cannot be scored as written. Never swallowed."""


class MissingPoseTraceError(ScoringError):
    """A turn's ``execution`` has no ``pose_trace`` (doc 06 §4, PLAN T4.1).

    Raised **loudly** rather than falling back to chord-summing the once-per-turn
    ``true_pose`` entries. That fallback would under-measure every within-turn
    curve (``send_velocity`` arcs with wz ≠ 0, deflections during a bumped
    ``move``), shrinking ``p`` and *inflating* SPL — i.e. it would make every
    model look better than it was, in the headline metric, with nothing to notice
    it by. An **empty** trace is legal and means "this turn stepped no physics";
    §4's widening #3 exists precisely so the two cannot look alike.
    """


class IncompleteTrialError(ScoringError):
    """No ``final`` block — doc 06 §9.1's resume check (see :func:`load_trial`)."""


# ---------------------------------------------------------------------------
# Frozen constants from configs/benchmark.yaml (doc 06 §6)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def scoring_config(path: Path | str = BENCHMARK_CONFIG) -> Mapping[str, object]:
    """The ``scoring:`` block of the frozen benchmark config, READ-ONLY.

    Cached: a 12-trial batch asks for the bootstrap constants once per metric per
    model, and re-reading a frozen file dozens of times only adds ways for the
    two reads to differ.

    Read rather than duplicated: the bootstrap seed exists in exactly one place
    (doc 06 §6 locks "fixed RNG seed" but names no value), and the two success
    radii are mirrored there for the freeze manifest. Literals here would be a
    third copy of numbers ``tests/test_loop.py`` already pins to the layout.

    Returned as a :class:`~types.MappingProxyType` because ``lru_cache`` caches
    the *object*, not the file read: every caller shares one dict, so a single
    ``config['bootstrap_seed'] = …`` anywhere in T4.2/T4.4 would silently re-seed
    every subsequent bootstrap in the run and the published intervals would stop
    being reproducible from the committed YAML. Writing through the proxy raises.
    """
    import yaml  # lazy, as in providers/base.py — pyyaml is not a hard dep

    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    return MappingProxyType(dict(document["scoring"]))


# ---------------------------------------------------------------------------
# Numeric hygiene: NaN and infinity are log corruption, never a metric
# ---------------------------------------------------------------------------


def _finite(value, where: str) -> float:
    """``float(value)``, rejecting NaN and ±inf with a :class:`ScoringError`.

    Every float the scorer reads out of a trial JSON goes through here. NaN is
    the one corrupt value the module would otherwise publish **silently and
    flatteringly**: ``min(1.0, nan)`` is ``1.0``, so ``progress`` reported a
    PERFECT 1.0 for a NaN distance, and ``estimate([1.0, nan, 1.0])`` returned a
    NaN mean and interval while still claiming three defined values. It is
    reachable — ``json.loads('{"x": NaN}')` succeeds, so a PhysX blow-up
    round-trips through the log — and ``json.dumps`` then writes ``NaN``, which
    is not valid JSON for any downstream reader.

    Note the asymmetry this removes: ``defined([None])`` already raised
    ``TypeError``, so ``None`` was loud and NaN was mute.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ScoringError(f"{where}: {value!r} is not a number") from error
    if not math.isfinite(number):
        raise ScoringError(
            f"{where}: {value!r} is not finite. NaN/inf is log corruption, not a "
            "metric — see doc 06 §9.1's 'a hostile log fails with a message'"
        )
    return number


# ---------------------------------------------------------------------------
# Trial loading and the resume check (doc 06 §9.1 "Schema/leak guard")
# ---------------------------------------------------------------------------


def is_complete(document: dict) -> bool:
    """Is this trial JSON a scoreable result?

    ``final`` is written only by ``TrialLog.finish``; a doc 05 §8 infra fault
    records ``infra_failure`` and deliberately leaves ``final`` absent so the
    trial reruns whole. Both halves are checked: an ``infra_failure`` that
    somehow acquired a ``final`` is still not a result.
    """
    return "final" in document and "infra_failure" not in document


def load_trial(path: Path | str) -> dict:
    """Load a trial JSON, rejecting an incomplete one (doc 06 §9.1).

    This is the predicate T4.2's resumable runner skips on, kept here so the
    runner and the scorer agree on what "done" means — the same argument doc 06
    §9.1(iii) makes for ``score_stage``.
    """
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not is_complete(document):
        detail = document.get("infra_failure", "no `final` block")
        raise IncompleteTrialError(f"{path}: incomplete trial ({detail})")
    return document


# ---------------------------------------------------------------------------
# 5.2 Progress and 5.3 SPL — the canonical formulas
# ---------------------------------------------------------------------------


def progress(d_initial: float, d_final: float) -> float:
    """doc 06 §5.2: ``clip(1 − d_final / d_initial, 0, 1)``.

    **No success override**: a success reports the same formula value as a
    failure, so every published number is reproducible from the formula alone.

    ``d_initial == 0`` returns 0.0. It is geometrically unreachable for stage 1
    (``tests/test_layout.py`` pins every spawn at > 3 × 0.35 m from the target)
    but is representable for an unrun stage 2, whose start pose is arbitrary; the
    value agrees with doc 06 §3.2's rule for that case ("the robot never moved,
    so d_final = d_initial ⇒ progress 0.0") instead of dividing by zero.

    A non-finite distance **raises**: ``min(1.0, nan)`` is ``1.0`` and
    ``max(0.0, 1.0)`` is ``1.0``, so a garbage distance would otherwise report
    the maximum possible progress (see :func:`_finite`).
    """
    d_initial = _finite(d_initial, "progress: d_initial")
    d_final = _finite(d_final, "progress: d_final")
    if d_initial <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - d_final / d_initial))


def spl(success: bool, oracle_m: float | None, path_m: float) -> float:
    """doc 06 §5.3: ``S × l / max(p, l)`` — Anderson et al. 2018 Eq. (1).

    The ``S = 0`` short-circuit happens **before** the division, which §9.1's
    stage-2 case requires: after a stage-1 failure the end pose is arbitrary, so
    ``l`` can be ~0 and ``max(p, l)`` can be 0/0. ``max(p, l)`` also caps the
    ratio at 1.0 when ``p < l`` (drift, rounding, or corner-cutting through the
    grid's 5 cm resolution).
    """
    if not success:
        return 0.0
    if oracle_m is None:
        raise ScoringError(
            "a successful stage has no oracle path length: `l` is undefined, so "
            "SPL cannot be computed (doc 06 §5.3)"
        )
    oracle_m = _finite(oracle_m, "spl: oracle_m")
    path_m = _finite(path_m, "spl: path_m")
    denominator = max(path_m, oracle_m)
    if denominator <= 0.0:
        # Success with a zero-length oracle AND a zero-length path: the robot
        # was already at the target and never moved. §9.1's "no division
        # blowup"; l / max(p, l) is 1 in the limit.
        return 1.0
    return oracle_m / denominator


# ---------------------------------------------------------------------------
# 5.3's path integral p — the 5 Hz pose_trace, never per-turn chords
# ---------------------------------------------------------------------------


def turn_pose_trace(turn: dict) -> list[tuple[float, float]]:
    """This turn's 5 Hz true-XY samples, or raise (doc 06 §4, PLAN T4.1)."""
    if turn.get("execution") is None:
        raise MissingPoseTraceError(
            f"turn {turn.get('global_turn_idx', '?')} has no `execution` block; "
            "doc 06 §4 requires one on EVERY turn (an empty pose_trace when no "
            "physics stepped). Refusing to fall back to per-turn true_pose "
            "chords, which would inflate SPL."
        )
    execution = turn["execution"]
    if "pose_trace" not in execution:
        raise MissingPoseTraceError(
            f"turn {turn.get('global_turn_idx', '?')} has `execution` but no "
            "`pose_trace` key. doc 06 §4: a turn that stepped no physics has an "
            "EMPTY trace, so 'no motion' and 'trace dropped' must not look "
            "alike. Refusing to fall back to per-turn true_pose chords, which "
            "would under-measure curvature and inflate SPL."
        )
    trace = execution["pose_trace"]
    if not isinstance(trace, list):
        # `null` is the same fault as a missing key wearing a different hat, and
        # the class docstring is entirely about keeping "no motion" and "trace
        # dropped" distinguishable. Without this it was a raw TypeError from
        # inside the path integral, with no trial named.
        raise MissingPoseTraceError(
            f"turn {turn.get('global_turn_idx', '?')} has a `pose_trace` that is "
            f"{type(trace).__name__}, not a list. doc 06 §4: a turn that stepped "
            "no physics has an EMPTY list. Refusing to fall back to per-turn "
            "true_pose chords, which would inflate SPL."
        )
    where = f"turn {turn.get('global_turn_idx', '?')} pose_trace"
    points: list[tuple[float, float]] = []
    for sample in trace:
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            raise ScoringError(f"{where}: {sample!r} is not an [x, y] sample")
        points.append((_finite(sample[0], where), _finite(sample[1], where)))
    return points


def path_length_m(turns: Sequence[dict], stage: str) -> float:
    """doc 06 §5.3's ``p`` for one stage: Σ ‖pose(t+1) − pose(t)‖.

    Segmented **per stage**, which §5.3 leaves implicit and §3.2's per-stage
    accounting requires: summing across the boundary would charge stage 2 with
    the (arbitrary) jump from wherever stage 1 ended.

    Concatenation is safe. Each motion call's trace is
    ``[start_xy, *sampled_xy, end_xy]`` (``sim/policy_wrapper.py``), so a
    boundary point is duplicated between consecutive calls and consecutive
    turns — and a duplicated point contributes exactly 0 to the sum.

    This is only half of ``p``'s guard: an *empty* trace is legal, so a recorder
    that wrote ``[]`` on every turn would report ``p = 0`` and — because
    ``max(p, l)`` caps SPL at 1.0 — a PERFECT headline score. See
    :func:`chord_floor_m`, which is what catches that.
    """
    trace: list[tuple[float, float]] = []
    for turn in turns:
        if turn.get("stage") != stage:
            continue
        trace.extend(turn_pose_trace(turn))
    return sum(math.dist(a, b) for a, b in zip(trace, trace[1:]))


#: Per-chord slack for :func:`chord_floor_m`. ``true_pose`` is logged at 4 dp
#: while ``pose_trace`` carries full precision, so each rounded endpoint moves by
#: ≤ 5e-5 per axis (≤ 7.08e-5 in the plane) and each chord can exceed the true
#: sub-path it bounds by ≤ 1.42e-4. [measured on the committed fixtures: the
#: margin is 0.000000 for find_kitchen and -1e-16 for return_home — straight-line
#: traces make the bound tight, which is exactly why the slack has to be small.]
CHORD_FLOOR_TOL_M = 1.5e-4
CHORD_FLOOR_BASE_TOL_M = 1e-3


def chord_floor_m(document: dict, stage: str) -> tuple[float, int]:
    """A LOWER bound on the stage's true path length, from a different field.

    ``Σ ‖true_pose(n) − true_pose(n−1)‖`` over the stage's turns, anchored at the
    stage's start. Physics advances only inside ``env.step()`` and every step is
    traced, so by the triangle inequality this can never exceed the traced path
    length ``p`` — it is a floor computed from ``turns[].true_pose`` rather than
    from ``turns[].execution.pose_trace``.

    Why it exists: every way of LOSING trace samples pushes SPL **up**, because
    ``max(p, l)`` caps the ratio at 1.0. Measured on the golden fixture: blanking
    every ``pose_trace`` to ``[]`` — the module's own documented value for "this
    turn stepped no physics", so nothing raised — gave ``p = 0`` and
    ``spl = 1.0`` for both stages, indistinguishable from a perfect run; keeping
    only each trace's last sample gave ``p = 1.4922`` against a true 2.2985
    (−35 %) and still ``spl = 1.0``. The module raised loudly on a *missing*
    ``pose_trace`` key while the likelier recorder fault — writing empty lists —
    was unguarded.

    Returns ``(floor, chord count)``; the count sizes the rounding slack.
    """
    start = stage_start_xy(document, stage)
    previous = start
    total = 0.0
    chords = 0
    for turn in _stage_turns(document, stage):
        pose = _pose_xy(turn.get("true_pose"))
        if pose is None:
            continue
        total += math.dist(previous, pose)
        previous = pose
        chords += 1
    return total, chords


def true_trace(document: dict, stage: str | None = None) -> list[tuple[float, float]]:
    """Every ground-truth XY the log records, in order.

    The 5 Hz ``pose_trace`` samples **plus** each turn's ``true_pose`` sibling
    plus the spawn. doc 06 §5.7/§5.9 say "the true trace" without saying which of
    the two series they mean; the union is the honest reading — the 5 Hz samples
    are the finest-grained record of where the robot actually was, and the
    per-turn ``true_pose`` covers turns that stepped no physics (which produce no
    samples at all). Room polygons tile the whole apartment floor, so a sample
    inside a room means the robot's centre was in that room; there is no
    corner-clipping case where the union over-reports a visit.
    """
    points: list[tuple[float, float]] = []
    spawn = document.get("config", {}).get("spawn", {}).get("xy")
    if spawn is not None and stage in (None, STAGE_FIND_KITCHEN):
        points.append((_finite(spawn[0], "config.spawn.xy"), _finite(spawn[1], "config.spawn.xy")))
    for turn in document.get("turns", []):
        if stage is not None and turn.get("stage") != stage:
            continue
        points.extend(turn_pose_trace(turn))
        pose = _pose_xy(turn.get("true_pose"))
        if pose is not None:
            points.append(pose)
    return points


def visited_rooms(document: dict) -> tuple[str, ...]:
    """True rooms the robot entered, trial-scoped, in first-visit order.

    Trial-scoped because memory is: doc 06 §5.7's recall denominator and Q3's
    gold answer are both about the whole run, and ``reset_for_stage()``
    deliberately keeps the map across the stage boundary.
    """
    seen: list[str] = []
    for x, y in true_trace(document):
        room = room_at(x, y)
        if room is not None and room not in seen:
            seen.append(room)
    return tuple(seen)


# ---------------------------------------------------------------------------
# Stage geometry: the specs, the start poses, and the per-stage d_initial / l
# ---------------------------------------------------------------------------


def spawn_xy(document: dict) -> tuple[float, float]:
    """The trial's spawn, cross-checked against the layout's own table.

    ``config.spawn`` is what the harness actually used; ``spawn_pose(seed)`` is
    what the frozen layout says it should have been. A disagreement means the
    trial did not start where the answer key thinks it did, which would silently
    corrupt ``d_initial``, the SPL oracle and Q4's gold bearing.
    """
    config = document["config"]
    raw = config["spawn"]["xy"]
    logged = (_finite(raw[0], "config.spawn.xy"), _finite(raw[1], "config.spawn.xy"))
    expected, _heading = spawn_pose(int(config["seed"]))
    if not math.isclose(logged[0], expected[0], abs_tol=1e-6) or not math.isclose(
        logged[1], expected[1], abs_tol=1e-6
    ):
        raise ScoringError(
            f"trial {document.get('trial_id')!r} logs spawn {tuple(logged)} but "
            f"the layout's spawn_points[{config['seed']}] is {expected}"
        )
    return logged


def stage_spec(document: dict, stage: str):
    """The stage's :class:`StageSpec` — goal and radius from the layout only."""
    if stage == STAGE_FIND_KITCHEN:
        return find_kitchen_spec()
    return return_home_spec(spawn_xy(document))


def _pose_xy(pose: dict | None) -> tuple[float, float] | None:
    if pose is None:
        return None
    return (_finite(pose["x"], "true_pose.x"), _finite(pose["y"], "true_pose.y"))


def _score_xy(score: dict) -> tuple[float, float]:
    return (
        _finite(score["true_xy"][0], "stage score.true_xy"),
        _finite(score["true_xy"][1], "stage score.true_xy"),
    )


def stage_start_xy(document: dict, stage: str) -> tuple[float, float]:
    """Where the stage began.

    doc 06 §5.2/§5.3 both word the start as "spawn", which is stage 1's start and
    is *also* stage 2's target — read literally, stage 2's ``d_initial`` and
    oracle length would both be zero, contradicting §9.1's own stage-2 case and
    doc 05 §3.3's measured 1.574 m floor. Per stage is the only consistent
    reading: ``find_kitchen`` starts at the spawn, ``return_home`` starts at the
    true pose the robot held at the stage boundary. (Reported by T4.1 as a
    proposed §5.2/§5.3 wording edit.)
    """
    if stage == STAGE_FIND_KITCHEN:
        return spawn_xy(document)
    stage1 = document["final"]["stages"][STAGE_FIND_KITCHEN]
    end = _pose_xy(stage1.get("true_pose"))
    if end is None:
        score = stage1.get("score")
        if score is None:
            raise ScoringError(
                "stage 1 has neither `true_pose` nor `score`: the return_home "
                "start pose is unrecoverable (doc 06 §5.2/§5.3)"
            )
        end = _score_xy(score)
    return end


def stage_end_xy(document: dict, stage: str) -> tuple[float, float]:
    """Where the stage ended — an unrun stage never moved from its start."""
    result = document["final"]["stages"][stage]
    score = result.get("score")
    if score is not None:
        return _score_xy(score)
    end = _pose_xy(result.get("true_pose"))
    if end is not None:
        return end
    # `not_run`: StageResult.not_run writes score=None and true_pose=None
    # precisely because nothing happened. doc 06 §3.2: d_final == d_initial.
    return stage_start_xy(document, stage)


# ---------------------------------------------------------------------------
# 5.1 Success criterion v2 — "any counter face" (post-batch change; see the
# module docstring and results/rerun_log.md)
# ---------------------------------------------------------------------------

# TR.2: the criterion, its geometry and its predicate all moved OUT of this
# module. `SUCCESS_CRITERION`, `position_success` and `score_stage` come from
# `tasks/find_kitchen.py` (the live gate consults them too, which is the F-02
# fix), and the counter rectangles from `env/apartment_layout.py` (the frozen
# ground truth). The names below stay importable from `duck_embody.scoring`
# because scripts, charts and tests already address them here — but each is now
# a re-export, not a second implementation. A criterion this file could define
# on its own is a criterion that can drift from the one the robot ran under.
__all_criterion_reexports__ = (
    "SUCCESS_CRITERION",
    "CRITERION_PREREGISTERED",
    "CRITERION_V2_ANY_COUNTER",
    "kitchen_counter_rects",
    "nearest_counter_face",
)


def position_success_v2(stage: str, xy: tuple[float, float], spec) -> bool:
    """The position half of criterion v2 — :func:`position_success` at v2.

    Retained as a named alias because "v2" is the vocabulary of
    ``results/rerun_log.md`` and ``docs/METRICS.md`` §2.1, and because the
    scorer must be able to ask for v2 EXPLICITLY when reading a legacy trial
    whose live gate ran the pre-registered predicate.
    """
    return position_success(stage, xy, spec, criterion=CRITERION_V2_ANY_COUNTER)


def trial_success_criterion(document: dict) -> str:
    """Which criterion did this trial's LIVE stage machine run under?

    New logs stamp ``config.success_criterion`` (TR.2). Logs written before
    that — the v4 baseline and ``raw_v5d_r2`` — carry no such field, and for
    them the answer is the pre-registered point disc, because that is what
    ``score_stage`` computed at the time. Defaulting the other way would
    silently re-interpret ``final.stages.*.success`` as a v2 verdict it never
    was, and every log-consistency cross-check below would then either raise on
    a healthy legacy log or, worse, pass while comparing two different
    predicates.
    """
    criterion = (document.get("config") or {}).get("success_criterion")
    if criterion is None:
        return CRITERION_PREREGISTERED
    if criterion not in (CRITERION_PREREGISTERED, CRITERION_V2_ANY_COUNTER):
        raise ScoringError(
            f"trial {document.get('trial_id')!r} stamps an unknown "
            f"success_criterion {criterion!r}; this scorer knows "
            f"{(CRITERION_PREREGISTERED, CRITERION_V2_ANY_COUNTER)}"
        )
    return str(criterion)


def region_oracle_length_m(start: tuple[float, float], spec) -> float | None:
    """SPL's ``l`` for stage 1 under v2: shortest path to the SUCCESS REGION.

    The v2 goal is a region (disc ∪ counter band), not a point, so ``l`` is the
    shortest achievable path from ``start`` to any pose satisfying
    :func:`position_success_v2` — the ObjectNav convention (Habitat: path to
    the nearest success viewpoint), where the pre-registered scoring used the
    PointNav convention (path to the goal point). Computed as a uniform-cost
    search over the same :class:`FreeSpaceGrid` the point oracle uses — same
    cells, same body-radius inflation, same no-corner-cutting rule — stopping
    at the first free cell whose centre satisfies the predicate. ``None`` if no
    free cell does (unreachable region — a layout defect, not a trial state).

    ``return_home`` keeps ``oracle_length`` (its criterion did not change).
    """
    g = grid()
    origin = g.nearest_free(*start)
    if origin is None:
        return None
    best: dict[tuple[int, int], float] = {origin: 0.0}
    heap: list[tuple[float, tuple[int, int]]] = [(0.0, origin)]
    diagonal = math.sqrt(2.0)
    while heap:
        cost, node = heapq.heappop(heap)
        if cost > best.get(node, math.inf):
            continue
        if position_success_v2(STAGE_FIND_KITCHEN, g.center(*node), spec):
            return cost
        i, j = node
        for di, dj in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ):
            ni, nj = i + di, j + dj
            if not (0 <= ni < g.nx and 0 <= nj < g.ny) or not g.free[nj][ni]:
                continue
            # Same rule as FreeSpaceGrid.path: never cut a corner diagonally.
            if di and dj and not (g.free[j][ni] and g.free[nj][i]):
                continue
            step = diagonal if (di and dj) else 1.0
            new_cost = cost + step * g.cell
            if new_cost < best.get((ni, nj), math.inf):
                best[(ni, nj)] = new_cost
                heapq.heappush(heap, (new_cost, (ni, nj)))
    return None


# ---------------------------------------------------------------------------
# 5.1/5.2/5.3/5.4/5.5 — per-stage metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageMetrics:
    """doc 06 §5.1–§5.5 and §5.8, for one stage of one trial."""

    stage: str
    outcome: str
    end_reason: str
    success: bool
    #: The AS-RUN verdicts (pre-registered point-disc criterion — what the live
    #: stage-2 gate consulted). Published beside the v2 fields so both readings
    #: stay reproducible from one scores.json.
    success_preregistered: bool
    outcome_preregistered: str
    #: Distance from the stage's end pose to the nearest kitchen-counter
    #: footprint (criterion v2's counter branch). ``NA`` for ``return_home``,
    #: whose criterion has no counter semantics.
    d_nearest_counter_face_m: float | str
    d_initial_m: float
    d_final_m: float
    progress: float
    oracle_path_m: float | str
    true_path_m: float
    spl: float
    #: §5.4 time-to-kitchen / time-to-home. ``NA`` on a failure, never 0.0.
    time_s: float | str
    turns_used: int
    #: §5.8, per stage. ``NA`` when the stage logged no turn (``not_run``).
    drift_m: float | str
    corrections: int
    correction_magnitudes_m: tuple[float, ...]

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "outcome": self.outcome,
            "end_reason": self.end_reason,
            "success": self.success,
            "success_preregistered": self.success_preregistered,
            "outcome_preregistered": self.outcome_preregistered,
            "d_nearest_counter_face_m": _round_or_na(self.d_nearest_counter_face_m, 4),
            "d_initial_m": round(self.d_initial_m, 4),
            "d_final_m": round(self.d_final_m, 4),
            "progress": round(self.progress, 4),
            "oracle_path_m": _round_or_na(self.oracle_path_m, 4),
            "true_path_m": round(self.true_path_m, 4),
            "spl": round(self.spl, 4),
            "time_s": _round_or_na(self.time_s, 4),
            "turns_used": self.turns_used,
            "drift_m": _round_or_na(self.drift_m, 4),
            "corrections": self.corrections,
            "correction_magnitudes_m": [
                round(m, 4) for m in self.correction_magnitudes_m
            ],
        }


def _round_or_na(value: float | str, digits: int) -> float | str:
    return value if isinstance(value, str) else round(value, digits)


def _stage_verdict(document: dict, stage: str, criterion: str) -> bool:
    """§5.1 for one stage under one named criterion: position AND declare_done.

    "The model must *know* it arrived" — criterion v2 widened only WHERE
    arrival counts, never HOW. Two distinct predicates live in the log and
    confusing them inflates SR: ``stages[*].score.success`` is the pure
    position test, while ``stages[*].success`` additionally requires
    ``declare_done``. A trial that times out standing inside the region logs
    the first as ``true`` and the second as ``false``.
    """
    result = document["final"]["stages"][stage]
    score = result.get("score")
    if score is None or result["end_reason"] != REASON_DECLARE_DONE:
        return False
    return bool(
        score_stage(
            stage_spec(document, stage), _score_xy(score), criterion=criterion
        ).success
    )


def validate_stage_log(document: dict, stage: str) -> str:
    """Cross-check the logged verdict against the criterion the trial RAN.

    doc 06 §9.1(iii): "the same predicate the live loop consulted, so the
    scorer and the gate cannot disagree". TR.2 makes the criterion explicit
    instead of implicit — the as-run predicate is now whatever
    :func:`trial_success_criterion` says the trial stamped, so this check is
    just as strict for a v2 trial as it was for a pre-registered one, and it
    NEVER re-interprets a legacy log's ``success`` field as a v2 verdict.

    Returns the as-run criterion so callers do not look it up twice.
    """
    criterion = trial_success_criterion(document)
    result = document["final"]["stages"][stage]
    logged = bool(result["success"])
    score = result.get("score")
    if score is None:
        # `not_run` only. StageResult.not_run is the sole writer of score=None.
        if logged:
            raise ScoringError(f"{stage}: success with no score block")
        return criterion
    recomputed = score_stage(
        stage_spec(document, stage), _score_xy(score), criterion=criterion
    )
    if not math.isclose(
        recomputed.distance_m,
        _finite(score["distance_m"], f"{stage}: score.distance_m"),
        abs_tol=1e-3,
    ):
        raise ScoringError(
            f"{stage}: logged distance {score['distance_m']} disagrees with "
            f"score_stage's {recomputed.distance_m:.4f} for true_xy "
            f"{tuple(score['true_xy'])} (doc 06 §9.1(iii))"
        )
    logged_criterion = score.get("criterion_version")
    if logged_criterion is not None and logged_criterion != criterion:
        raise ScoringError(
            f"{stage}: the score block claims criterion {logged_criterion!r} "
            f"but the trial config stamps {criterion!r} — one trial cannot have "
            "run two predicates"
        )
    expected = _stage_verdict(document, stage, criterion)
    if expected != logged:
        raise ScoringError(
            f"{stage}: logged success={logged} but the {criterion} predicate "
            f"says {recomputed.success} with end_reason={result['end_reason']!r}; "
            "doc 06 §5.1 requires BOTH the goal region and declare_done"
        )
    return criterion


def stage_success_preregistered(document: dict, stage: str) -> bool:
    """The PRE-REGISTERED §5.1 verdict — the point disc, recomputed.

    Published per trial (``success_preregistered``) so the original reading
    stays reproducible whatever the trial ran under, and validated on every
    scoring pass via :func:`validate_stage_log`.

    Recomputed, never read off ``final.stages[*].success``: for a v2-stamped
    trial that logged flag IS the v2 verdict, and returning it here would
    publish "pre-registered" numbers that are nothing of the sort.
    """
    validate_stage_log(document, stage)
    return _stage_verdict(document, stage, CRITERION_PREREGISTERED)


def stage_success(document: dict, stage: str) -> bool:
    """The PUBLISHED §5.1 success flag — criterion v2 (module docstring).

    Runs :func:`validate_stage_log` first, unconditionally: every
    log-consistency check still raises on a corrupt log, and the published
    verdict is only ever computed on a log that passed them.

    For a v2-stamped trial this equals the logged flag by construction — the
    live gate ran the same predicate, which is the whole point of TR.2. For a
    legacy trial it is the disclosed v2 SENSITIVITY result: what the published
    criterion says about a pose the live gate judged under the point disc. The
    live consequence of that gap (a return leg never offered) is not
    recoverable post hoc and is reported by
    ``stage1_successes_never_offered_return``.
    """
    validate_stage_log(document, stage)
    return _stage_verdict(document, stage, CRITERION_V2_ANY_COUNTER)


def check_stage_turns(document: dict, stage: str) -> list[dict]:
    """The stage's turns, cross-checked against ``final.stages[*].turns_used``.

    The same argument §5.6's bump cross-check already makes, applied to the field
    §5.5 publishes verbatim: ``turns_used`` is ``Counters.turns``, and every turn
    it counted was appended to ``turns[]``, so the two cannot disagree. Measured
    on the golden fixture before this check existed: setting
    ``turns_used = 1`` against four logged stage-1 turns published ``1`` with no
    complaint, while the analogous bump disagreement raised by design.

    Also rejects a ``declare_done`` stage with no logged turn — the shape a
    truncated log takes, which otherwise scores ``spl 1.0, progress 0.985,
    time_s 0.0`` for a robot that never moved (a teleport scores perfectly).
    """
    result = document["final"]["stages"][stage]
    turns = _stage_turns(document, stage)
    declared = int(result["turns_used"])
    if len(turns) != declared:
        raise ScoringError(
            f"{stage}: final.stages.{stage}.turns_used = {declared} but the log "
            f"holds {len(turns)} turns stamped with that stage (doc 06 §5.5); "
            "the same cross-check §5.6 makes for bumps"
        )
    if result["end_reason"] == REASON_DECLARE_DONE and not turns:
        raise ScoringError(
            f"{stage}: end_reason is {REASON_DECLARE_DONE!r} but no turn is "
            "stamped with this stage — `declare_done` is a tool call and tool "
            "calls live in turns, so this log is truncated, not a fast trial"
        )
    return turns


def stage_metrics(document: dict, stage: str) -> StageMetrics:
    """doc 06 §5.1–§5.5 + §5.8 for one stage — criterion v2 where it applies.

    v2 touches exactly four things here: ``success`` (the union predicate),
    ``outcome`` (recomputed from the v2 verdict), ``time_s`` (defined on the
    published success), and stage 1's ``oracle_path_m``/``spl`` (the region
    oracle). ``progress`` / ``d_initial_m`` / ``d_final_m`` deliberately keep
    the pre-registered point reference: they are continuous distance metrics
    whose comparability across the batch (and with the numbers published before
    the change) matters more than folding a discontinuous region distance —
    through a wall the nearest counter is metres of walking away at centimetres
    of Euclidean distance — into a gradient. The as-run verdicts are published
    beside the v2 ones, and the logged outcome is cross-checked against the
    as-run predicate so the log stays internally consistent under BOTH readings.
    """
    result = document["final"]["stages"][stage]
    as_run_criterion = validate_stage_log(document, stage)
    success = stage_success(document, stage)
    preregistered = stage_success_preregistered(document, stage)
    outcome = outcome_for(result["end_reason"], success)
    # Against the AS-RUN criterion, which for a legacy trial is the point disc
    # and for a v2-stamped trial is v2 — the same check either way, asked of
    # the predicate that actually wrote the field.
    expected_logged = outcome_for(
        result["end_reason"], _stage_verdict(document, stage, as_run_criterion)
    )
    if result["outcome"] != expected_logged:
        raise ScoringError(
            f"{stage}: logged outcome {result['outcome']!r} disagrees with the "
            f"as-run ({as_run_criterion}) predicate's {expected_logged!r} "
            f"(end_reason {result['end_reason']!r}); the log is internally "
            "inconsistent"
        )
    spec = stage_spec(document, stage)
    check_stage_turns(document, stage)

    start = stage_start_xy(document, stage)
    end = stage_end_xy(document, stage)
    d_initial = math.dist(start, spec.goal_xy)
    d_final = math.dist(end, spec.goal_xy)

    if stage == STAGE_FIND_KITCHEN:
        oracle = region_oracle_length_m(start, spec)
        d_counter: float | str = nearest_counter_face(end)[1]
    else:
        oracle = oracle_length(start, spec.goal_xy)
        d_counter = NA
    walked = path_length_m(document.get("turns", []), stage)
    floor, chords = chord_floor_m(document, stage)
    tolerance = CHORD_FLOOR_BASE_TOL_M + CHORD_FLOOR_TOL_M * chords
    if walked + tolerance < floor:
        raise ScoringError(
            f"{stage}: the traced path p = {walked:.4f} m is shorter than the "
            f"chord floor {floor:.4f} m implied by the same turns' `true_pose` "
            f"entries (tolerance {tolerance:.5f} m over {chords} chords). "
            "pose_trace samples were lost; every such loss INFLATES SPL, because "
            "max(p, l) caps the ratio at 1.0 (doc 06 §5.3)"
        )
    drift = stage_drift(document, stage)

    return StageMetrics(
        stage=stage,
        outcome=outcome,
        end_reason=result["end_reason"],
        success=success,
        success_preregistered=preregistered,
        # COMPUTED, not the logged field. For a legacy trial the two are equal
        # (the check above proves it); for a v2-stamped trial the logged
        # outcome IS the v2 outcome, so copying it would publish a
        # "pre-registered" column that silently duplicated the v2 one.
        outcome_preregistered=outcome_for(result["end_reason"], preregistered),
        d_nearest_counter_face_m=d_counter,
        d_initial_m=d_initial,
        d_final_m=d_final,
        progress=progress(d_initial, d_final),
        oracle_path_m=NA if oracle is None else oracle,
        true_path_m=walked,
        # `spl` short-circuits on S = 0 BEFORE touching `oracle`, so an unrun
        # stage with an unreachable/zero-length oracle scores 0.0, not a crash.
        spl=spl(success, oracle, walked),
        time_s=(
            _finite(result["policy_seconds_used"], f"{stage}: policy_seconds_used")
            if success
            else NA
        ),
        turns_used=int(result["turns_used"]),
        drift_m=drift.drift_m,
        corrections=drift.count,
        correction_magnitudes_m=drift.magnitudes_m,
    )


# ---------------------------------------------------------------------------
# 5.8 Dead-reckoning drift
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftMetrics:
    """doc 06 §5.8 for one stage: final drift + the corrections series."""

    stage: str
    drift_m: float | str
    estimate_xy: tuple[float, float] | None
    true_xy: tuple[float, float] | None
    count: int
    magnitudes_m: tuple[float, ...]
    #: True when a ``correct_position`` landed on the declaring turn and
    #: therefore supersedes ``obs.position_estimate`` (see :func:`stage_drift`).
    used_correction: bool = False
    #: Which of :func:`stage_drift`'s two pairings produced the number:
    #: ``"post_dispatch"`` (the logged end-of-turn belief against the same turn's
    #: ``true_pose``) or ``"pre_dispatch"`` (the start-of-turn belief against the
    #: true pose at that same instant, i.e. the PREVIOUS turn's). Published so a
    #: disputed drift number says which convention produced it.
    paired_at: str = "pre_dispatch"


def _stage_turns(document: dict, stage: str) -> list[dict]:
    return [t for t in document.get("turns", []) if t.get("stage") == stage]


def _final_corrections(document: dict) -> list[dict]:
    """The whole corrections series — the LAST turn's snapshot has all of it."""
    turns = document.get("turns", [])
    if not turns:
        return []
    return list(turns[-1].get("memory_snapshot", {}).get("corrections", []))


def stage_drift(document: dict, stage: str) -> DriftMetrics:
    """doc 06 §5.8: ‖position_estimate − true_pose‖ at the end of the stage.

    **The two halves must be sampled at the SAME INSTANT.** They are not stored
    that way: within one turn record ``obs.position_estimate`` is captured
    *before* the tool calls are dispatched and ``true_pose`` *after* them
    (``loop._run_turn`` steps 1 and 3c), while doc 05 §3.3 explicitly allows
    ``declare_done`` to follow a ``move`` in the same turn. Pairing them across
    that gap charges the model the whole length of its last move as "drift".

    Measured, before this was fixed: a trial whose ``position_estimate`` equals
    ground truth at the instant ``obs`` was captured on **every** turn — i.e. a
    mathematically perfect dead-reckoner — reported ``drift_m = 1.3583``, exactly
    ``math.dist((1.2, 0.9), (2.55, 0.75))``, the final move. The same two moves
    with ``declare_done`` split into its own turn reported 0.0000. That is a
    per-model *style* bias (bundling move + declare is a phrasing habit) on the
    headline metric of the whole memory-scaffolding claim.

    Two pairings, in preference order, both instant-consistent:

    1. ``turns[].position_estimate_end`` — the integrator AFTER dispatch, against
       that same turn's ``true_pose``. This is §5.8's "at the moment of
       ``declare_done``" verbatim. A ``correct_position`` on the turn is already
       folded in (it re-anchors the integrator), so nothing supersedes it. The
       key is optional because writing it is a ``loop.py`` change T4.1 reports
       rather than makes; every value it needs is already in memory
       (``tools._record_motion`` breadcrumbs ``integrator.xy`` after every
       motion) and simply not written out.
    2. Otherwise: the last turn's ``obs.position_estimate`` against the true pose
       at the instant that ``obs`` was captured — which is the PREVIOUS turn's
       logged ``true_pose`` (physics advances only inside ``env.step()`` and the
       sim is paused while the request is assembled, AGENTS.md §5), or the
       stage's start pose for the stage's first turn. A ``correct_position`` on
       that turn still supersedes the estimate: it is the model's corrected
       belief about the pose it was looking at. The cost of this fallback is
       stated plainly — it measures the belief one turn before the declaration,
       so drift accrued during a final bundled move is not counted. Under-
       measuring a fraction of one move is the honest error; the old pairing
       ADDED a whole move.

    **Which turn.** The stage's last logged turn — which *is* the declaring turn
    when the stage ended that way. §5.8's own rationale ("how honest the estimate
    ended up") applies just as much to a capped or fallen stage, and dropping
    those would delete the metric for exactly the trials most worth explaining.
    Only a stage with no turns at all (``not_run``) is ``NA``, which is what doc
    06 §3.2 requires and all it requires.

    **Corrections after the declaration are ignored** (§9.1). Filtering is by
    ``stage`` first — ``Correction.turn`` is stage-local, so two stages share
    turn numbers and only the stamped stage can split the series — then by
    ``turn <= declaring turn``.
    """
    turns = _stage_turns(document, stage)
    corrections = [c for c in _final_corrections(document) if c.get("stage") == stage]

    if not turns:
        # `not_run`: nothing was estimated and nothing was true. doc 06 §3.2
        # requires "—", excluded from means, never coerced to a number.
        return DriftMetrics(stage, NA, None, None, 0, ())

    last = turns[-1]
    cutoff = int(last["turn_idx"])
    in_scope = [c for c in corrections if int(c["turn"]) <= cutoff]
    magnitudes = tuple(
        math.dist(
            (_finite(c["old_xy"][0], "correction.old_xy"), _finite(c["old_xy"][1], "correction.old_xy")),
            (_finite(c["new_xy"][0], "correction.new_xy"), _finite(c["new_xy"][1], "correction.new_xy")),
        )
        for c in in_scope
    )

    end_estimate = last.get(POSITION_ESTIMATE_END)
    if isinstance(end_estimate, dict):
        estimate = (
            _finite(end_estimate["x"], f"{POSITION_ESTIMATE_END}.x"),
            _finite(end_estimate["y"], f"{POSITION_ESTIMATE_END}.y"),
        )
        true_xy = _pose_xy(last["true_pose"])
        return DriftMetrics(
            stage=stage,
            drift_m=math.dist(estimate, true_xy),
            estimate_xy=estimate,
            true_xy=true_xy,
            count=len(in_scope),
            magnitudes_m=magnitudes,
            used_correction=False,
            paired_at="post_dispatch",
        )

    shown = last["obs"]["position_estimate"]
    estimate = (
        _finite(shown["x"], "obs.position_estimate.x"),
        _finite(shown["y"], "obs.position_estimate.y"),
    )
    used_correction = False
    on_this_turn = [c for c in in_scope if int(c["turn"]) == cutoff]
    if on_this_turn:
        new = on_this_turn[-1]["new_xy"]
        estimate = (
            _finite(new[0], "correction.new_xy"),
            _finite(new[1], "correction.new_xy"),
        )
        used_correction = True

    # The true pose at the instant `obs` was captured = the end of the previous
    # turn, or the stage's own start when this is the stage's first turn.
    previous = _pose_xy(turns[-2]["true_pose"]) if len(turns) >= 2 else None
    true_xy = previous if previous is not None else stage_start_xy(document, stage)
    return DriftMetrics(
        stage=stage,
        drift_m=math.dist(estimate, true_xy),
        estimate_xy=estimate,
        true_xy=true_xy,
        count=len(in_scope),
        magnitudes_m=magnitudes,
        used_correction=used_correction,
        paired_at="pre_dispatch",
    )


# ---------------------------------------------------------------------------
# 5.6 Bumps and falls
# ---------------------------------------------------------------------------


def bumps(document: dict) -> int:
    """doc 06 §5.6's trial-scoped bump count, cross-checked against the turns.

    ``final.bumps`` is ``ToolContext.bumps``, which increments exactly when a
    call records ``counted_as_bump`` — §5.6 calls that flag "the ONLY per-turn
    source". Summing the flags must therefore reproduce the total, and a
    disagreement means the log is internally inconsistent (a lost turn, a merged
    file, a reset that should not have happened). That is exactly the class of
    defect this gate exists to catch before a paid batch, so it raises.
    """
    total = int(document["final"]["bumps"])
    counted = sum(
        1
        for turn in document.get("turns", [])
        for call in (turn.get("execution") or {}).get("calls", [])
        if call.get("counted_as_bump")
    )
    if counted != total:
        raise ScoringError(
            f"final.bumps = {total} but the turns record {counted} calls with "
            "counted_as_bump — doc 06 §5.6 names that flag the only per-turn "
            "source, so the two cannot disagree"
        )
    return total


def falls(document: dict) -> int:
    """doc 06 §5.6: ``falls ∈ {0, 1}`` — a fall ends the trial.

    The reason string comes from ``find_kitchen.REASON_FALL``, not a literal, so
    renaming the stop condition cannot silently zero this metric.
    """
    reasons = document["final"]["end_reason"].values()
    return 1 if any(reason == REASON_FALL for reason in reasons) else 0


# ---------------------------------------------------------------------------
# 5.7 Map accuracy
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    """prompts.normalize_room_name's cleaner: lowercase, punctuation → space."""
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return " ".join(cleaned.split())


def normalize_claim(name: str) -> str | None:
    """The frozen synonym table, applied to a model-coined room name.

    Imported, never re-authored: ``ROOM_SYNONYMS`` is frozen with the prompt
    (doc 06 §5.7, §2) and "kitchenette" is deliberately absent from it so §9.1's
    non-synonym near-string fixture exercises the *name* half of the matching
    rule. A second table here would let the two drift.
    """
    return ROOM_SYNONYMS.get(_clean(name))


def _name_grade(claimed: str, true_room: str) -> int:
    """doc 06 §5.7's "name-similarity" tie-break, made orderable.

    Under a fixed synonym table the name test is boolean, so "similarity" ranks
    nothing as written. The only two grades a frozen table can produce are: the
    claim IS the canonical name (2), or it reached it through a synonym (1).
    (T4.1 reports this as a proposed §5.7 wording edit.)
    """
    return 2 if _clean(claimed) == _clean(true_room) else 1


def _point_in_room(x: float, y: float, room: str) -> bool:
    """Through ``room_at``, not a second bounds test.

    ``room_at``'s boundaries are half-open on the max side so a point on a shared
    wall belongs to exactly one room. A closed-bounds copy here would credit a
    doorway-straddling pose to BOTH rooms, and §5.7's majority rule would then
    disagree with §2.10's visited set about where the robot was.
    """
    return room_at(x, y) == room


def executed_calls(turn: dict) -> list[dict]:
    """The turn's tool calls the harness actually RAN, in order.

    ``model_output.tool_calls`` is what the model emitted; ``dispatched`` is how
    many of them ``loop._run_turn`` got to before ``declare_done`` ended the
    stage, and every call after that is answered with ``not_executed``. Measured
    before this filter existed: a turn logged as ``[declare_done,
    set_current_room('kitchen')]`` still contributed a §5.7 evidence point for
    a call the harness had rejected, so the majority-of-evidence test could be
    tipped by claims that never ran.

    Known residual: a call with a ``parse_error`` IS counted in ``dispatched``
    (``tools.dispatch`` answers it with an error and never touches memory), and
    ``parse_errors`` records only names, not indices, so it cannot be excluded
    unambiguously. Such a call can only add evidence for a room name that some
    *other*, successful call already put in the memory snapshot — the snapshot is
    what ``claimed_rooms`` reads. Recorded in ``docs/METRICS.md`` §2.7.
    """
    output = turn.get("model_output") or {}
    calls = output.get("tool_calls") or []
    if not isinstance(calls, list):
        return []
    dispatched = output.get("dispatched")
    if isinstance(dispatched, int) and 0 <= dispatched <= len(calls):
        calls = calls[:dispatched]
    return [call for call in calls if isinstance(call, dict)]


def _claimed_room_name(call: dict) -> str | None:
    """The room name a room-claiming tool call names, or ``None``.

    ``args`` is tolerated as ``null`` rather than crashing the whole trial's
    score: the OpenAI adapter parses tool arguments out of a JSON string, so a
    provider-side artefact can land a ``null`` there, and one malformed call must
    not turn into an ``AttributeError`` in the middle of a 12-trial pass.
    """
    argument = ROOM_CLAIM_TOOLS.get(call.get("name", ""))
    if argument is None:
        return None
    args = call.get("args")
    if not isinstance(args, dict):
        return None
    name = args.get(argument)
    return name if isinstance(name, str) and name else None


def room_evidence(document: dict) -> dict[str, list[tuple[float, float]]]:
    """doc 06 §5.7's evidence points, per model-coined room name.

    Evidence = the true pose at the turns on which the model named that room via
    ``update_room`` / ``set_current_room`` / ``add_landmark``.

    **``add_landmark`` contributes the pose at the call, not a landmark
    position.** §5.7 also offers "the true positions of its claimed landmarks",
    which is not computable from anything the scorer may read: ``add_landmark``
    stores free text with no coordinates, the layout's per-room ``landmarks``
    lists are names only, and the sole coordinates live in
    ``LAYOUT["furniture"]``, whose own header says "SCENE SPEC ONLY — scoring
    never reads this". (T4.1 reports this as a proposed §5.7 wording edit.)

    The pose used is the turn's logged ``true_pose``, which is post-dispatch —
    the right vintage for the common case (drive into a room, then claim it) and
    at worst one turn's motion stale for a claim made before that turn's move.
    The majority rule below is what absorbs that.
    """
    evidence: dict[str, list[tuple[float, float]]] = {}
    for turn in document.get("turns", []):
        point = _pose_xy(turn.get("true_pose"))
        if point is None:
            continue
        for call in executed_calls(turn):
            name = _claimed_room_name(call)
            if name is not None:
                evidence.setdefault(name, []).append(point)
    return evidence


def claim_order(document: dict) -> dict[str, int]:
    """Global turn index of each room name's FIRST claim — §5.7's last tie-break."""
    order: dict[str, int] = {}
    for turn in document.get("turns", []):
        index = int(turn.get("global_turn_idx", turn.get("turn_idx", 0)))
        for call in executed_calls(turn):
            name = _claimed_room_name(call)
            if name is not None and name not in order:
                order[name] = index
    return order


def claimed_rooms(document: dict) -> list[str]:
    """The model's ``update_room`` entries, from the final memory snapshot."""
    turns = document.get("turns", [])
    if not turns:
        return []
    return list(turns[-1].get("memory_snapshot", {}).get("rooms", {}))


def claimed_edges(document: dict) -> list[tuple[str, str]]:
    """Undirected ``leads_to:`` edges, mirroring ``Memory.claimed_edges``.

    Unexplored exits are excluded — doc 06 §5.7: "claiming 'there's an unexplored
    exit north' is not an adjacency assertion". ``exit_status_target`` is
    imported from :mod:`duck_embody.agent.memory` so the ``leads_to:`` grammar
    has one parser.
    """
    turns = document.get("turns", [])
    if not turns:
        return []
    edges: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for exit_ in turns[-1].get("memory_snapshot", {}).get("exits", []):
        target = exit_status_target(exit_.get("status", ""))
        if target is None:
            continue
        key = frozenset((exit_["room"], target))
        if key in seen:
            continue
        seen.add(key)
        edges.append((exit_["room"], target))
    return edges


@dataclass(frozen=True)
class MapAccuracy:
    """doc 06 §5.7, trial-scoped (memory carries across the stage boundary)."""

    claimed: int
    matched: int
    true_rooms_visited: int
    #: ``NA`` when the model never called ``update_room`` — §5.7's edge
    #: convention: undefined, excluded from the aggregate, never coerced to 0.
    precision: float | str
    recall: float
    matches: tuple[tuple[str, str], ...]
    edges_claimed: int
    edges_correct: int
    #: ``NA`` when no ``leads_to:`` edge was ever claimed.
    edge_accuracy: float | str
    #: Claimed edges with an endpoint that matched no true room. They are counted
    #: WRONG (in ``edges_claimed``, not in ``edges_correct``), and this counter is
    #: published so a reader can recompute the other convention — §5.7's "edges
    #: between matched rooms" also reads as "exclude them", and the two
    #: conventions give different published numbers. See ``docs/METRICS.md`` §2.7.
    edges_unresolved: int = 0

    def as_dict(self) -> dict:
        return {
            "claimed": self.claimed,
            "matched": self.matched,
            "true_rooms_visited": self.true_rooms_visited,
            "precision": _round_or_na(self.precision, 4),
            "recall": round(self.recall, 4),
            "matches": [list(pair) for pair in self.matches],
            "edges_claimed": self.edges_claimed,
            "edges_correct": self.edges_correct,
            "edges_unresolved": self.edges_unresolved,
            "edge_accuracy": _round_or_na(self.edge_accuracy, 4),
        }


def match_rooms(
    claimed: Sequence[str],
    evidence: dict[str, Sequence[tuple[float, float]]],
    order: dict[str, int],
) -> list[tuple[str, str]]:
    """doc 06 §5.7's greedy one-to-one room matching.

    A claim matches a true room iff **both** halves hold: the name normalises to
    it through the frozen synonym table, AND a majority of the claim's evidence
    points fall inside that room's polygon. Because ``normalize_claim`` is a
    function, each claim has at most one candidate true room — so the one-to-one
    constraint only ever bites when two claims normalise to the same room, which
    is exactly §9.1's tie-break fixture.

    Greedy order: evidence count (desc), then name-similarity grade (desc), then
    earliest claim time (asc). Deterministic at every step, with the claim's own
    string as a final tie-break so two identical-looking claims cannot reorder
    between runs.
    """
    candidates: list[tuple[int, int, int, str, str]] = []
    for name in claimed:
        true_room = normalize_claim(name)
        if true_room is None:
            continue
        points = list(evidence.get(name, ()))
        if not points:
            # Nothing places this claim anywhere, so the polygon half of the
            # rule cannot be satisfied. (A room created by `update_room` always
            # has at least that call's pose; this is the defensive branch.)
            continue
        inside = sum(1 for x, y in points if _point_in_room(x, y, true_room))
        if inside * 2 <= len(points):
            continue  # doc 06 §5.7: MAJORITY of evidence must be inside
        candidates.append(
            (len(points), _name_grade(name, true_room), order.get(name, 0), name, true_room)
        )

    candidates.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))
    used_claims: set[str] = set()
    used_rooms: set[str] = set()
    matches: list[tuple[str, str]] = []
    for _count, _grade, _when, name, true_room in candidates:
        if name in used_claims or true_room in used_rooms:
            continue
        used_claims.add(name)
        used_rooms.add(true_room)
        matches.append((name, true_room))
    return matches


def map_accuracy(document: dict) -> MapAccuracy:
    """doc 06 §5.7: room-node precision/recall + adjacency edge accuracy."""
    claimed = claimed_rooms(document)
    evidence = room_evidence(document)
    matches = match_rooms(claimed, evidence, claim_order(document))
    visited = visited_rooms(document)

    precision: float | str = NA if not claimed else len(matches) / len(claimed)
    recall = 0.0 if not visited else len(matches) / len(visited)

    resolved = {name: true_room for name, true_room in matches}
    graph = adjacency()
    edges = claimed_edges(document)
    correct = 0
    unresolved = 0
    for a, b in edges:
        room_a, room_b = resolved.get(a), resolved.get(b)
        # An edge whose endpoint matched no true room is WRONG, not excluded:
        # it is still an adjacency assertion, and excluding it would make an
        # unmatched claim free after §5.7 already counted it against precision.
        # Counted separately so the other reading of §5.7 ("edges between matched
        # rooms" = only matched pairs are claimed edges) stays recomputable from
        # the published numbers. (T4.1 reports this as an open §5.7 wording point.)
        if room_a is None or room_b is None:
            unresolved += 1
            continue
        if room_a == room_b:
            continue
        if room_b in graph[room_a]:
            correct += 1
    edge_accuracy: float | str = NA if not edges else correct / len(edges)

    return MapAccuracy(
        claimed=len(claimed),
        matched=len(matches),
        true_rooms_visited=len(visited),
        precision=precision,
        recall=recall,
        matches=tuple(matches),
        edges_claimed=len(edges),
        edges_correct=correct,
        edge_accuracy=edge_accuracy,
        edges_unresolved=unresolved,
    )


# ---------------------------------------------------------------------------
# 5.9 Layout QA — shared text machinery
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: Sentence/clause boundaries for the QA matchers. A period between digits is
#: NOT a boundary ("22.5 degrees"), which is why this is not a bare ``[.]``.
#: Em and en dashes ARE boundaries — "Not north — the kitchen is northeast"
#: starts a new clause there — but the ASCII hyphen is not, because Q5 answers
#: use it as a list glue ("Sofa - living room").
_SENTENCE_BREAK_RE = re.compile(r"(?<!\d)\.(?!\d)|[;!?\n\r•—–]")


class Tokens:
    """One tokenisation shared by every matcher, so their indices always agree.

    Every Q2/Q4/Q5 matcher works in this index space and marks tokens it
    consumed, which is what makes "longest phrase wins, no overlaps" hold across
    tables that were written independently.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        matches = list(_TOKEN_RE.finditer(text))
        self.raw = [m.group(0) for m in matches]
        self.low = [word.lower() for word in self.raw]
        self.starts = [m.start() for m in matches]
        #: Per token: is there a comma immediately before it? Commas end a
        #: negation's scope but are NOT sentence boundaries — Q5 segments by
        #: sentence and "A sofa in the living room, a fridge in the kitchen."
        #: must stay one segmentation unit.
        commas = [m.start() for m in re.finditer(r",", text)]
        self.after_comma: list[bool] = []
        seen = 0
        for start in self.starts:
            passed = bisect.bisect_left(commas, start)
            self.after_comma.append(passed > seen)
            seen = passed
        breaks = [m.start() for m in _SENTENCE_BREAK_RE.finditer(text)]
        #: Per token, the index of the first token of the sentence it sits in.
        self.sentence_start: list[int] = []
        first_of_sentence = 0
        sentence_id = -1
        for index, start in enumerate(self.starts):
            current = bisect.bisect_left(breaks, start)
            if current != sentence_id:
                sentence_id = current
                first_of_sentence = index
            self.sentence_start.append(first_of_sentence)

    def __len__(self) -> int:
        return len(self.low)

    def index_at_char(self, position: int) -> int:
        return bisect.bisect_left(self.starts, position)


def _wrap180(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def _phrase_order(table) -> list[str]:
    """Longest phrase first (words, then characters) — no-overlap scan order."""
    return sorted(table, key=lambda s: (-len(s.split()), -len(s)))


_SYNONYM_ORDER = _phrase_order(ROOM_SYNONYMS)

#: ``ROOM_SYNONYMS`` entries that are ordinary English words for a *doorway*
#: rather than room names, and are therefore skipped when scanning FREE PROSE.
#: :func:`normalize_claim` — which normalises a whole claimed room NAME, §5.7 —
#: keeps the full frozen table, so a model that literally names a room "Entry"
#: is still matched. This is a scoring-local list, deliberately not an edit to
#: ``prompts.ROOM_SYNONYMS``: that table is a doc 06 §2 frozen fairness item
#: shared with T2.3's already-passed scene gate.
#:
#: Measured on the committed table before this existed: "walk east through the
#: ENTRY into the kitchen" parsed as living_room → hallway → kitchen, which cost
#: Q2 the exact-route defect AND recomputed the gold bearing to the hallway
#: doorway (69.810°), turning the correct "east" into a second defect — 1.0 → 0.0
#: against the identical sentence with "doorway". Also
#: ``extract_room_mentions('the landing gear') == ['hallway']``.
#:
#: Residual risk, stated rather than discovered later (``docs/METRICS.md`` §5):
#: "hall", "passage", "corridor" and "den" stay in the prose vocabulary because
#: they are genuine room words here far more often than not.
PROSE_AMBIGUOUS_SYNONYMS: frozenset[str] = frozenset({"entry", "entryway", "landing"})

#: Words that flip a clause into "I did NOT go/see there". A room or compass word
#: after one of these, inside the same sentence, is not a claim — it is the
#: opposite of one. Bare "no" is deliberately ABSENT: "No, I visited the living
#: room and the kitchen." is an affirmative answer that starts with it. The
#: apostrophe forms appear as their tokenised stems ("didn't" → "didn", "t").
NEGATION_CUES: frozenset[str] = frozenset({
    "not", "never", "nor", "neither", "without", "except", "excluding",
    "besides", "unlike", "avoid", "avoided", "skip", "skipped",
    "didn", "don", "doesn", "wasn", "weren", "isn", "aren",
    "couldn", "wouldn", "shouldn", "hadn", "haven", "hasn",
})

#: Contrastive conjunctions that END a negation's scope: "It is not clear, BUT I
#: visited the living room and the kitchen." Without these the negation would run
#: to the end of the sentence and delete two correct room names.
NEGATION_RESET_CUES: frozenset[str] = frozenset({
    "but", "however", "though", "although", "yet", "whereas", "instead", "rather",
})


def negated_tokens(tokens: Tokens) -> list[bool]:
    """Per token: is it inside the scope of a negation?

    Scope runs from the cue to the end of its **clause**, where a clause ends at
    a sentence boundary (including an em dash), a comma, or a contrastive
    conjunction. Clause-scoped rather than a fixed token window because the
    failing case needs the reach: in "I did not see the bedroom or the hallway."
    the second room sits six tokens past the cue, so a short window misses it —
    while "Not north — the kitchen is northeast" and "It is not clear, but I
    visited the living room and the kitchen" must NOT have their scope run on.

    Residual, recorded in ``docs/METRICS.md`` §2.9: a comma-separated negated
    list ("I did not see the bedroom, the hallway") only negates up to the comma.
    That direction is deliberate — it errs towards today's behaviour of counting
    a mention, and the harm it leaves is half a point rather than a full one.
    """
    flags = [False] * len(tokens)
    active = False
    previous_sentence = None
    for index, word in enumerate(tokens.low):
        sentence = tokens.sentence_start[index]
        if sentence != previous_sentence:
            active = False
            previous_sentence = sentence
        if tokens.after_comma[index] or word in NEGATION_RESET_CUES:
            active = False
        if word in NEGATION_CUES:
            active = True
            flags[index] = True
            continue
        flags[index] = active
    return flags


@dataclass(frozen=True)
class RoomMention:
    """One room named in free prose, in the shared token index space."""

    start: int
    end: int  # exclusive
    room: str
    negated: bool


def room_mention_spans(text: str) -> list[RoomMention]:
    """Every room named in ``text``, in positional order.

    A plural sibling of ``prompts.extract_room_mention`` over the *same* frozen
    table: whole-word, longest-phrase-first, non-overlapping, in positional
    order. Whole-word matching is why "kitchenette" still does not match, which
    doc 06 §12 requires and §9.1 fixtures.

    :data:`PROSE_AMBIGUOUS_SYNONYMS` are skipped here and only here; each hit
    carries whether it sits under a negation (:func:`negated_tokens`) so each
    question can decide what that means for it.
    """
    tokens = Tokens(text)
    taken = [False] * len(tokens)
    negated = negated_tokens(tokens)
    hits: list[RoomMention] = []
    for synonym in _SYNONYM_ORDER:
        if synonym in PROSE_AMBIGUOUS_SYNONYMS:
            continue
        parts = synonym.split()
        width = len(parts)
        for i in range(len(tokens) - width + 1):
            if any(taken[i : i + width]) or tokens.low[i : i + width] != parts:
                continue
            for k in range(i, i + width):
                taken[k] = True
            hits.append(
                RoomMention(i, i + width, ROOM_SYNONYMS[synonym], negated[i])
            )
    hits.sort(key=lambda mention: mention.start)
    return hits


def extract_room_mentions(text: str, *, drop_negated: bool = True) -> list[str]:
    """Rooms named in ``text``, in order, with immediate repeats collapsed.

    ``drop_negated`` removes rooms the answer explicitly says it did NOT visit —
    "I visited the living room and the kitchen. I did not see the bedroom or the
    hallway." is a fully correct Q3 answer that scored **0.0** while its mention
    set counted all four rooms. Enumerating what you did not visit is ordinary
    LLM answer style, so this fires on real answers.

    Q5 passes ``drop_negated=False``: there, mentions are structural anchors for
    segmenting the answer, not the answer itself.
    """
    sequence: list[str] = []
    for mention in room_mention_spans(text):
        if drop_negated and mention.negated:
            continue
        if not sequence or sequence[-1] != mention.room:
            sequence.append(mention.room)
    return sequence


# ---------------------------------------------------------------------------
# 5.9 Q2 — the direction-vocabulary parse rules (doc 06 §12, RESOLVED by T4.1)
# ---------------------------------------------------------------------------
#
# Committed BEFORE the batch, per PLAN T4.1's ordering, with 30 fixtures in
# tests/fixtures/qa_q2_answers.json. Everything below is measured against the
# COMMITTED layout, never against doc 06 §11's representative fixture layout —
# §11's own Q2 row (living_room -> hallway -> kitchen) does not reproduce here.

#: Degrees CCW from east — the system prompt's own convention, so a model that
#: answers in the frame it was driven in is read correctly.
ABSOLUTE_WORDS: dict[str, float] = {
    "east": 0.0, "eastward": 0.0, "eastwards": 0.0,
    "northeast": 45.0, "north east": 45.0, "northeastward": 45.0,
    "north": 90.0, "northward": 90.0, "northwards": 90.0,
    "northwest": 135.0, "north west": 135.0, "northwestward": 135.0,
    "west": 180.0, "westward": 180.0, "westwards": 180.0,
    "southwest": 225.0, "south west": 225.0, "southwestward": 225.0,
    "south": 270.0, "southward": 270.0, "southwards": 270.0,
    "southeast": 315.0, "south east": 315.0, "southeastward": 315.0,
}

#: Matched against the ORIGINAL text, UPPERCASE ONLY. Lowercase would read the
#: "e" of "i.e." as east — fixtured as ``q2_1_uppercase_abbrev``.
ABSOLUTE_ABBREV: dict[str, float] = {
    "E": 0.0, "NE": 45.0, "N": 90.0, "NW": 135.0,
    "W": 180.0, "SW": 225.0, "S": 270.0, "SE": 315.0,
}

#: Offsets from the current facing, CCW positive.
RELATIVE_WORDS: dict[str, float] = {
    "straight": 0.0, "straight ahead": 0.0, "straight on": 0.0,
    "forward": 0.0, "forwards": 0.0, "ahead": 0.0, "onward": 0.0, "onwards": 0.0,
    "left": 90.0, "leftward": 90.0, "leftwards": 90.0,
    "right": -90.0, "rightward": -90.0, "rightwards": -90.0,
    "slight left": 45.0, "slightly left": 45.0, "bear left": 45.0,
    "veer left": 45.0, "half left": 45.0, "diagonally left": 45.0,
    "slight right": -45.0, "slightly right": -45.0, "bear right": -45.0,
    "veer right": -45.0, "half right": -45.0, "diagonally right": -45.0,
    "sharp left": 135.0, "hard left": 135.0,
    "sharp right": -135.0, "hard right": -135.0,
    # Bare "back" and "behind" are DELIBERATELY absent: "the back wall" and
    # "walk behind the counter" are not 180° turns. Only unambiguous forms:
    "backward": 180.0, "backwards": 180.0,
    "go back": 180.0, "head back": 180.0, "walk back": 180.0, "turn back": 180.0,
    "double back": 180.0, "doubling back": 180.0,
    "turn around": 180.0, "u turn": 180.0, "about face": 180.0, "reverse": 180.0,
}

#: "right" is also an intensifier ("go RIGHT through the door"). Blocked only
#: before adverbial heads that can never follow a turn instruction. "at" and
#: "into" are deliberately ABSENT — "turn right at the doorway" and "turn right
#: into the kitchen" are real turns.
RIGHT_ADVERB_BLOCKLIST: frozenset[str] = frozenset({
    "through", "past", "away", "up", "down", "before", "after", "next",
    "alongside", "by", "over", "across", "back", "along", "outside", "inside",
})

#: A direction token counts only if a motion cue occurs within
#: :data:`CUE_WINDOW` tokens before it, or the token's own first word is a cue
#: ("bear left", "turn around"). This is what stops "The sofa is against the
#: WEST wall" being read as the initial direction. There is NO fallback to
#: un-cued tokens: an answer that only describes the layout has a "missing turn
#: direction", which the frozen 0.5 anchor already covers verbatim.
CUE_WORDS: frozenset[str] = frozenset({
    "turn", "turns", "turning", "head", "heads", "heading", "face", "facing",
    "go", "goes", "going", "walk", "walks", "walking", "move", "moves",
    "moving", "proceed", "proceeds", "continue", "continues", "bear", "veer",
    "travel", "step", "steps", "drive", "exit", "exits", "toward", "towards",
    "direction", "aim", "keep", "follow", "rotate", "pivot",
})
CUE_WINDOW = 4

#: Half of one 45° compass sector on each side, applied to the CONTINUOUS
#: bearing rather than to ``compass_8`` buckets. WHY not buckets: the oracle A*
#: path's first 1.0 m bears 336.80°, which is 0.70° from the SE/E bucket
#: boundary — bucketing would make Q2's headline bit turn on 0.7°. The wedge
#: accepts "east" (17.0° off) and "southeast" (28.0° off — the leg that actually
#: clears the coffee table, whose inflated SW corner bears 330.8° from the
#: start), and rejects "northeast" (62.1°) and "left"/"north" (72.9°).
#: [measured 2026-07-26 from the committed layout with its own helpers]
DIRECTION_TOL_DEG = 45.0

#: doc 06 §5.9's "Decide before freeze whether Q2 accepts any collision-free
#: route that reaches the fridge". DECIDED: one extra room (the hallway detour,
#: 3.611 m against 3.152 m direct) is a VALID route but not the ORACLE route, so
#: it costs exactly one defect — 0.5, never 1 and never 0. That is the only
#: reading consistent with the frozen anchors: the 1 anchor requires the
#: sequence to match the *oracle* route, and the 0 anchor is "route would not
#: reach the kitchen", which the hallway route plainly does.
MAX_EXTRA_ROOMS = 1

#: East. Only "straight/forward/ahead" and the ±45 relative tokens depend on it:
#: "left" and "right" are wrong under BOTH readings of "the front of the sofa"
#: (facing away from it, or facing it), so the constant is not load-bearing for
#: the common cases. East is the sofa's front: its long axis runs north-south
#: (footprint 0.391 × 0.975) against the west wall, with the coffee table
#: (0.88, 1.60) and the blue rug (0.95, 1.60) directly east of it.
INITIAL_FACING_DEG = 0.0

GOAL_OBJECT_TERMS: tuple[str, ...] = ("fridge", "refrigerator")

_ABS_DEG_RE = re.compile(
    r"(?:heading|bearing|azimuth)\s+(?:of\s+)?(-?\d+(?:\.\d+)?)"
    r"|(-?\d+(?:\.\d+)?)\s*(?:deg\b|degs\b|degree\b|degrees\b|°)",
    re.I,
)
_REL_DEG_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:deg\b|degs\b|degree\b|degrees\b|°)"
    r"\s*(?:to\s+the\s+)?(left|right)"
    r"|turn\w*\s+(left|right)\s+(?:by\s+)?(-?\d+(?:\.\d+)?)"
    r"\s*(?:deg\b|degs\b|degree\b|degrees\b|°)",
    re.I,
)

_REL_ORDER = _phrase_order(RELATIVE_WORDS)
_ABS_ORDER = _phrase_order(ABSOLUTE_WORDS)


@dataclass(frozen=True)
class DirectionToken:
    """One direction the answer expresses, in the shared token index space."""

    index: int
    kind: str  # "abs" (a heading) | "rel" (an offset from the current facing)
    token: str
    value: float
    cued: bool
    #: Under a negation ("NOT north — the kitchen is northeast"). Q4 skips these;
    #: Q2's cue anchoring already handles the instruction case.
    negated: bool = False

    def resolve(self, facing_deg: float) -> float:
        if self.kind == "abs":
            return self.value % 360.0
        return (facing_deg + self.value) % 360.0


def direction_tokens(text: str) -> list[DirectionToken]:
    """Every direction the answer expresses, in reading order.

    Explicit degrees are matched first and their tokens consumed, so
    "turn 90 degrees left" is never *also* read as the absolute heading 90.
    """
    tokens = Tokens(text)
    taken = [False] * len(tokens)
    found: list[tuple[int, str, str, float]] = []

    def consume(start_char: int, end_char: int) -> int:
        """Mark every token that STARTS inside ``[start_char, end_char)``.

        ``end_char`` is the match's exclusive end, so ``index_at_char(end_char)``
        is the first token *past* the span: marking it swallowed a direction word
        that immediately followed an explicit-degrees phrase. Measured before the
        clamp: "turn 90 degrees left east then stop" yielded one token and lost
        "east" — invisible in ``first_direction`` (the swallowed token is never
        first) but a silently missing leg in ``direction_sequence``, which doc 06
        §10.4's figures and any per-answer audit consume.
        """
        first = tokens.index_at_char(start_char)
        last = tokens.index_at_char(end_char) - 1
        for k in range(first, min(last, len(taken) - 1) + 1):
            taken[k] = True
        return first

    for match in _REL_DEG_RE.finditer(text):
        degrees = match.group(1) or match.group(4)
        side = (match.group(2) or match.group(3)).lower()
        index = consume(match.start(), match.end())
        found.append(
            (index, "rel", match.group(0).strip(),
             float(degrees) * (1.0 if side == "left" else -1.0))
        )
    for match in _ABS_DEG_RE.finditer(text):
        index = tokens.index_at_char(match.start())
        if any(taken[index : index + 4]):
            continue
        degrees = match.group(1) or match.group(2)
        consume(match.start(), match.end())
        found.append((index, "abs", match.group(0).strip(), float(degrees)))

    for table, kind, order in (
        (RELATIVE_WORDS, "rel", _REL_ORDER),
        (ABSOLUTE_WORDS, "abs", _ABS_ORDER),
    ):
        for phrase in order:
            parts = phrase.split()
            width = len(parts)
            for i in range(len(tokens) - width + 1):
                if any(taken[i : i + width]) or tokens.low[i : i + width] != parts:
                    continue
                if (
                    phrase == "right"
                    and i + 1 < len(tokens)
                    and tokens.low[i + 1] in RIGHT_ADVERB_BLOCKLIST
                ):
                    continue
                for k in range(i, i + width):
                    taken[k] = True
                found.append((i, kind, phrase, table[phrase]))

    for i, raw in enumerate(tokens.raw):  # uppercase-only abbreviations
        if taken[i] or raw not in ABSOLUTE_ABBREV:
            continue
        if not _is_standalone_abbrev(text, tokens.starts[i] + len(raw)):
            continue
        taken[i] = True
        found.append((i, "abs", raw, ABSOLUTE_ABBREV[raw]))

    negated = negated_tokens(tokens)
    out: list[DirectionToken] = []
    for index, kind, token, value in found:
        own = token.split()[0].lower()
        window = tokens.low[max(0, index - CUE_WINDOW) : index]
        cued = own in CUE_WORDS or any(word in CUE_WORDS for word in window)
        out.append(
            DirectionToken(index, kind, token, value, cued, negated[index])
        )
    out.sort(key=lambda token: token.index)
    return out


def _is_standalone_abbrev(text: str, end: int) -> bool:
    """Is the abbreviation ending at ``end`` a compass claim, or punctuation?

    The uppercase-only rule was written to stop the "e" of "i.e." being read as
    east, and the docstring of :data:`ABSOLUTE_ABBREV` says so — but the
    sentence-initial "E.g." defeats it, and "N/A" is worse: measured on seed 101
    (gold NE), ``score_q4('N/A') == 0.5``, i.e. a model that declined to answer
    collected half a point, and ``score_q4('E.g. somewhere to the southwest…')``
    scored the abbreviation of *exempli gratia* rather than the model's actual
    answer.

    Rejected: followed by "/" ("N/A", "S/N"), or by "." immediately followed by
    an alphanumeric ("E.g.", "N.B."). A trailing "." at a sentence end is fine —
    "The kitchen is NE." must still score.
    """
    tail = text[end : end + 2]
    if tail[:1] == "/":
        return False
    if tail[:1] == "." and tail[1:2].isalnum():
        return False
    return True


def first_direction(text: str) -> DirectionToken | None:
    """The first CUE-ANCHORED direction — the only one Q2's rubric scores.

    "initial direction correct" is all the frozen anchor asks for, and the
    answers carry no reliable distances, so simulating the whole route would
    invent geometry the model never stated.
    """
    for token in direction_tokens(text):
        if token.cued:
            return token
    return None


def direction_sequence(text: str) -> list[tuple[str, float]]:
    """The full resolved heading sequence — for the record and T4.4's figures.

    Deliberately NOT scored: see :func:`first_direction`.
    """
    facing = INITIAL_FACING_DEG
    sequence: list[tuple[str, float]] = []
    for token in direction_tokens(text):
        if not token.cued:
            continue
        facing = token.resolve(facing)
        sequence.append((token.token, facing))
    return sequence


def _furniture(name: str) -> dict:
    for item in LAYOUT["furniture"]:
        if item["name"] == name:
            return item
    raise ScoringError(f"the layout has no furniture item named {name!r}")


def q2_start_point() -> tuple[float, float]:
    """"The front of the sofa": the midpoint of the sofa's EAST face.

    Derived from ``LAYOUT`` so the scene spec and the answer key cannot drift
    (AGENTS.md §2). [measured 2026-07-26: (0.4955, 1.60); the sofa CENTRE gives a
    gold bearing 2.1° away and the front face plus one body radius 1.0° away —
    all three inside :data:`DIRECTION_TOL_DEG`, so the choice moves no fixture
    except the two deliberate tolerance-boundary probes.]
    """
    sofa = _furniture("sofa")
    return (sofa["pos"][0] + sofa["footprint"][0] / 2.0, sofa["pos"][1])


def q2_start_room() -> str:
    return room_at(*_furniture("sofa")["pos"])


def q2_goal_room() -> str:
    return room_at(*_furniture("fridge")["pos"])


def q2_oracle_route() -> list[str]:
    """``["living_room", "kitchen"]`` on the committed layout — COMPUTED.

    Never transcribed from doc 06 §11, whose row was produced against that
    section's own representative layout dict and says
    ``living_room → hallway → kitchen``. The committed layout has a direct
    living_room↔kitchen doorway at (1.8, 1.2). ``tests/test_layout.py`` already
    pins the two-room path. [measured: oracle_length(sofa, fridge) = 3.1521 m
    direct vs 3.6107 m via the hallway waypoints, +14.55 %]
    """
    return room_path(q2_start_room(), q2_goal_room())


def doorway_center(a: str, b: str) -> tuple[float, float] | None:
    for door in LAYOUT["doorways"]:
        if set(door["between"]) == {a, b}:
            return (float(door["center"][0]), float(door["center"][1]))
    return None


def q2_gold_bearing(route: Sequence[str]) -> float:
    """The gold initial bearing FOR THE ROUTE THE ANSWER DESCRIBES.

    An answer that offers the hallway detour is scored against the bearing to the
    hallway doorway (0.9, 2.7) = 69.810°, not the direct doorway (1.8, 1.2) =
    342.953°. Otherwise a hallway answer would be penalised twice — once for the
    extra room, once for the direction that is correct *for* that room — for a
    single decision.
    """
    oracle = q2_oracle_route()
    if len(route) >= 2 and route[0] == oracle[0] and doorway_center(route[0], route[1]):
        pair = (route[0], route[1])
    else:
        pair = (oracle[0], oracle[1])
    return bearing_deg(q2_start_point(), doorway_center(*pair))


@dataclass(frozen=True)
class Q2Parse:
    """Everything :func:`score_q2` decided, so a disputed score is auditable."""

    rooms: tuple[str, ...]
    normalized: tuple[str, ...]
    reversed_order: bool
    reaches_goal: bool
    walkable: bool
    exact_route: bool
    direction: str | None
    direction_deg: float | None
    gold_bearing_deg: float
    direction_ok: bool
    names_goal_object: bool
    defects: int | None
    score: float


def parse_q2(answer: str) -> Q2Parse:
    """Q2's full parse. See :func:`score_q2` for the ladder and the anchors."""
    start, goal = q2_start_room(), q2_goal_room()
    oracle = q2_oracle_route()
    graph = adjacency()

    raw = extract_room_mentions(answer)
    sequence = list(raw)
    direction = first_direction(answer)
    reversed_order = False

    # N1 (in extract_room_mentions): collapse runs of the same room.
    # N2: drop a goal-room-only preamble ("The fridge is in the kitchen. From
    #     the living room, ..."), ONLY if the remainder still reaches the goal —
    #     that guard is what keeps N2 from destroying a reverse-order answer.
    if sequence and sequence[0] == goal and start in sequence[1:]:
        cut = sequence.index(start)
        if all(room == goal for room in sequence[:cut]) and goal in sequence[cut:]:
            sequence = sequence[cut:]
    # N3: prepend the implied start room, ONLY when the answer names no start
    #     room AND gave a cue-anchored direction. Without the direction
    #     condition, "The fridge is in the kitchen. I do not remember the way."
    #     would be normalized into the oracle route.
    if sequence and start not in sequence and direction is not None and sequence[0] in graph[start]:
        sequence = [start] + sequence

    def walkable(candidate: list[str]) -> bool:
        return (
            bool(candidate)
            and candidate[-1] == goal
            and len(candidate) <= len(oracle) + MAX_EXTRA_ROOMS
            and all(b in graph[a] for a, b in zip(candidate, candidate[1:]))
        )

    # N4: reverse-order salvage ("walk into the kitchen from the living room").
    #     ALWAYS counts as a defect below, so a backwards-phrased answer can
    #     never score 1.0 — the parser cannot tell a phrasing quirk from a
    #     genuinely reversed route.
    if not walkable(sequence) and walkable(list(reversed(sequence))):
        sequence = list(reversed(sequence))
        reversed_order = True

    reaches = goal in sequence
    walks = walkable(sequence)
    exact = sequence == oracle and not reversed_order
    gold = q2_gold_bearing(sequence)
    resolved = None if direction is None else direction.resolve(INITIAL_FACING_DEG)
    direction_ok = resolved is not None and abs(_wrap180(resolved - gold)) <= DIRECTION_TOL_DEG
    names_goal = any(term in answer.lower() for term in GOAL_OBJECT_TERMS)

    if not reaches or not walks:
        defects = None
        score = 0.0
    else:
        defects = (
            (0 if exact else 1)
            + (0 if direction_ok else 1)
            + (0 if names_goal else 1)
        )
        score = {0: 1.0, 1: 0.5}.get(defects, 0.0)

    return Q2Parse(
        rooms=tuple(raw),
        normalized=tuple(sequence),
        reversed_order=reversed_order,
        reaches_goal=reaches,
        walkable=walks,
        exact_route=exact,
        direction=None if direction is None else direction.token,
        direction_deg=resolved,
        gold_bearing_deg=gold,
        direction_ok=direction_ok,
        names_goal_object=names_goal,
        defects=defects,
        score=score,
    )


def score_q2(answer: str) -> float:
    """doc 06 §5.9 Q2, reproducing all three frozen anchors exactly.

    Floor first — "route would not reach the kitchen" is the 0 anchor, so an
    answer whose room sequence is not a graph walk ending in the kitchen scores
    0 whatever else it got right. Then one point per independent defect:

    * 0 defects → 1.0 = "room sequence matches the oracle route AND initial
      direction correct AND ends at the fridge";
    * 1 defect → 0.5 = "correct room sequence but a wrong/missing turn
      direction" **or** "correct directions with one wrong room name";
    * ≥2 defects → 0.0. This is the one EXTENSION beyond the frozen anchors (they
      enumerate no multi-defect case) and is recorded in ``docs/METRICS.md``.
    """
    return parse_q2(answer).score


# ---------------------------------------------------------------------------
# 5.9 Q1 / Q3 / Q4 / Q5
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QAContext:
    """The ground truth the five questions are scored against."""

    spawn_xy: tuple[float, float]
    visited: tuple[str, ...]


def qa_context(document: dict) -> QAContext:
    return QAContext(spawn_xy=spawn_xy(document), visited=visited_rooms(document))


#: Q1 restates its own two rooms ("Which room connects the BEDROOM to the
#: KITCHEN?"), and models answer in full sentences, so the first room mentioned
#: is usually one of the two from the question rather than the answer. These are
#: skipped when picking the answered room; if nothing else was named, the first
#: mention is used, so an answer of literally "the kitchen" still scores.
Q1_QUESTION_ROOMS: tuple[str, str] = ("bedroom", "kitchen")


def score_q1(answer: str) -> float:
    """"Which room connects the bedroom to the kitchen?" (doc 06 §5.9).

    1: the unique connector per the adjacency graph. 0.5: a room adjacent to
    exactly one of the two. 0: anything else — including an unparseable answer.
    """
    mentions = extract_room_mentions(answer)
    if not mentions:
        return 0.0
    named = next(
        (room for room in mentions if room not in Q1_QUESTION_ROOMS), mentions[0]
    )
    if named in connecting_rooms(*Q1_QUESTION_ROOMS):
        return 1.0
    neighbours = adjacency()[named] & set(Q1_QUESTION_ROOMS)
    return 0.5 if len(neighbours) == 1 else 0.0


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
#: "three rooms", "3 rooms", "four different rooms", "all 4 rooms". Anchored on
#: the word "room(s)" so an ordinal inside prose ("Room 1 was the kitchen")
#: cannot be read as the count.
_COUNT_RE = re.compile(
    r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\b\s+"
    r"(?:different\s+|distinct\s+|separate\s+|other\s+|total\s+)?rooms?\b",
    re.I,
)


def stated_count(answer: str) -> int | None:
    """The count Q3 asked for, if the answer states one near the word "room"."""
    match = _COUNT_RE.search(answer)
    if match is None:
        return None
    token = match.group(1).lower()
    return int(token) if token.isdigit() else _NUMBER_WORDS[token]


def score_q3(answer: str, context: QAContext) -> float:
    """"How many rooms did you visit? Name them." (doc 06 §5.9).

    1: count and names match the true visited set. A count the answer never
    states is taken as implied by the naming — a model that names exactly the
    right rooms has answered "how many" — so only a *stated* count that is wrong
    can cost anything. 0.5: names correct but the stated count is off by one, or
    one room missing/extra. 0: otherwise.

    Rooms the answer explicitly says it did **not** visit are not counted (see
    :func:`extract_room_mentions`): "I visited two rooms, the living room and the
    kitchen. I did not see the bedroom or the hallway." is a fully correct answer
    that scored 0.0 when every mention counted.
    """
    named = set(extract_room_mentions(answer))
    truth = set(context.visited)
    if not named:
        return 0.0
    difference = len(named ^ truth)
    count = stated_count(answer)
    if difference == 0:
        if count is None or count == len(truth):
            return 1.0
        return 0.5 if abs(count - len(truth)) == 1 else 0.0
    if difference == 1:
        return 0.5
    return 0.0


def compass_tokens(text: str) -> list[str]:
    """Compass words / uppercase abbreviations in ``text``, in reading order.

    Reuses Q2's direction tokenizer, restricted to *absolute* tokens: Q4 asks for
    a compass direction, not an instruction, so "left" is not an answer and no
    cue anchoring applies. Bucketing goes through
    ``apartment_layout.compass_8`` — doc 06 §5.9 requires that exact function
    rather than a second bucketer.

    A token the answer **negates** is dropped: "Not north — the kitchen is
    northeast of the spawn." states NE unambiguously and used to score 0.5,
    because the rejected direction is the one that got scored.
    """
    return [
        compass_8(token.value % 360.0)
        for token in direction_tokens(text)
        if token.kind == "abs" and not token.negated
    ]


def q4_gold(context: QAContext) -> str:
    """spawn → kitchen centroid, bucketed by ``compass_8``.

    [measured 2026-07-26 from the committed layout: seed 101 = NE (22.521°, i.e.
    0.021° past the E/NE boundary), 102 = SW, 103 = SE, 104 = SE. Doc 06 §11's
    E/E/SE/SE was computed against its representative layout and does not
    transfer.]
    """
    return compass_8(bearing_deg(context.spawn_xy, room_centroid("kitchen")))


def score_q4(answer: str, context: QAContext) -> float:
    """"Which direction (compass) is the kitchen from your spawn point?"

    1: matches the bucketed true bearing. 0.5: an adjacent bucket. 0: otherwise.

    The **first** surviving token is scored: it is the answer's leading claim,
    and the alternative (score the last) reads "Northeast. The bedroom is to the
    south." as an answer of *south*. Negated tokens are already gone
    (:func:`compass_tokens`), which is what makes the first token the model's
    actual claim in the common contrast phrasings. The residual, recorded in
    ``docs/METRICS.md`` §2.9: an un-negated self-correction ("east… more
    precisely northeast") is scored on the first form.
    """
    tokens = compass_tokens(answer)
    if not tokens:
        return 0.0
    gold = q4_gold(context)
    given = tokens[0]
    if given == gold:
        return 1.0
    separation = abs(COMPASS_8.index(given) - COMPASS_8.index(gold))
    return 0.5 if min(separation, 8 - separation) == 1 else 0.0


def _landmark_terms() -> dict[str, tuple[tuple[str, ...], ...]]:
    """Per-room landmark phrases plus their head nouns, from the layout.

    The head noun ("table" for "coffee table") is accepted only because every
    head noun in the committed layout is unique across rooms; an ambiguous one is
    dropped rather than guessed. ``tests/test_scoring.py`` asserts the uniqueness
    holds, so a layout edit that breaks it fails the gate instead of silently
    crediting the wrong room.
    """
    heads: dict[str, set[str]] = {}
    for room, spec in LAYOUT["rooms"].items():
        for landmark in spec["landmarks"]:
            heads.setdefault(_clean(landmark).split()[-1], set()).add(room)
    terms: dict[str, list[tuple[str, ...]]] = {}
    for room, spec in LAYOUT["rooms"].items():
        room_terms: list[tuple[str, ...]] = []
        for landmark in spec["landmarks"]:
            words = tuple(_clean(landmark).split())
            room_terms.append(words)
            head = words[-1]
            if len(words) > 1 and len(heads[head]) == 1:
                room_terms.append((head,))
        terms[room] = room_terms
    return {room: tuple(value) for room, value in terms.items()}


LANDMARK_TERMS: dict[str, tuple[tuple[str, ...], ...]] = _landmark_terms()


def _contains_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    width = len(phrase)
    return any(
        tuple(tokens[i : i + width]) == tuple(phrase)
        for i in range(len(tokens) - width + 1)
    )


def score_q5(answer: str, context: QAContext) -> float:
    """"Name one landmark in each room you visited." (doc 06 §5.9).

    1: a true layout landmark for every visited room. 0.5: correct for all but
    one room. 0: otherwise.

    The answer is segmented by room mention, so a landmark is credited to the
    room it was actually attached to — which is what "in EACH room" means, and is
    why "Living room: the fridge. Kitchen: the blue rug." scores 0.

    **Each mention owns text on BOTH sides of itself**, because English attaches
    landmarks to rooms in both orders and a model picks one habitually:

    * forward, to the next mention — "Living room: the sofa. Kitchen: the fridge.";
    * backward, but no further than the previous mention and **never across a
      sentence boundary** — "The sofa is in the living room and the fridge is in
      the kitchen."

    Forward-only was the original rule and scored every backward phrasing 0.0
    while the answer named a true landmark for every visited room: measured, "A
    sofa in the living room, a fridge in the kitchen." → 0.0, "Sofa - living
    room. Fridge - kitchen." → 0.0, "I saw a blue rug (living room) and a fridge
    (kitchen)." → 0.0. The sentence clamp is what keeps the swap case honest: it
    stops "Kitchen:" from reaching back over the full stop and claiming the
    living room's landmark.

    A visited room the answer never names by a synonym-table name is simply not
    credited; models coin their own room names (the system prompt tells them to),
    and that limitation is recorded in ``docs/METRICS.md``.

    Negated mentions are kept here (unlike Q1/Q3/Q4): a mention is a structural
    anchor for the segmentation, not a claim about having visited.
    """
    if not context.visited or not answer.strip():
        # The blank guard is duplicated from `score_answer` deliberately, and is
        # belt-and-braces rather than load-bearing: the `correct >= 1` condition
        # below is the general form of the same rule and already returns 0.0
        # here. Before that rule existed,
        # `score_q5('', QAContext(visited=('living_room',)))` returned 0.5 — half
        # credit for an empty answer — and ONLY `score_answer`'s guard, which no
        # fixture stressed, stood between that and a published number.
        return 0.0
    tokens = Tokens(answer)
    spans = room_mention_spans(answer)
    segments: dict[str, list[str]] = {}
    for position, mention in enumerate(spans):
        forward = (
            spans[position + 1].start if position + 1 < len(spans) else len(tokens)
        )
        backward = max(
            spans[position - 1].end if position else 0,
            tokens.sentence_start[mention.start],
        )
        segments.setdefault(mention.room, []).extend(tokens.low[backward:forward])

    correct = 0
    for room in context.visited:
        words = segments.get(room)
        if words and any(
            _contains_phrase(words, phrase) for phrase in LANDMARK_TERMS[room]
        ):
            correct += 1
    missing = len(context.visited) - correct
    if missing == 0:
        return 1.0
    # "correct for all but one room" presupposes at least one correct room; with
    # a single visited room, `missing == 1` means nothing was right.
    return 0.5 if missing == 1 and correct >= 1 else 0.0


# ---------------------------------------------------------------------------
# 5.9 — the whole QA block
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QAResult:
    """doc 06 §5.9: five 0/0.5/1 scores and their mean."""

    scores: tuple[float, ...]
    answers: tuple[str, ...]

    @property
    def score(self) -> float:
        return sum(self.scores) / len(self.scores)

    def as_dict(self) -> dict:
        return {
            "per_question": [round(s, 3) for s in self.scores],
            "score": round(self.score, 4),
        }


def score_answer(number: int, answer: str, context: QAContext) -> float:
    """Score one QA answer. ``""`` — the loop's honest record of an unparseable
    answer — scores 0 rather than the harness inventing text (doc 06 §5.9)."""
    if not answer or not answer.strip():
        return 0.0
    if number == 1:
        return score_q1(answer)
    if number == 2:
        return score_q2(answer)
    if number == 3:
        return score_q3(answer, context)
    if number == 4:
        return score_q4(answer, context)
    if number == 5:
        return score_q5(answer, context)
    raise ScoringError(f"no rubric for QA question {number}")


def score_qa(document: dict) -> QAResult:
    """Score ``final.qa`` against the layout + the logged true trace."""
    entries = document["final"].get("qa")
    expected = [question.number for question in LAYOUT_QA_QUESTIONS]
    if not isinstance(entries, list) or [e.get("number") for e in entries] != expected:
        raise ScoringError(
            "final.qa must be the five frozen questions numbered "
            f"{expected} (doc 06 §5.9); got "
            f"{None if entries is None else [e.get('number') for e in entries]}"
        )
    context = qa_context(document)
    answers = tuple(entry.get("answer") or "" for entry in entries)
    scores = tuple(
        score_answer(entry["number"], answer, context)
        for entry, answer in zip(entries, answers)
    )
    return QAResult(scores=scores, answers=answers)


# ---------------------------------------------------------------------------
# The whole trial
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialMetrics:
    """doc 06 §5 for one trial — what fills ``final.metrics``."""

    trial_id: str
    model: str
    seed: int
    stages: dict[str, StageMetrics]
    bumps: int
    falls: int
    map_accuracy: MapAccuracy
    qa: QAResult
    visited_rooms: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "model": self.model,
            "seed": self.seed,
            "stages": {name: value.as_dict() for name, value in self.stages.items()},
            "bumps": self.bumps,
            "falls": self.falls,
            "map_accuracy": self.map_accuracy.as_dict(),
            "qa": self.qa.as_dict(),
            "visited_rooms": list(self.visited_rooms),
        }


def score_trial(document: dict) -> TrialMetrics:
    """Every doc 06 §5 metric for one trial JSON."""
    if not is_complete(document):
        raise IncompleteTrialError(
            f"trial {document.get('trial_id')!r} has no `final` block "
            f"({document.get('infra_failure', 'incomplete')})"
        )
    for turn in document.get("turns", []):
        # A turn stamped with an unknown stage vanishes from BOTH stages' path
        # integrals. Measured on the golden trial: relabelling the last stage-1
        # turn dropped p from 2.2985 to 1.8174 (-21 %) and raised SPL, silently.
        # A dropped tail is pure SPL inflation, so this raises.
        if turn.get("stage") not in STAGES:
            raise ScoringError(
                f"turn {turn.get('global_turn_idx', '?')} is stamped stage "
                f"{turn.get('stage')!r}, which is not one of {STAGES}; such a "
                "turn is invisible to every per-stage metric (doc 06 §3.2)"
            )
    config = document["config"]
    return TrialMetrics(
        trial_id=document["trial_id"],
        model=config["model"],
        seed=int(config["seed"]),
        stages={stage: stage_metrics(document, stage) for stage in STAGES},
        bumps=bumps(document),
        falls=falls(document),
        map_accuracy=map_accuracy(document),
        qa=score_qa(document),
        visited_rooms=visited_rooms(document),
    )


# ---------------------------------------------------------------------------
# 6. Statistics — mean ± bootstrap 95% CI
# ---------------------------------------------------------------------------


def defined(values: Sequence[float | str]) -> list[float]:
    """Drop the "—" cells. They are excluded from means and CIs (§5.4/§5.7).

    A non-finite value raises rather than passing through: one corrupt trial
    would otherwise turn a model's whole published mean and interval into NaN
    while still reporting ``n_defined = 3`` (measured). ``None`` already raised
    ``TypeError`` here, so this only removes the asymmetry that made NaN mute.
    """
    return [
        _finite(v, "per-trial value")
        for v in values
        if not isinstance(v, str)
    ]


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile with LINEAR interpolation (numpy's default).

    Pinned because doc 06 §6 says "the 2.5th and 97.5th percentiles" without
    naming an estimator, and at N=4 the choice between linear interpolation and
    nearest-rank moves the reported interval. Reproducibility is the stated
    reason for the locked seed, so the estimator is locked too — and recorded in
    ``docs/METRICS.md``.
    """
    if not sorted_values:
        raise ScoringError("percentile of an empty sample")
    position = (len(sorted_values) - 1) * (q / 100.0)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[int(position)]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (
        position - low
    )


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int | None = None,
    seed: int | None = None,
) -> tuple[float, float] | None:
    """doc 06 §6's percentile bootstrap over the per-trial values.

    Resample the values with replacement ``resamples`` times, take each
    resample's mean, and report the 2.5th/97.5th percentiles. Percentile method,
    not BCa — §6: "at N=4 the sophistication of the interval method is noise;
    reproducibility is not."

    ``None`` when fewer than :data:`MIN_CI_VALUES` values are defined ("a
    bootstrap over one value is theatre").

    Deterministic, and deliberately **per call**: each interval is drawn from its
    own ``random.Random(seed)``, so it is a pure function of
    (values, resamples, seed) and can be re-derived in isolation from the
    committed config. (``docs/METRICS.md`` §4 used to say "one
    ``random.Random(seed)`` drives the whole run", which described a threading
    this code never did; the doc was corrected to match, AGENTS.md rule 5.)
    """
    sample = list(values)
    if len(sample) < MIN_CI_VALUES:
        return None
    config = scoring_config()
    resamples = int(config["bootstrap_resamples"]) if resamples is None else resamples
    seed = int(config["bootstrap_seed"]) if seed is None else seed
    rng = random.Random(seed)
    size = len(sample)
    means = sorted(
        sum(rng.choice(sample) for _ in range(size)) / size for _ in range(resamples)
    )
    return (percentile(means, 2.5), percentile(means, 97.5))


@dataclass(frozen=True)
class Estimate:
    """One metric's mean ± CI over N trials, with the "—" cells accounted for."""

    values: tuple[float, ...]
    n_total: int
    mean: float | str
    ci: tuple[float, float] | None

    def as_dict(self) -> dict:
        return {
            "mean": _round_or_na(self.mean, 4),
            "ci95": None if self.ci is None else [round(self.ci[0], 4), round(self.ci[1], 4)],
            "n_defined": len(self.values),
            "n_total": self.n_total,
        }


def estimate(
    values: Sequence[float | str],
    *,
    resamples: int | None = None,
    seed: int | None = None,
) -> Estimate:
    """Mean ± bootstrap CI over the DEFINED values (doc 06 §6)."""
    usable = defined(values)
    return Estimate(
        values=tuple(usable),
        n_total=len(values),
        mean=sum(usable) / len(usable) if usable else NA,
        ci=bootstrap_ci(usable, resamples=resamples, seed=seed),
    )


@dataclass(frozen=True)
class Ratio:
    """A success rate printed with its denominator, per doc 06 §3.2/§6."""

    numerator: int
    denominator: int

    @property
    def value(self) -> float | str:
        return NA if self.denominator == 0 else self.numerator / self.denominator

    def as_dict(self) -> dict:
        return {
            "successes": self.numerator,
            "n": self.denominator,
            "rate": _round_or_na(self.value, 4),
            "printed": f"{self.numerator}/{self.denominator}"
            if self.denominator
            else NA,
        }


#: Every column doc 06 §6/§10 publishes with a mean ± CI, as
#: ``flat key -> per-trial getter``. ONE table, consumed by both
#: :func:`metric_estimates` (what T4.4's figures read) and :func:`summarise`
#: (what the results table reads), so a figure cannot disagree with the number
#: printed beside it. Success rate is in here too: §10's README-table row asks
#: for "SR (both stages), progress, SPL, … each as mean ± 95 % CI", and the
#: bootstrap over the binary per-trial indicator is what produces that interval.
#: §3.2's "no CI when k < 3" needs no special case — :data:`MIN_CI_VALUES`
#: already enforces it for every column here.
def _metric_columns() -> dict[str, Callable[[TrialMetrics], float | str]]:
    columns: dict[str, Callable[[TrialMetrics], float | str]] = {}
    for stage in STAGES:
        columns[f"{stage}.success_rate"] = (
            lambda t, s=stage: 1.0 if t.stages[s].success else 0.0
        )
        columns[f"{stage}.progress"] = lambda t, s=stage: t.stages[s].progress
        columns[f"{stage}.spl"] = lambda t, s=stage: t.stages[s].spl
        columns[f"{stage}.time_s"] = lambda t, s=stage: t.stages[s].time_s
        columns[f"{stage}.turns_used"] = (
            lambda t, s=stage: float(t.stages[s].turns_used)
        )
        columns[f"{stage}.drift_m"] = lambda t, s=stage: t.stages[s].drift_m
        columns[f"{stage}.corrections"] = (
            lambda t, s=stage: float(t.stages[s].corrections)
        )
    columns["bumps"] = lambda t: float(t.bumps)
    columns["falls"] = lambda t: float(t.falls)
    columns["map_precision"] = lambda t: t.map_accuracy.precision
    columns["map_recall"] = lambda t: t.map_accuracy.recall
    columns["edge_accuracy"] = lambda t: t.map_accuracy.edge_accuracy
    columns["qa"] = lambda t: t.qa.score
    return columns


METRIC_COLUMNS: dict[str, Callable[[TrialMetrics], float | str]] = _metric_columns()


def metric_estimates(
    trials: Sequence[TrialMetrics],
    *,
    resamples: int | None = None,
    seed: int | None = None,
) -> dict[str, Estimate]:
    """Flat metric key → :class:`Estimate` for one model's trials.

    The accessor ``charts.py`` needs. ``summarise`` converts every ``Estimate``
    to a plain dict for the results JSON, so a figure fed from ``summarise``
    would have to rebuild the columns itself — exactly the duplication
    ``charts.py``'s docstring forbids ("Nothing here re-derives a metric, so a
    figure can never disagree with the table beside it"). Both now come from
    :data:`METRIC_COLUMNS`.
    """
    return {
        key: estimate(
            [getter(trial) for trial in trials], resamples=resamples, seed=seed
        )
        for key, getter in METRIC_COLUMNS.items()
    }


def summarise(
    model: str,
    trials: Sequence[TrialMetrics],
    *,
    resamples: int | None = None,
    seed: int | None = None,
) -> dict:
    """doc 06 §6's per-model aggregate. The per-trial table always ships with it.

    Two success rates are reported for ``return_home``, exactly as doc 06 §3.2
    requires: ``x/N`` with N = every trial and an unrun stage counted a failure
    (denominator printed literally), plus a conditional ``x/k`` over the stage-1
    successes with ``k`` printed — ``—`` when k = 0, and no CI when k < 3.

    Every ``success_rate`` block carries **both**: the ``x/N`` ratio (the honest
    thing to print) and the bootstrap over the binary per-trial indicator, which
    is the ``± 95 % CI`` doc 06 §10 asks the README table for.
    """
    estimates = metric_estimates(trials, resamples=resamples, seed=seed)

    def stage_estimates(stage: str) -> dict:
        block = {
            "success_rate": {
                **Ratio(
                    sum(1 for t in trials if t.stages[stage].success), len(trials)
                ).as_dict(),
                **estimates[f"{stage}.success_rate"].as_dict(),
            }
        }
        for metric in (
            "progress", "spl", "time_s", "turns_used", "drift_m", "corrections",
        ):
            block[metric] = estimates[f"{stage}.{metric}"].as_dict()
        return block

    stage1_successes = [t for t in trials if t.stages[STAGE_FIND_KITCHEN].success]
    # Criterion v2 can grant a stage-1 success the LIVE gate (pre-registered
    # predicate) denied — such a trial was never offered its return leg, and
    # counting an unrun stage in the conditional denominator would report a
    # return "failure" for a leg the model never got to attempt. The
    # conditional is therefore over stage-1 successes whose return leg RAN;
    # the excluded count is published beside it rather than vanishing.
    offered = [
        t
        for t in stage1_successes
        if t.stages[STAGE_RETURN_HOME].end_reason != REASON_NOT_RUN
    ]
    conditional = Ratio(
        sum(1 for t in offered if t.stages[STAGE_RETURN_HOME].success),
        len(offered),
    )
    conditional_ci = estimate(
        [1.0 if t.stages[STAGE_RETURN_HOME].success else 0.0 for t in offered],
        resamples=resamples,
        seed=seed,
    )
    summary = {
        "model": model,
        "n_trials": len(trials),
        "trials": [trial.trial_id for trial in trials],
        STAGE_FIND_KITCHEN: stage_estimates(STAGE_FIND_KITCHEN),
        STAGE_RETURN_HOME: {
            **stage_estimates(STAGE_RETURN_HOME),
            # doc 06 §3.2: reported BESIDE the x/N rate, never instead of it.
            "success_rate_given_stage1": {
                **conditional.as_dict(),
                **conditional_ci.as_dict(),
            },
            "stage1_successes_never_offered_return": len(stage1_successes)
            - len(offered),
        },
    }
    for metric in ("bumps", "falls", "map_precision", "map_recall", "edge_accuracy", "qa"):
        summary[metric] = estimates[metric].as_dict()
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Score trial JSONs named on the command line and print the metrics.

    PLAN T4.1's smoke step ("score the T3.5 sanity JSON end to end and eyeball
    every number") needs an entry point, and ``scripts/`` is T3.4-owned.
    """
    import sys

    paths = list(argv if argv is not None else sys.argv[1:])
    if not paths:
        print("usage: python -m duck_embody.scoring TRIAL.json [TRIAL.json ...]")
        return 2
    trials = [score_trial(load_trial(path)) for path in paths]
    # allow_nan=False: `json.dumps({'spl': float('nan')})` emits a bare `NaN`,
    # which is not valid JSON for any downstream reader. Nothing should reach
    # here non-finite (see `_finite`); this is the last gate that says so.
    print(
        json.dumps(
            [trial.as_dict() for trial in trials],
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
