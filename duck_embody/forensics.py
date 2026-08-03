"""One reader of the raw trial schema, for audits, tests, and remediation work.

Why this module exists (PLAN TR.0, forensics F-08): `scripts/auto_audit.sh`
reads a top-level ``corrections`` field and ``stages.*.drift_m`` — neither is in
doc 06 §4's schema — so it published "0 corrections" for the two trials holding
the batch's worst loop-closure regressions. Three consumers had each guessed at
the schema separately. This module is the one that reads it, and
``tests/test_forensics.py`` pins its output against the frozen `v5d_r2` batch so
a later remediation task cannot quietly move the baseline it claims to improve.

**Pure by contract.** No Isaac, no kit, no GPU: it imports only the standard
library plus ``duck_embody.scoring`` (itself pure — layout + task predicate +
prompt constants). That is what lets the baseline be re-derived in a unit test
instead of only after a multi-minute Kit boot.

**Read-only by contract.** Nothing here writes to a results directory. The raw
trials are evidence (AGENTS.md rule 7); the analysis lives beside them.

Three schema facts the callers keep getting wrong, all measured against
``results/raw_v5d_r2`` and re-asserted by the tests:

1. ``turns[].execution.calls[]`` holds **motion calls only** — 343 entries
   across the batch for 343 `turn_to_heading`/`move`/`send_velocity` tool calls,
   and nothing for the other 12 tools. It is not positionally aligned with
   ``model_output.tool_calls``.
2. Historical ``model_output.dispatched`` is a prefix count and excludes
   ``declare_done`` or calls after a fall. Remediated logs can reject an
   interior second motion while continuing to later non-motion calls, so their
   positional ``tool_results`` records are the exact per-call source.
3. ``memory_snapshot.corrections`` is **cumulative and stage-local**: its
   ``turn`` is ``turn_idx``, which restarts at the stage boundary, so records
   must be keyed on ``(stage, turn)``. ``opus5_seed104`` has a ``find_kitchen``
   turn 3 and a ``return_home`` turn 3, and only one of them corrected.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from duck_embody import scoring

#: The three tools that move the robot, in the order doc 05 §4 lists them. Kept
#: local rather than imported from ``agent/tools.py`` on purpose: this module
#: describes *logs already written*, so it must keep reading historical batches
#: unchanged even after a remediation task edits the live tool set.
MOTION_TOOLS = ("turn_to_heading", "move", "send_velocity", "turn_and_move")

#: The terminal tool. It ends the stage instead of being dispatched, which is
#: the whole reason ``dispatched`` can be less than ``len(tool_calls)``.
DECLARE_DONE = "declare_done"

#: A visual audit that still says this has not been done, whatever the wording.
#: Matching the literal ``_pending visual pass_`` placeholder would let a
#: reworded placeholder read as a completed audit — the exact class of silent
#: pass that F-08 is about.
_PENDING_AUDIT = re.compile(r"pending\s+visual", re.IGNORECASE)


class ForensicsError(ValueError):
    """A trial document does not match doc 06 §4's schema.

    Always raised with the trial id, the JSON path, and the values that
    disagree — a parser that says only "invalid trial" makes the reader redo the
    investigation this module exists to stop repeating.
    """


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """One entry of ``turns[].model_output.tool_calls``, with its turn context."""

    trial_id: str
    stage: str
    turn_idx: int
    global_turn_idx: int
    call_index: int
    name: str
    args: dict[str, Any]
    dispatched: bool

    @property
    def is_motion(self) -> bool:
        return self.name in MOTION_TOOLS


@dataclass(frozen=True)
class MotionCall:
    """A motion :class:`ToolCall` joined to its ``execution.calls[]`` record."""

    call: ToolCall
    motion_index: int
    execution: dict[str, Any]

    @property
    def true_pose(self) -> tuple[float, float, float]:
        return tuple(self.execution["true_pose"])  # type: ignore[return-value]

    @property
    def true_xy(self) -> tuple[float, float]:
        pose = self.execution["true_pose"]
        return (float(pose[0]), float(pose[1]))


@dataclass(frozen=True)
class CorrectionEvent:
    """A ``correct_position`` call placed at its true physical instant.

    ``true_xy`` is where the robot actually was when the call executed,
    reconstructed per the TR.0 recipe: start from the previous turn's logged
    true pose (or the spawn, on turn 1), then advance through every motion call
    listed *earlier in the same turn* using that call's scoring-only end pose.
    ``true_xy_source`` records which of those three branches was taken, so a
    surprising number can be traced without re-reading the log.

    ``accepted`` is False when the harness refused the write (F-10's blank
    ``place`` with explicit x/y); then ``old_xy``/``new_xy`` are None and no
    error effect is defined, because nothing moved.
    """

    trial_id: str
    stage: str
    turn_idx: int
    global_turn_idx: int
    call_index: int
    motion_calls_before: int
    args: dict[str, Any]
    accepted: bool
    true_xy: tuple[float, float]
    true_xy_source: str
    old_xy: tuple[float, float] | None = None
    new_xy: tuple[float, float] | None = None
    reason: str | None = None

    @property
    def place(self) -> str | None:
        value = self.args.get("place")
        return value if isinstance(value, str) else None


@dataclass(frozen=True)
class CorrectionEffect:
    """What a correction did to true localization error. Positive is harmful."""

    event: CorrectionEvent
    error_before_m: float | None
    error_after_m: float | None

    @property
    def effect_m(self) -> float | None:
        if self.error_before_m is None or self.error_after_m is None:
            return None
        return self.error_after_m - self.error_before_m

    @property
    def worsened(self) -> bool:
        effect = self.effect_m
        return effect is not None and effect > 0

    @property
    def improved(self) -> bool:
        effect = self.effect_m
        return effect is not None and effect <= 0


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def _fail(document: Any, path: str, detail: str) -> ForensicsError:
    trial = "<unknown trial>"
    if isinstance(document, dict):
        trial = str(document.get("trial_id", trial))
    return ForensicsError(f"{trial}: {path}: {detail}")


def _require(document: dict, container: Any, key: str, path: str) -> Any:
    if not isinstance(container, dict) or key not in container:
        raise _fail(document, f"{path}.{key}", "missing (doc 06 §4)")
    return container[key]


def validate_document(document: Any, *, require_final: bool = True) -> dict:
    """Structural check of one trial JSON. Raises :class:`ForensicsError`.

    Deliberately structural, not semantic: it asserts the shapes every function
    here dereferences, so a malformed log fails at load with a pointer instead
    of surfacing as a plausible-looking wrong number three functions later.
    ``require_final=False`` is for synthetic in-test documents that describe
    turns only.
    """
    if not isinstance(document, dict):
        raise ForensicsError(
            f"trial document must be a JSON object, got {type(document).__name__}"
        )
    for key in ("trial_id", "config", "turns"):
        _require(document, document, key, "<root>")
    config = document["config"]
    for key in ("config_hash", "seed", "spawn"):
        _require(document, config, key, "config")
    spawn_xy = _require(document, config["spawn"], "xy", "config.spawn")
    if not (isinstance(spawn_xy, (list, tuple)) and len(spawn_xy) == 2):
        raise _fail(document, "config.spawn.xy", f"expected [x, y], got {spawn_xy!r}")
    turns = document["turns"]
    if not isinstance(turns, list):
        raise _fail(document, "turns", f"expected a list, got {type(turns).__name__}")
    for index, turn in enumerate(turns):
        _validate_turn(document, index, turn)
    if require_final:
        final = _require(document, document, "final", "<root>")
        stages = _require(document, final, "stages", "final")
        if not isinstance(stages, dict) or not stages:
            raise _fail(document, "final.stages", "expected a non-empty object")
    return document


def _validate_turn(document: dict, index: int, turn: Any) -> None:
    path = f"turns[{index}]"
    if not isinstance(turn, dict):
        raise _fail(document, path, f"expected an object, got {type(turn).__name__}")
    for key in ("stage", "turn_idx", "model_output", "execution", "memory_snapshot"):
        _require(document, turn, key, path)
    tool_calls = turn["model_output"].get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise _fail(
            document,
            f"{path}.model_output.tool_calls",
            f"expected a list, got {type(tool_calls).__name__}",
        )
    for call_index, call in enumerate(tool_calls):
        if not isinstance(call, dict) or "name" not in call:
            raise _fail(
                document,
                f"{path}.model_output.tool_calls[{call_index}]",
                f"expected an object with a 'name', got {call!r}",
            )
    calls = turn["execution"].get("calls") or []
    if not isinstance(calls, list):
        raise _fail(
            document,
            f"{path}.execution.calls",
            f"expected a list, got {type(calls).__name__}",
        )
    dispatch_mask = _dispatch_mask(document, index, turn, tool_calls)
    motion_names = [
        call["name"]
        for call, ran in zip(tool_calls, dispatch_mask)
        if ran and call["name"] in MOTION_TOOLS
    ]
    if len(motion_names) != len(calls):
        raise _fail(
            document,
            f"{path}.execution.calls",
            f"{len(calls)} execution record(s) for {len(motion_names)} dispatched "
            f"motion tool call(s) {motion_names} — the correction reconstruction "
            "pairs these positionally and cannot proceed",
        )
    for call_index, (name, record) in enumerate(zip(motion_names, calls)):
        if not isinstance(record, dict):
            raise _fail(
                document,
                f"{path}.execution.calls[{call_index}]",
                f"expected an object, got {type(record).__name__}",
            )
        if record.get("tool") != name:
            raise _fail(
                document,
                f"{path}.execution.calls[{call_index}].tool",
                f"is {record.get('tool')!r} but tool call {call_index} of this "
                f"turn is {name!r} — execution records are out of order",
            )
        pose = record.get("true_pose")
        if not (isinstance(pose, (list, tuple)) and len(pose) == 3):
            raise _fail(
                document,
                f"{path}.execution.calls[{call_index}].true_pose",
                f"expected [x, y, heading_deg], got {pose!r}",
            )
    for record_index, record in enumerate(turn["memory_snapshot"].get("corrections") or []):
        if not isinstance(record, dict):
            raise _fail(
                document,
                f"{path}.memory_snapshot.corrections[{record_index}]",
                f"expected an object, got {type(record).__name__}",
            )
        for key in ("turn", "stage", "old_xy", "new_xy"):
            if key not in record:
                raise _fail(
                    document,
                    f"{path}.memory_snapshot.corrections[{record_index}].{key}",
                    "missing (written by agent/memory.py::correct_position)",
                )


def _dispatch_mask(
    document: dict, index: int, turn: dict, tool_calls: list[dict]
) -> list[bool]:
    """Which listed calls ran, with an auditable historical fallback."""
    dispatched = turn["model_output"].get("dispatched")
    if dispatched is None:
        dispatched = len(tool_calls)
    if not isinstance(dispatched, int) or not 0 <= dispatched <= len(tool_calls):
        raise _fail(
            document,
            f"turns[{index}].model_output.dispatched",
            f"is {dispatched!r}, not a count within {len(tool_calls)} tool call(s)",
        )
    results = turn.get("tool_results")
    if isinstance(results, list):
        if len(results) != len(tool_calls):
            raise _fail(
                document,
                f"turns[{index}].tool_results",
                f"has {len(results)} record(s) for {len(tool_calls)} tool call(s)",
            )
        for call_index, (call, result) in enumerate(zip(tool_calls, results)):
            if not isinstance(result, dict) or result.get("name") != call["name"]:
                raise _fail(
                    document,
                    f"turns[{index}].tool_results[{call_index}]",
                    f"does not positionally match tool call {call['name']!r}",
                )
        mask = scoring.executed_call_mask(turn)
        if sum(mask) != dispatched:
            raise _fail(
                document,
                f"turns[{index}].model_output.dispatched",
                f"is {dispatched}, but positional tool results prove {sum(mask)} "
                "call(s) ran",
            )
        return mask

    # Historical logs predate positional tool results. In those logs the loop
    # stopped at the first declare_done or fall, so dispatched is a prefix
    # count. Both truncation causes are explicit in doc 06 §4.
    undispatched = [call["name"] for call in tool_calls[dispatched:]]
    records = turn["execution"].get("calls") or []
    ended_by_fall = bool(records and records[-1].get("fell"))
    ended_by_declare = bool(undispatched and undispatched[0] == DECLARE_DONE)
    if undispatched and not (ended_by_declare or ended_by_fall):
        raise _fail(
            document,
            f"turns[{index}].model_output.dispatched",
            f"stops before {undispatched} without a {DECLARE_DONE!r} or a falling "
            "last execution record (doc 06 §4)",
        )
    return [call_index < dispatched for call_index in range(len(tool_calls))]


def load_trial(path: str | Path) -> dict:
    """Read and validate one trial JSON."""
    path = Path(path)
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ForensicsError(f"{path}: not valid JSON: {exc}") from exc
    try:
        return validate_document(document)
    except ForensicsError as exc:
        raise ForensicsError(f"{path}: {exc}") from None


def load_batch(directory: str | Path) -> list[dict]:
    """Read every trial JSON in a batch directory, sorted by trial id.

    Skips the ``*_audit.txt`` siblings and anything that is not a trial (the
    directory also holds a ``frames/`` tree).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ForensicsError(f"{directory}: not a directory")
    documents = [load_trial(path) for path in sorted(directory.glob("*.json"))]
    if not documents:
        raise ForensicsError(f"{directory}: contains no *.json trial documents")
    return sorted(documents, key=lambda doc: doc["trial_id"])


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


def iter_tool_calls(document: dict) -> Iterator[ToolCall]:
    """Every tool call in the trial, in the order the model listed them."""
    trial_id = str(document.get("trial_id", "<unknown trial>"))
    for index, turn in enumerate(document.get("turns") or []):
        tool_calls = turn["model_output"].get("tool_calls") or []
        dispatch_mask = _dispatch_mask(document, index, turn, tool_calls)
        for call_index, call in enumerate(tool_calls):
            yield ToolCall(
                trial_id=trial_id,
                stage=turn["stage"],
                turn_idx=int(turn["turn_idx"]),
                global_turn_idx=int(turn.get("global_turn_idx", turn["turn_idx"])),
                call_index=call_index,
                name=str(call["name"]),
                args=dict(call.get("args") or {}),
                dispatched=dispatch_mask[call_index],
            )


def iter_motion_calls(document: dict) -> Iterator[MotionCall]:
    """Every dispatched motion call, joined to its scoring-only execution record.

    The join is the point: ``execution.calls[]`` carries the true pose,
    displacement, contact and stop reason, but not which listed tool call it
    came from. Everything downstream that needs "where was the robot when the
    model did X" goes through here.
    """
    trial_id = str(document.get("trial_id", "<unknown trial>"))
    for index, turn in enumerate(document.get("turns") or []):
        tool_calls = turn["model_output"].get("tool_calls") or []
        dispatch_mask = _dispatch_mask(document, index, turn, tool_calls)
        records = turn["execution"].get("calls") or []
        motion_index = 0
        for call_index, (call, ran) in enumerate(zip(tool_calls, dispatch_mask)):
            if not ran or call["name"] not in MOTION_TOOLS:
                continue
            if motion_index >= len(records):
                raise _fail(
                    document,
                    f"turns[{index}].execution.calls",
                    f"has {len(records)} record(s) but tool call {call_index} "
                    f"({call['name']}) is dispatched motion #{motion_index + 1}",
                )
            yield MotionCall(
                call=ToolCall(
                    trial_id=trial_id,
                    stage=turn["stage"],
                    turn_idx=int(turn["turn_idx"]),
                    global_turn_idx=int(
                        turn.get("global_turn_idx", turn["turn_idx"])
                    ),
                    call_index=call_index,
                    name=str(call["name"]),
                    args=dict(call.get("args") or {}),
                    dispatched=True,
                ),
                motion_index=motion_index,
                execution=records[motion_index],
            )
            motion_index += 1


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------


def correction_events(document: dict) -> list[CorrectionEvent]:
    """Every ``correct_position`` call, placed at its true physical instant.

    The reconstruction the whole loop-closure finding (F-01) rests on. Within
    one turn the model may bundle motion and correction in any order, and the
    log records only an end-of-turn ``true_pose``; using that end pose would
    measure a correction against a position the robot reached *after* it, which
    for ``sonnet5_seed101`` t21 would silently change the batch's worst
    regression.
    """
    trial_id = str(document.get("trial_id", "<unknown trial>"))
    spawn = document["config"]["spawn"]["xy"]
    events: list[CorrectionEvent] = []
    previous_true_xy: tuple[float, float] | None = None
    for index, turn in enumerate(document.get("turns") or []):
        tool_calls = turn["model_output"].get("tool_calls") or []
        dispatch_mask = _dispatch_mask(document, index, turn, tool_calls)
        records = turn["execution"].get("calls") or []
        written = _corrections_written(turn)
        write_index = 0
        motion_index = 0
        for call_index, (call, ran) in enumerate(zip(tool_calls, dispatch_mask)):
            if not ran:
                continue
            name = call["name"]
            if name in MOTION_TOOLS:
                motion_index += 1
                continue
            if name != "correct_position":
                continue
            if motion_index == 0:
                if previous_true_xy is None:
                    true_xy = (float(spawn[0]), float(spawn[1]))
                    source = "spawn"
                else:
                    true_xy = previous_true_xy
                    source = "prior_turn"
            else:
                if motion_index > len(records):
                    raise _fail(
                        document,
                        f"turns[{index}].execution.calls",
                        f"has {len(records)} record(s) but tool call {call_index} "
                        f"follows {motion_index} dispatched motion call(s) — the "
                        "correction cannot be placed",
                    )
                pose = records[motion_index - 1]["true_pose"]
                true_xy = (float(pose[0]), float(pose[1]))
                source = f"motion_call[{motion_index - 1}]"
            record = written[write_index] if write_index < len(written) else None
            if record is not None:
                write_index += 1
            events.append(
                CorrectionEvent(
                    trial_id=trial_id,
                    stage=turn["stage"],
                    turn_idx=int(turn["turn_idx"]),
                    global_turn_idx=int(
                        turn.get("global_turn_idx", turn["turn_idx"])
                    ),
                    call_index=call_index,
                    motion_calls_before=motion_index,
                    args=dict(call.get("args") or {}),
                    accepted=record is not None,
                    true_xy=true_xy,
                    true_xy_source=source,
                    old_xy=_xy(record["old_xy"]) if record else None,
                    new_xy=_xy(record["new_xy"]) if record else None,
                    reason=record.get("reason") if record else None,
                )
            )
        if write_index < len(written):
            raise _fail(
                document,
                f"turns[{index}].memory_snapshot.corrections",
                f"{len(written)} record(s) written this turn but only "
                f"{write_index} dispatched correct_position call(s) to match",
            )
        previous_true_xy = _turn_true_xy(turn, records, previous_true_xy)
    return events


def _corrections_written(turn: dict) -> list[dict]:
    """The correction records this turn added — keyed on (stage, turn_idx).

    ``memory_snapshot.corrections`` is cumulative, so filtering is unavoidable;
    filtering on ``turn`` alone would mix a stage-1 and a stage-2 turn 3.
    """
    stage = turn["stage"]
    turn_idx = int(turn["turn_idx"])
    return [
        record
        for record in (turn["memory_snapshot"].get("corrections") or [])
        if int(record["turn"]) == turn_idx and record["stage"] == stage
    ]


def _turn_true_xy(
    turn: dict,
    records: list[dict],
    fallback: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Where the robot was at the end of this turn.

    Prefers the turn-level ``true_pose``; falls back to the last motion record
    (perception-only turns in some historical logs omit the turn-level pose) and
    finally to the previous value, since a turn with no motion cannot have moved.
    """
    pose = turn.get("true_pose")
    if isinstance(pose, dict) and "x" in pose and "y" in pose:
        return (float(pose["x"]), float(pose["y"]))
    if records:
        last = records[-1]["true_pose"]
        return (float(last[0]), float(last[1]))
    return fallback


def _xy(value: Iterable[float]) -> tuple[float, float]:
    pair = list(value)
    return (float(pair[0]), float(pair[1]))


def correction_error_effects(document: dict) -> list[CorrectionEffect]:
    """Each correction's change in true localization error. Positive is harmful.

    Rejected calls are kept, with ``None`` errors: they are part of the ledger
    (F-10) and dropping them would make "16 calls, 15 accepted" unverifiable
    from this function's output alone.
    """
    effects: list[CorrectionEffect] = []
    for event in correction_events(document):
        if not event.accepted:
            effects.append(CorrectionEffect(event, None, None))
            continue
        effects.append(
            CorrectionEffect(
                event=event,
                error_before_m=math.dist(event.old_xy, event.true_xy),
                error_after_m=math.dist(event.new_xy, event.true_xy),
            )
        )
    return effects


def correction_summary(documents: Iterable[dict]) -> dict[str, Any]:
    """Batch totals for the loop-closure finding (forensics F-01)."""
    calls = accepted = rejected = worsened = improved = 0
    before = after = 0.0
    ledger: list[dict[str, Any]] = []
    for document in documents:
        for effect in correction_error_effects(document):
            calls += 1
            event = effect.event
            if not event.accepted:
                rejected += 1
            else:
                accepted += 1
                before += effect.error_before_m
                after += effect.error_after_m
                if effect.worsened:
                    worsened += 1
                else:
                    improved += 1
            ledger.append(
                {
                    "trial_id": event.trial_id,
                    "stage": event.stage,
                    "turn_idx": event.turn_idx,
                    "global_turn_idx": event.global_turn_idx,
                    "place": event.place,
                    "accepted": event.accepted,
                    "motion_calls_before": event.motion_calls_before,
                    "true_xy": list(event.true_xy),
                    "true_xy_source": event.true_xy_source,
                    "old_xy": list(event.old_xy) if event.old_xy else None,
                    "new_xy": list(event.new_xy) if event.new_xy else None,
                    "error_before_m": effect.error_before_m,
                    "error_after_m": effect.error_after_m,
                    "effect_m": effect.effect_m,
                    "reason": event.reason,
                }
            )
    return {
        "calls": calls,
        "accepted": accepted,
        "rejected": rejected,
        "worsened": worsened,
        "improved": improved,
        "error_before_sum_m": before,
        "error_after_sum_m": after,
        "net_added_error_m": after - before,
        "ledger": ledger,
    }


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def published_and_live_outcomes(document: dict) -> dict[str, Any]:
    """The F-02 split: what the live gate decided vs what the scorer publishes.

    Both verdicts come from this one document. The live values are what the loop
    wrote at the time (``final.stages[*]``); the published ones are recomputed
    through ``duck_embody.scoring``, which needs only the log plus the committed
    layout — so this cannot drift from a stale scores file.
    """
    trial_id = str(document.get("trial_id", "<unknown trial>"))
    final = document.get("final") or {}
    stages = final.get("stages") or {}
    out: dict[str, Any] = {"trial_id": trial_id, "stages": {}}
    for stage, result in stages.items():
        score = result.get("score") or {}
        published = scoring.stage_success(document, stage)
        preregistered = scoring.stage_success_preregistered(document, stage)
        out["stages"][stage] = {
            "ran": result.get("end_reason") != "not_run",
            "live_outcome": result.get("outcome"),
            "live_end_reason": result.get("end_reason"),
            "live_success": bool(result.get("success")),
            "live_distance_m": score.get("distance_m"),
            "live_radius_m": score.get("radius_m"),
            "published_success_v2": published,
            "published_success_preregistered": preregistered,
            "criterion_split": published != bool(result.get("success")),
        }
    stage1 = out["stages"].get(scoring.STAGE_FIND_KITCHEN, {})
    stage2 = out["stages"].get(scoring.STAGE_RETURN_HOME, {})
    out["stage1_success_never_offered_return"] = bool(
        stage1.get("published_success_v2") and not stage2.get("ran", False)
    )
    return out


# ---------------------------------------------------------------------------
# Batch-level checks
# ---------------------------------------------------------------------------


def batch_integrity(
    documents: Iterable[dict], manifest: dict | None = None
) -> dict[str, Any]:
    """Is this a single-freeze, complete, schema-valid batch?

    ``manifest`` is a ``results/freeze.json`` document (schema
    ``duck-embody-freeze-v1``). It is optional because a batch directory can
    outlive the manifest that produced it — F-06's whole point — but when it is
    absent the hash agreement simply cannot be asserted, and this says so rather
    than reporting a pass.
    """
    documents = list(documents)
    trial_ids = sorted(str(doc.get("trial_id", "<unknown>")) for doc in documents)
    hashes = sorted({doc["config"]["config_hash"] for doc in documents})
    commits = sorted({doc["config"].get("freeze_commit") for doc in documents})
    complete = [doc["trial_id"] for doc in documents if _is_complete(doc)]
    incomplete = sorted(set(trial_ids) - set(complete))
    report: dict[str, Any] = {
        "trials": len(documents),
        "trial_ids": trial_ids,
        "complete_trials": len(complete),
        "incomplete_trial_ids": incomplete,
        "total_turns": sum(len(doc.get("turns") or []) for doc in documents),
        "config_hashes": hashes,
        "single_config_hash": len(hashes) == 1,
        "freeze_commits": commits,
        "manifest": None,
    }
    if manifest is None:
        return report
    expected_hash = manifest.get("config_hash")
    matrix = manifest.get("matrix") or {}
    models = list(matrix.get("models") or [])
    seeds = list(matrix.get("seeds") or [])
    expected_cells = [f"{model}_seed{seed}" for model in models for seed in seeds]
    report["manifest"] = {
        "config_hash": expected_hash,
        "freeze_commit": manifest.get("freeze_commit"),
        "config_hash_matches": hashes == [expected_hash] if expected_hash else False,
        "freeze_commit_matches": commits == [manifest.get("freeze_commit")],
        "expected_cells": len(expected_cells),
        "missing_cells": sorted(set(expected_cells) - set(trial_ids)),
        "unexpected_trials": sorted(set(trial_ids) - set(expected_cells)),
        "frozen_files": len(manifest.get("files") or {}),
    }
    return report


def _is_complete(document: dict) -> bool:
    final = document.get("final")
    return isinstance(final, dict) and bool(final.get("stages"))


def visual_audit_status(batch_dir: str | Path) -> dict[str, Any]:
    """Which rule-11 visual audits are still unwritten.

    Accepts either the audit directory itself or the batch's raw directory —
    ``results/raw_v5d_r2`` resolves to its ``results/audits_v5d_r2`` sibling —
    because every caller already has the raw path in hand. A pending verdict is
    an audit failure (F-08), not a warning, so the caller gets the file names.
    """
    directory = _resolve_audit_dir(Path(batch_dir))
    pending: list[str] = []
    completed: list[str] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(errors="replace")
        (pending if _PENDING_AUDIT.search(text) else completed).append(path.name)
    return {
        "audit_dir": str(directory),
        "total": len(pending) + len(completed),
        "pending": pending,
        "completed": completed,
        "complete": not pending and bool(completed),
    }


def _resolve_audit_dir(batch_dir: Path) -> Path:
    if not batch_dir.exists():
        raise ForensicsError(f"{batch_dir}: does not exist")
    if any(batch_dir.glob("*.md")):
        return batch_dir
    if batch_dir.name.startswith("raw_"):
        sibling = batch_dir.parent / f"audits_{batch_dir.name[len('raw_'):]}"
        if sibling.is_dir():
            return sibling
        raise ForensicsError(
            f"{batch_dir}: holds no *.md audits and its sibling {sibling} does "
            "not exist — pass the audit directory explicitly"
        )
    raise ForensicsError(
        f"{batch_dir}: holds no *.md audits and its name does not follow "
        "'raw_<batch>', so the audit directory cannot be derived"
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def trial_report(document: dict) -> dict[str, Any]:
    """Everything this module knows about one trial, as plain JSON."""
    tool_counts: dict[str, int] = {}
    for call in iter_tool_calls(document):
        tool_counts[call.name] = tool_counts.get(call.name, 0) + 1
    motion = list(iter_motion_calls(document))
    multi_motion = [
        int(turn["turn_idx"])
        for turn in document.get("turns") or []
        if len(turn["execution"].get("calls") or []) > 1
    ]
    return {
        "trial_id": document["trial_id"],
        "config_hash": document["config"]["config_hash"],
        "freeze_commit": document["config"].get("freeze_commit"),
        "model": document["config"].get("model"),
        "seed": document["config"].get("seed"),
        "turns": len(document.get("turns") or []),
        "complete": _is_complete(document),
        "tool_call_counts": dict(sorted(tool_counts.items())),
        "motion_calls": len(motion),
        "multi_motion_turns": len(multi_motion),
        "multi_motion_turn_idx": multi_motion,
        "bumped_motion_calls": sum(
            1 for call in motion if call.execution.get("bumped")
        ),
        "counted_bumps": sum(
            1 for call in motion if call.execution.get("counted_as_bump")
        ),
        "falls": sum(1 for call in motion if call.execution.get("fell")),
        "corrections": correction_summary([document]),
        "outcomes": published_and_live_outcomes(document),
    }


def batch_report(
    documents: Iterable[dict],
    *,
    manifest: dict | None = None,
    batch_dir: str | Path | None = None,
) -> dict[str, Any]:
    """The forensic baseline: one dict the tests and the analyzer both pin."""
    documents = list(documents)
    trials = [trial_report(doc) for doc in documents]
    tool_counts: dict[str, int] = {}
    for trial in trials:
        for name, count in trial["tool_call_counts"].items():
            tool_counts[name] = tool_counts.get(name, 0) + count
    report: dict[str, Any] = {
        "schema": "duck-embody-forensics-v1",
        "integrity": batch_integrity(documents, manifest),
        "tool_call_counts": dict(sorted(tool_counts.items())),
        "motion_calls": sum(trial["motion_calls"] for trial in trials),
        "multi_motion_turns": sum(trial["multi_motion_turns"] for trial in trials),
        "falls": sum(trial["falls"] for trial in trials),
        "counted_bumps": sum(trial["counted_bumps"] for trial in trials),
        "bumped_motion_calls": sum(trial["bumped_motion_calls"] for trial in trials),
        "corrections": correction_summary(documents),
        "outcomes": [trial["outcomes"] for trial in trials],
        "stage1_success_never_offered_return": [
            trial["trial_id"]
            for trial in trials
            if trial["outcomes"]["stage1_success_never_offered_return"]
        ],
        "trials": trials,
    }
    if batch_dir is not None:
        report["visual_audits"] = visual_audit_status(batch_dir)
    return report


def as_json(value: Any) -> Any:
    """Dataclass-aware conversion for the analyzer's output."""
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(item) for item in value]
    return value
