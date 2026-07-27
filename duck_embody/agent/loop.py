"""Tool-use episode loop: the paused-sim protocol, the context policy, the
stage machine, the caps, the post-episode layout-QA exchange, and doc 06 §4's
per-trial log.

Doc 05 §3.1's pseudocode is **normative** and this module follows its four
numbered phases literally — assemble while the sim is paused, call the model and
count the turn, dispatch in listed order, check caps last. Where the shipped
code deviates from the pseudocode the deviation is recorded in doc 05 in the
same commit (AGENTS.md rule 5); the five deviations are listed in §3.1's
implementation note and each is repeated at its site below.

Four things this module owns that nothing else can:

**1. The turn counter is the cap.** ``state.turns += 1`` fires immediately after
``provider.send`` returns and before any dispatch, so *every* model turn counts —
malformed calls, refusals and prose-only turns included. That is doc 05 §8's
agency line made mechanical: backoff retries live inside ``provider.send`` and
therefore never increment it, while a turn the model wasted always does. Nothing
in ``tools.py`` touches either counter (T3.2 left both to the loop), so a loop
that forgot one would run an unbounded, unbudgeted episode with a budget line
frozen at ``turns 0/40``.

**2. The stage ends on ``status.fell``** (doc 05 §4.1's residual, assigned here
explicitly). ``dispatch`` already refuses *motion* after a fall, but perception
and memory tools still answer — so a ``get_observation`` listed after the
falling command would render its frame from the spawn point Isaac teleported the
robot to, and that frame would be logged as the trial's final observation. Only
the loop can stop dispatching the rest of the turn.

**3. The context policy, including the one rule that is easy to get subtly
wrong.** Doc 05 §5.2: system prompt + first turn + last K=10 turns, memory block
regenerated fresh into every request. The first turn is pinned but the K window
is computed over ``transcript[1:]``, so that while the transcript is short the
first turn is emitted **once**, not twice — see :func:`context_messages`. It
costs a duplicated spawn image per turn to get wrong, and nothing fails.

**4. The scoring channel reaches the JSON and only the JSON.**
``ToolOutcome.execution`` carries the 5 Hz ``pose_trace`` doc 06 §5.3 pins SPL
to and the guarded post-fall ``true_pose``; :meth:`ToolOutcome.to_block` cannot
carry it, and this module must never put it anywhere near a request. The trial
log is built from ``execution`` + the model-facing payloads; the request is built
from ``SYSTEM_PROMPT`` + the transcript + the memory block, and those two
assemblies share no code.

**What is NOT here, deliberately.** No ``try``/``except`` around
``provider.send`` or ``dispatch``. Doc 05 §8 routes a render error or a physics
NaN to the infra path — the trial reruns whole — and catching them would launder
a broken GPU into a model failure. Model-attributable faults never reach an
exception in the first place: ``dispatch`` returns a structured error for
everything a model can emit, and this module appends it like any other result.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from duck_embody.agent.memory import (
    STAGE_RETURN_HOME,
    Counters,
    Memory,
    PositionIntegrator,
)
from duck_embody.agent.prompts import (
    DERAILMENT_NUDGE,
    LAYOUT_QA_QUESTIONS,
    SYSTEM_PROMPT,
    render_memory_block,
    render_qa_prompt,
)
from duck_embody.agent.providers.base import (
    AssistantMessage,
    Block,
    ImageBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    Usage,
    UserMessage,
)
from duck_embody.agent.tools import (
    DECLARE_DONE,
    TOOL_SCHEMAS,
    ToolContext,
    ToolOutcome,
    dispatch,
    not_executed,
    observed_compass_deg,
    stage_end_result,
)
from duck_embody.tasks.find_kitchen import (
    REASON_DECLARE_DONE,
    REASON_FALL,
    REASON_MOTION_CAP,
    REASON_NOT_RUN,
    REASON_TURN_CAP,
    StageScore,
    StageSpec,
    outcome_for,
    runs_return_home,
    score_stage,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The context window, in transcript entries (doc 05 §5.2, §2, §12).
#:
#: FROZEN AT 10, not configurable. Doc 05 §7.1 lists "context K" as a
#: ``configs/models/*.yaml`` key, but §5.2, §2, §12 and — decisively — the frozen
#: ``SYSTEM_PROMPT`` all hard-state 10, and the system prompt is byte-identical
#: for all three models (AGENTS.md rule 4). Reading K from per-model YAML would
#: let a typo in one file contradict the sentence every model is *told*, with
#: nothing failing; ``tests/test_loop.py`` asserts this constant still matches
#: the prompt's own promise.
K_CONTEXT_TURNS = 10

#: The system parameter for the post-episode QA exchange (doc 06 §5.9).
#:
#: EMPTY, not ``SYSTEM_PROMPT``. §5.9 calls the exchange "fresh": the model sees
#: "only its own final map/memory block — no new camera frames, no sim access".
#: ``LAYOUT_QA_PREAMBLE`` is written as the standalone framing for exactly this
#: call and ``render_qa_prompt`` already embeds it, so re-sending the driving
#: prompt would tell a model with no robot that it is still driving one — and
#: would put 3,000 tokens of navigation doctrine in front of a map-reading probe.
#: ``tests/test_memory.py`` already treats the preamble as a *peer* of
#: ``SYSTEM_PROMPT`` for leak checks, which is the same reading.
QA_SYSTEM_PROMPT = ""

#: Tools for the QA exchange: none. The preamble tells the model "you have no
#: camera, no robot, and no tools now", and a tool-equipped QA call invites a
#: ``get_observation`` instead of an answer — which would score 0 on five
#: questions for a reason that is the harness's fault, not the model's.
QA_TOOLS: list[dict] = []

#: Files inside doc 06 §2's fairness contract, hashed into ``config.config_hash``
#: so a trial records exactly what it ran under.
#:
#: PROVISIONAL: T4.2 owns §7's freeze guard and the authoritative manifest
#: (``results/freeze.json``) and may move this. It is here because a trial JSON
#: written before that exists still needs the field doc 06 §4 requires, and a
#: placeholder string would be worse than a real hash of a possibly-incomplete
#: list — the hash at least changes when the prompt or the tool schema does.
#:
#: **The list must cover the file that ENFORCES each frozen item, not the file
#: that documents it** (doc 06 §2's file list, extended in the same commit —
#: AGENTS.md rule 5). The four additions T3.4's review pass found missing:
#:
#: * ``agent/memory.py`` holds ``TURN_CAP``/``POLICY_SECONDS_CAP`` — §2's "40
#:   model turns per stage; 240 policy-seconds" — plus ``PLAN_MAX_CHARS`` and
#:   the exit-direction quantum. ``configs/benchmark.yaml`` mirrors the caps but
#:   is only checked by a unit test; ``Counters`` is what the loop compares
#:   against and what the model reads in its budget line.
#: * ``sim/policy_wrapper.py`` holds ``MOVE_MAX_DISTANCE_M``/``MOVE_SPEED_MPS``
#:   and the command-hull clamp — the numbers §2's tool-schema row states to the
#:   model ("Max 1.5 m per call", "vx in (-0.148, 0.222)").
#: * ``agent/loop.py`` holds ``K_CONTEXT_TURNS = 10``, which the frozen
#:   ``SYSTEM_PROMPT`` promises the model verbatim, plus the context policy and
#:   the QA exchange.
#: * ``agent/providers/*.py`` shape every request; a change there is a change to
#:   what the models were asked, which is the whole fairness contract.
#:
#: Without them an uncommitted mid-batch edit to any of those files left
#: ``config.freeze_commit`` and ``config.config_hash`` byte-identical across
#: trials, so doc 06 §7's guard would see nothing and the published table would
#: average two incomparable halves. ``tests/test_loop.py`` now asserts that
#: editing EVERY listed file moves the hash.
FROZEN_FILES: tuple[str, ...] = (
    "duck_embody/agent/prompts.py",
    "duck_embody/agent/tools.py",
    "duck_embody/agent/memory.py",
    "duck_embody/agent/loop.py",
    "duck_embody/agent/providers/base.py",
    "duck_embody/agent/providers/anthropic.py",
    "duck_embody/agent/providers/openai.py",
    "duck_embody/sim/policy_wrapper.py",
    "duck_embody/env/camera.py",
    "duck_embody/env/apartment_layout.py",
    "duck_embody/tasks/find_kitchen.py",
    "configs/benchmark.yaml",
    "configs/models/fable5.yaml",
    "configs/models/opus5.yaml",
    "configs/models/gpt56sol.yaml",
)

#: Anything shaped like a provider key, for the log scrubber below. Both vendors
#: currently issue ``sk-``-prefixed keys; the env-var values are substituted
#: first, so this is only the belt-and-braces half.
_SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")

#: Env vars whose VALUES must never reach a tracked file (AGENTS.md rule 6).
SECRET_ENV_VARS: tuple[str, ...] = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")


def redact_secrets(text: str) -> str:
    """Strip anything key-shaped out of text bound for the trial JSON.

    AGENTS.md rule 6 forbids a key in a tracked file and rule 7 commits
    ``results/`` to a public repo — so the infra-failure traceback, which is
    third-party exception text the harness does not author, cannot be trusted to
    be key-free. No current anthropic/openai/httpx exception embeds the
    ``x-api-key`` header in its ``str()``, but nothing enforces that, and a
    wrapped ``httpx.Request`` repr in a future SDK would put a live key into a
    published file. This runs before the traceback is stored OR printed.

    The env values are replaced by name only — the key itself is never echoed,
    logged, or included in the marker.
    """
    for var in SECRET_ENV_VARS:
        value = os.environ.get(var)
        if value and len(value) >= 8:
            text = text.replace(value, f"<redacted:{var}>")
    return _SECRET_RE.sub("<redacted>", text)


# ---------------------------------------------------------------------------
# Transcript entries and the context policy (doc 05 §3.1, §5.2)
# ---------------------------------------------------------------------------


def _without_images(block: ToolResultBlock) -> ToolResultBlock:
    """The same tool_result with its frames stripped (doc 05 §2, §5.2, §12).

    "Images older than the K=10-turn window are dropped; the map text is the
    durable record." The rule is **uniform** and applies to the pinned first turn
    too (§12: "the kept first turn loses its image too … Currently: dropped, per
    the uniform rule"), which is what the frozen ``SYSTEM_PROMPT`` promises the
    model in as many words — "only the first turn and the last 10 turns are kept,
    and their images are dropped as they age out". Pinning the spawn frame
    permanently would help ``return_home`` re-recognition and would make that
    sentence false for every model in the batch, so it stays dropped.

    The JSON status text survives, so an aged first turn still says *what* the
    model observed; only the pixels go.
    """
    return ToolResultBlock(
        tool_use_id=block.tool_use_id,
        tool_name=block.tool_name,
        text=block.text,
        images=[],
        is_error=block.is_error,
    )


@dataclass
class TranscriptEntry:
    """One exchange: the model's turn, echoed native, plus its answer message.

    ``native`` is ``AssistantTurn.raw`` and is replayed **unchanged** — Anthropic
    requires thinking blocks to come back byte-identical and the OpenAI adapter
    has the same requirement for its ``reasoning`` items (doc 05 §7.2/§7.3).

    Exactly one of ``results`` / ``note`` is populated in practice. ``results``
    holds one ``tool_result`` per ``tool_use`` block in the turn — doc 05 §7.2:
    an unanswered one is an API error, i.e. an infra rerun of a trial the model
    actually finished. ``note`` holds ``DERAILMENT_NUDGE`` for a turn that made
    no tool call at all, which §3.1's pseudocode has no branch for; see
    :meth:`EpisodeRunner._run_turn`.
    """

    native: Any
    results: list[ToolResultBlock] = field(default_factory=list)
    note: str | None = None

    def messages(self, *, keep_images: bool) -> list[Message]:
        blocks: list[Block] = [
            block if keep_images else _without_images(block) for block in self.results
        ]
        if self.note:
            blocks.append(TextBlock(self.note))
        if not blocks:
            # Unreachable by construction (a turn either made tool calls, so
            # `results` is non-empty, or it did not, so `note` is set). Raising
            # rather than emitting an empty user message, which Anthropic
            # rejects: a 400 here is doc 05 §8's infra path, and it would rerun a
            # whole trial for a harness bug that is trivially detectable.
            raise ValueError(
                "transcript entry has neither tool results nor a note — an empty "
                "user message is an API error (doc 05 §7.2)"
            )
        return [AssistantMessage(native=self.native), UserMessage(blocks)]


def context_messages(
    transcript: list[TranscriptEntry], k: int = K_CONTEXT_TURNS
) -> list[Message]:
    """Doc 05 §3.1's ``[first_turn] + last_k_turns(transcript[1:], K=10)``.

    Two rules, both of which fail silently if fumbled:

    * **The first turn is never emitted twice.** §3.1 slices ``transcript[1:]``
      for exactly this reason — "no duplicate first turn while it is still inside
      the K window (early turns would otherwise pay the spawn image's token cost
      twice)". The effective window is therefore ``1 + min(K, len - 1)`` entries,
      never ``K + 1`` with one of them repeated. PLAN T3.4 names this as a
      required unit test.
    * **Images age out by position in the whole transcript**, not by whether the
      entry survived the window. An entry keeps its frames iff it is among the
      last ``k`` entries, so the pinned first turn keeps its spawn image while
      the transcript is short and loses it the moment it ages past K — the
      uniform rule of doc 05 §12 and the sentence the frozen system prompt
      promises the model.

    Returns provider-neutral messages. The memory block is NOT here: it is
    regenerated per request and appended by :func:`build_request`, because doc 05
    §5.2 exempts it from truncation and storing it in the transcript would both
    duplicate it and freeze a stale budget line into the context.
    """
    if not transcript:
        return []
    n = len(transcript)
    # `max(1, ...)`: the tail is taken over transcript[1:], so it can never
    # reach back to index 0 and re-emit the pinned first turn.
    tail_start = max(1, n - k) if k > 0 else n
    image_floor = n - k
    messages: list[Message] = []
    for index in [0, *range(tail_start, n)]:
        messages += transcript[index].messages(keep_images=index >= image_floor)
    return messages


def build_request(
    memory_block: str,
    transcript: list[TranscriptEntry],
    k: int = K_CONTEXT_TURNS,
) -> list[Message]:
    """The messages for one request: context window, then the memory block.

    The block rides in a **trailing, non-persisted user message**, never
    concatenated into the ``system`` string. Two reasons, and the second is
    money: (a) doc 05 §5.2 regenerates it fresh every turn, so a copy in the
    transcript would show the model a stale budget line and a stale position
    right beside the live one; (b) the system prompt plus the tool schema is the
    stable prefix doc 06 §8 counts on for prompt caching, and a block that
    changes every turn would invalidate that prefix on every single request.

    Consecutive user messages (the tool results, then this) are merged
    server-side, which is the shape doc 05 §3.3 explicitly sanctions — the thing
    that must not happen is splitting *tool results* across messages.
    """
    return [*context_messages(transcript, k), UserMessage([TextBlock(memory_block)])]


# ---------------------------------------------------------------------------
# doc 06 §4's per-trial log
# ---------------------------------------------------------------------------


def _json_safe(value):
    """Make model-authored values JSON-serialisable without ``allow_nan``.

    ``json.loads`` accepts the bare ``NaN``/``Infinity`` literals, so a model
    *can* put a non-finite float into a tool call's arguments. ``dispatch``
    rejects it as ``invalid_args`` (``memory.number_arg``), but doc 06 §4 still
    logs ``model_output.tool_calls`` verbatim — and ``json.dump(..., allow_nan=
    False)`` would then raise while writing the log, crashing the harness on
    doc 05 §8's *infra* path for a fault whose agency is entirely the model's.
    That is §8's line crossed backwards: the trial would be rerun whole and the
    malformed call would never be scored.

    Non-finite floats therefore become their repr as a string. The value is
    preserved and visibly not-a-number; nothing downstream can mistake
    ``"nan"`` for a quantity.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def config_hash(files: Iterable[str] = FROZEN_FILES, root: Path = REPO_ROOT) -> str:
    """sha256 over the frozen files, path-prefixed so a rename is a change."""
    digest = hashlib.sha256()
    for relative in files:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def frozen_files_dirty(
    files: Iterable[str] = FROZEN_FILES, root: Path = REPO_ROOT
) -> bool:
    """True iff any frozen file has uncommitted changes. Never raises."""
    out = _git(root, "status", "--porcelain", "--", *files)
    return bool(out and out.strip())


def freeze_commit(root: Path = REPO_ROOT, files: Iterable[str] = FROZEN_FILES) -> str:
    """Current git sha (``-dirty`` if a frozen file is uncommitted), or ``"unknown"``.

    Never raises: a trial that ran is worth logging even from a tree with no git,
    and T4.2's runner is where a missing/dirty commit becomes a hard refusal
    (doc 06 §7).

    DEVIATION from doc 06 §4's ``"<git sha>"``, recorded in §2 in the same
    commit: a bare ``rev-parse HEAD`` is returned regardless of working-tree
    state, and AGENTS.md §5 records that this very tree "carries large
    uncommitted work". Without the marker, an uncommitted edit to a frozen file
    mid-batch leaves ``freeze_commit`` byte-identical across trials that ran
    different code — the exact silent incomparability §7's guard exists to
    prevent, and the guard cannot see it either.
    """
    out = _git(root, "rev-parse", "HEAD")
    if out is None or not out.strip():
        return "unknown"
    sha = out.strip()
    return f"{sha}-dirty" if frozen_files_dirty(files, root) else sha


class TrialLog:
    """doc 06 §4's per-trial JSON, written incrementally.

    "Written incrementally (turn-by-turn flush, so a crash loses at most the
    in-flight turn)." The whole document is rewritten through a temp file and
    ``os.replace`` after every turn rather than appended to: a trial is at most a
    few MB, and an atomic replace means a crash mid-write cannot leave a
    half-written file that ``json.load`` chokes on — which for a resumable runner
    (doc 06 §7) would be indistinguishable from a corrupt result.

    ``final`` is written **only** by :meth:`finish`. That is the resume key: doc
    06 §9.1 requires an incomplete trial JSON (no ``final``) to be rejected by
    the resume check, so an infra-failed trial must never acquire one.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        trial_id: str,
        model_id: str,
        model_name: str,
        seed: int,
        spawn_xy: tuple[float, float],
        spawn_heading_deg: float,
        frames_root: Path | str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Frames live in a per-trial subdirectory so 12 trials can share one
        # results directory; doc 06 §4's example path is `frames/t007_0.png`,
        # and the recorded paths stay relative to this JSON's directory so the
        # results tree can be moved or archived whole.
        self.frames_rel = f"frames/{trial_id}"
        root = Path(frames_root) if frames_root is not None else self.path.parent
        self.frames_dir = root / "frames" / trial_id
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.document: dict = {
            "trial_id": trial_id,
            "config": {
                "freeze_commit": freeze_commit(),
                "config_hash": config_hash(),
                "model": model_id,
                "model_config": model_name,
                "seed": seed,
                "spawn": {
                    "xy": [spawn_xy[0], spawn_xy[1]],
                    "heading_deg": spawn_heading_deg,
                },
            },
            "turns": [],
            "video_path": None,
        }
        self.flush()

    # -- writes -------------------------------------------------------------

    def save_frames(self, global_turn: int, images: list[ImageBlock]) -> list[str]:
        """Write this turn's frames and return doc 06 §4's ``obs.frame_paths``.

        The bytes written are **base64-decoded from the exact block that was
        sent to the model**, never re-captured: re-rendering would produce a
        different frame (the robot has moved on, the renderer is stochastic in
        its streaming) and the saved image would then illustrate a decision it
        did not inform.

        DEVIATION from doc 06 §4, recorded there in the same commit: the
        extension is ``.jpg``, not the example's ``.png``. The pipeline is JPEG
        end to end (``camera.encode_jpeg``, quality 85), and writing ``.png``
        would require re-encoding — i.e. a different file from the one the model
        saw, under a name that claims otherwise.
        """
        paths: list[str] = []
        for index, image in enumerate(images):
            name = f"t{global_turn:03d}_{index}.jpg"
            (self.frames_dir / name).write_bytes(base64.b64decode(image.data_b64))
            paths.append(f"{self.frames_rel}/{name}")
        return paths

    def append_turn(self, record: dict) -> None:
        self.document["turns"].append(record)
        self.flush()

    def set_video(self, video_path: str | None) -> None:
        self.document["video_path"] = video_path
        self.flush()

    def note_infra_failure(self, detail: str) -> None:
        """Record a doc 05 §8 infra fault WITHOUT writing ``final``.

        The trial reruns whole (§8, doc 06 §7), so it must stay incomplete: an
        infra-failed JSON that carried a ``final`` block would be skipped by the
        resume check and silently reported as a result.

        The detail is a third-party traceback and this file is committed to a
        public repo, so it goes through :func:`redact_secrets` first — see that
        function for why the harness cannot assume an SDK exception is key-free.
        """
        self.document["infra_failure"] = redact_secrets(detail)
        self.flush()

    def finish(self, final: dict) -> dict:
        self.document["final"] = final
        self.flush()
        return self.document

    def flush(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.document, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# Turn-level records
# ---------------------------------------------------------------------------


def motion_phrase(tool_name: str, args: dict, execution: dict) -> str:
    """One motion call's line of doc 06 §4's ``execution.result``.

    T3.2 delegated this string to T3.4 (``tools.py``'s ``_record_motion`` repeats
    the model-facing facts into ``execution`` precisely "so the trial log's
    ``execution.result`` line can be written without re-parsing the payload")
    and pinned no format. Authored here, pinned by ``tests/test_loop.py``, and
    modelled on §4's own example — ``"moved 0.82 m, auto-stop on collision
    (bump)"``. It is human-readable only: every number in it is also a typed
    field of the same ``execution`` block, so no scorer ever has to parse it.
    """
    distance = execution.get("distance_moved_m", 0.0)
    seconds = execution.get("policy_seconds_used", 0.0)
    if tool_name == "move":
        parts = [f"moved {distance:.2f} m"]
    elif tool_name == "turn_to_heading":
        parts = [f"turned toward {args.get('heading_deg')} deg"]
    else:
        parts = [f"drove {distance:.2f} m"]
    parts.append(f"{seconds:.1f} policy-s")
    reason = execution.get("stop_reason", "")
    if execution.get("fell"):
        parts.append("FELL — trial over")
    elif reason == "bump":
        parts.append("auto-stop on collision (bump)")
    elif execution.get("bumped"):
        parts.append("collision reported (bump)")
    elif reason == "timeout":
        parts.append("timed out")
    return f"{tool_name}: " + ", ".join(parts)


def merge_executions(
    calls: list[tuple[str, dict, dict]], non_motion: list[str]
) -> dict:
    """doc 06 §4's singular ``turns[].execution``, from N motion calls.

    §4 has exactly ONE ``execution`` object per turn while doc 05 §3.3 explicitly
    allows several motion tools in one turn — "motion tools inside one turn
    execute sequentially and each advances physics; policy-seconds accumulate
    across them". The mismatch is resolved by **merging**, with the per-call
    records kept alongside (doc 06 §4 widened in the same commit, AGENTS.md
    rule 5):

    * ``policy_seconds_used`` sums, which is what the cap charges;
    * ``pose_trace`` concatenates in call order, which is exactly what §5.3's SPL
      path integral wants — it is one turn's true trajectory either way;
    * ``calls`` keeps every per-call ``execution`` dict verbatim, because a merge
      alone would destroy ``stop_reason`` and ``counted_as_bump``, and
      ``counted_as_bump`` is the **only** per-turn source for §5.6's two-source
      bump metric.

    ``motion_calls`` is recorded because doc 05 §12's open question — cap motion
    tools at one per turn? — is designated for T3.5's smoke to answer, and it can
    only answer it from data. Along with the per-turn ``policy_seconds_used`` and
    the stage total in ``budget``, this is the evidence for how far a chained
    turn overshoots the 240 s cap (which is checked after the whole turn, so an
    overshoot is doc-sanctioned and must be visible rather than clipped).

    ``execution`` is **always an object, never ``null`` and never absent**, even
    on a turn that stepped no physics: T4.1's scorer raises on a missing
    ``pose_trace``, so "no motion this turn" must be an empty trace rather than a
    missing key that is indistinguishable from a dropped one.
    """
    pose_trace: list[list[float]] = []
    seconds = 0.0
    records: list[dict] = []
    phrases: list[str] = []
    for tool_name, args, execution in calls:
        seconds += float(execution.get("policy_seconds_used", 0.0))
        pose_trace.extend(execution.get("pose_trace", []))
        records.append({"tool": tool_name, **execution})
        phrases.append(motion_phrase(tool_name, args, execution))
    if not phrases:
        phrases = [
            "no motion commanded"
            if not non_motion
            else "no motion; " + ", ".join(non_motion)
        ]
    return {
        "result": "; ".join(phrases),
        "policy_seconds_used": round(seconds, 4),
        "pose_trace": pose_trace,
        "motion_calls": len(calls),
        "calls": records,
    }


def memory_snapshot(memory: Memory, memory_block: str) -> dict:
    """doc 06 §4's ``turns[].memory_snapshot`` — widened, in the same commit.

    §4 shows ``{rooms, exits, trajectory, plan}`` while describing it as "full
    memory block as re-injected", which is a *string*. Both are logged: the
    rendered ``block`` is literally what the model read, and the structured
    fields are what a scorer can join on without re-parsing prose.

    Two corrections §4's shape needed:

    * there is no ``trajectory`` field on :class:`Memory`; ``room_sequence`` is
      what renders as ``Trajectory:``, so it is logged under §4's key name;
    * ``corrections`` was **absent from §4 entirely**, and doc 06 §5.8 requires
      "the count of ``correct_position`` calls and the magnitude of each
      correction" *per stage*. ``Correction`` already carries ``turn`` (stage-
      local) and ``stage``; after the batch nothing else can recover which stage
      a correction belonged to.

    Breadcrumbs are deliberately not logged: the series is exactly the per-turn
    ``obs.position_estimate`` column, and re-emitting the whole growing list into
    every turn would make the file quadratic in the episode length for no
    information.

    **Two vintages in one object, deliberately** (recorded in doc 06 §4 in the
    same commit). ``block`` is the PRE-dispatch rendering — literally the bytes
    re-injected into this turn's request, which is what §4 describes it as —
    while the structured fields are read POST-dispatch, so they include this
    turn's ``update_room`` / ``mark_exit`` / ``correct_position`` writes. Both
    are the right vintage for their consumer: §5.8's correction series needs the
    writes the turn made, and an audit of what the model read needs the block it
    read. The consequence to know is that a turn which creates a room lists it
    under ``rooms`` while ``block`` does not yet mention it — the two halves
    disagree by exactly one turn's writes, by construction.
    """
    return {
        "rooms": {
            name: {
                "name": room.name,
                "description": room.description,
                "landmarks": list(room.landmarks),
            }
            for name, room in memory.rooms.items()
        },
        "exits": [
            {"room": e.room, "direction_deg": e.direction_deg, "status": e.status}
            for e in memory.exits
        ],
        "trajectory": list(memory.room_sequence),
        "current_room": memory.current_room,
        "plan": memory.plan,
        "corrections": [
            {
                "turn": c.turn,
                "old_xy": [c.old_xy[0], c.old_xy[1]],
                "new_xy": [c.new_xy[0], c.new_xy[1]],
                "reason": c.reason,
                "stage": c.stage,
            }
            for c in memory.corrections
        ],
        "stage": memory.stage,
        "block": memory_block,
    }


# ---------------------------------------------------------------------------
# The QA exchange (doc 06 §5.9)
# ---------------------------------------------------------------------------

#: One answer marker's body: "1.", "1)", "**1.**", "- 1:", "Q1.",
#: "**Question 1:**". Deliberately permissive about decoration and, in the
#: line-anchored form below, strict about position: the preamble asks for
#: answers "numbered 1 to 5", and a digit mid-sentence must not split an answer.
#:
#: The ``Q``/``Question`` prefix was added by T3.4's review pass: measured
#: against the shipped matcher, a reply formatted ``**Question 1:** …`` matched
#: NO marker at all, so all five answers came back ``""`` and T4.1 would have
#: scored the trial 0/5 for a formatting reason rather than a map-quality one.
_ANSWER_MARKER = (
    r"(?:[*#>\-]+[ \t]*)?\*{{0,2}}(?:Q(?:uestion)?[ \t]*)?{number}\*{{0,2}}"
    r"[ \t]*[.):]\**[ \t]*"
)
_LINE_MARKER = r"^[ \t]*" + _ANSWER_MARKER
#: Same body, anywhere after whitespace — the last-resort pass, see below.
_INLINE_MARKER = r"(?:^|(?<=[\s]))" + _ANSWER_MARKER


def _find_line_marker(text: str, number: int, cursor: int):
    """First line-anchored marker for ``number`` at/after ``cursor``, column 0 first.

    Preferring an UNINDENTED match is what stops a nested numbered list from
    stealing a boundary. Measured on the shipped matcher, ``"1. Rooms:\\n   1.
    living\\n   2. kitchen\\n2. two…"`` split answer 1 to ``"Rooms:\\n   1.
    living"`` and answer 2 to ``"kitchen\\n2. two"`` — the sub-list's ``2.`` won
    because it came first. An indented match is still accepted when there is no
    column-0 one anywhere after the cursor, so a wholly-indented reply
    (``"  1 . alpha"``) parses exactly as before.
    """
    pattern = re.compile(_LINE_MARKER.format(number=number), re.MULTILINE)
    first = None
    for match in pattern.finditer(text, cursor):
        if first is None:
            first = match
        if not match.group(0)[:1].isspace():
            return match
    return first


def _scan_markers(text: str, wanted: list[int], inline: bool) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for number in wanted:
        if inline:
            pattern = re.compile(_INLINE_MARKER.format(number=number))
            match = pattern.search(text, cursor)
        else:
            match = _find_line_marker(text, number, cursor)
        if match is None:
            continue
        spans.append((number, match.start(), match.end()))
        cursor = match.end()
    return spans


def split_qa_answers(text: str, numbers: Iterable[int] = (1, 2, 3, 4, 5)) -> dict[int, str]:
    """Split one numbered blob into per-question answers (doc 06 §4's ``final.qa``).

    ``render_qa_prompt`` sends all five questions in ONE user message, so the
    model returns ONE text blob — but ``final.qa`` is a per-question array with a
    per-question ``answer``. Something has to split it, and it is this task
    rather than T4.1 because §4's shape is what T4.3 freezes and spends money
    against; the raw blob is logged beside it (``final.qa_raw``) so a scorer that
    disagrees with this split can always redo it from the original.

    Markers are searched **in ascending order, each after the previous one**, so
    a "2." inside answer 1's prose cannot steal the boundary, and a **column-0
    match wins over an indented one** so a nested numbered list cannot either
    (see :func:`_find_line_marker`). A question whose marker never appears gets
    ``""`` — the honest record of an unparseable answer, which T4.1 then scores
    0 rather than the harness inventing text.

    **The one-line fallback** (added by T3.4's review pass, doc 06 §5.9 amended
    in the same commit): if the line-anchored pass finds fewer than two markers,
    the whole scan is redone allowing a marker anywhere after whitespace. A
    model that answers all five on one line — ``"1. hallway 2. left 3. two
    rooms 4. NE 5. sofa"`` — otherwise scores 1/5, and formatting habits differ
    systematically between the three contestants, so that is a per-model
    penalty on a published metric rather than a random one. The fallback only
    runs when the strict pass has already failed, so prose digits inside a
    normally-formatted reply still cannot steal a boundary.
    """
    wanted = list(numbers)
    if not text:
        return {number: "" for number in wanted}
    spans = _scan_markers(text, wanted, inline=False)
    if len(spans) < 2:
        loose = _scan_markers(text, wanted, inline=True)
        if len(loose) > len(spans):
            spans = loose
    answers = {number: "" for number in wanted}
    for index, (number, _start, end) in enumerate(spans):
        stop = spans[index + 1][1] if index + 1 < len(spans) else len(text)
        answers[number] = text[end:stop].strip()
    return answers


# ---------------------------------------------------------------------------
# Stage results
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """How one stage ended, in both vocabularies (doc 05 §3.2 / doc 06 §4)."""

    stage: str
    end_reason: str
    outcome: str
    success: bool
    turns_used: int
    policy_seconds_used: float
    score: StageScore | None
    true_pose: tuple[float, float, float] | None

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "end_reason": self.end_reason,
            "outcome": self.outcome,
            "success": self.success,
            "turns_used": self.turns_used,
            "policy_seconds_used": round(self.policy_seconds_used, 4),
            # The decision inputs, so T4.1 recomputes the verdict instead of
            # trusting it (doc 06 §4: the log is the single source for scoring).
            "score": None if self.score is None else self.score.as_dict(),
            "true_pose": None
            if self.true_pose is None
            else {
                "x": round(self.true_pose[0], 4),
                "y": round(self.true_pose[1], 4),
                "heading_deg": round(self.true_pose[2], 2),
            },
        }

    @classmethod
    def not_run(cls, spec: StageSpec) -> "StageResult":
        """The stage that never started — doc 06 §12's "—" convention.

        Zero turns and zero policy-seconds are the literal truth (nothing ran),
        which is also what makes doc 06 §5.2's canonical progress formula give
        0.0 without a special case: the robot never moved, so
        ``d_final == d_initial``. Time-to-home and stage-2 drift stay genuinely
        undefined and are the scorer's "—" cells; they are not manufactured here.
        """
        return cls(
            stage=spec.name,
            end_reason=REASON_NOT_RUN,
            outcome=outcome_for(REASON_NOT_RUN, False),
            success=False,
            turns_used=0,
            policy_seconds_used=0.0,
            score=None,
            true_pose=None,
        )


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class EpisodeRunner:
    """One trial: two stages in one episode, then the QA exchange.

    The sim is reached only through ``context.playback`` / ``context.camera``,
    which is what lets ``tests/test_loop.py`` run the whole thing with no kit
    process (PLAN T3.4: "provider AND sim mocked").
    """

    def __init__(
        self,
        *,
        provider,
        context: ToolContext,
        stages: tuple[StageSpec, StageSpec],
        log: TrialLog,
        system_prompt: str = SYSTEM_PROMPT,
        tool_schemas: list[dict] | None = None,
        k: int = K_CONTEXT_TURNS,
        clock: Callable[[], str] = _utc_now,
        on_turn: Callable[[dict], None] | None = None,
    ) -> None:
        self.provider = provider
        self.context = context
        self.stages = stages
        self.log = log
        self.system_prompt = system_prompt
        self.tool_schemas = TOOL_SCHEMAS if tool_schemas is None else tool_schemas
        self.k = k
        self.clock = clock
        self.on_turn = on_turn

        self.transcript: list[TranscriptEntry] = []
        self.episode_usage = Usage()
        self.qa_usage = Usage()
        self.global_turn = 0
        #: Last TRUE pose observed while the episode was live. Maintained rather
        #: than re-read at the end because a fall has already teleported the
        #: robot back to spawn inside `env.step()` — `ExecResult.true_pose` is
        #: guarded against that and `playback.true_xy()` is not, so reading the
        #: live sensor after a fall would log the spawn point as the fall
        #: location and hand doc 06 §5.2's progress metric a free 1.0.
        self.last_true_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # -- convenience --------------------------------------------------------

    @property
    def memory(self) -> Memory:
        return self.context.memory

    @property
    def counters(self) -> Counters:
        return self.context.counters

    @property
    def integrator(self) -> PositionIntegrator:
        return self.context.integrator

    def _refresh_true_pose(self) -> None:
        """Sample ground truth while it is trustworthy (i.e. before any fall)."""
        x, y = self.context.playback.true_xy()
        self.last_true_pose = (x, y, self.context.playback.compass_deg())

    # -- the episode --------------------------------------------------------

    def run(self) -> dict:
        """Run both stages, then the QA exchange; return doc 06 §4's ``final``."""
        self._refresh_true_pose()
        stage1_spec, stage2_spec = self.stages
        stage1 = self.run_stage(stage1_spec)

        if runs_return_home(stage1.end_reason, stage1.success):
            # doc 05 §3.3's four steps: (1) stage 1 is scored above and the model
            # never sees the number, (2) the per-stage counters reset, (3) the
            # memory object and the transcript window are kept intact, (4) the
            # new objective already went out as declare_done's tool_result.
            #
            # `reset_for_stage()`, NEVER a rebuilt ToolContext: doc 05 §4.1 pins
            # that it zeroes exactly `turn`, `last_bumped`,
            # `last_contact_groups`, `last_distance_moved_m` and the two
            # Counters fields, and deliberately leaves `bumps` alone because
            # doc 06 §5.6 counts bumps over the TRIAL. Rebuilding would drop
            # every stage-1 collision from a headline metric with no test and
            # no traceback.
            self.context.reset_for_stage()
            # One line, and nothing else sets it: `reset_for_stage` deliberately
            # does not (it owns the *budget*, not the protocol). `Correction.turn`
            # is stage-local, so without this stamp doc 06 §5.8's per-stage
            # correction series cannot be split after the batch — there is no
            # other record of where the boundary was.
            self.memory.stage = STAGE_RETURN_HOME
            stage2 = self.run_stage(stage2_spec)
        else:
            stage2 = StageResult.not_run(stage2_spec)

        # doc 06 §5.9 says "after the episode ends" — a cap-out and a fall are
        # both ways for an episode to end, and PLAN T3.4's acceptance wants a
        # populated `final.qa` in every trial JSON. Skipping the failures would
        # drop a headline metric for exactly the trials most worth explaining,
        # and it would bias the QA aggregate toward models that finished.
        qa, qa_raw = self.run_qa()

        total = self.episode_usage + self.qa_usage
        return {
            "outcome": {stage1.stage: stage1.outcome, stage2.stage: stage2.outcome},
            "end_reason": {
                stage1.stage: stage1.end_reason,
                stage2.stage: stage2.end_reason,
            },
            "stages": {stage1.stage: stage1.as_dict(), stage2.stage: stage2.as_dict()},
            "bumps": self.context.bumps,
            # doc 06 §4: "§5 values, computed post-hoc by scorer". T4.1 fills it.
            "metrics": {},
            # WIDENED from doc 06 §4's {input, output, cost_usd_estimate} to
            # Usage.as_dict()'s five keys, recorded in §4 in the same commit. The
            # two cache fields are not decoration: doc 06 §8 names prompt caching
            # on the stable prefix as the main lever on Anthropic input cost, and
            # a batch that cannot see whether it hit the cache cannot report what
            # it actually spent. `input_tokens` here is the provider's own name,
            # so no remapping can silently transpose the two columns.
            "tokens": total.as_dict(),
            "tokens_breakdown": {
                "episode": self.episode_usage.as_dict(),
                "qa": self.qa_usage.as_dict(),
            },
            "qa": qa,
            "qa_raw": qa_raw,
            # WIDENED (doc 06 §4/§5.9, same commit): loud when the split found
            # fewer than five answers. `split_qa_answers` records an unparseable
            # answer as "" and T4.1 scores that 0, which is indistinguishable
            # from a model that answered badly — so a pure formatting mismatch
            # would silently cost up to 0.8 of a headline metric with nothing to
            # notice it by. `qa_raw` makes it recoverable; this flag makes it
            # VISIBLE, before scoring rather than after.
            "qa_parse_failed": any(not q["answer"] for q in qa),
        }

    # -- one stage ----------------------------------------------------------

    def run_stage(self, spec: StageSpec) -> StageResult:
        """doc 05 §3.1's ``run_stage``, phase for phase."""
        while True:
            result = self._run_turn(spec)
            if result is not None:
                return result

    def _run_turn(self, spec: StageSpec) -> StageResult | None:
        context = self.context
        counters = self.counters

        # --- 1. Assemble the request — the sim is PAUSED here ---------------
        #
        # `observed_compass_deg` and `integrator.xy` are the LIVE sensors, which
        # is doc 05 §5.2's recorded deviation from §3.1's pseudocode: the block
        # must not be rendered from the last breadcrumb, because
        # `correct_position` re-anchors the integrator WITHOUT appending a crumb,
        # so a crumb-rendered block would show the model the number it had just
        # overwritten. A swapped argument here is invisible except in the drift
        # metric.
        compass = observed_compass_deg(context)
        position = self.integrator.xy
        memory_block = render_memory_block(self.memory, counters, position, compass)
        # doc 06 §4's `obs` — "what the model was shown". Captured BEFORE the
        # model decides, because that is what "shown" means: the status triple
        # describes the previous turn's motion, which is exactly what the model
        # is reading when it chooses this turn's action.
        obs = {
            "frame_paths": [],
            "compass_deg": round(compass, 1),
            "position_estimate": {"x": round(position[0], 2), "y": round(position[1], 2)},
            "status": {
                "bumped": context.last_bumped,
                # Same carried reading `_state_payload` shows the model
                # (T3.5's contact field, recorded into doc 06 §4 in the same
                # commit): without it the log's "what the model was shown"
                # summary silently under-reports the one status field that says
                # WHICH way was blocked.
                "contact": list(context.last_contact_groups),
                "fell": bool(context.playback.fell),
                "distance_moved_m": round(context.last_distance_moved_m, 3),
            },
        }
        messages = build_request(memory_block, self.transcript, self.k)

        # --- 2. Model call --------------------------------------------------
        #
        # Not wrapped: the SDK already retries 429/5xx with backoff (doc 05 §7.1,
        # `max_retries`) and those retries are INSIDE this call, so they never
        # reach the counter below. An exhausted retry raises, which is doc 05
        # §8's infra path — the trial reruns whole, and catching it here would
        # convert a network fault into a scored model failure.
        turn = self.provider.send(self.system_prompt, messages, self.tool_schemas)
        counters.turns += 1
        context.turn += 1
        self.global_turn += 1
        self.episode_usage = self.episode_usage + turn.usage
        # The two counters count the same thing for two consumers — the budget
        # line the model reads and the stage-local index `correct_position`
        # stamps into every drift record. They are incremented at this one site;
        # the assertion is what makes a future second site fail loudly instead of
        # silently mislabelling a correction's turn.
        assert context.turn == counters.turns, (
            f"turn counters diverged: ToolContext.turn={context.turn}, "
            f"Counters.turns={counters.turns}"
        )

        # --- 3. Dispatch tool calls IN ORDER --------------------------------
        results: list[ToolResultBlock] = []
        motion: list[tuple[str, dict, dict]] = []
        non_motion: list[str] = []
        frames: list[ImageBlock] = []
        end_reason: str | None = None
        score: StageScore | None = None
        dispatched = 0

        for index, call in enumerate(turn.tool_calls):
            if call.name == DECLARE_DONE:
                # Branched on BEFORE `dispatch` (doc 05 §3.1): the result is the
                # stage OUTCOME, which no tool can compute. Scored at this call's
                # position in the list, so a `move` bundled ahead of it in the
                # same turn counts — the model declared from where it ended up.
                self._refresh_true_pose()
                score = score_stage(spec, (self.last_true_pose[0], self.last_true_pose[1]))
                results.append(
                    ToolOutcome(
                        payload=stage_end_result(
                            spec.name,
                            # T3.4's resolution of doc 05 §12 / doc 06 §12,
                            # asked of the SAME predicate `run()` uses to decide
                            # whether the stage actually runs — a second copy
                            # here could offer an objective for a leg that never
                            # happens. Passed explicitly: see
                            # `stage_end_result`'s docstring for why its default
                            # must never be the value that decides.
                            continue_to_return_home=runs_return_home(
                                REASON_DECLARE_DONE, bool(score.success)
                            ),
                        )
                    ).to_block(call.id, call.name)
                )
                # Every remaining tool_use block still gets an answer. doc 05 §8:
                # not a model failure to score — the stage simply ended first —
                # but an unanswered tool_use is an API error, i.e. an infra rerun
                # of a trial the model actually finished.
                for later in turn.tool_calls[index + 1 :]:
                    results.append(
                        ToolOutcome(
                            payload=not_executed(later.name), is_error=True
                        ).to_block(later.id, later.name)
                    )
                end_reason = REASON_DECLARE_DONE
                break

            outcome = dispatch(call, context)
            dispatched += 1
            results.append(outcome.to_block(call.id, call.name))
            frames.extend(outcome.images)
            if outcome.execution is not None:
                motion.append((call.name, dict(call.args), outcome.execution))
                pose = outcome.execution.get("true_pose")
                if pose:
                    # Guarded against the post-fall teleport by
                    # `policy_wrapper.execute`; the live sensor is not.
                    self.last_true_pose = (float(pose[0]), float(pose[1]), float(pose[2]))
            else:
                non_motion.append(call.name)

            if context.playback.fell:
                # doc 05 §4.1's residual, assigned to T3.4: `dispatch` refuses
                # further MOTION after a fall, but a `get_observation` listed
                # after the falling command would still answer — rendering the
                # trial's final frame from the spawn point Isaac teleported the
                # robot to. Only the loop can stop the rest of the turn.
                end_reason = REASON_FALL
                break

        # --- 3b. The transcript ---------------------------------------------
        #
        # doc 05 §3.1 returns on a fall WITHOUT appending, and justifies the drop:
        # no further model call happens, so the dropped turn is moot. The turn's
        # doc 06 §4 record is still written below — the provider transcript and
        # the scoring log are different artifacts, and T4.1's scorer raises on a
        # missing `pose_trace` for a turn that plainly stepped physics.
        if end_reason != REASON_FALL:
            if turn.tool_calls:
                self.transcript.append(
                    TranscriptEntry(native=turn.raw, results=results)
                )
            else:
                # doc 05 §8's refusal/derailment row, which §3.1's pseudocode has
                # no branch for: as literally written it would append an
                # assistant turn with no tool_use and an EMPTY user message.
                # Amended in doc 05 §3.1 in this commit (AGENTS.md rule 5). The
                # nudge is fixed text and the episode continues to its caps —
                # never a retry, because retrying only the trials that derailed
                # is selection bias in the model's favour.
                # `turn.raw` may be EMPTY here and legitimately so: an Anthropic
                # refusal is HTTP 200 with an empty `content` array, which is
                # exactly the shape that reaches this branch. Echoing it back
                # would emit `{"role": "assistant", "content": []}` on the next
                # request — an API 400, i.e. §8's INFRA path for a model failure
                # §8 says must be scored. The guard lives in
                # `AnthropicProvider.to_native` so it holds for every producer
                # of an empty turn, not just this one.
                self.transcript.append(
                    TranscriptEntry(native=turn.raw, results=[], note=DERAILMENT_NUDGE)
                )

        # --- 4. Caps (checked AFTER execution; never retried) ---------------
        #
        # Evaluated BEFORE the record is built so the turn that ends the stage
        # carries the reason, which is what doc 06 §4's annotation promises
        # ("§3.2's stop reason, on the turn that ends the stage"). Nothing about
        # the turn changes: it has already been dispatched in full and appended
        # to the transcript, and the stage still returns below.
        #
        # Read off `Counters`, not the module constants, so the caps the loop
        # enforces are byte-identical to the ones rendered into the budget line
        # the model budgets against.
        if end_reason is None:
            if counters.turns >= counters.turn_cap:
                end_reason = REASON_TURN_CAP
            elif counters.policy_seconds >= counters.policy_seconds_cap:
                end_reason = REASON_MOTION_CAP

        # --- 3c. The doc 06 §4 turn record ----------------------------------
        #
        # DEVIATION recorded in doc 06 §4 in the same commit: `obs.frame_paths`
        # lists the frames this turn's tool calls PRODUCED, which the model
        # reads on the NEXT request — while the rest of `obs` is the state it
        # read before deciding this turn. Attaching a frame to the turn that
        # captured it is what makes "show me the look_around from turn 7"
        # answerable; a qualitative audit asking "what was it looking at when it
        # decided" must read the PREVIOUS turn's paths.
        obs["frame_paths"] = self.log.save_frames(self.global_turn, frames)
        record = {
            "stage": spec.name,
            # STAGE-LOCAL, so it joins `Correction.turn` — the key doc 06 §5.8
            # needs to split the drift series per stage. The trial-global index
            # is carried alongside because it is what the frame filenames use and
            # what makes two stages' turn 7 distinguishable at a glance.
            "turn_idx": context.turn,
            "global_turn_idx": self.global_turn,
            "timestamp": self.clock(),
            "obs": obs,
            "model_output": {
                "thought": turn.thinking or "",
                "text": turn.text or "",
                "stop_reason": turn.stop_reason,
                "refusal": turn.refusal,
                "tool_calls": [
                    {"name": call.name, "args": _json_safe(call.args)}
                    for call in turn.tool_calls
                ],
                "dispatched": dispatched,
                "parse_errors": [
                    {"name": call.name, "detail": call.parse_error}
                    for call in turn.tool_calls
                    if call.parse_error
                ],
                "nudged": not turn.tool_calls,
            },
            "execution": merge_executions(motion, non_motion),
            # doc 06 §4's sibling of `execution`, logged EVERY turn — including
            # the turns that stepped no physics, where it comes from the live
            # sensors sampled at the top of this turn. DEVIATION from T3.2's
            # `execution["true_pose"]`, which is a 3-element list: §4 wants the
            # object, and T4.1 reads §4.
            "true_pose": {
                "x": round(self.last_true_pose[0], 4),
                "y": round(self.last_true_pose[1], 4),
                "heading_deg": round(self.last_true_pose[2], 2),
            },
            "memory_snapshot": memory_snapshot(self.memory, memory_block),
            # The evidence doc 05 §12's motion-tools-per-turn question needs, and
            # the only place a cap overshoot is visible: caps are checked AFTER
            # the whole turn, so one chained turn can legitimately end at 251 s.
            "budget": {
                "stage_turns_used": counters.turns,
                "stage_turn_cap": counters.turn_cap,
                "stage_policy_seconds_used": round(counters.policy_seconds, 4),
                "stage_policy_seconds_cap": counters.policy_seconds_cap,
            },
            "usage": turn.usage.as_dict(),
            "end_reason": end_reason,
        }
        self.log.append_turn(record)
        if self.on_turn is not None:
            self.on_turn(record)

        # --- 3d/4b. Terminal stop conditions ---------------------------------
        if end_reason == REASON_DECLARE_DONE:
            return self._stage_result(spec, end_reason, score)
        if end_reason is not None:
            # fall, turn_cap, motion_cap — all scored failures with partial
            # progress, none of them ever retried (doc 05 §3.2, §8).
            return self._stage_result(spec, end_reason, None)
        return None

    def _stage_result(
        self, spec: StageSpec, end_reason: str, score: StageScore | None
    ) -> StageResult:
        if score is None:
            # A capped or fallen stage is a scored failure with partial progress
            # (doc 06 §3.2), but the distance still goes in the log: doc 06 §5.2
            # needs d_final, and after a fall the only honest end pose is the
            # guarded pre-teleport one tracked above.
            score = score_stage(spec, (self.last_true_pose[0], self.last_true_pose[1]))
            success = False
        else:
            success = bool(score.success)
        return StageResult(
            stage=spec.name,
            end_reason=end_reason,
            outcome=outcome_for(end_reason, success),
            success=success,
            turns_used=self.counters.turns,
            policy_seconds_used=self.counters.policy_seconds,
            score=score,
            true_pose=self.last_true_pose,
        )

    # -- the QA exchange (doc 06 §5.9) --------------------------------------

    def run_qa(self) -> tuple[list[dict], str]:
        """Ask the 5 frozen questions in a FRESH exchange; return ``final.qa``.

        Fresh means what §5.9 says: no transcript, no camera, no sim, no tools —
        only the model's own final memory block, rendered by exactly the same
        function that rendered it into every request, so the QA measures the map
        the model built rather than a second round of exploration.

        The compass comes from :func:`observed_compass_deg`, so a trial that
        ended in a fall shows the latched pre-fall heading rather than the spawn
        heading Isaac teleported the robot to.

        ``score`` is left ``None``: it is T4.1's output, and a number written
        here would be a rubric applied by the producer of the answers.
        """
        block = render_memory_block(
            self.memory,
            self.counters,
            self.integrator.xy,
            observed_compass_deg(self.context),
        )
        prompt = render_qa_prompt(block)
        turn = self.provider.send(
            QA_SYSTEM_PROMPT, [UserMessage([TextBlock(prompt)])], QA_TOOLS
        )
        self.qa_usage = self.qa_usage + turn.usage
        answers = split_qa_answers(turn.text or "")
        qa = [
            {
                "number": question.number,
                "question": question.text,
                "answer": answers.get(question.number, ""),
                # T4.1's field. Explicitly null rather than absent, so a scorer
                # that never ran is distinguishable from one that scored 0.
                "score": None,
            }
            for question in LAYOUT_QA_QUESTIONS
        ]
        return qa, turn.text or ""


def run_trial(
    *,
    provider,
    context: ToolContext,
    stages: tuple[StageSpec, StageSpec],
    log: TrialLog,
    on_turn: Callable[[dict], None] | None = None,
) -> dict:
    """Run one whole trial and write doc 06 §4's ``final``. Returns the document."""
    runner = EpisodeRunner(provider=provider, context=context, stages=stages, log=log, on_turn=on_turn)
    final = runner.run()
    return log.finish(final)
