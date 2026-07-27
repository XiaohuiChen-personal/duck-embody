"""Episode-loop unit tests: context policy, stage machine, caps, error routing,
the QA exchange, and doc 06 §4's log schema.

**Provider AND sim are mocked** (PLAN T3.4): no API call, no kit process, no
Isaac import, no money. The fakes below reproduce only the *shapes* the real
components return, and they reuse the wrapper's own pure functions so no rule is
tested against a second implementation of itself.

The guards here protect failures that leave a green suite and a plausible-looking
transcript:

* **The first turn emitted twice.** Doc 05 §3.1 slices ``transcript[1:]`` for
  exactly this reason. Duplicating it costs the spawn image's tokens on every
  early turn and shows the model its own first move twice; nothing raises.

* **Images that never age out** (or that age out of the wrong entries). The
  frozen ``SYSTEM_PROMPT`` *promises* the model "only the first turn and the last
  10 turns are kept, and their images are dropped as they age out" — a harness
  that quietly kept them would make a frozen, unchangeable sentence false for
  every trial in the batch.

* **Ground truth reaching the model.** ``ExecResult`` carries four scoring-only
  fields and ``true_xy()`` is the answer key. The leak test below serialises
  everything the fake provider was ever sent and looks for sentinel values;
  nothing crashes when a benchmark hands over its own answer key, the numbers
  just stop meaning what they say (doc 06 §4).

* **A model failure taking the infra path.** Doc 05 §8 draws the line at agency:
  a bad tool call is scored, a GPU fault reruns the trial. Either direction
  crossed silently invalidates the N=4 numbers — a swallowed render error
  launders broken hardware into a model failure, and a raised ``invalid_args``
  hands a malformed call a free retry.

* **The stage that ends without ending.** ``dispatch`` refuses motion after a
  fall, but perception still answers; only the loop can stop the turn, and doc 05
  §4.1 assigns that to T3.4 by name.

* **`turns[].execution.pose_trace` missing.** T4.1's scorer is specified to RAISE
  on it rather than fall back, so an episode whose log drops it fails the hard
  gate — after the paid batch.
"""

from __future__ import annotations

import base64
import html
import json
import math
import re
from pathlib import Path

import pytest

from duck_embody.agent.loop import (
    FROZEN_FILES,
    K_CONTEXT_TURNS,
    QA_SYSTEM_PROMPT,
    QA_TOOLS,
    EpisodeRunner,
    TranscriptEntry,
    TrialLog,
    _json_safe,
    build_request,
    config_hash,
    context_messages,
    freeze_commit,
    merge_executions,
    motion_phrase,
    redact_secrets,
    run_trial,
    split_qa_answers,
)
from duck_embody.agent.memory import (
    POLICY_SECONDS_CAP,
    STAGE_FIND_KITCHEN,
    STAGE_RETURN_HOME,
    TURN_CAP,
    Counters,
    Memory,
    PositionIntegrator,
)
from duck_embody.agent.prompts import (
    DERAILMENT_NUDGE,
    LAYOUT_QA_PREAMBLE,
    LAYOUT_QA_QUESTIONS,
    STAGE2_OBJECTIVE_TOOL_RESULT,
    SYSTEM_PROMPT,
    render_memory_block,
)
from duck_embody.agent.providers.base import (
    AssistantMessage,
    AssistantTurn,
    ImageBlock,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    Usage,
    UserMessage,
)
from duck_embody.agent.tools import (
    DECLARE_DONE,
    TOOL_NAMES,
    TOOL_SCHEMAS,
    TRIAL_OVER_DETAIL,
    ToolContext,
    observed_compass_deg,
)
from duck_embody.env.apartment_layout import LAYOUT, spawn_pose, target_point
from duck_embody.env.camera import encode_b64
from duck_embody.sim.policy_wrapper import (
    CONTROL_DT,
    MACRO_CHUNK_S,
    MOVE_SPEED_MPS,
    ExecResult,
    clamp_command,
    duration_to_steps,
)
from duck_embody.tasks.find_kitchen import (
    OUTCOME_DECLARED_ELSEWHERE,
    OUTCOME_FALL,
    OUTCOME_NOT_RUN,
    OUTCOME_SUCCESS,
    OUTCOME_TIMEOUT_MOTION,
    OUTCOME_TIMEOUT_TURNS,
    REASON_DECLARE_DONE,
    REASON_FALL,
    REASON_MOTION_CAP,
    REASON_NOT_RUN,
    REASON_TURN_CAP,
    STAGE2_REQUIRES_STAGE1_SUCCESS,
    StageSpec,
    find_kitchen_spec,
    outcome_for,
    return_home_spec,
    score_stage,
    stage_specs,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_YAML = REPO_ROOT / "configs" / "benchmark.yaml"
DOC_06 = REPO_ROOT / "docs" / "designs" / "06-benchmark-evaluation.html"

SEED = 101
SPAWN_XY, SPAWN_HEADING = spawn_pose(SEED)

#: Ground truth planted in the fakes. Chosen so no legitimate model-facing number
#: can collide with them, the way tests/test_tools.py does it.
TRUE_XY = (7.77, -8.88)
TRUE_HEADING = 123.45
TRUE_POSE = (TRUE_XY[0], TRUE_XY[1], TRUE_HEADING)
TRUE_DISPLACEMENT_M = 9.99
POSE_TRACE_SENTINEL = [(6.66, -5.55), (7.77, -8.88)]

START_COMPASS_DEG = 90.0


# ---------------------------------------------------------------------------
# Fakes — sim
# ---------------------------------------------------------------------------


def _exec_result(**kwargs) -> ExecResult:
    base = dict(
        commanded=(0.0, 0.0, 0.0),
        duration_s=0.0,
        steps=0,
        policy_seconds=0.0,
        bumped=False,
        fell=False,
        pose_trace=list(POSE_TRACE_SENTINEL),
        sampled_xy=[TRUE_XY],
        true_pose=TRUE_POSE,
        true_displacement_m=TRUE_DISPLACEMENT_M,
    )
    base.update(kwargs)
    return ExecResult(**base)


class FakePlayback:
    """``PolicyPlayback``'s surface, with the post-fall teleport modelled.

    ``true_xy`` is a settable sentinel so a test can put the robot exactly on or
    off the success radius; ``compass_deg`` jumps to ``spawn_compass_deg`` after
    a fall, which is what Isaac's in-``step()`` auto-reset really does.
    """

    def __init__(self, compass_deg: float = START_COMPASS_DEG):
        #: Mirrors PolicyPlayback.fall_diagnostics — the real one is a
        #: property over instance state that survives the call, which is
        #: what tools.py falls back to when an ExecResult lacks it.
        self.fall_diagnostics: dict | None = None
        self._compass = compass_deg
        self._fell = False
        self._true_xy = TRUE_XY
        self.spawn_compass_deg = 45.0
        self.calls: list[tuple[str, dict]] = []
        self.move_falls = False
        self.bumped = False
        #: Sampled at the bump by the real `execute()` — non-empty iff bumped,
        #: mirroring `tests/test_tools.py`'s fake.
        self.bump_contact_groups: list[str] = ["torso"]
        self.move_policy_seconds = 3 * MACRO_CHUNK_S

    # -- sensors (SCORING ONLY except compass_deg) --------------------------

    def true_xy(self) -> tuple[float, float]:
        return self._true_xy

    def set_true_xy(self, xy) -> None:
        self._true_xy = (float(xy[0]), float(xy[1]))

    def compass_deg(self) -> float:
        return self._compass

    @property
    def fell(self) -> bool:
        return self._fell

    def _teleport_to_spawn(self) -> None:
        self._compass = self.spawn_compass_deg
        self._true_xy = (SPAWN_XY[0], SPAWN_XY[1])

    # -- execution ----------------------------------------------------------

    def execute(self, vx, vy, wz, duration_s, stop_on_bump=False, stop_predicate=None):
        self.calls.append(("execute", dict(vx=vx, vy=vy, wz=wz, duration_s=duration_s)))
        commanded, notes = clamp_command(vx, vy, wz)
        steps = duration_to_steps(duration_s)
        return _exec_result(
            commanded=commanded,
            duration_s=duration_s,
            steps=steps,
            policy_seconds=steps * CONTROL_DT,
            bumped=self.bumped,
            fell=self._fell,
            clamp_notes=notes,
            stop_reason="",
        )

    def turn_to_heading(self, heading_deg, **kwargs):
        self.calls.append(("turn_to_heading", dict(heading_deg=heading_deg)))
        self._compass = heading_deg
        return _exec_result(
            commanded=(0.0, 0.0, 0.3),
            steps=duration_to_steps(4 * MACRO_CHUNK_S),
            policy_seconds=4 * MACRO_CHUNK_S,
            fell=self._fell,
            stop_reason="reached",
        )

    def move(self, distance_m, hold_heading=True, stop_on_bump=True, on_chunk=None):
        self.calls.append(("move", dict(distance_m=distance_m)))
        fell = self.move_falls
        drive_s = self.move_policy_seconds
        settle_s = 0.0 if fell else MACRO_CHUNK_S
        travelled = MOVE_SPEED_MPS * drive_s
        if fell:
            self._fell = True
        result = _exec_result(
            commanded=(MOVE_SPEED_MPS, 0.0, 0.0),
            steps=duration_to_steps(drive_s + settle_s),
            policy_seconds=drive_s + settle_s,
            bumped=self.bumped,
            contact_groups=list(self.bump_contact_groups) if self.bumped else [],
            fell=fell,
            stop_reason="fell" if fell else ("bump" if self.bumped else "reached"),
            dead_reckoned_distance_m=travelled,
        )
        if fell:
            self._teleport_to_spawn()
        return result


class FakeCamera:
    """``HeadCamera``'s capture surface. Real arrays, so the JPEG encoder runs."""

    def __init__(self):
        self.captures = 0

    @staticmethod
    def _frame(seed: int):
        import numpy as np

        return np.full((8, 8, 3), seed % 256, dtype=np.uint8)

    def capture_b64(self, quality: int = 85) -> str:
        self.captures += 1
        return encode_b64(self._frame(self.captures), quality)

    def look_around(self, bearings_deg=(0, 90, 180, 270)):
        return [
            (bearing, self._frame(i + 2), (TRUE_XY[0], TRUE_XY[1], 0.0))
            for i, bearing in enumerate(bearings_deg)
        ]


class ExplodingCamera(FakeCamera):
    """A render failure — doc 05 §8's infra path. Must NOT be caught."""

    def capture_b64(self, quality: int = 85) -> str:
        raise RuntimeError("RTX render failed: no frame produced")


# ---------------------------------------------------------------------------
# Fakes — provider
# ---------------------------------------------------------------------------


def call(name: str, call_id: str | None = None, /, **args) -> ToolCall:
    """Build a ``ToolCall``. Positional-only, because ``update_room`` and
    ``set_current_room`` both take an argument literally called ``name``."""
    return ToolCall(id=call_id or f"call_{name}", name=name, args=dict(args))


def turn(*tool_calls: ToolCall, text: str = "", thinking: str = "",
         stop_reason: str = "tool_use", refusal: str | None = None) -> AssistantTurn:
    return AssistantTurn(
        text=text,
        tool_calls=list(tool_calls),
        usage=Usage(input_tokens=100, output_tokens=20, cost_usd=0.001),
        # `raw` is echoed back verbatim; a marker string is enough to prove it
        # travelled unchanged.
        raw=[{"type": "marker", "calls": [c.name for c in tool_calls], "text": text}],
        stop_reason=stop_reason,
        thinking=thinking,
        refusal=refusal,
    )


class FakeProvider:
    """Replays a script of ``AssistantTurn``s and records every request."""

    name = "fake"
    model_id = "fake-model-1"

    def __init__(self, script, qa_text: str = ""):
        self.script = list(script)
        self.qa_text = qa_text
        self.requests: list[dict] = []
        self.index = 0

    def send(self, system, messages, tools) -> AssistantTurn:
        self.requests.append({"system": system, "messages": messages, "tools": tools})
        if tools == [] and system == QA_SYSTEM_PROMPT:
            return AssistantTurn(
                text=self.qa_text,
                tool_calls=[],
                usage=Usage(input_tokens=500, output_tokens=200, cost_usd=0.01),
                raw=[],
                stop_reason="end_turn",
            )
        if self.index >= len(self.script):
            # An exhausted script means the loop asked for a turn the test did
            # not anticipate — always a bug in the test or the loop, never
            # something to paper over with a default reply.
            raise AssertionError(
                f"provider script exhausted after {self.index} turns; the loop "
                "asked for another"
            )
        item = self.script[self.index]
        self.index += 1
        return item() if callable(item) else item


def standing_at(playback, xy, assistant_turn):
    """A script entry that puts the fake robot at ``xy`` before the turn runs.

    The fake sim has no physics, so a test that needs the robot to *be*
    somewhere when ``declare_done`` is scored says so explicitly rather than
    pretending a ``move`` carried it there.
    """

    def item() -> AssistantTurn:
        playback.set_true_xy(xy)
        return assistant_turn

    return item


class ExplodingProvider:
    """An exhausted-retry API failure — doc 05 §8's infra path."""

    name = "boom"
    model_id = "boom-1"

    def send(self, system, messages, tools):
        raise ConnectionError("connection reset by peer (retries exhausted)")


# ---------------------------------------------------------------------------
# Harness helpers
# ---------------------------------------------------------------------------


def make_context(playback=None, camera=None, memory=None, counters=None,
                 integrator=None) -> ToolContext:
    return ToolContext(
        playback=playback or FakePlayback(),
        camera=camera or FakeCamera(),
        memory=memory or Memory(),
        integrator=integrator or PositionIntegrator(*SPAWN_XY),
        counters=counters or Counters(),
    )


def make_log(tmp_path: Path, trial_id: str = "fake_seed101") -> TrialLog:
    return TrialLog(
        tmp_path / f"{trial_id}.json",
        trial_id=trial_id,
        model_id="fake-model-1",
        model_name="fake",
        seed=SEED,
        spawn_xy=SPAWN_XY,
        spawn_heading_deg=SPAWN_HEADING,
    )


def make_runner(tmp_path, script, *, qa_text="", context=None, trial_id="fake_seed101"):
    context = context or make_context()
    provider = FakeProvider(script, qa_text=qa_text)
    log = make_log(tmp_path, trial_id)
    runner = EpisodeRunner(
        provider=provider, context=context, stages=stage_specs(SEED), log=log
    )
    return runner, provider, log


def entry(index: int, *, with_image: bool = True) -> TranscriptEntry:
    """A transcript entry whose text and image both identify their turn."""
    images = [ImageBlock(data_b64=f"IMAGE-{index}")] if with_image else []
    return TranscriptEntry(
        native=[{"turn": index}],
        results=[
            ToolResultBlock(
                tool_use_id=f"call_{index}",
                tool_name="get_observation",
                text=json.dumps({"turn": index}),
                images=images,
            )
        ],
    )


def target_true_xy(offset_m: float = 0.0) -> tuple[float, float]:
    """A true pose ``offset_m`` east of the kitchen-counter target point."""
    tx, ty = target_point()
    return (tx + offset_m, ty)


def flatten(value) -> str:
    """Everything in a request, as one string — for the leak sweep."""
    return json.dumps(value, default=repr)


def request_text(request: dict) -> str:
    parts = [request["system"], flatten(request["tools"])]
    for message in request["messages"]:
        if isinstance(message, AssistantMessage):
            parts.append(flatten(message.native))
            continue
        for block in message.blocks:
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, ImageBlock):
                parts.append(block.data_b64)
                parts.append(block.label or "")
            elif isinstance(block, ToolResultBlock):
                parts.append(block.text)
                for image in block.images:
                    parts.append(image.data_b64)
                    parts.append(image.label or "")
    return "\n".join(parts)


def user_blocks(messages) -> list:
    out = []
    for message in messages:
        if isinstance(message, UserMessage):
            out.extend(message.blocks)
    return out


# ---------------------------------------------------------------------------
# Helpers: pull doc 06's own schema out of the HTML, so the test cites the source
# ---------------------------------------------------------------------------


def doc_06_schema_block() -> str:
    """Doc 06 §4's annotated JSON schema, extracted from the design doc itself.

    Extracted rather than transcribed, for the reason ``tests/test_tools.py``
    and ``tests/test_memory.py`` extract theirs: a hand-copied golden can be
    edited in the same commit as the code it is supposed to police, and nothing
    fails. It was: six doc-mandated fields (``usage``,
    ``memory_snapshot.current_room``, ``corrections[].old_xy/new_xy``,
    ``execution.calls[].pose_trace``, ``stages[].true_pose``, ``video_path``)
    could each be deleted with the whole suite still green, because
    ``REQUIRED_TURN_KEYS`` below is a partial hand-copy.
    """
    source = DOC_06.read_text(encoding="utf-8")
    start = source.index('<h2 id="logs">')
    match = re.search(r"<pre><code>(.*?)</code></pre>", source[start:], flags=re.DOTALL)
    assert match, "doc 06 §4's annotated schema block is no longer in the HTML"
    return html.unescape(match.group(1))


def doc_06_key_paths() -> set[str]:
    """Every ``"key":`` in §4's block, as a dotted path with arrays transparent.

    The block is annotated pseudo-JSON (``//`` comments, ``...`` ellipses, bare
    ``x``/``y`` placeholders), so it is scanned rather than parsed. Containers
    push the key that owns them; a container with no owning key (an array's
    elements) pushes a transparent frame, so ``"calls": [ {"tool": ...} ]``
    yields ``execution.calls.tool`` rather than an empty segment.
    """
    text = doc_06_schema_block()
    paths: set[str] = set()
    stack: list[str | None] = []
    pending: str | None = None
    index, size = 0, len(text)
    while index < size:
        char = text[index]
        if char == '"':
            end = index + 1
            while end < size and text[end] != '"':
                end += 2 if text[end] == "\\" else 1
            token = text[index + 1 : end]
            index = end + 1
            after = index
            while after < size and text[after] in " \t\n":
                after += 1
            if after < size and text[after] == ":":
                paths.add(".".join([*(s for s in stack if s), token]))
                pending = token
                index = after + 1
            continue
        if text[index : index + 2] == "//":
            newline = text.find("\n", index)
            if newline == -1:
                break
            index = newline
            continue
        if char in "{[":
            stack.append(pending)
            pending = None
        elif char in "}]":
            if stack:
                stack.pop()
            pending = None
        elif char == ",":
            pending = None
        index += 1
    return paths


def has_key_path(document, path: str) -> bool:
    """Does ``path`` exist in ``document``? Lists are walked transparently."""
    parts = path.split(".")

    def walk(node, depth: int) -> bool:
        if depth == len(parts):
            return True
        if isinstance(node, list):
            return any(walk(item, depth) for item in node)
        if isinstance(node, dict) and parts[depth] in node:
            return walk(node[parts[depth]], depth + 1)
        return False

    return walk(document, 0)


def doc_06_manifest_files() -> set[str]:
    """The frozen-file manifest doc 06 §2 names, expanded out of its brace form."""
    source = DOC_06.read_text(encoding="utf-8")
    start = source.index("THE HASHED MANIFEST")
    # The <ul> only — the surrounding prose also mentions files, but the list is
    # the manifest and a test that scraped the prose too would be untestable.
    bullets = source[source.index("<ul>", start) : source.index("</ul>", start)]
    listed: set[str] = set()
    for item in bullets.split("<li>")[1:]:
        for raw in re.findall(r"<code>([^<]+)</code>", item):
            text = html.unescape(raw).strip()
            if not re.fullmatch(r"[\w./{},\-]+\.(py|yaml)", text):
                continue
            brace = re.search(r"\{([^}]*)\}", text)
            if brace:
                listed.update(
                    text[: brace.start()] + option.strip() + text[brace.end() :]
                    for option in brace.group(1).split(",")
                )
            else:
                listed.add(text)
    assert listed, "doc 06 §2's hashed-manifest list is no longer in the HTML"
    return listed


# ===========================================================================
# 1. Context assembly — doc 05 §3.1, §5.2
# ===========================================================================


class TestContextWindow:
    def test_empty_transcript_yields_no_context_messages(self):
        assert context_messages([]) == []

    @pytest.mark.parametrize("n", range(1, 26))
    def test_the_first_turn_is_never_emitted_twice(self, n):
        """PLAN T3.4 names this test explicitly.

        Doc 05 §3.1 computes the K window over ``transcript[1:]`` so that while
        the transcript is short the pinned first turn is not ALSO the first entry
        of the window. Get it wrong and every early turn pays the spawn image's
        token cost twice, the model sees its own opening move duplicated, and
        nothing fails.
        """
        transcript = [entry(i) for i in range(n)]
        messages = context_messages(transcript)
        natives = [
            m.native[0]["turn"] for m in messages if isinstance(m, AssistantMessage)
        ]
        assert natives == sorted(natives)
        assert len(natives) == len(set(natives)), f"duplicate turn in {natives}"
        assert natives[0] == 0, "the first turn must always be pinned"

    @pytest.mark.parametrize(
        "n,expected",
        [(1, 1), (5, 5), (10, 10), (11, 11), (12, 11), (20, 11), (60, 11)],
    )
    def test_window_size_is_one_plus_min_k_and_the_rest(self, n, expected):
        """``1 + min(K, len-1)`` entries — never K+1 distinct with a duplicate."""
        messages = context_messages([entry(i) for i in range(n)])
        assert sum(isinstance(m, AssistantMessage) for m in messages) == expected

    def test_the_oldest_turns_are_the_ones_dropped(self):
        transcript = [entry(i) for i in range(15)]
        messages = context_messages(transcript)
        kept = [m.native[0]["turn"] for m in messages if isinstance(m, AssistantMessage)]
        assert kept == [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    def test_every_assistant_turn_is_answered(self):
        """Doc 05 §7.2: an unanswered ``tool_use`` is an API error — i.e. an
        infra rerun of a trial the model actually finished."""
        messages = context_messages([entry(i) for i in range(15)])
        roles = ["assistant" if isinstance(m, AssistantMessage) else "user" for m in messages]
        assert roles == ["assistant", "user"] * 11

    def test_k_is_configurable_for_tests_but_defaults_to_ten(self):
        assert K_CONTEXT_TURNS == 10
        messages = context_messages([entry(i) for i in range(9)], k=3)
        kept = [m.native[0]["turn"] for m in messages if isinstance(m, AssistantMessage)]
        assert kept == [0, 6, 7, 8]


class TestImageAging:
    def _images(self, messages) -> dict[int, list[str]]:
        found: dict[int, list[str]] = {}
        turn_index = None
        for message in messages:
            if isinstance(message, AssistantMessage):
                turn_index = message.native[0]["turn"]
                continue
            for block in message.blocks:
                if isinstance(block, ToolResultBlock):
                    found[turn_index] = [i.data_b64 for i in block.images]
        return found

    def test_all_images_survive_while_the_transcript_is_short(self):
        images = self._images(context_messages([entry(i) for i in range(10)]))
        assert all(images[i] == [f"IMAGE-{i}"] for i in range(10))

    def test_images_older_than_k_are_dropped_but_their_text_survives(self):
        transcript = [entry(i) for i in range(15)]
        messages = context_messages(transcript)
        images = self._images(messages)
        # Only the last K entries carry frames.
        assert [i for i, imgs in images.items() if imgs] == list(range(5, 15))
        # The JSON status text of the aged first turn is still there — the map
        # text is the durable record (doc 05 §2's keyframe row).
        assert '{"turn": 0}' in request_text(
            {"system": "", "tools": [], "messages": messages}
        )

    def test_the_pinned_first_turn_loses_its_image_too(self):
        """Doc 05 §12: "the kept first turn loses its image too … Currently:
        dropped, per the uniform rule" — and the FROZEN system prompt promises
        the model exactly that. Pinning the spawn frame permanently would make an
        unchangeable sentence false for every trial in the batch."""
        assert self._images(context_messages([entry(i) for i in range(11)]))[0] == []
        # ...while it is still inside the window, it keeps it.
        assert self._images(context_messages([entry(i) for i in range(10)]))[0] == [
            "IMAGE-0"
        ]

    def test_a_fourteen_turn_transcript_carries_exactly_the_last_ten_images(self):
        """The exact image set at an arbitrary mid-episode length, hunting the
        off-by-one at the window edge: entries 4-13 (the last K=10) keep their
        frames, the pinned first turn is emitted but bare, entries 1-3 are gone
        entirely. Off by one in ``image_floor`` and either entry 3's image
        reappears without its entry or entry 4 goes dark while in-window."""
        images = self._images(context_messages([entry(i) for i in range(14)]))
        assert sorted(images) == [0, *range(4, 14)], "entries emitted"
        assert images[0] == [], "pinned first turn is bare at n=14"
        assert all(images[i] == [f"IMAGE-{i}"] for i in range(4, 14))

    @pytest.mark.parametrize(
        "n,first_turn_keeps_image", [(10, True), (11, False), (12, False)]
    )
    def test_the_first_turn_image_drops_exactly_when_it_leaves_the_window(
        self, n, first_turn_keeps_image
    ):
        """K, K+1, K+2: the spawn frame survives while entry 0 is among the
        last K entries (n <= 10) and drops the moment it is not (n = 11), and
        stays dropped — never flickering back at n = 12."""
        images = self._images(context_messages([entry(i) for i in range(n)]))
        expected = ["IMAGE-0"] if first_turn_keeps_image else []
        assert images[0] == expected

    def test_the_prompt_promise_and_the_constant_agree(self):
        """The one sentence the model is told about the context policy."""
        assert (
            "only the first turn and the last 10 turns are kept, and their "
            "images are dropped as they age out" in " ".join(SYSTEM_PROMPT.split())
        )
        assert K_CONTEXT_TURNS == 10


class TestMemoryBlockInEveryRequest:
    def test_build_request_appends_the_block_as_a_trailing_user_message(self):
        messages = build_request("== YOUR MAP ==", [entry(0), entry(1)])
        assert isinstance(messages[-1], UserMessage)
        assert messages[-1].blocks == [TextBlock("== YOUR MAP ==")]

    def test_the_block_is_not_persisted_into_the_transcript(self):
        transcript = [entry(0)]
        build_request("BLOCK-A", transcript)
        second = build_request("BLOCK-B", transcript)
        assert "BLOCK-A" not in request_text(
            {"system": "", "tools": [], "messages": second}
        )

    def test_first_request_of_a_trial_is_the_block_alone(self):
        messages = build_request("BLOCK", [])
        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)

    def test_every_request_of_a_real_episode_carries_a_fresh_block(self, tmp_path):
        runner, provider, _ = make_runner(
            tmp_path,
            [
                turn(call("get_observation")),
                turn(call("update_plan", text="explore east")),
                turn(call(DECLARE_DONE)),
            ],
        )
        runner.context.playback.set_true_xy(target_true_xy(5.0))
        runner.run()
        episode_requests = [
            r for r in provider.requests if r["system"] == SYSTEM_PROMPT
        ]
        assert len(episode_requests) == 3
        for index, request in enumerate(episode_requests):
            trailing = request["messages"][-1]
            assert isinstance(trailing, UserMessage)
            text = trailing.blocks[0].text
            assert text.startswith("== YOUR MAP")
            assert "== STATE" in text and "== YOUR PLAN" in text
            # Turn counter is live: the block re-renders, it is not cached.
            assert f"Budget: turns {index}/{TURN_CAP}" in text

    def test_the_block_is_never_folded_into_the_system_string(self, tmp_path):
        """doc 06 §8's prompt-cache lever: the system prompt + tool schema is the
        stable prefix. A memory block concatenated into it would change every
        turn and invalidate the cache on every single request."""
        runner, provider, _ = make_runner(tmp_path, [turn(call(DECLARE_DONE))])
        runner.run()
        for request in provider.requests:
            # `== YOUR MAP ==` appears in SYSTEM_PROMPT as prose describing the
            # block, so the marker has to be a line only the RENDERER produces.
            assert "Position estimate: x=" not in request["system"]
            assert "Budget: turns" not in request["system"]
            assert request["system"] in (SYSTEM_PROMPT, QA_SYSTEM_PROMPT)


# ===========================================================================
# 2. The stage machine — doc 05 §3.3, doc 06 §3.1
# ===========================================================================


class TestStageTransition:
    def test_success_offers_the_return_leg_and_runs_stage_two(self, tmp_path):
        playback = FakePlayback()
        context = make_context(playback=playback)
        provider = FakeProvider(
            [
                turn(call("move", distance_m=1.0)),
                # stage 1 — declared at the counter
                standing_at(playback, target_true_xy(0.1), turn(call(DECLARE_DONE))),
                turn(call("move", distance_m=1.0)),
                # stage 2 — declared back at spawn
                standing_at(playback, SPAWN_XY, turn(call(DECLARE_DONE))),
            ]
        )
        log = make_log(tmp_path)
        runner = EpisodeRunner(
            provider=provider, context=context, stages=stage_specs(SEED), log=log
        )
        final = runner.run()

        assert final["outcome"][STAGE_FIND_KITCHEN] == OUTCOME_SUCCESS
        assert final["outcome"][STAGE_RETURN_HOME] == OUTCOME_SUCCESS
        assert log.document["turns"][1]["end_reason"] == REASON_DECLARE_DONE
        # doc 05 §3.3 item (4): the new objective arrives as declare_done's own
        # tool_result, in the same user message as the turn's other results.
        blocks = [
            b
            for r in provider.requests
            for b in user_blocks(r["messages"])
            if isinstance(b, ToolResultBlock) and b.tool_name == DECLARE_DONE
        ]
        assert STAGE2_OBJECTIVE_TOOL_RESULT in blocks[0].text

    def test_counters_reset_but_memory_bumps_and_transcript_do_not(self, tmp_path):
        """Doc 05 §3.3 items (2) and (3), plus doc 06 §5.6's trial-scoped bumps.

        Rebuilding the ToolContext instead of calling ``reset_for_stage()`` would
        zero ``bumps`` and halve a published headline metric with no traceback —
        the failure doc 05 §4.1 pins by name.
        """
        playback = FakePlayback()
        context = make_context(playback=playback)
        provider = FakeProvider(
            [
                turn(call("update_room", name="start", description="sofa")),
                standing_at(playback, target_true_xy(0.0), turn(call(DECLARE_DONE))),
                turn(call("get_observation")),
                standing_at(playback, SPAWN_XY, turn(call(DECLARE_DONE))),
            ]
        )
        log = make_log(tmp_path)
        runner = EpisodeRunner(
            provider=provider, context=context, stages=stage_specs(SEED), log=log
        )
        context.bumps = 5
        transcript_len_before = None

        original = runner.run_stage

        def run_stage(spec):
            nonlocal transcript_len_before
            result = original(spec)
            if spec.name == STAGE_FIND_KITCHEN:
                transcript_len_before = len(runner.transcript)
            return result

        runner.run_stage = run_stage
        final = runner.run()

        assert runner.context.bumps == 5, "bumps are TRIAL-scoped (doc 06 §5.6)"
        assert "start" in runner.memory.rooms, "the memory object survives"
        assert len(runner.transcript) > transcript_len_before, "transcript kept"
        assert runner.memory.stage == STAGE_RETURN_HOME
        # Stage 2's own budget starts at zero.
        stage2_turns = [t for t in log.document["turns"] if t["stage"] == STAGE_RETURN_HOME]
        assert stage2_turns[0]["turn_idx"] == 1
        assert stage2_turns[0]["budget"]["stage_turns_used"] == 1
        assert stage2_turns[0]["budget"]["stage_policy_seconds_used"] == 0.0
        assert final["stages"][STAGE_RETURN_HOME]["outcome"] == OUTCOME_SUCCESS

    def test_a_stage_one_contact_list_does_not_leak_into_stage_two(self, tmp_path):
        """T3.5 added ``last_contact_groups`` AFTER ``reset_for_stage`` was
        written. Left carried, a trial whose last stage-1 motion bumped would
        open stage 2 with ``bumped: false, contact: ["torso"]`` — a contact
        list from a bump the same payload no longer reports — shown identically
        to all three models at the one boundary every successful trial crosses.
        """
        playback = FakePlayback()
        playback.bumped = True  # stage 1's move bumps and samples ["torso"]
        context = make_context(playback=playback)
        provider = FakeProvider(
            [
                turn(call("move", distance_m=1.0)),
                standing_at(playback, target_true_xy(0.1), turn(call(DECLARE_DONE))),
                turn(call("get_observation")),
                standing_at(playback, SPAWN_XY, turn(call(DECLARE_DONE))),
            ]
        )
        log = make_log(tmp_path)
        runner = EpisodeRunner(
            provider=provider, context=context, stages=stage_specs(SEED), log=log
        )
        final = runner.run()
        assert final["outcome"][STAGE_FIND_KITCHEN] == OUTCOME_SUCCESS

        # Positive control: the bumped move really did carry its contact into
        # the next payload the model read (turn 2's request echoes turn 1's
        # tool_result via the transcript).
        move_result = json.loads(runner.transcript[0].results[0].text)
        assert move_result["status"]["contact"] == ["torso"]

        # The stage-2 get_observation — the first payload after the boundary —
        # must NOT report stage-1's contact beside a reset `bumped: false`.
        stage2_obs = json.loads(runner.transcript[2].results[0].text)
        assert stage2_obs["status"]["bumped"] is False
        assert stage2_obs["status"]["contact"] == []
        # And the log's pre-decision obs for that turn says the same.
        stage2_turns = [
            t for t in log.document["turns"] if t["stage"] == STAGE_RETURN_HOME
        ]
        assert stage2_turns[0]["obs"]["status"]["contact"] == []

    def test_the_first_stage_two_block_reads_zero_of_forty(self, tmp_path):
        """Q2's resolution, made visible: per-stage means the budget RESETS and
        the caps do not change, so the model's first return-home request says
        ``turns 0/40``. A cumulative line would read ``turns 31/40`` — an
        immediate false cap on a stage that has not started."""
        runner, provider, _ = make_runner(
            tmp_path,
            [
                turn(call("move", distance_m=1.0)),
                turn(call(DECLARE_DONE)),
                turn(call("get_observation")),
                turn(call(DECLARE_DONE)),
            ],
        )
        runner.context.playback.set_true_xy(target_true_xy(0.0))
        runner.run()
        blocks = [
            b.text
            for r in provider.requests
            if r["system"] == SYSTEM_PROMPT
            for b in [r["messages"][-1].blocks[0]]
        ]
        assert f"Budget: turns 0/{TURN_CAP}, policy-seconds 0.0/240" in blocks[2]

    def test_the_memory_stage_stamp_splits_the_correction_series(self, tmp_path):
        """Without ``memory.stage = STAGE_RETURN_HOME`` at the boundary,
        ``Correction.turn`` is stage-local and doc 06 §5.8's per-stage drift
        series cannot be split after the batch — nothing else records where the
        boundary was."""
        runner, _, log = make_runner(
            tmp_path,
            [
                turn(call("correct_position", x=1.0, y=1.0, reason="saw the rug")),
                turn(call(DECLARE_DONE)),
                turn(call("correct_position", x=2.0, y=2.0, reason="saw the sofa")),
                turn(call(DECLARE_DONE)),
            ],
        )
        runner.context.playback.set_true_xy(target_true_xy(0.0))
        runner.run()
        stages = [c.stage for c in runner.memory.corrections]
        assert stages == [STAGE_FIND_KITCHEN, STAGE_RETURN_HOME]
        # Both are turn 1 of their own stage — which is exactly why the stamp
        # is the only thing that can separate them.
        assert [c.turn for c in runner.memory.corrections] == [1, 1]
        logged = log.document["turns"][-1]["memory_snapshot"]["corrections"]
        assert [c["stage"] for c in logged] == [STAGE_FIND_KITCHEN, STAGE_RETURN_HOME]


class TestStageTwoGate:
    """T3.4's resolution of doc 05 §12 / doc 06 §12."""

    def test_the_rule_is_recorded_in_the_hashed_config(self):
        text = BENCHMARK_YAML.read_text()
        assert "stage2_requires_stage1_success: true" in text
        assert STAGE2_REQUIRES_STAGE1_SUCCESS is True

    def test_declaring_elsewhere_ends_the_trial_and_says_nothing_about_failing(
        self, tmp_path
    ):
        runner, provider, _ = make_runner(
            tmp_path, [turn(call(DECLARE_DONE))], qa_text="1. a"
        )
        runner.context.playback.set_true_xy(target_true_xy(2.0))
        final = runner.run()

        assert final["outcome"][STAGE_FIND_KITCHEN] == OUTCOME_DECLARED_ELSEWHERE
        assert final["outcome"][STAGE_RETURN_HOME] == OUTCOME_NOT_RUN
        assert final["stages"][STAGE_RETURN_HOME]["turns_used"] == 0
        assert final["stages"][STAGE_RETURN_HOME]["policy_seconds_used"] == 0.0

        # The declare_done result must be outcome-NEUTRAL: it says the trial
        # ended, never that the model was wrong. That is the mitigation for the
        # one thing this resolution costs (doc 05 §3.3 item (1) narrows to "the
        # only signal is whether the return leg is offered"). It lands in the
        # transcript rather than in a request, because the trial is over — on
        # this branch the model is never sent anything again, which bounds the
        # signal even further.
        answered = json.loads(runner.transcript[-1].results[0].text)
        assert answered["detail"] == TRIAL_OVER_DETAIL
        assert answered["stage_ended"] == STAGE_FIND_KITCHEN
        for word in ("fail", "wrong", "missed", "incorrect", "score"):
            assert word not in TRIAL_OVER_DETAIL.lower()
        sent = "\n".join(request_text(r) for r in provider.requests)
        assert STAGE2_OBJECTIVE_TOOL_RESULT not in sent

    def test_a_cap_out_does_not_run_the_return_leg(self, tmp_path):
        counters = Counters(turn_cap=2)
        runner, _, _ = make_runner(
            tmp_path,
            [turn(call("get_observation")), turn(call("get_observation"))],
            context=make_context(counters=counters),
        )
        final = runner.run()
        assert final["outcome"][STAGE_FIND_KITCHEN] == OUTCOME_TIMEOUT_TURNS
        assert final["outcome"][STAGE_RETURN_HOME] == OUTCOME_NOT_RUN

    def test_a_fall_does_not_run_the_return_leg(self, tmp_path):
        runner, _, _ = make_runner(tmp_path, [turn(call("move", distance_m=1.0))])
        runner.context.playback.move_falls = True
        final = runner.run()
        assert final["outcome"][STAGE_FIND_KITCHEN] == OUTCOME_FALL
        assert final["outcome"][STAGE_RETURN_HOME] == OUTCOME_NOT_RUN

    def test_a_zero_motion_return_home_success_is_geometrically_impossible(self):
        """The measured reason the gate is "succeeded" and not "declared".

        Under "any declare_done", a wrong declare inside the 0.5 m home disc
        would score ``return_home`` a success with zero motion — 25 percentage
        points of an N=4 SR. Under the committed gate, stage 2 always begins
        inside the 0.35 m counter disc, whose worst case is still far outside
        every spawn's return radius.
        """
        target = target_point()
        home_radius = LAYOUT["return_home_radius"]
        find_radius = LAYOUT["target"]["radius"]
        worst = min(
            math.dist(spawn_pose(seed)[0], target) - find_radius
            for seed in LAYOUT["spawn_points"]
        )
        assert worst > home_radius, (
            f"minimum stage-2 d_initial {worst:.3f} m must clear the "
            f"{home_radius} m return radius"
        )
        assert round(worst, 3) == 1.574


class TestSuccessPredicate:
    def test_the_boundary_is_inclusive(self):
        """doc 06 §3.1 says "within 0.35 m", so exactly 0.35 m succeeds. Tested
        on clean numbers because ``2.55 + 0.35`` is not exactly 2.9 in binary —
        an exactness test on the real target would be testing float layout, not
        the operator."""
        spec = StageSpec(
            name="unit", objective="", goal_xy=(0.0, 0.0),
            success_radius_m=1.0, goal_label="unit",
        )
        assert score_stage(spec, (1.0, 0.0)).success
        assert score_stage(spec, (0.0, -1.0)).success
        assert not score_stage(spec, (1.0000001, 0.0)).success

    def test_one_predicate_two_consumers(self):
        spec = find_kitchen_spec()
        assert score_stage(spec, target_true_xy(0.30)).success
        assert not score_stage(spec, target_true_xy(0.40)).success
        # The decision inputs travel with the verdict, so T4.1 recomputes it.
        score = score_stage(spec, target_true_xy(0.40))
        assert score.radius_m == spec.success_radius_m
        assert score.goal_xy == spec.goal_xy
        assert round(score.distance_m, 6) == 0.4

    def test_radii_agree_between_the_layout_and_the_frozen_config(self):
        """The layout dict is the ground truth (AGENTS.md §2); benchmark.yaml
        mirrors it for the freeze hash. Before T3.4 the two were duplicated with
        NO agreement test — a drift there would let a trial be logged
        ``find_kitchen: success`` (and run a stage 2) while T4.1's scorer, reading
        the other copy, published a failure."""
        text = BENCHMARK_YAML.read_text()
        assert f"find_kitchen_success_radius_m: {LAYOUT['target']['radius']}" in text
        assert f"return_home_success_radius_m: {LAYOUT['return_home_radius']}" in text
        assert find_kitchen_spec().success_radius_m == LAYOUT["target"]["radius"]
        assert (
            return_home_spec(SPAWN_XY).success_radius_m == LAYOUT["return_home_radius"]
        )

    def test_the_return_home_goal_is_this_seeds_spawn(self):
        for seed in LAYOUT["spawn_points"]:
            spawn, _ = spawn_pose(seed)
            assert return_home_spec(spawn).goal_xy == (float(spawn[0]), float(spawn[1]))

    def test_arriving_without_declaring_is_never_promoted_to_a_success(self):
        """doc 06 §3.1 requires BOTH; quietly promoting a capped stage that
        happened to stop on target would inflate the headline SR."""
        with pytest.raises(ValueError):
            outcome_for(REASON_TURN_CAP, True)
        assert outcome_for(REASON_TURN_CAP, False) == OUTCOME_TIMEOUT_TURNS
        assert outcome_for(REASON_MOTION_CAP, False) == OUTCOME_TIMEOUT_MOTION
        assert outcome_for(REASON_FALL, False) == OUTCOME_FALL
        assert outcome_for(REASON_NOT_RUN, False) == OUTCOME_NOT_RUN
        assert outcome_for(REASON_DECLARE_DONE, True) == OUTCOME_SUCCESS


# ===========================================================================
# 3. Caps — doc 05 §3.1 phase 4, doc 06 §3.2
# ===========================================================================


class TestCaps:
    def test_the_turn_cap_is_checked_after_execution_not_before(self, tmp_path):
        """Doc 05 §3.1: "Caps (checked after execution …)". The turn that trips
        the cap is fully executed and fully kept."""
        counters = Counters(turn_cap=3)
        runner, _, log = make_runner(
            tmp_path,
            [turn(call("update_room", name=f"r{i}", description="d")) for i in range(3)],
            context=make_context(counters=counters),
        )
        result = runner.run_stage(runner.stages[0])
        assert result.end_reason == REASON_TURN_CAP
        assert result.turns_used == 3
        assert len(log.document["turns"]) == 3
        assert set(runner.memory.rooms) == {"r0", "r1", "r2"}

    def test_every_model_turn_counts_including_wasted_ones(self, tmp_path):
        """Doc 05 §8: malformed calls and derailments consume budget — that is
        the point. ``state.turns += 1`` fires before any dispatch, so a refusal
        costs exactly as much as a good turn."""
        counters = Counters(turn_cap=4)
        runner, _, _ = make_runner(
            tmp_path,
            [
                turn(text="I would rather not.", stop_reason="refusal", refusal="refusal"),
                turn(call("no_such_tool")),
                turn(call("move", distance_m="not a number")),
                turn(text="thinking out loud"),
            ],
            context=make_context(counters=counters),
        )
        result = runner.run_stage(runner.stages[0])
        assert result.end_reason == REASON_TURN_CAP
        assert result.turns_used == 4

    def test_the_motion_cap_ends_the_stage(self, tmp_path):
        # One fake `move` spends 3 drive chunks + 1 settle chunk = 0.8 policy-s.
        counters = Counters(policy_seconds_cap=0.5)
        runner, _, _ = make_runner(
            tmp_path,
            [turn(call("move", distance_m=1.0))],
            context=make_context(counters=counters),
        )
        result = runner.run_stage(runner.stages[0])
        assert result.end_reason == REASON_MOTION_CAP
        assert result.policy_seconds_used >= 0.5

    def test_a_chained_turn_may_overshoot_the_motion_cap_and_it_is_logged(
        self, tmp_path
    ):
        """Doc-sanctioned: caps are checked after the WHOLE turn, so several
        motion tools in one turn can pass 240 s together. Doc 05 §12's open
        question (cap motion tools per turn?) is designated for T3.5's smoke,
        which can only answer it from data — hence ``execution.motion_calls``
        and the per-turn policy-seconds recorded here."""
        counters = Counters(policy_seconds_cap=1.0, turn_cap=7)
        runner, _, log = make_runner(
            tmp_path,
            [turn(call("move", "a", distance_m=1.0), call("move", "b", distance_m=1.0))],
            context=make_context(counters=counters),
        )
        result = runner.run_stage(runner.stages[0])
        record = log.document["turns"][0]
        assert record["execution"]["motion_calls"] == 2
        # 2 x 0.8 policy-s in ONE turn, against a 1.0 s cap checked afterwards.
        assert record["execution"]["policy_seconds_used"] > counters.policy_seconds_cap
        assert result.end_reason == REASON_MOTION_CAP
        assert result.policy_seconds_used > counters.policy_seconds_cap
        # BOTH cap columns come from the live Counters, not the module
        # constants: every smoke run constructs a non-default cap, and a budget
        # line reading `40` while the loop enforced 7 is evidence that
        # contradicts the run that produced it.
        assert record["budget"]["stage_policy_seconds_cap"] == 1.0
        assert record["budget"]["stage_turn_cap"] == counters.turn_cap == 7

    @pytest.mark.parametrize(
        "counters,script,expected",
        [
            (
                Counters(turn_cap=2),
                [turn(call("get_observation")), turn(call("get_observation"))],
                REASON_TURN_CAP,
            ),
            (
                Counters(policy_seconds_cap=0.5),
                [turn(call("move", distance_m=1.0))],
                REASON_MOTION_CAP,
            ),
        ],
    )
    def test_the_turn_that_ends_a_stage_by_cap_carries_the_reason(
        self, tmp_path, counters, script, expected
    ):
        """doc 06 §4 annotates `turns[].end_reason` as "§3.2's stop reason, on
        the turn that ends the stage", and §3.2 lists the two caps among its
        four stop conditions. They were null: the caps were evaluated after the
        record had already been flushed, so a cap-ended stage — roughly half the
        expected trials — had NO turn marked terminal, and any per-turn scorer
        locating the boundary by `end_reason` silently treated the last turn as
        an ordinary one."""
        runner, _, log = make_runner(
            tmp_path, script, context=make_context(counters=counters)
        )
        result = runner.run_stage(runner.stages[0])
        assert result.end_reason == expected
        assert log.document["turns"][-1]["end_reason"] == expected
        assert all(t["end_reason"] is None for t in log.document["turns"][:-1])

    def test_caps_come_from_counters_so_they_match_the_budget_line(self):
        """The model budgets against the numbers in its block; the loop must
        enforce the same objects, not a second copy of the constants."""
        counters = Counters()
        assert counters.turn_cap == TURN_CAP
        assert counters.policy_seconds_cap == POLICY_SECONDS_CAP

    def test_both_caps_apply_to_stage_two_as_well(self, tmp_path):
        counters = Counters(turn_cap=2)
        runner, _, _ = make_runner(
            tmp_path,
            [
                turn(call(DECLARE_DONE)),
                turn(call("get_observation")),
                turn(call("get_observation")),
            ],
            context=make_context(counters=counters),
        )
        runner.context.playback.set_true_xy(target_true_xy(0.0))
        final = runner.run()
        assert final["outcome"][STAGE_FIND_KITCHEN] == OUTCOME_SUCCESS
        assert final["outcome"][STAGE_RETURN_HOME] == OUTCOME_TIMEOUT_TURNS
        assert final["stages"][STAGE_RETURN_HOME]["turns_used"] == 2


# ===========================================================================
# 4. declare_done's transcript shape — doc 05 §3.1, §3.3 item (4), §4.4
# ===========================================================================


class TestDeclareDoneShape:
    def test_every_call_in_the_declaring_turn_is_answered(self, tmp_path):
        """Doc 05 §7.2 / §8: an unanswered ``tool_use`` block is an API error —
        i.e. an infra rerun of a trial the model actually finished."""
        runner, _, _ = make_runner(
            tmp_path,
            [
                turn(
                    call("update_room", "a", name="kitchen", description="tiles"),
                    call(DECLARE_DONE, "b"),
                    call("move", "c", distance_m=1.0),
                    call("look_around", "d"),
                )
            ],
        )
        runner.context.playback.set_true_xy(target_true_xy(0.0))
        runner.run_stage(runner.stages[0])
        blocks = runner.transcript[-1].results
        assert [b.tool_use_id for b in blocks] == ["a", "b", "c", "d"]
        # `dispatched` is the only field separating "the model emitted 4 calls"
        # from "the harness ran 1 and answered the rest not_executed" — which is
        # exactly what a declare_done mid-turn does.
        record = runner.log.document["turns"][0]["model_output"]
        assert len(record["tool_calls"]) == 4
        assert record["dispatched"] == 1
        # Positional: the real result, the stage outcome, then not_executed.
        assert json.loads(blocks[0].text)["ok"] is True
        assert json.loads(blocks[1].text)["stage_ended"] == STAGE_FIND_KITCHEN
        for block in blocks[2:]:
            payload = json.loads(block.text)
            assert payload["error"] == "not_executed"
            assert block.is_error is True

    def test_calls_after_declare_done_never_touch_the_sim(self, tmp_path):
        runner, _, _ = make_runner(
            tmp_path,
            [turn(call(DECLARE_DONE, "a"), call("move", "b", distance_m=1.0))],
        )
        runner.context.playback.set_true_xy(target_true_xy(0.0))
        runner.run_stage(runner.stages[0])
        assert runner.context.playback.calls == []

    def test_a_move_bundled_before_declare_done_counts_toward_the_verdict(
        self, tmp_path
    ):
        """Scored at the declare_done call's POSITION in the list, so the model
        is judged from where it actually ended up."""
        runner, _, _ = make_runner(
            tmp_path,
            [turn(call("move", "a", distance_m=1.0), call(DECLARE_DONE, "b"))],
        )
        playback = runner.context.playback
        playback.set_true_xy(target_true_xy(9.0))
        original_move = playback.move

        def move(distance_m, **kwargs):
            result = original_move(distance_m, **kwargs)
            playback.set_true_xy(target_true_xy(0.0))
            return result

        playback.move = move
        result = runner.run_stage(runner.stages[0])
        assert result.success is True

    def test_the_declaring_turn_is_kept_in_the_transcript(self, tmp_path):
        runner, _, _ = make_runner(
            tmp_path,
            [turn(call("update_plan", "a", text="done"), call(DECLARE_DONE, "b"))],
        )
        runner.context.playback.set_true_xy(target_true_xy(0.0))
        before = len(runner.transcript)
        runner.run_stage(runner.stages[0])
        assert len(runner.transcript) == before + 1


# ===========================================================================
# 5. The fall path — doc 05 §3.1, §4.1's residual
# ===========================================================================


class TestFall:
    def test_a_get_observation_after_the_falling_command_is_never_dispatched(
        self, tmp_path
    ):
        """Doc 05 §4.1's residual, assigned to T3.4 by name: ``dispatch`` refuses
        further MOTION after a fall, but perception still answers — so without
        the loop's check the trial's final logged frame would be rendered from
        the spawn point Isaac teleported the robot to."""
        runner, _, log = make_runner(
            tmp_path,
            [turn(call("move", "a", distance_m=1.0), call("get_observation", "b"))],
        )
        runner.context.playback.move_falls = True
        result = runner.run_stage(runner.stages[0])
        assert result.end_reason == REASON_FALL
        assert runner.context.camera.captures == 0
        assert log.document["turns"][0]["obs"]["frame_paths"] == []
        # ...and the log says so: 2 calls emitted, 1 executed.
        model_output = log.document["turns"][0]["model_output"]
        assert len(model_output["tool_calls"]) == 2
        assert model_output["dispatched"] == 1

    def test_the_fall_turn_is_dropped_from_the_transcript_but_logged(self, tmp_path):
        """Doc 05 §3.1 drops the provider transcript entry ("no further model
        calls, so the dropped turn is moot") — but doc 06 §4 still needs
        ``turns[].execution`` / ``pose_trace`` for it, and T4.1's scorer RAISES
        on a missing trace. The two artifacts are not the same artifact."""
        runner, _, log = make_runner(tmp_path, [turn(call("move", distance_m=1.0))])
        runner.context.playback.move_falls = True
        runner.run_stage(runner.stages[0])
        assert runner.transcript == []
        assert len(log.document["turns"]) == 1
        assert log.document["turns"][0]["execution"]["pose_trace"]
        assert log.document["turns"][0]["end_reason"] == REASON_FALL

    def test_the_end_pose_is_the_pre_teleport_one(self, tmp_path):
        """Isaac auto-resets a terminated env inside ``env.step()``. Reading
        ``true_xy()`` afterwards logs the SPAWN point as the fall location, which
        hands doc 06 §5.2's progress metric a free 1.0 on seed 101 and corrupts
        the trajectory figure."""
        runner, _, log = make_runner(tmp_path, [turn(call("move", distance_m=1.0))])
        runner.context.playback.move_falls = True
        result = runner.run_stage(runner.stages[0])
        assert result.true_pose == TRUE_POSE
        logged = log.document["turns"][0]["true_pose"]
        assert (logged["x"], logged["y"]) == TRUE_XY
        assert (result.score.true_xy) == TRUE_XY

    def test_the_qa_block_after_a_fall_shows_the_latched_compass(self, tmp_path):
        """``observed_compass_deg`` is the single implementation of the latch. A
        QA prompt built from the live sensor would show the spawn heading in
        every trial that ended in a fall."""
        runner, provider, _ = make_runner(
            tmp_path, [turn(call("move", distance_m=1.0))], qa_text="1. x"
        )
        playback = runner.context.playback
        playback.move_falls = True
        playback._compass = 200.0
        runner.run()
        qa_request = [r for r in provider.requests if r["system"] == QA_SYSTEM_PROMPT][0]
        text = qa_request["messages"][0].blocks[0].text
        assert "Compass heading: 200 deg" in text
        assert f"Compass heading: {int(playback.spawn_compass_deg)} deg" not in text


# ===========================================================================
# 6. Doc 05 §8 error routing — model-attributable vs infra
# ===========================================================================


class TestErrorPolicy:
    @pytest.mark.parametrize(
        "bad_call,expected",
        [
            (call("no_such_tool"), "unknown_tool"),
            (call("move", distance_m="banana"), "invalid_args"),
            (call("move"), "invalid_args"),
            (ToolCall(id="x", name="move", args={}, parse_error="bad json"), "invalid_args"),
        ],
    )
    def test_a_model_attributable_fault_is_a_scored_result_not_an_exception(
        self, tmp_path, bad_call, expected
    ):
        """Doc 05 §8's first row. An escaping exception would be classified as an
        infra fault and rerun the trial WHOLE — handing a malformed call the free
        retry §8's agency line exists to prevent."""
        runner, _, log = make_runner(tmp_path, [turn(bad_call), turn(call(DECLARE_DONE))])
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        runner.run_stage(runner.stages[0])
        block = runner.transcript[0].results[0]
        assert json.loads(block.text)["error"] == expected
        assert block.is_error is True
        assert log.document["turns"][0]["execution"]["motion_calls"] == 0

    def test_a_bad_call_still_burns_a_turn(self, tmp_path):
        runner, _, _ = make_runner(
            tmp_path, [turn(call("no_such_tool")), turn(call(DECLARE_DONE))]
        )
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        result = runner.run_stage(runner.stages[0])
        assert result.turns_used == 2

    def test_a_turn_with_no_tool_call_gets_the_fixed_nudge_and_continues(
        self, tmp_path
    ):
        """Doc 05 §8's derailment row. §3.1's pseudocode has NO branch for this —
        as literally written it would append an assistant turn with no
        ``tool_use`` plus an EMPTY user message, which is an API error. §3.1 is
        amended in the same commit."""
        runner, provider, log = make_runner(
            tmp_path,
            [turn(text="Let me think about this."), turn(call(DECLARE_DONE))],
        )
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        runner.run_stage(runner.stages[0])
        assert runner.transcript[0].note == DERAILMENT_NUDGE
        assert log.document["turns"][0]["model_output"]["nudged"] is True
        # And the model actually receives it on the next request.
        second = provider.requests[1]
        assert DERAILMENT_NUDGE in request_text(second)

    def test_a_refusal_with_empty_native_content_is_scored_not_retried(self, tmp_path):
        """The shape the guard above never exercised: `turn()`'s fixture always
        sets a non-empty `raw`, but an Anthropic refusal is HTTP 200 with an
        EMPTY `content` array, so `raw` is `[]`.

        What must hold is that the refusal stays on doc 05 §8's DERAILMENT
        path: "scored as a failure … never selectively retried". The failure
        mode is a 400 anywhere in the next request, which
        `scripts/run_trial.py` catches at the trial boundary and records as an
        infra failure with no `final` — so doc 06 §9.1's resume check would
        rerun a trial the model actually failed, which is selection bias in
        that model's favour.

        Measured 2026-07-26: the empty ASSISTANT turn is not itself a 400
        (it is accepted, even with tools + adaptive thinking). The 400 came
        from the empty USER message the missing derailment branch appended.
        Both are covered here. It was also asymmetric — harmless on the OpenAI
        adapter — so the same behaviour cost only the Anthropic contestants.
        """
        from duck_embody.agent.providers.anthropic import AnthropicProvider
        from duck_embody.agent.providers.base import ModelConfig
        from duck_embody.agent.providers.openai import OpenAIProvider

        refusal = AssistantTurn(
            text="",
            tool_calls=[],
            usage=Usage(input_tokens=100, output_tokens=0),
            raw=[],  # the real wire shape of a pre-output decline
            stop_reason="refusal",
            refusal="refusal (category=cyber)",
        )
        runner, provider, log = make_runner(
            tmp_path, [refusal, turn(call(DECLARE_DONE))]
        )
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        result = runner.run_stage(runner.stages[0])
        assert result.end_reason == REASON_DECLARE_DONE  # scored, not rerun
        assert runner.transcript[0].note == DERAILMENT_NUDGE
        assert log.document["turns"][0]["model_output"]["nudged"] is True

        cfg = ModelConfig(
            name="t", provider="anthropic", model_id="claude-fable-5",
            max_tokens=8, price_in_per_mtok=1.0, price_out_per_mtok=1.0,
        )
        adapter = AnthropicProvider.__new__(AnthropicProvider)
        adapter.cfg = cfg
        second = provider.requests[1]["messages"]
        body = adapter.to_native(second)
        assert body, "the second request must still carry the nudge"
        assert all(m["content"] for m in body), (
            f"an empty-content message reached the Anthropic body: {body}"
        )
        # ...and the nudge the model must actually read survived the drop.
        assert DERAILMENT_NUDGE in json.dumps(body, ensure_ascii=False)
        # The OpenAI adapter was always unaffected; assert the symmetry holds.
        openai = OpenAIProvider.__new__(OpenAIProvider)
        assert DERAILMENT_NUDGE in json.dumps(
            openai.to_native(second), ensure_ascii=False
        )

    def test_a_refusal_is_treated_as_a_derailment_not_an_infra_fault(self, tmp_path):
        runner, _, log = make_runner(
            tmp_path,
            [
                turn(text="", stop_reason="refusal", refusal="refusal (category=x)"),
                turn(call(DECLARE_DONE)),
            ],
        )
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        result = runner.run_stage(runner.stages[0])
        assert result.end_reason == REASON_DECLARE_DONE
        assert log.document["turns"][0]["model_output"]["refusal"] == "refusal (category=x)"
        assert runner.transcript[0].note == DERAILMENT_NUDGE

    def test_a_render_failure_propagates(self, tmp_path):
        """Doc 05 §8's last row + its implementation note: faults with no model
        agency are deliberately NOT caught. Swallowing a render error would
        launder a broken GPU into a model failure."""
        runner, _, _ = make_runner(
            tmp_path,
            [turn(call("get_observation"))],
            context=make_context(camera=ExplodingCamera()),
        )
        with pytest.raises(RuntimeError, match="RTX render failed"):
            runner.run_stage(runner.stages[0])

    def test_an_exhausted_api_retry_propagates(self, tmp_path):
        log = make_log(tmp_path)
        runner = EpisodeRunner(
            provider=ExplodingProvider(),
            context=make_context(),
            stages=stage_specs(SEED),
            log=log,
        )
        with pytest.raises(ConnectionError):
            runner.run()
        assert "final" not in log.document, (
            "an infra-failed trial must stay incomplete so doc 06 §9.1's resume "
            "check rejects it and the trial reruns whole"
        )

    def test_an_infra_failure_note_does_not_create_a_final_block(self, tmp_path):
        log = make_log(tmp_path)
        log.note_infra_failure("traceback here")
        on_disk = json.loads(log.path.read_text())
        assert on_disk["infra_failure"] == "traceback here"
        assert "final" not in on_disk

    def test_a_model_supplied_non_finite_argument_cannot_crash_the_log(self, tmp_path):
        """``json.loads`` accepts the bare ``NaN``/``Infinity`` literals, so a
        model CAN emit one. ``dispatch`` rejects it as ``invalid_args``, but
        writing it into ``model_output.tool_calls`` with ``allow_nan=False``
        would raise — crashing the harness onto doc 05 §8's INFRA path for a
        fault whose agency is entirely the model's, i.e. §8's line crossed
        backwards."""
        runner, _, log = make_runner(
            tmp_path,
            [turn(call("move", distance_m=float("nan")))],
            context=make_context(counters=Counters(turn_cap=1)),
        )
        runner.run_stage(runner.stages[0])
        logged = log.document["turns"][0]["model_output"]["tool_calls"][0]
        assert logged["args"]["distance_m"] == "nan"
        json.loads(log.path.read_text())  # parses; no bare NaN token on disk
        assert "NaN" not in log.path.read_text()


# ===========================================================================
# 7. The benchmark's validity — no ground truth may reach the model
# ===========================================================================


class TestNoGroundTruthReachesTheModel:
    #: Rounded and derived renderings of the planted sentinels. The exact-value
    #: sweep below is necessary but NOT sufficient: two injected leaks —
    #: appending the true pose to the memory block at 1 dp, and appending a
    #: `math.dist(true_pose, goal)` range-to-goal oracle — both passed the whole
    #: suite, because `str(7.77) not in sent` says nothing about "7.8".
    ROUNDED_SENTINELS = ("7.8", "7.77", "-8.9", "-8.88", "123.4", "123.5", "9.99")

    def test_the_trailing_user_message_is_byte_equal_to_the_rendered_block(
        self, tmp_path
    ):
        """STRUCTURAL, not value-exact: nothing may be appended to the block.

        A sentinel sweep can only find leaks it was told to look for; this
        asserts the model-facing text is re-assemblable from the frozen pieces,
        so ANY extra byte — rounded ground truth, a derived oracle, a debug
        marker — fails regardless of what it says. The block is re-rendered
        inside `send`, where the memory state is exactly the state phase 1
        rendered from (nothing mutates between the two), so equality is a real
        comparison rather than a tautology.
        """
        context = make_context()
        checked = []

        class Checking(FakeProvider):
            def send(self, system, messages, tools):
                if system == SYSTEM_PROMPT:
                    expected = render_memory_block(
                        context.memory,
                        context.counters,
                        context.integrator.xy,
                        observed_compass_deg(context),
                    )
                    last = messages[-1]
                    assert isinstance(last, UserMessage)
                    assert len(last.blocks) == 1
                    assert isinstance(last.blocks[0], TextBlock)
                    assert last.blocks[0].text == expected, (
                        "the trailing user message must be the rendered memory "
                        "block and nothing else"
                    )
                    checked.append(len(messages))
                return super().send(system, messages, tools)

        provider = Checking(
            [
                turn(call("get_observation")),
                turn(call("update_room", name="start", description="sofa"),
                     call("move", distance_m=1.0)),
                turn(call("correct_position", x=1.0, y=2.0, reason="the rug")),
                turn(text="thinking out loud"),
                turn(call(DECLARE_DONE)),
                turn(call("look_around")),
                turn(call(DECLARE_DONE)),
            ],
            qa_text="1. a\n2. b\n3. c\n4. d\n5. e",
        )
        log = make_log(tmp_path)
        context.playback.set_true_xy(target_true_xy(0.0))
        runner = EpisodeRunner(
            provider=provider, context=context, stages=stage_specs(SEED), log=log
        )
        runner.run()
        assert len(checked) == 7

    def test_every_model_facing_part_is_re_assemblable_from_the_frozen_pieces(
        self, tmp_path
    ):
        """The other half of the structural guard: the system string, the tool
        schemas and every free-text block a request carries are exactly the
        frozen artifacts — a leak cannot hide in an extra text block either."""
        runner, provider, _ = make_runner(
            tmp_path,
            [
                turn(call("get_observation")),
                turn(text="no tools this turn"),
                turn(call("move", distance_m=1.0)),
                turn(call(DECLARE_DONE)),
            ],
            qa_text="1. a\n2. b\n3. c\n4. d\n5. e",
        )
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        runner.run()
        driving = [r for r in provider.requests if r["system"] == SYSTEM_PROMPT]
        assert driving
        for request in driving:
            assert request["tools"] == TOOL_SCHEMAS
            for block in user_blocks(request["messages"]):
                if isinstance(block, TextBlock):
                    assert block.text == DERAILMENT_NUDGE or block.text.startswith(
                        "== YOUR MAP"
                    ), f"unexpected free text in a request: {block.text[:120]!r}"
                elif isinstance(block, ToolResultBlock):
                    # Built by ToolOutcome.to_block, whose keys tests/test_tools.py
                    # pins against doc 05 §4's frozen payload key by key.
                    json.loads(block.text)

    def test_no_sentinel_ground_truth_appears_in_any_request(self, tmp_path):
        runner, provider, _ = make_runner(
            tmp_path,
            [
                turn(call("get_observation")),
                turn(call("move", distance_m=1.0)),
                turn(call("look_around")),
                turn(call("send_velocity", vx=0.1, vy=0.0, wz=0.2, duration_s=1.0)),
                turn(call("turn_to_heading", heading_deg=180)),
                turn(call(DECLARE_DONE)),      # stage 1 succeeds -> stage 2 runs
                turn(call("get_observation")),
                turn(call(DECLARE_DONE)),
            ],
            qa_text="1. a 2. b 3. c 4. d 5. e",
        )
        runner.context.playback.set_true_xy(target_true_xy(0.0))
        runner.run()
        sent = "\n".join(request_text(r) for r in provider.requests)
        for value in (TRUE_XY[0], TRUE_XY[1], TRUE_DISPLACEMENT_M, TRUE_HEADING):
            assert str(value) not in sent, f"ground truth {value} leaked"
        for point in POSE_TRACE_SENTINEL:
            assert str(point[0]) not in sent
        # ...and their ROUNDED forms, which an exact-substring sweep misses: a
        # leak rendered "%.1f" is still the benchmark's answer key.
        for rounded in self.ROUNDED_SENTINELS:
            assert not re.search(rf"(?<![\d.]){re.escape(rounded)}(?![\d])", sent), (
                f"a rounded form of the ground truth ({rounded}) leaked"
            )

    def test_no_layout_geometry_or_room_label_reaches_the_model(self, tmp_path):
        """Doc 05 §1 names "real room labels" as ground-truth injection. Three of
        the four true room names must appear nowhere; ``kitchen`` is the task
        statement, not a layout label."""
        runner, provider, _ = make_runner(
            tmp_path,
            [
                turn(call("get_observation")),
                turn(call(DECLARE_DONE)),
                turn(call("look_around")),
                turn(call(DECLARE_DONE)),
            ],
            qa_text="1. a",
        )
        runner.context.playback.set_true_xy(target_true_xy(0.0))
        runner.run()
        # DRIVING requests only. The post-episode QA questions are frozen doc 06
        # §5.9 text and DO name real rooms ("Which room connects the bedroom to
        # the kitchen?") — that is the probe, asked after the episode has ended,
        # with no sim to act on. The boundary that matters is that no room label
        # reaches the model while it is still navigating.
        sent = "\n".join(
            request_text(r) for r in provider.requests if r["system"] == SYSTEM_PROMPT
        )
        for room in ("bedroom", "hallway", "living_room", "living room"):
            assert room not in sent.lower()
        target = target_point()
        assert f"{target[0]}" not in sent and f"{target[1]}" not in sent
        # ...and the QA request is the only place any of them may appear.
        qa = [r for r in provider.requests if r["system"] == QA_SYSTEM_PROMPT]
        assert len(qa) == 1

    def test_the_scoring_channel_never_becomes_a_tool_result(self, tmp_path):
        runner, provider, log = make_runner(
            tmp_path,
            [
                turn(call("move", distance_m=1.0)),
                turn(call(DECLARE_DONE)),
                turn(call("move", distance_m=1.0)),
                turn(call(DECLARE_DONE)),
            ],
            qa_text="1. a",
        )
        runner.context.playback.set_true_xy(target_true_xy(0.0))
        runner.run()
        sent = "\n".join(request_text(r) for r in provider.requests)
        for key in ("pose_trace", "sampled_xy", "true_pose", "true_displacement_m"):
            assert key not in sent
        # ...but it IS in the log, which is what T4.1 reads.
        assert log.document["turns"][0]["execution"]["pose_trace"]

    def test_the_qa_exchange_shows_only_the_models_own_block(self, tmp_path):
        runner, provider, _ = make_runner(
            tmp_path, [turn(call(DECLARE_DONE))], qa_text="1. a"
        )
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        runner.run()
        qa_request = [r for r in provider.requests if r["system"] == QA_SYSTEM_PROMPT][0]
        assert qa_request["tools"] == QA_TOOLS == []
        assert len(qa_request["messages"]) == 1
        text = qa_request["messages"][0].blocks[0].text
        assert text.startswith(LAYOUT_QA_PREAMBLE)
        assert "== YOUR MAP" in text
        for value in (TRUE_XY[0], TRUE_HEADING, target_point()[0]):
            assert str(value) not in text


# ===========================================================================
# 8. The QA exchange — doc 06 §5.9, §4's final.qa
# ===========================================================================


class TestQaExchange:
    ANSWER_BLOB = (
        "1. The hallway connects them.\n"
        "2. Turn left, walk 2 m, then right into the tiled room.\n"
        "3. I visited 3 rooms: start_room, corridor, tiled_room.\n"
        "4. North-east.\n"
        "5. start_room: sofa. corridor: plant. tiled_room: counter.\n"
    )

    def test_it_fires_after_a_success_and_lands_in_final_qa(self, tmp_path):
        runner, _, log = make_runner(
            tmp_path,
            [turn(call(DECLARE_DONE)), turn(call(DECLARE_DONE))],
            qa_text=self.ANSWER_BLOB,
        )
        runner.context.playback.set_true_xy(target_true_xy(0.0))
        final = runner.run()
        log.finish(final)
        qa = json.loads(log.path.read_text())["final"]["qa"]
        assert len(qa) == 5
        assert [q["question"] for q in qa] == [q.text for q in LAYOUT_QA_QUESTIONS]
        assert qa[0]["answer"] == "The hallway connects them."
        assert qa[3]["answer"] == "North-east."
        # `score` is T4.1's output — explicitly null, never guessed here.
        assert all(q["score"] is None for q in qa)

    @pytest.mark.parametrize(
        "script,label",
        [
            ([turn(call("move", distance_m=1.0))], "fall"),
            ([turn(call("get_observation"))], "cap"),
        ],
    )
    def test_it_fires_after_a_failure_too(self, tmp_path, script, label):
        """§5.9 says "after the episode ends" — a cap-out and a fall both are.
        Skipping the failures would drop a headline metric for exactly the trials
        most worth explaining, and bias the aggregate toward finishers."""
        counters = Counters(turn_cap=1)
        runner, _, _ = make_runner(
            tmp_path, script, qa_text=self.ANSWER_BLOB,
            context=make_context(counters=counters),
        )
        if label == "fall":
            runner.context.playback.move_falls = True
        final = runner.run()
        assert final["outcome"][STAGE_FIND_KITCHEN] in (OUTCOME_FALL, OUTCOME_TIMEOUT_TURNS)
        assert sum(1 for q in final["qa"] if q["answer"]) == 5

    def test_the_raw_blob_is_kept_so_a_scorer_can_redo_the_split(self, tmp_path):
        runner, _, _ = make_runner(
            tmp_path, [turn(call(DECLARE_DONE))], qa_text=self.ANSWER_BLOB
        )
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        final = runner.run()
        assert final["qa_raw"] == self.ANSWER_BLOB

    def test_an_unsplittable_reply_raises_the_parse_flag(self, tmp_path):
        """T4.1 scores an empty answer 0, which is indistinguishable from a bad
        map — so a formatting mismatch must be visible in `final`, not only
        recoverable from `qa_raw` by somebody who happened to notice."""
        runner, _, _ = make_runner(
            tmp_path, [turn(call(DECLARE_DONE))],
            qa_text="I'd rather describe it in prose: the tiled room was north.",
        )
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        final = runner.run()
        assert final["qa_parse_failed"] is True
        assert all(q["answer"] == "" for q in final["qa"])
        assert final["qa_raw"].startswith("I'd rather")

    def test_a_fully_parsed_reply_leaves_the_flag_down(self, tmp_path):
        runner, _, _ = make_runner(
            tmp_path, [turn(call(DECLARE_DONE))], qa_text=self.ANSWER_BLOB
        )
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        assert runner.run()["qa_parse_failed"] is False

    def test_qa_usage_is_accounted_separately_and_in_the_total(self, tmp_path):
        runner, _, _ = make_runner(
            tmp_path,
            [turn(call("get_observation")), turn(call(DECLARE_DONE))],
            qa_text=self.ANSWER_BLOB,
        )
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        final = runner.run()
        episode = final["tokens_breakdown"]["episode"]
        qa = final["tokens_breakdown"]["qa"]
        assert episode["input_tokens"] == 200 and qa["input_tokens"] == 500
        assert final["tokens"]["input_tokens"] == 700
        assert final["tokens"]["output_tokens"] == qa["output_tokens"] + episode["output_tokens"]


class TestQaAnswerSplitting:
    def test_plain_numbering(self):
        answers = split_qa_answers("1. one\n2. two\n3. three\n4. four\n5. five")
        assert answers == {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    @pytest.mark.parametrize("marker", ["1.", "1)", "**1.**", "- 1:", "  1 ."])
    def test_common_decorations(self, marker):
        answers = split_qa_answers(f"{marker} alpha\n2. beta")
        assert answers[1] == "alpha"
        assert answers[2] == "beta"

    def test_a_digit_inside_prose_cannot_steal_the_boundary(self):
        """Markers are searched in ascending order, each after the previous, and
        anchored to the start of a line."""
        text = "1. I walked 2. something like 2 m.\n2. Then I turned."
        answers = split_qa_answers(text)
        assert answers[1] == "I walked 2. something like 2 m."
        assert answers[2] == "Then I turned."

    def test_a_missing_answer_is_empty_not_invented(self):
        answers = split_qa_answers("1. one\n3. three")
        assert answers[1] == "one"
        assert answers[2] == ""
        assert answers[3] == "three"
        assert answers[4] == answers[5] == ""

    def test_empty_text(self):
        assert split_qa_answers("") == {n: "" for n in range(1, 6)}

    def test_multiline_answers_are_kept_whole(self):
        text = "1. First line\n   continued here\n2. Second"
        assert split_qa_answers(text)[1] == "First line\n   continued here"

    def test_each_marker_is_searched_after_the_previous_one(self):
        """The stated invariant, pinned against a naive per-marker search.

        Replacing `search(text, cursor)` with `search(text)` left the suite
        green: the prose-digit case above passes either way because its stray
        "2." is mid-line. On an out-of-order reply the mutant scores answer 1's
        text as answer 2 — so the shipped behaviour (answer 2 empty rather than
        swallowing answer 1) is asserted here explicitly."""
        text = "2. I turned right.\n1. The hallway connects them.\n3. Three rooms."
        answers = split_qa_answers(text)
        assert answers[1] == "The hallway connects them."
        assert answers[2] == ""
        assert answers[3] == "Three rooms."

    @pytest.mark.parametrize("marker", ["Q1.", "Q 1)", "**Question 1:**", "Question 1."])
    def test_a_question_prefixed_marker_is_recognised(self, marker):
        """Measured on the pre-fix matcher: `**Question 1:** …` matched NO
        marker, so all five answers came back "" and T4.1 scored the trial 0/5
        for a formatting reason rather than a map-quality one."""
        answers = split_qa_answers(f"{marker} alpha\n2. beta")
        assert answers[1] == "alpha"
        assert answers[2] == "beta"

    def test_a_nested_numbered_list_cannot_steal_a_boundary(self):
        """doc 06 §5.9 claimed this could not happen; measured, it did — the
        sub-list's indented "2." won because it came first. A column-0 match now
        wins over an indented one."""
        text = (
            "1. Rooms:\n"
            "   1. living\n"
            "   2. kitchen\n"
            "2. two\n3. three\n4. four\n5. five"
        )
        answers = split_qa_answers(text)
        assert answers[1] == "Rooms:\n   1. living\n   2. kitchen"
        assert answers[2] == "two"
        assert answers[5] == "five"

    def test_all_five_answers_on_one_line_still_split(self):
        """A per-model penalty otherwise: formatting habits differ between the
        three contestants, and the shipped matcher scored this 1/5 — 0.8 of a
        published metric lost to whitespace."""
        answers = split_qa_answers("1. hallway 2. left 3. two rooms 4. NE 5. sofa")
        assert answers == {
            1: "hallway", 2: "left", 3: "two rooms", 4: "NE", 5: "sofa",
        }

    def test_the_inline_fallback_does_not_fire_on_a_well_formed_reply(self):
        """It only runs when the strict pass found fewer than two markers, so a
        prose digit inside a normally-numbered reply still cannot steal."""
        text = "1. I walked 2. something like 2 m.\n2. Then I turned."
        assert split_qa_answers(text)[1] == "I walked 2. something like 2 m."


# ===========================================================================
# 9. doc 06 §4's log schema
# ===========================================================================


REQUIRED_TURN_KEYS = {
    "stage",
    "turn_idx",
    "timestamp",
    "obs",
    "model_output",
    "execution",
    "true_pose",
    "memory_snapshot",
}


class TestTrialLogFramesHygiene:
    """A trial's frames dir is cleared at TrialLog init, never merged into.

    `mkdir(exist_ok=True)` let every attempt at a trial_id ACCUMULATE into one
    directory: fable5_seed101 mixed frames from >=3 T3.5 attempts, with a
    deleted attempt's JSON referencing filenames whose bytes a later attempt
    had overwritten. A T4.2 resume that reruns an incomplete trial would
    contaminate the rerun's frame evidence the same way (gap G6)."""

    def test_a_previous_attempts_frames_are_cleared_at_init(self, tmp_path):
        stale_dir = tmp_path / "frames" / "fake_seed101"
        stale_dir.mkdir(parents=True)
        stale = stale_dir / "t007_0.jpg"
        stale.write_bytes(b"previous attempt's bytes")

        log = make_log(tmp_path)

        assert log.frames_dir == stale_dir
        assert log.frames_dir.is_dir()
        assert not stale.exists(), "stale frame survived into the new attempt"
        assert list(log.frames_dir.iterdir()) == []

    def test_frames_written_by_this_attempt_survive(self, tmp_path):
        """Positive control: the clear happens at INIT only, never later."""
        log = make_log(tmp_path)
        paths = log.save_frames(1, [ImageBlock(data_b64=base64.b64encode(b"x").decode())])
        assert paths == ["frames/fake_seed101/t001_0.jpg"]
        assert (log.frames_dir / "t001_0.jpg").read_bytes() == b"x"


class TestTrialLogStrictJson:
    """G12: a physics NaN in a scoring-only field must RAISE at flush (doc 05
    §8's infra path) rather than silently write the non-RFC `NaN` token — a
    strict parser would otherwise crash at scoring time, weeks after the
    batch. Model-authored NaN is already neutralised by `_json_safe`, so
    strictness cannot misfire on a model fault."""

    def test_a_physics_nan_in_a_scoring_field_raises_at_flush(self, tmp_path):
        log = make_log(tmp_path)
        with pytest.raises(ValueError):
            log.append_turn({"execution": {"pose_trace": [[float("nan"), 0.0]]}})
        # The file on disk is still the last GOOD state — parseable, no final —
        # which doc 06 §9.1's resume check treats as an infra rerun.
        document = json.loads(log.path.read_text())
        assert document["turns"] == []
        assert "final" not in document

    def test_a_model_authored_nan_still_logs_as_a_string(self, tmp_path):
        """The other half of §8's agency line: the model's own non-finite
        argument is `_json_safe`d into its repr and the turn records fine."""
        log = make_log(tmp_path)
        record = {
            "model_output": {
                "tool_calls": [{"name": "move", "args": _json_safe({"distance_m": float("nan")})}]
            }
        }
        log.append_turn(record)
        document = json.loads(log.path.read_text())
        assert (
            document["turns"][0]["model_output"]["tool_calls"][0]["args"]["distance_m"]
            == "nan"
        )


class TestTrialLogSchema:
    @pytest.fixture
    def document(self, tmp_path):
        """One whole trial, exercising every field doc 06 §4's block names.

        The bundled memory writes are not decoration: ``corrections[]`` only
        appears in the log if the episode makes one, and §5.8 needs its
        magnitude, so a fixture without a ``correct_position`` cannot notice
        that ``old_xy``/``new_xy`` stopped being written. Calls are bundled into
        the existing turns rather than added as new ones, so the turn indices
        the neighbouring tests pin stay put.
        """
        context = make_context()
        provider = FakeProvider(
            [
                turn(call("get_observation"), thinking="looking around"),
                turn(
                    call("update_room", "a", name="start", description="sofa"),
                    call("add_landmark", "b", room="start", description="a red rug"),
                    call("mark_exit", "c", room="start", direction_deg=90,
                         status="unexplored"),
                    call("correct_position", "d", x=1.0, y=2.0, reason="the rug"),
                    call("move", "e", distance_m=1.0),
                ),
                turn(call(DECLARE_DONE)),
                turn(call("look_around", "f"), call("move", "g", distance_m=1.0)),
                turn(call(DECLARE_DONE)),
            ],
            qa_text=TestQaExchange.ANSWER_BLOB,
        )
        log = make_log(tmp_path)
        context.playback.set_true_xy(target_true_xy(0.0))
        return run_trial(
            provider=provider, context=context, stages=stage_specs(SEED), log=log
        )

    def test_top_level_shape(self, document):
        assert set(document) >= {"trial_id", "config", "turns", "final", "video_path"}
        assert document["trial_id"] == "fake_seed101"
        config = document["config"]
        assert set(config) >= {"freeze_commit", "config_hash", "model", "seed", "spawn"}
        assert config["seed"] == SEED
        assert config["spawn"] == {
            "xy": [SPAWN_XY[0], SPAWN_XY[1]],
            "heading_deg": SPAWN_HEADING,
        }
        assert re.fullmatch(r"[0-9a-f]{64}", config["config_hash"])

    def test_every_key_path_doc_06_s4_documents_exists_in_the_log(self, document):
        """The schema golden is EXTRACTED from doc 06 §4, not transcribed here.

        ``REQUIRED_TURN_KEYS`` below is a partial hand-copy, and a partial copy
        polices only what somebody remembered to copy: deleting ``usage``,
        ``memory_snapshot.current_room``, ``corrections[].old_xy/new_xy``,
        ``execution.calls[].pose_trace``, ``stages[].true_pose`` or
        ``video_path`` each left the whole suite green — six fields T4.1 needs,
        discovered after the paid batch. Path-aware, so a field is not excused
        by a same-named sibling elsewhere in the document.
        """
        documented = doc_06_key_paths()
        assert len(documented) > 60, "the extractor stopped seeing §4's block"
        missing = sorted(p for p in documented if not has_key_path(document, p))
        assert not missing, f"doc 06 §4 documents fields the log does not write: {missing}"

    def test_the_per_turn_usage_is_recorded_for_doc_06_s8_cost_accounting(self, document):
        for record in document["turns"]:
            assert set(record["usage"]) == {
                "input_tokens", "output_tokens", "cache_read_tokens",
                "cache_write_tokens", "cost_usd_estimate",
            }
            assert record["usage"]["input_tokens"] == 100

    def test_the_video_path_key_is_always_present(self, document, tmp_path):
        """Rule-11's evidence link. Absent (rather than null) would make an
        un-recorded trial indistinguishable from a mis-written one."""
        assert "video_path" in document
        log = make_log(tmp_path, "video_seed101")
        log.set_video("results/videos/x.mp4")
        assert json.loads(log.path.read_text())["video_path"] == "results/videos/x.mp4"

    def test_every_turn_of_both_stages_has_the_required_keys(self, document):
        stages = {t["stage"] for t in document["turns"]}
        assert stages == {STAGE_FIND_KITCHEN, STAGE_RETURN_HOME}
        for record in document["turns"]:
            assert REQUIRED_TURN_KEYS <= set(record)
            assert set(record["obs"]) == {
                "frame_paths", "compass_deg", "position_estimate", "status",
            }
            assert set(record["obs"]["position_estimate"]) == {"x", "y"}
            assert set(record["obs"]["status"]) == {
                "bumped", "contact", "fell", "distance_moved_m",
            }
            assert set(record["true_pose"]) == {"x", "y", "heading_deg"}
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", record["timestamp"])

    def test_execution_is_always_an_object_with_a_pose_trace(self, document):
        """T4.1's scorer RAISES on a missing ``pose_trace`` rather than falling
        back to per-turn chords (which would shrink p and inflate SPL). So a
        turn that stepped no physics must carry an EMPTY trace, never a null or
        an absent key that is indistinguishable from a dropped one."""
        for record in document["turns"]:
            execution = record["execution"]
            assert isinstance(execution, dict)
            assert set(execution) >= {
                "result", "policy_seconds_used", "pose_trace", "motion_calls", "calls",
            }
            assert isinstance(execution["pose_trace"], list)
            if execution["motion_calls"] == 0:
                assert execution["pose_trace"] == []
                assert execution["policy_seconds_used"] == 0.0

    def test_the_bump_metric_has_a_per_turn_source(self, document):
        """doc 06 §5.6's two-source rule survives only in ``counted_as_bump``;
        §4 as written gave the scorer no per-turn source for a headline metric."""
        motion = [
            c for t in document["turns"] for c in t["execution"]["calls"]
        ]
        assert motion, "the fixture must exercise at least one motion tool"
        for record in motion:
            assert set(record) >= {
                "tool", "stop_reason", "bumped", "counted_as_bump", "fell",
                "distance_moved_m", "true_pose", "sampled_xy",
                # `pose_trace` and `policy_seconds_used` are the two the merge
                # would otherwise hide: the merged block keeps a concatenated
                # trace, so stripping them from the PER-CALL records is
                # invisible at the turn level — and attributing a trace to a
                # call within a chained turn is exactly the data doc 05 §12's
                # open question (cap motion tools per turn?) is designated to
                # answer from T3.5's smoke.
                "pose_trace", "policy_seconds_used",
            }
        assert isinstance(document["final"]["bumps"], int)

    def test_the_merged_trace_is_exactly_its_calls_concatenated(self, document):
        """So the merge can neither drop a call's trace nor double-count one —
        either would move doc 06 §5.3's SPL path integral."""
        for record in document["turns"]:
            execution = record["execution"]
            assert sum(len(c["pose_trace"]) for c in execution["calls"]) == len(
                execution["pose_trace"]
            )
            assert round(
                sum(c["policy_seconds_used"] for c in execution["calls"]), 4
            ) == execution["policy_seconds_used"]

    def test_turn_idx_is_stage_local_and_joins_the_correction_log(self, document):
        by_stage: dict[str, list[int]] = {}
        for record in document["turns"]:
            by_stage.setdefault(record["stage"], []).append(record["turn_idx"])
        assert by_stage[STAGE_FIND_KITCHEN] == [1, 2, 3]
        assert by_stage[STAGE_RETURN_HOME] == [1, 2]
        globals_ = [t["global_turn_idx"] for t in document["turns"]]
        assert globals_ == [1, 2, 3, 4, 5]

    def test_frames_are_written_and_referenced_relatively(self, document, tmp_path):
        paths = [p for t in document["turns"] for p in t["obs"]["frame_paths"]]
        assert paths, "the fixture calls get_observation"
        for relative in paths:
            assert relative.startswith("frames/fake_seed101/")
            assert relative.endswith(".jpg")
            assert (tmp_path / relative).exists()
        # UNIQUE, which prefix/suffix/existence checks cannot see: a filename
        # collision inside one turn leaves `frame_paths` listing four entries
        # that all point at one file, overwritten three times — three quarters
        # of a `look_around` panorama gone while the log claims otherwise.
        assert len(set(paths)) == len(paths), f"frame path collision in {paths}"

    def test_a_look_around_turn_keeps_all_four_bearings(self, document, tmp_path):
        panorama = [
            t for t in document["turns"]
            if any(c["name"] == "look_around" for c in t["model_output"]["tool_calls"])
        ]
        assert panorama, "the fixture must exercise look_around"
        paths = panorama[0]["obs"]["frame_paths"]
        assert len(paths) == 4
        blobs = [(tmp_path / p).read_bytes() for p in paths]
        assert len(set(blobs)) == 4, "the four bearings must be four distinct images"

    def test_the_saved_frame_is_the_exact_image_the_model_saw(self, tmp_path):
        context = make_context()
        provider = FakeProvider([turn(call("get_observation")), turn(call(DECLARE_DONE))])
        log = make_log(tmp_path)
        context.playback.set_true_xy(target_true_xy(9.0))
        document = run_trial(
            provider=provider, context=context, stages=stage_specs(SEED), log=log
        )
        relative = document["turns"][0]["obs"]["frame_paths"][0]
        on_disk = (tmp_path / relative).read_bytes()
        sent = [
            image
            for request in provider.requests
            for block in user_blocks(request["messages"])
            if isinstance(block, ToolResultBlock)
            for image in block.images
        ][0]
        assert on_disk == base64.b64decode(sent.data_b64)

    def test_memory_snapshot_carries_the_block_and_the_correction_series(self, document):
        snapshot = document["turns"][-1]["memory_snapshot"]
        assert set(snapshot) >= {
            "rooms", "exits", "trajectory", "plan", "corrections", "stage", "block",
        }
        assert snapshot["block"].startswith("== YOUR MAP")
        assert isinstance(snapshot["corrections"], list)
        assert "start" in snapshot["rooms"]

    def test_the_nested_map_content_survives_not_just_the_top_level_keys(self, document):
        """doc 06 §5.9's layout QA is scored by joining the model's answers
        against exactly these fields — question 5 asks for landmarks per room,
        and an unexplored exit is what makes a frontier a frontier. A key-set
        check on the snapshot's top level cannot see either of them vanish."""
        snapshot = document["turns"][-1]["memory_snapshot"]
        assert snapshot["rooms"]["start"]["landmarks"] == ["a red rug"]
        assert snapshot["rooms"]["start"]["description"] == "sofa"
        assert snapshot["exits"], "the fixture calls mark_exit"
        assert set(snapshot["exits"][0]) == {"room", "direction_deg", "status"}
        assert snapshot["exits"][0]["status"] == "unexplored"

    def test_the_correction_series_keeps_the_magnitude_of_each_correction(self, document):
        """doc 06 §5.8 needs "the count of correct_position calls and the
        magnitude of each correction"; nothing else in the log can recover the
        magnitude once `old_xy`/`new_xy` are gone."""
        corrections = [
            c for t in document["turns"] for c in t["memory_snapshot"]["corrections"]
        ]
        assert corrections, "the fixture calls correct_position"
        first = corrections[0]
        assert set(first) == {"turn", "old_xy", "new_xy", "reason", "stage"}
        assert first["new_xy"] == [1.0, 2.0]
        assert first["old_xy"] != first["new_xy"]
        assert first["stage"] == STAGE_FIND_KITCHEN
        assert first["turn"] == 2  # stage-local, joins turns[].turn_idx

    def test_final_shape(self, document):
        final = document["final"]
        assert set(final) >= {
            "outcome", "end_reason", "stages", "metrics", "tokens", "qa", "qa_raw",
        }
        assert set(final["outcome"]) == {STAGE_FIND_KITCHEN, STAGE_RETURN_HOME}
        assert final["metrics"] == {}, "§5 values are computed post-hoc by T4.1"
        assert set(final["tokens"]) == {
            "input_tokens", "output_tokens", "cache_read_tokens",
            "cache_write_tokens", "cost_usd_estimate",
        }
        # The decision inputs, so the scorer recomputes rather than trusts.
        score = final["stages"][STAGE_FIND_KITCHEN]["score"]
        assert set(score) == {"success", "distance_m", "radius_m", "goal_xy", "true_xy"}
        assert score["radius_m"] == LAYOUT["target"]["radius"]

    def test_the_document_on_disk_matches_the_one_returned(self, document, tmp_path):
        assert json.loads((tmp_path / "fake_seed101.json").read_text()) == document

    def test_the_log_is_flushed_turn_by_turn(self, tmp_path):
        """"A crash loses at most the in-flight turn" — so the file must be
        complete and parseable after every turn, not only at the end."""
        seen: list[int] = []
        context = make_context(counters=Counters(turn_cap=2))
        log = make_log(tmp_path)

        def check(_record) -> None:
            on_disk = json.loads(log.path.read_text())
            seen.append(len(on_disk["turns"]))
            assert "final" not in on_disk

        runner = EpisodeRunner(
            provider=FakeProvider(
                [turn(call("get_observation")), turn(call("get_observation"))]
            ),
            context=context,
            stages=stage_specs(SEED),
            log=log,
            on_turn=check,
        )
        runner.run_stage(runner.stages[0])
        assert seen == [1, 2]


class TestObsRecordsWhatTheModelWasShown:
    """``obs`` is checked for VALUES here, not only for key sets.

    A key-set check cannot see the log misreport what the model saw, and both
    of the failures below were survivors: replacing ``distance_moved_m`` with a
    constant ``0.0`` and rendering the position from the last breadcrumb instead
    of the integrator each left the suite green.
    """

    def test_the_status_triple_describes_the_previous_turns_motion(self, tmp_path):
        runner, _, log = make_runner(
            tmp_path,
            [
                turn(call("move", distance_m=1.0)),
                turn(call("get_observation")),
                turn(call(DECLARE_DONE)),
            ],
        )
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        runner.run_stage(runner.stages[0])
        turns = log.document["turns"]
        # Turn 1 was shown "you have not moved"; turn 2 was shown the move.
        assert turns[0]["obs"]["status"]["distance_moved_m"] == 0.0
        moved = turns[0]["execution"]["calls"][-1]["distance_moved_m"]
        assert moved > 0.0
        assert turns[1]["obs"]["status"]["distance_moved_m"] == round(moved, 3)

    def test_the_obs_contact_column_records_what_the_status_payload_showed(
        self, tmp_path
    ):
        """T3.5 added ``status.contact`` to the model-facing payload but not to
        the log's ``obs.status`` — so the audit's "what was it reading when it
        decided" column silently omitted the one field that says WHICH way was
        blocked. The obs must carry the same carried reading, same vintage."""
        playback = FakePlayback()
        runner, _, log = make_runner(
            tmp_path,
            [
                turn(call("move", distance_m=1.0)),
                turn(call("get_observation")),
                turn(call(DECLARE_DONE)),
            ],
            context=make_context(playback=playback),
        )
        playback.bumped = True
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        runner.run_stage(runner.stages[0])
        turns = log.document["turns"]
        # Turn 1 was shown the zero state; turn 2 was shown the bump's contact.
        assert turns[0]["obs"]["status"]["contact"] == []
        assert turns[1]["obs"]["status"]["bumped"] is True
        assert turns[1]["obs"]["status"]["contact"] == ["torso"]

    def test_the_position_estimate_follows_the_integrator_not_the_breadcrumb(
        self, tmp_path
    ):
        """The exact failure loop.py's phase-1 comment documents:
        ``correct_position`` re-anchors the integrator WITHOUT appending a
        crumb, so a crumb-rendered block shows the model the number it had just
        overwritten — invisible except in doc 06 §5.8's drift metric."""
        runner, _, log = make_runner(
            tmp_path,
            [
                turn(call("move", distance_m=1.0)),
                turn(call("correct_position", x=1.0, y=2.0, reason="the rug")),
                turn(call("get_observation")),
                turn(call(DECLARE_DONE)),
            ],
        )
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        runner.run_stage(runner.stages[0])
        crumb = runner.memory.breadcrumbs[-1]
        assert (round(crumb.x, 2), round(crumb.y, 2)) != (1.0, 2.0), (
            "the fixture must leave the breadcrumb and the integrator disagreeing"
        )
        shown = log.document["turns"][2]["obs"]["position_estimate"]
        assert shown == {"x": 1.0, "y": 2.0}
        assert shown == {
            "x": round(runner.integrator.xy[0], 2),
            "y": round(runner.integrator.xy[1], 2),
        }
        # ...and the model was actually shown that, not just the log.
        block = log.document["turns"][2]["memory_snapshot"]["block"]
        assert "Position estimate: x=1.00, y=2.00" in block

    def test_the_compass_column_is_the_observed_heading(self, tmp_path):
        runner, _, log = make_runner(
            tmp_path,
            [turn(call("turn_to_heading", heading_deg=210)), turn(call("get_observation"))],
            context=make_context(counters=Counters(turn_cap=2)),
        )
        runner.run_stage(runner.stages[0])
        turns = log.document["turns"]
        assert turns[0]["obs"]["compass_deg"] == START_COMPASS_DEG
        assert turns[1]["obs"]["compass_deg"] == 210.0


class TestFairnessContract:
    """``config.config_hash`` and the manifest it hashes (doc 06 §2, §7).

    The hash was only asserted to be 64 hex characters, so the LIST was
    untested: deleting ``prompts.py`` from ``FROZEN_FILES`` left the suite
    green, and after that a batch run with an edited system prompt would record
    the same ``config_hash`` as a clean one — the trial JSON asserting a
    fairness contract it is no longer checking (AGENTS.md rule 4).
    """

    def test_the_manifest_matches_doc_06_s2(self):
        assert set(FROZEN_FILES) == doc_06_manifest_files()

    def test_every_frozen_file_exists_on_disk(self):
        missing = [rel for rel in FROZEN_FILES if not (REPO_ROOT / rel).exists()]
        assert not missing, f"frozen files are hashed as b'<missing>': {missing}"

    def test_the_manifest_covers_every_enforcement_site(self):
        """Doc 06 §2 freezes ITEMS; the hash covers FILES. These are the four
        the two lists disagreed about — the caps, the motion clamps, K=10, and
        the request shaping — each of which lives in code no earlier version of
        the manifest hashed."""
        for rel in (
            "duck_embody/agent/memory.py",        # TURN_CAP / POLICY_SECONDS_CAP
            "duck_embody/sim/policy_wrapper.py",  # MOVE_MAX_DISTANCE_M / hull clamp
            "duck_embody/agent/loop.py",          # K_CONTEXT_TURNS
            "duck_embody/agent/providers/anthropic.py",
            "duck_embody/agent/providers/openai.py",
            "duck_embody/agent/prompts.py",
            "duck_embody/agent/tools.py",
        ):
            assert rel in FROZEN_FILES

    def test_editing_any_frozen_file_moves_the_hash(self, tmp_path):
        for rel in FROZEN_FILES:
            destination = tmp_path / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO_ROOT / rel).read_bytes())
        baseline = config_hash(FROZEN_FILES, tmp_path)
        assert re.fullmatch(r"[0-9a-f]{64}", baseline)
        for rel in FROZEN_FILES:
            path = tmp_path / rel
            original = path.read_bytes()
            path.write_bytes(original + b"\n# mid-batch edit\n")
            assert config_hash(FROZEN_FILES, tmp_path) != baseline, (
                f"editing {rel} does not change config_hash — a mid-batch change "
                "to it would be invisible to doc 06 §7's guard"
            )
            path.write_bytes(original)
        assert config_hash(FROZEN_FILES, tmp_path) == baseline

    def test_a_renamed_frozen_file_is_a_change(self, tmp_path):
        """Paths are hashed alongside contents, so swapping two files' names
        cannot leave the hash where it was."""
        for rel in FROZEN_FILES:
            destination = tmp_path / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO_ROOT / rel).read_bytes())
        baseline = config_hash(FROZEN_FILES, tmp_path)
        renamed = (FROZEN_FILES[1], FROZEN_FILES[0], *FROZEN_FILES[2:])
        assert config_hash(renamed, tmp_path) != baseline

    def test_the_frozen_files_are_the_hashed_ones(self, tmp_path):
        """A file OUTSIDE the manifest must not move the hash — otherwise the
        test above would pass for the wrong reason."""
        for rel in FROZEN_FILES:
            destination = tmp_path / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO_ROOT / rel).read_bytes())
        baseline = config_hash(FROZEN_FILES, tmp_path)
        (tmp_path / "duck_embody" / "agent" / "unrelated.py").write_text("x = 1\n")
        assert config_hash(FROZEN_FILES, tmp_path) == baseline

    def test_freeze_commit_never_raises_without_git(self, tmp_path):
        assert freeze_commit(tmp_path, FROZEN_FILES) in {"unknown"} or re.fullmatch(
            r"[0-9a-f]{7,40}(-dirty)?", freeze_commit(tmp_path, FROZEN_FILES)
        )

    def test_freeze_commit_marks_an_uncommitted_frozen_file(self):
        """AGENTS.md §5 records that this tree carries large uncommitted work,
        so a bare sha would claim trials ran code the commit does not contain."""
        value = freeze_commit(REPO_ROOT, FROZEN_FILES)
        assert value == "unknown" or re.fullmatch(r"[0-9a-f]{40}(-dirty)?", value)


class TestInfraFailureRedaction:
    """AGENTS.md rule 6 vs rule 7: the traceback is third-party text, and
    ``results/`` is committed to a public repo. Nothing but exception
    formatting was keeping a key out of a tracked file."""

    def test_a_planted_key_never_reaches_the_json(self, tmp_path, monkeypatch):
        secret = "sk-ant-test-DEADBEEFdeadbeef0123456789"
        monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
        log = make_log(tmp_path)
        log.note_infra_failure(
            f"Traceback (most recent call last):\n  auth failed for {secret}\n"
        )
        on_disk = log.path.read_text()
        assert secret not in on_disk
        assert "<redacted:ANTHROPIC_API_KEY>" in on_disk
        assert "final" not in json.loads(on_disk)

    def test_a_key_shaped_string_is_scrubbed_even_when_the_env_is_unset(
        self, monkeypatch
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        text = "httpx.Request(headers={'x-api-key': 'sk-proj-ABCDEFGHIJKLMNOPQRSTUV'})"
        assert "sk-proj-" not in redact_secrets(text)
        assert "<redacted>" in redact_secrets(text)

    def test_ordinary_tracebacks_survive_intact(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-abcdefghijklmnopqrst")
        text = "RuntimeError: RTX render failed: no frame produced"
        assert redact_secrets(text) == text


class TestExecutionResultLine:
    def test_the_doc_example_shape(self):
        execution = {
            "distance_moved_m": 0.82,
            "policy_seconds_used": 4.1,
            "bumped": True,
            "fell": False,
            "stop_reason": "bump",
        }
        assert motion_phrase("move", {}, execution) == (
            "move: moved 0.82 m, 4.1 policy-s, auto-stop on collision (bump)"
        )

    def test_a_fall_is_shouted(self):
        execution = {
            "distance_moved_m": 0.4, "policy_seconds_used": 2.0,
            "bumped": False, "fell": True, "stop_reason": "fell",
        }
        assert "FELL — trial over" in motion_phrase("move", {}, execution)

    def test_a_turn_reports_its_requested_heading(self):
        execution = {
            "distance_moved_m": 0.0, "policy_seconds_used": 0.8,
            "bumped": False, "fell": False, "stop_reason": "timeout",
        }
        line = motion_phrase("turn_to_heading", {"heading_deg": 270}, execution)
        assert line == "turn_to_heading: turned toward 270 deg, 0.8 policy-s, timed out"

    def test_a_turn_with_no_motion_says_so(self):
        assert merge_executions([], [])["result"] == "no motion commanded"
        assert (
            merge_executions([], ["get_observation", "update_plan"])["result"]
            == "no motion; get_observation, update_plan"
        )

    def test_merging_sums_seconds_and_concatenates_the_trace(self):
        left = {"policy_seconds_used": 1.0, "pose_trace": [[0.0, 0.0]],
                "distance_moved_m": 0.2, "bumped": False, "fell": False,
                "stop_reason": "reached"}
        right = {"policy_seconds_used": 2.0, "pose_trace": [[1.0, 1.0], [2.0, 2.0]],
                 "distance_moved_m": 0.4, "bumped": False, "fell": False,
                 "stop_reason": "reached"}
        merged = merge_executions([("move", {}, left), ("move", {}, right)], [])
        assert merged["policy_seconds_used"] == 3.0
        assert merged["pose_trace"] == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        assert merged["motion_calls"] == 2
        assert merged["result"].count("move:") == 2


# ===========================================================================
# 10. Turn counters
# ===========================================================================


class TestTurnCounters:
    def test_context_turn_and_counters_turns_advance_together(self, tmp_path):
        """They count the same thing for two consumers — the budget line the
        model reads, and the stage-local index ``correct_position`` stamps into
        every drift record. Nothing in ``tools.py`` touches either."""
        runner, _, _ = make_runner(
            tmp_path,
            [turn(call("get_observation")) for _ in range(3)] + [turn(call(DECLARE_DONE))],
        )
        runner.context.playback.set_true_xy(target_true_xy(9.0))
        runner.run_stage(runner.stages[0])
        assert runner.context.turn == runner.counters.turns == 4

    def test_the_turn_is_counted_before_dispatch(self, tmp_path):
        """So the stage-local index stamped into a correction made on turn 1 is
        1, not 0 — the ordering doc 05 §3.1 pins."""
        runner, _, _ = make_runner(
            tmp_path,
            [turn(call("correct_position", x=1.0, y=2.0, reason="the rug"))],
        )
        runner.context.counters.turn_cap = 1
        runner.run_stage(runner.stages[0])
        assert runner.memory.corrections[0].turn == 1

    def test_the_global_index_spans_both_stages(self, tmp_path):
        runner, _, _ = make_runner(
            tmp_path,
            [
                turn(call("get_observation")),
                turn(call(DECLARE_DONE)),
                turn(call("get_observation")),
                turn(call(DECLARE_DONE)),
            ],
        )
        runner.context.playback.set_true_xy(target_true_xy(0.0))
        runner.run()
        assert runner.global_turn == 4
        assert runner.context.turn == 2  # stage-local, reset at the boundary


# ===========================================================================
# 11. Transcript entries
# ===========================================================================


class TestTranscriptEntry:
    def test_the_native_turn_is_echoed_unchanged(self):
        native = [{"type": "thinking", "signature": "opaque"}]
        messages = TranscriptEntry(native=native, note="x").messages(keep_images=True)
        assert messages[0].native is native

    def test_an_entry_with_neither_results_nor_a_note_refuses_to_render(self):
        """An empty user message is an API 400, i.e. doc 05 §8's infra path — a
        whole trial rerun for a trivially detectable harness bug."""
        with pytest.raises(ValueError, match="empty user message"):
            TranscriptEntry(native=[]).messages(keep_images=True)


# ===========================================================================
# 12. The CLI (no kit, no API call)
# ===========================================================================


class TestRunTrialCli:
    """``scripts/run_trial.py`` parses and imports without launching anything.

    The script is the entry point T3.5 runs for real, minutes of kit startup and
    dollars of API into a session. A typo in its argument wiring is not something
    to discover there.
    """

    @staticmethod
    def _module():
        import importlib.util

        path = REPO_ROOT / "scripts" / "run_trial.py"
        spec = importlib.util.spec_from_file_location("run_trial_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_module_imports_without_isaac_or_a_provider(self):
        """Every heavy import is inside ``main()``: ``AppLauncher`` must be
        constructed before torch/isaaclab, and a config typo should fail in a
        second rather than after a multi-minute cold start."""
        module = self._module()
        assert module.TASK_ID == "DuckEmbody-Apartment-v0"
        assert module.REPO_ROOT == REPO_ROOT

    def test_the_required_arguments_are_model_and_seed(self):
        parser = self._module().build_parser()
        args = parser.parse_args(["--model", "fable5", "--seed", "101"])
        assert args.model == "fable5" and args.seed == 101
        assert args.task == "find_kitchen"
        assert args.no_video is False and args.video_every_n == 1
        with pytest.raises(SystemExit):
            parser.parse_args(["--seed", "101"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "fable5"])

    def test_the_scoring_artifact_is_written_before_the_video_artifacts(self):
        """Source-level, because `main()` needs a kit process — but the ORDER is
        the contract, and reversing it is a one-line edit with no other symptom.

        `recorder.encode()` runs `subprocess.run(..., check=True)` and `_ffmpeg()`
        raises when the binary is missing, so a video fault after a completed
        episode would leave the JSON with every turn and no `final` — byte-for-
        byte an infra failure, so doc 06 §9.1's resume check would move a
        finished, fully paid trial to `results/incomplete/` and rerun it. Video
        is rule-11 evidence, not the trial result.
        """
        source = (REPO_ROOT / "scripts" / "run_trial.py").read_text()
        body = source[source.index("def main("):].splitlines()
        # CODE lines only — the prose above each block names the same calls, and
        # a test that matched those would pass on a file whose code was reordered.
        code = [
            (n, line.strip())
            for n, line in enumerate(body)
            if line.strip() and not line.strip().startswith("#")
        ]

        def line_of(fragment: str) -> int:
            hits = [n for n, text in code if fragment in text]
            assert hits, f"{fragment!r} is no longer in run_trial.main()"
            return hits[0]

        assert line_of("log.finish(final)") < line_of("recorder.encode()")
        # ...and the GPU is released on every path, including a Ctrl-C in the
        # artifact block: a surviving kit process holds the machine's only GPU
        # and the rerun cannot start at all (AGENTS.md rule 1).
        finallys = [n for n, text in code if text == "finally:"]
        assert finallys, "session.close() must run from a finally block"
        assert max(finallys) < line_of("session.close()")
        assert line_of("except BaseException") < max(finallys)

    def test_unknown_flags_are_left_for_the_kit_launcher(self):
        """``AppLauncher`` parses ``sys.argv`` for its own flags, so ours are
        taken with ``parse_known_args`` and the rest handed back untouched."""
        parser = self._module().build_parser()
        args, rest = parser.parse_known_args(
            ["--model", "opus5", "--seed", "104", "--headless", "--device", "cuda:0"]
        )
        assert args.model == "opus5"
        assert rest == ["--headless", "--device", "cuda:0"]

    def test_every_seed_and_model_in_the_frozen_matrix_is_accepted(self):
        parser = self._module().build_parser()
        for seed in LAYOUT["spawn_points"]:
            for model in ("fable5", "opus5", "gpt56sol"):
                args = parser.parse_args(["--model", model, "--seed", str(seed)])
                assert (args.model, args.seed) == (model, seed)

    def test_the_out_of_benchmark_judge_cannot_be_run_as_a_contestant(self):
        """`configs/models/` also holds `judge.yaml` — the Sonnet 5 scene judge
        (doc 04 §8), deliberately NOT one of the three contestants. Accepting it
        here would spend a full paid trial and write a benchmark-shaped
        `results/raw/judge_seed101.json` with a valid `final` and the same
        `config_hash` as the real trials, which a glob-based aggregator would
        fold into the comparison as a fourth model. The existing guard
        (`tests/test_providers.py::test_the_judge_is_not_a_contestant`) covers
        the config file, not the entry point."""
        assert (REPO_ROOT / "configs" / "models" / "judge.yaml").exists()
        parser = self._module().build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "judge", "--seed", "101"])

    def test_an_off_matrix_seed_is_rejected(self):
        parser = self._module().build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "fable5", "--seed", "999"])

    def test_the_accepted_matrix_is_read_from_the_hashed_config(self):
        """Not hardcoded: `configs/benchmark.yaml` is inside doc 06 §2's
        fairness contract, so the entry point and the contract cannot drift."""
        import yaml

        module = self._module()
        raw = yaml.safe_load(BENCHMARK_YAML.read_text())
        models, seeds = module.frozen_matrix()
        assert list(models) == raw["models"]
        assert list(seeds) == raw["seeds"]


# ===========================================================================
# 13. The per-trial video hook (T3.4's only way to get one at all)
# ===========================================================================


class RecorderSpy:
    def __init__(self):
        self.grabs = 0

    def grab(self, env) -> None:
        self.grabs += 1


class TestAttachRecorder:
    """``recorder.attach_recorder`` — the seam that makes a trial mp4 possible.

    Without it there is NO per-trial video: ``agent/tools.py`` drives motion
    through ``playback.move`` / ``turn_to_heading`` / ``execute`` and never
    passes the ``on_chunk=`` callback, while ``SimSession``'s only per-step
    grabber is reachable through ``scripted_drive``, which the LLM path never
    uses. AGENTS.md rule 11 makes the video the acceptance evidence for any run
    that steps simulation — and "when metrics and video disagree, the video
    wins".
    """

    def test_a_macro_records_at_the_sub_chunk_rate_not_the_servo_rate(self):
        from duck_embody.sim.recorder import RECORD_CHUNK_S, attach_recorder

        playback, recorder = FakePlayback(), RecorderSpy()
        attach_recorder(playback, object(), recorder)
        # `move` is a real doc 02 §6.2 macro in production; the fake stands in
        # for its INTERNAL `self.execute(...)` calls, which is the level the
        # patch has to reach. One 0.2 s servo chunk = 5 recording chunks.
        playback.execute(0.0, 0.0, 0.0, MACRO_CHUNK_S)
        assert recorder.grabs == MACRO_CHUNK_S / RECORD_CHUNK_S == 5

    def test_the_merged_result_is_indistinguishable_from_an_unchunked_one(self):
        from duck_embody.sim.recorder import attach_recorder

        plain = FakePlayback().execute(0.1, 0.0, 0.0, 1.0)
        playback = FakePlayback()
        attach_recorder(playback, object(), RecorderSpy())
        chunked = playback.execute(0.1, 0.0, 0.0, 1.0)
        assert chunked.steps == plain.steps
        assert round(chunked.policy_seconds, 9) == round(plain.policy_seconds, 9)
        assert chunked.commanded == plain.commanded
        assert chunked.duration_s == plain.duration_s

    def test_the_pose_trace_keeps_its_five_hertz_sampling(self):
        """Concatenating each chunk's FULL ``pose_trace`` would add its
        start/end bookends — two extra points per 2-step chunk, i.e. a ~50 Hz
        trace of per-step gait sway, which inflates doc 06 §5.3's SPL path
        integral. Only the 5 Hz samples merge; the bookends are added once."""
        from duck_embody.sim.recorder import attach_recorder

        playback = FakePlayback()
        attach_recorder(playback, object(), RecorderSpy())
        result = playback.execute(0.1, 0.0, 0.0, 0.2)
        # 5 chunks x 1 sampled point each, bracketed by start and end.
        assert len(result.sampled_xy) == 5
        assert result.pose_trace[0] == POSE_TRACE_SENTINEL[0]
        assert result.pose_trace[-1] == (TRUE_POSE[0], TRUE_POSE[1])
        assert len(result.pose_trace) == 7

    def test_a_stop_predicate_falls_through_unrecorded_rather_than_breaking(self):
        """The predicate is given the step index WITHIN one ``execute``, and
        chunking restarts it — so a chunked predicate would silently never fire.
        No tool passes one today; the fallback keeps it correct if one ever
        does, at the cost of frames."""
        from duck_embody.sim.recorder import attach_recorder

        playback, recorder = FakePlayback(), RecorderSpy()
        attach_recorder(playback, object(), recorder)
        playback.execute(0.1, 0.0, 0.0, 1.0, stop_predicate=lambda i: False)
        assert recorder.grabs == 0

    def test_detach_restores_the_unpatched_method(self):
        from duck_embody.sim.recorder import attach_recorder

        playback, recorder = FakePlayback(), RecorderSpy()
        detach = attach_recorder(playback, object(), recorder)
        detach()
        playback.execute(0.1, 0.0, 0.0, 1.0)
        assert recorder.grabs == 0


class TestStageTwoGatePredicate:
    """``runs_return_home`` — one predicate, two consumers inside the loop.

    The tool_result the model is handed and the decision to actually run the
    stage come from the same call, so they cannot drift into offering an
    objective for a leg that never happens.
    """

    def test_only_a_successful_declare_done_continues(self):
        from duck_embody.tasks.find_kitchen import runs_return_home

        assert runs_return_home(REASON_DECLARE_DONE, True) is True
        assert runs_return_home(REASON_DECLARE_DONE, False) is False

    @pytest.mark.parametrize("reason", [REASON_TURN_CAP, REASON_MOTION_CAP, REASON_FALL])
    def test_no_cap_or_fall_ever_continues_under_either_rule(self, reason):
        """All three docs already agreed on this before T3.4; the resolution did
        not change it."""
        from duck_embody.tasks.find_kitchen import runs_return_home

        assert runs_return_home(reason, False) is False
        assert runs_return_home(reason, True) is False
