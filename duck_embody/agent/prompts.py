"""Frozen prompt artifacts: the system prompt, the memory renderer, the layout-QA
questions and rubric, and the room-name synonym table.

Everything in this module is part of the **fairness contract** (doc 06 §2): one
prompt template for all three models, frozen in a single commit before the batch,
with no per-model tuning. Changing anything here after the freeze invalidates
every trial that came before it.

.. note::
   **Plan-ordering correction (T2.3).** PLAN assigns the synonym table to T3.1,
   but T2.3's survey gate scores the judge's answers against it and runs first.
   The table was therefore authored here early, as T2.3's dependency; T3.1
   filled in the system prompt, the memory renderer, and the QA rubric around
   it. One source of truth either way — the table is not duplicated.

.. warning::
   **Two audiences, one file, and they must not mix.** :data:`ROOM_SYNONYMS`,
   :func:`normalize_room_name`, :func:`extract_room_mention` and
   :data:`SURVEY_ROOM_QUESTION` enumerate the apartment's *true* room names.
   They exist for the post-hoc scorer (doc 06 §5.7) and the out-of-benchmark
   scene-survey judge (doc 03 §8.2) — **never** for the driving model. Doc 05 §1
   names "real room labels" as ground-truth injection: the model must coin its
   own room names from what it sees. Of the four true room names, three —
   ``bedroom``, ``hallway``, ``living_room`` — appear nowhere in
   :data:`SYSTEM_PROMPT` or in the memory block; ``kitchen`` appears only as the
   stage-1 objective ("Find the kitchen and walk to the counter"), which is the
   task statement, not a layout label. ``tests/test_memory.py`` asserts exactly
   that split stays true.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from duck_embody.agent.memory import Counters, Memory
from duck_embody.sim.policy_wrapper import wrap_deg

# ---------------------------------------------------------------------------
# Room-name synonym table (doc 06 §5.7, §12)
# ---------------------------------------------------------------------------
#
# Used in two places, which is why it is frozen alongside the prompt rather than
# living in either consumer:
#   * T2.3's scene-recognition gate, mapping a judge's free-text answer to a room;
#   * T4.1's map-accuracy matching, deciding whether a model's claimed room name
#     refers to a true room.
#
# Deliberately a FIXED synonym list, never fuzzy or embedding-based matching
# (doc 06 §5.7): a fuzzy matcher would quietly accept "kitchenette" claimed in
# the bedroom, and the metric would stop meaning what it says. Entries are
# lowercase; matching is case-insensitive and punctuation-insensitive.
# Every key is its own *cleaned* form (lowercase, punctuation replaced by
# spaces, runs of whitespace collapsed) — both matchers clean their input before
# looking it up, so a key that is not already in that form can never be reached.
# `tests/test_memory.py` asserts the invariant; a `"living_room"` key sat here
# unreachable until T3.1's review pass removed it (the cleaner turns `_` into a
# space, so `"living room"` is the key that actually matches it).
ROOM_SYNONYMS: dict[str, str] = {
    # living_room
    "living room": "living_room",
    "livingroom": "living_room",
    "lounge": "living_room",
    "living area": "living_room",
    "sitting room": "living_room",
    "family room": "living_room",
    "front room": "living_room",
    "den": "living_room",
    "parlor": "living_room",
    "parlour": "living_room",
    # kitchen
    #
    # "kitchenette" is deliberately ABSENT. doc 06 §9.1 names it as *the*
    # canonical non-synonym near-string that must not match, and the comment
    # above quotes that rationale — an earlier revision of this table listed it
    # anyway (T2.3), which would have made §9.1's fixture pass for the wrong
    # reason: the fixture's claim also fails the polygon half of the rule, so
    # the name half would have gone untested. Removing it is evidence-neutral
    # for T2.3's passed gate: `grep -rli kitchenette results/` matches nothing,
    # so no judge reply ever used the word. T3.1 removal.
    "kitchen": "kitchen",
    "galley": "kitchen",
    "cooking area": "kitchen",
    # bedroom
    "bedroom": "bedroom",
    "bed room": "bedroom",
    "bedchamber": "bedroom",
    "sleeping area": "bedroom",
    "sleeping room": "bedroom",
    "guest room": "bedroom",
    # hallway
    "hallway": "hallway",
    "hall": "hallway",
    "corridor": "hallway",
    "passage": "hallway",
    "passageway": "hallway",
    "entryway": "hallway",
    "entry": "hallway",
    "foyer": "hallway",
    "landing": "hallway",
}

#: Scan order for :func:`extract_room_mention`, longest synonym first. The
#: winner there is chosen by POSITION (first mention wins); this order only
#: breaks a tie between two synonyms matching at the same index, which needs one
#: key to be a whole-word prefix of another — no such pair exists in the table
#: above, so today this is purely defensive against a future entry.
_SYNONYMS_BY_LENGTH = sorted(ROOM_SYNONYMS, key=len, reverse=True)


def normalize_room_name(text: str) -> str | None:
    """Map a free-text room name to a canonical room, or ``None``.

    Exact match after lowercasing and stripping punctuation — the doc 06 §5.7
    rule. Returns ``None`` rather than guessing when nothing matches; an
    unmatched claim is a real signal (the model named a room that does not
    exist) and must not be silently coerced into one that does.
    """
    if not text:
        return None
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    cleaned = " ".join(cleaned.split())
    return ROOM_SYNONYMS.get(cleaned)


def extract_room_mention(text: str) -> str | None:
    """Find a room named anywhere inside a longer reply.

    Needed because judges do not reliably answer in one word even when asked:
    T3.3's probe asked Sonnet 5 for a one-word room name and got a full
    sentence back. Scoring the whole string would fail a correct answer, so the
    gate looks for a room term *within* the reply — while still using the same
    frozen synonym table, so nothing is matched that
    :func:`normalize_room_name` would reject.

    Returns the FIRST room mentioned (by position), so a reply that names one
    room and then speculates about others is scored on its actual answer.
    """
    if not text:
        return None
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    cleaned = f" {' '.join(cleaned.split())} "

    best: tuple[int, str] | None = None
    for synonym in _SYNONYMS_BY_LENGTH:
        idx = cleaned.find(f" {synonym} ")
        if idx == -1:
            continue
        if best is None or idx < best[0]:
            best = (idx, ROOM_SYNONYMS[synonym])
    return None if best is None else best[1]


#: The question the out-of-benchmark judge is asked about each survey frame.
#: Wording authored by T2.3; doc 03 §8.2 fixes only the acceptance criterion
#: ("a VLM, shown each room's sweep frames cold, names the room correctly for
#: all four rooms"), not this text — the earlier "doc 03 §8.2, doc 04 §8"
#: citation overclaimed and is corrected here (AGENTS.md rule 3).
#:
#: NOT a doc 06 §2 frozen item, but **frozen by inclusion**: §7's config-hash
#: guard hashes whole files, so editing this string after the freeze
#: hard-refuses the runner and invalidates every completed trial. Re-run the
#: scene survey before the freeze, never after. (Flagged for T4.2's freeze
#: manifest, which may prefer to move it into ``scripts/judge_scene_survey.py``
#: — its only consumer.)
SURVEY_ROOM_QUESTION = (
    "What room of a home is this? Answer with a single word: "
    "kitchen, bedroom, hallway, or living room."
)


# ===========================================================================
# FROZEN BEFORE THE BATCH — the prompt template (doc 06 §2, AGENTS.md rule 4)
# ===========================================================================
#
# Everything from here to the end of the file is one frozen artifact: one
# system prompt, one memory-block format, one set of QA questions, for all
# three models, with no per-model tuning. It is frozen in a single commit
# before the first batch trial; the commit hash goes into every result file
# (`config.freeze_commit`) and the runner refuses to run if the working tree
# differs from it. **Changing any of it after the freeze invalidates every
# trial that came before** — the correct response is "revert" or "start a new
# batch directory", never "--force" (doc 06 §2, §7).

#: Stage-1 objective (doc 05 §3). The stage-2 objective is *not* in the system
#: prompt on purpose: doc 05 §3.3 delivers it as ``declare_done``'s tool_result
#: mid-episode, so a model that knew about it in advance would map differently
#: from the design the benchmark describes.
STAGE1_OBJECTIVE = "Find the kitchen and walk to the counter"

#: Delivered verbatim as the stage-1 ``declare_done`` call's tool_result, in the
#: same user message as the turn's other results (doc 05 §3.3).
STAGE2_OBJECTIVE_TOOL_RESULT = (
    "New objective: return to your starting position and call declare_done "
    "when you are there."
)

#: Appended when a turn arrives with no tool call (prose only, empty, off-task)
#: or a refusal. Fixed text, doc 05 §8. The episode then runs to its caps and is
#: scored as a failure if it never succeeds — never selectively retried, because
#: retrying only the trials that derailed is selection bias in the model's
#: favour.
DERAILMENT_NUDGE = (
    "No tool call received — you must act via tools; "
    "call get_observation if unsure"
)

#: Breadcrumb window rendered into the memory block. The label in doc 05 §5.2's
#: worked example is literally ``Breadcrumbs (last 5)``, which is the only place
#: the size is stated. The label is FIXED at 5 even when fewer crumbs exist: it
#: names the window, not the count, so the block's grammar does not change shape
#: under the model between turn 1 and turn 40.
BREADCRUMB_WINDOW = 5

#: Rendered wherever a fixed line of the block has nothing to show yet. One
#: token everywhere rather than bespoke wording per slot, so the model learns a
#: single "nothing here" signal. doc 05 §5.2's example is mid-trial and pins
#: only ONE empty case — a room with no landmarks omits its ``landmarks:`` line
#: entirely — so the turn-1 shape below is T3.1's choice, recorded in §5.2 in
#: the same commit (AGENTS.md rule 5).
EMPTY_SLOT = "(none yet)"


SYSTEM_PROMPT = f"""\
You are driving a small bipedal robot through an apartment you have never seen.
You act only through tool calls; prose alone moves nothing and observes nothing.

There is no map, no floor plan, and no localization system. You are your own
SLAM: you author the map, you estimate where you are, and you decide where to
go. The harness only stores what you assert and re-displays it. You choose your
own names for the rooms you find — there is no list of room names to pick from.

**Your objective: {STAGE1_OBJECTIVE}.**
Call `declare_done` when you believe you are standing at the counter. Declaring
in the wrong place is simply a scored failure — there is no retry and no hint.

Every turn you are shown one text block with three sections. `== YOUR MAP ==`
holds the rooms, connections, trajectory and unexplored exits YOU recorded,
numbered as `Place N`. `== STATE ==` holds your compass heading, your
dead-reckoned position estimate, your last breadcrumbs, and your remaining
budget. `== YOUR PLAN ==` holds your standing plan. That block is regenerated
from your own records every turn, so whatever you write into it stays in front
of you for the rest of the episode. Camera frames and older turns do not: only
the first turn and the last 10 turns are kept, and their images are dropped as
they age out. If a detail matters, record it before it scrolls away.

Your position estimate is integrated from the velocities you *commanded*, not
from any measurement of where the robot actually went. It drifts — further with
every metre walked, every turn taken, and every bump. The compass does not
drift; it is absolute. When you recognize a place you already mapped, re-anchor
with `correct_position` instead of trusting the drifting number.

Headings everywhere — compass, breadcrumbs, exit directions, `turn_to_heading`
— are degrees counter-clockwise from east: 0 = east, 90 = north, 180 = west,
270 = south. Positions are metres in that same frame.

1. **Embodiment & physics.** You are a 42 cm bipedal robot; camera at ~0.36 m,
90° HFOV — furniture looks large; doorways are ~0.35 m wide vs your ~0.16 m
body, so center yourself before passing. Walking is slow (max 0.222 m/s
forward); turning in place then driving beats sideways motion. A fall ends the
trial — prefer short moves near obstacles. You cannot get up, you cannot open
doors, you cannot move furniture, and looking up at a counter from 0.36 m is
the normal view, not a sign that something is wrong.

2. **Tool documentation.** The full JSON schemas come attached to every
request; these are the notes that are not in them.

   *Perception.* `get_observation` returns one 512x512 frame plus your compass
   heading, position estimate and status flags. `look_around` returns four
   frames at headings 0/90/180/270 deg, captured by rotating the camera while
   the robot stands still — no fall risk and no motion budget spent, at the
   cost of four images of context.

   *Motion.* `turn_to_heading(heading_deg)` rotates in place, closed-loop on
   the compass to within 5 deg, and reports `timed_out` rather than spinning
   forever. `move(distance_m)` walks forward at 0.2 m/s, at most 1.5 m per
   call, and auto-stops when contact PERSISTS (about half a second of steady
   force), returning `bumped: true` plus the dead-reckoned distance it covered.
   A passing graze does not stop the walk — it is reported, not acted on.
   `status.contact` names which parts of YOUR OWN body felt contact — any of
   `head`, `torso`, `left_leg`, `right_leg`. It tells you what you touched
   with, never what you touched: `head` means something at your own height,
   `torso` something lower, and a single leg means one leg caught a low edge. `send_velocity(vx, vy, wz, duration_s)` is the raw escape
   hatch and does **not** auto-stop — it runs its full duration even through a
   collision. Every command is clamped to the policy's trained envelope and the
   clamp is echoed back: if a command returns changed, the changed one is what
   ran.

   *Memory.* `update_room`, `add_landmark`, `mark_exit`, `set_current_room`,
   `correct_position` and `update_plan` write YOUR map. They spend no motion
   budget and advance no physics. `mark_exit` snaps directions to the nearest
   15 deg, and re-marking the same snapped direction updates that exit rather
   than adding a near-duplicate.

   *Control.* `declare_done` ends the stage.

   Usage notes:
   - Wherever you can, bundle memory writes with your motion command — turns
     are your scarcest budget. Several tool calls in one turn run in the order
     you list them and all their results come back together.
   - A rejected call (unknown room, malformed exit status) comes back as a
     structured error with a `hint`, not an exception. Read it and fix the next
     call; like every turn, it counts against your turn budget.
   - Distances you are told you moved are dead-reckoned, not measured. Treat
     them the same way you treat your position estimate.

3. **Navigation doctrine — a CogNav-style state machine.** Operate in three
explicit modes and name your current mode in your plan: *broad search* (sweep
unexplored exits to grow the map), *contextual search* (kitchen-correlated
evidence seen — tile, counters, appliances — bias exploration toward it),
*verify* (believe you're at the counter: confirm with an observation before
`declare_done`).

4. **Frontier scoring.** A *frontier* is the boundary between explored and
unexplored space — here, the unexplored-exits list. Each turn, briefly rate
each unexplored exit for kitchen-likelihood from visual evidence and pick
deliberately, rather than defaulting to the nearest.

5. **Plan carry-forward.** Your plan is re-shown every turn; either act
consistently with it or call `update_plan` — silent divergence is the failure
pattern to avoid. A plan that says what you are looking for, which exit you
chose and why, and which mode you are in is worth more than one that says what
you are about to do next.

6. **Honesty rules.** Record only what you observed: never invent a room you
haven't entered; mark uncertain exits `unexplored` rather than guessing
`leads_to`; when lost, say so in the plan and navigate to a mapped landmark to
re-anchor. An entry you are unsure of is worth less to you than an absent one:
this map is the only thing you will still have in forty turns, and it is only
useful if you can trust it.
"""


# ---------------------------------------------------------------------------
# The rendered memory block (doc 05 §5.2)
# ---------------------------------------------------------------------------


def _fmt_m(value: float) -> str:
    """Metres, always 2 decimals — ``x=0.90`` keeps its trailing zero.

    ``-0.00`` is folded to ``0.00``: a position estimate a hair west of the
    origin would otherwise render with a minus sign that means nothing and
    invites the model to reason about a sign that is pure float noise.
    """
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


def _one_line(text: str) -> str:
    """Model-authored text, flattened to one line for the block's grammar.

    The block is a line-oriented format with fixed ``== … ==`` section headers,
    and everything in it except the plan occupies exactly one line. A newline
    inside a landmark or a room description would otherwise let a model's own
    text forge block structure: a landmark reading
    ``"== STATE …\\nPosition estimate: x=9.99, y=9.99  (dead-reckoned…)"``
    renders a complete counterfeit STATE header and position line ABOVE the real
    one, so the model — and any post-hoc parser of the block, and the QA prompt,
    which embeds it verbatim — sees two contradictory position estimates.

    This is formatting, not editing: only line breaks change (every other
    character, including runs of spaces, survives), and the stored assertion is
    untouched (doc 05 §1 — the harness stores what the model asserts). The plan
    is deliberately NOT passed through here; §4.3 pins it as verbatim and §5.2's
    worked example is multi-line.
    """
    return " ".join(text.splitlines())


def _fmt_deg(value: float) -> str:
    """Degrees as an integer in [0, 360) — the block never shows ``360 deg``.

    Half-up, then wrapped: 359.7 renders ``0``, not ``360``. Python's ``round``
    is banker's rounding, which would render 88.5 as 88 and 89.5 as 90; the
    model has no way to know that, so it would read as noise in the compass.
    """
    return str(int(wrap_deg(math.floor(wrap_deg(value) + 0.5))))


def render_memory_block(
    memory: Memory,
    counters: Counters,
    position_estimate: tuple[float, float],
    compass_deg: float,
) -> str:
    """Render ``Memory`` into the one text block re-injected every turn.

    Doc 05 §5.2 is the specification and its worked seed-101 example is the
    golden output; ``tests/test_memory.py`` extracts that example from the HTML
    doc itself and asserts this function reproduces it byte for byte, so the doc
    and the renderer cannot drift apart.

    The block is the one artifact exempt from context truncation: it is
    regenerated fresh into every request, so a fact asserted on turn 3 is still
    in front of the model on turn 38 (doc 05 §5.2). Its cost is constant *in the
    length of the episode* — the block does not grow with the transcript — but
    not constant absolutely: it grows with what the model itself writes into it.
    The plan, the only unbounded free-text field, is capped
    (``memory.PLAN_MAX_CHARS``); rooms, landmarks and exits are not, because
    refusing to record an observation the model made would be the harness
    editing the model's map (doc 05 §1).

    DEVIATION from doc 05 §3.1's pseudocode, which calls
    ``render_memory_block(state.memory, state.counters)``: the live compass
    reading and the integrator's (x, y) are passed explicitly rather than read
    off the last breadcrumb. They usually agree — the sim is paused between
    turns, so nothing moves — but ``correct_position`` re-anchors the integrator
    *without* appending a breadcrumb (doc 05 §5.1: crumbs are appended after
    motion commands), and a block that rendered the pre-correction position
    right after the model corrected it would be showing the model a number it
    had just overwritten. Recorded in §5.2 in the same commit.

    Ordering rules, none of which doc 05 states outright (recorded in §5.2 in
    the same commit — the §5.2 example is consistent with all of them):

    * ``Place N`` follows ``Memory.rooms`` insertion order, 1-based, so a room's
      number is stable for the whole episode.
    * ``Connections:`` renders one line per undirected edge — "explicit
      adjacency lines", doc 05 §5.2 — in **exit-creation order**, source room on
      the left; see :meth:`Memory.claimed_edges`, and §5.2, whose "the order the
      model first asserted it" T3.1's review pass amended to match. Reciprocal
      assertions collapse to one line.
    * ``Unexplored exits`` sorts by room name, then ascending direction. That is
      the only rule that reproduces §5.2's example (hallway 0, hallway 270,
      living_room 0 — living_room is Place 1 yet renders last, so it is neither
      insertion nor Place order).
    * Every model-authored string except the plan is flattened to one line
      (:func:`_one_line`), so no assertion can forge a block section.
    * ``Re-anchored:`` renders only once at least one ``correct_position`` call
      exists, like ``Connections:`` and ``Trajectory:`` (added by T3.1's review
      pass, recorded in §5.2 in the same commit).
    """
    lines: list[str] = [
        "== YOUR MAP (authored by you; rendered verbatim by the harness) =="
    ]

    if memory.rooms:
        for number, room in enumerate(memory.rooms.values(), start=1):
            lines.append(
                f"Place {number}: {_one_line(room.name)} -- "
                f"{_one_line(room.description)}"
            )
            if room.landmarks:
                # The one empty case doc 05 §5.2 pins: no landmarks, no line.
                lines.append(
                    "  landmarks: " + "; ".join(_one_line(m) for m in room.landmarks)
                )
    else:
        lines.append(EMPTY_SLOT)

    for left, right in memory.claimed_edges():
        lines.append(f"Connections: {_one_line(left)} <-> {_one_line(right)}")

    if memory.room_sequence:
        lines.append(
            "Trajectory: " + " -> ".join(_one_line(r) for r in memory.room_sequence)
        )

    unexplored = sorted(
        memory.unexplored_exits(), key=lambda e: (e.room, e.direction_deg)
    )
    if unexplored:
        lines.append("Unexplored exits:")
        for exit_ in unexplored:
            lines.append(
                f"  - {_one_line(exit_.room)}: "
                f"exit at {_fmt_deg(exit_.direction_deg)} deg "
                f"({_one_line(exit_.status)})"
            )

    x, y = position_estimate
    crumbs = memory.breadcrumbs[-BREADCRUMB_WINDOW:]
    crumb_text = (
        " ".join(
            f"({_fmt_m(c.x)},{_fmt_m(c.y)},{_fmt_deg(c.heading_deg)})" for c in crumbs
        )
        or EMPTY_SLOT
    )
    current_room = (
        EMPTY_SLOT if memory.current_room is None else _one_line(memory.current_room)
    )
    lines += [
        "",
        "== STATE (sensor-derived; the declared exceptions) ==",
        # The double space before "(dead-reckoned" is verbatim from doc 05 §5.2.
        # It is not a typo to tidy: the golden test compares against the doc.
        f"Position estimate: x={_fmt_m(x)}, y={_fmt_m(y)}  "
        "(dead-reckoned from commanded velocity; drifts)",
        f"Compass heading: {_fmt_deg(compass_deg)} deg",
        f"Current room (your assertion): {current_room}",
        f"Breadcrumbs (last {BREADCRUMB_WINDOW}): {crumb_text}",
    ]
    if memory.corrections:
        # Why this line exists: correct_position re-anchors the estimate without
        # rewriting the breadcrumbs already recorded (they are the honest
        # record), so the crumb series legitimately contains a jump no motion
        # command explains — 1.3 m backwards inside a 0.5 m move, in the worked
        # case. The tool_result that explained it is dropped once its turn ages
        # past the K=10 context window, after which the discontinuity is
        # indistinguishable from a bad integration, in exactly the metric
        # (doc 06 §5.8) correct_position exists to serve. Conditional, so the
        # §5.2 golden block — which has no corrections — is unchanged.
        last = memory.corrections[-1]
        moved = math.dist(last.old_xy, last.new_xy)
        lines.append(
            f"Re-anchored: {len(memory.corrections)} time"
            f"{'' if len(memory.corrections) == 1 else 's'} "
            f"(latest moved the estimate {_fmt_m(moved)} m; "
            "breadcrumbs before it are in the old frame)"
        )
    lines += [
        f"Budget: turns {counters.turns}/{counters.turn_cap}, "
        f"policy-seconds {counters.policy_seconds:.1f}/{counters.policy_seconds_cap:g}",
        "",
        "== YOUR PLAN (update_plan to change; carried forward otherwise) ==",
        # Verbatim: `update_plan` replaces the plan verbatim (doc 05 §4.3), so
        # the renderer must not re-wrap it either.
        memory.plan if memory.plan else EMPTY_SLOT,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layout QA — 5 fixed questions + rubric anchors (doc 06 §5.9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QAQuestion:
    """One frozen layout-QA question and its 1 / 0.5 / 0 rubric anchors.

    Frozen dataclass because these are contract text, not configuration: doc 06
    §5.9 says the questions and rubrics are frozen with the prompt template, and
    a mutated question would make two trials incomparable while looking fine.
    """

    number: int
    text: str
    rubric_1: str
    rubric_half: str
    rubric_0: str


#: The 5 questions, verbatim from doc 06 §5.9's "Question (exact text)" column,
#: with the rubric anchors split out of its "Rubric (1 / 0.5 / 0)" column.
#: ``tests/test_memory.py`` asserts every string below still appears in the doc.
#:
#: Scoring them is T4.1's (``scoring.py``); this module owns only the frozen
#: text. The two Q2 rubric operationalizations an earlier revision of this
#: comment reported open in doc 06 §12 are both RESOLVED by T4.1 in
#: ``scoring.py``, fixture-pinned in ``tests/fixtures/qa_q2_answers.json``:
#: the direction-vocabulary parse rules (``ABSOLUTE_WORDS`` /
#: ``ABSOLUTE_ABBREV`` / the relative tokens, wedge tolerance
#: ``DIRECTION_TOL_DEG``), and the route tolerance — ``MAX_EXTRA_ROOMS = 1``,
#: so the hallway detour (3.611 m vs 3.152 m direct through the committed
#: layout's living_room<->kitchen doorway) costs exactly one defect and scores
#: 0.5, never 0: the 0 anchor is "route would not reach the kitchen", which
#: that route plainly does. Q4's bucketing was never open:
#: ``apartment_layout.compass_8`` pins it (half-open buckets, 22.5° rounds
#: up), so seed 101's 22.521° bearing makes NE the gold answer — 0.021° past
#: the boundary, close enough that the rubric's adjacent-bucket 0.5 is doing
#: the real work there.
#: [measured: docs/designs/06-benchmark-evaluation.html §5.9 + scoring.py, same commit]
LAYOUT_QA_QUESTIONS: tuple[QAQuestion, ...] = (
    QAQuestion(
        number=1,
        text="Which room connects the bedroom to the kitchen?",
        rubric_1="names the unique connector room (per adjacency graph).",
        rubric_half="names a room that is adjacent to exactly one of the two.",
        rubric_0="anything else.",
    ),
    QAQuestion(
        number=2,
        text=(
            "Starting at the front of the sofa, give turn-by-turn directions "
            "to the fridge."
        ),
        rubric_1=(
            "room sequence matches the oracle route AND initial direction "
            "correct AND ends at the fridge."
        ),
        rubric_half=(
            "correct room sequence but a wrong/missing turn direction, or "
            "correct directions with one wrong room name."
        ),
        rubric_0="route would not reach the kitchen.",
    ),
    QAQuestion(
        number=3,
        text="How many rooms did you visit? Name them.",
        rubric_1=(
            "count and names match the true visited set (from the true trace "
            "∩ room polygons)."
        ),
        rubric_half="names correct but count off by one, or one room missing/extra.",
        rubric_0="otherwise.",
    ),
    QAQuestion(
        number=4,
        text="Which direction (compass) is the kitchen from your spawn point?",
        rubric_1=(
            "matches the true bearing bucketed to 8-way compass (spawn → "
            "kitchen centroid)."
        ),
        rubric_half="adjacent bucket (e.g. E vs NE).",
        rubric_0="otherwise.",
    ),
    QAQuestion(
        number=5,
        text="Name one landmark in each room you visited.",
        rubric_1="a true layout landmark for every visited room.",
        rubric_half="correct for all but one room.",
        rubric_0="otherwise.",
    ),
)

#: Preamble for the post-episode QA exchange (doc 06 §5.9). The exchange is
#: FRESH: the model sees only its own final memory block — no new camera frames,
#: no sim access, no tools — so the QA measures the map it built, not another
#: round of exploration.
#:
#: Free-text answers scored by a rubric parser, per doc 06 §12's stated leaning
#: ("to keep the QA a genuine map-reading probe"); numbering the answers is the
#: minimum structure the parser needs to attribute an answer to a question, and
#: is deliberately the *only* structure imposed.
LAYOUT_QA_PREAMBLE = (
    "The run is over. Below is the final map and memory block you authored "
    "during it. Answer the five questions that follow using that block alone — "
    "you have no camera, no robot, and no tools now. Answer each question in "
    "turn, numbered 1 to 5, in plain language."
)


def render_qa_prompt(memory_block: str) -> str:
    """The single user message of the post-episode QA exchange (doc 06 §5.9)."""
    questions = "\n".join(f"{q.number}. {q.text}" for q in LAYOUT_QA_QUESTIONS)
    return f"{LAYOUT_QA_PREAMBLE}\n\n{memory_block}\n\n{questions}"
