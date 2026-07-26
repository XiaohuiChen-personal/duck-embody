"""Frozen prompt artifacts: the system prompt, the memory renderer, the layout-QA
questions and rubric, and the room-name synonym table.

Everything in this module is part of the **fairness contract** (doc 06 §2): one
prompt template for all three models, frozen in a single commit before the batch,
with no per-model tuning. Changing anything here after the freeze invalidates
every trial that came before it.

.. note::
   **Plan-ordering correction (T2.3).** PLAN assigns the synonym table to T3.1,
   but T2.3's survey gate scores the judge's answers against it and runs first.
   The table is therefore authored here now, as T2.3's dependency; T3.1 fills in
   the system prompt, the memory renderer, and the QA rubric around it. One
   source of truth either way — the table is not duplicated.
"""

from __future__ import annotations

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
ROOM_SYNONYMS: dict[str, str] = {
    # living_room
    "living_room": "living_room",
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
    "kitchen": "kitchen",
    "kitchenette": "kitchen",
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

#: Longest-first, so "living room" is matched before the bare "room" inside it
#: and "bed room" before "bed".
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


#: The question the out-of-benchmark judge is asked about each survey frame
#: (doc 03 §8.2, doc 04 §8). Frozen with the gate.
SURVEY_ROOM_QUESTION = (
    "What room of a home is this? Answer with a single word: "
    "kitchen, bedroom, hallway, or living room."
)
