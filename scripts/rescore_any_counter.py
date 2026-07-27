"""POST-HOC sensitivity analysis: stage-1 success under "any counter face".

**STATUS: superseded — the criterion this script explored WAS ADOPTED as the
published scoring** (criterion v2, 2026-07-27, owner-directed;
``results/rerun_log.md``, ``docs/METRICS.md`` §2.1, ``duck_embody/scoring.py``).
The adopted form is the UNION of the pre-registered disc and this script's
counter band (the two regions are not nested — the pinned target point is
0.397 m from the nearest counter footprint), which is identical on this batch.
The script and its output are kept as the provenance record of how the decision
was reached: it ran, and was adversarially verified, BEFORE adoption. The
docstring below is the original analysis framing, edited only from present to
past tense where adoption has since made a statement false; the analysis
itself is unchanged.

**This was not the benchmark's scoring when it ran.** The then-primary
criterion — Euclidean distance to the single target point
``LAYOUT["target"]["point"]``, never to a cabinet footprint — was pinned
*before the batch* (doc 03 §4; the layout's own ``target`` comment; a required
fixture in ``tests/test_scoring.py``) and its published numbers in
``results/scores.json`` stood unchanged at analysis time. This script existed
because the batch surfaced a case the pinned choice decides: ``gpt56sol_seed103``
declared done 5 cm from the face of ``counter_5`` (the east-wall run) and scored
``declared_elsewhere`` against a goal point that sits before the *south* run.
The objective text — "Find the kitchen and walk to the counter" — cannot name
one run without leaking layout knowledge, so "which counter" is genuinely
underdetermined for the model. This analysis asks: **how would stage-1 SR read
if any kitchen-counter face counted?**

Decided AFTER results were visible, so at the time it could only be reported
as a sensitivity analysis, clearly labeled, re-scoring all 12 trials of all
three models together (the same all-models-together rule
``duck_embody/scoring.py``'s header sets for post-batch scorer changes; the
subsequent adoption as criterion v2 is logged in ``results/rerun_log.md``).

Alternative criterion (stage 1 only)
------------------------------------
success_alt :=
    end_reason == declare_done                    (unchanged: the model must
                                                   *know* it arrived — doc 06 §3.1)
AND room_at(end_xy) == "kitchen"                  (a face reached through a wall
                                                   is not "at the counter":
                                                   counter_4/5 back onto the
                                                   bedroom partition, and the
                                                   south run backs onto the
                                                   apartment's south wall)
AND min distance from end_xy to any kitchen sektion_cabinet footprint
    <= LAYOUT["target"]["radius"]                 (same 0.35 m radius; only the
                                                   goal geometry changes:
                                                   point -> five rectangles)

What this CANNOT recover: ``return_home``. The live gate
(``STAGE2_REQUIRES_STAGE1_SUCCESS``) consulted the primary predicate at run
time, so a trial that flips to success here never ran its return leg — that
data does not exist and only a rerun could create it. The alternative SR is a
stage-1 number only.

Reads: results/raw/*.json (immutable), the frozen layout's own helpers.
Writes: results/rescore_any_counter.json — and nothing else.

Run:  python3 scripts/rescore_any_counter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from duck_embody.env.apartment_layout import (  # noqa: E402
    LAYOUT,
    _dist_point_rect,
    room_at,
)
from duck_embody.scoring import (  # noqa: E402
    NA,
    load_trial,
    stage_end_xy,
)
from duck_embody.agent.memory import STAGE_FIND_KITCHEN  # noqa: E402
from duck_embody.tasks.find_kitchen import REASON_DECLARE_DONE  # noqa: E402

RAW_DIR = REPO_ROOT / "results" / "raw"
OUT_PATH = REPO_ROOT / "results" / "rescore_any_counter.json"


def counter_rects() -> list[tuple[str, tuple[float, float, float, float]]]:
    """The five kitchen counters' footprints, from the frozen layout only.

    Selected structurally (kitchen + sektion_cabinet asset), not by name, so
    the list cannot silently miss a counter someone renames. Footprints in
    ``LAYOUT["furniture"]`` are already world-axis extents (the same reading
    ``furniture_rects`` uses), so the rectangle is axis-aligned by construction.
    """
    rects = []
    for item in LAYOUT["furniture"]:
        if item["room"] == "kitchen" and item["asset"] == "sektion_cabinet":
            cx, cy = item["pos"]
            w, d = item["footprint"]
            rects.append(
                (item["name"], (cx - w / 2.0, cy - d / 2.0, cx + w / 2.0, cy + d / 2.0))
            )
    if len(rects) != 5:
        raise RuntimeError(
            f"expected 5 kitchen sektion_cabinet counters in the frozen layout, "
            f"found {len(rects)} — the criterion definition no longer matches the scene"
        )
    return rects


def rescore_trial(path: Path, rects) -> dict:
    document = load_trial(path)
    stage = document["final"]["stages"][STAGE_FIND_KITCHEN]
    end_reason = stage["end_reason"]
    end_xy = stage_end_xy(document, STAGE_FIND_KITCHEN)
    distances = {name: _dist_point_rect(end_xy[0], end_xy[1], rect) for name, rect in rects}
    nearest = min(distances, key=distances.get)
    in_kitchen = room_at(end_xy[0], end_xy[1]) == "kitchen"
    radius = float(LAYOUT["target"]["radius"])
    declared = end_reason == REASON_DECLARE_DONE
    success_alt = declared and in_kitchen and distances[nearest] <= radius
    return {
        "trial_id": document["trial_id"],
        "model": document["config"]["model"],
        "seed": document["config"]["seed"],
        "primary_outcome": stage["outcome"],
        "primary_success": bool(stage["success"]),
        "end_reason": end_reason,
        "end_xy": [round(end_xy[0], 4), round(end_xy[1], 4)],
        "end_room": room_at(end_xy[0], end_xy[1]),
        "nearest_counter": nearest,
        "distance_to_nearest_counter_face_m": round(distances[nearest], 4),
        "distance_to_primary_target_m": (
            round(float(stage["score"]["distance_m"]), 4)
            if stage.get("score")
            else NA
        ),
        "success_alt": success_alt,
        "flipped": success_alt != bool(stage["success"]),
    }


def main() -> int:
    rects = counter_rects()
    paths = sorted(RAW_DIR.glob("*.json"))
    if len(paths) != 12:
        raise RuntimeError(f"expected the 12 immutable batch trials, found {len(paths)}")
    rows = [rescore_trial(path, rects) for path in paths]

    by_model: dict[str, dict] = {}
    for row in rows:
        agg = by_model.setdefault(
            row["model"], {"trials": 0, "primary_successes": 0, "alt_successes": 0}
        )
        agg["trials"] += 1
        agg["primary_successes"] += int(row["primary_success"])
        agg["alt_successes"] += int(row["success_alt"])

    freeze = json.loads((REPO_ROOT / "results" / "freeze.json").read_text())
    result = {
        "analysis": "post_hoc_sensitivity_any_counter_face",
        "provenance": {
            "post_hoc": True,
            "decided_after_results_were_visible": True,
            "motivation": (
                "gpt56sol_seed103 declared done 0.05 m from counter_5 (east-wall "
                "run) and scored declared_elsewhere against the pinned south-run "
                "target point; the objective text does not disambiguate the runs"
            ),
            "primary_scoring_unchanged": "results/scores.json",
            "batch_config_hash": freeze.get("config_hash"),
            "criterion": (
                "end_reason == declare_done AND end position inside the kitchen "
                "room polygon AND distance from end position to the nearest of "
                "the five kitchen sektion_cabinet footprint rectangles <= 0.35 m "
                "(the primary radius, unchanged)"
            ),
            "return_home_note": (
                "not recoverable post hoc: the live stage-2 gate consulted the "
                "primary predicate, so a trial that flips here never ran its "
                "return leg"
            ),
        },
        "trials": rows,
        "per_model": by_model,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    flips = [r for r in rows if r["flipped"]]
    print(f"rescored {len(rows)} trials -> {OUT_PATH.relative_to(REPO_ROOT)}")
    for model, agg in sorted(by_model.items()):
        print(
            f"  {model}: primary {agg['primary_successes']}/{agg['trials']}"
            f" -> alt {agg['alt_successes']}/{agg['trials']}"
        )
    for row in flips:
        print(
            f"  FLIP {row['trial_id']}: {row['primary_outcome']} -> success_alt "
            f"({row['distance_to_nearest_counter_face_m']} m from {row['nearest_counter']})"
        )
    if not flips:
        print("  no trial changes verdict under the alternative criterion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
