"""Is contact SIDE distinguishable, or would reporting it be noise?

The T3.5 trials showed the model bumping repeatedly with only a boolean to go
on: `status.bumped`. 6 of 13 moves stopped under 0.11 m. The harness senses far
more than that — `contact_report()` already returns per-body forces across 22
bodies, several of which are left/right distinguishable — but it is marked
"debug only" and the model never sees it.

Reporting WHERE contact happened would be a doc 05 §1 formatting of sensed
state, not a decision made for the model. But that is only worth doing if the
signal is real. This drives at obstacles from a spread of approach angles and
reports which bodies fire, so the decision rests on a measurement rather than on
the fact that left/right body names exist.

Run:  PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/debug_contact_side.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "figures" / "smoke" / "contact_side_report.json"

#: Approach a wall head-on and at a spread of yaw offsets. If side is real, a
#: left-biased approach should light up left-side bodies more than right.
APPROACHES = [
    ("wall head-on", (2.55, 1.70, 90.0)),
    ("wall from the left", (2.55, 1.70, 65.0)),
    ("wall from the right", (2.55, 1.70, 115.0)),
    ("sofa head-on", (0.30, 0.75, 90.0)),
    ("counter head-on", (2.72, 0.95, 270.0)),
]


def group_of(pb, names: list[str]) -> dict[str, str]:
    """body name -> kinematic group, from the SHIPPED tree grouping.

    ``pb._contact_groups`` is grouped by position in the articulation tree —
    the grouping the model-facing ``status.contact`` actually uses. The
    name-suffix heuristic this probe originally carried (`"_2"`/`"_3"` => R)
    is exactly backwards for `knee_and_ankle_assembly_2`, which sits under
    `left_roll_to_pitch_assembly` (policy_wrapper.__init__ documents the
    dump), so its verdict would have contradicted the signal being validated.
    Summing forces by the shipped groups means the verdict and the shipped
    tool report the same thing, or the probe is measuring nothing.
    """
    by_group: dict[str, str] = {}
    for group, ids in pb._contact_groups.items():
        for idx in ids:
            by_group[names[idx]] = group
    return by_group


def main() -> int:
    from duck_embody.sim.session import SimSession, SpawnPose

    session = SimSession.launch(task_id="DuckEmbody-Apartment-v0", headless=True)
    pb = session.playback
    report: dict = {"approaches": []}

    from duck_embody.env.apartment_layout import grid

    g = grid()
    names = list(pb._contact_sensor.body_names)
    print("  GROUPING the code actually uses:")
    for grp, ids in pb._contact_groups.items():
        print(f"    {grp:<10} {[names[i] for i in ids]}")
    body_group = group_of(pb, names)

    for label, (x, y, heading) in APPROACHES:
        session.reset(seed=101, spawn=SpawnPose(x, y, heading))
        pb.settle(0.4)
        peak: dict[str, float] = {}
        groups_seen: set[str] = set()
        # Step count DERIVED from the standoff, never a magic number. Three
        # separate probes in this project have silently measured nothing because
        # a hardcoded step budget did not cover the distance to the obstacle —
        # the run looks clean and every body reads zero, which is
        # indistinguishable from "nothing collided".

        # Distance ALONG THE HEADING, not `clearance()` (nearest obstacle in
        # any direction) — that mistake made three earlier probes measure
        # nothing while looking clean.
        import math

        dx, dy = math.cos(math.radians(heading)), math.sin(math.radians(heading))
        reach = 0.0
        while reach < 2.0 and g.is_free(x + dx * (reach + 0.05), y + dy * (reach + 0.05)):
            reach += 0.05
        steps = int((reach + 0.45) / 0.2 * 50) + 60
        # `reach`, the variable actually computed above. The first run of this
        # probe crashed HERE with `NameError: standoff` — after the multi-minute
        # cold start, before measuring anything — because the print referenced
        # a name from an earlier draft. Recorded so nobody "simplifies" the
        # f-string back.
        print(f"    reach {reach:.2f} m -> {steps} steps "
              f"({steps / 50 * 0.2:.2f} m of commanded travel)")
        for _ in range(steps):
            pb.execute(0.2, 0.0, 0.0, 1 / 50)
            for n, f in pb.contact_report().items():
                peak[n] = max(peak.get(n, 0.0), f)
            for grp in pb.contact_groups():
                groups_seen.add(grp)
        loud = {n: v for n, v in sorted(peak.items(), key=lambda kv: -kv[1])}
        if not loud:
            print("    *** NOTHING registered — the probe did not reach the obstacle")
        # Summed by the SHIPPED tree grouping (see group_of), never by name
        # suffix: the verdict must be about the signal the model actually gets.
        left = sum(v for n, v in loud.items() if body_group.get(n) == "left_leg")
        right = sum(v for n, v in loud.items() if body_group.get(n) == "right_leg")
        front = sum(v for n, v in loud.items() if body_group.get(n) == "head")
        torso = sum(v for n, v in loud.items() if body_group.get(n) == "torso")
        report["approaches"].append(
            {"label": label, "spawn": [x, y, heading], "peak_by_body": loud,
             "left_total": round(left, 1), "right_total": round(right, 1),
             "head_total": round(front, 1), "torso_total": round(torso, 1),
             "contact_groups": sorted(groups_seen)}
        )
        print(f"\n  {label}  (heading {heading:g} deg)")
        print(f"    bodies over threshold: {list(loud)[:6]}")
        print(f"    L={left:7.1f}  R={right:7.1f}  head={front:7.1f}  torso={torso:7.1f}")
        print(f"    contact_groups() reported: {sorted(groups_seen) or 'NONE'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    print("\n== verdict ==")
    head_on = next(a for a in report["approaches"] if a["label"] == "wall head-on")
    from_left = next(a for a in report["approaches"] if a["label"] == "wall from the left")
    from_right = next(a for a in report["approaches"] if a["label"] == "wall from the right")
    print(f"  head-on    L/R = {head_on['left_total']:.0f}/{head_on['right_total']:.0f}")
    print(f"  from left  L/R = {from_left['left_total']:.0f}/{from_left['right_total']:.0f}")
    print(f"  from right L/R = {from_right['left_total']:.0f}/{from_right['right_total']:.0f}")
    print("  -> side is USABLE only if the L/R ratio actually tracks the approach angle")
    print(f"  wrote {OUT.relative_to(REPO_ROOT)}")
    print("  closing app (nothing after this line runs)")
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
