"""Which bodies actually touch a wall? (T2.4 diagnostic, no assertions.)

T2.4 found the duck walking into a wall, TOPPLING, and reporting ``bumped=False``
— because bump detection reads only ``trunk_assembly`` (doc 02 §6.2). The contact
sensor covers ``/Robot/base/.*``, i.e. every body, so this drives at a wall and
logs the per-body force history to show what the trunk-only filter is missing.

Feet are expected to be loud the entire time (they carry the robot); the question
is which OTHER body sees the wall, and how many steps before the topple.

Run:  PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/debug_bump_bodies.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "figures" / "smoke" / "bump_bodies_report.json"


def main() -> int:
    import torch

    from duck_embody.sim.session import SimSession, SpawnPose

    session = SimSession.launch(task_id="DuckEmbody-Apartment-v0", headless=True)
    pb = session.playback
    sensor = pb._contact_sensor
    names = list(sensor.body_names)
    print(f"contact sensor bodies ({len(names)}):")
    for i, n in enumerate(names):
        print(f"  [{i:2d}] {n}")

    report: dict = {"body_names": names, "runs": []}

    # Two approaches known to topple the duck in the T2.4 pass, plus the sofa,
    # which the trunk-only filter DID catch — a control case.
    runs = [
        # Step counts must COVER the standoff distance, or the run ends before
        # the obstacle is reached and every body looks quiet. (First cut used 130
        # steps for a 1.09 m standoff = 0.52 m of walking, and "proved" the wall
        # touches nothing.)
        {"name": "wall_A", "spawn": (2.55, 1.60, 90.0), "steps": 450},
        {"name": "fridge_proxy", "spawn": (2.60, 2.30, 0.0), "steps": 250},
        {"name": "sofa_control", "spawn": (0.30, 0.75, 90.0), "steps": 250},
    ]

    for run in runs:
        x, y, h = run["spawn"]
        session.reset(seed=101, spawn=SpawnPose(x, y, h))
        print(f"\n== {run['name']}  spawn=({x}, {y}) heading={h} ==")

        peak = torch.zeros(len(names))
        first_step = {}
        fell_at = None
        trace = []

        for step in range(run["steps"]):
            pb.execute(0.2, 0.0, 0.0, 1.0 / 50.0)
            forces = sensor.data.net_forces_w[0].norm(dim=-1).cpu()
            peak = torch.maximum(peak, forces)
            for i, f in enumerate(forces.tolist()):
                if f > 1.0 and i not in first_step:
                    first_step[i] = step
            height = pb.true_height()
            if step % 10 == 0:
                loud = [
                    (names[i], round(f, 1))
                    for i, f in enumerate(forces.tolist())
                    if f > 1.0
                ]
                trace.append({"step": step, "height": round(height, 4), "loud": loud})
            if height < 0.09 and fell_at is None:
                fell_at = step
                print(f"  FELL at step {step} (t={step / 50.0:.2f} s)")
                break  # env auto-resets on termination; anything after is a lie

        non_foot = {
            i: st for i, st in first_step.items() if "foot" not in names[i]
        }
        if non_foot:
            lead = min(non_foot.values())
            print(f"  FIRST NON-FOOT CONTACT at step {lead} "
                  f"({names[min(non_foot, key=non_foot.get)]}), "
                  f"fall at {fell_at} -> {(fell_at - lead) if fell_at else 'n/a'} "
                  f"steps of warning")
        else:
            print("  NO non-foot body ever exceeded 1 N")
        print(f"  peak force per body (N), and the step it first exceeded 1 N:")
        for i, n in enumerate(names):
            p = peak[i].item()
            if p > 1.0:
                print(f"    {n:<28} peak {p:8.1f}  first@step {first_step.get(i, '-')}")

        report["runs"].append(
            {
                "name": run["name"],
                "spawn": [x, y, h],
                "fell_at_step": fell_at,
                "peak_force_n": {names[i]: round(peak[i].item(), 2) for i in range(len(names))},
                "first_contact_step": {names[i]: first_step.get(i) for i in range(len(names))},
                "trace": trace,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")
    print("closing app (nothing after this line runs)")
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
