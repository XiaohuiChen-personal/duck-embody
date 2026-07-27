"""T3.5 tool-coverage gate: dispatch all 12 tools against a LIVE session.

PLAN T3.5 requires every tool exercised at least once. The sanity trial cannot be
relied on for that — the model fell on turn 8 and never reached `declare_done`,
and three tools went untouched.

The tempting fix is to nudge the system prompt until the model calls them all.
That is precisely what AGENTS.md rule 4's pre-freeze criteria forbid: tuning the
scaffold to manufacture a coverage result, fitted to whichever contestant
happened to fall early. Whether a *model* chooses to call `correct_position` is
data about the model. Whether `correct_position` *works* is data about the
harness, and that is what this gate is for.

So the coverage check is scripted, deterministic and model-free: every tool is
dispatched with valid arguments against a real kit session, and every result is
checked for a structured non-error outcome. No LLM, no money.

Run:  PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/smoke_tool_surface.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "figures" / "smoke" / "tool_surface_report.json"


def main() -> int:
    from duck_embody.agent.memory import Counters, Memory, PositionIntegrator
    from duck_embody.agent.tools import (
        TOOL_SCHEMAS,
        ToolCall,
        ToolContext,
        dispatch,
    )
    from duck_embody.env.camera import HeadCamera
    from duck_embody.sim.session import SimSession, SpawnPose

    failures: list[str] = []
    report: dict = {"calls": []}

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            failures.append(label)

    session = SimSession.launch(task_id="DuckEmbody-Apartment-v0", headless=True)
    session.reset(seed=101, spawn=SpawnPose(0.5, 0.5, 90.0))
    camera = HeadCamera(session.env)
    camera.warmup()

    context = ToolContext(
        playback=session.playback,
        camera=camera,
        memory=Memory(),
        counters=Counters(),
        integrator=PositionIntegrator(0.5, 0.5),
    )

    # Ordered so the memory tools have something to refer to, and so the two
    # tools that can END the run (a fall, or declare_done) come last.
    plan = [
        ("get_observation", {}),
        ("look_around", {}),
        ("update_room", {"name": "start_room", "description": "beige floor, red sofa"}),
        ("add_landmark", {"room": "start_room", "description": "red sofa to the west"}),
        ("mark_exit", {"room": "start_room", "direction_deg": 90, "status": "unexplored"}),
        ("set_current_room", {"name": "start_room"}),
        ("update_plan", {"text": "sweep east along the south wall"}),
        ("turn_to_heading", {"heading_deg": 0}),
        ("move", {"distance_m": 0.4}),
        ("send_velocity", {"vx": 0.1, "vy": 0.0, "wz": 0.0, "duration_s": 0.5}),
        ("correct_position", {"x": 0.9, "y": 0.5, "reason": "re-anchored on the sofa corner"}),
        ("declare_done", {}),
    ]

    # Deliberate negative: a wrong argument NAME must come back as a structured
    # doc 05 §8 error with a hint, never an exception. This started as a bug in
    # this script (it passed `plan`, the schema says `text`) and the tool
    # handled it correctly — worth keeping as an assertion rather than a
    # coincidence.
    negative = ("update_plan", {"plan": "wrong argument name on purpose"})

    declared = {name for name, _ in plan}
    schema_names = {s["name"] for s in TOOL_SCHEMAS}
    print(f"== tool surface: {len(schema_names)} tools declared ==")
    check(declared == schema_names, "the plan covers EVERY declared tool",
          f"missing: {sorted(schema_names - declared)}" if schema_names - declared else "")

    print("\n== dispatching ==")
    for name, args in plan:
        # `id` is required: the loop threads it back as `tool_use_id` so a
        # result can be matched to its call (both providers reject a mismatch).
        outcome = dispatch(
            ToolCall(id=f"call_{name}", name=name, args=args), context
        )
        payload = outcome.payload
        entry = {
            "tool": name,
            "args": args,
            "is_error": bool(outcome.is_error),
            "payload_keys": sorted(payload) if isinstance(payload, dict) else None,
            "n_images": len(outcome.images or []),
            "has_execution": outcome.execution is not None,
        }
        report["calls"].append(entry)
        print(f"  {name:<18} error={entry['is_error']!s:<5} images={entry['n_images']} "
              f"exec={entry['has_execution']!s:<5} keys={(entry['payload_keys'] or [])[:4]}")
        check(not outcome.is_error, f"{name}: returned a non-error result",
              json.dumps(payload)[:90] if outcome.is_error else "")

    name, args = negative
    outcome = dispatch(ToolCall(id="call_negative", name=name, args=args), context)
    print(f"\n== deliberate bad-argument call: {name}({list(args)}) ==")
    check(outcome.is_error, "a wrong argument name is reported as an error")
    check(isinstance(outcome.payload, dict) and "hint" in outcome.payload,
          "the error carries a `hint` (doc 05 §8)",
          json.dumps(outcome.payload)[:80])
    report["negative_case"] = {"tool": name, "args": args,
                               "is_error": bool(outcome.is_error),
                               "payload": outcome.payload}

    # Frame-bearing tools must actually carry frames — an observation tool that
    # returns a clean payload and no image is the failure T1.4 already caught
    # once, and it is invisible in a pass/fail on `is_error` alone.
    for name, expected in (("get_observation", 1), ("look_around", 4)):
        got = next(c["n_images"] for c in report["calls"] if c["tool"] == name)
        check(got == expected, f"{name}: carries {expected} frame(s)", f"got {got}")

    # Motion tools must produce a scoring record; memory tools must not.
    for c in report["calls"]:
        if c["tool"] in ("move", "turn_to_heading", "send_velocity"):
            check(c["has_execution"], f"{c['tool']}: has a scoring `execution` record")
        elif c["tool"] in ("update_room", "add_landmark", "mark_exit",
                           "set_current_room", "update_plan", "correct_position"):
            check(not c["has_execution"], f"{c['tool']}: steps no physics")

    report["failures"] = failures
    report["acceptance"] = "PASS" if not failures else "FAIL"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n== summary ==\n  {len(plan)} tools dispatched, {len(failures)} failures")
    print(f"  wrote {OUT.relative_to(REPO_ROOT)}")
    if failures:
        for f in failures:
            print(f"    FAIL {f}")
    else:
        print("  OK - every declared tool dispatches against a live session")
    print("  closing app (nothing after this line runs)")

    session.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
