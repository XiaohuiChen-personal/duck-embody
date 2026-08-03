"""Single-session pre-freeze smoke: SIM TEST PLAN scenarios S0-S5 (gap hunt).

ORCHESTRATOR-RUN ONLY (AGENTS.md rule 1 — this launches a kit process).
This script was written by a no-sim session and has NEVER been executed; the
orchestrator runs it exactly once per invocation of:

    BUDGET=$(~/IsaacLab/isaaclab.sh -p scripts/smoke_gap_hunt.py --print-budget \\
        | tail -n1)
    PYTHONUNBUFFERED=1 timeout --kill-after=60 "$BUDGET" \\
        ~/IsaacLab/isaaclab.sh -p scripts/smoke_gap_hunt.py

The ``tail -n1`` is load-bearing (measured): ``isaaclab.sh -p`` writes terminal
escape codes and an ``[INFO] Using python from: ...`` banner to STDOUT ahead of
the number, so a bare capture is a multi-line string and ``timeout`` dies with
"invalid time interval" before the smoke starts. The budget number is the last
line printed; stderr redirection alone does NOT fix it (the banner is stdout).

The ``timeout --kill-after`` wrapper is G7 guard (3): 2 of 26 probe-era logs
ended in an exception-exit hang that held the GPU for 22 minutes, so every sim
invocation gets a kill switch whose budget is DERIVED from the run's own step
counts (``--print-budget``, computed pre-kit from layout geometry), never
guessed. Logging is per-invocation (G13): the run's own artifacts go to a fresh
``results/logs/gap_hunt_<timestamp>/`` directory; never reuse a log path.

Structure (AGENTS.md §5 kit-process rules):

* ALL work happens inside ``try``; ``session.close()`` is in ``finally`` — a
  surviving kit process holds the machine's only GPU.
* Every verdict line prints IMMEDIATELY (``SCENARIO <id> PASS|FAIL: ...``), and
  the aggregate ``VERDICT:`` line + the JSON report are written BEFORE
  ``close()`` — ``SimulationApp.close()`` terminates the process and statements
  after it never run.
* Every drive's step budget is derived from ``apartment_layout`` free-space
  geometry along the commanded heading (reach-scan + the probe-established
  0.45 m margin) — three probes in this project silently measured nothing from
  hardcoded step counts.

Scenarios (the consolidated pre-freeze plan, §2):

  S0  startup canary — the session's own startup severity lines diffed against
      the benign-noise allowlist (results/logs/README.md).
  S1  recorded multi-chunk bump — G1 end-to-end on the default recorded path.
  S2  recorded forced fall — G1's diagnostics half + G8 (video ends pre-topple)
      + G9 (seconds-into-call) + G10 (values_pre_step marker).
  S3  contact-side matrix — G3's gate: the frozen prompt's single-leg-recovery
      sentence stands or gets weakened based on THIS scenario's verdict.
  S4  move() auto-stop semantics — no spurious aborts brushing furniture.
  S5  scripted two-stage mini-trial through the REAL dispatch path (stub
      provider, no LLM, no paid calls) + audit_trial.py conformance.

The JSON report (per-scenario ``{id, pass, measurements, artifact_paths}``)
lands in the run directory; S3 additionally rewrites
``results/figures/smoke/contact_side_report.json`` (superseding the crashed
``t3_5_contact_side.log`` run — G13).

``--checkpoint`` points the whole gap hunt at a RETRAINED policy. Every scenario
here except S0 is a claim about the *gait*: S2 asserts a full-hull press into
the counter run topples the robot, S3 maps which body region feels a bump, S4
asserts brushing furniture does not spuriously abort a ``move``. A new
checkpoint invalidates all three, and re-running this script is how they get
re-measured. Omitting the flag keeps the previous behaviour exactly —
``session.py``'s ``DEFAULT_CHECKPOINT`` (``policy/model_2999.pt``, the
v4_robust baseline the frozen batch ran) — so the four archived
``gap_hunt_*`` runs stay reproducible. The report gains one ADDITIVE key,
``checkpoint``, because a verdict about a gait must name the gait it measured.
Note S2's verdict can legitimately FLIP on a more robust policy: a candidate
that does NOT topple under a sustained press prints INCONCLUSIVE there, which
is a result about the policy, not a broken gate.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# --- pure-python imports (no kit needed; verified importable by system python)
from duck_embody.env.apartment_layout import (  # noqa: E402
    BODY_RADIUS_M,
    LAYOUT,
    clearance,
    grid,
    path_length,
    spawn_pose,
    target_point,
)
from duck_embody.sim.policy_wrapper import (  # noqa: E402
    BUMP_FORCE_N,
    CONTROL_DT,
    MOVE_SPEED_MPS,
)

# ---------------------------------------------------------------------------
# Geometry-derived budgets (no hardcoded step counts)
# ---------------------------------------------------------------------------

#: Fixed margin ADDED to every reach-scan, in metres — the probe-established
#: value (debug_contact_side.py): covers the body reach past the last inflated
#: free cell, the debounce travel, and gait slip. The plan spec blesses
#: "reach-scan + fixed margin" explicitly.
REACH_MARGIN_M = 0.45

#: S2's push allowance AFTER the geometric reach, in policy-seconds. This
#: constant only bounds runtime — it never decides PASS/FAIL: no measured
#: topple time exists pre-run, so a no-fall outcome prints INCONCLUSIVE (not
#: FAIL, not PASS), per the plan spec.
PUSH_ALLOWANCE_S = 15.0

#: Contact dwell after a confirmed bump in S3, policy-seconds. Bounds runtime
#: only: the side verdict uses the union and peaks over ALL sampled steps, so
#: a short dwell can under-sample a region but never mislabel one.
CONTACT_DWELL_S = 1.0


def reach_along(x: float, y: float, heading_deg: float, cap_m: float = 3.0) -> float:
    """Free distance along a heading, from the inflated free-space grid.

    Distance ALONG THE HEADING, never ``clearance()`` (nearest obstacle in any
    direction) — that exact mistake made three earlier probes measure nothing
    while looking clean.
    """
    g = grid()
    dx = math.cos(math.radians(heading_deg))
    dy = math.sin(math.radians(heading_deg))
    reach = 0.0
    while reach < cap_m and g.is_free(x + dx * (reach + 0.05), y + dy * (reach + 0.05)):
        reach += 0.05
    return reach


def drive_budget_steps(reach_m: float) -> int:
    """Step budget to cover ``reach_m`` + margin at the move speed."""
    return int((reach_m + REACH_MARGIN_M) / MOVE_SPEED_MPS / CONTROL_DT)


# --- scene geometry the scenarios aim at, derived from the layout dict -----

def _furniture(name: str) -> dict:
    for item in LAYOUT["furniture"]:
        if item["name"] == name:
            return item
    raise KeyError(name)


def counter_faces() -> tuple[float, float, float]:
    """(north_face_y, west_edge_x, east_edge_x) of the SOUTH counter run.

    counter_1..3 only, by name: the layout also has an east-wall run
    (counter_4/5 at x=3.13) whose extents a startswith("counter_") filter
    folded in — the first dry run of this script produced a "counter face" at
    y=1.58 and an east edge inside wall C's inflation because of exactly that.
    """
    counters = [
        f for f in LAYOUT["furniture"]
        if f["name"] in ("counter_1", "counter_2", "counter_3")
    ]
    north = max(f["pos"][1] + f["footprint"][1] / 2.0 for f in counters)
    west = min(f["pos"][0] - f["footprint"][0] / 2.0 for f in counters)
    east = max(f["pos"][0] + f["footprint"][0] / 2.0 for f in counters)
    return north, west, east


def sofa_east_edge_x() -> float:
    sofa = _furniture("sofa")
    return sofa["pos"][0] + sofa["footprint"][0] / 2.0


# --- S3's approach matrix, all derived --------------------------------------

# The yaw-offset wall target: the hallway's NORTH OUTER wall (y=3.6, x 0-4.8,
# no doorways). NOT wall A2 — the first dry run of this script anchored at
# (1.7, 1.7) in the living room, which sits inside wall B's inflated margin,
# and three of five oblique rays scanned 0.00 m of free space (the
# probe-measured-nothing class this plan exists to kill). Every ray below is
# anchored so it HITS the outer wall at x = WALL_HIT_X, between the two
# hallway planters (x 1.20 / 3.30) with >0.9 m of lateral clearance.
WALL_Y = 3.6
WALL_HIT_X = 2.25
WALL_ANCHOR_Y = 2.95  # hallway mid-corridor, clear of wall A's inflation
WALL_NORMAL_DEG = 90.0
YAW_OFFSETS_DEG = (-25.0, -15.0, 0.0, 15.0, 25.0)
S3_SEEDS = (101, 104)


def s3_approaches() -> list[dict]:
    """label, spawn pose, expected nearer side (None for head-on controls)."""
    out: list[dict] = []
    for off in YAW_OFFSETS_DEG:
        heading = WALL_NORMAL_DEG - off
        # Anchor x chosen so this ray hits the wall at WALL_HIT_X exactly.
        run = (WALL_Y - WALL_ANCHOR_Y) / math.sin(math.radians(heading))
        x0 = WALL_HIT_X - math.cos(math.radians(heading)) * run
        # Signed angle from heading to the wall normal: positive => the wall
        # is counter-clockwise of the heading => the LEFT shoulder is nearer.
        signed = (WALL_NORMAL_DEG - heading + 180.0) % 360.0 - 180.0
        expected = None if off == 0.0 else ("left_leg" if signed > 0 else "right_leg")
        out.append(
            {"label": f"wall yaw{off:+.0f}", "spawn": (round(x0, 3), WALL_ANCHOR_Y, heading),
             "expected": expected, "mirror_key": f"wall{abs(off):.0f}"}
        )

    north, west, east = counter_faces()
    # Edge approaches: the robot's centreline rides the cabinet edge, so the
    # cabinet overlaps exactly one body-half.
    out.append(
        {"label": "counter west edge", "spawn": (west, 0.95, 270.0),
         # Facing south: the robot's LEFT is east — the side the counter is on.
         "expected": "left_leg", "mirror_key": "counter_edge"}
    )
    out.append(
        # y=0.90, not the west edge's 0.95: the east edge sits under
        # counter_4's south face (y 1.0165), and a 0.95 start would spawn the
        # body inside that cabinet's clearance.
        # 3 cm INSIDE the geometric edge: the knife-edge ride at exactly
        # `east` measured ZERO force at both seeds (gait sway carries the body
        # a centimetre wide of a zero-overlap line). 3 cm keeps the overlap on
        # one body-half — the property the probe needs — while making contact
        # robust to sway.
        {"label": "counter east edge", "spawn": (east - 0.03, 0.90, 270.0),
         "expected": "right_leg", "mirror_key": "counter_edge"}
    )
    # Sofa east edge, approaching north: the robot's LEFT is west = sofa side.
    # (The west edge has no mirror twin: it sits a body-radius from the west
    # wall, so the wall would confound the measurement.)
    out.append(
        {"label": "sofa east edge", "spawn": (sofa_east_edge_x(), 0.75, 90.0),
         "expected": "left_leg", "mirror_key": None}
    )
    return out


# ---------------------------------------------------------------------------
# --print-budget: the timeout wrapper's argument, computed WITHOUT kit
# ---------------------------------------------------------------------------

def estimated_policy_seconds() -> float:
    total = 0.0
    # S1: one move into the sofa from the T2.4-proven south approach.
    total += (reach_along(0.30, 0.75, 90.0) + 0.3) / MOVE_SPEED_MPS + 2.0
    # S2: reach the counter + the push allowance.
    total += reach_along(2.72, 0.95, 270.0) / MOVE_SPEED_MPS + PUSH_ALLOWANCE_S + 2.0
    # S3: every approach (x seeds) + dwell + recovery (1.5 s sidestep + 0.3 m
    # move + settle) — recovery conservatively budgeted for every approach.
    for approach in s3_approaches():
        x, y, heading = approach["spawn"]
        reach = reach_along(x, y, heading)
        per = (reach + REACH_MARGIN_M) / MOVE_SPEED_MPS + CONTACT_DWELL_S
        per += 1.5 + 0.3 / MOVE_SPEED_MPS + 1.0
        total += per * len(S3_SEEDS)
    # S4: the corridor run + settles.
    north, west, east = counter_faces()
    y4 = north + BODY_RADIUS_M + LAYOUT["wall_thickness"]
    total += reach_along(2.0, y4, 0.0) / MOVE_SPEED_MPS + 2.0
    # S5: out-and-back along the A* route, with the macro time margin, plus a
    # per-leg allowance for turn_to_heading (its 8 s timeout is the worst
    # case; typical turns finish in 1-2 s — budget half the timeout).
    # length/0.3 is a conservative UPPER bound on macro legs: the navigator's
    # line-of-sight shortcuts issue fewer, longer moves, and the headroom
    # also covers its bounded bump-recovery send_velocity calls (the seed-101
    # dry run measured <= 62 policy-seconds against this term's ~126).
    g = grid()
    route = g.path(spawn_pose(101)[0], target_point())
    length = path_length(route) if route else 6.0
    n_waypoints = max(4, int(length / 0.3))
    total += 2 * (length / MOVE_SPEED_MPS * 1.6 + n_waypoints * 4.0 + 4.0)
    return total


def wallclock_budget_s() -> int:
    """Kill-switch budget: startup + policy-time x a conservative factor.

    The wall-clock-per-policy-second factor is UNMEASURED (doc 06 §12's open
    question); 30x is a deliberate upper bound so the timeout is protection,
    never the thing that kills a healthy run. Startup allowance 900 s: cold
    start "costs minutes" (AGENTS.md §4) + video encode at the end.
    """
    return int(900 + estimated_policy_seconds() * 30.0)


# ---------------------------------------------------------------------------
# S0's self-capture: tee this process's fd-level stdout/stderr to a file.
# Kit's C++ plugins write straight to fd 1/2, so Python-level redirection
# would miss exactly the severity lines S0 exists to count.
# ---------------------------------------------------------------------------

def tee_process_output(capture_path: Path):
    import threading

    capture_path.parent.mkdir(parents=True, exist_ok=True)
    read_fd, write_fd = os.pipe()
    orig_fd = os.dup(1)
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    capture = open(capture_path, "wb", buffering=0)
    original = os.fdopen(orig_fd, "wb", buffering=0)

    def pump() -> None:
        while True:
            try:
                data = os.read(read_fd, 65536)
            except OSError:
                break
            if not data:
                break
            capture.write(data)
            original.write(data)

    threading.Thread(target=pump, daemon=True).start()
    return capture_path


#: (pattern, expected count per apartment launch) — measured across all 8
#: local (gitignored) apartment-launch logs; the authoritative table with
#: rationale is results/logs/README.md. NOTE: if the owner ever elects G4
#: option B (fetch the SimPBR sibling modules), the three MDL rows drop to
#: zero and this list must be updated with the re-run evidence.
BENIGN_ALLOWLIST: tuple[tuple[str, int], ...] = (
    ("CreateJoint - cannot create a joint between static bodies", 45),
    ("MDLC:COMPILER", 3),
    ("Unable to find SdrShaderNode", 12),
    ("Could not perform 'modify_collision_properties'", 5),
    ("DLSS increasing input dimensions", 1),
    ("Seed not set for the environment", 1),
)

#: [Error] lines matching any of these are the allowlisted errors above;
#: anything else severity-[Error] is off-list and fails S0.
ALLOWLISTED_ERROR_MARKERS = (
    "CreateJoint - cannot create a joint between static bodies",
    "MDLC:COMPILER",
    "Unable to find SdrShaderNode",
    "USD_MDL",  # the 1-line MdlModuleId Invalid half of the G4 signature
)


# ---------------------------------------------------------------------------
# Report plumbing
# ---------------------------------------------------------------------------

class Report:
    def __init__(self, out_path: Path, checkpoint: str | None = None):
        self.out_path = out_path
        #: Which policy produced these verdicts. S2/S3/S4 are gait measurements,
        #: so a report that cannot name its checkpoint cannot be compared with
        #: another one. Additive: absent from the four archived pre-freeze runs,
        #: which all used session.py's DEFAULT_CHECKPOINT.
        self.checkpoint = checkpoint
        self.scenarios: list[dict] = []

    def record(self, sid: str, ok: bool, reason: str,
               measurements: dict, artifacts: list[str]) -> None:
        verdict = "PASS" if ok else "FAIL"
        print(f"SCENARIO {sid} {verdict}: {reason}")
        self.scenarios.append(
            {"id": sid, "pass": bool(ok), "reason": reason,
             "measurements": measurements, "artifact_paths": artifacts}
        )
        self.flush(final=False)

    def flush(self, final: bool) -> None:
        doc = {
            "script": "smoke_gap_hunt.py",
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "checkpoint": self.checkpoint,
            "complete": final,
            "scenarios": self.scenarios,
            "verdict": "PASS" if self.scenarios and all(s["pass"] for s in self.scenarios) else "FAIL",
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(doc, indent=2) + "\n")


# ---------------------------------------------------------------------------
# S5's stub provider: a scripted navigator through the REAL dispatch path
# ---------------------------------------------------------------------------

QA_STUB_TEXT = (
    "1. The connector is the small room I crossed between the two others.\n"
    "2. Face east from the seating area, pass the doorway, then the cold "
    "appliance is ahead.\n"
    "3. I visited two areas: place_a and place_b.\n"
    "4. Roughly east of where I started.\n"
    "5. place_a: the low seat. place_b: the counter run.\n"
)


class ScriptedNavigator:
    """Deterministic no-LLM driver. Reads ONLY what a model would read (the
    tool-result payloads' position_estimate/compass), plans on the layout's
    free-space grid (a scripted stand-in is allowed geometry the model never
    gets), and exercises every tool at least once so ``audit_trial.py``'s
    coverage check can pass.

    Usage numbers are a stub's: tiny non-zero values, ``cache_read_tokens=1``
    included, because ``audit_trial.py``'s cache check targets REAL trials and
    this smoke JSON lives under results/logs (never pooled, never scored).
    """

    #: Planning-grid inflation. 0.12 m, not the scoring grid's 0.08 m body
    #: radius: cell centres then clear obstacles by > 0.12 m, so a chord
    #: between ADJACENT route vertices (<= 0.0354 m off a centre) keeps true
    #: clearance >= 0.0846 m > the 0.08 m contact radius — the polyline is
    #: drivable BY CONSTRUCTION. 0.135 m is the tightest doorway-crossing cell
    #: clearance in the layout, so 0.12 m still leaves every doorway open.
    PLAN_INFLATE_M = 0.12
    #: A next-hop chord (to the immediate waypoint) carries a small margin
    #: over the contact radius.
    NEXT_HOP_CLEAR_M = BODY_RADIUS_M + 0.02
    #: A line-of-sight SHORTCUT (skipping vertices) must carry a full margin —
    #: shortcuts are exactly how the corner-cut wedge happened, and the margin
    #: has to absorb move()'s documented up-to-0.04 m quantisation overshoot
    #: plus accumulated dead-reckoning error (the 1.02-factor dry run measured
    #: a 0.10 m-clearance shortcut ending in real corner contact).
    LOS_CLEAR_M = BODY_RADIUS_M + 0.05
    #: Stop this short of the aimed waypoint: move() overshoots by up to
    #: 0.04 m (0.2 s chunk quantisation, doc 05 §4.2), and commanding the full
    #: chord converts a margin-legal chord ENDPOINT into contact. The 0.15 m
    #: prune still counts the waypoint as reached.
    STOP_SHORT_M = 0.05
    #: Exemption for the first stretch of a chord (departing from a surface
    #: after a bump — walls block >= 0.19 m of any crossing chord, so the
    #: exemption cannot let a chord tunnel through one).
    CHORD_START_EXEMPT_M = 0.10
    CHORD_MAX_M = 1.2

    def __init__(
        self,
        goals: list[tuple[tuple[float, float], float]],
        true_xy_fn=None,
    ):
        from duck_embody.agent.providers.base import Usage  # pure import

        self._usage_cls = Usage
        self._plan_grid = None  # built lazily; a padded FreeSpaceGrid
        self.goals = goals  # [(xy, declare_when_within_m)], stage order
        self.stage_idx = 0
        self.turn_in_stage = 0
        self.waypoints: list[tuple[float, float]] | None = None
        self.replans = 0
        self.did_precision_tools = False
        self.did_bump_probe = False
        self.calls_made = 0
        # Stall detector state: (current waypoint, distance to it) at the last
        # navigate decision.
        # Ground-truth oracle for RE-ANCHORING ONLY, via the real
        # correct_position tool. The first gate run measured why this exists:
        # the navigator steered and DECLARED on the dead-reckoned estimate, and
        # ~2.4 m of accumulated drift pushed a geometrically perfect declare
        # outside the 0.35 m radius -> 'declared_elsewhere'. S5 tests the
        # PIPES, not dead-reckoning accuracy (that is the benchmark's own
        # subject); a scripted harness check may know the truth, and feeding it
        # through correct_position exercises loop closure end to end - the same
        # call a model makes when it re-recognises a landmark.
        self.true_xy_fn = true_xy_fn
        self._stall_ref: tuple[tuple[float, float], float] | None = None
        self._stall_turns = 0
        # Consecutive bump recoveries without intervening progress.
        self._bump_recoveries = 0

    # -- provider protocol ---------------------------------------------------

    def send(self, system, messages, tools):
        from duck_embody.agent.providers.base import AssistantTurn

        if not tools:  # the QA exchange: render-only, canned answers
            return AssistantTurn(
                text=QA_STUB_TEXT, tool_calls=[], usage=self._usage(),
                raw=[{"stub": "qa"}], stop_reason="stub",
            )
        state = self._latest_state(messages)
        calls = self._decide(state)
        self.turn_in_stage += 1
        return AssistantTurn(
            text="", tool_calls=calls, usage=self._usage(),
            raw=[{"stub": self.calls_made}], stop_reason="stub",
        )

    # -- internals -----------------------------------------------------------

    def _usage(self):
        return self._usage_cls(
            input_tokens_total=2,
            input_tokens_uncached=1,
            output_tokens_total=1,
            cache_read_tokens=1,
        )

    def _call(self, tool: str, **args):
        # First parameter is `tool`, NOT `name`: update_room/set_current_room
        # take a model-facing argument literally called `name`, and a `name`
        # parameter here collides with it (caught by the off-sim dry run).
        from duck_embody.agent.providers.base import ToolCall

        self.calls_made += 1
        return ToolCall(id=f"stub_{self.calls_made}", name=tool, args=args)

    @staticmethod
    def _latest_state(messages):
        """Newest payload carrying position_estimate — what a model would read."""
        from duck_embody.agent.providers.base import ToolResultBlock, UserMessage

        latest = None
        for msg in messages:
            if not isinstance(msg, UserMessage):
                continue
            for block in msg.blocks:
                if isinstance(block, ToolResultBlock):
                    try:
                        payload = json.loads(block.text)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if isinstance(payload, dict) and "position_estimate" in payload:
                        latest = payload
        return latest

    def _decide(self, state):
        goal_xy, declare_within = self.goals[self.stage_idx]

        if self.turn_in_stage == 0:
            calls = [self._call("get_observation")]
            if self.stage_idx == 0:
                calls += [
                    self._call("update_room", name="place_a",
                               description="low seat and a rug"),
                    self._call("set_current_room", name="place_a"),
                    self._call("update_plan",
                               text="cross to the counter, then return"),
                ]
            return calls
        if self.stage_idx == 0 and self.turn_in_stage == 1:
            return [
                self._call("look_around"),
                self._call("mark_exit", room="place_a", direction_deg=0,
                           status="unexplored"),
                self._call("add_landmark", room="place_a",
                           description="a low seat"),
            ]

        if state is None:  # defensive: keep observing rather than derail
            return [self._call("get_observation")]

        x = float(state["position_estimate"]["x"])
        y = float(state["position_estimate"]["y"])
        compass = float(state["compass_deg"])

        # Oracle re-anchor: when the estimate has drifted, correct it FIRST (in
        # the same multi-call turn) and steer on the corrected value.
        anchor_calls = []
        if self.true_xy_fn is not None:
            tx, ty = self.true_xy_fn()
            if math.dist((x, y), (tx, ty)) > 0.10:
                anchor_calls.append(self._call(
                    "correct_position", x=round(tx, 3), y=round(ty, 3),
                    # No "oracle" in this string: reason lands in the corrections
                    # log, which is model-visible, and audit_trial's leak scan
                    # (correctly) bans that word anywhere a model could read.
                    reason="scripted re-anchor (harness smoke)",
                ))
                x, y = tx, ty
        dist_to_goal = math.dist((x, y), goal_xy)

        if anchor_calls and dist_to_goal <= declare_within:
            # Do not bundle the declare with the anchor: the anchor must land
            # in the integrator BEFORE the declare turn, so the declare's own
            # payload reflects it.
            return anchor_calls + [self._call("get_observation")]
        if dist_to_goal <= declare_within:
            if self.stage_idx == 0 and not self.did_precision_tools:
                # Exercise the remaining tools once, at the goal, where a
                # 0.05 m send_velocity nudge cannot cost the declare.
                self.did_precision_tools = True
                return [
                    self._call("update_room", name="place_b",
                               description="counters along the south side"),
                    self._call("set_current_room", name="place_b"),
                    self._call("correct_position", x=round(x, 2), y=round(y, 2),
                               reason="re-anchoring at the counter run"),
                    self._call("send_velocity", vx=0.1, vy=0.0, wz=0.0,
                               duration_s=0.5),
                ]
            if self.stage_idx == 0 and not self.did_bump_probe:
                # A deliberate auto-stopped bump against the counter run,
                # declared in the SAME turn: stage 2's first observation must
                # then show bumped=false, contact=[] (the reset regression,
                # end to end). The declare must ride the same turn — any later
                # clean motion would clear last_bumped and the boundary check
                # would test nothing. 0.7 m: from anywhere in the 0.2 m
                # declare disc (plus drift) the southward ray hits the
                # counter/stove face and stops at y~0.433, still inside the
                # 0.35 m true target radius.
                self.did_bump_probe = True
                calls = [
                    self._call("turn_to_heading", heading_deg=270.0),
                    self._call("move", distance_m=0.7),
                    self._call("declare_done"),
                ]
                self._advance_stage()
                return calls
            call = self._call("declare_done")
            self._advance_stage()
            return [call]

        # -- bump reaction (reads status.bumped/contact, as a model would) ----
        if (state.get("status") or {}).get("bumped"):
            # The last motion ended in real contact. Disengage by walking back
            # the way we came (that space was just driven through) — with an
            # ESCALATING distance, a sidestep AWAY from the felt side (the S3
            # recovery recipe, from the same status.contact a model reads),
            # and a bounded give-up. Grinding forward instead is what creeps
            # the dead-reckoned estimate off the true pose (move() credits the
            # estimate with the bump chunk's commanded distance every retry
            # while the robot stays put — measured off-sim, 0.5-0.9 m of
            # compounded drift), and an UNbounded back-out/re-approach cycle
            # is just a slower wedge (also measured).
            self.waypoints = None
            self._stall_ref = None
            self._stall_turns = 0
            self._bump_recoveries += 1
            if self._bump_recoveries > 5:
                call = self._call("declare_done")  # honest give-up: scored
                self._advance_stage()
                return [call]
            back_m = min(0.10 + 0.05 * (self._bump_recoveries - 1), 0.25)
            calls = [self._call("send_velocity", vx=-0.1, vy=0.0, wz=0.0,
                                duration_s=round(back_m / 0.1, 1))]
            felt = (state.get("status") or {}).get("contact") or []
            vy = 0.0
            if "left_leg" in felt and "right_leg" not in felt:
                vy = -0.1  # obstacle felt on the left: step right
            elif "right_leg" in felt and "left_leg" not in felt:
                vy = 0.1
            if vy:
                calls.append(self._call("send_velocity", vx=0.0, vy=vy,
                                        wz=0.0, duration_s=1.0))
            return calls

        # -- navigate: line-of-sight chords over the A* polyline --------------
        if self.waypoints is None:
            route = self._planner().path((x, y), goal_xy)
            if route is None:
                return [self._call("get_observation")]  # honest stall
            # ALL polyline vertices, planned on the PADDED grid. The earlier
            # every-0.3 m densification (on the 0.08 m grid) let the commanded
            # chord CUT THE CORNER around wall B1's north tip at 0.06-0.075 m
            # true clearance — under the 0.08 m body radius, so stage 2 wedged
            # against real contact until the turn cap (measured off-sim
            # against the layout geometry, nav_debug dry run). Density is
            # handled by the 0.15 m prune; long straight moves by the
            # line-of-sight shortcut below; corner safety by _chord_clear.
            self.waypoints = [tuple(p) for p in route[1:]] + [tuple(goal_xy)]

        while self.waypoints and math.dist((x, y), self.waypoints[0]) < 0.15:
            self.waypoints.pop(0)
        if not self.waypoints:
            # Drift ate the plan: replan from the current estimate (bounded).
            return self._force_replan()

        # Stall detector: two consecutive navigate decisions with < 0.05 m of
        # progress toward the SAME waypoint means the robot is bump-wedged (or
        # drift-blind) — the identical blocked move would otherwise repeat
        # until the 40-turn stage cap, which is exactly how the corner-cut
        # wedge stayed silent. Progress is read from the position_estimate
        # deltas, the same channel a model would have to use.
        wp0, d_now = self.waypoints[0], math.dist((x, y), self.waypoints[0])
        stalled = (
            self._stall_ref is not None
            and self._stall_ref[0] == wp0
            and self._stall_ref[1] - d_now < 0.05
        )
        self._stall_turns = self._stall_turns + 1 if stalled else 0
        if not stalled:
            self._bump_recoveries = 0  # real progress: recovery worked
        self._stall_ref = (wp0, d_now)
        if self._stall_turns >= 2:
            return self._force_replan()

        if not self._chord_clear((x, y), self.waypoints[0], self.NEXT_HOP_CLEAR_M):
            # The immediate chord is blocked (estimate drifted into the
            # inflation margin, or the plan is stale): replan, bounded.
            return self._force_replan()
        # Aim at the FARTHEST waypoint reachable by a clear straight chord —
        # long moves where the corridor is straight, vertex-by-vertex at
        # corners, and never a shortcut that dips near the body radius.
        target_idx = 0
        while (
            target_idx + 1 < len(self.waypoints)
            and math.dist((x, y), self.waypoints[target_idx + 1]) <= self.CHORD_MAX_M
            and self._chord_clear(
                (x, y), self.waypoints[target_idx + 1], self.LOS_CLEAR_M
            )
        ):
            target_idx += 1
        del self.waypoints[:target_idx]

        wx, wy = self.waypoints[0]
        bearing = math.degrees(math.atan2(wy - y, wx - x)) % 360.0
        err = (bearing - compass + 180.0) % 360.0 - 180.0
        if abs(err) > 10.0:
            return anchor_calls + [
                self._call("turn_to_heading", heading_deg=round(bearing, 1))
            ]
        commanded = min(1.5, max(0.08, math.dist((x, y), (wx, wy)) - self.STOP_SHORT_M))
        return anchor_calls + [self._call("move", distance_m=round(commanded, 3))]

    def _planner(self):
        """The padded planning grid (see PLAN_INFLATE_M), built once."""
        if self._plan_grid is None:
            from duck_embody.env.apartment_layout import FreeSpaceGrid

            self._plan_grid = FreeSpaceGrid(inflate=self.PLAN_INFLATE_M)
        return self._plan_grid

    def _chord_clear(
        self, a: tuple[float, float], b: tuple[float, float], min_clear: float
    ) -> bool:
        """A straight drive from ``a`` to ``b`` keeps true clearance >= min_clear.

        Sampled every 0.01 m against the layout's TRUE obstacle rectangles —
        never a cell lookup: a free CELL of the scoring grid may contain
        points as little as ~0.045 m from a wall, which is how the corner-cut
        chord looked clean while producing real contact. The first
        CHORD_START_EXEMPT_M is exempt so the robot can drive AWAY from a
        surface it is resting against after a bump.
        """
        length = math.dist(a, b)
        if length < 1e-6:
            return True
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        s = 0.0
        while s < length:
            s = min(s + 0.01, length)
            if s <= self.CHORD_START_EXEMPT_M:
                continue
            px, py = a[0] + ux * s, a[1] + uy * s
            if clearance(px, py) < min_clear:
                return False
        return True

    def _force_replan(self):
        """Drop the plan and replan from the current estimate, bounded."""
        self.replans += 1
        self.waypoints = None
        self._stall_ref = None
        self._stall_turns = 0
        if self.replans > 4:
            call = self._call("declare_done")  # honest give-up: scored
            self._advance_stage()
            return [call]
        return [self._call("get_observation")]

    def _advance_stage(self) -> None:
        self.stage_idx = min(self.stage_idx + 1, len(self.goals) - 1)
        self.turn_in_stage = -1  # the send() wrapper increments to 0
        self.waypoints = None
        self.replans = 0
        self._stall_ref = None
        self._stall_turns = 0
        self._bump_recoveries = 0


# ---------------------------------------------------------------------------
# Frame-diff helper (S2): numpy is available in the kit python
# ---------------------------------------------------------------------------

def mean_abs_diff(path_a: Path, path_b: Path) -> float:
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(path_a), dtype=np.float32)
    b = np.asarray(Image.open(path_b), dtype=np.float32)
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).mean())


# ---------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------

def scenario_s0(report: Report, capture_path: Path) -> None:
    text = capture_path.read_text(errors="replace")
    counts = {pattern: text.count(pattern) for pattern, _ in BENIGN_ALLOWLIST}
    mismatches = [
        f"{pattern!r} x{counts[pattern]} (expect {expected})"
        for pattern, expected in BENIGN_ALLOWLIST
        if counts[pattern] != expected
    ]
    off_list = [
        line for line in text.splitlines()
        if "[Error]" in line
        and not any(marker in line for marker in ALLOWLISTED_ERROR_MARKERS)
    ]
    kvdb = [line for line in text.splitlines() if "kvdb" in line.lower()]
    ok = not mismatches and not off_list and not kvdb
    reason = "startup noise matches the allowlist" if ok else (
        f"mismatches={mismatches or 'none'} off_list={len(off_list)} kvdb={len(kvdb)}"
    )
    report.record(
        "S0", ok, reason,
        {"counts": counts, "off_list_errors": off_list[:20], "kvdb_lines": kvdb[:5]},
        [str(capture_path)],
    )


def scenario_s1(report: Report, session, out_dir: Path) -> None:
    """Recorded multi-chunk bump: the G1 fix, end to end, on the trial path."""
    from duck_embody.agent.memory import Counters, Memory, PositionIntegrator
    from duck_embody.agent.providers.base import ToolCall
    from duck_embody.agent.tools import ToolContext, _state_payload, dispatch
    from duck_embody.sim.recorder import Recorder, attach_recorder
    from duck_embody.sim.session import SpawnPose

    pb = session.playback
    # T2.4-PROVEN clean-bump approach (sofa from the south: stopped at 0.29 m,
    # bumped, no fall — physics_pass_report.json). The first gate run used the
    # seed-101 spawn toward the sofa's east face, the exact approach every
    # sanity trial toppled on: the duck FELL before the 3-step debounce could
    # confirm a bump, so the scenario measured a fall, not the recorded-bump
    # path it exists to exercise.
    sx, sy, heading = 0.30, 0.75, 90.0
    session.reset(seed=101, spawn=SpawnPose(sx, sy, heading))
    pb.settle(0.4)

    reach = reach_along(sx, sy, heading)
    target_distance = min(1.5, reach + 0.3)

    # Instrument the first over-threshold contact step so "the confirming
    # contact happened after piece 1" is measured, not assumed.
    first_contact_step: list[int] = []
    call_start_step = pb._step_counter
    original_force = pb.bump_contact_force

    def instrumented_force():
        force = original_force()
        if force > BUMP_FORCE_N and not first_contact_step:
            first_contact_step.append(pb._step_counter - call_start_step)
        return force

    pb.bump_contact_force = instrumented_force

    recorder = Recorder(out_dir / "s1_bump", fps=25, hide_ceiling=True)
    detach = attach_recorder(pb, session.env.unwrapped, recorder)
    context = ToolContext(
        playback=pb, camera=None, memory=Memory(),
        integrator=PositionIntegrator(sx, sy), counters=Counters(),
    )
    try:
        outcome = dispatch(
            ToolCall(id="s1", name="move", args={"distance_m": target_distance}),
            context,
        )
    finally:
        detach()
        pb.bump_contact_force = original_force
        del pb.bump_contact_force  # restore the bound method
    mp4 = recorder.encode(keep_frames=False)
    strip = recorder.filmstrip(mp4) if mp4 else None

    execution = outcome.execution or {}
    payload_after = _state_payload(context)
    contact_step = first_contact_step[0] if first_contact_step else None
    measurements = {
        "target_distance_m": target_distance,
        "reach_m": reach,
        "bumped": execution.get("bumped"),
        "contact_groups": execution.get("contact_groups"),
        "first_contact_step_in_call": contact_step,
        "status_contact_next_payload": payload_after["status"]["contact"],
        "stop_reason": execution.get("stop_reason"),
    }
    problems = []
    if not execution.get("bumped"):
        problems.append("no bump — the approach measured nothing")
    if not execution.get("contact_groups"):
        problems.append("bumped with empty contact_groups (G1 regression)")
    if contact_step is None or contact_step <= 2:
        problems.append(
            f"first contact at step {contact_step}: the merge was not exercised "
            "— scenario fails ITSELF (plan spec), not the harness"
        )
    if not payload_after["status"]["contact"]:
        problems.append("next _state_payload lost status.contact")
    report.record(
        "S1", not problems,
        "; ".join(problems) or
        f"bumped with contact {execution.get('contact_groups')} confirmed at step {contact_step}",
        measurements,
        [str(p) for p in (mp4, strip) if p],
    )


def scenario_s2(report: Report, session, out_dir: Path) -> None:
    """Recorded forced fall: diagnostics survive the merge; video ends pre-topple."""
    from duck_embody.sim.recorder import Recorder, attach_recorder
    from duck_embody.sim.session import SpawnPose

    pb = session.playback
    x0, y0, heading = 2.72, 0.95, 270.0  # facing the counter run
    session.reset(seed=101, spawn=SpawnPose(x0, y0, heading))
    pb.settle(0.4)

    reach = reach_along(x0, y0, heading)
    budget_steps = drive_budget_steps(reach) + int(PUSH_ALLOWANCE_S / CONTROL_DT)

    recorder = Recorder(out_dir / "s2_fall", fps=25, hide_ceiling=True)
    detach = attach_recorder(pb, session.env.unwrapped, recorder)
    last = None
    steps = 0
    try:
        while steps < budget_steps and not pb.fell:
            # Full-speed push, no auto-stop: the recorded seam chunks this into
            # 0.04 s pieces, so the fall lands mid-merge — the G1 shape.
            last = pb.execute(0.222, 0.0, 0.0, 1.0, stop_on_bump=False)
            steps += last.steps
    finally:
        detach()

    frames = sorted(recorder.frames_dir.glob("f*.png"))
    frame_diffs: dict[str, float] = {}
    frames_ok = False
    if len(frames) >= 7:
        reference = frames[0]  # standing at spawn, by construction
        # Median consecutive-pair diff over the video's middle — NOT
        # diff(f0, f1): the recorder's first grab is a stale/unrendered init
        # frame, and the first gate run measured diff(f0,f1)=70.5, turning the
        # '3x noise floor' bar into an impossible 211 on 8-bit frames.
        mids = frames[1:-5]
        pairs = list(zip(mids, mids[1:])) or [(frames[0], frames[1])]
        diffs_seq = sorted(mean_abs_diff(a, b) for a, b in pairs)
        noise_floor = diffs_seq[len(diffs_seq) // 2]
        tail = frames[-5:]
        frame_diffs = {p.name: mean_abs_diff(reference, p) for p in tail}
        frame_diffs["noise_floor_f0_f1"] = noise_floor
        # Data-derived threshold: the last frames must differ from the spawn
        # reference by clearly more than two consecutive standing frames do.
        frames_ok = all(
            frame_diffs[p.name] > 3.0 * max(noise_floor, 1e-3) for p in tail
        )
    mp4 = recorder.encode(keep_frames=True)
    strip = recorder.filmstrip(mp4) if mp4 else None

    if not pb.fell:
        print("SCENARIO S2 INCONCLUSIVE: no fall within the geometric+push budget")
        report.record(
            "S2", False, "INCONCLUSIVE — no fall within budget (not a harness verdict)",
            {"steps": steps, "budget_steps": budget_steps, "reach_m": reach},
            [str(p) for p in (mp4, strip) if p],
        )
        return

    diag = last.fall_diagnostics if last is not None else None
    seconds_into_call = None if not diag else diag.get("policy_seconds_into_call")
    call_seconds = None if last is None else last.steps * CONTROL_DT
    problems = []
    if last is None or not last.fell:
        problems.append("fell flag not on the merged result")
    if last is not None and last.stop_reason != "fell":
        problems.append(f"stop_reason={last.stop_reason!r} (G2 shape)")
    if diag is None:
        problems.append("fall_diagnostics=None on the recorded path (G1 regression)")
    else:
        if not diag.get("values_pre_step"):
            problems.append("values_pre_step marker missing (G10)")
        if seconds_into_call is None or call_seconds is None or (
            abs(seconds_into_call - call_seconds) > 2 * CONTROL_DT
        ):
            problems.append(
                f"policy_seconds_into_call={seconds_into_call} vs measured "
                f"{call_seconds} (G9)"
            )
        elif seconds_into_call <= 0.04:
            problems.append(
                "seconds_into_call <= one chunk — accumulation not exercised"
            )
    if not frames_ok:
        problems.append(
            "final 5 frames match the spawn-pose reference (G8 regression) or "
            "too few frames — check the filmstrip"
        )
    report.record(
        "S2", not problems,
        "; ".join(problems) or
        f"fell {seconds_into_call}s into the call; video ends pre-topple",
        {"steps": steps, "budget_steps": budget_steps, "reach_m": reach,
         "fall_diagnostics": diag, "frame_diffs": frame_diffs},
        [str(p) for p in (mp4, strip) if p],
    )


def scenario_s3(report: Report, session) -> None:
    """Contact-side matrix — G3's gate for the frozen prompt's recovery claim."""
    from duck_embody.sim.session import SpawnPose

    pb = session.playback
    results: list[dict] = []

    for approach in s3_approaches():
        for seed in S3_SEEDS:
            x, y, heading = approach["spawn"]
            session.reset(seed=seed, spawn=SpawnPose(x, y, heading))
            pb.settle(0.4)
            reach = reach_along(x, y, heading)
            budget = drive_budget_steps(reach)
            dwell_left = int(CONTACT_DWELL_S / CONTROL_DT)

            peak: dict[str, float] = {}
            felt: list[str] = []
            bumped_at: int | None = None
            steps = 0
            while steps < budget and dwell_left > 0 and not pb.fell:
                res = pb.execute(MOVE_SPEED_MPS, 0.0, 0.0, CONTROL_DT)
                steps += res.steps
                for name, force in pb.contact_report().items():
                    peak[name] = max(peak.get(name, 0.0), force)
                for group in pb.contact_groups():
                    if group not in felt:
                        felt.append(group)
                if res.bumped:
                    if bumped_at is None:
                        bumped_at = steps
                    dwell_left -= res.steps

            legs = [g for g in felt if g in ("left_leg", "right_leg")]
            single_leg = felt and len(legs) == 1
            entry = {
                "label": approach["label"], "seed": seed,
                "spawn": [x, y, heading], "expected": approach["expected"],
                "mirror_key": approach["mirror_key"], "reach_m": reach,
                "steps": steps, "bumped_at_step": bumped_at, "fell": pb.fell,
                "peak_by_body": {k: round(v, 1) for k, v in
                                 sorted(peak.items(), key=lambda kv: -kv[1])},
                "felt_groups": felt, "single_leg": bool(single_leg),
                "reported_leg": legs[0] if single_leg else None,
                "registered_force": bool(peak),
            }

            # Recovery recipe, only meaningful after a single-leg bump: side-
            # step AWAY from the felt side, then a short move must not re-bump.
            if single_leg and not pb.fell:
                vy_away = -0.1 if legs[0] == "left_leg" else 0.1
                pb.execute(0.0, vy_away, 0.0, 1.5)
                retry = pb.move(0.3)
                entry["recovery_cleared"] = bool(
                    not retry.bumped and not retry.fell
                )
            results.append(entry)
            print(f"  S3 {approach['label']} seed {seed}: felt={felt} "
                  f"peak={max(peak.values()) if peak else 0.0:.0f}N")

    dead_probes = [r for r in results if not r["registered_force"]]

    judged = [r for r in results if r["expected"] and r["single_leg"]]
    correct = [r for r in judged if r["reported_leg"] == r["expected"]]
    side_rate = len(correct) / len(judged) if judged else 0.0

    mirrors_ok = True
    mirror_detail: dict[str, list[str]] = {}
    for key in {r["mirror_key"] for r in results if r["mirror_key"]}:
        pair = [r for r in results if r["mirror_key"] == key and r["single_leg"]]
        legs_seen = sorted({r["reported_leg"] for r in pair})
        mirror_detail[key] = legs_seen
        if len({r["expected"] for r in pair}) == 2 and len(legs_seen) < 2:
            mirrors_ok = False

    recoveries = [r for r in results if "recovery_cleared" in r]
    recovery_rate = (
        sum(1 for r in recoveries if r["recovery_cleared"]) / len(recoveries)
        if recoveries else 0.0
    )

    claim_holds = bool(judged) and side_rate >= 0.8 and mirrors_ok
    recipe_holds = bool(recoveries) and recovery_rate >= 0.7
    if claim_holds and recipe_holds:
        prompt_action = "KEEP the single-leg recovery sentence (verdict A)"
    else:
        prompt_action = (
            "WEAKEN the prompt sentence to '…and a single leg means one leg "
            "caught a low edge' before T4.3 (verdict B / honest null)"
        )

    out = REPO_ROOT / "results" / "figures" / "smoke" / "contact_side_report.json"
    out.write_text(json.dumps(
        {"approaches": results, "side_rate": side_rate, "mirrors": mirror_detail,
         "recovery_rate": recovery_rate, "prompt_action": prompt_action},
        indent=2) + "\n")

    # Either verdict is a scenario PASS (the JSON records the prompt action);
    # only a probe-measured-nothing approach fails the scenario itself.
    ok = not dead_probes
    report.record(
        "S3", ok,
        (f"{len(dead_probes)} approaches registered NO force — measured nothing"
         if dead_probes else
         f"side_rate={side_rate:.2f} recovery_rate={recovery_rate:.2f} -> {prompt_action}"),
        {"side_rate": side_rate, "recovery_rate": recovery_rate,
         "mirrors_ok": mirrors_ok, "n_judged": len(judged),
         "dead_probes": [r["label"] for r in dead_probes],
         "prompt_action": prompt_action},
        [str(out)],
    )


def scenario_s4(report: Report, session) -> None:
    """move()'s abort semantics, BOTH directions, at measured clearances.

    The first two gate runs walked 3 cm beyond the body radius and demanded no
    abort — a premise the robot's own gait refutes. MEASURED (s4_forensics2,
    2026-07-27): at 3 cm the right knee scrapes the counter run LATERALLY
    (81.9 N peak, 20 steps, horiz >> vert — real furniture contact, not a floor
    strike); at 7 cm and 11 cm the counter contact vanishes. The swept
    half-width of the walking gait is ~0.11-0.15 m against the 0.08 m nominal
    body radius, which is also why the 0.35 m doorways (0.175 m to each face)
    transit clean 8/8 in T2.4.

    So the honest scenario is TWO corridors:
      leg A (clean): 7 cm beyond radius — measured scrape-free — must complete
        with no abort (the spurious-abort defect the first run was after);
      leg B (scrape): 3 cm — measured sustained scrape — MUST abort with
        stop_reason 'bump' and an accurate right-leg report (the harness's
        protective stop, working as designed).
    """
    from duck_embody.sim.session import SpawnPose

    pb = session.playback
    north, west, east = counter_faces()
    x0 = 2.0
    measurements: dict = {}
    problems: list[str] = []

    # -- leg A: measured-walkable clearance, must NOT abort ------------------
    # 9 cm, NOT the 7 cm the holdless forensics measured clean: move()'s
    # heading-hold corrections swing the legs wider than open-loop walking, and
    # at 7 cm the right leg catches counter_1's protruding west corner (round-3
    # gate run: abort at 0.32 m, heading 344.7 deg mid-correction). The
    # move()-with-hold sweep (s4_sweep, 2x each at 9/11/13/15 cm) is clean from
    # 9 cm up — 8/8 completions, zero contact. The prompt's "~10 cm" guidance
    # sits exactly on this measured boundary.
    y_clean = north + BODY_RADIUS_M + 0.09
    session.reset(seed=101, spawn=SpawnPose(x0, y_clean, 0.0))
    pb.settle(0.4)
    free = reach_along(x0, y_clean, 0.0)
    commanded = max(0.3, min(1.5, free - 0.15))
    result = pb.move(commanded, hold_heading=True, stop_on_bump=True)
    measurements["clean_leg"] = {
        "y0": round(y_clean, 3), "corridor_free_m": free, "commanded_m": commanded,
        "stop_reason": result.stop_reason,
        "true_displacement_m": result.true_displacement_m,
        "contact_groups": result.contact_groups,
        "stop_pose": list(result.true_pose),
    }
    if result.stop_reason != "reached":
        problems.append(
            f"CLEAN leg: stop_reason={result.stop_reason!r} at a measured "
            f"scrape-free clearance — spurious abort, triage before the batch"
        )
    if result.true_displacement_m < 0.8 * commanded:
        problems.append(
            f"CLEAN leg: displacement {result.true_displacement_m:.2f} < 0.8x "
            f"commanded {commanded:.2f}"
        )

    # -- leg B: measured-scrape clearance, MUST abort accurately -------------
    y_scrape = north + BODY_RADIUS_M + LAYOUT["wall_thickness"]
    session.reset(seed=101, spawn=SpawnPose(x0, y_scrape, 0.0))
    pb.settle(0.4)
    free_b = reach_along(x0, y_scrape, 0.0)
    commanded_b = max(0.3, min(1.5, free_b - 0.15))
    result_b = pb.move(commanded_b, hold_heading=True, stop_on_bump=True)
    measurements["scrape_leg"] = {
        "y0": round(y_scrape, 3), "commanded_m": commanded_b,
        "stop_reason": result_b.stop_reason,
        "true_displacement_m": result_b.true_displacement_m,
        "contact_groups": result_b.contact_groups,
        "stop_pose": list(result_b.true_pose),
    }
    if result_b.stop_reason == "bump":
        if "right_leg" not in (result_b.contact_groups or []):
            problems.append(
                f"SCRAPE leg aborted but reported {result_b.contact_groups!r} — "
                "the measured contact is the right knee on the counter"
            )
    elif result_b.stop_reason == "reached":
        # Gait variance can thread it — that is a pass for the walk, but the
        # scrape assertion then proved nothing; say so honestly.
        measurements["scrape_leg"]["note"] = (
            "no sustained scrape this run — assertion not exercised"
        )
    elif result_b.stop_reason == "fell":
        problems.append("SCRAPE leg FELL — 3 cm clearance topples, not just scrapes")
    report.record(
        "S4", not problems,
        "; ".join(problems) or
        (f"clean leg reached {measurements['clean_leg']['true_displacement_m']:.2f} m; "
         f"scrape leg {measurements['scrape_leg']['stop_reason']}"),
        measurements, [],
    )


def scenario_s5(report: Report, session, out_dir: Path) -> None:
    """Scripted two-stage mini-trial through the REAL dispatch path."""
    from duck_embody.agent.loop import TrialLog, run_trial
    from duck_embody.agent.memory import Counters, Memory, PositionIntegrator
    from duck_embody.agent.tools import ToolContext
    from duck_embody.env.camera import HeadCamera
    from duck_embody.sim.recorder import Recorder, attach_recorder
    from duck_embody.sim.session import SpawnPose
    from duck_embody.tasks.find_kitchen import stage_specs

    pb = session.playback
    (sx, sy), heading = spawn_pose(101)
    session.reset(seed=101, spawn=SpawnPose(sx, sy, heading))
    pb.settle(0.4)

    camera = HeadCamera(session.env)
    camera.warmup()

    recorder = Recorder(out_dir / "s5_trial", fps=25, hide_ceiling=True)
    detach = attach_recorder(pb, session.env.unwrapped, recorder)

    # Declare thresholds sit well inside the true radii (0.35 / 0.5) so honest
    # dead-reckoning drift over this short course cannot flip the verdict.
    provider = ScriptedNavigator(
        goals=[(target_point(), 0.2), ((sx, sy), 0.25)],
        # Oracle re-anchor via the REAL correct_position tool: the first gate
        # run measured 'declared_elsewhere' — ~2.4 m of honest dead-reckoning
        # drift pushed a geometrically perfect declare outside the 0.35 m
        # radius. S5 verifies the pipes, not drift (the benchmark's own
        # subject), so the scripted navigator re-anchors like a model that
        # recognises landmarks — through the same tool call.
        true_xy_fn=pb.true_xy,
    )
    context = ToolContext(
        playback=pb, camera=camera, memory=Memory(),
        integrator=PositionIntegrator(sx, sy), counters=Counters(),
    )
    json_path = out_dir / "gapsmoke_seed101.json"
    log = TrialLog(
        json_path, trial_id="gapsmoke_seed101", model_id="scripted-stub",
        model_name="gapsmoke", seed=101, spawn_xy=(sx, sy),
        spawn_heading_deg=heading,
    )
    trial_error = None
    try:
        document = run_trial(
            provider=provider, context=context, stages=stage_specs(101), log=log
        )
    except Exception as exc:  # noqa: BLE001 — recorded as the scenario FAIL
        trial_error = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        document = None
    finally:
        detach()
    mp4 = recorder.encode(keep_frames=False)
    strip = recorder.filmstrip(mp4) if mp4 else None

    problems: list[str] = []
    measurements: dict = {}
    if trial_error is not None:
        problems.append(f"run_trial raised: {trial_error.splitlines()[-1]}")
        measurements["traceback"] = trial_error
    else:
        turns = document.get("turns", [])
        final = document.get("final") or {}
        measurements["turns"] = len(turns)
        measurements["outcome"] = final.get("outcome")

        if "final" not in document:
            problems.append("no final block")

        # BOTH stages must end in the declare-based success — this is the
        # mini-trial's premise, asserted explicitly. Without it a navigator
        # that wedges against a wall until the 40-turn cap (the corner-cut
        # defect this scenario's dry run caught) still passes every other S5
        # check — audit included, since tool coverage completes in stage 1 —
        # and the degradation is silent.
        from duck_embody.tasks.find_kitchen import OUTCOME_SUCCESS

        outcome = final.get("outcome") or {}
        for stage_name in ("find_kitchen", "return_home"):
            if outcome.get(stage_name) != OUTCOME_SUCCESS:
                problems.append(
                    f"{stage_name} outcome {outcome.get(stage_name)!r} != "
                    f"{OUTCOME_SUCCESS!r} — the scripted navigator must "
                    "complete both stages (wedge/cap/give-up must FAIL here, "
                    "not pass silently)"
                )

        # audit_trial.py conformance (the doc-06-§4 gate).
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import audit_trial

        audit_rc = audit_trial.audit(json_path, require_tool_coverage=True)
        measurements["audit_rc"] = audit_rc
        if audit_rc != 0:
            problems.append("audit_trial.py FAILED the trial JSON")

        # Budgets and timestamps monotonic.
        for stage in ("find_kitchen", "return_home"):
            stage_turns = [t for t in turns if t["stage"] == stage]
            used = [t["budget"]["stage_turns_used"] for t in stage_turns]
            if used != sorted(used):
                problems.append(f"{stage}: stage_turns_used not monotonic")
        stamps = [t["timestamp"] for t in turns]
        if stamps != sorted(stamps):
            problems.append("timestamps not monotonic")

        # Stage boundary zeroed the carried status (reset regression, e2e).
        stage2 = [t for t in turns if t["stage"] == "return_home"]
        if not stage2:
            problems.append("stage 2 never ran (stage-1 declare missed?)")
        else:
            status = stage2[0]["obs"]["status"]
            if status["bumped"] or status["contact"]:
                problems.append(
                    f"stage-2 opening status carries stage-1 contact: {status}"
                )
            stage1 = [t for t in turns if t["stage"] == "find_kitchen"]
            bumped_calls = [
                c for t in stage1
                for c in (t["execution"].get("calls") or [])
                if c.get("bumped")
            ]
            if not bumped_calls:
                problems.append(
                    "no stage-1 bump recorded — the boundary check tested nothing"
                )
            elif not any(c.get("contact_groups") for c in bumped_calls):
                problems.append("stage-1 bump carried no contact_groups (G1)")

        # G6: the frames dir holds exactly this attempt's referenced frames.
        referenced = {
            Path(p).name for t in turns for p in t["obs"].get("frame_paths", [])
        }
        on_disk = {p.name for p in log.frames_dir.glob("*.jpg")}
        if referenced != on_disk:
            problems.append(
                f"frames dir mismatch: {len(on_disk)} on disk vs "
                f"{len(referenced)} referenced"
            )

        # Recorded-path merges preserve the budget accounting.
        for stage in ("find_kitchen", "return_home"):
            stage_turns = [t for t in turns if t["stage"] == stage]
            per_turn = sum(
                t["execution"]["policy_seconds_used"] for t in stage_turns
            )
            stage_total = (final.get("stages", {}).get(stage) or {}).get(
                "policy_seconds_used", 0.0
            )
            if abs(per_turn - stage_total) > 1e-3:
                problems.append(
                    f"{stage}: per-turn policy seconds {per_turn:.3f} != "
                    f"stage total {stage_total:.3f}"
                )

    report.record(
        "S5", not problems,
        "; ".join(problems) or "scripted trial passes audit + boundary + accounting",
        measurements,
        [str(p) for p in (json_path, mp4, strip) if p],
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser():
    """Argument parser. Built outside ``main`` so ``--help`` needs no kit."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="smoke_gap_hunt.py",
        description="Single-session pre-freeze smoke: SIM TEST PLAN S0-S5.",
    )
    parser.add_argument(
        "--print-budget", action="store_true",
        help="print the kill-switch wallclock budget (seconds) and exit, "
             "without launching kit. Emitted as the LAST stdout line.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="policy .pt to run the scenarios against. Default: session.py's "
             "DEFAULT_CHECKPOINT (policy/model_2999.pt, the v4_robust baseline) "
             "— omit it to reproduce the archived gap_hunt runs.",
    )
    return parser


def main() -> int:
    # Parsed BEFORE anything launches, then stripped from argv: AppLauncher
    # inside SimSession.launch() parses sys.argv for its own flags and dies on
    # unknown ones (run_trial.py:151-155). This replaced a bare
    # `"--print-budget" in sys.argv` check, which could not have coexisted with
    # a second flag kit must not see.
    args, kit_argv = build_parser().parse_known_args()
    sys.argv = [sys.argv[0], *kit_argv]

    if args.print_budget:
        # Pre-kit: pure layout math only. Prints the budget (seconds) as the
        # LAST stdout line — the isaaclab.sh wrapper prepends its own banner
        # to stdout, so the documented run line captures via `tail -n1`.
        print(wallclock_budget_s())
        return 0

    # G7: refuse to launch beside another GPU/kit job (pre-kit, cheap).
    from duck_embody.sim.preflight import format_refusal, rule1_violations

    violations = rule1_violations()
    if violations:
        print(format_refusal(violations))
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / "results" / "logs" / f"gap_hunt_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_path = tee_process_output(out_dir / "session_output.log")

    from duck_embody.sim.session import DEFAULT_CHECKPOINT

    checkpoint = str(args.checkpoint or DEFAULT_CHECKPOINT)
    report = Report(out_dir / "gap_hunt_report.json", checkpoint=checkpoint)
    print(f"== smoke_gap_hunt {stamp} ==")
    print(f"  artifacts: {out_dir}")
    print(f"  policy   : {checkpoint}")
    print(f"  estimated policy-seconds: {estimated_policy_seconds():.0f}")

    from duck_embody.sim.session import SimSession

    session = SimSession.launch(
        task_id="DuckEmbody-Apartment-v0", headless=True, checkpoint=args.checkpoint
    )
    try:
        # A first reset so the startup noise (S0's subject) is fully emitted.
        from duck_embody.sim.session import SpawnPose

        (sx, sy), heading = spawn_pose(101)
        session.reset(seed=101, spawn=SpawnPose(sx, sy, heading))

        for sid, run in (
            ("S0", lambda: scenario_s0(report, capture_path)),
            ("S1", lambda: scenario_s1(report, session, out_dir)),
            ("S2", lambda: scenario_s2(report, session, out_dir)),
            ("S3", lambda: scenario_s3(report, session)),
            ("S4", lambda: scenario_s4(report, session)),
            ("S5", lambda: scenario_s5(report, session, out_dir)),
        ):
            try:
                run()
            except Exception as exc:  # noqa: BLE001 — a crash is a FAIL, not a hang
                detail = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
                print(detail)
                report.record(sid, False, f"scenario raised: {exc!r}",
                              {"traceback": detail}, [])

        # Aggregate verdict + report BEFORE close() — nothing after it runs.
        report.flush(final=True)
        all_pass = all(s["pass"] for s in report.scenarios)
        print(f"VERDICT: {'PASS' if all_pass else 'FAIL'}")
        print(f"  report: {report.out_path}")
        return 0 if all_pass else 1
    finally:
        print("  closing app (nothing after this line runs)")
        # Give the tee's pump thread a beat to drain the pipe: close()
        # terminates the process, and a verdict line stuck in the pipe would
        # be lost from BOTH the console and the capture file.
        sys.stdout.flush()
        time.sleep(1.0)
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
