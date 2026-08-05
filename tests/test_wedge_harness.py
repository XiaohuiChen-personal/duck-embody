"""B3 + A1 furniture-wedge harness pins (no kit).

B3: pre-latched ``sustained_contact`` must not abort ``execute`` on step 0/1;
rising-edge into sustained during the call still stops; reconfirm after
``CONTACT_SUSTAINED_STEPS`` of this call's own sustained steps still stops.

A1: ``status.progress`` aggregates consecutive blocked no-progress motions.
"""

from __future__ import annotations

import inspect

from duck_embody.agent.tools import (
    NO_PROGRESS_EPSILON_M,
    NO_PROGRESS_STREAK_THRESHOLD,
    dispatch,
    status_payload,
)
from duck_embody.sim.policy_wrapper import (
    BUMP_DEBOUNCE_STEPS,
    CONTACT_REVERSE_GRACE_STEPS,
    CONTACT_SUSTAINED_STEPS,
    CONTROL_DT,
    duration_to_steps,
)

from tests.test_tools import FakePlayback, call, make_context
from tests.test_wrapper_math import _curve, _scripted_playback


def _contact_playback(*, force_n: float, prelatched: bool = False):
    """Real ``execute`` loop against a fixed pose; contact force is injectable."""
    # Long enough for rising-edge + reverse-grace reconfirm windows.
    traj = _curve(CONTACT_REVERSE_GRACE_STEPS * 4)
    pb = _scripted_playback(traj, seed=101)
    force = {"n": force_n}
    pb.bump_contact_force = lambda: force["n"]
    pb.contact_groups = lambda: ["torso"] if force["n"] > 0 else []
    if prelatched:
        # Leave the machine in the same state a prior bump-stop would.
        pb._contact_state = "sustained_contact"
        pb._bump_run = CONTACT_SUSTAINED_STEPS
        pb._contact_event_id_counter = 1
        pb._contact_event_id = 1
        pb._contact_event_onset_step = 1
        pb._contact_event_regions = ["torso"]
        pb._contact_candidate_regions = ["torso"]
        pb._last_contact_event = {
            "contact_event_id": 1,
            "onset_step": 1,
            "release_step": None,
            "regions": ["torso"],
            "state": "sustained_contact",
        }
    return pb, force


class TestB3PreLatchedStopPolicy:
    def test_prelatched_reverse_does_not_break_on_step_0_or_1(self):
        """The R3 defect: already-latched + stop_on_bump aborted after ~1 step."""
        pb, force = _contact_playback(force_n=0.0, prelatched=True)
        # Clear force immediately (space behind); reverse must be allowed to run
        # past the old step-0/1 abort and into the release window.
        duration_s = (CONTACT_SUSTAINED_STEPS + 2) * CONTROL_DT
        result = pb.execute(-0.10, 0.0, 0.0, duration_s, stop_on_bump=True)

        assert result.steps > 2, (
            f"pre-latched reverse aborted after {result.steps} step(s); "
            "B3 must not stop solely on prior latch"
        )
        assert result.stop_reason != "sustained_contact"
        assert result.contact_state == "free"

    def test_prelatched_still_touching_reconfirms_after_sustained_steps(self):
        pb, force = _contact_playback(force_n=500.0, prelatched=True)
        duration_s = (CONTACT_SUSTAINED_STEPS + 5) * CONTROL_DT
        result = pb.execute(0.2, 0.0, 0.0, duration_s, stop_on_bump=True)

        assert result.stop_reason == "sustained_contact"
        assert result.stopped_early is True
        assert result.steps == CONTACT_SUSTAINED_STEPS

    def test_rising_edge_into_sustained_still_stops(self):
        pb, force = _contact_playback(force_n=500.0, prelatched=False)
        # Free → debounce → candidate → sustained needs CONTACT_SUSTAINED_STEPS
        # of continuous contact from onset (after BUMP_DEBOUNCE gets us into
        # candidate). Provide enough duration for the rising edge.
        duration_s = (CONTACT_SUSTAINED_STEPS + BUMP_DEBOUNCE_STEPS + 5) * CONTROL_DT
        result = pb.execute(0.2, 0.0, 0.0, duration_s, stop_on_bump=True)

        assert result.stop_reason == "sustained_contact"
        assert result.stopped_early is True
        assert result.contact_state == "sustained_contact"
        # Must stop at/near the rising edge, not run the full duration.
        assert result.steps < duration_to_steps(duration_s)

    def test_send_velocity_still_ignores_bump_stop(self):
        pb, force = _contact_playback(force_n=500.0, prelatched=True)
        duration_s = (CONTACT_SUSTAINED_STEPS + 5) * CONTROL_DT
        result = pb.execute(0.2, 0.0, 0.0, duration_s, stop_on_bump=False)

        assert result.stop_reason == ""
        assert result.stopped_early is False
        assert result.steps == duration_to_steps(duration_s)

    def test_execute_stop_predicate_is_rising_edge_or_reconfirm(self):
        """Replace the brittle source-assert that required the old one-liner."""
        from duck_embody.sim import policy_wrapper

        exec_src = inspect.getsource(policy_wrapper.PolicyPlayback.execute)
        assert "began_in_sustained" in exec_src
        assert "rose_into_sustained" in exec_src
        assert "reconfirmed" in exec_src
        assert "sustained_steps_this_call" in exec_src
        assert "if rose_into_sustained or reconfirmed:" in exec_src
        # Rising-edge is onset from free/candidate_contact only — not
        # candidate_release (same-event hysteresis).
        assert '"free"' in exec_src and '"candidate_contact"' in exec_src
        assert "candidate_release" in exec_src  # named in the guard comment

    def test_force_trough_same_event_does_not_rising_edge_abort_reverse(self):
        """candidate_release → sustained is hysteresis, not a new face.

        Bouncing contact (furniture wedge) produces short force troughs. Treating
        re-entry as rising-edge aborted reverse at step ~5 — reintroducing the
        mm-scale backup no-op B3 exists to fix.
        """
        pb, _force = _contact_playback(force_n=500.0, prelatched=True)
        counter = {"i": 0}
        held = {"f": 500.0}

        def scheduled_force():
            counter["i"] += 1
            held["f"] = 0.0 if counter["i"] == 4 else 500.0
            return held["f"]

        pb.bump_contact_force = scheduled_force
        pb.contact_groups = lambda: ["torso"] if held["f"] > 0 else []

        duration_s = (CONTACT_REVERSE_GRACE_STEPS + 5) * CONTROL_DT
        result = pb.execute(-0.10, 0.0, 0.0, duration_s, stop_on_bump=True)

        assert result.steps > 5, (
            f"same-event trough aborted reverse after {result.steps} step(s); "
            "candidate_release→sustained must not rising-edge stop"
        )
        # Reverse reconfirms after CONTACT_REVERSE_GRACE_STEPS (not the shorter
        # forward window); the trough pauses the sustained counter by one.
        assert result.stop_reason == "sustained_contact"
        assert result.steps == CONTACT_REVERSE_GRACE_STEPS + 1

    def test_prelatched_reverse_survives_forward_reconfirm_window(self):
        """Kit smoke gate: 0.4 s reverse cap is ≤0.04 m — must use grace."""
        pb, _force = _contact_playback(force_n=500.0, prelatched=True)
        duration_s = (CONTACT_SUSTAINED_STEPS + 5) * CONTROL_DT
        result = pb.execute(-0.10, 0.0, 0.0, duration_s, stop_on_bump=True)

        assert result.steps > CONTACT_SUSTAINED_STEPS, (
            f"reverse reconfirmed at forward window ({result.steps} steps); "
            "CONTACT_REVERSE_GRACE_STEPS must apply when vx < 0"
        )
        assert result.stop_reason != "sustained_contact"

    def test_prelatched_reverse_eventually_reconfirms_after_grace(self):
        pb, _force = _contact_playback(force_n=500.0, prelatched=True)
        duration_s = (CONTACT_REVERSE_GRACE_STEPS + 5) * CONTROL_DT
        result = pb.execute(-0.10, 0.0, 0.0, duration_s, stop_on_bump=True)

        assert result.stop_reason == "sustained_contact"
        assert result.steps == CONTACT_REVERSE_GRACE_STEPS

    def test_forward_reconfirm_window_unchanged(self):
        """Forward still reconfirms at CONTACT_SUSTAINED_STEPS (0.4 s)."""
        assert CONTACT_SUSTAINED_STEPS == round(0.4 * (1.0 / CONTROL_DT))
        assert CONTACT_REVERSE_GRACE_STEPS > CONTACT_SUSTAINED_STEPS

    def test_move_accepts_hold_heading_deg_override(self):
        from duck_embody.sim import policy_wrapper

        src = inspect.getsource(policy_wrapper.PolicyPlayback.move)
        assert "hold_heading_deg" in src
        assert "wrap_deg(float(hold_heading_deg))" in src


class TestA1ProgressStreak:
    def test_three_blocked_motions_set_no_progress(self):
        playback = FakePlayback()
        playback.bumped = True
        playback.stop_after_chunks = 1  # 0.04 m forward ≤ ε
        context = make_context(playback=playback)

        for _ in range(NO_PROGRESS_STREAK_THRESHOLD):
            dispatch(call("move", distance_m=0.4), context)

        progress = status_payload(context)["progress"]
        assert progress["consecutive_no_progress"] == NO_PROGRESS_STREAK_THRESHOLD
        assert progress["no_progress"] is True
        assert progress["last_measured_m"] <= NO_PROGRESS_EPSILON_M
        assert "hint" in progress

    def test_measured_above_eps_without_block_resets_streak(self):
        playback = FakePlayback()
        playback.bumped = True
        playback.stop_after_chunks = 1
        context = make_context(playback=playback)

        for _ in range(NO_PROGRESS_STREAK_THRESHOLD):
            dispatch(call("move", distance_m=0.4), context)
        assert context.consecutive_no_progress == NO_PROGRESS_STREAK_THRESHOLD

        playback.bumped = False
        playback.stop_after_chunks = None
        dispatch(call("move", distance_m=0.4), context)

        progress = status_payload(context)["progress"]
        assert progress["consecutive_no_progress"] == 0
        assert progress["no_progress"] is False
        assert progress["last_measured_m"] > NO_PROGRESS_EPSILON_M
        assert "hint" not in progress

    def test_stage_reset_clears_streak(self):
        playback = FakePlayback()
        playback.bumped = True
        playback.stop_after_chunks = 1
        context = make_context(playback=playback)
        for _ in range(NO_PROGRESS_STREAK_THRESHOLD):
            dispatch(call("move", distance_m=0.4), context)
        assert context.consecutive_no_progress == NO_PROGRESS_STREAK_THRESHOLD

        context.reset_for_stage()
        assert context.consecutive_no_progress == 0
        assert status_payload(context)["progress"]["no_progress"] is False

    def test_turn_while_bumped_does_not_inflate_streak(self):
        """Recovery pattern bump→turn→backup must not trip no_progress early.

        turn_to_heading reports distance_moved_m=0 and may surface bumped from a
        sticky latch while counts_bump=False — that is not a blocked translation.
        """
        playback = FakePlayback()
        playback.bumped = True
        playback.stop_after_chunks = 1
        context = make_context(playback=playback)

        dispatch(call("move", distance_m=0.4), context)
        assert context.consecutive_no_progress == 1
        dispatch(call("turn_to_heading", heading_deg=90.0), context)
        assert context.consecutive_no_progress == 1
        dispatch(call("move", distance_m=0.4), context)
        assert context.consecutive_no_progress == 2
        assert status_payload(context)["progress"]["no_progress"] is False

    def test_mid_streak_tiny_unblocked_move_holds_streak(self):
        """measured≤ε without block neither increments nor resets (hysteresis)."""
        playback = FakePlayback()
        playback.bumped = True
        playback.stop_after_chunks = 1
        context = make_context(playback=playback)
        for _ in range(2):
            dispatch(call("move", distance_m=0.4), context)
        assert context.consecutive_no_progress == 2

        playback.bumped = False
        playback.stop_after_chunks = 1  # 0.04 m ≤ ε, unblocked
        dispatch(call("move", distance_m=0.4), context)
        assert context.consecutive_no_progress == 2
