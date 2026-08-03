"""Unit tests for the kit-free math in ``duck_embody.sim.policy_wrapper``.

These functions decide what command the robot actually receives and how long it
runs, so they are worth testing without paying for a kit launch. The module
keeps them at module scope (all Isaac imports are deferred into
``PolicyPlayback.__init__``) precisely so this file can import it.
"""

from __future__ import annotations

import math
import random

import pytest

from duck_embody.sim.policy_wrapper import (
    CONTACT_SUSTAINED_STEPS,
    CONTROL_HZ,
    ExecResult,
    PolicyPlayback,
    REVERSE_MOVE_SPEED_MPS,
    VX_RANGE,
    VY_RANGE,
    WZ_RANGE,
    clamp_command,
    duration_to_steps,
    shortest_angle_diff_deg,
    wrap_deg,
)


class TestClampCommand:
    def test_in_hull_passes_through_unchanged_and_silently(self):
        (vx, vy, wz), notes = clamp_command(0.2, 0.05, -0.3)
        assert (vx, vy, wz) == (0.2, 0.05, -0.3)
        assert notes == []

    def test_each_axis_clamps_to_its_own_bound(self):
        (vx, vy, wz), notes = clamp_command(5.0, 5.0, 5.0)
        assert vx == VX_RANGE[1]
        assert vy == VY_RANGE[1]
        assert wz == WZ_RANGE[1]
        assert len(notes) == 3

    def test_negative_bounds_clamp_too(self):
        (vx, vy, wz), _ = clamp_command(-5.0, -5.0, -5.0)
        assert (vx, vy, wz) == (VX_RANGE[0], VY_RANGE[0], WZ_RANGE[0])

    def test_clamping_is_reported_not_silent(self):
        """The model must be able to see that it did not get what it asked for."""
        _, notes = clamp_command(0.9, 0.0, 0.0)
        assert len(notes) == 1
        assert "vx" in notes[0] and "0.222" in notes[0]

    def test_asymmetric_vx_hull_is_respected(self):
        """vx is NOT symmetric: backwards is limited to -0.148, forwards 0.222."""
        (fwd, _, _), _ = clamp_command(1.0, 0.0, 0.0)
        (back, _, _), _ = clamp_command(-1.0, 0.0, 0.0)
        assert fwd == pytest.approx(0.222)
        assert back == pytest.approx(-0.148)
        assert abs(back) != pytest.approx(fwd)

    def test_exact_bounds_are_not_reported_as_clamped(self):
        _, notes = clamp_command(VX_RANGE[1], VY_RANGE[0], WZ_RANGE[1])
        assert notes == []


class TestDurationToSteps:
    @pytest.mark.parametrize(
        "duration_s,expected",
        [(1.0, 50), (0.2, 10), (3.0, 150), (20.0, 1000), (120.0, 6000)],
    )
    def test_50hz_conversion(self, duration_s, expected):
        assert duration_to_steps(duration_s) == expected

    def test_rounds_rather_than_truncates(self):
        # 0.019 s is under half a control step; 0.021 s is over.
        assert duration_to_steps(0.019) == 1  # floor would give 0 -> no motion
        assert duration_to_steps(0.021) == 1
        assert duration_to_steps(0.03) == 2  # round-half-even at exactly 1.5

    def test_never_returns_zero_steps(self):
        """A zero-step command would silently do nothing at all."""
        assert duration_to_steps(0.0) == 1
        assert duration_to_steps(-1.0) == 1

    def test_control_rate_matches_the_policy_contract(self):
        assert CONTROL_HZ == 50.0


class TestHeadingMath:
    @pytest.mark.parametrize(
        "raw,expected", [(0, 0), (90, 90), (360, 0), (450, 90), (-90, 270), (-450, 270)]
    )
    def test_wrap_to_0_360(self, raw, expected):
        assert wrap_deg(raw) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "target,current,expected",
        [
            (90, 0, 90),
            (0, 90, -90),
            (350, 10, -20),  # across the 0/360 seam, the short way
            (10, 350, 20),
            (0, 0, 0),
        ],
    )
    def test_shortest_diff_takes_the_short_way(self, target, current, expected):
        assert shortest_angle_diff_deg(target, current) == pytest.approx(expected)

    def test_exact_half_turn_resolves_deterministically(self):
        """Both directions are equally short at 180 deg; the tie must break the
        SAME way every time, or the turn P-loop can dither at the boundary."""
        assert shortest_angle_diff_deg(180, 0) == pytest.approx(-180.0)
        assert shortest_angle_diff_deg(180, 0) == shortest_angle_diff_deg(180, 0)

    def test_result_is_always_within_half_turn(self):
        # Half-open [-180, 180): +180 is normalised to -180 (see the docstring).
        for target in range(0, 360, 7):
            for current in range(0, 360, 11):
                diff = shortest_angle_diff_deg(target, current)
                assert -180.0 <= diff < 180.0

    def test_turning_toward_target_reduces_error(self):
        """The sign convention the turn_to_heading P-loop relies on."""
        current, target = 10.0, 90.0
        diff = shortest_angle_diff_deg(target, current)
        assert diff > 0  # positive error -> positive wz -> CCW
        moved = wrap_deg(current + diff * 0.5)
        assert abs(shortest_angle_diff_deg(target, moved)) < abs(diff)


class TestHullMatchesTrainedPolicy:
    """These bounds are read from policy/params/env.yaml; drifting from them
    means commanding a gait the policy was never trained to produce."""

    def test_hull_values(self):
        assert VX_RANGE == (-0.148, 0.222)
        assert VY_RANGE == (-0.111, 0.111)
        assert WZ_RANGE == (-0.5, 0.5)

    def test_sideways_motion_is_near_useless(self):
        """Doc 05 §4 tells the model to prefer turn-then-drive; this is why."""
        seconds_per_metre = 1.0 / VY_RANGE[1]
        assert seconds_per_metre > 8.0
        assert math.isclose(seconds_per_metre, 9.009, rel_tol=1e-3)


class TestFallThresholdsMirrorTheEnvCfg:
    """`policy_wrapper` duplicates the fall thresholds instead of importing them
    (importing `embody_env_cfg` pulls in the parent repo and needs a running kit
    app, while this module's pure half must stay unit-testable).

    A drift here would not crash: the fall REPORT would simply cite a threshold
    that is not the one that actually fired, so an audit of the most
    consequential event in a trial would be quietly reading fiction.
    """

    def test_the_two_copies_agree(self):
        import re
        from pathlib import Path

        from duck_embody.sim.policy_wrapper import (
            FALL_MIN_HEIGHT_M,
            FALL_TILT_LIMIT_DEG,
        )

        src = (
            Path(__file__).resolve().parent.parent
            / "duck_embody" / "env" / "embody_env_cfg.py"
        ).read_text()

        height = re.search(r"^FALL_MIN_HEIGHT_M\s*=\s*([0-9.]+)", src, re.M)
        tilt = re.search(r"^FALL_TILT_LIMIT_RAD\s*=\s*math\.radians\(([0-9.]+)\)", src, re.M)
        assert height, "FALL_MIN_HEIGHT_M not found in embody_env_cfg.py"
        assert tilt, "FALL_TILT_LIMIT_RAD not found in embody_env_cfg.py"

        assert float(height.group(1)) == FALL_MIN_HEIGHT_M
        assert float(tilt.group(1)) == FALL_TILT_LIMIT_DEG


class TestFallDiagnosticsAreScoringOnly:
    """The diagnostics exist to audit a fall, not to inform the model.

    Height, tilt and the termination term are ground truth the model has no
    sensor for. The model-facing half of this rule is asserted in
    tests/test_tools.py, where the per-tool payload allowlist lives.
    """

    def test_execresult_carries_them_and_they_default_to_none(self):
        from duck_embody.sim.policy_wrapper import ExecResult

        assert "fall_diagnostics" in ExecResult.__dataclass_fields__
        assert ExecResult(
            commanded=(0.0, 0.0, 0.0), duration_s=0.0, steps=0,
            policy_seconds=0.0, bumped=False, fell=False,
        ).fall_diagnostics is None


def _macro_result(**kw) -> ExecResult:
    base = dict(
        commanded=(0.0, 0.0, 0.0),
        duration_s=0.2,
        steps=10,
        policy_seconds=0.2,
        bumped=False,
        fell=False,
        true_pose=(0.0, 0.0, 0.0),
    )
    base.update(kw)
    return ExecResult(**base)


class TestMeasuredDistanceMove:
    @staticmethod
    def _playback(fake_execute):
        pb = PolicyPlayback.__new__(PolicyPlayback)
        pb.compass_deg = lambda: 0.0
        pb.true_xy = lambda: (0.0, 0.0)
        pb.execute = fake_execute
        return pb

    def test_point_one_metres_cannot_reach_a_point_four_request(self):
        """The timeout forecast may use k; completion must use odometry only."""
        drive_calls = {"n": 0}

        def fake_execute(vx, vy, wz, duration_s, **kwargs):
            if vx == 0.0:
                return _macro_result()
            drive_calls["n"] += 1
            measured = 0.10 if drive_calls["n"] == 1 else 0.0
            return _macro_result(
                commanded=(vx, vy, wz),
                odom_dxy=(measured, 0.0),
                odom_distance_m=measured,
            )

        result = self._playback(fake_execute).move(0.40)

        assert result.measured_distance_m == pytest.approx(0.10)
        assert result.requested_distance_m == pytest.approx(0.40)
        assert result.target_reached is False
        assert result.stop_reason == "timeout"

    def test_reverse_uses_the_conservative_cap_and_measured_progress(self):
        commands = []

        def fake_execute(vx, vy, wz, duration_s, **kwargs):
            commands.append(vx)
            measured = 0.05 if vx < 0.0 else 0.0
            return _macro_result(
                commanded=(vx, vy, wz),
                odom_dxy=(-measured, 0.0),
                odom_distance_m=measured,
            )

        result = self._playback(fake_execute).move(-0.10)

        assert commands[:2] == [-REVERSE_MOVE_SPEED_MPS, -REVERSE_MOVE_SPEED_MPS]
        assert result.requested_distance_m == pytest.approx(-0.10)
        assert result.measured_distance_m == pytest.approx(0.10)
        assert result.target_reached is True
        assert result.stop_reason == "reached"


class TestPersistentContactMachine:
    def test_one_event_spans_calls_then_release_allows_the_next_id(self):
        pb = _scripted_playback(_curve(80))
        force = {"n": 500.0}
        pb.bump_contact_force = lambda: force["n"]
        pb.contact_groups = lambda: ["head"]
        chunk_s = (CONTACT_SUSTAINED_STEPS // 2) / CONTROL_HZ

        first = pb.execute(0.0, 0.0, 0.0, chunk_s)
        second = pb.execute(0.0, 0.0, 0.0, chunk_s)

        assert first.contact_state == "candidate_contact"
        assert first.contact_event_id is None
        assert second.contact_state == "sustained_contact"
        assert second.contact_event_id == 1
        assert second.contact_onset_step == 1
        assert second.contact_event_regions == ["head"]

        force["n"] = 0.0
        releasing = pb.execute(0.0, 0.0, 0.0, chunk_s)
        released = pb.execute(0.0, 0.0, 0.0, chunk_s)
        assert releasing.contact_state == "candidate_release"
        assert released.contact_state == "free"
        assert released.contact_event_id == 1
        assert released.contact_release_step == 2 * CONTACT_SUSTAINED_STEPS

        force["n"] = 500.0
        pb.execute(0.0, 0.0, 0.0, chunk_s)
        next_event = pb.execute(0.0, 0.0, 0.0, chunk_s)
        assert next_event.contact_event_id == 2
        assert pb.last_contact_event["contact_event_id"] == 2


class TestTurnAndMoveMacro:
    def test_turn_aborts_on_sustained_contact_after_a_graze(self):
        pb = PolicyPlayback.__new__(PolicyPlayback)
        pb.true_xy = lambda: (0.0, 0.0)
        pb.compass_deg = lambda: 0.0
        turns = {"n": 0}

        def fake_execute(vx, vy, wz, duration_s, **kwargs):
            if wz != 0.0:
                turns["n"] += 1
                state = (
                    "candidate_contact"
                    if turns["n"] == 1
                    else "sustained_contact"
                )
                return _macro_result(
                    commanded=(vx, vy, wz),
                    bumped=True,
                    contact_state=state,
                    stop_reason=(
                        "sustained_contact"
                        if state == "sustained_contact"
                        else ""
                    ),
                )
            return _macro_result(contact_state="candidate_release")

        pb.execute = fake_execute
        result = pb.turn_to_heading(90.0)

        assert turns["n"] == 2
        assert result.target_reached is False
        assert result.stop_reason == "sustained_contact"

    def test_turn_and_move_runs_in_order_and_returns_phase_summaries(self):
        pb = PolicyPlayback.__new__(PolicyPlayback)
        calls = []

        def turn(heading_deg, on_chunk=None):
            calls.append(("turn", heading_deg))
            return _macro_result(stop_reason="reached", target_reached=True)

        def move(distance_m, **kwargs):
            calls.append(("move", distance_m))
            return _macro_result(
                commanded=(0.2, 0.0, 0.0),
                stop_reason="reached",
                target_reached=True,
                requested_distance_m=distance_m,
                measured_distance_m=distance_m,
                odom_dxy=(distance_m, 0.0),
                odom_distance_m=distance_m,
            )

        pb.turn_to_heading = turn
        pb.move = move
        result = pb.turn_and_move(90.0, 0.40)

        assert calls == [("turn", 90.0), ("move", 0.40)]
        assert [phase["phase"] for phase in result.phase_results] == [
            "turn",
            "move",
        ]
        assert result.target_reached is True
        assert result.measured_distance_m == pytest.approx(0.40)


# ===========================================================================
# TR.3 — the odometry process is a property of the STEPS, not of the CALLS
# ===========================================================================


class _NoGradCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTorch:
    @staticmethod
    def no_grad():
        return _NoGradCtx()


def _scripted_playback(trajectory, seed: int = 101, scale: float = 1.0):
    """A ``PolicyPlayback`` whose physics is a fixed list of true XY poses.

    Deliberately NOT a mock of ``execute()``: this drives the REAL step loop
    (clamping, termination check, bump debounce, pose sampling, the odometer)
    against a scripted trajectory, so the invariance being asserted is the
    shipped one. ``env.step`` advances an index into ``trajectory``; every call
    on the same instance continues from where the previous one stopped, which is
    exactly what makes "one call vs 75 calls over the same steps" a fair test.
    """
    from types import SimpleNamespace

    from duck_embody.sim.policy_wrapper import PolicyPlayback

    pb = PolicyPlayback.__new__(PolicyPlayback)
    state = {"i": 0}

    pb._torch = _FakeTorch
    pb._obs = object()
    pb.policy = lambda obs: "actions"
    def _step(actions):
        state["i"] += 1
        return (object(), None, None, None)

    pb.env = SimpleNamespace(step=_step)
    pb.base_env = SimpleNamespace(
        termination_manager=SimpleNamespace(
            terminated=[False], active_terms=(), get_term=lambda name: [False]
        )
    )
    pb.set_command = lambda vx, vy, wz: None
    pb.true_xy = lambda: trajectory[min(state["i"], len(trajectory) - 1)]
    pb.compass_deg = lambda: 0.0
    pb.true_height = lambda: 0.17
    pb.tilt_deg = lambda: 0.0
    pb.bump_contact_force = lambda: 0.0
    pb.contact_groups = lambda: []
    pb._bump_run = 0
    pb._step_counter = 0
    pb._fell = False
    pb._fall_diagnostics = None
    pb._odom_rng = random.Random(seed)
    pb._odom_scale = scale
    return pb


def _curve(n_steps: int) -> list[tuple[float, float]]:
    """A deterministic curved trajectory, ~4 mm per step (0.2 m/s at 50 Hz)."""
    out = []
    for i in range(n_steps + 1):
        t = i * 0.004
        out.append((t, 0.15 * math.sin(t * 2.0)))
    return out


class TestOdometryIsChunkInvariant:
    """Forensics F-03's falsifier, as a unit test.

    The pre-TR.3 odometer drew ONE Gaussian per ``execute()`` call with an
    ADDITIVE sigma (``FRAC*distance + RATE*seconds``). Sigma does not add —
    variance does — so slicing a command into N pieces and vector-summing gave
    ``sigma/sqrt(N)``: with the recorder's 75 pieces per 3 s command, the paid
    batch ran an 8.7x quieter odometry sensor than every unit test and the first
    odometry smoke did. Recording changed the measurement.

    Noise is now drawn per CONTROL STEP with additive variance, so the total is
    fixed by the step sequence alone.
    """

    @staticmethod
    def _run_split(n_calls: int, total_steps: int = 150, seed: int = 101):
        from duck_embody.sim.policy_wrapper import CONTROL_DT

        traj = _curve(total_steps)
        pb = _scripted_playback(traj, seed=seed)
        per_call = total_steps // n_calls
        assert per_call * n_calls == total_steps, "choose a divisible split"
        dx = dy = 0.0
        dist = 0.0
        for _ in range(n_calls):
            r = pb.execute(0.2, 0.0, 0.0, per_call * CONTROL_DT)
            dx += r.odom_dxy[0]
            dy += r.odom_dxy[1]
            dist += r.odom_distance_m
        return dx, dy, dist

    def test_one_five_and_seventyfive_calls_give_identical_odometry(self):
        one = self._run_split(1)
        five = self._run_split(5)
        seventyfive = self._run_split(75)
        assert five[0] == pytest.approx(one[0], abs=1e-12)
        assert five[1] == pytest.approx(one[1], abs=1e-12)
        assert seventyfive[0] == pytest.approx(one[0], abs=1e-12)
        assert seventyfive[1] == pytest.approx(one[1], abs=1e-12)

    def test_the_odometry_actually_moved_so_the_test_is_not_vacuous(self):
        """A split-invariance assertion passes trivially on zeros."""
        dx, dy, _ = self._run_split(1)
        assert math.hypot(dx, dy) > 0.4, (dx, dy)

    def test_noise_is_present_at_every_splitting(self):
        """Invariance must not have been bought by deleting the noise.

        The estimate has to keep drifting or the research question (can an LLM
        close loops against real drift?) disappears — AGENTS.md rule 5.
        """
        from duck_embody.sim.policy_wrapper import CONTROL_DT

        traj = _curve(150)
        truth = (traj[150][0] - traj[0][0], traj[150][1] - traj[0][1])
        for n_calls in (1, 5, 75):
            dx, dy, _ = self._run_split(n_calls)
            assert (dx, dy) != truth
            assert math.hypot(dx - truth[0], dy - truth[1]) > 1e-6

    def test_a_different_seed_gives_different_noise(self):
        assert self._run_split(5, seed=101)[:2] != self._run_split(5, seed=104)[:2]

    def test_the_per_trial_scale_is_systematic_not_per_call(self):
        """One scale draw per trial, applied at every step: a 5% short sensor
        stays 5% short whether the motion arrived in one call or seventy-five."""
        from duck_embody.sim.policy_wrapper import CONTROL_DT

        traj = _curve(100)
        results = []
        for n_calls in (1, 4):
            pb = _scripted_playback(traj, seed=7, scale=0.95)
            dx = 0.0
            for _ in range(n_calls):
                dx += pb.execute(0.2, 0.0, 0.0, (100 // n_calls) * CONTROL_DT).odom_dxy[0]
            results.append(dx)
        assert results[1] == pytest.approx(results[0], abs=1e-12)
        # 0.95 scale on a 0.6 m x-run, plus a few cm of noise.
        assert results[0] == pytest.approx(0.95 * traj[100][0], abs=0.05)


class TestOdometryVarianceRates:
    def test_the_rates_are_the_squares_of_the_legacy_sigmas(self):
        """Calibration pin: at 1 m travelled the per-axis sigma is unchanged at
        ODOM_NOISE_FRAC, so TR.3 changed the SHAPE of the process (sqrt-of-
        distance random walk instead of linear-in-distance) without silently
        changing its magnitude."""
        from duck_embody.sim.policy_wrapper import (
            ODOM_NOISE_FLOOR_RATE_MPS,
            ODOM_NOISE_FRAC,
            ODOM_VAR_PER_M,
            ODOM_VAR_PER_S,
        )

        assert ODOM_VAR_PER_M == pytest.approx(ODOM_NOISE_FRAC ** 2)
        assert ODOM_VAR_PER_S == pytest.approx(ODOM_NOISE_FLOOR_RATE_MPS ** 2)
        assert math.sqrt(ODOM_VAR_PER_M * 1.0) == pytest.approx(ODOM_NOISE_FRAC)

    def test_variance_adds_over_steps_so_sigma_grows_as_sqrt_distance(self):
        """Measured over 400 seeded trials rather than asserted from the
        formula: this is the property that failed before (aggregate sigma
        depending on the partitioning), so it is checked on outputs."""
        from duck_embody.sim.policy_wrapper import CONTROL_DT, ODOM_VAR_PER_M

        def spread(total_steps: int) -> float:
            errs = []
            traj = _curve(total_steps)
            truth = traj[total_steps][0] - traj[0][0]
            for seed in range(400):
                pb = _scripted_playback(traj, seed=seed)
                r = pb.execute(0.2, 0.0, 0.0, total_steps * CONTROL_DT)
                errs.append(r.odom_dxy[0] - truth)
            mean = sum(errs) / len(errs)
            return math.sqrt(sum((e - mean) ** 2 for e in errs) / (len(errs) - 1))

        near, far = spread(100), spread(400)
        # Path lengths are ~0.4 m and ~1.6 m (4 mm/step): 4x the distance is
        # 4x the variance, i.e. 2x the sigma.
        assert far / near == pytest.approx(2.0, rel=0.25), (near, far)
        # And the absolute scale matches the declared rate (0.4 m -> ~1.9 cm).
        assert near == pytest.approx(math.sqrt(ODOM_VAR_PER_M * 0.4), rel=0.25)

    def test_a_standing_robot_still_accrues_a_small_time_floor(self):
        """Standing/slipping error is a RATE. Zero here would mean a duck that
        stands for a minute is certain it stands exactly where it started."""
        from duck_embody.sim.policy_wrapper import CONTROL_DT

        still = [(0.0, 0.0)] * 200
        pb = _scripted_playback(still, seed=3)
        r = pb.execute(0.0, 0.0, 0.0, 150 * CONTROL_DT)
        assert 0.0 < r.odom_distance_m < 0.02, r.odom_distance_m
