"""Unit tests for the kit-free math in ``duck_embody.sim.policy_wrapper``.

These functions decide what command the robot actually receives and how long it
runs, so they are worth testing without paying for a kit launch. The module
keeps them at module scope (all Isaac imports are deferred into
``PolicyPlayback.__init__``) precisely so this file can import it.
"""

from __future__ import annotations

import math

import pytest

from duck_embody.sim.policy_wrapper import (
    CONTROL_HZ,
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
