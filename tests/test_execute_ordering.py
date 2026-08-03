"""Forensic pins for the motion execution path (sweep 3, static deep-read).

Each test here asserts the behaviour the design docs require of the
execute/merge/settle path, mocked so no kit process is needed. The first five
were authored RED against live defects and landed in the same commit as their
fixes (the consolidated pre-freeze plan's hygiene rule: pins enter the tree
green, never as standing reds). The defects they now pin closed:

* ``chunked_execute`` (recorder.py) merged ``fell``/``bumped`` across chunks
  but never copied ``fall_diagnostics`` or ``contact_groups`` — so in any
  recorded run (the paid batch records), a fall or bump confirmed after the
  first 2-step chunk returned ``fell=True`` with ``fall_diagnostics=None`` and
  ``bumped=True`` with ``contact_groups=[]``. That was T3.5's "``fell: true``
  with no diagnostics" artifact exactly. Fixed by routing BOTH stitching
  layers through one shared merge, ``policy_wrapper.merge_exec_results``.
* ``move()``/``turn_to_heading()`` overwrote ``stop_reason`` AFTER merging the
  trailing settle chunk, so a fall DURING settle was recorded as
  ``reached``/``timeout`` in the doc 06 §4 audit record (and could flip
  ``timed_out`` shown to the model) even though ``fell=True``.
* the merges replaced ``contact_groups`` wholesale, dropping regions felt at
  an earlier chunk's confirmed bump. The ratified policy (doc 06 §4) is a
  union preserving first-seen order, applied identically in the shared merge
  and inside ``execute()`` itself.

Also pinned: ``fall_diagnostics.policy_seconds_into_call`` accumulates across
merged chunks (chunk-local, it bounded every recorded fall at 0.04 s), the
``values_pre_step`` self-description marker, and the clamp-note dedupe that
keeps a 75-chunk recorded command from echoing one clamp note 75 times.

The passing-from-birth tests document orderings that were audited and found
sound (the terminated-before-bump precedence inside ``execute()``).
"""

from __future__ import annotations

from types import SimpleNamespace

import random
import copy

from duck_embody.sim.policy_wrapper import (
    BUMP_DEBOUNCE_STEPS,
    ExecResult,
    PolicyPlayback,
    merge_exec_results,
)
from duck_embody.sim.recorder import chunked_execute


def _res(**kw) -> ExecResult:
    base = dict(
        commanded=(0.2, 0.0, 0.0),
        duration_s=0.04,
        steps=2,
        policy_seconds=0.04,
        bumped=False,
        fell=False,
    )
    base.update(kw)
    return ExecResult(**base)


def _fell_res(diag: dict, **kw) -> ExecResult:
    base = dict(
        steps=1,
        policy_seconds=0.02,
        fell=True,
        stopped_early=True,
        stop_reason="fell",
        fall_diagnostics=diag,
    )
    base.update(kw)
    return _res(**base)


class TestChunkedExecuteMerge:
    """recorder.chunked_execute — **LEGACY, NON-BENCHMARK since TR.3.**

    It was the merge every RECORDED command ran through while
    ``attach_recorder`` patched ``playback.execute``; that seam is gone (see
    ``TestAttachIsObservational`` below) and the helper now serves only
    ``scripts/replay_falls.py``, which replays the frozen v5d batches the way
    they really executed. These pins stay because that replay tool is how a
    frozen trial gets re-examined, and the merge losing ``fall_diagnostics`` or
    ``contact_groups`` is exactly what made T3.5's fall unauditable."""

    @staticmethod
    def _run(parts):
        calls = iter(parts)

        def fake_execute(vx, vy, wz, duration_s, **kwargs):
            return next(calls)

        recorder = SimpleNamespace(grab=lambda env: None)
        # 0.08 s = 4 control steps = two 2-step chunks at the default chunk_s.
        return chunked_execute(
            fake_execute, object(), recorder, 0.2, 0.0, 0.0, 0.08
        )

    def test_fall_diagnostics_survive_a_multi_chunk_merge(self):
        """A fall in chunk 2+ must not return fell=True with diagnostics=None.

        BUMP_DEBOUNCE_STEPS=3 > the 2-step chunk, so in a recorded run the
        terminating/confirming chunk is almost never the first — this is the
        common case, not a corner. tools._record_motion papers over the None
        via the playback-instance fallback, but the ExecResult itself is the
        contract (scripted_drive consumers read it directly), and the fallback
        comment says this exact artifact could not be explained.
        """
        diag = {"height_m": 0.05, "tilt_deg": 75.0}
        merged = self._run(
            [
                _res(pose_trace=[(0.0, 0.0), (0.1, 0.0)], true_pose=(0.1, 0.0, 0.0)),
                _fell_res(
                    diag,
                    pose_trace=[(0.1, 0.0), (0.12, 0.0)],
                    true_pose=(0.12, 0.0, 0.0),
                ),
            ]
        )
        assert merged.fell is True
        assert merged.stop_reason == "fell"
        assert merged.fall_diagnostics == diag

    def test_contact_groups_survive_a_multi_chunk_merge(self):
        """A bump confirmed in chunk 2+ must keep its contact regions.

        Nothing downstream recovers these: _state_payload renders
        status.contact from ExecResult.contact_groups, so losing them here
        shows the model bumped=true, contact=[] in every recorded trial —
        the bump-body blindness class reintroduced by the recording seam.
        """
        merged = self._run(
            [
                _res(pose_trace=[(0.0, 0.0), (0.1, 0.0)], true_pose=(0.1, 0.0, 0.0)),
                _res(
                    bumped=True,
                    contact_groups=["head"],
                    stopped_early=True,
                    stop_reason="bump",
                    pose_trace=[(0.1, 0.0), (0.12, 0.0)],
                    true_pose=(0.12, 0.0, 0.0),
                ),
            ]
        )
        assert merged.bumped is True
        assert merged.contact_groups == ["head"]

    def test_policy_seconds_into_call_accumulates_across_chunks(self):
        """G9: `execute()` can only know its own chunk, so a recorded fall's
        `policy_seconds_into_call` was bounded at 0.04 s no matter how deep
        into the command it happened. The merge re-stamps it with the seconds
        accumulated into the whole call."""
        diag = {"height_m": 0.05, "tilt_deg": 75.0, "policy_seconds_into_call": 0.02}
        merged = self._run(
            [
                _res(pose_trace=[(0.0, 0.0), (0.1, 0.0)], true_pose=(0.1, 0.0, 0.0)),
                _fell_res(
                    diag,
                    pose_trace=[(0.1, 0.0), (0.12, 0.0)],
                    true_pose=(0.12, 0.0, 0.0),
                ),
            ]
        )
        # 0.04 s (chunk 1) + 0.02 s (the terminating chunk's one step).
        assert merged.fall_diagnostics["policy_seconds_into_call"] == 0.06

    def test_a_repeated_clamp_note_is_echoed_once_not_per_chunk(self):
        """Every 0.04 s piece of one out-of-hull command carries the identical
        clamp note; extending blindly would echo it ~75 times per 3 s command,
        turning the model-facing `notes` key from a signal into noise."""
        note = "vx +0.500 clamped to +0.222 (hull [-0.148, 0.222])"
        merged = self._run(
            [
                _res(clamp_notes=[note], pose_trace=[(0.0, 0.0)], true_pose=(0.0, 0.0, 0.0)),
                _res(
                    clamp_notes=[note],
                    stopped_early=True,
                    stop_reason="bump",
                    pose_trace=[(0.0, 0.0)],
                    true_pose=(0.0, 0.0, 0.0),
                ),
            ]
        )
        assert merged.clamp_notes == [note]


class TestFallFrameGrab:
    """G8: no viewport grab after the falling piece. Isaac auto-reset the
    terminated env inside step(), so a post-fall grab films a healthy duck at
    spawn — every fall video ENDED on a post-teleport frame, structurally
    deleting rule 11's 'video wins' evidence for the event that ends trials."""

    @staticmethod
    def _run_counting_grabs(parts):
        calls = iter(parts)

        def fake_execute(vx, vy, wz, duration_s, **kwargs):
            return next(calls)

        grabs: list[int] = []
        recorder = SimpleNamespace(grab=lambda env: grabs.append(1))
        chunked_execute(fake_execute, object(), recorder, 0.2, 0.0, 0.0, 0.08)
        return len(grabs)

    def test_no_grab_for_the_falling_piece(self):
        n = self._run_counting_grabs(
            [
                _res(pose_trace=[(0.0, 0.0), (0.1, 0.0)], true_pose=(0.1, 0.0, 0.0)),
                _fell_res(
                    {"height_m": 0.05, "tilt_deg": 75.0},
                    pose_trace=[(0.1, 0.0), (0.12, 0.0)],
                    true_pose=(0.12, 0.0, 0.0),
                ),
            ]
        )
        assert n == 1, "the falling piece must not be grabbed (post-teleport frame)"

    def test_every_upright_piece_is_still_grabbed(self):
        """Positive control: without a fall, one grab per piece as always."""
        n = self._run_counting_grabs(
            [
                _res(pose_trace=[(0.0, 0.0), (0.1, 0.0)], true_pose=(0.1, 0.0, 0.0)),
                _res(pose_trace=[(0.1, 0.0), (0.2, 0.0)], true_pose=(0.2, 0.0, 0.0)),
            ]
        )
        assert n == 2


class TestSharedMergeIsShared:
    """The recorder's chunk merge and the macros' `_merge` used to be two
    hand-mirrored copies, and the copies drifted (that drift IS gap G1). Both
    now route through `merge_exec_results`; this pins the macro entry point to
    the shared behaviour so a re-fork fails loudly."""

    def test_macro_merge_and_shared_merge_agree_field_for_field(self):
        diag = {"height_m": 0.05, "tilt_deg": 75.0, "policy_seconds_into_call": 0.02}
        parts = [
            _res(bumped=True, contact_groups=["head"]),
            _fell_res(diag, contact_groups=["torso"]),
        ]
        via_macro = PolicyPlayback.__new__(PolicyPlayback)._merge(
            copy.deepcopy(parts[0]), copy.deepcopy(parts[1])
        )
        via_shared = merge_exec_results(
            copy.deepcopy(parts[0]), copy.deepcopy(parts[1])
        )
        for field_name in (
            "steps", "policy_seconds", "bumped", "fell", "contact_groups",
            "fall_diagnostics", "stopped_early", "stop_reason", "clamp_notes",
        ):
            assert getattr(via_macro, field_name) == getattr(via_shared, field_name), field_name


class TestSettleFallReporting:
    """A fall during the trailing zero-command settle chunk must be reported as
    a fall. Both macros currently overwrite stop_reason/stopped_early AFTER
    merging the settle part, so the audit record says reached/timeout while
    fell=True — and turn_to_heading's model-facing `timed_out` key is derived
    from that stop_reason."""

    @staticmethod
    def _playback(compass_deg: float, fake_execute):
        pb = PolicyPlayback.__new__(PolicyPlayback)
        # Real execute() consumes the leg-odometry error model that
        # __init__ seeds; a __new__-built instance must supply it.
        pb._odom_rng = random.Random(0)
        pb._odom_scale = 1.0
        pb.compass_deg = lambda: compass_deg
        pb.true_xy = lambda: (0.0, 0.0)
        # Instance attribute shadows the bound method — the same seam
        # attach_recorder itself relies on.
        pb.execute = fake_execute
        return pb

    def test_move_settle_fall_reports_fell(self):
        diag = {"height_m": 0.05, "tilt_deg": 75.0}

        def fake_execute(vx, vy, wz, duration_s, stop_on_bump=False, stop_predicate=None):
            if vx > 0.0:  # the driving chunk: covers the target immediately
                return _res(
                    steps=10,
                    policy_seconds=0.2,
                    pose_trace=[(0.0, 0.0), (0.04, 0.0)],
                    true_pose=(0.04, 0.0, 0.0),
                )
            # the settle chunk: the robot topples while stopping
            return _fell_res(diag, true_pose=(0.05, 0.0, 0.0))

        pb = self._playback(0.0, fake_execute)
        result = pb.move(0.04)

        assert result.fell is True
        assert result.fall_diagnostics == diag
        assert result.stop_reason == "fell"
        assert result.stopped_early is True

    def test_turn_settle_fall_reports_fell(self):
        diag = {"height_m": 0.05, "tilt_deg": 75.0}

        def fake_execute(vx, vy, wz, duration_s, stop_on_bump=False, stop_predicate=None):
            # Already within tolerance, so the ONLY execute is the settle.
            return _fell_res(diag, true_pose=(0.0, 0.0, 90.0))

        pb = self._playback(90.0, fake_execute)
        result = pb.turn_to_heading(90.0)

        assert result.fell is True
        assert result.fall_diagnostics == diag
        # tools.py derives the model-facing `timed_out` from this exact string.
        assert result.stop_reason == "fell"


class TestMergeContactGroups:
    def test_merge_keeps_every_contact_region_seen(self):
        """Chunk 3 bumps with the head, chunk 4 with the torso: the merged
        report must not silently lose the head contact. `_merge` currently
        replaces the list wholesale, so the earlier region vanishes."""
        pb = PolicyPlayback.__new__(PolicyPlayback)
        # Real execute() consumes the leg-odometry error model that
        # __init__ seeds; a __new__-built instance must supply it.
        pb._odom_rng = random.Random(0)
        pb._odom_scale = 1.0
        total = _res(bumped=True, contact_groups=["head"])
        part = _res(bumped=True, contact_groups=["torso"])
        merged = pb._merge(total, part)
        assert "head" in merged.contact_groups
        assert "torso" in merged.contact_groups


class _NoGradCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTorch:
    @staticmethod
    def no_grad():
        return _NoGradCtx()


class TestExecuteTerminationOrdering:
    """Audit of execute()'s step loop, fully mocked: the terminated check runs
    BEFORE bump detection and before the pose sample, so a step that both
    terminates and would confirm a bump still sets _fell, captures diagnostics
    from the PRE-step snapshot, and never reads the teleported pose."""

    @staticmethod
    def _playback(terminated: bool):
        pb = PolicyPlayback.__new__(PolicyPlayback)
        # Real execute() consumes the leg-odometry error model that
        # __init__ seeds; a __new__-built instance must supply it.
        pb._odom_rng = random.Random(0)
        pb._odom_scale = 1.0
        pb._torch = _FakeTorch
        pb._obs = object()
        pb.policy = lambda obs: "actions"
        pb.env = SimpleNamespace(step=lambda actions: (object(), None, None, None))
        pb.base_env = SimpleNamespace(
            termination_manager=SimpleNamespace(
                terminated=[terminated],
                active_terms=("base_height", "bad_orientation"),
                get_term=lambda name: [terminated],
            )
        )
        pb.set_command = lambda vx, vy, wz: None
        pb.true_xy = lambda: (1.0, 2.0)
        pb.compass_deg = lambda: 45.0
        pb.true_height = lambda: 0.05
        pb.tilt_deg = lambda: 80.0
        pb.bump_contact_force = lambda: 500.0
        pb.contact_groups = lambda: ["head"]
        pb._bump_run = BUMP_DEBOUNCE_STEPS - 1  # would confirm on this step
        pb._step_counter = 0
        pb._fell = False
        pb._fall_diagnostics = None
        return pb

    def test_terminated_step_sets_fell_and_captures_pre_step_state(self):
        pb = self._playback(terminated=True)
        result = pb.execute(0.2, 0.0, 0.0, 0.2, stop_on_bump=True)

        # The terminated check ran (it precedes the bump-stop break).
        assert pb._fell is True
        assert result.fell is True
        assert result.stop_reason == "fell"
        assert result.stopped_early is True
        assert result.steps == 1
        # Diagnostics captured on the terminating call, from PRE-step reads.
        assert result.fall_diagnostics is not None
        assert result.fall_diagnostics["tilt_deg"] == 80.0
        assert result.fall_diagnostics["height_m"] == 0.05
        # G10: the dict says its values are pre-step, because a tilt of 59.4
        # beside a 60.0 threshold and fell_over=true is otherwise inexplicable.
        assert result.fall_diagnostics["values_pre_step"] is True
        assert result.fall_diagnostics["terms"] == {
            "base_height": True,
            "bad_orientation": True,
        }
        # The final pose is the pre-step snapshot, not the teleported spawn.
        assert result.true_pose == (1.0, 2.0, 45.0)
        # Fall takes precedence over the same-step bump confirm: documented
        # ordering, the collision surfaces via fall_diagnostics instead.
        assert result.bumped is False


class TestExecuteContactUnion:
    """`execute()` samples contact regions on every confirmed-contact step
    (under stop_on_bump=False the command keeps driving through contact) and
    must accumulate them as a first-seen-order union — the same rule the merge
    layers apply across chunks — rather than overwrite with the latest sample,
    which silently destroyed the earlier felt region."""

    def test_regions_felt_on_successive_contact_steps_all_survive(self):
        pb = TestExecuteTerminationOrdering._playback(terminated=False)
        samples = iter([["head"], ["torso"], ["torso"]])
        pb.contact_groups = lambda: next(samples)
        pb._bump_run = BUMP_DEBOUNCE_STEPS - 1  # confirms on the first step

        # 3 control steps at 50 Hz; contact force stays above threshold
        # throughout (the fake returns 500 N), so every step re-samples.
        result = pb.execute(0.2, 0.0, 0.0, 0.06, stop_on_bump=False)

        assert result.bumped is True
        assert result.contact_groups == ["head", "torso"]


class TestAttachIsObservational:
    """TR.3 / forensics F-03: attaching a recorder must change NOTHING.

    ``attach_recorder`` used to replace ``playback.execute`` with a wrapper that
    re-entered it in 0.04 s pieces. Moving the command boundary moved five
    things that hang off it — the bump debounce window, the pose-trace phase,
    the clamp-note list, the fall-diagnostics stamp and the odometry noise draw
    — and each was found and fixed as its own bug before the seam itself was
    recognised as the defect. A recorded run and an unrecorded run were
    different experiments; the paid batch only ever ran the recorded one.
    """

    @staticmethod
    def _playback():
        pb = PolicyPlayback.__new__(PolicyPlayback)
        pb._odom_rng = random.Random(0)
        pb._odom_scale = 1.0
        return pb

    def test_attach_does_not_replace_execute(self):
        from duck_embody.sim.recorder import attach_recorder

        pb = self._playback()
        before = pb.execute
        attach_recorder(pb, object(), SimpleNamespace(grab=lambda env: None))
        # Same bound method, and no shadowing instance attribute.
        assert pb.execute.__func__ is before.__func__
        assert "execute" not in vars(pb)

    def test_attach_registers_a_step_observer_and_detach_removes_it(self):
        from duck_embody.sim.recorder import attach_recorder

        pb = self._playback()
        assert pb.step_observer is None
        detach = attach_recorder(pb, object(), SimpleNamespace(grab=lambda env: None))
        assert callable(pb.step_observer)
        detach()
        assert pb.step_observer is None

    def test_a_second_attach_refuses_rather_than_starving_a_consumer(self):
        """Two observers each believing they get every step means one silently
        gets none — a video that is quietly empty, which is the exact rule-11
        failure mode. runner.py's nested-recorder hazard, made loud."""
        import pytest

        from duck_embody.sim.recorder import attach_recorder

        pb = self._playback()
        detach = attach_recorder(pb, object(), SimpleNamespace(grab=lambda env: None))
        with pytest.raises(RuntimeError, match="already registered"):
            attach_recorder(pb, object(), SimpleNamespace(grab=lambda env: None))
        detach()
        attach_recorder(pb, object(), SimpleNamespace(grab=lambda env: None))

    def test_attaching_does_not_change_duration_to_steps(self):
        from duck_embody.sim.policy_wrapper import duration_to_steps
        from duck_embody.sim.recorder import attach_recorder

        pb = self._playback()
        before = [duration_to_steps(d) for d in (0.04, 0.2, 3.0)]
        attach_recorder(pb, object(), SimpleNamespace(grab=lambda env: None))
        assert [duration_to_steps(d) for d in (0.04, 0.2, 3.0)] == before == [2, 10, 150]


class TestStepGrabberSampling:
    """The frame-sampling rule, tested on synthetic observations (no kit)."""

    @staticmethod
    def _obs(step_index: int, terminated: bool = False):
        from duck_embody.sim.policy_wrapper import StepObservation

        return StepObservation(
            step_index=step_index,
            terminated=terminated,
            true_pose=(0.0, 0.0, 0.0),
            contact_force_n=0.0,
            contact_groups=(),
            in_contact=False,
        )

    def test_it_grabs_every_second_control_step_for_25_fps(self):
        from duck_embody.sim.recorder import step_grabber

        grabs = []
        observe = step_grabber(object(), SimpleNamespace(grab=lambda env: grabs.append(1)))
        for i in range(1, 11):
            observe(self._obs(i))
        assert len(grabs) == 5

    def test_a_terminating_step_is_never_grabbed(self):
        """Isaac auto-resets inside step(), so the viewport already shows a
        healthy duck at spawn: grabbing here ended every fall video on a
        post-teleport frame and deleted rule 11's evidence for the event that
        ends trials."""
        from duck_embody.sim.recorder import step_grabber

        grabs = []
        observe = step_grabber(object(), SimpleNamespace(grab=lambda env: grabs.append(1)))
        observe(self._obs(2))
        observe(self._obs(4, terminated=True))
        assert len(grabs) == 1

    def test_the_grid_is_indexed_by_the_trial_step_counter_not_by_call(self):
        """The stride must not restart when a command does — a per-call grid
        would sample at ~50 Hz under the old 2-step chunking, which is the
        pose-trace bug's shape applied to frames."""
        from duck_embody.sim.recorder import step_grabber

        grabs = []
        observe = step_grabber(object(), SimpleNamespace(grab=lambda env: grabs.append(1)))
        # Three separate "calls" covering steps 1-2, 3-4, 5-6: one grab each,
        # never two, because the grid belongs to the trial.
        for i in (1, 2, 3, 4, 5, 6):
            observe(self._obs(i))
        assert len(grabs) == 3

    def test_the_stride_follows_the_requested_interval(self):
        from duck_embody.sim.recorder import step_grabber, steps_per_grab

        assert steps_per_grab(0.04) == 2
        assert steps_per_grab(0.2) == 10
        grabs = []
        observe = step_grabber(
            object(), SimpleNamespace(grab=lambda env: grabs.append(1)), chunk_s=0.2
        )
        for i in range(1, 21):
            observe(self._obs(i))
        assert len(grabs) == 2


class TestExecuteEmitsStepObservations:
    """``execute()`` emits one observation per non-teleported control step."""

    def test_every_step_is_observed_with_the_trial_step_index(self):
        pb = TestExecuteTerminationOrdering._playback(terminated=False)
        pb.bump_contact_force = lambda: 0.0
        pb.contact_groups = lambda: []
        pb._bump_run = 0
        seen = []
        pb.step_observer = seen.append

        pb.execute(0.2, 0.0, 0.0, 0.06)  # 3 control steps

        assert [o.step_index for o in seen] == [1, 2, 3]
        assert all(o.terminated is False for o in seen)
        assert all(o.true_pose == (1.0, 2.0, 45.0) for o in seen)

    def test_the_terminating_step_is_emitted_and_flagged(self):
        pb = TestExecuteTerminationOrdering._playback(terminated=True)
        seen = []
        pb.step_observer = seen.append

        result = pb.execute(0.2, 0.0, 0.0, 0.2)

        assert result.fell is True
        assert len(seen) == 1
        assert seen[0].terminated is True
        # The PRE-step pose, like fall_diagnostics: live state is the teleport.
        assert seen[0].true_pose == (1.0, 2.0, 45.0)

    def test_contact_state_reaches_the_observer(self):
        pb = TestExecuteTerminationOrdering._playback(terminated=False)
        pb._bump_run = BUMP_DEBOUNCE_STEPS - 1  # confirms on the first step
        seen = []
        pb.step_observer = seen.append

        pb.execute(0.2, 0.0, 0.0, 0.04, stop_on_bump=False)

        assert seen[0].in_contact is True
        assert seen[0].contact_groups == ("head",)
        assert seen[0].contact_force_n == 500.0

    def test_an_observer_cannot_change_the_result(self):
        """Passive by construction: the return value is ignored and the
        observation is frozen, so a consumer cannot reach back into physics."""
        import dataclasses

        import pytest

        pb = TestExecuteTerminationOrdering._playback(terminated=False)
        pb.bump_contact_force = lambda: 0.0
        pb.contact_groups = lambda: []
        pb._bump_run = 0
        quiet = pb.execute(0.2, 0.0, 0.0, 0.06)

        pb2 = TestExecuteTerminationOrdering._playback(terminated=False)
        pb2.bump_contact_force = lambda: 0.0
        pb2.contact_groups = lambda: []
        pb2._bump_run = 0
        captured = []

        def greedy(obs):
            captured.append(obs)
            return "ignored"

        pb2.step_observer = greedy
        observed = pb2.execute(0.2, 0.0, 0.0, 0.06)

        assert dataclasses.is_dataclass(captured[0])
        with pytest.raises(dataclasses.FrozenInstanceError):
            captured[0].terminated = True
        for field_name in ("steps", "policy_seconds", "bumped", "fell",
                           "contact_steps", "odom_dxy", "odom_distance_m",
                           "sampled_xy", "true_pose", "stop_reason"):
            assert getattr(observed, field_name) == getattr(quiet, field_name), field_name


class TestSustainedContactAbort:
    """`move` aborts on SUSTAINED contact, never on a single-chunk graze.

    MEASURED (gap-hunt S4, 2026-07-27): under abort-on-first-bump, a 60 ms
    right-leg graze stopped a 1.0 m move at 0.347 m with 0.80 m geometrically
    free ahead. The model was told `bumped: true` in open space — the harness
    conflating CONTACT with BLOCKED and refusing a viable command, which is the
    harness making the model's decision (doc 05 §1). One bumping chunk now
    reports contact without stopping; MOVE_ABORT_SUSTAINED_CHUNKS (=2)
    consecutive bumping chunks abort.
    """

    @staticmethod
    def _playback(fake_execute):
        pb = PolicyPlayback.__new__(PolicyPlayback)
        # Real execute() consumes the leg-odometry error model that
        # __init__ seeds; a __new__-built instance must supply it.
        pb._odom_rng = random.Random(0)
        pb._odom_scale = 1.0
        pb.compass_deg = lambda: 0.0
        pb.true_xy = lambda: (0.0, 0.0)
        pb.execute = fake_execute
        return pb

    @staticmethod
    def _chunk(bumped: bool, x: float, groups=None):
        return _res(
            steps=10,
            policy_seconds=0.2,
            bumped=bumped,
            contact_groups=list(groups or ([] if not bumped else ["right_leg"])),
            pose_trace=[(x, 0.0), (x + 0.04, 0.0)],
            true_pose=(x + 0.04, 0.0, 0.0),
        )

    def test_a_single_graze_chunk_does_not_abort_the_move(self):
        """Chunk 2 bumps, chunk 3 is clean: the move must run to its target,
        still REPORT the contact (bumped + groups survive the merge), and end
        with stop_reason 'reached' — not 'bump' at a third of the distance."""
        calls = {"n": 0}

        def fake_execute(vx, vy, wz, duration_s, stop_on_bump=False, **kw):
            if vx == 0.0:  # settle
                return self._chunk(False, 0.24)
            calls["n"] += 1
            return self._chunk(calls["n"] == 2, 0.04 * (calls["n"] - 1))

        pb = self._playback(fake_execute)
        result = pb.move(0.20, stop_on_bump=True)

        assert result.stop_reason == "reached", result.stop_reason
        assert result.bumped is True, "the graze must still be reported"
        assert "right_leg" in result.contact_groups
        assert calls["n"] >= 5, "the move gave up early on a single graze"

    def test_two_consecutive_bumping_chunks_abort(self):
        calls = {"n": 0}

        def fake_execute(vx, vy, wz, duration_s, stop_on_bump=False, **kw):
            if vx == 0.0:
                return self._chunk(False, 0.12)
            calls["n"] += 1
            return self._chunk(calls["n"] >= 2, 0.04 * (calls["n"] - 1))

        pb = self._playback(fake_execute)
        result = pb.move(1.0, stop_on_bump=True)

        assert result.stop_reason == "bump"
        assert result.stopped_early is True
        assert calls["n"] == 3, f"expected abort on the 3rd chunk (2nd consecutive bump), got {calls['n']}"

    def test_separated_grazes_never_accumulate(self):
        """bump, clean, bump, clean ... must not count as 'sustained'."""
        calls = {"n": 0}

        def fake_execute(vx, vy, wz, duration_s, stop_on_bump=False, **kw):
            if vx == 0.0:
                return self._chunk(False, 0.4)
            calls["n"] += 1
            return self._chunk(calls["n"] % 2 == 1, 0.04 * (calls["n"] - 1))

        pb = self._playback(fake_execute)
        result = pb.move(0.32, stop_on_bump=True)

        assert result.stop_reason == "reached"
        assert result.bumped is True

    def test_send_velocity_semantics_unchanged(self):
        """The sustained gate is a MOVE policy; execute()'s own per-chunk bump
        reporting (what send_velocity returns) is untouched — pinned by reading
        the constant is only consumed inside move()."""
        import inspect

        from duck_embody.sim import policy_wrapper

        src = inspect.getsource(policy_wrapper)
        uses = [
            line for line in src.splitlines()
            if "MOVE_ABORT_SUSTAINED_CHUNKS" in line
            and not line.strip().startswith("#")
            and "=" not in line.split("MOVE_ABORT_SUSTAINED_CHUNKS")[0] + "X"
        ]
        move_src = inspect.getsource(policy_wrapper.PolicyPlayback.move)
        assert "MOVE_ABORT_SUSTAINED_CHUNKS" in move_src
        exec_src = inspect.getsource(policy_wrapper.PolicyPlayback.execute)
        assert "MOVE_ABORT_SUSTAINED_CHUNKS" not in exec_src
