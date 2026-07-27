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
    """recorder.chunked_execute is the merge every RECORDED command runs through
    (attach_recorder patches playback.execute, and run_trial attaches unless
    --no-video). Its merge block must not lose fields that only the
    terminating/bumping chunk carries."""

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
