"""Video recording for rule-11 verification.

AGENTS.md rule 11 exists because of a specific, expensive lesson from the parent
project: **a run passed every aggregate metric while the robot was crawling**,
and only the video caught it. The owner works over SSH with no live viewport, so
every simulation smoke test must leave behind an mp4 and a filmstrip that can be
inspected frame by frame.

Two things this module depends on, both owned by earlier tasks:

* ``gym.make(..., render_mode="rgb_array")`` — without it
  ``ManagerBasedRLEnv.render()`` returns ``None`` and every frame grab is a
  silent no-op. ``session.py`` passes it.
* ``ViewerCfg(origin_type="asset_body", ...)`` — without it the viewport is a
  fixed shot and the duck walks out of frame. ``DuckEmbodyEnvCfg`` sets it.

Frames are written as PNGs and encoded with ffmpeg at the end rather than piped
live, so a crashed run still leaves inspectable frames behind.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

#: imageio-ffmpeg's static build; verified present (T0.0: ffmpeg 7.0.2-static).
FFMPEG = os.environ.get("DUCK_EMBODY_FFMPEG") or str(Path.home() / ".local/bin/ffmpeg")


def _ffmpeg() -> str:
    if Path(FFMPEG).exists():
        return FFMPEG
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        f"ffmpeg not found at {FFMPEG} and not on PATH. "
        "Set DUCK_EMBODY_FFMPEG to a working binary."
    )


class Recorder:
    """Grabs viewport frames during a run and encodes them afterwards.

    Usage::

        rec = Recorder(out_dir / "run", fps=25)
        ...
        rec.grab(env)          # once per control step, or every Nth
        mp4 = rec.encode()
        strip = rec.filmstrip(mp4)
    """

    def __init__(
        self,
        out_prefix: Path | str,
        fps: int = 25,
        every_n: int = 2,
        hide_ceiling: bool = False,
    ):
        #: Hide the apartment ceiling for the duration of each viewport grab.
        #:
        #: The audit view and the model's view want opposite things. The chase
        #: camera sits 2 m up so it can see over the 0.7 m walls (anything lower
        #: is buried in a wall slab or jammed against furniture — measured in
        #: scripts/debug_viewer_offset.py), which puts it ABOVE the ceiling the
        #: T2.3 gate added; with the roof on, every audit frame is a photo of the
        #: roof. The head camera the models actually see needs that same ceiling,
        #: or the judge reads each room as an outdoor terrace.
        #:
        #: They never render at the same instant — viewport grabs happen on chunk
        #: boundaries, head captures only on an explicit look/look_around — so
        #: the roof comes off for the grab and goes straight back on. Restoration
        #: is in a `finally`, because a roofless stage would silently degrade
        #: every subsequent observation rather than fail.
        self.hide_ceiling = hide_ceiling
        self.out_prefix = Path(out_prefix)
        self.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.out_prefix.parent / f"{self.out_prefix.name}_frames"
        self.fps = fps
        #: Grab every Nth call. Control runs at 50 Hz; every 2nd step gives a
        #: 25 fps video, which is smooth enough to judge gait without doubling
        #: the render cost of a 120 s hold.
        self.every_n = max(1, every_n)
        self._calls = 0
        self._n_frames = 0
        self._warned_none = False

        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
        self.frames_dir.mkdir(parents=True)

    def grab(self, env) -> None:
        """Capture one viewport frame, if this call lands on the sampling grid.

        The explicit ``env.sim.render()`` is load-bearing and easy to lose.
        ``ManagerBasedRLEnv.render()`` only renders for you when the scene has
        **no** RTX sensors::

            if not self.sim.has_rtx_sensors() and not recompute:
                self.sim.render()

        Once the head camera exists (T1.4), ``has_rtx_sensors()`` is True and
        ``render()`` assumes the step loop already produced a frame at
        ``sim.render_interval`` — but we deliberately raised that interval to
        10,000 so RTX does not run at 50 Hz. The two settings combine to hand
        back a **stale viewport buffer**: every mp4 from T2.4 onward would have
        been frozen frames while every metric looked healthy, which is the exact
        failure mode rule 11 exists to catch.

        So: render explicitly, then ask render() only to read the annotator
        (``recompute=True`` suppresses its own render call, avoiding a double
        render). Correct whether or not a camera is present.
        """
        self._calls += 1
        if (self._calls - 1) % self.every_n:
            return
        if self.hide_ceiling:
            from duck_embody.env.scene_builder import ceiling_hidden

            with ceiling_hidden(env.sim.stage, verbose=False):
                env.sim.render()
                frame = env.render(recompute=True)
        else:
            env.sim.render()
            frame = env.render(recompute=True)
        if frame is None:
            # Loud once, not per step: a silent None here is exactly how a
            # rule-11 smoke test ends up with an empty video and a green tick.
            if not self._warned_none:
                print(
                    "  [recorder] WARNING: env.render() returned None — was the env "
                    'created with render_mode="rgb_array"? No video will be produced.'
                )
                self._warned_none = True
            return
        from PIL import Image

        Image.fromarray(frame).save(self.frames_dir / f"f{self._n_frames:06d}.png")
        self._n_frames += 1

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def encode(self, keep_frames: bool = False) -> Path | None:
        """Encode the grabbed frames into an mp4. Returns None if none were grabbed."""
        if self._n_frames == 0:
            print(f"  [recorder] no frames captured for {self.out_prefix.name}; no mp4 written")
            return None
        mp4 = self.out_prefix.with_suffix(".mp4")
        cmd = [
            _ffmpeg(), "-y", "-loglevel", "error",
            "-framerate", str(self.fps),
            "-i", str(self.frames_dir / "f%06d.png"),
            # yuv420p + even dimensions: without both, the mp4 plays in ffplay
            # but not in a browser or QuickTime, which is where it gets reviewed.
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(mp4),
        ]
        subprocess.run(cmd, check=True)
        if not keep_frames:
            shutil.rmtree(self.frames_dir, ignore_errors=True)
        print(f"  [recorder] {mp4.name}: {self._n_frames} frames @ {self.fps} fps")
        return mp4

    def filmstrip(self, mp4: Path, fps: float = 1.0, cols: int = 6) -> Path | None:
        """Extract a 1-frame-per-second contact sheet from ``mp4``.

        This is the artifact rule 11(b) asks for: the grid is what makes
        "trunk upright, feet alternating, no drag" checkable at a glance.
        """
        if mp4 is None or not Path(mp4).exists():
            return None
        out = Path(mp4).with_name(Path(mp4).stem + "_filmstrip.png")

        # Rows must be given EXPLICITLY. ffmpeg's documented `tile=COLSx0`
        # ("as many rows as needed") is rejected by this build —
        # 'Unable to parse option value "6x0" as image size' (ffmpeg 7.0.2
        # static, verified). Compute the row count from how many frames the
        # fps filter will actually emit.
        duration_s = self._n_frames / float(self.fps) if self.fps else 0.0
        n_sampled = max(1, math.ceil(duration_s * fps))
        rows = max(1, math.ceil(n_sampled / cols))

        cmd = [
            _ffmpeg(), "-y", "-loglevel", "error",
            "-i", str(mp4),
            "-vf", f"fps={fps},scale=320:-1,tile={cols}x{rows}",
            "-frames:v", "1",
            str(out),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"  [recorder] filmstrip failed for {mp4.name}: {exc}")
            return None
        print(f"  [recorder] {out.name}")
        return out


def record_and_strip(out_prefix: Path | str, fps: int = 25) -> Recorder:
    """Convenience constructor mirroring how the smoke scripts use it."""
    return Recorder(out_prefix, fps=fps)


# ---------------------------------------------------------------------------
# Recording an LLM-driven episode (T3.4)
# ---------------------------------------------------------------------------

#: Recording interval: 0.04 s = 2 control steps at 50 Hz, i.e. one grab per
#: 25 fps frame. Historically the length of the execution CHUNK the recorder cut
#: commands into; since TR.3 it is a *sampling stride* over control steps and
#: nothing about execution depends on it. Kept under the old name because
#: callers (runner, smokes, tests) pass it as a duration.
RECORD_CHUNK_S = 0.04


def steps_per_grab(chunk_s: float = RECORD_CHUNK_S) -> int:
    """Sampling stride in control steps for a desired grab interval.

    0.04 s -> 2 steps at 50 Hz, which is the 25 fps the mp4 is encoded at.
    """
    from duck_embody.sim.policy_wrapper import CONTROL_HZ

    return max(1, round(chunk_s * CONTROL_HZ))


def chunked_execute(execute, env, recorder, vx, vy, wz, duration_s, *,
                    chunk_s: float = RECORD_CHUNK_S, **kwargs):
    """**NON-BENCHMARK / LEGACY.** Run a command in pieces, grabbing between them.

    RETIRED FROM THE BENCHMARK PATH by TR.3 (forensics F-03). It survives for
    exactly one purpose: replaying the frozen `results/raw_v5d*` batches, whose
    trials really did execute this way (`scripts/replay_falls.py` — a historical
    forensic tool, not a benchmark path). Do NOT reintroduce it into
    `attach_recorder`, `SimSession`, `runner.py` or any new smoke.

    Why it is wrong to record with: cutting one semantic command into 0.04 s
    executes moves the command boundary, and the command boundary carries the
    bump debounce window, the pose-trace phase, the clamp-note list, the fall
    diagnostics stamp and (worst) the odometry noise draw. Recorded and
    unrecorded runs were therefore different experiments — and the paid batch
    only ever ran the recorded one. Use :func:`attach_recorder`, which observes
    fixed control steps and changes nothing.

    Frame grabbing has to interleave with stepping, so a long command cannot be
    recorded from the outside — it has to be cut up. ``execute`` is passed in
    (rather than a ``PolicyPlayback``) so :func:`attach_recorder` can hand over
    the *unpatched* bound method and not recurse into itself.

    The chunk-fold itself is ``policy_wrapper.merge_exec_results`` — the SAME
    function the motion macros merge through, on purpose. This block used to
    carry its own hand-mirrored field list, and the copy drifted: it never
    merged ``contact_groups`` or ``fall_diagnostics``, so every recorded run
    (the default, and the rule-11-mandatory batch path) showed the model
    ``bumped: true, contact: []`` and logged ``fell: true`` with no
    diagnostics whenever the confirming chunk was not the first — the common
    case, since BUMP_DEBOUNCE_STEPS=3 exceeds a 2-step chunk. That is T3.5's
    "fell with no diagnostics" artifact, root-caused at last (the
    stale-bytecode theory run_trial.py used to state was wrong).

    What stays here is the trace bookkeeping: only the 5 Hz ``sampled_xy``
    samples are concatenated (by the merge), and the start/end bookends are
    added once at the end. Concatenating each chunk's full ``pose_trace``
    instead would contribute two extra points per 2-step chunk — a ~50 Hz
    trace of per-step gait sway — which inflates the SPL path integral
    doc 06 §5.3 pins to 5 Hz.
    """
    import math

    from duck_embody.sim.policy_wrapper import (
        CONTROL_DT,
        ExecResult,
        duration_to_steps,
        merge_exec_results,
    )

    total_steps = duration_to_steps(duration_s)
    done_steps = 0
    merged: "ExecResult | None" = None
    start_xy: tuple[float, float] | None = None

    while done_steps < total_steps:
        remaining = (total_steps - done_steps) * CONTROL_DT
        part = execute(vx, vy, wz, min(chunk_s, remaining), **kwargs)
        if not part.fell:
            # No grab after the falling piece: Isaac auto-reset the terminated
            # env INSIDE step(), so the viewport now shows a healthy duck
            # standing at spawn — grabbing it made every fall video END on a
            # post-teleport frame, structurally deleting rule 11's "video wins"
            # evidence for the most consequential event a trial has. Skipping
            # leaves the previous piece's frame (<= 40 ms before the topple
            # began) as the last one; S2 of scripts/smoke_gap_hunt.py verifies
            # the final frames no longer show the spawn pose.
            recorder.grab(env)
        done_steps += part.steps

        if start_xy is None:
            start_xy = part.pose_trace[0]
        merged = merge_exec_results(merged, part)

        if part.fell or part.stopped_early:
            break

    end_xy = (merged.true_pose[0], merged.true_pose[1])
    merged.duration_s = duration_s
    merged.pose_trace = [start_xy, *merged.sampled_xy, end_xy]
    merged.true_displacement_m = math.dist(start_xy, end_xy)
    return merged


def step_grabber(env, recorder, chunk_s: float = RECORD_CHUNK_S):
    """Build the ``StepObservation`` handler that grabs viewport frames.

    Pure of any playback state, so ``tests/test_execute_ordering.py`` can feed
    it synthetic observations and assert the sampling rule without a kit
    process. Two rules, both learned expensively:

    * **Sample on a fixed control-step grid**, ``chunk_s`` wide (2 steps =
      25 fps). The grid is indexed by the trial-scoped step counter, so it does
      not move when a command starts or ends — which is what makes recorded and
      unrecorded runs the same experiment (forensics F-03).
    * **Never grab on a terminating step.** Isaac Lab auto-resets a terminated
      env inside ``step()``, so the live viewport already shows a healthy duck
      at spawn; grabbing it made every fall video END on a post-teleport frame,
      structurally deleting rule 11's "video wins" evidence for the single most
      consequential event a trial has. The previous frame (<= 40 ms before the
      topple) stays the last one.
    """
    stride = steps_per_grab(chunk_s)

    def observe(obs) -> None:
        if obs.terminated:
            return
        if obs.step_index % stride:
            return
        recorder.grab(env)

    return observe


def attach_recorder(playback, env, recorder, chunk_s: float = RECORD_CHUNK_S):
    """Record video by OBSERVING control steps. Returns a detach callable.

    Drop-in replacement for the pre-TR.3 signature, with one load-bearing
    difference: it **does not replace** ``playback.execute``. It registers a
    passive per-control-step observer (``PolicyPlayback.register_step_observer``)
    that grabs a frame every ``chunk_s`` of simulated time and returns nothing.
    Attaching therefore cannot change command boundaries, bump timing, pose
    sampling, clamp notes, fall diagnostics or the odometry noise process —
    which the patch it replaces changed, all of them, and each was fixed as a
    separate bug before the seam itself was recognised as the defect
    (forensics F-03; `tests/test_execute_ordering.py::TestAttachIsObservational`
    pins that ``execute`` is untouched).

    Frame rate is unchanged: the old patch grabbed once per 0.04 s execution
    chunk, this grabs once per 0.04 s of stepping. ``Recorder.every_n`` still
    decimates on top, so ``Recorder(..., every_n=1)`` remains 25 fps.

    Without a recorder attached somewhere there is **no per-trial video at
    all**: ``agent/tools.py`` drives motion through ``playback.move()`` /
    ``turn_to_heading()`` / ``execute()`` and never passes an ``on_chunk``
    callback, and ``SimSession``'s scripted-drive grabber is not on the LLM
    path. AGENTS.md rule 11 makes a video the acceptance evidence for any run
    that steps simulation.

    ``stop_predicate`` needs no special case any more (the pre-TR.3 patch had to
    fall through unrecorded, because chunking restarted the per-call step index
    a predicate is defined over). Observation is orthogonal to it, so a
    predicate-driven servo now records normally.
    """
    return playback.register_step_observer(step_grabber(env, recorder, chunk_s))
