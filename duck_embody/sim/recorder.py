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
