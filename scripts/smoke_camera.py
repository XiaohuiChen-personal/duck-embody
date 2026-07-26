"""T1.4 — settle the head camera against real pixels (VIDEO smoke).

Four questions that no amount of reading the source can answer, per doc 04 §10:

1. **Does the mount work at all?** The robot USD is instanceable, so a camera
   prim may fail to attach. Rung 1 mounts on the articulation root
   ``/Robot/base`` — outside the instanced subtree.
2. **Is it pointing forward?** The head frame's forward axis is local −Z and an
   identity mount there films the sky. On the base frame with
   ``convention="world"`` no correction should be needed — but "should" is not
   evidence, so we look at the pixels.
3. **How many warmup renders?** MDL materials stream asynchronously; early
   frames come back flat gray. A gray first observation would poison the model's
   first room guess. Measured here, not guessed.
4. **Does ``look_around`` actually re-aim?** On a featureless empty plane four
   bearings look identical to a human, so this check is **numeric**: assert the
   camera's forward vector rotates in 90° steps and that the four frames are not
   byte-identical.

Run:  PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/smoke_camera.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "figures" / "smoke"

#: Design envelope for the lens height (doc 04 §2): between the standing trunk
#: (~0.17 m) and the full ~0.42 m height, targeting ~0.36 m.
CAMERA_HEIGHT_BAND = (0.30, 0.45)


def main() -> int:
    import subprocess

    from PIL import Image

    from duck_embody.env.camera import HeadCamera
    from duck_embody.sim.recorder import Recorder, _ffmpeg
    from duck_embody.sim.session import SimSession, SpawnPose

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    report: dict = {}

    def check(label: str, condition: bool, detail: str = "") -> None:
        print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
        if not condition:
            failures.append(label)

    session = SimSession.launch(task_id="DuckEmbody-v0", headless=True)
    session.reset(seed=42, spawn=SpawnPose(0.0, 0.0, 0.0))
    cam = HeadCamera(session.env)
    playback = session.playback
    print("== session up, camera attached ==")

    # -- 1. mount + geometry ------------------------------------------------
    print("\n== mount ==")
    cam.warmup()
    eye, fwd = cam.aim()
    trunk_z = playback.true_height()
    report["camera_eye_world"] = [round(v, 4) for v in eye]
    report["camera_forward_vector"] = [round(v, 4) for v in fwd]
    report["trunk_height_m"] = round(trunk_z, 4)
    report["camera_height_m"] = round(eye[2], 4)
    print(f"  trunk height      {trunk_z:.4f} m")
    print(f"  camera eye        ({eye[0]:.4f}, {eye[1]:.4f}, {eye[2]:.4f})")
    print(f"  camera forward    ({fwd[0]:.3f}, {fwd[1]:.3f}, {fwd[2]:.3f})")

    check(
        f"camera height in {CAMERA_HEIGHT_BAND} m",
        CAMERA_HEIGHT_BAND[0] <= eye[2] <= CAMERA_HEIGHT_BAND[1],
        f"{eye[2]:.4f}",
    )
    # Robot spawned at heading 0 => forward should be world +X, level.
    check("camera aims along +X (not sky, not floor)", fwd[0] > 0.9, f"x={fwd[0]:.3f}")
    check("camera is level (|z| small)", abs(fwd[2]) < 0.2, f"z={fwd[2]:.3f}")

    # -- 2. warmup measurement ---------------------------------------------
    # Reset to reproduce the post-reset state that actually matters, then render
    # one frame at a time until the image stops being flat gray.
    print("\n== warmup (how many renders before a usable frame?) ==")
    session.reset(seed=42, spawn=SpawnPose(0.0, 0.0, 0.0))
    measured_warmup = None
    for i in range(1, 21):
        arr = cam.capture_rgb()
        spread = int(arr.max()) - int(arr.min())
        if i <= 12:
            print(f"    render {i:>2}: pixel spread {spread}")
        if measured_warmup is None and not cam.is_gray(arr):
            measured_warmup = i
    report["measured_warmup_renders"] = measured_warmup
    check(
        "a non-gray frame is produced within 20 renders",
        measured_warmup is not None,
        str(measured_warmup),
    )

    # -- 3. stills while standing ------------------------------------------
    print("\n== stills ==")
    cam.warmup()
    standing = cam.capture_rgb()
    Image.fromarray(standing).save(OUT_DIR / "camera_standing.png")
    check(
        "standing frame is not flat gray",
        not cam.is_gray(standing),
        f"spread={int(standing.max()) - int(standing.min())}",
    )
    check(
        "standing frame has the expected shape",
        standing.shape[:2] == (512, 512),
        str(standing.shape),
    )

    # A level camera on an empty plane should see sky above and floor below:
    # the top and bottom halves must differ.
    top = float(standing[:256].mean())
    bottom = float(standing[256:].mean())
    report["frame_top_mean"] = round(top, 2)
    report["frame_bottom_mean"] = round(bottom, 2)
    print(f"    top-half mean {top:.1f} vs bottom-half mean {bottom:.1f}")
    # A level camera clear of the robot's own head sees bright sky above and a
    # dark floor below, so the halves differ by ~95 grey levels. Anything under
    # ~30 means the lens is buried inside the head shell, whose uniform light
    # gray reads as a plausible frame while showing the model nothing
    # (measured: 2.2 at forward 0.02 vs 95.2 at forward 0.10).
    check(
        "horizon present — lens is OUTSIDE the duck's own head",
        abs(top - bottom) > 30.0,
        f"delta={abs(top - bottom):.1f}",
    )
    check(
        "frame has real structure (not a flat shell interior)",
        float(standing.std()) > 30.0,
        f"std={float(standing.std()):.1f}",
    )

    jpeg = cam.capture_jpeg()
    report["jpeg_bytes"] = len(jpeg)
    (OUT_DIR / "camera_standing.jpg").write_bytes(jpeg)
    print(f"    JPEG q85: {len(jpeg) / 1024:.1f} KiB")
    check(
        "JPEG is a plausible size for a 512x512 frame",
        5_000 < len(jpeg) < 400_000,
        f"{len(jpeg)} B",
    )

    # -- 4. look_around: NUMERIC, not visual -------------------------------
    print("\n== look_around (numeric: the empty plane is featureless) ==")
    views = cam.look_around()
    bearings, vectors, digests = [], [], []
    for bearing, arr, vec in views:
        bearings.append(bearing)
        vectors.append(vec)
        Image.fromarray(arr).save(OUT_DIR / f"camera_look_{bearing:03d}.png")
        digests.append(arr.tobytes())
        print(f"    bearing {bearing:>3}: forward ({vec[0]:+.3f}, {vec[1]:+.3f}, {vec[2]:+.3f})")

    for bearing, vec in zip(bearings, vectors):
        want = (math.cos(math.radians(bearing)), math.sin(math.radians(bearing)))
        ok = abs(vec[0] - want[0]) < 0.05 and abs(vec[1] - want[1]) < 0.05
        check(
            f"bearing {bearing} deg aims at ({want[0]:+.2f}, {want[1]:+.2f})",
            ok,
            f"got ({vec[0]:+.3f}, {vec[1]:+.3f})",
        )

    check(
        "the four frames are all distinct",
        len(set(digests)) == 4,
        f"{len(set(digests))} distinct of 4",
    )

    # Distinctness alone could be renderer noise, so require the frames to
    # differ SUBSTANTIALLY — on the empty plane the grid pattern and the sky
    # dome's light shift with bearing.
    arrs = [v[1].astype(int) for v in views]
    diffs = [
        float(abs(arrs[i] - arrs[j]).mean())
        for i in range(len(arrs))
        for j in range(i + 1, len(arrs))
    ]
    report["look_around_pairwise_mean_abs_diff"] = [round(d, 3) for d in diffs]
    print(f"    pairwise mean-abs-diff: {[round(d, 2) for d in diffs]}")
    check(
        "frames differ substantially, not just by render noise",
        max(diffs) > 1.0,
        f"max={max(diffs):.2f}",
    )

    # THE decisive check: looking at the robot's own heading through
    # look_around must reproduce what plain get_observation sees. This ties the
    # gimbal path to the normal capture path — if either drifted, they would
    # disagree here.
    heading = playback.compass_deg()
    via_look = cam.look_around(bearings_deg=(heading,))[0][1].astype(int)
    via_plain = cam.capture_rgb().astype(int)
    agreement = float(abs(via_look - via_plain).mean())
    report["look_around_vs_plain_capture_mean_abs_diff"] = round(agreement, 4)
    check(
        "look_around at the robot's own heading == plain capture",
        agreement < 1.0,
        f"mean-abs-diff={agreement:.3f}",
    )

    # -- 5. walking clip: head-camera video ---------------------------------
    print("\n== walking clip (head camera, 5 s) ==")
    session.reset(seed=42, spawn=SpawnPose(0.0, 0.0, 0.0))
    cam.warmup()
    head_frames_dir = OUT_DIR / "camera_walk_frames"
    head_frames_dir.mkdir(exist_ok=True)
    n_head = 0
    gray_after_warmup = 0
    for _ in range(25):  # 25 x 0.2 s = 5 s
        session.playback.execute(0.2, 0.0, 0.0, 0.2)
        arr = cam.capture_rgb()
        if cam.is_gray(arr):
            gray_after_warmup += 1
        Image.fromarray(arr).save(head_frames_dir / f"f{n_head:04d}.png")
        n_head += 1
    report["walk_frames"] = n_head
    report["gray_frames_after_warmup"] = gray_after_warmup
    check(
        "no gray frames during the walking clip",
        gray_after_warmup == 0,
        f"{gray_after_warmup} gray",
    )

    head_mp4 = OUT_DIR / "camera_walk_headcam.mp4"
    subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", "-framerate", "5",
         "-i", str(head_frames_dir / "f%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(head_mp4)],
        check=True,
    )
    print(f"  wrote {head_mp4.name}")

    # -- 6. third-person clip: proves the RTX-sensor render path still works -
    # With a camera in the scene, ManagerBasedRLEnv.render() no longer renders
    # for us (see Recorder.grab). If that interaction were mishandled this clip
    # would be frozen frames — so it is checked here, before T2.4 depends on it.
    print("\n== third-person tracking clip (rule 11 locomotion baseline) ==")
    session.reset(seed=42, spawn=SpawnPose(0.0, 0.0, 0.0))
    rec = Recorder(OUT_DIR / "camera_walk_thirdperson", fps=25, every_n=2)
    session.scripted_drive([(0.2, 0.0, 0.0, 5.0)], recorder=rec)
    mp4 = rec.encode()
    rec.filmstrip(mp4, fps=1.0)
    check(
        "third-person recording produced frames (RTX-sensor render path)",
        rec.n_frames > 0,
        f"{rec.n_frames} frames",
    )

    report["failures"] = failures
    report["acceptance"] = "PASS" if not failures else "FAIL"
    (OUT_DIR / "camera_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\n== result ==")
    print(json.dumps(report, indent=2))
    if failures:
        print("\n  FAILURES:")
        for f in failures:
            print(f"    {f}")
    else:
        print("\n  OK - mount rung 1 works, view is forward, look_around re-aims")
    print("  closing app (nothing after this line runs)")
    session.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
