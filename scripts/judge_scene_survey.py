"""T2.3 part 2 (NO kit): the out-of-benchmark room-recognition gate.

Runs offline from the PNGs ``render_scene_survey.py`` saved, after that process
has exited.

**The judge is deliberately not a contestant.** Doc 04 §8 requires "a VLM that is
not one of the three benchmark models ... to avoid tuning the scene to any
contestant's strengths", and this task iterates the layout until the probe
passes — so using Fable 5, Opus 5, or GPT 5.6 sol would tune the *world* to that
model, which is a benchmark-integrity defect, not a shortcut. The judge is
Sonnet 5 (``configs/models/judge.yaml``), named in the methods write-up.

Each **pose** is judged once from its whole four-bearing sweep, because that is
what ``look_around()`` hands a contestant — judging isolated frames would set a
bar the benchmark never asks a model to clear. A room's verdict is the majority
across its poses.

**Acceptance: all four rooms named correctly.** Do not silently relax it; any
relaxation updates doc 03 §8.2 and doc 04 §8 in the same commit, with rationale.

Run:  ~/IsaacLab/isaaclab.sh -p scripts/judge_scene_survey.py
"""

from __future__ import annotations

import base64
import io
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SURVEY_DIR = REPO_ROOT / "results" / "figures" / "survey"
OUT = SURVEY_DIR / "judge_result.json"

JPEG_QUALITY = 85


def to_jpeg_b64(path: Path) -> str:
    """Match the wire format a contestant sees: 512x512 JPEG q85."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_gate(judge, survey, question, extract) -> tuple[dict, list[dict], int]:
    """One full pass of the gate. Returns (per-room verdicts, per-pose, n_pass)."""
    poses: dict[tuple[str, int], list[dict]] = {}
    for frame in survey["frames"]:
        poses.setdefault((frame["room"], frame["pose_index"]), []).append(frame)

    results: list[dict] = []
    for (room, pose_idx), frames in sorted(poses.items()):
        frames = sorted(frames, key=lambda f: f["bearing_deg"])
        images = [to_jpeg_b64(SURVEY_DIR / f["file"]) for f in frames]
        labels = [f"view at compass {f['bearing_deg']} deg" for f in frames]
        reply = judge.ask_about_images(question, images, labels=labels, max_tokens=200)
        # Judges do not reliably answer in one word even when asked (T3.3
        # measured exactly that), so extract the room term using the SAME frozen
        # synonym table the scorer uses.
        guess = extract(reply)
        results.append(
            {
                "room": room, "pose_index": pose_idx, "xy": frames[0]["xy"],
                "guess": guess, "correct": guess == room, "reply": reply,
            }
        )
        print(f"  {'OK  ' if guess == room else 'MISS'} {room:<12} pose {pose_idx} -> {guess!r}")

    verdicts: dict[str, dict] = {}
    for room in sorted({r["room"] for r in results}):
        room_results = [r for r in results if r["room"] == room]
        votes = Counter(r["guess"] for r in room_results)
        winner, count = votes.most_common(1)[0]
        # A TIE IS NOT A MAJORITY. Counter.most_common breaks ties by insertion
        # order, so a 1/1/1 split would silently "pass" whichever guess arrived
        # first — the gate would report 4/4 on an unrecognisable room.
        has_majority = count * 2 > len(room_results)
        passed = has_majority and winner == room
        verdicts[room] = {
            "majority_guess": winner,
            "votes": {str(k): v for k, v in votes.items()},
            "n_poses": len(room_results),
            "has_strict_majority": has_majority,
            "passed": passed,
        }
        note = "" if has_majority else "  [NO MAJORITY - tie]"
        print(
            f"  {'PASS' if passed else 'FAIL'}  {room:<12} top={winner!r} "
            f"({count}/{len(room_results)})  votes={dict(votes)}{note}"
        )
    return verdicts, results, sum(1 for v in verdicts.values() if v["passed"])


def main() -> int:
    import argparse

    from duck_embody.agent.prompts import SURVEY_ROOM_QUESTION, extract_room_mention
    from duck_embody.agent.providers.base import build_provider, load_model_config

    parser = argparse.ArgumentParser()
    # No locked model supports deterministic decoding (T3.3), so one pass of the
    # gate is a sample, not a verdict. Repeating it and requiring EVERY pass to
    # succeed is a stricter bar than a single run, not a weaker one.
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    manifest_path = SURVEY_DIR / "survey_manifest.json"
    if not manifest_path.exists():
        print(f"FATAL: {manifest_path} missing — run scripts/render_scene_survey.py first")
        return 1
    survey = json.loads(manifest_path.read_text())

    cfg = load_model_config("judge")
    judge = build_provider("judge")
    print(f"judge: {cfg.model_id} (out-of-benchmark)")
    print(f"repeats: {args.repeats} — ALL must pass\n")

    runs = []
    for i in range(1, args.repeats + 1):
        print(f"== run {i}/{args.repeats} ==")
        verdicts, results, n_pass = run_gate(
            judge, survey, SURVEY_ROOM_QUESTION, extract_room_mention
        )
        runs.append({"run": i, "per_room": verdicts, "per_pose": results, "rooms_correct": n_pass})
        print(f"  -> {n_pass}/{len(verdicts)}\n")

    n_rooms = len(runs[0]["per_room"])
    all_passed = all(r["rooms_correct"] == n_rooms for r in runs)

    print("== stability across runs ==")
    for room in sorted(runs[0]["per_room"]):
        per_run = [r["per_room"][room]["passed"] for r in runs]
        print(f"  {room:<12} {['PASS' if p else 'FAIL' for p in per_run]}")

    report = {
        "judge_model": cfg.model_id,
        "question": SURVEY_ROOM_QUESTION,
        "repeats": args.repeats,
        "runs": runs,
        "rooms_total": n_rooms,
        "gate": "PASS" if all_passed else "FAIL",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")

    print(f"\n== GATE over {args.repeats} runs ==")
    if all_passed:
        print(f"  PASS — all four rooms named correctly in every one of {args.repeats} runs.")
        print("  Layout may freeze.")
        return 0
    print("  FAIL — fix the SCENE (props, colours, landmarks at duck height),")
    print("         never the per-model camera. Then re-render and re-judge.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
