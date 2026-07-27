"""Single-episode entry point: one model x one seed, inside one kit process.

Writes doc 06 §4's per-trial JSON **incrementally** (turn by turn, atomically —
a crash loses at most the in-flight turn) plus a rule-11 audit mp4 and filmstrip.
Runs both stages of doc 06 §3.1 and the post-episode layout-QA exchange (§5.9),
so the JSON it leaves behind is the complete artifact T4.1 scores.

**This spends real money.** Every invocation is 1-80 model turns of paid API on
one of the three locked contestants plus the 5-question QA exchange. There is no
dry-run mode for the model call by design: a fake response would exercise none
of the paths T3.5 exists to prove. Use ``--help`` to check wiring without
launching anything.

**One GPU job at a time** (AGENTS.md rule 1) — check ``nvidia-smi`` first. The
batch runner (T4.2) launches ONE kit process and loops the matrix inside it;
this script is the single-trial path T3.5 uses, and it launches and closes its
own.

Two hazards this script is shaped around:

* ``SimulationApp.close()`` terminates the process — nothing after it runs. The
  JSON, the mp4, the filmstrip and every verdict are written **before**
  ``session.close()``.
* kit buffers stdout aggressively, so the run line below sets
  ``PYTHONUNBUFFERED=1``. Without it the whole log is discarded at exit.

Disk: the recorder writes one PNG per 0.04 s of simulated motion and deletes
them after encoding. A trial that spends its full 2 x 240 policy-second budget
is ~12,000 frames at peak; pass ``--video-every-n 2`` to halve that, or
``--no-video`` to skip it (which forfeits the rule-11 evidence).

Run:  PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/run_trial.py \\
          --model fable5 --seed 101
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "raw"
DEFAULT_VIDEO_DIR = REPO_ROOT / "results" / "videos"

#: The apartment task. The empty-plane `DuckEmbody-v0` exists for the smoke
#: tests; a benchmark trial in it would have no rooms to find.
TASK_ID = "DuckEmbody-Apartment-v0"

#: Stage-1 task name. `--task` exists because PLAN T3.4 specifies the flag, and
#: because T3.5's GPT dry run wants a few turns without committing to the whole
#: two-stage protocol. There is exactly one benchmark task.
TASKS = ("find_kitchen",)


def frozen_matrix() -> tuple[tuple[str, ...], tuple[int, ...]]:
    """doc 06 §2's frozen roster and seed set, read from the hashed config.

    Read, never hardcoded: `configs/benchmark.yaml` is inside the fairness
    contract §7's guard hashes, so the entry point and the contract cannot
    drift apart. Enforcing it HERE and not only in the tests matters because
    `load_model_config` globs `configs/models/*.yaml`, which also contains
    `judge.yaml` — the out-of-benchmark Sonnet 5 scene judge (doc 04 §8).
    `--model judge` would otherwise run a full paid trial and write a
    benchmark-shaped `results/raw/judge_seed101.json` with a valid `final` and
    the same `config_hash` as the real trials, which a glob-based aggregator
    would fold into the comparison as a fourth, non-roster model.
    """
    import yaml

    raw = yaml.safe_load((REPO_ROOT / "configs" / "benchmark.yaml").read_text())
    return tuple(raw["models"]), tuple(int(s) for s in raw["seeds"])


def build_parser():
    """Argument parser. Built outside ``main`` so ``--help`` needs no kit."""
    import argparse

    models, seeds = frozen_matrix()
    parser = argparse.ArgumentParser(
        prog="run_trial.py",
        description="Run one Duck Embody benchmark trial (one model x one seed).",
    )
    parser.add_argument(
        "--model", required=True, choices=models,
        help="model config name in configs/models/ (the frozen roster)",
    )
    parser.add_argument(
        "--seed", required=True, type=int, choices=seeds,
        help="trial seed (the frozen seed set)",
    )
    parser.add_argument("--task", default="find_kitchen", choices=TASKS)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--checkpoint", default=None, help="policy .pt (default: policy/model_2999.pt)")
    parser.add_argument(
        "--max-turns", type=int, default=None,
        help="per-stage turn cap override for smoke runs ONLY. A benchmark "
             "trial must use the frozen 40 (doc 06 §2); the override is recorded "
             "in the JSON (config.turn_cap_override) so a capped smoke run can "
             "never be mistaken for a result.",
    )
    parser.add_argument("--video-every-n", type=int, default=1,
                        help="grab every Nth recording chunk (1 = 25 fps)")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--headed", action="store_true", help="run with a viewport")
    return parser


def main() -> int:
    # Parsed BEFORE anything launches: AppLauncher inside SimSession.launch()
    # parses sys.argv for its own flags and would choke on ours. Stripping them
    # afterwards leaves kit exactly the argv it expects.
    args, kit_argv = build_parser().parse_known_args()
    sys.argv = [sys.argv[0], *kit_argv]

    if args.max_turns is not None and args.max_turns <= 0:
        print("FATAL: --max-turns must be positive")
        return 2

    # Imports that need no kit, so a config typo fails in a second rather than
    # after a multi-minute cold start.
    from duck_embody.agent.memory import Counters, Memory, PositionIntegrator
    from duck_embody.agent.providers.base import (
        build_provider,
        load_model_config,
        preflight_provider,
    )
    from duck_embody.tasks.find_kitchen import spawn_for_seed, stage_specs

    cfg = load_model_config(args.model)
    spawn_xy, spawn_heading = spawn_for_seed(args.seed)
    trial_id = f"{args.model}_seed{args.seed}"
    out_dir = Path(args.out_dir)
    video_dir = Path(args.video_dir)
    json_path = out_dir / f"{trial_id}.json"

    print(f"== trial {trial_id} ==")
    print(f"  model    : {cfg.model_id} ({cfg.provider})")
    print(f"  seed     : {args.seed}  spawn ({spawn_xy[0]}, {spawn_xy[1]}) @ {spawn_heading} deg")
    print(f"  json     : {json_path}")
    if args.max_turns is not None:
        print(f"  WARNING  : --max-turns {args.max_turns} — SMOKE ONLY, not a benchmark result")

    # Fail fast on a missing key or an unknown provider — WITHOUT importing the
    # vendor SDK, which must not be imported before kit starts (see below).
    preflight_provider(args.model)

    # Heavy imports last, and only after the provider config is known good.
    # Stale-bytecode guard. `isaaclab.sh -p` does NOT set
    # PYTHONDONTWRITEBYTECODE, and Python validates a cached module on
    # (source mtime truncated to SECONDS, source size) — so an edit made in the
    # same second as the previous run, at the same byte count, silently runs the
    # OLD module. AGENTS.md §5 records this biting the test suite; it then bit a
    # T3.5 trial, which recorded `fell: true` with no diagnostics because the
    # running process predated the code that captures them. Cheap to assert,
    # expensive to discover from a finished trial.
    from duck_embody.sim.policy_wrapper import ExecResult as _ExecResult

    if "fall_diagnostics" not in _ExecResult.__dataclass_fields__:
        print("FATAL: loaded policy_wrapper is STALE (no fall_diagnostics). "
              "Clear __pycache__ and re-run.")
        return 2

    from duck_embody.agent.loop import EpisodeRunner, TrialLog
    from duck_embody.agent.tools import ToolContext
    from duck_embody.env.camera import HeadCamera
    from duck_embody.sim.recorder import Recorder, attach_recorder
    from duck_embody.sim.session import SimSession, SpawnPose

    session = SimSession.launch(
        task_id=TASK_ID, checkpoint=args.checkpoint, headless=not args.headed
    )
    print("  kit up; resetting to the seed's spawn pose")

    # THE PROVIDER IS BUILT HERE, AFTER kit, AND THAT ORDER IS LOAD-BEARING.
    #
    # MEASURED (T3.5, first sanity run): importing the `anthropic` SDK BEFORE
    # `AppLauncher` leaves it unable to strip its own unset-parameter defaults.
    # Twelve `omit` sentinels — cache_control, container, inference_geo,
    # metadata, output_config, service_tier, stop_sequences, stream,
    # temperature, tool_choice, top_k, top_p — then survive into the request
    # body and the first call dies with
    #   TypeError: Object of type Omit is not JSON serializable
    # Import the SDK after kit and they are stripped correctly.
    #
    # This kills the trial on turn 1, so it cannot corrupt results — but it
    # would have killed all 12 of them, one cold start at a time.
    provider = build_provider(args.model)
    session.reset(
        seed=args.seed,
        spawn=SpawnPose(spawn_xy[0], spawn_xy[1], spawn_heading),
    )

    camera = HeadCamera(session.env)
    camera.warmup()

    recorder = None
    detach = None
    if not args.no_video:
        # hide_ceiling=True is not optional in the apartment: the chase camera
        # sits above the 0.7 m walls, which puts it above the ceiling the T2.3
        # gate added — with the roof on, every audit frame is a photo of the
        # roof (T2.4, recorder.py's own docstring).
        recorder = Recorder(
            video_dir / trial_id, fps=25, every_n=args.video_every_n, hide_ceiling=True
        )
        detach = attach_recorder(session.playback, session.env.unwrapped, recorder)

    counters = Counters()
    if args.max_turns is not None:
        counters.turn_cap = args.max_turns
    context = ToolContext(
        playback=session.playback,
        camera=camera,
        memory=Memory(),
        # The seed's spawn coordinates are the integrator's t=0 anchor — the ONE
        # thing the dead-reckoned estimate takes from ground truth (doc 05 §5.1).
        integrator=PositionIntegrator(*spawn_xy),
        counters=counters,
    )

    log = TrialLog(
        json_path,
        trial_id=trial_id,
        model_id=cfg.model_id,
        model_name=args.model,
        seed=args.seed,
        spawn_xy=spawn_xy,
        spawn_heading_deg=spawn_heading,
    )
    if args.max_turns is not None:
        log.document["config"]["turn_cap_override"] = args.max_turns
        log.flush()

    def announce(record: dict) -> None:
        names = ", ".join(c["name"] for c in record["model_output"]["tool_calls"]) or "(none)"
        print(
            f"  [{record['stage']} t{record['turn_idx']:02d}] {names}"
            f"  |  {record['execution']['result']}"
            f"  |  budget {record['budget']['stage_turns_used']}"
            f"/{record['budget']['stage_turn_cap']} turns,"
            f" {record['budget']['stage_policy_seconds_used']:.1f}"
            f"/{record['budget']['stage_policy_seconds_cap']:g} s"
        )

    runner = EpisodeRunner(
        provider=provider,
        context=context,
        stages=stage_specs(args.seed),
        log=log,
        on_turn=announce,
    )

    import traceback

    from duck_embody.agent.loop import redact_secrets

    exit_code = 0
    final = None
    video_rel = None
    try:
        try:
            final = runner.run()
        except BaseException as exc:  # noqa: BLE001 — deliberate, see below
            # doc 05 §8's INFRA path, taken at the only place it is allowed to
            # be taken: the trial boundary. §8 forbids catching a render error
            # or a physics NaN *inside the loop*, where it would be laundered
            # into a model-visible result; its actual policy for one is "log,
            # rerun whole", which is what this does. The JSON deliberately keeps
            # NO `final` block, so doc 06 §9.1's resume check rejects it and
            # T4.2 reruns the trial.
            #
            # `BaseException`, not `Exception`: a Ctrl-C that skipped
            # `session.close()` would leave a kit process holding the machine's
            # single GPU (AGENTS.md rule 1) with no way to start the next run.
            #
            # The traceback is scrubbed (`redact_secrets`) before it is stored
            # OR printed: it is third-party exception text and this JSON is
            # committed to a public repo (AGENTS.md rules 6 and 7).
            detail = redact_secrets(
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            )
            log.note_infra_failure(detail)
            print("\n  INFRA FAILURE — trial is incomplete and must be rerun whole:")
            print(detail)
            exit_code = 1

        # --- the SCORING artifact first ------------------------------------
        #
        # `log.finish(final)` runs before any video work and inside no other
        # guard. A completed episode is 40-80 turns of paid API plus the QA
        # exchange; if an ffmpeg fault could reach this line the JSON would hold
        # every turn and no `final`, which is byte-for-byte what an infra-failed
        # trial looks like — doc 06 §9.1's resume check would move a finished,
        # paid result to `results/incomplete/` and rerun it.
        if final is not None:
            log.finish(final)

        # --- then the rule-11 audit artifacts, guarded separately -----------
        #
        # Video is EVIDENCE, not the result. `recorder.encode()` shells out to
        # ffmpeg with `check=True` and `_ffmpeg()` raises when the binary is
        # missing, so a full disk or an absent ffmpeg raises here — and that
        # must not invalidate a completed trial or strand the GPU.
        try:
            if detach is not None:
                detach()
            if recorder is not None:
                mp4 = recorder.encode()
                if mp4 is not None:
                    recorder.filmstrip(mp4)
                    try:
                        video_rel = str(mp4.relative_to(REPO_ROOT))
                    except ValueError:
                        # --video-dir outside the repo: absolute is still a
                        # usable pointer, and `relative_to` raising here would
                        # cost the trial for a path preference.
                        video_rel = str(mp4)
        except Exception:  # noqa: BLE001 — evidence failure, not a trial failure
            print("\n  WARNING: video artifacts failed; the trial result stands:")
            traceback.print_exc()
        log.set_video(video_rel)

        if final is not None:
            print("\n== outcome ==")
            for stage, outcome in final["outcome"].items():
                detail = final["stages"][stage]
                distance = detail["score"]["distance_m"] if detail["score"] else None
                print(
                    f"  {stage:<13} {outcome:<20} "
                    f"turns {detail['turns_used']}, "
                    f"policy-s {detail['policy_seconds_used']:.1f}"
                    + (f", d_final {distance:.3f} m" if distance is not None else "")
                )
            print(f"  bumps (trial)  {final['bumps']}")
            answered = sum(1 for q in final["qa"] if q["answer"])
            print(f"  layout QA      {answered}/{len(final['qa'])} answers parsed")
            if final.get("qa_parse_failed"):
                # Loud, because T4.1 scores an unparsed answer 0 and a pure
                # formatting mismatch is then indistinguishable from a bad map.
                # `final.qa_raw` holds the reply, so a re-split is always
                # possible — but only if somebody notices before the batch ends.
                print(
                    "  WARNING: the QA reply did not split into 5 answers "
                    "(final.qa_parse_failed=true) — re-split from final.qa_raw "
                    "before scoring"
                )
            tokens = final["tokens"]
            print(
                f"  tokens         in {tokens['input_tokens']} / out "
                f"{tokens['output_tokens']} / cached {tokens['cache_read_tokens']}"
                f"  ~${tokens['cost_usd_estimate']:.4f}"
            )

        print(f"\n  wrote {json_path}")
        if video_rel:
            print(f"  wrote {video_rel}")
    finally:
        # `finally`, so the kit process is released on EVERY path — including a
        # Ctrl-C during the artifact block. A surviving kit process holds the
        # machine's single GPU and the rerun cannot start at all (rule 1).
        print("  closing app (nothing after this line runs)")
        session.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
