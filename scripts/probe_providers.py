"""T3.3 live probe: prove both wire formats work BEFORE the batch is frozen.

Doc 04 §6.1 makes this a freeze condition, and the reason is arithmetic: the
OpenAI image path is structurally different from Anthropic's (tool-role content
is string-only, so frames travel in an adjacent user message). If that path
first executed inside the frozen batch, a wire-shape error would cost four
trials and a re-freeze.

Each contestant gets one real call carrying a **real 512x512 JPEG** — the
standing frame captured by T1.4, not a synthetic pixel — plus a dummy tool, and
must come back with a well-formed tool call whose arguments parse. The judge
model gets its own probe because T2.3's gate depends on it.

Also runs doc 05 §7.1's open question: does GPT 5.6 sol accept `temperature=0`?
The answer is written into `configs/models/gpt56sol.yaml` so the decision is
recorded in the frozen config rather than assumed in code.

Costs a handful of cents. Run:
    ~/IsaacLab/isaaclab.sh -p scripts/probe_providers.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAME = REPO_ROOT / "results" / "figures" / "smoke" / "camera_standing.jpg"
OUT = REPO_ROOT / "results" / "figures" / "smoke" / "provider_probe.json"

#: A dummy tool with a required argument, so "did it come back well-formed?"
#: is actually checkable rather than vacuous.
PROBE_TOOL = {
    "name": "report_scene",
    "description": (
        "Report what the camera image shows. Call this exactly once, immediately, "
        "using the image provided."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "primary_subject": {
                "type": "string",
                "description": "The main thing visible in the image, in one or two words.",
            },
            "horizon_visible": {
                "type": "boolean",
                "description": "True if a horizon line is visible.",
            },
        },
        "required": ["primary_subject", "horizon_visible"],
    },
}

SYSTEM = "You are a vision probe. Use the report_scene tool to describe the image."

STATUS_JSON = json.dumps(
    {
        "compass_deg": 87.4,
        "position_estimate": {"x": 1.42, "y": -0.31},
        "status": {"bumped": False, "fell": False, "distance_moved_m": 0.48},
    },
    sort_keys=True,
)


def main() -> int:
    from duck_embody.agent.providers.base import (
        ImageBlock,
        ToolResultBlock,
        UserMessage,
        build_provider,
        load_model_config,
    )

    if not FRAME.exists():
        print(f"FATAL: no probe frame at {FRAME}. Run scripts/smoke_camera.py first.")
        return 1

    jpeg_b64 = base64.b64encode(FRAME.read_bytes()).decode("ascii")
    print(f"probe frame: {FRAME.name} ({FRAME.stat().st_size / 1024:.1f} KiB)")

    failures: list[str] = []
    report: dict = {"frame": FRAME.name, "frame_bytes": FRAME.stat().st_size, "models": {}}

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            failures.append(label)

    # The image is delivered the way the harness will actually deliver it: as a
    # tool_result for a get_observation-shaped call, NOT as a bare user image.
    # That is the path doc 04 §6.1 is worried about.
    def observation_turn(call_id: str) -> list:
        return [
            UserMessage(
                blocks=[
                    ToolResultBlock(
                        tool_use_id=call_id,
                        tool_name="get_observation",
                        text=STATUS_JSON,
                        images=[ImageBlock(data_b64=jpeg_b64, label="view at compass 0 deg")],
                    )
                ]
            )
        ]

    for name in ("fable5", "opus5", "gpt56sol"):
        cfg = load_model_config(name)
        print(f"\n== {name} ({cfg.model_id}) ==")
        entry: dict = {"model_id": cfg.model_id}
        try:
            provider = build_provider(name)
        except Exception as exc:  # noqa: BLE001
            check(f"{name}: adapter constructs", False, str(exc)[:120])
            report["models"][name] = {**entry, "error": str(exc)}
            continue

        # A tool_result must answer a tool_use, so seed a minimal exchange the
        # provider will accept: assistant asks, harness answers with the frame.
        if cfg.provider == "anthropic":
            seed_assistant = [
                {
                    "type": "tool_use",
                    "id": "probe_call_1",
                    "name": "get_observation",
                    "input": {},
                }
            ]
        else:
            # Responses API: the assistant turn is a LIST of output items, and
            # a function_call_output must reference `call_id`.
            seed_assistant = [
                {
                    "type": "function_call",
                    "call_id": "probe_call_1",
                    "name": "get_observation",
                    "arguments": "{}",
                }
            ]

        from duck_embody.agent.providers.base import AssistantMessage, TextBlock

        messages = [
            UserMessage(blocks=[TextBlock(text="Look at the camera and report the scene.")]),
            AssistantMessage(native=seed_assistant),
            *observation_turn("probe_call_1"),
        ]
        tools = [
            {
                "name": "get_observation",
                "description": "Return one camera frame plus status.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            PROBE_TOOL,
        ]

        try:
            turn = provider.send(SYSTEM, messages, tools)
        except Exception as exc:  # noqa: BLE001
            check(f"{name}: live call succeeds", False, f"{type(exc).__name__}: {exc}"[:200])
            report["models"][name] = {**entry, "error": f"{type(exc).__name__}: {exc}"}
            continue

        check(f"{name}: live call succeeds (image path executed)", True)
        check(f"{name}: no refusal", turn.refusal is None, str(turn.refusal))

        calls = [c for c in turn.tool_calls if c.name == "report_scene"]
        check(f"{name}: returned a report_scene tool call", bool(calls),
              f"{[c.name for c in turn.tool_calls]}")
        if calls:
            call = calls[0]
            check(f"{name}: arguments parsed to a dict", call.parse_error is None,
                  str(call.parse_error))
            check(f"{name}: required args present",
                  {"primary_subject", "horizon_visible"} <= set(call.args),
                  str(sorted(call.args)))
            entry["tool_args"] = call.args
            print(f"    model saw: {call.args}")

        entry["stop_reason"] = turn.stop_reason
        entry["usage"] = turn.usage.as_dict()
        if turn.thinking:
            entry["thinking_chars"] = len(turn.thinking)
        report["models"][name] = entry
        print(f"    usage: {turn.usage.as_dict()}")

    # -- judge probe (T2.3 depends on it) ----------------------------------
    print("\n== judge (out-of-benchmark, T2.3 gate) ==")
    judge_cfg = load_model_config("judge")
    try:
        judge = build_provider("judge")
        answer = judge.ask_about_image(
            "What room of a home is this? Answer with one word.", jpeg_b64
        )
        check("judge: answers a one-word room question", bool(answer), repr(answer))
        report["judge"] = {"model_id": judge_cfg.model_id, "answer": answer}
        print(f"    judge answered: {answer!r} (empty plane — no room expected)")
    except Exception as exc:  # noqa: BLE001
        check("judge: live call succeeds", False, f"{type(exc).__name__}: {exc}"[:200])
        report["judge"] = {"model_id": judge_cfg.model_id, "error": str(exc)}

    # -- doc 05 §7.1 open question: temperature=0 on GPT 5.6 sol -----------
    print("\n== temperature probe (doc 05 §7.1 open question) ==")
    try:
        import openai

        from duck_embody.agent.providers.base import load_env

        load_env()
        client = openai.OpenAI(max_retries=0)
        client.chat.completions.create(
            model=judge_cfg and load_model_config("gpt56sol").model_id,
            messages=[{"role": "user", "content": "Say OK."}],
            max_completion_tokens=16,
            temperature=0,
        )
        accepted, detail = True, "accepted"
    except Exception as exc:  # noqa: BLE001
        accepted, detail = False, f"{type(exc).__name__}: {str(exc)[:160]}"

    report["gpt56sol_temperature_zero_accepted"] = accepted
    report["gpt56sol_temperature_detail"] = detail
    print(f"  temperature=0 accepted by gpt-5.6-sol: {accepted}")
    print(f"    {detail}")
    print(
        "  -> "
        + (
            "record temperature: 0 in gpt56sol.yaml params"
            if accepted
            else "NO locked model supports deterministic decoding; "
            "reproducibility rests on sim seeds alone (doc 05 §7.1)"
        )
    )

    report["failures"] = failures
    report["acceptance"] = "PASS" if not failures else "FAIL"
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")

    if failures:
        print("\n  FAILURES:")
        for f in failures:
            print(f"    {f}")
        return 1
    print("\n  OK - both wire formats verified with a real JPEG, pre-freeze")
    return 0


if __name__ == "__main__":
    sys.exit(main())
