#!/usr/bin/env python3
"""Cheap two-turn wire probe for one model from each benchmark provider."""

from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FRAME = REPO / "results/figures/smoke/camera_standing.jpg"
OUT = REPO / "results/probes/provider_roundtrip_20260802.json"
ALIASES = ("sonnet5", "gpt56sol")
SYSTEM = (
    "You are a provider wire probe. On every request, call report_scene exactly "
    "once. Keep the label under five words."
)
TOOLS = [
    {
        "name": "report_scene",
        "description": "Acknowledge the image or prior tool result.",
        "input_schema": {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
    }
]


def _complete_usage(usage: dict) -> bool:
    required = {
        "input_tokens_total",
        "input_tokens_uncached",
        "cache_read_tokens",
        "cache_write_tokens",
        "output_tokens_total",
        "cost_usd_estimate",
        "pricing_version",
        "pricing_source",
    }
    return (
        required <= set(usage)
        and usage["input_tokens_total"]
        == usage["input_tokens_uncached"]
        + usage["cache_read_tokens"]
        + usage["cache_write_tokens"]
        and bool(usage["pricing_version"])
        and bool(usage["pricing_source"])
    )


def main() -> int:
    from duck_embody.agent.loop import (
        build_neutral_request_manifest,
        reconstruct_neutral_request,
    )
    from duck_embody.agent.providers.base import (
        AssistantMessage,
        ImageBlock,
        TextBlock,
        ToolResultBlock,
        UserMessage,
        build_provider,
    )

    if not FRAME.exists():
        raise SystemExit(f"missing probe frame: {FRAME}")
    frame_bytes = FRAME.read_bytes()
    frame_b64 = base64.b64encode(frame_bytes).decode("ascii")
    frame_rel = str(FRAME.relative_to(REPO))
    report = {
        "schema": "duck-embody-provider-roundtrip-v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frame": frame_rel,
        "providers": {},
        "acceptance": "PASS",
    }

    def save_image(_raw: bytes, _label: str, _extension: str) -> str:
        return frame_rel

    for alias in ALIASES:
        provider = build_provider(alias)
        messages = [
            UserMessage(
                [
                    TextBlock("Inspect this real harness frame and acknowledge it."),
                    ImageBlock(frame_b64, label="standing camera frame"),
                ]
            )
        ]
        manifests = []
        turns = []
        first_manifest = build_neutral_request_manifest(
            trial_id=f"probe_{alias}",
            request_index=0,
            kind="probe",
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            memory_block="",
            context={"probe": True, "turn": 1},
            save_image=save_image,
        )
        manifests.append(first_manifest)
        first = provider.send(SYSTEM, messages, TOOLS)
        if not first.tool_calls:
            raise RuntimeError(f"{alias}: first response returned no tool call")
        first_call = first.tool_calls[0]
        messages.extend(
            [
                AssistantMessage(
                    native=first.raw,
                    context_index=0,
                    global_turn_index=1,
                    native_response_sha256=first.metadata.get(
                        "native_response_sha256"
                    ),
                ),
                UserMessage(
                    [
                        ToolResultBlock(
                            tool_use_id=first_call.id,
                            tool_name=first_call.name,
                            text=json.dumps({"ok": True, "continue": True}),
                        )
                    ]
                ),
            ]
        )
        second_manifest = build_neutral_request_manifest(
            trial_id=f"probe_{alias}",
            request_index=1,
            kind="probe",
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            memory_block="",
            context={"probe": True, "turn": 2},
            save_image=save_image,
        )
        manifests.append(second_manifest)
        second = provider.send(SYSTEM, messages, TOOLS)
        if second.refusal is not None:
            raise RuntimeError(f"{alias}: follow-up was refused: {second.refusal}")

        document = {"requests": manifests}
        reconstructed = [
            reconstruct_neutral_request(
                document,
                index,
                lambda _relative: frame_bytes,
            )["request_sha256"]
            for index in range(2)
        ]
        for turn in (first, second):
            usage = turn.usage.as_dict()
            if not _complete_usage(usage):
                raise RuntimeError(f"{alias}: incomplete normalized usage")
            turns.append(
                {
                    "resolved_model_id": turn.metadata.get("resolved_model_id"),
                    "native_response_sha256": turn.metadata.get(
                        "native_response_sha256"
                    ),
                    "tool_names": [call.name for call in turn.tool_calls],
                    "usage": usage,
                }
            )
        report["providers"][alias] = {
            "request_sha256": [item["request_sha256"] for item in manifests],
            "reconstructed_sha256": reconstructed,
            "hashes_match": reconstructed
            == [item["request_sha256"] for item in manifests],
            "turns": turns,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: two-turn probes complete -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
