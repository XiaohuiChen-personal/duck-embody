"""Wire-shape tests for both provider adapters. No live API calls.

The thing being protected here is the fairness contract: the *logical* payload
is frozen — one image and one JSON status object per observation, identical
field names and values for all three models — while the wire encoding differs
per provider (AGENTS.md rule 4, doc 04 §6.1). These tests pin the translation in
both directions and assert the two adapters carry the same information.

Adapters are exercised via their pure shaping methods, which are deliberately
static/instance methods that touch no network and no client.
"""

from __future__ import annotations

import json

import pytest

from duck_embody.agent.providers.base import (
    AssistantMessage,
    ImageBlock,
    ModelConfig,
    TextBlock,
    ToolResultBlock,
    Usage,
    UserMessage,
)

# A 1x1 JPEG is not needed — the adapters never decode, they only transport.
FAKE_JPEG_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgH"

CANONICAL_TOOLS = [
    {
        "name": "get_observation",
        "description": "One egocentric camera frame plus compass and status.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "move",
        "description": "Walk forward, closed-loop on dead-reckoned distance.",
        "input_schema": {
            "type": "object",
            "properties": {"distance_m": {"type": "number"}},
            "required": ["distance_m"],
        },
    },
]

STATUS_JSON = json.dumps(
    {
        "compass_deg": 87.4,
        "position_estimate": {"x": 1.42, "y": -0.31},
        "status": {"bumped": False, "fell": False, "distance_moved_m": 0.48},
    },
    sort_keys=True,
)


def _cfg(provider: str, model_id: str) -> ModelConfig:
    return ModelConfig(
        name="test",
        provider=provider,
        model_id=model_id,
        max_tokens=16000,
        price_in_per_mtok=10.0,
        price_out_per_mtok=50.0,
    )


@pytest.fixture
def anthropic_adapter():
    """Construct the adapter without touching the network or the API key."""
    from duck_embody.agent.providers.anthropic import AnthropicProvider

    adapter = AnthropicProvider.__new__(AnthropicProvider)
    adapter.cfg = _cfg("anthropic", "claude-fable-5")
    adapter.name, adapter.model_id = "fable5", "claude-fable-5"
    return adapter


@pytest.fixture
def openai_adapter():
    from duck_embody.agent.providers.openai import OpenAIProvider

    adapter = OpenAIProvider.__new__(OpenAIProvider)
    adapter.cfg = _cfg("openai", "gpt-5.6-sol")
    adapter.name, adapter.model_id = "gpt56sol", "gpt-5.6-sol"
    return adapter


@pytest.fixture
def observation_message():
    """One turn's worth of results: a tool_result carrying a frame + status."""
    return UserMessage(
        blocks=[
            ToolResultBlock(
                tool_use_id="call_abc",
                tool_name="get_observation",
                text=STATUS_JSON,
                images=[ImageBlock(data_b64=FAKE_JPEG_B64)],
            )
        ]
    )


class TestToolSchemaTranslation:
    def test_anthropic_tools_are_near_identity(self, anthropic_adapter):
        out = anthropic_adapter.to_native_tools(CANONICAL_TOOLS)
        assert out[0]["name"] == "get_observation"
        assert out[0]["input_schema"] == CANONICAL_TOOLS[0]["input_schema"]
        assert "function" not in out[0]

    def test_openai_tools_are_flat_not_nested(self, openai_adapter):
        """Responses API tools are FLAT — chat completions nested them under
        `function`, and that difference is a silent 400 if confused."""
        out = openai_adapter.to_native_tools(CANONICAL_TOOLS)
        assert out[0]["type"] == "function"
        assert out[0]["name"] == "get_observation"
        assert "function" not in out[0]
        # input_schema -> parameters, contents unchanged
        assert out[0]["parameters"] == CANONICAL_TOOLS[0]["input_schema"]

    def test_both_expose_the_same_tool_names_and_schemas(
        self, anthropic_adapter, openai_adapter
    ):
        """The tool surface is frozen and identical across models (rule 4)."""
        a = anthropic_adapter.to_native_tools(CANONICAL_TOOLS)
        o = openai_adapter.to_native_tools(CANONICAL_TOOLS)
        assert [t["name"] for t in a] == [t["name"] for t in o]
        assert [t["input_schema"] for t in a] == [t["parameters"] for t in o]


class TestAnthropicMessageShaping:
    def test_tool_result_is_a_block_in_one_user_message(
        self, anthropic_adapter, observation_message
    ):
        out = anthropic_adapter.to_native([observation_message])
        assert len(out) == 1
        assert out[0]["role"] == "user"
        assert out[0]["content"][0]["type"] == "tool_result"
        assert out[0]["content"][0]["tool_use_id"] == "call_abc"

    def test_image_rides_inside_the_tool_result(self, anthropic_adapter, observation_message):
        """The capability OpenAI lacks — worth pinning explicitly."""
        content = anthropic_adapter.to_native([observation_message])[0]["content"][0]["content"]
        kinds = [c["type"] for c in content]
        assert kinds == ["text", "image"]
        assert content[1]["source"]["data"] == FAKE_JPEG_B64
        assert content[1]["source"]["media_type"] == "image/jpeg"

    def test_all_results_land_in_a_single_user_message(self, anthropic_adapter):
        """Splitting them degrades parallel tool use (doc 05 §3.3)."""
        msg = UserMessage(
            blocks=[
                ToolResultBlock("a", "update_room", "ok"),
                ToolResultBlock("b", "move", STATUS_JSON),
                ToolResultBlock("c", "get_observation", STATUS_JSON),
            ]
        )
        out = anthropic_adapter.to_native([msg])
        assert len(out) == 1
        assert len(out[0]["content"]) == 3

    def test_assistant_turn_is_echoed_verbatim(self, anthropic_adapter):
        """Thinking blocks must be replayed unchanged for multi-turn continuity."""
        native = [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "hi"}]
        out = anthropic_adapter.to_native([AssistantMessage(native=native)])
        assert out == [{"role": "assistant", "content": native}]

    def test_error_results_are_flagged(self, anthropic_adapter):
        msg = UserMessage(blocks=[ToolResultBlock("x", "move", "bad args", is_error=True)])
        assert anthropic_adapter.to_native([msg])[0]["content"][0]["is_error"] is True

    def test_look_around_labels_precede_each_frame(self, anthropic_adapter):
        msg = UserMessage(
            blocks=[
                ToolResultBlock(
                    "la",
                    "look_around",
                    STATUS_JSON,
                    images=[
                        ImageBlock(FAKE_JPEG_B64, label="view at compass 0 deg"),
                        ImageBlock(FAKE_JPEG_B64, label="view at compass 90 deg"),
                    ],
                )
            ]
        )
        content = anthropic_adapter.to_native([msg])[0]["content"][0]["content"]
        assert [c["type"] for c in content] == ["text", "text", "image", "text", "image"]
        assert content[1]["text"] == "view at compass 0 deg"


class TestOpenAIMessageShaping:
    """Responses API item shapes (verified live 2026-07-26)."""

    def test_tool_result_becomes_a_function_call_output_item(
        self, openai_adapter, observation_message
    ):
        out = openai_adapter.to_native([observation_message])
        outputs = [m for m in out if m.get("type") == "function_call_output"]
        assert len(outputs) == 1
        assert outputs[0]["call_id"] == "call_abc"
        # `output` is string-only — the constraint that forces the carrier design.
        assert isinstance(outputs[0]["output"], str)

    def test_image_is_carried_in_an_adjacent_user_message(
        self, openai_adapter, observation_message
    ):
        out = openai_adapter.to_native([observation_message])
        user_msgs = [m for m in out if m.get("role") == "user"]
        assert len(user_msgs) == 1
        parts = user_msgs[0]["content"]
        images = [p for p in parts if p["type"] == "input_image"]
        assert len(images) == 1
        assert images[0]["image_url"] == f"data:image/jpeg;base64,{FAKE_JPEG_B64}"

    def test_content_parts_use_input_prefixed_types(
        self, openai_adapter, observation_message
    ):
        """Responses API uses input_text/input_image, not text/image_url."""
        parts = [m for m in openai_adapter.to_native([observation_message])
                 if m.get("role") == "user"][0]["content"]
        assert {p["type"] for p in parts} <= {"input_text", "input_image"}

    def test_carrier_message_names_the_originating_call(
        self, openai_adapter, observation_message
    ):
        """Without this the model cannot tell which call a frame came from."""
        carrier = [m for m in openai_adapter.to_native([observation_message])
                   if m.get("role") == "user"][0]
        assert "call_abc" in carrier["content"][0]["text"]
        assert "get_observation" in carrier["content"][0]["text"]

    def test_one_output_item_per_call(self, openai_adapter):
        msg = UserMessage(
            blocks=[
                ToolResultBlock("a", "update_room", "ok"),
                ToolResultBlock("b", "move", STATUS_JSON),
            ]
        )
        out = openai_adapter.to_native([msg])
        assert len([m for m in out if m.get("type") == "function_call_output"]) == 2

    def test_assistant_output_items_are_echoed_verbatim(self, openai_adapter):
        """Reasoning items must survive replay, like Anthropic thinking blocks."""
        native = [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "function_call", "call_id": "c1", "name": "move", "arguments": "{}"},
        ]
        out = openai_adapter.to_native([AssistantMessage(native=native)])
        assert out == native

    def test_look_around_labels_survive_the_carrier(self, openai_adapter):
        msg = UserMessage(
            blocks=[
                ToolResultBlock(
                    "la",
                    "look_around",
                    STATUS_JSON,
                    images=[
                        ImageBlock(FAKE_JPEG_B64, label="view at compass 0 deg"),
                        ImageBlock(FAKE_JPEG_B64, label="view at compass 90 deg"),
                    ],
                )
            ]
        )
        parts = [m for m in openai_adapter.to_native([msg]) if m.get("role") == "user"][0][
            "content"
        ]
        texts = [p["text"] for p in parts if p["type"] == "input_text"]
        assert "view at compass 0 deg" in texts
        assert "view at compass 90 deg" in texts


class TestSemanticEquivalence:
    """The freeze condition: same image, same JSON, both providers."""

    def test_both_transmit_the_identical_image_payload(
        self, anthropic_adapter, openai_adapter, observation_message
    ):
        a_content = anthropic_adapter.to_native([observation_message])[0]["content"][0][
            "content"
        ]
        a_b64 = [c["source"]["data"] for c in a_content if c["type"] == "image"]

        o_out = openai_adapter.to_native([observation_message])
        o_parts = [m for m in o_out if m.get("role") == "user"][0]["content"]
        o_b64 = [
            p["image_url"].split("base64,", 1)[1]
            for p in o_parts
            if p["type"] == "input_image"
        ]

        assert a_b64 == o_b64 == [FAKE_JPEG_B64]

    def test_both_transmit_the_identical_status_json(
        self, anthropic_adapter, openai_adapter, observation_message
    ):
        a_content = anthropic_adapter.to_native([observation_message])[0]["content"][0][
            "content"
        ]
        a_text = next(c["text"] for c in a_content if c["type"] == "text")

        o_out = openai_adapter.to_native([observation_message])
        o_text = next(
            m["output"] for m in o_out if m.get("type") == "function_call_output"
        )

        assert a_text == o_text == STATUS_JSON
        # And it must survive as parseable JSON with the frozen field names.
        assert json.loads(a_text)["compass_deg"] == 87.4
        assert set(json.loads(o_text)) == {"compass_deg", "position_estimate", "status"}

    def test_image_count_matches_across_providers_for_look_around(
        self, anthropic_adapter, openai_adapter
    ):
        msg = UserMessage(
            blocks=[
                ToolResultBlock(
                    "la",
                    "look_around",
                    STATUS_JSON,
                    images=[ImageBlock(FAKE_JPEG_B64, label=f"view at compass {b} deg")
                            for b in (0, 90, 180, 270)],
                )
            ]
        )
        a_content = anthropic_adapter.to_native([msg])[0]["content"][0]["content"]
        n_a = sum(1 for c in a_content if c["type"] == "image")

        o_parts = [m for m in openai_adapter.to_native([msg]) if m.get("role") == "user"][0][
            "content"
        ]
        n_o = sum(1 for p in o_parts if p["type"] == "input_image")

        assert n_a == n_o == 4


class TestCostAccounting:
    def test_cost_uses_the_configured_price_sheet(self):
        cfg = _cfg("anthropic", "claude-fable-5")  # $10 in / $50 out
        usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
        assert cfg.cost(usage) == pytest.approx(10.0 + 5.0)

    def test_usage_accumulates_across_turns(self):
        total = Usage(input_tokens=10, output_tokens=1, cost_usd=0.5) + Usage(
            input_tokens=20, output_tokens=2, cost_usd=0.25
        )
        assert (total.input_tokens, total.output_tokens) == (30, 3)
        assert total.cost_usd == pytest.approx(0.75)

    def test_usage_serialises_for_the_trial_json(self):
        d = Usage(input_tokens=5, output_tokens=6, cost_usd=0.123456789).as_dict()
        assert d["input_tokens"] == 5
        assert d["cost_usd_estimate"] == 0.123457


class TestModelConfigs:
    """The committed configs are part of the frozen fairness contract."""

    @pytest.mark.parametrize(
        "name,provider,model_id",
        [
            ("fable5", "anthropic", "claude-fable-5"),
            ("opus5", "anthropic", "claude-opus-5"),
            ("gpt56sol", "openai", "gpt-5.6-sol"),
            ("judge", "anthropic", "claude-sonnet-5"),
        ],
    )
    def test_configs_load_with_the_locked_ids(self, name, provider, model_id):
        from duck_embody.agent.providers.base import load_model_config

        cfg = load_model_config(name)
        assert cfg.provider == provider
        assert cfg.model_id == model_id

    def test_no_anthropic_contestant_sets_sampling_params(self):
        """temperature/top_p/top_k return HTTP 400 on the locked models."""
        from duck_embody.agent.providers.base import load_model_config

        for name in ("fable5", "opus5"):
            params = load_model_config(name).params
            assert not ({"temperature", "top_p", "top_k"} & set(params))

    def test_the_judge_is_not_a_contestant(self):
        """doc 04 §8: tuning the scene to a contestant is an integrity defect."""
        from duck_embody.agent.providers.base import load_model_config

        contestants = {load_model_config(n).model_id for n in ("fable5", "opus5", "gpt56sol")}
        assert load_model_config("judge").model_id not in contestants
