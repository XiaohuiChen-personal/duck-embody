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

from duck_embody.agent.prompts import DERAILMENT_NUDGE
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


def _refusal_response(content):
    """The shape a pre-output classifier decline actually returns: HTTP 200,
    ``stop_reason: "refusal"``, and an EMPTY ``content`` array."""
    from types import SimpleNamespace

    return SimpleNamespace(
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber"),
        content=content,
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


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

    @pytest.mark.parametrize("native", [[], None])
    def test_an_empty_assistant_turn_is_dropped_not_echoed(
        self, anthropic_adapter, native
    ):
        """A refusal is HTTP 200 with an EMPTY `content` array (doc 05 §7.2), so
        `AssistantTurn.raw` is `[]` and the loop routes it to §8's DERAILMENT
        path — scored, never retried.

        This asserts BEHAVIOUR, not a wire requirement. Echoing the empty turn
        back is NOT rejected: measured 2026-07-26 against the live API with the
        12 tool schemas and adaptive thinking, `{"role": "assistant",
        "content": []}` is ACCEPTED. An earlier version of this docstring
        claimed it was a 400; that was wrong, and the real 400 came from the
        empty USER message the missing derailment branch appended (covered by
        `test_the_derailment_nudge_is_never_empty`).

        Dropping is still correct: `content: None` IS rejected (400 "Input
        should be a valid list"), and an empty turn is a no-op that costs
        tokens and tells the model nothing."""
        out = anthropic_adapter.to_native(
            [
                AssistantMessage(native=native),
                UserMessage(blocks=[TextBlock("No tool call received — …")]),
            ]
        )
        assert all(m["content"] for m in out), f"empty message content in {out}"
        assert [m["role"] for m in out] == ["user"]

    def test_the_derailment_nudge_is_never_empty(self, anthropic_adapter):
        """THE actual wire requirement a refusal has to satisfy.

        Measured 2026-07-26 against the live API (claude-opus-5): an empty USER
        message is rejected — both ``content: []`` and ``content: ""`` return
        400 "user messages must ...". An empty ASSISTANT turn is accepted. So
        the thing that genuinely broke a refusal was doc 05 §3.1's pseudocode
        appending an empty user message when it had no derailment branch, and
        the thing that keeps it fixed is that the nudge carries text.

        A 400 here would be caught at the trial boundary and logged as an infra
        failure, so doc 06 §9.1's resume check would rerun a trial the model
        actually failed — selection bias in that model's favour, which doc 05
        §8 forbids in as many words.
        """
        assert DERAILMENT_NUDGE.strip(), "the nudge must carry text or it is a 400"

        out = anthropic_adapter.to_native(
            [
                AssistantMessage(native=[]),
                UserMessage(blocks=[TextBlock(DERAILMENT_NUDGE)]),
            ]
        )
        assert out, "the nudge turn must survive"
        for message in out:
            assert message["content"], f"empty content on the wire: {message}"
            if message["role"] == "user":
                # An all-whitespace text block is the same 400 as an empty one.
                assert any(
                    part.get("text", "").strip()
                    for part in message["content"]
                    if part.get("type") == "text"
                ), f"user message carries no text: {message}"

    def test_a_non_empty_assistant_turn_is_still_echoed(self, anthropic_adapter):
        """The guard must not swallow a partial refusal that DID return blocks
        (e.g. thinking only) — those still have to be replayed unchanged."""
        native = [{"type": "thinking", "thinking": "..."}]
        out = anthropic_adapter.to_native([AssistantMessage(native=native)])
        assert out == [{"role": "assistant", "content": native}]

    def test_a_refusal_parses_to_an_empty_list_never_none(self, anthropic_adapter):
        """`_parse` normalises `raw`, so a `None` content cannot become
        `"content": null` on the wire either."""
        for content in ([], None):
            turn = anthropic_adapter._parse(_refusal_response(content))
            assert turn.raw == []
            assert turn.tool_calls == []
            assert turn.refusal == "refusal (category=cyber)"
            assert turn.stop_reason == "refusal"

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

    def test_the_error_channel_asymmetry_is_confined_to_the_protocol_flag(
        self, anthropic_adapter, openai_adapter
    ):
        """The ONE place AGENTS.md rule 4's "one tool set" is not byte-identical
        across contestants — recorded here and in doc 05 §7.3 so it cannot be
        mistaken for a harness choice.

        `tools.dispatch` sets `is_error=True` on every `invalid_args` /
        `unknown_tool` / rejected-memory-write result. Anthropic's Messages API
        has a protocol-level `tool_result.is_error`; the Responses API's
        `function_call_output` has no equivalent field, so GPT 5.6 sol receives
        the same information through the JSON body alone. The asymmetry is
        API-imposed, not a design decision, and the mitigation is that the body —
        which BOTH models read — is byte-identical. That is what this pins: the
        text channel cannot be allowed to drift as well.
        """
        body = '{"error": "invalid_args", "detail": "d", "hint": "h"}'
        msg = UserMessage(blocks=[ToolResultBlock("x", "mark_exit", body, is_error=True)])

        a_result = anthropic_adapter.to_native([msg])[0]["content"][0]
        o_result = next(
            m
            for m in openai_adapter.to_native([msg])
            if m.get("type") == "function_call_output"
        )
        # The shared channel: identical bytes.
        a_text = next(c["text"] for c in a_result["content"] if c["type"] == "text")
        assert a_text == o_result["output"] == body
        # The unshared one: a flag on one side, nothing to carry it on the other.
        assert a_result["is_error"] is True
        assert "is_error" not in o_result

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


class TestRequestBodies:
    """The exact kwargs each adapter sends (T3.4).

    Split out of ``send`` so the QA exchange's shape is checkable without a
    network call — which matters because that call happens once per trial, at
    the very end, after every dollar of the trial has been spent. A 400 there
    would cost a headline metric (doc 06 §5.9's QA score) on a trial that had
    already finished.
    """

    @pytest.fixture
    def message(self):
        return UserMessage(blocks=[TextBlock("hello")])

    def test_anthropic_driving_request_carries_system_and_tools(
        self, anthropic_adapter, message
    ):
        kwargs = anthropic_adapter.request_kwargs("SYS", [message], CANONICAL_TOOLS)
        # The system prompt is a BLOCK LIST, not a bare string, because it
        # carries the prompt-cache breakpoint. The text must survive verbatim —
        # caching is a billing change and must never alter what the model reads.
        assert kwargs["system"] == [
            {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}
        ]
        assert [t["name"] for t in kwargs["tools"]] == ["get_observation", "move"]
        assert kwargs["model"] == "claude-fable-5"
        assert kwargs["max_tokens"] == 16000
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}

    def test_the_cache_breakpoint_is_present_and_covers_system_plus_tools(
        self, anthropic_adapter, message
    ):
        """The system+tools prefix is 3,919 MEASURED tokens, byte-identical on
        every call of a trial — 22% of all input tokens.

        Marking the system block caches everything earlier in Anthropic's prompt
        hierarchy (tools, then system), so one breakpoint covers both. Without
        the marker the whole prefix is re-billed at full rate every turn; the
        adapter already reads `cache_read_tokens` back, so a silent regression
        here would show up only as a larger invoice.
        """
        kwargs = anthropic_adapter.request_kwargs("SYS", [message], CANONICAL_TOOLS)
        blocks = kwargs["system"]
        assert isinstance(blocks, list), "system must be blocks to carry cache_control"
        marked = [b for b in blocks if b.get("cache_control")]
        assert len(marked) == 1, f"expected exactly one breakpoint, got {blocks}"
        assert marked[0]["cache_control"] == {"type": "ephemeral"}
        # Tools must still be sent, since the breakpoint's value is that it
        # covers them too.
        assert kwargs["tools"], "tools must accompany the cached system prefix"

    def test_caching_does_not_change_what_the_model_reads(
        self, anthropic_adapter, message
    ):
        """The fairness half of the caching change (AGENTS.md rule 4).

        Whatever the billing, the assembled prompt text must be identical to the
        uncached form — otherwise 'we enabled caching' would quietly be 'we
        changed the two Anthropic contestants' prompt'.
        """
        kwargs = anthropic_adapter.request_kwargs("SYS", [message], CANONICAL_TOOLS)
        text = "".join(b["text"] for b in kwargs["system"])
        assert text == "SYS"

    def test_openai_driving_request_carries_instructions_and_tools(
        self, openai_adapter, message
    ):
        kwargs = openai_adapter.request_kwargs("SYS", [message], CANONICAL_TOOLS)
        assert kwargs["instructions"] == "SYS"
        assert [t["name"] for t in kwargs["tools"]] == ["get_observation", "move"]
        assert kwargs["max_output_tokens"] == 16000

    @pytest.mark.parametrize("adapter_name", ["anthropic_adapter", "openai_adapter"])
    def test_the_qa_exchange_omits_empty_system_and_tools(
        self, adapter_name, message, request
    ):
        """doc 06 §5.9's exchange is toolless and system-less: "you have no
        camera, no robot, and no tools now". Omitting the keys states that,
        rather than asking the API to interpret ``""`` and ``[]``."""
        adapter = request.getfixturevalue(adapter_name)
        kwargs = adapter.request_kwargs("", [message], [])
        assert "tools" not in kwargs
        assert "system" not in kwargs and "instructions" not in kwargs
        # The message itself still goes, under each API's own key.
        assert kwargs.get("messages") or kwargs.get("input")

    def test_config_params_still_win(self, anthropic_adapter, message):
        """``cfg.params`` is applied last so a recorded per-provider setting can
        never be silently overridden by a default above it."""
        anthropic_adapter.cfg.params = {"max_tokens": 99}
        assert anthropic_adapter.request_kwargs("SYS", [message], [])["max_tokens"] == 99


class TestBlankTextIsRejectedProviderNeutrally:
    """The cross-provider asymmetry that makes a harness bug a FAIRNESS bug.

    MEASURED 2026-07-26 against both live APIs:

        shape                 Anthropic                      OpenAI
        user text ""          400 "must be non-empty"        ACCEPTED
        user text "   "       400 "must contain non-ws"      ACCEPTED
        user content []       400 "must have non-empty"      ACCEPTED

    So a blank block the harness emitted by mistake would cost the two
    Anthropic contestants a trial (400 -> caught at the trial boundary ->
    logged infra_failure -> doc 06 §9.1 reruns it) and cost GPT 5.6 sol
    nothing. Same defect, different price per model — exactly what AGENTS.md
    rule 4's "one prompt template, one tool set, frozen" is protecting.

    The guard therefore lives in the provider-NEUTRAL layer, so both providers
    fail identically and the failure surfaces here rather than mid-batch.
    """

    @pytest.mark.parametrize("bad", ["", " ", "\n", "\t  \n"])
    def test_a_blank_text_block_cannot_be_constructed(self, bad):
        with pytest.raises(ValueError, match="non-empty"):
            TextBlock(bad)

    def test_real_text_is_unaffected(self):
        assert TextBlock(" the map so far ").text == " the map so far "

    def test_a_blank_image_label_is_dropped_by_both_adapters(
        self, anthropic_adapter, openai_adapter
    ):
        """A label is optional decoration, so a blank one is dropped rather than
        raised on — but it must be dropped by BOTH adapters, or the Anthropic
        request 400s on a payload OpenAI happily accepts."""
        msg = UserMessage(blocks=[ImageBlock(data_b64=FAKE_JPEG_B64, label="   ")])

        a_content = anthropic_adapter.to_native([msg])[0]["content"]
        assert all(part["type"] != "text" for part in a_content), a_content

        o_parts = openai_adapter.to_native([msg])[0]["content"]
        blank = [
            p
            for p in o_parts
            if p.get("type") == "input_text" and not str(p.get("text", "")).strip()
        ]
        assert not blank, f"OpenAI adapter emitted a blank text part: {o_parts}"


class TestCostIncludesCacheTerms:
    """Enabling prompt caching without fixing cost accounting silently
    understates every reported cost.

    MEASURED on a live call before the fix: with caching on, `input_tokens`
    counts only the UNCACHED remainder (84) while the 3,856-token prefix lands
    in `cache_write_tokens` / `cache_read_tokens`. Summing only input+output
    reported the cache-write call and the cache-read call at exactly the same
    $0.0010 — the tell that both cache columns were being ignored.

    Cost is a published per-model column, so this is a wrong number in the
    write-up, not a crash.
    """

    def _cfg(self):
        return _cfg("anthropic", "claude-fable-5")  # $10 in / $50 out per MTok

    def test_a_cache_write_costs_more_than_the_same_tokens_uncached(self):
        cfg = self._cfg()
        write = cfg.cost(Usage(input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000))
        plain = cfg.cost(Usage(input_tokens=1_000_000, output_tokens=0))
        assert write == pytest.approx(plain * 1.25)

    def test_a_cache_read_costs_a_tenth_of_the_same_tokens_uncached(self):
        cfg = self._cfg()
        read = cfg.cost(Usage(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000))
        plain = cfg.cost(Usage(input_tokens=1_000_000, output_tokens=0))
        assert read == pytest.approx(plain * 0.1)

    def test_the_write_call_and_the_read_call_do_not_cost_the_same(self):
        """The exact symptom that exposed the bug on the live API."""
        cfg = self._cfg()
        first = cfg.cost(Usage(input_tokens=84, output_tokens=4, cache_write_tokens=3856))
        second = cfg.cost(Usage(input_tokens=84, output_tokens=4, cache_read_tokens=3856))
        assert first > second, "a cache write must cost more than a cache read"
        assert second > cfg.cost(Usage(input_tokens=84, output_tokens=4)), (
            "a cache read is cheap but NOT free — ignoring it understates the bill"
        )

    def test_caching_actually_saves_money_over_the_uncached_equivalent(self):
        """Sanity on the direction: 50 calls sharing a 3,919-token prefix should
        cost less cached than paying full rate for it every time."""
        cfg = self._cfg()
        prefix, calls = 3919, 50
        uncached = sum(
            cfg.cost(Usage(input_tokens=prefix + 84, output_tokens=4)) for _ in range(calls)
        )
        cached = cfg.cost(Usage(input_tokens=84, output_tokens=4, cache_write_tokens=prefix)) + sum(
            cfg.cost(Usage(input_tokens=84, output_tokens=4, cache_read_tokens=prefix))
            for _ in range(calls - 1)
        )
        assert cached < uncached


class TestPreflightDoesNotNeedTheVendorSDK:
    """`preflight_provider` exists so run_trial.py can fail fast on a bad model
    name or a missing key WITHOUT importing the vendor SDK — which must not be
    imported before AppLauncher (measured T3.5: the SDK then leaks a dozen
    `omit` sentinels into the request body and every trial dies on turn 1).

    Two things have to hold together, and the first attempt got the second
    wrong: preflight must not pull in the SDK, AND it must still see the key.
    `.env` used to be loaded only as a side effect of importing an adapter
    module, so skipping that import made a perfectly good key look missing.
    """

    def test_preflight_loads_dotenv_itself(self, monkeypatch):
        import duck_embody.agent.providers.base as base

        called = {"n": 0}
        monkeypatch.setattr(base, "load_env", lambda: called.__setitem__("n", called["n"] + 1))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
        base.preflight_provider("fable5")
        assert called["n"] == 1, "preflight must load .env, not inherit it from an import"

    def test_preflight_does_not_import_the_vendor_sdk(self, monkeypatch):
        import sys

        import duck_embody.agent.providers.base as base

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
        for mod in ("anthropic", "duck_embody.agent.providers.anthropic"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
        base.preflight_provider("fable5")
        assert "duck_embody.agent.providers.anthropic" not in sys.modules, (
            "preflight imported the adapter module — the SDK import must wait for kit"
        )

    def test_preflight_still_rejects_a_missing_key(self, monkeypatch):
        import duck_embody.agent.providers.base as base

        monkeypatch.setattr(base, "load_env", lambda: None)  # simulate no .env
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            base.preflight_provider("fable5")

    def test_preflight_rejects_an_unknown_provider(self, monkeypatch, tmp_path):
        import duck_embody.agent.providers.base as base

        cfg = ModelConfig(
            name="bogus", provider="acme", model_id="x",
            max_tokens=1, price_in_per_mtok=0.0, price_out_per_mtok=0.0,
        )
        monkeypatch.setattr(base, "load_model_config", lambda _n: cfg)
        with pytest.raises(ValueError, match="Unknown provider"):
            base.preflight_provider("bogus")
