"""OpenAI adapter for GPT 5.6 sol — **Responses API**, not chat completions.

Doc 05 §7.3 specified chat completions. That does not work for this model, and
the failure is not subtle. Measured 2026-07-26 (T3.3 probe):

    POST /v1/chat/completions  ->  400
    "Function tools with reasoning_effort are not supported for gpt-5.6-sol in
     /v1/chat/completions. To use function tools, use /v1/responses or set
     reasoning_effort to 'none'."

Of the two escapes the error offers, only one is admissible. Setting
``reasoning_effort='none'`` would run the OpenAI contestant **with reasoning
disabled** while both Anthropic contestants run with thinking on at the API
default — that is not a wire-format detail, it is a handicap that would make the
cross-lab comparison meaningless. So the harness uses ``/v1/responses``, where
function tools and reasoning coexist (verified: a probe call returned a
well-formed tool call alongside 19 reasoning tokens).

Shapes, all verified against the live API rather than assumed:

* system prompt -> the top-level ``instructions`` parameter;
* tools are **flat**: ``{type: "function", name, description, parameters}`` —
  note this differs from chat completions, where they nest under ``function``;
* user content parts are ``input_text`` / ``input_image`` (not ``text`` /
  ``image_url``), and images are data URLs;
* the model's turn comes back as a list of *output items* (``reasoning``,
  ``function_call``, ``message``) which are echoed back verbatim — the
  ``reasoning`` items matter for multi-turn continuity the same way Anthropic's
  thinking blocks do;
* a tool result is its own item: ``{type: "function_call_output", call_id,
  output}`` where ``output`` is a **string**.

That last point preserves the asymmetry doc 04 §6.1 flagged: a tool result
still cannot carry an image. Frames follow in an adjacent user message,
labelled and attributed to the call they came from, so the *logical* payload
(one image + one JSON status object, identical field names and values) is
identical to Anthropic's while the envelope differs.
"""

from __future__ import annotations

import json

from duck_embody.agent.providers.base import (
    AssistantMessage,
    AssistantTurn,
    ImageBlock,
    Message,
    ModelConfig,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    Usage,
    UserMessage,
    load_env,
    require_key,
)

load_env()

#: Introduces the frames belonging to the preceding tool result. Without this
#: attribution the model cannot tell which call a frame came from — the tool
#: result itself is text-only.
FRAME_CARRIER_PREFIX = "Camera frames for tool result {tool_use_id} ({tool_name}):"


class OpenAIProvider:
    def __init__(self, cfg: ModelConfig, max_retries: int = 5):
        import openai

        require_key("OPENAI_API_KEY", "OpenAI")
        self.cfg = cfg
        self.name = cfg.name
        self.model_id = cfg.model_id
        self.client = openai.OpenAI(max_retries=max_retries)

    # -- request shaping ----------------------------------------------------

    @staticmethod
    def _image_part(img: ImageBlock) -> dict:
        return {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{img.data_b64}",
        }

    @staticmethod
    def _text_part(text: str) -> dict:
        return {"type": "input_text", "text": text}

    def to_native(self, messages: list[Message]) -> list[dict]:
        """Neutral messages -> Responses API ``input`` items."""
        out: list[dict] = []

        for msg in messages:
            if isinstance(msg, AssistantMessage):
                # `native` is the model's list of output items, echoed verbatim
                # (reasoning items included — dropping them breaks continuity).
                out.extend(msg.native)
                continue

            plain_parts: list[dict] = []
            carried_images: list[dict] = []

            for block in msg.blocks:
                if isinstance(block, TextBlock):
                    plain_parts.append(self._text_part(block.text))
                elif isinstance(block, ImageBlock):
                    if block.label:
                        plain_parts.append(self._text_part(block.label))
                    plain_parts.append(self._image_part(block))
                elif isinstance(block, ToolResultBlock):
                    out.append(
                        {
                            "type": "function_call_output",
                            "call_id": block.tool_use_id,
                            "output": block.text,
                        }
                    )
                    if block.images:
                        carried_images.append(
                            self._text_part(
                                FRAME_CARRIER_PREFIX.format(
                                    tool_use_id=block.tool_use_id,
                                    tool_name=block.tool_name,
                                )
                            )
                        )
                        for img in block.images:
                            if img.label:
                                carried_images.append(self._text_part(img.label))
                            carried_images.append(self._image_part(img))

            # Frames first — they belong to the tool outputs just emitted.
            if carried_images:
                out.append({"role": "user", "content": carried_images})
            if plain_parts:
                out.append({"role": "user", "content": plain_parts})

        return out

    @staticmethod
    def to_native_tools(tools: list[dict]) -> list[dict]:
        """Canonical schema -> Responses API tools (FLAT, not nested)."""
        return [
            {
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
            for t in tools
        ]

    # -- the call -----------------------------------------------------------

    def send(self, system: str, messages: list[Message], tools: list[dict]) -> AssistantTurn:
        kwargs = {
            "model": self.model_id,
            "instructions": system,
            "input": self.to_native(messages),
            "tools": self.to_native_tools(tools),
            "max_output_tokens": self.cfg.max_tokens,
        }
        # temperature is NOT set: gpt-5.6-sol rejects any non-default value
        # ("does not support 0 with this model. Only the default (1) value is
        # supported" — measured). Anything provider-specific that IS accepted
        # lives in the config so the decision is recorded, not hardcoded.
        kwargs.update(self.cfg.params)

        response = self.client.responses.create(**kwargs)
        return self._parse(response)

    def _parse(self, response) -> AssistantTurn:
        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
        )
        in_details = getattr(response.usage, "input_tokens_details", None)
        if in_details is not None:
            usage.cache_read_tokens = getattr(in_details, "cached_tokens", 0) or 0
        usage.cost_usd = self.cfg.cost(usage)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        native: list[dict] = []
        refusal = None

        for item in response.output or []:
            # Echo every item back verbatim next turn.
            native.append(item.model_dump() if hasattr(item, "model_dump") else item)

            itype = getattr(item, "type", None)
            if itype == "function_call":
                raw_args = item.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                    parse_error = None
                    if not isinstance(args, dict):
                        args, parse_error = (
                            {},
                            f"arguments were {type(args).__name__}, not an object",
                        )
                except json.JSONDecodeError as exc:
                    # The MODEL's error to recover from (doc 05 §8), returned as
                    # a structured tool error rather than aborting the trial.
                    args, parse_error = {}, f"could not parse arguments as JSON: {exc}"
                tool_calls.append(
                    ToolCall(
                        # call_id is what a function_call_output must reference;
                        # item.id is a different, unusable identifier.
                        id=item.call_id,
                        name=item.name,
                        args=args,
                        parse_error=parse_error,
                    )
                )
            elif itype == "message":
                for part in getattr(item, "content", None) or []:
                    ptype = getattr(part, "type", None)
                    if ptype == "output_text":
                        text_parts.append(part.text)
                    elif ptype == "refusal":
                        refusal = getattr(part, "refusal", "refusal")

        return AssistantTurn(
            text="\n".join(p for p in text_parts if p),
            tool_calls=tool_calls,
            usage=usage,
            raw=native,
            stop_reason=getattr(response, "status", "") or "",
            refusal=refusal,
        )

    # -- one-shot helper, mirroring the Anthropic adapter -------------------

    def ask_about_image(self, prompt: str, jpeg_b64: str, max_tokens: int = 2000) -> str:
        response = self.client.responses.create(
            model=self.model_id,
            max_output_tokens=max_tokens,
            input=[
                {
                    "role": "user",
                    "content": [
                        self._image_part(ImageBlock(data_b64=jpeg_b64)),
                        self._text_part(prompt),
                    ],
                }
            ],
        )
        return (getattr(response, "output_text", "") or "").strip()
