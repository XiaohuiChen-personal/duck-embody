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
    response_metadata,
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
            "image_url": f"data:{img.media_type};base64,{img.data_b64}",
        }

    @staticmethod
    def _text_part(text: str) -> dict:
        return {"type": "input_text", "text": text}

    @staticmethod
    def _item_type(item) -> str | None:
        """Type of one echoed output item — dict from ``model_dump`` normally,
        an SDK object if a future item type ever lacks ``model_dump``."""
        if isinstance(item, dict):
            return item.get("type")
        return getattr(item, "type", None)

    @classmethod
    def _without_trailing_reasoning(cls, native: list) -> list:
        """Drop reasoning items that no non-reasoning item follows.

        The defensive half of the pair the Anthropic adapter's empty-turn drop
        is the other half of (gap G5). A gpt-5.6-sol turn that exhausts
        ``max_output_tokens`` DURING reasoning yields reasoning-only output —
        commonly on the derailment path, where the turn is echoed back
        verbatim — and the Responses API documents rejecting a ``reasoning``
        item without its required follow-up item (400). That 400 would surface
        as a doc 05 §8 INFRA rerun of the whole trial, converting one
        contestant's own budget exhaustion (a §8 *scored* model failure — the
        burned turn still counts either way) into a free retry.

        Model-neutral by construction: it fires only on shapes that would
        otherwise 400, and never touches a reasoning item that keeps its
        follower — mid-turn continuity is preserved verbatim (§7.3). Interior
        reasoning items are left alone even in the trailing scan: the scan
        stops at the first non-reasoning item from the end.
        """
        end = len(native)
        while end > 0 and cls._item_type(native[end - 1]) == "reasoning":
            end -= 1
        return native[:end] if end < len(native) else native

    def to_native(self, messages: list[Message]) -> list[dict]:
        """Neutral messages -> Responses API ``input`` items."""
        out: list[dict] = []

        for msg in messages:
            if isinstance(msg, AssistantMessage):
                # `native` is the model's list of output items, echoed verbatim
                # (reasoning items included — dropping them breaks continuity)
                # — EXCEPT trailing reasoning-only items, which the API rejects
                # with a 400 on the echo (see _without_trailing_reasoning). A
                # turn that was reasoning-only becomes an empty extend, the
                # exact analogue of the Anthropic adapter's empty-turn drop.
                out.extend(self._without_trailing_reasoning(msg.native))
                continue

            plain_parts: list[dict] = []
            carried_images: list[dict] = []

            for block in msg.blocks:
                if isinstance(block, TextBlock):
                    plain_parts.append(self._text_part(block.text))
                elif isinstance(block, ImageBlock):
                    # Matched to the Anthropic adapter's rule so a blank
                    # label behaves identically on both providers.
                    if block.label and block.label.strip():
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
                            if img.label and img.label.strip():
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

    def request_kwargs(self, system: str, messages: list[Message], tools: list[dict]) -> dict:
        """The exact request body, split out of :meth:`send` so it is testable.

        ``instructions`` and ``tools`` are **omitted when empty**, matching the
        Anthropic adapter: T3.4's post-episode layout-QA exchange (doc 06 §5.9)
        is a toolless, system-less call, and it runs once per trial at the very
        end — after every dollar of that trial has been spent. Sending `[]` /
        `""` there would be asking the API to interpret an absence rather than
        stating one. The benchmark path passes both non-empty and is unchanged.
        """
        kwargs = {
            "model": self.model_id,
            "input": self.to_native(messages),
            "max_output_tokens": self.cfg.max_tokens,
        }
        if system:
            kwargs["instructions"] = system
        if tools:
            kwargs["tools"] = self.to_native_tools(tools)
        # temperature is NOT set: gpt-5.6-sol rejects any non-default value
        # ("does not support 0 with this model. Only the default (1) value is
        # supported" — measured). Anything provider-specific that IS accepted
        # lives in the config so the decision is recorded, not hardcoded.
        kwargs.update(self.cfg.params)
        return kwargs

    def send(self, system: str, messages: list[Message], tools: list[dict]) -> AssistantTurn:
        response = self.client.responses.create(
            **self.request_kwargs(system, messages, tools)
        )
        return self._parse(response)

    def _parse(self, response) -> AssistantTurn:
        # OpenAI's `input_tokens` is the TOTAL. Both cached reads and GPT-5.6
        # cache writes are subsets in `input_tokens_details`; subtract them to
        # obtain the disjoint uncached bucket before pricing. Treating the total
        # like Anthropic's uncached field double-charges every cache read.
        input_total = getattr(response.usage, "input_tokens", 0) or 0
        output_total = getattr(response.usage, "output_tokens", 0) or 0
        provider_total = getattr(response.usage, "total_tokens", None)
        in_details = getattr(response.usage, "input_tokens_details", None)
        cache_read = (
            getattr(in_details, "cached_tokens", 0) or 0
            if in_details is not None
            else 0
        )
        cache_write = (
            getattr(in_details, "cache_write_tokens", 0) or 0
            if in_details is not None
            else 0
        )
        output_details = getattr(response.usage, "output_tokens_details", None)
        reasoning_tokens = (
            getattr(output_details, "reasoning_tokens", None)
            if output_details is not None
            else None
        )
        usage = Usage(
            input_tokens_total=input_total,
            input_tokens_uncached=input_total - cache_read - cache_write,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            output_tokens_total=output_total,
            reasoning_tokens=reasoning_tokens,
            provider_reported_total_tokens=provider_total,
        )
        self.cfg.price_usage(usage)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        native: list[dict] = []
        refusal = None

        for item in response.output or []:
            # Echo every item back next turn — minus the server-only fields.
            #
            # MEASURED (T3.5 GPT dry run, first multi-turn echo): the SDK's
            # model_dump() includes `status: "completed"` on output items, and
            # replaying it verbatim is a 400 — "Unknown parameter:
            # 'input[0].status'". The reasoning-only probe missed this because
            # its whole turn was stripped; a normal turn (reasoning +
            # function_call) echoes its items and dies on the FIRST follow-up
            # request of every trial. `exclude_none` for the same reason:
            # nullable fields the API rejects as explicit nulls on input.
            dumped = (
                item.model_dump(exclude_none=True)
                if hasattr(item, "model_dump")
                else dict(item)
            )
            dumped.pop("status", None)
            native.append(dumped)

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
            metadata=response_metadata(
                response, alias=self.name, model_id=self.model_id
            ),
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
