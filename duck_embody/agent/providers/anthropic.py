"""Anthropic Messages API adapter (Fable 5, Opus 5, and the Sonnet 5 judge).

Wire shape (doc 05 §7.2):

* tools pass through nearly verbatim — ``{name, description, input_schema}``;
* the assistant turn carries ``tool_use`` blocks whose ``input`` is already a
  parsed object;
* results go back as **one user message** containing every ``tool_result``
  block — splitting them across messages is not rejected by the API but
  measurably degrades the model's parallel-tool-use behaviour;
* camera frames ride *inside* a ``tool_result``'s content as base64 image
  blocks, which is the thing the OpenAI adapter cannot do.

Three model facts this adapter is built around, all of which are 400s if
ignored on the locked models:

* **No sampling parameters.** ``temperature`` / ``top_p`` / ``top_k`` were
  removed on Fable 5 and Opus 5. Doc 05 §7.1's warning is correct: determinism
  of decoding is simply unavailable here, and trial-level reproducibility comes
  from the fixed sim seeds instead.
* **Thinking is not configurable the old way.** ``budget_tokens`` is removed;
  Fable 5 thinks always. We request ``{"type": "adaptive", "display":
  "summarized"}`` so the transcript records *some* reasoning for the qualitative
  audit — display affects visibility only, never how much thinking is billed.
* **``max_tokens`` caps thinking + text together**, so it is sized with headroom
  rather than around the (small) tool-call output.

``stop_reason`` is checked before ``content`` is read: a refusal returns HTTP
200 with empty or partial content, so indexing ``content[0]`` blindly would
crash on exactly the runs we most want to record.
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


class AnthropicProvider:
    def __init__(self, cfg: ModelConfig, max_retries: int = 5):
        import anthropic

        require_key("ANTHROPIC_API_KEY", "Anthropic")
        self.cfg = cfg
        self.name = cfg.name
        self.model_id = cfg.model_id
        # The SDK retries 408/409/429/5xx and connection errors with
        # exponential backoff natively; we only choose how many times.
        self.client = anthropic.Anthropic(max_retries=max_retries)
        self.retries_seen = 0

    # -- request shaping ----------------------------------------------------

    @staticmethod
    def _image_block(img: ImageBlock) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.media_type,
                "data": img.data_b64,
            },
        }

    def _tool_result_content(self, block: ToolResultBlock) -> list[dict]:
        """A tool_result's content: the JSON status text plus any frames.

        Each image is preceded by its label when one exists, so `look_around`'s
        four bearings are distinguishable.
        """
        content: list[dict] = [{"type": "text", "text": block.text}]
        for img in block.images:
            # Nested inside a tool_result, where the API's non-empty-text rule
            # does NOT apply (measured: the same blank block 400s at the top
            # level and is accepted here). Guarded anyway so both label sites
            # read the same and neither invites a "why only that one?" edit.
            if img.label and img.label.strip():
                content.append({"type": "text", "text": img.label})
            content.append(self._image_block(img))
        return content

    def to_native(self, messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for msg in messages:
            if isinstance(msg, AssistantMessage):
                if not msg.native:
                    # An EMPTY assistant turn is dropped rather than echoed.
                    #
                    # NOT because echoing it is rejected - it is not. MEASURED 2026-07-26 against the live API (claude-opus-5, with the
                    # 12 tool schemas and adaptive thinking, i.e. the exact trial config):
                    #
                    #     {"role": "assistant", "content": []}   -> ACCEPTED (stop=end_turn)
                    #     {"role": "user",      "content": []}   -> 400 "user messages must ..."
                    #     {"role": "assistant", "content": None} -> 400 "Input should be a valid list"
                    #
                    # So the 400 a refusal used to cause came from the EMPTY USER MESSAGE that
                    # doc 05 §3.1's pseudocode appended when it had no derailment branch — not
                    # from the empty assistant turn. The derailment branch (which sends a
                    # non-empty nudge) is what actually fixes it.
                    #
                    # Dropping is kept anyway, on its own merits: `content:
                    # None` IS rejected, so normalising through this path is a
                    # real guard; and an empty turn is a no-op that costs
                    # tokens and tells the model nothing. Synthesising a text
                    # block instead would put words the model never wrote into
                    # its own transcript.
                    #
                    # Safe because the turn has no `tool_use` blocks (that is
                    # why it took the derailment branch), so nothing is
                    # orphaned, and the nudge that follows is a user message -
                    # consecutive user messages are accepted (measured above).
                    continue
                # Echoed back unchanged — thinking blocks included. Rebuilding
                # them would break multi-turn continuity on Fable 5.
                out.append({"role": "assistant", "content": msg.native})
                continue

            content: list[dict] = []
            for block in msg.blocks:
                if isinstance(block, TextBlock):
                    content.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    # `.strip()`, not truthiness: a whitespace-only label
                    # is a 400 here and silently fine on OpenAI (measured).
                    if block.label and block.label.strip():
                        content.append({"type": "text", "text": block.label})
                    content.append(self._image_block(block))
                elif isinstance(block, ToolResultBlock):
                    entry = {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": self._tool_result_content(block),
                    }
                    if block.is_error:
                        entry["is_error"] = True
                    content.append(entry)
            out.append({"role": "user", "content": content})
        return out

    @staticmethod
    def to_native_tools(tools: list[dict]) -> list[dict]:
        """Canonical schema -> Anthropic tools (essentially identity)."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in tools
        ]

    # -- the call -----------------------------------------------------------

    def request_kwargs(self, system: str, messages: list[Message], tools: list[dict]) -> dict:
        """The exact request body, split out of :meth:`send` so it is testable.

        ``system`` and ``tools`` are **omitted when empty** rather than sent as
        `""` / `[]`. T3.4's post-episode layout-QA exchange (doc 06 §5.9) is a
        deliberately toolless, system-less call — "you have no camera, no robot,
        and no tools now" — and an empty string for `system` risks the API's
        non-empty-text-block rule on a request that runs once per trial, at the
        very end, after all the money has been spent. Omitting the key asks for
        exactly what the exchange means. The benchmark path always passes both
        non-empty, so its body is unchanged.
        """
        kwargs = {
            "model": self.model_id,
            "max_tokens": self.cfg.max_tokens,
            "messages": self.to_native(messages),
            # Adaptive is the only supported mode on the locked models; the
            # summary is free (display controls visibility, not billing) and
            # gives the write-up something to quote.
            "thinking": {"type": "adaptive", "display": "summarized"},
        }
        if system:
            # PROMPT CACHING. The system prompt + 12 tool schemas are 3,919
            # MEASURED tokens (anthropic count_tokens, 2026-07-26) and are
            # byte-identical on every one of a trial's ~50 calls — 22% of all
            # input tokens, re-billed at full rate every turn without this.
            #
            # Marking the system block caches everything before it in the
            # prompt hierarchy (tools, then system), so one breakpoint covers
            # both. Reads bill at 0.1x, the one-time write at 1.25x.
            #
            # BENCHMARK-SAFE, which is the only reason it is allowed inside the
            # freeze: caching changes what we are BILLED, never what any model
            # sees — the assembled prompt is byte-for-byte identical with and
            # without the marker. It is applied to both Anthropic contestants
            # identically. OpenAI needs no equivalent: the Responses API caches
            # automatically (gpt-5.6-sol cached input $0.50/M vs $5.00/M), which
            # is why that contestant already got the discount and these two did
            # not — an asymmetry in the BILL, not in the measurement.
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if tools:
            kwargs["tools"] = self.to_native_tools(tools)
        # NOTE: `effort` is deliberately NOT set — the API default (high) is
        # used for both Anthropic models. Picking a level would be an arbitrary
        # constant applied to two of the three contestants; leaving the default
        # keeps "no per-model tuning" (rule 4) literally true.
        kwargs.update(self.cfg.params)
        return kwargs

    def send(self, system: str, messages: list[Message], tools: list[dict]) -> AssistantTurn:
        response = self.client.messages.create(
            **self.request_kwargs(system, messages, tools)
        )
        return self._parse(response)

    def _parse(self, response) -> AssistantTurn:
        # Anthropic's `input_tokens` is ONLY the uncached remainder. Cache reads
        # and cache creations are additional, disjoint buckets, so normalized
        # total input is their sum (Messages API usage documentation, 2026-08-02).
        input_uncached = getattr(response.usage, "input_tokens", 0) or 0
        cache_read = (
            getattr(response.usage, "cache_read_input_tokens", 0) or 0
        )
        cache_write = (
            getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        )
        usage = Usage(
            input_tokens_total=input_uncached + cache_read + cache_write,
            input_tokens_uncached=input_uncached,
            output_tokens_total=(
                getattr(response.usage, "output_tokens", 0) or 0
            ),
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
        self.cfg.price_usage(usage)

        stop_reason = response.stop_reason or ""

        # Refusals arrive as HTTP 200 with empty or partial content. Check the
        # stop reason BEFORE touching content.
        refusal = None
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            refusal = f"refusal (category={category})"

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, args=dict(block.input or {}))
                )

        return AssistantTurn(
            text="\n".join(p for p in text_parts if p),
            tool_calls=tool_calls,
            usage=usage,
            # Normalised to a list, never `None`: a refusal is HTTP 200 with an
            # empty (or absent) `content`, and `to_native` must be able to tell
            # "nothing to echo" from a real turn without special-casing `None`.
            # It drops an empty one rather than emitting invalid content — see
            # there for why that is a scored-vs-rerun distinction.
            raw=list(response.content or []),
            stop_reason=stop_reason,
            thinking="\n".join(p for p in thinking_parts if p),
            refusal=refusal,
            metadata=response_metadata(
                response, alias=self.name, model_id=self.model_id
            ),
        )

    # -- one-shot helper for the out-of-benchmark judge ---------------------

    def ask_about_image(self, prompt: str, jpeg_b64: str, max_tokens: int = 64) -> str:
        """Single question about a single image; returns the text answer.

        Used by T2.3's scene-recognition gate, where the judge is deliberately
        NOT one of the three contestants (doc 04 §8) and there is no tool use.
        """
        return self.ask_about_images(prompt, [jpeg_b64], max_tokens=max_tokens)

    def ask_about_images(
        self, prompt: str, jpegs_b64: list[str], labels: list[str] | None = None,
        max_tokens: int = 64,
    ) -> str:
        """Ask one question about several images at once.

        T2.3's gate shows the judge a whole four-bearing sweep in one call,
        because that is what ``look_around()`` gives a contestant — judging
        single frames would set a bar the benchmark never actually asks a model
        to clear.
        """
        content: list[dict] = []
        for i, b64 in enumerate(jpegs_b64):
            if labels and i < len(labels):
                content.append({"type": "text", "text": labels[i]})
            content.append(self._image_block(ImageBlock(data_b64=b64)))
        content.append({"type": "text", "text": prompt})

        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(
            b.text for b in (response.content or []) if getattr(b, "type", None) == "text"
        ).strip()


def json_dumps(obj) -> str:
    """Deterministic serialisation for status payloads (stable prompt bytes)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ": "))
