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
                "media_type": "image/jpeg",
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
            if img.label:
                content.append({"type": "text", "text": img.label})
            content.append(self._image_block(img))
        return content

    def to_native(self, messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for msg in messages:
            if isinstance(msg, AssistantMessage):
                # Echoed back unchanged — thinking blocks included. Rebuilding
                # them would break multi-turn continuity on Fable 5.
                out.append({"role": "assistant", "content": msg.native})
                continue

            content: list[dict] = []
            for block in msg.blocks:
                if isinstance(block, TextBlock):
                    content.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    if block.label:
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

    def send(self, system: str, messages: list[Message], tools: list[dict]) -> AssistantTurn:
        kwargs = {
            "model": self.model_id,
            "max_tokens": self.cfg.max_tokens,
            "system": system,
            "messages": self.to_native(messages),
            "tools": self.to_native_tools(tools),
            # Adaptive is the only supported mode on the locked models; the
            # summary is free (display controls visibility, not billing) and
            # gives the write-up something to quote.
            "thinking": {"type": "adaptive", "display": "summarized"},
        }
        # NOTE: `effort` is deliberately NOT set — the API default (high) is
        # used for both Anthropic models. Picking a level would be an arbitrary
        # constant applied to two of the three contestants; leaving the default
        # keeps "no per-model tuning" (rule 4) literally true.
        kwargs.update(self.cfg.params)

        response = self.client.messages.create(**kwargs)
        return self._parse(response)

    def _parse(self, response) -> AssistantTurn:
        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        )
        usage.cost_usd = self.cfg.cost(usage)

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
            raw=response.content,
            stop_reason=stop_reason,
            thinking="\n".join(p for p in thinking_parts if p),
            refusal=refusal,
        )

    # -- one-shot helper for the out-of-benchmark judge ---------------------

    def ask_about_image(self, prompt: str, jpeg_b64: str, max_tokens: int = 64) -> str:
        """Single question about a single image; returns the text answer.

        Used by T2.3's scene-recognition gate, where the judge is deliberately
        NOT one of the three contestants (doc 04 §8) and there is no tool use.
        """
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        self._image_block(ImageBlock(data_b64=jpeg_b64)),
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(
            b.text for b in (response.content or []) if getattr(b, "type", None) == "text"
        ).strip()


def json_dumps(obj) -> str:
    """Deterministic serialisation for status payloads (stable prompt bytes)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ": "))
