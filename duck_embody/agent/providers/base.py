"""Provider-neutral message format and the ``Provider`` interface.

Two APIs, one benchmark. Anthropic and OpenAI disagree about almost every
surface detail of tool use — where results go, whether arguments arrive parsed
or as a JSON string, and (the one that actually bites) **whether an image may
ride inside a tool result at all**. OpenAI's tool-role messages accept string
content only, so a camera frame cannot be returned the way Anthropic's
``tool_result`` image block allows.

The rule that keeps the comparison honest (AGENTS.md rule 4, doc 04 §6.1): the
**logical** payload is frozen — one image and one JSON status object per
observation, identical field names, identical values, for all three models. Only
the wire encoding differs, and each adapter is responsible for delivering the
same information.

Everything above this layer speaks the neutral format defined here; the adapters
translate. Assistant turns are the exception — they are echoed back
provider-native (``AssistantTurn.raw``), because Anthropic requires thinking
blocks to be replayed unchanged and re-serialising them would corrupt the turn.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_env() -> None:
    """Load ``.env`` (gitignored) so API keys never live in a tracked file.

    Called at adapter import. Keys are read from the environment and never
    logged, echoed, or written anywhere (AGENTS.md rule 6).
    """
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")


# ---------------------------------------------------------------------------
# Neutral content blocks
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    """A top-level text block in a message the harness composes.

    Never nested inside a tool result — ``ToolResultBlock`` carries its own
    ``text`` — so the rule below can be strict without constraining what a tool
    is allowed to return.
    """

    text: str

    def __post_init__(self) -> None:
        # Rejected HERE, in the provider-neutral layer, rather than in the
        # Anthropic adapter where the symptom shows up. MEASURED 2026-07-26
        # against both live APIs:
        #
        #   shape                        Anthropic          OpenAI
        #   user text ""                 400 non-empty      ACCEPTED
        #   user text "   "              400 non-whitespace ACCEPTED
        #   user content []              400 non-empty      ACCEPTED
        #
        # A harness bug that emitted a blank block would therefore 400 the two
        # Anthropic contestants and cost GPT 5.6 sol nothing — the same defect
        # scored as an infra failure for two models and invisible for the third,
        # which is the asymmetry AGENTS.md rule 4 exists to prevent. Guarding at
        # the neutral layer makes the failure identical for all three, and makes
        # it a loud local error in a unit test rather than a remote 400 mid-batch.
        #
        # Every construction site is harness-controlled (the memory block, the
        # derailment nudge, the QA prompt), so this can only ever fire on our
        # own bug — never on anything a model produced.
        if not self.text or not self.text.strip():
            raise ValueError(
                "TextBlock text must be non-empty and not whitespace-only: "
                "the Anthropic Messages API rejects such a block (400 'text "
                "content blocks must contain non-whitespace text') while the "
                "OpenAI Responses API accepts it, so letting one through would "
                "penalise only the Anthropic contestants."
            )


@dataclass
class ImageBlock:
    """A camera frame, base64-encoded JPEG."""

    data_b64: str
    #: Optional caption, used by look_around to label each bearing.
    label: str | None = None


@dataclass
class ToolResultBlock:
    """The result of one tool call, as the model will see it."""

    tool_use_id: str
    tool_name: str
    #: The JSON status object, already serialised.
    text: str
    images: list[ImageBlock] = field(default_factory=list)
    is_error: bool = False


Block = TextBlock | ImageBlock | ToolResultBlock


@dataclass
class UserMessage:
    """A turn from the harness to the model."""

    blocks: list[Block]


@dataclass
class AssistantMessage:
    """A turn from the model, echoed back verbatim in provider-native form."""

    native: Any


Message = UserMessage | AssistantMessage


# ---------------------------------------------------------------------------
# Model output
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    #: Always a parsed dict. OpenAI delivers a JSON *string*; the adapter parses
    #: it, and a parse failure becomes `parse_error` rather than an exception —
    #: a malformed call is the model's mistake to recover from (doc 05 §8), not
    #: an infrastructure failure.
    args: dict
    parse_error: str | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def as_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd_estimate": round(self.cost_usd, 6),
        }


@dataclass
class AssistantTurn:
    #: Any prose the model wrote (visible text, not reasoning).
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    #: Provider-native content, echoed back on the next request unchanged.
    raw: Any
    stop_reason: str
    #: Model's summarised reasoning, when the provider exposes it. Recorded for
    #: the qualitative audit; never fed back as anything but `raw`.
    thinking: str = ""
    #: Populated when the provider declined the request outright.
    refusal: str | None = None


class Provider(Protocol):
    """One model, one frozen configuration."""

    name: str
    model_id: str

    def send(
        self, system: str, messages: list[Message], tools: list[dict]
    ) -> AssistantTurn: ...


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    name: str
    provider: str
    model_id: str
    max_tokens: int
    price_in_per_mtok: float
    price_out_per_mtok: float
    #: Extra provider-specific request parameters, verbatim from the YAML.
    params: dict = field(default_factory=dict)
    notes: str = ""

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.price_in_per_mtok
            + usage.output_tokens * self.price_out_per_mtok
        ) / 1_000_000


def load_model_config(name: str) -> ModelConfig:
    """Read ``configs/models/<name>.yaml``.

    Model IDs are configured values, never hardcoded in the loop (doc 05 §7.3):
    IDs live in one place next to their price sheet, and the batch reads them at
    run time so `results/freeze.json` can hash exactly what ran.
    """
    import yaml

    path = REPO_ROOT / "configs" / "models" / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (REPO_ROOT / "configs" / "models").glob("*.yaml"))
        raise FileNotFoundError(f"No model config '{name}'. Available: {available}")
    raw = yaml.safe_load(path.read_text())
    return ModelConfig(
        name=name,
        provider=raw["provider"],
        model_id=raw["model_id"],
        max_tokens=raw["max_tokens"],
        price_in_per_mtok=raw["price_in_per_mtok"],
        price_out_per_mtok=raw["price_out_per_mtok"],
        params=raw.get("params") or {},
        notes=raw.get("notes", ""),
    )


def build_provider(name: str, max_retries: int = 5):
    """Construct the adapter named by ``configs/models/<name>.yaml``."""
    cfg = load_model_config(name)
    if cfg.provider == "anthropic":
        from duck_embody.agent.providers.anthropic import AnthropicProvider

        return AnthropicProvider(cfg, max_retries=max_retries)
    if cfg.provider == "openai":
        from duck_embody.agent.providers.openai import OpenAIProvider

        return OpenAIProvider(cfg, max_retries=max_retries)
    raise ValueError(f"Unknown provider '{cfg.provider}' in config '{name}'")


def require_key(env_var: str, provider: str) -> None:
    """Fail early and clearly, without ever revealing the key's value."""
    if not os.environ.get(env_var):
        raise RuntimeError(
            f"{env_var} is not set — the {provider} adapter cannot run. "
            "Copy .env.example to .env and fill it in (see AGENTS.md rule 6)."
        )
