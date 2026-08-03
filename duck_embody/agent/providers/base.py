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

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

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
    """A camera frame, base64-encoded without a provider-specific envelope."""

    data_b64: str
    #: Optional caption, used by look_around to label each bearing.
    label: str | None = None
    media_type: str = "image/jpeg"


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
    #: Audit-only descriptors. Adapters ignore these fields and replay only
    #: ``native``; the neutral request manifest uses them instead of logging
    #: provider-native reasoning content.
    context_index: int | None = None
    global_turn_index: int | None = None
    native_response_sha256: str | None = None


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
    #: Provider response identifiers and hashes only. Provider-native content
    #: can contain reasoning and therefore never belongs in this mapping.
    metadata: dict[str, Any] = field(default_factory=dict)


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON used by request and native-response hashes."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _native_json_value(value: Any) -> Any:
    """Convert an SDK response to stable JSON without exposing it in the log."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if dataclasses.is_dataclass(value):
        return _native_json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _native_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native_json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json", exclude_none=False)
        except TypeError:
            dumped = value.model_dump(exclude_none=False)
        return _native_json_value(dumped)
    if hasattr(value, "to_dict"):
        return _native_json_value(value.to_dict())
    if hasattr(value, "__dict__"):
        return {
            key: _native_json_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def native_response_sha256(response: Any) -> str:
    """Hash the complete native response while logging none of its content."""
    return hashlib.sha256(canonical_json_bytes(_native_json_value(response))).hexdigest()


def response_metadata(response: Any, *, alias: str, model_id: str) -> dict[str, Any]:
    """Provider-neutral response provenance with no API keys or reasoning."""

    def first(*names: str) -> Any:
        for name in names:
            value = getattr(response, name, None)
            if value is not None:
                return _native_json_value(value)
        return None

    return {
        "configured_alias": alias,
        "resolved_model_id": first("model") or model_id,
        "response_id": first("id"),
        "provider_request_id": first("_request_id", "request_id"),
        "created": first("created_at", "created"),
        "fingerprint": first("system_fingerprint", "fingerprint"),
        "native_response_sha256": native_response_sha256(response),
    }


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

    #: Prompt-cache multipliers on the input rate. Anthropic bills a cache READ
    #: at 0.1x and the one-time WRITE at 1.25x. OpenAI's published cached-input
    #: rate for gpt-5.6-sol is $0.50 against $5.00, i.e. the same 0.1x, and it
    #: has no separate write charge — so `cache_write_tokens` is simply 0 there
    #: and the write term vanishes without needing a per-provider branch.
    CACHE_READ_MULTIPLIER = 0.1
    CACHE_WRITE_MULTIPLIER = 1.25

    def cost(self, usage: Usage) -> float:
        """USD for one call, INCLUDING the prompt-cache terms.

        The cache terms are not decoration. With caching on, `input_tokens`
        counts only the UNCACHED remainder — 84 tokens on a measured trial call
        — while the 3,856-token cached prefix is reported separately in
        `cache_read_tokens` / `cache_write_tokens`. Summing only input+output
        would therefore report a trial's cost as a small fraction of the real
        invoice, and would report the cached and uncached calls as costing
        exactly the same amount (measured: both $0.0010 before this fix), which
        is the tell that the cache columns were being dropped on the floor.

        Cost is a published per-model column, so an error here is a wrong number
        in the write-up rather than a crash.
        """
        return (
            usage.input_tokens * self.price_in_per_mtok
            + usage.cache_read_tokens * self.price_in_per_mtok * self.CACHE_READ_MULTIPLIER
            + usage.cache_write_tokens * self.price_in_per_mtok * self.CACHE_WRITE_MULTIPLIER
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


def preflight_provider(name: str) -> ModelConfig:
    """Validate a model config WITHOUT importing the vendor SDK.

    Split out from `build_provider` because of a measured kit hazard (T3.5):
    importing the `anthropic` SDK before `AppLauncher` starts leaves it unable
    to strip its own unset-parameter defaults, so a dozen `omit` sentinels
    survive into the request body and the first call dies with
    ``TypeError: Object of type Omit is not JSON serializable``.

    The caller still wants to fail in a second on a typo'd model name or a
    missing key rather than after a multi-minute cold start — and none of those
    checks need the SDK. So they happen here, pre-kit, and the client itself is
    constructed by `build_provider` after kit is up.
    """
    # `.env` is loaded HERE as well as at adapter import. Until this function
    # existed, the only path to `require_key` ran through a vendor adapter
    # module, whose import called `load_env()` as a side effect. Preflight
    # deliberately does NOT import that module (the whole point — see the
    # docstring), so relying on that side effect made the key look unset and
    # every trial died claiming ANTHROPIC_API_KEY was missing when it was not.
    # `load_dotenv` does not override an already-set variable, so calling it
    # twice is free.
    load_env()

    cfg = load_model_config(name)
    if cfg.provider == "anthropic":
        require_key("ANTHROPIC_API_KEY", "Anthropic")
    elif cfg.provider == "openai":
        require_key("OPENAI_API_KEY", "OpenAI")
    else:
        raise ValueError(f"Unknown provider '{cfg.provider}' in config '{name}'")
    return cfg


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
