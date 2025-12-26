"""Types, constants, model registries, and routing for OpenAI API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, TypedDict

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Status codes for retry
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Parameters forbidden in text mode (tool-calling/multimodality)
# When calling call(), these parameters will raise an error
_FORBIDDEN_TEXT_KWARGS = frozenset(
    {
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "functions",
        "function_call",
        "modalities",
        "audio",
        "input_audio_format",
        "output_audio_format",
    }
)

# Allowed roles in text mode
_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})


# =============================================================================
# Types
# =============================================================================


class ToolChoice(StrEnum):
    """Tool usage strategy."""

    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"


class SearchContextSize(StrEnum):
    """Context size for Web Search."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Message(TypedDict, total=False):
    """Message structure for Chat Completions API."""

    role: Literal["system", "user", "assistant"]
    content: str
    name: str  # optional


class ModelCategory(StrEnum):
    """OpenAI model category."""

    STANDARD = "standard"  # Chat Completions API
    SEARCH = "search"  # Chat Completions + web search
    DEEP_RESEARCH = "deep_research"  # Responses API


@dataclass(frozen=True)
class ModelInfo:
    """OpenAI model information."""

    snapshot: str
    category: ModelCategory
    input_cost_per_m: float  # $/1M tokens
    output_cost_per_m: float  # $/1M tokens


# =============================================================================
# OpenAI Model Registry
# =============================================================================

MODELS_REGISTRY: dict[str, ModelInfo] = {
    # Standard models
    "gpt-4o": ModelInfo("gpt-4o-2024-11-20", ModelCategory.STANDARD, 2.50, 10.00),
    "gpt-4o-mini": ModelInfo(
        "gpt-4o-mini-2024-07-18", ModelCategory.STANDARD, 0.15, 0.60
    ),
    "gpt-4.1": ModelInfo("gpt-4.1-2025-04-14", ModelCategory.STANDARD, 2.00, 8.00),
    "gpt-4.1-mini": ModelInfo(
        "gpt-4.1-mini-2025-04-14", ModelCategory.STANDARD, 0.40, 1.60
    ),
    "o4-mini": ModelInfo("o4-mini-2025-04-16", ModelCategory.STANDARD, 1.10, 4.40),
    "o3-mini": ModelInfo("o3-mini-2025-01-31", ModelCategory.STANDARD, 1.10, 4.40),
    # Search models
    "gpt-4o-search-preview": ModelInfo(
        "gpt-4o-search-preview", ModelCategory.SEARCH, 2.50, 10.00
    ),
    "gpt-4o-mini-search-preview": ModelInfo(
        "gpt-4o-mini-search-preview", ModelCategory.SEARCH, 0.15, 0.60
    ),
    "gpt-5-search-api": ModelInfo(
        "gpt-5-search-api", ModelCategory.SEARCH, 2.50, 10.00
    ),
    # Deep Research
    "o3-deep-research": ModelInfo(
        "o3-deep-research", ModelCategory.DEEP_RESEARCH, 1.10, 4.40
    ),
    "o4-mini-deep-research": ModelInfo(
        "o4-mini-deep-research", ModelCategory.DEEP_RESEARCH, 1.10, 4.40
    ),
}

# Backward compatibility
MODELS_ALL: dict[str, str] = {
    alias: info.snapshot for alias, info in MODELS_REGISTRY.items()
}
MODELS_STANDARD: dict[str, str] = {
    alias: info.snapshot
    for alias, info in MODELS_REGISTRY.items()
    if info.category == ModelCategory.STANDARD
}
MODELS_SEARCH: dict[str, str] = {
    alias: info.snapshot
    for alias, info in MODELS_REGISTRY.items()
    if info.category == ModelCategory.SEARCH
}
MODELS_DEEP_RESEARCH: dict[str, str] = {
    alias: info.snapshot
    for alias, info in MODELS_REGISTRY.items()
    if info.category == ModelCategory.DEEP_RESEARCH
}


# =============================================================================
# Model Routing
# =============================================================================


@dataclass(frozen=True)
class RoutingConfig:
    """Configuration for automatic model selection.

    Attributes:
        default_cheap: Model for simple requests (default "gpt-4o-mini").
        default_capable: Model for complex requests (default "gpt-4.1-mini").
        token_threshold: Token threshold for selecting capable model.
        instruction_chars_threshold: Instruction characters threshold.
        force_capable_on_strict_schema: Select capable with response_format.
        high_risk_keywords: Keywords signaling complexity.

    Warning:
        When overriding high_risk_keywords, consider:
        - Words are checked via `kw in text.lower()` (substring)
        - Too generic words ("is", "the") will cause false positives
        - Empty list disables instruction complexity criterion
        - Incorrect keywords cause hidden routing errors
        - If response quality issues occur, check model selection logs

        Recommendation: use default values unless you have
        specific language or domain requirements.

    Example:
        >>> # Add domain terms (keeping defaults)
        >>> custom_cfg = RoutingConfig(
        ...     high_risk_keywords=RoutingConfig().high_risk_keywords + (
        ...         "compliance", "regulatory", "audit",
        ...     )
        ... )
    """

    default_cheap: str = "gpt-4o-mini"  # Default cheap model
    default_capable: str = "gpt-4.1-mini"  # Default capable model
    token_threshold: int = 60000  # Token threshold for capable model
    instruction_chars_threshold: int = 1800  # Instruction characters threshold
    force_capable_on_strict_schema: bool = True  # Select capable with strict schema
    high_risk_keywords: tuple[str, ...] = (
        # Russian
        "строго",
        "обязательно",
        "не допускается",
        "валид",
        "схем",
        "json schema",
        "jsonl",
        "якор",
        "цитат",
        "пункт",
        "критер",
        "без домысл",
        "без галлюцин",
        "только",
        "запрещено",
        
        # English
        "strict",
        "required",
        "must not",
        "valid",
        "schema",
        "anchor",
        "citation",
        "point",
        "criteria",
        "no hallucin",
        "only",
        "forbidden",
        "prohibited",
        "mandatory",
        "exactly",
        "precisely",
    )


# Module-level constant for reuse
_DEFAULT_ROUTING_CONFIG = RoutingConfig()


def estimate_tokens(text: str) -> int:
    """Approximate token estimation (~4 characters per token)."""
    return max(1, len(text) // 4) if text else 0


def choose_model(
    system_prompt: str = "",
    user_prompt: str = "",
    attachments_text: str = "",
    *,
    strict_schema: bool = False,
    cfg: RoutingConfig = _DEFAULT_ROUTING_CONFIG,
) -> str:
    """Automatic model selection based on complexity criteria.

    Args:
        system_prompt: System prompt.
        user_prompt: User prompt.
        attachments_text: Attachments text.
        strict_schema: Whether strict JSON schema is used.
        cfg: Routing configuration.

    Returns:
        Model from cfg.default_cheap for simple requests
        Model from cfg.default_capable for complex/long requests
    """
    full_text = f"{system_prompt}\n{user_prompt}\n{attachments_text}".strip()
    token_est = estimate_tokens(full_text)

    # Calculate keyword_hits once at the start
    instruction_block = f"{system_prompt}\n{user_prompt}"
    keyword_hits = 0
    if len(instruction_block) >= cfg.instruction_chars_threshold:
        lowered = instruction_block.lower()
        keyword_hits = sum(1 for kw in cfg.high_risk_keywords if kw in lowered)

    chosen_model = cfg.default_cheap
    reason = "default"

    # Criterion 1: Context size
    if token_est >= cfg.token_threshold:
        chosen_model = cfg.default_capable
        reason = "tokens"

    # Criterion 2: Strict schema
    elif strict_schema and cfg.force_capable_on_strict_schema:
        chosen_model = cfg.default_capable
        reason = "strict_schema"

    # Criterion 3: Instruction complexity
    elif keyword_hits >= 2:
        chosen_model = cfg.default_capable
        reason = "keywords"

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Model routing: tokens=%d, strict=%s, keyword_hits=%d, chosen=%s (reason=%s)",
            token_est,
            strict_schema,
            keyword_hits,  # Use already calculated value
            chosen_model,
            reason,
        )

    return chosen_model


def maybe_escalate(
    output_text: str,
    strict_schema: bool = False,
    cfg: RoutingConfig = _DEFAULT_ROUTING_CONFIG,
) -> str | None:
    """Escalation after failure: if JSON is invalid, retry with capable model.

    Args:
        output_text: Model response text.
        strict_schema: Whether strict schema was used.
        cfg: Routing configuration (for selecting capable model).

    Returns:
        Model from cfg.default_capable if escalation needed, None if result is OK.

    Example (pattern for pipeline):
        >>> model = choose_model(system_prompt, user_prompt, strict_schema=True)
        >>> result = call_openai(prompt, model=model)
        >>>
        >>> escalate_to = maybe_escalate(result, strict_schema=True)
        >>> if escalate_to:
        ...     result = call_openai(prompt, model=escalate_to)
    """
    if strict_schema:
        try:
            json.loads(output_text)
        except (json.JSONDecodeError, TypeError):
            return cfg.default_capable
    return None


# =============================================================================
# Web Search Helpers
# =============================================================================


def _get_output_list(response: Any) -> list[dict[str, Any]]:
    """Extract output list from Responses API response."""
    payload = response.model_dump() if hasattr(response, "model_dump") else response
    output = payload.get("output") if isinstance(payload, dict) else None
    return output if isinstance(output, list) else []


def extract_web_sources(response: Any) -> list[dict[str, Any]]:
    """Extract sources from Responses API response with Web Search.

    Requires request with include_sources=True (or include=["web_search_call.action.sources"]).

    Args:
        response: Raw response from Responses API (return_raw=True).

    Returns:
        List of sources [{url, title, snippet, ...}, ...].

    Example:
        >>> raw = call_openai_web_search("...", return_raw=True, include_sources=True)
        >>> sources = extract_web_sources(raw)
        >>> for s in sources:
        ...     print(s.get("url"), s.get("title"))
    """
    sources: list[dict[str, Any]] = []
    for item in _get_output_list(response):
        if not isinstance(item, dict) or item.get("type") != "web_search_call":
            continue
        action = item.get("action") or {}
        if not isinstance(action, dict):
            continue
        lst = action.get("sources")
        if isinstance(lst, list):
            sources.extend(s for s in lst if isinstance(s, dict))
    return sources


def extract_url_citations(response: Any) -> list[dict[str, Any]]:
    """Extract url_citation annotations (inline citations) from Responses API response.

    Args:
        response: Raw response from Responses API (return_raw=True).

    Returns:
        List of citations [{url, title, start_index, end_index, ...}, ...].

    Example:
        >>> raw = call_openai_web_search("...", return_raw=True)
        >>> citations = extract_url_citations(raw)
        >>> for c in citations:
        ...     print(c.get("url"), c.get("title"))
    """
    citations: list[dict[str, Any]] = []
    for item in _get_output_list(response):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            annotations = part.get("annotations") if isinstance(part, dict) else None
            if isinstance(annotations, list):
                citations.extend(
                    a
                    for a in annotations
                    if isinstance(a, dict) and a.get("type") == "url_citation"
                )
    return citations


__all__ = [
    # Constants
    "_RETRYABLE_STATUS_CODES",
    "_FORBIDDEN_TEXT_KWARGS",
    "_ALLOWED_ROLES",
    # Types
    "ToolChoice",
    "SearchContextSize",
    "ModelCategory",
    "Message",
    "ModelInfo",
    # Registries
    "MODELS_REGISTRY",
    "MODELS_ALL",
    "MODELS_STANDARD",
    "MODELS_SEARCH",
    "MODELS_DEEP_RESEARCH",
    # Routing
    "RoutingConfig",
    "choose_model",
    "maybe_escalate",
    "estimate_tokens",
    # Web Search helpers
    "extract_web_sources",
    "extract_url_citations",
]
