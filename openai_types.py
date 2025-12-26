"""Типы, константы, справочники моделей и роутинг для OpenAI API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, TypedDict

logger = logging.getLogger(__name__)

# =============================================================================
# Константы
# =============================================================================

# Статус-коды для retry
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Параметры, запрещённые в текстовом режиме (tool-calling/мультимодальность)
# При вызове call() эти параметры вызовут ошибку
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

# Допустимые роли в текстовом режиме
_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})


# =============================================================================
# Типы
# =============================================================================


class ToolChoice(StrEnum):
    """Стратегия использования инструментов."""

    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"


class SearchContextSize(StrEnum):
    """Размер контекста для Web Search."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Message(TypedDict, total=False):
    """Структура сообщения для Chat Completions API."""

    role: Literal["system", "user", "assistant"]
    content: str
    name: str  # опционально


class ModelCategory(StrEnum):
    """Категория модели OpenAI."""

    STANDARD = "standard"  # Chat Completions API
    SEARCH = "search"  # Chat Completions + web search
    DEEP_RESEARCH = "deep_research"  # Responses API


@dataclass(frozen=True)
class ModelInfo:
    """Информация о модели OpenAI."""

    snapshot: str
    category: ModelCategory
    input_cost_per_m: float  # $/1M tokens
    output_cost_per_m: float  # $/1M tokens


# =============================================================================
# Справочник моделей OpenAI
# =============================================================================

MODELS_REGISTRY: dict[str, ModelInfo] = {
    # Стандартные модели
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
    # Search-модели
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

# Обратная совместимость
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
# Роутинг моделей
# =============================================================================


@dataclass(frozen=True)
class RoutingConfig:
    """Конфигурация автоматического выбора модели.

    Attributes:
        default_cheap: Модель для простых запросов (по умолчанию "gpt-4o-mini").
        default_capable: Модель для сложных запросов (по умолчанию "gpt-4.1-mini").
        token_threshold: Порог токенов для выбора capable модели.
        instruction_chars_threshold: Порог символов инструкций.
        force_capable_on_strict_schema: Выбирать capable при response_format.
        high_risk_keywords: Ключевые слова, сигнализирующие о сложности.

    Warning:
        При переопределении high_risk_keywords учитывайте:
        - Слова проверяются через `kw in text.lower()` (подстрока)
        - Слишком общие слова ("is", "the") вызовут ложные срабатывания
        - Пустой список отключит критерий сложности инструкций
        - Неправильные ключевые слова — причина скрытых ошибок роутинга
        - При проблемах с качеством ответов проверьте логи выбора модели

        Рекомендация: используйте дефолтные значения, если нет
        специфических требований к языку или домену.

    Example:
        >>> # Добавить доменные термины (сохраняя дефолтные)
        >>> custom_cfg = RoutingConfig(
        ...     high_risk_keywords=RoutingConfig().high_risk_keywords + (
        ...         "compliance", "regulatory", "audit",
        ...     )
        ... )
    """

    default_cheap: str = "gpt-4o-mini"  # Дешёвая модель по умолчанию
    default_capable: str = "gpt-4.1-mini"  # Мощная модель по умолчанию
    token_threshold: int = 60000  # Порог токенов для capable модели
    instruction_chars_threshold: int = 1800  # Порог символов инструкций
    force_capable_on_strict_schema: bool = True  # Выбирать capable при строгой схеме
    high_risk_keywords: tuple[str, ...] = (
        # Русский
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


# Module-level константа для переиспользования
_DEFAULT_ROUTING_CONFIG = RoutingConfig()


def estimate_tokens(text: str) -> int:
    """Приблизительная оценка токенов (~4 символа на токен)."""
    return max(1, len(text) // 4) if text else 0


def choose_model(
    system_prompt: str = "",
    user_prompt: str = "",
    attachments_text: str = "",
    *,
    strict_schema: bool = False,
    cfg: RoutingConfig = _DEFAULT_ROUTING_CONFIG,
) -> str:
    """Автоматический выбор модели по критериям сложности.

    Args:
        system_prompt: Системный промпт.
        user_prompt: Пользовательский промпт.
        attachments_text: Текст вложений.
        strict_schema: Используется ли строгая JSON схема.
        cfg: Конфигурация роутинга.

    Returns:
        Модель из cfg.default_cheap для простых запросов
        Модель из cfg.default_capable для сложных/длинных запросов
    """
    full_text = f"{system_prompt}\n{user_prompt}\n{attachments_text}".strip()
    token_est = estimate_tokens(full_text)

    # Вычисляем keyword_hits один раз в начале
    instruction_block = f"{system_prompt}\n{user_prompt}"
    keyword_hits = 0
    if len(instruction_block) >= cfg.instruction_chars_threshold:
        lowered = instruction_block.lower()
        keyword_hits = sum(1 for kw in cfg.high_risk_keywords if kw in lowered)

    chosen_model = cfg.default_cheap
    reason = "default"

    # Критерий 1: Размер контекста
    if token_est >= cfg.token_threshold:
        chosen_model = cfg.default_capable
        reason = "tokens"

    # Критерий 2: Строгая схема
    elif strict_schema and cfg.force_capable_on_strict_schema:
        chosen_model = cfg.default_capable
        reason = "strict_schema"

    # Критерий 3: Сложность инструкций
    elif keyword_hits >= 2:
        chosen_model = cfg.default_capable
        reason = "keywords"

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Model routing: tokens=%d, strict=%s, keyword_hits=%d, chosen=%s (reason=%s)",
            token_est,
            strict_schema,
            keyword_hits,  # Используем уже вычисленное значение
            chosen_model,
            reason,
        )

    return chosen_model


def maybe_escalate(
    output_text: str,
    strict_schema: bool = False,
    cfg: RoutingConfig = _DEFAULT_ROUTING_CONFIG,
) -> str | None:
    """Эскалация после неудачи: если JSON невалиден, повторить на capable модели.

    Args:
        output_text: Текст ответа модели.
        strict_schema: Использовалась ли строгая схема.
        cfg: Конфигурация роутинга (для выбора capable модели).

    Returns:
        Модель из cfg.default_capable если нужна эскалация, None если результат OK.

    Example (паттерн для pipeline):
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
    """Извлекает output list из ответа Responses API."""
    payload = response.model_dump() if hasattr(response, "model_dump") else response
    output = payload.get("output") if isinstance(payload, dict) else None
    return output if isinstance(output, list) else []


def extract_web_sources(response: Any) -> list[dict[str, Any]]:
    """Извлекает источники (sources) из ответа Responses API с Web Search.

    Для работы требуется запрос с include_sources=True (или include=["web_search_call.action.sources"]).

    Args:
        response: Сырой ответ от Responses API (return_raw=True).

    Returns:
        Список источников [{url, title, snippet, ...}, ...].

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
    """Извлекает url_citation аннотации (inline citations) из ответа Responses API.

    Args:
        response: Сырой ответ от Responses API (return_raw=True).

    Returns:
        Список цитат [{url, title, start_index, end_index, ...}, ...].

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
    # Константы
    "_RETRYABLE_STATUS_CODES",
    "_FORBIDDEN_TEXT_KWARGS",
    "_ALLOWED_ROLES",
    # Типы
    "ToolChoice",
    "SearchContextSize",
    "ModelCategory",
    "Message",
    "ModelInfo",
    # Справочники
    "MODELS_REGISTRY",
    "MODELS_ALL",
    "MODELS_STANDARD",
    "MODELS_SEARCH",
    "MODELS_DEEP_RESEARCH",
    # Роутинг
    "RoutingConfig",
    "choose_model",
    "maybe_escalate",
    "estimate_tokens",
    # Web Search helpers
    "extract_web_sources",
    "extract_url_citations",
]
