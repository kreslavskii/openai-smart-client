"""Универсальный модуль для работы с OpenAI API.

⚠️ ДЛЯ AI АГЕНТОВ: Перед использованием модуля прочитайте файл `00_AGENT_INSTRUCTIONS.md`
   в директории модуля. Там содержится полная документация для автоматического использования.

Этот модуль предоставляет удобный интерфейс для вызова моделей OpenAI
с поддержкой автоматического выбора модели, Web Search, structured outputs
и других возможностей API.

Основные компоненты:
- OpenAIClient: класс для работы с OpenAI API
- call_openai(): функциональный API для простых вызовов
- call_openai_structured(): вызовы с JSON structured outputs
- call_openai_web_search(): вызовы с Web Search через Responses API
- RoutingConfig: конфигурация автоматического выбора модели

Example:
    >>> from openai_client_module import call_openai
    >>> result = call_openai("Привет, как дела?")
    >>> print(result)
"""

from .openai_api import (
    call_openai,
    call_openai_markdown,
    call_openai_structured,
    call_openai_web_search,
)
from .openai_client import (
    OpenAIClient,
    openai_client,
)
from .openai_config import (
    OpenAIConfig,
    WebSearchConfig,
    parse_json,
)
from .openai_types import (
    MODELS_ALL,
    MODELS_DEEP_RESEARCH,
    MODELS_REGISTRY,
    MODELS_SEARCH,
    MODELS_STANDARD,
    Message,
    ModelCategory,
    ModelInfo,
    RoutingConfig,
    SearchContextSize,
    ToolChoice,
    choose_model,
    estimate_tokens,
    extract_url_citations,
    extract_web_sources,
    maybe_escalate,
)

__all__ = [
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
    # Конфигурация
    "WebSearchConfig",
    "OpenAIConfig",
    # Клиент
    "OpenAIClient",
    "openai_client",
    # Функциональный API
    "call_openai",
    "call_openai_structured",
    "call_openai_web_search",
    "call_openai_markdown",
    # Helpers
    "extract_web_sources",
    "extract_url_citations",
    # JSON parsing (для Schema-модуля)
    "parse_json",
]

__version__ = "1.0.0"
