"""Конфигурация для OpenAI API (YAML, dataclasses)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    import yaml
except ImportError as err:
    raise ImportError(
        "Библиотека pyyaml не установлена. Установите: pip install pyyaml"
    ) from err

logger = logging.getLogger(__name__)


# =============================================================================
# Конфигурация
# =============================================================================


@dataclass
class WebSearchConfig:
    """Конфигурация Web Search."""

    tool_choice: Literal["auto", "required", "none"] = "auto"
    search_context_size: Literal["low", "medium", "high"] = "medium"
    user_location: dict[str, Any] | None = None


@dataclass
class OpenAIConfig:
    """Конфигурация для вызова OpenAI API."""

    api_key: str | None = None
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_completion_tokens: int | None = None
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    max_retry_delay: float = 60.0  # Верхний предел задержки retry
    base_url: str | None = None
    organization: str | None = None
    defaults: dict[str, Any] | None = None
    web_search: WebSearchConfig | None = None

    @classmethod
    def _find_config_path(cls) -> Path | None:
        """Ищет файл конфигурации в стандартных местах."""
        env_path = os.getenv("OPENAI_CONFIG_PATH")
        if env_path:
            return Path(env_path)

        current = Path("openai_config.yaml")
        if current.exists():
            return current

        module_config = Path(__file__).parent / "openai_config.yaml"
        if module_config.exists():
            return module_config

        return None

    @classmethod
    def from_yaml(cls, config_path: str | Path | None = None) -> OpenAIConfig:
        """Загружает конфигурацию из YAML файла.

        Args:
            config_path: Путь к файлу. Если None, ищет автоматически.

        Returns:
            Экземпляр OpenAIConfig.

        Raises:
            FileNotFoundError: Файл конфигурации не найден.
            yaml.YAMLError: Ошибка парсинга YAML.
        """
        if config_path is None:
            config_path = cls._find_config_path()
        if config_path is None:
            return cls()

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Ошибка парсинга YAML: {path}: {e}") from e

        # API ключ берётся ТОЛЬКО из переменной окружения (безопасность)
        api_key = os.getenv("OPENAI_API_KEY")

        # max_completion_tokens с fallback на max_tokens для совместимости
        mct = data.get("max_completion_tokens")
        if mct is None:
            mct = data.get("max_tokens")

        # Web Search конфигурация
        web_search_data = data.get("web_search")
        web_search_config = None
        if web_search_data and isinstance(web_search_data, dict):
            web_search_config = WebSearchConfig(
                tool_choice=web_search_data.get("tool_choice", "auto"),
                search_context_size=web_search_data.get(
                    "search_context_size", "medium"
                ),
                user_location=web_search_data.get("user_location"),
            )

        return cls(
            api_key=api_key,
            model=data.get("model", "gpt-4o"),
            temperature=data.get("temperature", 0.7),
            max_completion_tokens=mct,
            timeout=data.get("timeout", 60.0),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 1.0),
            max_retry_delay=data.get("max_retry_delay", 60.0),
            base_url=data.get("base_url"),
            organization=data.get("organization"),
            defaults=data.get("defaults"),
            web_search=web_search_config,
        )


# =============================================================================
# Helper функции
# =============================================================================


def _resolve_config(
    config: OpenAIConfig | None,
    config_path: str | Path | None,
) -> OpenAIConfig | None:
    """Разрешает конфигурацию из различных источников."""
    if config is not None:
        return config
    if config_path is not None:
        return OpenAIConfig.from_yaml(config_path)
    try:
        return OpenAIConfig.from_yaml()
    except FileNotFoundError:
        return None
    except yaml.YAMLError as e:
        logger.warning("Ошибка парсинга конфига, используются дефолты: %s", e)
        return None


def parse_json(text: str, *, max_error_snippet: int = 400) -> Any:
    """Парсит JSON с диагностикой ошибок.

    Публичная функция для использования в Schema-модуле (SGR).

    Args:
        text: JSON строка для парсинга.
        max_error_snippet: Максимальная длина фрагмента текста в сообщении об ошибке.

    Returns:
        Распарсенный JSON объект.

    Raises:
        ValueError: Невалидный JSON с диагностикой ошибки.

    Example:
        >>> from openai_client_module import parse_json
        >>> obj = parse_json('{"key": "value"}')
        >>> print(obj["key"])
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        snippet = text[:max_error_snippet]
        raise ValueError(
            f"Невалидный JSON в ответе модели: {e.msg} (pos={e.pos}). "
            f"Начало ответа: {snippet!r}"
        ) from e


__all__ = [
    "WebSearchConfig",
    "OpenAIConfig",
    "_resolve_config",
    "parse_json",  # Публичный API для Schema-модуля
]
