"""Клиент OpenAI API (класс OpenAIClient и контекстный менеджер)."""

from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal, overload

try:
    from openai import (
        APIConnectionError,
        APIError,
        APIStatusError,
        APITimeoutError,
        OpenAI,
        RateLimitError,
    )
except ImportError as err:
    raise ImportError(
        "Библиотека openai не установлена. Установите: pip install openai"
    ) from err

from .openai_types import (
    _RETRYABLE_STATUS_CODES,
    _FORBIDDEN_TEXT_KWARGS,
    _ALLOWED_ROLES,
    Message,
    ToolChoice,
    SearchContextSize,
    MODELS_REGISTRY,
    choose_model,
)
from .openai_config import (
    OpenAIConfig,
    WebSearchConfig,
    _resolve_config,
    parse_json as _parse_json,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper функции
# =============================================================================


def _get_or_create_client(
    client: "OpenAIClient | None",
    config: OpenAIConfig | None,
    config_path: str | Path | None,
) -> tuple["OpenAIClient", bool]:
    """Возвращает (клиент, own_client) для управления жизненным циклом.

    Args:
        client: Переиспользуемый клиент (если передан).
        config: Конфигурация для создания нового клиента.
        config_path: Путь к YAML файлу конфигурации.

    Returns:
        Кортеж (клиент, own_client), где own_client=True означает,
        что клиент был создан внутри функции и должен быть закрыт.
    """
    if client is not None:
        return client, False
    resolved = _resolve_config(config, config_path)
    return OpenAIClient(resolved), True


# =============================================================================
# Клиент OpenAI
# =============================================================================


class OpenAIClient:
    """Клиент для работы с OpenAI API."""

    def __init__(self, config: OpenAIConfig | None = None):
        """Инициализация клиента OpenAI.

        Args:
            config: Конфигурация. Если None, используются значения по умолчанию.

        Raises:
            ValueError: API ключ не найден.
        """
        self.config = config or OpenAIConfig()

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API ключ не найден. Установите OPENAI_API_KEY "
                "или передайте api_key в конфигурации."
            )

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.config.timeout,
            "max_retries": 0,  # Отключаем SDK retry, используем свой
        }

        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        if self.config.organization:
            client_kwargs["organization"] = self.config.organization

        self.client = OpenAI(**client_kwargs)

    def close(self) -> None:
        """Закрывает HTTP клиент."""
        try:
            self.client.close()
        except Exception:
            pass

    def _merge_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Объединяет параметры: kwargs > defaults > config."""
        defaults = self.config.defaults or {}
        result = {
            **defaults,
            "model": self.config.model,
            "temperature": self.config.temperature,
            **kwargs,
        }
        if self.config.max_completion_tokens is not None:
            result.setdefault(
                "max_completion_tokens", self.config.max_completion_tokens
            )

        # Валидация модели
        model = result.get("model", self.config.model)
        if model not in MODELS_REGISTRY:
            if not any(model.startswith(p) for p in ("gpt-", "o3-", "o4-", "ft:")):
                logger.warning("Неизвестная модель: %s (возможно опечатка)", model)

        return result

    def _validate_text_only(
        self,
        messages: list[Message],
        request_params: dict[str, Any],
    ) -> None:
        """Валидация режима 'только текст'.

        Запрещает tool-calling/мультимодальные параметры.
        Проверяет, что каждое сообщение имеет content=str.
        """
        forbidden = _FORBIDDEN_TEXT_KWARGS.intersection(request_params.keys())
        if forbidden:
            raise ValueError(
                f"Запрещённые параметры для текстового режима: {sorted(forbidden)}. "
                "Используйте call_web_search() для инструментов или return_raw=True."
            )

        for i, m in enumerate(messages):
            if not isinstance(m, dict):
                raise ValueError(f"messages[{i}] должен быть dict")
            role = m.get("role")
            if role not in _ALLOWED_ROLES:
                raise ValueError(
                    f"messages[{i}].role должен быть одним из {sorted(_ALLOWED_ROLES)}"
                )
            content = m.get("content")
            if not isinstance(content, str):
                raise ValueError(
                    f"messages[{i}].content должен быть строкой (текст-only режим)"
                )

    def _build_messages(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        messages: list[Message] | None = None,
    ) -> list[Message]:
        """Строит список messages.

        Args:
            prompt: Пользовательский промпт.
            system_prompt: Системный промпт.
            messages: Готовый список сообщений.

        Returns:
            Список сообщений для API.

        Raises:
            ValueError: Некорректные параметры.
        """
        if messages is not None:
            return messages

        result: list[Message] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        if prompt:
            result.append({"role": "user", "content": prompt})

        if not result:
            raise ValueError(
                "Необходимо указать prompt или messages с хотя бы одним сообщением"
            )
        return result

    def _calc_wait_time(self, attempt: int) -> float:
        """Вычисляет время ожидания для retry с jitter."""
        base = self.config.retry_delay * (2**attempt)
        base = min(base, self.config.max_retry_delay)
        jitter = 1.0 + random.random() * 0.1  # +0-10%
        return base * jitter

    def _execute_with_retry(
        self,
        request_params: dict[str, Any],
        use_responses_api: bool = False,
    ) -> Any:
        """Выполняет запрос с retry логикой.

        Args:
            request_params: Параметры запроса.
            use_responses_api: Использовать Responses API вместо Chat Completions.

        Returns:
            Ответ от API.

        Raises:
            RateLimitError: Превышен лимит запросов.
            APIConnectionError: Ошибка соединения.
            APITimeoutError: Таймаут запроса.
            APIStatusError: Ошибка HTTP статуса.
            APIError: Ошибка API.
        """
        attempts = max(1, int(self.config.max_retries))
        for attempt in range(attempts):
            try:
                if use_responses_api:
                    return self.client.responses.create(**request_params)
                return self.client.chat.completions.create(**request_params)
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                if attempt == attempts - 1:
                    logger.error("All %d retries exhausted: %s", attempts, e)
                    raise
                wait_time = self._calc_wait_time(attempt)
                logger.warning(
                    "Retry %d/%d in %.2fs (%s): %s",
                    attempt + 1,
                    attempts,
                    wait_time,
                    type(e).__name__,
                    e,
                )
                time.sleep(wait_time)
            except APIStatusError as e:
                status = getattr(e, "status_code", None)
                if status not in _RETRYABLE_STATUS_CODES:
                    raise
                if attempt == attempts - 1:
                    logger.error(
                        "All %d retries exhausted (HTTP %s)",
                        attempts,
                        status,
                    )
                    raise
                wait_time = self._calc_wait_time(attempt)
                logger.warning(
                    "Retry %d/%d in %.2fs (HTTP %s)",
                    attempt + 1,
                    attempts,
                    wait_time,
                    status,
                )
                time.sleep(wait_time)
        raise RuntimeError("Unreachable")

    @overload
    def call(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        messages: list[Message] | None = None,
        *,
        stream: Literal[False] = False,
        return_raw: Literal[False] = False,
        auto_model: bool = False,
        resolve_snapshot: bool = False,
        **kwargs: Any,
    ) -> str: ...

    @overload
    def call(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        messages: list[Message] | None = None,
        *,
        stream: Literal[True],
        return_raw: Literal[False] = False,
        auto_model: bool = False,
        resolve_snapshot: bool = False,
        **kwargs: Any,
    ) -> Iterator[str]: ...

    @overload
    def call(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        messages: list[Message] | None = None,
        *,
        stream: bool = False,
        return_raw: Literal[True],
        auto_model: bool = False,
        resolve_snapshot: bool = False,
        **kwargs: Any,
    ) -> Any: ...

    def call(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        messages: list[Message] | None = None,
        stream: bool = False,
        return_raw: bool = False,
        auto_model: bool = False,
        resolve_snapshot: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str] | Any:
        """Вызывает модель OpenAI через Chat Completions API.

        Args:
            prompt: Пользовательский промпт.
            system_prompt: Системный промпт.
            messages: Список сообщений (альтернатива prompt/system_prompt).
            stream: Если True, возвращает итератор.
            return_raw: Если True, возвращает сырой объект ответа.
            auto_model: Автоматический выбор модели по сложности запроса.
            resolve_snapshot: Заменить alias модели на snapshot (детерминированность).
            **kwargs: Все параметры OpenAI Chat Completions API.

        Returns:
            Текст ответа, итератор при stream=True, или сырой объект при return_raw=True.

        Raises:
            ValueError: Некорректные параметры или ответ содержит tool_calls.
            APIError: Ошибка OpenAI API.

        See Also:
            https://platform.openai.com/docs/api-reference/chat/create
        """
        # Строим messages один раз
        msgs = self._build_messages(prompt, system_prompt, messages)

        # Автовыбор модели (использует уже построенные msgs)
        if auto_model and "model" not in kwargs:
            strict = "response_format" in kwargs
            if messages is None:
                # Передаём только исходные промпты (без дублирования)
                kwargs["model"] = choose_model(
                    system_prompt=system_prompt or "",
                    user_prompt=prompt or "",
                    strict_schema=strict,
                )
            else:
                # Передаём полный текст как attachments_text (без дублирования)
                full_text = "\n".join(m.get("content", "") for m in msgs)
                kwargs["model"] = choose_model(
                    attachments_text=full_text,
                    strict_schema=strict,
                )

        request_params = self._merge_params(kwargs)

        # Разрешение snapshot
        if resolve_snapshot:
            model = request_params.get("model", self.config.model)
            if model in MODELS_REGISTRY:
                request_params["model"] = MODELS_REGISTRY[model].snapshot

        # Валидация текст-only режима (если не return_raw)
        if not return_raw:
            self._validate_text_only(msgs, request_params)

        request_params["messages"] = msgs

        if stream:
            request_params["stream"] = True
            stream_obj = self._execute_with_retry(request_params)
            if return_raw:
                return stream_obj
            return self._stream_response(stream_obj)

        response = self._execute_with_retry(request_params)

        if return_raw:
            return response

        if not response.choices:
            raise ValueError("Ответ от API не содержит choices")

        msg = response.choices[0].message
        content = msg.content

        if content is None:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                raise ValueError(
                    "Ответ содержит tool_calls при content=None. "
                    "Используйте return_raw=True и обработайте tool_calls."
                )
            raise ValueError("Ответ от API не содержит content")

        return content

    def call_structured(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        messages: list[Message] | None = None,
        *,
        response_format: dict[str, Any],
        parse: bool = True,
        auto_model: bool = False,
        resolve_snapshot: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Вызов с ожиданием JSON в content (structured outputs).

        Args:
            prompt: Пользовательский промпт.
            system_prompt: Системный промпт.
            messages: Список сообщений.
            response_format: Формат ответа (json_object или json_schema).
            parse: Если True, парсит JSON и возвращает объект.
            auto_model: Автоматический выбор модели по сложности запроса.
            resolve_snapshot: Заменить alias модели на snapshot (детерминированность).
            **kwargs: Дополнительные параметры API.

        Returns:
            Распарсенный JSON (parse=True) или JSON-строка (parse=False).

        Raises:
            ValueError: Невалидный JSON или некорректные параметры.
        """
        if kwargs.get("stream") is True:
            raise ValueError(
                "call_structured не поддерживает stream=True (нужен цельный JSON)"
            )

        text = self.call(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            stream=False,
            return_raw=False,
            response_format=response_format,
            auto_model=auto_model,
            resolve_snapshot=resolve_snapshot,
            **kwargs,
        )

        if not isinstance(text, str):
            raise ValueError("Ожидалась строка, но получен нестроковый результат")

        return _parse_json(text) if parse else text

    def call_web_search(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        messages: list[Message] | None = None,
        model: str | None = None,
        tool_choice: ToolChoice | str | None = None,
        search_context_size: SearchContextSize | str | None = None,
        user_location: dict[str, Any] | None = None,
        include_sources: bool = False,
        return_raw: bool = False,
        resolve_snapshot: bool = False,
        **kwargs: Any,
    ) -> str | Any:
        """Вызывает модель OpenAI с Web Search через Responses API.

        Args:
            prompt: Пользовательский промпт.
            system_prompt: Системный промпт.
            messages: Список сообщений (альтернатива prompt/system_prompt).
            model: Модель (по умолчанию из конфига).
            tool_choice: Стратегия использования web_search: "auto", "required", "none".
            search_context_size: Размер контекста поиска: "low", "medium", "high".
            user_location: Локация пользователя для персонализации результатов.
            include_sources: Если True, запрашивает полный список источников.
                Используйте extract_web_sources(response) для извлечения URL.
            return_raw: Если True, возвращает сырой объект ответа.
            resolve_snapshot: Заменить alias модели на snapshot (детерминированность).
            **kwargs: Дополнительные параметры Responses API.

        Returns:
            Текст ответа или сырой объект при return_raw=True.
            При return_raw=True ответ содержит url_citation аннотации.
            При include_sources=True и return_raw=True — также содержит sources.

        Raises:
            ValueError: Некорректные параметры.
            APIError: Ошибка OpenAI API.

        See Also:
            https://platform.openai.com/docs/guides/tools-web-search
        """
        # Собираем input для Responses API (переиспользуем _build_messages)
        input_content = self._build_messages(prompt, system_prompt, messages)

        # Получаем настройки web_search из конфига или параметров
        ws_config = self.config.web_search or WebSearchConfig()

        effective_tool_choice = tool_choice or ws_config.tool_choice
        effective_search_context_size = (
            search_context_size or ws_config.search_context_size
        )
        effective_user_location = user_location or ws_config.user_location

        # Определяем модель с учётом resolve_snapshot
        effective_model = model or self.config.model
        if resolve_snapshot and effective_model in MODELS_REGISTRY:
            effective_model = MODELS_REGISTRY[effective_model].snapshot

        # Формируем web_search tool
        web_search_tool: dict[str, Any] = {
            "type": "web_search",
            "search_context_size": effective_search_context_size,
        }
        if effective_user_location:
            web_search_tool["user_location"] = effective_user_location

        # Параметры запроса
        request_params: dict[str, Any] = {
            "model": effective_model,
            "input": input_content,
            "tools": [web_search_tool],
            **kwargs,
        }

        # tool_choice для Responses API
        if effective_tool_choice != "auto":
            request_params["tool_choice"] = effective_tool_choice

        # include_sources для получения списка URL
        if include_sources:
            include_list = list(request_params.get("include") or [])
            if "web_search_call.action.sources" not in include_list:
                include_list.append("web_search_call.action.sources")
            request_params["include"] = include_list

        response = self._execute_with_retry(request_params, use_responses_api=True)

        if return_raw:
            return response

        # Извлекаем текстовый ответ из Responses API
        return self._extract_responses_content(response)

    def _extract_responses_content(self, response: Any) -> str:
        """Извлекает текстовый контент из ответа Responses API.

        Args:
            response: Ответ от Responses API.

        Returns:
            Текстовый контент ответа.

        Raises:
            ValueError: Не удалось извлечь контент.
        """
        # Responses API возвращает output как список элементов
        output = getattr(response, "output", None)
        if not output:
            raise ValueError("Ответ Responses API не содержит output")

        # Ищем message с текстом
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                content = getattr(item, "content", None)
                if content:
                    # content может быть списком частей
                    if isinstance(content, list):
                        texts = []
                        for part in content:
                            if getattr(part, "type", None) == "output_text":
                                texts.append(getattr(part, "text", ""))
                        if texts:
                            return "\n".join(texts)
                    elif isinstance(content, str):
                        return content

        raise ValueError("Не удалось извлечь текстовый контент из ответа")

    def _stream_response(self, stream: Any) -> Iterator[str]:
        """Обрабатывает streaming ответ."""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


# =============================================================================
# Контекстный менеджер
# =============================================================================


@contextmanager
def openai_client(
    config: OpenAIConfig | None = None,
    config_path: str | Path | None = None,
):
    """Контекстный менеджер для создания клиента.

    Args:
        config: Готовая конфигурация.
        config_path: Путь к YAML файлу.

    Yields:
        Экземпляр OpenAIClient.

    Example:
        >>> with openai_client() as client:
        ...     response = client.call("Привет")
        ...     search_result = client.call_web_search("Погода в Москве")
    """
    resolved_config = _resolve_config(config, config_path)
    client = OpenAIClient(resolved_config)
    try:
        yield client
    finally:
        client.close()


__all__ = [
    "OpenAIClient",
    "openai_client",
    "_get_or_create_client",
]
