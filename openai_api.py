"""Функциональный API для OpenAI (call_openai_* функции и CLI)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Iterator, Literal, overload

try:
    from openai import APIError
except ImportError as err:
    raise ImportError(
        "Библиотека openai не установлена. Установите: pip install openai"
    ) from err

from .openai_client import (
    OpenAIClient,
    _get_or_create_client,
)
from .openai_config import OpenAIConfig
from .openai_types import (
    Message,
    SearchContextSize,
    ToolChoice,
)

# =============================================================================
# Функциональный API
# =============================================================================


@overload
def call_openai(
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    *,
    stream: Literal[False] = False,
    return_raw: Literal[False] = False,
    auto_model: bool = False,
    resolve_snapshot: bool = False,
    config: OpenAIConfig | None = None,
    config_path: str | Path | None = None,
    client: OpenAIClient | None = None,
    **kwargs: Any,
) -> str: ...


@overload
def call_openai(
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    *,
    stream: Literal[True],
    return_raw: Literal[False] = False,
    auto_model: bool = False,
    resolve_snapshot: bool = False,
    config: OpenAIConfig | None = None,
    config_path: str | Path | None = None,
    client: OpenAIClient | None = None,
    **kwargs: Any,
) -> Iterator[str]: ...


@overload
def call_openai(
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    *,
    stream: bool = False,
    return_raw: Literal[True],
    auto_model: bool = False,
    resolve_snapshot: bool = False,
    config: OpenAIConfig | None = None,
    config_path: str | Path | None = None,
    client: OpenAIClient | None = None,
    **kwargs: Any,
) -> Any: ...


def call_openai(
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    stream: bool = False,
    return_raw: bool = False,
    auto_model: bool = False,
    resolve_snapshot: bool = False,
    config: OpenAIConfig | None = None,
    config_path: str | Path | None = None,
    client: OpenAIClient | None = None,
    **kwargs: Any,
) -> str | Iterator[str] | Any:
    """Вызов OpenAI API через Chat Completions.

    Для web search можно использовать search-модели:
    - gpt-4o-search-preview
    - gpt-4o-mini-search-preview
    - gpt-5-search-api (может требовать верификации)

    Args:
        prompt: Пользовательский промпт.
        system_prompt: Системный промпт.
        messages: Список сообщений (альтернатива prompt/system_prompt).
        model: Модель для использования.
        temperature: Температура генерации.
        max_completion_tokens: Максимальное количество токенов.
        max_tokens: Устаревший параметр, используйте max_completion_tokens.
        stream: Если True, возвращает итератор.
        return_raw: Если True, возвращает сырой объект ответа.
        auto_model: Автоматический выбор модели по сложности запроса.
        resolve_snapshot: Заменить alias модели на snapshot (детерминированность).
        config: Конфигурация.
        config_path: Путь к YAML файлу конфигурации.
        client: Переиспользуемый клиент (для batch операций).
            При передаче client игнорируются config/config_path.
        **kwargs: Все параметры OpenAI Chat Completions API.

    Returns:
        Текст ответа, итератор при stream=True, или сырой объект при return_raw=True.

    Raises:
        ValueError: Некорректные параметры.
        APIError: Ошибка OpenAI API.

    See Also:
        https://platform.openai.com/docs/api-reference/chat/create
    """
    # Deprecated max_tokens
    if max_tokens is not None:
        warnings.warn(
            "max_tokens устарел, используйте max_completion_tokens",
            DeprecationWarning,
            stacklevel=2,
        )
        if max_completion_tokens is None:
            max_completion_tokens = max_tokens

    # Переиспользование клиента или создание нового
    _client, own_client = _get_or_create_client(client, config, config_path)

    if model is not None:
        kwargs["model"] = model
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens

    # При stream=True оборачиваем итератор для корректного закрытия клиента
    if stream:
        iterator = _client.call(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            stream=True,
            return_raw=return_raw,
            auto_model=auto_model,
            resolve_snapshot=resolve_snapshot,
            **kwargs,
        )

        def _gen() -> Iterator[str]:
            try:
                for part in iterator:
                    yield part
            finally:
                if own_client:
                    _client.close()

        return _gen()

    # При stream=False закрываем клиент после вызова (если создали сами)
    try:
        return _client.call(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            stream=False,
            return_raw=return_raw,
            auto_model=auto_model,
            resolve_snapshot=resolve_snapshot,
            **kwargs,
        )
    finally:
        if own_client:
            _client.close()


def call_openai_structured(
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    *,
    response_format: dict[str, Any],
    parse: bool = True,
    model: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    auto_model: bool = False,
    resolve_snapshot: bool = False,
    config: OpenAIConfig | None = None,
    config_path: str | Path | None = None,
    client: OpenAIClient | None = None,
    **kwargs: Any,
) -> Any:
    """Вызов OpenAI API с ожиданием JSON (structured outputs).

    Args:
        prompt: Пользовательский промпт.
        system_prompt: Системный промпт.
        messages: Список сообщений.
        response_format: Формат ответа (json_object или json_schema).
        parse: Если True, парсит JSON и возвращает объект.
        model: Модель.
        temperature: Температура.
        max_completion_tokens: Максимум токенов.
        auto_model: Автоматический выбор модели по сложности запроса.
        resolve_snapshot: Заменить alias модели на snapshot (детерминированность).
        config: Конфигурация.
        config_path: Путь к YAML файлу.
        client: Переиспользуемый клиент (для batch операций).
        **kwargs: Дополнительные параметры API.

    Returns:
        Распарсенный JSON (parse=True) или JSON-строка (parse=False).

    Raises:
        ValueError: Невалидный JSON или некорректные параметры.

    Example:
        >>> obj = call_openai_structured(
        ...     prompt="Верни JSON с полями name и age",
        ...     response_format={"type": "json_object"},
        ... )
    """
    _client, own_client = _get_or_create_client(client, config, config_path)

    try:
        if model is not None:
            kwargs["model"] = model
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens

        return _client.call_structured(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            response_format=response_format,
            parse=parse,
            auto_model=auto_model,
            resolve_snapshot=resolve_snapshot,
            **kwargs,
        )
    finally:
        if own_client:
            _client.close()


def call_openai_web_search(
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    model: str | None = None,
    tool_choice: ToolChoice | str = ToolChoice.AUTO,
    search_context_size: SearchContextSize | str = SearchContextSize.MEDIUM,
    user_location: dict[str, Any] | None = None,
    include_sources: bool = False,
    return_raw: bool = False,
    resolve_snapshot: bool = False,
    config: OpenAIConfig | None = None,
    config_path: str | Path | None = None,
    client: OpenAIClient | None = None,
    **kwargs: Any,
) -> str | Any:
    """Вызов OpenAI API с Web Search через Responses API.

    Это рекомендуемый современный способ использования web search.
    Модель сама решает, когда искать в интернете (tool_choice="auto"),
    или можно принудительно направить к поиску (tool_choice="required").

    Args:
        prompt: Пользовательский промпт.
        system_prompt: Системный промпт.
        messages: Список сообщений.
        model: Модель (gpt-4o, gpt-4.1, o4-mini и др.).
        tool_choice: Стратегия использования web_search:
            - "auto" - модель сама решает искать или нет
            - "required" - принудительный поиск
            - "none" - отключить поиск
        search_context_size: Размер контекста поиска:
            - "low" - минимум контекста, быстрее
            - "medium" - баланс (по умолчанию)
            - "high" - максимум контекста, медленнее
        user_location: Локация для персонализации результатов:
            {"type": "approximate", "country": "RU", "city": "Moscow", ...}
        include_sources: Если True, запрашивает полный список источников.
            Используйте extract_web_sources(response) для извлечения URL.
        return_raw: Если True, возвращает сырой объект с url_citation аннотациями.
        resolve_snapshot: Заменить alias модели на snapshot (детерминированность).
        config: Конфигурация.
        config_path: Путь к YAML файлу конфигурации.
        client: Переиспользуемый клиент (для batch операций).
        **kwargs: Дополнительные параметры Responses API.

    Returns:
        Текст ответа или сырой объект при return_raw=True.

    Raises:
        ValueError: Некорректные параметры.
        APIError: Ошибка OpenAI API.

    See Also:
        https://platform.openai.com/docs/guides/tools-web-search

    Example:
        >>> # Простой поиск
        >>> result = call_openai_web_search("Какая погода в Москве сегодня?")

        >>> # Принудительный поиск с локацией
        >>> result = call_openai_web_search(
        ...     "Последние новости",
        ...     tool_choice=ToolChoice.REQUIRED,
        ...     user_location={"type": "approximate", "country": "RU"},
        ... )

        >>> # Получить сырой ответ с источниками
        >>> raw = call_openai_web_search("...", return_raw=True, include_sources=True)
        >>> sources = extract_web_sources(raw)
        >>> citations = extract_url_citations(raw)
    """
    _client, own_client = _get_or_create_client(client, config, config_path)

    try:
        return _client.call_web_search(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            model=model,
            tool_choice=str(tool_choice),
            search_context_size=str(search_context_size),
            user_location=user_location,
            include_sources=include_sources,
            return_raw=return_raw,
            resolve_snapshot=resolve_snapshot,
            **kwargs,
        )
    finally:
        if own_client:
            _client.close()


def call_openai_markdown(
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    **kwargs: Any,
) -> str:
    """Вызывает OpenAI с форматированием Markdown.

    Args:
        prompt: Пользовательский промпт.
        system_prompt: Системный промпт.
        messages: Список сообщений.
        **kwargs: Все параметры OpenAI API.

    Returns:
        Ответ в формате Markdown.

    Raises:
        ValueError: Если передан response_format (несовместимо с Markdown).
    """
    # Проверка несовместимости с response_format
    if "response_format" in kwargs:
        raise ValueError(
            "call_openai_markdown несовместим с response_format (structured outputs). "
            "Используйте call_openai_structured() для JSON."
        )

    markdown_instruction = (
        "Отвечай строго в формате Markdown. Используй заголовки, списки, "
        "жирный и курсивный текст, блоки кода где это уместно."
    )

    if messages is None:
        full_system = (
            f"{system_prompt}\n\n{markdown_instruction}"
            if system_prompt
            else markdown_instruction
        )
        result = call_openai(prompt=prompt, system_prompt=full_system, **kwargs)
        if not isinstance(result, str):
            raise ValueError("Ожидалась строка в Markdown")
        return result

    modified = [dict(m) for m in messages]
    if modified and modified[0].get("role") == "system":
        modified[0]["content"] = f"{modified[0]['content']}\n\n{markdown_instruction}"
    else:
        modified.insert(0, {"role": "system", "content": markdown_instruction})

    result = call_openai(messages=modified, **kwargs)
    if not isinstance(result, str):
        raise ValueError("Ожидалась строка в Markdown")
    return result


# =============================================================================
# CLI
# =============================================================================


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python -m openai_client_module.openai_api <prompt>")
        print(
            "              python -m openai_client_module.openai_api --search <prompt>"
        )
        print("              python -m openai_client_module.openai_api --json <prompt>")
        print()
        print("Примеры:")
        print(
            "  python -m openai_client_module.openai_api 'Объясни квантовую механику'"
        )
        print(
            "  python -m openai_client_module.openai_api --search 'Какая погода в Москве?'"
        )
        print(
            "  python -m openai_client_module.openai_api --json 'Верни JSON с полем greeting'"
        )
        sys.exit(1)

    mode = "markdown"
    args = sys.argv[1:]

    if args[0] == "--search":
        mode = "search"
        args = args[1:]
    elif args[0] == "--json":
        mode = "json"
        args = args[1:]

    user_prompt = " ".join(args)

    try:
        print("Вызов OpenAI API...")
        if mode == "search":
            print("(с Web Search)")
        elif mode == "json":
            print("(JSON mode)")
        print("-" * 60)

        if mode == "search":
            result = call_openai_web_search(
                user_prompt,
                system_prompt="Ты полезный ассистент. Отвечай на русском языке.",
            )
        elif mode == "json":
            result = call_openai_structured(
                user_prompt,
                system_prompt="Ты полезный ассистент. Возвращай валидный JSON.",
                response_format={"type": "json_object"},
                parse=False,  # Выводим как строку для CLI
            )
        else:
            result = call_openai_markdown(
                user_prompt,
                system_prompt="Ты полезный ассистент. Отвечай на русском языке.",
            )

        print(result)
        print("-" * 60)
        print("Готово!")

    except (APIError, ValueError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


__all__ = [
    "call_openai",
    "call_openai_structured",
    "call_openai_web_search",
    "call_openai_markdown",
]
