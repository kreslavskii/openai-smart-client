"""Functional API for OpenAI (call_openai_* functions and CLI)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Iterator, Literal, overload

try:
    from openai import APIError
except ImportError as err:
    raise ImportError(
        "The openai library is not installed. Install it: pip install openai"
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
# Functional API
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
    """Call OpenAI API via Chat Completions.

    For web search, you can use search models:
    - gpt-4o-search-preview
    - gpt-4o-mini-search-preview
    - gpt-5-search-api (may require verification)

    Args:
        prompt: User prompt.
        system_prompt: System prompt.
        messages: List of messages (alternative to prompt/system_prompt).
        model: Model to use.
        temperature: Generation temperature.
        max_completion_tokens: Maximum number of tokens.
        max_tokens: Deprecated parameter, use max_completion_tokens.
        stream: If True, returns an iterator.
        return_raw: If True, returns raw response object.
        auto_model: Automatic model selection based on request complexity.
        resolve_snapshot: Replace model alias with snapshot (determinism).
        config: Configuration.
        config_path: Path to YAML configuration file.
        client: Reusable client (for batch operations).
            When client is passed, config/config_path are ignored.
        **kwargs: All OpenAI Chat Completions API parameters.

    Returns:
        Response text, iterator if stream=True, or raw object if return_raw=True.

    Raises:
        ValueError: Invalid parameters.
        APIError: OpenAI API error.

    See Also:
        https://platform.openai.com/docs/api-reference/chat/create
    """
    # Deprecated max_tokens
    if max_tokens is not None:
        warnings.warn(
            "max_tokens is deprecated, use max_completion_tokens",
            DeprecationWarning,
            stacklevel=2,
        )
        if max_completion_tokens is None:
            max_completion_tokens = max_tokens

    # Client reuse or creation
    _client, own_client = _get_or_create_client(client, config, config_path)

    if model is not None:
        kwargs["model"] = model
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens

    # For stream=True, wrap iterator for proper client closure
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

    # For stream=False, close client after call (if we created it)
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
    """Call OpenAI API expecting JSON (structured outputs).

    Args:
        prompt: User prompt.
        system_prompt: System prompt.
        messages: List of messages.
        response_format: Response format (json_object or json_schema).
        parse: If True, parses JSON and returns object.
        model: Model.
        temperature: Temperature.
        max_completion_tokens: Maximum tokens.
        auto_model: Automatic model selection based on request complexity.
        resolve_snapshot: Replace model alias with snapshot (determinism).
        config: Configuration.
        config_path: Path to YAML file.
        client: Reusable client (for batch operations).
        **kwargs: Additional API parameters.

    Returns:
        Parsed JSON (parse=True) or JSON string (parse=False).

    Raises:
        ValueError: Invalid JSON or incorrect parameters.

    Example:
        >>> obj = call_openai_structured(
        ...     prompt="Return JSON with name and age fields",
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
    """Call OpenAI API with Web Search via Responses API.

    This is the recommended modern way to use web search.
    The model decides when to search the internet (tool_choice="auto"),
    or you can force search (tool_choice="required").

    Args:
        prompt: User prompt.
        system_prompt: System prompt.
        messages: List of messages.
        model: Model (gpt-4o, gpt-4.1, o4-mini, etc.).
        tool_choice: Strategy for using web_search:
            - "auto" - model decides whether to search
            - "required" - forced search
            - "none" - disable search
        search_context_size: Search context size:
            - "low" - minimal context, faster
            - "medium" - balanced (default)
            - "high" - maximum context, slower
        user_location: User location for result personalization:
            {"type": "approximate", "country": "US", "city": "New York", ...}
        include_sources: If True, requests full list of sources.
            Use extract_web_sources(response) to extract URLs.
        return_raw: If True, returns raw object with url_citation annotations.
        resolve_snapshot: Replace model alias with snapshot (determinism).
        config: Configuration.
        config_path: Path to YAML configuration file.
        client: Reusable client (for batch operations).
        **kwargs: Additional Responses API parameters.

    Returns:
        Response text or raw object if return_raw=True.

    Raises:
        ValueError: Invalid parameters.
        APIError: OpenAI API error.

    See Also:
        https://platform.openai.com/docs/guides/tools-web-search

    Example:
        >>> # Simple search
        >>> result = call_openai_web_search("What's the weather in Moscow today?")

        >>> # Forced search with location
        >>> result = call_openai_web_search(
        ...     "Latest news",
        ...     tool_choice=ToolChoice.REQUIRED,
        ...     user_location={"type": "approximate", "country": "US"},
        ... )

        >>> # Get raw response with sources
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
    """Call OpenAI with Markdown formatting.

    Args:
        prompt: User prompt.
        system_prompt: System prompt.
        messages: List of messages.
        **kwargs: All OpenAI API parameters.

    Returns:
        Response in Markdown format.

    Raises:
        ValueError: If response_format is passed (incompatible with Markdown).
    """
    # Check incompatibility with response_format
    if "response_format" in kwargs:
        raise ValueError(
            "call_openai_markdown is incompatible with response_format (structured outputs). "
            "Use call_openai_structured() for JSON."
        )

    markdown_instruction = (
        "Respond strictly in Markdown format. Use headings, lists, "
        "bold and italic text, code blocks where appropriate."
    )

    if messages is None:
        full_system = (
            f"{system_prompt}\n\n{markdown_instruction}"
            if system_prompt
            else markdown_instruction
        )
        result = call_openai(prompt=prompt, system_prompt=full_system, **kwargs)
        if not isinstance(result, str):
            raise ValueError("Expected string in Markdown")
        return result

    modified = [dict(m) for m in messages]
    if modified and modified[0].get("role") == "system":
        modified[0]["content"] = f"{modified[0]['content']}\n\n{markdown_instruction}"
    else:
        modified.insert(0, {"role": "system", "content": markdown_instruction})

    result = call_openai(messages=modified, **kwargs)
    if not isinstance(result, str):
        raise ValueError("Expected string in Markdown")
    return result


# =============================================================================
# CLI
# =============================================================================


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m openai_client_module.openai_api <prompt>")
        print("       python -m openai_client_module.openai_api --search <prompt>")
        print("       python -m openai_client_module.openai_api --json <prompt>")
        print()
        print("Examples:")
        print("  python -m openai_client_module.openai_api 'Explain quantum mechanics'")
        print(
            "  python -m openai_client_module.openai_api --search 'What is the weather in Moscow?'"
        )
        print(
            "  python -m openai_client_module.openai_api --json 'Return JSON with greeting field'"
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
        print("Calling OpenAI API...")
        if mode == "search":
            print("(with Web Search)")
        elif mode == "json":
            print("(JSON mode)")
        print("-" * 60)

        if mode == "search":
            result = call_openai_web_search(
                user_prompt,
                system_prompt="You are a helpful assistant. Respond in English.",
            )
        elif mode == "json":
            result = call_openai_structured(
                user_prompt,
                system_prompt="You are a helpful assistant. Return valid JSON.",
                response_format={"type": "json_object"},
                parse=False,  # Output as string for CLI
            )
        else:
            result = call_openai_markdown(
                user_prompt,
                system_prompt="You are a helpful assistant. Respond in English.",
            )

        print(result)
        print("-" * 60)
        print("Done!")

    except (APIError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


__all__ = [
    "call_openai",
    "call_openai_structured",
    "call_openai_web_search",
    "call_openai_markdown",
]
