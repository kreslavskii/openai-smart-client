"""OpenAI API client (OpenAIClient class and context manager)."""

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
        "The openai library is not installed. Install it: pip install openai"
    ) from err

from .openai_config import (
    OpenAIConfig,
    WebSearchConfig,
    _resolve_config,
)
from .openai_config import (
    parse_json as _parse_json,
)
from .openai_types import (
    _ALLOWED_ROLES,
    _FORBIDDEN_TEXT_KWARGS,
    _RETRYABLE_STATUS_CODES,
    MODELS_REGISTRY,
    Message,
    SearchContextSize,
    ToolChoice,
    choose_model,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def _get_or_create_client(
    client: "OpenAIClient | None",
    config: OpenAIConfig | None,
    config_path: str | Path | None,
) -> tuple["OpenAIClient", bool]:
    """Return (client, own_client) for lifecycle management.

    Args:
        client: Reusable client (if provided).
        config: Configuration for creating new client.
        config_path: Path to YAML configuration file.

    Returns:
        Tuple (client, own_client), where own_client=True means
        the client was created internally and should be closed.
    """
    if client is not None:
        return client, False
    resolved = _resolve_config(config, config_path)
    return OpenAIClient(resolved), True


# =============================================================================
# OpenAI Client
# =============================================================================


class OpenAIClient:
    """Client for working with OpenAI API."""

    def __init__(self, config: OpenAIConfig | None = None):
        """Initialize OpenAI client.

        Args:
            config: Configuration. If None, defaults are used.

        Raises:
            ValueError: API key not found.
        """
        self.config = config or OpenAIConfig()

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not found. Set OPENAI_API_KEY "
                "or pass api_key in configuration."
            )

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.config.timeout,
            "max_retries": 0,  # Disable SDK retry, use our own
        }

        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        if self.config.organization:
            client_kwargs["organization"] = self.config.organization

        self.client = OpenAI(**client_kwargs)

    def close(self) -> None:
        """Close HTTP client."""
        try:
            self.client.close()
        except Exception:
            pass

    def _merge_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge parameters: kwargs > defaults > config."""
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

        # Model validation
        model = result.get("model", self.config.model)
        if model not in MODELS_REGISTRY:
            if not any(model.startswith(p) for p in ("gpt-", "o3-", "o4-", "ft:")):
                logger.warning("Unknown model: %s (possible typo)", model)

        return result

    def _validate_text_only(
        self,
        messages: list[Message],
        request_params: dict[str, Any],
    ) -> None:
        """Validate 'text-only' mode.

        Forbids tool-calling/multimodal parameters.
        Checks that each message has content=str.
        """
        forbidden = _FORBIDDEN_TEXT_KWARGS.intersection(request_params.keys())
        if forbidden:
            raise ValueError(
                f"Forbidden parameters for text mode: {sorted(forbidden)}. "
                "Use call_web_search() for tools or return_raw=True."
            )

        for i, m in enumerate(messages):
            if not isinstance(m, dict):
                raise ValueError(f"messages[{i}] must be dict")
            role = m.get("role")
            if role not in _ALLOWED_ROLES:
                raise ValueError(
                    f"messages[{i}].role must be one of {sorted(_ALLOWED_ROLES)}"
                )
            content = m.get("content")
            if not isinstance(content, str):
                raise ValueError(
                    f"messages[{i}].content must be string (text-only mode)"
                )

    def _build_messages(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        messages: list[Message] | None = None,
    ) -> list[Message]:
        """Build messages list.

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            messages: Ready message list.

        Returns:
            Message list for API.

        Raises:
            ValueError: Invalid parameters.
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
                "Must specify prompt or messages with at least one message"
            )
        return result

    def _calc_wait_time(self, attempt: int) -> float:
        """Calculate wait time for retry with jitter."""
        base = self.config.retry_delay * (2**attempt)
        base = min(base, self.config.max_retry_delay)
        jitter = 1.0 + random.random() * 0.1  # +0-10%
        return base * jitter

    def _execute_with_retry(
        self,
        request_params: dict[str, Any],
        use_responses_api: bool = False,
    ) -> Any:
        """Execute request with retry logic.

        Args:
            request_params: Request parameters.
            use_responses_api: Use Responses API instead of Chat Completions.

        Returns:
            API response.

        Raises:
            RateLimitError: Rate limit exceeded.
            APIConnectionError: Connection error.
            APITimeoutError: Request timeout.
            APIStatusError: HTTP status error.
            APIError: API error.
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
        """Call OpenAI model via Chat Completions API.

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            messages: Message list (alternative to prompt/system_prompt).
            stream: If True, returns iterator.
            return_raw: If True, returns raw response object.
            auto_model: Automatic model selection based on request complexity.
            resolve_snapshot: Replace model alias with snapshot (determinism).
            **kwargs: All OpenAI Chat Completions API parameters.

        Returns:
            Response text, iterator if stream=True, or raw object if return_raw=True.

        Raises:
            ValueError: Invalid parameters or response contains tool_calls.
            APIError: OpenAI API error.

        See Also:
            https://platform.openai.com/docs/api-reference/chat/create
        """
        # Build messages once
        msgs = self._build_messages(prompt, system_prompt, messages)

        # Auto model selection (uses already built msgs)
        if auto_model and "model" not in kwargs:
            strict = "response_format" in kwargs
            if messages is None:
                # Pass only original prompts (no duplication)
                kwargs["model"] = choose_model(
                    system_prompt=system_prompt or "",
                    user_prompt=prompt or "",
                    strict_schema=strict,
                )
            else:
                # Pass full text as attachments_text (no duplication)
                full_text = "\n".join(m.get("content", "") for m in msgs)
                kwargs["model"] = choose_model(
                    attachments_text=full_text,
                    strict_schema=strict,
                )

        request_params = self._merge_params(kwargs)

        # Snapshot resolution
        if resolve_snapshot:
            model = request_params.get("model", self.config.model)
            if model in MODELS_REGISTRY:
                request_params["model"] = MODELS_REGISTRY[model].snapshot

        # Text-only mode validation (if not return_raw)
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
            raise ValueError("API response contains no choices")

        msg = response.choices[0].message
        content = msg.content

        if content is None:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                raise ValueError(
                    "Response contains tool_calls with content=None. "
                    "Use return_raw=True and handle tool_calls."
                )
            raise ValueError("API response contains no content")

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
        """Call expecting JSON in content (structured outputs).

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            messages: Message list.
            response_format: Response format (json_object or json_schema).
            parse: If True, parses JSON and returns object.
            auto_model: Automatic model selection based on request complexity.
            resolve_snapshot: Replace model alias with snapshot (determinism).
            **kwargs: Additional API parameters.

        Returns:
            Parsed JSON (parse=True) or JSON string (parse=False).

        Raises:
            ValueError: Invalid JSON or incorrect parameters.
        """
        if kwargs.get("stream") is True:
            raise ValueError(
                "call_structured does not support stream=True (needs complete JSON)"
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
            raise ValueError("Expected string, but got non-string result")

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
        """Call OpenAI model with Web Search via Responses API.

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            messages: Message list (alternative to prompt/system_prompt).
            model: Model (default from config).
            tool_choice: Strategy for using web_search: "auto", "required", "none".
            search_context_size: Search context size: "low", "medium", "high".
            user_location: User location for result personalization.
            include_sources: If True, requests full list of sources.
                Use extract_web_sources(response) to extract URLs.
            return_raw: If True, returns raw response object.
            resolve_snapshot: Replace model alias with snapshot (determinism).
            **kwargs: Additional Responses API parameters.

        Returns:
            Response text or raw object if return_raw=True.
            With return_raw=True, response contains url_citation annotations.
            With include_sources=True and return_raw=True — also contains sources.

        Raises:
            ValueError: Invalid parameters.
            APIError: OpenAI API error.

        See Also:
            https://platform.openai.com/docs/guides/tools-web-search
        """
        # Build input for Responses API (reuse _build_messages)
        input_content = self._build_messages(prompt, system_prompt, messages)

        # Get web_search settings from config or parameters
        ws_config = self.config.web_search or WebSearchConfig()

        effective_tool_choice = tool_choice or ws_config.tool_choice
        effective_search_context_size = (
            search_context_size or ws_config.search_context_size
        )
        effective_user_location = user_location or ws_config.user_location

        # Determine model with resolve_snapshot consideration
        effective_model = model or self.config.model
        if resolve_snapshot and effective_model in MODELS_REGISTRY:
            effective_model = MODELS_REGISTRY[effective_model].snapshot

        # Build web_search tool
        web_search_tool: dict[str, Any] = {
            "type": "web_search",
            "search_context_size": effective_search_context_size,
        }
        if effective_user_location:
            web_search_tool["user_location"] = effective_user_location

        # Request parameters
        request_params: dict[str, Any] = {
            "model": effective_model,
            "input": input_content,
            "tools": [web_search_tool],
            **kwargs,
        }

        # tool_choice for Responses API
        if effective_tool_choice != "auto":
            request_params["tool_choice"] = effective_tool_choice

        # include_sources to get URL list
        if include_sources:
            include_list = list(request_params.get("include") or [])
            if "web_search_call.action.sources" not in include_list:
                include_list.append("web_search_call.action.sources")
            request_params["include"] = include_list

        response = self._execute_with_retry(request_params, use_responses_api=True)

        if return_raw:
            return response

        # Extract text response from Responses API
        return self._extract_responses_content(response)

    def _extract_responses_content(self, response: Any) -> str:
        """Extract text content from Responses API response.

        Args:
            response: Response from Responses API.

        Returns:
            Text content of response.

        Raises:
            ValueError: Could not extract content.
        """
        # Responses API returns output as list of items
        output = getattr(response, "output", None)
        if not output:
            raise ValueError("Responses API response contains no output")

        # Find message with text
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                content = getattr(item, "content", None)
                if content:
                    # content can be a list of parts
                    if isinstance(content, list):
                        texts = []
                        for part in content:
                            if getattr(part, "type", None) == "output_text":
                                texts.append(getattr(part, "text", ""))
                        if texts:
                            return "\n".join(texts)
                    elif isinstance(content, str):
                        return content

        raise ValueError("Could not extract text content from response")

    def _stream_response(self, stream: Any) -> Iterator[str]:
        """Process streaming response."""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


# =============================================================================
# Context Manager
# =============================================================================


@contextmanager
def openai_client(
    config: OpenAIConfig | None = None,
    config_path: str | Path | None = None,
):
    """Context manager for creating client.

    Args:
        config: Ready configuration.
        config_path: Path to YAML file.

    Yields:
        OpenAIClient instance.

    Example:
        >>> with openai_client() as client:
        ...     response = client.call("Hello")
        ...     search_result = client.call_web_search("Weather in Moscow")
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
