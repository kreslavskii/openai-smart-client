"""Universal module for working with OpenAI API.

⚠️ FOR AI AGENTS: Before using the module, read the file `00_AGENT_INSTRUCTIONS.md`
   in the module directory. It contains complete documentation for automatic use.

This module provides a convenient interface for calling OpenAI models
with support for automatic model selection, Web Search, structured outputs,
and other API capabilities.

Main components:
- OpenAIClient: class for working with OpenAI API
- call_openai(): functional API for simple calls
- call_openai_structured(): calls with JSON structured outputs
- call_openai_web_search(): calls with Web Search via Responses API
- RoutingConfig: configuration for automatic model selection

Example:
    >>> from openai_client_module import call_openai
    >>> result = call_openai("Hello, how are you?")
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
    # Configuration
    "WebSearchConfig",
    "OpenAIConfig",
    # Client
    "OpenAIClient",
    "openai_client",
    # Functional API
    "call_openai",
    "call_openai_structured",
    "call_openai_web_search",
    "call_openai_markdown",
    # Helpers
    "extract_web_sources",
    "extract_url_citations",
    # JSON parsing (for Schema module)
    "parse_json",
]

__version__ = "1.0.0"
