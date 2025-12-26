# openai_client_module — Technical Documentation for Agents

> **IMPORTANT:** If you are an agent and just opened the codebase, start with the file [`00_AGENT_INSTRUCTIONS.md`](./00_AGENT_INSTRUCTIONS.md)  
> **TARGET AUDIENCE:** AI agents, automated systems, code generators  
> **PURPOSE:** Machine-readable instructions for connecting and using the module without codebase analysis  
> **FORMAT:** Structured function signatures, code examples, technical details

**Version:** 1.0.0 | **Python:** 3.10+ | **Dependencies:** openai, pyyaml

**Module Structure:**
- `openai_types.py` (425 lines) — types, model registries, routing, Web Search helpers
- `openai_config.py` (194 lines) — configuration (YAML, dataclasses), `parse_json()`
- `openai_client.py` (645 lines) — `OpenAIClient` class, `openai_client()` context manager
- `openai_api.py` (506 lines) — functional API (`call_openai_*`), CLI
- `__init__.py` (91 lines) — public API re-export

**Total:** ~1861 lines

---

## Agent Instructions: Module Connection

**Action algorithm for agents:**

1. Check dependencies (Python 3.10+, openai, pyyaml)
2. Install dependencies if necessary (`pip install openai pyyaml`)
3. Import required components from `openai_client_module`
4. Ensure the `OPENAI_API_KEY` environment variable is set
5. Use functions according to the signatures below

### Step 1: Check Dependencies

```python
import sys
if sys.version_info < (3, 10):
    raise RuntimeError("Python 3.10+ required")

try:
    import openai
    import yaml
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}")
```

### Step 2: Install Dependencies (if necessary)

```bash
pip install openai pyyaml
```

### Step 3: Import Module

```python
from openai_client_module import (
    call_openai,
    call_openai_structured,
    call_openai_web_search,
    call_openai_markdown,
    OpenAIClient,
    openai_client,
    OpenAIConfig,
    RoutingConfig,
    Message,
    parse_json,
)
```

### Step 4: API Key Setup

**REQUIRED:** The `OPENAI_API_KEY` environment variable must be set:

```python
import os
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set")
```

**PRIORITY:** API key is taken ONLY from `OPENAI_API_KEY`. Any value in YAML is ignored.

### Step 5: Configuration (optional)

The module looks for `openai_config.yaml` in order:
1. `os.getenv("OPENAI_CONFIG_PATH")`
2. Current working directory
3. `openai_client_module/`

If file is not found, defaults are used (see "Configuration" section).

---

## Public API: Function Signatures

> **IMPORTANT FOR AGENTS:** All functions below are available via `from openai_client_module import ...`  
> Signatures match the actual code. Use them to generate calls.

### call_openai()

**Purpose:** Basic Chat Completions API call.

**Signatures (overload):**

```python
# Streaming: stream=True
def call_openai(
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    stream: bool = True,
    auto_model: bool = False,
    resolve_snapshot: bool = False,
    return_raw: bool = False,
    config: OpenAIConfig | None = None,
    config_path: str | Path | None = None,
    client: OpenAIClient | None = None,
    **kwargs: Any,
) -> Iterator[str]

# Non-streaming: stream=False or not specified
def call_openai(
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    stream: bool = False,
    auto_model: bool = False,
    resolve_snapshot: bool = False,
    return_raw: bool = False,
    config: OpenAIConfig | None = None,
    config_path: str | Path | None = None,
    client: OpenAIClient | None = None,
    **kwargs: Any,
) -> str

# Raw response: return_raw=True
def call_openai(
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    stream: bool = False,
    auto_model: bool = False,
    resolve_snapshot: bool = False,
    return_raw: bool = True,
    config: OpenAIConfig | None = None,
    config_path: str | Path | None = None,
    client: OpenAIClient | None = None,
    **kwargs: Any,
) -> Any
```

**Parameters:**
- `prompt: str | None` — user prompt (if `messages=None`)
- `system_prompt: str | None` — system prompt (if `messages=None`)
- `messages: list[Message] | None` — message list (alternative to `prompt`/`system_prompt`)
- `model: str | None` — model (if `None`, taken from config or `gpt-4o`)
- `temperature: float | None` — temperature (0-2)
- `max_completion_tokens: int | None` — maximum tokens in response
- `stream: bool` — streaming output
- `auto_model: bool` — automatic model selection based on complexity
- `resolve_snapshot: bool` — resolve alias to snapshot (e.g., `gpt-4o` → `gpt-4o-2024-11-20`)
- `return_raw: bool` — return raw response object instead of string
- `config: OpenAIConfig | None` — configuration object
- `config_path: str | Path | None` — path to YAML config
- `client: OpenAIClient | None` — reuse existing client
- `**kwargs: Any` — additional OpenAI API parameters

**Returns:**
- `str` — if `stream=False` and `return_raw=False`
- `Iterator[str]` — if `stream=True` and `return_raw=False`
- `Any` — if `return_raw=True` (raw response object)

**Examples:**

```python
# Simple call
result = call_openai("Explain Python decorators")

# With system prompt
result = call_openai(
    prompt="Write a function",
    system_prompt="You are a Python expert",
    model="gpt-4o-mini",
)

# Streaming
for chunk in call_openai("Tell a story", stream=True):
    print(chunk, end="")

# With messages
from openai_client_module import Message
messages = [
    Message(role="system", content="You are an assistant"),
    Message(role="user", content="The Ultimate Question of Life, the Universe, and Everything?"),
]
result = call_openai(messages=messages)

# Client reuse
with openai_client() as client:
    r1 = call_openai("Prompt 1", client=client)
    r2 = call_openai("Prompt 2", client=client)
```

### call_openai_structured()

**Purpose:** Call with JSON structured outputs.

**Signature:**

```python
def call_openai_structured(
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
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
) -> dict | list | str
```

**Parameters:**
- All parameters from `call_openai()`, except `stream` and `return_raw`
- `response_format: dict[str, Any]` — response format (required)
  - `{"type": "json_object"}` — simple JSON
  - `{"type": "json_schema", "json_schema": {...}}` — JSON Schema
- `parse: bool` — parse JSON (`True` → `dict|list`, `False` → `str`)

**Returns:**
- `dict | list` — if `parse=True` (parsed JSON)
- `str` — if `parse=False` (JSON string)

**Examples:**

```python
# Simple JSON
obj = call_openai_structured(
    prompt="Return JSON with name and age fields",
    response_format={"type": "json_object"},
    parse=True,
)
# obj: {"name": "...", "age": ...}

# JSON Schema
schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "person",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        },
    },
}
obj = call_openai_structured(
    prompt="Create a person object",
    response_format=schema,
    parse=True,
)

# Without parsing (get string)
json_str = call_openai_structured(
    prompt="Return JSON",
    response_format={"type": "json_object"},
    parse=False,
)
```

### call_openai_web_search()

**Purpose:** Web Search via Responses API.

**Signature:**

```python
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
) -> str | Any
```

**Parameters:**
- All parameters from `call_openai()`, except `stream`
- `tool_choice: ToolChoice | str` — search strategy (`"auto"`, `"required"`, `"none"`)
- `search_context_size: SearchContextSize | str` — context size (`"low"`, `"medium"`, `"high"`)
- `user_location: dict[str, Any] | None` — user location for personalization
- `include_sources: bool` — include sources in response
- `return_raw: bool` — return raw object (for `extract_web_sources`)

**Returns:**
- `str` — if `return_raw=False` (response text)
- `Any` — if `return_raw=True` (raw object for source extraction)

**Examples:**

```python
# Simple search
result = call_openai_web_search("Latest news about Python 3.13")

# Forced search
result = call_openai_web_search(
    "What's the weather in Moscow?",
    tool_choice="required",
    search_context_size="high",
)

# With source extraction
from openai_client_module import extract_web_sources, extract_url_citations

raw = call_openai_web_search(
    "Python news",
    return_raw=True,
    include_sources=True,
)
sources = extract_web_sources(raw)
citations = extract_url_citations(raw)
```

### call_openai_markdown()

**Purpose:** Call with Markdown formatting.

**Signature:**

```python
def call_openai_markdown(
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    auto_model: bool = False,
    resolve_snapshot: bool = False,
    config: OpenAIConfig | None = None,
    config_path: str | Path | None = None,
    client: OpenAIClient | None = None,
    **kwargs: Any,
) -> str
```

**Parameters:** Similar to `call_openai()`, but without `stream` and `return_raw`.

**Limitation:** If `response_format` is present in `kwargs`, raises `ValueError`.

**Returns:** `str` (Markdown text)

**Example:**

```python
markdown = call_openai_markdown("Write an article about Python")
```

---

## Data Types

### Message

```python
from typing import TypedDict

class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
```

**Constraints:**
- `role` — only `"system"`, `"user"`, `"assistant"`
- `content` — strictly `str` (not list, not dict)

### ModelInfo

```python
from dataclasses import dataclass

@dataclass
class ModelInfo:
    alias: str  # e.g., "gpt-4o"
    snapshot: str  # e.g., "gpt-4o-2024-11-20"
    category: ModelCategory  # standard, search, deep_research
    input_cost_per_m: float  # $ per 1M input tokens
    output_cost_per_m: float  # $ per 1M output tokens
```

### RoutingConfig

```python
@dataclass
class RoutingConfig:
    default_cheap: str = "gpt-4o-mini"
    default_capable: str = "gpt-4.1-mini"
    token_threshold: int = 50000
    instruction_chars_threshold: int = 2000
    high_risk_keywords: list[str] = field(default_factory=lambda: [...])
    force_capable_on_strict_schema: bool = True
```

### OpenAIConfig

```python
@dataclass
class OpenAIConfig:
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_completion_tokens: int | None = None
    timeout: float = 60.0
    base_url: str | None = None
    organization: str | None = None
    max_retries: int = 3
    retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    api_key: str | None = None  # Ignored, taken from env
```

### Enums

```python
class ToolChoice(StrEnum):
    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"

class SearchContextSize(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ModelCategory(StrEnum):
    STANDARD = "standard"
    SEARCH = "search"
    DEEP_RESEARCH = "deep_research"
```

---

## Model Registries

### MODELS_REGISTRY

```python
MODELS_REGISTRY: dict[str, ModelInfo]
```

Complete registry of all models: `{alias: ModelInfo}`.

**Example:**

```python
from openai_client_module import MODELS_REGISTRY

info = MODELS_REGISTRY["gpt-4o"]
print(info.snapshot)  # "gpt-4o-2024-11-20"
print(info.category)  # ModelCategory.STANDARD
print(info.input_cost_per_m)  # 2.5
```

### MODELS_ALL

```python
MODELS_ALL: dict[str, str]  # {alias: snapshot}
```

Backward compatibility: `{"gpt-4o": "gpt-4o-2024-11-20", ...}`

### MODELS_STANDARD, MODELS_SEARCH, MODELS_DEEP_RESEARCH

```python
MODELS_STANDARD: dict[str, str]  # Standard models
MODELS_SEARCH: dict[str, str]  # Search models
MODELS_DEEP_RESEARCH: dict[str, str]  # Deep Research models
```

---

## Model Routing

### choose_model()

**Signature:**

```python
def choose_model(
    system_prompt: str = "",
    user_prompt: str = "",
    attachments_text: str = "",
    *,
    strict_schema: bool = False,
    cfg: RoutingConfig = _DEFAULT_ROUTING_CONFIG,
) -> str
```

**Selection Logic:**
1. If `estimate_tokens(...) >= cfg.token_threshold` → `cfg.default_capable`
2. If `strict_schema and cfg.force_capable_on_strict_schema` → `cfg.default_capable`
3. If `keyword_hits >= 2` → `cfg.default_capable`
4. Otherwise → `cfg.default_cheap`

**Returns:** `str` (model alias)

### maybe_escalate()

**Signature:**

```python
def maybe_escalate(
    output_text: str,
    strict_schema: bool = False,
    cfg: RoutingConfig = _DEFAULT_ROUTING_CONFIG,
) -> str | None
```

**Logic:** If JSON is invalid and `strict_schema=True`, returns `cfg.default_capable`, otherwise `None`.

### estimate_tokens()

**Signature:**

```python
def estimate_tokens(text: str) -> int
```

**Formula:** `len(text) // 4` (rough estimate: ~4 characters = 1 token)

---

## Helpers

### parse_json()

**Signature:**

```python
def parse_json(text: str, *, max_error_snippet: int = 400) -> Any
```

**Purpose:** JSON parsing with error diagnostics.

**Raises:** `ValueError` if JSON is invalid (with text snippet in message).

### extract_web_sources()

**Signature:**

```python
def extract_web_sources(response: Any) -> list[dict[str, Any]]
```

**Purpose:** Extract sources from Web Search response.

**Returns:** `list[dict]` with fields `title`, `url`, etc.

### extract_url_citations()

**Signature:**

```python
def extract_url_citations(response: Any) -> list[dict[str, Any]]
```

**Purpose:** Extract URL citations from response.

---

## OpenAIClient Class

### Initialization

```python
client = OpenAIClient(
    config: OpenAIConfig | None = None,
    config_path: str | Path | None = None,
)
```

### Methods

```python
# Chat Completions
def call(
    self,
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    model: str | None = None,
    stream: bool = False,
    auto_model: bool = False,
    resolve_snapshot: bool = False,
    return_raw: bool = False,
    **kwargs: Any,
) -> str | Iterator[str] | Any

# Structured Outputs
def call_structured(
    self,
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    response_format: dict[str, Any],
    parse: bool = True,
    model: str | None = None,
    auto_model: bool = False,
    resolve_snapshot: bool = False,
    **kwargs: Any,
) -> dict | list | str

# Web Search
def call_web_search(
    self,
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    model: str | None = None,
    tool_choice: str = "auto",
    search_context_size: str = "medium",
    user_location: dict[str, Any] | None = None,
    include_sources: bool = False,
    return_raw: bool = False,
    resolve_snapshot: bool = False,
    **kwargs: Any,
) -> str | Any

# Close client
def close(self) -> None
```

### Context Manager

```python
from openai_client_module import openai_client

with openai_client() as client:
    result1 = client.call("Prompt 1")
    result2 = client.call("Prompt 2")
    # Client is automatically closed
```

---

## Configuration

### Settings Priority

1. Function parameters (highest)
2. YAML configuration (`openai_config.yaml`)
3. Environment variables
4. Default values (lowest)

### Configuration File Search

```python
import os
from pathlib import Path

# Search order:
config_paths = [
    os.getenv("OPENAI_CONFIG_PATH"),  # 1. Environment variable
    Path.cwd() / "openai_config.yaml",  # 2. Current directory
    Path(__file__).parent / "openai_config.yaml",  # 3. Module directory
]
```

### YAML Format

```yaml
model: "gpt-4o"
temperature: 0.7
max_completion_tokens: null
timeout: 60.0
max_retries: 3
retry_delay: 1.0
max_retry_delay: 60.0

web_search:
  tool_choice: "auto"
  search_context_size: "medium"
```

### Loading Configuration

```python
from openai_client_module import OpenAIConfig

# From YAML
config = OpenAIConfig.from_yaml("path/to/config.yaml")

# Programmatically
config = OpenAIConfig(
    model="gpt-4o-mini",
    temperature=0.5,
    max_retries=5,
)
```

---

## Error Handling

### Exception Types

```python
from openai import (
    RateLimitError,
    APIConnectionError,
    APIError,
    APITimeoutError,
    APIStatusError,
)
```

### Retry Logic

The module automatically retries the following errors:
- `RateLimitError` — rate limit exceeded
- `APIConnectionError` — network issues
- `APITimeoutError` — request timeout
- `APIStatusError` — if `status_code in {429, 500, 502, 503, 504}`

**Retry parameters:**
- `max_retries` — number of attempts (default 3)
- `retry_delay` — base delay (default 1.0 sec)
- `max_retry_delay` — upper delay limit (default 60.0 sec)

**Delay formula:** `min(retry_delay * (2 ** attempt) + jitter, max_retry_delay)`

### Parameter Validation

The module validates input parameters:

```python
# Forbidden parameters in text-only mode:
_FORBIDDEN_TEXT_KWARGS = {
    "tools", "tool_choice", "parallel_tool_calls",
    "prompt_cache_key", "prompt_cache_retention",
}

# Allowed roles:
_ALLOWED_ROLES = {"system", "user", "assistant"}
```

**Raises:** `ValueError` on constraint violation.

---

## Limitations

| Limitation | Where checked | Exception |
|------------|---------------|-----------|
| Tool-calling forbidden | `_validate_text_only()` | `ValueError` |
| `response_format` incompatible with markdown | `call_openai_markdown()` | `ValueError` |
| Invalid role in `Message` | `_validate_text_only()` | `ValueError` |
| `content` not `str` in `Message` | `_validate_text_only()` | `ValueError` |
| `max_retries: 0` in SDK | `_execute_with_retry()` | Uses own retry logic |

---

## Dependency Graph

```
openai_types.py     ← no dependencies
       ↓
openai_config.py    ← no dependencies
       ↓
openai_client.py    ← types, config, openai SDK
       ↓
openai_api.py       ← types, config, client
```

**No circular dependencies.**

---

## Usage Examples for Agents

> **AGENT INSTRUCTIONS:** Use the examples below as templates for code generation.  
> All examples are verified and work when conditions from "Agent Instructions: Module Connection" are met.

### Minimal Working Example

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from openai_client_module import call_openai
result = call_openai("The Ultimate Question of Life, the Universe, and Everything?")
print(result)
```

### Batch Operations

```python
from openai_client_module import openai_client

prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
results = []

with openai_client() as client:
    for prompt in prompts:
        result = client.call(prompt)
        results.append(result)
```

### Structured Outputs with Error Handling

```python
from openai_client_module import call_openai_structured, parse_json

try:
    obj = call_openai_structured(
        prompt="Return JSON",
        response_format={"type": "json_object"},
        parse=True,
    )
except ValueError as e:
    # JSON parsing error
    print(f"Error: {e}")
```

### Web Search with Sources

```python
from openai_client_module import (
    call_openai_web_search,
    extract_web_sources,
    extract_url_citations,
)

raw = call_openai_web_search(
    "Python news",
    return_raw=True,
    include_sources=True,
)

sources = extract_web_sources(raw)
citations = extract_url_citations(raw)

for source in sources:
    print(f"{source.get('title')}: {source.get('url')}")
```

---

## Exported Components

Full list available in `__init__.py`:

```python
__all__ = [
    # Types
    "ToolChoice", "SearchContextSize", "ModelCategory",
    "Message", "ModelInfo",
    # Registries
    "MODELS_REGISTRY", "MODELS_ALL", "MODELS_STANDARD",
    "MODELS_SEARCH", "MODELS_DEEP_RESEARCH",
    # Routing
    "RoutingConfig", "choose_model", "maybe_escalate", "estimate_tokens",
    # Configuration
    "WebSearchConfig", "OpenAIConfig",
    # Client
    "OpenAIClient", "openai_client",
    # Functional API
    "call_openai", "call_openai_structured",
    "call_openai_web_search", "call_openai_markdown",
    # Helpers
    "extract_web_sources", "extract_url_citations", "parse_json",
]
```

---

## See Also

- `openai_config.yaml` — configuration example
- `README.md` — usage examples for users
