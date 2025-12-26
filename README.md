# OpenAI Client Module

**Universal module for working with OpenAI API.**

> **For AI agents:** Start with the file [`00_AGENT_INSTRUCTIONS.md`](./00_AGENT_INSTRUCTIONS.md)

## Main Features

- **Chat Completions API**: standard text requests
- **Web Search**: internet search via Responses API
- **Structured Outputs**: JSON with schema validation
- **Automatic Model Selection**: intelligent routing based on request complexity
- **Batch Operations**: HTTP client reuse for performance

## Module Structure
The module is split into 4 files for better maintainability:

- `openai_types.py` — types, model registries, routing, Web Search helpers
- `openai_config.py` — configuration (YAML, dataclasses), `parse_json()`
- `openai_client.py` — `OpenAIClient` class, `openai_client()` context manager
- `openai_api.py` — functional API (`call_openai_*`), CLI
- `__init__.py` — public API re-export

## Installation

The module doesn't require installation, just copy the `openai_client_module` folder to your project.

## Quick Start

```python
from openai_client_module import call_openai

# Simple call
result = call_openai("The Ultimate Question of Life, the Universe, and Everything?")
print(result)
```

## Usage Examples

### Standard Call

```python
from openai_client_module import call_openai

response = call_openai("Explain quantum mechanics")
```

### Web Search

```python
from openai_client_module import call_openai_web_search

result = call_openai_web_search("What's the weather in New York today?")
```

### Structured Outputs (JSON)

```python
from openai_client_module import call_openai_structured

obj = call_openai_structured(
    prompt="Return JSON with name and age fields",
    response_format={"type": "json_object"},
)
```

### Batch Operations (Performance)

```python
from openai_client_module import openai_client

# HTTP client reuse for multiple requests
with openai_client() as client:
    for prompt in prompts:
        result = client.call(prompt)
```

### Automatic Model Selection

```python
from openai_client_module import call_openai, RoutingConfig, choose_model

# Automatic model selection based on complexity
result = call_openai("Complex request", auto_model=True)

# Manual model selection
model = choose_model(
    system_prompt="You are a Python expert",
    user_prompt="Explain decorators",
    strict_schema=False,
)
result = call_openai("Explain decorators", model=model)

# Custom routing configuration
custom_cfg = RoutingConfig(
    default_cheap="gpt-4o-mini",
    default_capable="gpt-4.1-mini",
    token_threshold=50000,
)
```

### Working with Web Search Sources

```python
from openai_client_module import call_openai_web_search, extract_web_sources, extract_url_citations

# Get response with sources
raw_response = call_openai_web_search(
    "Latest news about Python 3.13",
    return_raw=True,
    include_sources=True,
)

# Extract sources and citations
sources = extract_web_sources(raw_response)
citations = extract_url_citations(raw_response)

for source in sources:
    print(f"{source.get('title')}: {source.get('url')}")
```

### Structured Outputs with Validation

```python
from openai_client_module import call_openai_structured, parse_json

# Automatic JSON parsing
obj = call_openai_structured(
    prompt="Return JSON with name and age fields",
    response_format={"type": "json_object"},
    parse=True,  # Returns dict
)

# Manual parsing with error diagnostics
json_str = call_openai_structured(
    prompt="Return JSON",
    response_format={"type": "json_object"},
    parse=False,  # Returns string
)
try:
    obj = parse_json(json_str)
except ValueError as e:
    print(f"Parsing error: {e}")
```

## Configuration

The module looks for `openai_config.yaml` in the following locations:
1. Path from `OPENAI_CONFIG_PATH` environment variable
2. Current directory
3. Module directory

Configuration example:

```yaml
model: "gpt-4o"
temperature: 0.7
max_completion_tokens: 2000
timeout: 60.0
max_retries: 3
retry_delay: 1.0
max_retry_delay: 60.0

web_search:
  tool_choice: "auto"
  search_context_size: "medium"
```

**Important**: API key is taken **ONLY** from the `OPENAI_API_KEY` environment variable. Never store keys in configuration files!

## Documentation

- **Full documentation:** `OPENAI_CLIENT_DOC.md` — architecture, API, limitations
- **Docstrings:** available via `help()`:

```python
from openai_client_module import call_openai, OpenAIClient, RoutingConfig
help(call_openai)
help(OpenAIClient.call)
help(RoutingConfig)
```

## Supported Models

### Standard Models (Chat Completions API)
- `gpt-4o`, `gpt-4o-mini` — GPT-4o family
- `gpt-4.1`, `gpt-4.1-mini` — GPT-4.1 family
- `o3-mini`, `o4-mini` — O-series

### Search Models (Chat Completions + web search)
- `gpt-4o-search-preview`, `gpt-4o-mini-search-preview` — available without org verification
- `gpt-5-search-api` — may require org verification

### Deep Research Models (Responses API)
- `o3-deep-research`, `o4-mini-deep-research` — long-running agentic search

### Model Registries

```python
from openai_client_module import (
    MODELS_REGISTRY,      # {alias: ModelInfo} — full information
    MODELS_ALL,           # {alias: snapshot} — backward compatibility
    MODELS_STANDARD,      # Standard models
    MODELS_SEARCH,        # Search models
    MODELS_DEEP_RESEARCH, # Deep Research models
)

# Get model information
info = MODELS_REGISTRY["gpt-4o"]
print(f"Snapshot: {info.snapshot}, Category: {info.category}")
print(f"Cost: ${info.input_cost_per_m}/1M input, ${info.output_cost_per_m}/1M output")
```

## Module Structure

```
openai_client_module/
├── __init__.py           # Public API re-export
├── openai_types.py       # Types, registries, routing
├── openai_config.py      # Configuration (YAML, dataclasses)
├── openai_client.py      # OpenAIClient class
├── openai_api.py         # Functional API + CLI
├── openai_config.yaml    # Default configuration
├── OPENAI_CLIENT_DOC.md  # Full documentation
└── README.md             # This file
```

**Dependency Graph:**
```
openai_types.py     ← no dependencies
       ↓
openai_config.py    ← no dependencies
       ↓
openai_client.py    ← types, config, openai SDK
       ↓
openai_api.py       ← types, config, client
```

## Public API

### Functional API
- `call_openai()` — basic call
- `call_openai_structured()` — JSON structured outputs
- `call_openai_web_search()` — Web Search via Responses API
- `call_openai_markdown()` — Markdown output

### Class and Context Manager
- `OpenAIClient` — class for batch operations
- `openai_client()` — context manager

### Model Routing
- `choose_model()` — automatic model selection
- `maybe_escalate()` — escalation on invalid JSON
- `estimate_tokens()` — token estimation

### Helpers
- `parse_json()` — JSON parsing with diagnostics
- `extract_web_sources()` — extract sources from web search
- `extract_url_citations()` — extract URL citations

See `__init__.py` or `OPENAI_CLIENT_DOC.md` for the full list of exported components.

## License

## Version
**Version:** 1.0.0 | **Python:** 3.10+ | **Dependencies:** openai, pyyaml
