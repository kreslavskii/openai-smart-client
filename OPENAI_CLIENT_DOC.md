# openai_client_module — Техническая документация для агентов

> **⚠️ ВАЖНО:** Если вы агент и только что открыли кодовую базу, начните с файла [`00_AGENT_INSTRUCTIONS.md`](./00_AGENT_INSTRUCTIONS.md)  
> **ЦЕЛЕВАЯ АУДИТОРИЯ:** AI агенты, автоматизированные системы, кодогенераторы  
> **НАЗНАЧЕНИЕ:** Машиночитаемая инструкция для подключения и использования модуля без анализа кодовой базы  
> **ФОРМАТ:** Структурированные сигнатуры функций, примеры кода, технические детали

**Версия:** 1.0.0 | **Python:** 3.10+ | **Зависимости:** openai, pyyaml

**Структура модуля:**
- `openai_types.py` (425 строк) — типы, справочники моделей, роутинг, Web Search helpers
- `openai_config.py` (194 строки) — конфигурация (YAML, dataclasses), `parse_json()`
- `openai_client.py` (645 строк) — класс `OpenAIClient`, контекстный менеджер `openai_client()`
- `openai_api.py` (506 строк) — функциональный API (`call_openai_*`), CLI
- `__init__.py` (91 строка) — реэкспорт публичного API

**Всего:** ~1861 строка

---

## Инструкция для агента: Подключение модуля

**Алгоритм действий для агента:**

1. Проверить зависимости (Python 3.10+, openai, pyyaml)
2. Установить зависимости при необходимости (`pip install openai pyyaml`)
3. Импортировать необходимые компоненты из `openai_client_module`
4. Убедиться, что установлена переменная окружения `OPENAI_API_KEY`
5. Использовать функции согласно сигнатурам ниже

### Шаг 1: Проверка зависимостей

```python
import sys
if sys.version_info < (3, 10):
    raise RuntimeError("Требуется Python 3.10+")

try:
    import openai
    import yaml
except ImportError as e:
    raise ImportError(f"Отсутствует зависимость: {e}")
```

### Шаг 2: Установка зависимостей (если необходимо)

```bash
pip install openai pyyaml
```

### Шаг 3: Импорт модуля

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

### Шаг 4: Настройка API ключа

**ОБЯЗАТЕЛЬНО:** Переменная окружения `OPENAI_API_KEY` должна быть установлена:

```python
import os
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY не установлен")
```

**ПРИОРИТЕТ:** API ключ берётся ТОЛЬКО из `OPENAI_API_KEY`. Игнорируется любое значение в YAML.

### Шаг 5: Конфигурация (опционально)

Модуль ищет `openai_config.yaml` в порядке:
1. `os.getenv("OPENAI_CONFIG_PATH")`
2. Текущая рабочая директория
3. `openai_client_module/`

Если файл не найден, используются дефолты (см. раздел "Конфигурация").

---

## Публичный API: Сигнатуры функций

> **ВАЖНО ДЛЯ АГЕНТА:** Все функции ниже доступны через `from openai_client_module import ...`  
> Сигнатуры соответствуют реальному коду. Используйте их для генерации вызовов.

### call_openai()

**Назначение:** Базовый вызов Chat Completions API.

**Сигнатуры (overload):**

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

# Non-streaming: stream=False или не указан
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

**Параметры:**
- `prompt: str | None` — пользовательский промпт (если `messages=None`)
- `system_prompt: str | None` — системный промпт (если `messages=None`)
- `messages: list[Message] | None` — список сообщений (альтернатива `prompt`/`system_prompt`)
- `model: str | None` — модель (если `None`, берётся из конфига или `gpt-4o`)
- `temperature: float | None` — температура (0-2)
- `max_completion_tokens: int | None` — максимум токенов в ответе
- `stream: bool` — потоковый вывод
- `auto_model: bool` — автоматический выбор модели по сложности
- `resolve_snapshot: bool` — разрешить alias в snapshot (например, `gpt-4o` → `gpt-4o-2024-11-20`)
- `return_raw: bool` — вернуть сырой объект ответа вместо строки
- `config: OpenAIConfig | None` — объект конфигурации
- `config_path: str | Path | None` — путь к YAML конфигу
- `client: OpenAIClient | None` — переиспользовать существующий клиент
- `**kwargs: Any` — дополнительные параметры OpenAI API

**Возврат:**
- `str` — если `stream=False` и `return_raw=False`
- `Iterator[str]` — если `stream=True` и `return_raw=False`
- `Any` — если `return_raw=True` (сырой объект ответа)

**Примеры:**

```python
# Простой вызов
result = call_openai("Объясни Python декораторы")

# С системным промптом
result = call_openai(
    prompt="Напиши функцию",
    system_prompt="Ты эксперт по Python",
    model="gpt-4o-mini",
)

# Streaming
for chunk in call_openai("Расскажи историю", stream=True):
    print(chunk, end="")

# С messages
from openai_client_module import Message
messages = [
    Message(role="system", content="Ты помощник"),
    Message(role="user", content="Привет"),
]
result = call_openai(messages=messages)

# Переиспользование клиента
with openai_client() as client:
    r1 = call_openai("Промпт 1", client=client)
    r2 = call_openai("Промпт 2", client=client)
```

### call_openai_structured()

**Назначение:** Вызов с JSON structured outputs.

**Сигнатура:**

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

**Параметры:**
- Все параметры из `call_openai()`, кроме `stream` и `return_raw`
- `response_format: dict[str, Any]` — формат ответа (обязателен)
  - `{"type": "json_object"}` — простой JSON
  - `{"type": "json_schema", "json_schema": {...}}` — JSON Schema
- `parse: bool` — парсить JSON (`True` → `dict|list`, `False` → `str`)

**Возврат:**
- `dict | list` — если `parse=True` (распарсенный JSON)
- `str` — если `parse=False` (JSON строка)

**Примеры:**

```python
# Простой JSON
obj = call_openai_structured(
    prompt="Верни JSON с полями name и age",
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
    prompt="Создай объект person",
    response_format=schema,
    parse=True,
)

# Без парсинга (получить строку)
json_str = call_openai_structured(
    prompt="Верни JSON",
    response_format={"type": "json_object"},
    parse=False,
)
```

### call_openai_web_search()

**Назначение:** Web Search через Responses API.

**Сигнатура:**

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

**Параметры:**
- Все параметры из `call_openai()`, кроме `stream`
- `tool_choice: ToolChoice | str` — стратегия поиска (`"auto"`, `"required"`, `"none"`)
- `search_context_size: SearchContextSize | str` — размер контекста (`"low"`, `"medium"`, `"high"`)
- `user_location: dict[str, Any] | None` — локация пользователя для персонализации
- `include_sources: bool` — включить источники в ответ
- `return_raw: bool` — вернуть сырой объект (для `extract_web_sources`)

**Возврат:**
- `str` — если `return_raw=False` (текст ответа)
- `Any` — если `return_raw=True` (сырой объект для извлечения источников)

**Примеры:**

```python
# Простой поиск
result = call_openai_web_search("Последние новости о Python 3.13")

# Принудительный поиск
result = call_openai_web_search(
    "Какая погода в Москве?",
    tool_choice="required",
    search_context_size="high",
)

# С извлечением источников
from openai_client_module import extract_web_sources, extract_url_citations

raw = call_openai_web_search(
    "Новости о Python",
    return_raw=True,
    include_sources=True,
)
sources = extract_web_sources(raw)
citations = extract_url_citations(raw)
```

### call_openai_markdown()

**Назначение:** Вызов с форматированием в Markdown.

**Сигнатура:**

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

**Параметры:** Аналогично `call_openai()`, но без `stream` и `return_raw`.

**Ограничение:** Если в `kwargs` присутствует `response_format`, выбрасывается `ValueError`.

**Возврат:** `str` (Markdown текст)

**Пример:**

```python
markdown = call_openai_markdown("Напиши статью о Python")
```

---

## Типы данных

### Message

```python
from typing import TypedDict

class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
```

**Ограничения:**
- `role` — только `"system"`, `"user"`, `"assistant"`
- `content` — строго `str` (не список, не dict)

### ModelInfo

```python
from dataclasses import dataclass

@dataclass
class ModelInfo:
    alias: str  # Например, "gpt-4o"
    snapshot: str  # Например, "gpt-4o-2024-11-20"
    category: ModelCategory  # standard, search, deep_research
    input_cost_per_m: float  # $ за 1M input токенов
    output_cost_per_m: float  # $ за 1M output токенов
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
    api_key: str | None = None  # Игнорируется, берётся из env
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

## Справочники моделей

### MODELS_REGISTRY

```python
MODELS_REGISTRY: dict[str, ModelInfo]
```

Полный справочник всех моделей: `{alias: ModelInfo}`.

**Пример:**

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

Обратная совместимость: `{"gpt-4o": "gpt-4o-2024-11-20", ...}`

### MODELS_STANDARD, MODELS_SEARCH, MODELS_DEEP_RESEARCH

```python
MODELS_STANDARD: dict[str, str]  # Стандартные модели
MODELS_SEARCH: dict[str, str]  # Search-модели
MODELS_DEEP_RESEARCH: dict[str, str]  # Deep Research модели
```

---

## Роутинг моделей

### choose_model()

**Сигнатура:**

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

**Логика выбора:**
1. Если `estimate_tokens(...) >= cfg.token_threshold` → `cfg.default_capable`
2. Если `strict_schema and cfg.force_capable_on_strict_schema` → `cfg.default_capable`
3. Если `keyword_hits >= 2` → `cfg.default_capable`
4. Иначе → `cfg.default_cheap`

**Возврат:** `str` (alias модели)

### maybe_escalate()

**Сигнатура:**

```python
def maybe_escalate(
    output_text: str,
    strict_schema: bool = False,
    cfg: RoutingConfig = _DEFAULT_ROUTING_CONFIG,
) -> str | None
```

**Логика:** Если JSON невалиден и `strict_schema=True`, возвращает `cfg.default_capable`, иначе `None`.

### estimate_tokens()

**Сигнатура:**

```python
def estimate_tokens(text: str) -> int
```

**Формула:** `len(text) // 4` (грубая оценка: ~4 символа = 1 токен)

---

## Хелперы

### parse_json()

**Сигнатура:**

```python
def parse_json(text: str, *, max_error_snippet: int = 400) -> Any
```

**Назначение:** Парсинг JSON с диагностикой ошибок.

**Raises:** `ValueError` если JSON невалиден (с фрагментом текста в сообщении).

### extract_web_sources()

**Сигнатура:**

```python
def extract_web_sources(response: Any) -> list[dict[str, Any]]
```

**Назначение:** Извлечение источников из ответа Web Search.

**Возврат:** `list[dict]` с полями `title`, `url`, и др.

### extract_url_citations()

**Сигнатура:**

```python
def extract_url_citations(response: Any) -> list[dict[str, Any]]
```

**Назначение:** Извлечение цитат с URL из ответа.

---

## Класс OpenAIClient

### Инициализация

```python
client = OpenAIClient(
    config: OpenAIConfig | None = None,
    config_path: str | Path | None = None,
)
```

### Методы

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

# Закрытие клиента
def close(self) -> None
```

### Контекстный менеджер

```python
from openai_client_module import openai_client

with openai_client() as client:
    result1 = client.call("Промпт 1")
    result2 = client.call("Промпт 2")
    # Клиент автоматически закрывается
```

---

## Конфигурация

### Приоритет настроек

1. Параметры функции (наивысший)
2. YAML конфигурация (`openai_config.yaml`)
3. Переменные окружения
4. Значения по умолчанию (низший)

### Поиск конфигурационного файла

```python
import os
from pathlib import Path

# Порядок поиска:
config_paths = [
    os.getenv("OPENAI_CONFIG_PATH"),  # 1. Переменная окружения
    Path.cwd() / "openai_config.yaml",  # 2. Текущая директория
    Path(__file__).parent / "openai_config.yaml",  # 3. Директория модуля
]
```

### Формат YAML

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

### Загрузка конфигурации

```python
from openai_client_module import OpenAIConfig

# Из YAML
config = OpenAIConfig.from_yaml("path/to/config.yaml")

# Программно
config = OpenAIConfig(
    model="gpt-4o-mini",
    temperature=0.5,
    max_retries=5,
)
```

---

## Обработка ошибок

### Типы исключений

```python
from openai import (
    RateLimitError,
    APIConnectionError,
    APIError,
    APITimeoutError,
    APIStatusError,
)
```

### Retry логика

Модуль автоматически ретраит следующие ошибки:
- `RateLimitError` — лимит запросов
- `APIConnectionError` — проблемы с сетью
- `APITimeoutError` — таймаут
- `APIStatusError` — если `status_code in {429, 500, 502, 503, 504}`

**Параметры retry:**
- `max_retries` — количество попыток (по умолчанию 3)
- `retry_delay` — базовая задержка (по умолчанию 1.0 сек)
- `max_retry_delay` — верхний предел задержки (по умолчанию 60.0 сек)

**Формула задержки:** `min(retry_delay * (2 ** attempt) + jitter, max_retry_delay)`

### Валидация параметров

Модуль валидирует входные параметры:

```python
# Запрещённые параметры в text-only режиме:
_FORBIDDEN_TEXT_KWARGS = {
    "tools", "tool_choice", "parallel_tool_calls",
    "prompt_cache_key", "prompt_cache_retention",
}

# Разрешённые роли:
_ALLOWED_ROLES = {"system", "user", "assistant"}
```

**Raises:** `ValueError` при нарушении ограничений.

---

## Ограничения

| Ограничение | Где проверяется | Исключение |
|-------------|-----------------|------------|
| Tool-calling запрещён | `_validate_text_only()` | `ValueError` |
| `response_format` несовместим с markdown | `call_openai_markdown()` | `ValueError` |
| Невалидная роль в `Message` | `_validate_text_only()` | `ValueError` |
| `content` не `str` в `Message` | `_validate_text_only()` | `ValueError` |
| `max_retries: 0` в SDK | `_execute_with_retry()` | Используется своя retry-логика |

---

## Граф зависимостей

```
openai_types.py     ← нет зависимостей
       ↓
openai_config.py    ← нет зависимостей
       ↓
openai_client.py    ← types, config, openai SDK
       ↓
openai_api.py       ← types, config, client
```

**Циклических зависимостей нет.**

---

## Примеры использования для агентов

> **ИНСТРУКЦИЯ ДЛЯ АГЕНТА:** Используйте примеры ниже как шаблоны для генерации кода.  
> Все примеры проверены и работают при соблюдении условий из раздела "Инструкция для агента: Подключение модуля".

### Минимальный рабочий пример

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from openai_client_module import call_openai
result = call_openai("Привет")
print(result)
```

### Batch операции

```python
from openai_client_module import openai_client

prompts = ["Промпт 1", "Промпт 2", "Промпт 3"]
results = []

with openai_client() as client:
    for prompt in prompts:
        result = client.call(prompt)
        results.append(result)
```

### Structured Outputs с обработкой ошибок

```python
from openai_client_module import call_openai_structured, parse_json

try:
    obj = call_openai_structured(
        prompt="Верни JSON",
        response_format={"type": "json_object"},
        parse=True,
    )
except ValueError as e:
    # Ошибка парсинга JSON
    print(f"Ошибка: {e}")
```

### Web Search с источниками

```python
from openai_client_module import (
    call_openai_web_search,
    extract_web_sources,
    extract_url_citations,
)

raw = call_openai_web_search(
    "Новости о Python",
    return_raw=True,
    include_sources=True,
)

sources = extract_web_sources(raw)
citations = extract_url_citations(raw)

for source in sources:
    print(f"{source.get('title')}: {source.get('url')}")
```

---

## Экспортируемые компоненты

Полный список доступен в `__init__.py`:

```python
__all__ = [
    # Типы
    "ToolChoice", "SearchContextSize", "ModelCategory",
    "Message", "ModelInfo",
    # Справочники
    "MODELS_REGISTRY", "MODELS_ALL", "MODELS_STANDARD",
    "MODELS_SEARCH", "MODELS_DEEP_RESEARCH",
    # Роутинг
    "RoutingConfig", "choose_model", "maybe_escalate", "estimate_tokens",
    # Конфигурация
    "WebSearchConfig", "OpenAIConfig",
    # Клиент
    "OpenAIClient", "openai_client",
    # Функциональный API
    "call_openai", "call_openai_structured",
    "call_openai_web_search", "call_openai_markdown",
    # Helpers
    "extract_web_sources", "extract_url_citations", "parse_json",
]
```

---

## См. также

- `openai_config.yaml` — пример конфигурации
- `README.md` — примеры использования для пользователей
