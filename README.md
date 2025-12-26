# OpenAI Client Module

**Универсальный модуль для работы с OpenAI API.**

> **Для AI агентов:** Начните с файла [`00_AGENT_INSTRUCTIONS.md`](./00_AGENT_INSTRUCTIONS.md)

## Основные возможности

- **Chat Completions API**: стандартные текстовые запросы
- **Web Search**: поиск в интернете через Responses API
- **Structured Outputs**: JSON с валидацией схемы
- **Автоматический выбор модели**: интеллектуальный роутинг по сложности запроса
- **Batch операции**: переиспользование HTTP клиента для производительности

## Структура модуля
Модуль разделён на 4 файла для лучшей поддерживаемости:

- `openai_types.py` — типы, справочники моделей, роутинг, Web Search helpers
- `openai_config.py` — конфигурация (YAML, dataclasses), `parse_json()`
- `openai_client.py` — класс `OpenAIClient`, контекстный менеджер `openai_client()`
- `openai_api.py` — функциональный API (`call_openai_*`), CLI
- `__init__.py` — реэкспорт публичного API

## Установка

Модуль не требует установки, достаточно скопировать папку `openai_client_module` в ваш проект.

## Быстрый старт

```python
from openai_client_module import call_openai

# Простой вызов
result = call_openai("The Ultimate Question of Life, the Universe, and Everything?")
print(result)
```

## Примеры использования

### Стандартный вызов

```python
from openai_client_module import call_openai

response = call_openai("Объясни квантовую механику")
```

### Web Search

```python
from openai_client_module import call_openai_web_search

result = call_openai_web_search("Какая погода в Москве сегодня?")
```

### Structured Outputs (JSON)

```python
from openai_client_module import call_openai_structured

obj = call_openai_structured(
    prompt="Верни JSON с полями name и age",
    response_format={"type": "json_object"},
)
```

### Batch операции (производительность)

```python
from openai_client_module import openai_client

# Переиспользование HTTP клиента для множества запросов
with openai_client() as client:
    for prompt in prompts:
        result = client.call(prompt)
```

### Автоматический выбор модели

```python
from openai_client_module import call_openai, RoutingConfig, choose_model

# Автоматический выбор модели по сложности
result = call_openai("Сложный запрос", auto_model=True)

# Ручной выбор модели
model = choose_model(
    system_prompt="Ты эксперт по Python",
    user_prompt="Объясни декораторы",
    strict_schema=False,
)
result = call_openai("Объясни декораторы", model=model)

# Кастомная конфигурация роутинга
custom_cfg = RoutingConfig(
    default_cheap="gpt-4o-mini",
    default_capable="gpt-4.1-mini",
    token_threshold=50000,
)
```

### Работа с Web Search источниками

```python
from openai_client_module import call_openai_web_search, extract_web_sources, extract_url_citations

# Получить ответ с источниками
raw_response = call_openai_web_search(
    "Последние новости о Python 3.13",
    return_raw=True,
    include_sources=True,
)

# Извлечь источники и цитаты
sources = extract_web_sources(raw_response)
citations = extract_url_citations(raw_response)

for source in sources:
    print(f"{source.get('title')}: {source.get('url')}")
```

### Structured Outputs с валидацией

```python
from openai_client_module import call_openai_structured, parse_json

# Автоматический парсинг JSON
obj = call_openai_structured(
    prompt="Верни JSON с полями name и age",
    response_format={"type": "json_object"},
    parse=True,  # Возвращает dict
)

# Ручной парсинг с диагностикой ошибок
json_str = call_openai_structured(
    prompt="Верни JSON",
    response_format={"type": "json_object"},
    parse=False,  # Возвращает строку
)
try:
    obj = parse_json(json_str)
except ValueError as e:
    print(f"Ошибка парсинга: {e}")
```

## Конфигурация

Модуль ищет файл `openai_config.yaml` в следующих местах:
1. Путь из переменной окружения `OPENAI_CONFIG_PATH`
2. Текущая директория
3. Директория модуля

Пример конфигурации:

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

**Важно**: API ключ берётся **ТОЛЬКО** из переменной окружения `OPENAI_API_KEY`. Никогда не храните ключи в конфигурационных файлах!

## Документация

- **Полная документация:** `OPENAI_CLIENT_DOC.md` — архитектура, API, ограничения
- **Docstrings:** доступны через `help()`:

```python
from openai_client_module import call_openai, OpenAIClient, RoutingConfig
help(call_openai)
help(OpenAIClient.call)
help(RoutingConfig)
```

## Поддерживаемые модели

### Стандартные модели (Chat Completions API)
- `gpt-4o`, `gpt-4o-mini` — GPT-4o семейство
- `gpt-4.1`, `gpt-4.1-mini` — GPT-4.1 семейство
- `o3-mini`, `o4-mini` — O-серия

### Search-модели (Chat Completions + web search)
- `gpt-4o-search-preview`, `gpt-4o-mini-search-preview` — доступны без верификации org
- `gpt-5-search-api` — может требовать верификации org

### Deep Research модели (Responses API)
- `o3-deep-research`, `o4-mini-deep-research` — длительный agentic поиск

### Справочники моделей

```python
from openai_client_module import (
    MODELS_REGISTRY,      # {alias: ModelInfo} — полная информация
    MODELS_ALL,           # {alias: snapshot} — обратная совместимость
    MODELS_STANDARD,      # Стандартные модели
    MODELS_SEARCH,        # Search-модели
    MODELS_DEEP_RESEARCH, # Deep Research модели
)

# Получить информацию о модели
info = MODELS_REGISTRY["gpt-4o"]
print(f"Snapshot: {info.snapshot}, Category: {info.category}")
print(f"Cost: ${info.input_cost_per_m}/1M input, ${info.output_cost_per_m}/1M output")
```

## Структура модуля

```
openai_client_module/
├── __init__.py           # Реэкспорт публичного API
├── openai_types.py       # Типы, справочники, роутинг
├── openai_config.py      # Конфигурация (YAML, dataclasses)
├── openai_client.py      # Класс OpenAIClient
├── openai_api.py         # Функциональный API + CLI
├── openai_config.yaml    # Конфигурация по умолчанию
├── OPENAI_CLIENT_DOC.md  # Полная документация
└── README.md             # Этот файл
```

**Граф зависимостей:**
```
openai_types.py     ← нет зависимостей
       ↓
openai_config.py    ← нет зависимостей
       ↓
openai_client.py    ← types, config, openai SDK
       ↓
openai_api.py       ← types, config, client
```

## Публичный API

### Функциональный API
- `call_openai()` — базовый вызов
- `call_openai_structured()` — JSON structured outputs
- `call_openai_web_search()` — Web Search через Responses API
- `call_openai_markdown()` — вывод в Markdown

### Класс и контекстный менеджер
- `OpenAIClient` — класс для batch операций
- `openai_client()` — контекстный менеджер

### Роутинг моделей
- `choose_model()` — автоматический выбор модели
- `maybe_escalate()` — эскалация при невалидном JSON
- `estimate_tokens()` — оценка токенов

### Хелперы
- `parse_json()` — парсинг JSON с диагностикой
- `extract_web_sources()` — извлечение источников из web search
- `extract_url_citations()` — извлечение цитат с URL

Полный список экспортируемых компонентов см. в `__init__.py` или `OPENAI_CLIENT_DOC.md`.

## Лицензия

##Версия
**Версия:** 1.0.0 | **Python:** 3.10+ | **Зависимости:** openai, pyyaml