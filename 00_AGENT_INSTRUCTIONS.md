# IMPORTANT FOR AGENTS: Start Here

> **THIS FILE IS THE FIRST ENTRY POINT FOR AI AGENTS**  
> **READ THIS FILE FIRST BEFORE WORKING WITH THE MODULE**

## For Agents: How to Use This Module

1. **Read this file completely** — all necessary information for connecting and using the module is here
2. **Don't analyze the source code** — all function signatures, data types, and examples are already described here
3. **Follow the step-by-step instructions** — from dependency checking to usage examples

## Full Documentation

**Main documentation file:** [`OPENAI_CLIENT_DOC.md`](./OPENAI_CLIENT_DOC.md)

This file contains:
- ✅ Step-by-step instructions for module connection
- ✅ Complete signatures of all public functions
- ✅ Description of all data types
- ✅ Usage examples for each scenario
- ✅ Error handling information
- ✅ Model registries and configuration

## Quick Start for Agents

```python
# 1. Check dependencies
import sys
if sys.version_info < (3, 10):
    raise RuntimeError("Python 3.10+ required")

# 2. Install dependencies (if necessary)
# pip install openai pyyaml

# 3. Import module
from openai_client_module import call_openai

# 4. Check API key
import os
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not set")

# 5. Use
result = call_openai("Hello!")
print(result)
```

## Module Structure

```
openai_client_module/
├── 00_AGENT_INSTRUCTIONS.md  ← YOU ARE HERE (read first)
├── OPENAI_CLIENT_DOC.md      ← Full documentation (read second)
├── README.md                  ← For users (optional)
├── __init__.py                ← API re-export
├── openai_types.py           ← Types and registries
├── openai_config.py          ← Configuration
├── openai_client.py          ← OpenAIClient class
└── openai_api.py             ← Functional API
```

## What's Next?

1. Open [`OPENAI_CLIENT_DOC.md`](./OPENAI_CLIENT_DOC.md) for complete information
2. Use the "Public API: Function Signatures" section to generate calls
3. See examples in the "Usage Examples for Agents" section

---

**Module version:** 1.0.0 | **Python:** 3.10+ | **Dependencies:** openai, pyyaml
