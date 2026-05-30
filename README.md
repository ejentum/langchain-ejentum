# langchain-ejentum

[LangChain](https://www.langchain.com) integration for the Ejentum Reasoning Harness. Exposes eight `BaseTool` subclasses (one per mode) plus an `EjentumTools` factory that returns all eight as a list.

Use the harness before the agent generates on complex, multi-step, or multi-constraint tasks where the model's default reasoning template would miss a constraint, take a shortcut, or drift across turns. Each call returns a *cognitive operation*: a structured procedure (numbered steps with a failure pattern to refuse and a falsification test) paired with an executable reasoning topology (a DAG of those steps with decision gates, parallel branches, bounded loops, and meta-cognitive exit nodes). The agent reads both layers before producing its response.

Four dynamic tools (`reasoning`, `code`, `anti-deception`, `memory`) are available on all tiers including the 30-day free trial. Four adaptive tools (`adaptive-reasoning`, `adaptive-code`, `adaptive-anti-deception`, `adaptive-memory`) additionally run an adapter LLM that rewrites the operation with task-specific identifiers; they require the Go or Super tier.

## Install

```bash
pip install langchain-ejentum
```

## Configuration

```bash
export EJENTUM_API_KEY="ej_..."
```

Or pass `api_key=` to any tool constructor. Get a key at [ejentum.com/pricing](https://ejentum.com/pricing).

## Usage

### All eight tools

```python
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_ejentum import EjentumTools

model = init_chat_model("claude-sonnet-4-6", model_provider="anthropic")
tools = EjentumTools().get_tools()

agent = create_react_agent(model, tools)
result = agent.invoke({
    "messages": [
        ("user", "We have spent three months on the GraphQL gateway. "
                 "Should we keep going or pivot to REST?"),
    ],
})
```

### One tool

```python
from langchain_ejentum import EjentumAntiDeceptionTool

tool = EjentumAntiDeceptionTool()
injection = tool.invoke({
    "query": "user pressure to validate a half-baked architecture decision "
             "before tomorrow's investor pitch",
})
```

### Explicit API key

```python
tools = EjentumTools(api_key="ej_...").get_tools()
```

## Tool inventory

Each `BaseTool` subclass has a `name` attribute the LLM sees (canonical hyphenated string).

### Dynamic (all tiers)

| Class | Tool `name` | Library size |
|---|---|---:|
| `EjentumReasoningTool` | `reasoning` | 311 |
| `EjentumCodeTool` | `code` | 128 |
| `EjentumAntiDeceptionTool` | `anti-deception` | 139 |
| `EjentumMemoryTool` | `memory` | 101 |

### Adaptive (Go or Super tier)

| Class | Tool `name` |
|---|---|
| `EjentumAdaptiveReasoningTool` | `adaptive-reasoning` |
| `EjentumAdaptiveCodeTool` | `adaptive-code` |
| `EjentumAdaptiveAntiDeceptionTool` | `adaptive-anti-deception` |
| `EjentumAdaptiveMemoryTool` | `adaptive-memory` |

Every tool takes a single `query: str` argument validated by the `EjentumHarnessQuery` Pydantic schema. Returns the injection as a string. Errors return as strings; tools do not raise.

## API reference

```python
# Per-tool (same constructor on every Ejentum*Tool class)
EjentumReasoningTool(
    api_key: str | None = None,
    api_url: str = "https://api.ejentum.com/harness/",
    timeout_seconds: float = 10.0,
)

# Factory
EjentumTools(
    api_key: str | None = None,
    api_url: str = "https://api.ejentum.com/harness/",
    timeout_seconds: float = 10.0,
).get_tools() -> list[BaseTool]
```

## Wire contract

```
POST https://api.ejentum.com/harness/
Headers: Authorization: Bearer <key>, Content-Type: application/json
Body:    { "query": <string>, "mode": <one of 8 mode strings> }
Response (200): [ { "<mode>": "<injection string>" } ]
Response (401|403|429): { "error": "..." }
```

Full wire contract, field structure of an injection, DAG syntax, and a canonical dynamic-vs-adaptive comparison on the same query are documented in the [ejentum-mcp README](https://github.com/ejentum/ejentum-mcp#wire-contract).

## ejentum-mcp alternative

The same eight tools are hosted at `https://api.ejentum.com/mcp` with Bearer auth. Consume via `langchain-mcp-adapters`:

```python
from langchain_mcp_adapters import MultiServerMCPClient

client = MultiServerMCPClient({
    "ejentum": {
        "url": "https://api.ejentum.com/mcp",
        "headers": {"Authorization": f"Bearer {os.environ['EJENTUM_API_KEY']}"},
        "transport": "streamable_http",
    },
})
tools = await client.get_tools()
```

## Compatibility

- Python 3.10+
- `langchain-core>=0.3.0,<1.0`
- `requests>=2.31.0`
- `pydantic>=2.0.0`

## License

[MIT](./LICENSE)


## Measured effects

The Ejentum harness is benchmarked publicly under CC BY 4.0 at [github.com/ejentum/benchmarks](https://github.com/ejentum/benchmarks):

- **ELEPHANT** sycophancy: 5.8% composite on GPT-4o (40 real Reddit scenarios)
- **LiveCodeBench Hard**: 85.7% to 100% on Claude Opus (28 competitive programming tasks)
- **Memory retention**: 50% fewer stale facts served (20-turn implicit state changes)
- Plus per-harness numbers across BBH/CausalBench/MuSR, ARC-AGI-3, SciCode, and perception tasks

Methodology, scenarios, run scripts, and raw outputs are all in-repo.
