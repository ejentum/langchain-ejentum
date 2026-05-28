# langchain-ejentum

[LangChain](https://www.langchain.com) integration for the [Ejentum](https://ejentum.com) Reasoning Harness. Exposes eight `BaseTool` subclasses an agent calls before generating, plus an `EjentumTools` factory that returns all eight with shared config: four dynamic (`reasoning`, `code`, `anti-deception`, `memory`) plus four adaptive (`adaptive-reasoning`, `adaptive-code`, `adaptive-anti-deception`, `adaptive-memory`) that pre-fit the cognitive operation to the caller's task via an adapter LLM.

Each operation in the Ejentum library (679 of them, organized across four cognitive harnesses each with dynamic and adaptive variants) is engineered in **two layers**:

- a **natural-language procedure** the model can read, naming the steps to take and the failure pattern to refuse, and
- an **executable reasoning topology**: a graph-shaped plan over those steps. The plan names explicit decision points where the model branches, parallel branches that run and rejoin, bounded loops that run until convergence, named meta-cognitive moments where the model is asked to stop, look at its own working, and re-enter at a specific step, plus escape paths for when the prescribed plan stops fitting the task at hand.

The natural-language layer tells the model *what* to do. The topology layer pins down *how* those steps connect: where to decide, where to loop, where to stop and look at itself. Together they act as a persistent attention anchor that survives long context windows and multi-turn execution chains, which is precisely where a model's own reasoning template typically decays.

## Installation

```bash
pip install langchain-ejentum
```

## Configuration

Get an Ejentum API key at <https://ejentum.com/pricing>. The 30-day free trial (no card required) includes 1,000 dynamic reasoning calls; adaptive tools require Go or Super.

```bash
export EJENTUM_API_KEY="ej_..."
```

## Usage

### Drop-in: all eight tools

```python
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_ejentum import EjentumTools

model = init_chat_model("claude-sonnet-4-6", model_provider="anthropic")
tools = EjentumTools().get_tools()  # reads EJENTUM_API_KEY from env

agent = create_react_agent(model, tools)
result = agent.invoke({
    "messages": [
        ("user", "We've spent three months on the GraphQL gateway. "
                 "Should we keep going or pivot to REST?"),
    ],
})
```

The agent reads each tool's description and routes to the matching tool (`anti-deception` fires on the sunk-cost framing, `reasoning` fires on a clean analytical question, etc.).

### Pick one tool

```python
from langchain_ejentum import EjentumAntiDeceptionTool

tool = EjentumAntiDeceptionTool()  # reads EJENTUM_API_KEY from env

injection = tool.invoke({
    "query": "user pressure to validate a half-baked architecture decision "
             "before tomorrow's investor pitch",
})
```

### Explicit API key

```python
from langchain_ejentum import EjentumTools

tools = EjentumTools(api_key="ej_...").get_tools()
```

## The eight tools

Each tool is registered with a distinct `name`, so a tool-calling agent picks one per turn.

### Dynamic (single retrieval, all tiers including the 30-day free trial)

| Class | Tool name (LLM-visible) | Best for | Library size |
|---|---|---|---|
| `EjentumReasoningTool` | `reasoning` | Analytical, diagnostic, planning, multi-step tasks spanning abstraction, time, causality, simulation, spatial, and metacognition | 311 operations |
| `EjentumCodeTool` | `code` | Code generation, refactoring, review, and debugging across the software-engineering layer | 128 operations |
| `EjentumAntiDeceptionTool` | `anti-deception` | Prompts that pressure the agent to validate, certify, or soften an honest assessment | 139 operations |
| `EjentumMemoryTool` | `memory` | Sharpening an observation already formed about cross-turn drift. Filter-oriented, not write-oriented. Format query as `"I noticed X. This might mean Y. Sharpen: Z."` | 101 operations |

### Adaptive (top-k retrieval + adapter LLM rewrites operation to fit the task; Go or Super tier required)

| Class | Tool name | When to prefer over the dynamic version |
|---|---|---|
| `EjentumAdaptiveReasoningTool` | `adaptive-reasoning` | High-stakes analytical work where every DAG node should be mapped to your specifics before generation. |
| `EjentumAdaptiveCodeTool` | `adaptive-code` | Security-critical reviews, refactor-heavy diffs, or any code work where every verification step should be concretized. |
| `EjentumAdaptiveAntiDeceptionTool` | `adaptive-anti-deception` | When the stakes of a soft or sycophantic answer are high. |
| `EjentumAdaptiveMemoryTool` | `adaptive-memory` | When the dynamic memory tool's general scaffold is not sharp enough for the perception being formed. |

## What an injection looks like

A real `reasoning` mode response on the query `investigate why our nightly ETL job has started failing intermittently over the past two weeks; nothing in the code or schema has changed`:

```
[NEGATIVE GATE]
The server's response time was accepted as average, despite a suspicious
rhythm break in its timing pattern.

[PROCEDURE]
Step 1: Establish baseline timing profiles by extracting historical
durations and intervals for each event type. Step 2: Compare each observed
timing against its baseline and compute deviation magnitude. ...

[REASONING TOPOLOGY]
S1:durations -> FIXED_POINT[baselines] -> N{dismiss_timing_deviations_
without_investigation} -> for_each: S2:compare -> S3:deviation ->
G1{>2sigma?} --yes-> S4:classify -> S5:probe_cause -> FLAG -> continue --no->
S6:validate -> continue -> all_checked -> OUT:anomaly_report

[FALSIFICATION TEST]
If no event timing is flagged as suspiciously fast or slow relative to
baseline, temporal anomaly detection was not active.

Amplify: timing baseline comparison; anomaly classification
Suppress: average timing acceptance; outlier normalization
```

The agent reads both the natural-language `[PROCEDURE]` and the graph-logic `[REASONING TOPOLOGY]` before generating its user-facing answer. The bracketed labels are instructions to the agent, not content to display; the user sees a naturally-phrased answer shaped by the injection.

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

Every tool takes a single `query: str` argument validated by the `EjentumHarnessQuery` Pydantic schema. Errors are returned as human-readable strings (no exceptions cross the tool boundary, so an agent step never crashes the run).

> **MCP alternative.** This package wraps the Ejentum API REST gateway. If you prefer the MCP route (to share one server across frameworks), the same eight harness tools are hosted at `https://api.ejentum.com/mcp` with Bearer auth. Use `langchain-mcp-adapters` to consume.

## Compatibility

- Python 3.10+
- `langchain-core>=0.3.0,<1.0`
- `requests>=2.31.0`
- `pydantic>=2.0.0`

## Resources

- Ejentum homepage: <https://ejentum.com>
- Pricing: <https://ejentum.com/pricing>
- API reference: <https://ejentum.com/docs/api_reference>
- "Why LLM Agents Fail" essay: <https://ejentum.com/blog/why-llm-agents-fail>
- "Under Pressure" research paper: <https://doi.org/10.5281/zenodo.19392715>
- LangChain documentation: <https://docs.langchain.com>

## License

[MIT](./LICENSE)
