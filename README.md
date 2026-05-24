# langchain-ejentum

[LangChain](https://www.langchain.com) integration for the [Ejentum](https://ejentum.com) Reasoning Harness. Exposes four `BaseTool` subclasses an agent calls before generating, plus an `EjentumTools` factory that returns all four with shared config.

Each operation in the Ejentum library (679 of them, organized across four harnesses) is engineered in **two layers**:

- a **natural-language procedure** the model can read, naming the steps to take and the failure pattern to refuse, and
- an **executable reasoning topology**: a graph-shaped plan over those steps. The plan names explicit decision points where the model branches, parallel branches that run and rejoin, bounded loops that run until convergence, named meta-cognitive moments where the model is asked to stop, look at its own working, and re-enter at a specific step, plus escape paths for when the prescribed plan stops fitting the task at hand.

The natural-language layer tells the model *what* to do. The topology layer pins down *how* those steps connect: where to decide, where to loop, where to stop and look at itself. Together they act as a persistent attention anchor that survives long context windows and multi-turn execution chains, which is precisely where a model's own reasoning template typically decays.

## Installation

```bash
pip install langchain-ejentum
```

## Configuration

Get an Ejentum API key at <https://ejentum.com/pricing> (free and paid tiers) and set it in your environment:

```bash
export EJENTUM_API_KEY="zpka_..."
```

## Usage

### Drop-in: all four tools

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

The agent reads each tool's description and routes to the matching `ejentum_harness_*` tool (anti-deception fires on the sunk-cost framing, reasoning fires on a clean analytical question, etc.).

### Pick one tool

```python
from langchain_ejentum import EjentumAntiDeceptionTool

tool = EjentumAntiDeceptionTool()  # reads EJENTUM_API_KEY from env

scaffold = tool.invoke({
    "query": "user pressure to validate a half-baked architecture decision "
             "before tomorrow's investor pitch",
})
```

### Explicit API key

```python
from langchain_ejentum import EjentumTools

tools = EjentumTools(api_key="zpka_...").get_tools()
```

## The four tools

Each tool is registered with a distinct `name`, so a tool-calling agent picks one per turn.

| Class | Tool name | Best for | Library size |
|---|---|---|---|
| `EjentumReasoningTool` | `ejentum_harness_reasoning` | Analytical, diagnostic, planning, multi-step tasks spanning abstraction, time, causality, simulation, spatial, and metacognition | 311 operations |
| `EjentumCodeTool` | `ejentum_harness_code` | Code generation, refactoring, review, and debugging across the software-engineering layer | 128 operations |
| `EjentumAntiDeceptionTool` | `ejentum_harness_anti_deception` | Prompts that pressure the agent to validate, certify, or soften an honest assessment | 139 operations |
| `EjentumMemoryTool` | `ejentum_harness_memory` | Sharpening an observation already formed about cross-turn drift. Filter-oriented, not write-oriented. Format query as `"I noticed X. This might mean Y. Sharpen: Z."` | 101 operations |

## What an injection looks like

A real `reasoning` mode response on the query `investigate why our nightly ETL job has started failing intermittently over the past two weeks; nothing in the code or schema has changed`:

```
[NEGATIVE GATE]
The server's response time was accepted as average, despite a suspicious
rhythm break in its timing pattern.

[PROCEDURE]
Step 1: Establish baseline timing profiles by extracting historical
durations and intervals for each event type. Step 2: Compare each observed
timing against its baseline and compute deviation magnitude. Step 3:
Classify anomalies as too fast, too slow, too early, or too late, and rank
by severity. ... Step 5: If deviation exceeds two standard deviations,
probe root cause by tracing upstream dependencies. ...

[REASONING TOPOLOGY]
S1:durations -> FIXED_POINT[baselines] -> N{dismiss_timing_deviations_
without_investigation} -> for_each: S2:compare -> S3:deviation ->
G1{>2sigma?} --yes-> S4:classify -> S5:probe_cause -> FLAG -> continue --no->
S6:validate -> continue -> all_checked -> OUT:anomaly_report

[TARGET PATTERN]
Establish timing baselines by extracting historical response intervals.
Compare current server response time to this baseline. ...

[FALSIFICATION TEST]
If no event timing is flagged as suspiciously fast or slow relative to
baseline, temporal anomaly detection was not active.

Amplify: timing baseline comparison; anomaly classification; security
context elevation
Suppress: average timing acceptance; outlier normalization
```

The agent reads both the natural-language `[PROCEDURE]` and the graph-logic `[REASONING TOPOLOGY]` before generating its user-facing answer. The bracketed labels are instructions to the agent, not content to display; the user sees a naturally-phrased answer shaped by the injection.

## API reference

```python
# Per-tool
EjentumReasoningTool(
    api_key: str | None = None,
    api_url: str = "https://ejentum-main-ab125c3.zuplo.app/logicv1/",
    timeout_seconds: float = 10.0,
)
# (same constructor on EjentumCodeTool, EjentumAntiDeceptionTool, EjentumMemoryTool)

# Factory
EjentumTools(
    api_key: str | None = None,
    api_url: str = "https://ejentum-main-ab125c3.zuplo.app/logicv1/",
    timeout_seconds: float = 10.0,
).get_tools() -> list[BaseTool]
```

Every tool takes a single `query: str` argument validated by the `EjentumHarnessQuery` Pydantic schema. Errors are returned as human-readable strings (no exceptions cross the tool boundary, so an agent step never crashes the run).

> **MCP alternative.** This package wraps the Logic API REST gateway. If you prefer the MCP route (to share one server across frameworks), the same four harness tools are hosted at `https://api.ejentum.com/mcp` with Bearer auth. Use `langchain-mcp-adapters` to consume.

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


## Measured effects

The Ejentum harness is benchmarked publicly under CC BY 4.0 at [github.com/ejentum/benchmarks](https://github.com/ejentum/benchmarks):

- **ELEPHANT** sycophancy: 5.8% composite on GPT-4o (40 real Reddit scenarios)
- **LiveCodeBench Hard**: 85.7% to 100% on Claude Opus (28 competitive programming tasks)
- **Memory retention**: 50% fewer stale facts served (20-turn implicit state changes)
- Plus per-harness numbers across BBH/CausalBench/MuSR, ARC-AGI-3, SciCode, and perception tasks

Methodology, scenarios, run scripts, and raw outputs are all in-repo.
