"""langchain-ejentum: LangChain integration for the Ejentum Reasoning Harness.

Exposes eight ``BaseTool`` subclasses, one per harness mode:

Dynamic (single retrieval, all tiers including the 30-day free trial):

- :class:`EjentumReasoningTool` (311 operations: abstraction, time, causality,
  simulation, spatial, metacognition)
- :class:`EjentumCodeTool` (128 operations: software-engineering layer)
- :class:`EjentumAntiDeceptionTool` (139 operations: sycophancy,
  hallucination, deception, adversarial framing, judgment, executive control)
- :class:`EjentumMemoryTool` (101 operations: perception layer; filter-oriented)

Adaptive (top-k retrieval + adapter LLM rewrites the operation to fit the
specific task; requires Go or Super tier):

- :class:`EjentumAdaptiveReasoningTool`
- :class:`EjentumAdaptiveCodeTool`
- :class:`EjentumAdaptiveAntiDeceptionTool`
- :class:`EjentumAdaptiveMemoryTool`

Plus :class:`EjentumTools`, a factory that returns all eight with shared config.

Pricing at https://ejentum.com/pricing.
"""

from langchain_ejentum.tools import (
    EjentumAdaptiveAntiDeceptionTool,
    EjentumAdaptiveCodeTool,
    EjentumAdaptiveMemoryTool,
    EjentumAdaptiveReasoningTool,
    EjentumAntiDeceptionTool,
    EjentumCodeTool,
    EjentumMemoryTool,
    EjentumReasoningTool,
    EjentumTools,
)
from langchain_ejentum._base import (
    DEFAULT_API_URL,
    DEFAULT_TIMEOUT_SECONDS,
    EjentumHarnessQuery,
)

__all__ = [
    "EjentumReasoningTool",
    "EjentumCodeTool",
    "EjentumAntiDeceptionTool",
    "EjentumMemoryTool",
    "EjentumAdaptiveReasoningTool",
    "EjentumAdaptiveCodeTool",
    "EjentumAdaptiveAntiDeceptionTool",
    "EjentumAdaptiveMemoryTool",
    "EjentumTools",
    "EjentumHarnessQuery",
    "DEFAULT_API_URL",
    "DEFAULT_TIMEOUT_SECONDS",
]
__version__ = "0.2.0"
