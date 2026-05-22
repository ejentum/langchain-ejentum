"""langchain-ejentum: LangChain integration for the Ejentum Reasoning Harness.

Exposes four ``BaseTool`` subclasses, one per harness:

- :class:`EjentumReasoningTool` (311 operations: abstraction, time, causality,
  simulation, spatial, metacognition)
- :class:`EjentumCodeTool` (128 operations: software-engineering layer)
- :class:`EjentumAntiDeceptionTool` (139 operations: sycophancy,
  hallucination, deception, adversarial framing, judgment, executive control)
- :class:`EjentumMemoryTool` (101 operations: perception layer; filter-oriented)

Plus :class:`EjentumTools`, a factory that returns all four with shared config.

Free and paid tiers at https://ejentum.com/pricing.
"""

from langchain_ejentum.tools import (
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
    "EjentumTools",
    "EjentumHarnessQuery",
    "DEFAULT_API_URL",
    "DEFAULT_TIMEOUT_SECONDS",
]
__version__ = "0.1.0"
