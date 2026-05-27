"""Public tool classes for the Ejentum Reasoning Harness LangChain integration.

Eight tools total: four dynamic (`reasoning`, `code`, `anti-deception`,
`memory`) and four adaptive (`adaptive-reasoning`, `adaptive-code`,
`adaptive-anti-deception`, `adaptive-memory`) that pre-fit the cognitive
operation to the caller's task via an adapter LLM. Adaptive tools require
the Go or Super tier.

Tool ``name`` (the LLM-facing string) equals the API mode string. Python
class names use CapitalCase for developer ergonomics.

The bracketed labels in the returned injection (``[NEGATIVE GATE]``,
``[PROCEDURE]``, ``[REASONING TOPOLOGY]``, ``[FALSIFICATION TEST]``, etc.)
are instructions to the agent, not content to display.
"""

from __future__ import annotations

from typing import ClassVar, List, Optional

from langchain_core.tools import BaseTool

from langchain_ejentum._base import _EjentumBaseTool


# ---------------------------------------------------------------------------
# Dynamic tools (single retrieval, all tiers including the 30-day free trial)
# ---------------------------------------------------------------------------


class EjentumReasoningTool(_EjentumBaseTool):
    """Retrieve a reasoning injection before an analytical or planning step.

    Call BEFORE the agent performs analysis, diagnosis, planning, or any
    multi-step task. 311 operations spanning abstraction, time, causality,
    simulation, spatial, and metacognition.
    """

    name: str = "reasoning"
    description: str = (
        "Retrieve a reasoning injection before any analytical, diagnostic, "
        "planning, or multi-step task. Returns a structured injection with a "
        "named failure pattern, an executable procedure, a reasoning "
        "topology (graph DAG), and a falsification test from a library of "
        "311 reasoning operations."
    )

    mode: ClassVar[str] = "reasoning"


class EjentumCodeTool(_EjentumBaseTool):
    """Retrieve a code injection before generating, refactoring, or reviewing code.

    Call BEFORE the agent produces or reviews code. 128 operations in the
    software-engineering layer.
    """

    name: str = "code"
    description: str = (
        "Retrieve a code injection before any code generation, refactoring, "
        "review, or debugging task. Returns a structured injection with a "
        "named code-failure pattern, an engineering procedure, a reasoning "
        "topology (graph DAG), and a verification step from a library of "
        "128 code operations."
    )

    mode: ClassVar[str] = "code"


class EjentumAntiDeceptionTool(_EjentumBaseTool):
    """Retrieve an anti-deception injection when the prompt pressures the agent.

    Call BEFORE responding to prompts that pressure validation, manufactured
    agreement, authority appeals, or any setup where the obvious helpful
    answer would compromise honesty. 139 operations spanning sycophancy,
    hallucination, deception, adversarial framing, judgment, executive control.
    """

    name: str = "anti-deception"
    description: str = (
        "Retrieve an anti-deception injection before responding to any "
        "prompt that pressures the agent to validate, certify, or soften an "
        "honest assessment. Returns a structured injection with a named "
        "deception pattern, an integrity procedure, a detection topology "
        "(graph DAG with omission-bias gates), and an integrity check from a "
        "library of 139 operations."
    )

    mode: ClassVar[str] = "anti-deception"


class EjentumMemoryTool(_EjentumBaseTool):
    """Retrieve a memory injection to sharpen a cross-turn observation already formed.

    Call ONLY when sharpening an observation already formed. Filter-oriented
    (101 perception operations), NOT write-oriented; do not call for fact
    extraction or storing structured data.

    The query MUST be in the format: "I noticed [observation]. This might
    mean [interpretation]. Sharpen: [what to see deeper into]."
    """

    name: str = "memory"
    description: str = (
        "Retrieve a memory injection ONLY when sharpening an observation "
        "the agent has already formed about cross-turn drift or pattern. "
        "Filter-oriented, not write-oriented. Format 'query' as: 'I "
        "noticed [X]. This might mean [Y]. Sharpen: [Z].' Library of 101 "
        "perception operations."
    )

    mode: ClassVar[str] = "memory"


# ---------------------------------------------------------------------------
# Adaptive tools (top-k retrieval + LLM adapter rewrites operation to fit
# the specific task; requires Go or Super tier)
# ---------------------------------------------------------------------------


class EjentumAdaptiveReasoningTool(_EjentumBaseTool):
    """Same triggers as :class:`EjentumReasoningTool`, but the operation is rewritten by an adapter LLM.

    Procedure steps and topology DAG nodes are concretized with task-specific
    language. Use when the dynamic tool is too generic, or for high-stakes
    analytical work. Requires Go or Super tier. Cost ~2-3 seconds vs ~1
    second for dynamic.
    """

    name: str = "adaptive-reasoning"
    description: str = (
        "Same triggers as `reasoning`, but the returned operation is "
        "REWRITTEN by an adapter LLM to fit the specific task. Procedure "
        "steps and topology DAG nodes are concretized with task-specific "
        "language. Use when the dynamic tool is too generic or for "
        "high-stakes analytical work. Requires Go or Super tier."
    )

    mode: ClassVar[str] = "adaptive-reasoning"


class EjentumAdaptiveCodeTool(_EjentumBaseTool):
    """Same triggers as :class:`EjentumCodeTool`, but the operation is rewritten by an adapter LLM.

    Language, framework, and failure modes are concretized in every step.
    Use for security-critical reviews or refactor-heavy diffs. Requires Go
    or Super tier.
    """

    name: str = "adaptive-code"
    description: str = (
        "Same triggers as `code`, but the returned operation is REWRITTEN "
        "by an adapter LLM to fit the specific code task: language, "
        "framework, and failure modes are concretized in every step. Use "
        "for security-critical reviews or refactor-heavy diffs. Requires Go "
        "or Super tier."
    )

    mode: ClassVar[str] = "adaptive-code"


class EjentumAdaptiveAntiDeceptionTool(_EjentumBaseTool):
    """Same triggers as :class:`EjentumAntiDeceptionTool`, but the operation is rewritten by an adapter LLM.

    Detection topology gates are concretized to the exact pressure at play.
    Use when the stakes of a soft answer are high. Requires Go or Super tier.
    """

    name: str = "adaptive-anti-deception"
    description: str = (
        "Same triggers as `anti-deception`, but the returned operation is "
        "REWRITTEN by an adapter LLM to fit the specific integrity dynamic. "
        "Detection topology gates are concretized to the exact pressure at "
        "play. Requires Go or Super tier."
    )

    mode: ClassVar[str] = "adaptive-anti-deception"


class EjentumAdaptiveMemoryTool(_EjentumBaseTool):
    """Same triggers as :class:`EjentumMemoryTool`, but the operation is rewritten by an adapter LLM.

    Perception topology nodes are concretized to the specific signal.
    Observe FIRST, then call. Requires Go or Super tier.
    """

    name: str = "adaptive-memory"
    description: str = (
        "Same triggers as `memory`, but the returned operation is REWRITTEN "
        "by an adapter LLM to fit the specific observation. Perception "
        "topology nodes are concretized to the specific signal. Observe "
        "FIRST, then call. Requires Go or Super tier."
    )

    mode: ClassVar[str] = "adaptive-memory"


class EjentumTools:
    """Grouping helper that returns all eight harness tools with shared config.

    LangChain 1.x dropped ``BaseToolkit`` in favor of plain factory classes.
    This helper exists so a user can write::

        from langchain_ejentum import EjentumTools

        tools = EjentumTools(api_key="...").get_tools()
        agent = create_react_agent(model, tools)

    instead of instantiating all eight classes individually.

    :param api_key: Ejentum Logic API key. If omitted, each tool reads from
        the ``EJENTUM_API_KEY`` environment variable at call time. Pricing
        at https://ejentum.com/pricing.
    :param api_url: Override only if you self-host the Ejentum Logic API
        gateway.
    :param timeout_seconds: Per-call HTTP timeout shared across all tools.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://ejentum-main-ab125c3.zuplo.app/logicv1/",
        timeout_seconds: float = 10.0,
    ) -> None:
        self._kwargs = dict(
            api_key=api_key,
            api_url=api_url,
            timeout_seconds=timeout_seconds,
        )

    def get_tools(self) -> List[BaseTool]:
        """Return the eight harness tools as a list, configured with shared kwargs."""
        return [
            EjentumReasoningTool(**self._kwargs),
            EjentumCodeTool(**self._kwargs),
            EjentumAntiDeceptionTool(**self._kwargs),
            EjentumMemoryTool(**self._kwargs),
            EjentumAdaptiveReasoningTool(**self._kwargs),
            EjentumAdaptiveCodeTool(**self._kwargs),
            EjentumAdaptiveAntiDeceptionTool(**self._kwargs),
            EjentumAdaptiveMemoryTool(**self._kwargs),
        ]
