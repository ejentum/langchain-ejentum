"""Public tool classes for the Ejentum Reasoning Harness LangChain integration.

Each tool is a :class:`langchain_core.tools.BaseTool` subclass an agent calls
before generating. Pick the tool that matches what the agent is about to do
(or pass all four via :class:`EjentumTools` and let the agent route).

The bracketed labels in the returned scaffold (``[NEGATIVE GATE]``,
``[PROCEDURE]``, ``[REASONING TOPOLOGY]``, ``[FALSIFICATION TEST]``, etc.)
are instructions to the agent, not content to display.
"""

from __future__ import annotations

from typing import ClassVar, List, Optional

from langchain_core.tools import BaseTool

from langchain_ejentum._base import _EjentumBaseTool


class EjentumReasoningTool(_EjentumBaseTool):
    """Retrieve a reasoning-mode scaffold before an analytical or planning step.

    Call BEFORE the agent performs analysis, diagnosis, planning, or any
    multi-step task. The Ejentum reasoning harness contains 311 operations
    spanning abstraction, time, causality, simulation, spatial, and
    metacognition. The returned scaffold is engineered in two layers: a
    natural-language procedure (named failure pattern, executable steps,
    suppression vectors, falsification test) and an executable reasoning
    topology (graph DAG with decision gates, parallel branches, bounded
    loops, and meta-cognitive exit nodes). Read both before generating.
    """

    name: str = "ejentum_harness_reasoning"
    description: str = (
        "Retrieve a reasoning scaffold before any analytical, diagnostic, "
        "planning, or multi-step task. Returns a structured scaffold with a "
        "named failure pattern, an executable procedure, a reasoning "
        "topology (graph DAG), and a falsification test. Use 'query' to "
        "describe what the agent is about to work on in 1-2 sentences."
    )

    mode: ClassVar[str] = "reasoning"


class EjentumCodeTool(_EjentumBaseTool):
    """Retrieve a code-mode scaffold before generating, refactoring, or reviewing code.

    Call BEFORE the agent produces or reviews code. The Ejentum code harness
    contains 128 operations in the software-engineering layer (correctness,
    refactor safety, contract preservation, edge case coverage, error path
    discipline). The returned scaffold is engineered in two layers as
    described above. Read both before emitting code.
    """

    name: str = "ejentum_harness_code"
    description: str = (
        "Retrieve a code scaffold before any code generation, refactoring, "
        "review, or debugging task. Returns a structured scaffold with a "
        "named code-failure pattern, an engineering procedure, a reasoning "
        "topology (graph DAG), and a verification step. Use 'query' to "
        "describe what the agent is coding or reviewing in 1-2 sentences; "
        "include the failure risk to avoid where possible."
    )

    mode: ClassVar[str] = "code"


class EjentumAntiDeceptionTool(_EjentumBaseTool):
    """Retrieve an anti-deception scaffold when the prompt pressures the agent to soften an honest assessment.

    Call BEFORE the agent responds to prompts that pressure validation,
    manufactured agreement, authority appeals, fabricated commitments, or
    any setup where the obvious helpful answer would compromise honesty.
    The Ejentum anti-deception harness contains 139 operations spanning
    sycophancy, hallucination, deception, adversarial framing, judgment,
    and executive control.
    """

    name: str = "ejentum_harness_anti_deception"
    description: str = (
        "Retrieve an anti-deception scaffold before responding to any "
        "prompt that pressures the agent to validate, certify, or soften an "
        "honest assessment. Returns a structured scaffold with a named "
        "deception pattern, an integrity procedure, a detection topology "
        "(graph DAG with omission-bias gates), and an integrity check. Use "
        "'query' to describe the integrity dynamic at play in 1-2 sentences."
    )

    mode: ClassVar[str] = "anti-deception"


class EjentumMemoryTool(_EjentumBaseTool):
    """Retrieve a memory-mode scaffold to sharpen a cross-turn observation already formed.

    Call ONLY when sharpening an observation the agent has already formed
    about conversation state, drift, or cross-turn pattern. The Ejentum
    memory harness is filter-oriented (101 perception operations), NOT
    write-oriented; do not call for fact extraction, summarization, or
    storing structured data, those produce scaffold paralysis.

    The query MUST be in the format: "I noticed [observation]. This might
    mean [tentative interpretation]. Sharpen: [what to see deeper into]."
    Calling with an empty mind defeats the harness. Observe first.
    """

    name: str = "ejentum_harness_memory"
    description: str = (
        "Retrieve a memory-mode scaffold ONLY when sharpening an observation "
        "the agent has already formed about cross-turn drift or pattern. "
        "Filter-oriented, not write-oriented; do not call for fact "
        "extraction. Format 'query' as: 'I noticed [X]. This might mean "
        "[Y]. Sharpen: [Z].' Calling with an empty mind defeats the harness."
    )

    mode: ClassVar[str] = "memory"


class EjentumTools:
    """Grouping helper that returns all four harness tools with shared config.

    LangChain 1.x dropped ``BaseToolkit`` in favor of plain factory classes.
    This helper exists so a user can write::

        from langchain_ejentum import EjentumTools

        tools = EjentumTools(api_key="...").get_tools()
        agent = create_react_agent(model, tools)

    instead of instantiating all four classes individually.

    :param api_key: Ejentum Logic API key. If omitted, each tool reads from
        the ``EJENTUM_API_KEY`` environment variable at call time. Free and
        paid tiers at https://ejentum.com/pricing.
    :param api_url: Override only if you self-host the Ejentum Logic API
        gateway.
    :param timeout_seconds: Per-call HTTP timeout shared across all four
        tools.
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
        """Return the four harness tools as a list, configured with shared kwargs."""
        return [
            EjentumReasoningTool(**self._kwargs),
            EjentumCodeTool(**self._kwargs),
            EjentumAntiDeceptionTool(**self._kwargs),
            EjentumMemoryTool(**self._kwargs),
        ]
