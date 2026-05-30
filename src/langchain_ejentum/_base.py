"""Shared base class and HTTP helper for the four Ejentum harness tools.

Underscore-prefixed module: not part of the public API. Users import the
four concrete tool classes (``EjentumReasoningTool`` and friends) from
``langchain_ejentum`` directly.
"""

from __future__ import annotations

import os
from typing import ClassVar, Optional

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


DEFAULT_API_URL = "https://api.ejentum.com/harness/"
DEFAULT_TIMEOUT_SECONDS = 10.0


class EjentumHarnessQuery(BaseModel):
    """Single-argument schema for every harness tool."""

    query: str = Field(
        ...,
        description=(
            "A 1-2 sentence description of the task the agent is about to "
            "work on. Be specific about the failure mode to avoid. For the "
            "memory harness, format as: 'I noticed [X]. This might mean [Y]. "
            "Sharpen: [Z].'"
        ),
        min_length=1,
    )


class _EjentumBaseTool(BaseTool):
    """Internal base. Concrete subclasses fix ``mode`` via a class attribute.

    The constructor surfaces three knobs (``api_key``, ``api_url``,
    ``timeout_seconds``). All four concrete tools share the same HTTP path
    and the same error-as-string contract: errors are returned as
    human-readable strings from ``_run`` so the calling agent never crashes
    the run.
    """

    args_schema: type[BaseModel] = EjentumHarnessQuery

    mode: ClassVar[str] = ""

    api_key: Optional[str] = Field(default=None, exclude=True)
    api_url: str = DEFAULT_API_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def _resolve_key(self) -> Optional[str]:
        return self.api_key or os.environ.get("EJENTUM_API_KEY")

    def _run(self, query: str, **_: object) -> str:
        clean = query.strip() if isinstance(query, str) else ""
        if not clean:
            return "Ejentum harness call failed: 'query' is required."

        api_key = self._resolve_key()
        if not api_key:
            return (
                "Ejentum harness call failed: EJENTUM_API_KEY environment "
                "variable is not set. Free and paid tiers at "
                "https://ejentum.com/pricing."
            )

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"query": clean, "mode": self.mode},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            return f"Ejentum harness call failed: network error: {exc}"

        if response.status_code == 401:
            return (
                "Ejentum harness call failed: unauthorized (401). Check the "
                "EJENTUM_API_KEY value. Free and paid tiers at "
                "https://ejentum.com/pricing."
            )
        if response.status_code != 200:
            return (
                f"Ejentum harness call failed: HTTP {response.status_code}. "
                f"Response: {response.text[:300]}"
            )

        try:
            data = response.json()
        except ValueError:
            return (
                f"Ejentum harness call failed: response is not valid JSON. "
                f"Body: {response.text[:300]}"
            )

        if isinstance(data, list) and data and isinstance(data[0], dict):
            scaffold = data[0].get(self.mode)
            if isinstance(scaffold, str) and scaffold:
                return scaffold

        return (
            f"Ejentum harness call returned an unexpected response shape: "
            f"{str(data)[:300]}"
        )
