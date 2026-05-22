# Changelog

All notable changes to `langchain-ejentum` are documented here. This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-22

### Added

- Initial release.
- Four `langchain_core.tools.BaseTool` subclasses: `EjentumReasoningTool`, `EjentumCodeTool`, `EjentumAntiDeceptionTool`, `EjentumMemoryTool`. Each is a single-argument tool (`query: str`) that calls the Ejentum Logic API and returns a structured scaffold.
- `EjentumTools(api_key=...)` factory class with `get_tools() -> list[BaseTool]` for the common case of passing all four to an agent with shared configuration.
- `EjentumHarnessQuery` Pydantic schema used as `args_schema` on every tool, with the memory-mode docstring guidance ("I noticed X. This might mean Y. Sharpen: Z.").
- Construction-time and call-time validation: empty/whitespace query returns an actionable error without spending a paid API call. Missing `EJENTUM_API_KEY` returns an actionable error pointing to https://ejentum.com/pricing.
- Errors returned as human-readable strings from `_run` for every failure path (no exceptions cross the tool boundary so an agent step never crashes the run).
- Unit tests cover the failure surface: missing key, empty/whitespace/non-string query, 401, non-200, invalid JSON, unexpected response shape, non-string scaffold value, network error, the `EjentumTools` factory, and per-tool mode dispatch.
- Published to PyPI with OIDC trusted-publisher provenance attestation via GitHub Actions.

### Background

LangChain's third-party tool convention is one PyPI package per integration under the `langchain-<vendor>` namespace (`langchain-openai`, `langchain-anthropic`, `langchain-tavily`, etc.), each exposing `BaseTool` subclasses users instantiate directly. This package follows that pattern.
