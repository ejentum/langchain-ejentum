"""Unit tests for langchain-ejentum BaseTool subclasses and the EjentumTools factory."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool

from langchain_ejentum import (
    EjentumAntiDeceptionTool,
    EjentumCodeTool,
    EjentumMemoryTool,
    EjentumReasoningTool,
    EjentumTools,
)


def _mock_response(
    status_code: int = 200, json_data=None, text: str = ""
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text or (str(json_data) if json_data else "")
    resp.json.return_value = json_data if json_data is not None else []
    return resp


def test_each_tool_is_basetool_subclass():
    for cls in (
        EjentumReasoningTool,
        EjentumCodeTool,
        EjentumAntiDeceptionTool,
        EjentumMemoryTool,
    ):
        assert issubclass(cls, BaseTool), f"{cls.__name__} must subclass BaseTool"


def test_each_tool_has_distinct_name_and_description():
    instances = [
        EjentumReasoningTool(),
        EjentumCodeTool(),
        EjentumAntiDeceptionTool(),
        EjentumMemoryTool(),
    ]
    names = {t.name for t in instances}
    assert names == {
        "ejentum_harness_reasoning",
        "ejentum_harness_code",
        "ejentum_harness_anti_deception",
        "ejentum_harness_memory",
    }
    for t in instances:
        assert len(t.description) > 50, f"{t.name} description too short"


def test_ejentum_tools_factory_returns_four_tools():
    factory = EjentumTools()
    tools = factory.get_tools()
    assert len(tools) == 4
    assert all(isinstance(t, BaseTool) for t in tools)
    assert {t.name for t in tools} == {
        "ejentum_harness_reasoning",
        "ejentum_harness_code",
        "ejentum_harness_anti_deception",
        "ejentum_harness_memory",
    }


def test_factory_propagates_shared_config():
    factory = EjentumTools(
        api_key="shared-key",
        api_url="https://example.com/api/",
        timeout_seconds=42.0,
    )
    tools = factory.get_tools()
    for t in tools:
        assert t.api_key == "shared-key"
        assert t.api_url == "https://example.com/api/"
        assert t.timeout_seconds == 42.0


def test_missing_api_key_returns_actionable_error(monkeypatch):
    monkeypatch.delenv("EJENTUM_API_KEY", raising=False)
    tool = EjentumReasoningTool()
    result = tool.invoke({"query": "diagnose 503s under load"})
    assert "EJENTUM_API_KEY" in result
    assert "ejentum.com/pricing" in result


def test_empty_query_validation_error(monkeypatch):
    monkeypatch.setenv("EJENTUM_API_KEY", "test-key")
    tool = EjentumReasoningTool()
    with patch("langchain_ejentum._base.requests.post") as mock_post:
        with pytest.raises(Exception):
            tool.invoke({"query": ""})
        mock_post.assert_not_called()


def test_whitespace_only_query_returns_validation_error(monkeypatch):
    """Whitespace-only input passes Pydantic min_length=1 but must NOT trigger a paid request."""
    monkeypatch.setenv("EJENTUM_API_KEY", "test-key")
    tool = EjentumAntiDeceptionTool()
    with patch("langchain_ejentum._base.requests.post") as mock_post:
        result = tool.invoke({"query": "   \t\n  "})
    assert "query" in result.lower()
    assert "required" in result.lower()
    mock_post.assert_not_called()


@pytest.mark.parametrize(
    "cls,mode",
    [
        (EjentumReasoningTool, "reasoning"),
        (EjentumCodeTool, "code"),
        (EjentumAntiDeceptionTool, "anti-deception"),
        (EjentumMemoryTool, "memory"),
    ],
)
@patch("langchain_ejentum._base.requests.post")
def test_each_tool_dispatches_correct_mode(mock_post, cls, mode, monkeypatch):
    monkeypatch.setenv("EJENTUM_API_KEY", "test-key")
    mock_post.return_value = _mock_response(
        status_code=200,
        json_data=[{mode: f"[NEGATIVE GATE] sample {mode} scaffold"}],
    )

    tool = cls()
    query = (
        "I noticed drift. This might mean Y. Sharpen: Z."
        if mode == "memory"
        else "sample task"
    )
    result = tool.invoke({"query": query})

    assert f"sample {mode} scaffold" in result
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"]["mode"] == mode
    assert kwargs["json"]["query"] == query


@patch("langchain_ejentum._base.requests.post")
def test_explicit_api_key_arg_overrides_env(mock_post, monkeypatch):
    monkeypatch.setenv("EJENTUM_API_KEY", "env-key")
    mock_post.return_value = _mock_response(
        status_code=200,
        json_data=[{"reasoning": "scaffold"}],
    )

    tool = EjentumReasoningTool(api_key="explicit-key")
    tool.invoke({"query": "anything"})

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer explicit-key"


@patch("langchain_ejentum._base.requests.post")
def test_401_returns_actionable_error(mock_post, monkeypatch):
    monkeypatch.setenv("EJENTUM_API_KEY", "bad-key")
    mock_post.return_value = _mock_response(status_code=401, text="Unauthorized")

    tool = EjentumAntiDeceptionTool()
    result = tool.invoke({"query": "anything"})

    assert "401" in result
    assert "EJENTUM_API_KEY" in result


@patch("langchain_ejentum._base.requests.post")
def test_non_200_returns_status_and_body(mock_post, monkeypatch):
    monkeypatch.setenv("EJENTUM_API_KEY", "test-key")
    mock_post.return_value = _mock_response(status_code=500, text="boom")

    tool = EjentumCodeTool()
    result = tool.invoke({"query": "anything"})

    assert "500" in result
    assert "boom" in result


@patch("langchain_ejentum._base.requests.post")
def test_unexpected_response_shape_is_handled(mock_post, monkeypatch):
    monkeypatch.setenv("EJENTUM_API_KEY", "test-key")
    mock_post.return_value = _mock_response(
        status_code=200, json_data={"wrong": "shape"}
    )

    tool = EjentumCodeTool()
    result = tool.invoke({"query": "anything"})

    assert "unexpected response shape" in result.lower()


@patch("langchain_ejentum._base.requests.post")
def test_non_string_scaffold_value_is_handled(mock_post, monkeypatch):
    monkeypatch.setenv("EJENTUM_API_KEY", "test-key")
    mock_post.return_value = _mock_response(
        status_code=200,
        json_data=[{"reasoning": ["not", "a", "string"]}],
    )

    tool = EjentumReasoningTool()
    result = tool.invoke({"query": "anything"})

    assert "unexpected response shape" in result.lower()


@patch("langchain_ejentum._base.requests.post")
def test_invalid_json_response_is_handled(mock_post, monkeypatch):
    monkeypatch.setenv("EJENTUM_API_KEY", "test-key")
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "<html>not json</html>"
    resp.json.side_effect = ValueError("not json")
    mock_post.return_value = resp

    tool = EjentumReasoningTool()
    result = tool.invoke({"query": "anything"})

    assert "not valid json" in result.lower()


@patch("langchain_ejentum._base.requests.post")
def test_network_error_is_caught(mock_post, monkeypatch):
    import requests

    monkeypatch.setenv("EJENTUM_API_KEY", "test-key")
    mock_post.side_effect = requests.ConnectionError("simulated")

    tool = EjentumMemoryTool()
    result = tool.invoke(
        {"query": "I noticed drift. This might mean Y. Sharpen: Z."}
    )

    assert "network error" in result.lower()
    assert "simulated" in result
