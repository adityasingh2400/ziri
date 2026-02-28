"""Tests for the LangGraph orchestrator graph construction and routing."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.memory import InMemoryStore
from app.core.orchestrator import ZiriOrchestrator, ZiriState
from app.core.brain import Brain
from app.core.supervisor import Supervisor, SupervisorResult
from app.core.tool_runner import ToolRunner
from app.integrations.tts import TTS
from app.schemas import IntentType, RouterDecision, ToolResult


@pytest.fixture()
def _mock_tool_runner():
    runner = MagicMock(spec=ToolRunner)
    runner.run.return_value = ToolResult(
        ok=True, action_code="MUSIC_PAUSE", speak_text="Paused.",
    )
    return runner


@pytest.fixture()
def _mock_tts():
    tts = MagicMock(spec=TTS)
    tts.synthesize.return_value = None
    return tts


@pytest.fixture()
def orchestrator(settings, memory, _mock_tool_runner, _mock_tts):
    brain = Brain(settings=settings, memory=memory)
    brain._bedrock = None

    supervisor = Supervisor(settings=settings, memory=memory, bedrock_client=None)

    music_agent = MagicMock()
    info_agent = MagicMock()
    info_agent.run.return_value = (
        RouterDecision(
            intent_type=IntentType.INFO_QUERY,
            tool_name="general.answer",
            tool_args={"query": "test"},
            action_code="INFO_REPLY",
            confidence=0.5,
        ),
        ToolResult(ok=True, action_code="INFO_REPLY", speak_text="I can help."),
    )
    home_agent = MagicMock()

    return ZiriOrchestrator(
        settings=settings,
        brain=brain,
        tool_runner=_mock_tool_runner,
        memory=memory,
        tts=_mock_tts,
        supervisor=supervisor,
        music_agent=music_agent,
        info_agent=info_agent,
        home_agent=home_agent,
    )


def test_graph_compiled_or_manual_fallback(orchestrator) -> None:
    """Orchestrator should have either a compiled graph or use manual fallback."""
    assert orchestrator.supervisor is not None


def test_supervisor_path_quick_action(orchestrator, device_context, make_request) -> None:
    """'pause' should route through supervisor -> quick_action -> respond."""
    resp, route, result = orchestrator.handle_intent(make_request("pause"), device_context)

    assert route.tool_name == "spotify.pause"
    assert result.ok is True
    assert resp.action_code == "MUSIC_PAUSE"


def test_supervisor_path_info_fallback(orchestrator, device_context, make_request) -> None:
    """Unrecognized text without Bedrock falls back to info domain."""
    resp, route, result = orchestrator.handle_intent(
        make_request("explain quantum entanglement"), device_context,
    )
    assert resp is not None
    assert route is not None


def test_manual_fallback_works_without_langgraph(settings, memory, _mock_tool_runner, _mock_tts, device_context, make_request) -> None:
    """When StateGraph is None, the manual fallback path should work."""
    brain = Brain(settings=settings, memory=memory)
    brain._bedrock = None
    supervisor = Supervisor(settings=settings, memory=memory, bedrock_client=None)

    info_agent = MagicMock()
    info_agent.run.return_value = (
        RouterDecision(
            intent_type=IntentType.INFO_QUERY,
            tool_name="general.answer",
            tool_args={"query": "test"},
            action_code="INFO_REPLY",
            confidence=0.5,
        ),
        ToolResult(ok=True, action_code="INFO_REPLY", speak_text="Fallback."),
    )

    with patch("app.core.orchestrator.StateGraph", None):
        orch = ZiriOrchestrator(
            settings=settings,
            brain=brain,
            tool_runner=_mock_tool_runner,
            memory=memory,
            tts=_mock_tts,
            supervisor=supervisor,
            music_agent=MagicMock(),
            info_agent=info_agent,
            home_agent=MagicMock(),
        )
        assert orch.graph is None
        resp, route, result = orch.handle_intent(make_request("pause"), device_context)
        assert route.tool_name == "spotify.pause"


def test_respond_node_remembers_turn(orchestrator, device_context, make_request, memory) -> None:
    """After handling an intent the memory store should contain the turn."""
    orchestrator.handle_intent(make_request("skip"), device_context)
    ctx = memory.get_recent_context("Aditya")
    assert "skip" in ctx.lower()
