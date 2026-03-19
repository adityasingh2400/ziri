"""Tests for the linear route-execute-respond pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.memory import InMemoryStore
from app.core.orchestrator import ZiriOrchestrator, ZiriState
from app.core.brain import Brain
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
    
    # Mock brain.route_intent to bypass actual bedrock/regex logic for orchestrator tests
    brain.route_intent = MagicMock()
    brain.route_intent.return_value = RouterDecision(
        intent_type=IntentType.MUSIC_COMMAND,
        tool_name="spotify.pause",
        tool_args={},
        action_code="MUSIC_PAUSE",
        confidence=1.0,
    )

    return ZiriOrchestrator(
        settings=settings,
        brain=brain,
        tool_runner=_mock_tool_runner,
        memory=memory,
        tts=_mock_tts,
    )


def test_handle_intent_routes_and_executes(orchestrator, device_context, make_request) -> None:
    """Orchestrator should route through brain and pass decision to tool runner."""
    resp, route, result = orchestrator.handle_intent(make_request("pause"), device_context)

    orchestrator.brain.route_intent.assert_called_once()
    orchestrator.tool_runner.run.assert_called_once()
    assert route.tool_name == "spotify.pause"
    assert result.ok is True
    assert resp.action_code == "MUSIC_PAUSE"


def test_respond_node_remembers_turn(orchestrator, device_context, make_request, memory) -> None:
    """After handling an intent the memory store should contain the turn."""
    orchestrator.handle_intent(make_request("skip"), device_context)
    ctx = memory.get_recent_context("Aditya")
    assert "skip" in ctx.lower()

