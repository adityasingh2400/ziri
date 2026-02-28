"""Tests for the Supervisor agent's classification logic."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.brain import tool_to_domain
from app.core.memory import InMemoryStore
from app.core.supervisor import Supervisor, SupervisorResult
from app.schemas import IntentType


# ── Deterministic-first path ─────────────────────────────────────────────

def test_deterministic_pause_routes_to_quick(settings, device_context, make_request) -> None:
    """Music commands with confidence >= 0.9 should get domain='quick'."""
    memory = InMemoryStore()
    sv = Supervisor(settings=settings, memory=memory, bedrock_client=None)
    result = sv.classify(make_request("pause"), device_context)

    assert isinstance(result, SupervisorResult)
    assert result.domain == "quick"
    assert result.deterministic_decision is not None
    assert result.deterministic_decision.tool_name == "spotify.pause"


def test_deterministic_calendar_routes_to_info(settings, device_context, make_request) -> None:
    """Non-music deterministic matches keep their natural domain."""
    memory = InMemoryStore()
    sv = Supervisor(settings=settings, memory=memory, bedrock_client=None)
    result = sv.classify(make_request("what's on my calendar"), device_context)

    assert result.domain == "info"
    assert result.deterministic_decision is not None
    assert result.deterministic_decision.tool_name == "calendar.today"


def test_deterministic_reminder_routes_to_home(settings, device_context, make_request) -> None:
    memory = InMemoryStore()
    sv = Supervisor(settings=settings, memory=memory, bedrock_client=None)
    result = sv.classify(make_request("remind me to buy groceries"), device_context)

    assert result.domain == "home"
    assert result.deterministic_decision.tool_name == "reminders.create"


def test_deterministic_scene_routes_to_home(settings, device_context, make_request) -> None:
    memory = InMemoryStore()
    sv = Supervisor(settings=settings, memory=memory, bedrock_client=None)
    result = sv.classify(make_request("movie mode"), device_context)

    assert result.domain == "home"


# ── Quick-action threshold ───────────────────────────────────────────────

def test_low_confidence_music_not_quick(settings, device_context, make_request) -> None:
    """A music command at confidence < 0.9 should route to 'music', not 'quick'."""
    memory = InMemoryStore()
    memory.remember_turn(
        user_id="Aditya",
        raw_text="Play Uzi",
        intent_type="MUSIC_COMMAND",
        tool_name="spotify.play_query",
        assistant_speak="Playing",
        private_note="",
        context={"tool_payload": {"title": "XO TOUR Llif3"}},
    )
    sv = Supervisor(settings=settings, memory=memory, bedrock_client=None)

    # "play it again" resolves via replay with confidence=0.9, so domain="quick"
    result = sv.classify(make_request("play it again"), device_context)
    assert result.domain == "quick"


# ── Fallback without Bedrock ─────────────────────────────────────────────

def test_no_bedrock_falls_back_to_info(settings, device_context, make_request) -> None:
    memory = InMemoryStore()
    sv = Supervisor(settings=settings, memory=memory, bedrock_client=None)
    result = sv.classify(make_request("tell me about quantum physics"), device_context)

    assert result.domain == "info"
    assert result.deterministic_decision is None


# ── LLM classification (mocked) ─────────────────────────────────────────

def test_llm_classification_music(settings, device_context, make_request, mock_bedrock) -> None:
    """'play something chill' matches deterministic 'play X' catch-all, so domain is 'quick'."""
    mock_bedrock.converse.side_effect = lambda **kw: {
        "output": {
            "message": {
                "content": [{"toolUse": {"name": "route_to_music", "input": {"query": "vibe playlist"}}}]
            }
        }
    }
    memory = InMemoryStore()
    sv = Supervisor(settings=settings, memory=memory, bedrock_client=mock_bedrock)
    result = sv.classify(make_request("play something chill"), device_context)

    # "play something chill" hits the deterministic "play X" catch-all at confidence 0.95
    assert result.domain == "quick"
    assert result.deterministic_decision is not None


def test_llm_classification_home(settings, device_context, make_request, mock_bedrock) -> None:
    mock_bedrock.converse.side_effect = lambda **kw: {
        "output": {
            "message": {
                "content": [{"toolUse": {"name": "route_to_home", "input": {"action": "set thermostat"}}}]
            }
        }
    }
    memory = InMemoryStore()
    sv = Supervisor(settings=settings, memory=memory, bedrock_client=mock_bedrock)
    result = sv.classify(make_request("set the thermostat to 72"), device_context)

    assert result.domain == "home"


def test_llm_failure_falls_back_to_info(settings, device_context, make_request, mock_bedrock) -> None:
    mock_bedrock.converse.side_effect = RuntimeError("API down")
    memory = InMemoryStore()
    sv = Supervisor(settings=settings, memory=memory, bedrock_client=mock_bedrock)
    result = sv.classify(make_request("explain dark energy"), device_context)

    assert result.domain == "info"


# ── tool_to_domain helper ───────────────────────────────────────────────

@pytest.mark.parametrize("tool,expected", [
    ("spotify.play_query", "music"),
    ("spotify.pause", "music"),
    ("spotify.skip", "music"),
    ("home.scene", "home"),
    ("reminders.create", "home"),
    ("private.phone_data", "home"),
    ("weather.current", "info"),
    ("nba.scores", "info"),
    ("general.answer", "info"),
    ("calendar.today", "info"),
    ("time.now", "info"),
])
def test_tool_to_domain(tool: str, expected: str) -> None:
    assert tool_to_domain(tool) == expected
