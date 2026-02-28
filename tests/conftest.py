"""Shared fixtures for the Ziri test suite."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.core.device_registry import DeviceContext
from app.core.memory import InMemoryStore
from app.schemas import IntentRequest
from app.settings import Settings


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def settings() -> Settings:
    """Minimal settings that do not require real API keys."""
    return Settings(
        supabase_url=None,
        supabase_service_role_key=None,
        elevenlabs_api_key=None,
        enable_polly=False,
        semantic_memory_enabled=False,
        return_audio_url=False,
    )


@pytest.fixture()
def memory() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture()
def device_context() -> DeviceContext:
    return DeviceContext(
        device_id="iPhone_Kitchen",
        room_name="Kitchen",
        default_speaker="Living_Room_Sonos",
        spotify_device_id=None,
    )


@pytest.fixture()
def intent_request() -> IntentRequest:
    return _make_request("hello")


def _make_request(text: str, user_id: str = "Aditya") -> IntentRequest:
    return IntentRequest(
        user_id=user_id,
        device_id="iPhone_Kitchen",
        room="Kitchen",
        raw_text=text,
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture()
def make_request():
    """Factory fixture for building IntentRequest objects."""
    return _make_request


@pytest.fixture()
def mock_bedrock() -> MagicMock:
    """A mock Bedrock runtime client that returns a plausible tool-use response."""
    client = MagicMock()

    def _default_converse(**kwargs):
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "route_to_info",
                                "input": {"query": "test"},
                            }
                        }
                    ]
                }
            }
        }

    client.converse.side_effect = _default_converse
    return client


@pytest.fixture()
def routing_eval_cases() -> list[dict[str, Any]]:
    """Load the 25 JSONL eval cases from fixtures."""
    path = FIXTURES_DIR / "routing_eval.jsonl"
    cases: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases
