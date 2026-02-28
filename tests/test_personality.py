"""Tests for the personality module's quick-reply logic."""
from __future__ import annotations

import pytest

from app.core.personality import QUICK_REPLIES, QUICK_REPLY_ACTIONS, rewrite_response


@pytest.mark.parametrize("action_code", list(QUICK_REPLY_ACTIONS))
def test_quick_reply_returns_from_pool(action_code: str) -> None:
    result = rewrite_response(
        bedrock_client=None,
        model_id="",
        raw_text="some raw text",
        action_code=action_code,
        user_text="test",
    )
    assert result in QUICK_REPLIES[action_code]


def test_non_quick_action_passes_through() -> None:
    result = rewrite_response(
        bedrock_client=None,
        model_id="",
        raw_text="The weather is sunny and 72F.",
        action_code="WEATHER_CURRENT",
        user_text="what's the weather",
    )
    assert result == "The weather is sunny and 72F."


def test_empty_text_returns_empty() -> None:
    result = rewrite_response(
        bedrock_client=None,
        model_id="",
        raw_text="",
        action_code="MUSIC_PAUSE",
        user_text="pause",
    )
    assert result == ""


def test_unknown_action_passes_through() -> None:
    result = rewrite_response(
        bedrock_client=None,
        model_id="",
        raw_text="Here are your results.",
        action_code="SEARCH_RESULTS",
        user_text="search",
    )
    assert result == "Here are your results."
