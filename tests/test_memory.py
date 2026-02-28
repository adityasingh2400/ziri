"""Tests for the InMemoryStore and MemoryStore interface."""
from __future__ import annotations

import pytest

from app.core.memory import InMemoryStore, MemoryStore


def test_inmemory_implements_interface() -> None:
    store = InMemoryStore()
    assert isinstance(store, MemoryStore)


def test_remember_and_retrieve(memory: InMemoryStore) -> None:
    memory.remember_turn(
        user_id="u1",
        raw_text="play jazz",
        intent_type="MUSIC_COMMAND",
        tool_name="spotify.play_query",
        assistant_speak="Playing jazz.",
        private_note="",
        context={"tool_payload": {"title": "Jazz FM"}},
    )
    ctx = memory.get_recent_context("u1")
    assert "play jazz" in ctx
    assert "MUSIC_COMMAND" in ctx


def test_get_recent_context_empty(memory: InMemoryStore) -> None:
    assert memory.get_recent_context("nobody") == ""


def test_get_recent_context_respects_limit(memory: InMemoryStore) -> None:
    for i in range(10):
        memory.remember_turn(
            user_id="u1",
            raw_text=f"turn {i}",
            intent_type="INFO_QUERY",
            tool_name="general.answer",
            assistant_speak=f"resp {i}",
            private_note="",
        )
    ctx = memory.get_recent_context("u1", limit=3)
    lines = [l for l in ctx.split("\n") if l.strip()]
    assert len(lines) == 3
    assert "turn 9" in ctx
    assert "turn 7" in ctx


def test_get_last_music_context(memory: InMemoryStore) -> None:
    memory.remember_turn(
        user_id="u1",
        raw_text="play Uzi",
        intent_type="MUSIC_COMMAND",
        tool_name="spotify.play_query",
        assistant_speak="Playing",
        private_note="",
        context={"tool_payload": {"title": "XO TOUR Llif3"}},
    )
    memory.remember_turn(
        user_id="u1",
        raw_text="what time is it",
        intent_type="INFO_QUERY",
        tool_name="time.now",
        assistant_speak="3pm",
        private_note="",
    )
    last = memory.get_last_music_context("u1")
    assert last is not None
    assert last["tool_payload"]["title"] == "XO TOUR Llif3"


def test_get_last_music_context_none(memory: InMemoryStore) -> None:
    assert memory.get_last_music_context("nobody") is None


def test_user_isolation(memory: InMemoryStore) -> None:
    memory.remember_turn(
        user_id="alice",
        raw_text="hello from alice",
        intent_type="INFO_QUERY",
        tool_name="general.answer",
        assistant_speak="hi",
        private_note="",
    )
    assert memory.get_recent_context("bob") == ""
    assert "alice" in memory.get_recent_context("alice")


def test_maxlen_eviction() -> None:
    store = InMemoryStore()
    for i in range(25):
        store.remember_turn(
            user_id="u1",
            raw_text=f"msg {i}",
            intent_type="INFO_QUERY",
            tool_name="general.answer",
            assistant_speak=f"r {i}",
            private_note="",
        )
    ctx = store.get_recent_context("u1", limit=100)
    assert "msg 0" not in ctx
    assert "msg 24" in ctx


def test_semantic_context_base_returns_empty() -> None:
    """Base MemoryStore.get_semantic_context returns empty string."""
    store = InMemoryStore()
    assert store.get_semantic_context("u1", [0.1] * 1536) == ""
