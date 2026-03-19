"""Tests for Spotify skip_next dead-end handling (single-track / empty queue)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.integrations.spotify_controller import SpotifyController
from app.settings import Settings


def _minimal_settings() -> Settings:
    return Settings(
        spotify_client_id="x",
        spotify_client_secret="y",
        spotify_redirect_uri="http://localhost/cb",
    )


@pytest.fixture
def controller(monkeypatch: pytest.MonkeyPatch) -> SpotifyController:
    monkeypatch.setattr(
        "app.integrations.spotify_controller.spotify",
        MagicMock(),
        raising=False,
    )
    monkeypatch.setattr("app.integrations.spotify_controller.time.sleep", lambda _s: None)
    return SpotifyController(_minimal_settings())


def test_skip_next_when_next_track_plays_returns_music_skip(controller: SpotifyController) -> None:
    client = MagicMock()
    playing = {
        "is_playing": True,
        "item": {"id": "track_b", "name": "Next", "artists": [{"name": "Artist"}]},
    }
    client.current_playback.side_effect = [
        {"is_playing": True, "item": {"id": "track_a", "artists": [{"name": "Artist"}]}},
        playing,
    ]

    controller._build_client = MagicMock(return_value=(client, True))  # type: ignore[method-assign]
    controller._ensure_active_device = MagicMock(return_value=None)  # type: ignore[method-assign]

    result = controller.skip_next()

    assert result.ok is True
    assert result.action_code == "MUSIC_SKIP"
    client.next_track.assert_called_once()


def test_skip_next_same_track_still_playing_is_dead_end(controller: SpotifyController) -> None:
    """Spotify often keeps the same track as is_playing briefly when there is no 'next'."""
    client = MagicMock()
    before = {
        "is_playing": True,
        "item": {
            "id": "track_a",
            "name": "Money Longer",
            "artists": [{"name": "Lil Uzi Vert"}],
        },
    }
    stale = {
        "is_playing": True,
        "item": {
            "id": "track_a",
            "name": "Money Longer",
            "artists": [{"name": "Lil Uzi Vert"}],
        },
    }
    client.current_playback.side_effect = [before] + [stale] * 25

    controller._build_client = MagicMock(return_value=(client, True))  # type: ignore[method-assign]
    controller._ensure_active_device = MagicMock(return_value=None)  # type: ignore[method-assign]

    result = controller.skip_next()

    assert result.ok is True
    assert result.action_code == "MUSIC_SKIP_NO_NEXT"
    assert "paused" in result.speak_text.lower()


def test_skip_next_dead_end_returns_music_skip_no_next(controller: SpotifyController) -> None:
    client = MagicMock()
    before = {
        "is_playing": True,
        "item": {
            "id": "track_a",
            "name": "Money Longer",
            "artists": [{"name": "Lil Uzi Vert"}],
        },
    }
    stopped = {"is_playing": False, "item": None}
    client.current_playback.side_effect = [before] + [stopped] * 20

    controller._build_client = MagicMock(return_value=(client, True))  # type: ignore[method-assign]
    controller._ensure_active_device = MagicMock(return_value=None)  # type: ignore[method-assign]

    result = controller.skip_next()

    assert result.ok is True
    assert result.action_code == "MUSIC_SKIP_NO_NEXT"
    assert "paused" in result.speak_text.lower()
    assert "Lil Uzi Vert" in result.speak_text
    assert "playlist" in result.speak_text.lower()
    client.next_track.assert_called_once()
