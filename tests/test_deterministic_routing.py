"""Parametrized tests for the deterministic phrase-matching router.

Covers every phrase table in brain.py's ``deterministic_route()`` to ensure
the zero-LLM fast path resolves correctly without any Bedrock calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.brain import deterministic_route
from app.schemas import IntentType


# ── Pause ────────────────────────────────────────────────────────────────

_PAUSE = [
    "pause", "stop", "stop playing", "stop music", "stop the music",
    "pause music", "pause the music", "shut up", "be quiet", "mute",
    "hold", "stop aux", "cut the music", "silence", "pause it",
    "stop it", "stop the song", "pause the song", "kill the music",
    "turn it off", "turn off music", "enough", "that's enough",
]


@pytest.mark.parametrize("phrase", _PAUSE)
def test_pause(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "spotify.pause"
    assert d.intent_type == IntentType.MUSIC_COMMAND


# ── Resume ───────────────────────────────────────────────────────────────

_RESUME = [
    "resume", "unpause", "continue", "continue playing", "keep playing",
    "start again", "go on", "resume playing", "resume music",
    "resume playback", "unpause music", "play again",
]


@pytest.mark.parametrize("phrase", _RESUME)
def test_resume(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "spotify.resume"


# ── Skip ─────────────────────────────────────────────────────────────────

_SKIP = [
    "skip", "next", "next song", "next track", "skip this",
    "skip song", "skip track", "play next", "go next", "skip it",
    "next one", "skip this one", "skip this song",
]


@pytest.mark.parametrize("phrase", _SKIP)
def test_skip(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "spotify.skip"


# ── Previous ─────────────────────────────────────────────────────────────

_PREVIOUS = [
    "previous", "go back", "last song", "last track", "previous song",
    "previous track", "rewind", "play previous", "back", "go back one",
    "previous one", "play the last song",
]


@pytest.mark.parametrize("phrase", _PREVIOUS)
def test_previous(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "spotify.previous"


# ── Volume ───────────────────────────────────────────────────────────────

_VOLUME_UP = [
    "louder", "volume up", "turn it up", "crank it", "crank it up",
    "bump it up", "raise volume", "raise the volume", "more volume",
    "pump it up", "turn up the volume", "bit louder", "a bit louder",
]

_VOLUME_DOWN = [
    "quieter", "softer", "volume down", "turn it down", "lower volume",
    "lower the volume", "less volume", "bring it down",
    "turn down the volume", "bit quieter", "a bit quieter", "not so loud",
]

_VOLUME_MAX = [
    "volume max", "max volume", "full volume", "volume full",
    "turn it all the way up", "crank it to the max", "maximum volume",
]

_VOLUME_CALM = [
    "volume calm", "calm volume", "quiet mode", "background volume",
    "background music", "chill volume", "low volume",
]


@pytest.mark.parametrize("phrase", _VOLUME_UP)
def test_volume_up(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "spotify.adjust_volume"
    assert d.tool_args["delta_percent"] == 20


@pytest.mark.parametrize("phrase", _VOLUME_DOWN)
def test_volume_down(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "spotify.adjust_volume"
    assert d.tool_args["delta_percent"] == -20


@pytest.mark.parametrize("phrase", _VOLUME_MAX)
def test_volume_max(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "spotify.set_volume"
    assert d.tool_args["percent"] == 100


@pytest.mark.parametrize("phrase", _VOLUME_CALM)
def test_volume_calm(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "spotify.set_volume"
    assert d.tool_args["percent"] == 35


@pytest.mark.parametrize("text,expected", [
    ("set volume to 50", 50),
    ("volume to 80", 80),
    ("set volume 30", 30),
    ("volume at 100", 100),
    ("set volume to 0", 0),
])
def test_set_volume(text: str, expected: int) -> None:
    d = deterministic_route(text)
    assert d is not None
    assert d.tool_name == "spotify.set_volume"
    assert d.tool_args["percent"] == expected


# ── Shuffle / Repeat ─────────────────────────────────────────────────────

_SHUFFLE_ON = [
    "shuffle", "shuffle on", "turn on shuffle", "mix it up",
    "randomize", "shuffle mode", "shuffle the music", "shuffle songs",
    "enable shuffle",
]

_SHUFFLE_OFF = [
    "shuffle off", "turn off shuffle", "stop shuffle", "no shuffle",
    "unshuffle", "disable shuffle", "stop shuffling",
]


@pytest.mark.parametrize("phrase", _SHUFFLE_ON)
def test_shuffle_on(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "spotify.shuffle"
    assert d.tool_args["state"] is True


@pytest.mark.parametrize("phrase", _SHUFFLE_OFF)
def test_shuffle_off(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "spotify.shuffle"
    assert d.tool_args["state"] is False


_REPEAT_ON = [
    "repeat", "repeat on", "loop", "loop this", "on repeat",
    "repeat track", "repeat song", "repeat this", "loop this song",
    "put this on repeat", "loop it",
]

_REPEAT_OFF = [
    "repeat off", "stop repeating", "no repeat", "no loop",
    "stop repeat", "turn off repeat", "disable repeat", "stop looping",
]


@pytest.mark.parametrize("phrase", _REPEAT_ON)
def test_repeat_on(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "spotify.repeat"
    assert d.tool_args["mode"] == "track"


@pytest.mark.parametrize("phrase", _REPEAT_OFF)
def test_repeat_off(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "spotify.repeat"
    assert d.tool_args["mode"] == "off"


# ── What's playing / Like ───────────────────────────────────────────────

_WHATS_PLAYING = [
    "what's playing", "what is playing", "current song", "what song",
    "what track", "who is this", "who's singing", "what am i listening to",
    "song name", "track name", "what's this song", "what is this song",
    "who sings this", "what's on", "what song is this",
]


@pytest.mark.parametrize("phrase", _WHATS_PLAYING)
def test_whats_playing(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "spotify.current_track"


_LIKE = [
    "like this", "like this song", "save this", "save this song",
    "heart this", "add to liked", "save song", "like it",
    "heart this song", "like the song", "save the song", "like track",
    "add to liked songs",
]


@pytest.mark.parametrize("phrase", _LIKE)
def test_like(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "spotify.like"


# ── Replay with context ─────────────────────────────────────────────────

def test_replay_with_context() -> None:
    ctx = {"tool_payload": {"title": "Blinding Lights"}}
    d = deterministic_route("play it again", ctx)
    assert d is not None
    assert d.tool_name == "spotify.play_query"
    assert d.tool_args["query"] == "Blinding Lights"


def test_replay_without_context() -> None:
    d = deterministic_route("replay")
    assert d is not None
    assert d.tool_name == "spotify.play_query"
    assert d.tool_args["query"] == "my last track"


# ── Prefix-based music routes ───────────────────────────────────────────

@pytest.mark.parametrize("text,expected_tool,expected_query", [
    ("queue Bohemian Rhapsody", "spotify.queue", "Bohemian Rhapsody"),
    ("add to queue Money Trees", "spotify.queue", "Money Trees"),
    ("play artist Taylor Swift", "spotify.play_artist", "Taylor Swift"),
    ("play playlist chill vibes", "spotify.play_playlist", "chill vibes"),
    ("shuffle my playlist workout", "spotify.play_playlist", "workout"),
    ("shuffle songs by Kendrick", "spotify.play_artist", "Kendrick"),
])
def test_prefix_music_routes(text: str, expected_tool: str, expected_query: str) -> None:
    d = deterministic_route(text)
    assert d is not None
    assert d.tool_name == expected_tool
    assert d.tool_args["query"] == expected_query


def test_after_skip_dead_end_hint_maps_bare_playlist_name() -> None:
    d = deterministic_route(
        "exotic melodies",
        {},
        listener_route_hint="after_skip_no_next",
    )
    assert d is not None
    assert d.tool_name == "spotify.play_playlist"
    assert d.tool_args["query"] == "exotic melodies"
    assert d.tool_args.get("shuffle") is True


def test_after_skip_hint_ignores_one_word_yeah() -> None:
    assert deterministic_route(
        "yeah",
        {},
        listener_route_hint="after_skip_no_next",
    ) is None


def test_shuffle_artist_after_assistant_echo_in_transcript() -> None:
    """Follow-up STT may include TTS bleed; match the user's trailing shuffle request."""
    blob = (
        "There wasn't another track lined up after that one, so Spotify paused. "
        "Want me to play more from Lou Val or shuffle one of your playlists? "
        "Yeah, shuffle songs by Lou Val"
    )
    d = deterministic_route(blob)
    assert d is not None
    assert d.tool_name == "spotify.play_artist"
    assert d.tool_args.get("shuffle") is True
    assert "Lou Val" in d.tool_args["query"]


def test_shuffle_playlist_inline_after_filler_phrase() -> None:
    """'playlist' contains substring 'play' — must not route to play_query."""
    d = deterministic_route(
        "Yeah, could you shuffle my playlist exotic melodies?",
    )
    assert d is not None
    assert d.tool_name == "spotify.play_playlist"
    assert d.tool_args["query"].lower() == "exotic melodies"
    assert d.tool_args.get("shuffle") is True


def test_playlist_request_not_play_query_catchall() -> None:
    d = deterministic_route("shuffle the playlist workout mix")
    assert d is not None
    assert d.tool_name == "spotify.play_playlist"
    assert d.tool_args["query"] == "workout mix"
    assert d.tool_args.get("shuffle") is True


def test_play_my_playlist_inline() -> None:
    d = deterministic_route("hey could you play my playlist focus")
    assert d is not None
    assert d.tool_name == "spotify.play_playlist"
    assert d.tool_args["query"] == "focus"
    assert not d.tool_args.get("shuffle")


# ── Catch-all "play X" ──────────────────────────────────────────────────

def test_play_catchall() -> None:
    d = deterministic_route("play some jazz")
    assert d is not None
    assert d.tool_name == "spotify.play_query"
    assert "jazz" in d.tool_args["query"].lower()


# ── Non-music deterministic routes ──────────────────────────────────────

_WEATHER = [
    "weather", "what's the weather", "temperature",
    "is it raining", "do i need a jacket",
]


@pytest.mark.parametrize("phrase", _WEATHER)
def test_weather(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "weather.current"
    assert d.intent_type == IntentType.WEATHER


_SUNRISE = ["sunrise", "sunset", "when is sunrise", "sun times"]


@pytest.mark.parametrize("phrase", _SUNRISE)
def test_sunrise_sunset(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "weather.sunrise_sunset"


_NBA = ["nba scores", "basketball scores", "nba tonight"]


@pytest.mark.parametrize("phrase", _NBA)
def test_nba_scores(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "nba.scores"


def test_nba_team() -> None:
    d = deterministic_route("how did the Lakers game")
    assert d is not None
    assert d.tool_name == "nba.team"
    assert d.tool_args["team"] == "Lakers"


_NEWS = ["news", "headlines", "what's happening", "brief me"]


@pytest.mark.parametrize("phrase", _NEWS)
def test_news_headlines(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "news.headlines"


def test_news_topic() -> None:
    d = deterministic_route("news about AI")
    assert d is not None
    assert d.tool_name == "news.topic"
    assert d.tool_args["query"] == "AI"


_CALENDAR = ["calendar", "meeting", "what's on my calendar", "my schedule"]


@pytest.mark.parametrize("phrase", _CALENDAR)
def test_calendar(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "calendar.today"


def test_reminder() -> None:
    d = deterministic_route("remind me to buy milk")
    assert d is not None
    assert d.tool_name == "reminders.create"
    assert d.tool_args["text"] == "buy milk"


def test_scene() -> None:
    d = deterministic_route("movie mode")
    assert d is not None
    assert d.tool_name == "home.scene"


def test_private_phone() -> None:
    d = deterministic_route("read my texts")
    assert d is not None
    assert d.tool_name == "private.phone_data"
    assert d.requires_private_display is True


_TIME = ["what time is it", "what's the time", "current time"]


@pytest.mark.parametrize("phrase", _TIME)
def test_time(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "time.now"


_DATE = ["what day is it", "what's the date", "today's date"]


@pytest.mark.parametrize("phrase", _DATE)
def test_date(phrase: str) -> None:
    d = deterministic_route(phrase)
    assert d is not None
    assert d.tool_name == "time.date"


# ── No match returns None ───────────────────────────────────────────────

@pytest.mark.parametrize("phrase", [
    "tell me a joke",
    "what is the meaning of life",
    "how do black holes form",
])
def test_no_match(phrase: str) -> None:
    assert deterministic_route(phrase) is None


# ── JSONL eval fixture integration ──────────────────────────────────────

def _load_eval_cases() -> list[tuple[str, str]]:
    path = Path(__file__).parent / "fixtures" / "routing_eval.jsonl"
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                cases.append((c["input"], c["expected_tool"]))
    return cases


@pytest.mark.parametrize("text,expected_tool", _load_eval_cases())
def test_eval_fixture(text: str, expected_tool: str) -> None:
    """All 25 eval fixture cases should resolve via deterministic routing."""
    d = deterministic_route(text)
    if expected_tool == "general.answer":
        # general.answer is the LLM fallback -- deterministic returns None
        if d is None:
            return
    assert d is not None, f"No deterministic route for: {text!r}"
    assert d.tool_name == expected_tool
