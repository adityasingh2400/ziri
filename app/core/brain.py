from __future__ import annotations

import json
import logging
import re
from typing import Any

import boto3

from app.settings import Settings
from app.core.device_registry import DeviceContext
from app.core.memory import MemoryStore
from app.schemas import IntentRequest, IntentType, RouterDecision

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """\
You are Ziri, the intent router for a distributed home voice OS.

Given the user request and context, choose exactly ONE tool to call.
Rules:
- Prefer music tools for playback/volume/skip/pause actions.
- If user asks for private phone-only data (texts, OTP, sensitive info), use private_phone_data.
- For pronouns like "it" / "that", use memory_context to resolve the previous media command.
- Keep speak_text concise and spoken-friendly.
"""

# ======================================================================
# Bedrock Converse toolConfig definitions
# ======================================================================

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {"toolSpec": {"name": "spotify_play_query", "description": "Search Spotify and start playing a track.", "inputSchema": {"json": {"type": "object", "properties": {"query": {"type": "string", "description": "What to search for"}}, "required": ["query"]}}}},
    {"toolSpec": {"name": "spotify_play_artist", "description": "Play an artist's top tracks on Spotify.", "inputSchema": {"json": {"type": "object", "properties": {"query": {"type": "string", "description": "Artist name"}}, "required": ["query"]}}}},
    {"toolSpec": {"name": "spotify_play_playlist", "description": "Search and play a playlist on Spotify.", "inputSchema": {"json": {"type": "object", "properties": {"query": {"type": "string", "description": "Playlist name"}}, "required": ["query"]}}}},
    {"toolSpec": {"name": "spotify_pause", "description": "Pause the currently playing Spotify track.", "inputSchema": {"json": {"type": "object", "properties": {}}}}},
    {"toolSpec": {"name": "spotify_resume", "description": "Resume/unpause Spotify playback.", "inputSchema": {"json": {"type": "object", "properties": {}}}}},
    {"toolSpec": {"name": "spotify_skip", "description": "Skip to the next track on Spotify.", "inputSchema": {"json": {"type": "object", "properties": {}}}}},
    {"toolSpec": {"name": "spotify_previous", "description": "Go back to the previous track on Spotify.", "inputSchema": {"json": {"type": "object", "properties": {}}}}},
    {"toolSpec": {"name": "spotify_adjust_volume", "description": "Raise or lower the Spotify volume by a delta.", "inputSchema": {"json": {"type": "object", "properties": {"delta_percent": {"type": "integer", "description": "Positive to raise, negative to lower"}}, "required": ["delta_percent"]}}}},
    {"toolSpec": {"name": "spotify_set_volume", "description": "Set the Spotify volume to an exact percentage.", "inputSchema": {"json": {"type": "object", "properties": {"percent": {"type": "integer", "description": "0-100"}}, "required": ["percent"]}}}},
    {"toolSpec": {"name": "spotify_shuffle", "description": "Toggle shuffle mode on Spotify.", "inputSchema": {"json": {"type": "object", "properties": {"state": {"type": "boolean", "description": "true=on, false=off"}}, "required": ["state"]}}}},
    {"toolSpec": {"name": "spotify_repeat", "description": "Set repeat mode on Spotify.", "inputSchema": {"json": {"type": "object", "properties": {"mode": {"type": "string", "description": "off, track, or context"}}, "required": ["mode"]}}}},
    {"toolSpec": {"name": "spotify_current_track", "description": "Get info about the currently playing Spotify track.", "inputSchema": {"json": {"type": "object", "properties": {}}}}},
    {"toolSpec": {"name": "spotify_queue", "description": "Search for a track and add it to the Spotify queue.", "inputSchema": {"json": {"type": "object", "properties": {"query": {"type": "string", "description": "What to search and queue"}}, "required": ["query"]}}}},
    {"toolSpec": {"name": "spotify_like", "description": "Like/save the currently playing track to the user's library.", "inputSchema": {"json": {"type": "object", "properties": {}}}}},
    {"toolSpec": {"name": "calendar_today", "description": "Retrieve today's calendar events.", "inputSchema": {"json": {"type": "object", "properties": {}}}}},
    {"toolSpec": {"name": "reminders_create", "description": "Create a new reminder.", "inputSchema": {"json": {"type": "object", "properties": {"text": {"type": "string", "description": "The reminder content"}, "due_at": {"type": "string", "description": "Optional ISO8601 due date/time"}}, "required": ["text"]}}}},
    {"toolSpec": {"name": "home_scene", "description": "Activate a home automation scene.", "inputSchema": {"json": {"type": "object", "properties": {"scene_name": {"type": "string", "description": "Name of the scene"}}, "required": ["scene_name"]}}}},
    {"toolSpec": {"name": "private_phone_data", "description": "Fetch private/sensitive data -- displayed only on phone, never spoken aloud.", "inputSchema": {"json": {"type": "object", "properties": {"query": {"type": "string", "description": "What the user is asking for"}}, "required": ["query"]}}}},
    {"toolSpec": {"name": "general_answer", "description": "General-purpose answer for questions that don't match other tools.", "inputSchema": {"json": {"type": "object", "properties": {"query": {"type": "string", "description": "The user's question"}}, "required": ["query"]}}}},
    {"toolSpec": {"name": "weather_current", "description": "Get the current weather conditions (temperature, conditions, wind).", "inputSchema": {"json": {"type": "object", "properties": {}}}}},
    {"toolSpec": {"name": "weather_sunrise_sunset", "description": "Get today's sunrise and sunset times.", "inputSchema": {"json": {"type": "object", "properties": {}}}}},
    {"toolSpec": {"name": "nba_scores", "description": "Get today's NBA game scores.", "inputSchema": {"json": {"type": "object", "properties": {}}}}},
    {"toolSpec": {"name": "nba_team", "description": "Get the score for a specific NBA team's game today.", "inputSchema": {"json": {"type": "object", "properties": {"team": {"type": "string", "description": "Team name, e.g. Lakers, Celtics"}}, "required": ["team"]}}}},
    {"toolSpec": {"name": "news_headlines", "description": "Get top news headlines.", "inputSchema": {"json": {"type": "object", "properties": {"category": {"type": "string", "description": "Category: general, business, technology, sports, entertainment, health, science"}}}}}},
    {"toolSpec": {"name": "news_topic", "description": "Search news about a specific topic.", "inputSchema": {"json": {"type": "object", "properties": {"query": {"type": "string", "description": "Topic to search"}}, "required": ["query"]}}}},
]

_TOOL_NAME_MAP: dict[str, str] = {
    "spotify_play_query": "spotify.play_query",
    "spotify_play_artist": "spotify.play_artist",
    "spotify_play_playlist": "spotify.play_playlist",
    "spotify_pause": "spotify.pause",
    "spotify_resume": "spotify.resume",
    "spotify_skip": "spotify.skip",
    "spotify_previous": "spotify.previous",
    "spotify_adjust_volume": "spotify.adjust_volume",
    "spotify_set_volume": "spotify.set_volume",
    "spotify_shuffle": "spotify.shuffle",
    "spotify_repeat": "spotify.repeat",
    "spotify_current_track": "spotify.current_track",
    "spotify_queue": "spotify.queue",
    "spotify_like": "spotify.like",
    "calendar_today": "calendar.today",
    "reminders_create": "reminders.create",
    "home_scene": "home.scene",
    "private_phone_data": "private.phone_data",
    "general_answer": "general.answer",
    "weather_current": "weather.current",
    "weather_sunrise_sunset": "weather.sunrise_sunset",
    "nba_scores": "nba.scores",
    "nba_team": "nba.team",
    "news_headlines": "news.headlines",
    "news_topic": "news.topic",
}

_TOOL_INTENT_MAP: dict[str, IntentType] = {
    "spotify.play_query": IntentType.MUSIC_COMMAND,
    "spotify.play_artist": IntentType.MUSIC_COMMAND,
    "spotify.play_playlist": IntentType.MUSIC_COMMAND,
    "spotify.pause": IntentType.MUSIC_COMMAND,
    "spotify.resume": IntentType.MUSIC_COMMAND,
    "spotify.skip": IntentType.MUSIC_COMMAND,
    "spotify.previous": IntentType.MUSIC_COMMAND,
    "spotify.adjust_volume": IntentType.MUSIC_COMMAND,
    "spotify.set_volume": IntentType.MUSIC_COMMAND,
    "spotify.shuffle": IntentType.MUSIC_COMMAND,
    "spotify.repeat": IntentType.MUSIC_COMMAND,
    "spotify.current_track": IntentType.MUSIC_COMMAND,
    "spotify.queue": IntentType.MUSIC_COMMAND,
    "spotify.like": IntentType.MUSIC_COMMAND,
    "calendar.today": IntentType.INFO_QUERY,
    "reminders.create": IntentType.PERSONAL_REMINDER,
    "home.scene": IntentType.HOME_SCENE,
    "private.phone_data": IntentType.INFO_QUERY,
    "general.answer": IntentType.INFO_QUERY,
    "weather.current": IntentType.WEATHER,
    "weather.sunrise_sunset": IntentType.WEATHER,
    "nba.scores": IntentType.SPORTS,
    "nba.team": IntentType.SPORTS,
    "news.headlines": IntentType.NEWS,
    "news.topic": IntentType.NEWS,
}

_ACTION_CODES: dict[str, str] = {
    "spotify.play_query": "MUSIC_START",
    "spotify.play_artist": "MUSIC_START",
    "spotify.play_playlist": "MUSIC_START",
    "spotify.pause": "MUSIC_PAUSE",
    "spotify.resume": "MUSIC_RESUME",
    "spotify.skip": "MUSIC_SKIP",
    "spotify.previous": "MUSIC_PREVIOUS",
    "spotify.adjust_volume": "MUSIC_VOLUME",
    "spotify.set_volume": "MUSIC_VOLUME",
    "spotify.shuffle": "MUSIC_SHUFFLE",
    "spotify.repeat": "MUSIC_REPEAT",
    "spotify.current_track": "MUSIC_INFO",
    "spotify.queue": "MUSIC_QUEUE",
    "spotify.like": "MUSIC_LIKE",
    "calendar.today": "CALENDAR_READ",
    "reminders.create": "REMINDER_CREATE",
    "home.scene": "SCENE_APPLY",
    "private.phone_data": "PRIVATE_NOTE",
    "general.answer": "INFO_REPLY",
    "weather.current": "WEATHER_CURRENT",
    "weather.sunrise_sunset": "WEATHER_SUN",
    "nba.scores": "NBA_SCORES",
    "nba.team": "NBA_SCORES",
    "news.headlines": "NEWS_HEADLINES",
    "news.topic": "NEWS_TOPIC",
}

# ======================================================================
# Deterministic phrase tables
# ======================================================================

_PAUSE_PHRASES = [
    "pause", "stop", "stop playing", "stop music", "stop the music",
    "pause music", "pause the music", "shut up", "be quiet", "mute",
    "hold", "stop aux", "cut the music", "silence", "pause it",
    "stop it", "stop the song", "pause the song", "kill the music",
    "turn it off", "turn off music", "enough", "that's enough",
]

_RESUME_PHRASES = [
    "resume", "unpause", "continue", "continue playing", "keep playing",
    "start again", "go on", "resume playing", "resume music",
    "resume playback", "unpause music", "play again",
]

_SKIP_PHRASES = [
    "skip", "next", "next song", "next track", "skip this",
    "skip song", "skip track", "play next", "go next", "skip it",
    "next one", "skip this one", "skip this song",
]

_PREVIOUS_PHRASES = [
    "previous", "go back", "last song", "last track", "previous song",
    "previous track", "rewind", "play previous", "back", "go back one",
    "previous one", "play the last song",
]

_VOLUME_UP_PHRASES = [
    "louder", "volume up", "turn it up", "crank it", "crank it up",
    "bump it up", "raise volume", "raise the volume", "more volume",
    "pump it up", "turn up the volume", "bit louder", "a bit louder",
]

_VOLUME_DOWN_PHRASES = [
    "quieter", "softer", "volume down", "turn it down", "lower volume",
    "lower the volume", "less volume", "bring it down",
    "turn down the volume", "bit quieter", "a bit quieter", "not so loud",
]

_SHUFFLE_ON_PHRASES = [
    "shuffle", "shuffle on", "turn on shuffle", "mix it up",
    "randomize", "shuffle mode", "shuffle the music", "shuffle songs",
    "enable shuffle",
]

_SHUFFLE_OFF_PHRASES = [
    "shuffle off", "turn off shuffle", "stop shuffle", "no shuffle",
    "unshuffle", "disable shuffle", "stop shuffling",
]

_REPEAT_ON_PHRASES = [
    "repeat", "repeat on", "loop", "loop this", "on repeat",
    "repeat track", "repeat song", "repeat this", "loop this song",
    "put this on repeat", "loop it",
]

_REPEAT_OFF_PHRASES = [
    "repeat off", "stop repeating", "no repeat", "no loop",
    "stop repeat", "turn off repeat", "disable repeat", "stop looping",
]

_WHATS_PLAYING_PHRASES = [
    "what's playing", "what is playing", "current song", "what song",
    "what track", "who is this", "who's singing", "what am i listening to",
    "song name", "track name", "what's this song", "what is this song",
    "who sings this", "what's on", "what song is this",
]

_LIKE_PHRASES = [
    "like this", "like this song", "save this", "save this song",
    "heart this", "add to liked", "save song", "like it",
    "heart this song", "like the song", "save the song", "like track",
    "add to liked songs",
]

_REPLAY_PHRASES = [
    "play it again", "play that again", "replay", "replay it",
    "one more time", "again", "replay this",
]

_QUEUE_PREFIXES = ["queue ", "add to queue ", "q up "]

_PLAY_ARTIST_PREFIXES = ["play artist ", "put on artist "]
_PLAY_PLAYLIST_PREFIXES = [
    "play playlist ", "put on playlist ", "play my playlist ",
    "play the playlist ",
]

_SHUFFLE_PLAYLIST_PREFIXES = [
    "shuffle my playlist ", "shuffle playlist ", "shuffle the playlist ",
]
_SHUFFLE_ARTIST_PREFIXES = [
    "shuffle songs by ", "shuffle music by ", "shuffle tracks by ",
]

_SET_VOLUME_RE = re.compile(
    r"^(?:set |turn )?volume (?:to |at )?(\d+)(?:\s*%| percent)?$"
)

# Non-music deterministic routes
_CALENDAR_PHRASES = [
    "calendar", "meeting", "meetings", "schedule", "events today",
    "what's on my calendar", "do i have any meetings", "my schedule",
]

_REMINDER_MARKERS = ["remind", "reminder", "remember to"]

_SCENE_PHRASES = [
    "scene", "lights", "movie mode", "goodnight", "good night",
    "turn on lights", "turn off lights", "dim the lights",
]

_PRIVATE_PHRASES = [
    "read my texts", "texts", "messages", "otp", "code from sms",
    "read my messages", "show my texts",
]

_WEATHER_PHRASES = [
    "weather", "what's the weather", "what is the weather",
    "how's the weather", "temperature", "how cold is it",
    "how hot is it", "is it cold", "is it hot", "is it raining",
    "do i need a jacket", "forecast",
]

_SUNRISE_SUNSET_PHRASES = [
    "sunrise", "sunset", "when does the sun set", "when does the sun rise",
    "what time is sunrise", "what time is sunset",
    "when is sunrise", "when is sunset",
    "sun times",
]

_NBA_PHRASES = [
    "nba scores", "nba games", "basketball scores",
    "nba games today", "basketball games today",
    "who's playing tonight", "nba tonight",
]

_NBA_TEAM_PREFIXES = [
    "how did the ", "how are the ", "did the ",
    "score for the ", "score for ",
]
_NBA_TEAM_SUFFIXES = [" score", " game"]

_NEWS_PHRASES = [
    "news", "headlines", "top stories", "what's happening",
    "what's going on", "what's in the news",
    "tell me the news", "news briefing", "brief me",
    "what's going on in the world",
]

_NEWS_TOPIC_PREFIXES = [
    "news about ", "news on ", "what's happening with ",
    "any news about ", "any news on ",
]


class Brain:
    def __init__(self, settings: Settings, memory: MemoryStore) -> None:
        self.settings = settings
        self.memory = memory
        self._bedrock = self._init_bedrock_client()

    def _init_bedrock_client(self) -> Any:
        kwargs: dict[str, Any] = {"region_name": self.settings.aws_region}
        aws_access_key = self.settings.aws_access_key_id or self.settings.aws_access_key
        if aws_access_key and self.settings.aws_secret_access_key:
            kwargs["aws_access_key_id"] = aws_access_key
            kwargs["aws_secret_access_key"] = self.settings.aws_secret_access_key
        try:
            return boto3.client("bedrock-runtime", **kwargs)
        except Exception as exc:
            logger.warning("Bedrock init failed; using heuristic router: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Main entry: deterministic first, Bedrock fallback
    # ------------------------------------------------------------------

    def route_intent(self, req: IntentRequest, device_context: DeviceContext) -> RouterDecision:
        memory_text = self.memory.get_recent_context(req.user_id, limit=self.settings.memory_window)
        last_music_context = self.memory.get_last_music_context(req.user_id) or {}

        deterministic = self._deterministic_route(req, last_music_context)
        if deterministic:
            return deterministic

        if self._bedrock:
            llm_decision = self._route_with_bedrock(req, device_context, memory_text, last_music_context)
            if llm_decision:
                return llm_decision

        return RouterDecision(
            intent_type=IntentType.INFO_QUERY,
            tool_name="general.answer",
            tool_args={"query": req.raw_text},
            action_code="INFO_REPLY",
            speak_text="I heard you. I can help with music, reminders, scenes, or calendar.",
            confidence=0.45,
        )

    # ------------------------------------------------------------------
    # Deterministic phrase matcher
    # ------------------------------------------------------------------

    def _deterministic_route(
        self, req: IntentRequest, last_music_context: dict[str, Any]
    ) -> RouterDecision | None:
        text = req.raw_text.lower().strip()

        # -- Replay (must come before generic "play" catch-all) --
        if any(text == p or text.startswith(p) for p in _REPLAY_PHRASES):
            previous_query = None
            tp = last_music_context.get("tool_payload")
            if isinstance(tp, dict):
                previous_query = tp.get("title") or tp.get("spotify_uri")
            query = previous_query or "my last track"
            return self._decision("spotify.play_query", {"query": query}, confidence=0.9)

        # -- Pause / Stop --
        if self._match_exact(text, _PAUSE_PHRASES):
            return self._decision("spotify.pause", {})

        # -- Resume --
        if self._match_exact(text, _RESUME_PHRASES):
            return self._decision("spotify.resume", {})

        # -- Skip / Next --
        if self._match_exact(text, _SKIP_PHRASES):
            return self._decision("spotify.skip", {})

        # -- Previous --
        if self._match_exact(text, _PREVIOUS_PHRASES):
            return self._decision("spotify.previous", {})

        # -- Volume: exact set --
        vol_match = _SET_VOLUME_RE.match(text)
        if vol_match:
            return self._decision("spotify.set_volume", {"percent": int(vol_match.group(1))})

        # -- Volume up --
        if self._match_exact(text, _VOLUME_UP_PHRASES):
            return self._decision("spotify.adjust_volume", {"delta_percent": 15})

        # -- Volume down --
        if self._match_exact(text, _VOLUME_DOWN_PHRASES):
            return self._decision("spotify.adjust_volume", {"delta_percent": -15})

        # -- Shuffle + playlist / artist (must come before simple shuffle) --
        for prefix in _SHUFFLE_PLAYLIST_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                query = req.raw_text[len(prefix):].strip()
                return self._decision("spotify.play_playlist", {"query": query, "shuffle": True})

        for prefix in _SHUFFLE_ARTIST_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                query = req.raw_text[len(prefix):].strip()
                return self._decision("spotify.play_artist", {"query": query, "shuffle": True})

        # -- Shuffle off (before shuffle on) --
        if self._match_exact(text, _SHUFFLE_OFF_PHRASES):
            return self._decision("spotify.shuffle", {"state": False})

        # -- Shuffle on --
        if self._match_exact(text, _SHUFFLE_ON_PHRASES):
            return self._decision("spotify.shuffle", {"state": True})

        # -- Repeat off (before repeat on) --
        if self._match_exact(text, _REPEAT_OFF_PHRASES):
            return self._decision("spotify.repeat", {"mode": "off"})

        # -- Repeat on --
        if self._match_exact(text, _REPEAT_ON_PHRASES):
            return self._decision("spotify.repeat", {"mode": "track"})

        # -- What's playing --
        if self._match_exact(text, _WHATS_PLAYING_PHRASES):
            return self._decision("spotify.current_track", {})

        # -- Like --
        if self._match_exact(text, _LIKE_PHRASES):
            return self._decision("spotify.like", {})

        # -- Queue --
        for prefix in _QUEUE_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                query = req.raw_text[len(prefix):].strip()
                return self._decision("spotify.queue", {"query": query})

        # -- Play playlist --
        for prefix in _PLAY_PLAYLIST_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                query = req.raw_text[len(prefix):].strip()
                return self._decision("spotify.play_playlist", {"query": query})

        # -- Play artist --
        for prefix in _PLAY_ARTIST_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                query = req.raw_text[len(prefix):].strip()
                return self._decision("spotify.play_artist", {"query": query})

        # -- Private data --
        if self._match_contains(text, _PRIVATE_PHRASES):
            return RouterDecision(
                intent_type=IntentType.INFO_QUERY,
                tool_name="private.phone_data",
                tool_args={"query": req.raw_text},
                action_code="PRIVATE_NOTE",
                speak_text="",
                private_note="Fetching private message summary to display on phone.",
                confidence=0.85,
                requires_private_display=True,
            )

        # -- Calendar --
        if self._match_contains(text, _CALENDAR_PHRASES):
            return self._decision("calendar.today", {})

        # -- Reminders --
        if any(k in text for k in _REMINDER_MARKERS):
            reminder_text = req.raw_text
            marker = "remind me to"
            if marker in text:
                reminder_text = req.raw_text.lower().split(marker, maxsplit=1)[1].strip()
            return RouterDecision(
                intent_type=IntentType.PERSONAL_REMINDER,
                tool_name="reminders.create",
                tool_args={"text": reminder_text},
                action_code="REMINDER_CREATE",
                speak_text="Okay, I will add that reminder.",
                confidence=0.85,
            )

        # -- Home scenes --
        if self._match_contains(text, _SCENE_PHRASES):
            return RouterDecision(
                intent_type=IntentType.HOME_SCENE,
                tool_name="home.scene",
                tool_args={"scene_name": req.raw_text},
                action_code="SCENE_APPLY",
                speak_text="Applying that home scene.",
                confidence=0.8,
            )

        # -- Weather --
        if self._match_contains(text, _WEATHER_PHRASES):
            return self._decision("weather.current", {})

        # -- Sunrise / Sunset --
        if self._match_contains(text, _SUNRISE_SUNSET_PHRASES):
            return self._decision("weather.sunrise_sunset", {})

        # -- NBA team-specific (before general NBA) --
        for prefix in _NBA_TEAM_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                team = req.raw_text[len(prefix):].strip()
                for suf in _NBA_TEAM_SUFFIXES + [""]:
                    if team.lower().endswith(suf):
                        team = team[:len(team)-len(suf)].strip()
                        break
                if team:
                    return self._decision("nba.team", {"team": team})

        # -- NBA scores --
        if self._match_contains(text, _NBA_PHRASES):
            return self._decision("nba.scores", {})

        # -- News topic --
        for prefix in _NEWS_TOPIC_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                query = req.raw_text[len(prefix):].strip()
                return self._decision("news.topic", {"query": query})

        # -- News headlines --
        if self._match_contains(text, _NEWS_PHRASES):
            return self._decision("news.headlines", {})

        # -- Generic "play ..." catch-all (must be last music match) --
        if text.startswith("play ") and len(text) > 5:
            query = req.raw_text[5:].strip() or "top tracks"
            return self._decision("spotify.play_query", {"query": query})

        if "play" in text or "spotify" in text:
            return self._decision("spotify.play_query", {"query": req.raw_text})

        # No deterministic match — fall through to Bedrock
        return None

    @staticmethod
    def _match_exact(text: str, phrases: list[str]) -> bool:
        return text in phrases

    @staticmethod
    def _match_contains(text: str, phrases: list[str]) -> bool:
        return any(p in text for p in phrases)

    def _decision(self, tool_name: str, tool_args: dict[str, Any], confidence: float = 0.95) -> RouterDecision:
        return RouterDecision(
            intent_type=_TOOL_INTENT_MAP.get(tool_name, IntentType.INFO_QUERY),
            tool_name=tool_name,
            tool_args=tool_args,
            action_code=_ACTION_CODES.get(tool_name, "NO_OP"),
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Bedrock Converse tool-use routing (fallback)
    # ------------------------------------------------------------------

    def _route_with_bedrock(
        self,
        req: IntentRequest,
        device_context: DeviceContext,
        memory_text: str,
        last_music_context: dict[str, Any],
    ) -> RouterDecision | None:
        user_prompt = json.dumps(
            {
                "user": req.user_id,
                "device_id": req.device_id,
                "room": req.room,
                "resolved_room": device_context.room_name,
                "default_speaker": device_context.default_speaker,
                "text": req.raw_text,
                "timestamp": req.timestamp.isoformat(),
                "memory_context": memory_text,
                "last_music_context": last_music_context,
            },
            ensure_ascii=True,
        )

        try:
            response = self._bedrock.converse(
                modelId=self.settings.bedrock_model_id,
                system=[{"text": ROUTER_SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                toolConfig={"tools": TOOL_DEFINITIONS},
                inferenceConfig={"maxTokens": 400, "temperature": 0.1},
            )
            return self._parse_tool_use_response(response)
        except Exception as exc:
            logger.warning("Bedrock routing failed: %s", exc)
            return None

    def _parse_tool_use_response(self, response: dict[str, Any]) -> RouterDecision | None:
        content_blocks = response.get("output", {}).get("message", {}).get("content", [])

        tool_use_block: dict[str, Any] | None = None
        assistant_text = ""

        for block in content_blocks:
            if "toolUse" in block:
                tool_use_block = block["toolUse"]
            elif "text" in block:
                assistant_text += block["text"]

        if not tool_use_block:
            parsed = self._parse_router_json(assistant_text)
            if parsed:
                return RouterDecision.model_validate(parsed)
            return None

        bedrock_tool_name = tool_use_block.get("name", "")
        tool_args = tool_use_block.get("input", {})
        if not isinstance(tool_args, dict):
            tool_args = {}

        runner_tool_name = _TOOL_NAME_MAP.get(bedrock_tool_name, bedrock_tool_name)
        intent_type = _TOOL_INTENT_MAP.get(runner_tool_name, IntentType.INFO_QUERY)
        is_private = runner_tool_name == "private.phone_data"

        return RouterDecision(
            intent_type=intent_type,
            tool_name=runner_tool_name,
            tool_args=tool_args,
            action_code=_ACTION_CODES.get(runner_tool_name, "NO_OP"),
            speak_text="" if is_private else "",
            private_note="" if not is_private else "Fetching private data for phone display.",
            confidence=0.9,
            requires_private_display=is_private,
        )

    # ------------------------------------------------------------------
    # Legacy JSON-in-text parser (compat fallback)
    # ------------------------------------------------------------------

    def _parse_router_json(self, text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()

        if stripped.startswith("{") and stripped.endswith("}"):
            candidate = stripped
        else:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            candidate = stripped[start : end + 1]

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                return None

        payload.setdefault("tool_args", {})
        payload.setdefault("action_code", "NO_OP")
        payload.setdefault("speak_text", "")
        payload.setdefault("private_note", "")
        payload.setdefault("confidence", 0.6)
        payload.setdefault("requires_private_display", False)
        return payload
