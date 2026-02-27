from __future__ import annotations

from app.core.device_registry import DeviceContext
from app.integrations.calendar_controller import CalendarController
from app.integrations.home_scene_controller import HomeSceneController
from app.integrations.nba import NBAController
from app.integrations.news import NewsController
from app.integrations.phone_bridge import PhoneBridge
from app.integrations.reminders_bridge import RemindersBridge
from app.integrations.spotify_controller import SpotifyController
from app.integrations.weather import WeatherController
from app.schemas import IntentRequest, RouterDecision, ToolResult

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class ToolRunner:
    def __init__(
        self,
        spotify: SpotifyController,
        calendar: CalendarController,
        reminders: RemindersBridge,
        home_scene: HomeSceneController,
        phone_bridge: PhoneBridge,
        weather: WeatherController,
        nba: NBAController,
        news: NewsController,
        bedrock_client: Any = None,
        bedrock_model_id: str = "",
    ) -> None:
        self.spotify = spotify
        self.calendar = calendar
        self.reminders = reminders
        self.home_scene = home_scene
        self.phone_bridge = phone_bridge
        self.weather = weather
        self.nba = nba
        self.news = news
        self._bedrock = bedrock_client
        self._model_id = bedrock_model_id

    def run(
        self,
        decision: RouterDecision,
        req: IntentRequest,
        device_context: DeviceContext,
    ) -> ToolResult:
        if decision.tool_name == "spotify.play_query":
            query = str(decision.tool_args.get("query") or req.raw_text)
            return self.spotify.play_query(
                query=query,
                speaker_name=device_context.default_speaker,
                spotify_device_id=device_context.spotify_device_id,
            )

        if decision.tool_name == "spotify.adjust_volume":
            delta = int(decision.tool_args.get("delta_percent", 10))
            return self.spotify.adjust_volume(
                delta_percent=delta,
                speaker_name=device_context.default_speaker,
                spotify_device_id=device_context.spotify_device_id,
            )

        if decision.tool_name == "spotify.pause":
            return self.spotify.pause(
                speaker_name=device_context.default_speaker,
                spotify_device_id=device_context.spotify_device_id,
            )

        if decision.tool_name == "spotify.skip":
            return self.spotify.skip_next(
                speaker_name=device_context.default_speaker,
                spotify_device_id=device_context.spotify_device_id,
            )

        if decision.tool_name == "spotify.current_track":
            return self.spotify.get_currently_playing()

        if decision.tool_name == "spotify.resume":
            return self.spotify.resume(
                speaker_name=device_context.default_speaker,
                spotify_device_id=device_context.spotify_device_id,
            )

        if decision.tool_name == "spotify.previous":
            return self.spotify.previous_track(
                speaker_name=device_context.default_speaker,
                spotify_device_id=device_context.spotify_device_id,
            )

        if decision.tool_name == "spotify.shuffle":
            state = bool(decision.tool_args.get("state", True))
            return self.spotify.shuffle(state)

        if decision.tool_name == "spotify.repeat":
            mode = str(decision.tool_args.get("mode", "track"))
            return self.spotify.repeat(mode)

        if decision.tool_name == "spotify.set_volume":
            percent = int(decision.tool_args.get("percent", 50))
            return self.spotify.set_volume(
                percent=percent,
                speaker_name=device_context.default_speaker,
                spotify_device_id=device_context.spotify_device_id,
            )

        if decision.tool_name == "spotify.queue":
            query = str(decision.tool_args.get("query") or req.raw_text)
            return self.spotify.add_to_queue(
                query=query,
                speaker_name=device_context.default_speaker,
                spotify_device_id=device_context.spotify_device_id,
            )

        if decision.tool_name == "spotify.play_artist":
            query = str(decision.tool_args.get("query") or req.raw_text)
            shuffle = bool(decision.tool_args.get("shuffle", False))
            return self.spotify.play_artist(
                query=query,
                speaker_name=device_context.default_speaker,
                spotify_device_id=device_context.spotify_device_id,
                shuffle=shuffle,
            )

        if decision.tool_name == "spotify.play_playlist":
            query = str(decision.tool_args.get("query") or req.raw_text)
            shuffle = bool(decision.tool_args.get("shuffle", False))
            return self.spotify.play_playlist(
                query=query,
                speaker_name=device_context.default_speaker,
                spotify_device_id=device_context.spotify_device_id,
                shuffle=shuffle,
            )

        if decision.tool_name == "spotify.like":
            return self.spotify.like_current_track()

        if decision.tool_name == "calendar.today":
            return self.calendar.get_todays_events()

        if decision.tool_name == "reminders.create":
            text = str(decision.tool_args.get("text") or req.raw_text)
            due_at = decision.tool_args.get("due_at")
            return self.reminders.create_reminder(text=text, due_at=due_at)

        if decision.tool_name in {"private.phone_texts", "private.phone_data"}:
            return self.phone_bridge.handle_private_query(
                str(decision.tool_args.get("query", req.raw_text))
            )

        if decision.tool_name == "home.scene":
            scene_name = str(decision.tool_args.get("scene_name") or req.raw_text)
            return self.home_scene.apply_scene(scene_name, device_context.room_name)

        if decision.tool_name == "general.answer":
            return self._answer_with_bedrock(decision, req)

        if decision.tool_name == "time.now":
            now = datetime.now()
            h = now.hour % 12 or 12
            ampm = "AM" if now.hour < 12 else "PM"
            return ToolResult(ok=True, action_code="TIME_NOW", speak_text=f"It's {h}:{now.strftime('%M')} {ampm}.", private_note="")

        if decision.tool_name == "time.date":
            now = datetime.now()
            return ToolResult(ok=True, action_code="TIME_DATE", speak_text=f"Today is {now.strftime('%A, %B %d, %Y')}.", private_note="")

        if decision.tool_name == "weather.current":
            return self.weather.get_current_weather()

        if decision.tool_name == "weather.sunrise_sunset":
            return self.weather.get_sunrise_sunset()

        if decision.tool_name == "nba.scores":
            return self.nba.get_todays_games()

        if decision.tool_name == "nba.team":
            team = str(decision.tool_args.get("team") or req.raw_text)
            return self.nba.get_team_score(team)

        if decision.tool_name == "news.headlines":
            category = str(decision.tool_args.get("category", "general"))
            return self.news.get_headlines(category=category)

        if decision.tool_name == "news.topic":
            query = str(decision.tool_args.get("query") or req.raw_text)
            return self.news.get_topic_news(query)

        return ToolResult(
            ok=False,
            action_code="UNKNOWN_TOOL",
            speak_text="I do not have that action configured yet.",
            private_note=f"Unknown tool: {decision.tool_name}",
            error="unknown_tool",
        )

    def _answer_with_bedrock(self, decision: RouterDecision, req: IntentRequest) -> ToolResult:
        query = str(decision.tool_args.get("query") or req.raw_text)

        if not self._bedrock:
            return ToolResult(ok=True, action_code="INFO_REPLY", speak_text="I'm not sure about that.", private_note="", payload={"query": query})

        try:
            response = self._bedrock.converse(
                modelId=self._model_id,
                system=[{"text": (
                    "You are Ziri, a concise voice assistant. Answer the user's question in 1-2 short sentences. "
                    "Be direct and accurate. Speak naturally as if talking to a friend. "
                    "Maximum 25 words. No filler."
                )}],
                messages=[{"role": "user", "content": [{"text": query}]}],
                inferenceConfig={"maxTokens": 80, "temperature": 0.5},
            )

            content = response.get("output", {}).get("message", {}).get("content", [])
            answer = "".join(part.get("text", "") for part in content if part.get("text")).strip()

            if not answer:
                return ToolResult(ok=True, action_code="INFO_REPLY", speak_text="I'm not sure about that.", private_note="")

            return ToolResult(ok=True, action_code="INFO_REPLY", speak_text=answer, private_note="", payload={"query": query})
        except Exception as exc:
            logger.warning("Bedrock general answer failed: %s", exc)
            return ToolResult(ok=True, action_code="INFO_REPLY", speak_text="I couldn't look that up right now.", private_note=str(exc))
