from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from app.core.brain import TOOL_DEFINITIONS, _TOOL_NAME_MAP, _TOOL_INTENT_MAP, _ACTION_CODES
from app.core.device_registry import DeviceContext
from app.core.tracing import trace_llm_call
from app.integrations.calendar_controller import CalendarController
from app.integrations.nba import NBAController
from app.integrations.news import NewsController
from app.integrations.weather import WeatherController
from app.schemas import IntentRequest, RouterDecision, IntentType, ToolResult
from app.settings import Settings

logger = logging.getLogger(__name__)

INFO_TOOL_NAMES = {
    "general_answer", "weather_current", "weather_sunrise_sunset",
    "nba_scores", "nba_team", "news_headlines", "news_topic", "calendar_today",
}

INFO_TOOLS = [t for t in TOOL_DEFINITIONS if t["toolSpec"]["name"] in INFO_TOOL_NAMES]

INFO_AGENT_SYSTEM = """\
You are Ziri's information sub-agent. You answer questions and fetch data.
Given the user's query, choose exactly ONE tool to call.
For general knowledge questions, use general_answer.
Keep any speak_text short and natural for voice output.
"""

MAX_REACT_ITERATIONS = 3


class InfoAgent:
    def __init__(
        self,
        settings: Settings,
        calendar: CalendarController,
        weather: WeatherController,
        nba: NBAController,
        news: NewsController,
    ) -> None:
        self.settings = settings
        self.calendar = calendar
        self.weather = weather
        self.nba = nba
        self.news = news
        self._bedrock: Any = None
        self._model_id: str = ""

    def set_bedrock_client(self, client: Any, model_id: str) -> None:
        self._bedrock = client
        self._model_id = model_id

    def run(
        self,
        query: str,
        req: IntentRequest,
        device_context: DeviceContext,
        trace: Any = None,
    ) -> tuple[RouterDecision, ToolResult]:
        scratchpad: list[dict[str, str]] = []

        for iteration in range(MAX_REACT_ITERATIONS):
            decision = self._think(query, scratchpad, trace)
            if decision is None:
                break

            result = self._act(decision, req)
            scratchpad.append({
                "thought": f"Call {decision.tool_name} with {decision.tool_args}",
                "action": decision.tool_name,
                "observation": f"ok={result.ok}, speak={result.speak_text[:100]}",
            })

            if result.ok or iteration == MAX_REACT_ITERATIONS - 1:
                return decision, result

        fallback_decision = RouterDecision(
            intent_type=IntentType.INFO_QUERY,
            tool_name="general.answer",
            tool_args={"query": query},
            action_code="INFO_REPLY",
            confidence=0.5,
        )
        return fallback_decision, self._general_answer(query, trace)

    def _think(
        self,
        query: str,
        scratchpad: list[dict[str, str]],
        trace: Any = None,
    ) -> RouterDecision | None:
        if not self._bedrock:
            return RouterDecision(
                intent_type=IntentType.INFO_QUERY,
                tool_name="general.answer",
                tool_args={"query": query},
                action_code="INFO_REPLY",
                confidence=0.6,
            )

        user_prompt_data: dict[str, Any] = {"text": query}
        if scratchpad:
            user_prompt_data["previous_attempts"] = scratchpad
        user_prompt = json.dumps(user_prompt_data, ensure_ascii=True)

        try:
            response = trace_llm_call(
                trace=trace,
                name="info_agent_think",
                model=self.settings.bedrock_model_id,
                system_prompt=INFO_AGENT_SYSTEM,
                user_prompt=user_prompt,
                bedrock_call=lambda: self._bedrock.converse(
                    modelId=self.settings.bedrock_model_id,
                    system=[{"text": INFO_AGENT_SYSTEM}],
                    messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                    toolConfig={"tools": INFO_TOOLS},
                    inferenceConfig={"maxTokens": 300, "temperature": 0.1},
                ),
                tool_config={"tools": INFO_TOOLS},
            )

            content_blocks = response.get("output", {}).get("message", {}).get("content", [])
            for block in content_blocks:
                if "toolUse" in block:
                    tu = block["toolUse"]
                    bedrock_name = tu.get("name", "")
                    tool_args = tu.get("input", {}) if isinstance(tu.get("input"), dict) else {}
                    runner_name = _TOOL_NAME_MAP.get(bedrock_name, bedrock_name)
                    return RouterDecision(
                        intent_type=_TOOL_INTENT_MAP.get(runner_name, IntentType.INFO_QUERY),
                        tool_name=runner_name,
                        tool_args=tool_args,
                        action_code=_ACTION_CODES.get(runner_name, "INFO_REPLY"),
                        confidence=0.85,
                    )
        except Exception as exc:
            logger.warning("InfoAgent think failed: %s", exc)

        return None

    def _act(self, decision: RouterDecision, req: IntentRequest) -> ToolResult:
        name = decision.tool_name
        args = decision.tool_args

        try:
            if name == "weather.current":
                return self.weather.get_current_weather()
            if name == "weather.sunrise_sunset":
                return self.weather.get_sunrise_sunset()
            if name == "nba.scores":
                return self.nba.get_todays_games()
            if name == "nba.team":
                return self.nba.get_team_score(str(args.get("team") or req.raw_text))
            if name == "news.headlines":
                return self.news.get_headlines(category=str(args.get("category", "general")))
            if name == "news.topic":
                return self.news.get_topic_news(str(args.get("query") or req.raw_text))
            if name == "calendar.today":
                return self.calendar.get_todays_events()
            if name == "time.now":
                now = datetime.now()
                h = now.hour % 12 or 12
                ampm = "AM" if now.hour < 12 else "PM"
                return ToolResult(ok=True, action_code="TIME_NOW", speak_text=f"It's {h}:{now.strftime('%M')} {ampm}.")
            if name == "time.date":
                now = datetime.now()
                return ToolResult(ok=True, action_code="TIME_DATE", speak_text=f"Today is {now.strftime('%A, %B %d, %Y')}.")
            if name == "general.answer":
                return self._general_answer(str(args.get("query") or req.raw_text))
        except Exception as exc:
            logger.warning("InfoAgent act failed for %s: %s", name, exc)
            return ToolResult(ok=False, action_code="ERROR", speak_text="I couldn't get that information.", error=str(exc))

        return ToolResult(ok=False, action_code="UNKNOWN_TOOL", speak_text="Unknown info tool.", error=f"Unknown: {name}")

    def _general_answer(self, query: str, trace: Any = None) -> ToolResult:
        if not self._bedrock:
            return ToolResult(ok=True, action_code="INFO_REPLY", speak_text="I'm not sure about that.")

        system_prompt = (
            "You are Ziri, a concise voice assistant. Answer the user's question in 1-2 short sentences. "
            "Be direct and accurate. Speak naturally as if talking to a friend. "
            "Maximum 25 words. No filler."
        )

        try:
            response = trace_llm_call(
                trace=trace,
                name="info_agent_answer",
                model=self._model_id,
                system_prompt=system_prompt,
                user_prompt=query,
                bedrock_call=lambda: self._bedrock.converse(
                    modelId=self._model_id,
                    system=[{"text": system_prompt}],
                    messages=[{"role": "user", "content": [{"text": query}]}],
                    inferenceConfig={"maxTokens": 80, "temperature": 0.5},
                ),
            )

            content = response.get("output", {}).get("message", {}).get("content", [])
            answer = "".join(p.get("text", "") for p in content if p.get("text")).strip()

            if not answer:
                return ToolResult(ok=True, action_code="INFO_REPLY", speak_text="I'm not sure about that.")

            return ToolResult(ok=True, action_code="INFO_REPLY", speak_text=answer, payload={"query": query})
        except Exception as exc:
            logger.warning("InfoAgent general answer failed: %s", exc)
            return ToolResult(ok=True, action_code="INFO_REPLY", speak_text="I couldn't look that up right now.", error=str(exc))

    def general_answer_streaming(self, query: str, trace: Any = None) -> Iterator[str]:
        """Yield text tokens from Bedrock converse_stream for real-time TTS.

        Falls back to yielding the full response as a single chunk if
        streaming is unavailable.
        """
        if not self._bedrock:
            yield "I'm not sure about that."
            return

        system_prompt = (
            "You are Ziri, a concise voice assistant. Answer the user's question in 1-2 short sentences. "
            "Be direct and accurate. Speak naturally as if talking to a friend. "
            "Maximum 25 words. No filler."
        )

        try:
            response = self._bedrock.converse_stream(
                modelId=self._model_id,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": query}]}],
                inferenceConfig={"maxTokens": 80, "temperature": 0.5},
            )

            stream = response.get("stream")
            if stream is None:
                yield from self._general_answer_fallback(query, trace)
                return

            for event in stream:
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                text = delta.get("text", "")
                if text:
                    yield text

        except Exception as exc:
            logger.warning("InfoAgent streaming answer failed, falling back: %s", exc)
            yield from self._general_answer_fallback(query, trace)

    def _general_answer_fallback(self, query: str, trace: Any = None) -> Iterator[str]:
        """Non-streaming fallback: yield the full answer as one chunk."""
        result = self._general_answer(query, trace=trace)
        if result.speak_text:
            yield result.speak_text
