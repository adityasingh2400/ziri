from __future__ import annotations

import json
import logging
from typing import Any

from app.core.brain import TOOL_DEFINITIONS, _TOOL_NAME_MAP, _TOOL_INTENT_MAP, _ACTION_CODES
from app.core.device_registry import DeviceContext
from app.core.tracing import trace_llm_call
from app.integrations.spotify_controller import SpotifyController
from app.schemas import IntentRequest, RouterDecision, IntentType, ToolResult
from app.settings import Settings

logger = logging.getLogger(__name__)

MUSIC_TOOL_NAMES = {
    "spotify_play_query", "spotify_play_artist", "spotify_play_playlist",
    "spotify_pause", "spotify_resume", "spotify_skip", "spotify_previous",
    "spotify_adjust_volume", "spotify_set_volume", "spotify_shuffle",
    "spotify_repeat", "spotify_current_track", "spotify_queue", "spotify_like",
}

MUSIC_TOOLS = [t for t in TOOL_DEFINITIONS if t["toolSpec"]["name"] in MUSIC_TOOL_NAMES]

MUSIC_AGENT_SYSTEM = """\
You are Ziri's music sub-agent. You control Spotify playback.
Given the user's music request, choose exactly ONE tool to call.
If the request is ambiguous, prefer spotify_play_query with a reasonable search query.
Keep speak_text short and natural for voice output.
"""

MAX_REACT_ITERATIONS = 3


class MusicAgent:
    def __init__(self, settings: Settings, spotify: SpotifyController) -> None:
        self.settings = settings
        self.spotify = spotify
        self._bedrock: Any = None

    def set_bedrock_client(self, client: Any) -> None:
        self._bedrock = client

    def run(
        self,
        query: str,
        req: IntentRequest,
        device_context: DeviceContext,
        trace: Any = None,
    ) -> tuple[RouterDecision, ToolResult]:
        """ReAct loop: Think -> Act -> Observe, up to MAX_REACT_ITERATIONS."""
        scratchpad: list[dict[str, str]] = []

        for iteration in range(MAX_REACT_ITERATIONS):
            decision = self._think(query, scratchpad, trace)
            if decision is None:
                break

            result = self._act(decision, req, device_context)
            scratchpad.append({
                "thought": f"Call {decision.tool_name} with {decision.tool_args}",
                "action": decision.tool_name,
                "observation": f"ok={result.ok}, speak={result.speak_text[:100]}",
            })

            if result.ok or iteration == MAX_REACT_ITERATIONS - 1:
                return decision, result

            if result.error and "not found" in (result.error or "").lower():
                query = f"{query} (retry: previous search found nothing)"
            else:
                break

        fallback_decision = RouterDecision(
            intent_type=IntentType.MUSIC_COMMAND,
            tool_name="spotify.play_query",
            tool_args={"query": query},
            action_code="MUSIC_START",
            confidence=0.5,
        )
        fallback_result = ToolResult(
            ok=False,
            action_code="MUSIC_START",
            speak_text="I had trouble with that music request.",
        )
        return fallback_decision, fallback_result

    def _think(
        self,
        query: str,
        scratchpad: list[dict[str, str]],
        trace: Any = None,
    ) -> RouterDecision | None:
        if not self._bedrock:
            return RouterDecision(
                intent_type=IntentType.MUSIC_COMMAND,
                tool_name="spotify.play_query",
                tool_args={"query": query},
                action_code="MUSIC_START",
                confidence=0.7,
            )

        user_prompt_data: dict[str, Any] = {"text": query}
        if scratchpad:
            user_prompt_data["previous_attempts"] = scratchpad

        user_prompt = json.dumps(user_prompt_data, ensure_ascii=True)

        try:
            response = trace_llm_call(
                trace=trace,
                name="music_agent_think",
                model=self.settings.bedrock_model_id,
                system_prompt=MUSIC_AGENT_SYSTEM,
                user_prompt=user_prompt,
                bedrock_call=lambda: self._bedrock.converse(
                    modelId=self.settings.bedrock_model_id,
                    system=[{"text": MUSIC_AGENT_SYSTEM}],
                    messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                    toolConfig={"tools": MUSIC_TOOLS},
                    inferenceConfig={"maxTokens": 300, "temperature": 0.1},
                ),
                tool_config={"tools": MUSIC_TOOLS},
            )

            content_blocks = response.get("output", {}).get("message", {}).get("content", [])
            for block in content_blocks:
                if "toolUse" in block:
                    tu = block["toolUse"]
                    bedrock_name = tu.get("name", "")
                    tool_args = tu.get("input", {}) if isinstance(tu.get("input"), dict) else {}
                    runner_name = _TOOL_NAME_MAP.get(bedrock_name, bedrock_name)
                    return RouterDecision(
                        intent_type=_TOOL_INTENT_MAP.get(runner_name, IntentType.MUSIC_COMMAND),
                        tool_name=runner_name,
                        tool_args=tool_args,
                        action_code=_ACTION_CODES.get(runner_name, "MUSIC_START"),
                        confidence=0.85,
                    )
        except Exception as exc:
            logger.warning("MusicAgent think failed: %s", exc)

        return None

    def _act(
        self,
        decision: RouterDecision,
        req: IntentRequest,
        device_context: DeviceContext,
    ) -> ToolResult:
        """Execute the Spotify tool."""
        name = decision.tool_name
        args = decision.tool_args
        speaker = device_context.default_speaker
        spotify_id = device_context.spotify_device_id

        dispatch = {
            "spotify.play_query": lambda: self.spotify.play_query(
                query=str(args.get("query") or req.raw_text),
                speaker_name=speaker, spotify_device_id=spotify_id,
            ),
            "spotify.play_artist": lambda: self.spotify.play_artist(
                query=str(args.get("query") or req.raw_text),
                speaker_name=speaker, spotify_device_id=spotify_id,
                shuffle=bool(args.get("shuffle", False)),
            ),
            "spotify.play_playlist": lambda: self.spotify.play_playlist(
                query=str(args.get("query") or req.raw_text),
                speaker_name=speaker, spotify_device_id=spotify_id,
                shuffle=bool(args.get("shuffle", False)),
            ),
            "spotify.pause": lambda: self.spotify.pause(
                speaker_name=speaker, spotify_device_id=spotify_id,
            ),
            "spotify.resume": lambda: self.spotify.resume(
                speaker_name=speaker, spotify_device_id=spotify_id,
            ),
            "spotify.skip": lambda: self.spotify.skip_next(
                speaker_name=speaker, spotify_device_id=spotify_id,
            ),
            "spotify.previous": lambda: self.spotify.previous_track(
                speaker_name=speaker, spotify_device_id=spotify_id,
            ),
            "spotify.adjust_volume": lambda: self.spotify.adjust_volume(
                delta_percent=int(args.get("delta_percent", 10)),
                speaker_name=speaker, spotify_device_id=spotify_id,
            ),
            "spotify.set_volume": lambda: self.spotify.set_volume(
                percent=int(args.get("percent", 50)),
                speaker_name=speaker, spotify_device_id=spotify_id,
            ),
            "spotify.shuffle": lambda: self.spotify.shuffle(bool(args.get("state", True))),
            "spotify.repeat": lambda: self.spotify.repeat(str(args.get("mode", "track"))),
            "spotify.current_track": lambda: self.spotify.get_currently_playing(),
            "spotify.queue": lambda: self.spotify.add_to_queue(
                query=str(args.get("query") or req.raw_text),
                speaker_name=speaker, spotify_device_id=spotify_id,
            ),
            "spotify.like": lambda: self.spotify.like_current_track(),
        }

        handler = dispatch.get(name)
        if handler:
            try:
                return handler()
            except Exception as exc:
                logger.warning("MusicAgent act failed for %s: %s", name, exc)
                return ToolResult(ok=False, action_code="ERROR", speak_text="That music action failed.", error=str(exc))

        return ToolResult(ok=False, action_code="UNKNOWN_TOOL", speak_text="Unknown music action.", error=f"Unknown: {name}")
