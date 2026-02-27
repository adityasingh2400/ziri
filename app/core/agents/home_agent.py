from __future__ import annotations

import json
import logging
from typing import Any

from app.core.brain import TOOL_DEFINITIONS, _TOOL_NAME_MAP, _TOOL_INTENT_MAP, _ACTION_CODES
from app.core.device_registry import DeviceContext
from app.core.tracing import trace_llm_call
from app.integrations.home_scene_controller import HomeSceneController
from app.integrations.phone_bridge import PhoneBridge
from app.integrations.reminders_bridge import RemindersBridge
from app.schemas import IntentRequest, RouterDecision, IntentType, ToolResult
from app.settings import Settings

logger = logging.getLogger(__name__)

HOME_TOOL_NAMES = {"home_scene", "reminders_create", "private_phone_data"}

HOME_TOOLS = [t for t in TOOL_DEFINITIONS if t["toolSpec"]["name"] in HOME_TOOL_NAMES]

HOME_AGENT_SYSTEM = """\
You are Ziri's home sub-agent. You handle home automation scenes, reminders, and private data.
Given the user's request, choose exactly ONE tool to call.
Keep speak_text short and natural for voice output.
"""

MAX_REACT_ITERATIONS = 2


class HomeAgent:
    def __init__(
        self,
        settings: Settings,
        home_scene: HomeSceneController,
        reminders: RemindersBridge,
        phone_bridge: PhoneBridge,
    ) -> None:
        self.settings = settings
        self.home_scene = home_scene
        self.reminders = reminders
        self.phone_bridge = phone_bridge
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

        fallback_decision = RouterDecision(
            intent_type=IntentType.HOME_SCENE,
            tool_name="home.scene",
            tool_args={"scene_name": query},
            action_code="SCENE_APPLY",
            confidence=0.5,
        )
        fallback_result = ToolResult(
            ok=False,
            action_code="SCENE_APPLY",
            speak_text="I had trouble with that home action.",
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
                intent_type=IntentType.HOME_SCENE,
                tool_name="home.scene",
                tool_args={"scene_name": query},
                action_code="SCENE_APPLY",
                confidence=0.6,
            )

        user_prompt_data: dict[str, Any] = {"text": query}
        if scratchpad:
            user_prompt_data["previous_attempts"] = scratchpad
        user_prompt = json.dumps(user_prompt_data, ensure_ascii=True)

        try:
            response = trace_llm_call(
                trace=trace,
                name="home_agent_think",
                model=self.settings.bedrock_model_id,
                system_prompt=HOME_AGENT_SYSTEM,
                user_prompt=user_prompt,
                bedrock_call=lambda: self._bedrock.converse(
                    modelId=self.settings.bedrock_model_id,
                    system=[{"text": HOME_AGENT_SYSTEM}],
                    messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                    toolConfig={"tools": HOME_TOOLS},
                    inferenceConfig={"maxTokens": 200, "temperature": 0.1},
                ),
                tool_config={"tools": HOME_TOOLS},
            )

            content_blocks = response.get("output", {}).get("message", {}).get("content", [])
            for block in content_blocks:
                if "toolUse" in block:
                    tu = block["toolUse"]
                    bedrock_name = tu.get("name", "")
                    tool_args = tu.get("input", {}) if isinstance(tu.get("input"), dict) else {}
                    runner_name = _TOOL_NAME_MAP.get(bedrock_name, bedrock_name)
                    is_private = runner_name == "private.phone_data"
                    return RouterDecision(
                        intent_type=_TOOL_INTENT_MAP.get(runner_name, IntentType.HOME_SCENE),
                        tool_name=runner_name,
                        tool_args=tool_args,
                        action_code=_ACTION_CODES.get(runner_name, "SCENE_APPLY"),
                        speak_text="" if is_private else "",
                        private_note="Fetching private data for phone display." if is_private else "",
                        confidence=0.85,
                        requires_private_display=is_private,
                    )
        except Exception as exc:
            logger.warning("HomeAgent think failed: %s", exc)

        return None

    def _act(
        self,
        decision: RouterDecision,
        req: IntentRequest,
        device_context: DeviceContext,
    ) -> ToolResult:
        name = decision.tool_name
        args = decision.tool_args

        try:
            if name == "home.scene":
                scene_name = str(args.get("scene_name") or req.raw_text)
                return self.home_scene.apply_scene(scene_name, device_context.room_name)

            if name == "reminders.create":
                text = str(args.get("text") or req.raw_text)
                due_at = args.get("due_at")
                return self.reminders.create_reminder(text=text, due_at=due_at)

            if name == "private.phone_data":
                return self.phone_bridge.handle_private_query(
                    str(args.get("query", req.raw_text))
                )
        except Exception as exc:
            logger.warning("HomeAgent act failed for %s: %s", name, exc)
            return ToolResult(ok=False, action_code="ERROR", speak_text="That action failed.", error=str(exc))

        return ToolResult(ok=False, action_code="UNKNOWN_TOOL", speak_text="Unknown home action.", error=f"Unknown: {name}")
