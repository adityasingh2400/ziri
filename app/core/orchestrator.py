from __future__ import annotations

from typing import TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional import for local dev
    END = "__end__"  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]

from app.settings import Settings
from app.core.brain import Brain
from app.core.device_registry import DeviceContext
from app.core.memory import MemoryStore
from app.core.personality import rewrite_response
from app.core.tool_runner import ToolRunner
from app.integrations.tts import PollyTTS
from app.schemas import IntentRequest, IntentResponse, RouterDecision, ToolResult


class AuraState(TypedDict, total=False):
    request: IntentRequest
    device_context: DeviceContext
    route_decision: RouterDecision
    tool_result: ToolResult
    response: IntentResponse


class AuraOrchestrator:
    def __init__(
        self,
        settings: Settings,
        brain: Brain,
        tool_runner: ToolRunner,
        memory: MemoryStore,
        tts: PollyTTS,
    ) -> None:
        self.settings = settings
        self.brain = brain
        self.tool_runner = tool_runner
        self.memory = memory
        self.tts = tts
        self.graph = self._build_graph() if StateGraph else None

    def _build_graph(self):
        graph = StateGraph(AuraState)
        graph.add_node("route", self._route_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("respond", self._respond_node)
        graph.add_edge(START, "route")
        graph.add_edge("route", "execute")
        graph.add_edge("execute", "respond")
        graph.add_edge("respond", END)
        return graph.compile()

    def _route_node(self, state: AuraState) -> dict[str, RouterDecision]:
        req = state["request"]
        device_context = state["device_context"]
        decision = self.brain.route_intent(req, device_context)
        return {"route_decision": decision}

    def _execute_node(self, state: AuraState) -> dict[str, ToolResult]:
        req = state["request"]
        device_context = state["device_context"]
        decision = state["route_decision"]
        result = self.tool_runner.run(decision, req, device_context)
        return {"tool_result": result}

    def _respond_node(self, state: AuraState) -> dict[str, IntentResponse]:
        req = state["request"]
        device_context = state["device_context"]
        route = state["route_decision"]
        tool_result = state["tool_result"]

        raw_speak = tool_result.speak_text or route.speak_text
        private_note = tool_result.private_note or route.private_note
        action_code = tool_result.action_code or route.action_code or "NO_OP"

        speak_text = rewrite_response(
            bedrock_client=self.brain._bedrock,
            model_id=self.settings.bedrock_model_id,
            raw_text=raw_speak,
            action_code=action_code,
            user_text=req.raw_text,
        )

        audio_url = None
        if self.settings.return_audio_url and speak_text:
            audio_url = self.tts.synthesize(speak_text)

        response = IntentResponse(
            speak_text=speak_text,
            private_note=private_note,
            action_code=action_code,
            audio_url=audio_url,
            target_device=device_context.default_speaker,
            metadata={
                "intent_type": route.intent_type.value,
                "tool": route.tool_name,
                "tool_payload": tool_result.payload,
                "confidence": route.confidence,
                "requires_private_display": route.requires_private_display,
                "resolved_room": device_context.room_name,
                "speaker": device_context.default_speaker,
                "spotify_device_id": device_context.spotify_device_id,
            },
        )

        self.memory.remember_turn(
            user_id=req.user_id,
            raw_text=req.raw_text,
            intent_type=route.intent_type.value,
            tool_name=route.tool_name,
            assistant_speak=speak_text,
            private_note=private_note,
            context={
                "tool_payload": tool_result.payload,
                "target_device": device_context.default_speaker,
                "room": device_context.room_name,
                "action_code": action_code,
            },
        )

        return {"response": response}

    def handle_intent(
        self,
        request: IntentRequest,
        device_context: DeviceContext,
    ) -> tuple[IntentResponse, RouterDecision, ToolResult]:
        if self.graph:
            output = self.graph.invoke({"request": request, "device_context": device_context})
            return output["response"], output["route_decision"], output["tool_result"]

        routed = self._route_node({"request": request, "device_context": device_context})
        executed = self._execute_node(
            {
                "request": request,
                "device_context": device_context,
                "route_decision": routed["route_decision"],
            }
        )
        responded = self._respond_node(
            {
                "request": request,
                "device_context": device_context,
                "route_decision": routed["route_decision"],
                "tool_result": executed["tool_result"],
            }
        )
        return responded["response"], routed["route_decision"], executed["tool_result"]
