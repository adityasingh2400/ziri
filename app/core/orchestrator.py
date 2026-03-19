from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, TypedDict

from app.settings import Settings

from app.core.brain import Brain
from app.core.device_registry import DeviceContext
from app.core.memory import MemoryStore
from app.core.metrics import (
    DETERMINISTIC_ROUTE_TOTAL,
    INTENT_ROUTING_DURATION,
    TOOL_EXECUTION_DURATION,
)
from app.core.personality import rewrite_response
from app.core.search import ElasticsearchStore, HybridSearcher
from app.core.tool_runner import ToolRunner
from app.core.tracing import create_trace
from app.integrations.tts import TTS
from app.schemas import IntentRequest, IntentResponse, IntentType, RouterDecision, ToolResult

logger = logging.getLogger(__name__)


class ZiriState(TypedDict, total=False):
    request: IntentRequest
    device_context: DeviceContext
    route_decision: RouterDecision
    tool_result: ToolResult
    response: IntentResponse
    trace: Any
    domain: str
    agent_scratchpad: list[dict[str, str]]


class ZiriOrchestrator:
    def __init__(
        self,
        settings: Settings,
        brain: Brain,
        tool_runner: ToolRunner,
        memory: MemoryStore,
        tts: TTS,
        hybrid_searcher: HybridSearcher | None = None,
        es_store: ElasticsearchStore | None = None,
    ) -> None:
        self.settings = settings
        self.brain = brain
        self.tool_runner = tool_runner
        self.memory = memory
        self.tts = tts
        self.hybrid_searcher = hybrid_searcher
        self.es_store = es_store

    # ------------------------------------------------------------------
    # Linear execution nodes
    # ------------------------------------------------------------------

    def _route_node(self, state: ZiriState) -> RouterDecision:
        req = state["request"]
        device_context = state["device_context"]
        trace = state.get("trace")
        return self.brain.route_intent(req, device_context, trace=trace)

    def _execute_node(self, state: ZiriState) -> ToolResult:
        req = state["request"]
        device_context = state["device_context"]
        decision = state["route_decision"]
        trace = state.get("trace")
        return self.tool_runner.run(decision, req, device_context, trace=trace)

    # ------------------------------------------------------------------
    # Shared respond node
    # ------------------------------------------------------------------

    def _respond_node(self, state: ZiriState) -> dict[str, IntentResponse]:
        req = state["request"]
        device_context = state["device_context"]
        route = state["route_decision"]
        tool_result = state["tool_result"]
        trace = state.get("trace")

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
            audio_url = self.tts.synthesize(speak_text, trace=trace)

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
                "domain": state.get("domain", "legacy"),
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

        if self.es_store:
            from datetime import datetime, timezone as _tz
            self.es_store.index_turn(
                user_id=req.user_id,
                raw_text=req.raw_text,
                intent_type=route.intent_type.value,
                tool_name=route.tool_name,
                assistant_speak=speak_text,
                created_at=datetime.now(_tz.utc).isoformat(),
            )

        return response

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def handle_intent(
        self,
        request: IntentRequest,
        device_context: DeviceContext,
    ) -> tuple[IntentResponse, RouterDecision, ToolResult]:
        trace = create_trace(
            name="handle_intent",
            user_id=request.user_id,
            metadata={
                "device_id": request.device_id,
                "room": request.room,
                "raw_text": request.raw_text,
                "multi_agent": False,
            },
            tags=["orchestrator"],
            settings=self.settings,
        )

        init_state: ZiriState = {
            "request": request,
            "device_context": device_context,
            "trace": trace,
        }

        return self._manual_linear(init_state)

    def _manual_linear(self, state: ZiriState) -> tuple[IntentResponse, RouterDecision, ToolResult]:
        routed = self._route_node(state)
        state["route_decision"] = routed
        
        executed = self._execute_node(state)
        state["tool_result"] = executed
        
        response = self._respond_node(state)
        return response, state["route_decision"], state["tool_result"]

    # ------------------------------------------------------------------
    # Streaming entry point (LLM tokens → TTS → speaker in real-time)
    # ------------------------------------------------------------------

    def handle_intent_streaming(
        self,
        request: IntentRequest,
        device_context: DeviceContext,
    ) -> tuple[IntentResponse, RouterDecision, ToolResult, bool]:
        """Like handle_intent, but streams the LLM → TTS path when possible.

        Returns (response, decision, result, did_stream).
        When did_stream is True, audio was already played during processing
        and the caller should NOT play audio_url again.
        """
        trace = create_trace(
            name="handle_intent_streaming",
            user_id=request.user_id,
            metadata={
                "device_id": request.device_id,
                "room": request.room,
                "raw_text": request.raw_text,
                "multi_agent": False,
                "streaming": True,
            },
            tags=["orchestrator", "streaming"],
            settings=self.settings,
        )

        init_state: ZiriState = {
            "request": request,
            "device_context": device_context,
            "trace": trace,
        }

        # Linear routing check first to see if it's a general info answer
        decision = self._route_node(init_state)
        init_state["route_decision"] = decision

        # Check if the routing decision says this is a general query we can stream
        if decision.tool_name == "general.answer" and decision.intent_type == IntentType.INFO_QUERY:
            return self._stream_info_path(init_state, decision, trace)

        # Non-streaming fallback execution
        executed = self._execute_node(init_state)
        init_state["tool_result"] = executed
        
        response = self._respond_node(init_state)
        return response, init_state["route_decision"], init_state["tool_result"], False

    def _stream_info_path(
        self,
        state: ZiriState,
        decision: RouterDecision,
        trace: Any,
    ) -> tuple[IntentResponse, RouterDecision, ToolResult, bool]:
        """Run the streaming text generation for general answers and route to TTS stream."""
        req = state["request"]
        device_context = state["device_context"]
        query = str(decision.tool_args.get("query") or req.raw_text)

        # Stream LLM → TTS
        text_iter = self.brain.general_answer_streaming(
            query=query,
            trace=trace,
        )

        spoken_text = self.tts.synthesize_streaming(text_iter, trace=trace)

        decision_final = RouterDecision(
            intent_type=IntentType.INFO_QUERY,
            tool_name="general.answer",
            tool_args={"query": query},
            action_code="INFO_REPLY",
            confidence=0.85,
        )
        result = ToolResult(
            ok=True,
            action_code="INFO_REPLY",
            speak_text=spoken_text,
            payload={"query": query, "streamed": True},
        )

        response = IntentResponse(
            speak_text=spoken_text,
            action_code="INFO_REPLY",
            audio_url=None,
            target_device=device_context.default_speaker,
            metadata={
                "intent_type": IntentType.INFO_QUERY.value,
                "tool": "general.answer",
                "tool_payload": result.payload,
                "confidence": 0.85,
                "resolved_room": device_context.room_name,
                "speaker": device_context.default_speaker,
                "domain": "info",
                "streamed": True,
            },
        )

        self.memory.remember_turn(
            user_id=req.user_id,
            raw_text=req.raw_text,
            intent_type=IntentType.INFO_QUERY.value,
            tool_name="general.answer",
            assistant_speak=spoken_text,
            private_note="",
            context={
                "tool_payload": result.payload,
                "target_device": device_context.default_speaker,
                "room": device_context.room_name,
                "action_code": "INFO_REPLY",
            },
        )

        if self.es_store:
            from datetime import datetime, timezone as _tz
            self.es_store.index_turn(
                user_id=req.user_id,
                raw_text=req.raw_text,
                intent_type=IntentType.INFO_QUERY.value,
                tool_name="general.answer",
                assistant_speak=spoken_text,
                created_at=datetime.now(_tz.utc).isoformat(),
            )

        return response, decision_final, result, True
