from __future__ import annotations

import logging
from typing import Any, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional import for local dev
    END = "__end__"  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]

from app.settings import Settings
from app.core.agents.home_agent import HomeAgent
from app.core.agents.info_agent import InfoAgent
from app.core.agents.music_agent import MusicAgent
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
from app.core.supervisor import Supervisor, SupervisorResult
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
    supervisor_result: SupervisorResult
    agent_scratchpad: list[dict[str, str]]


class ZiriOrchestrator:
    def __init__(
        self,
        settings: Settings,
        brain: Brain,
        tool_runner: ToolRunner,
        memory: MemoryStore,
        tts: TTS,
        supervisor: Supervisor | None = None,
        music_agent: MusicAgent | None = None,
        info_agent: InfoAgent | None = None,
        home_agent: HomeAgent | None = None,
        hybrid_searcher: HybridSearcher | None = None,
        es_store: ElasticsearchStore | None = None,
    ) -> None:
        self.settings = settings
        self.brain = brain
        self.tool_runner = tool_runner
        self.memory = memory
        self.tts = tts
        self.supervisor = supervisor
        self.music_agent = music_agent
        self.info_agent = info_agent
        self.home_agent = home_agent
        self.hybrid_searcher = hybrid_searcher
        self.es_store = es_store
        self.graph = self._build_graph() if StateGraph else None

    def _build_graph(self):
        graph = StateGraph(ZiriState)

        if self.supervisor:
            graph.add_node("supervisor", self._supervisor_node)
            graph.add_node("music_agent", self._music_agent_node)
            graph.add_node("info_agent", self._info_agent_node)
            graph.add_node("home_agent", self._home_agent_node)
            graph.add_node("quick_action", self._quick_action_node)
            graph.add_node("respond", self._respond_node)

            graph.add_edge(START, "supervisor")
            graph.add_conditional_edges("supervisor", self._route_to_agent, {
                "music": "music_agent",
                "info": "info_agent",
                "home": "home_agent",
                "quick": "quick_action",
            })
            graph.add_edge("music_agent", "respond")
            graph.add_edge("info_agent", "respond")
            graph.add_edge("home_agent", "respond")
            graph.add_edge("quick_action", "respond")
            graph.add_edge("respond", END)
        else:
            graph.add_node("route", self._route_node)
            graph.add_node("execute", self._execute_node)
            graph.add_node("respond", self._respond_node)
            graph.add_edge(START, "route")
            graph.add_edge("route", "execute")
            graph.add_edge("execute", "respond")
            graph.add_edge("respond", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Multi-agent nodes (supervisor path)
    # ------------------------------------------------------------------

    def _supervisor_node(self, state: ZiriState) -> dict:
        req = state["request"]
        device_context = state["device_context"]
        trace = state.get("trace")

        import time as _time
        _t0 = _time.monotonic()
        result = self.supervisor.classify(req, device_context, trace=trace)
        _elapsed = _time.monotonic() - _t0
        INTENT_ROUTING_DURATION.observe(_elapsed)
        if result.deterministic_decision is not None:
            DETERMINISTIC_ROUTE_TOTAL.labels(outcome="hit").inc()
        else:
            DETERMINISTIC_ROUTE_TOTAL.labels(outcome="miss").inc()

        return {
            "supervisor_result": result,
            "domain": result.domain,
        }

    @staticmethod
    def _route_to_agent(state: ZiriState) -> str:
        return state.get("domain", "info")

    def _music_agent_node(self, state: ZiriState) -> dict:
        req = state["request"]
        device_context = state["device_context"]
        trace = state.get("trace")
        sv = state.get("supervisor_result")
        query = sv.query if sv else req.raw_text

        decision, result = self.music_agent.run(query, req, device_context, trace=trace)
        return {"route_decision": decision, "tool_result": result}

    def _info_agent_node(self, state: ZiriState) -> dict:
        req = state["request"]
        device_context = state["device_context"]
        trace = state.get("trace")
        sv = state.get("supervisor_result")
        query = sv.query if sv else req.raw_text

        decision, result = self.info_agent.run(query, req, device_context, trace=trace)
        return {"route_decision": decision, "tool_result": result}

    def _home_agent_node(self, state: ZiriState) -> dict:
        req = state["request"]
        device_context = state["device_context"]
        trace = state.get("trace")
        sv = state.get("supervisor_result")
        query = sv.query if sv else req.raw_text

        decision, result = self.home_agent.run(query, req, device_context, trace=trace)
        return {"route_decision": decision, "tool_result": result}

    def _quick_action_node(self, state: ZiriState) -> dict:
        """Direct execution for deterministic phrase matches -- no LLM call."""
        req = state["request"]
        device_context = state["device_context"]
        trace = state.get("trace")
        sv = state.get("supervisor_result")

        if sv and sv.deterministic_decision:
            decision = sv.deterministic_decision
            import time as _time
            _t0 = _time.monotonic()
            result = self.tool_runner.run(decision, req, device_context, trace=trace)
            TOOL_EXECUTION_DURATION.labels(tool_name=decision.tool_name).observe(
                _time.monotonic() - _t0,
            )
            return {"route_decision": decision, "tool_result": result}

        fallback = RouterDecision(
            intent_type=IntentType.INFO_QUERY,
            tool_name="general.answer",
            tool_args={"query": req.raw_text},
            action_code="INFO_REPLY",
            confidence=0.5,
        )
        result = self.tool_runner.run(fallback, req, device_context, trace=trace)
        return {"route_decision": fallback, "tool_result": result}

    # ------------------------------------------------------------------
    # Legacy linear nodes (fallback when supervisor not configured)
    # ------------------------------------------------------------------

    def _route_node(self, state: ZiriState) -> dict[str, RouterDecision]:
        req = state["request"]
        device_context = state["device_context"]
        trace = state.get("trace")
        decision = self.brain.route_intent(req, device_context, trace=trace)
        return {"route_decision": decision}

    def _execute_node(self, state: ZiriState) -> dict[str, ToolResult]:
        req = state["request"]
        device_context = state["device_context"]
        decision = state["route_decision"]
        trace = state.get("trace")
        result = self.tool_runner.run(decision, req, device_context, trace=trace)
        return {"tool_result": result}

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

        return {"response": response}

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
                "multi_agent": self.supervisor is not None,
            },
            tags=["orchestrator"],
            settings=self.settings,
        )

        init_state: ZiriState = {
            "request": request,
            "device_context": device_context,
            "trace": trace,
        }

        if self.graph:
            output = self.graph.invoke(init_state)
            return output["response"], output["route_decision"], output["tool_result"]

        # Manual fallback when LangGraph is not installed
        if self.supervisor:
            return self._manual_multi_agent(init_state)
        return self._manual_linear(init_state)

    def _manual_multi_agent(self, state: ZiriState) -> tuple[IntentResponse, RouterDecision, ToolResult]:
        sv_out = self._supervisor_node(state)
        state.update(sv_out)  # type: ignore[arg-type]

        domain = state.get("domain", "info")
        if domain == "music":
            agent_out = self._music_agent_node(state)
        elif domain == "home":
            agent_out = self._home_agent_node(state)
        elif domain == "quick":
            agent_out = self._quick_action_node(state)
        else:
            agent_out = self._info_agent_node(state)

        state.update(agent_out)  # type: ignore[arg-type]
        responded = self._respond_node(state)
        return responded["response"], state["route_decision"], state["tool_result"]

    def _manual_linear(self, state: ZiriState) -> tuple[IntentResponse, RouterDecision, ToolResult]:
        routed = self._route_node(state)
        state.update(routed)  # type: ignore[arg-type]
        executed = self._execute_node(state)
        state.update(executed)  # type: ignore[arg-type]
        responded = self._respond_node(state)
        return responded["response"], state["route_decision"], state["tool_result"]
