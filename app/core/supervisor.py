from __future__ import annotations

import json
import logging
from typing import Any

from app.core.brain import (
    ROUTER_SYSTEM_PROMPT,
    TOOL_DEFINITIONS,
    _TOOL_INTENT_MAP,
    _TOOL_NAME_MAP,
    _ACTION_CODES,
    deterministic_route,
    tool_to_domain,
)
from app.core.device_registry import DeviceContext
from app.core.embeddings import embed_text
from app.core.memory import MemoryStore
from app.core.tracing import trace_llm_call
from app.schemas import IntentRequest, IntentType, RouterDecision
from app.settings import Settings

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """\
You are Ziri's supervisor agent. Your job is to classify the user's intent
into exactly ONE domain and extract the relevant query.

Available domains:
- route_to_music: Any music-related request (play, pause, skip, volume, queue, etc.)
- route_to_info: Questions, facts, weather, sports scores, news, calendar, time/date
- route_to_home: Home automation scenes, reminders, private phone data

Choose the single best domain tool to call. Pass the user's core query/intent as the argument.
"""

DOMAIN_TOOLS: list[dict[str, Any]] = [
    {"toolSpec": {
        "name": "route_to_music",
        "description": "Route to the music agent for Spotify playback, volume, queue, and track control.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The music-related request"},
        }, "required": ["query"]}},
    }},
    {"toolSpec": {
        "name": "route_to_info",
        "description": "Route to the info agent for questions, weather, sports, news, calendar, time.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The information query"},
        }, "required": ["query"]}},
    }},
    {"toolSpec": {
        "name": "route_to_home",
        "description": "Route to the home agent for scenes, reminders, or private phone data.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "action": {"type": "string", "description": "The home/device action request"},
        }, "required": ["action"]}},
    }},
]

_DOMAIN_MAP = {
    "route_to_music": "music",
    "route_to_info": "info",
    "route_to_home": "home",
}


class SupervisorResult:
    __slots__ = ("domain", "query", "deterministic_decision")

    def __init__(
        self,
        domain: str,
        query: str,
        deterministic_decision: RouterDecision | None = None,
    ) -> None:
        self.domain = domain
        self.query = query
        self.deterministic_decision = deterministic_decision


class Supervisor:
    def __init__(
        self,
        settings: Settings,
        memory: MemoryStore,
        bedrock_client: Any = None,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self._bedrock = bedrock_client

    def classify(
        self,
        req: IntentRequest,
        device_context: DeviceContext,
        trace: Any = None,
    ) -> SupervisorResult:
        last_music_context = self.memory.get_last_music_context(req.user_id) or {}
        det = deterministic_route(req.raw_text, last_music_context)

        if det is not None:
            domain = tool_to_domain(det.tool_name)
            return SupervisorResult(
                domain="quick" if domain == "music" and det.confidence >= 0.9 else domain,
                query=req.raw_text,
                deterministic_decision=det,
            )

        if not self._bedrock:
            return SupervisorResult(domain="info", query=req.raw_text)

        memory_text = self.memory.get_recent_context(
            req.user_id, limit=self.settings.memory_window,
        )

        semantic_text = ""
        if self.settings.semantic_memory_enabled:
            query_embedding = embed_text(
                self._bedrock, self.settings.embedding_model_id, req.raw_text,
            )
            if query_embedding:
                semantic_text = self.memory.get_semantic_context(
                    req.user_id, query_embedding,
                    top_k=self.settings.semantic_memory_top_k,
                )

        prompt_data: dict[str, Any] = {
            "user": req.user_id,
            "device_id": req.device_id,
            "room": req.room,
            "text": req.raw_text,
            "memory_context": memory_text,
        }
        if semantic_text:
            prompt_data["semantic_context"] = semantic_text

        user_prompt = json.dumps(prompt_data, ensure_ascii=True)

        try:
            response = trace_llm_call(
                trace=trace,
                name="supervisor_classify",
                model=self.settings.bedrock_model_id,
                system_prompt=SUPERVISOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                bedrock_call=lambda: self._bedrock.converse(
                    modelId=self.settings.bedrock_model_id,
                    system=[{"text": SUPERVISOR_SYSTEM_PROMPT}],
                    messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                    toolConfig={"tools": DOMAIN_TOOLS},
                    inferenceConfig={"maxTokens": 200, "temperature": 0.0},
                ),
                tool_config={"tools": DOMAIN_TOOLS},
            )

            content_blocks = response.get("output", {}).get("message", {}).get("content", [])
            for block in content_blocks:
                if "toolUse" in block:
                    tool_use = block["toolUse"]
                    tool_name = tool_use.get("name", "")
                    tool_input = tool_use.get("input", {})
                    domain = _DOMAIN_MAP.get(tool_name, "info")
                    query = tool_input.get("query") or tool_input.get("action") or req.raw_text
                    return SupervisorResult(domain=domain, query=query)

        except Exception as exc:
            logger.warning("Supervisor LLM classification failed: %s", exc)

        return SupervisorResult(domain="info", query=req.raw_text)
