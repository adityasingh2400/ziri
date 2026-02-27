from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.schemas import IntentRequest, IntentResponse, RouterDecision, ToolResult

try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover - optional import for local dev
    Client = Any  # type: ignore[misc,assignment]
    create_client = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class SessionRepository:
    def log_request(self, req: IntentRequest) -> str | None:
        raise NotImplementedError

    def finalize(
        self,
        session_id: str | None,
        response: IntentResponse,
        route: RouterDecision,
        tool_result: ToolResult,
    ) -> None:
        raise NotImplementedError


class InMemorySessionRepository(SessionRepository):
    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def log_request(self, req: IntentRequest) -> str:
        session_id = f"mem-{len(self._rows) + 1}"
        self._rows[session_id] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": req.model_dump(mode="json"),
            "status": "RECEIVED",
        }
        return session_id

    def finalize(
        self,
        session_id: str | None,
        response: IntentResponse,
        route: RouterDecision,
        tool_result: ToolResult,
    ) -> None:
        if not session_id:
            return
        if session_id not in self._rows:
            return
        self._rows[session_id].update(
            {
                "status": "COMPLETED" if tool_result.ok else "FAILED",
                "response": response.model_dump(mode="json"),
                "route": route.model_dump(mode="json"),
                "tool_result": tool_result.model_dump(mode="json"),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
        )


class SupabaseSessionRepository(SessionRepository):
    def __init__(self, url: str, service_role_key: str) -> None:
        if not create_client:
            raise RuntimeError("supabase package is not installed")
        self.client: Client = create_client(url, service_role_key)

    def log_request(self, req: IntentRequest) -> str | None:
        try:
            payload = {
                "user_id": req.user_id,
                "device_id": req.device_id,
                "room": req.room,
                "raw_text": req.raw_text,
                "request_ts": req.timestamp.isoformat(),
                "status": "RECEIVED",
            }
            res = self.client.table("sessions").insert(payload).execute()
            inserted = (res.data or [{}])[0]
            return str(inserted.get("id")) if inserted.get("id") is not None else None
        except Exception as exc:
            logger.warning("Failed to write session request to Supabase: %s", exc)
            return None

    def finalize(
        self,
        session_id: str | None,
        response: IntentResponse,
        route: RouterDecision,
        tool_result: ToolResult,
    ) -> None:
        if not session_id:
            return
        try:
            patch = {
                "status": "COMPLETED" if tool_result.ok else "FAILED",
                "response_json": response.model_dump(mode="json"),
                "route_json": route.model_dump(mode="json"),
                "tool_json": tool_result.model_dump(mode="json"),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
            self.client.table("sessions").update(patch).eq("id", session_id).execute()
        except Exception as exc:
            logger.warning("Failed to finalize session in Supabase: %s", exc)
