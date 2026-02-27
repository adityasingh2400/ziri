from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from app.core.embeddings import build_turn_text, embed_text

try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover - optional import for local dev
    Client = Any  # type: ignore[misc,assignment]
    create_client = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class MemoryStore:
    def get_recent_context(self, user_id: str, limit: int = 8) -> str:
        raise NotImplementedError

    def get_last_music_context(self, user_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def remember_turn(
        self,
        user_id: str,
        raw_text: str,
        intent_type: str,
        tool_name: str,
        assistant_speak: str,
        private_note: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def get_semantic_context(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> str:
        return ""


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._turns: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=20))

    def get_recent_context(self, user_id: str, limit: int = 8) -> str:
        turns = list(self._turns[user_id])[-limit:]
        if not turns:
            return ""
        return "\n".join(
            f"- user: {t['raw_text']} | intent: {t['intent_type']} | tool: {t['tool_name']} | assistant: {t['assistant_speak']}"
            for t in turns
        )

    def get_last_music_context(self, user_id: str) -> dict[str, Any] | None:
        turns = self._turns[user_id]
        for turn in reversed(turns):
            if turn.get("intent_type") != "MUSIC_COMMAND":
                continue
            context = turn.get("context_json")
            if isinstance(context, dict):
                return context
        return None

    def remember_turn(
        self,
        user_id: str,
        raw_text: str,
        intent_type: str,
        tool_name: str,
        assistant_speak: str,
        private_note: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self._turns[user_id].append(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "raw_text": raw_text,
                "intent_type": intent_type,
                "tool_name": tool_name,
                "assistant_speak": assistant_speak,
                "private_note": private_note,
                "context_json": context or {},
            }
        )


class SupabaseMemoryStore(MemoryStore):
    def __init__(
        self,
        url: str,
        service_role_key: str,
        bedrock_client: Any = None,
        embedding_model_id: str = "",
    ) -> None:
        if not create_client:
            raise RuntimeError("supabase package is not installed")
        self.client: Client = create_client(url, service_role_key)
        self._bedrock = bedrock_client
        self._embedding_model_id = embedding_model_id

    def get_recent_context(self, user_id: str, limit: int = 8) -> str:
        try:
            res = (
                self.client.table("conversation_turns")
                .select("raw_text,intent_type,tool_name,assistant_speak,created_at")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            turns = list(reversed(res.data or []))
            return "\n".join(
                f"- user: {row.get('raw_text', '')} | intent: {row.get('intent_type', '')} | tool: {row.get('tool_name', '')} | assistant: {row.get('assistant_speak', '')}"
                for row in turns
            )
        except Exception as exc:
            logger.warning("Failed to fetch memory context from Supabase: %s", exc)
            return ""

    def get_last_music_context(self, user_id: str) -> dict[str, Any] | None:
        try:
            res = (
                self.client.table("conversation_turns")
                .select("context_json,intent_type,created_at")
                .eq("user_id", user_id)
                .eq("intent_type", "MUSIC_COMMAND")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            rows = res.data or []
            if not rows:
                return None
            context = rows[0].get("context_json")
            return context if isinstance(context, dict) else None
        except Exception as exc:
            logger.warning("Failed to load last music context: %s", exc)
            return None

    def remember_turn(
        self,
        user_id: str,
        raw_text: str,
        intent_type: str,
        tool_name: str,
        assistant_speak: str,
        private_note: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        try:
            payload: dict[str, Any] = {
                "user_id": user_id,
                "raw_text": raw_text,
                "intent_type": intent_type,
                "tool_name": tool_name,
                "assistant_speak": assistant_speak,
                "private_note": private_note,
                "context_json": context or {},
            }

            if self._bedrock and self._embedding_model_id:
                turn_text = build_turn_text(raw_text, intent_type, tool_name, assistant_speak)
                vector = embed_text(self._bedrock, self._embedding_model_id, turn_text)
                if vector:
                    payload["embedding"] = vector

            self.client.table("conversation_turns").insert(payload).execute()
        except Exception as exc:
            logger.warning("Failed to persist memory turn to Supabase: %s", exc)

    def get_semantic_context(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> str:
        try:
            res = self.client.rpc(
                "match_conversation_turns",
                {
                    "query_embedding": query_embedding,
                    "match_user_id": user_id,
                    "match_count": top_k,
                },
            ).execute()
            rows = res.data or []
            if not rows:
                return ""
            return "\n".join(
                f"- [similarity={row.get('similarity', 0):.2f}] user: {row.get('raw_text', '')} | "
                f"intent: {row.get('intent_type', '')} | tool: {row.get('tool_name', '')} | "
                f"assistant: {row.get('assistant_speak', '')}"
                for row in rows
            )
        except Exception as exc:
            logger.warning("Semantic memory search failed: %s", exc)
            return ""
