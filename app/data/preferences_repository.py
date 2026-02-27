from __future__ import annotations

import logging
from typing import Any

try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover - optional import for local dev
    Client = Any  # type: ignore[misc,assignment]
    create_client = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class PreferencesRepository:
    def get_user_preferences(self, user_id: str) -> dict[str, Any]:
        raise NotImplementedError


class InMemoryPreferencesRepository(PreferencesRepository):
    def __init__(self) -> None:
        self._prefs: dict[str, dict[str, Any]] = {}

    def get_user_preferences(self, user_id: str) -> dict[str, Any]:
        return self._prefs.get(user_id, {})


class SupabasePreferencesRepository(PreferencesRepository):
    def __init__(self, url: str, service_role_key: str) -> None:
        if not create_client:
            raise RuntimeError("supabase package is not installed")
        self.client: Client = create_client(url, service_role_key)

    def get_user_preferences(self, user_id: str) -> dict[str, Any]:
        try:
            res = (
                self.client.table("user_preferences")
                .select("preferences")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            rows = res.data or []
            if not rows:
                return {}
            prefs = rows[0].get("preferences")
            return prefs if isinstance(prefs, dict) else {}
        except Exception as exc:
            logger.warning("Failed to load user preferences: %s", exc)
            return {}
