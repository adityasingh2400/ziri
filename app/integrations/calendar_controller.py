from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.settings import Settings
from app.schemas import ToolResult

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
except Exception:  # pragma: no cover - optional import for local dev
    service_account = None  # type: ignore[assignment]
    build = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class CalendarController:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get_todays_events(self, max_results: int = 3) -> ToolResult:
        if not self.settings.google_service_account_file:
            return ToolResult(
                ok=False,
                action_code="CALENDAR_UNCONFIGURED",
                speak_text="",
                private_note="Google Calendar is not configured yet.",
                error="calendar_not_configured",
            )
        if not service_account or not build:
            return ToolResult(
                ok=False,
                action_code="CALENDAR_ERROR",
                speak_text="",
                private_note="Google API client dependencies are missing.",
                error="calendar_deps_missing",
            )

        try:
            creds = service_account.Credentials.from_service_account_file(
                self.settings.google_service_account_file,
                scopes=["https://www.googleapis.com/auth/calendar.readonly"],
            )
            svc = build("calendar", "v3", credentials=creds, cache_discovery=False)

            now = datetime.now(timezone.utc)
            end = now + timedelta(days=1)
            events = (
                svc.events()
                .list(
                    calendarId=self.settings.google_calendar_id,
                    timeMin=now.isoformat(),
                    timeMax=end.isoformat(),
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            ).get("items", [])

            if not events:
                return ToolResult(
                    ok=True,
                    action_code="CALENDAR_EMPTY",
                    speak_text="You have no events in the next 24 hours.",
                    private_note="",
                    payload={"events": []},
                )

            summary_parts: list[str] = []
            for event in events:
                name = event.get("summary", "Untitled event")
                start = event.get("start", {}).get("dateTime") or event.get("start", {}).get("date")
                summary_parts.append(f"{name} at {start}")

            speak_text = "Here are your upcoming events: " + "; ".join(summary_parts)
            return ToolResult(
                ok=True,
                action_code="CALENDAR_READ",
                speak_text=speak_text,
                private_note="",
                payload={"events": events},
            )
        except Exception as exc:
            logger.exception("Google Calendar lookup failed")
            return ToolResult(
                ok=False,
                action_code="CALENDAR_ERROR",
                speak_text="I couldn't read your calendar right now.",
                private_note=str(exc),
                error="calendar_exception",
            )
