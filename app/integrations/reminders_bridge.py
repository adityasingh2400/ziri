from __future__ import annotations

from app.schemas import ToolResult


class RemindersBridge:
    """Produces payloads for iOS Shortcut to write iCloud reminders."""

    def create_reminder(self, text: str, due_at: str | None = None) -> ToolResult:
        payload = {
            "shortcut_action": "CREATE_ICLOUD_REMINDER",
            "text": text,
            "due_at": due_at,
        }
        return ToolResult(
            ok=True,
            action_code="REMINDER_CREATE",
            speak_text="Reminder queued.",
            private_note="Reminder details were sent to your phone for iCloud sync.",
            payload=payload,
        )
