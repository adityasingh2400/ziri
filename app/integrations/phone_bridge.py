from __future__ import annotations

from app.schemas import ToolResult


class PhoneBridge:
    """Maps private queries to iOS Shortcut-safe response payloads."""

    def handle_private_query(self, query: str) -> ToolResult:
        lowered = query.lower()

        if "text" in lowered or "message" in lowered:
            return ToolResult(
                ok=True,
                action_code="PRIVATE_NOTE",
                speak_text="",
                private_note="Private message summary prepared for phone display.",
                payload={
                    "shortcut_action": "SHOW_PRIVATE_PANEL",
                    "data_type": "messages",
                    "query": query,
                },
            )

        if "otp" in lowered or "code" in lowered:
            return ToolResult(
                ok=True,
                action_code="PRIVATE_NOTE",
                speak_text="",
                private_note="One-time code lookup prepared for your phone.",
                payload={
                    "shortcut_action": "SHOW_PRIVATE_PANEL",
                    "data_type": "otp",
                    "query": query,
                },
            )

        return ToolResult(
            ok=True,
            action_code="PRIVATE_NOTE",
            speak_text="",
            private_note="Private data was routed to your phone display.",
            payload={
                "shortcut_action": "SHOW_PRIVATE_PANEL",
                "data_type": "private_data",
                "query": query,
            },
        )
