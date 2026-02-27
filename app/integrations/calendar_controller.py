from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone

from app.schemas import ToolResult
from app.settings import Settings

logger = logging.getLogger(__name__)

_EVENTS_SCRIPT = '''
use AppleScript version "2.4"
use scripting additions

set now to current date
set tomorrow to now + (1 * days)
set output to ""

tell application "Calendar"
    repeat with cal in calendars
        try
            set evts to (every event of cal whose start date >= now and start date < tomorrow)
            repeat with evt in evts
                set evtName to summary of evt
                set evtStart to start date of evt
                set h to hours of evtStart
                set m to minutes of evtStart
                if h > 12 then
                    set h to h - 12
                    set ap to "PM"
                else if h = 0 then
                    set h to 12
                    set ap to "AM"
                else if h = 12 then
                    set ap to "PM"
                else
                    set ap to "AM"
                end if
                if m < 10 then
                    set mStr to "0" & (m as string)
                else
                    set mStr to m as string
                end if
                set timeStr to (h as string) & ":" & mStr & " " & ap
                set output to output & evtName & " at " & timeStr & linefeed
            end repeat
        end try
    end repeat
end tell

return output
'''


class CalendarController:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get_todays_events(self, max_results: int = 5) -> ToolResult:
        try:
            result = subprocess.run(
                ["osascript", "-e", _EVENTS_SCRIPT],
                capture_output=True, text=True, timeout=8,
            )

            if result.returncode != 0:
                logger.warning("Calendar AppleScript failed: %s", result.stderr)
                return ToolResult(
                    ok=False,
                    action_code="CALENDAR_ERROR",
                    speak_text="I couldn't access your calendar.",
                    private_note=result.stderr,
                    error="calendar_script_error",
                )

            raw = result.stdout.strip()
            if not raw:
                return ToolResult(
                    ok=True,
                    action_code="CALENDAR_EMPTY",
                    speak_text="No events on your calendar today.",
                    private_note="",
                    payload={"events": []},
                )

            events = [line.strip() for line in raw.split("\n") if line.strip()]
            events = events[:max_results]

            if len(events) == 1:
                speak = f"You have one event today: {events[0]}."
            else:
                speak = f"You have {len(events)} events today. " + ". ".join(events) + "."

            return ToolResult(
                ok=True,
                action_code="CALENDAR_READ",
                speak_text=speak,
                private_note="",
                payload={"events": events},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, action_code="CALENDAR_ERROR", speak_text="Calendar took too long to respond.", private_note="", error="calendar_timeout")
        except Exception as exc:
            logger.exception("Calendar lookup failed")
            return ToolResult(ok=False, action_code="CALENDAR_ERROR", speak_text="I couldn't read your calendar.", private_note=str(exc), error="calendar_exception")
