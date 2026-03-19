from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IntentType(str, Enum):
    MUSIC_COMMAND = "MUSIC_COMMAND"
    INFO_QUERY = "INFO_QUERY"
    PERSONAL_REMINDER = "PERSONAL_REMINDER"
    HOME_SCENE = "HOME_SCENE"
    WEATHER = "WEATHER"
    SPORTS = "SPORTS"
    NEWS = "NEWS"
    UNKNOWN = "UNKNOWN"


class IntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_id: str = Field(min_length=1, max_length=120)
    device_id: str = Field(min_length=1, max_length=120)
    room: str = Field(min_length=1, max_length=120)
    raw_text: str = Field(min_length=1, max_length=3000)
    timestamp: datetime
    # Set by always-on listener only: refines routing for the next utterance (e.g. after skip dead-end).
    listener_route_hint: str = Field(default="", max_length=64)

    @field_validator("timestamp")
    @classmethod
    def ensure_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class RouterDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_type: IntentType
    tool_name: str = Field(default="general.answer", max_length=120)
    tool_args: dict[str, Any] = Field(default_factory=dict)
    action_code: str = Field(default="NO_OP", max_length=120)
    speak_text: str = Field(default="", max_length=2000)
    private_note: str = Field(default="", max_length=4000)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    requires_private_display: bool = False


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    action_code: str = Field(default="NO_OP", max_length=120)
    speak_text: str = Field(default="", max_length=2000)
    private_note: str = Field(default="", max_length=4000)
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class IntentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speak_text: str = ""
    private_note: str = ""
    action_code: str = "NO_OP"
    audio_url: str | None = None
    target_device: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    service: str
    timestamp: datetime
    model: str
    version: str
    degraded: bool = False
    components: dict[str, str] = Field(default_factory=dict)
