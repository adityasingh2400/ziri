from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DeviceContext:
    device_id: str
    room_name: str
    default_speaker: str | None
    spotify_device_id: str | None


class DeviceRegistry:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._config: dict[str, Any] = {"devices": {}, "speakers": {}}
        self.reload()

    def reload(self) -> None:
        if not self.path.exists():
            self._config = {"devices": {}, "speakers": {}}
            return
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self._config = {
            "devices": raw.get("devices", {}),
            "speakers": raw.get("speakers", {}),
        }

    def resolve_context(self, device_id: str, room_fallback: str) -> DeviceContext:
        device_meta = self._config.get("devices", {}).get(device_id, {})
        room_name = device_meta.get("room_name") or room_fallback
        speaker_name = device_meta.get("default_speaker")
        speaker_meta = self.resolve_speaker(speaker_name) if speaker_name else {}
        spotify_id = speaker_meta.get("spotify_device_id") or None

        return DeviceContext(
            device_id=device_id,
            room_name=room_name,
            default_speaker=speaker_name,
            spotify_device_id=spotify_id,
        )

    def resolve_speaker(self, speaker_name: str | None) -> dict[str, Any]:
        if not speaker_name:
            return {}
        return self._config.get("speakers", {}).get(speaker_name, {}) or {}

    def as_dict(self) -> dict[str, Any]:
        return self._config
