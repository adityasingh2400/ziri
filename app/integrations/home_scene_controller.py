from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.schemas import ToolResult


class HomeSceneController:
    def __init__(self, scene_config_path: str) -> None:
        self.scene_config_path = Path(scene_config_path)
        self._scenes: dict[str, dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        if not self.scene_config_path.exists():
            self._scenes = {}
            return
        raw = yaml.safe_load(self.scene_config_path.read_text(encoding="utf-8")) or {}
        self._scenes = raw.get("scenes", {}) if isinstance(raw, dict) else {}

    def apply_scene(self, scene_name: str, room_name: str) -> ToolResult:
        normalized = scene_name.strip().lower()
        selected_key = None
        for key in self._scenes.keys():
            if normalized == key.lower() or normalized in key.lower():
                selected_key = key
                break

        if not selected_key:
            return ToolResult(
                ok=False,
                action_code="SCENE_NOT_FOUND",
                speak_text="I could not find that scene.",
                private_note="Try one of your configured scenes.",
                error="scene_not_found",
            )

        scene_payload = self._scenes[selected_key]
        return ToolResult(
            ok=True,
            action_code="SCENE_APPLY",
            speak_text=f"Applying {selected_key} in the {room_name}.",
            private_note="Scene payload sent to your home bridge.",
            payload={
                "shortcut_action": "RUN_HOME_SCENE",
                "scene_name": selected_key,
                "room": room_name,
                "scene_payload": scene_payload,
            },
        )
