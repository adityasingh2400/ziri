"""Gesture-based vision controller using MediaPipe hand tracking.

Detection approach: fist = command mode.
- Close fist → lock start position, begin tracking
- Move fist right/left → next/previous track
- Move fist up/down → volume up/down
- Open hand while fist locked → pause
- Close fist from open hand → resume

All directional gestures use relative displacement from the fist lock point,
making detection robust regardless of where the hand starts in frame.
"""

from __future__ import annotations

import collections
import enum
import logging
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions as _BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker as _HandLandmarker,
    HandLandmarkerOptions as _HandLandmarkerOptions,
    RunningMode as _RunningMode,
)

from app.schemas import IntentRequest, IntentType, RouterDecision, ToolResult

if TYPE_CHECKING:
    import asyncio

    from app.hub import ZiriHub
    from app.settings import Settings

logger = logging.getLogger(__name__)

# ── Tuning knobs ──────────────────────────────────────────────────────────
COOLDOWN_SECONDS = 0.5
TOGGLE_COOLDOWN_SECONDS = 1.5
FIST_MAX_FINGERS = 1
OPEN_MIN_FINGERS = 3
FIST_LOCK_FRAMES = 4
MOVE_THRESHOLD = 0.10
PINCH_DISTANCE = 0.07

_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task")

# ── Landmark indices ─────────────────────────────────────────────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_PIPS = (INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)

_HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
]

_COLOR_IDLE = (200, 200, 200)
_COLOR_TRACKING = (80, 255, 120)
_COLOR_GESTURE = (80, 180, 255)
_COLOR_LINE_IDLE = (120, 120, 120)
_COLOR_LINE_TRACKING = (50, 200, 80)
_COLOR_LINE_GESTURE = (60, 140, 220)
_COLOR_CROSSHAIR = (0, 200, 255)


class Gesture(enum.Enum):
    PAUSE = "pause"
    RESUME = "resume"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    NEXT_TRACK = "next_track"
    PREV_TRACK = "prev_track"


_GESTURE_TO_DECISION: dict[Gesture, tuple[str, dict[str, Any], str]] = {
    Gesture.PAUSE: ("spotify.pause", {}, "MUSIC_PAUSE"),
    Gesture.RESUME: ("spotify.resume", {}, "MUSIC_RESUME"),
    Gesture.VOLUME_UP: ("spotify.adjust_volume", {"delta_percent": 20}, "MUSIC_VOLUME"),
    Gesture.VOLUME_DOWN: ("spotify.adjust_volume", {"delta_percent": -20}, "MUSIC_VOLUME"),
    Gesture.NEXT_TRACK: ("spotify.skip", {}, "MUSIC_SKIP"),
    Gesture.PREV_TRACK: ("spotify.previous", {}, "MUSIC_PREVIOUS"),
}


class GestureRecord:
    __slots__ = ("timestamp", "gesture", "action_code", "ok")

    def __init__(self, timestamp: str, gesture: str, action_code: str = "", ok: bool = True):
        self.timestamp = timestamp
        self.gesture = gesture
        self.action_code = action_code
        self.ok = ok


class _TrackingState:
    """Fist-lock tracking state."""
    __slots__ = ("origin_x", "origin_y", "locked")

    def __init__(self) -> None:
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.locked = False

    def lock(self, x: float, y: float) -> None:
        self.origin_x = x
        self.origin_y = y
        self.locked = True

    def reset(self) -> None:
        self.locked = False

    def displacement(self, x: float, y: float) -> tuple[float, float]:
        return x - self.origin_x, y - self.origin_y


class GestureRecognizer:
    MAX_HISTORY = 50

    def __init__(self, hub: ZiriHub, settings: Settings) -> None:
        self.hub = hub
        self.settings = settings

        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._cooldowns: dict[Gesture, float] = {}
        self._last_toggle_time: float = 0.0
        self._last_gesture_flash: float = 0.0
        self._last_gesture_name: str = ""

        self._tracking = _TrackingState()
        self._fist_frames: int = 0
        self._paused: bool = False

        self.history: collections.deque[GestureRecord] = collections.deque(maxlen=self.MAX_HISTORY)
        self._gesture_count = 0
        self._frame_ts_ms = 0

        self._latest_jpeg: bytes = b""
        self._jpeg_lock = threading.Lock()

        self._prebuilt_decisions: dict[Gesture, tuple[RouterDecision, str]] = {}
        self._prebuilt_request: IntentRequest | None = None
        self._prebuilt_device_ctx: Any = None

    def _prebuild_dispatch_objects(self) -> None:
        for gesture, (tool_name, tool_args, action_code) in _GESTURE_TO_DECISION.items():
            self._prebuilt_decisions[gesture] = (
                RouterDecision(
                    intent_type=IntentType.MUSIC_COMMAND,
                    tool_name=tool_name, tool_args=tool_args,
                    action_code=action_code, confidence=1.0,
                ),
                action_code,
            )
        self._prebuilt_request = IntentRequest(
            user_id=self.settings.listener_user_id,
            device_id=self.settings.listener_device_id,
            room=self.settings.listener_room,
            raw_text="[gesture]",
            timestamp=datetime.now(timezone.utc),
        )
        self._prebuilt_device_ctx = self.hub.device_registry.resolve_context(
            self.settings.listener_device_id,
            self.settings.listener_room,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        if self._running:
            return
        self._loop = loop
        self._running = True
        self._prebuild_dispatch_objects()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="gesture-vision")
        self._thread.start()
        logger.info("GestureRecognizer started")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        logger.info("GestureRecognizer stopped")

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "tracking": self._tracking.locked,
            "gesture_count": self._gesture_count,
            "history": [
                {"timestamp": r.timestamp, "gesture": r.gesture, "action_code": r.action_code, "ok": r.ok}
                for r in self.history
            ],
        }

    def get_jpeg_frame(self) -> bytes:
        with self._jpeg_lock:
            return self._latest_jpeg

    # ------------------------------------------------------------------
    # Capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open camera")
            self._running = False
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._latest_landmarks: list | None = None
        self._latest_handedness: str = "Right"
        self._landmark_lock = threading.Lock()

        def _on_result(result, image, ts_ms):
            if result.hand_landmarks:
                h = "Right"
                if result.handedness and result.handedness[0]:
                    h = result.handedness[0][0].category_name
                with self._landmark_lock:
                    self._latest_landmarks = result.hand_landmarks[0]
                    self._latest_handedness = h
            else:
                with self._landmark_lock:
                    self._latest_landmarks = None

        landmarker = _HandLandmarker.create_from_options(_HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=_RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=_on_result,
        ))

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Vision capture running (%dx%d)", w, h)
        self._frame_ts_ms = 0
        enc = [cv2.IMWRITE_JPEG_QUALITY, 80]

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                self._frame_ts_ms += 33
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._frame_ts_ms)

                with self._landmark_lock:
                    lm = self._latest_landmarks
                    handed = self._latest_handedness

                now = time.monotonic()
                if lm is not None:
                    self._process(lm, handed, now)
                    self._draw_skeleton(frame, lm, w, h, now)
                else:
                    self._fist_frames = 0
                    if self._tracking.locked:
                        self._tracking.reset()

                mirrored = cv2.flip(frame, 1)
                self._draw_hud(mirrored, lm, w, h, now)
                _, jpg = cv2.imencode(".jpg", mirrored, enc)
                with self._jpeg_lock:
                    self._latest_jpeg = jpg.tobytes()
        finally:
            landmarker.close()
            cap.release()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_skeleton(self, frame: np.ndarray, lm: list, w: int, h: int, now: float) -> None:
        flash = (now - self._last_gesture_flash) < 0.4
        tracking = self._tracking.locked

        if flash:
            pc, lc = _COLOR_GESTURE, _COLOR_LINE_GESTURE
        elif tracking:
            pc, lc = _COLOR_TRACKING, _COLOR_LINE_TRACKING
        else:
            pc, lc = _COLOR_IDLE, _COLOR_LINE_IDLE

        pts = [(int(l.x * w), int(l.y * h)) for l in lm]
        for a, b in _HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], lc, 2, cv2.LINE_AA)
        for i, (px, py) in enumerate(pts):
            r = 6 if i in (*FINGER_TIPS, THUMB_TIP, WRIST) else 3
            cv2.circle(frame, (px, py), r, pc, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), r, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw crosshair at lock origin + displacement line
        if tracking:
            ox = int(self._tracking.origin_x * w)
            oy = int(self._tracking.origin_y * h)
            cx = int(lm[WRIST].x * w)
            cy = int(lm[WRIST].y * h)
            cv2.drawMarker(frame, (ox, oy), _COLOR_CROSSHAIR, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            cv2.line(frame, (ox, oy), (cx, cy), _COLOR_CROSSHAIR, 2, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, lm: list | None, w: int, h: int, now: float) -> None:
        flash = (now - self._last_gesture_flash) < 0.4
        tracking = self._tracking.locked

        if flash and self._last_gesture_name:
            hud = f">> {self._last_gesture_name.upper()} <<"
            color = _COLOR_GESTURE
        elif tracking and lm is not None:
            dx, dy = self._tracking.displacement(lm[WRIST].x, lm[WRIST].y)
            hud = f"TRACKING  dx:{dx:+.2f}  dy:{dy:+.2f}"
            color = _COLOR_TRACKING
        elif lm is not None:
            fc = self._count_fingers(lm)
            hud = f"{fc} fingers | {'FIST' if fc <= FIST_MAX_FINGERS else 'open'}"
            color = _COLOR_IDLE
        else:
            hud = "no hand"
            color = _COLOR_IDLE

        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Gesture logic
    # ------------------------------------------------------------------

    def _process(self, lm: list, handedness: str, now: float) -> None:
        fc = self._count_fingers(lm)
        is_fist = fc <= FIST_MAX_FINGERS
        is_open = fc >= OPEN_MIN_FINGERS
        wx, wy = lm[WRIST].x, lm[WRIST].y

        pinch_dist = math.hypot(
            lm[THUMB_TIP].x - lm[INDEX_TIP].x,
            lm[THUMB_TIP].y - lm[INDEX_TIP].y,
        )

        if self._tracking.locked:
            if pinch_dist < PINCH_DISTANCE:
                if (now - self._last_toggle_time) >= TOGGLE_COOLDOWN_SECONDS:
                    g = Gesture.RESUME if self._paused else Gesture.PAUSE
                    self._fire(g, now)
                    self._paused = not self._paused
                    self._last_toggle_time = now
                self._tracking.reset()
                self._fist_frames = 0
                return

            # Hand opened → just release tracking (no action)
            if is_open:
                self._tracking.reset()
                return

            if not is_fist:
                return

            dx, dy = self._tracking.displacement(wx, wy)
            adx, ady = abs(dx), abs(dy)

            if adx > ady and adx > MOVE_THRESHOLD:
                g = Gesture.NEXT_TRACK if dx > 0 else Gesture.PREV_TRACK
                if self._cooldown_ok(g, now):
                    self._fire(g, now)
                self._tracking.reset()
            elif ady > adx and ady > MOVE_THRESHOLD:
                g = Gesture.VOLUME_UP if dy < 0 else Gesture.VOLUME_DOWN
                if self._cooldown_ok(g, now):
                    self._fire(g, now)
                self._tracking.lock(wx, wy)
        else:
            if is_fist:
                if (now - self._last_toggle_time) < TOGGLE_COOLDOWN_SECONDS:
                    return
                self._fist_frames += 1
                if self._fist_frames >= FIST_LOCK_FRAMES:
                    self._tracking.lock(wx, wy)
                    logger.info("Fist locked at (%.2f, %.2f)", wx, wy)
            else:
                self._fist_frames = 0

    @staticmethod
    def _count_fingers(lm: list) -> int:
        count = 0
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
            if lm[tip].y < lm[pip].y:
                count += 1
        return count

    def _cooldown_ok(self, gesture: Gesture, now: float) -> bool:
        return (now - self._cooldowns.get(gesture, 0.0)) >= COOLDOWN_SECONDS

    # ------------------------------------------------------------------
    # Fire + dispatch
    # ------------------------------------------------------------------

    def _fire(self, gesture: Gesture, now: float) -> None:
        self._cooldowns[gesture] = now
        self._last_gesture_flash = now
        self._last_gesture_name = gesture.value
        self._gesture_count += 1
        logger.info("Gesture: %s", gesture.value)
        self.history.appendleft(GestureRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            gesture=gesture.value,
            action_code=_GESTURE_TO_DECISION[gesture][2],
            ok=True,
        ))
        threading.Thread(target=self._dispatch, args=(gesture,), daemon=True).start()

    def _dispatch(self, gesture: Gesture) -> None:
        pair = self._prebuilt_decisions.get(gesture)
        if not pair:
            return
        decision, _ = pair
        try:
            r = self.hub.orchestrator.tool_runner.run(
                decision, self._prebuilt_request, self._prebuilt_device_ctx,
            )
            logger.info("  -> %s (ok=%s)", decision.tool_name, r.ok)
        except Exception:
            logger.exception("Dispatch failed: %s", gesture.value)
