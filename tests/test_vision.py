"""Unit tests for the fist-based gesture recognition system."""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core.vision import (
    COOLDOWN_SECONDS,
    FIST_LOCK_FRAMES,
    FIST_MAX_FINGERS,
    MOVE_THRESHOLD,
    OPEN_MIN_FINGERS,
    FINGER_TIPS,
    FINGER_PIPS,
    THUMB_TIP,
    INDEX_TIP,
    WRIST,
    Gesture,
    GestureRecognizer,
    _TrackingState,
)


def _lm(x: float = 0.5, y: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0)


def _hand(
    fingers_up: int = 0,
    wrist_x: float = 0.5,
    wrist_y: float = 0.5,
    pinch: bool = False,
) -> list[SimpleNamespace]:
    """Build landmarks with a given number of fingers up (0-4)."""
    lm = [_lm() for _ in range(21)]
    lm[WRIST] = _lm(x=wrist_x, y=wrist_y)
    for i in range(4):
        lm[FINGER_PIPS[i]] = _lm(y=0.55)
        lm[FINGER_TIPS[i]] = _lm(y=0.30 if i < fingers_up else 0.75)
    # Thumb defaults: away from index
    lm[THUMB_TIP] = _lm(x=0.35, y=0.50)
    if pinch:
        lm[THUMB_TIP] = _lm(x=0.50, y=0.40)
        lm[FINGER_TIPS[0]] = _lm(x=0.51, y=0.41)  # index tip near thumb
    return lm


def _gr() -> GestureRecognizer:
    hub = MagicMock()
    hub.device_registry.resolve_context.return_value = MagicMock()
    hub.orchestrator.tool_runner.run.return_value = MagicMock(ok=True)
    s = MagicMock()
    s.listener_user_id = "test"
    s.listener_device_id = "dev"
    s.listener_room = "room"
    g = GestureRecognizer(hub=hub, settings=s)
    g._prebuild_dispatch_objects()
    return g


class TestFingerCount:
    def test_fist(self) -> None:
        assert _gr()._count_fingers(_hand(0)) == 0

    def test_open(self) -> None:
        assert _gr()._count_fingers(_hand(4)) == 4

    def test_two(self) -> None:
        assert _gr()._count_fingers(_hand(2)) == 2


class TestTrackingState:
    def test_lock_and_displacement(self) -> None:
        t = _TrackingState()
        t.lock(0.5, 0.5)
        assert t.locked
        dx, dy = t.displacement(0.7, 0.3)
        assert abs(dx - 0.2) < 1e-6
        assert abs(dy - (-0.2)) < 1e-6

    def test_reset(self) -> None:
        t = _TrackingState()
        t.lock(0.5, 0.5)
        t.reset()
        assert not t.locked


class TestFistLocking:
    def test_fist_locks_after_n_frames(self) -> None:
        g = _gr()
        fist = _hand(0)
        now = time.monotonic()
        for i in range(FIST_LOCK_FRAMES + 1):
            g._process(fist, "Right", now + i * 0.03)
        assert g._tracking.locked

    def test_open_hand_does_not_lock(self) -> None:
        g = _gr()
        open_h = _hand(4)
        now = time.monotonic()
        for i in range(FIST_LOCK_FRAMES + 2):
            g._process(open_h, "Right", now + i * 0.03)
        assert not g._tracking.locked

    def test_fist_not_enough_frames(self) -> None:
        g = _gr()
        fist = _hand(0)
        now = time.monotonic()
        for i in range(FIST_LOCK_FRAMES - 2):
            g._process(fist, "Right", now + i * 0.03)
        assert not g._tracking.locked


class TestDirectionalGestures:
    def _lock_fist(self, g: GestureRecognizer, wx: float = 0.5, wy: float = 0.5) -> float:
        """Lock a fist at (wx, wy) and return the timestamp."""
        now = time.monotonic()
        fist = _hand(0, wrist_x=wx, wrist_y=wy)
        for i in range(FIST_LOCK_FRAMES + 1):
            g._process(fist, "Right", now + i * 0.03)
        assert g._tracking.locked
        return now + FIST_LOCK_FRAMES * 0.03

    def test_move_right_next_track(self) -> None:
        g = _gr()
        t = self._lock_fist(g, 0.5, 0.5)
        moved = _hand(0, wrist_x=0.5 + MOVE_THRESHOLD + 0.05, wrist_y=0.5)
        g._process(moved, "Right", t + 0.5)
        assert any(r.gesture == "next_track" for r in g.history)

    def test_move_left_prev_track(self) -> None:
        g = _gr()
        t = self._lock_fist(g, 0.5, 0.5)
        moved = _hand(0, wrist_x=0.5 - MOVE_THRESHOLD - 0.05, wrist_y=0.5)
        g._process(moved, "Right", t + 0.5)
        assert any(r.gesture == "prev_track" for r in g.history)

    def test_move_up_volume_up(self) -> None:
        g = _gr()
        t = self._lock_fist(g, 0.5, 0.5)
        moved = _hand(0, wrist_x=0.5, wrist_y=0.5 - MOVE_THRESHOLD - 0.05)
        g._process(moved, "Right", t + 0.5)
        assert any(r.gesture == "volume_up" for r in g.history)

    def test_move_down_volume_down(self) -> None:
        g = _gr()
        t = self._lock_fist(g, 0.5, 0.5)
        moved = _hand(0, wrist_x=0.5, wrist_y=0.5 + MOVE_THRESHOLD + 0.05)
        g._process(moved, "Right", t + 0.5)
        assert any(r.gesture == "volume_down" for r in g.history)

    def test_small_movement_ignored(self) -> None:
        g = _gr()
        t = self._lock_fist(g, 0.5, 0.5)
        moved = _hand(0, wrist_x=0.53, wrist_y=0.5)
        g._process(moved, "Right", t + 0.5)
        assert g._gesture_count == 0

    def test_volume_resets_origin_for_chaining(self) -> None:
        g = _gr()
        t = self._lock_fist(g, 0.5, 0.5)
        up1 = _hand(0, wrist_x=0.5, wrist_y=0.5 - MOVE_THRESHOLD - 0.05)
        g._process(up1, "Right", t + 0.5)
        assert g._gesture_count == 1
        assert g._tracking.locked


class TestPauseResume:
    def test_pinch_while_tracking_pauses(self) -> None:
        g = _gr()
        now = time.monotonic()
        fist = _hand(0)
        for i in range(FIST_LOCK_FRAMES + 1):
            g._process(fist, "Right", now + i * 0.03)
        assert g._tracking.locked
        t = now + FIST_LOCK_FRAMES * 0.03
        pinch = _hand(0, pinch=True)
        g._process(pinch, "Right", t + 0.5)
        assert any(r.gesture == "pause" for r in g.history)
        assert not g._tracking.locked
        assert g._paused is True

    def test_pinch_toggles_resume(self) -> None:
        """Second pinch after pause should fire resume."""
        g = _gr()
        g._paused = True
        now = time.monotonic()
        fist = _hand(0)
        for i in range(FIST_LOCK_FRAMES + 1):
            g._process(fist, "Right", now + i * 0.03)
        t = now + FIST_LOCK_FRAMES * 0.03
        pinch = _hand(0, pinch=True)
        g._process(pinch, "Right", t + 0.5)
        assert any(r.gesture == "resume" for r in g.history)
        assert g._paused is False

    def test_open_hand_releases_without_action(self) -> None:
        g = _gr()
        now = time.monotonic()
        fist = _hand(0)
        for i in range(FIST_LOCK_FRAMES + 1):
            g._process(fist, "Right", now + i * 0.03)
        assert g._tracking.locked
        t = now + FIST_LOCK_FRAMES * 0.03
        open_h = _hand(4)
        g._process(open_h, "Right", t + 0.5)
        assert not g._tracking.locked
        assert g._gesture_count == 0

    def test_fist_lock_does_not_fire_resume(self) -> None:
        """Making a fist should only enter tracking, not fire resume."""
        g = _gr()
        now = time.monotonic()
        fist = _hand(0)
        for i in range(FIST_LOCK_FRAMES + 2):
            g._process(fist, "Right", now + i * 0.03)
        assert g._tracking.locked
        assert g._gesture_count == 0


class TestCooldown:
    def test_blocks_rapid(self) -> None:
        g = _gr()
        now = time.monotonic()
        g._cooldowns[Gesture.NEXT_TRACK] = now
        assert not g._cooldown_ok(Gesture.NEXT_TRACK, now + 0.1)

    def test_allows_after(self) -> None:
        g = _gr()
        now = time.monotonic()
        g._cooldowns[Gesture.NEXT_TRACK] = now
        assert g._cooldown_ok(Gesture.NEXT_TRACK, now + COOLDOWN_SECONDS + 0.01)
