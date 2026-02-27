"""Thin audio playback wrapper around sounddevice + soundfile."""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

_AUDIO_ROOT = Path(__file__).resolve().parent.parent / "static" / "audio"
_CHIME_FILE = _AUDIO_ROOT / "Single_short_soft_di_#2-1772184508490.mp3"
_THINKING_FILE = _AUDIO_ROOT / "Looping_beeping_sexy_#4-1772184390291.mp3"

_MIN_CHIME_VOLUME = 0.85


# ---- Wake word chime (from 11Labs sound effect) ----

_CHIME_SAMPLES: tuple[np.ndarray, int] | None = None


def _get_chime() -> tuple[np.ndarray, int]:
    global _CHIME_SAMPLES
    if _CHIME_SAMPLES is None:
        data, sr = sf.read(str(_CHIME_FILE), dtype="float32")
        peak = np.abs(data).max()
        if 0 < peak < _MIN_CHIME_VOLUME:
            data = data * (_MIN_CHIME_VOLUME / peak)
        _CHIME_SAMPLES = (data, sr)
    return _CHIME_SAMPLES


def play_chime(blocking: bool = False) -> None:
    """Play the wake-word confirmation blip."""
    data, sr = _get_chime()
    if blocking:
        sd.play(data, samplerate=sr)
        sd.wait()
    else:
        threading.Thread(
            target=lambda: (sd.play(data, samplerate=sr), sd.wait()),
            daemon=True,
        ).start()


# ---- General playback ----

def play_audio_file(path: str | Path, blocking: bool = True) -> None:
    """Play an MP3 or WAV file through the default output device."""
    data, sr = sf.read(str(path), dtype="float32")
    sd.play(data, samplerate=sr)
    if blocking:
        sd.wait()


# ---- Thinking loop (from 11Labs sound effect) ----

_THINKING_SAMPLES: tuple[np.ndarray, int] | None = None
_thinking_stop = threading.Event()
_thinking_thread: threading.Thread | None = None


def _get_thinking() -> tuple[np.ndarray, int]:
    global _THINKING_SAMPLES
    if _THINKING_SAMPLES is None:
        data, sr = sf.read(str(_THINKING_FILE), dtype="float32")
        _THINKING_SAMPLES = (data, sr)
    return _THINKING_SAMPLES


def start_thinking_sound() -> None:
    """Start looping the thinking sound effect. Call stop_thinking_sound() to end."""
    global _thinking_thread
    stop_thinking_sound()
    _thinking_stop.clear()
    data, sr = _get_thinking()

    def _loop():
        while not _thinking_stop.is_set():
            sd.play(data, samplerate=sr)
            sd.wait()

    _thinking_thread = threading.Thread(target=_loop, daemon=True, name="ziri-thinking")
    _thinking_thread.start()


def stop_thinking_sound() -> None:
    """Stop the thinking loop."""
    global _thinking_thread
    _thinking_stop.set()
    try:
        sd.stop()
    except Exception:
        pass
    if _thinking_thread and _thinking_thread.is_alive():
        _thinking_thread.join(timeout=1)
    _thinking_thread = None


# ---- Utility playback ----

def play_audio_bytes(audio_bytes: bytes, blocking: bool = True) -> None:
    """Play raw MP3/WAV bytes through the default output device."""
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    sd.play(data, samplerate=sr)
    if blocking:
        sd.wait()


def play_audio_url(url: str, blocking: bool = True) -> bool:
    """Download and play audio from a URL. Returns True on success."""
    try:
        import httpx
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        if resp.status_code != 200:
            logger.warning("Failed to fetch audio (status %s): %s", resp.status_code, url)
            return False
        play_audio_bytes(resp.content, blocking=blocking)
        return True
    except Exception as exc:
        logger.warning("Audio playback failed for %s: %s", url, exc)
        return False
