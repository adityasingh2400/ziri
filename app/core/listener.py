"""Always-on wake word listener with local STT and audio playback.

State machine:
    Idle  -->  (wake word detected)  -->  Listening
    Listening  -->  (silence / timeout)  -->  Transcribing
    Transcribing  -->  (transcript ready)  -->  Processing
    Processing  -->  (audio response)  -->  Speaking
    Speaking  -->  (playback done)  -->  Idle
"""

from __future__ import annotations

import asyncio
import collections
import enum
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from app.hub import AuraHub
    from app.settings import Settings

logger = logging.getLogger(__name__)

TARGET_RATE = 16000
MIC_CHANNELS = 1
WAKEWORD_CHUNK = 1280  # 80ms at 16kHz (openwakeword requirement)
MAX_LISTEN_SECONDS = 12
SILENCE_THRESHOLD_SECONDS = 1.2
ENERGY_SILENCE_THRESHOLD = 300
MAX_RETRIES = 5
RETRY_DELAY = 3.0
MAX_HISTORY = 50


class InteractionRecord:
    """One wake-word interaction."""
    __slots__ = ("timestamp", "transcript", "action_code", "speak_text", "latency_ms", "error")

    def __init__(self, timestamp: str, transcript: str, action_code: str = "",
                 speak_text: str = "", latency_ms: int = 0, error: str = ""):
        self.timestamp = timestamp
        self.transcript = transcript
        self.action_code = action_code
        self.speak_text = speak_text
        self.latency_ms = latency_ms
        self.error = error

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "transcript": self.transcript,
            "action_code": self.action_code,
            "speak_text": self.speak_text,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


def _downsample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Simple integer-ratio downsampling via decimation."""
    if src_rate == dst_rate:
        return audio
    ratio = src_rate / dst_rate
    if ratio == int(ratio):
        return audio[:: int(ratio)]
    # Non-integer ratio: linear interpolation
    n_out = int(len(audio) * dst_rate / src_rate)
    indices = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(indices, np.arange(len(audio)), audio.astype(np.float64)).astype(audio.dtype)


class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class Listener:
    """Runs a background thread that captures mic audio and drives the state machine."""

    def __init__(self, settings: Settings, hub: AuraHub) -> None:
        self.settings = settings
        self.hub = hub
        self.state = State.IDLE
        self._running = False

        self._wakeword_model = None
        self._whisper_model = None

        self._audio_buffer: collections.deque[np.ndarray] = collections.deque()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._native_rate: int = 0
        self._started_at: str = ""
        self._wake_count: int = 0
        self._current_transcript: str = ""
        self._current_response: str = ""
        self._last_active_ts: float = 0
        self._last_partial_ts: float = 0
        self.history: collections.deque[InteractionRecord] = collections.deque(maxlen=MAX_HISTORY)

    def _ensure_wakeword(self):
        if self._wakeword_model is not None:
            return
        from openwakeword.model import Model
        from openwakeword.utils import download_models
        download_models(model_names=[self.settings.wake_word_model])
        self._wakeword_model = Model(
            wakeword_models=[self.settings.wake_word_model],
            inference_framework="onnx",
        )
        logger.info("Wake word model loaded: %s", self.settings.wake_word_model)

    def _ensure_whisper(self):
        if self._whisper_model is not None:
            return
        from faster_whisper import WhisperModel
        self._whisper_model = WhisperModel(
            self.settings.whisper_model,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Whisper model loaded: %s", self.settings.whisper_model)

    def _detect_mic_rate(self) -> int:
        """Query the default input device's native sample rate."""
        try:
            info = sd.query_devices(kind="input")
            rate = int(info["default_samplerate"])
            logger.info(
                "Mic: %s (native rate=%d, channels=%d)",
                info["name"], rate, info["max_input_channels"],
            )
            return rate
        except Exception:
            logger.warning("Could not query mic info, assuming 48000")
            return 48000

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Begin listening. Call from the main async event loop."""
        self._loop = loop
        self._running = True
        self._ensure_wakeword()
        self._ensure_whisper()
        self._native_rate = self._detect_mic_rate()
        self._started_at = datetime.now(timezone.utc).isoformat()
        logger.info("Listener starting (wake word: %s)", self.settings.wake_word_model)
        import threading
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ziri-listener")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=3)
        try:
            sd.stop()
        except Exception:
            pass

    def get_status(self) -> dict:
        """Return listener state for the dashboard API."""
        return {
            "running": self._running,
            "state": self.state.value,
            "wake_word": self.settings.wake_word_model,
            "whisper_model": self.settings.whisper_model,
            "mic_rate": self._native_rate,
            "started_at": self._started_at,
            "wake_count": self._wake_count,
            "current_transcript": self._current_transcript,
            "current_response": self._current_response,
            "last_active_ts": self._last_active_ts,
            "history": [r.to_dict() for r in self.history],
        }

    # ------------------------------------------------------------------
    # Main audio loop (runs in its own thread)
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        # Record at the mic's native rate, downsample chunks to 16kHz for models
        native_rate = self._native_rate
        native_blocksize = int(native_rate * 0.03)  # 30ms in native samples

        ww_buffer = np.zeros(0, dtype=np.int16)

        def _audio_callback(indata: np.ndarray, frames: int, time_info, status):
            if status:
                logger.debug("Audio callback status: %s", status)
            raw = indata[:, 0].copy()
            downsampled = _downsample(raw, native_rate, TARGET_RATE)
            self._audio_buffer.append(downsampled)

        for attempt in range(1, MAX_RETRIES + 1):
            if not self._running:
                return
            try:
                logger.info(
                    "Opening mic stream (attempt %d, native_rate=%d, blocksize=%d)",
                    attempt, native_rate, native_blocksize,
                )
                with sd.InputStream(
                    samplerate=native_rate,
                    channels=MIC_CHANNELS,
                    dtype="int16",
                    blocksize=native_blocksize,
                    callback=_audio_callback,
                ):
                    logger.info("Microphone stream opened successfully")
                    while self._running:
                        if not self._audio_buffer:
                            time.sleep(0.01)
                            continue

                        chunk = self._audio_buffer.popleft()

                        if self.state == State.IDLE:
                            ww_buffer = np.concatenate([ww_buffer, chunk])
                            while len(ww_buffer) >= WAKEWORD_CHUNK:
                                frame = ww_buffer[:WAKEWORD_CHUNK]
                                ww_buffer = ww_buffer[WAKEWORD_CHUNK:]
                                self._check_wakeword(frame)

                        elif self.state == State.LISTENING:
                            self._capture_frames.append(chunk)
                            self._maybe_partial_transcribe()
                            self._check_end_of_speech(chunk)

                        elif self.state in (State.TRANSCRIBING, State.PROCESSING, State.SPEAKING):
                            pass  # drain buffer, ignore audio

                    return  # clean exit

            except Exception as exc:
                logger.warning("Mic stream failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("All mic stream attempts exhausted")

        self._running = False
        logger.info("Listener loop exited")

    # ------------------------------------------------------------------
    # Spotify volume ducking
    # ------------------------------------------------------------------

    def _duck_spotify(self) -> None:
        """Lower Spotify volume so Ziri can be heard."""
        try:
            self.hub.orchestrator.tool_runner.spotify.duck(duck_percent=20)
        except Exception as exc:
            logger.debug("Spotify duck failed (non-critical): %s", exc)

    def _unduck_spotify(self) -> None:
        """Gradually restore Spotify volume after interaction ends."""
        try:
            self.hub.orchestrator.tool_runner.spotify.unduck()
        except Exception as exc:
            logger.debug("Spotify unduck failed (non-critical): %s", exc)

    # ------------------------------------------------------------------
    # Idle -> Listening  (wake word detection)
    # ------------------------------------------------------------------

    def _check_wakeword(self, frame: np.ndarray) -> None:
        prediction = self._wakeword_model.predict(frame)
        for mdl_name, score in prediction.items():
            if score >= self.settings.wake_word_threshold:
                logger.info("Wake word detected! (%s, score=%.2f)", mdl_name, score)
                self._wakeword_model.reset()
                self._wake_count += 1
                self._transition_to_listening()
                return

    def _transition_to_listening(self) -> None:
        self.state = State.LISTENING
        self._capture_frames: list[np.ndarray] = []
        self._listen_start = time.monotonic()
        self._silence_start: float | None = None
        self._current_transcript = ""
        self._current_response = ""
        self._last_active_ts = time.time()
        self._last_partial_ts = time.monotonic()
        self._partial_busy = False

        self._duck_spotify()

        if self.settings.listener_chime_enabled:
            from app.core.audio_player import play_chime
            play_chime(blocking=True)

    # ------------------------------------------------------------------
    # Live partial transcription during listening (non-blocking)
    # ------------------------------------------------------------------

    _PARTIAL_INTERVAL = 1.5
    _MIN_AUDIO_FOR_PARTIAL = 1.5  # seconds of audio before first partial

    _HALLUCINATION_PATTERNS = [
        "thank", "thanks for watching", "subscribe", "like and subscribe",
        "okay, let", "okay let", "all right", "you", "bye",
        "the end", "i'm going to", "so,", "uh,", "um,",
    ]

    def _maybe_partial_transcribe(self) -> None:
        """Kick off a background partial transcription if conditions are met."""
        if self._partial_busy:
            return
        now = time.monotonic()
        if now - self._last_partial_ts < self._PARTIAL_INTERVAL:
            return
        if not self._capture_frames:
            return

        total_samples = sum(len(f) for f in self._capture_frames)
        audio_duration = total_samples / TARGET_RATE
        if audio_duration < self._MIN_AUDIO_FOR_PARTIAL:
            return

        self._last_partial_ts = now
        self._partial_busy = True
        snapshot = list(self._capture_frames)

        import threading
        threading.Thread(
            target=self._run_partial, args=(snapshot,), daemon=True,
        ).start()

    def _run_partial(self, frames: list[np.ndarray]) -> None:
        """Run whisper on a snapshot of frames (background thread)."""
        try:
            audio = np.concatenate(frames).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.005:
                return
            segments, _ = self._whisper_model.transcribe(
                audio, language="en", beam_size=1, vad_filter=False,
            )
            partial = " ".join(seg.text.strip() for seg in segments).strip()
            if partial and not self._is_hallucination(partial):
                self._current_transcript = partial
        except Exception:
            pass
        finally:
            self._partial_busy = False

    def _is_hallucination(self, text: str) -> bool:
        low = text.lower().strip()
        if len(low) < 3:
            return True
        for pattern in self._HALLUCINATION_PATTERNS:
            if low.startswith(pattern):
                return True
        return False

    # ------------------------------------------------------------------
    # Listening -> Transcribing  (energy-based silence detection)
    # ------------------------------------------------------------------

    @staticmethod
    def _rms_energy(samples: np.ndarray) -> float:
        """Root-mean-square energy of an int16 audio chunk."""
        return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

    def _check_end_of_speech(self, chunk: np.ndarray) -> None:
        elapsed = time.monotonic() - self._listen_start
        if elapsed > MAX_LISTEN_SECONDS:
            logger.info("Max listen time reached (%.1fs)", elapsed)
            self._transition_to_transcribing()
            return

        energy = self._rms_energy(chunk)
        is_speech = energy > ENERGY_SILENCE_THRESHOLD

        if is_speech:
            self._silence_start = None
        else:
            if self._silence_start is None:
                self._silence_start = time.monotonic()
            elif time.monotonic() - self._silence_start >= SILENCE_THRESHOLD_SECONDS:
                logger.info("Silence detected after %.1fs of listening", elapsed)
                self._transition_to_transcribing()

    # ------------------------------------------------------------------
    # Transcribing -> Processing  (faster-whisper STT)
    # ------------------------------------------------------------------

    def _transition_to_transcribing(self) -> None:
        self.state = State.TRANSCRIBING
        if not self._capture_frames:
            logger.info("No audio captured, returning to idle")
            self._unduck_spotify()
            self.state = State.IDLE
            return

        audio = np.concatenate(self._capture_frames).astype(np.float32) / 32768.0
        self._capture_frames = []

        segments, _info = self._whisper_model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()

        if not text or self._is_hallucination(text):
            logger.info("Empty or hallucinated transcript, returning to idle")
            self._unduck_spotify()
            self.state = State.IDLE
            return

        logger.info("Transcript: %s", text)
        self._current_transcript = text
        self._dispatch_intent(text)

    # ------------------------------------------------------------------
    # Processing -> Speaking  (intent dispatch + TTS playback)
    # ------------------------------------------------------------------

    def _dispatch_intent(self, text: str) -> None:
        self.state = State.PROCESSING
        if self._loop is None:
            logger.error("No event loop available")
            self._unduck_spotify()
            self.state = State.IDLE
            return

        future = asyncio.run_coroutine_threadsafe(self._async_handle(text), self._loop)
        try:
            future.result(timeout=30)
        except Exception:
            logger.exception("Intent processing failed")
        self._unduck_spotify()
        self.state = State.IDLE

    async def _async_handle(self, text: str) -> None:
        from app.schemas import IntentRequest
        from app.core.personality import QUICK_REPLY_ACTIONS

        t0 = time.monotonic()
        request = IntentRequest(
            user_id=self.settings.listener_user_id,
            device_id=self.settings.listener_device_id,
            room=self.settings.listener_room,
            raw_text=text,
            timestamp=datetime.now(timezone.utc),
        )

        try:
            response = await self.hub.handle_intent(request)
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "Response: action=%s speak=%s",
                response.action_code,
                response.speak_text[:80] if response.speak_text else "",
            )
            silent = response.action_code in QUICK_REPLY_ACTIONS
            self.history.appendleft(InteractionRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                transcript=text,
                action_code=response.action_code,
                speak_text="" if silent else (response.speak_text[:200] if response.speak_text else ""),
                latency_ms=latency_ms,
            ))
            self._current_response = "" if silent else (response.speak_text or "")
        except Exception as exc:
            self.history.appendleft(InteractionRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                transcript=text,
                error=str(exc)[:200],
            ))
            raise

        silent = response.action_code in QUICK_REPLY_ACTIONS
        if response.audio_url and not silent:
            self.state = State.SPEAKING
            from app.core.audio_player import play_audio_file
            from pathlib import Path
            static_root = Path(__file__).resolve().parent.parent / "static"
            if response.audio_url.startswith("/static/"):
                local_path = static_root / response.audio_url[len("/static/"):]
            else:
                local_path = static_root / "audio" / Path(response.audio_url).name
            if local_path.exists():
                play_audio_file(local_path, blocking=True)
            else:
                logger.warning("Audio file not found: %s", local_path)
