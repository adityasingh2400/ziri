"""Always-on wake word listener with local STT and audio playback.

State machine:
    Idle  -->  (wake word detected)  -->  Listening
    Listening  -->  (silence / timeout)  -->  Transcribing
    Transcribing  -->  (transcript ready)  -->  Processing
    Processing  -->  (audio response)  -->  Speaking
    Speaking  -->  (playback done)  -->  Idle

End-of-speech uses Silero VAD (ONNX) for robust utterance boundary
detection that works in noisy rooms with background chatter.
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
    from app.hub import ZiriHub
    from app.settings import Settings

logger = logging.getLogger(__name__)

TARGET_RATE = 16000
MIC_CHANNELS = 1
WAKEWORD_CHUNK = 1280  # 80ms at 16kHz (openwakeword requirement)
VAD_CHUNK = 512         # 32ms at 16kHz (Silero VAD requirement)
MAX_LISTEN_SECONDS = 10
MAX_RETRIES = 5
RETRY_DELAY = 3.0
MAX_HISTORY = 50

# --------------- Silero VAD tuning ---------------
VAD_THRESHOLD = 0.40            # prob above which a 32ms chunk is "speech"
VAD_SILENCE_AFTER_SPEECH = 0.55 # seconds of non-speech after user stops → end
VAD_NO_SPEECH_TIMEOUT = 3.0     # abort if nobody speaks within this window
VAD_MIN_SPEECH_CHUNKS = 5       # ~160ms of confirmed speech before allowing end


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

    def __init__(self, settings: Settings, hub: ZiriHub) -> None:
        self.settings = settings
        self.hub = hub
        self.state = State.IDLE
        self._running = False

        self._wakeword_model = None
        self._whisper_model = None
        self._vad_model = None

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

    # ------------------------------------------------------------------
    # Model loaders
    # ------------------------------------------------------------------

    def _ensure_vad(self):
        """Load Silero VAD (ONNX) once. ~2 MB model, <1 ms per chunk."""
        if self._vad_model is not None:
            return
        from silero_vad import load_silero_vad
        self._vad_model = load_silero_vad(onnx=True)
        import torch
        self._torch = torch
        self._vad_tensor = torch.zeros(VAD_CHUNK, dtype=torch.float32)
        self._vad_np_buf = np.zeros(VAD_CHUNK, dtype=np.float32)
        logger.info("Silero VAD loaded (ONNX)")

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
        if self.settings.elevenlabs_api_key:
            logger.info("Using ElevenLabs Scribe for cloud STT")
            self._whisper_model = "elevenlabs"
            return
        from faster_whisper import WhisperModel
        self._whisper_model = WhisperModel(
            self.settings.whisper_model,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Whisper model loaded (local): %s", self.settings.whisper_model)

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
        self._ensure_vad()
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
        vad_buffer = np.zeros(0, dtype=np.int16)

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
                            time.sleep(0.002)
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
                            if self._whisper_model == "elevenlabs":
                                self._send_audio_to_elevenlabs(chunk)
                            else:
                                self._maybe_partial_transcribe()

                            vad_buffer = np.concatenate([vad_buffer, chunk])
                            while len(vad_buffer) >= VAD_CHUNK:
                                vad_frame = vad_buffer[:VAD_CHUNK]
                                vad_buffer = vad_buffer[VAD_CHUNK:]
                                self._vad_step(vad_frame)
                                if self.state != State.LISTENING:
                                    vad_buffer = np.zeros(0, dtype=np.int16)
                                    break

                        elif self.state in (State.TRANSCRIBING, State.PROCESSING, State.SPEAKING):
                            vad_buffer = np.zeros(0, dtype=np.int16)

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
        logger.info("[DEBUG] === STATE -> LISTENING ===")
        self._capture_frames: list[np.ndarray] = []
        self._listen_start = time.monotonic()
        self._current_transcript = ""
        self._current_response = ""
        self._last_active_ts = time.time()
        self._last_partial_ts = time.monotonic()
        self._partial_busy = False
        self._el_chunks_sent = 0
        self._el_chunks_backlogged = 0

        # VAD sub-state for end-of-speech detection
        self._vad_speech_started = False
        self._vad_speech_chunks = 0
        self._vad_silence_start: float | None = None
        if self._vad_model is not None:
            self._vad_model.reset_states()

        # ElevenLabs realtime STT WebSocket state
        self._el_ws = None
        self._el_final_text = ""
        self._el_session_ready = False
        self._el_audio_backlog: list[np.ndarray] = []

        self._duck_spotify()

        # Start WS BEFORE chime so it connects during chime playback
        if self._whisper_model == "elevenlabs":
            logger.info("[DEBUG] Starting ElevenLabs WS thread (before chime)...")
            import threading
            threading.Thread(target=self._start_elevenlabs_ws, daemon=True).start()

        # Chime disabled — sd.play() opens an OutputStream that conflicts with
        # the Bluetooth mic InputStream, killing audio capture for ~3 seconds.

    # ------------------------------------------------------------------
    # ElevenLabs Realtime STT (official SDK)
    # ------------------------------------------------------------------

    def _start_elevenlabs_ws(self) -> None:
        """Open ElevenLabs Scribe Realtime via the official SDK."""
        try:
            from elevenlabs import AudioFormat, CommitStrategy, ElevenLabs, RealtimeEvents, RealtimeAudioOptions

            client = ElevenLabs(api_key=self.settings.elevenlabs_api_key)

            el_loop = asyncio.new_event_loop()
            self._el_loop = el_loop

            async def _run():
                connection = await client.speech_to_text.realtime.connect(
                    RealtimeAudioOptions(
                        model_id="scribe_v2_realtime",
                        audio_format=AudioFormat.PCM_16000,
                        sample_rate=TARGET_RATE,
                        language_code="en",
                        commit_strategy=CommitStrategy.VAD,
                        vad_silence_threshold_secs=0.5,
                    )
                )
                self._el_connection = connection

                def on_session_started(data):
                    logger.info("[DEBUG] SDK session_started, %.2fs since listen, backlog=%d",
                                time.monotonic() - self._listen_start, len(self._el_audio_backlog))
                    self._el_session_ready = True
                    asyncio.create_task(_flush_backlog())

                async def _flush_backlog():
                    backlog = self._el_audio_backlog
                    self._el_audio_backlog = []
                    logger.info("[DEBUG] Flushing %d backlogged chunks", len(backlog))
                    import base64
                    first = True
                    for chunk in backlog:
                        payload = {"audio_base_64": base64.b64encode(chunk.tobytes()).decode()}
                        if first:
                            payload["previous_text"] = (
                                "Ziri, Spotify, playlist, play, pause, resume, skip, "
                                "shuffle, volume, queue, weather, calendar, news, NBA"
                            )
                            first = False
                        await connection.send(payload)

                def on_partial(data):
                    text = data.get("text", "").strip() if isinstance(data, dict) else ""
                    if text:
                        self._current_transcript = text
                        logger.info("[DEBUG] SDK partial: '%s' (%.2fs)",
                                    text, time.monotonic() - self._listen_start)

                def on_committed(data):
                    text = data.get("text", "").strip() if isinstance(data, dict) else ""
                    if text:
                        self._el_final_text = text
                        self._current_transcript = text
                        logger.info("[DEBUG] SDK committed: '%s'", text)

                def on_error(error):
                    logger.warning("[DEBUG] SDK error: %s", error)

                def on_close():
                    logger.info("[DEBUG] SDK connection closed")

                connection.on(RealtimeEvents.SESSION_STARTED, on_session_started)
                connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, on_partial)
                connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed)
                connection.on(RealtimeEvents.ERROR, on_error)
                connection.on(RealtimeEvents.CLOSE, on_close)

                while self.state in (State.LISTENING, State.TRANSCRIBING) and self._running:
                    await asyncio.sleep(0.05)

            try:
                el_loop.run_until_complete(_run())
            finally:
                el_loop.close()
                self._el_loop = None

        except Exception as exc:
            logger.warning("[DEBUG] ElevenLabs SDK STT failed: %s", exc)
            self._el_connection = None

    def _send_audio_to_elevenlabs(self, chunk: np.ndarray, is_first: bool = False, loop=None) -> None:
        """Send a PCM chunk to the ElevenLabs SDK connection."""
        conn = getattr(self, '_el_connection', None)
        if not conn:
            return
        if not self._el_session_ready:
            self._el_audio_backlog.append(chunk)
            self._el_chunks_backlogged = len(self._el_audio_backlog)
            if self._el_chunks_backlogged % 20 == 1:
                logger.info("[DEBUG] Backlogging chunk (total=%d)", self._el_chunks_backlogged)
            return
        try:
            import base64
            payload = {"audio_base_64": base64.b64encode(chunk.tobytes()).decode()}
            el_loop = self._el_loop
            if el_loop and el_loop.is_running():
                asyncio.run_coroutine_threadsafe(conn.send(payload), el_loop)
            self._el_chunks_sent += 1
            if self._el_chunks_sent % 30 == 1:
                logger.info("[DEBUG] Sent chunk #%d (%.2fs)",
                            self._el_chunks_sent, time.monotonic() - self._listen_start)
        except Exception as exc:
            logger.warning("[DEBUG] SDK send failed: %s", exc)

    def _close_elevenlabs_ws(self) -> None:
        """Close the ElevenLabs SDK connection."""
        self._el_session_ready = False
        self._el_audio_backlog = []
        conn = getattr(self, '_el_connection', None)
        el_loop = getattr(self, '_el_loop', None)
        if conn and el_loop and el_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(conn.close(), el_loop).result(timeout=2)
            except Exception:
                pass
        self._el_connection = None

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
        """Run transcription on a snapshot of frames (background thread)."""
        try:
            audio = np.concatenate(frames).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.005:
                return
            if self._whisper_model == "elevenlabs":
                partial = self._transcribe_elevenlabs(audio)
            else:
                model = self._ensure_local_whisper()
                segments, _ = model.transcribe(
                    audio, language="en", beam_size=1, vad_filter=False,
                    initial_prompt=self._TRANSCRIPTION_PROMPT,
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
    # Listening -> Transcribing  (Silero VAD end-of-speech detection)
    #
    # Sub-states within LISTENING:
    #   1. WAITING  – wake word just fired, waiting for user to start talking
    #   2. ACTIVE   – user is mid-utterance (VAD confirms speech)
    #   3. TRAILING – user may have stopped; counting consecutive non-speech
    #                 chunks. If the gap exceeds VAD_SILENCE_AFTER_SPEECH → done.
    #                 If speech resumes → back to ACTIVE.
    # ------------------------------------------------------------------

    def _vad_step(self, frame: np.ndarray) -> None:
        """Run one 32 ms VAD frame through Silero and drive end-of-speech logic."""
        elapsed = time.monotonic() - self._listen_start

        if elapsed > MAX_LISTEN_SECONDS:
            logger.info("Max listen time reached (%.1fs)", elapsed)
            self._transition_to_transcribing()
            return

        # Reuse pre-allocated tensor to avoid allocation per chunk
        np.divide(frame, 32768.0, out=self._vad_np_buf)
        self._vad_tensor[:] = self._torch.from_numpy(self._vad_np_buf)
        prob = self._vad_model(self._vad_tensor, TARGET_RATE).item()

        is_speech = prob >= VAD_THRESHOLD

        if not self._vad_speech_started:
            # --- WAITING: no speech yet after wake word ---
            if is_speech:
                self._vad_speech_started = True
                self._vad_speech_chunks = 1
                self._vad_silence_start = None
                logger.info("VAD: speech started at %.2fs (prob=%.2f)", elapsed, prob)
            elif elapsed > VAD_NO_SPEECH_TIMEOUT:
                logger.info("VAD: no speech within %.1fs, aborting", VAD_NO_SPEECH_TIMEOUT)
                self._close_elevenlabs_ws()
                self._unduck_spotify()
                self.state = State.IDLE
            return

        # --- ACTIVE / TRAILING: user has spoken at least once ---
        if is_speech:
            self._vad_speech_chunks += 1
            self._vad_silence_start = None
            return

        # Non-speech frame while user was talking → start/continue trailing
        if self._vad_silence_start is None:
            self._vad_silence_start = time.monotonic()

        silence_dur = time.monotonic() - self._vad_silence_start

        # Only end if we've captured enough real speech to be meaningful
        if (self._vad_speech_chunks >= VAD_MIN_SPEECH_CHUNKS
                and silence_dur >= VAD_SILENCE_AFTER_SPEECH):
            logger.info(
                "VAD: end-of-speech at %.2fs (speech_chunks=%d, silence=%.2fs, last_prob=%.2f)",
                elapsed, self._vad_speech_chunks, silence_dur, prob,
            )
            self._transition_to_transcribing()

    # ------------------------------------------------------------------
    # Transcribing -> Processing  (faster-whisper STT)
    # ------------------------------------------------------------------

    # Domain vocabulary hint for local Whisper fallback
    _TRANSCRIPTION_PROMPT = (
        "Ziri, Spotify, playlist, play, pause, resume, skip, shuffle, "
        "volume, queue, weather, calendar, remind, reminder, news, "
        "NBA, Lakers, Celtics, Warriors, "
        "goodnight, movie mode, focus, "
        "Hey Jarvis"
    )

    _stt_http: "httpx.Client | None" = None
    _stt_ssl_ctx = None

    def _get_stt_http(self):
        """Long-lived httpx client with cached SSL for ElevenLabs STT REST fallback."""
        if self._stt_http is None or self._stt_http.is_closed:
            import ssl
            import certifi
            import httpx
            if self._stt_ssl_ctx is None:
                Listener._stt_ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            Listener._stt_http = httpx.Client(timeout=10, verify=self._stt_ssl_ctx)
        return self._stt_http

    def _transcribe_elevenlabs(self, audio_float: np.ndarray) -> str | None:
        """Send audio to ElevenLabs Scribe API. Returns transcript or None."""
        import io
        import struct

        pcm16 = (audio_float * 32767).astype(np.int16)
        buf = io.BytesIO()
        n = len(pcm16)
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + n * 2))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<IHHIIHH", 16, 1, 1, TARGET_RATE, TARGET_RATE * 2, 2, 16))
        buf.write(b"data")
        buf.write(struct.pack("<I", n * 2))
        buf.write(pcm16.tobytes())
        buf.seek(0)

        try:
            client = self._get_stt_http()
            resp = client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": self.settings.elevenlabs_api_key},
                files={"file": ("audio.wav", buf, "audio/wav")},
                data={
                    "model_id": "scribe_v2",
                    "language_code": "eng",
                    "tag_audio_events": "false",
                },
            )
            if resp.status_code == 200:
                text = resp.json().get("text", "").strip()
                return text if text else None
            logger.warning("ElevenLabs STT failed (status %d): %s", resp.status_code, resp.text[:200])
            return None
        except Exception as exc:
            logger.warning("ElevenLabs STT error: %s", exc)
            return None

    def _ensure_local_whisper(self):
        """Lazily load a real faster-whisper model for local fallback."""
        if hasattr(self, '_local_whisper') and self._local_whisper is not None:
            return self._local_whisper
        from faster_whisper import WhisperModel
        self._local_whisper = WhisperModel(
            self.settings.whisper_model,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Whisper model loaded (local fallback): %s", self.settings.whisper_model)
        return self._local_whisper

    def _transcribe_local(self, audio_float: np.ndarray) -> str | None:
        """Transcribe with local faster-whisper."""
        model = self._ensure_local_whisper()
        segments, _ = model.transcribe(
            audio_float,
            language="en",
            beam_size=1,
            vad_filter=True,
            initial_prompt=self._TRANSCRIPTION_PROMPT,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text if text else None

    def _transition_to_transcribing(self) -> None:
        self.state = State.TRANSCRIBING
        logger.info("[DEBUG] === STATE -> TRANSCRIBING === (chunks_sent=%d, backlogged=%d, ws_ready=%s, current_transcript='%s', el_final='%s')",
                    getattr(self, '_el_chunks_sent', 0), getattr(self, '_el_chunks_backlogged', 0),
                    self._el_session_ready, self._current_transcript, self._el_final_text)

        if not self._capture_frames:
            logger.info("[DEBUG] No audio captured, returning to idle")
            self._close_elevenlabs_ws()
            self._unduck_spotify()
            self.state = State.IDLE
            return

        audio = np.concatenate(self._capture_frames).astype(np.float32) / 32768.0
        self._capture_frames = []
        logger.info("[DEBUG] Audio: %.2fs, using %s path",
                    len(audio) / TARGET_RATE,
                    "elevenlabs-sdk" if (self._whisper_model == "elevenlabs" and getattr(self, '_el_connection', None)) else "fallback")

        text = None

        if self._whisper_model == "elevenlabs" and getattr(self, '_el_connection', None):
            conn = self._el_connection
            el_loop = getattr(self, '_el_loop', None)

            # Fast path: if we already have committed text, skip the commit round-trip.
            # If EL has nothing at all (no final, no partial), skip commit too --
            # the audio was too short for the WS to process, REST will be faster.
            if not self._el_final_text and self._current_transcript:
                try:
                    if el_loop and el_loop.is_running():
                        logger.info("[DEBUG] Sending commit via SDK")
                        fut = asyncio.run_coroutine_threadsafe(conn.commit(), el_loop)
                        fut.result(timeout=2)
                    for i in range(20):
                        if self._el_final_text:
                            logger.info("[DEBUG] Got committed text after %dms", i * 50)
                            break
                        time.sleep(0.05)
                except Exception as exc:
                    logger.warning("[DEBUG] SDK commit failed: %s", exc)

            text = self._el_final_text or self._current_transcript
            logger.info("[DEBUG] SDK text='%s' (final='%s', partial='%s')", text, self._el_final_text, self._current_transcript)
            self._close_elevenlabs_ws()

            if not text:
                logger.info("[DEBUG] No SDK text, falling back to batch REST API")
                text = self._transcribe_elevenlabs(audio)
        else:
            if self._whisper_model == "elevenlabs":
                logger.info("[DEBUG] ElevenLabs SDK not connected, trying REST API")
                text = self._transcribe_elevenlabs(audio)
            if not text:
                logger.info("[DEBUG] Using local whisper fallback")
                text = self._transcribe_local(audio)

        if not text or self._is_hallucination(text):
            logger.info("[DEBUG] Empty or hallucinated transcript ('%s'), returning to idle", text)
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
        logger.info("[DEBUG] === STATE -> PROCESSING === transcript='%s'", text)
        if self._loop is None:
            logger.error("No event loop available")
            self._unduck_spotify()
            self.state = State.IDLE
            return

        logger.info("[DEBUG] Submitting to event loop...")
        future = asyncio.run_coroutine_threadsafe(self._async_handle(text), self._loop)
        try:
            response, did_stream = future.result(timeout=30)
            logger.info("[DEBUG] Event loop returned response, action=%s, streamed=%s",
                        response.action_code if response else "None", did_stream)
        except Exception:
            logger.exception("Intent processing failed")
            self._unduck_spotify()
            self.state = State.IDLE
            return

        from app.core.personality import QUICK_REPLY_ACTIONS
        silent = response.action_code in QUICK_REPLY_ACTIONS
        if response.audio_url and not silent and not did_stream:
            self.state = State.SPEAKING
            logger.info("[DEBUG] === STATE -> SPEAKING === response='%s'", self._current_response[:80])
            self._play_response_audio(response.audio_url)
        elif did_stream:
            logger.info("[DEBUG] Audio was streamed during processing, skipping file playback")

        logger.info("[DEBUG] === STATE -> IDLE ===")
        if response.action_code == "MUSIC_VOLUME":
            logger.info("Skipping unduck — user changed volume intentionally")
        else:
            self._unduck_spotify()
        self._current_transcript = ""
        self._current_response = ""
        self.state = State.IDLE

    def _play_response_audio(self, audio_url: str) -> None:
        """Play TTS audio in the listener thread (not on the event loop)."""
        from app.core.audio_player import play_audio_file
        from pathlib import Path
        static_root = Path(__file__).resolve().parent.parent / "static"
        if audio_url.startswith("/static/"):
            local_path = static_root / audio_url[len("/static/"):]
        else:
            local_path = static_root / "audio" / Path(audio_url).name
        if local_path.exists():
            play_audio_file(local_path, blocking=True)
        else:
            logger.warning("Audio file not found: %s", local_path)

    async def _async_handle(self, text: str):
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
            loop = asyncio.get_event_loop()
            response, did_stream = await loop.run_in_executor(
                None, self._run_intent_sync, request,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "Response: action=%s speak=%s streamed=%s",
                response.action_code,
                response.speak_text[:80] if response.speak_text else "",
                did_stream,
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
            return response, did_stream
        except Exception as exc:
            self.history.appendleft(InteractionRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                transcript=text,
                error=str(exc)[:200],
            ))
            raise

    def _run_intent_sync(self, request):
        """Run hub.handle_intent_streaming in a fresh event loop inside a thread pool thread."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.hub.handle_intent_streaming(request))
        finally:
            loop.close()
