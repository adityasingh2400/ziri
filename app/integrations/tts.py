from __future__ import annotations

import hashlib
import io
import logging
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import sounddevice as sd
import soundfile as sf

from app.settings import Settings
from app.core.tracing import trace_tts_span
from app.core.metrics import TTS_TTFB

logger = logging.getLogger(__name__)

_AUDIO_DIR = Path(__file__).resolve().parent.parent / "static" / "audio"
_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR = _AUDIO_DIR / "cached"
_CACHE_DIR.mkdir(exist_ok=True)

_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+|(?<=\.\.\.)\s*')
_MIN_CHUNK_LEN = 20


class TTS:
    """Text-to-speech with pre-cached quick replies, ElevenLabs primary, Polly fallback.

    Supports two modes:
      1. synthesize()          — full text → file → URL (existing behavior)
      2. synthesize_streaming() — iterator of text chunks → play audio as it arrives
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._eleven_ok = bool(settings.elevenlabs_api_key)
        self._polly = None
        self._cache: dict[str, str] = {}
        self._http: "httpx.Client | None" = None

        if not self._eleven_ok and settings.enable_polly:
            try:
                import boto3
                kwargs = {"region_name": settings.aws_region}
                ak = settings.aws_access_key_id or settings.aws_access_key
                if ak and settings.aws_secret_access_key:
                    kwargs["aws_access_key_id"] = ak
                    kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
                self._polly = boto3.client("polly", **kwargs)
            except Exception as exc:
                logger.warning("Polly init failed: %s", exc)

        self._load_existing_cache()

    def _get_http(self) -> "httpx.Client":
        """Return a long-lived httpx client for ElevenLabs TTS requests."""
        if self._http is None or self._http.is_closed:
            import httpx
            self._http = httpx.Client(timeout=20)
        return self._http

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def _load_existing_cache(self) -> None:
        for f in _CACHE_DIR.glob("*.mp3"):
            key = f.stem
            self._cache[key] = f"/static/audio/cached/{f.name}"

    def precache_phrases(self, phrases: list[str]) -> int:
        """Pre-generate TTS for a list of phrases. Returns count of newly generated files."""
        if not self._eleven_ok:
            return 0

        count = 0
        for phrase in phrases:
            key = self._cache_key(phrase)
            if key in self._cache:
                continue
            cached_path = _CACHE_DIR / f"{key}.mp3"
            if cached_path.exists():
                self._cache[key] = f"/static/audio/cached/{key}.mp3"
                continue
            audio = self._elevenlabs_generate(phrase)
            if audio:
                cached_path.write_bytes(audio)
                self._cache[key] = f"/static/audio/cached/{key}.mp3"
                count += 1
                logger.info("Pre-cached TTS: %s", phrase)
        return count

    def synthesize(self, text: str, trace: Any = None) -> str | None:
        if not text:
            return None

        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        if self._eleven_ok:
            audio = self._elevenlabs_generate(text, trace=trace)
            if audio:
                filename = f"{uuid4().hex}.mp3"
                path = _AUDIO_DIR / filename
                path.write_bytes(audio)
                self._cleanup_old_files()
                return f"/static/audio/{filename}"

        if self._polly:
            return self._polly_synthesize(text)

        return None

    def _elevenlabs_generate(self, text: str, trace: Any = None) -> bytes | None:
        """Stream audio from ElevenLabs, collecting chunks as they arrive."""
        try:
            s = self.settings
            url = (
                f"https://api.elevenlabs.io/v1/text-to-speech/{s.elevenlabs_voice_id}/stream"
                f"?output_format={s.elevenlabs_output_format}"
                f"&optimize_streaming_latency={s.elevenlabs_streaming_latency}"
            )
            headers = {
                "xi-api-key": s.elevenlabs_api_key,
                "Content-Type": "application/json",
            }
            body = {
                "text": text,
                "model_id": s.elevenlabs_model_id,
                "voice_settings": {
                    "stability": s.elevenlabs_stability,
                    "similarity_boost": s.elevenlabs_similarity_boost,
                    "speed": s.elevenlabs_speed,
                },
            }

            with trace_tts_span(
                trace=trace,
                text=text,
                voice_id=s.elevenlabs_voice_id,
                model_id=s.elevenlabs_model_id,
            ) as timing:
                buf = bytearray()
                first_chunk = True
                stream_start = time.perf_counter()
                client = self._get_http()
                with client.stream("POST", url, headers=headers, json=body) as resp:
                    if resp.status_code != 200:
                        resp.read()
                        logger.warning(
                            "ElevenLabs TTS failed (status %s): %s",
                            resp.status_code,
                            resp.text[:200],
                        )
                        return None
                    for chunk in resp.iter_bytes(chunk_size=4096):
                        if first_chunk:
                            ttfb_ms = (time.perf_counter() - stream_start) * 1000
                            timing["ttfb_ms"] = ttfb_ms
                            TTS_TTFB.observe(ttfb_ms / 1000)
                            first_chunk = False
                        buf.extend(chunk)

                if not buf:
                    logger.warning("ElevenLabs TTS returned empty audio")
                    return None
                return bytes(buf)
        except Exception as exc:
            logger.warning("ElevenLabs TTS error: %s", exc)
            return None

    def _polly_synthesize(self, text: str) -> str | None:
        try:
            speech = self._polly.synthesize_speech(
                Text=text, OutputFormat="mp3",
                VoiceId=self.settings.polly_voice_id, Engine=self.settings.polly_engine,
            )
            stream = speech.get("AudioStream")
            if not stream:
                return None
            filename = f"{uuid4().hex}.mp3"
            path = _AUDIO_DIR / filename
            path.write_bytes(stream.read())
            self._cleanup_old_files()
            return f"/static/audio/{filename}"
        except Exception as exc:
            logger.warning("Polly TTS error: %s", exc)
            return None

    @staticmethod
    def _cleanup_old_files(max_files: int = 50) -> None:
        files = sorted(_AUDIO_DIR.glob("*.mp3"), key=lambda f: f.stat().st_mtime)
        while len(files) > max_files:
            files.pop(0).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Streaming TTS: text iterator → real-time audio playback
    # ------------------------------------------------------------------

    def synthesize_streaming(
        self,
        text_iter: Iterator[str],
        trace: Any = None,
        on_sentence: Any = None,
    ) -> str:
        """Stream LLM text tokens through ElevenLabs TTS and play audio in real-time.

        Args:
            text_iter: yields text fragments (tokens) from the LLM.
            trace: optional Langfuse trace for observability.
            on_sentence: optional callback(sentence: str) fired as each sentence is flushed.

        Returns:
            The full assembled text that was spoken.
        """
        if not self._eleven_ok:
            full_text = "".join(text_iter)
            if on_sentence:
                on_sentence(full_text)
            self.synthesize(full_text, trace=trace)
            return full_text

        return self._elevenlabs_streaming_pipeline(text_iter, trace=trace, on_sentence=on_sentence)

    def _elevenlabs_streaming_pipeline(
        self,
        text_iter: Iterator[str],
        trace: Any = None,
        on_sentence: Any = None,
    ) -> str:
        """Send text chunks to ElevenLabs streaming TTS; decode + play MP3 in real-time."""
        import httpx

        s = self.settings
        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{s.elevenlabs_voice_id}/stream"
            f"?output_format=pcm_24000"
            f"&optimize_streaming_latency={s.elevenlabs_streaming_latency}"
        )
        headers = {
            "xi-api-key": s.elevenlabs_api_key,
            "Content-Type": "application/json",
        }

        full_text_parts: list[str] = []
        sentence_buffer = ""
        sentences: list[str] = []

        for token in text_iter:
            full_text_parts.append(token)
            sentence_buffer += token
            parts = _SENTENCE_BOUNDARY.split(sentence_buffer)
            if len(parts) > 1:
                for complete_sentence in parts[:-1]:
                    stripped = complete_sentence.strip()
                    if stripped and len(stripped) >= _MIN_CHUNK_LEN:
                        sentences.append(stripped)
                        if on_sentence:
                            on_sentence(stripped)
                sentence_buffer = parts[-1]

        if sentence_buffer.strip():
            sentences.append(sentence_buffer.strip())
            if on_sentence:
                on_sentence(sentence_buffer.strip())

        full_text = "".join(full_text_parts)

        if not sentences:
            return full_text

        stream_start = time.perf_counter()
        first_audio = True
        output_stream = None
        pcm_rate = 24000

        try:
            for sentence in sentences:
                body = {
                    "text": sentence,
                    "model_id": s.elevenlabs_model_id,
                    "voice_settings": {
                        "stability": s.elevenlabs_stability,
                        "similarity_boost": s.elevenlabs_similarity_boost,
                        "speed": s.elevenlabs_speed,
                    },
                }

                client = self._get_http()
                with client.stream("POST", url, headers=headers, json=body, timeout=20) as resp:
                    if resp.status_code != 200:
                        resp.read()
                        logger.warning(
                            "ElevenLabs streaming TTS failed (status %s): %s",
                            resp.status_code, resp.text[:200],
                        )
                        continue

                    for chunk in resp.iter_bytes(chunk_size=4096):
                        if first_audio:
                            ttfb_ms = (time.perf_counter() - stream_start) * 1000
                            TTS_TTFB.observe(ttfb_ms / 1000)
                            logger.info("Streaming TTS TTFB: %.0fms", ttfb_ms)
                            first_audio = False

                        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                        if output_stream is None:
                            output_stream = sd.OutputStream(
                                samplerate=pcm_rate,
                                channels=1,
                                dtype="float32",
                            )
                            output_stream.start()
                        output_stream.write(samples.reshape(-1, 1))

        except Exception as exc:
            logger.warning("Streaming TTS playback error: %s", exc)
        finally:
            if output_stream is not None:
                try:
                    sd.sleep(200)
                    output_stream.stop()
                    output_stream.close()
                except Exception:
                    pass

            total_ms = (time.perf_counter() - stream_start) * 1000
            logger.info("Streaming TTS complete: %.0fms, %d sentences, %d chars",
                        total_ms, len(sentences), len(full_text))

            if trace is not None:
                try:
                    trace.span(
                        name="tts_streaming",
                        input={"text": full_text, "text_length": len(full_text), "sentences": len(sentences)},
                        output={"voice_id": s.elevenlabs_voice_id, "model_id": s.elevenlabs_model_id},
                        metadata={"total_ms": round(total_ms, 1), "streaming": True},
                    )
                except Exception:
                    pass

        return full_text
