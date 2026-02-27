from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from uuid import uuid4

from app.settings import Settings

logger = logging.getLogger(__name__)

_AUDIO_DIR = Path(__file__).resolve().parent.parent / "static" / "audio"
_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR = _AUDIO_DIR / "cached"
_CACHE_DIR.mkdir(exist_ok=True)


class TTS:
    """Text-to-speech with pre-cached quick replies, ElevenLabs primary, Polly fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._eleven_ok = bool(settings.elevenlabs_api_key)
        self._polly = None
        self._cache: dict[str, str] = {}

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

    def synthesize(self, text: str) -> str | None:
        if not text:
            return None

        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        if self._eleven_ok:
            audio = self._elevenlabs_generate(text)
            if audio:
                filename = f"{uuid4().hex}.mp3"
                path = _AUDIO_DIR / filename
                path.write_bytes(audio)
                self._cleanup_old_files()
                return f"/static/audio/{filename}"

        if self._polly:
            return self._polly_synthesize(text)

        return None

    def _elevenlabs_generate(self, text: str) -> bytes | None:
        try:
            import httpx
            resp = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self.settings.elevenlabs_voice_id}?output_format=mp3_44100_128",
                headers={
                    "xi-api-key": self.settings.elevenlabs_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": self.settings.elevenlabs_model_id,
                    "voice_settings": {
                        "stability": 0.55,
                        "similarity_boost": 0.85,
                        "speed": 1.2,
                    },
                },
                timeout=12,
            )
            if resp.status_code != 200:
                logger.warning("ElevenLabs TTS failed (status %s): %s", resp.status_code, resp.text[:200])
                return None
            return resp.content
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


PollyTTS = TTS
