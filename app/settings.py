from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Ziri"
    app_env: str = "dev"
    ziri_port: int = 8000
    log_level: str = "INFO"
    cors_allow_origins: str = "*"

    aws_region: str = "us-east-1"
    aws_access_key: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    polly_voice_id: str = "Joanna"
    polly_engine: str = "neural"
    s3_tts_bucket: Optional[str] = None
    s3_tts_public_base_url: Optional[str] = None

    supabase_url: Optional[str] = None
    supabase_service_role_key: Optional[str] = None

    spotify_client_id: Optional[str] = None
    spotify_client_secret: Optional[str] = None
    spotify_redirect_uri: str = "http://127.0.0.1:8000/spotify/callback"
    spotify_refresh_token: Optional[str] = None
    spotify_user_access_token: Optional[str] = None
    spotify_default_device_id: Optional[str] = None

    google_service_account_file: Optional[str] = None
    google_calendar_id: str = "primary"

    device_map_path: str = "app/config/device_map.yaml"
    scene_map_path: str = "app/config/scenes.yaml"
    return_audio_url: bool = True
    enable_polly: bool = True
    memory_window: int = 8

    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: str = "yj30vwTGJxSHezdAGsv9"
    elevenlabs_model_id: str = "eleven_turbo_v2_5"
    elevenlabs_output_format: str = "mp3_44100_192"
    elevenlabs_streaming_latency: int = 4
    elevenlabs_stability: float = 0.55
    elevenlabs_similarity_boost: float = 0.85
    elevenlabs_speed: float = 0.85

    user_latitude: float = 40.7128
    user_longitude: float = -74.0060
    news_api_key: Optional[str] = None

    # Langfuse observability
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Semantic memory (pgvector)
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    semantic_memory_top_k: int = 3
    semantic_memory_enabled: bool = True

    # Elasticsearch (hybrid search)
    elasticsearch_url: Optional[str] = None
    elasticsearch_index: str = "ziri_conversation_turns"

    # Always-on listener
    wake_word_model: str = "hey_jarvis"
    wake_word_threshold: float = 0.5
    whisper_model: str = "base.en"
    listener_device_id: str = "Mac_Listener"
    listener_room: str = "Office"
    listener_user_id: str = "Aditya"
    listener_chime_enabled: bool = True
    
    # Speaker filtering: only transcribe the person who said the wake word
    speaker_filter_enabled: bool = True
    speaker_filter_pre_wakeword_secs: float = 1.5  # seconds of audio before wake word to capture

    # Siri Shortcuts
    siri_user_id: str = "Aditya"
    siri_device_id: str = "iPhone_Siri"
    siri_room: str = "Kitchen"


@lru_cache
def get_settings() -> Settings:
    return Settings()
