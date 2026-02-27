from __future__ import annotations

import logging
from datetime import datetime, timezone

from app import __version__
from app.settings import Settings
from app.core.brain import Brain
from app.core.device_registry import DeviceContext, DeviceRegistry
from app.core.memory import InMemoryStore, MemoryStore, SupabaseMemoryStore
from app.core.orchestrator import AuraOrchestrator
from app.core.tool_runner import ToolRunner
from app.data.preferences_repository import (
    InMemoryPreferencesRepository,
    PreferencesRepository,
    SupabasePreferencesRepository,
)
from app.data.session_repository import (
    InMemorySessionRepository,
    SessionRepository,
    SupabaseSessionRepository,
)
from app.integrations.calendar_controller import CalendarController
from app.integrations.home_scene_controller import HomeSceneController
from app.integrations.nba import NBAController
from app.integrations.news import NewsController
from app.integrations.phone_bridge import PhoneBridge
from app.integrations.reminders_bridge import RemindersBridge
from app.integrations.spotify_controller import SpotifyController
from app.integrations.tts import PollyTTS
from app.integrations.weather import WeatherController
from app.schemas import (
    IntentRequest,
    IntentResponse,
    IntentType,
    RouterDecision,
    StatusResponse,
    ToolResult,
)

logger = logging.getLogger(__name__)


class AuraHub:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device_registry = DeviceRegistry(settings.device_map_path)
        self.session_repository = self._build_session_repository()
        self.memory_store = self._build_memory_store()
        self.preferences_repository = self._build_preferences_repository()

        self.brain = Brain(settings=settings, memory=self.memory_store)
        spotify = SpotifyController(settings=settings)
        calendar = CalendarController(settings=settings)
        reminders = RemindersBridge()
        home_scene = HomeSceneController(settings.scene_map_path)
        phone_bridge = PhoneBridge()
        weather = WeatherController(settings=settings)
        nba = NBAController()
        news = NewsController(settings=settings)
        tool_runner = ToolRunner(
            spotify=spotify,
            calendar=calendar,
            reminders=reminders,
            home_scene=home_scene,
            phone_bridge=phone_bridge,
            weather=weather,
            nba=nba,
            news=news,
        )
        tts = PollyTTS(settings=settings)

        self.orchestrator = AuraOrchestrator(
            settings=settings,
            brain=self.brain,
            tool_runner=tool_runner,
            memory=self.memory_store,
            tts=tts,
        )

    def _build_session_repository(self) -> SessionRepository:
        if self.settings.supabase_url and self.settings.supabase_service_role_key:
            try:
                return SupabaseSessionRepository(
                    url=self.settings.supabase_url,
                    service_role_key=self.settings.supabase_service_role_key,
                )
            except Exception as exc:
                logger.warning("Using in-memory sessions repository: %s", exc)
        return InMemorySessionRepository()

    def _build_memory_store(self) -> MemoryStore:
        if self.settings.supabase_url and self.settings.supabase_service_role_key:
            try:
                return SupabaseMemoryStore(
                    url=self.settings.supabase_url,
                    service_role_key=self.settings.supabase_service_role_key,
                )
            except Exception as exc:
                logger.warning("Using in-memory memory store: %s", exc)
        return InMemoryStore()

    def _build_preferences_repository(self) -> PreferencesRepository:
        if self.settings.supabase_url and self.settings.supabase_service_role_key:
            try:
                return SupabasePreferencesRepository(
                    url=self.settings.supabase_url,
                    service_role_key=self.settings.supabase_service_role_key,
                )
            except Exception as exc:
                logger.warning("Using in-memory preferences repository: %s", exc)
        return InMemoryPreferencesRepository()

    def _apply_user_preferences(self, request: IntentRequest, context: DeviceContext) -> DeviceContext:
        prefs = self.preferences_repository.get_user_preferences(request.user_id)
        preferred_speaker = prefs.get("default_speaker")
        if not preferred_speaker:
            return context

        speaker_meta = self.device_registry.resolve_speaker(str(preferred_speaker))
        context.default_speaker = str(preferred_speaker)
        context.spotify_device_id = speaker_meta.get("spotify_device_id") or context.spotify_device_id
        return context

    async def handle_intent(self, request: IntentRequest) -> IntentResponse:
        session_id = self.session_repository.log_request(request)
        fallback_route = RouterDecision(
            intent_type=IntentType.INFO_QUERY,
            tool_name="general.answer",
            action_code="ERROR",
            speak_text="",
            private_note="",
            confidence=0.0,
        )
        fallback_result = ToolResult(ok=False, action_code="ERROR")

        try:
            device_context = self.device_registry.resolve_context(request.device_id, request.room)
            device_context = self._apply_user_preferences(request, device_context)
            response, route_decision, tool_result = self.orchestrator.handle_intent(
                request=request,
                device_context=device_context,
            )
            self.session_repository.finalize(session_id, response, route_decision, tool_result)
            return response
        except Exception as exc:
            logger.exception("Intent processing failed")
            response = IntentResponse(
                speak_text="I hit an internal error while processing that request.",
                private_note=str(exc),
                action_code="ERROR",
                metadata={"exception": exc.__class__.__name__},
            )
            self.session_repository.finalize(session_id, response, fallback_route, fallback_result)
            return response

    def status(self) -> StatusResponse:
        using_supabase = isinstance(self.session_repository, SupabaseSessionRepository)
        using_bedrock = self.brain._bedrock is not None

        degraded = not using_bedrock
        components = {
            "router": "bedrock" if using_bedrock else "heuristic_fallback",
            "sessions": "supabase" if using_supabase else "in_memory",
            "memory": self.memory_store.__class__.__name__,
            "preferences": self.preferences_repository.__class__.__name__,
            "tts": "polly" if self.settings.enable_polly else "disabled",
        }

        return StatusResponse(
            status="ok" if not degraded else "degraded",
            service=self.settings.app_name,
            timestamp=datetime.now(timezone.utc),
            model=self.settings.bedrock_model_id,
            version=__version__,
            degraded=degraded,
            components=components,
        )
