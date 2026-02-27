from datetime import datetime, timezone

from app.core.brain import Brain
from app.core.device_registry import DeviceContext
from app.core.memory import InMemoryStore
from app.schemas import IntentRequest, IntentType
from app.settings import Settings


def _request(text: str) -> IntentRequest:
    return IntentRequest(
        user_id="Aditya",
        device_id="iPhone_Kitchen",
        room="Kitchen",
        raw_text=text,
        timestamp=datetime.now(timezone.utc),
    )


def _device() -> DeviceContext:
    return DeviceContext(
        device_id="iPhone_Kitchen",
        room_name="Kitchen",
        default_speaker="Living_Room_Sonos",
        spotify_device_id=None,
    )


def test_private_queries_route_to_phone_data() -> None:
    memory = InMemoryStore()
    brain = Brain(settings=Settings(), memory=memory)
    brain._bedrock = None

    decision = brain.route_intent(_request("Read my texts"), _device())

    assert decision.intent_type == IntentType.INFO_QUERY
    assert decision.tool_name == "private.phone_data"
    assert decision.speak_text == ""
    assert decision.requires_private_display is True


def test_play_again_uses_last_music_context() -> None:
    memory = InMemoryStore()
    brain = Brain(settings=Settings(), memory=memory)
    brain._bedrock = None

    memory.remember_turn(
        user_id="Aditya",
        raw_text="Play Uzi",
        intent_type="MUSIC_COMMAND",
        tool_name="spotify.play_query",
        assistant_speak="Playing XO TOUR Llif3",
        private_note="",
        context={"tool_payload": {"title": "XO TOUR Llif3"}},
    )

    decision = brain.route_intent(_request("Play it again"), _device())

    assert decision.intent_type == IntentType.MUSIC_COMMAND
    assert decision.tool_name == "spotify.play_query"
    assert decision.tool_args.get("query") == "XO TOUR Llif3"
