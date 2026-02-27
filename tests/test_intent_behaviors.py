from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _payload(raw_text: str) -> dict[str, str]:
    return {
        "user_id": "Aditya",
        "device_id": "iPhone_Kitchen",
        "room": "Kitchen",
        "raw_text": raw_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def test_private_note_is_screen_only() -> None:
    response = client.post("/intent", json=_payload("Read my texts"))
    assert response.status_code == 200
    data = response.json()
    assert data["speak_text"] == ""
    assert data["action_code"] == "PRIVATE_NOTE"
    assert "private" in data["private_note"].lower()


def test_device_mapping_targets_default_house_speaker() -> None:
    response = client.post("/intent", json=_payload("Play Uzi"))
    assert response.status_code == 200
    data = response.json()
    assert data["target_device"] == "Living_Room_Sonos"
