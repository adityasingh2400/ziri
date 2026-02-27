from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_status_endpoint_returns_health() -> None:
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data


def test_intent_endpoint_accepts_payload() -> None:
    payload = {
        "user_id": "Aditya",
        "device_id": "iPhone_Kitchen",
        "room": "Kitchen",
        "raw_text": "Play Uzi",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    response = client.post("/intent", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "action_code" in data
    assert "speak_text" in data
    assert "private_note" in data
