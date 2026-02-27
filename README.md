# Aura Hub

Starter repository for **Aura**, a distributed home voice OS:

- Front Door: iOS Shortcut (dictation + device metadata)
- Brain: FastAPI + Bedrock Claude intent router + LangGraph orchestration
- Mouth: House speaker control + optional AWS Polly audio URL

## Spec Coverage

- `POST /intent` endpoint with strict Pydantic validation
- `GET /status` endpoint for runtime health/component mode
- Session logging to `sessions` table (Supabase) with in-memory fallback
- LLM router via Bedrock (`Brain` class using `boto3`)
- Stateful follow-up context (`play it again`, `play it louder`) via memory store
- Tooling: Spotify, Calendar, iCloud reminders bridge, home scene bridge
- Privacy routing (`Read my texts` => `speak_text=""`, data in `private_note`)
- Device/room/speaker mapping via YAML config
- Polly TTS generation with S3 URL return option
- `.env.example` includes placeholders for AWS, Spotify, Supabase
- iOS Shortcut build flow docs

## Layout

- `app/main.py`: FastAPI entrypoint and middleware
- `app/hub.py`: dependency wiring and request lifecycle
- `app/core/brain.py`: Bedrock router and heuristic fallback
- `app/core/orchestrator.py`: LangGraph graph (`route -> execute -> respond`)
- `app/core/tool_runner.py`: tool dispatcher
- `app/core/memory.py`: in-memory + Supabase memory stores
- `app/core/device_registry.py`: `device_id -> room/speaker` resolution
- `app/data/session_repository.py`: session persistence
- `app/data/preferences_repository.py`: user preference lookup
- `app/integrations/spotify_controller.py`: search/playback/volume
- `app/integrations/calendar_controller.py`: Google Calendar read
- `app/integrations/reminders_bridge.py`: iCloud reminder bridge payload
- `app/integrations/home_scene_controller.py`: scene execution payloads
- `app/integrations/phone_bridge.py`: private phone-only data responses
- `app/integrations/tts.py`: Polly + S3 audio URL generation
- `app/config/device_map.yaml`: source device mapping
- `app/config/scenes.yaml`: home scene definitions
- `sql/001_init.sql`: Supabase schema (sessions, turns, preferences)
- `docs/ios_shortcut_flow.md`: Shortcut implementation

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker compose up --build
```

## Environment Variables

See `.env.example`. Important keys:

- `BEDROCK_MODEL_ID`
- `AWS_ACCESS_KEY` or `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`
- `SPOTIFY_USER_ACCESS_TOKEN` or `SPOTIFY_REFRESH_TOKEN`
- `DEVICE_MAP_PATH`
- `SCENE_MAP_PATH`
- `ENABLE_POLLY`
- `S3_TTS_BUCKET`

## API Contract

### `POST /intent`

Request:

```json
{
  "user_id": "Aditya",
  "device_id": "iPhone_Kitchen",
  "room": "Kitchen",
  "raw_text": "Play Uzi",
  "timestamp": "2026-02-26T18:15:00Z"
}
```

Response shape:

```json
{
  "speak_text": "Playing ...",
  "private_note": "...",
  "action_code": "MUSIC_START",
  "audio_url": "https://...",
  "target_device": "Living_Room_Sonos",
  "metadata": {}
}
```

### `GET /status`

Returns component mode (`bedrock` vs fallback, Supabase vs in-memory, etc.).

## Supabase Setup

Run:

```sql
-- sql/001_init.sql
```

Tables created:

- `sessions`
- `conversation_turns` (`context_json`, `embedding vector(1536)`)
- `user_preferences`

## Model Notes

`BEDROCK_MODEL_ID` is configurable. Use whichever Claude model ID you have on Bedrock. The router prompt expects strict JSON output and is model-agnostic.

## Tests

```bash
pytest -q
```
