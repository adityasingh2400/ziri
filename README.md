# Ziri

**Ziri** is a distributed home voice OS — an always-on, AI-powered voice assistant that runs locally on your Mac, controls your smart home, manages your music, and answers your questions through natural conversation.

Built with a focus on low latency, beautiful audio, and seamless ambient integration.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Input Sources                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ Always-On   │  │ Siri         │  │ Browser  │  │ REST API     │ │
│  │ Mic Listener│  │ Shortcuts    │  │ /listen  │  │ POST /intent │ │
│  └──────┬──────┘  └──────┬───────┘  └────┬─────┘  └──────┬───────┘ │
│         │                │               │                │         │
│         ▼                ▼               ▼                ▼         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  AuraHub.handle_intent()                                    │    │
│  │  ┌───────────┐  ┌───────────────┐  ┌──────────────────┐    │    │
│  │  │   Brain   │→ │  Tool Runner  │→ │   Personality    │    │    │
│  │  │ (Bedrock  │  │ (Spotify, Cal,│  │  (Quick Replies, │    │    │
│  │  │  Claude)  │  │  Weather, NBA,│  │   Rewriter)      │    │    │
│  │  │           │  │  News, Scenes)│  │                  │    │    │
│  │  └───────────┘  └───────────────┘  └──────────────────┘    │    │
│  └─────────────────────────┬───────────────────────────────────┘    │
│                            │                                        │
│                            ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  ElevenLabs TTS (Streaming, 192kbps, Custom Voice)          │    │
│  └─────────────────────────┬───────────────────────────────────┘    │
│                            │                                        │
│         ┌──────────────────┼──────────────────┐                     │
│         ▼                  ▼                  ▼                     │
│  Local Speaker       Static File         JSON Response              │
│  (sounddevice)       (/static/audio/)    (Siri / API)              │
└──────────────────────────────────────────────────────────────────────┘
```

## Features

### Voice Pipeline
- **Always-on wake word detection** — "Hey Jarvis" via [openwakeword](https://github.com/dscripka/openWakeWord) (ONNX)
- **Local speech-to-text** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) running on-device, no cloud STT
- **Live partial transcription** — words appear on the dashboard as you speak, not after
- **ElevenLabs streaming TTS** — 192kbps MP3 via the `/stream` endpoint with `optimize_streaming_latency=3`
- **Custom 11Labs sound effects** — wake word blip and thinking pulse generated with ElevenLabs SFX
- **Spotify volume ducking** — music drops to 20% on wake, gradually restores after response
- **Silent quick commands** — "skip", "pause", "resume" execute instantly with no voice response

### Intelligence
- **Claude on Bedrock** — intent routing via Anthropic Claude with tool-use API
- **Deterministic fast path** — common music/calendar/scene commands bypass the LLM entirely
- **LangGraph orchestration** — structured `route → execute → respond` pipeline
- **Conversational memory** — resolves "play it again", "turn it up" from context
- **Hallucination filtering** — whisper output is screened for common noise artifacts

### Integrations
- **Spotify** — search, play, pause, skip, queue, volume, shuffle, repeat, like, device control
- **Google Calendar** — today's events, upcoming schedule
- **iCloud Reminders** — create reminders via macOS bridge
- **Home Scenes** — trigger smart home scenes (lights, movie mode, goodnight)
- **Weather** — current conditions and forecasts
- **NBA Scores** — live game scores and standings
- **News** — top headlines via NewsAPI
- **Phone Bridge** — private data (texts, OTPs) displayed only on phone

### Frontend
- **Fluid WebGL dashboard** (`/listen`) — album art with reactive fluid dynamics driven by mic audio
- **Real-time Spotify now-playing** — album art crossfade, progress bar, playback controls
- **Live voice overlay** — typewriter transcription + Ziri's response with smooth animations
- **Conversation history modal** — frosted glass slide-up panel with full interaction log
- **Color extraction** — UI theme adapts to album art palette in real-time

### Infrastructure
- **FastAPI** backend with async request handling
- **Pydantic v2** strict validation on all schemas
- **Multi-input** — always-on mic, Siri Shortcuts, browser, REST API all share one pipeline
- **Session logging** — Supabase persistence with in-memory fallback
- **User preferences** — per-user default speaker, room preferences
- **Device registry** — YAML-based device → room → speaker mapping
- **Pre-cached TTS** — common phrases pre-generated at startup for instant playback

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Claude (Anthropic) via AWS Bedrock |
| Orchestration | LangGraph |
| TTS | ElevenLabs (streaming, 192kbps) with AWS Polly fallback |
| STT | faster-whisper (local, CPU, int8) |
| Wake Word | openwakeword (ONNX) |
| Backend | FastAPI + Uvicorn |
| Music | Spotify Web API (spotipy) |
| Calendar | Google Calendar API |
| Database | Supabase (PostgreSQL) |
| Frontend | Vanilla JS + WebGL Fluid Simulation |
| Audio | sounddevice + soundfile |
| Sound FX | ElevenLabs Sound Effects API |

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

**Server + always-on listener:**
```bash
python3 run_listener.py
```

**Server only (no wake word):**
```bash
python3 run_listener.py --no-listener
```

Then open `http://localhost:8000/listen` for the dashboard.

## Environment Variables

See `.env.example` for all options. Key configuration:

| Variable | Description |
|----------|-------------|
| `ELEVENLABS_API_KEY` | ElevenLabs API key (primary TTS) |
| `ELEVENLABS_VOICE_ID` | Voice to use for TTS |
| `ELEVENLABS_MODEL_ID` | Model (`eleven_multilingual_v2` or `eleven_turbo_v2_5`) |
| `ELEVENLABS_SPEED` | Speech rate (1.0 = normal) |
| `BEDROCK_MODEL_ID` | Claude model ARN for intent routing |
| `AWS_ACCESS_KEY_ID` | AWS credentials for Bedrock |
| `SPOTIFY_CLIENT_ID` | Spotify app credentials |
| `SPOTIFY_REFRESH_TOKEN` | Long-lived Spotify auth |

## API

### `POST /intent`
Full intent processing pipeline. Accepts any voice command as text.

### `POST /siri?text=...`
Simplified endpoint for iOS Siri Shortcuts integration.

### `GET /listen`
WebGL fluid dashboard with Spotify now-playing and voice interaction overlay.

### `GET /dashboard/api`
JSON API returning listener state, transcript, response, and interaction history.

### `GET /status`
Runtime health check with component status.

## Project Structure

```
app/
├── main.py                    # FastAPI app, routes, middleware
├── hub.py                     # Dependency wiring, request lifecycle
├── settings.py                # Pydantic settings (env-driven)
├── schemas.py                 # Request/response models
├── core/
│   ├── brain.py               # Bedrock Claude router + deterministic fast path
│   ├── orchestrator.py        # LangGraph pipeline (route → execute → respond)
│   ├── tool_runner.py         # Tool dispatcher
│   ├── listener.py            # Always-on wake word + STT + playback
│   ├── audio_player.py        # sounddevice playback + 11Labs sound effects
│   ├── personality.py         # Quick replies, response rewriting
│   ├── memory.py              # Conversational memory (in-memory + Supabase)
│   └── device_registry.py     # Device → room → speaker resolution
├── integrations/
│   ├── tts.py                 # ElevenLabs streaming TTS + Polly fallback
│   ├── spotify_controller.py  # Spotify Web API (search, playback, volume)
│   ├── calendar_controller.py # Google Calendar
│   ├── weather.py             # Weather forecasts
│   ├── nba.py                 # NBA scores
│   ├── news.py                # News headlines
│   ├── reminders_bridge.py    # iCloud Reminders
│   ├── home_scene_controller.py # Smart home scenes
│   └── phone_bridge.py        # Private phone data
├── static/
│   ├── listen.html            # WebGL fluid dashboard
│   └── audio/                 # TTS output + sound effects
├── config/
│   ├── device_map.yaml        # Device/room/speaker mapping
│   └── scenes.yaml            # Home scene definitions
└── data/
    ├── session_repository.py  # Session persistence
    └── preferences_repository.py # User preferences
```

## Roadmap

- [ ] ElevenLabs Conversational AI (real-time voice-to-voice)
- [ ] Multi-voice contexts (different voices per room/mood)
- [ ] Voice cloning for unique Ziri identity
- [ ] Multi-room audio with distributed speakers
- [ ] Proactive notifications (calendar reminders, weather alerts)
- [ ] HomeKit / Matter smart home integration
- [ ] Mobile companion app
- [ ] Wake word customization ("Hey Ziri")

## License

Private. All rights reserved.
