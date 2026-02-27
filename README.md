# Ziri

**Ziri** is a distributed home voice OS — an always-on, AI-powered voice assistant that runs locally on your Mac, controls your smart home, manages your music, and answers your questions through natural conversation.

Built with a focus on low latency, beautiful audio, seamless ambient integration, and enterprise-grade AI engineering: multi-agent orchestration, semantic memory, and full LLM observability.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Input Sources                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐         │
│  │ Always-On   │  │ Siri         │  │ Browser  │  │ REST API     │         │
│  │ Mic Listener│  │ Shortcuts    │  │ /listen  │  │ POST /intent │         │
│  └──────┬──────┘  └──────┬───────┘  └────┬─────┘  └──────┬───────┘         │
│         └────────────────┴───────────────┴───────────────┘                   │
│                                  │                                           │
│                                  ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │  AuraHub.handle_intent()                                              │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │  │  LangGraph Orchestrator                                         │  │   │
│  │  │                                                                 │  │   │
│  │  │  ┌──────────────────┐    ┌────────────────────────────────────┐ │  │   │
│  │  │  │   Supervisor     │───▶│  Conditional Domain Router         │ │  │   │
│  │  │  │  (Deterministic  │    │                                    │ │  │   │
│  │  │  │   + Bedrock LLM) │    │  ┌────────────┐ ┌──────────────┐  │ │  │   │
│  │  │  └──────────────────┘    │  │ MusicAgent │ │  InfoAgent   │  │ │  │   │
│  │  │                          │  │ (ReAct x3) │ │  (ReAct x3)  │  │ │  │   │
│  │  │                          │  ├────────────┤ ├──────────────┤  │ │  │   │
│  │  │                          │  │ HomeAgent  │ │ Quick Action │  │ │  │   │
│  │  │                          │  │ (ReAct x2) │ │ (Zero LLM)  │  │ │  │   │
│  │  │                          │  └────────────┘ └──────────────┘  │ │  │   │
│  │  │                          └────────────────────────────────────┘ │  │   │
│  │  │                                          │                      │  │   │
│  │  │                          ┌───────────────▼───────────────────┐  │  │   │
│  │  │                          │  Respond (Personality + TTS +     │  │  │   │
│  │  │                          │          Memory + Langfuse Trace) │  │  │   │
│  │  │                          └───────────────────────────────────┘  │  │   │
│  │  └─────────────────────────────────────────────────────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                  │                                           │
│  ┌───────────────────────────────▼───────────────────────────────────────┐   │
│  │  ElevenLabs TTS (Streaming, 192kbps, TTFB-tracked via Langfuse)       │   │
│  └───────────────────────────────┬───────────────────────────────────────┘   │
│                                  │                                           │
│         ┌────────────────────────┼────────────────────────┐                  │
│         ▼                        ▼                        ▼                  │
│  Local Speaker             Static File              JSON Response            │
│  (sounddevice)             (/static/audio/)         (Siri / API)            │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  Persistence & Observability                                                 │
│  ┌────────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐  │
│  │ Supabase       │  │ pgvector         │  │ Langfuse                    │  │
│  │ (Sessions,     │  │ (Semantic Memory, │  │ (LLM traces, token usage,  │  │
│  │  Turns, Prefs) │  │  1536-dim HNSW)  │  │  TTS TTFB, eval scores)    │  │
│  └────────────────┘  └──────────────────┘  └─────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
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

### Multi-Agent AI System
- **Supervisor-Worker architecture** — a Supervisor agent classifies intent into domains, then delegates to specialized sub-agents
- **ReAct reasoning loops** — each sub-agent runs a Think/Act/Observe cycle with up to 3 iterations, retrying on failures
- **Domain-specialized agents:**
  - **MusicAgent** — handles all 14 Spotify tools (play, pause, skip, volume, queue, shuffle, repeat, like, etc.)
  - **InfoAgent** — handles weather, NBA scores, news, calendar, time/date, and general Q&A via Claude
  - **HomeAgent** — handles home automation scenes, iCloud reminders, and private phone data
- **Zero-latency fast path** — deterministic phrase matching (200+ patterns) routes common commands directly to execution, bypassing the LLM entirely for sub-100ms responses
- **LangGraph orchestration** — `supervisor → conditional_edges → [music|info|home|quick] → respond` with full state passing
- **Graceful degradation** — falls back to legacy linear pipeline if LangGraph is unavailable; in-memory stores if Supabase is unreachable; heuristic routing if Bedrock is down

### Semantic Memory (pgvector)
- **Vector embeddings** — every conversation turn is embedded using Amazon Titan Embeddings v1 (1536-dim) and stored in Supabase PostgreSQL via pgvector
- **HNSW similarity search** — before routing, the user's query is embedded and matched against past conversation turns using cosine similarity
- **Hybrid context injection** — both recency-based context (last N turns) and semantically relevant past context are injected into the LLM prompt
- **Optimized token usage** — instead of stuffing the entire chat history into the context window, only the most relevant past interactions are retrieved

### Observability & Evaluation (Langfuse)
- **End-to-end tracing** — every request creates a Langfuse trace spanning supervisor classification, sub-agent reasoning, tool execution, and TTS synthesis
- **LLM generation spans** — token usage (`inputTokens`, `outputTokens`), model ID, latency, and full prompt I/O are recorded for every Bedrock call
- **TTS TTFB tracking** — time-to-first-byte and total synthesis latency are captured for ElevenLabs streaming calls
- **Offline evaluation** — `scripts/eval_tool_routing.py` runs 25 test cases through the routing pipeline, scoring tool-name accuracy and argument correctness, with results logged to Langfuse
- **Zero-overhead when disabled** — all tracing is no-op when Langfuse keys are not configured

### Integrations
- **Spotify** — search, play, pause, skip, queue, volume, shuffle, repeat, like, device control
- **Google Calendar** — today's events, upcoming schedule
- **iCloud Reminders** — create reminders via macOS bridge
- **Home Scenes** — trigger smart home scenes (lights, movie mode, goodnight)
- **Weather** — current conditions and forecasts via Open-Meteo
- **NBA Scores** — live game scores via ESPN
- **News** — top headlines via NewsAPI + GNews fallback
- **Phone Bridge** — private data (texts, OTPs) displayed only on phone, never spoken aloud

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
| Orchestration | LangGraph (Supervisor + Conditional Edges + ReAct Sub-Agents) |
| Embeddings | Amazon Titan Embeddings v1 (1536-dim) via AWS Bedrock |
| Vector Store | pgvector on Supabase PostgreSQL (HNSW index) |
| Observability | Langfuse (traces, generations, scores) |
| TTS | ElevenLabs (streaming, 192kbps) with AWS Polly fallback |
| STT | faster-whisper (local, CPU, int8) |
| Wake Word | openwakeword (ONNX) |
| Backend | FastAPI + Uvicorn |
| Music | Spotify Web API (spotipy) |
| Calendar | Google Calendar API |
| Database | Supabase (PostgreSQL + pgvector) |
| Frontend | Vanilla JS + WebGL Fluid Simulation |
| Audio | sounddevice + soundfile |
| Sound FX | ElevenLabs Sound Effects API |

## Multi-Agent Architecture

Ziri uses a **Supervisor-Worker multi-agent pattern** with **ReAct reasoning loops**:

```
                         ┌─────────────┐
                         │   START     │
                         └──────┬──────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │     Supervisor        │
                    │  1. Deterministic     │
                    │     phrase match      │
                    │  2. Bedrock domain    │
                    │     classification    │
                    └───────────┬───────────┘
                                │
              ┌─────────┬───────┴───────┬──────────┐
              ▼         ▼               ▼          ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Music   │ │   Info   │ │   Home   │ │  Quick   │
        │  Agent   │ │  Agent   │ │  Agent   │ │  Action  │
        │          │ │          │ │          │ │  (no LLM)│
        │ Think    │ │ Think    │ │ Think    │ │          │
        │   ↓      │ │   ↓      │ │   ↓      │ │ Direct   │
        │ Act      │ │ Act      │ │ Act      │ │ execute  │
        │   ↓      │ │   ↓      │ │   ↓      │ │          │
        │ Observe  │ │ Observe  │ │ Observe  │ │          │
        │ (loop≤3) │ │ (loop≤3) │ │ (loop≤2) │ │          │
        └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
              └─────────┴───────────┴──────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │       Respond         │
                    │  Personality rewrite  │
                    │  ElevenLabs TTS       │
                    │  Memory + Embedding   │
                    │  Langfuse trace       │
                    └───────────┬───────────┘
                                │
                                ▼
                         ┌─────────────┐
                         │     END     │
                         └─────────────┘
```

**Domain routing:**
- **MusicAgent** — all `spotify.*` tools (14 tools)
- **InfoAgent** — `general.answer`, `weather.*`, `nba.*`, `news.*`, `calendar.today`, `time.*`
- **HomeAgent** — `home.scene`, `reminders.create`, `private.phone_data`
- **QuickAction** — deterministic matches (pause, skip, volume, etc.) execute with zero LLM calls

## Semantic Memory

Ziri implements a hybrid memory system combining recency-based and semantic retrieval:

1. **On every conversation turn**, the user's text, intent, tool name, and assistant response are concatenated and embedded via Amazon Titan Embeddings v1 (1536 dimensions)
2. **The embedding is stored** in the `conversation_turns` table's `embedding vector(1536)` column, indexed with an HNSW index (`vector_cosine_ops`, m=16, ef_construction=64)
3. **Before each LLM routing call**, the user's current query is embedded and matched against past turns using a Supabase RPC function (`match_conversation_turns`)
4. **Both contexts are injected** into the prompt:
   - `memory_context` — last N turns (chronological, for immediate context)
   - `semantic_context` — top-K most similar past turns (for long-range recall)

## Observability

Ziri uses [Langfuse](https://langfuse.com) for full-stack LLM observability:

| What's Traced | Metrics Captured |
|---------------|-----------------|
| Supervisor classification | Input/output tokens, latency, domain decision |
| Sub-agent Think steps | Token usage, tool selection, ReAct iteration count |
| General answer generation | Token usage, latency, answer text |
| ElevenLabs TTS | Time-to-first-byte (TTFB), total synthesis time, text length |
| End-to-end request | User ID, device, room, domain routed to, full pipeline latency |

**Offline evaluation** (`scripts/eval_tool_routing.py`):
- 25 test cases covering all tool categories
- Scores: tool name match, argument correctness, overall accuracy
- Results logged to Langfuse as scores on each trace

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

**Run the pgvector migration** (if using Supabase):
```bash
# In your Supabase SQL editor, run:
# sql/001_init.sql  (if not already applied)
# sql/002_vector_index.sql
```

**Server + always-on listener:**
```bash
python3 run_listener.py
```

**Server only (no wake word):**
```bash
python3 run_listener.py --no-listener
```

**Run the routing evaluation:**
```bash
python3 scripts/eval_tool_routing.py --verbose
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
| `AWS_ACCESS_KEY_ID` | AWS credentials for Bedrock + Titan Embeddings |
| `SPOTIFY_CLIENT_ID` | Spotify app credentials |
| `SPOTIFY_REFRESH_TOKEN` | Long-lived Spotify auth |
| `SUPABASE_URL` | Supabase project URL (for sessions, memory, pgvector) |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key (observability) |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse host (default: `https://cloud.langfuse.com`) |
| `EMBEDDING_MODEL_ID` | Bedrock embedding model (default: `amazon.titan-embed-text-v1:0`) |
| `SEMANTIC_MEMORY_ENABLED` | Enable/disable vector memory search (default: `true`) |
| `SEMANTIC_MEMORY_TOP_K` | Number of similar turns to retrieve (default: `3`) |

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
Runtime health check with component status. Now includes `router` (shows `multi_agent_supervisor`), `semantic_memory`, and `tracing` fields.

### `GET /debug/connections`
Connection status for Spotify, Bedrock, TTS, and memory store.

## Project Structure

```
app/
├── main.py                        # FastAPI app, routes, middleware
├── hub.py                         # Dependency wiring, request lifecycle
├── settings.py                    # Pydantic settings (env-driven)
├── schemas.py                     # Request/response models
├── core/
│   ├── orchestrator.py            # LangGraph pipeline (supervisor → agents → respond)
│   ├── supervisor.py              # Supervisor agent (deterministic + Bedrock domain classifier)
│   ├── brain.py                   # Bedrock Claude router + deterministic fast path
│   ├── tool_runner.py             # Legacy tool dispatcher (used by quick_action path)
│   ├── tracing.py                 # Langfuse observability (traces, generations, spans)
│   ├── embeddings.py              # Amazon Titan embedding wrapper
│   ├── memory.py                  # Conversational memory (in-memory + Supabase + pgvector)
│   ├── personality.py             # Quick replies, response rewriting
│   ├── listener.py                # Always-on wake word + STT + playback
│   ├── audio_player.py            # sounddevice playback + 11Labs sound effects
│   ├── device_registry.py         # Device → room → speaker resolution
│   └── agents/
│       ├── __init__.py            # Sub-agent package exports
│       ├── music_agent.py         # Music domain ReAct sub-agent (14 Spotify tools)
│       ├── info_agent.py          # Info domain ReAct sub-agent (weather, NBA, news, etc.)
│       └── home_agent.py          # Home domain ReAct sub-agent (scenes, reminders, phone)
├── integrations/
│   ├── tts.py                     # ElevenLabs streaming TTS + Polly fallback + TTFB tracking
│   ├── spotify_controller.py      # Spotify Web API (search, playback, volume)
│   ├── calendar_controller.py     # Google Calendar
│   ├── weather.py                 # Weather via Open-Meteo
│   ├── nba.py                     # NBA scores via ESPN
│   ├── news.py                    # News headlines via NewsAPI + GNews
│   ├── reminders_bridge.py        # iCloud Reminders
│   ├── home_scene_controller.py   # Smart home scenes
│   └── phone_bridge.py            # Private phone data
├── static/
│   ├── listen.html                # WebGL fluid dashboard
│   └── audio/                     # TTS output + sound effects + cached phrases
├── config/
│   ├── device_map.yaml            # Device/room/speaker mapping
│   └── scenes.yaml                # Home scene definitions
└── data/
    ├── session_repository.py      # Session persistence (Supabase + in-memory)
    └── preferences_repository.py  # User preferences

sql/
├── 001_init.sql                   # Base schema (sessions, conversation_turns, user_preferences)
└── 002_vector_index.sql           # HNSW vector index + match_conversation_turns RPC

scripts/
└── eval_tool_routing.py           # Offline routing accuracy evaluation (25 test cases)

tests/
├── test_brain.py                  # Brain unit tests (deterministic routing)
├── test_intent_behaviors.py       # Integration tests via FastAPI TestClient
├── test_api.py                    # API endpoint tests
└── fixtures/
    └── routing_eval.jsonl         # Evaluation test cases (input → expected tool + args)

docs/
├── architecture.md                # Detailed architecture documentation
├── deployment.md                  # Deployment guide
└── ios_shortcut_flow.md           # iOS Siri Shortcuts integration
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
- [ ] Langfuse prompt management (version-controlled system prompts)
- [ ] A/B testing on routing models via Langfuse experiments
- [ ] Streaming LLM responses with partial TTS synthesis

## License

Private. All rights reserved.
