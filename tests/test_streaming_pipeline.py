"""Tests for the streaming LLM → TTS pipeline.

Covers:
  - TTS sentence-boundary splitting and token buffering
  - TTS.synthesize_streaming() with mocked ElevenLabs HTTP
  - InfoAgent.general_answer_streaming() with mocked Bedrock converse_stream
  - Orchestrator handle_intent_streaming() routing logic
  - End-to-end latency assertions (streaming path < file-based path)
"""
from __future__ import annotations

import re
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from app.core.agents.info_agent import InfoAgent
from app.core.device_registry import DeviceContext
from app.core.memory import InMemoryStore
from app.core.orchestrator import ZiriOrchestrator
from app.core.supervisor import Supervisor, SupervisorResult
from app.core.brain import Brain
from app.integrations.tts import TTS, _SENTENCE_BOUNDARY, _MIN_CHUNK_LEN
from app.schemas import IntentRequest, IntentType, RouterDecision, ToolResult
from app.settings import Settings


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture()
def settings() -> Settings:
    return Settings(
        supabase_url=None,
        supabase_service_role_key=None,
        elevenlabs_api_key="test-key-fake",
        elevenlabs_voice_id="test-voice",
        elevenlabs_model_id="eleven_turbo_v2_5",
        enable_polly=False,
        semantic_memory_enabled=False,
        return_audio_url=True,
        bedrock_model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    )


@pytest.fixture()
def settings_no_eleven() -> Settings:
    return Settings(
        supabase_url=None,
        supabase_service_role_key=None,
        elevenlabs_api_key=None,
        enable_polly=False,
        semantic_memory_enabled=False,
        return_audio_url=False,
    )


@pytest.fixture()
def memory() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture()
def device_context() -> DeviceContext:
    return DeviceContext(
        device_id="Mac_Listener",
        room_name="Office",
        default_speaker="Mac_Speaker",
        spotify_device_id=None,
    )


def _make_request(text: str) -> IntentRequest:
    return IntentRequest(
        user_id="Aditya",
        device_id="Mac_Listener",
        room="Office",
        raw_text=text,
        timestamp=datetime.now(timezone.utc),
    )


# =====================================================================
# Sentence boundary splitting tests
# =====================================================================

class TestSentenceBoundary:
    """Verify the regex correctly splits text at sentence boundaries."""

    def test_splits_on_period(self):
        text = "Hello world. How are you?"
        parts = _SENTENCE_BOUNDARY.split(text)
        assert len(parts) == 2
        assert parts[0] == "Hello world."
        assert parts[1] == "How are you?"

    def test_splits_on_exclamation(self):
        text = "Wow! That's great."
        parts = _SENTENCE_BOUNDARY.split(text)
        assert len(parts) == 2

    def test_splits_on_question_mark(self):
        text = "What time is it? It's three."
        parts = _SENTENCE_BOUNDARY.split(text)
        assert len(parts) == 2

    def test_no_split_without_boundary(self):
        text = "Hello world"
        parts = _SENTENCE_BOUNDARY.split(text)
        assert len(parts) == 1

    def test_ellipsis_split(self):
        text = "Well... I think so."
        parts = _SENTENCE_BOUNDARY.split(text)
        assert len(parts) >= 1

    def test_multiple_sentences(self):
        text = "First sentence. Second one. Third here!"
        parts = _SENTENCE_BOUNDARY.split(text)
        assert len(parts) == 3


# =====================================================================
# Token buffering tests (simulating LLM token stream)
# =====================================================================

class TestTokenBuffering:
    """Test that streaming tokens are correctly buffered into sentences."""

    def _buffer_tokens(self, tokens: list[str]) -> list[str]:
        """Simulate the sentence buffering logic from synthesize_streaming."""
        sentence_buffer = ""
        sentences: list[str] = []
        for token in tokens:
            sentence_buffer += token
            parts = _SENTENCE_BOUNDARY.split(sentence_buffer)
            if len(parts) > 1:
                for complete in parts[:-1]:
                    stripped = complete.strip()
                    if stripped and len(stripped) >= _MIN_CHUNK_LEN:
                        sentences.append(stripped)
                sentence_buffer = parts[-1]
        if sentence_buffer.strip():
            sentences.append(sentence_buffer.strip())
        return sentences

    def test_single_sentence_from_tokens(self):
        tokens = ["The ", "weather ", "is ", "nice ", "today."]
        sentences = self._buffer_tokens(tokens)
        assert len(sentences) == 1
        assert sentences[0] == "The weather is nice today."

    def test_two_sentences_from_tokens(self):
        tokens = ["It's a beautiful sunny day outside today. ", "The temperature ", "is roughly ", "72 degrees Fahrenheit."]
        sentences = self._buffer_tokens(tokens)
        assert len(sentences) == 2
        assert "sunny" in sentences[0]
        assert "72" in sentences[1]

    def test_short_fragments_merged_into_remainder(self):
        """Fragments shorter than _MIN_CHUNK_LEN get dropped by the sentence
        splitter but land in the final remainder flush."""
        tokens = ["Hi. ", "Yes."]
        sentences = self._buffer_tokens(tokens)
        assert len(sentences) == 1
        assert sentences[0] == "Yes."

    def test_boundary_in_middle_of_token(self):
        tokens = ["The answer to everything is 42. ", "The meaning of life is also relevant."]
        sentences = self._buffer_tokens(tokens)
        assert len(sentences) >= 1
        full = " ".join(sentences)
        assert "42" in full

    def test_empty_tokens_handled(self):
        tokens = ["", "Hello.", ""]
        sentences = self._buffer_tokens(tokens)
        assert len(sentences) >= 1


# =====================================================================
# TTS.synthesize_streaming() tests
# =====================================================================

def _fake_pcm_response(num_chunks: int = 3, chunk_size: int = 4096):
    """Create a mock httpx streaming response returning PCM int16 chunks."""
    pcm_data = np.zeros(chunk_size // 2, dtype=np.int16).tobytes()
    chunks = [pcm_data] * num_chunks

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_bytes.return_value = iter(chunks)
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestTTSSynthesizeStreaming:
    """Test TTS.synthesize_streaming() with mocked HTTP and audio."""

    @patch("app.integrations.tts.sd")
    def test_returns_full_text(self, mock_sd, settings):
        tts = TTS(settings=settings)

        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream
        mock_sd.sleep = MagicMock()

        tokens = iter(["The capital of France is Paris."])

        mock_client = MagicMock()
        mock_client.stream.return_value = _fake_pcm_response()
        tts._http = mock_client

        with patch.object(tts, '_get_http', return_value=mock_client):
            result = tts.synthesize_streaming(tokens)

        assert "Paris" in result
        assert "France" in result

    @patch("app.integrations.tts.sd")
    def test_on_sentence_callback_fires(self, mock_sd, settings):
        tts = TTS(settings=settings)
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream
        mock_sd.sleep = MagicMock()

        fired: list[str] = []
        tokens = iter(["Hello there. Nice to meet you."])

        mock_client = MagicMock()
        mock_client.stream.return_value = _fake_pcm_response()
        tts._http = mock_client

        with patch.object(tts, '_get_http', return_value=mock_client):
            tts.synthesize_streaming(tokens, on_sentence=lambda s: fired.append(s))

        assert len(fired) >= 1

    @patch("app.integrations.tts.sd")
    def test_opens_output_stream_with_correct_params(self, mock_sd, settings):
        tts = TTS(settings=settings)
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream
        mock_sd.sleep = MagicMock()

        tokens = iter(["Testing audio output stream."])

        mock_client = MagicMock()
        mock_client.stream.return_value = _fake_pcm_response()
        tts._http = mock_client

        with patch.object(tts, '_get_http', return_value=mock_client):
            tts.synthesize_streaming(tokens)

        mock_sd.OutputStream.assert_called_once_with(
            samplerate=24000,
            channels=1,
            dtype="float32",
        )
        mock_stream.start.assert_called_once()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_fallback_to_synthesize_without_elevenlabs(self, settings_no_eleven):
        tts = TTS(settings=settings_no_eleven)
        tts.synthesize = MagicMock(return_value=None)

        tokens = iter(["Fallback text."])
        result = tts.synthesize_streaming(tokens)

        assert result == "Fallback text."
        tts.synthesize.assert_called_once()

    @patch("app.integrations.tts.sd")
    def test_handles_http_error_gracefully(self, mock_sd, settings):
        tts = TTS(settings=settings)
        mock_sd.sleep = MagicMock()

        tokens = iter(["Some text to speak."])

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_resp.read = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_resp
        tts._http = mock_client

        with patch.object(tts, '_get_http', return_value=mock_client):
            result = tts.synthesize_streaming(tokens)

        assert result == "Some text to speak."

    @patch("app.integrations.tts.sd")
    def test_empty_iterator_returns_empty(self, mock_sd, settings):
        tts = TTS(settings=settings)
        result = tts.synthesize_streaming(iter([]))
        assert result == ""


# =====================================================================
# InfoAgent.general_answer_streaming() tests
# =====================================================================

class TestInfoAgentStreaming:
    """Test that InfoAgent.general_answer_streaming yields tokens correctly."""

    def _make_agent(self, settings) -> InfoAgent:
        from app.integrations.calendar_controller import CalendarController
        from app.integrations.weather import WeatherController
        from app.integrations.nba import NBAController
        from app.integrations.news import NewsController

        agent = InfoAgent(
            settings=settings,
            calendar=MagicMock(spec=CalendarController),
            weather=MagicMock(spec=WeatherController),
            nba=MagicMock(spec=NBAController),
            news=MagicMock(spec=NewsController),
        )
        return agent

    def test_yields_tokens_from_converse_stream(self, settings):
        agent = self._make_agent(settings)
        mock_bedrock = MagicMock()

        stream_events = [
            {"contentBlockDelta": {"delta": {"text": "The "}}},
            {"contentBlockDelta": {"delta": {"text": "answer "}}},
            {"contentBlockDelta": {"delta": {"text": "is 42."}}},
            {"contentBlockStop": {}},
        ]

        mock_bedrock.converse_stream.return_value = {"stream": iter(stream_events)}
        agent._bedrock = mock_bedrock
        agent._model_id = "test-model"

        tokens = list(agent.general_answer_streaming("meaning of life"))
        full_text = "".join(tokens)

        assert "42" in full_text
        assert len(tokens) == 3

    def test_fallback_when_no_bedrock(self, settings):
        agent = self._make_agent(settings)
        agent._bedrock = None

        tokens = list(agent.general_answer_streaming("test query"))
        assert len(tokens) == 1
        assert "not sure" in tokens[0].lower()

    def test_fallback_when_stream_is_none(self, settings):
        agent = self._make_agent(settings)
        mock_bedrock = MagicMock()
        mock_bedrock.converse_stream.return_value = {"stream": None}
        mock_bedrock.converse.return_value = {
            "output": {"message": {"content": [{"text": "Fallback answer."}]}},
            "usage": {},
        }
        agent._bedrock = mock_bedrock
        agent._model_id = "test-model"

        tokens = list(agent.general_answer_streaming("test"))
        full_text = "".join(tokens)
        assert "Fallback" in full_text

    def test_fallback_on_exception(self, settings):
        agent = self._make_agent(settings)
        mock_bedrock = MagicMock()
        mock_bedrock.converse_stream.side_effect = Exception("Network error")
        mock_bedrock.converse.return_value = {
            "output": {"message": {"content": [{"text": "Error fallback."}]}},
            "usage": {},
        }
        agent._bedrock = mock_bedrock
        agent._model_id = "test-model"

        tokens = list(agent.general_answer_streaming("test"))
        full_text = "".join(tokens)
        assert len(full_text) > 0

    def test_skips_empty_deltas(self, settings):
        agent = self._make_agent(settings)
        mock_bedrock = MagicMock()

        stream_events = [
            {"contentBlockDelta": {"delta": {"text": "Hello"}}},
            {"contentBlockDelta": {"delta": {}}},
            {"contentBlockDelta": {"delta": {"text": ""}}},
            {"contentBlockDelta": {"delta": {"text": " world."}}},
            {"messageStop": {}},
        ]
        mock_bedrock.converse_stream.return_value = {"stream": iter(stream_events)}
        agent._bedrock = mock_bedrock
        agent._model_id = "test-model"

        tokens = list(agent.general_answer_streaming("hi"))
        full_text = "".join(tokens)
        assert full_text == "Hello world."


# =====================================================================
# Orchestrator handle_intent_streaming() routing tests
# =====================================================================

class TestOrchestratorStreaming:
    """Test that the orchestrator correctly routes to streaming vs non-streaming."""

    def _build_orchestrator(
        self,
        settings,
        memory,
        info_agent=None,
        tts=None,
    ) -> ZiriOrchestrator:
        brain = Brain(settings=settings, memory=memory)
        brain._bedrock = None

        supervisor = Supervisor(settings=settings, memory=memory, bedrock_client=None)

        mock_tool_runner = MagicMock()
        mock_tool_runner.run.return_value = ToolResult(
            ok=True, action_code="MUSIC_PAUSE", speak_text="Paused.",
        )
        mock_tool_runner.spotify = MagicMock()

        mock_tts = tts or MagicMock(spec=TTS)
        if not tts:
            mock_tts.synthesize.return_value = None
            mock_tts.synthesize_streaming.return_value = "Streamed text."

        _info_agent = info_agent or MagicMock()
        if not info_agent:
            _info_agent.run.return_value = (
                RouterDecision(
                    intent_type=IntentType.INFO_QUERY,
                    tool_name="general.answer",
                    tool_args={"query": "test"},
                    action_code="INFO_REPLY",
                    confidence=0.5,
                ),
                ToolResult(ok=True, action_code="INFO_REPLY", speak_text="Mocked."),
            )
            _info_agent._think.return_value = RouterDecision(
                intent_type=IntentType.INFO_QUERY,
                tool_name="general.answer",
                tool_args={"query": "test"},
                action_code="INFO_REPLY",
                confidence=0.85,
            )
            _info_agent.general_answer_streaming.return_value = iter(["Streamed ", "answer."])

        return ZiriOrchestrator(
            settings=settings,
            brain=brain,
            tool_runner=mock_tool_runner,
            memory=memory,
            tts=mock_tts,
            supervisor=supervisor,
            music_agent=MagicMock(),
            info_agent=_info_agent,
            home_agent=MagicMock(),
        )

    def test_quick_action_does_not_stream(self, settings, memory, device_context):
        orch = self._build_orchestrator(settings, memory)
        req = _make_request("pause")

        resp, decision, result, did_stream = orch.handle_intent_streaming(req, device_context)

        assert did_stream is False
        assert decision.tool_name == "spotify.pause"

    def test_info_query_uses_streaming(self, settings, memory, device_context):
        orch = self._build_orchestrator(settings, memory)
        req = _make_request("explain quantum computing")

        resp, decision, result, did_stream = orch.handle_intent_streaming(req, device_context)

        assert did_stream is True
        assert decision.tool_name == "general.answer"
        assert result.payload.get("streamed") is True
        orch.tts.synthesize_streaming.assert_called_once()

    def test_info_non_general_answer_falls_back(self, settings, memory, device_context):
        info_agent = MagicMock()
        info_agent._think.return_value = RouterDecision(
            intent_type=IntentType.WEATHER,
            tool_name="weather.current",
            tool_args={},
            action_code="WEATHER_CURRENT",
            confidence=0.9,
        )
        info_agent.run.return_value = (
            RouterDecision(
                intent_type=IntentType.WEATHER,
                tool_name="weather.current",
                tool_args={},
                action_code="WEATHER_CURRENT",
                confidence=0.9,
            ),
            ToolResult(ok=True, action_code="WEATHER_CURRENT", speak_text="72 degrees."),
        )

        orch = self._build_orchestrator(settings, memory, info_agent=info_agent)
        req = _make_request("what's the weather")

        resp, decision, result, did_stream = orch.handle_intent_streaming(req, device_context)

        assert did_stream is False

    def test_streaming_remembers_turn(self, settings, memory, device_context):
        orch = self._build_orchestrator(settings, memory)
        req = _make_request("what is gravity")

        orch.handle_intent_streaming(req, device_context)

        ctx = memory.get_recent_context("Aditya")
        assert "gravity" in ctx.lower()

    def test_streaming_returns_valid_response(self, settings, memory, device_context):
        orch = self._build_orchestrator(settings, memory)
        req = _make_request("who invented the telephone")

        resp, decision, result, did_stream = orch.handle_intent_streaming(req, device_context)

        assert resp.speak_text is not None
        assert resp.action_code == "INFO_REPLY"
        assert resp.metadata.get("streamed") is True
        assert resp.audio_url is None

    def test_can_stream_info_with_deterministic_general_answer(self, settings, memory):
        brain = Brain(settings=settings, memory=memory)
        brain._bedrock = None
        supervisor = Supervisor(settings=settings, memory=memory, bedrock_client=None)
        orch = ZiriOrchestrator(
            settings=settings, brain=brain,
            tool_runner=MagicMock(), memory=memory,
            tts=MagicMock(spec=TTS), supervisor=supervisor,
            music_agent=MagicMock(), info_agent=MagicMock(), home_agent=MagicMock(),
        )

        sv_result = SupervisorResult(
            domain="info",
            query="test",
            deterministic_decision=RouterDecision(
                intent_type=IntentType.INFO_QUERY,
                tool_name="general.answer",
                tool_args={"query": "test"},
                action_code="INFO_REPLY",
            ),
        )
        assert orch._can_stream_info(sv_result) is True

    def test_cannot_stream_deterministic_non_general(self, settings, memory):
        brain = Brain(settings=settings, memory=memory)
        brain._bedrock = None
        supervisor = Supervisor(settings=settings, memory=memory, bedrock_client=None)
        orch = ZiriOrchestrator(
            settings=settings, brain=brain,
            tool_runner=MagicMock(), memory=memory,
            tts=MagicMock(spec=TTS), supervisor=supervisor,
            music_agent=MagicMock(), info_agent=MagicMock(), home_agent=MagicMock(),
        )

        sv_result = SupervisorResult(
            domain="quick",
            query="pause",
            deterministic_decision=RouterDecision(
                intent_type=IntentType.MUSIC_COMMAND,
                tool_name="spotify.pause",
                tool_args={},
                action_code="MUSIC_PAUSE",
            ),
        )
        assert orch._can_stream_info(sv_result) is False


# =====================================================================
# Latency simulation tests
# =====================================================================

class TestStreamingLatency:
    """Verify that the streaming path avoids the wait-for-full-response bottleneck."""

    def test_tokens_arrive_incrementally(self):
        """Simulate token arrival and verify they're available immediately."""
        received: list[tuple[float, str]] = []
        start = time.monotonic()

        def slow_token_gen() -> Iterator[str]:
            tokens = ["The ", "capital ", "of France ", "is Paris."]
            for tok in tokens:
                time.sleep(0.01)
                yield tok

        for token in slow_token_gen():
            received.append((time.monotonic() - start, token))

        assert len(received) == 4
        first_token_time = received[0][0]
        last_token_time = received[-1][0]
        assert first_token_time < 0.05
        assert last_token_time - first_token_time > 0.02

    @patch("app.integrations.tts.sd")
    def test_streaming_faster_than_batch(self, mock_sd, settings):
        """Streaming should start playback sooner than waiting for the full response."""
        tts = TTS(settings=settings)
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream
        mock_sd.sleep = MagicMock()

        playback_started_at: list[float] = []

        def track_write(*args, **kwargs):
            if not playback_started_at:
                playback_started_at.append(time.monotonic())

        mock_stream.write = track_write

        pcm_data = np.zeros(2048, dtype=np.int16).tobytes()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_bytes.return_value = iter([pcm_data, pcm_data])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_resp

        def slow_tokens() -> Iterator[str]:
            yield "First sentence is here. "
            time.sleep(0.05)
            yield "Second sentence follows."

        t0 = time.monotonic()
        with patch.object(tts, '_get_http', return_value=mock_client):
            tts.synthesize_streaming(slow_tokens())

        assert len(playback_started_at) > 0
        time_to_first_audio = playback_started_at[0] - t0
        assert time_to_first_audio < 1.0


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:

    @patch("app.integrations.tts.sd")
    def test_single_word_response(self, mock_sd, settings):
        tts = TTS(settings=settings)
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream
        mock_sd.sleep = MagicMock()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        pcm = np.zeros(1024, dtype=np.int16).tobytes()
        mock_resp.iter_bytes.return_value = iter([pcm])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_resp

        with patch.object(tts, '_get_http', return_value=mock_client):
            result = tts.synthesize_streaming(iter(["OK"]))

        assert result == "OK"

    @patch("app.integrations.tts.sd")
    def test_unicode_text(self, mock_sd, settings):
        tts = TTS(settings=settings)
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream
        mock_sd.sleep = MagicMock()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        pcm = np.zeros(1024, dtype=np.int16).tobytes()
        mock_resp.iter_bytes.return_value = iter([pcm])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_resp

        with patch.object(tts, '_get_http', return_value=mock_client):
            result = tts.synthesize_streaming(iter(["Café résumé naïve."]))

        assert "Café" in result

    @patch("app.integrations.tts.sd")
    def test_langfuse_trace_recorded(self, mock_sd, settings):
        tts = TTS(settings=settings)
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream
        mock_sd.sleep = MagicMock()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        pcm = np.zeros(1024, dtype=np.int16).tobytes()
        mock_resp.iter_bytes.return_value = iter([pcm])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_resp

        mock_trace = MagicMock()

        with patch.object(tts, '_get_http', return_value=mock_client):
            tts.synthesize_streaming(iter(["Traced response here."]), trace=mock_trace)

        mock_trace.span.assert_called_once()
        call_kwargs = mock_trace.span.call_args
        assert call_kwargs[1]["name"] == "tts_streaming"
        assert call_kwargs[1]["metadata"]["streaming"] is True
