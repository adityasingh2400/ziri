from __future__ import annotations

import logging
from typing import Any

import httpx

from app.schemas import ToolResult
from app.settings import Settings

logger = logging.getLogger(__name__)


class NewsController:
    """Fetches headlines via NewsAPI.org (free tier: 100 req/day)."""

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.news_api_key

    def get_headlines(self, category: str = "general", count: int = 5) -> ToolResult:
        if not self.api_key:
            return self._fallback_headlines(count)

        try:
            resp = httpx.get(
                "https://newsapi.org/v2/top-headlines",
                params={"country": "us", "category": category, "pageSize": count, "apiKey": self.api_key},
                timeout=5,
            )
            data = resp.json()
            articles = data.get("articles", [])
            if not articles:
                return ToolResult(ok=True, action_code="NEWS_HEADLINES", speak_text="No headlines available right now.", private_note="")

            titles = [a.get("title", "").split(" - ")[0] for a in articles[:count]]
            summary = ". ".join(t for t in titles if t)
            return ToolResult(
                ok=True,
                action_code="NEWS_HEADLINES",
                speak_text=summary,
                private_note="",
                payload={"headlines": titles},
            )
        except Exception as exc:
            logger.exception("News fetch failed")
            return ToolResult(ok=False, action_code="NEWS_ERROR", speak_text="I could not get the news right now.", private_note=str(exc), error="news_exception")

    def get_topic_news(self, query: str, count: int = 3) -> ToolResult:
        if not self.api_key:
            return ToolResult(ok=False, action_code="NEWS_ERROR", speak_text="News is not configured yet. Add NEWS_API_KEY to .env.", private_note="", error="no_api_key")

        try:
            resp = httpx.get(
                "https://newsapi.org/v2/everything",
                params={"q": query, "sortBy": "publishedAt", "pageSize": count, "language": "en", "apiKey": self.api_key},
                timeout=5,
            )
            data = resp.json()
            articles = data.get("articles", [])
            if not articles:
                return ToolResult(ok=True, action_code="NEWS_TOPIC", speak_text=f"No recent news about {query}.", private_note="")

            titles = [a.get("title", "").split(" - ")[0] for a in articles[:count]]
            summary = ". ".join(t for t in titles if t)
            return ToolResult(
                ok=True,
                action_code="NEWS_TOPIC",
                speak_text=summary,
                private_note="",
                payload={"headlines": titles, "query": query},
            )
        except Exception as exc:
            logger.exception("Topic news fetch failed")
            return ToolResult(ok=False, action_code="NEWS_ERROR", speak_text=f"I could not find news about {query}.", private_note=str(exc), error="news_topic_exception")

    def _fallback_headlines(self, count: int = 5) -> ToolResult:
        """Use free GNews API as fallback when NewsAPI key is missing."""
        try:
            resp = httpx.get(
                "https://gnews.io/api/v4/top-headlines",
                params={"lang": "en", "country": "us", "max": count, "apikey": ""},
                timeout=5,
            )
            if resp.status_code != 200:
                return ToolResult(ok=False, action_code="NEWS_ERROR", speak_text="News is not configured. Add NEWS_API_KEY to .env.", private_note="", error="no_api_key")
            data = resp.json()
            articles = data.get("articles", [])
            titles = [a.get("title", "") for a in articles[:count]]
            summary = ". ".join(t for t in titles if t)
            return ToolResult(ok=True, action_code="NEWS_HEADLINES", speak_text=summary or "No headlines available.", private_note="", payload={"headlines": titles})
        except Exception:
            return ToolResult(ok=False, action_code="NEWS_ERROR", speak_text="News is not configured. Add NEWS_API_KEY to .env.", private_note="", error="no_api_key")
