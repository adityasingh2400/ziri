"""Elasticsearch-backed keyword search and hybrid RRF re-ranking.

Provides ``ElasticsearchStore`` for indexing / querying conversation turns
and ``HybridSearcher`` that fuses keyword hits from ES with semantic hits
from pgvector using Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None  # type: ignore[misc,assignment]

INDEX_NAME = "ziri_conversation_turns"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "user_id": {"type": "keyword"},
            "raw_text": {"type": "text", "analyzer": "standard"},
            "assistant_speak": {"type": "text", "analyzer": "standard"},
            "intent_type": {"type": "keyword"},
            "tool_name": {"type": "keyword"},
            "created_at": {"type": "date"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}


class ElasticsearchStore:
    """Thin wrapper around the Elasticsearch Python client."""

    def __init__(self, url: str, index: str = INDEX_NAME) -> None:
        if Elasticsearch is None:
            raise RuntimeError("elasticsearch package is not installed")
        self.client = Elasticsearch(url)
        self.index = index
        self._ensure_index()

    def _ensure_index(self) -> None:
        try:
            if not self.client.indices.exists(index=self.index):
                self.client.indices.create(index=self.index, body=INDEX_MAPPING)
                logger.info("Created ES index: %s", self.index)
        except Exception as exc:
            logger.warning("ES index creation failed: %s", exc)

    def index_turn(
        self,
        user_id: str,
        raw_text: str,
        intent_type: str,
        tool_name: str,
        assistant_speak: str,
        created_at: str,
    ) -> None:
        try:
            self.client.index(
                index=self.index,
                body={
                    "user_id": user_id,
                    "raw_text": raw_text,
                    "intent_type": intent_type,
                    "tool_name": tool_name,
                    "assistant_speak": assistant_speak,
                    "created_at": created_at,
                },
            )
        except Exception as exc:
            logger.warning("ES index_turn failed: %s", exc)

    def keyword_search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        try:
            resp = self.client.search(
                index=self.index,
                body={
                    "size": top_k,
                    "query": {
                        "bool": {
                            "must": {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["raw_text", "assistant_speak"],
                                }
                            },
                            "filter": {"term": {"user_id": user_id}},
                        }
                    },
                },
            )
            return [
                {
                    "raw_text": hit["_source"].get("raw_text", ""),
                    "intent_type": hit["_source"].get("intent_type", ""),
                    "tool_name": hit["_source"].get("tool_name", ""),
                    "assistant_speak": hit["_source"].get("assistant_speak", ""),
                    "score": hit["_score"],
                }
                for hit in resp["hits"]["hits"]
            ]
        except Exception as exc:
            logger.warning("ES keyword_search failed: %s", exc)
            return []


class HybridSearcher:
    """Combines pgvector semantic results with ES keyword results via RRF.

    Reciprocal Rank Fusion:
        score(doc) = sum(1 / (k + rank)) for each result list the doc appears in
    """

    RRF_K = 60

    def __init__(self, es_store: ElasticsearchStore | None = None) -> None:
        self.es = es_store

    def search(
        self,
        user_id: str,
        query_text: str,
        semantic_results: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        keyword_results = []
        if self.es:
            keyword_results = self.es.keyword_search(user_id, query_text, top_k=top_k)

        if not keyword_results:
            return semantic_results[:top_k]
        if not semantic_results:
            return keyword_results[:top_k]

        return self._rrf_merge(semantic_results, keyword_results, top_k)

    def _rrf_merge(
        self,
        semantic: list[dict[str, Any]],
        keyword: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}
        k = self.RRF_K

        for rank, doc in enumerate(semantic, start=1):
            key = doc.get("raw_text", "")
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map.setdefault(key, doc)

        for rank, doc in enumerate(keyword, start=1):
            key = doc.get("raw_text", "")
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map.setdefault(key, doc)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in ranked[:top_k]]

    def format_context(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return ""
        return "\n".join(
            f"- user: {r.get('raw_text', '')} | intent: {r.get('intent_type', '')} | "
            f"tool: {r.get('tool_name', '')} | assistant: {r.get('assistant_speak', '')}"
            for r in results
        )
