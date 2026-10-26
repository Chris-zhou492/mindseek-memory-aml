"""FastAPI entry point for the local SQLite text-memory challenge service."""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Callable

import uvicorn
from dotenv import dotenv_values
from fastapi import FastAPI

from memory_service.app import create_app
from memory_service.embeddings import QwenEmbedding
from memory_service.query_planner import FollowupQueryPlanner
from memory_service.reranker import DeepSeekReranker
from memory_service.sqlite_storage import SQLiteStorage
from memory_service.structured_profile import extract_structured_profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_local_env(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.is_file():
        return
    for name, value in dotenv_values(path, encoding="utf-8-sig").items():
        # Local reranker settings are pinned in this file. Reload them even if
        # an earlier dotenv import expanded a key reference using stale process credentials.
        pinned_reranker = name in {"MEMORY_RERANK_API_KEY", "MEMORY_RERANK_BASE_URL", "MEMORY_RERANK_MODEL"}
        if value is not None and (name not in os.environ or pinned_reranker):
            os.environ[name] = value


def create_local_app(
    storage: SQLiteStorage | None = None,
    *,
    db_path: str | Path | None = None,
    profile_extractor: Callable[[list[dict[str, Any]]], dict[str, list[str]]] = extract_structured_profile,
) -> FastAPI:
    """Use the shared FastAPI routes with a local SQLite storage backend."""
    load_local_env()
    from memory_service.directed_evidence import DirectedEvidence
    from memory_service.event_memory import EventMemory
    from memory_service.evidence_selector import DeepSeekEvidenceSelector
    from memory_service.game_log_evidence import GameLogEvidence
    from memory_service.record_calculator import (
        DeepSeekCalculationPlanner,
        RecordCalculator,
    )
    from memory_service.state_evidence import StateEvidence

    owns_storage = storage is None
    if storage is None:
        mode = os.environ.get("MEMORY_RETRIEVAL_MODE", "hybrid")
        if mode not in {"hybrid", "lexical"}:
            raise ValueError("MEMORY_RETRIEVAL_MODE must be hybrid or lexical")
        embedder = QwenEmbedding() if mode == "hybrid" else None
        reranker = (
            DeepSeekReranker()
            if mode == "hybrid" and os.environ.get("MEMORY_RERANK_ENABLED", "true").lower() == "true"
            else None
        )
        storage = SQLiteStorage(
            db_path or os.environ.get("SQLITE_PATH") or PROJECT_ROOT / "state" / "memory.sqlite3",
            embedder=embedder,
            embedding_model=embedder.model_id if embedder else "",
            reranker=reranker,
            rerank_candidates=int(os.environ.get("MEMORY_RERANK_CANDIDATES", "60")),
            evidence_coverage=os.environ.get("MEMORY_EVIDENCE_COVERAGE", "false").lower() == "true",
            context_window=int(os.environ.get("MEMORY_CONTEXT_WINDOW", "1")),
            evidence_char_budget=int(os.environ.get("MEMORY_EVIDENCE_CHAR_BUDGET", "24000")),
            query_planner=(
                FollowupQueryPlanner()
                if mode == "hybrid" and os.environ.get("MEMORY_MULTI_SEARCH", "false").lower() == "true"
                else None
            ),
            evidence_selector=(
                DeepSeekEvidenceSelector()
                if os.environ.get("MEMORY_EVIDENCE_SELECTION", "false").lower() == "true"
                else None
            ),
            source_record_index=os.environ.get("MEMORY_SOURCE_RECORD_INDEX", "false").lower() == "true",
            record_calculator=(
                RecordCalculator(DeepSeekCalculationPlanner())
                if os.environ.get("MEMORY_RECORD_CALCULATIONS", "false").lower() == "true"
                else None
            ),
            state_reconciler=StateEvidence()
            if os.environ.get("MEMORY_STATE_EVIDENCE", "false").lower() == "true"
            else None,
            hybrid_fusion=os.environ.get("MEMORY_HYBRID_FUSION", "legacy"),
            evidence_dedup=os.environ.get("MEMORY_EVIDENCE_DEDUP", "false").lower() == "true",
            directed_evidence=DirectedEvidence()
            if os.environ.get("MEMORY_DIRECTED_EVIDENCE", "false").lower() == "true"
            else None,
            game_log_evidence=GameLogEvidence()
            if os.environ.get("MEMORY_GAME_LOG_EVIDENCE", "false").lower() == "true"
            else None,
            event_memory=EventMemory(
                validation_policy=os.environ.get("MEMORY_EVENT_VALIDATION_POLICY", "strict"),
                max_input_chars=int(os.environ.get("MEMORY_EVENT_MAX_INPUT_CHARS", "160000")),
                max_source_chars=int(os.environ.get("MEMORY_EVENT_MAX_SOURCE_CHARS", "24000")),
            )
            if os.environ.get("MEMORY_EVENT_MEMORY", "false").lower() == "true"
            else None,
        )
    return create_app(storage, profile_extractor=profile_extractor, close_storage=owns_storage)


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--db", default=os.environ.get("SQLITE_PATH") or str(PROJECT_ROOT / "state" / "memory.sqlite3"))
    parser.add_argument("--reindex-vectors", action="store_true", help="Backfill missing vectors, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    app = create_local_app(db_path=args.db)
    if args.reindex_vectors:
        try:
            print("Indexed memories:", app.state.challenge_storage.reindex_vectors())
        finally:
            app.state.challenge_storage.close()
        return
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
