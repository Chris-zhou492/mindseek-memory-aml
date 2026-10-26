from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from memory_service.credential_filter import filter_credentials
from memory_service.errors import RequestConflictError
from memory_service.profile import extract_profile
from memory_service.sqlite_storage import SQLiteStorage
from memory_service.structured_profile import ProfileValidationError

if TYPE_CHECKING:
    from memory_service.storage import ChallengeStorage

load_dotenv()
logger = logging.getLogger(__name__)


class ChallengeMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    timestamp: StrictInt | None = None

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must be a non-empty string")
        return value


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    messages: list[ChallengeMessage] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)

    @field_validator("request_id", "user_id", "session_id")
    @classmethod
    def identifier_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier must be a non-empty string")
        return value


class AddResponse(BaseModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    options: list[str] | None = None
    user_id: str = Field(min_length=1)
    top_k: StrictInt = Field(gt=0)

    @field_validator("query", "user_id")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be a non-empty string")
        return value


class FactSource(BaseModel):
    source_message_id: str
    source_ref: str | None = None
    source_date: str | None = None
    source_timestamp_ms: int | None = None
    quote: str


class SearchItem(BaseModel):
    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float | None = None
    created_at: str | None = None
    entity: str | None = None
    attribute: str | None = None
    value: str | None = None
    sources: list[FactSource] = Field(default_factory=list)
    retrieval_score: float | None = None
    retrieval_rank: int | None = None
    semantic_score: float | None = None
    bm25_score: float | None = None
    semantic_rank: int | None = None
    bm25_rank: int | None = None
    fusion_score: float | None = None
    fusion_mode: str | None = None
    recall_channels: list[str] = Field(default_factory=list)
    context_message_ids: list[str] = Field(default_factory=list)
    evidence_need: str | None = None
    group_member_ids: list[str] = Field(default_factory=list)
    budget_omitted_ids: list[str] = Field(default_factory=list)
    record_sources: list[dict] = Field(default_factory=list)
    oversized_records_omitted: int | None = None
    calculation_run_id: str | None = None
    calculation_complete: bool | None = None
    calculation_status: str | None = None
    calculation_needs: list[dict] = Field(default_factory=list)
    state_run_id: str | None = None
    memory_kind: str | None = None
    event_id: str | None = None
    event_statement_count: int | None = None
    omitted_event_statements: int | None = None
    event_states: list[str] = Field(default_factory=list)
    dedup_removed_statement_ids: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    data: list[SearchItem]


def require_challenge_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    expected = os.environ.get("CHALLENGE_API_KEY", "")
    if not expected:
        if os.environ.get("CHALLENGE_ALLOW_UNAUTHENTICATED", "").lower() == "true":
            return
        raise HTTPException(status_code=503, detail="Challenge API key is not configured")
    provided = x_api_key
    if provided is None and authorization:
        scheme, separator, value = authorization.partition(" ")
        if separator and scheme.lower() in {"bearer", "token"}:
            provided = value
    if provided is None or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


def get_storage(request: Request) -> ChallengeStorage | SQLiteStorage:
    storage = request.app.state.challenge_storage
    if storage is None:
        raise HTTPException(status_code=503, detail="Challenge storage is unavailable")
    return storage


def create_app(
    storage: ChallengeStorage | SQLiteStorage | None = None,
    profile_extractor: Callable[[list[dict]], dict[str, list[str]]] = extract_profile,
    close_storage: bool = False,
    credential_filter: bool | None = None,
) -> FastAPI:
    filter_enabled = (
        os.environ.get("MEMORY_CREDENTIAL_FILTER", "false").lower() == "true"
        if credential_filter is None
        else credential_filter
    )
    @asynccontextmanager
    async def lifespan(instance: FastAPI):
        if instance.state.challenge_storage is None:
            from memory_service.storage import ChallengeStorage

            instance.state.challenge_storage = ChallengeStorage(
                host=os.environ.get("POSTGRES_HOST", "postgres"),
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
                dbname=os.environ.get("POSTGRES_DB", "postgres"),
                user=os.environ.get("POSTGRES_USER", "postgres"),
                password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
                schema=os.environ.get("CHALLENGE_SCHEMA", "challenge_memory"),
            )
            owns_storage = True
        else:
            owns_storage = False
        try:
            yield
        finally:
            if owns_storage or close_storage:
                instance.state.challenge_storage.close()

    instance = FastAPI(title="MindSeek Recall Challenge API", version="1.0.0", lifespan=lifespan)
    instance.state.challenge_storage = storage

    @instance.get("/health")
    def health(challenge_storage: ChallengeStorage = Depends(get_storage)):
        try:
            challenge_storage.health()
        except Exception:
            logger.error("Challenge health check failed")
            raise HTTPException(status_code=503, detail="Challenge storage is unavailable")
        return {"status": "ok"}

    @instance.post("/add", response_model=AddResponse, dependencies=[Depends(require_challenge_key)])
    def add(payload: AddRequest, challenge_storage: ChallengeStorage = Depends(get_storage)):
        try:
            messages = [message.model_dump(exclude_none=True) for message in payload.messages]
            if challenge_storage.is_duplicate(
                request_id=payload.request_id,
                user_id=payload.user_id,
                session_id=payload.session_id,
                messages=messages,
            ):
                return AddResponse(
                    request_id=payload.request_id, user_id=payload.user_id, session_id=payload.session_id
                )
            profile_messages = (
                challenge_storage.prepare_profile_messages(messages)
                if hasattr(challenge_storage, "prepare_profile_messages")
                else messages
            )
            try:
                profile = profile_extractor(profile_messages)
            except ProfileValidationError:
                logger.warning(
                    "Structured profile validation failed; storing source messages without derived profile facts"
                )
                profile = {}
            challenge_storage.add(
                request_id=payload.request_id,
                user_id=payload.user_id,
                session_id=payload.session_id,
                messages=messages,
                profile=profile,
            )
        except RequestConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except Exception:
            logger.exception("Challenge Add failed")
            raise HTTPException(status_code=503, detail="Challenge Add failed")
        return AddResponse(request_id=payload.request_id, user_id=payload.user_id, session_id=payload.session_id)

    @instance.post(
        "/search",
        response_model=SearchResponse,
        response_model_exclude_unset=True,
        dependencies=[Depends(require_challenge_key)],
    )
    def search(payload: SearchRequest, challenge_storage: ChallengeStorage = Depends(get_storage)):
        try:
            results = challenge_storage.search(
                user_id=payload.user_id,
                query=payload.query,
                options=payload.options,
                top_k=payload.top_k,
            )
        except Exception:
            logger.exception("Challenge Search failed")
            raise HTTPException(status_code=503, detail="Challenge Search failed")
        if len(results) > payload.top_k:
            raise HTTPException(status_code=503, detail="Challenge Search exceeded top_k")
        if filter_enabled:
            results = filter_credentials(results)
        return SearchResponse(data=[SearchItem(**item) for item in results])

    return instance


app = create_app()
