"""Start the memory service with an explicit Academic configuration."""

import argparse
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
MAX_UNIT_BYTES = 24000
OPTIONAL_LLM_FEATURES = (
    "MEMORY_MULTI_SEARCH",
    "MEMORY_EVIDENCE_SELECTION",
    "MEMORY_RECORD_CALCULATIONS",
    "MEMORY_STATE_EVIDENCE",
    "MEMORY_STATE_STAGE_RECALL",
    "MEMORY_STATE_PHASE_PIPELINE",
    "MEMORY_EVENT_MEMORY",
    "MEMORY_DIRECTED_EVIDENCE",
    "MEMORY_GAME_LOG_EVIDENCE",
)


def configure():
    # An explicitly supplied local file wins over stale shell credentials.
    load_dotenv(os.environ.get("MEMORY_ENV_FILE", str(ROOT / ".env")), override=True)
    defaults = {
        "CHALLENGE_PROFILE_MODEL": "gpt-4o-mini",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "DEEPSEEK_API_BASE": "https://api.deepseek.com",
        "MEMORY_EMBEDDING_MODEL": "text-embedding-v4",
        "MEMORY_EMBEDDING_DIMENSIONS": "1024",
        "MEMORY_RETRIEVAL_MODE": "hybrid",
        "MEMORY_SOURCE_RECORD_INDEX": "true",
        "MEMORY_RERANK_ENABLED": "true",
        "MEMORY_RERANK_MODEL": "deepseek-chat",
        "MEMORY_RERANK_CANDIDATES": "100",
        "SQLITE_PATH": "/data/memory.sqlite3",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("PROFILE_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    os.environ.setdefault("PROFILE_LLM_BASE_URL", os.environ["OPENAI_BASE_URL"])
    os.environ.setdefault("MEMORY_RERANK_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))
    os.environ.setdefault("MEMORY_RERANK_BASE_URL", os.environ["DEEPSEEK_API_BASE"])
    for name in ("CHALLENGE_API_KEY", "PROFILE_LLM_API_KEY", "MEMORY_RERANK_API_KEY"):
        if not os.environ.get(name, "").strip() or os.environ[name] in {"change-me", "replace-me"}:
            raise ValueError(f"Configure {name} before starting the service")
    if not (os.environ.get("MEMORY_EMBEDDING_API_KEY") or os.environ.get("QIAN_WEN_API_KEY")):
        raise ValueError("Configure QIAN_WEN_API_KEY or MEMORY_EMBEDDING_API_KEY")
    if os.environ["CHALLENGE_PROFILE_MODEL"] != "gpt-4o-mini":
        raise ValueError("This Academic package requires CHALLENGE_PROFILE_MODEL=gpt-4o-mini")
    if os.environ["MEMORY_EMBEDDING_MODEL"] != "text-embedding-v4":
        raise ValueError("This Academic package requires MEMORY_EMBEDDING_MODEL=text-embedding-v4")
    if os.environ["MEMORY_RETRIEVAL_MODE"] != "hybrid":
        raise ValueError("This package requires hybrid retrieval")
    if os.environ["MEMORY_SOURCE_RECORD_INDEX"].lower() != "true":
        raise ValueError("Keep MEMORY_SOURCE_RECORD_INDEX=true to preserve and index long sources")
    for name in OPTIONAL_LLM_FEATURES:
        if os.environ.get(name, "false").lower() == "true":
            raise ValueError(f"{name} is outside this frozen Academic configuration")
        os.environ[name] = "false"
    for name in ("PROFILE_LLM_BASE_URL", "MEMORY_RERANK_BASE_URL", "MEMORY_EMBEDDING_BASE_URL"):
        value = os.environ.get(name)
        if value:
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
            ):
                raise ValueError(f"Invalid provider URL in {name}")


def bounded_profile(messages):
    """Bound each profile prompt and retain indexes into the full source-chunk list."""
    from memory_service.structured_profile import ProfileValidationError, StructuredProfile, extract_structured_profile

    facts, start = [], 0
    while start < len(messages):
        end, size = start, 0
        while end < len(messages):
            next_size = len(json.dumps(messages[end], ensure_ascii=False).encode("utf-8"))
            if end > start and size + next_size > MAX_UNIT_BYTES:
                break
            size += next_size
            end += 1
        try:
            batch = extract_structured_profile(messages[start:end])
        except ProfileValidationError:
            logging.getLogger(__name__).warning("Profile validation failed; retaining this batch's source messages")
        else:
            facts.extend({**fact, "source_index": fact["source_index"] + start} for fact in batch.facts)
        start = end
    return StructuredProfile(facts)


def create_submission_app(storage=None, profile_extractor=bounded_profile):
    configure()
    from fastapi.responses import JSONResponse

    from memory_service.app import AddRequest, SearchRequest
    from memory_service.chunking import SOURCE_PREFIX
    from memory_service.local_server import create_local_app
    from memory_service.record_index import split_documents

    app = create_local_app(storage=storage, profile_extractor=profile_extractor)

    @app.middleware("http")
    async def validate_supported_units(request, call_next):
        if request.method == "POST" and request.url.path in {"/add", "/search"}:
            try:
                payload = await request.json()
                if request.url.path == "/add":
                    parsed = AddRequest.model_validate(payload)
                    chunks, _ = split_documents([m.model_dump(exclude_none=True) for m in parsed.messages])
                    texts = [SOURCE_PREFIX.sub("", item["content"], count=1) for item in chunks]
                else:
                    texts = [SearchRequest.model_validate(payload).query]
            except (ValueError, TypeError, KeyError):
                # Let the normal FastAPI schema handler explain malformed payloads.
                return await call_next(request)
            if any(len(text.encode("utf-8")) > MAX_UNIT_BYTES for text in texts):
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": {
                            "reason": "unsupported_source_unit",
                            "message": "An indivisible source unit or query exceeds 24000 UTF-8 bytes; no text was truncated or stored.",
                        }
                    },
                )
        return await call_next(request)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    configure()
    if args.check_config:
        print(
            json.dumps(
                {
                    name: os.environ[name]
                    for name in (
                        "CHALLENGE_PROFILE_MODEL",
                        "MEMORY_RERANK_MODEL",
                        "MEMORY_EMBEDDING_MODEL",
                        "MEMORY_SOURCE_RECORD_INDEX",
                    )
                }
            )
        )
        return
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_submission_app(), host="0.0.0.0", port=8000, workers=1, access_log=False)


if __name__ == "__main__":
    main()
