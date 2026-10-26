"""Qwen embeddings for the local memory service."""

import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from memory_service.chunking import SOURCE_PREFIX


class QwenEmbedding:
    def __init__(self):
        self.api_key = os.environ.get("MEMORY_EMBEDDING_API_KEY") or os.environ.get("QIAN_WEN_API_KEY")
        if not self.api_key:
            raise RuntimeError("QIAN_WEN_API_KEY or MEMORY_EMBEDDING_API_KEY is required for hybrid retrieval")
        self.base_url = os.environ.get(
            "MEMORY_EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
        self.model = os.environ.get("MEMORY_EMBEDDING_MODEL", "text-embedding-v4")
        self.dimensions = int(os.environ.get("MEMORY_EMBEDDING_DIMENSIONS", "1024"))
        self.model_id = f"{self.base_url}/{self.model}:{self.dimensions}:source-boundary-input-v2"
        self.query_url = os.environ.get("MEMORY_EMBEDDING_QUERY_URL") or (
            self.base_url.removesuffix("/compatible-mode/v1")
            + "/api/v1/services/embeddings/text-embedding/text-embedding"
        )

    def embed_query(self, query: str) -> list[float]:
        """Use DashScope's retrieval-specific query mode (not document mode)."""
        payload = {
            "model": self.model,
            "input": {"texts": [query]},
            "parameters": {
                "dimension": self.dimensions,
                "text_type": "query",
                "instruct": "Given a question about a person or past conversations, retrieve the relevant "
                "facts, personal attributes, or event details that answer the question.",
            },
        }
        for attempt in range(3):
            request = Request(
                self.query_url,
                data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            try:
                with urlopen(request, timeout=60) as response:
                    items = json.load(response)["output"]["embeddings"]
                if len(items) != 1 or items[0]["text_index"] != 0:
                    raise ValueError("Query embedding count does not match input")
                vector = items[0]["embedding"]
                if len(vector) != self.dimensions or not all(math.isfinite(v) for v in vector) or not any(vector):
                    raise ValueError("Invalid query vector")
                return vector
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f"Query embedding provider returned HTTP {error.code}") from None
            except (URLError, TimeoutError, ConnectionError):
                if attempt == 2:
                    raise RuntimeError("Query embedding provider is unavailable") from None
            time.sleep(2**attempt)
        raise RuntimeError("Query embedding provider is unavailable")

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        prepared = [SOURCE_PREFIX.sub("", text, count=1) for text in texts]
        for start in range(0, len(texts), 10):
            batch = prepared[start : start + 10]
            payload = {"model": self.model, "input": batch, "dimensions": self.dimensions}
            for attempt in range(3):
                request = Request(
                    self.base_url + "/embeddings",
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                )
                try:
                    with urlopen(request, timeout=60) as response:
                        data = json.load(response)["data"]
                    break
                except HTTPError as error:
                    if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                        raise RuntimeError(f"Embedding provider returned HTTP {error.code}") from None
                except (URLError, TimeoutError, ConnectionError):
                    if attempt == 2:
                        raise RuntimeError("Embedding provider is unavailable") from None
                time.sleep(2**attempt)
            ordered = sorted(data, key=lambda item: item["index"])
            if [item["index"] for item in ordered] != list(range(len(batch))):
                raise ValueError("Embedding response indices do not match input")
            for item in ordered:
                vector = item["embedding"]
                if len(vector) != self.dimensions or not all(math.isfinite(v) for v in vector) or not any(vector):
                    raise ValueError("Embedding provider returned an invalid vector")
                vectors.append(vector)
        return vectors
