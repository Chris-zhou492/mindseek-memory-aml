import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from memory_service.errors import RequestConflictError


class ChallengeStorage:
    def __init__(self, *, host: str, port: int, dbname: str, user: str, password: str, schema: str):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
            raise ValueError("Invalid challenge schema name")
        self.schema = schema
        self.requests_table = f"{schema}.add_requests"
        self.messages_table = f"{schema}.messages"
        self.profiles_table = f"{schema}.profile_items"
        self.pool = ConnectionPool(
            conninfo=make_conninfo(host=host, port=port, dbname=dbname, user=user, password=password),
            kwargs={"row_factory": dict_row},
            min_size=1,
            max_size=8,
            open=True,
        )
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.requests_table} (
                        request_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        payload_hash TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.messages_table} (
                        position BIGSERIAL PRIMARY KEY,
                        id TEXT NOT NULL UNIQUE,
                        request_id TEXT NOT NULL REFERENCES {self.requests_table}(request_id),
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        message_index INT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp_ms BIGINT,
                        text_lexeme TSVECTOR NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (request_id, message_index)
                    )
                    """
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_challenge_user_position "
                    f"ON {self.messages_table} (user_id, position DESC)"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_challenge_lexeme "
                    f"ON {self.messages_table} USING GIN (text_lexeme)"
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.profiles_table} (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        source_request_id TEXT NOT NULL REFERENCES {self.requests_table}(request_id),
                        section TEXT NOT NULL,
                        content TEXT NOT NULL,
                        text_lexeme TSVECTOR NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_challenge_profile_user "
                    f"ON {self.profiles_table} (user_id)"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_challenge_profile_lexeme "
                    f"ON {self.profiles_table} USING GIN (text_lexeme)"
                )

    def close(self) -> None:
        self.pool.close()

    def health(self) -> None:
        with self.pool.connection() as conn:
            conn.execute("SELECT 1")

    @staticmethod
    def _payload_hash(*, user_id: str, session_id: str, messages: list[dict[str, Any]]) -> str:
        canonical = json.dumps(
            {"user_id": user_id, "session_id": session_id, "messages": messages},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def is_duplicate(self, *, request_id: str, user_id: str, session_id: str, messages: list[dict[str, Any]]) -> bool:
        payload_hash = self._payload_hash(user_id=user_id, session_id=session_id, messages=messages)
        with self.pool.connection() as conn:
            row = conn.execute(
                f"SELECT payload_hash FROM {self.requests_table} WHERE request_id = %s", (request_id,)
            ).fetchone()
        if row is None:
            return False
        if row["payload_hash"] != payload_hash:
            raise RequestConflictError("request_id was already used with a different payload")
        return True

    def add(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        profile: dict[str, list[str]],
    ) -> None:
        payload_hash = self._payload_hash(user_id=user_id, session_id=session_id, messages=messages)
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.requests_table} (request_id, user_id, session_id, payload_hash)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (request_id) DO NOTHING
                    RETURNING request_id
                    """,
                    (request_id, user_id, session_id, payload_hash),
                )
                if cur.fetchone() is None:
                    cur.execute(
                        f"SELECT payload_hash FROM {self.requests_table} WHERE request_id = %s",
                        (request_id,),
                    )
                    prior = cur.fetchone()
                    if prior is None or prior["payload_hash"] != payload_hash:
                        raise RequestConflictError("request_id was already used with a different payload")
                    return

                for index, message in enumerate(messages):
                    memory_id = "mem_" + hashlib.sha256(f"{request_id}:{index}".encode("utf-8")).hexdigest()[:24]
                    cur.execute(
                        f"""
                        INSERT INTO {self.messages_table}
                            (id, request_id, user_id, session_id, message_index, role,
                             content, timestamp_ms, text_lexeme)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))
                        """,
                        (
                            memory_id,
                            request_id,
                            user_id,
                            session_id,
                            index,
                            message["role"],
                            message["content"],
                            message.get("timestamp"),
                            message["content"],
                        ),
                    )

                for section, values in profile.items():
                    for value in values:
                        memory_id = "profile_" + hashlib.sha256(
                            f"{user_id}:{section}:{value.casefold()}".encode("utf-8")
                        ).hexdigest()[:24]
                        cur.execute(
                            f"""
                            INSERT INTO {self.profiles_table}
                                (id, user_id, source_request_id, section, content, text_lexeme)
                            VALUES (%s, %s, %s, %s, %s, to_tsvector('simple', %s))
                            ON CONFLICT (id) DO NOTHING
                            """,
                            (memory_id, user_id, request_id, section, value, value),
                        )

    def search(self, *, user_id: str, query: str, options: list[str] | None, top_k: int) -> list[dict[str, Any]]:
        retrieval_query = query
        if options:
            retrieval_query += "\n" + "\n".join(options)
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, role, NULL::text AS section, content, timestamp_ms, created_at,
                           ts_rank_cd(text_lexeme, websearch_to_tsquery('simple', %s)) AS score
                    FROM {self.messages_table} WHERE user_id = %s
                    UNION ALL
                    SELECT id, NULL::text AS role, section, content, NULL::bigint AS timestamp_ms, created_at,
                           ts_rank_cd(text_lexeme, websearch_to_tsquery('simple', %s)) AS score
                    FROM {self.profiles_table} WHERE user_id = %s
                    ORDER BY score DESC, created_at DESC
                    LIMIT %s
                    """,
                    (retrieval_query, user_id, retrieval_query, user_id, top_k),
                )
                rows = cur.fetchall()
        results = []
        for row in rows:
            source_time = row["created_at"]
            if row["timestamp_ms"] is not None:
                try:
                    source_time = datetime.fromtimestamp(row["timestamp_ms"] / 1000, tz=timezone.utc)
                except (OverflowError, OSError, ValueError):
                    pass
            source_time = source_time.astimezone(timezone.utc)
            timestamp = source_time.isoformat().replace("+00:00", "Z")
            results.append(
                {
                    "id": row["id"],
                    "content": (
                        f"[profile:{row['section']} | {timestamp}] {row['content']}"
                        if row["section"] else f"[{row['role']} | {timestamp}] {row['content']}"
                    ),
                    "score": float(row["score"] or 0.0),
                    "created_at": timestamp,
                }
            )
        return results
