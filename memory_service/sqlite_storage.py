"""SQLite storage for the local Add/Search service."""

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory_service.errors import RequestConflictError


class SQLiteStorage:
    def __init__(
        self,
        path: str | Path,
        *,
        embedder: Callable | None = None,
        embedding_model: str = "",
        reranker: Callable | None = None,
        rerank_candidates: int = 60,
        evidence_coverage: bool = False,
        context_window: int = 1,
        evidence_char_budget: int = 24000,
        query_planner: Callable | None = None,
        evidence_selector: Callable | None = None,
        source_record_index: bool = False,
        record_calculator: Callable | None = None,
        state_reconciler: Callable | None = None,
        hybrid_fusion: str = "legacy",
        event_memory: Any = None,
        evidence_dedup: bool = False,
        directed_evidence: Callable | None = None,
        game_log_evidence: Callable | None = None,
    ):
        from memory_service.hybrid_recall import MODES

        if hybrid_fusion not in MODES:
            raise ValueError("hybrid_fusion must be legacy, vector or bm25_rrf")
        self.hybrid_fusion = hybrid_fusion
        if embedder is not None and not embedding_model:
            raise ValueError("embedding_model must identify the vector model")
        if source_record_index and embedder is None:
            raise ValueError("Source record index requires hybrid retrieval")
        if record_calculator is not None and not source_record_index:
            raise ValueError("Record calculations require source_record_index")
        if state_reconciler is not None and not source_record_index:
            raise ValueError("State evidence requires source_record_index")
        if event_memory is not None and not source_record_index:
            raise ValueError("Write-time event memory requires source_record_index")
        self.event_memory = event_memory
        self.evidence_dedup = evidence_dedup
        self.embedder = embedder
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.rerank_candidates = max(1, rerank_candidates)
        if not 0 <= context_window <= 3 or evidence_char_budget < 1000:
            raise ValueError("context_window must be 0..3 and evidence_char_budget must be at least 1000")
        self.evidence_coverage = evidence_coverage
        self.context_window = context_window
        self.evidence_char_budget = evidence_char_budget
        self.query_planner = query_planner
        self.evidence_selector = evidence_selector
        self.source_record_index = source_record_index
        self.record_calculator = record_calculator
        self.state_reconciler = state_reconciler
        if directed_evidence is not None and not source_record_index:
            raise ValueError("Directed evidence requires source_record_index")
        self.directed_evidence = directed_evidence
        if game_log_evidence is not None and not source_record_index:
            raise ValueError("Game log evidence requires source_record_index")
        self.game_log_evidence = game_log_evidence
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS add_requests (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    position INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    request_id TEXT NOT NULL REFERENCES add_requests(request_id),
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp_ms INTEGER,
                    created_at TEXT NOT NULL,
                    UNIQUE(request_id, message_index)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_user_position
                    ON messages(user_id, position DESC);
                CREATE TABLE IF NOT EXISTS profile_items (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_request_id TEXT NOT NULL REFERENCES add_requests(request_id),
                    section TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_profile_user ON profile_items(user_id);
                CREATE TABLE IF NOT EXISTS profile_facts (
                    memory_id TEXT NOT NULL REFERENCES profile_items(id),
                    entity TEXT NOT NULL,
                    attribute TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source_message_id TEXT NOT NULL REFERENCES messages(id),
                    source_ref TEXT,
                    source_date TEXT,
                    source_timestamp_ms INTEGER,
                    quote TEXT NOT NULL,
                    PRIMARY KEY (memory_id, source_message_id)
                );
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    memory_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    PRIMARY KEY (memory_id, model)
                );
                CREATE INDEX IF NOT EXISTS idx_vectors_user_model ON memory_vectors(user_id, model);
                CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
                    content, memory_id UNINDEXED, tokenize='unicode61'
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS profile_fts USING fts5(
                    content, memory_id UNINDEXED, tokenize='unicode61'
                );
                """
            )
            from memory_service.record_index import SCHEMA

            conn.executescript(SCHEMA)
            if self.event_memory is not None:
                from memory_service.event_memory import SCHEMA as EVENT_SCHEMA

                conn.executescript(EVENT_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def close(self) -> None:
        pass

    def health(self) -> None:
        with self._connect() as conn:
            conn.execute("SELECT 1").fetchone()

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
        with self._connect() as conn:
            row = conn.execute("SELECT payload_hash FROM add_requests WHERE request_id = ?", (request_id,)).fetchone()
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
        if self.is_duplicate(request_id=request_id, user_id=user_id, session_id=session_id, messages=messages):
            return
        original_messages = messages
        documents = None
        if self.source_record_index:
            from memory_service.record_index import split_documents

            messages, documents = split_documents(original_messages)
        event_plan = self.event_memory.prepare(self, user_id, session_id, request_id, documents) if self.event_memory else None
        vector_items = [
            ("mem_" + hashlib.sha256(f"{request_id}:{i}".encode()).hexdigest()[:24], message["content"])
            for i, message in enumerate(messages)
        ]
        vector_items.extend(
            (self._profile_id(user_id, section, value), value)
            for section, values in profile.items()
            for value in values
        )
        if event_plan is not None:
            vector_items.extend(event_plan["vectors"])
        # Provider failures happen before the transaction: an Add never commits
        # messages without their vectors when hybrid retrieval is enabled.
        vectors = self._embed([text for _, text in vector_items]) if self.embedder else []
        payload_hash = self._payload_hash(user_id=user_id, session_id=session_id, messages=original_messages)
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT payload_hash FROM add_requests WHERE request_id = ?", (request_id,)).fetchone()
            if row is not None:
                if row["payload_hash"] != payload_hash:
                    raise RequestConflictError("request_id was already used with a different payload")
                return

            if event_plan is not None:
                self.event_memory.check_snapshot(conn, user_id, session_id, event_plan)
            conn.execute(
                "INSERT INTO add_requests VALUES (?, ?, ?, ?, ?)",
                (request_id, user_id, session_id, payload_hash, created_at),
            )
            for index, message in enumerate(messages):
                memory_id = "mem_" + hashlib.sha256(f"{request_id}:{index}".encode()).hexdigest()[:24]
                conn.execute(
                    """INSERT INTO messages
                    (id, request_id, user_id, session_id, message_index, role, content, timestamp_ms, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        memory_id,
                        request_id,
                        user_id,
                        session_id,
                        index,
                        message["role"],
                        message["content"],
                        message.get("timestamp"),
                        created_at,
                    ),
                )
                conn.execute(
                    "INSERT INTO message_fts (content, memory_id) VALUES (?, ?)",
                    (message["content"], memory_id),
                )

            if documents is not None:
                from memory_service.record_index import insert_documents

                sources = [
                    (
                        original,
                        [
                            (
                                "mem_" + hashlib.sha256(f"{request_id}:{index}".encode()).hexdigest()[:24],
                                start,
                                end,
                                a,
                                b,
                            )
                            for index, start, end, a, b in spans
                        ],
                    )
                    for original, spans in documents
                ]
                insert_documents(conn, user_id, session_id, sources, created_at)
            if event_plan is not None:
                self.event_memory.persist(conn, user_id, session_id, request_id, event_plan, created_at)

            for section, values in profile.items():
                for value in values:
                    memory_id = self._profile_id(user_id, section, value)
                    inserted = conn.execute(
                        """INSERT OR IGNORE INTO profile_items
                        (id, user_id, source_request_id, section, content, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                        (memory_id, user_id, request_id, section, value, created_at),
                    )
                    if inserted.rowcount:
                        conn.execute(
                            "INSERT INTO profile_fts (content, memory_id) VALUES (?, ?)",
                            (value, memory_id),
                        )
            for (memory_id, _), vector in zip(vector_items, vectors):
                if event_plan is not None and memory_id.startswith("event_"):
                    conn.execute("INSERT OR REPLACE INTO memory_vectors VALUES (?,?,?,?)",
                                 (memory_id, user_id, self.embedding_model, json.dumps(vector)))
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO memory_vectors VALUES (?, ?, ?, ?)",
                    (memory_id, user_id, self.embedding_model, json.dumps(vector)),
                )
            for fact in getattr(profile, "facts", []):
                source_index = fact["source_index"]
                if (
                    type(source_index) is not int
                    or not 0 <= source_index < len(messages)
                    or messages[source_index]["role"] != "user"
                    or not fact["quote"]
                    or fact["quote"] not in messages[source_index]["content"]
                ):
                    raise ValueError("Structured fact has an invalid source")
                source_id = "mem_" + hashlib.sha256(f"{request_id}:{source_index}".encode()).hexdigest()[:24]
                conn.execute(
                    "INSERT OR IGNORE INTO profile_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        self._profile_id(user_id, fact["section"], fact["content"]),
                        fact["entity"],
                        fact["attribute"],
                        fact["value"],
                        source_id,
                        fact["source_ref"],
                        fact["source_date"],
                        fact["source_timestamp_ms"],
                        fact["quote"],
                    ),
                )

    @staticmethod
    def _profile_id(user_id: str, section: str, value: str) -> str:
        return "profile_" + hashlib.sha256(f"{user_id}:{section}:{value.casefold()}".encode()).hexdigest()[:24]

    def prepare_profile_messages(self, messages):
        if not self.source_record_index:
            return messages
        from memory_service.record_index import split_documents

        return split_documents(messages)[0]

    def get_event_memory(self, *, user_id, event_id):
        if self.event_memory is None:
            return None
        with self._connect() as conn:
            event = conn.execute("SELECT * FROM event_memories WHERE id=? AND user_id=?", (event_id, user_id)).fetchone()
            if event is None:
                return None
            statements = [dict(s) for s in conn.execute(
                "SELECT s.*,d.role,d.timestamp_ms FROM event_statements s JOIN source_documents d ON d.id=s.document_id "
                "WHERE s.event_id=? AND s.user_id=? AND d.user_id=? ORDER BY s.source_order,s.quote_start,s.id",
                (event_id, user_id, user_id))]
            return {**dict(event), "statements": statements,
                    "scope": "Observed source statements and model interpretations, not an authoritative latest truth"}

    def backfill_source_records(self, *, user_id, session_id, originals):
        """Validate exact historical chunks before atomically replacing only the sidecar."""
        from memory_service.record_index import align_documents, insert_documents

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT * FROM messages WHERE user_id=? AND session_id=? ORDER BY position", (user_id, session_id)
            ).fetchall()
            if not rows or not originals:
                raise ValueError("Cannot backfill an empty or missing source history")
            documents = align_documents(originals, rows)
            conn.execute("DELETE FROM source_documents WHERE user_id=? AND session_id=?", (user_id, session_id))
            return insert_documents(conn, user_id, session_id, documents, rows[0]["created_at"])

    def _fit_source_records(self, evidence, top_k):
        selected, remaining = [], self.evidence_char_budget
        covered = set()
        for item in evidence:
            if len(selected) >= top_k:
                break
            keys = set()
            if self.evidence_dedup:
                from memory_service.evidence_dedup import project_event

                item, keys = project_event(item, covered)
                if item is None:
                    continue
            if len(item["content"]) <= remaining:
                selected.append(item)
                remaining -= len(item["content"])
                covered.update(keys)
        return selected

    def get_source_record(self, *, user_id, record_id, include_parent=False):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT r.*,d.content,d.role,d.session_id,d.timestamp_ms FROM source_records r JOIN source_documents d ON d.id=r.document_id WHERE r.id=? AND r.user_id=? AND d.user_id=?",
                (record_id, user_id, user_id),
            ).fetchone()
            if row is None:
                return None
            result = {
                "record_id": row["id"],
                "document_id": row["document_id"],
                "kind": row["kind"],
                "start": row["start"],
                "end": row["end"],
                "content": row["content"][row["start"] : row["end"]],
                "role": row["role"],
                "session_id": row["session_id"],
                "timestamp_ms": row["timestamp_ms"],
            }
            if include_parent:
                result["parent_content"] = row["content"]
            return result

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embedder(texts)
        if len(vectors) != len(texts):
            raise ValueError("Embedding count does not match input")
        dimensions = len(vectors[0]) if vectors else 0
        for vector in vectors:
            if (
                not dimensions
                or len(vector) != dimensions
                or not all(math.isfinite(v) for v in vector)
                or not any(vector)
            ):
                raise ValueError("Invalid memory embedding")
        return vectors

    def reindex_vectors(self) -> int:
        """Explicitly backfill the configured vector model for existing memories."""
        if self.embedder is None:
            raise RuntimeError("Hybrid embeddings must be configured for vector backfill")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id,user_id,content FROM messages UNION ALL SELECT id,user_id,content FROM profile_items"
            ).fetchall()
            if self.event_memory is not None:
                from memory_service.event_memory import index_texts

                rows.extend({"id": ident, "user_id": owner, "content": text} for ident, owner, text in index_texts(conn))
            existing = {
                r[0]
                for r in conn.execute("SELECT memory_id FROM memory_vectors WHERE model=?", (self.embedding_model,))
            }
        missing = [row for row in rows if row["id"] not in existing]
        for start in range(0, len(missing), 50):
            batch = missing[start : start + 50]
            vectors = self._embed([row["content"] for row in batch])
            with self._connect() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO memory_vectors VALUES (?, ?, ?, ?)",
                    [
                        (row["id"], row["user_id"], self.embedding_model, json.dumps(vector))
                        for row, vector in zip(batch, vectors, strict=True)
                    ],
                )
        return len(missing)

    @staticmethod
    def _match_query(query: str) -> str:
        terms = dict.fromkeys(term.casefold() for term in re.findall(r"\w+", query, flags=re.UNICODE))
        return " OR ".join('"' + term.replace('"', '""') + '"' for term in list(terms)[:32])

    @staticmethod
    def _result(row: sqlite3.Row, score: float) -> dict[str, Any]:
        timestamp = None
        if row["timestamp_ms"] is not None:
            try:
                timestamp = (
                    datetime.fromtimestamp(row["timestamp_ms"] / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
                )
            except (OverflowError, OSError, ValueError):
                pass
        label = f"profile:{row['section']}" if row["section"] else row["role"]
        return {
            "id": row["id"],
            "content": f"[{label} | source_time: {timestamp or 'unknown'}] {row['content']}",
            "score": score,
            "created_at": row["created_at"],
        }

    def _format_results(self, ranked):
        results = [self._result(row, score) for row, score in ranked]
        profiles = {item["id"]: item for item in results if item["id"].startswith("profile_")}
        keys = list(profiles)
        with self._connect() as conn:
            for start in range(0, len(keys), 200):
                batch = keys[start : start + 200]
                placeholders = ",".join("?" for _ in batch)
                facts = conn.execute(
                    f"SELECT * FROM profile_facts WHERE memory_id IN ({placeholders}) ORDER BY source_message_id",
                    batch,
                ).fetchall()
                for fact in facts:
                    item = profiles[fact["memory_id"]]
                    item.update(entity=fact["entity"], attribute=fact["attribute"], value=fact["value"])
                    item.setdefault("sources", []).append(
                        {
                            "source_message_id": fact["source_message_id"],
                            "source_ref": fact["source_ref"],
                            "source_date": fact["source_date"],
                            "source_timestamp_ms": fact["source_timestamp_ms"],
                            "quote": fact["quote"],
                        }
                    )
        for item in profiles.values():
            for source in item.get("sources", [])[:3]:
                item["content"] += (
                    f"\n[source_date: {source['source_date'] or 'unknown'}; "
                    f"source_ref: {source['source_ref'] or source['source_message_id']}] "
                    f"Evidence quote: {source['quote']}"
                )
        return results

    def search(self, *, user_id: str, query: str, options: list[str] | None, top_k: int) -> list[dict[str, Any]]:
        if self.embedder is not None:
            candidates = self._hybrid_search(user_id=user_id, query=query, options=options, top_k=top_k)
            if self.query_planner is not None and candidates:
                queries = self.query_planner(query, candidates[:10], options)
                if not isinstance(queries, list) or len(queries) > 2:
                    raise ValueError("Query planner must return at most two search queries")
                merged = {item["id"]: dict(item) for item in candidates}
                fusion = {item["id"]: 1 / (60 + i) for i, item in enumerate(candidates, 1)}
                seen_queries = {query.strip().casefold()}
                for followup in queries:
                    if not isinstance(followup, str) or not followup.strip() or len(followup) > 1000:
                        raise ValueError("Invalid follow-up search query")
                    if followup.strip().casefold() in seen_queries:
                        continue
                    seen_queries.add(followup.strip().casefold())
                    hits = self._hybrid_search(user_id=user_id, query=followup, options=None, top_k=top_k)
                    for i, item in enumerate(hits, 1):
                        merged.setdefault(item["id"], dict(item))
                        fusion[item["id"]] = fusion.get(item["id"], 0) + 1 / (60 + i)
                candidates = sorted(merged.values(), key=lambda item: (-fusion[item["id"]], item["id"]))
            candidates = self._candidate_quota(candidates, max(top_k, self.rerank_candidates))
            if self.reranker is not None:
                candidates = self.reranker(query, candidates, options)
            if self.evidence_selector is not None and candidates:
                with self._connect() as conn:
                    positions = {
                        row["id"]: row["position"]
                        for row in conn.execute("SELECT id,position FROM messages WHERE user_id=?", (user_id,))
                    }
                candidates = [{**item, "source_order": positions.get(item["id"])} for item in candidates]
                groups = self.evidence_selector(query, candidates, options, top_k, self.evidence_char_budget)
                if not isinstance(groups, list) or len(groups) > top_k:
                    raise ValueError("Evidence selector exceeded top_k")
                evidence = self._pack_evidence_groups(user_id, candidates, groups)
                return self._calculate(user_id, query, options, evidence, top_k)
            if self.source_record_index:
                evidence = self._fit_source_records(candidates, top_k)
                return self._calculate(user_id, query, options, evidence, top_k)
            if self.evidence_coverage:
                return self._cover_evidence(user_id, candidates, top_k)
            return candidates[:top_k]
        retrieval_query = query + ("\n" + "\n".join(options) if options else "")
        match_query = self._match_query(retrieval_query)
        candidates: dict[str, tuple[sqlite3.Row, float]] = {}
        with self._connect() as conn:
            if match_query:
                for fts, table, rank in (
                    ("message_fts", "messages", "bm25(message_fts)"),
                    ("profile_fts", "profile_items", "bm25(profile_fts)"),
                ):
                    rows = conn.execute(
                        f"""SELECT item.id, item.content, item.created_at,
                        {"item.role" if table == "messages" else "NULL"} AS role,
                        {"NULL" if table == "messages" else "item.section"} AS section,
                        {"item.timestamp_ms" if table == "messages" else "NULL"} AS timestamp_ms,
                        {rank} AS rank
                        FROM {fts} JOIN {table} AS item ON item.id = {fts}.memory_id
                        WHERE {fts} MATCH ? AND item.user_id = ?
                        ORDER BY rank ASC LIMIT ?""",
                        (match_query, user_id, top_k),
                    ).fetchall()
                    for row in rows:
                        candidates[row["id"]] = (row, max(0.0, -float(row["rank"])))

            if len(candidates) < top_k:
                for table in ("messages", "profile_items"):
                    rows = conn.execute(
                        f"""SELECT id, content, created_at,
                        {"role" if table == "messages" else "NULL"} AS role,
                        {"NULL" if table == "messages" else "section"} AS section,
                        {"timestamp_ms" if table == "messages" else "NULL"} AS timestamp_ms
                        FROM {table} WHERE user_id = ? ORDER BY created_at DESC LIMIT ?""",
                        (user_id, top_k),
                    ).fetchall()
                    for row in rows:
                        candidates.setdefault(row["id"], (row, 0.0))

        ranked = sorted(candidates.values(), key=lambda pair: (pair[1], pair[0]["created_at"]), reverse=True)
        evidence = self._format_results(ranked[: max(top_k, self.rerank_candidates)])
        return self._cover_evidence(user_id, evidence, top_k) if self.evidence_coverage else evidence[:top_k]

    def _calculate(self, user_id, query, options, evidence, top_k):
        if self.record_calculator is not None:
            evidence = self.record_calculator(self, user_id, query, options, evidence, top_k)
        if self.directed_evidence is not None:
            evidence = self.directed_evidence(self, user_id, query, options, evidence, top_k)
        if self.game_log_evidence is not None:
            evidence = self.game_log_evidence(self, user_id, query, options, evidence, top_k)
        if self.state_reconciler is not None:
            evidence = self.state_reconciler(self, user_id, query, options, evidence, top_k)
        return evidence

    def get_state_run(self, *, user_id, run_id):
        from memory_service.state_evidence import get_run

        return get_run(self, user_id, run_id)

    def get_calculation_run(self, *, user_id, run_id):
        from memory_service.record_calculator import get_run

        return get_run(self, user_id, run_id)

    def _hybrid_search(self, *, user_id: str, query: str, options: list[str] | None, top_k: int) -> list[dict]:
        if self.source_record_index:
            from memory_service.record_index import candidates

            return candidates(self, user_id, query, options, top_k)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id,user_id,content,created_at,role,NULL AS section,timestamp_ms FROM messages WHERE user_id=? "
                "UNION ALL SELECT id,user_id,content,created_at,NULL AS role,section,NULL AS timestamp_ms "
                "FROM profile_items WHERE user_id=?",
                (user_id, user_id),
            ).fetchall()
            if not rows:
                return []
            vectors = {
                row[0]: json.loads(row[1])
                for row in conn.execute(
                    "SELECT memory_id,vector FROM memory_vectors WHERE user_id=? AND model=?",
                    (user_id, self.embedding_model),
                )
            }
            if any(row["id"] not in vectors for row in rows):
                raise RuntimeError("Memory vectors are missing for the configured model; run --reindex-vectors")
            lexical = {}
            match = self._match_query(query + ("\n" + "\n".join(options) if options else ""))
            pool = max(50, top_k * 3)
            if match:
                for fts, table in (("message_fts", "messages"), ("profile_fts", "profile_items")):
                    hits = conn.execute(
                        f"SELECT item.id FROM {fts} JOIN {table} item ON item.id={fts}.memory_id "
                        f"WHERE {fts} MATCH ? AND item.user_id=? ORDER BY bm25({fts}) LIMIT ?",
                        (match, user_id, pool),
                    ).fetchall()
                    for rank, hit in enumerate(hits, 1):
                        lexical[hit[0]] = 61 / (60 + rank)
        query_vector = (
            self.embedder.embed_query(query) if hasattr(self.embedder, "embed_query") else self._embed([query])[0]
        )
        query_norm = math.sqrt(sum(v * v for v in query_vector))
        semantic = {}
        for row in rows:
            vector = vectors[row["id"]]
            if len(vector) != len(query_vector):
                raise ValueError("Stored vector dimensions differ from the query; reindex required")
            semantic[row["id"]] = sum(a * b for a, b in zip(vector, query_vector)) / (
                math.sqrt(sum(v * v for v in vector)) * query_norm
            )
        if self.hybrid_fusion != "legacy":
            from memory_service.hybrid_recall import fuse
            from memory_service.record_index import local_bm25

            evidence = self._format_results([(row, semantic[row["id"]]) for row in rows])
            bm25 = local_bm25([row["content"] for row in rows], query + "\n" + "\n".join(options or []))
            return fuse(evidence, [semantic[row["id"]] for row in rows], bm25,
                        max(top_k, self.rerank_candidates), self.hybrid_fusion)
        # Semantic recall scans all memories belonging to this user, before any
        # keyword cutoff. A small rank-based boost retains exact keyword matches
        # without comparing raw BM25 scores from different tables.
        semantic_ids = sorted(semantic, key=semantic.get, reverse=True)[:pool]
        if self.evidence_coverage:
            ordered = sorted(rows, key=lambda row: semantic[row["id"]], reverse=True)
            semantic_ids = [row["id"] for row in ordered if row["section"] is None][:pool]
            semantic_ids += [row["id"] for row in ordered if row["section"] is not None][:pool]
        candidates = set(semantic_ids) | lexical.keys()
        scores = {key: semantic[key] + 0.05 * lexical.get(key, 0) for key in candidates}
        ranked = sorted(
            (row for row in rows if row["id"] in candidates),
            key=lambda row: (scores[row["id"]], row["id"]),
            reverse=True,
        )
        size = max(top_k, self.rerank_candidates)
        if self.evidence_coverage:
            raw = [row for row in ranked if row["section"] is None]
            profiles = [row for row in ranked if row["section"] is not None]
            # Reserve separate candidate slots before the ranker, not only at output.
            chosen = raw[: size - size // 4] + profiles[: size // 4]
            chosen_ids = {row["id"] for row in chosen}
            chosen += [row for row in ranked if row["id"] not in chosen_ids][: size - len(chosen)]
            ranked = chosen
        return self._format_results([(row, scores[row["id"]]) for row in ranked[:size]])

    def _candidate_quota(self, candidates, size):
        if not self.evidence_coverage:
            return candidates[:size]
        raw = [item for item in candidates if not item["id"].startswith("profile_")]
        profiles = [item for item in candidates if item["id"].startswith("profile_")]
        selected = raw[: size - size // 4] + profiles[: size // 4]
        selected_ids = {item["id"] for item in selected}
        selected += [item for item in candidates if item["id"] not in selected_ids][: size - len(selected)]
        return selected

    def _pack_evidence_groups(self, user_id, candidates, groups):
        # Validate even injected selectors. Content is always reconstructed here.
        from memory_service.evidence_selector import DeepSeekEvidenceSelector

        DeepSeekEvidenceSelector.validate(
            json.dumps({"groups": groups}), candidates, len(groups), self.evidence_char_budget
        )
        selected_ids = {candidates[i]["id"] for group in groups for i in group["indices"]}
        anchors, extras = {}, {}
        with self._connect() as conn:
            for ident in selected_ids:
                row = conn.execute(
                    "SELECT *,NULL AS section FROM messages WHERE user_id=? AND id=?", (user_id, ident)
                ).fetchone()
                if row is None:
                    owner = conn.execute(
                        "SELECT id FROM profile_items WHERE user_id=? AND id=?", (user_id, ident)
                    ).fetchone()
                    if owner is None:
                        owner = conn.execute(
                            "SELECT id FROM source_records WHERE user_id=? AND id=?", (user_id, ident)
                        ).fetchone()
                    if owner is None:
                        raise ValueError("Evidence candidate does not belong to the requested user")
                    continue
                anchors[ident] = row
                extras[ident] = conn.execute(
                    "SELECT *,NULL AS section FROM messages WHERE user_id=? AND session_id=? AND position!=? "
                    "ORDER BY abs(position-?),position LIMIT ?",
                    (user_id, row["session_id"], row["position"], row["position"], 2 * self.context_window),
                ).fetchall()
        # Reserve all selected passages first. Optional context can use only the rest.
        remaining = self.evidence_char_budget - sum(
            len(candidates[i]["content"]) + len(candidates[i]["id"]) + 100 for group in groups for i in group["indices"]
        )
        used = set(selected_ids) | {
            ident
            for group in groups
            for i in group["indices"]
            for ident in candidates[i].get("context_message_ids", [])
        }
        output = []
        for group in groups:
            parts, source_ids = [], []
            for index in group["indices"]:
                item = candidates[index]
                order = f"; source_order: {anchors[item['id']]['position']}" if item["id"] in anchors else ""
                parts.append(f"[memory_id: {item['id']}{order}]\n{item['content']}")
                if item["id"] in anchors:
                    source_ids.append(item["id"])
                else:
                    source_ids.extend(item.get("context_message_ids", []))
                for row in sorted(extras.get(item["id"], []), key=lambda value: value["position"]):
                    if row["id"] in used:
                        continue
                    text = (
                        f"[memory_id: {row['id']}; source_order: {row['position']}]\n" + self._result(row, 0)["content"]
                    )
                    if len(text) + 2 > remaining:
                        continue
                    parts.append(text)
                    source_ids.append(row["id"])
                    used.add(row["id"])
                    remaining -= len(text) + 2
            first = candidates[group["indices"][0]]
            output.append(
                {
                    **first,
                    "content": "\n\n".join(parts),
                    "context_message_ids": source_ids,
                    "evidence_need": group["need"],
                    "group_member_ids": [candidates[i]["id"] for i in group["indices"]],
                    "budget_omitted_ids": [candidates[i]["id"] for i in group.get("omitted_indices", [])],
                    "record_sources": [
                        record for i in group["indices"] for record in candidates[i].get("record_sources", [])
                    ],
                }
            )
        if sum(len(item["content"]) for item in output) > self.evidence_char_budget:
            raise ValueError("Packed evidence exceeds source budget")
        return output

    def _cover_evidence(self, user_id, candidates, top_k):
        raw = [item for item in candidates if not item["id"].startswith("profile_")]
        profiles = [item for item in candidates if item["id"].startswith("profile_")]
        profile_limit = top_k // 4
        profiles = profiles[:profile_limit]
        # Provenance adds retrieval seeds; it never replaces the original source.
        seeds = list(raw)
        known = {item["id"] for item in seeds}
        with self._connect() as conn:
            for profile in profiles:
                for source in profile.get("sources", [])[:3]:
                    key = source["source_message_id"]
                    if key in known:
                        continue
                    row = conn.execute(
                        "SELECT *,NULL AS section FROM messages WHERE user_id=? AND id=?", (user_id, key)
                    ).fetchone()
                    if row is not None:
                        seeds.append(self._result(row, profile.get("score", 0)))
                        known.add(key)
            selected, used_ids = [], set()
            remaining = self.evidence_char_budget
            profile_budget = min(sum(len(p["content"]) for p in profiles), remaining // 4)
            raw_budget = remaining - profile_budget
            for seed in seeds:
                if len(selected) >= top_k - len(profiles):
                    break
                if seed["id"] in used_ids:
                    continue
                anchor = conn.execute(
                    "SELECT * FROM messages WHERE user_id=? AND id=?", (user_id, seed["id"])
                ).fetchone()
                if anchor is None:
                    continue
                args = (user_id, anchor["session_id"], anchor["position"], self.context_window)
                before = conn.execute(
                    "SELECT *,NULL AS section FROM messages WHERE user_id=? AND session_id=? "
                    "AND position<? ORDER BY position DESC LIMIT ?",
                    args,
                ).fetchall()
                after = conn.execute(
                    "SELECT *,NULL AS section FROM messages WHERE user_id=? AND session_id=? "
                    "AND position>? ORDER BY position LIMIT ?",
                    args,
                ).fetchall()
                anchor = conn.execute(
                    "SELECT *,NULL AS section FROM messages WHERE user_id=? AND id=?", (user_id, seed["id"])
                ).fetchone()
                neighbors = sorted([*before, anchor, *after], key=lambda row: row["position"])
                # Keep whole source messages. If the full window does not fit, try
                # the intact anchor alone; never silently truncate source text.
                options = [neighbors, [anchor]]
                for group in options:
                    group = [row for row in group if row["id"] not in used_ids]
                    content = "\n\n".join(
                        f"[message_id: {row['id']}; source_order: {row['position']}] " + self._result(row, 0)["content"]
                        for row in group
                    )
                    if not content or len(content) > raw_budget:
                        continue
                    selected.append({**seed, "content": content, "context_message_ids": [r["id"] for r in group]})
                    used_ids.update(row["id"] for row in group)
                    raw_budget -= len(content)
                    remaining -= len(content)
                    break
            for profile in profiles:
                if len(profile["content"]) <= remaining and len(selected) < top_k:
                    selected.append(profile)
                    remaining -= len(profile["content"])
        return selected
