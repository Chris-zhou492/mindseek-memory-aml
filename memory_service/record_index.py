"""Lossless original documents, record spans and links to native retrieval chunks."""

import hashlib
import json
import math
import re
from collections import Counter

from memory_service.chunking import SOURCE_PREFIX, TARGET_CHARS, chunk_spans
from memory_service.source_units import source_units

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_documents (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL,session_id TEXT NOT NULL,
 role TEXT NOT NULL,content TEXT NOT NULL,timestamp_ms INTEGER,created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_documents_user ON source_documents(user_id,session_id);
CREATE TABLE IF NOT EXISTS source_records (
 id TEXT PRIMARY KEY,document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
 user_id TEXT NOT NULL,kind TEXT NOT NULL,start INTEGER NOT NULL,end INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_records_user ON source_records(user_id);
CREATE TABLE IF NOT EXISTS source_spans (
 document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
 message_id TEXT NOT NULL REFERENCES messages(id),document_start INTEGER NOT NULL,document_end INTEGER NOT NULL,
 message_start INTEGER NOT NULL,message_end INTEGER NOT NULL,
 PRIMARY KEY(document_id,message_id)
);
CREATE TABLE IF NOT EXISTS record_chunks (
 record_id TEXT NOT NULL REFERENCES source_records(id) ON DELETE CASCADE,
 message_id TEXT NOT NULL REFERENCES messages(id),document_start INTEGER NOT NULL,document_end INTEGER NOT NULL,
 message_start INTEGER NOT NULL,message_end INTEGER NOT NULL,
 PRIMARY KEY(record_id,message_id)
);
CREATE INDEX IF NOT EXISTS idx_record_chunks_message ON record_chunks(message_id);
"""


def split_documents(messages, target_chars=TARGET_CHARS):
    """Use source boundaries for embedding/profile chunks and retain exact offsets."""
    chunks, documents = [], []
    for original in messages:
        text = original["content"]
        match = SOURCE_PREFIX.match(text)
        prefix = match.group() if match else ""
        body = text[len(prefix) :]
        spans = []
        for start, end in chunk_spans(body, target_chars):
            part = body[start:end]
            if not part.strip():
                continue
            chunk = {"role": original["role"], "content": prefix + part}
            if "timestamp" in original:
                chunk["timestamp"] = original["timestamp"]
            first = start == 0
            spans.append(
                (
                    len(chunks),
                    0 if first else len(prefix) + start,
                    len(prefix) + start + len(part),
                    0 if first else len(prefix),
                    len(prefix) + len(part),
                )
            )
            chunks.append(chunk)
        if not spans:
            raise ValueError("Original source has no nonempty chunks")
        documents.append((dict(original), spans))
    return chunks, documents


def align_documents(originals, stored):
    """Align exact originals with whole, historical or boundary-based stored chunks."""
    documents, cursor = [], 0
    for original in originals:
        text = original["content"]
        match = SOURCE_PREFIX.match(text)
        prefix = match.group() if match else ""
        body, consumed, links = text[len(prefix):], 0, []
        while consumed < len(body):
            if cursor >= len(stored):
                raise ValueError("Stored history ends before the original source")
            row = stored[cursor]
            content = row["content"]
            part = content[len(prefix):] if content.startswith(prefix) else ""
            if (
                row["role"] != original["role"]
                or row["timestamp_ms"] != original.get("timestamp")
                or not part
                or not body.startswith(part, consumed)
            ):
                raise ValueError("Original source does not match stored history")
            start = 0 if consumed == 0 else len(prefix) + consumed
            end = len(prefix) + consumed + len(part)
            a = 0 if consumed == 0 else len(prefix)
            if text[start:end] != content[a:]:
                raise ValueError("Source span does not match stored chunk")
            links.append((row["id"], start, end, a, len(content)))
            consumed += len(part)
            cursor += 1
        if not links:
            raise ValueError("Original source has no nonempty chunks")
        documents.append((original, links))
    if cursor != len(stored):
        raise ValueError("Stored history contains unmatched messages")
    return documents


def source_document_id(user_id, session_id, first_message_id, content):
    key = f"{user_id}:{session_id}:{first_message_id}:{content}"
    return "doc_" + hashlib.sha256(key.encode()).hexdigest()[:24]


def insert_documents(conn, user_id, session_id, documents, created_at):
    counts = {"documents": 0, "records": 0, "chunk_links": 0}
    for original, links in documents:
        document_id = source_document_id(user_id, session_id, links[0][0], original["content"])
        conn.execute(
            "INSERT INTO source_documents VALUES (?,?,?,?,?,?,?)",
            (
                document_id,
                user_id,
                session_id,
                original["role"],
                original["content"],
                original.get("timestamp"),
                created_at,
            ),
        )
        counts["documents"] += 1
        for message_id, start, end, a, b in links:
            row = conn.execute(
                "SELECT user_id,session_id,role,content FROM messages WHERE id=?", (message_id,)
            ).fetchone()
            if (
                row is None
                or row["user_id"] != user_id
                or row["session_id"] != session_id
                or row["role"] != original["role"]
                or original["content"][start:end] != row["content"][a:b]
            ):
                raise ValueError("Invalid source ownership or offsets")
            conn.execute("INSERT INTO source_spans VALUES (?,?,?,?,?,?)", (document_id, message_id, start, end, a, b))
        units = source_units(original["content"])
        for unit in units:
            ident = "record_" + hashlib.sha256(f"{document_id}:{unit.start}:{unit.end}".encode()).hexdigest()[:24]
            conn.execute(
                "INSERT INTO source_records VALUES (?,?,?,?,?,?)",
                (ident, document_id, user_id, unit.kind, unit.start, unit.end),
            )
            counts["records"] += 1
            for message_id, start, end, a, _ in links:
                lo, hi = max(unit.start, start), min(unit.end, end)
                if lo < hi:
                    conn.execute(
                        "INSERT INTO record_chunks VALUES (?,?,?,?,?,?)",
                        (ident, message_id, lo, hi, a + lo - start, a + hi - start),
                    )
                    counts["chunk_links"] += 1
    return counts


def local_bm25(texts, query):
    """Corpus statistics belong only to the requested user's records/profiles."""
    terms = list(dict.fromkeys(re.findall(r"\w+", query.casefold())))[:32]
    frequencies, lengths = [], []
    for text in texts:
        tokens = re.findall(r"\w+", text.casefold())
        counts = Counter(tokens)
        frequencies.append({term: counts.get(term, 0) for term in terms})
        lengths.append(len(tokens))
    average = sum(lengths) / max(1, len(lengths)) or 1
    dfs = {term: sum(row[term] > 0 for row in frequencies) for term in terms}
    result = []
    for row, length in zip(frequencies, lengths, strict=True):
        score = 0.0
        for term in terms:
            freq = row[term]
            if freq:
                idf = math.log(1 + (len(texts) - dfs[term] + 0.5) / (dfs[term] + 0.5))
                score += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * length / average))
        result.append(score)
    return result


def deduplicate_profiles(items):
    """Merge only equal structured claims; retain every exact source and ID.

    Similar wording and shared attributes do not establish equivalent states.
    Positive/negative or old/new values must remain separate evidence.
    """
    output, known = [], {}
    for item in items:
        if not all(item.get(key) is not None for key in ("entity", "attribute", "value")):
            output.append(item)
            continue
        key = json.dumps([item[k] for k in ("entity", "attribute", "value")], ensure_ascii=False, sort_keys=True)
        if key not in known:
            merged = {**item, "sources": list(item.get("sources", [])), "duplicate_profile_ids": []}
            known[key] = merged
            output.append(merged)
            continue
        merged = known[key]
        merged["duplicate_profile_ids"].append(item["id"])
        for source in item.get("sources", []):
            if source not in merged["sources"]:
                merged["sources"].append(source)
                merged["content"] += "\n[Additional source for identical profile claim] " + json.dumps(
                    source, ensure_ascii=False
                )
    return output


def candidates(store, user_id, query, options, top_k):
    with store._connect() as conn:
        rows = conn.execute("SELECT * FROM source_records WHERE user_id=? ORDER BY id", (user_id,)).fetchall()
        documents = {
            row["id"]: dict(row) for row in conn.execute("SELECT * FROM source_documents WHERE user_id=?", (user_id,))
        }
        if not rows:
            if conn.execute("SELECT 1 FROM messages WHERE user_id=? LIMIT 1", (user_id,)).fetchone():
                raise RuntimeError("Source record index is missing; backfill exact originals or use a new indexed Add")
            return []
        missing = conn.execute(
            "SELECT 1 FROM messages m WHERE m.user_id=? AND NOT EXISTS (SELECT 1 FROM source_spans s WHERE s.message_id=m.id) LIMIT 1",
            (user_id,),
        ).fetchone()
        if missing:
            raise RuntimeError("Source record index is incomplete")
        messages = {
            row["id"]: dict(row)
            for row in conn.execute("SELECT id,position,content FROM messages WHERE user_id=?", (user_id,))
        }
        profiles = conn.execute(
            "SELECT *,NULL AS role,NULL AS timestamp_ms FROM profile_items WHERE user_id=?", (user_id,)
        ).fetchall()
        vectors = {
            row["memory_id"]: json.loads(row["vector"])
            for row in conn.execute(
                "SELECT memory_id,vector FROM memory_vectors WHERE user_id=? AND model=?",
                (user_id, store.embedding_model),
            )
        }
        links = {}
        for row in conn.execute(
            "SELECT c.* FROM record_chunks c JOIN source_records r ON r.id=c.record_id WHERE r.user_id=?", (user_id,)
        ):
            links.setdefault(row["record_id"], []).append(dict(row))
    expected = set(messages) | {row["id"] for row in profiles}
    if not expected <= vectors.keys():
        raise RuntimeError("Source record chunk/profile vectors are missing")
    query_vector = (
        store.embedder.embed_query(query) if hasattr(store.embedder, "embed_query") else store._embed([query])[0]
    )
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    similarity = {}
    for ident in expected:
        vector = vectors[ident]
        if len(vector) != len(query_vector):
            raise ValueError("Source record vector dimensions differ")
        similarity[ident] = sum(a * b for a, b in zip(vector, query_vector, strict=True)) / (
            math.sqrt(sum(value * value for value in vector)) * query_norm
        )
    items = []
    oversized = 0
    for row in rows:
        document = documents[row["document_id"]]
        body = document["content"][row["start"] : row["end"]]
        if len(body) + 300 > store.evidence_char_budget:
            oversized += 1
            continue
        chunk_links = links.get(row["id"], [])
        if not chunk_links:
            continue
        ids = sorted({link["message_id"] for link in chunk_links}, key=lambda ident: messages[ident]["position"])
        score = max(similarity[ident] for ident in ids)
        label = f"[source_record: {row['id']}; kind: {row['kind']}; source_order: {messages[ids[0]]['position']}] "
        if row["kind"] == "json_fragment":
            label += (
                "[Recovered valid JSON fragment; surrounding document failed parsing. Population coverage is unknown.] "
            )
        formatted = store._result(
            {
                "id": row["id"],
                "role": document["role"],
                "section": None,
                "timestamp_ms": document["timestamp_ms"],
                "content": body,
                "created_at": document["created_at"],
            },
            score,
        )
        items.append(
            {
                "id": row["id"],
                "content": label + formatted["content"],
                "score": score,
                "created_at": document["created_at"],
                "context_message_ids": ids,
                "record_sources": [
                    {
                        "record_id": row["id"],
                        "document_id": row["document_id"],
                        "kind": row["kind"],
                        "start": row["start"],
                        "end": row["end"],
                        "complete": True,
                        "document_parse_complete": row["kind"] not in {"json_fragment", "unparsed_context"},
                        "source_spans": chunk_links,
                    }
                ],
                "_lexical_text": body,
            }
        )
    formatted_profiles = deduplicate_profiles(store._format_results([(row, similarity[row["id"]]) for row in profiles]))
    items.extend({**item, "_lexical_text": item["content"]} for item in formatted_profiles)
    if store.event_memory is not None:
        from memory_service.event_memory import retrieval_items

        with store._connect() as conn:
            items.extend(retrieval_items(conn, user_id, store.embedding_model, query_vector, vectors, store.evidence_char_budget,
                                         with_dedup_metadata=store.evidence_dedup))
    lexical = local_bm25([item["_lexical_text"] for item in items], query + "\n" + "\n".join(options or []))
    if store.hybrid_fusion != "legacy":
        from memory_service.hybrid_recall import fuse

        for item in items:
            item["oversized_records_omitted"] = oversized
            item.pop("_lexical_text")
        return fuse(items, [item["score"] for item in items], lexical,
                    max(top_k, store.rerank_candidates), store.hybrid_fusion)
    ranks = {
        index: rank
        for rank, index in enumerate(
            sorted(range(len(items)), key=lambda index: (-lexical[index], items[index]["id"])), 1
        )
        if lexical[index] > 0
    }
    for index, item in enumerate(items):
        item["score"] += 0.05 * 61 / (60 + ranks[index]) if index in ranks else 0
        item["oversized_records_omitted"] = oversized
        item.pop("_lexical_text")
    # Union semantic and lexical record recall before the candidate cutoff.
    size = max(top_k, store.rerank_candidates)
    semantic = sorted(items, key=lambda item: (-item["score"], item["id"]))[:size]
    by_id = {item["id"]: item for item in semantic}
    for index in sorted(ranks, key=ranks.get)[:size]:
        by_id.setdefault(items[index]["id"], items[index])
    # Preserve lexical-only additions by reciprocal ranks, rather than dropping them again.
    sem_rank = {
        item["id"]: rank for rank, item in enumerate(sorted(items, key=lambda item: (-item["score"], item["id"])), 1)
    }
    lex_rank = {items[index]["id"]: rank for index, rank in ranks.items()}
    return sorted(
        by_id.values(),
        key=lambda item: (
            -(1 / (60 + sem_rank[item["id"]]) + (1 / (60 + lex_rank[item["id"]]) if item["id"] in lex_rank else 0)),
            item["id"],
        ),
    )[:size]
