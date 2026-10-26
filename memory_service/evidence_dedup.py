"""Conservative event evidence projection with source/state-aware coverage."""

VERSION = "event-source-state-dedup-v1"


def fragments(item):
    """Reject incomplete/legacy metadata instead of guessing semantic equality."""
    if item.get("memory_kind") not in {"event", "event_fact"}:
        return []
    content, sources = item.get("content"), item.get("record_sources")
    if not isinstance(content, str) or not isinstance(sources, list) or not sources:
        return []
    parts = []
    cursor = None
    for source in sources:
        try:
            start, end = source["display_start"], source["display_end"]
            left, right = source["start"], source["end"]
            labels = tuple(source[k] for k in ("document_id", "role", "state", "subject", "attribute"))
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 < start < end <= len(content)
                or (cursor is not None and start != cursor)
                or type(left) is not int
                or type(right) is not int
                or not 0 <= left < right
                or not all(isinstance(x, str) and x for x in labels)
                or not isinstance(source["statement_id"], str)
                or not isinstance(source["message_ids"], list)
            ):
                return []
            key = (*labels, left, right)
            parts.append((key, source, content[start:end]))
            cursor = end
        except (KeyError, TypeError):
            return []
    return parts if cursor == len(content) else []


def project_event(item, covered):
    """Remove only complete repeated event fragments, never rewrite a quote.

    Raw records and profiles retain their original qualification and context.
    Different states/roles/subjects remain independent even on the same span.
    The caller updates coverage only after accepting a packet into its budget.
    """
    parts = fragments(item)
    if not parts:
        return item, set()
    novel, removed, local = [], [], set(covered)
    for key, source, text in parts:
        if key in local:
            removed.append(source["statement_id"])
        else:
            novel.append((key, source, text))
            local.add(key)
    if not novel:
        return None, set()
    keys = {key for key, _, _ in novel}
    if not removed:
        return item, keys
    content = item["content"][: parts[0][1]["display_start"]]
    sources = []
    for _, source, text in novel:
        start = len(content)
        content += text
        sources.append({**source, "display_start": start, "display_end": len(content)})
    return {
        **item,
        "content": content,
        "record_sources": sources,
        "context_message_ids": sorted({mid for s in sources for mid in s["message_ids"]}),
        "group_member_ids": [s["statement_id"] for s in sources],
        "event_states": sorted({s["state"] for s in sources}),
        "omitted_event_statements": item.get("omitted_event_statements", 0) + len(removed),
        "dedup_removed_statement_ids": [*item.get("dedup_removed_statement_ids", []), *removed],
    }, keys
