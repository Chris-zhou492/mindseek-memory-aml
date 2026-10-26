"""Bounded, source-exact recall of user preferences, action cues and stages."""

import re

from memory_service.record_index import local_bm25


def excerpt(source, start, end, kind):
    return {
        **source,
        "start": source["start"] + start,
        "end": source["start"] + end,
        "content": source["content"][start:end],
        "support_kind": kind,
    }


def prose_ranges(text):
    fences = list(re.finditer(r"```[\s\S]*?(?:```|\Z)", text))
    cursor = 0
    for fence in fences:
        if cursor < fence.start():
            yield cursor, fence.start()
        cursor = fence.end()
    if cursor < len(text):
        yield cursor, len(text)


def preferences(items, query):
    """Only recognised response formats, never arbitrary instructions."""
    kinds = []
    if re.search(r"\b(?:librar\w*|dependenc\w*|software|version\w*)\b|依赖|库|版本", query, re.I):
        kinds.append(
            ("versions", r"\b(?:include|provide|show|state|list)(?:\s+\w+){0,3}\s+version (?:numbers|details)\b")
        )
    if re.search(r"\b(?:implement\w*|code|example)\b|代码|实现", query, re.I):
        kinds.append(("code_format", r"(?:always|please|must|prefer|format)[^\n.!?]{0,160}\bsyntax highlighting\b"))
    found, seen = [], set()
    for item in items:
        if item["role"] != "user":
            continue
        for lo, hi in prose_ranges(item["content"]):
            for label, pattern in kinds:
                for match in re.finditer(pattern, item["content"][lo:hi], re.I):
                    start = lo + match.start()
                    end = min(hi, start + 400)
                    boundary = re.search(r"[\n!?]|\.(?:\s|$)", item["content"][lo + match.end() : end])
                    if boundary:
                        end = lo + match.end() + boundary.end()
                    key = (item["document_id"], item["start"] + start)
                    if key not in seen:
                        found.append({**excerpt(item, start, end, "response_preference"), "preference_type": label})
                        seen.add(key)
    return sorted(found, key=lambda s: (s["source_order"], s["start"]))[-4:]


def action_cues(items, query):
    found = []
    patterns = [
        ("negative_wording", r"\b(?:I(?:'ve| have)? never|I have not|I haven't|not yet)\b"),
        (
            "action_wording",
            r"\bI(?:'ve| have)? (?:already |also )?(?:implemented|completed|integrated|deployed|added)\b",
        ),
    ]
    for item in items:
        if item["role"] != "user":
            continue
        for lo, hi in prose_ranges(item["content"]):
            for label, pattern in patterns:
                for match in re.finditer(pattern, item["content"][lo:hi], re.I):
                    cue_start = lo + match.start()
                    # Include the object before relative wording such as
                    # "Library, which I've never integrated".
                    start = max(lo, cue_start - 60)
                    prefix = item["content"][start:cue_start]
                    boundaries = list(re.finditer(r"[\n!?]|\.(?:\s|$)", prefix))
                    if boundaries:
                        start += boundaries[-1].end()
                    end = min(hi, cue_start + 300)
                    boundary = re.search(r"[\n!?]|\.(?:\s|$)", item["content"][lo + match.end() : end])
                    if boundary:
                        end = lo + match.end() + boundary.end()
                    found.append({**excerpt(item, start, end, "user_action_cue"), "cue": label})
    scores = local_bm25([s["content"] for s in found], query)
    result, seen = [], set()
    for label, _ in patterns:
        ranked = sorted(range(len(found)), key=lambda i: (-scores[i], found[i]["source_order"]))
        count = 0
        for i in ranked:
            source = found[i]
            key = (source["document_id"], source["start"], source["end"])
            if scores[i] <= 0 or source["cue"] != label or key in seen:
                continue
            result.append(source)
            seen.add(key)
            count += 1
            if count == 3:
                break
    return result


def version_mentions(items, query):
    if not re.search(r"\b(?:librar\w*|dependenc\w*|software|version\w*)\b|依赖|版本", query, re.I):
        return []
    found = {}
    pattern = r"\b([A-Za-z][A-Za-z0-9.+_-]{1,40})\s+v?(\d+(?:\.\d+){1,3})\b"
    for item in sorted(items, key=lambda s: (s["source_order"], s["start"])):
        if item["role"] != "user":
            continue
        for lo, hi in prose_ranges(item["content"]):
            for match in re.finditer(pattern, item["content"][lo:hi]):
                # Exact bounded context retains requests/qualifications. A mention
                # does not establish installation, adoption or a current version.
                start, end = max(lo, lo + match.start() - 70), min(hi, lo + match.end() + 50)
                found[(match[1].casefold(), match[2])] = {
                    **excerpt(item, start, end, "version_mention"),
                    "name": match[1],
                    "version": match[2],
                }
    return list(found.values())[-12:]


def chronology_sources(items):
    """Every user document gets a short opening before code-heavy repetitions."""
    documents = {}
    for item in items:
        if item["role"] != "user":
            continue
        prior = documents.get(item["document_id"])
        if prior is None or (item["start"], -len(item["content"])) < (prior["start"], -len(prior["content"])):
            documents[item["document_id"]] = item
    ordered = sorted(documents.values(), key=lambda s: (s["source_order"], s["start"]))
    # Bound cost. Beyond 96 documents, preserve a declared partial timeline.
    if len(ordered) > 96:
        ordered = [ordered[round(i * (len(ordered) - 1) / 95)] for i in range(96)]
    result = []
    width = min(300, max(100, 17000 // max(1, len(ordered))))
    for item in ordered:
        for lo, hi in prose_ranges(item["content"]):
            if item["content"][lo:hi].strip():
                result.append(excerpt(item, lo, min(hi, lo + width), "chronology_opening"))
                break
    return result


def source_fact(source):
    return {
        "quote": source["content"],
        "source": {k: source[k] for k in ("document_id", "record_id", "role", "source_order", "start", "end")},
        **{k: source[k] for k in ("cue", "preference_type", "name", "version") if k in source},
    }
