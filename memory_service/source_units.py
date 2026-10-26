"""Lossless source-span segmentation; no question, gold answer or model inference."""

import json
import re
from dataclasses import dataclass

from memory_service.source_recovery import feed_spans, json_fragments


def with_gaps(spans, size, gap_kind):
    units, cursor = [], 0
    for start, end, kind in spans:
        if cursor < start:
            units.append(SourceUnit(cursor, start, gap_kind))
        units.append(SourceUnit(start, end, kind))
        cursor = end
    if cursor < size:
        units.append(SourceUnit(cursor, size, gap_kind))
    return units


@dataclass(frozen=True)
class SourceUnit:
    start: int
    end: int
    kind: str


def source_units(text: str) -> list[SourceUnit]:
    """Prefer explicit record boundaries, then paragraphs. Offsets index original text.

    JSON containers are traversed to find array element boundaries without
    serializing values. Nested arrays stay inside their parent record. Textual
    headers cover common PGN, hand-history, turn-log and RSS record formats.
    """
    if not text:
        return []
    decoder = json.JSONDecoder()

    def walk(start, depth=0):
        start = skip_space(start)
        if depth > 50:
            raise ValueError("JSON nesting limit")
        opening = text[start]
        if opening not in "[{":
            _, end = decoder.raw_decode(text, start)
            return end, []
        array = opening == "["
        closing = "]" if array else "}"
        pos = skip_space(start + 1)
        spans = []
        while text[pos] != closing:
            if not array:
                key, pos = decoder.raw_decode(text, pos)
                if not isinstance(key, str):
                    raise ValueError("Invalid JSON object key")
                pos = skip_space(pos)
                if text[pos] != ":":
                    raise ValueError("Invalid JSON object")
                pos = skip_space(pos + 1)
            item_start = pos
            end, children = walk(pos, depth + 1)
            if array:
                spans.append(SourceUnit(item_start, end, "json_record"))
            else:
                spans.extend(children)
            pos = skip_space(end)
            if text[pos] == closing:
                break
            if text[pos] != ",":
                raise ValueError("Invalid JSON separator")
            pos = skip_space(pos + 1)
        return pos + 1, spans

    def skip_space(pos):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        return pos

    try:
        begin = skip_space(0)
        if text[begin] in "[{":
            end, units = walk(begin)
            if units:
                if end < len(text):
                    units.append(SourceUnit(end, len(text), "trailing_text"))
                return units
    except (ValueError, IndexError, RecursionError):
        if text.lstrip().startswith(("{", "[")):
            fragments = json_fragments(text)
            if fragments:
                return with_gaps([(a, b, "json_fragment") for a, b in fragments], len(text), "unparsed_context")

    feeds = feed_spans(text)
    if feeds:
        return with_gaps(feeds, len(text), "header")

    patterns = (
        (r"(?m)^(?:Game )?Hand #[^\n]+", "hand"),
        (r'(?m)^\[Event "[^\n]*', "pgn_game"),
        (r"(?m)^=== TURN \d+ ===[^\n]*", "turn"),
        (r"(?m)^[ \t]*<item(?:\s[^>]*)?>", "xml_item"),
    )
    for pattern, kind in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            boundaries = [m.start() for m in matches] + [len(text)]
            units = [SourceUnit(a, b, kind) for a, b in zip(boundaries, boundaries[1:])]
            if boundaries[0]:
                units.insert(0, SourceUnit(0, boundaries[0], "header"))
            return units
    # Preserve ordinary short conversation turns. Larger prose is segmented on
    # actual paragraph breaks, never on an arbitrary character count.
    if len(text) <= 6000:
        return [SourceUnit(0, len(text), "message")]
    boundaries = [0] + [m.end() for m in re.finditer(r"\n[ \t]*\n", text)] + [len(text)]
    return [SourceUnit(a, b, "paragraph") for a, b in zip(boundaries, boundaries[1:]) if a < b]


def complete_units(text: str, start: int, end: int, max_chars: int = 6000):
    """Expand only units intersecting a hit. Refuse partial/oversized records.

    A refusal returns None, allowing the caller to retain the original evidence.
    JSON punctuation outside array records may have no matching complete unit.
    """
    if not 0 <= start < end <= len(text):
        raise ValueError("Invalid source span")
    matches = [u for u in source_units(text) if u.start < end and start < u.end]
    if not matches:
        return None
    pieces = [text[u.start : u.end] for u in matches]
    if sum(map(len, pieces)) > max_chars:
        return None
    return matches
