"""Build retrieval spans at source boundaries without fixed-width slicing."""

import json
import re

from memory_service.source_units import source_units

VERSION = "source-boundary-chunks-v1"
TARGET_CHARS = 6000  # Soft grouping target, never a slicing position or provider token limit.
SOURCE_PREFIX = re.compile(r"\[source_date: [^\n]*?\] ")
STRUCTURED = {"json_record", "json_fragment", "hand", "pgn_game", "turn", "xml_item"}


def _prose_units(text, start, end):
    """Keep fenced code intact; split other text on existing paragraph breaks."""
    units, cursor, fence = [], start, None
    position = start
    for line in text[start:end].splitlines(keepends=True):
        match = re.match(r"[ \t]{0,3}(`{3,}|~{3,})", line)
        if match and fence is None:
            if cursor < position:
                units.extend(_paragraphs(text, cursor, position))
            cursor, fence = position, match.group(1)
        elif match and fence and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence):
            units.append((cursor, position + len(line), True))
            cursor, fence = position + len(line), None
        position += len(line)
    if cursor < end:
        units.extend([(cursor, end, True)] if fence else _paragraphs(text, cursor, end))
    return units


def _paragraphs(text, start, end):
    cuts = [start] + [start + m.end() for m in re.finditer(r"\n[ \t]*\n", text[start:end])] + [end]
    return [(a, b, False) for a, b in zip(cuts, cuts[1:]) if a < b]


def _bounded_prose(text, start, end, limit):
    pieces = []
    for a, b, protected in _prose_units(text, start, end):
        if b - a <= limit:
            pieces.append((a, b))
            continue
        if protected:
            pieces.append((a, b))
            continue
        # Heuristic punctuation/line boundaries, not an inferred semantic topic split.
        boundaries = [a]
        boundaries.extend(
            a + m.end()
            for m in re.finditer(r'''[。！？!?]+[”’"']*\s*|\.(?=\s|$)[”’"']*\s*|\n''', text[a:b])
        )
        boundaries.append(b)
        sentences = [(lo, hi) for lo, hi in zip(boundaries, boundaries[1:]) if lo < hi]
        pieces.extend(sentences)
    # An indivisible long sentence stays whole. The provider enforces its token limit.
    packed = []
    for a, b in pieces:
        if packed and b - packed[-1][0] <= limit:
            packed[-1] = (packed[-1][0], b)
        else:
            packed.append((a, b))
    return packed


def chunk_spans(text, target_chars=TARGET_CHARS):
    if type(target_chars) is not int or target_chars < 1:
        raise ValueError("target_chars must be a positive integer")
    units = source_units(text)
    if not units:
        return []
    if not any(unit.kind in STRUCTURED for unit in units):
        if text.lstrip().startswith(("{", "[")):
            try:
                json.loads(text)
            except ValueError:
                pass
            else:
                return [(0, len(text))]
        if len(text) <= target_chars:
            return [(0, len(text))]
        # Inspect all prose together, so paragraph breaks inside code fences
        # cannot become chunk boundaries before fences have been recognised.
        return _bounded_prose(text, 0, len(text), target_chars)
    # Include wrappers, punctuation and whitespace in adjacent source regions.
    boundaries = sorted({0, len(text), *(unit.start for unit in units if unit.start > 0)})
    protected = [(unit.start, unit.end) for unit in units if unit.kind in STRUCTURED]
    output = []
    for a, b in zip(boundaries, boundaries[1:]):
        if b - a <= target_chars:
            output.append((a, b))
        elif any(lo < b and a < hi for lo, hi in protected):
            output.append((a, b))
        else:
            output.extend(_bounded_prose(text, a, b, target_chars))
    return output
