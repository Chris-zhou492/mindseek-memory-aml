"""Recover bounded, independently valid records without rewriting source bytes."""

import json
import re


def json_fragments(text, max_chars=6000):
    """Maximal disjoint valid containers inside a damaged/oversized JSON payload.

    A fragment proves only its own fields, never that the surrounding population
    is complete. Brackets inside strings are ignored. Original offsets survive.
    """
    stack, fragments = [], []
    quoted = escaped = False
    decoder = json.JSONDecoder(parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
    for pos, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "[{":
            stack.append((pos, char, len(fragments)))
        elif char in "]}":
            if not stack or stack[-1][1] != ("[" if char == "]" else "{"):
                stack.clear()
                continue
            start, _, index = stack.pop()
            end = pos + 1
            if end - start > max_chars:
                continue
            try:
                decoder.decode(text[start:end])
            except (ValueError, RecursionError):
                continue
            fragments[index:] = [(start, end)]
    return fragments


def feed_spans(text):
    """Detect XML, escaped XML and rendered activity items in mixed documents."""
    found = []
    for pattern, close in ((r"<item(?:\s[^>]*)?>", "</item>"), (r"&lt;item(?:\s[^&]*?)?&gt;", "&lt;/item&gt;")):
        for match in re.finditer(pattern, text):
            end = text.find(close, match.end())
            if end >= 0:
                found.append((match.start(), end + len(close), "xml_item"))
    headers = list(re.finditer(r"(?m)([A-Za-z0-9_][A-Za-z0-9_ -]{0,79}) avatar \1[ \t]*\r?$", text))
    for index, match in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        # A rendered item ends at its timestamp. Do not absorb a trailing task.
        date = re.search(r"\b\d{1,2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2}\b", text[match.end() : end])
        if date:
            found.append((match.start(), match.end() + date.end(), "activity_item"))
    found.sort()
    # Rendered headers inside a genuine XML description are not independent items.
    result = []
    for item in found:
        if result and item[0] < result[-1][1]:
            continue
        result.append(item)
    return result
