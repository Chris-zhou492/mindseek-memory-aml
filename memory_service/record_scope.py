"""Bind explicit ordinal mode labels to observed fields; no external mode codes."""

import re

from memory_service.calculations import MISSING, lookup, serial


def grounded_scopes(pools, history):
    if len(pools["json"]) != 1 or not isinstance(pools["json"][0]["data"], list):
        return []
    source = pools["json"][0]
    records = source["data"]
    if not records or len(records) > 100 or not all(isinstance(r, dict) for r in records):
        return []
    paths = set()

    def visit(value, path=(), depth=0):
        if not isinstance(value, dict) or depth > 3:
            return
        for key, item in value.items():
            current = (*path, key)
            if isinstance(item, dict):
                visit(item, current, depth + 1)
            elif (
                re.search(r"queue|mode|category|match.?id|game.?id", key, re.I)
                and isinstance(item, (str, int))
                and not isinstance(item, bool)
            ):
                paths.add(current)

    for record in records:
        visit(record)
    bindings = []
    pattern = re.compile(
        r"(?:(?:matches|games|records)\s+)?(\d+)\s*[-–]\s*(\d+)\s+(?:were|are)\s+(competitive|casual|ranked|unranked)\b",
        re.I,
    )
    for entry in history:
        if entry["role"] != "user" or not re.search(r"\b(matches|games|records)\b", entry["excerpt"], re.I):
            continue
        text = entry["excerpt"]
        for match in pattern.finditer(text):
            start, end = int(match[1]), int(match[2])
            if not 1 <= start <= end <= len(records):
                continue
            selectors = []
            for path in sorted(paths):
                values = [lookup(r, list(path)) for r in records]
                selected = values[start - 1 : end]
                if any(v is MISSING or not isinstance(v, (str, int)) or isinstance(v, bool) for v in values):
                    continue
                chosen = list(dict.fromkeys(selected))
                # A selector must identify exactly the declared record subset.
                if set(chosen).intersection(values[: start - 1] + values[end:]):
                    continue
                if len(chosen) > 20:
                    continue
                selectors.append(
                    {
                        "path": ["*", *path],
                        "op": "eq" if len(chosen) == 1 else "in",
                        "value": serial(chosen[0] if len(chosen) == 1 else chosen),
                    }
                )
            if selectors:
                bindings.append(
                    {
                        "label": match[3].casefold(),
                        "record_range": [start, end],
                        "selectors": selectors,
                        "source": {
                            "document_id": entry["document_id"],
                            "role": "user",
                            "start": entry["start"] + match.start(),
                            "end": entry["start"] + match.end(),
                            "quote": match.group(),
                        },
                        "json_source": source["source"]["document_id"],
                    }
                )
    return bindings


def scope_errors(plan, question, bindings):
    errors = {}
    relevant = [b for b in bindings if re.search(r"\b" + re.escape(b["label"]) + r"\b", question, re.I)]
    # Conflicting user declarations must not silently become an authoritative map.
    relevant = [
        b for b in relevant if all(x["label"] != b["label"] or x["record_range"] == b["record_range"] for x in relevant)
    ]
    for query in plan["queries"]:
        if query["parser"] != "json":
            continue
        for binding in relevant:
            accepted = False
            contradicted = False
            for condition in query.get("where", []):
                column = query["columns"][condition["field"]]
                path = column.get("path", [])
                if column.get("scope", "row") == "row":
                    path = query.get("rows_path", []) + path
                elif column["scope"] == "ancestor":
                    up = column.get("up", 1)
                    path = (query.get("rows_path", [])[:-up] if up else query.get("rows_path", [])) + path
                for selector in binding["selectors"]:
                    if path != selector["path"] or condition["op"] not in {"eq", "in"}:
                        continue
                    values = condition["value"] if condition["op"] == "in" else [condition["value"]]
                    allowed = selector["value"] if selector["op"] == "in" else [selector["value"]]
                    if values and all(v in allowed for v in values):
                        accepted = True
                    else:
                        contradicted = True
            if not accepted or contradicted:
                errors[query["name"]] = (
                    "Filter does not bind requested "
                    + binding["label"]
                    + " records to the user-grounded selectors: "
                    + str(binding["selectors"])
                )
    return errors
