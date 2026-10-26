"""Bounded, source-audited calculations. No generated Python, SQL or regex."""

from __future__ import annotations

import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from decimal import ROUND_FLOOR, Decimal, InvalidOperation, localcontext
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, urlsplit

from memory_service.source_recovery import feed_spans

MISSING = object()
MAX_ROWS = 20000
MAX_GROUPS = 100
PARSERS = {"json", "pgn", "hand", "xml"}
OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "contains", "not_contains", "exists"}
METRICS = {"count", "distinct_count", "values", "sum", "mean", "min", "max", "first", "last"}


def serial(value):
    if isinstance(value, Decimal):
        # Decimal.normalize() applies the ambient precision and can round an audit value.
        text = format(value, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    return value


def number(value):
    if isinstance(value, bool) or value is MISSING or value is None:
        return None
    if not isinstance(value, (int, float, str, Decimal)) or len(str(value)) > 100:
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    return result if result.is_finite() and result.adjusted() <= 50 and result.adjusted() >= -50 else None


def json_value(text):
    begin = len(text) - len(text.lstrip())
    if begin == len(text) or text[begin] not in "[{":
        return None
    decoder = json.JSONDecoder(
        parse_float=Decimal, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON number"))
    )
    value, end = decoder.raw_decode(text, begin)
    return value, begin, end


def parsed_record(parser, text):
    if parser == "json":
        result = json_value(text)
        return result[0] if result else None
    if parser == "pgn":
        tags = dict(re.findall(r'^\[([A-Za-z0-9_]+) "([^"\n]*)"\][ \t]*$', text, re.M))
        if not tags:
            raise ValueError("Invalid PGN header")
        return tags
    if parser == "hand":
        hand = re.search(r"(?m)^(?:Game )?Hand #([A-Za-z0-9_-]+)", text)
        if not hand:
            raise ValueError("Invalid hand header")
        dealt = re.search(r"(?m)^Dealt to (.+?) \[", text)
        actor = dealt.group(1) if dealt else None
        return {
            "hand_id": hand.group(1),
            "dealt_to": actor,
            "showdown": bool(re.search(r"\*\*\* SHOW\s*DOWN \*\*\*", text)),
            "dealt_player_folded": bool(re.search(r"(?m)^" + re.escape(actor) + r":? folds\b", text))
            if actor
            else None,
            "dealt_player_showed": bool(re.search(r"(?m)^" + re.escape(actor) + r":? shows\b", text))
            if actor
            else None,
        }
    if parser == "xml":
        if re.match(r"([^\n<>]{1,80}) avatar \1(?:\r?\n)", text):
            lines = text.splitlines()
            account = lines[0].split(" avatar ", 1)[0]
            stamp = re.search(r"\b\d{1,2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2}\b", text)
            if not stamp or len(lines) < 2:
                raise ValueError("Incomplete rendered activity item")
            return {
                "account": account,
                "title": lines[1].strip(),
                "description": text[text.index(lines[1]) + len(lines[1]) : stamp.start()].strip(),
                "pubDate": stamp.group(),
                "date_iso": datetime.strptime(stamp.group(), "%d-%b-%Y %H:%M").isoformat(),
                "link": None,
                "guid": None,
                "source_format": "rendered_activity",
                "timezone_known": False,
            }
        if "&lt;item" in text:
            text = html.unescape(text)
        match = re.search(r"<item(?:\s[^>]*)?>.*?</item>", text, re.S)
        if not match or len(match.group()) > 100000 or "<!" in match.group():
            raise ValueError("Unsupported XML item")
        item = ET.fromstring(match.group())
        row = {name: item.findtext(name) for name in ("title", "description", "link", "pubDate", "guid")}
        row["account"] = parse_qs(urlsplit(row.get("link") or "").query).get("searchName", [None])[0]
        try:
            row["date_iso"] = parsedate_to_datetime(row["pubDate"]).isoformat()
        except (TypeError, ValueError, OverflowError):
            row["date_iso"] = None
        return row
    raise ValueError("Unsupported parser")


def load_sources(store, user_id):
    with store._connect() as conn:
        documents = {
            row["id"]: dict(row)
            for row in conn.execute("SELECT * FROM source_documents WHERE user_id=? ORDER BY id", (user_id,))
        }
        records = conn.execute(
            "SELECT * FROM source_records WHERE user_id=? ORDER BY document_id,start", (user_id,)
        ).fetchall()
    if sum(len(row["content"].encode()) for row in documents.values()) > 8_000_000:
        raise ValueError("Calculation source size limit exceeded")
    pools = {name: [] for name in PARSERS}
    errors = Counter()
    for document in documents.values():
        if document["role"] != "user":
            continue
        text = document["content"]
        try:
            value = json_value(text)
        except (ValueError, RecursionError):
            errors["json"] += 1
            continue
        if value:
            obj, start, end = value
            if text[end:].lstrip().startswith(("[", "{")):
                errors["json"] += 1
            pools["json"].append(
                {
                    "data": obj,
                    "source": {
                        "document_id": document["id"],
                        "record_id": None,
                        "start": start,
                        "end": end,
                        "sha256": hashlib.sha256(text[start:end].encode()).hexdigest(),
                        "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    },
                    "trailing_characters": len(text) - end,
                }
            )
    kinds = {"pgn_game": "pgn", "hand": "hand", "xml_item": "xml", "activity_item": "xml"}
    # Parse feed formats from original documents, including pre-existing indexes
    # whose final XML item absorbed a rendered/escaped suffix. Do not double count.
    for document in documents.values():
        if document["role"] != "user" or document["content"].lstrip().startswith(("{", "[")):
            continue
        text = document["content"]
        spans = feed_spans(text)
        openings = len(re.findall(r"<item(?:\s[^>]*)?>|&lt;item(?:\s[^&]*?)?&gt;", text))
        unclosed = max(0, openings - sum(kind == "xml_item" for _, _, kind in spans))
        if unclosed:
            errors["xml"] += unclosed
        for start, end, _ in spans:
            try:
                data = parsed_record("xml", text[start:end])
            except (ValueError, ET.ParseError, RecursionError):
                errors["xml"] += 1
                continue
            pools["xml"].append(
                {
                    "data": data,
                    "source": {
                        "document_id": document["id"],
                        "record_id": None,
                        "start": start,
                        "end": end,
                        "sha256": hashlib.sha256(text[start:end].encode()).hexdigest(),
                    },
                }
            )
    for record in records:
        if documents[record["document_id"]]["role"] != "user":
            continue
        parser = kinds.get(record["kind"])
        if parser is None or parser == "xml":
            continue
        text = documents[record["document_id"]]["content"][record["start"] : record["end"]]
        try:
            data = parsed_record(parser, text)
        except (ValueError, ET.ParseError, RecursionError):
            errors[parser] += 1
            continue
        pools[parser].append(
            {
                "data": data,
                "source": {
                    "document_id": record["document_id"],
                    "record_id": record["id"],
                    "start": record["start"],
                    "end": record["end"],
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                },
            }
        )
    signature = hashlib.sha256(
        json.dumps(
            {
                "documents": [
                    (key, hashlib.sha256(row["content"].encode()).hexdigest()) for key, row in documents.items()
                ],
                "records": [(row["id"], row["document_id"], row["kind"], row["start"], row["end"]) for row in records],
            }
        ).encode()
    ).hexdigest()
    return pools, dict(errors), signature


def schema_of(value, path=(), depth=0):
    if depth > 7:
        return []
    if isinstance(value, dict):
        entries = list(value.items())[:18]
        if entries and all(str(key).isdigit() for key, _ in entries):
            entries = [("*", item) for _, item in entries[:1]]
        return [leaf for key, item in entries for leaf in schema_of(item, (*path, str(key)), depth + 1)][:90]
    if isinstance(value, list):
        return schema_of(value[0], (*path, "*"), depth + 1) if value else [{"path": list(path), "example": []}]
    display = serial(value)
    if isinstance(display, str):
        display = display[:160]
    return [{"path": list(path), "example": display}]


def inventory_of(pools, errors):
    inventory = {
        parser: {
            "sources": len(rows),
            "parse_errors": errors.get(parser, 0),
            "examples": [schema_of(row["data"]) for row in rows[:2]],
        }
        for parser, rows in sorted(pools.items())
        if rows or errors.get(parser)
    }
    # Examples expose observed identifiers, queue codes and event types beyond the first row.
    for parser, rows in pools.items():
        categories, scanned = {}, 0

        def visit(value, path=(), depth=0):
            nonlocal scanned
            scanned += 1
            if scanned > 200000 or depth > 10:
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(item, (*path, "*" if str(key).isdigit() else str(key)), depth + 1)
            elif isinstance(value, list):
                for item in value:
                    visit(item, (*path, "*"), depth + 1)
            elif isinstance(value, (str, int, float, Decimal, bool)) and 0 < len(str(value)) <= 100:
                value = serial(value)
                key = json.dumps(path)
                if key not in categories and len(categories) < 90:
                    categories[key] = []
                if key in categories and value not in categories[key] and len(categories[key]) < 16:
                    categories[key].append(value)

        for row in rows:
            visit(row["data"])
        if parser in inventory:
            inventory[parser]["observed_scalar_examples_not_exhaustive"] = [
                {"path": json.loads(path), "examples": values} for path, values in categories.items()
            ]
    return inventory


def lookup(value, path):
    for part in path:
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return MISSING
    return value


def walk_rows(root, path):
    frontier = [(root, [])]
    for part in path:
        following = []
        for value, actual in frontier:
            if part == "*":
                entries = (
                    value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else []
                )
                following.extend((item, actual + [str(key)]) for key, item in entries)
            else:
                item = lookup(value, [part])
                if item is not MISSING:
                    following.append((item, actual + [part]))
        if len(following) > MAX_ROWS:
            raise ValueError("Calculation row limit exceeded")
        frontier = following
    return frontier


def valid_path(path, *, wildcard=False):
    if (
        not isinstance(path, list)
        or len(path) > 12
        or any(
            not isinstance(part, str) or len(part) > 100 or part in {"..", "**"} or (part == "*" and not wildcard)
            for part in path
        )
    ):
        raise ValueError("Invalid calculation field path")


def validate_plan(plan):
    if (
        not isinstance(plan, dict)
        or "queries" not in plan
        or set(plan) - {"queries", "needs", "reference_scope", "planning_failures"}
        or not isinstance(plan["queries"], list)
        or len(plan["queries"]) > 4
    ):
        raise ValueError("Invalid calculation plan")
    if "reference_scope" in plan and (
        not isinstance(plan["reference_scope"], dict) or len(json.dumps(plan["reference_scope"])) > 20000
    ):
        raise ValueError("Invalid reference scope metadata")
    names = set()
    for query in plan["queries"]:
        if not isinstance(query, dict) or set(query) - {
            "name",
            "parser",
            "rows_path",
            "columns",
            "where",
            "group_by",
            "metrics",
            "derived",
            "order_by",
            "sort_by",
            "descending",
            "limit",
            "required_order_bounds",
            "compare_to",
            "event_state",
        }:
            raise ValueError("Unknown calculation query fields")
        name = query.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", name) or name in names:
            raise ValueError("Invalid calculation query name")
        names.add(name)
        if not isinstance(query.get("parser"), str) or query.get("parser") not in PARSERS:
            raise ValueError("Unsupported calculation parser")
        valid_path(query.get("rows_path", []), wildcard=True)
        if query["parser"] != "json" and query.get("rows_path", []):
            raise ValueError("pgn/hand/xml already represent one row per source; rows_path must be []")
        columns = query.get("columns")
        if not isinstance(columns, dict) or not 1 <= len(columns) <= 12:
            raise ValueError("Invalid calculation columns")
        for alias, column in columns.items():
            if (
                not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", alias)
                or not isinstance(column, dict)
                or set(column) - {"path", "scope", "up", "bucket_width"}
            ):
                raise ValueError("Invalid calculation column")
            valid_path(column.get("path", []))
            if (
                column.get("scope", "row") not in {"row", "record", "ancestor"}
                or type(column.get("up", 1)) is not int
                or not 0 <= column.get("up", 1) <= 12
            ):
                raise ValueError("Invalid calculation column scope")
            if "bucket_width" in column and (
                number(column["bucket_width"]) is None or number(column["bucket_width"]) <= 0
            ):
                raise ValueError("Invalid numeric bucket width")
        group = query.get("group_by", [])
        if not isinstance(group, list) or len(group) > 3 or any(alias not in columns for alias in group):
            raise ValueError("Invalid calculation grouping")
        metrics = query.get("metrics", [])
        if not isinstance(metrics, list) or not 1 <= len(metrics) <= 12:
            raise ValueError("Invalid calculation metrics")
        aliases = set()
        for metric in metrics:
            if (
                not isinstance(metric, dict)
                or set(metric) - {"name", "op", "field", "where"}
                or metric.get("op") not in METRICS
            ):
                raise ValueError("Unknown calculation metric")
            alias = metric.get("name")
            if not isinstance(alias, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", alias) or alias in aliases:
                raise ValueError("Invalid metric name")
            aliases.add(alias)
            if metric["op"] != "count" and metric.get("field") not in columns:
                raise ValueError("Unknown metric column")
        for conditions in [query.get("where", []), *[metric.get("where", []) for metric in metrics]]:
            if not isinstance(conditions, list) or len(conditions) > 10:
                raise ValueError("Invalid filter count")
            for condition in conditions:
                if (
                    not isinstance(condition, dict)
                    or set(condition) != {"field", "op", "value"}
                    or condition["field"] not in columns
                    or condition["op"] not in OPS
                ):
                    raise ValueError(
                        "Invalid calculation filter: use declared column aliases and supported operators eq,ne,gt,gte,lt,lte,in,contains,not_contains,exists"
                    )
                if len(json.dumps(condition["value"])) > 2000:
                    raise ValueError("Filter value too large")
                if condition["op"] == "exists" and type(condition["value"]) is not bool:
                    raise ValueError("exists requires a boolean")
                if condition["op"] == "in" and not isinstance(condition["value"], list):
                    raise ValueError("in requires a list")
        if query.get("order_by") is not None and query["order_by"] not in columns:
            raise ValueError("Unknown order column")
        if "required_order_bounds" in query:
            bounds = query["required_order_bounds"]
            if (
                not query.get("order_by")
                or not isinstance(bounds, list)
                or len(bounds) != 2
                or any(number(value) is None for value in bounds)
            ):
                raise ValueError("Invalid required order bounds")
        if type(query.get("limit", 40)) is not int or not 1 <= query.get("limit", 40) <= MAX_GROUPS:
            raise ValueError("Invalid result limit")
        if type(query.get("descending", False)) is not bool:
            raise ValueError("Invalid sort direction")
        derived = query.get("derived", {})
        if not isinstance(derived, dict) or len(derived) > 12:
            raise ValueError("Invalid derived metrics")
        for alias, expression in derived.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", alias) or alias in aliases:
                raise ValueError("Invalid derived metric name")
            validate_expression(expression, aliases, names - {name})
            aliases.add(alias)
        if query.get("sort_by") is not None and query["sort_by"] not in aliases:
            raise ValueError("Unknown result sort metric")
        if "compare_to" in query:
            comparison = query["compare_to"]
            if (
                not isinstance(comparison, dict)
                or set(comparison) != {"query", "metrics"}
                or comparison["query"] not in names - {name}
                or not isinstance(comparison["metrics"], list)
                or not 1 <= len(comparison["metrics"]) <= 12
                or any(metric not in aliases for metric in comparison["metrics"])
            ):
                raise ValueError("Invalid baseline comparison")
        if "event_state" in query:
            from memory_service.event_state import validate

            validate(query["event_state"], columns)
            if (
                number(query["event_state"]["at"]) is None
                or query.get("where")
                or query.get("group_by")
                or "compare_to" in query
            ):
                raise ValueError(
                    "event_state needs full unfiltered event rows, no group_by/compare_to and a numeric cutoff"
                )
    needs = plan.get("needs", [])
    if not isinstance(needs, list) or len(needs) > 12:
        raise ValueError("Invalid calculation needs")
    for need in needs:
        if (
            not isinstance(need, dict)
            or set(need) != {"need", "status", "reason", "queries"}
            or not isinstance(need["need"], str)
            or not 1 <= len(need["need"]) <= 300
            or need["status"]
            not in {
                "planned",
                "unsupported",
                "missing_context",
                "missing_field",
                "ambiguous_identity",
                "parse_error",
                "invalid_plan",
            }
            or not isinstance(need["reason"], str)
            or not 1 <= len(need["reason"]) <= 700
            or not isinstance(need["queries"], list)
            or any(not isinstance(name, str) or name not in names for name in need["queries"])
        ):
            raise ValueError("Invalid calculation need")
        if (need["status"] == "planned") != bool(need["queries"]):
            raise ValueError("Planned needs must name queries; abstentions must not")
    return plan


def compare_metrics(result, spec, previous):
    baseline = previous.get(spec["query"], {})
    groups = baseline.get("groups", [])
    baseline_metrics = groups[0]["metrics"] if len(groups) == 1 and baseline.get("complete") else {}
    differences, unavailable = {}, []
    for metric in spec["metrics"]:
        value, mean = number(result.get(metric)), number(baseline_metrics.get(metric))
        if value is None or mean is None or mean == 0:
            differences[metric] = None
            unavailable.append(metric)
        else:
            differences[metric] = (value - mean) / abs(mean) * 100
    largest = None
    if not unavailable:
        greatest = max(abs(value) for value in differences.values())
        largest = [key for key, value in differences.items() if abs(value) == greatest]
    return {
        "baseline_query": spec["query"],
        "formula": "100 * (value - baseline) / abs(baseline)",
        "percent_deviations": differences,
        "largest_absolute_deviation_metrics": largest,
        "unavailable_metrics": unavailable,
        "complete": not unavailable,
    }


def validate_expression(expr, metrics, previous, depth=0):
    if depth > 8 or not isinstance(expr, dict):
        raise ValueError("Invalid arithmetic expression")
    if set(expr) == {"constant"} and number(expr["constant"]) is not None:
        return
    if set(expr) == {"metric"} and isinstance(expr["metric"], str) and expr["metric"] in metrics:
        return
    if (
        set(expr) == {"query", "metric"}
        and isinstance(expr["query"], str)
        and expr["query"] in previous
        and isinstance(expr["metric"], str)
    ):
        return
    op = expr.get("op")
    if (
        set(expr) != {"op", "args"}
        or not isinstance(op, str)
        or op not in {"add", "subtract", "multiply", "divide", "abs", "max", "min"}
    ):
        raise ValueError("Unknown arithmetic expression")
    args = expr["args"]
    if (
        not isinstance(args, list)
        or not 1 <= len(args) <= 12
        or (op == "abs" and len(args) != 1)
        or (op in {"add", "subtract", "multiply", "divide"} and len(args) != 2)
    ):
        raise ValueError("Invalid arithmetic arity")
    for arg in args:
        validate_expression(arg, metrics, previous, depth + 1)


def matches(row, conditions):
    unknown = False
    for condition in conditions:
        actual, expected, op = row.get(condition["field"], MISSING), condition["value"], condition["op"]
        if op == "exists":
            accepted = (actual is not MISSING and actual is not None) == expected
        elif actual is MISSING or actual is None:
            unknown = True
            continue
        elif op in {"eq", "ne"}:
            equal = actual == expected and (
                not isinstance(actual, bool) and not isinstance(expected, bool) or type(actual) is type(expected)
            )
            accepted = equal if op == "eq" else not equal
        elif op in {"contains", "not_contains"}:
            if not isinstance(actual, str) or not isinstance(expected, str):
                unknown = True
                continue
            accepted = expected in actual if op == "contains" else expected not in actual
        elif op == "in":
            if not isinstance(expected, list):
                raise ValueError("in filter requires a list")
            accepted = any(
                actual == item
                and (not isinstance(actual, bool) and not isinstance(item, bool) or type(actual) is type(item))
                for item in expected
            )
        else:
            a, b = number(actual), number(expected)
            if a is None or b is None:
                if isinstance(actual, str) and isinstance(expected, str):
                    a, b = actual, expected
                else:
                    unknown = True
                    continue
            accepted = {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b}[op]
        if not accepted:
            return False
    return None if unknown else True


def formula(expression, metrics, previous, depth=0):
    if depth > 8 or not isinstance(expression, dict):
        raise ValueError("Invalid arithmetic expression")
    if set(expression) == {"constant"}:
        value = number(expression["constant"])
        if value is None:
            raise ValueError("Invalid arithmetic constant")
        return value
    if set(expression) == {"metric"}:
        return number(metrics.get(expression["metric"]))
    if set(expression) == {"query", "metric"}:
        groups = previous.get(expression["query"], {}).get("groups", [])
        if len(groups) != 1:
            return None
        return number(groups[0]["metrics"].get(expression["metric"]))
    if set(expression) != {"op", "args"} or expression["op"] not in {
        "add",
        "subtract",
        "multiply",
        "divide",
        "abs",
        "max",
        "min",
    }:
        raise ValueError("Unknown arithmetic operation")
    args = expression["args"]
    if not isinstance(args, list) or not 1 <= len(args) <= 12:
        raise ValueError("Invalid arithmetic arguments")
    values = [formula(item, metrics, previous, depth + 1) for item in args]
    if any(value is None for value in values):
        return None
    op = expression["op"]
    if op == "abs" and len(values) == 1:
        return abs(values[0])
    if op in {"max", "min"}:
        return max(values) if op == "max" else min(values)
    if len(values) != 2:
        raise ValueError("Binary arithmetic requires two arguments")
    a, b = values
    if op == "add":
        return a + b
    if op == "subtract":
        return a - b
    if op == "multiply":
        return a * b
    return a / b if b else None


def evaluate(plan, pools, parse_errors):
    validate_plan(plan)
    results, ledger, ledger_size = {}, [], 0
    with localcontext() as context:
        context.prec = 60
        for query in plan["queries"]:
            projected, unknown_rows, visited, missing_order = [], 0, 0, 0
            for source in pools[query["parser"]]:
                for _, actual_path in walk_rows(source["data"], query.get("rows_path", [])):
                    visited += 1
                    if visited > MAX_ROWS:
                        raise ValueError("Calculation row limit exceeded")
                    values, paths, raw_values = {}, {}, {}
                    for alias, column in query["columns"].items():
                        scope = column.get("scope", "row")
                        parent = (
                            []
                            if scope == "record"
                            else actual_path[: -column.get("up", 1)]
                            if scope == "ancestor" and column.get("up", 1)
                            else actual_path
                        )
                        path = parent + column.get("path", [])
                        values[alias] = lookup(source["data"], path)
                        if isinstance(values[alias], (dict, list)) or (
                            isinstance(values[alias], str) and len(values[alias]) > 10000
                        ):
                            raise ValueError("Calculation columns must be bounded scalar fields")
                        raw_values[alias] = values[alias]
                        if "bucket_width" in column:
                            numeric = number(values[alias])
                            values[alias] = (
                                (numeric / number(column["bucket_width"])).to_integral_value(rounding=ROUND_FLOOR)
                                if numeric is not None
                                else MISSING
                            )
                        paths[alias] = path
                    accepted = matches(values, query.get("where", []))
                    ledger.append(
                        {
                            "query": query["name"],
                            "source": source["source"],
                            "filter_match": accepted,
                            "fields": {
                                alias: {
                                    "path": paths[alias],
                                    "value": None if raw_values[alias] is MISSING else raw_values[alias],
                                    "projected_value": None if value is MISSING else value,
                                    "bucket_width": query["columns"][alias].get("bucket_width"),
                                    "missing": value is MISSING,
                                }
                                for alias, value in values.items()
                            },
                        }
                    )
                    ledger_size += len(json.dumps(serial(ledger[-1]), ensure_ascii=False))
                    if ledger_size > 16_000_000:
                        raise ValueError("Calculation audit size limit exceeded")
                    if accepted is None:
                        unknown_rows += 1
                    elif accepted:
                        projected.append((values, source["source"], paths))
            if query.get("order_by"):
                field = query["order_by"]
                missing_order = sum(
                    row[0].get(field, MISSING) is MISSING or row[0].get(field) is None for row in projected
                )

                def order(row):
                    value = row[0].get(field, MISSING)
                    numeric = number(value)
                    return (
                        value is MISSING or value is None,
                        0 if numeric is not None else 1,
                        numeric if numeric is not None else str(value),
                    )

                projected.sort(key=order)
            grouped = {}
            for row in projected:
                values = [row[0].get(alias, MISSING) for alias in query.get("group_by", [])]
                if any(value is MISSING or value is None or isinstance(value, (dict, list)) for value in values):
                    unknown_rows += 1
                    continue
                key = json.dumps(serial(values), ensure_ascii=False)
                grouped.setdefault(key, {"key": dict(zip(query.get("group_by", []), values, strict=True)), "rows": []})[
                    "rows"
                ].append(row)
                if len(grouped) > MAX_GROUPS:
                    raise ValueError("Calculation group limit exceeded")
            outputs = []
            for group in grouped.values():
                result, missing = {}, {}
                bounds_missing = False
                if "required_order_bounds" in query:
                    observed = [number(row[0].get(query["order_by"])) for row in group["rows"]]
                    bounds = [number(value) for value in query["required_order_bounds"]]
                    bounds_missing = (
                        any(value is None for value in observed)
                        or not observed
                        or min(observed) != bounds[0]
                        or max(observed) != bounds[1]
                    )
                for metric in query["metrics"]:
                    selected, uncertain = [], 0
                    for values, source, paths in group["rows"]:
                        accepted = matches(values, metric.get("where", []))
                        if accepted is None:
                            uncertain += 1
                        elif accepted:
                            selected.append(values.get(metric.get("field"), MISSING))
                    op = metric["op"]
                    absent = (
                        uncertain + sum(value is MISSING or value is None for value in selected)
                        if op != "count"
                        else uncertain
                    )
                    if op in {"sum", "mean", "min", "max"}:
                        numbers = [number(value) for value in selected]
                        absent = uncertain + sum(value is None for value in numbers)
                        if not numbers or absent:
                            value = None
                        else:
                            value = (
                                sum(numbers)
                                if op == "sum"
                                else sum(numbers) / len(numbers)
                                if op == "mean"
                                else min(numbers)
                                if op == "min"
                                else max(numbers)
                            )
                    elif op == "count":
                        value = len(selected) if not uncertain else None
                    elif absent:
                        value = None
                    elif op in {"first", "last"}:
                        value = selected[0 if op == "first" else -1] if selected else None
                    else:
                        distinct = list(
                            dict.fromkeys(json.dumps(serial(value), ensure_ascii=False) for value in selected)
                        )
                        value = (
                            len(distinct) if op == "distinct_count" else [json.loads(value) for value in distinct[:40]]
                        )
                        if op == "values" and len(distinct) > 40:
                            absent += len(distinct) - 40
                            value = None
                    missing[metric["name"]] = absent
                    result[metric["name"]] = value
                incomplete = bool(
                    unknown_rows
                    or missing_order
                    or parse_errors.get(query["parser"], 0)
                    or any(missing.values())
                    or any(value is None for value in result.values())
                    or bounds_missing
                )
                # Unknown filtered rows can affect every group: do not present partial numbers as complete.
                if unknown_rows or missing_order or bounds_missing or parse_errors.get(query["parser"], 0):
                    result = {alias: None for alias in result}
                for alias, expression in query.get("derived", {}).items():
                    result[alias] = formula(expression, result, results)
                    if result[alias] is None:
                        incomplete = True
                output = {
                    "key": group["key"],
                    "metrics": result,
                    "complete": not incomplete,
                    "missing_values": missing,
                    "input_rows": len(group["rows"]),
                    "required_bounds_missing": bounds_missing,
                }
                if "compare_to" in query:
                    output["comparison"] = compare_metrics(result, query["compare_to"], results)
                    output["complete"] = output["complete"] and output["comparison"]["complete"]
                outputs.append(output)
            sort = query.get("sort_by")
            if sort:
                present = [row for row in outputs if number(row["metrics"].get(sort)) is not None]
                absent = [row for row in outputs if number(row["metrics"].get(sort)) is None]
                outputs = (
                    sorted(
                        present, key=lambda row: number(row["metrics"][sort]), reverse=query.get("descending", False)
                    )
                    + absent
                )
            results[query["name"]] = {
                "groups": outputs,
                "display_limit": query.get("limit", 40),
                "visited_rows": visited,
                "matched_rows": len(projected),
                "unknown_filter_rows": unknown_rows,
                "missing_order_rows": missing_order,
                "parse_errors": parse_errors.get(query["parser"], 0),
                "scope": "only declared parser, paths and filters; JSON uses the first complete root value at the start of each original, excluding trailing prose",
                "complete": bool(outputs) and all(row["complete"] for row in outputs),
            }
            if "event_state" in query:
                from memory_service.event_state import reduce_events

                state = reduce_events(
                    projected, query["event_state"], number, MISSING, parse_errors.get(query["parser"], 0)
                )
                if len(state["groups"]) > MAX_GROUPS:
                    raise ValueError("Calculation group limit exceeded")
                results[query["name"]].update(groups=state.pop("groups"), complete=state["complete"], event_state=state)
    data = serial({"results": results, "inputs": ledger})
    if len(json.dumps(data, ensure_ascii=False)) > 16_000_000:
        raise ValueError("Calculation audit size limit exceeded")
    return data


def evaluate_partial(plan, pools, parse_errors):
    """Execute independent valid needs even if another query exceeds an execution bound.

    Replaying at most four successful query prefixes preserves dependency semantics
    while keeping the strict evaluator shared with offline tooling.
    """
    validate_plan(plan)
    successful, failures = [], {}
    audit = {"results": {}, "inputs": []}
    for query in plan["queries"]:
        try:
            attempt = evaluate({"queries": successful + [query]}, pools, parse_errors)
        except (ValueError, ArithmeticError, TypeError, KeyError) as error:
            failures[query["name"]] = {"status": "execution_failed", "reason": str(error)[:300]}
            continue
        successful.append(query)
        audit = attempt
    audit["query_failures"] = {**plan.get("planning_failures", {}), **failures}
    needs = []
    for need in plan.get("needs", []):
        resolved = dict(need)
        if need["status"] == "planned":
            completed = [name for name in need["queries"] if audit["results"].get(name, {}).get("complete")]
            resolved["completed_queries"] = completed
            resolved["status"] = (
                "completed" if len(completed) == len(need["queries"]) else "partial" if completed else "incomplete"
            )
        needs.append(resolved)
    audit["needs"] = needs
    return audit
