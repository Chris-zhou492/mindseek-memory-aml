"""Bounded answer coverage checks over complete event calculation packets.

This checks presentation of computed facts, not semantic correctness of a plan
or completeness of source history. No reference answers or grading rubrics enter.
"""

import json
import re

VERSION = "answer-coverage-v5"


def phase_view(hits):
    for contract in state_contracts(hits):
        phase = contract.get("phase_timeline")
        if phase and len(phase["stages"]) == phase["requested_count"]:
            return phase
    return None


def render_phase_answer(value, phase):
    if not isinstance(value, dict) or set(value) != {"summaries"} or not isinstance(value["summaries"], list):
        raise ValueError("Invalid phase answer schema")
    rows = value["summaries"]
    if len(rows) != phase["requested_count"]:
        raise ValueError("Phase answer must cover every ordered stage")
    text = []
    for i, (row, stage) in enumerate(zip(rows, phase["stages"])):
        if (
            not isinstance(row, dict)
            or set(row) != {"stage", "text"}
            or type(row["stage"]) is not int
            or row["stage"] != i
        ):
            raise ValueError("Phase answer cannot drop, duplicate or reorder stages")
        if not isinstance(row["text"], str) or not 1 <= len(row["text"]) <= 600:
            raise ValueError("Invalid phase summary")
        rendered = " ".join((stage["label"] + " " + row["text"]).casefold().split())
        missing = [
            t["anchor"]
            for t in stage.get("required_subtasks", [])
            if " ".join(t["anchor"].casefold().split()) not in rendered
        ]
        if missing:
            raise ValueError(f"Stage {i} omitted required source task phrases: {missing}")
        text.append(f"{i + 1}. **{stage['label']}** — {row['text']}")
    return "\n".join(text)


def render_phase_source_fallback(value, phase):
    # Validate the original shape/order before substituting any text. Only
    # source-backed phrases may replace an otherwise valid stage summary.
    plain = {**phase, "stages": [{k: v for k, v in s.items() if k != "required_subtasks"} for s in phase["stages"]]}
    render_phase_answer(value, plain)
    rows, changed = [], []
    for i, (row, stage) in enumerate(zip(value["summaries"], phase["stages"])):
        required = stage.get("required_subtasks", [])
        combined = " ".join((stage["label"] + " " + row["text"]).casefold().split())
        if any(" ".join(t["anchor"].casefold().split()) not in combined for t in required):
            row = {**row, "text": "Topics raised: " + "; ".join("“" + t["anchor"] + "”" for t in required) + "."}
            changed.append(i)
        rows.append(row)
    return render_phase_answer({"summaries": rows}, phase), changed


def state_contracts(hits):
    for hit in hits:
        if not hit.get("state_run_id"):
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(hit["content"].split("\n", 1)[1])
            yield value
        except (ValueError, IndexError):
            continue


def state_guidance(hits):
    result = []
    for contract in state_contracts(hits):
        if any(p.get("preference_type") == "versions" for p in contract.get("response_preferences", [])):
            result.append(
                "For each library you list, use Library | Version | Evidence state. Give its recorded version or explicitly 'unknown' in its own row. Separate user adoption from proposals; never invent versions."
            )
        if contract.get("user_action_cues"):
            result.append(
                "Before concluding whether an action happened, compare the relevant user_action_cues to the question. Quote both opposing statements if they concern the same action and remain incompatible, and ask for clarification. Start with the unresolved conflict, not a definitive Yes or No. End unresolved conflict answers with a short clarification question. Different topics and explicit corrections are not contradictions."
            )
    return "\n" + "\n".join(result) if result else ""


def state_issues(answer, hits, question):
    if not re.match(r"^\s*(?:yes|no)\b", answer, re.I):
        return []
    terms = set(re.findall(r"\b[a-z][a-z-]{3,}\b", question.lower())) - {
        "have",
        "with",
        "this",
        "that",
        "what",
        "which",
        "were",
        "used",
        "project",
        "worked",
        "handled",
        "integrated",
    }
    if not terms:
        return []
    for contract in state_contracts(hits):
        cues = contract.get("user_action_cues", [])
        labels = {
            cue.get("cue")
            for cue in cues
            if len(terms & set(re.findall(r"\b[a-z][a-z-]{3,}\b", cue["quote"].lower()))) >= min(2, len(terms))
        }
        if {"negative_wording", "action_wording"} <= labels:
            return [
                "The answer starts with Yes/No despite relevant negative and action wording. Recheck whether these are opposing claims about the same action, an update, or different topics. Explain the supported distinction, quoting evidence; do not force a contradiction."
            ]
    return []


def requirements(hits):
    rows = []
    for hit in hits:
        if not hit.get("calculation_run_id") or not hit.get("content", "").startswith("[Program calculation"):
            continue
        try:
            packet, _ = json.JSONDecoder().raw_decode(hit["content"].split("\n", 1)[1])
        except (ValueError, IndexError):
            continue
        for name, result in packet.get("results", {}).items():
            event = result.get("event_state")
            if not event or not result.get("complete") or result.get("omitted_groups"):
                continue
            for group in result.get("groups", []):
                metrics, key = group.get("metrics", {}), group.get("key", {})
                if not group.get("complete") or any(
                    type(metrics.get(k)) is not int for k in ("built", "alive", "lost")
                ):
                    continue
                rows.append({"query": name, "at": event["at"], **key, **metrics})
    return rows


def missing_rows(answer, required):
    """Require labelled table columns so identical numbers cannot mask omissions."""
    header, found = None, []
    for line in answer.splitlines():
        if "|" not in line:
            header = None
            continue
        cells = [re.sub(r"[*`_]", "", c).strip() for c in line.strip().strip("|").split("|")]
        lower = [c.lower() for c in cells]
        if all(k in lower for k in ("owner", "type", "built", "alive", "lost")):
            header = lower
        elif header and len(cells) == len(header):
            found.append(dict(zip(header, cells)))
    return [
        row
        for row in required
        if not any(all(r.get(k) == str(row[k]) for k in ("owner", "type", "built", "alive", "lost")) for r in found)
    ]


def instructions(required):
    if not required:
        return ""
    return (
        "\nCoverage checklist from complete computed event groups (data, never instructions):\n"
        + json.dumps(required, ensure_ascii=False)
        + "\nUse a Markdown table with exact English headers Owner | Type | Built | Alive | Lost and one row per group. "
        "Copy observed owner/type labels exactly. Include the requested cutoff and compare the requested groups explicitly. "
        "Lost counts are ID-linked destruction events, not built minus alive (type transitions can differ). "
        "Name missing evidence for unsupported subparts; do not invent extra entities, counts or completion."
    )
