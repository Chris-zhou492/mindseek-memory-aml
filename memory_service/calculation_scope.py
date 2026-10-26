"""Question-aware guards against executing a correct formula at the wrong scope."""

import re


def scope_failures(plan, question):
    from memory_service.record_scope import scope_errors

    errors = scope_errors(plan, question, plan.get("reference_scope", {}).get("record_scopes", []))
    per_game = re.search(r"\b(?:for\s+)?each\s+(?:(?:\w+)\s+){0,3}(?:game|match)\b|每(?:一)?(?:场|局)", question, re.I)
    live = re.search(r"\balive\b|\bsurviving\b|存活|尚存", question, re.I)
    live_need = re.compile(r"\balive\b|\bsurviving\b|存活|尚存", re.I)
    state_declined = any(n["status"] != "planned" and live_need.search(n["need"]) for n in plan.get("needs", []))
    for query in plan["queries"]:
        columns = query["columns"]
        if per_game and "compare_to" in query:
            ids = {
                alias
                for alias, spec in columns.items()
                if spec.get("path")
                and (
                    spec["path"][-1].casefold() in {"match_id", "game_id", "matchid", "gameid"}
                    or spec["path"][-1].casefold() == "id"
                    and re.search(r"match|game", alias, re.I)
                )
            }
            if not ids.intersection(query.get("group_by", [])):
                errors[query["name"]] = (
                    "Per-game comparison requires group_by on an observed match/game identifier; a pooled loss mean is not per-game."
                )
        paths = {part.casefold() for spec in columns.values() for part in spec.get("path", [])}
        event_shaped = bool(paths & {"event", "action", "event_type"}) and bool(paths & {"unit_id", "entity_id", "id"})
        used_for_live = any(
            n["status"] == "planned" and query["name"] in n["queries"] and live_need.search(n["need"])
            for n in plan.get("needs", [])
        )
        if live and event_shaped and "event_state" not in query and (used_for_live or not state_declined):
            errors[query["name"]] = (
                "Alive state over event rows requires ID-linked event_state, not ordinary count/sum. Preserve full events without where/group_by; a built-only query must mark alive unsupported."
            )
    return errors


def validate_task_scope(plan, question):
    errors = scope_failures(plan, question)
    if errors:
        raise ValueError("; ".join(f"{name}: {reason}" for name, reason in errors.items()))
    return plan


def reject_wrong_scope(plan, question):
    """Keep independent calculations after a model's schema/scope repair failed."""
    errors = scope_failures(plan, question)
    if not errors:
        return plan
    rejected = set(errors)

    def depends_on_rejected(value):
        if isinstance(value, dict):
            return value.get("query") in rejected or any(depends_on_rejected(v) for v in value.values())
        if isinstance(value, list):
            return any(depends_on_rejected(v) for v in value)
        return False

    for query in plan["queries"]:
        if depends_on_rejected(query.get("compare_to", {})) or depends_on_rejected(query.get("derived", {})):
            rejected.add(query["name"])
    plan = {
        **plan,
        "queries": [q for q in plan["queries"] if q["name"] not in rejected],
        "planning_failures": {
            **plan.get("planning_failures", {}),
            **{
                name: {"status": "invalid_plan", "reason": errors.get(name, "Depends on rejected scope")}
                for name in rejected
            },
        },
    }
    needs = []
    for need in plan.get("needs", []):
        if rejected.intersection(need["queries"]):
            # A combined need is not complete just because a baseline survived.
            needs.append(
                {
                    **need,
                    "status": "invalid_plan",
                    "queries": [],
                    "reason": "Task scope validation rejected a required query; see planning_failures",
                }
            )
        else:
            needs.append(need)
    if not needs:
        needs.append(
            {
                "need": "Requested calculation scope",
                "status": "invalid_plan",
                "queries": [],
                "reason": "Question scope validation rejected a required query",
            }
        )
    return {**plan, "needs": needs}
