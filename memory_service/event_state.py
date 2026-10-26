"""Bounded event reduction by entity ID, with explicit coverage assumptions."""

from collections import Counter


def validate(spec, columns):
    required = {"id", "owner", "type", "time", "action", "creates", "destroys", "changes", "at", "target_types"}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError("Invalid event_state fields")
    for field in ("id", "owner", "type", "time", "action"):
        if not isinstance(spec[field], str) or spec[field] not in columns:
            raise ValueError("event_state must name projected columns")
    actions = []
    for field in ("creates", "destroys", "changes"):
        if (
            not isinstance(spec[field], list)
            or len(spec[field]) > 8
            or any(not isinstance(v, str) or not 1 <= len(v) <= 100 for v in spec[field])
        ):
            raise ValueError("Invalid event action values")
        actions.extend(spec[field])
    if not spec["creates"] or not spec["destroys"] or len(actions) != len(set(actions)):
        raise ValueError("Event action sets must be disjoint")
    if (
        not isinstance(spec["target_types"], list)
        or not 1 <= len(spec["target_types"]) <= 12
        or any(not isinstance(v, str) or not 1 <= len(v) <= 100 for v in spec["target_types"])
    ):
        raise ValueError("Invalid target event types")


def reduce_events(rows, spec, number, missing, parse_errors=0):
    state, built, lost, keys, errors = {}, Counter(), Counter(), set(), Counter()
    ordered = []
    actions = set(spec["creates"] + spec["destroys"] + spec["changes"])
    targets = set(spec["target_types"])
    cutoff = number(spec["at"])

    def issue(reason, kind):
        if kind in targets or kind is missing or kind is None:
            errors[reason] += 1

    for index, row in enumerate(rows):
        value = row[0]
        action = value.get(spec["action"], missing)
        if action not in actions:
            continue
        time = number(value.get(spec["time"], missing))
        if time is None:
            issue("missing_event_time", value.get(spec["type"], missing))
        elif time <= cutoff:
            ordered.append((time, index, value))
    for _, _, value in sorted(ordered, key=lambda item: (item[0], item[1])):
        entity = value.get(spec["id"], missing)
        kind, owner = value.get(spec["type"], missing), value.get(spec["owner"], missing)
        action = value[spec["action"]]
        if entity is missing or entity is None or isinstance(entity, bool):
            issue("missing_entity_id", kind)
            continue
        identity = (type(entity).__name__, str(entity))
        prior = state.get(identity)
        if action in spec["creates"]:
            if not isinstance(kind, str) or not isinstance(owner, str) or not kind or not owner:
                issue("missing_creation_owner_or_type", kind)
                continue
            if prior:
                if prior != (owner, kind, True):
                    issue("conflicting_or_reused_entity_id", kind)
                continue
            state[identity] = (owner, kind, True)
            built[(owner, kind)] += 1
            keys.add((owner, kind))
        elif action in spec["destroys"]:
            if not prior:
                issue("destruction_without_observed_creation", kind)
                continue
            old_owner, old_kind, alive = prior
            if not alive:
                # Repeated identical destruction reports are idempotent.
                continue
            if isinstance(owner, str) and owner != old_owner or isinstance(kind, str) and kind != old_kind:
                issue("destruction_identity_mismatch", old_kind)
                continue
            state[identity] = (old_owner, old_kind, False)
            lost[(old_owner, old_kind)] += 1
        else:
            if not prior or not prior[2]:
                issue("transition_without_live_entity", kind)
                continue
            if not isinstance(kind, str) or not isinstance(owner, str) or not kind or not owner:
                issue("missing_transition_owner_or_type", kind)
                continue
            state[identity] = (owner, kind, True)
            keys.add((owner, kind))
    alive = Counter((owner, kind) for owner, kind, exists in state.values() if exists)
    if parse_errors:
        errors["source_parse_errors"] = parse_errors
    outputs = []
    for owner, kind in sorted(keys):
        if kind not in targets:
            continue
        outputs.append(
            {
                "key": {"owner": owner, "type": kind},
                "metrics": {
                    "built": built[(owner, kind)],
                    "alive": None if errors else alive[(owner, kind)],
                    "lost": None if errors else lost[(owner, kind)],
                },
                "complete": not errors,
            }
        )
    return {
        "groups": outputs,
        "complete": bool(outputs) and not errors,
        "errors": dict(errors),
        "processed_events": len(ordered),
        "coverage_assumption": "Counts describe observed creations and ID-linked transitions only; entities present before the log or omitted events cannot be ruled out. A morph changes current type, not the original construction counter.",
        "at": spec["at"],
    }
