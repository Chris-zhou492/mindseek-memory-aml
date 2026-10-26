"""Literal action bindings and game-clock order; neither roster nor causal inference."""

import re

NAME = r"[A-Z]\.[A-Za-z](?:[A-Za-z'.-]*[A-Za-z])?"
PERIODS = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "OT": 5}


def primary_description(description):
    """A conversion attempt following a touchdown is a separate, untimed subevent."""
    td = re.search(r"\bTOUCHDOWN\b", description, re.I)
    if td:
        conversion = re.search(rf"{NAME}\s+extra point\b|TWO[- ]POINT CONVERSION", description[td.end():], re.I)
        if conversion:
            return description[:td.end() + conversion.start()].rstrip()
    return description


def bind_roles(description, offset):
    """Return named role evidence at exact document offsets; unknown roles stay absent."""
    roles, sources = {}, []

    def bind(role, match, group=1):
        if match is None:
            return
        name = match[group]
        if name not in roles.setdefault(role, []):
            roles[role].append(name)
            sources.append({"role": role, "name": name,
                            "start": offset + match.start(group), "end": offset + match.end(group)})

    passer = re.search(rf"({NAME})\s+pass\b", description)
    sacked = re.search(rf"({NAME})\s+sacked\b", description)
    rusher = re.search(rf"({NAME})\s+(?:left\b|right\b|up the middle\b|scrambles\b|kneels\b)", description)
    bind("passer", passer)
    bind("sacked_player", sacked)
    bind("rusher", rusher)
    if passer:
        target = re.search(rf"\b(?:to|intended for) ({NAME})", description[passer.end():])
        if target:
            # Re-run at the full-string position to keep exact capture offsets.
            target = re.compile(rf"\b(?:to|intended for) ({NAME})").search(description, passer.end())
            bind("target", target)
            if not re.search(r"\b(?:incomplete|INTERCEPTED|spikes)\b", description, re.I):
                bind("receiver", target)
    bind("interceptor", re.search(rf"\bINTERCEPTED by ({NAME})", description))
    bind("recoverer", re.search(rf"\bRECOVERED by (?:[A-Z0-9]{{2,4}}-)?({NAME})", description))
    forcer = re.search(r"\bFUMBLES\s*\(([^)]+)\)", description)
    if forcer:
        for match in re.compile(NAME).finditer(description, forcer.start(1), forcer.end(1)):
            bind("fumble_forcer", match, 0)
    if sacked:
        tacklers = re.search(r"\([^)]*\)", description[sacked.end():])
        if tacklers:
            start = sacked.end() + tacklers.start()
            for match in re.compile(NAME).finditer(description, start, sacked.end() + tacklers.end()):
                bind("sacker", match, 0)
    kicker = re.search(rf"({NAME})\s+(?:kicks|\d+ yard field goal)\b", description)
    bind("kicker", kicker)
    bind("punter", re.search(rf"({NAME})\s+punts\b", description))
    primary = passer or sacked or rusher or kicker or re.search(rf"({NAME})\s+punts\b", description)
    return primary[1] if primary else None, roles, sources


def annotate_chronology(events):
    """Regulation elapsed time assumes the supported format's 15-minute quarters."""
    previous_ot_clock, ambiguous_ot = None, False
    for event in events:
        if event["quarter"] == "OT":
            if previous_ot_clock is not None and event["clock_seconds_remaining"] > previous_ot_clock:
                ambiguous_ot = True
            previous_ot_clock = event["clock_seconds_remaining"]
    for event in events:
        period = PERIODS[event["quarter"]]
        remaining = event["clock_seconds_remaining"]
        event["clock_order_key"] = [period, -remaining]
        event["elapsed_regulation_seconds"] = (period - 1) * 900 + 900 - remaining if period <= 4 else None
        event["time_order_resolved"] = not (period == 5 and ambiguous_ot)
    ordered = sorted(events, key=lambda e: (e["clock_order_key"], e["start"]))
    for index, event in enumerate(ordered, 1):
        # Display rank only: same clock does not establish strict temporal order.
        event["display_order"] = index
    first_td = next((e for e in ordered if e["status"] == "valid" and "touchdown" in e["kinds"]
                     and set(e["kinds"]) & {"run", "pass"}
                     and not set(e["kinds"]) & {"interception", "fumble_lost", "kickoff", "punt"}
                     and e["time_order_resolved"]), None)
    for event in events:
        if not first_td or not event["time_order_resolved"]:
            relation = "unknown"
        elif event is first_td:
            relation = "first_offensive_td"
        elif event["clock_order_key"] == first_td["clock_order_key"]:
            relation = "same_clock_order_unknown"
        else:
            relation = "before" if event["clock_order_key"] < first_td["clock_order_key"] else "after"
        event["relative_to_first_offensive_td"] = relation
    return {
        "first_valid_offensive_touchdown_order": first_td["display_order"] if first_td else None,
        "overtime_order_ambiguous": ambiguous_ot,
        "rules": "Within each quarter the larger remaining clock is earlier. Equal clocks do not prove strict "
                 "order; display rank breaks ties by source offset only. Elapsed seconds assume 15-minute "
                 "regulation quarters; overtime duration/period identity is unknown. Before is not proof of causation.",
    }
