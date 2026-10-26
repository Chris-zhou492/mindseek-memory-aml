"""Source-bound football play index and deterministic evidence packets (opt-in)."""

import hashlib
import json
import re
from collections import Counter

from memory_service.game_event_semantics import (
    NAME,
    annotate_chronology,
    bind_roles,
    primary_description,
)

VERSION = "football-clock-events-v3.1"
CLOCK = re.compile(r"(?m)^(\d{1,2}):([0-5]\d) - (1st|2nd|3rd|4th|OT)[ \t]*\r?$")
DRIVE = re.compile(
    r"(?m)^([^\n\r]+)\r?\n([^\n\r]+)\r?\n\d+ plays?,[^\n\r]*"
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS structured_game_documents (
 document_id TEXT PRIMARY KEY REFERENCES source_documents(id) ON DELETE CASCADE,
 user_id TEXT NOT NULL, version TEXT NOT NULL, fingerprint TEXT NOT NULL, audit TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS structured_game_events (
 id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
 user_id TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL, payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_structured_game_user ON structured_game_events(user_id, document_id, start);
"""


def parse_log(text):
    """Only explicit clocked paragraphs; source order never assigns an event to a quarter."""
    clocks = list(CLOCK.finditer(text))
    drives = list(DRIVE.finditer(text))
    if not re.search(r"play[- ]by[- ]play", text, re.I) or len(clocks) < 2 or not drives:
        return [], {"supported": False, "reason": "No supported clocked play-by-play with drive headers"}
    events, errors = [], []
    for block in re.finditer(r"\S[\s\S]*?(?=\r?\n[ \t]*\r?\n|\Z)", text):
        quote = block.group()
        times = list(CLOCK.finditer(quote))
        if not times:
            continue
        if len(times) != 1 or int(times[0][1]) * 60 + int(times[0][2]) > 900:
            errors.append({"start": block.start(), "end": block.end(), "reason": "ambiguous or invalid clock"})
            continue
        clock = times[0]
        title = quote[:clock.start()].strip()
        raw_description = quote[clock.end():]
        description_offset = block.start() + clock.end() + len(raw_description) - len(raw_description.lstrip())
        description = primary_description(raw_description.strip())
        # A clock without a play description is not an observed event.
        if not description:
            errors.append({"start": block.start(), "end": block.end(), "reason": "missing play description"})
            continue
        cleaned = re.sub(r"^(?:\([^)]*\)\s*)+", "", description)
        actor_match = re.match(NAME, cleaned)
        actor = actor_match.group() if actor_match else None
        bound_actor, roles, role_sources = bind_roles(description, description_offset)
        actor = bound_actor or actor
        lower = description.lower()
        kinds = []
        for kind, present in (
            ("sack", "sacked" in lower),
            ("interception", "intercepted" in lower),
            ("fumble", "fumbles" in lower),
            ("fumble_lost", "fumbles" in lower and "opp fumble recovery" in title.lower()),
            ("run", bool(re.search(r"\bRun\b", title)) or "scrambles" in lower or "rusher" in roles),
            ("pass", bool(re.search(r"\bpass\b", lower))),
            ("incompletion", "incomplete" in lower),
            ("touchdown", "touchdown" in lower),
            ("field_goal", "field goal" in lower),
            ("punt", bool(re.search(r"\bpunts\b", lower))),
            ("kickoff", title.lower() == "kickoff"),
            ("penalty", "penalty" in lower),
        ):
            if present:
                kinds.append(kind)
        status = "valid"
        if re.search(r"\bno play\b|\bnullified\b", description, re.I):
            status = "cancelled"
        elif re.search(r"\breversed\b|\boverturned\b|\bunder review\b", description, re.I):
            status = "review_required"
        gain = re.search(r"\bfor (-?\d+) yards?\b", description)
        loss = re.search(r"\bfor (?:a )?loss of (\d+) yards?\b", description)
        yards = int(gain[1]) if gain else -int(loss[1]) if loss else 0 if "no gain" in lower else None
        return_yards = yards if "interception" in kinds else None
        if "interception" in kinds or not set(kinds) & {"run", "pass", "sack"} or "incompletion" in kinds:
            yards = None
        prior_drives = [d for d in drives if d.start() < block.start()]
        possession = prior_drives[-1][1].strip() if prior_drives else None
        # Drive context is not a roster binding: e.g. a kickoff actor may be on the other team.
        down = re.search(r"(?m)^([1-4](?:st|nd|rd|th) & [^\n\r]+)", description)
        events.append({"start": block.start(), "end": block.end(), "quarter": clock[3],
                       "clock": f"{int(clock[1])}:{clock[2]}",
                       "clock_seconds_remaining": int(clock[1]) * 60 + int(clock[2]),
                       "possession_context": possession, "actor": actor,
                       "receiver": next(iter(roles.get("receiver", [])), None),
                       "target": next(iter(roles.get("target", [])), None),
                       "roles": roles, "role_sources": role_sources,
                       "kinds": kinds, "status": status, "yards": yards, "return_yards": return_yards,
                       "down_and_field": down[1] if down else None,
                       "quote": quote})
    chronology = annotate_chronology(events)
    return events, {"supported": True, "clock_lines": len(clocks), "parsed_events": len(events),
                    "chronology": chronology,
                    "unclassified_events": sum(not e["kinds"] for e in events),
                    "unparsed": errors, "recognized_clocks_complete": len(events) == len(clocks) and not errors,
                    "population_complete": False,
                    "scope": "Observed explicit clock lines only; not proof of complete game history. "
                             "Possession is nearest preceding drive header, not independently verified actor team. "
                             "No Play/nullified excluded from valid counts; review outcomes remain unresolved."}


def summarize(events):
    """Counts retain quarter/actor identity and excluded events; no pressure/blame inference."""
    counts = Counter()
    for event in events:
        for kind in event["kinds"]:
            if kind in {"sack", "interception", "fumble", "fumble_lost"}:
                counts[(event["quarter"], event["actor"], kind, event["status"])] += 1
    return [{"quarter": q, "actor": actor, "kind": kind, "status": status, "count": count}
            for (q, actor, kind, status), count in sorted(counts.items(), key=lambda p: str(p[0]))]


def segment_totals(events):
    """Explicit overlapping scopes; never sum fumble and fumble_lost as two turnovers."""
    outcomes = [e for e in events if set(e["kinds"]) & {"sack", "interception", "fumble"}]
    actors = sorted({e["actor"] for e in outcomes}, key=str)
    scopes = {"first_half": {"1st", "2nd"}, "second_half": {"3rd", "4th"},
              "fourth_quarter": {"4th"}, "all_observed": {"1st", "2nd", "3rd", "4th", "OT"}}
    rows = []
    for actor in actors:
        for scope, quarters in scopes.items():
            selected = [e for e in outcomes if e["actor"] == actor and e["quarter"] in quarters]
            valid = [e for e in selected if e["status"] == "valid"]
            sacks = sum("sack" in e["kinds"] for e in valid)
            interceptions = sum("interception" in e["kinds"] for e in valid)
            lost = sum("fumble_lost" in e["kinds"] for e in valid)
            rows.append({"actor": actor, "scope": scope, "valid_sacks": sacks,
                         "cancelled_sacks": sum("sack" in e["kinds"] and e["status"] == "cancelled" for e in selected),
                         "valid_interceptions": interceptions, "confirmed_fumbles_lost": lost,
                         "confirmed_turnovers": interceptions + lost,
                         "fumbles_without_confirmed_loss": sum("fumble" in e["kinds"] and "fumble_lost" not in e["kinds"] for e in valid),
                         "review_required_events": sum(e["status"] == "review_required" for e in selected)})
    return rows


class GameLogEvidence:
    def __init__(self):
        self.last_trace = None

    def index(self, store, user_id):
        """Lazily derive versioned indexes for existing raw user documents, never assistant summaries."""
        indexed = []
        with store._connect() as conn:
            conn.executescript(SCHEMA)
            docs = conn.execute("SELECT * FROM source_documents WHERE user_id=? AND role='user' ORDER BY id",
                                (user_id,)).fetchall()
            for doc in docs:
                fingerprint = hashlib.sha256(doc["content"].encode()).hexdigest()
                saved = conn.execute("SELECT * FROM structured_game_documents WHERE document_id=? AND user_id=?",
                                     (doc["id"], user_id)).fetchone()
                if saved and saved["version"] == VERSION and saved["fingerprint"] == fingerprint:
                    audit = json.loads(saved["audit"])
                else:
                    events, audit = parse_log(doc["content"])
                    conn.execute("DELETE FROM structured_game_events WHERE document_id=? AND user_id=?", (doc["id"], user_id))
                    for event in events:
                        ident = "play_" + hashlib.sha256(f"{doc['id']}:{event['start']}:{VERSION}".encode()).hexdigest()[:24]
                        payload = {k: v for k, v in event.items() if k != "quote"}
                        payload["id"] = ident
                        conn.execute("INSERT INTO structured_game_events VALUES(?,?,?,?,?,?)",
                                     (ident, doc["id"], user_id, event["start"], event["end"], json.dumps(payload)))
                    conn.execute("INSERT OR REPLACE INTO structured_game_documents VALUES(?,?,?,?,?)",
                                 (doc["id"], user_id, VERSION, fingerprint, json.dumps(audit)))
                if audit["supported"]:
                    events = [json.loads(r["payload"]) for r in conn.execute(
                        "SELECT payload FROM structured_game_events WHERE user_id=? AND document_id=? ORDER BY start",
                        (user_id, doc["id"]))]
                    for event in events:
                        event["quote"] = doc["content"][event["start"]:event["end"]]
                    indexed.append((dict(doc), events, audit))
        return indexed

    def __call__(self, store, user_id, query, options, evidence, top_k):
        self.last_trace = {"version": VERSION, "documents": []}
        if not re.search(r"\b(sacks?|turnovers?|quarter|touchdowns?|football|offensive|play.by.play)\b", query, re.I):
            self.last_trace["skipped"] = "no supported query concepts"
            return evidence
        packets = []
        refreshed_documents = set()
        for doc, events, audit in self.index(store, user_id):
            refreshed_documents.add(doc["id"])
            groups = [
                ("outcomes", [e for e in events if set(e["kinds"]) & {"sack", "interception", "fumble"}]),
                ("scoring", [e for e in events if set(e["kinds"]) & {"field_goal", "touchdown"}]),
                # Explicit ranking, no gold-derived yard threshold or named player selection.
                ("largest_gains", sorted([e for e in events if e["status"] == "valid" and e["yards"] is not None
                                          and set(e["kinds"]) & {"run", "pass"}],
                                         key=lambda e: (-e["yards"], e["start"]))[:24]),
            ]
            doc_trace = {"document_id": doc["id"], "audit": audit, "counts": summarize(events),
                         "segment_totals": segment_totals(events), "groups": []}
            self.last_trace["documents"].append(doc_trace)
            for label, selected in groups:
                if not selected:
                    continue
                header = {"document_id": doc["id"], "session_id": doc["session_id"], "kind": label,
                          "coverage": audit, "selected_events": len(selected)}
                if label == "outcomes":
                    header["counts_by_quarter_actor_status"] = summarize(events)
                    header["program_segment_totals"] = segment_totals(events)
                    header["counting_rules"] = (
                        "Scopes overlap: do not add their totals. first_half=Q1+Q2; fourth_quarter=Q4 only; "
                        "all_observed includes all parsed quarters/OT. Confirmed turnovers=valid interceptions "
                        "+ explicitly opponent-recovered fumbles, without counting fumble and fumble_lost twice. "
                        "Cancelled and unresolved-review events are excluded from valid totals."
                    )
                if label == "largest_gains":
                    header["selection"] = "Top 24 observed valid run/pass gains across both teams; not all plays or all player plays."
                parts, sources = [], []
                for event in sorted(selected, key=lambda e: e["display_order"]):
                    bindings = "; ".join(f"{role}={','.join(names)}" for role, names in event["roles"].items())
                    fields = (f"[play {event['display_order']}; {event['quarter']} {event['clock']}; "
                              f"elapsed_s={event['elapsed_regulation_seconds']}; "
                              f"first_offensive_td_relation={event['relative_to_first_offensive_td']}; "
                              f"{event['status']}; yards={event['yards']}; "
                              f"possession_context={event['possession_context']}; {bindings}]")
                    parts.append(fields + "\n" + event["quote"])
                    sources.append({"document_id": doc["id"], "kind": "structured_play", "start": event["start"],
                                    "end": event["end"], "complete": True, "event_id": event["id"],
                                    "role_sources": event["role_sources"],
                                    "possession_context": event["possession_context"]})
                content = ("[Parsed original user log. Counts cover observed events, not total pressures or proof of "
                           "blocking failure. Quarter is attached to each play, never carried across a boundary. "
                           "Historical assistant summaries are not source measurements.]\n"
                           + json.dumps(header, ensure_ascii=False) + "\n" + "\n".join(parts))
                packet = {"id": f"game_{doc['id']}_{label}", "content": content, "score": 1.0,
                          "record_sources": sources, "created_at": doc["created_at"]}
                packets.append(packet)
                doc_trace["groups"].append({"kind": label, "events": [e["id"] for e in selected], "characters": len(content)})
        if not packets:
            return evidence
        # Refresh existing derived packets instead of returning stale v2 alongside v3.
        refreshed_ids = {f"game_{doc}_{label}" for doc in refreshed_documents
                         for label in ("outcomes", "scoring", "largest_gains")}
        fitted = store._fit_source_records(packets + [e for e in evidence if e["id"] not in refreshed_ids], top_k)
        self.last_trace["returned_packet_ids"] = [p["id"] for p in fitted if p["id"].startswith("game_")]
        return fitted
