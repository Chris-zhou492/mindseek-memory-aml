"""Offline evidence diagnostics. Never imported by Add, Search or answer code."""


def span_present(span, hits, messages):
    """A citation counts only when its exact source text is actually delivered."""
    source = messages[span["message_id"]]
    if not 0 <= span["start"] < span["end"] <= len(source["content"]):
        raise ValueError("Invalid evidence offsets")
    quote = source["content"][span["start"] : span["end"]]
    if not quote:
        raise ValueError("Empty evidence span")
    for hit in hits:
        if hit["id"] == span["message_id"] and quote in hit["content"]:
            return True
        if span["message_id"] in hit.get("context_message_ids", []) and quote in hit["content"]:
            return True
        for citation in hit.get("sources", []):
            if citation["source_message_id"] == span["message_id"] and quote in citation["quote"]:
                return True
        intervals = sorted(
            (s["start"], s["end"])
            for unit in hit.get("complete_source_units", [])
            for s in unit["source_spans"]
            if s["message_id"] == span["message_id"]
        )
        cursor = span["start"]
        for start, end in intervals:
            if start > cursor:
                break
            cursor = max(cursor, end)
        if cursor >= span["end"] and quote in hit["content"]:
            return True
    return False


def assess_needs(needs, hits, messages):
    result = []
    for need in needs:
        alternatives = need["alternatives"]
        if not alternatives or any(not alt for alt in alternatives):
            raise ValueError("A positive evidence need requires source alternatives")
        complete = sum(all(span_present(s, hits, messages) for s in alt) for alt in alternatives)
        result.append(
            {
                "name": need["name"],
                "covered": complete >= need.get("minimum", 1),
                "complete_alternatives": complete,
                "required_alternatives": need.get("minimum", 1),
            }
        )
    return result


def classify_stages(candidate, reranked, delivered, score, annotation_status, complete=False):
    if annotation_status != "source_verified":
        return "not_classified_" + annotation_status
    for label, stage in (("candidate_missing", candidate), ("rerank_loss", reranked), ("client_cutoff", delivered)):
        if not stage or not all(n["covered"] for n in stage):
            return label if score < 1 else "scored_correct_despite_missing_annotated_evidence"
    if not complete:
        return "annotated_subset_present_requires_review"
    return "evidence_present_answer_or_scoring_failure" if score < 1 else "evidence_present_scored_correct"
