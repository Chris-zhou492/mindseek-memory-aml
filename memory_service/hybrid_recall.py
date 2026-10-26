"""Independent dense/BM25 pools combined by reciprocal rank, scoped by caller."""

MODES = {"legacy", "vector", "bm25_rrf"}
VERSION = "independent-bm25-rrf-v1"


def fuse(items, dense, bm25, size, mode, k=60):
    if mode not in {"vector", "bm25_rrf"} or size < 1:
        raise ValueError("Invalid hybrid recall mode or pool size")
    dense_order = sorted(range(len(items)), key=lambda i: (-dense[i], items[i]["id"]))[:size]
    lexical_order = (
        sorted((i for i in range(len(items)) if bm25[i] > 0), key=lambda i: (-bm25[i], items[i]["id"]))[:size]
        if mode == "bm25_rrf" else []
    )
    dr = {i: r for r, i in enumerate(dense_order, 1)}
    br = {i: r for r, i in enumerate(lexical_order, 1)}
    candidates = []
    for i in dr.keys() | br.keys():
        channels = [name for name, ranks in (("vector", dr), ("bm25", br)) if i in ranks]
        score = sum(1 / (k + ranks[i]) for ranks in (dr, br) if i in ranks)
        candidates.append({
            **items[i], "score": score, "fusion_score": score,
            "semantic_score": dense[i], "bm25_score": bm25[i],
            "semantic_rank": dr.get(i), "bm25_rank": br.get(i),
            "recall_channels": channels, "fusion_mode": mode,
        })
    return sorted(candidates, key=lambda item: (-item["score"], item["id"]))[:size]
