"""Standard ranking metrics. ``ranked`` is a list of doc ids, best first;
``relevant`` maps each relevant doc id to its relevance grade (>= 1)."""

import math


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Share of the relevant documents that appear in the top k."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant.keys()) / len(relevant)


def mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """1 / position of the first relevant document in the top k (0 if none)."""
    for position, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Normalised discounted cumulative gain: rewards relevant documents near the top."""
    dcg = sum(
        relevant.get(doc_id, 0) / math.log2(position + 1)
        for position, doc_id in enumerate(ranked[:k], start=1)
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(position + 1) for position, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0
