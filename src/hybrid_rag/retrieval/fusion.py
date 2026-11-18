"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

Each ranked list votes for its items with ``weight / (k + rank)``. Only ranks
are used, never raw scores, which is what makes it safe to fuse retrievers whose
scores live on different scales (cosine similarity in [-1, 1] vs. unbounded BM25).
A larger ``k`` flattens the curve so that lower-ranked items still count.
"""

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists (best first) into one list of (id, score), best first.

    Ties keep the order in which ids were first seen, so results are deterministic.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("need exactly one weight per ranking")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
