"""
Evaluation metrics (algorithm 2.0, Phase 0).

Pure functions — they take plain lists (ids, vectors, listener counts) and return
a float. All DB/API access lives in run_eval.py so these stay trivially testable.
Each takes one seed's recommendations vs. its ground-truth target set; run_eval
averages them across the whole ground-truth set.
"""
import itertools

import numpy as np


def recall_at_k(recommended: list[str], target: set, k: int) -> float:
    """Fraction of the target set retrieved within the top-k recommendations.

    Denominator is min(len(target), k): a model can't surface more than k of the
    targets in k slots, so we don't penalize it for a target set larger than k.
    """
    target = set(target)
    if not target:
        return 0.0
    top = recommended[:k]
    hits = sum(1 for tid in top if tid in target)
    return hits / min(len(target), k)


def mrr(recommended: list[str], target: set) -> float:
    """Reciprocal rank of the first recommended item that is in the target set.

    1.0 if the first rec is a hit, 0.5 if the second, ... 0.0 if none hit.
    """
    target = set(target)
    for rank, tid in enumerate(recommended, start=1):
        if tid in target:
            return 1.0 / rank
    return 0.0


def intra_list_distance(vectors: list[list]) -> float:
    """Mean pairwise cosine *distance* (1 - cosine similarity) across the rec list.

    A diversity proxy: higher = the recommendations spread across more of the
    space. Returns 0.0 for fewer than two vectors (no pairs).
    """
    vecs = [np.asarray(v, dtype=float) for v in vectors if v is not None]
    if len(vecs) < 2:
        return 0.0

    distances = []
    for a, b in itertools.combinations(vecs, 2):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            distances.append(1.0)
            continue
        cos = float(np.dot(a, b) / (na * nb))
        distances.append(1.0 - cos)
    return float(np.mean(distances))


def median_listeners(listeners: list[int]) -> float:
    """Median listener count of the recommendations — underground health.

    A model that lifts recall by returning popular tracks is a REGRESSION for this
    product; watching this number alongside recall catches that.
    """
    vals = [x for x in listeners if x is not None]
    if not vals:
        return 0.0
    return float(np.median(vals))
