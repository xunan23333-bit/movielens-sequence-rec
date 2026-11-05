import numpy as np


def fit_scale(scores):
    return float(np.mean(scores)), max(float(np.std(scores)), 1e-8)


def blend_scores(a, b, weight, scales):
    ma, sa = scales[0]
    mb, sb = scales[1]
    return (1 - weight) * (a - ma) / sa + weight * (b - mb) / sb


def rank_metrics(scores, positives, known, k=10):
    scores = np.array(scores, copy=True)
    positives = np.asarray(positives, dtype=int) - 1
    mask = np.array(known, copy=True)
    mask[positives] = False
    scores[mask] = -np.inf
    top = np.argsort(-scores, kind='stable')[:k]
    hits = np.isin(top, positives).astype(float)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    ap = np.sum(precision * hits) / min(len(positives), k)
    discount = 1 / np.log2(np.arange(2, len(hits) + 2))
    ideal = np.sum(discount[:min(len(positives), k)])
    return float(ap), float(np.sum(hits * discount) / ideal)
