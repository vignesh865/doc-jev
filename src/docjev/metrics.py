"""Evaluation metrics for typed decisions.

Every question is a closed-set decision: a vector of logits over its K options
and the index of the correct option. K may differ between questions, so inputs
are lists of 1-D arrays. A yes/no question with a single logit z is the 2-class
case [0, z] (softmax([0, z] / T) == sigmoid(z / T)).

Decode is always argmax; the reported confidence is max(p), the probability of
the answer actually returned. That is the quantity we calibrate and gate on.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Sequence

import numpy as np

T_MIN, T_MAX = 0.05, 20.0


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, float) / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def noul_logits(z: float) -> np.ndarray:
    """Single yes/no logit -> 2-class logits [false, true]."""
    return np.array([0.0, float(z)])


def _probs(logits_list: Sequence[np.ndarray], T: float = 1.0) -> list[np.ndarray]:
    return [softmax(l, T) for l in logits_list]


def _conf_correct(probs: Sequence[np.ndarray], labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    conf = np.array([p.max() for p in probs])
    correct = np.array([float(p.argmax() == y) for p, y in zip(probs, labels)])
    return conf, correct


def nll(probs: Sequence[np.ndarray], labels: Sequence[int]) -> float:
    return float(-np.mean([np.log(max(p[y], 1e-12)) for p, y in zip(probs, labels)]))


def brier(probs: Sequence[np.ndarray], labels: Sequence[int]) -> float:
    """Multi-class Brier: sum over options of (p_k - y_k)^2, averaged over questions."""
    out = []
    for p, y in zip(probs, labels):
        t = np.zeros_like(p)
        t[y] = 1.0
        out.append(((p - t) ** 2).sum())
    return float(np.mean(out))


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    """Top-label expected calibration error with equal-width bins.

    Returns (ece, rows); rows are the reliability-diagram table. Bins with
    n < 30 are flagged `noisy`.
    """
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total, rows = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        c, a, n = conf[m].mean(), correct[m].mean(), int(m.sum())
        total += n / len(conf) * abs(c - a)
        rows.append({"lo": edges[b], "hi": edges[b + 1], "conf": c, "acc": a, "n": n, "noisy": n < 30})
    return float(total), rows


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Answer the most confident first. coverage[i], risk[i] after answering i+1 questions."""
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    errors = np.cumsum(1.0 - np.asarray(correct, float)[order])
    n = np.arange(1, len(order) + 1)
    return n / len(order), errors / n


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, risk = risk_coverage(conf, correct)
    return float(risk.mean())


def threshold_for_risk(conf: np.ndarray, correct: np.ndarray, max_risk: float) -> float:
    """Lowest confidence threshold whose answered set has risk <= max_risk.

    Fit on a calibration split, then apply to test with `coverage_and_risk_at`.
    Returns +inf (answer nothing) when no threshold qualifies.
    """
    conf = np.asarray(conf, float)
    order = np.argsort(-conf, kind="stable")
    _, risk = risk_coverage(conf, correct)
    # Only cut between distinct confidence values; ties are answered together.
    sorted_conf = conf[order]
    valid = np.r_[sorted_conf[1:] < sorted_conf[:-1], True]
    ok = np.where(valid & (risk <= max_risk))[0]
    return float(sorted_conf[ok.max()]) if ok.size else float("inf")


def coverage_and_risk_at(conf: np.ndarray, correct: np.ndarray, threshold: float) -> tuple[float, float]:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    m = conf >= threshold
    return float(m.mean()), float(1.0 - correct[m].mean()) if m.any() else 0.0


def coverage_at_risk(conf: np.ndarray, correct: np.ndarray, max_risk: float) -> float:
    """Oracle coverage at risk <= max_risk (threshold chosen on the same data).

    Optimistic by construction; for an honest number fit the threshold on a
    calibration split with `threshold_for_risk`.
    """
    return coverage_and_risk_at(conf, correct, threshold_for_risk(conf, correct, max_risk))[0]


def evaluate(logits_list: Sequence[np.ndarray], labels: Sequence[int], T: float = 1.0,
             risks: Sequence[float] = (0.01, 0.05)) -> dict:
    probs = _probs(logits_list, T)
    conf, correct = _conf_correct(probs, labels)
    out = {
        "n": len(labels),
        "accuracy": float(correct.mean()),
        "nll": nll(probs, labels),
        "brier": brier(probs, labels),
        "ece": ece(conf, correct)[0],
        "aurc": aurc(conf, correct),
    }
    for r in risks:
        out[f"cov@{r:g}"] = coverage_at_risk(conf, correct, r)
    return out


def fit_temperature(logits_list: Sequence[np.ndarray], labels: Sequence[int]) -> float:
    """Single temperature minimising NLL (convex in 1/T): log-grid, then golden-section refine."""
    def loss(log_t: float) -> float:
        return nll(_probs(logits_list, float(np.exp(log_t))), labels)

    grid = np.linspace(np.log(T_MIN), np.log(T_MAX), 41)
    i = int(np.argmin([loss(g) for g in grid]))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (np.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(40):
        if loss(a) < loss(b):
            hi, b = b, a
            a = hi - g * (hi - lo)
        else:
            lo, a = a, b
            b = lo + g * (hi - lo)
    return float(np.exp((lo + hi) / 2))


def option_bucket(k: int) -> str:
    return "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"


def fit_bucketed_temperatures(logits_list: Sequence[np.ndarray], labels: Sequence[int],
                              keys: Sequence[Hashable], min_n: int = 200) -> dict:
    """One temperature per bucket key (e.g. (qtype, option_bucket(k))).

    Buckets with fewer than `min_n` examples fall back to the global temperature,
    stored under the key "*".
    """
    temps = {"*": fit_temperature(logits_list, labels)}
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    for k, idx in groups.items():
        temps[k] = (fit_temperature([logits_list[i] for i in idx], [labels[i] for i in idx])
                    if len(idx) >= min_n else temps["*"])
    return temps


def apply_temperatures(logits_list: Sequence[np.ndarray], keys: Sequence[Hashable],
                       temps: dict) -> list[np.ndarray]:
    """Rescale logits by their bucket's temperature (unknown buckets use the global one)."""
    return [np.asarray(l, float) / temps.get(k, temps["*"]) for l, k in zip(logits_list, keys)]
