import numpy as np
import pytest

from docjev import metrics as M


def synthetic(n=4000, k=4, sharpness=3.0, seed=0):
    """Labels drawn from softmax(base); returned logits are base * sharpness (over-confident)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1.5, size=(n, k))
    labels = [int(rng.choice(k, p=M.softmax(b))) for b in base]
    return list(base * sharpness), labels


def test_noul_is_sigmoid():
    for z in (-3.0, 0.0, 1.7):
        for T in (0.5, 1.0, 2.0):
            assert M.softmax(M.noul_logits(z), T)[1] == pytest.approx(1 / (1 + np.exp(-z / T)))


def test_temperature_recovers_overconfidence_and_keeps_argmax():
    logits, labels = synthetic()
    T = M.fit_temperature(logits, labels)
    assert T == pytest.approx(3.0, rel=0.15)
    before, after = M.evaluate(logits, labels), M.evaluate(logits, labels, T=T)
    assert after["accuracy"] == before["accuracy"]          # argmax invariance
    assert after["ece"] < before["ece"] and after["nll"] < before["nll"]
    assert after["ece"] < 0.03


def test_ece_perfect_and_worst():
    conf = np.full(100, 0.9)
    assert M.ece(conf, np.r_[np.ones(90), np.zeros(10)])[0] == pytest.approx(0.0)
    assert M.ece(np.ones(50), np.zeros(50))[0] == pytest.approx(1.0)


def test_risk_coverage_and_threshold():
    conf = np.array([0.99, 0.95, 0.9, 0.8, 0.6, 0.55])
    correct = np.array([1, 1, 1, 0, 1, 0])
    cov, risk = M.risk_coverage(conf, correct)
    assert cov[-1] == 1.0 and risk[2] == 0.0 and risk[3] == pytest.approx(0.25)
    assert M.coverage_at_risk(conf, correct, 0.0) == pytest.approx(0.5)
    assert M.coverage_at_risk(conf, correct, 0.2) == pytest.approx(5 / 6)
    assert M.threshold_for_risk(np.array([0.9]), np.array([0]), 0.0) == float("inf")


def test_threshold_does_not_split_ties():
    conf = np.array([0.9, 0.9, 0.5])
    correct = np.array([1, 0, 1])
    # Cannot answer only the first 0.9 without the second; risk at the tie is 0.5.
    assert M.threshold_for_risk(conf, correct, 0.1) == float("inf")


def test_bucketed_temperatures_fallback():
    logits, labels = synthetic(n=1000)
    keys = [("choice", M.option_bucket(4))] * 900 + [("noul", "2")] * 100
    temps = M.fit_bucketed_temperatures(logits, labels, keys, min_n=200)
    assert temps[("noul", "2")] == temps["*"]
    assert temps[("choice", "3-5")] != temps["*"]
    scaled = M.apply_temperatures(logits, keys, temps)
    assert [s.argmax() for s in scaled] == [l.argmax() for l in logits]


def test_variable_option_counts():
    logits = [np.array([2.0, 0.0]), np.array([0.0, 1.0, 3.0, -1.0]), np.array([1.0] * 12)]
    out = M.evaluate(logits, [0, 2, 5])
    assert out["n"] == 3 and 0 <= out["ece"] <= 1
