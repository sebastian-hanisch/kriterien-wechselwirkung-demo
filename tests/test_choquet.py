"""Choquet-Integral, Kapazitäten, Shapley-Wert und Interaktionsindex (Handwerte der Mini-Instanzen und Identitäten)."""

import numpy as np
import pytest

import kwi_choquet as ch
import kwi_scenario as sn


def test_choquet_of_the_two_additive_capacity_by_hand(capacity3):
    mi, mij, x = capacity3
    mu = ch.moebius_capacity(mi, mij)
    assert mu(frozenset({0, 1})) == pytest.approx(0.8) and mu(frozenset({0, 2})) == pytest.approx(0.5) and mu(frozenset({1, 2})) == pytest.approx(0.5)
    assert mu(frozenset({0, 1, 2})) == pytest.approx(1.0)
    assert ch.choquet(x, mu) == pytest.approx(0.56)
    assert ch.choquet_2additive(x[None, :], np.array([0.3, 0.2, 0.3]), mij).tolist() == pytest.approx([0.56])


def test_shapley_and_interaction_of_the_two_additive_capacity(capacity3):
    mi, mij, _ = capacity3
    mu = ch.moebius_capacity(mi, mij)
    phi = ch.shapley(mu, 3)
    assert phi.tolist() == pytest.approx([0.40, 0.35, 0.25])
    assert phi.sum() == pytest.approx(1.0)
    I = ch.interaction_index(mu, 3)
    assert I[0, 1] == pytest.approx(0.3) and I[0, 2] == pytest.approx(-0.1) and I[1, 2] == pytest.approx(0.0)
    assert (I == I.T).all() and np.diag(I).tolist() == [0.0, 0.0, 0.0]


def test_the_weighted_sum_with_shapley_weights_is_not_the_choquet_integral(capacity3):
    mi, mij, x = capacity3
    mu = ch.moebius_capacity(mi, mij)
    assert float(x @ ch.shapley(mu, 3)) == pytest.approx(0.61)
    assert ch.choquet(x, mu) != pytest.approx(0.61)


def test_additive_capacity_is_the_weighted_sum():
    w = np.array([0.5, 0.3, 0.2])
    x = np.array([0.2, 0.9, 0.4])
    assert ch.choquet(x, lambda S: float(sum(w[i] for i in S))) == pytest.approx(float(x @ w))


def test_maximum_minimum_and_mean_are_choquet_integrals():
    x = np.array([0.9, 0.1, 0.5])
    assert ch.choquet(x, lambda S: 1.0 if S else 0.0) == pytest.approx(0.9)
    assert ch.choquet(x, lambda S: 1.0 if len(S) == 3 else 0.0) == pytest.approx(0.1)
    assert ch.choquet(x, lambda S: len(S) / 3) == pytest.approx(0.5)


def test_redundancy_and_synergy_order_the_pair_differently(ab_pair):
    A, B = ab_pair
    assert A.mean() == pytest.approx(B.mean())
    assert A.max() > B.max() and A.min() < B.min()
    cap_max = ch.aspect_capacity((2,), np.array([1.0]), 1.0, "max")
    cap_min = ch.aspect_capacity((2,), np.array([1.0]), 1.0, "min")
    assert ch.choquet(A, cap_max) == pytest.approx(0.9) and ch.choquet(B, cap_max) == pytest.approx(0.5)
    assert ch.choquet(A, cap_min) == pytest.approx(0.1) and ch.choquet(B, cap_min) == pytest.approx(0.5)


def test_aspect_capacity_values_for_a_pair_aspect():
    mu = ch.aspect_capacity((2, 1), np.array([0.6, 0.4]), 0.5, "max")
    assert mu(frozenset()) == pytest.approx(0.0)
    assert mu(frozenset({0})) == pytest.approx(0.6 * (0.5 * 0.5 + 0.5 * 1.0))
    assert mu(frozenset({2})) == pytest.approx(0.4)
    assert mu(frozenset({0, 1, 2})) == pytest.approx(1.0)
    assert ch.is_monotone(mu, 3)


def test_aspect_utility_is_the_choquet_integral_of_the_aspect_capacity():
    rng = np.random.default_rng(11)
    for kind in ("max", "min"):
        for lam in (0.0, 0.5, 1.0):
            sizes = (3, 2, 1)
            v = np.array([0.5, 0.3, 0.2])
            X = rng.random((6, 6))
            mu = ch.aspect_capacity(sizes, v, lam, kind)
            assert sn.aspect_utility(X, sizes, v, lam, kind).tolist() == pytest.approx(ch.choquet_rows(X, mu).tolist())


def test_shapley_of_an_aspect_capacity_splits_the_aspect_weight_evenly():
    sizes, v = (2, 1), np.array([0.6, 0.4])
    for kind in ("max", "min"):
        phi = ch.shapley(ch.aspect_capacity(sizes, v, 1.0, kind), 3)
        assert phi.tolist() == pytest.approx([0.3, 0.3, 0.4])


def test_interaction_sign_of_redundancy_and_synergy():
    sizes, v = (2, 1), np.array([0.6, 0.4])
    assert ch.interaction_index(ch.aspect_capacity(sizes, v, 1.0, "max"), 3)[0, 1] < 0
    assert ch.interaction_index(ch.aspect_capacity(sizes, v, 1.0, "min"), 3)[0, 1] > 0
    assert ch.interaction_index(ch.aspect_capacity(sizes, v, 0.0, "max"), 3)[0, 1] == pytest.approx(0.0)


def test_is_monotone_detects_a_decreasing_capacity():
    assert not ch.is_monotone(lambda S: 1.0 if len(S) == 1 else 0.0, 2)


def test_parameter_counts():
    assert ch.parameter_counts(6) == {"summe": 6, "zwei_additiv": 21, "voll": 62}
    assert ch.parameter_counts(3) == {"summe": 3, "zwei_additiv": 6, "voll": 6}
    assert ch.parameter_counts(8)["voll"] == 254
