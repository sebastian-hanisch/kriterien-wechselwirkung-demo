"""Szenario: Strukturen, Blockkorrelation, Welt und Nutzen des Entscheiders."""

import numpy as np
import pytest

import kwi_scenario as sn
from entscheidungsmatrix_generator import SplitMix64


def test_parse_structure():
    assert sn.parse_structure("3-2-1") == (3, 2, 1)
    assert sn.parse_structure("4") == (4,)
    for bad in ("0-2", "2--1", "a-b", "5-4"):
        with pytest.raises(ValueError):
            sn.parse_structure(bad)


def test_blocks_are_consecutive_columns():
    assert [b.tolist() for b in sn.blocks_of((3, 2, 1))] == [[0, 1, 2], [3, 4], [5]]


def test_block_correlation_matrix():
    c = sn.block_correlation((2, 1), 0.6)
    assert c.tolist() == [[1.0, 0.6, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for bad in (1.0, -0.5):
        with pytest.raises(ValueError):
            sn.block_correlation((3, 2), bad)                    # Untergrenze -1/(3-1) = -0,5
    assert sn.block_correlation((2, 1), -0.99)[0, 1] == pytest.approx(-0.99)


def test_correlated_goodness_has_the_requested_block_correlation():
    g = sn.correlated_goodness(20000, (3, 2, 1), 0.6, SplitMix64(7))
    r = np.corrcoef(g.T)
    assert r[0, 1] == pytest.approx(0.58, abs=0.03) and r[0, 2] == pytest.approx(0.58, abs=0.03) and r[3, 4] == pytest.approx(0.58, abs=0.03)
    assert abs(r[0, 3]) < 0.03 and abs(r[2, 5]) < 0.03
    assert g.min() > 0 and g.max() < 1


def test_make_world_is_deterministic_and_weights_sum_to_one():
    a = sn.make_world(20, (3, 2, 1), 0.6, 5)
    b = sn.make_world(20, (3, 2, 1), 0.6, 5)
    assert (a.goodness == b.goodness).all() and (a.S == b.S).all() and (a.v == b.v).all()
    assert a.v.sum() == pytest.approx(1.0) and a.wc.sum() == pytest.approx(1.0) and len(a.v) == 3 and len(a.wc) == 6
    assert a.S.shape == (20, 6) and a.m == 6
    c = sn.make_world(20, (3, 2, 1), 0.6, 6)
    assert not (a.goodness == c.goodness).all()


def test_aspect_utility_by_hand():
    g = np.array([[0.9, 0.1, 0.5]])
    v = np.array([0.6, 0.4])
    assert sn.aspect_utility(g, (2, 1), v)[0] == pytest.approx(0.6 * 0.5 + 0.4 * 0.5)
    assert sn.aspect_utility(g, (2, 1), v, 1.0, "max")[0] == pytest.approx(0.6 * 0.9 + 0.4 * 0.5)
    assert sn.aspect_utility(g, (2, 1), v, 1.0, "min")[0] == pytest.approx(0.6 * 0.1 + 0.4 * 0.5)
    assert sn.aspect_utility(g, (2, 1), v, 0.5, "max")[0] == pytest.approx(0.6 * (0.5 * 0.5 + 0.5 * 0.9) + 0.4 * 0.5)
    with pytest.raises(ValueError):
        sn.aspect_utility(g, (2, 1), v, 1.0, "median")


def test_decider_utility_orders_synergy_mean_redundancy():
    w = sn.make_world(30, (2, 2, 2), 0.0, 3)
    low = sn.decider_utility(w, "synergie", 1.0)
    mid = sn.decider_utility(w, "aspekt")
    high = sn.decider_utility(w, "redundanz", 1.0)
    assert (low <= mid + 1e-12).all() and (mid <= high + 1e-12).all() and (low < high).any()
    assert sn.decider_utility(w, "redundanz", 0.0).tolist() == pytest.approx(mid.tolist())
    assert sn.decider_utility(w, "einzeln").tolist() == pytest.approx((w.goodness @ w.wc).tolist())
    with pytest.raises(ValueError):
        sn.decider_utility(w, "orakel")
