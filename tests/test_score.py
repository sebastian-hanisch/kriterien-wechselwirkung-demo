"""Normierungen, Platz 1, Kendalls tau-b, Nutzenverlust (Handwerte)."""

import numpy as np
import pytest

import kwi_score as sc


def test_minmax_by_hand_and_constant_column():
    S = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    assert sc.minmax(S).tolist() == [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]


def test_average_ranks_with_ties():
    assert sc.average_ranks(np.array([10.0, 20.0, 20.0, 30.0])).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert sc.average_ranks(np.array([1.0, 2.0, 2.0])).tolist() == [1.0, 2.5, 2.5]
    assert sc.average_ranks(np.array([5.0, 5.0, 5.0])).tolist() == [2.0, 2.0, 2.0]


def test_percentile_normalize_with_two_alternatives_and_one():
    assert sc.percentile_normalize(np.array([[3.0, 9.0], [1.0, 9.5]])).tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert sc.percentile_normalize(np.array([[3.0, 9.0]])).tolist() == [[0.0, 0.0]]
    assert sc.percentile_normalize(np.array([[1.0], [2.0], [2.0], [4.0]])).ravel().tolist() == pytest.approx([0.0, 0.5, 0.5, 1.0])


def test_normalize_dispatch():
    S = np.array([[1.0], [2.0], [4.0]])
    assert sc.normalize(S, "minmax").ravel().tolist() == pytest.approx([0.0, 1 / 3, 1.0])
    assert sc.normalize(S, "perzentil").ravel().tolist() == pytest.approx([0.0, 0.5, 1.0])
    with pytest.raises(ValueError):
        sc.normalize(S, "z")


def test_top1_takes_the_lowest_index_at_a_tie():
    assert sc.top1(np.array([0.3, 0.9, 0.9])) == 1


def test_kendall_tau_b_by_hand():
    assert sc.kendall_tau_b(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])) == pytest.approx(1.0)
    assert sc.kendall_tau_b(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)
    assert sc.kendall_tau_b(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) == 0.0
    assert sc.kendall_tau_b(np.array([1.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(2 / np.sqrt(2 * 3))


def test_weighted_score_and_utility_loss():
    N = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert sc.weighted_score(N, np.array([0.3, 0.7])).tolist() == pytest.approx([0.3, 0.7])
    assert sc.utility_loss(np.array([0.2, 0.9, 0.5]), 2) == pytest.approx(0.4)
