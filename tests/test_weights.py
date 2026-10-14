"""Gewichte: erhoben, richtig geteilt, CRITIC, Entropie, Gruppierung, Aufblähung (Handwerte der Mini-Instanzen)."""

import numpy as np
import pytest

import kwi_weights as wt


def test_expand_weights_asked_per_criterion_and_split(counting):
    sizes, v, _ = counting
    assert wt.expand_weights(v, sizes, False).tolist() == pytest.approx([0.375, 0.375, 0.25])
    assert wt.expand_weights(v, sizes, True).tolist() == pytest.approx([0.3, 0.3, 0.4])


def test_asked_per_criterion_equals_split_when_aspects_are_equally_large():
    v = np.array([0.5, 0.3, 0.2])
    assert wt.expand_weights(v, (2, 2, 2), False).tolist() == pytest.approx(wt.expand_weights(v, (2, 2, 2), True).tolist())
    assert wt.expand_weights(v, (1, 1, 1), False).tolist() == pytest.approx(v.tolist())


def test_aspect_share_and_inflation(counting):
    sizes, v, _ = counting
    erhoben = wt.expand_weights(v, sizes, False)
    assert wt.aspect_share(erhoben, sizes).tolist() == pytest.approx([0.75, 0.25])
    assert wt.inflation(erhoben, sizes, v).tolist() == pytest.approx([1.25, 0.625])
    assert wt.inflation(wt.expand_weights(v, sizes, True), sizes, v).tolist() == pytest.approx([1.0, 1.0])


def test_equal_weights_inflate_an_aspect_by_its_size_over_the_mean_size():
    sizes = (3, 2, 1)
    equal = np.full(6, 1 / 6)
    infl = wt.aspect_share(equal, sizes) / np.full(3, 1 / 3)
    assert infl.tolist() == pytest.approx([1.5, 1.0, 0.5])


def test_critic_counts_a_duplicate_pair_like_one_criterion(dup_cols):
    w = wt.critic_weights(dup_cols)
    assert w.tolist() == pytest.approx([0.25, 0.25, 0.5])
    assert w.sum() == pytest.approx(1.0)


def test_critic_falls_back_to_equal_weights_without_variation():
    assert wt.critic_weights(np.ones((4, 3))).tolist() == pytest.approx([1 / 3] * 3)
    assert wt.critic_weights(np.array([[0.2, 0.7]])).tolist() == pytest.approx([0.5, 0.5])


def test_entropy_weights_hand_values(dup_cols, entropy_cols):
    assert wt.entropy_weights(dup_cols).tolist() == pytest.approx([1 / 3] * 3)
    assert wt.entropy_weights(entropy_cols).tolist() == pytest.approx([2 / 3, 1 / 3])


def test_entropy_weight_of_a_constant_zero_column_is_zero():
    N = np.array([[1.0, 0.0], [0.0, 0.0], [0.5, 0.0]])
    assert wt.entropy_weights(N).tolist() == pytest.approx([1.0, 0.0])


def test_entropy_weights_on_ranks_are_all_equal():
    rng = np.random.default_rng(3)
    ranks = np.argsort(np.argsort(rng.random((20, 5)), axis=0), axis=0) / 19.0
    w = wt.entropy_weights(ranks)
    assert w.tolist() == pytest.approx([0.2] * 5)


def test_components_group_correlated_columns(dup_cols):
    assert wt.components(dup_cols, 0.5) == [[0, 1], [2]]
    assert wt.components(dup_cols, 1.0) == [[0, 1], [2]]       # r = 1 erfüllt |r| >= 1: die Schwelle gilt inklusive
    assert wt.cluster_weights(dup_cols, 0.5).tolist() == pytest.approx([0.25, 0.25, 0.5])


def test_cluster_weights_with_a_high_threshold_give_equal_weights():
    N = np.array([[1.0, 0.9, 0.2], [0.0, 0.1, 0.9], [0.5, 0.2, 0.1], [0.8, 0.6, 0.4], [0.3, 0.4, 0.8]])
    r = np.abs(np.corrcoef(N.T))
    assert r[0, 1] < 1.0
    assert wt.cluster_weights(N, 0.999).tolist() == pytest.approx([1 / 3] * 3)


def test_aspect_equal_weights():
    assert wt.aspect_equal_weights((2, 1)).tolist() == pytest.approx([0.25, 0.25, 0.5])
    assert wt.aspect_equal_weights((3, 2, 1)).tolist() == pytest.approx([1 / 9, 1 / 9, 1 / 9, 1 / 6, 1 / 6, 1 / 3])


def test_combine_multiplies_and_normalizes():
    assert wt.combine(np.array([0.5, 0.5]), np.array([0.2, 0.6])).tolist() == pytest.approx([0.25, 0.75])
    assert wt.combine(np.array([0.0, 0.0]), np.array([1.0, 1.0])).tolist() == pytest.approx([0.5, 0.5])
