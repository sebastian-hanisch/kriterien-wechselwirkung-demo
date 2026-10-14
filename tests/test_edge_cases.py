"""Randfälle der Kernroutinen (DEMO-PLAYBOOK §12): n = 1, n = 2, konstante Spalten, identische Alternativen, ein Kriterium je Aspekt, inklusive Schwellen."""

import numpy as np
import pytest

import kwi_choquet as ch
import kwi_evaluation as ev
import kwi_scenario as sn
import kwi_score as sc
import kwi_weights as wt


def test_exactly_two_alternatives():
    """n = 2: die kleinste Instanz, in der es etwas zu vergleichen gibt (Standardabweichung mit ddof = 1, Entropie mit ln 2, Perzentil mit n - 1 = 1)."""
    N = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    w = wt.critic_weights(N)
    assert w.sum() == pytest.approx(1.0) and w[2] == pytest.approx(0.0)                 # die konstante Spalte hat Standardabweichung 0
    assert wt.entropy_weights(N).tolist() == pytest.approx([0.5, 0.5, 0.0])               # e = 0 bei Anteilen (1, 0), 1 bei (0,5; 0,5)
    assert wt.components(N, 0.5) == [[0, 1], [2]]                                         # r(1, 2) = -1: |r| >= 0,5 gruppiert; die konstante Spalte bleibt allein
    a = ev.analyse(ev.Settings(n=2, structure="2-2-2", seed=4))
    assert sorted(a.utility.argsort().tolist()) == [0, 1]


def test_single_alternative():
    N = np.array([[0.3, 0.8, 0.1]])
    assert wt.critic_weights(N).tolist() == pytest.approx([1 / 3] * 3)
    assert wt.entropy_weights(N).tolist() == pytest.approx([1 / 3] * 3)
    assert wt.components(N, 0.5) == [[0], [1], [2]]
    assert sc.percentile_normalize(N).tolist() == [[0.0, 0.0, 0.0]]
    assert sc.kendall_tau_b(np.array([1.0]), np.array([1.0])) == 0.0


def test_constant_columns_do_not_divide_by_zero():
    N = np.column_stack([np.full(5, 0.4), np.linspace(0, 1, 5), np.linspace(1, 0, 5) ** 2])
    for w in (wt.critic_weights(N), wt.entropy_weights(N), wt.cluster_weights(N, 0.5)):
        assert np.isfinite(w).all() and w.sum() == pytest.approx(1.0)
    assert wt.critic_weights(N)[0] == pytest.approx(0.0)                                  # Standardabweichung 0: kein Gewicht
    assert wt.entropy_weights(np.zeros((5, 2))).tolist() == pytest.approx([0.5, 0.5])     # nichts unterscheidet: gleiche Gewichte


def test_identical_alternatives_tie_and_pick_the_lowest_index():
    N = np.array([[0.5, 0.5], [0.5, 0.5], [0.1, 0.9]])
    s = sc.weighted_score(N, np.array([0.5, 0.5]))
    assert s[0] == s[1] and sc.top1(s) == 0
    assert ch.choquet(N[0], lambda S: len(S) / 2) == ch.choquet(N[1], lambda S: len(S) / 2)


def test_an_aspect_with_one_criterion_has_equal_mean_maximum_and_minimum():
    g = np.array([[0.2, 0.7], [0.9, 0.1]])
    v = np.array([0.5, 0.5])
    base = sn.aspect_utility(g, (1, 1), v)
    assert sn.aspect_utility(g, (1, 1), v, 1.0, "max").tolist() == pytest.approx(base.tolist())
    assert sn.aspect_utility(g, (1, 1), v, 1.0, "min").tolist() == pytest.approx(base.tolist())
    assert ch.interaction_index(ch.aspect_capacity((1, 1), v, 1.0, "max"), 2)[0, 1] == pytest.approx(0.0)


def test_threshold_of_the_grouping_is_inclusive():
    N = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])           # r = 1 genau
    assert wt.components(N, 1.0) == [[0, 1]]
    N2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]])          # r = 0,866: unter 0,9, über 0,8
    assert wt.components(N2, 0.9) == [[0], [1]]
    assert wt.components(N2, 0.8) == [[0, 1]]


def test_percentile_with_all_values_equal_and_two_values():
    assert sc.percentile_normalize(np.array([[2.0], [2.0], [2.0]])).ravel().tolist() == [0.5, 0.5, 0.5]
    assert sc.percentile_normalize(np.array([[1.0], [2.0]])).ravel().tolist() == [0.0, 1.0]


def test_correlation_bounds_are_exclusive():
    for bad in (1.0, -1.0):
        with pytest.raises(ValueError):
            sn.block_correlation((2, 2), bad)
    assert sn.block_correlation((1, 1, 1), 0.99)[0, 0] == 1.0
    with pytest.raises(ValueError):
        sn.parse_structure("3-3-3")


def test_choquet_with_equal_values_equals_that_value():
    x = np.array([0.4, 0.4, 0.4])
    for mu in (lambda S: len(S) / 3, lambda S: 1.0 if S else 0.0, lambda S: 1.0 if len(S) == 3 else 0.0):
        assert ch.choquet(x, mu) == pytest.approx(0.4)


def test_a_single_criterion_with_several_alternatives():
    """m = 1 und n >= 2: die Korrelationsmatrix ist 1 x 1 (numpy gibt einen Skalar zurück); CRITIC, Entropie und Gruppierung geben das Gewicht 1."""
    N = np.array([[0.2], [0.7], [0.4]])
    assert wt.critic_weights(N).tolist() == pytest.approx([1.0])
    assert wt.entropy_weights(N).tolist() == pytest.approx([1.0])
    assert wt.components(N, 0.5) == [[0]]
    assert wt.cluster_weights(N, 0.5).tolist() == pytest.approx([1.0])
    assert ev._correlation(N).tolist() == [[1.0]]


def test_degenerate_inputs_raise_no_numerical_warnings():
    """n = 1, konstante Spalten und Nullspalten laufen über die ausdrücklichen Zweige, nicht über NaN-Rückfälle mit Warnungen."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        one = np.array([[0.3, 0.8, 0.1]])
        assert wt.critic_weights(one).tolist() == pytest.approx([1 / 3] * 3)
        assert wt.entropy_weights(one).tolist() == pytest.approx([1 / 3] * 3)
        assert wt.components(one, 0.5) == [[0], [1], [2]]
        assert ev._correlation(one).tolist() == np.eye(3).tolist()
        assert wt.entropy_weights(np.zeros((4, 2))).tolist() == pytest.approx([0.5, 0.5])
        const = np.column_stack([np.full(5, 0.4), np.linspace(0, 1, 5)])
        assert wt.critic_weights(const).tolist() == pytest.approx([0.0, 1.0])
        assert wt.cluster_weights(const, 0.5).sum() == pytest.approx(1.0)


def test_monotonicity_check_sees_violations_that_involve_the_last_criterion():
    """is_monotone prüft alle Mengen und alle hinzugefügten Kriterien, auch das letzte (Index m - 1)."""
    table = {(): 0.0, (0,): 0.5, (1,): 0.1, (2,): 0.1, (0, 1): 0.6, (0, 2): 0.3, (1, 2): 0.3, (0, 1, 2): 1.0}
    adding_last = lambda S: table[tuple(sorted(S))]  # noqa: E731   # einzige Verletzung: {0} -> {0, 2} (das letzte Kriterium kommt dazu)
    from_last = lambda S: 0.5 if S == frozenset({2}) else (0.2 if S == frozenset({0, 2}) else 0.0)  # noqa: E731
    assert not ch.is_monotone(adding_last, 3)
    assert not ch.is_monotone(from_last, 3)
    assert ch.is_monotone(lambda S: len(S) / 3, 3)
