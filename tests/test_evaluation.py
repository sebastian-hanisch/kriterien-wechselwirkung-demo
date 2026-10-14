"""Auswertung: Fall der App, Grenzfälle der Verfahren und die Verdikt-Texte."""

import numpy as np
import pytest

import kwi_constants as C
import kwi_evaluation as ev


def test_de_uses_a_decimal_comma():
    assert ev.de(3.14159, 2) == "3,14"
    assert ev.de(2.0, 0) == "2"
    assert ev.de(12.0, 1) == "12,0"


def test_analysis_of_the_default_case_is_consistent():
    a = ev.analyse(ev.Settings())
    s = a.settings
    assert a.world.n == s.n and a.world.m == 6 and a.sizes == (3, 2, 1)
    assert a.N.shape == (s.n, 6) and len(a.utility) == s.n and a.best == int(np.argmax(a.utility))
    assert set(a.results) == set(ev.METHODS) and set(a.weights) == set(ev.WEIGHTED)
    for w in a.weights.values():
        assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
    assert a.right.sum() == pytest.approx(1.0)
    assert a.shapley.sum() == pytest.approx(1.0) and a.interaction.shape == (6, 6)
    assert a.corr.shape == (6, 6) and np.diag(a.corr).tolist() == pytest.approx([1.0] * 6)


def test_asked_per_criterion_inflates_and_split_does_not():
    a = ev.analyse(ev.Settings(structure="4-1-1", survey="kriterium"))
    share_asked = a.weights["erhoben"][:4].sum()
    share_right = a.right[:4].sum()
    assert share_asked > share_right
    b = ev.analyse(ev.Settings(structure="4-1-1", survey="geteilt"))
    assert b.weights["erhoben"].tolist() == pytest.approx(b.right.tolist())
    assert b.results["erhoben"].top1 == b.results["richtig"].top1


def test_a_decider_valuing_each_criterion_has_the_true_weights_as_asked_weights():
    a = ev.analyse(ev.Settings(decider="einzeln"))
    assert a.weights["erhoben"].tolist() == pytest.approx(a.world.wc.tolist())
    assert a.right.tolist() == pytest.approx(a.world.wc.tolist())
    assert a.results["choquet"].tau == pytest.approx(a.results["richtig"].tau)


def test_without_interaction_choquet_equals_the_correctly_split_sum():
    a = ev.analyse(ev.Settings(decider="aspekt"))
    assert a.results["choquet"].score.tolist() == pytest.approx(a.results["richtig"].score.tolist())
    assert a.interaction[np.triu_indices(6, 1)].tolist() == pytest.approx([0.0] * 15)


def test_with_interaction_choquet_is_the_deciders_own_function_on_a_common_scale():
    a = ev.analyse(ev.Settings(structure="2-2-2", rho=0.0, decider="redundanz", lam=1.0, survey="geteilt", norm="perzentil"))
    assert a.interaction[0, 1] < 0 and a.interaction[0, 2] == pytest.approx(0.0)
    b = ev.analyse(ev.Settings(structure="2-2-2", rho=0.0, decider="synergie", lam=1.0, survey="geteilt", norm="perzentil"))
    assert b.interaction[0, 1] > 0


def test_every_decider_norm_and_structure_analyses_without_error():
    for dec in C.DECIDERS:
        for norm in C.NORMS:
            for st in C.STRUCTURES:
                a = ev.analyse(ev.Settings(n=6, structure=st, decider=dec, norm=norm, seed=3))
                assert 0 <= a.results["choquet"].top1 < 6


def test_two_alternatives_and_the_minimum_n_analyse():
    a = ev.analyse(ev.Settings(n=2, structure="2-2-2", seed=1))
    assert a.N.shape == (2, 6) and a.results["richtig"].top1 in (0, 1)
    b = ev.analyse(ev.Settings(n=C.N_MIN, seed=2))
    assert len(b.utility) == 5


def test_experiments_accept_a_small_set_of_seeds():
    seeds = C.SWEEP_SEEDS[:5]
    rows = ev.experiment_counting(structures=("3-2-1",), rhos=(0.0, 0.9), seeds=seeds)
    assert len(rows) == 2 * len(ev.COUNTING_METHODS)
    assert all(0.0 <= r["hit"] <= 1.0 and r["se"] >= 0.0 for r in rows)
    assert len(ev.experiment_loss_by_size(ks=(1, 3), seeds=seeds)) == 2
    assert len(ev.experiment_inflation(structures=("4-1-1",), rhos=(0.0,), seeds=seeds)) == 4
    d = ev.experiment_data_driven(structures=("2-2-2",), seeds=seeds)
    assert d[0]["m"] == 6
    assert len(ev.experiment_separate(rhos=(0.0,), seeds=seeds)) == len(ev.SEPARATE_METHODS)
    i = ev.experiment_interaction(lams=(0.0, 1.0), seeds=seeds)
    assert len(i) == 2 * 2 * 2 * len(ev.INTERACTION_METHODS) - 2 * 2 * 1 * 1       # 'anpassung' gibt es nur bei Perzentil
    assert len(ev.experiment_correlation(rhos=(0.0, 0.9), seeds=seeds)) == 2


def test_verdicts_name_their_numbers():
    counting = ev.experiment_counting(structures=C.EXP_STRUCTURES, rhos=C.EXP_RHOS, seeds=C.SWEEP_SEEDS[:20])
    loss = ev.experiment_loss_by_size(seeds=C.SWEEP_SEEDS[:20])
    infl = ev.experiment_inflation(seeds=C.SWEEP_SEEDS[:20])
    kind, text = ev.verdict_counting(counting, loss, infl)
    assert kind == "warning" and "zählt so oft, wie er Kriterien hat" in text and "(4, 1, 1)" in text and "2,00-fach" in text
    data = ev.experiment_data_driven(seeds=C.SWEEP_SEEDS[:20])
    bundled = ev.experiment_separate(seeds=C.SWEEP_SEEDS[:20])
    kind, text = ev.verdict_data_driven(infl, data, bundled)
    assert kind == "warning" and "CRITIC" in text and "Entropie" in text and "Korrelation ist nicht Redundanz" in text
    inter = ev.experiment_interaction(seeds=C.SWEEP_SEEDS[:20])
    kind, text = ev.verdict_interaction(inter)
    assert kind == "warning" and "Redundanz" in text and "falschem" in text
    corr = ev.experiment_correlation(seeds=C.SWEEP_SEEDS[:20])
    kind, text = ev.verdict_correlation(corr, ev.parameter_table())
    assert kind == "info" and "21 Parameter" in text and "62" in text
