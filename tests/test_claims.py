"""Jede Zahl aus README.md und App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SWEEP_SEEDS (200 feste Matrizen), n = 20, nie über einen einzelnen Seed. Prozentwerte mit Toleranz 0,5 Punkte (Aufblähung 0,01), damit Rundung und
Bibliotheksversionen (CI installiert ungepinnt) nichts kippen. Der Standardfehler einer Trefferquote ist 2,5 bis 3,5 Punkte: Unterschiede unter etwa 6 Punkten sind NICHT belastbar
und stehen nirgends als Aussage.
"""

import pytest

import kwi_constants as C
import kwi_evaluation as ev


@pytest.fixture(scope="module")
def counting():
    return ev.experiment_counting()


@pytest.fixture(scope="module")
def loss():
    return ev.experiment_loss_by_size()


@pytest.fixture(scope="module")
def inflation():
    return ev.experiment_inflation()


@pytest.fixture(scope="module")
def data():
    return ev.experiment_data_driven()


@pytest.fixture(scope="module")
def bundled():
    return ev.experiment_separate(bundled=True)


@pytest.fixture(scope="module")
def random_weights():
    return ev.experiment_separate(rhos=(0.0, 0.3, 0.6, 0.9), bundled=False)


@pytest.fixture(scope="module")
def interaction():
    return ev.experiment_interaction()


@pytest.fixture(scope="module")
def correlation():
    return ev.experiment_correlation()


def cell(rows, **keys):
    hits = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(hits) == 1, (keys, len(hits))
    return hits[0]


def pct(x):
    return 100 * x


# ----------------------------------------------------------------------------- Experiment 1: ein Aspekt zählt so oft, wie er Kriterien hat

def test_equal_aspect_sizes_make_the_ask_per_criterion_correct(counting):
    for rho in C.EXP_RHOS:
        e = cell(counting, structure="2-2-2", rho=rho, method="erhoben")
        r = cell(counting, structure="2-2-2", rho=rho, method="richtig")
        assert e["hit"] == r["hit"] and e["tau"] == pytest.approx(r["tau"])
    assert [round(pct(cell(counting, structure="2-2-2", rho=rho, method="richtig")["hit"]), 1) for rho in C.EXP_RHOS] == [79.5, 80.0, 81.5, 88.0]


@pytest.mark.parametrize("structure,expected_asked,expected_right", [
    ("3-2-1", [58.5, 58.5, 65.5, 67.5], [72.0, 70.5, 75.0, 84.0]),
    ("4-1-1", [53.5, 59.5, 65.5, 67.5], [74.5, 80.5, 85.0, 84.0]),
])
def test_asking_per_criterion_loses_platz_1_when_aspects_differ_in_size(counting, structure, expected_asked, expected_right):
    asked = [pct(cell(counting, structure=structure, rho=rho, method="erhoben")["hit"]) for rho in C.EXP_RHOS]
    right = [pct(cell(counting, structure=structure, rho=rho, method="richtig")["hit"]) for rho in C.EXP_RHOS]
    assert asked == pytest.approx(expected_asked, abs=0.5)
    assert right == pytest.approx(expected_right, abs=0.5)
    assert all(r - e >= 9.0 for e, r in zip(asked, right))


def test_loss_grows_with_the_number_of_criteria_in_the_large_aspect(loss):
    assert [pct(r["loss"]) for r in loss] == pytest.approx([0.0, 3.5, 10.5, 16.5, 29.5], abs=0.5)
    assert [pct(r["erhoben"]) for r in loss] == pytest.approx([85.0, 67.5, 60.0, 67.5, 57.5], abs=0.5)
    assert [pct(r["richtig"]) for r in loss] == pytest.approx([85.0, 71.0, 70.5, 84.0, 87.0], abs=0.5)


def test_inflation_of_equal_weights_is_the_size_over_the_mean_size(inflation):
    for rho in C.EXP_RHOS:
        assert cell(inflation, structure="3-2-1", rho=rho, method="gleich")["inflation"] == pytest.approx(1.5, abs=1e-9)
        assert cell(inflation, structure="4-1-1", rho=rho, method="gleich")["inflation"] == pytest.approx(2.0, abs=1e-9)


# ----------------------------------------------------------------------------- Experiment 2: was CRITIC und Entropie korrigieren

def test_critic_corrects_only_partly_entropy_not_at_all(inflation):
    big = lambda m: [cell(inflation, structure="4-1-1", rho=rho, method=m)["inflation"] for rho in C.EXP_RHOS]  # noqa: E731
    assert big("critic") == pytest.approx([1.99, 1.87, 1.71, 1.49], abs=0.01)
    assert big("entropie") == pytest.approx([1.97, 1.96, 1.96, 1.95], abs=0.02)
    assert big("cluster") == pytest.approx([2.00, 1.84, 1.24, 1.03], abs=0.02)
    mid = lambda m: [cell(inflation, structure="3-2-1", rho=rho, method=m)["inflation"] for rho in C.EXP_RHOS]  # noqa: E731
    assert mid("critic") == pytest.approx([1.51, 1.45, 1.37, 1.27], abs=0.01)
    assert mid("cluster") == pytest.approx([1.49, 1.42, 1.08, 1.01], abs=0.02)


def test_critic_factor_helps_only_at_high_correlation(counting):
    asked = [pct(cell(counting, structure="4-1-1", rho=rho, method="erhoben")["hit"]) for rho in C.EXP_RHOS]
    factor = [pct(cell(counting, structure="4-1-1", rho=rho, method="erhoben_critic")["hit"]) for rho in C.EXP_RHOS]
    assert factor == pytest.approx([52.5, 63.0, 69.0, 80.5], abs=0.5)
    assert factor[0] - asked[0] < 1.0 and factor[3] - asked[3] > 12.0


def test_data_driven_methods_are_not_better_than_equal_weights(counting):
    for st, rho in (("3-2-1", 0.6), ("4-1-1", 0.6)):
        eq = pct(cell(counting, structure=st, rho=rho, method="gleich")["hit"])
        for m in ("critic", "entropie", "cluster"):
            assert pct(cell(counting, structure=st, rho=rho, method=m)["hit"]) < eq + 6.0


def test_data_driven_weights_say_nothing_about_importance(data):
    assert max(abs(r["critic_corr"]) for r in data) < 0.06
    assert max(abs(r["entropy_corr"]) for r in data) < 0.04
    assert max(r["entropy_rank_range"] for r in data) < 1e-3


def test_correlation_is_not_redundancy(bundled, random_weights):
    wahre = [pct(cell(bundled, rho=rho, method="wahre")["hit"]) for rho in C.EXP_CRITERION_RHOS]
    mit = [pct(cell(bundled, rho=rho, method="wahre_critic")["hit"]) for rho in C.EXP_CRITERION_RHOS]
    assert wahre == pytest.approx([81.5, 82.5, 85.5], abs=0.5)
    assert mit == pytest.approx([74.0, 81.0, 82.0], abs=0.5)
    assert wahre[0] - mit[0] > 7.0
    assert [pct(cell(random_weights, rho=rho, method="wahre")["hit"]) for rho in (0.0, 0.3, 0.6, 0.9)] == pytest.approx([76.0, 75.0, 77.0, 83.5], abs=0.5)


# ----------------------------------------------------------------------------- Experiment 3: wo die Summe nicht reicht

def _series(rows, kind, norm, method):
    return [round(pct(cell(rows, kind=kind, lam=lam, norm=norm, method=method)["hit"]), 1) for lam in C.EXP_LAMS]


def test_redundancy_on_minmax(interaction):
    assert _series(interaction, "max", "minmax", "summe") == pytest.approx([79.5, 77.5, 66.0, 54.0, 36.5], abs=0.5)
    assert _series(interaction, "max", "minmax", "choquet") == pytest.approx([79.5, 78.5, 75.0, 68.0, 63.5], abs=0.5)
    assert _series(interaction, "max", "minmax", "halb") == pytest.approx([79.5, 78.0, 70.5, 60.0, 46.0], abs=0.5)
    assert _series(interaction, "max", "minmax", "falsch") == pytest.approx([79.5, 71.5, 56.5, 42.5, 26.5], abs=0.5)


def test_synergy_has_no_advantage_on_minmax_but_on_a_common_scale(interaction):
    assert _series(interaction, "min", "minmax", "summe") == pytest.approx([79.5, 76.5, 74.0, 71.0, 67.5], abs=0.5)
    assert _series(interaction, "min", "minmax", "choquet") == pytest.approx([79.5, 73.0, 70.5, 68.5, 65.0], abs=0.5)
    assert _series(interaction, "min", "minmax", "falsch") == pytest.approx([79.5, 74.5, 62.5, 47.0, 27.0], abs=0.5)
    assert _series(interaction, "min", "perzentil", "summe") == pytest.approx([77.0, 77.5, 73.5, 71.5, 69.0], abs=0.5)
    assert _series(interaction, "min", "perzentil", "choquet") == pytest.approx([77.0, 77.5, 76.0, 76.5, 76.0], abs=0.5)


def test_redundancy_on_percentile_ranks_and_the_best_additive_fit(interaction):
    assert _series(interaction, "max", "perzentil", "summe") == pytest.approx([77.0, 75.5, 67.0, 52.0, 36.0], abs=0.5)
    assert _series(interaction, "max", "perzentil", "choquet") == pytest.approx([77.0, 76.5, 75.0, 72.0, 68.5], abs=0.5)
    assert _series(interaction, "max", "perzentil", "anpassung") == pytest.approx([77.0, 72.0, 65.0, 51.5, 39.5], abs=0.5)
    assert _series(interaction, "min", "perzentil", "anpassung") == pytest.approx([77.0, 76.5, 75.0, 73.0, 65.0], abs=0.5)


def test_how_often_the_sum_is_enough(interaction):
    same = lambda kind: [round(pct(cell(interaction, kind=kind, lam=lam, norm="perzentil", method="choquet")["same_as_summe"]), 1) for lam in C.EXP_LAMS]  # noqa: E731
    assert same("max") == pytest.approx([100.0, 88.0, 73.5, 53.0, 33.5], abs=0.5)
    assert same("min") == pytest.approx([100.0, 89.5, 84.5, 81.5, 76.0], abs=0.5)


# ----------------------------------------------------------------------------- Experiment 4: Korrelation und Wechselwirkung

def test_the_advantage_of_choquet_shrinks_with_correlation(correlation):
    assert [pct(r["summe"]) for r in correlation] == pytest.approx([17.0, 36.0, 46.0, 58.0, 72.0], abs=0.5)
    assert [pct(r["choquet"]) for r in correlation] == pytest.approx([65.5, 68.5, 72.0, 72.0, 75.0], abs=0.5)
    assert [pct(r["gap"]) for r in correlation] == pytest.approx([48.5, 32.5, 26.0, 14.0, 3.0], abs=0.5)


def test_parameter_counts_table():
    rows = ev.parameter_table()
    assert [(r["m"], r["summe"], r["zwei_additiv"], r["voll"]) for r in rows] == [(3, 3, 6, 6), (4, 4, 10, 14), (5, 5, 15, 30), (6, 6, 21, 62), (8, 8, 36, 254)]
