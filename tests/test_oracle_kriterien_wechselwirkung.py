"""Orakel: Kernroutinen gegen einen unabhängigen Rechenweg (DEMO-PLAYBOOK §12).

1. Choquet-Integral (sortierte Definition) gegen die Möbius-Darstellung: C(x) = Summe_T m(T) min_{i in T} x_i mit nichtnegativen Massen m(T) (Belief-Funktion, monoton).
2. Shapley-Wert gegen den Mittelwert der Grenzbeiträge über alle Reihenfolgen; Interaktionsindex gegen die Möbius-Formel Summe_{T enthält i, j} m(T) / (|T| - 1).
3. CRITIC und Entropie gegen die Definition mit einfachen Schleifen.
4. Aufblähung gegen die geschlossene Formel; erhoben = richtig bei gleich großen Aspekten.
5. Aspektweise Mischung (Mittel, Maximum, Minimum) ist das Choquet-Integral der Aspekt-Kapazität.
"""

import itertools
import math

import numpy as np
import pytest

import kwi_choquet as ch
import kwi_scenario as sn
import kwi_weights as wt


def random_belief(rng, m):
    """Nichtnegative Möbius-Massen auf allen nichtleeren Teilmengen, Summe 1; ergibt eine monotone Kapazität mit mu(alle) = 1."""
    sets = [frozenset(c) for r in range(1, m + 1) for c in itertools.combinations(range(m), r)]
    mass = rng.random(len(sets)) * (rng.random(len(sets)) < 0.5)
    if mass.sum() == 0:
        mass[-1] = 1.0
    mass = mass / mass.sum()
    return dict(zip(sets, mass))


def mu_of(mass):
    return lambda S: sum(v for T, v in mass.items() if T <= frozenset(S))


def test_choquet_equals_the_moebius_form_for_random_capacities():
    rng = np.random.default_rng(101)
    for _ in range(400):
        m = int(rng.integers(2, 6))
        mass = random_belief(rng, m)
        x = rng.random(m)
        expected = sum(v * min(x[i] for i in T) for T, v in mass.items())
        assert ch.choquet(x, mu_of(mass)) == pytest.approx(expected, abs=1e-12)


def test_choquet_is_monotone_in_every_argument():
    rng = np.random.default_rng(102)
    for _ in range(300):
        m = int(rng.integers(2, 6))
        mu = mu_of(random_belief(rng, m))
        x = rng.random(m)
        j = int(rng.integers(m))
        y = x.copy()
        y[j] = min(1.0, y[j] + rng.random() * 0.3)
        assert ch.choquet(y, mu) >= ch.choquet(x, mu) - 1e-12


def test_shapley_equals_the_average_marginal_contribution_over_all_orders():
    rng = np.random.default_rng(103)
    for _ in range(40):
        m = int(rng.integers(2, 6))
        mu = mu_of(random_belief(rng, m))
        phi = np.zeros(m)
        orders = list(itertools.permutations(range(m)))
        for order in orders:
            seen = frozenset()
            for i in order:
                phi[i] += mu(seen | {i}) - mu(seen)
                seen = seen | {i}
        phi /= len(orders)
        assert ch.shapley(mu, m).tolist() == pytest.approx(phi.tolist(), abs=1e-12)
        assert ch.shapley(mu, m).sum() == pytest.approx(mu(frozenset(range(m))), abs=1e-12)


def test_interaction_index_equals_the_moebius_formula():
    rng = np.random.default_rng(104)
    for _ in range(40):
        m = int(rng.integers(2, 6))
        mass = random_belief(rng, m)
        I = ch.interaction_index(mu_of(mass), m)
        for i, j in itertools.combinations(range(m), 2):
            expected = sum(v / (len(T) - 1) for T, v in mass.items() if i in T and j in T)
            assert I[i, j] == pytest.approx(expected, abs=1e-12)


def test_critic_and_entropy_follow_the_definition():
    rng = np.random.default_rng(105)
    for _ in range(100):
        n, m = int(rng.integers(3, 15)), int(rng.integers(2, 7))
        N = rng.random((n, m))
        # CRITIC mit Schleifen
        s = [math.sqrt(sum((N[i, j] - N[:, j].mean()) ** 2 for i in range(n)) / (n - 1)) for j in range(m)]
        c = []
        for j in range(m):
            total = 0.0
            for k in range(m):
                cov = sum((N[i, j] - N[:, j].mean()) * (N[i, k] - N[:, k].mean()) for i in range(n)) / (n - 1)
                r = cov / (s[j] * s[k])
                total += 1.0 - r
            c.append(s[j] * total)
        assert wt.critic_weights(N).tolist() == pytest.approx((np.array(c) / sum(c)).tolist(), abs=1e-9)
        # Entropie mit Schleifen
        d = []
        for j in range(m):
            col = N[:, j] / N[:, j].sum()
            e = -sum(p * math.log(p) for p in col if p > 0) / math.log(n)
            d.append(1.0 - e)
        assert wt.entropy_weights(N).tolist() == pytest.approx((np.array(d) / sum(d)).tolist(), abs=1e-9)


def test_inflation_follows_the_closed_formula_and_is_one_for_equal_sizes():
    rng = np.random.default_rng(106)
    for _ in range(100):
        p = int(rng.integers(2, 5))
        sizes = tuple(int(x) for x in rng.integers(1, 3, size=p))
        v = rng.dirichlet(np.ones(p))
        asked = wt.expand_weights(v, sizes, False)
        denominators = sum(s * va for s, va in zip(sizes, v))
        expected = np.array([s * va / denominators / va for s, va in zip(sizes, v)])
        assert wt.inflation(asked, sizes, v).tolist() == pytest.approx(expected.tolist())
        assert wt.inflation(wt.expand_weights(v, sizes, True), sizes, v).tolist() == pytest.approx([1.0] * p)
    v = np.array([0.5, 0.3, 0.2])
    assert wt.expand_weights(v, (3, 3, 3), False).tolist() == pytest.approx(wt.expand_weights(v, (3, 3, 3), True).tolist())


def test_aspect_utility_is_the_choquet_integral_of_the_aspect_capacity():
    rng = np.random.default_rng(107)
    for _ in range(60):
        sizes = tuple(int(x) for x in rng.integers(1, 4, size=int(rng.integers(1, 4))))
        v = rng.dirichlet(np.ones(len(sizes)))
        lam = float(rng.choice([0.0, 0.25, 0.5, 1.0]))
        kind = str(rng.choice(["max", "min"]))
        X = rng.random((5, int(sum(sizes))))
        mu = ch.aspect_capacity(sizes, v, lam, kind)
        assert sn.aspect_utility(X, sizes, v, lam, kind).tolist() == pytest.approx(ch.choquet_rows(X, mu).tolist(), abs=1e-12)
        assert ch.is_monotone(mu, int(sum(sizes)))
        assert mu(frozenset(range(int(sum(sizes))))) == pytest.approx(1.0)
