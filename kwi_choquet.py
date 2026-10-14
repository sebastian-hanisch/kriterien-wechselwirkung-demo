"""Choquet-Integral, Kapazitäten (aspektweise Mischung aus Mittel, Maximum, Minimum und 2-additive Möbius-Form), Shapley-Wert und Interaktionsindex. Nur numpy und itertools.

Eine Kapazität ist eine Funktion mu(S) auf Mengen von Kriterien (frozenset von Indizes), monoton, mu(leer) = 0, mu(alle) = 1."""

from __future__ import annotations

import itertools
import math

import numpy as np

from kwi_scenario import blocks_of


def choquet(x: np.ndarray, mu) -> float:
    """Choquet-Integral nach der Definition: aufsteigend sortiert, Summe (x_(i) - x_(i-1)) * mu(Menge der Kriterien mit Wert >= x_(i))."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    prev, total = 0.0, 0.0
    for pos, i in enumerate(order):
        total += (x[i] - prev) * mu(frozenset(int(j) for j in order[pos:]))
        prev = x[i]
    return float(total)


def choquet_rows(X: np.ndarray, mu) -> np.ndarray:
    return np.array([choquet(row, mu) for row in np.asarray(X, dtype=float)])


def aspect_capacity(sizes, v: np.ndarray, lam: float, kind: str):
    """Kapazität des Entscheiders: mu(S) = Summe_a v_a * mu_a(|S in Aspekt a|) mit mu_a(k) = (1 - lam) * k/s + lam * (1, wenn k >= 1 | 1, wenn k = s) für Maximum bzw. Minimum
    (s Kriterien im Aspekt). Das Choquet-Integral zu dieser Kapazität ist `kwi_scenario.aspect_utility` (Linearität in der Kapazität)."""
    blocks = [set(int(i) for i in b) for b in blocks_of(sizes)]
    if kind not in ("max", "min"):
        raise ValueError(f"unbekannte Wechselwirkung: {kind}")

    def mu(S) -> float:
        total = 0.0
        for a, b in enumerate(blocks):
            k = len(b & set(S))
            s = len(b)
            spec = (1.0 if k >= 1 else 0.0) if kind == "max" else (1.0 if k == s else 0.0)
            total += v[a] * ((1.0 - lam) * k / s + lam * spec)
        return total

    return mu


def moebius_capacity(m_single: dict, m_pair: dict):
    """2-additive Kapazität aus den Möbius-Massen: mu(S) = Summe_{i in S} m_i + Summe_{i<j in S} m_ij."""
    def mu(S) -> float:
        S = sorted(S)
        return sum(m_single[i] for i in S) + sum(m_pair[(i, j)] for a, i in enumerate(S) for j in S[a + 1:])

    return mu


def choquet_2additive(X: np.ndarray, m_single: np.ndarray, m_pair: dict) -> np.ndarray:
    """Choquet-Integral einer 2-additiven Kapazität in Möbius-Form: Summe m_i x_i + Summe m_ij min(x_i, x_j), je Zeile."""
    X = np.asarray(X, dtype=float)
    out = X @ np.asarray(m_single, dtype=float)
    for (i, j), mij in m_pair.items():
        out = out + mij * np.minimum(X[:, i], X[:, j])
    return out


def shapley(mu, m: int) -> np.ndarray:
    """Shapley-Wert (Wichtigkeit) je Kriterium: gewichteter Mittelwert der Grenzbeiträge über alle Mengen ohne das Kriterium. Summe = mu(alle)."""
    out = np.zeros(m)
    for i in range(m):
        others = [j for j in range(m) if j != i]
        for r in range(m):
            wgt = math.factorial(r) * math.factorial(m - r - 1) / math.factorial(m)
            for S in itertools.combinations(others, r):
                s = frozenset(S)
                out[i] += wgt * (mu(s | {i}) - mu(s))
    return out


def interaction_index(mu, m: int) -> np.ndarray:
    """Interaktionsindex I_ij (Grabisch) je Paar: positiv = Synergie, negativ = Redundanz, 0 = additiv. Symmetrische m x m Matrix, Diagonale 0."""
    I = np.zeros((m, m))
    for i, j in itertools.combinations(range(m), 2):
        others = [k for k in range(m) if k not in (i, j)]
        total = 0.0
        for r in range(len(others) + 1):
            wgt = math.factorial(r) * math.factorial(m - r - 2) / math.factorial(m - 1)
            for S in itertools.combinations(others, r):
                s = frozenset(S)
                total += wgt * (mu(s | {i, j}) - mu(s | {i}) - mu(s | {j}) + mu(s))
        I[i, j] = I[j, i] = total
    return I


def is_monotone(mu, m: int, tol: float = 1e-12) -> bool:
    """mu(S) <= mu(T) für S eine Teilmenge von T, geprüft über alle Mengen (m <= 8)."""
    for r in range(m):
        for S in itertools.combinations(range(m), r):
            s = frozenset(S)
            for j in range(m):
                if j not in s and mu(s | {j}) < mu(s) - tol:
                    return False
    return True


def parameter_counts(m: int) -> dict:
    """Zahl der Parameter: gewichtete Summe, 2-additive Kapazität, volle Kapazität (ohne mu(leer) und mu(alle))."""
    return {"summe": m, "zwei_additiv": m + m * (m - 1) // 2, "voll": 2 ** m - 2}
