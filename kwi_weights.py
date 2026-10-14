"""Gewichte je Kriterium: erhoben, richtig geteilt, CRITIC, Entropie, Gruppierung aus den Daten; Aspektanteile und Aufblähung. Nur numpy."""

from __future__ import annotations

import math

import numpy as np

from kwi_scenario import blocks_of


def expand_weights(v: np.ndarray, sizes, split: bool) -> np.ndarray:
    """Aspektgewichte auf Kriterien, auf Summe 1: split = Gewicht des Aspekts durch die Zahl seiner Kriterien (richtig), sonst die Wichtigkeit des Aspekts für JEDES Kriterium
    (die Erhebung je Kriterium)."""
    w = np.zeros(int(sum(sizes)))
    for a, b in enumerate(blocks_of(sizes)):
        w[b] = v[a] / len(b) if split else v[a]
    return w / w.sum()


def aspect_share(w: np.ndarray, sizes) -> np.ndarray:
    """Anteil der Gesamtgewichte je Aspekt."""
    w = np.asarray(w, dtype=float)
    return np.array([w[b].sum() for b in blocks_of(sizes)])


def inflation(w: np.ndarray, sizes, v: np.ndarray) -> np.ndarray:
    """Aufblähung je Aspekt: Anteil des Aspekts im Verfahren durch den richtigen Anteil v_a (1 = richtig gezählt, 2 = doppelt)."""
    return aspect_share(w, sizes) / np.asarray(v, dtype=float)


def _normalize(c: np.ndarray, m: int) -> np.ndarray:
    total = float(c.sum())
    return np.full(m, 1.0 / m) if not total > 0 else c / total


def critic_weights(N: np.ndarray) -> np.ndarray:
    """CRITIC: w_j proportional s_j * Summe_k (1 - r_jk), s_j die Standardabweichung (ddof = 1) der normierten Spalte, r die Korrelation. Entartung: gleiche Gewichte."""
    N = np.asarray(N, dtype=float)
    m = N.shape[1]
    if len(N) < 2:
        return np.full(m, 1.0 / m)
    sd = N.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.nan_to_num(np.atleast_2d(np.corrcoef(N.T)), nan=0.0)
    np.fill_diagonal(R, 1.0)
    return _normalize(sd * (1.0 - R).sum(axis=0), m)


def entropy_weights(N: np.ndarray) -> np.ndarray:
    """Entropie-Gewichte: p_ij = N_ij / Spaltensumme, e_j = -Summe p ln p / ln n, w_j proportional 1 - e_j. Eine Spalte ohne Summe trägt keine Information (e = 1)."""
    N = np.asarray(N, dtype=float)
    n, m = N.shape
    if n < 2:
        return np.full(m, 1.0 / m)
    colsum = N.sum(axis=0)
    P = N / np.where(colsum == 0, 1.0, colsum)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log(P), 0.0)
    e = np.where(colsum == 0, 1.0, -t.sum(axis=0) / math.log(n))
    return _normalize(np.clip(1.0 - e, 0.0, None), m)


def components(N: np.ndarray, thr: float) -> list[list[int]]:
    """Gruppen von Kriterien aus den Daten: zwei Kriterien gehören zusammen, wenn |r| >= thr (zusammenhängende Komponenten)."""
    N = np.asarray(N, dtype=float)
    m = N.shape[1]
    if len(N) < 2:
        return [[j] for j in range(m)]
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.abs(np.nan_to_num(np.atleast_2d(np.corrcoef(N.T)), nan=0.0))
    parent = list(range(m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(m):
        for j in range(i + 1, m):
            if R[i, j] >= thr:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(m):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def cluster_weights(N: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Jede Gruppe aus `components` bekommt das gleiche Gewicht, innerhalb der Gruppe geteilt."""
    groups = components(N, thr)
    w = np.zeros(np.asarray(N).shape[1])
    for gidx in groups:
        w[gidx] = 1.0 / len(groups) / len(gidx)
    return w


def aspect_equal_weights(sizes) -> np.ndarray:
    """Jeder Aspekt (bekannte Struktur) bekommt das gleiche Gewicht, innerhalb geteilt."""
    w = np.zeros(int(sum(sizes)))
    for b in blocks_of(sizes):
        w[b] = 1.0 / len(sizes) / len(b)
    return w


def combine(w: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Gewichte mal Faktor, auf Summe 1 (z. B. erhobene Gewichte mal CRITIC-Gewichte)."""
    p = np.asarray(w, dtype=float) * np.asarray(factor, dtype=float)
    return _normalize(p, len(p))
