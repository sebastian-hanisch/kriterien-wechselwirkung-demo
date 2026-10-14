"""Normierungen (Min-Max, Perzentil-Ränge), gewichtete Summe, Kendalls tau-b, Nutzenverlust. Nur numpy; eigene Kopie der kleinen Helfer aus den Vorgängern (kein Import über Repos).
Alle Funktionen arbeiten im vorzeichenbehafteten Raum, in dem HÖHER BESSER gilt."""

from __future__ import annotations

import numpy as np

NORMS = ("perzentil", "minmax")


def minmax(S: np.ndarray) -> np.ndarray:
    """Min-Max-Normierung je Kriterium über die Stichprobe; konstante Spalte -> lauter Nullen."""
    S = np.asarray(S, dtype=float)
    rng = S.max(axis=0) - S.min(axis=0)
    return (S - S.min(axis=0)) / np.where(rng == 0, 1.0, rng)


def average_ranks(col: np.ndarray) -> np.ndarray:
    """Ränge 1..n mit Mittelwert bei Gleichständen (wie scipy.stats.rankdata, method='average')."""
    col = np.asarray(col, dtype=float)
    order = np.argsort(col, kind="mergesort")
    sorted_col = col[order]
    ranks = np.empty(len(col), dtype=float)
    i, n = 0, len(col)
    while i < n:
        j = i
        while j + 1 < n and sorted_col[j + 1] == sorted_col[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def percentile_normalize(S: np.ndarray) -> np.ndarray:
    """Perzentil-Normierung: (mittlerer Rang - 1) / (n - 1) je Kriterium; bei n < 2 lauter Nullen."""
    S = np.asarray(S, dtype=float)
    n = len(S)
    if n < 2:
        return np.zeros_like(S)
    return np.column_stack([(average_ranks(S[:, k]) - 1.0) / (n - 1.0) for k in range(S.shape[1])])


def normalize(S: np.ndarray, how: str) -> np.ndarray:
    if how == "minmax":
        return minmax(S)
    if how == "perzentil":
        return percentile_normalize(S)
    raise ValueError(f"unbekannte Normierung: {how}")


def weighted_score(N: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(N, dtype=float) @ np.asarray(w, dtype=float)


def top1(score: np.ndarray) -> int:
    """Index des höchsten Scores; bei Gleichstand die kleinste Nummer."""
    return int(np.argmax(score))


def utility_loss(utility_values: np.ndarray, chosen: int) -> float:
    return float(np.max(utility_values) - utility_values[chosen])


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendalls tau-b (Gleichstände berücksichtigt); 0, wenn eine Seite keine Unterschiede hat."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = np.sign(a[:, None] - a[None, :])
    sb = np.sign(b[:, None] - b[None, :])
    denom = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return 0.0 if denom == 0 else float((sa * sb).sum() / denom)
