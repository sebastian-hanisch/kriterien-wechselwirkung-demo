"""Szenario: Kriterien in Aspekten (Blockkorrelation), Aspektgewichte des Entscheiders und sein Nutzen (additiv in Aspekten, Redundanz, Synergie, je Kriterium).

Der Generator Version 1 kennt nur Gleichkorrelation über alle Kriterien; die Blockmatrix, die Aspekte und die Entscheider entstehen hier aus seinen Bausteinen
(SplitMix64, normal_cdf, to_raw, directions, decision_maker_weights). Der Entscheider rechnet auf der Güte (Quantil), die Verfahren sehen nur die Rohwerte."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from entscheidungsmatrix_generator import M_MAX, SplitMix64, decision_maker_weights, directions, normal_cdf, to_raw


def parse_structure(text: str) -> tuple[int, ...]:
    """'3-2-1' -> (3, 2, 1): Zahl der Kriterien je Aspekt; höchstens M_MAX Kriterien insgesamt."""
    sizes = tuple(int(p) for p in text.split("-"))
    if not sizes or any(s < 1 for s in sizes):
        raise ValueError(f"ungültige Struktur: {text}")
    if sum(sizes) > M_MAX:
        raise ValueError(f"höchstens {M_MAX} Kriterien, Struktur {text} hat {sum(sizes)}")
    return sizes


def blocks_of(sizes) -> list[np.ndarray]:
    """Spaltenindizes je Aspekt (aufeinanderfolgende Kriterien)."""
    out, k = [], 0
    for s in sizes:
        out.append(np.arange(k, k + s))
        k += s
    return out


def block_correlation(sizes, rho: float) -> np.ndarray:
    """Korrelationsmatrix: rho innerhalb eines Aspekts, 0 zwischen den Aspekten. Zulässig ist -1/(k-1) < rho < 1 für den größten Aspekt der Größe k."""
    k = max(sizes)
    lower = -1.0 if k <= 1 else -1.0 / (k - 1)
    if not (lower < rho < 1.0):
        raise ValueError(f"rho = {rho} liegt nicht in ({lower:.3f}, 1) für einen Aspekt mit {k} Kriterien")
    m = int(sum(sizes))
    corr = np.eye(m)
    for b in blocks_of(sizes):
        for i in b:
            for j in b:
                if i != j:
                    corr[i, j] = rho
    return corr


def correlated_goodness(n: int, sizes, rho: float, rng: SplitMix64) -> np.ndarray:
    """n x m Güten, Randverteilungen gleichverteilt auf (0, 1), Blockkorrelation rho der zugrunde liegenden Normalverteilten."""
    m = int(sum(sizes))
    z = rng.normal(n * m).reshape(n, m)
    z = z @ np.linalg.cholesky(block_correlation(sizes, rho)).T
    return np.clip(normal_cdf(z), 1e-12, 1.0 - 1e-12)


@dataclass(frozen=True)
class World:
    """Eine Matrix mit Aspekten und simuliertem Entscheider (die Nutzen eines Entscheiders kommen aus `decider_utility`)."""

    n: int
    m: int
    sizes: tuple
    rho: float
    goodness: np.ndarray     # n x m Güte in (0, 1)
    S: np.ndarray            # n x m Rohwerte mit Vorzeichen der Richtung (höher = besser)
    v: np.ndarray            # Aspektgewichte des Entscheiders (Summe 1)
    wc: np.ndarray           # Kriteriengewichte des Entscheiders, wenn er jedes Kriterium für sich bewertet (Summe 1)


def make_world(n: int, sizes, rho: float, seed: int) -> World:
    """Gleiche Reihenfolge der Zufallszahlen wie in der Vorab-Messreihe: Normalverteilte, dann Aspektgewichte, dann Kriteriengewichte."""
    sizes = tuple(int(s) for s in sizes)
    m = int(sum(sizes))
    rng = SplitMix64(seed)
    g = correlated_goodness(n, sizes, rho, rng)
    v = decision_maker_weights(len(sizes), rng)
    wc = decision_maker_weights(m, rng)
    S = to_raw(g, m) * directions(m)
    return World(n, m, sizes, float(rho), g, S, v, wc)


def aspect_utility(g: np.ndarray, sizes, v: np.ndarray, lam: float = 0.0, kind: str = "max") -> np.ndarray:
    """u = Summe_a v_a * ((1 - lam) * Mittel + lam * (Maximum | Minimum)) über die Werte der Kriterien des Aspekts: ein Choquet-Integral (Konvexkombination von Kapazitäten).
    Auf Güten ist das der Nutzen des Entscheiders, auf normierten Rohwerten die Rechnung des Verfahrens mit der richtigen Wechselwirkung."""
    g = np.asarray(g, dtype=float)
    if kind not in ("max", "min"):
        raise ValueError(f"unbekannte Wechselwirkung: {kind}")
    u = np.zeros(len(g))
    for a, b in enumerate(blocks_of(sizes)):
        x = g[:, b]
        spec = x.max(axis=1) if kind == "max" else x.min(axis=1)
        u += v[a] * ((1.0 - lam) * x.mean(axis=1) + lam * spec)
    return u


def decider_utility(world: World, decider: str, lam: float = 1.0) -> np.ndarray:
    """Nutzen des Entscheiders: additiv in Aspekten (Mittel), Redundanz (Maximum), Synergie (Minimum) oder jedes Kriterium für sich (Gewichte wc)."""
    if decider == "aspekt":
        return aspect_utility(world.goodness, world.sizes, world.v)
    if decider == "redundanz":
        return aspect_utility(world.goodness, world.sizes, world.v, lam, "max")
    if decider == "synergie":
        return aspect_utility(world.goodness, world.sizes, world.v, lam, "min")
    if decider == "einzeln":
        return world.goodness @ world.wc
    raise ValueError(f"unbekannter Entscheider: {decider}")
