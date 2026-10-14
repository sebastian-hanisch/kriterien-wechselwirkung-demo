"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def dup_cols():
    """Vier Alternativen, drei Kriterien: Spalte 1 und 2 sind gleich (1, 0, 1, 0), Spalte 3 (1, 1, 0, 0) ist zu ihnen unkorreliert (r = 0).
    Standardabweichung (ddof = 1) je Spalte 0,5774; Summe(1 - r) = 1 / 1 / 2. CRITIC = 0,25 / 0,25 / 0,50 (das Duplikat zählt zusammen wie ein Kriterium);
    Entropie = 1/3 je Spalte (e = ln 2 / ln 4 = 0,5 bei allen); Gruppen {1, 2} und {3}, Gruppengewichte 0,25 / 0,25 / 0,50."""
    return np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


@pytest.fixture
def entropy_cols():
    """Spalten (1, 0, 0, 0) und (0,5; 0,5; 0; 0): e = 0 und 0,5, Entropie-Gewichte 2/3 und 1/3."""
    return np.array([[1.0, 0.5], [0.0, 0.5], [0.0, 0.0], [0.0, 0.0]])


@pytest.fixture
def counting():
    """Aspekte (2, 1), v = (0,6; 0,4); erhoben je Kriterium (0,375; 0,375; 0,25), richtig geteilt (0,3; 0,3; 0,4).
    P = (0,9; 0,9; 0), Q = (0,4; 0,4; 1): erhoben P = 0,675 vor Q = 0,55, richtig P = 0,54 hinter Q = 0,64 (der Nutzen des Entscheiders ist 'richtig')."""
    return (2, 1), np.array([0.6, 0.4]), np.array([[0.9, 0.9, 0.0], [0.4, 0.4, 1.0]])


@pytest.fixture
def capacity3():
    """2-additive Kapazität mit m = (0,3; 0,2; 0,3), m12 = 0,3, m13 = -0,1, m23 = 0: mu(1) = 0,3, mu(2) = 0,2, mu(3) = 0,3, mu(12) = 0,8, mu(13) = 0,5, mu(23) = 0,5, mu(123) = 1.
    x = (0,8; 0,4; 0,6): Choquet = 0,4 * 1 + 0,2 * 0,5 + 0,2 * 0,3 = 0,56; Shapley 0,40 / 0,35 / 0,25 (Summe 1); I12 = 0,3, I13 = -0,1, I23 = 0;
    gewichtete Summe mit den Shapley-Gewichten 0,61 (nicht 0,56)."""
    return {0: 0.3, 1: 0.2, 2: 0.3}, {(0, 1): 0.3, (0, 2): -0.1, (1, 2): 0.0}, np.array([0.8, 0.4, 0.6])


@pytest.fixture
def ab_pair():
    """A = (0,9; 0,1), B = (0,5; 0,5): Mittel 0,5 / 0,5 (Gleichstand), Maximum 0,9 / 0,5 (A vorn: Redundanz), Minimum 0,1 / 0,5 (B vorn: Synergie)."""
    return np.array([[0.9, 0.1], [0.5, 0.5]])
