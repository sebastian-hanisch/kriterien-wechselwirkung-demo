"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 5, 40, 12
SEED_MAX, DEFAULT_SEED = 999999, 1           # je Preset geprüft (tests/test_presets.py)

# Aspekte: Zahl der Kriterien je Aspekt, von links nach rechts. Höchstens 8 Kriterien (Länge der Kriterientabelle des Generators).
STRUCTURES = ("2-2-2", "3-2-1", "4-1-1", "2-2", "3-1-1", "3-3", "2-2-2-2", "1-1-1-1")
DEFAULT_STRUCTURE = "3-2-1"
RHO_CHOICES = (-0.3, 0.0, 0.3, 0.6, 0.9)     # Gleichkorrelation INNERHALB eines Aspekts (zwischen Aspekten 0); -0,3 liegt über der Untergrenze -1/(k-1) auch bei k = 4
DEFAULT_RHO = 0.6

DECIDERS = ("aspekt", "redundanz", "synergie", "einzeln")
DECIDER_LABELS = {
    "aspekt": "Additiv in Aspekten (Mittel der Kriterien je Aspekt)",
    "redundanz": "Redundanz: im Aspekt zählt vor allem das Beste (Maximum)",
    "synergie": "Synergie: im Aspekt zählt vor allem das Schlechteste (Minimum)",
    "einzeln": "Jedes Kriterium für sich (eigenes Gewicht je Kriterium)",
}
DECIDER_SHORT = {"aspekt": "additiv in Aspekten", "redundanz": "Redundanz", "synergie": "Synergie", "einzeln": "je Kriterium"}
DEFAULT_DECIDER = "aspekt"
LAM_CHOICES = (0.25, 0.5, 0.75, 1.0)         # Stärke der Wechselwirkung: 0 = Mittel, 1 = reines Maximum bzw. Minimum
DEFAULT_LAM = 1.0

SURVEYS = ("kriterium", "geteilt")
SURVEY_LABELS = {
    "kriterium": "Wichtigkeit je Kriterium erfragt (jedes Kriterium bekommt die Wichtigkeit seines Aspekts)",
    "geteilt": "Aspektgewicht auf die Kriterien des Aspekts geteilt",
}
DEFAULT_SURVEY = "kriterium"

NORMS = ("perzentil", "minmax")
NORM_LABELS = {"perzentil": "Perzentil-Ränge (gemeinsame Skala)", "minmax": "Min-Max (je Kriterium)"}
DEFAULT_NORM = "perzentil"

THR_CHOICES = (0.3, 0.5, 0.7)                # Schwelle |r| für die Gruppierung der Kriterien aus den Daten
DEFAULT_THR = 0.5

# Feste Instanzen für alle Experimente: Mittel über SWEEP_SEEDS, nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(200000, 200200))
EXP_N = 20                                     # Experimente: feste Instanz mit 20 Alternativen, unabhängig vom Regler
EXP_STRUCTURES = ("2-2-2", "3-2-1", "4-1-1")
EXP_RHOS = (0.0, 0.3, 0.6, 0.9)
EXP_LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXP_LOSS_KS = (1, 2, 3, 4, 5)
EXP_INTERACTION_STRUCTURE = "2-2-2"
EXP_DATA_STRUCTURES = ("1-1-1-1-1-1", "2-2-2", "3-2-1")
EXP_BUNDLED = (0.20, 0.20, 0.20, 0.10, 0.10, 0.20)   # gebündelte Gewichte je Kriterium bei (3, 2, 1): 60 % auf den drei korrelierten Kriterien
EXP_CRITERION_RHOS = (0.0, 0.6, 0.9)
EXP_CORRELATION_RHOS = (-0.4, 0.0, 0.3, 0.6, 0.9)     # Aspekte der Größe 2: Untergrenze -1

COLOR_CHOICE = "#2F6B65"
COLOR_BEST = "#D68A2E"
COLOR_SUM = "#8A96A6"
COLOR_WARN = "#c0392b"
COLOR_LINE = "#4c78a8"

PRESETS = {
    "Standardfall ((3, 2, 1), Korrelation 0,6)": {"n": DEFAULT_N, "structure": DEFAULT_STRUCTURE, "rho": DEFAULT_RHO, "decider": DEFAULT_DECIDER, "lam": DEFAULT_LAM,
                                                  "survey": DEFAULT_SURVEY, "norm": DEFAULT_NORM, "thr": DEFAULT_THR, "seed": DEFAULT_SEED},
    "Zählt dreifach": {"n": 12, "structure": "4-1-1", "rho": 0.9, "decider": "aspekt", "lam": 1.0, "survey": "kriterium", "norm": "perzentil", "thr": 0.5, "seed": 4},
    "CRITIC hilft": {"n": 12, "structure": "4-1-1", "rho": 0.9, "decider": "aspekt", "lam": 1.0, "survey": "kriterium", "norm": "perzentil", "thr": 0.5, "seed": 14},
    "Redundanz": {"n": 12, "structure": "2-2-2", "rho": 0.0, "decider": "redundanz", "lam": 1.0, "survey": "geteilt", "norm": "perzentil", "thr": 0.5, "seed": 1},
    "Synergie": {"n": 12, "structure": "2-2-2", "rho": 0.0, "decider": "synergie", "lam": 1.0, "survey": "geteilt", "norm": "perzentil", "thr": 0.5, "seed": 1},
}
PRESET_HELP = {
    "Standardfall ((3, 2, 1), Korrelation 0,6)": "12 Alternativen, 6 Kriterien in Aspekten mit 3, 2 und 1 Kriterien, Korrelation 0,6 im Aspekt; der Entscheider denkt in Aspekten, die Wichtigkeit wird je Kriterium erfragt.",
    "Zählt dreifach": "Ein Aspekt mit vier Kriterien (Korrelation 0,9), Wichtigkeit je Kriterium erfragt: Der Aspekt zählt vierfach, die erhobene Summe verfehlt den Besten, die richtig geteilte trifft.",
    "CRITIC hilft": "Derselbe Fall: Der CRITIC-Faktor nimmt dem korrelierten Aspekt einen Teil des Übergewichts und trifft den Besten, den die erhobene Summe verfehlt.",
    "Redundanz": "Im Aspekt zählt das Beste: Keine gewichtete Summe trifft, das Choquet-Integral mit der richtigen Wechselwirkung trifft.",
    "Synergie": "Im Aspekt zählt das Schlechteste: Die gewichtete Summe verfehlt, das Choquet-Integral mit der richtigen Wechselwirkung trifft (auf gemeinsamer Skala).",
}
