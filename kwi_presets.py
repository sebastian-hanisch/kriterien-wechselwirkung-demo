"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler werden immer gezeichnet (die Stärke der Wechselwirkung ist ohne Wechselwirkung deaktiviert); es gibt deshalb kein KEPT-Muster."""

import random
from dataclasses import dataclass

import streamlit as st

import kwi_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "structure_select": SettingSpec("st", str, C.DEFAULT_STRUCTURE, choices=C.STRUCTURES),
    "rho_select": SettingSpec("rho", float, C.DEFAULT_RHO, choices=C.RHO_CHOICES),
    "decider_select": SettingSpec("dc", str, C.DEFAULT_DECIDER, choices=C.DECIDERS),
    "lam_select": SettingSpec("lam", float, C.DEFAULT_LAM, choices=C.LAM_CHOICES),
    "survey_select": SettingSpec("sv", str, C.DEFAULT_SURVEY, choices=C.SURVEYS),
    "norm_select": SettingSpec("nm", str, C.DEFAULT_NORM, choices=C.NORMS),
    "thr_select": SettingSpec("thr", float, C.DEFAULT_THR, choices=C.THR_CHOICES),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "structure": "structure_select", "rho": "rho_select", "decider": "decider_select", "lam": "lam_select", "survey": "survey_select",
               "norm": "norm_select", "thr": "thr_select", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
