"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt seinen Effekt."""

import kwi_constants as C
import kwi_evaluation as ev
import kwi_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_and_choices():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["structure"] in C.STRUCTURES and p["rho"] in C.RHO_CHOICES and p["decider"] in C.DECIDERS and p["lam"] in C.LAM_CHOICES, name
        assert p["survey"] in C.SURVEYS and p["norm"] in C.NORMS and p["thr"] in C.THR_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall ((3, 2, 1), Korrelation 0,6)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_labels_cover_every_choice():
    assert set(C.DECIDER_LABELS) == set(C.DECIDERS) == set(C.DECIDER_SHORT)
    assert set(C.SURVEY_LABELS) == set(C.SURVEYS) and set(C.NORM_LABELS) == set(C.NORMS)


def test_every_structure_is_valid_and_fits_the_generator():
    import kwi_scenario as sn

    for st in C.STRUCTURES:
        sizes = sn.parse_structure(st)
        assert sum(sizes) <= 8
        for rho in C.RHO_CHOICES:
            sn.block_correlation(sizes, rho)


def test_sweep_seeds_are_the_documented_fixed_set():
    assert len(C.SWEEP_SEEDS) == 200 and len(set(C.SWEEP_SEEDS)) == 200 and C.SWEEP_SEEDS[0] == 200000


def _analysis(name, **override):
    p = dict(C.PRESETS[name])
    p.update(override)
    return ev.analyse(ev.Settings(**p))


def test_every_preset_shows_the_effect_it_is_named_for():
    """Die Presets sind auf einen Seed abgestimmt, bei dem der Effekt sichtbar ist; ändert sich Generator oder Auswertung, soll das hier auffallen."""
    a = _analysis("Standardfall ((3, 2, 1), Korrelation 0,6)")
    assert a.results["erhoben"].hit and a.results["richtig"].hit

    a = _analysis("Zählt dreifach")
    assert not a.results["erhoben"].hit and a.results["richtig"].hit and a.results["erhoben"].top1 != a.results["richtig"].top1
    assert a.sizes == (4, 1, 1)

    a = _analysis("CRITIC hilft")
    assert not a.results["erhoben"].hit and a.results["erhoben_critic"].hit and a.results["richtig"].hit

    a = _analysis("Redundanz")
    assert a.settings.decider == "redundanz" and not a.results["richtig"].hit and a.results["choquet"].hit

    a = _analysis("Synergie")
    assert a.settings.decider == "synergie" and not a.results["richtig"].hit and a.results["choquet"].hit
