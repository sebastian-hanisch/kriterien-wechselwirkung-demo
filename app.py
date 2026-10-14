"""Kriterien sind nicht unabhängig: Doppelzählung und Choquet - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Achtes Stück der Linie Mehrkriterien-Ranking (Kind der gewichteten Summe). Dieselbe neutrale Entscheidungsmatrix, aber die Kriterien sind in Aspekten gebündelt: Mehrere Kriterien
messen denselben Aspekt, und ein simulierter Entscheider denkt in Aspekten. Gezeigt wird, was die gewichtete Summe daraus macht (Doppelzählung), was CRITIC, Entropie und eine
Gruppierung aus den Daten korrigieren, und was das Choquet-Integral kann, wenn die Präferenz selbst nicht additiv ist.
Kern: kwi_weights.py, kwi_choquet.py, kwi_scenario.py; Generator: entscheidungsmatrix_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import kwi_constants as C
import kwi_scenario as sn
from kwi_evaluation import (
    Settings,
    analyse,
    de,
    experiment_correlation,
    experiment_counting,
    experiment_data_driven,
    experiment_inflation,
    experiment_interaction,
    experiment_loss_by_size,
    experiment_separate,
    parameter_table,
    verdict_correlation,
    verdict_counting,
    verdict_data_driven,
    verdict_interaction,
)
from kwi_pdf_export import build_pdf
from kwi_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from kwi_visualization import (
    agreement_table,
    alt_names,
    build_corr_map,
    build_correlation,
    build_counting,
    build_inflation,
    build_interaction,
    build_interaction_chart,
    build_method_table,
    build_shapley,
    build_table,
    build_weights,
    correlation_table,
    counting_table,
    data_table,
    inflation_table,
    interaction_table,
    loss_table,
    pair_details,
    parameter_table as parameter_frame,
    separate_table,
    style_pair,
    style_table,
    style_weights,
    weights_table,
)

st.set_page_config(page_title="Kriterien-Wechselwirkung – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Die vier Experimente rechnen …")
def _experiments():
    return {
        "counting": experiment_counting(),
        "loss": experiment_loss_by_size(),
        "inflation": experiment_inflation(),
        "data": experiment_data_driven(),
        "random_weights": experiment_separate(rhos=(0.0, 0.3, 0.6, 0.9), bundled=False),
        "bundled": experiment_separate(bundled=True),
        "interaction": experiment_interaction(),
        "correlation": experiment_correlation(),
        "params": parameter_table(),
    }


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


STRUCTURE_LABELS = {s: f"({', '.join(s.split('-'))}): {sum(int(x) for x in s.split('-'))} Kriterien in {len(s.split('-'))} Aspekten" for s in C.STRUCTURES}

st.title("🔗 Kriterien sind nicht unabhängig: Doppelzählung und Choquet")
st.markdown(
    """
**Korrelation ist eine Eigenschaft der Daten, Wechselwirkung eine der Präferenz, und sie brauchen verschiedene Mittel.** Dies ist das achte Stück der Linie **Mehrkriterien-Ranking**:
Messen mehrere Kriterien denselben **Aspekt**, zählt die gewichtete Summe ihn so oft, wie er Kriterien hat, wenn jedes Kriterium sein volles Gewicht bekommt (**Doppelzählung**).
Datengetriebene Gewichte (**CRITIC**, **Entropie**) und eine Gruppierung aus den Daten versprechen Abhilfe, korrigieren aber nur teilweise oder gar nicht. Ist die Präferenz selbst nicht
additiv, weil im Aspekt das Beste zählt (**Redundanz**) oder alle gut sein müssen (**Synergie**), hilft nur ein anderes Aggregationsmodell: das **Choquet-Integral**. Der Entscheider ist simuliert.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Doppelzählung", expanded=True):
    st.markdown(
        """
1. **Aspekte:** Der Entscheider denkt in Aspekten (zum Beispiel Kosten, Qualität, Risiko). Ein Aspekt kann in mehreren Kriterien stecken; Kriterien desselben Aspekts sind korreliert.
2. **Erhebung je Kriterium:** Fragt man je Kriterium nach der Wichtigkeit, bekommt jedes Kriterium die Wichtigkeit seines Aspekts. **Das ist hier eine Annahme über das Verhalten, nicht
   gemessen.** Ein Aspekt mit vier Kriterien zählt dann vierfach; bei gleich großen Aspekten bleibt es richtig.
3. **Datengetriebene Gegenmaßnahmen:** CRITIC gewichtet nach Streuung und Unähnlichkeit (w ∝ s · Σ(1 − r)), Entropie nach der Streuung der Anteile, die Gruppierung gibt jeder
   Gruppe korrelierter Kriterien ein Gewicht. Entropie-Gewichte haben mit Korrelation nichts zu tun; sie stehen hier als Gegenbeispiel.
4. **Wechselwirkung:** Im Aspekt zählt statt des Mittels das Maximum (Redundanz) oder das Minimum (Synergie). Das ist ein Choquet-Integral: Eine **Kapazität** μ(S) gibt jeder Menge S
   von Kriterien ein Gewicht, nicht nur jedem Kriterium. Der **Shapley-Wert** ist die Wichtigkeit, der **Interaktionsindex** zeigt Synergie und Redundanz je Paar.
5. **Gemessen wird** am simulierten Entscheider; ob Platz 1 getroffen wird, hängt an der Struktur der Aspekte, an der Korrelation und an einer gemeinsamen Skala.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Matrix**")
    n = st.slider("Alternativen n", *bounds("n_slider"), key="n_slider", help="Wie viele Alternativen zur Wahl stehen.")
    structure = st.selectbox("Aspekte (Kriterien je Aspekt)", C.STRUCTURES, key="structure_select", format_func=lambda s: STRUCTURE_LABELS[s],
                             help="Wie viele Kriterien denselben Aspekt messen. Bei gleich großen Aspekten ist die Erhebung je Kriterium richtig; bei ungleichen zählt der große Aspekt mehrfach.")
    rho = st.selectbox("Korrelation im Aspekt", C.RHO_CHOICES, key="rho_select", format_func=lambda v: de(v, 1),
                       help="Wie stark Kriterien desselben Aspekts zusammenhängen (zwischen Aspekten 0).")
    st.markdown("**Entscheider (simuliert)**")
    decider = st.selectbox("Entscheider", C.DECIDERS, key="decider_select", format_func=lambda k: C.DECIDER_LABELS[k],
                           help="Wie der verborgene Entscheider bewertet: additiv im Mittel der Kriterien je Aspekt, mit Redundanz oder Synergie im Aspekt, oder jedes Kriterium für sich.")
    lam = st.selectbox("Stärke der Wechselwirkung", C.LAM_CHOICES, key="lam_select", format_func=lambda v: de(v, 2), disabled=decider not in ("redundanz", "synergie"),
                       help="0 wäre das Mittel, 1 das reine Maximum bzw. Minimum im Aspekt. Nur bei Redundanz und Synergie wirksam.")
    st.markdown("**Verfahren**")
    survey = st.selectbox("Erhebung der Wichtigkeit", C.SURVEYS, key="survey_select", format_func=lambda k: C.SURVEY_LABELS[k], disabled=decider == "einzeln",
                          help="Je Kriterium erfragt: jedes Kriterium bekommt die Wichtigkeit seines Aspekts (Annahme). Geteilt: das Aspektgewicht wird auf die Kriterien verteilt.")
    norm = st.selectbox("Normierung", C.NORMS, key="norm_select", format_func=lambda k: C.NORM_LABELS[k],
                        help="Perzentil-Ränge bringen die Kriterien auf eine gemeinsame Skala; Maximum und Minimum brauchen sie. Min-Max skaliert je Kriterium nach den Extremen.")
    thr = st.selectbox("Schwelle |r| für die Gruppierung", C.THR_CHOICES, key="thr_select", format_func=lambda v: de(v, 1),
                       help="Kriterien mit einer Korrelation von mindestens dieser Größe (Betrag) gelten in der Gruppierung aus den Daten als ein Aspekt.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Matrix würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "structure_select": structure, "rho_select": float(rho), "decider_select": decider, "lam_select": float(lam), "survey_select": survey,
                   "norm_select": norm, "thr_select": float(thr), "seed_input": int(seed)})

settings = Settings(int(n), structure, float(rho), decider, float(lam), survey, norm, float(thr), int(seed))
a = _analysis(settings)
names = alt_names(settings.n)
R_E, R_R, R_C = a.results["erhoben"], a.results["richtig"], a.results["choquet"]
infl_er = a.weights["erhoben"][: a.sizes[0]].sum() / a.right[: a.sizes[0]].sum()

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Platz 1: Summe wie erhoben", names[R_E.top1], delta=f"bester nach Nutzen: {names[a.best]} ({'getroffen' if R_E.hit else 'verfehlt'})", delta_color="off",
          help="Gewichtete Summe auf den normierten Werten mit den Gewichten, die die Erhebung ergibt (je Kriterium oder geteilt).")
c2.metric("Platz 1: Summe richtig geteilt", names[R_R.top1], delta=f"τ gegen Nutzen: {de(R_R.tau, 2)} (erhoben {de(R_E.tau, 2)})", delta_color="off",
          help="Gewichtete Summe mit den richtigen Gewichten: Aspektgewicht durch die Zahl der Kriterien des Aspekts. τ ist Kendalls τ-b der ganzen Rangfolge gegen den verborgenen Nutzen.")
c3.metric("Platz 1: Choquet", names[R_C.top1], delta=f"τ {de(R_C.tau, 2)} ({'getroffen' if R_C.hit else 'verfehlt'})", delta_color="off",
          help="Choquet-Integral mit der richtigen Kapazität des Entscheiders. Ohne Wechselwirkung ist es die richtig geteilte gewichtete Summe.")
c4.metric("Aufblähung des ersten Aspekts", f"{de(infl_er, 2)}-fach", delta=f"{a.sizes[0]} von {a.world.m} Kriterien", delta_color="off",
          help="Anteil des ersten Aspekts an den erhobenen Gewichten durch seinen richtigen Anteil. 1 heißt richtig gezählt.")

st.caption("Links die Struktur der Daten, rechts die Gewichte: Der Entscheider denkt in Aspekten (Buchstaben), die Verfahren sehen Kriterien. Die Aufblähung vergleicht den Anteil eines "
           "Aspekts im Verfahren mit seinem richtigen Anteil.")
left, right = st.columns(2)
with left:
    st.plotly_chart(build_corr_map(a), width="stretch", key="corr_chart")
    st.plotly_chart(build_weights(a), width="stretch", key="weights_chart")
with right:
    st.plotly_chart(build_inflation(a), width="stretch", key="inflation_chart")
    if settings.decider in ("redundanz", "synergie"):
        st.plotly_chart(build_interaction(a), width="stretch", key="interaction_chart")
    else:
        st.plotly_chart(build_shapley(a), width="stretch", key="shapley_chart")

with st.expander("Ein Paar im Einzelnen: Mittel, Maximum und Minimum im Aspekt"):
    pa, pb, pc = st.columns(3)
    ia = pa.selectbox("Alternative a", list(range(settings.n)), index=0, format_func=lambda i: names[i], key="pair_a")
    ib = pb.selectbox("Alternative b", list(range(settings.n)), index=min(1, settings.n - 1), format_func=lambda i: names[i], key="pair_b")
    asp = pc.selectbox("Aspekt", list(range(len(a.sizes))), index=0, format_func=lambda i: f"Aspekt {'ABCDEFGH'[i]}", key="pair_aspect")
    if ia == ib:
        st.info("Bitte zwei verschiedene Alternativen wählen.")
    else:
        st.dataframe(style_pair(pair_details(a, int(ia), int(ib), int(asp))), hide_index=True, width="stretch")
        st.caption("Gleiches Mittel kann verschiedene Maxima und Minima haben: Bei Redundanz entscheidet das Maximum, bei Synergie das Minimum, bei additiver Präferenz das Mittel.")

tab_t, tab_m, tab_w = st.tabs(["Alle Alternativen", "Verfahren im Vergleich", "Gewichte"])
with tab_t:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch")
with tab_m:
    st.dataframe(build_method_table(a), hide_index=True, width="stretch")
    st.caption("Dieselbe Matrix, derselbe Entscheider. Ohne Wechselwirkung ist das Choquet-Integral die Summe mit richtig geteilten Gewichten.")
with tab_w:
    st.dataframe(style_weights(weights_table(a)), hide_index=True, width="stretch")
    groups = ", ".join("{" + ", ".join(f"K{j + 1}" for j in g) + "}" for g in a.groups)
    st.caption(f"Gruppen aus den Daten (|r| ≥ {de(settings.thr, 1)}): {groups}. Entscheider-Aspekte: "
               + ", ".join("{" + ", ".join(f"K{int(j) + 1}" for j in b) + "}" for b in sn.blocks_of(a.sizes)) + ".")

if st.button("📄 PDF des aktuellen Falls erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="kriterien-wechselwirkung.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was kostet es, Kriterien einzeln zu zählen?")
if settings.decider == "einzeln":
    st.info("Der Entscheider bewertet jedes Kriterium für sich: Die Gewichte je Kriterium sind richtig, auch bei hoher Korrelation. Korrelation ist nicht Redundanz; eine Korrektur "
            "aus den Daten (CRITIC, Gruppierung) nähme hier Gewicht weg, das der Entscheider wirklich vergibt (Experiment 2).")
elif R_E.top1 == R_R.top1:
    st.info(f"Die erhobenen und die richtig geteilten Gewichte wählen dieselbe Alternative ({names[R_E.top1]}); der große Aspekt ist {de(infl_er, 2)}-fach gewichtet, was hier das Ergebnis "
            "nicht kippt (Experiment 1 zeigt, wie oft es kippt).")
else:
    st.warning(f"Die erhobenen Gewichte wählen {names[R_E.top1]}, die richtig geteilten {names[R_R.top1]}; der Beste des Entscheiders ist {names[a.best]} "
               f"({'die richtigen Gewichte treffen' if R_R.hit else 'beide verfehlen'}). Der große Aspekt ist {de(infl_er, 2)}-fach gewichtet.")
if settings.decider in ("redundanz", "synergie"):
    st.caption(f"Bei Wechselwirkung trifft die Summe mit richtigen Gewichten {'Platz 1' if R_R.hit else 'Platz 1 nicht'}, das Choquet-Integral "
               f"{'trifft' if R_C.hit else 'verfehlt ihn ebenfalls'}; keine Wahl der Gewichte ersetzt die Wechselwirkung (Experiment 3).")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die vier Experimente")
st.caption("200 feste Matrizen (SWEEP_SEEDS), n = 20, unabhängig von den Reglern. Die Experimente rechnen zusammen rund 10 Sekunden und starten erst auf Klick. Standardfehler der "
           "Trefferquoten 2,5 bis 3,5 Punkte: Unterschiede unter etwa 6 Punkten sind nicht belastbar.")
if st.button("▶ Experimente rechnen", key="exp_start"):
    st.session_state["exp_rows"] = _experiments()
exp = st.session_state.get("exp_rows")
if exp:
    st.markdown("### Experiment 1: Ein Aspekt zählt so oft, wie er Kriterien hat")
    st.caption("Der Entscheider denkt in Aspekten (additiv), Min-Max, Platz 1 getroffen. Die Erhebung je Kriterium ist eine Annahme; bei (2, 2, 2) ist sie richtig (Kontrolle).")
    st.plotly_chart(build_counting(exp["counting"], "4-1-1"), width="stretch", key="counting_chart")
    verdict(verdict_counting(exp["counting"], exp["loss"], exp["inflation"]))
    st.dataframe(counting_table(exp["counting"]), hide_index=True, width="stretch")
    lc, ic = st.columns(2)
    with lc:
        st.markdown("**Verlust je Zahl der Kriterien im großen Aspekt** (Aspekte (k, 1, 1), Korrelation 0,9)")
        st.dataframe(loss_table(exp["loss"]), hide_index=True, width="stretch")
    with ic:
        st.markdown("**Aufblähung des großen Aspekts** (gleiche Aspektgewichte; richtig ist 1)")
        st.dataframe(inflation_table(exp["inflation"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 2: Was CRITIC und Entropie korrigieren")
    st.caption("Aufblähung und Trefferquote der datengetriebenen Verfahren; Korrelation der Gewichte mit der wahren Wichtigkeit; und der Fall, in dem der Entscheider korrelierte Kriterien einzeln bewertet.")
    verdict(verdict_data_driven(exp["inflation"], exp["data"], exp["bundled"]))
    st.dataframe(data_table(exp["data"]), hide_index=True, width="stretch")
    rc, bc = st.columns(2)
    with rc:
        st.markdown("**Jedes Kriterium für sich, zufällige Gewichte** (Aspekte (3, 2, 1) als Korrelationsstruktur): Platz 1 getroffen")
        st.dataframe(separate_table(exp["random_weights"]), hide_index=True, width="stretch")
    with bc:
        st.markdown("**Jedes Kriterium für sich, 60 % auf den drei korrelierten**: Platz 1 getroffen")
        st.dataframe(separate_table(exp["bundled"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 3: Wo die gewichtete Summe nicht reicht")
    st.caption("Redundanz (Maximum im Aspekt) und Synergie (Minimum), Aspekte (2, 2, 2), Korrelation 0. Gemeinsame Skala: Perzentil-Ränge; Min-Max zum Vergleich in der Tabelle.")
    verdict(verdict_interaction(exp["interaction"]))
    kc, nc = st.columns(2)
    with kc:
        st.plotly_chart(build_interaction_chart(exp["interaction"], "max", "perzentil"), width="stretch", key="interaction_max")
    with nc:
        st.plotly_chart(build_interaction_chart(exp["interaction"], "min", "perzentil"), width="stretch", key="interaction_min")
    for kind, norm_ in (("max", "perzentil"), ("max", "minmax"), ("min", "perzentil"), ("min", "minmax")):
        st.markdown(f"**{'Redundanz' if kind == 'max' else 'Synergie'}, {C.NORM_LABELS[norm_]}**")
        st.dataframe(interaction_table(exp["interaction"], kind, norm_), hide_index=True, width="stretch")
    st.markdown("**Gleicher Platz 1 von Summe (richtige Gewichte) und Choquet** (Perzentil-Ränge)")
    st.dataframe(agreement_table(exp["interaction"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 4: Korrelation und Wechselwirkung sind zwei Dinge")
    st.caption("Redundanz (Stärke 1), Perzentil-Ränge, Aspekte (2, 2, 2): Platz 1 der Summe und des Choquet je Korrelation im Aspekt; dazu der Preis in Parametern.")
    verdict(verdict_correlation(exp["correlation"], exp["params"]))
    cc, pc_ = st.columns(2)
    with cc:
        st.plotly_chart(build_correlation(exp["correlation"]), width="stretch", key="correlation_chart")
        st.dataframe(correlation_table(exp["correlation"]), hide_index=True, width="stretch")
    with pc_:
        st.markdown("**Zahl der Parameter** (Gewichte der Summe, 2-additive Kapazität m + m(m − 1)/2, volle Kapazität 2^m − 2)")
        st.dataframe(parameter_frame(exp["params"]), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Entscheider denkt in Aspekten, die Erhebung vergibt die Aspekt-Wichtigkeit je Kriterium** | Beides ist angenommen, nicht gemessen. Dass Menschen so antworten, ist hier nicht belegt; die Aussage ist „wenn, dann so viel“. | Gewichte erheben |
| **Die Aspekte sind bekannt** | In der Praxis ist die Zuordnung der Kriterien zu Aspekten selbst eine Modellierung; die Gruppierung aus den Daten findet sie erst bei hoher Korrelation (Experiment 1 und 2). | AHP (Hierarchie) |
| **Die Kapazität ist gegeben** | Das Choquet-Integral wird mit der richtigen Kapazität des Entscheiders gerechnet. Sie aus Fragen oder Beispielen zu erheben, ist der eigentliche Preis und hier nicht gemessen. | Aus Beispielen lernen |
| **Eine gemeinsame Skala ist vorhanden** | Maximum und Minimum brauchen Kriterien auf vergleichbarer Skala; in der Simulation ist das die Güte. Perzentil-Ränge nähern sie an und hängen an der Stichprobe (Experiment 3). | Rangumkehr |
| **Wechselwirkung nur innerhalb der Aspekte** | Mittel, Maximum und Minimum und ihre Mischung; keine Wechselwirkung zwischen Aspekten, keine gemischten Vorzeichen. | – |
| **Die Verfahren zählen ebenfalls je Kriterium** | Die Aufblähung gilt für jedes additive Verfahren (Abstand zum Ideal, Präferenzfluss); hier ist sie nur an der Summe gemessen. | TOPSIS, PROMETHEE |
| **Ein Verfahren am Schluss: additiv genug?** | Wie oft die Summe reicht, hängt von der Stärke der Wechselwirkung ab (Experiment 3); ein Vergleich über viele Verfahren steht im letzten Stück der Linie. | Verfahrensvergleich |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Aspekte.** $m$ Kriterien in Aspekten $A_1, \dots, A_p$ mit $s_a = |A_a|$. Güten $g_{ij} \in (0, 1)$ mit Gleichkorrelation $\rho$ innerhalb eines Aspekts und 0 zwischen Aspekten
(Gauß-Copula). Aspektgewichte $v_a \ge 0$, $\sum_a v_a = 1$. Nutzen des additiven Entscheiders: $u_i = \sum_a v_a \cdot \frac{1}{s_a} \sum_{j \in A_a} g_{ij}$.

**Doppelzählung.** Richtige Kriteriengewichte $w_j = v_a / s_a$ für $j \in A_a$. Erhoben je Kriterium: $\tilde w_j \propto v_a$. Der Anteil des Aspekts $a$ an den erhobenen Gewichten ist
$s_a v_a / \sum_b s_b v_b$, die **Aufblähung** das Verhältnis zu $v_a$; bei gleichen $s_a$ ist sie 1.

**CRITIC.** $w_j \propto s_j \sum_k (1 - r_{jk})$ mit der Standardabweichung $s_j$ der normierten Spalte und der Korrelation $r$. **Entropie.** $p_{ij} = n_{ij} / \sum_i n_{ij}$,
$e_j = -\frac{1}{\ln n} \sum_i p_{ij} \ln p_{ij}$, $w_j \propto 1 - e_j$; auf Perzentil-Rängen sind alle $e_j$ gleich. **Gruppierung.** Kriterien mit $|r_{jk}| \ge \theta$ bilden (transitiv)
eine Gruppe; jede Gruppe bekommt das gleiche Gewicht, innerhalb geteilt.

**Choquet-Integral.** Eine Kapazität $\mu: 2^{\{1..m\}} \to [0, 1]$ ist monoton mit $\mu(\emptyset) = 0$, $\mu(\{1..m\}) = 1$. Für $x$ aufsteigend sortiert $x_{(1)} \le \dots \le x_{(m)}$:
$C_\mu(x) = \sum_{i} (x_{(i)} - x_{(i-1)})\, \mu(\{(i), \dots, (m)\})$ mit $x_{(0)} = 0$. Ist $\mu$ additiv, ist $C_\mu$ die gewichtete Summe; $\mu \equiv 1$ auf nichtleeren Mengen
gibt das Maximum, $\mu = 1$ nur auf der vollen Menge das Minimum, $\mu(S) = |S|/m$ das Mittel. Das Integral ist linear in der Kapazität; der Entscheider mit Stärke $\lambda$ hat im Aspekt
$(1 - \lambda)\,\text{Mittel} + \lambda\,(\text{Maximum} \mid \text{Minimum})$.

**Möbius-Form, Shapley, Interaktion.** Eine 2-additive Kapazität hat $\mu(S) = \sum_{i \in S} m_i + \sum_{\{i,j\} \subseteq S} m_{ij}$ und $C_\mu(x) = \sum_i m_i x_i + \sum_{i<j} m_{ij}
\min(x_i, x_j)$, also $m + \binom{m}{2}$ Parameter (volle Kapazität $2^m - 2$). Shapley-Wert $\varphi_i = \sum_{S \not\ni i} \frac{|S|!\,(m - |S| - 1)!}{m!} (\mu(S \cup i) - \mu(S))$, Summe 1;
Interaktionsindex $I_{ij}$ (positiv: Synergie, negativ: Redundanz); bei 2-additiven Kapazitäten ist $I_{ij} = m_{ij}$.

**Literatur:** CRITIC wurde von Diakoulaki, Mavrotas und Papayannakis (1995) vorgeschlagen, das Choquet-Integral für wechselwirkende Kriterien geht auf Grabisch und Mitarbeiter zurück;
Titel und Fundstellen sind hier nicht im Original geprüft und deshalb nicht angegeben.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
