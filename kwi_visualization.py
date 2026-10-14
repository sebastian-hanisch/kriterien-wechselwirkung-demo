"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import kwi_constants as C
import kwi_scenario as sn
import kwi_weights as wt
from kwi_evaluation import METHOD_LABELS, WEIGHTED, Analysis, de

ARROW = {1: "↑", -1: "↓"}
ASPECT_LETTERS = "ABCDEFGH"
METHOD_COLORS = {"erhoben": C.COLOR_WARN, "richtig": C.COLOR_CHOICE, "erhoben_critic": C.COLOR_BEST, "critic": C.COLOR_LINE, "entropie": C.COLOR_SUM, "cluster": "#7a5195",
                 "gleich": "#bbbbbb", "aspekte_gleich": "#999999"}
WEIGHT_METHODS = ("erhoben", "richtig", "critic", "entropie", "cluster")


def alt_names(n: int) -> list[str]:
    return [f"A{i + 1}" for i in range(n)]


def criterion_names(sizes) -> list[str]:
    """K1 ... Km mit dem Buchstaben des Aspekts: K1 (A), K2 (A), K3 (B) ..."""
    out = []
    for a, b in enumerate(sn.blocks_of(sizes)):
        out += [f"K{int(j) + 1} ({ASPECT_LETTERS[a]})" for j in b]
    return out


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_corr_map(a: Analysis) -> go.Figure:
    """Korrelation der normierten Kriterien; Aspekte als Blöcke auf der Diagonale."""
    names = criterion_names(a.sizes)
    z = a.corr
    fig = go.Figure(go.Heatmap(z=z, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu_r", text=np.round(z, 2), texttemplate="%{text}", textfont=dict(size=10),
                               hovertemplate="%{y} und %{x}: r = %{z:.2f}<extra></extra>", colorbar=dict(title="r", thickness=10)))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=dict(text="Korrelation der Kriterien (Buchstabe: Aspekt des Entscheiders)", font=dict(size=13)))
    return _fixed(fig, 360)


def build_weights(a: Analysis) -> go.Figure:
    names = criterion_names(a.sizes)
    fig = go.Figure()
    for k in WEIGHT_METHODS:
        fig.add_trace(go.Bar(x=names, y=a.weights[k], name=METHOD_LABELS[k].replace("Summe, ", ""), marker_color=METHOD_COLORS[k]))
    fig.update_layout(barmode="group", title=dict(text="Gewichte je Kriterium und Verfahren", font=dict(size=13)))
    fig.update_yaxes(title="Gewicht")
    return _fixed(fig, 360)


def build_inflation(a: Analysis) -> go.Figure:
    """Anteil je Aspekt im Verfahren durch den richtigen Anteil: 1 = richtig gezählt."""
    share_right = wt.aspect_share(a.right, a.sizes)
    labels = [f"Aspekt {ASPECT_LETTERS[i]} ({s} Kriterien)" for i, s in enumerate(a.sizes)]
    fig = go.Figure()
    for k in WEIGHT_METHODS:
        fig.add_trace(go.Bar(x=labels, y=wt.aspect_share(a.weights[k], a.sizes) / share_right, name=METHOD_LABELS[k].replace("Summe, ", ""), marker_color=METHOD_COLORS[k]))
    fig.add_hline(y=1.0, line=dict(color="rgba(80,80,80,0.7)", dash="dot"))
    fig.update_layout(barmode="group", title=dict(text="Aufblähung je Aspekt (1 = richtig gezählt)", font=dict(size=13)))
    fig.update_yaxes(title="Anteil im Verfahren / richtiger Anteil", rangemode="tozero")
    return _fixed(fig, 360)


def build_shapley(a: Analysis) -> go.Figure:
    names = criterion_names(a.sizes)
    fig = go.Figure(go.Bar(x=names, y=a.shapley, marker_color=C.COLOR_CHOICE))
    fig.update_yaxes(title="Shapley-Wert (Wichtigkeit)", rangemode="tozero")
    fig.update_layout(title=dict(text="Wichtigkeit je Kriterium in der Kapazität des Entscheiders", font=dict(size=13)))
    return _fixed(fig, 300)


def build_interaction(a: Analysis) -> go.Figure:
    """Interaktionsindex je Paar: positiv Synergie (beide müssen gut sein), negativ Redundanz (eines genügt)."""
    names = criterion_names(a.sizes)
    z = a.interaction
    lim = max(0.05, float(np.abs(z).max()))
    fig = go.Figure(go.Heatmap(z=z, x=names, y=names, zmin=-lim, zmax=lim, colorscale="RdBu", text=np.round(z, 2), texttemplate="%{text}", textfont=dict(size=10),
                               hovertemplate="%{y} und %{x}: I = %{z:.3f}<extra></extra>", colorbar=dict(title="I", thickness=10)))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=dict(text="Wechselwirkung je Paar (blau: Synergie, rot: Redundanz)", font=dict(size=13)))
    return _fixed(fig, 340)


def build_table(a: Analysis) -> pd.DataFrame:
    n = a.settings.n
    rank = lambda v: np.argsort(np.argsort(-np.asarray(v), kind="stable"), kind="stable") + 1  # noqa: E731
    return pd.DataFrame({
        "Alternative": alt_names(n), "Nutzen (Entscheider)": a.utility, "Rang Entscheider": rank(a.utility), "Rang Summe (wie erhoben)": rank(a.results["erhoben"].score),
        "Rang Summe (richtig geteilt)": rank(a.results["richtig"].score), "Rang Choquet": rank(a.results["choquet"].score),
    })


def style_table(df: pd.DataFrame):
    return df.style.format({"Nutzen (Entscheider)": lambda v: de(v, 3)})


def build_method_table(a: Analysis) -> pd.DataFrame:
    names = alt_names(a.settings.n)
    rows = []
    for k, label in METHOD_LABELS.items():
        r = a.results[k]
        rows.append({"Verfahren": label, "Platz 1": names[r.top1] + (" (Gleichstand)" if r.tie_at_top else ""), "Trifft den Besten": "ja" if r.hit else "nein",
                     "τ gegen Nutzen": de(r.tau, 2), "Nutzenverlust": de(r.loss, 3)})
    return pd.DataFrame(rows)


def weights_table(a: Analysis) -> pd.DataFrame:
    names = criterion_names(a.sizes)
    out = {"Kriterium": names}
    for k in WEIGHTED:
        out[METHOD_LABELS[k].replace("Summe, ", "")] = a.weights[k]
    return pd.DataFrame(out)


def style_weights(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 3)) for c in df.columns if c != "Kriterium"})


def pair_details(a: Analysis, i: int, j: int, aspect: int) -> pd.DataFrame:
    """Je Kriterium des Aspekts: Werte (normiert) der beiden Alternativen; dazu Mittel, Maximum und Minimum je Alternative im Aspekt."""
    b = sn.blocks_of(a.sizes)[aspect]
    names = alt_names(a.settings.n)
    rows = []
    for k in b:
        rows.append({"Zeile": f"K{int(k) + 1}", names[i]: float(a.N[i, k]), names[j]: float(a.N[j, k])})
    xi, xj = a.N[i, b], a.N[j, b]
    for label, f in (("Mittel", np.mean), ("Maximum", np.max), ("Minimum", np.min)):
        rows.append({"Zeile": f"{label} im Aspekt", names[i]: float(f(xi)), names[j]: float(f(xj))})
    return pd.DataFrame(rows)


def style_pair(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 3)) for c in df.columns if c != "Zeile"})


# ----------------------------------------------------------------------------- Experimente

COUNT_LABELS = {"erhoben": "erhoben je Kriterium", "richtig": "richtig geteilt", "erhoben_critic": "erhoben mal CRITIC", "gleich": "gleich je Kriterium", "critic": "CRITIC",
                "entropie": "Entropie", "cluster": "gleich je Gruppe (|r| ≥ 0,5)", "aspekte_gleich": "gleich je Aspekt (Struktur bekannt)"}


def counting_table(rows: list[dict]) -> pd.DataFrame:
    structures = list(dict.fromkeys(r["structure"] for r in rows))
    rhos = list(dict.fromkeys(r["rho"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    out = []
    for st in structures:
        for k in methods:
            row = {"Aspekte": f"({', '.join(st.split('-'))})", "Verfahren": COUNT_LABELS[k]}
            for rho in rhos:
                r = next(x for x in rows if x["structure"] == st and x["rho"] == rho and x["method"] == k)
                row[f"Korr. {de(rho, 1)}"] = f"{de(100 * r['hit'], 1)} %"
            out.append(row)
    return pd.DataFrame(out)


def build_counting(rows: list[dict], structure: str = "4-1-1") -> go.Figure:
    rhos = list(dict.fromkeys(r["rho"] for r in rows))
    fig = go.Figure()
    for k in ("erhoben", "richtig", "erhoben_critic", "gleich"):
        ys = [100 * next(x for x in rows if x["structure"] == structure and x["rho"] == rho and x["method"] == k)["hit"] for rho in rhos]
        fig.add_trace(go.Scatter(x=[de(r, 1) for r in rhos], y=ys, mode="lines+markers", name=COUNT_LABELS[k], line=dict(color=METHOD_COLORS[k], width=3)))
    fig.update_xaxes(title="Korrelation im Aspekt")
    fig.update_yaxes(title="Platz 1 getroffen (%)", range=[0, 102])
    fig.update_layout(title=dict(text=f"Aspekte ({', '.join(structure.split('-'))}): erhobene gegen richtig geteilte Gewichte", font=dict(size=13)))
    return _fixed(fig, 380)


def loss_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Kriterien im großen Aspekt": r["k"], "erhoben": f"{de(100 * r['erhoben'], 1)} %", "richtig geteilt": f"{de(100 * r['richtig'], 1)} %",
                          "Verlust (Punkte)": de(100 * r["loss"], 1)} for r in rows])


def inflation_table(rows: list[dict]) -> pd.DataFrame:
    labels = {"gleich": "gleich je Kriterium", "critic": "CRITIC", "entropie": "Entropie", "cluster": "gleich je Gruppe (|r| ≥ 0,5)"}
    structures = list(dict.fromkeys(r["structure"] for r in rows))
    rhos = list(dict.fromkeys(r["rho"] for r in rows))
    out = []
    for st in structures:
        for k, label in labels.items():
            row = {"Aspekte": f"({', '.join(st.split('-'))})", "Verfahren": label}
            for rho in rhos:
                row[f"Korr. {de(rho, 1)}"] = de(next(x for x in rows if x["structure"] == st and x["rho"] == rho and x["method"] == k)["inflation"], 2)
            out.append(row)
    return pd.DataFrame(out)


def data_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Aspekte": f"({', '.join(r['structure'].split('-'))})", "Korrelation CRITIC-Gewichte mit wahren": de(r["critic_corr"], 2),
                          "Korrelation Entropie-Gewichte mit wahren": de(r["entropy_corr"], 2), "Spannweite Entropie-Gewichte auf Rängen": de(r["entropy_rank_range"], 4)} for r in rows])


def separate_table(rows: list[dict]) -> pd.DataFrame:
    labels = {"wahre": "wahre Gewichte je Kriterium", "wahre_critic": "wahre Gewichte mal CRITIC", "critic": "CRITIC allein", "cluster": "gleich je Gruppe", "gleich": "gleich je Kriterium"}
    rhos = list(dict.fromkeys(r["rho"] for r in rows))
    out = []
    for k, label in labels.items():
        row = {"Verfahren": label}
        for rho in rhos:
            row[f"Korr. {de(rho, 1)}"] = f"{de(100 * next(x for x in rows if x['rho'] == rho and x['method'] == k)['hit'], 1)} %"
        out.append(row)
    return pd.DataFrame(out)


INTERACTION_LABELS = {"summe": "Summe, richtige Gewichte", "anpassung": "beste additive Anpassung", "choquet": "Choquet, richtige Wechselwirkung", "halb": "Choquet, halbe Stärke",
                      "falsch": "Choquet, falsches Vorzeichen"}


def interaction_table(rows: list[dict], kind: str, norm: str) -> pd.DataFrame:
    lams = sorted({r["lam"] for r in rows})
    out = []
    for k, label in INTERACTION_LABELS.items():
        sel = [r for r in rows if r["kind"] == kind and r["norm"] == norm and r["method"] == k]
        if not sel:
            continue
        row = {"Verfahren": label}
        for lam in lams:
            row[f"Stärke {de(lam, 2)}"] = f"{de(100 * next(x for x in sel if x['lam'] == lam)['hit'], 1)} %"
        out.append(row)
    return pd.DataFrame(out)


def agreement_table(rows: list[dict]) -> pd.DataFrame:
    lams = sorted({r["lam"] for r in rows if r["lam"] > 0})
    out = []
    for kind, label in (("max", "Redundanz (Maximum)"), ("min", "Synergie (Minimum)")):
        row = {"Wechselwirkung": label}
        for lam in lams:
            r = next(x for x in rows if x["kind"] == kind and x["lam"] == lam and x["norm"] == "perzentil" and x["method"] == "choquet")
            row[f"Stärke {de(lam, 2)}"] = f"{de(100 * r['same_as_summe'], 1)} %"
        out.append(row)
    return pd.DataFrame(out)


def build_interaction_chart(rows: list[dict], kind: str = "max", norm: str = "perzentil") -> go.Figure:
    fig = go.Figure()
    colors = {"summe": C.COLOR_SUM, "anpassung": "#bbbbbb", "choquet": C.COLOR_CHOICE, "halb": C.COLOR_BEST, "falsch": C.COLOR_WARN}
    for k, label in INTERACTION_LABELS.items():
        sel = sorted((r for r in rows if r["kind"] == kind and r["norm"] == norm and r["method"] == k), key=lambda r: r["lam"])
        if sel:
            fig.add_trace(go.Scatter(x=[de(r["lam"], 2) for r in sel], y=[100 * r["hit"] for r in sel], mode="lines+markers", name=label, line=dict(color=colors[k], width=3)))
    fig.update_xaxes(title="Stärke der Wechselwirkung")
    fig.update_yaxes(title="Platz 1 getroffen (%)", range=[0, 102])
    name = "Redundanz (Maximum im Aspekt)" if kind == "max" else "Synergie (Minimum im Aspekt)"
    fig.update_layout(title=dict(text=f"{name}, {C.NORM_LABELS[norm]}", font=dict(size=13)))
    return _fixed(fig, 380)


def correlation_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Korrelation im Aspekt": de(r["rho"], 1), "Summe": f"{de(100 * r['summe'], 1)} %", "Choquet": f"{de(100 * r['choquet'], 1)} %",
                          "Abstand (Punkte)": de(100 * r["gap"], 1)} for r in rows])


def build_correlation(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[de(r["rho"], 1) for r in rows], y=[100 * r["summe"] for r in rows], name="gewichtete Summe", marker_color=C.COLOR_SUM))
    fig.add_trace(go.Bar(x=[de(r["rho"], 1) for r in rows], y=[100 * r["choquet"] for r in rows], name="Choquet", marker_color=C.COLOR_CHOICE))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Korrelation im Aspekt")
    fig.update_yaxes(title="Platz 1 getroffen (%)", range=[0, 102])
    return _fixed(fig, 360)


def parameter_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Kriterien m": r["m"], "gewichtete Summe": r["summe"], "2-additive Kapazität": r["zwei_additiv"], "volle Kapazität": r["voll"]} for r in rows])
