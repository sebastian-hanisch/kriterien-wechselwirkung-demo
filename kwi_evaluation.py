"""Auswertung: Fall der App, Experimente über feste Matrizen (SWEEP_SEEDS) und Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import kwi_choquet as ch
import kwi_constants as C
import kwi_scenario as sn
import kwi_score as sc
import kwi_weights as wt


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    structure: str = C.DEFAULT_STRUCTURE
    rho: float = C.DEFAULT_RHO
    decider: str = C.DEFAULT_DECIDER
    lam: float = C.DEFAULT_LAM
    survey: str = C.DEFAULT_SURVEY
    norm: str = C.DEFAULT_NORM
    thr: float = C.DEFAULT_THR
    seed: int = C.DEFAULT_SEED


METHODS = ("erhoben", "richtig", "erhoben_critic", "critic", "entropie", "cluster", "gleich", "aspekte_gleich", "choquet")
METHOD_LABELS = {
    "erhoben": "Summe, Gewichte wie erhoben",
    "richtig": "Summe, richtig geteilte Gewichte",
    "erhoben_critic": "Summe, erhoben mal CRITIC-Faktor",
    "critic": "Summe, CRITIC-Gewichte",
    "entropie": "Summe, Entropie-Gewichte",
    "cluster": "Summe, gleich je Gruppe aus den Daten",
    "gleich": "Summe, gleich je Kriterium",
    "aspekte_gleich": "Summe, gleich je Aspekt (Struktur bekannt)",
    "choquet": "Choquet-Integral, richtige Kapazität",
}
WEIGHTED = ("erhoben", "richtig", "erhoben_critic", "critic", "entropie", "cluster", "gleich", "aspekte_gleich")


@dataclass
class Result:
    score: np.ndarray          # höher ist besser
    top1: int
    hit: bool
    tau: float
    loss: float
    tie_at_top: bool


def _result(score: np.ndarray, utility: np.ndarray, best: int) -> Result:
    t = sc.top1(score)
    return Result(score, t, t == best, sc.kendall_tau_b(score, utility), sc.utility_loss(utility, t), int((score == score.max()).sum()) > 1)


@dataclass
class Analysis:
    settings: Settings
    world: sn.World
    sizes: tuple
    N: np.ndarray
    utility: np.ndarray
    best: int
    weights: dict              # Methode -> Gewichte je Kriterium (nur gewichtete Verfahren)
    right: np.ndarray          # richtige Gewichte je Kriterium
    results: dict              # Methode -> Result
    corr: np.ndarray           # Korrelation der normierten Kriterien
    groups: list               # Gruppen aus den Daten (Schwelle thr)
    shapley: np.ndarray
    interaction: np.ndarray


def method_scores(N: np.ndarray, world_v: np.ndarray, wc: np.ndarray, sizes, decider: str, survey: str, thr: float) -> tuple[dict, np.ndarray]:
    """Gewichte je Verfahren und die richtigen Gewichte. Bei jedem Kriterium für sich (decider einzeln) sind die erhobenen Gewichte die wahren."""
    if decider == "einzeln":
        right = np.asarray(wc, dtype=float)
        erhoben = right
    else:
        right = wt.expand_weights(world_v, sizes, True)
        erhoben = wt.expand_weights(world_v, sizes, survey == "geteilt")
    critic = wt.critic_weights(N)
    weights = {
        "erhoben": erhoben,
        "richtig": right,
        "erhoben_critic": wt.combine(erhoben, critic),
        "critic": critic,
        "entropie": wt.entropy_weights(N),
        "cluster": wt.cluster_weights(N, thr),
        "gleich": np.full(N.shape[1], 1.0 / N.shape[1]),
        "aspekte_gleich": wt.aspect_equal_weights(sizes),
    }
    return weights, right


def _correlation(N: np.ndarray) -> np.ndarray:
    """Korrelation der Spalten; konstante Spalten (Division durch 0) und n = 1 geben 0 außerhalb der Diagonalen."""
    m = N.shape[1]
    if len(N) < 2:
        return np.eye(m)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.nan_to_num(np.atleast_2d(np.corrcoef(N.T)), nan=0.0)
    np.fill_diagonal(R, 1.0)
    return R


def analyse(settings: Settings) -> Analysis:
    s = settings
    sizes = sn.parse_structure(s.structure)
    world = sn.make_world(s.n, sizes, s.rho, s.seed)
    N = sc.normalize(world.S, s.norm)
    u = sn.decider_utility(world, s.decider, s.lam)
    best = sc.top1(u)
    weights, right = method_scores(N, world.v, world.wc, sizes, s.decider, s.survey, s.thr)
    results = {k: _result(sc.weighted_score(N, w), u, best) for k, w in weights.items()}
    if s.decider in ("redundanz", "synergie"):
        kind = "max" if s.decider == "redundanz" else "min"
        lam = s.lam
        mu = ch.aspect_capacity(sizes, world.v, lam, kind)
        score = sn.aspect_utility(N, sizes, world.v, lam, kind)
    elif s.decider == "aspekt":
        mu = ch.aspect_capacity(sizes, world.v, 0.0, "max")
        score = sn.aspect_utility(N, sizes, world.v)
    else:
        w = world.wc
        mu = lambda S: float(sum(w[i] for i in S))  # noqa: E731
        score = sc.weighted_score(N, w)
    results["choquet"] = _result(score, u, best)
    m = world.m
    return Analysis(s, world, sizes, N, u, best, weights, right, results, _correlation(N),
                    wt.components(N, s.thr), ch.shapley(mu, m), ch.interaction_index(mu, m))


# ----------------------------------------------------------------------------- Hilfen für die Experimente

def _world_n(sizes, rho, seed, n=C.EXP_N):
    return sn.make_world(n, sizes, rho, seed)


def _hits(rows_acc: dict, name: str, score: np.ndarray, best: int, u: np.ndarray) -> None:
    h, t = rows_acc.setdefault(name, ([], []))
    h.append(sc.top1(score) == best)
    t.append(sc.kendall_tau_b(score, u))


def _rows(acc: dict, **keys) -> list[dict]:
    out = []
    for name, (h, t) in acc.items():
        p = float(np.mean(h))
        out.append({**keys, "method": name, "hit": p, "tau": float(np.mean(t)), "se": float(np.sqrt(p * (1 - p) / len(h)))})
    return out


# ----------------------------------------------------------------------------- Experiment 1: ein Aspekt zählt so oft, wie er Kriterien hat

COUNTING_METHODS = ("erhoben", "richtig", "erhoben_critic", "gleich", "critic", "entropie", "cluster", "aspekte_gleich")


def experiment_counting(structures=C.EXP_STRUCTURES, rhos=C.EXP_RHOS, seeds=C.SWEEP_SEEDS, n=C.EXP_N, thr=0.5) -> list[dict]:
    """Der Entscheider denkt in Aspekten (additiv); Min-Max. Platz 1 getroffen je Verfahren, Aspekte (k, ...), Korrelation im Aspekt."""
    rows = []
    for st in structures:
        sizes = sn.parse_structure(st)
        for rho in rhos:
            acc: dict = {}
            for s in seeds:
                w = _world_n(sizes, rho, s, n)
                N = sc.minmax(w.S)
                u = sn.aspect_utility(w.goodness, sizes, w.v)
                best = sc.top1(u)
                weights, _ = method_scores(N, w.v, w.wc, sizes, "aspekt", "kriterium", thr)
                for name in COUNTING_METHODS:
                    _hits(acc, name, N @ weights[name], best, u)
            rows += _rows(acc, structure=st, rho=rho)
    return rows


def experiment_loss_by_size(ks=C.EXP_LOSS_KS, rho=0.9, seeds=C.SWEEP_SEEDS, n=C.EXP_N) -> list[dict]:
    """Verlust von Platz 1 (erhoben gegen richtig geteilt) je Zahl k der Kriterien im großen Aspekt, Aspekte (k, 1, 1)."""
    rows = []
    for k in ks:
        sizes = (k, 1, 1)
        he, hr = [], []
        for s in seeds:
            w = _world_n(sizes, rho, s, n)
            N = sc.minmax(w.S)
            u = sn.aspect_utility(w.goodness, sizes, w.v)
            best = sc.top1(u)
            he.append(sc.top1(N @ wt.expand_weights(w.v, sizes, False)) == best)
            hr.append(sc.top1(N @ wt.expand_weights(w.v, sizes, True)) == best)
        rows.append({"k": k, "erhoben": float(np.mean(he)), "richtig": float(np.mean(hr)), "loss": float(np.mean(hr) - np.mean(he))})
    return rows


def experiment_inflation(structures=("3-2-1", "4-1-1"), rhos=C.EXP_RHOS, seeds=C.SWEEP_SEEDS, n=C.EXP_N, thr=0.5) -> list[dict]:
    """Aufblähung des ersten (größten) Aspekts gegen den richtigen Anteil bei gleichen Aspektgewichten (1 / Zahl der Aspekte)."""
    rows = []
    for st in structures:
        sizes = sn.parse_structure(st)
        right = 1.0 / len(sizes)
        for rho in rhos:
            acc = {"gleich": [], "critic": [], "entropie": [], "cluster": []}
            for s in seeds:
                N = sc.minmax(_world_n(sizes, rho, s, n).S)
                m = N.shape[1]
                acc["gleich"].append(wt.aspect_share(np.full(m, 1.0 / m), sizes)[0] / right)
                acc["critic"].append(wt.aspect_share(wt.critic_weights(N), sizes)[0] / right)
                acc["entropie"].append(wt.aspect_share(wt.entropy_weights(N), sizes)[0] / right)
                acc["cluster"].append(wt.aspect_share(wt.cluster_weights(N, thr), sizes)[0] / right)
            for k, v in acc.items():
                rows.append({"structure": st, "rho": rho, "method": k, "inflation": float(np.mean(v))})
    return rows


# ----------------------------------------------------------------------------- Experiment 2: was CRITIC und Entropie korrigieren

def experiment_data_driven(structures=C.EXP_DATA_STRUCTURES, seeds=C.SWEEP_SEEDS, n=C.EXP_N) -> list[dict]:
    """Sagen CRITIC und Entropie etwas über die Wichtigkeit? Korrelation der Verfahrensgewichte mit den wahren Kriteriengewichten (Korrelation 0,5 im Aspekt) und Spannweite der
    Entropie-Gewichte auf Perzentil-Rängen."""
    rows = []
    for st in structures:
        sizes = sn.parse_structure(st)
        c, e, r = [], [], []
        for s in seeds:
            w = _world_n(sizes, 0.5, s, n)
            N = sc.minmax(w.S)
            c.append(float(np.corrcoef(wt.critic_weights(N), w.wc)[0, 1]))
            e.append(float(np.corrcoef(wt.entropy_weights(N), w.wc)[0, 1]))
            r.append(float(np.ptp(wt.entropy_weights(sc.percentile_normalize(w.S)))))
        rows.append({"structure": st, "m": int(sum(sizes)), "critic_corr": float(np.mean(c)), "entropy_corr": float(np.mean(e)), "entropy_rank_range": float(np.mean(r))})
    return rows


SEPARATE_METHODS = ("wahre", "wahre_critic", "critic", "cluster", "gleich")


def experiment_separate(rhos=C.EXP_CRITERION_RHOS, bundled: bool = True, seeds=C.SWEEP_SEEDS, n=C.EXP_N, thr=0.5) -> list[dict]:
    """Korrelation ohne Redundanz: der Entscheider bewertet jedes Kriterium für sich (Aspekte (3, 2, 1) nur als Korrelationsstruktur). bundled: 60 % der Gewichte auf den drei
    korrelierten Kriterien (je 20 %); sonst zufällige Kriteriengewichte."""
    sizes = (3, 2, 1)
    wf = np.asarray(C.EXP_BUNDLED, dtype=float)
    rows = []
    for rho in rhos:
        acc: dict = {}
        for s in seeds:
            w = _world_n(sizes, rho, s, n)
            N = sc.minmax(w.S)
            wc = wf if bundled else w.wc
            u = w.goodness @ wc
            best = sc.top1(u)
            critic = wt.critic_weights(N)
            cand = {"wahre": N @ wc, "wahre_critic": N @ wt.combine(wc, critic), "critic": N @ critic, "cluster": N @ wt.cluster_weights(N, thr), "gleich": N.mean(axis=1)}
            for name in SEPARATE_METHODS:
                _hits(acc, name, cand[name], best, u)
        rows += _rows(acc, rho=rho, bundled=bundled)
    return rows


# ----------------------------------------------------------------------------- Experiment 3: wo die Summe nicht reicht

def _best_additive_fit(N: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Kleinste-Quadrate-Anpassung der Nutzen auf die Werte dieser Matrix (mit Absolutglied); keine Obergrenze für die Trefferquote einer gewichteten Summe."""
    X = np.hstack([N, np.ones((len(N), 1))])
    b = np.linalg.lstsq(X, target, rcond=None)[0]
    return N @ b[:-1]


INTERACTION_METHODS = ("summe", "choquet", "halb", "falsch", "anpassung")


def experiment_interaction(structure=C.EXP_INTERACTION_STRUCTURE, lams=C.EXP_LAMS, seeds=C.SWEEP_SEEDS, n=C.EXP_N) -> list[dict]:
    """Redundanz (Maximum) und Synergie (Minimum) innerhalb der Aspekte, Korrelation 0: Platz 1 getroffen und gleicher Platz 1 von Summe und Choquet, je Normierung."""
    sizes = sn.parse_structure(structure)
    acc: dict = {}
    same: dict = {}
    for s in seeds:
        w = _world_n(sizes, 0.0, s, n)
        norms = {"minmax": sc.minmax(w.S), "perzentil": sc.percentile_normalize(w.S)}
        for kind in ("max", "min"):
            other = "min" if kind == "max" else "max"
            for lam in lams:
                u = sn.aspect_utility(w.goodness, sizes, w.v, lam, kind)
                best = sc.top1(u)
                for norm, N in norms.items():
                    key = (kind, lam, norm)
                    cand = {
                        "summe": N @ wt.expand_weights(w.v, sizes, True),
                        "choquet": sn.aspect_utility(N, sizes, w.v, lam, kind),
                        "halb": sn.aspect_utility(N, sizes, w.v, lam / 2, kind),
                        "falsch": sn.aspect_utility(N, sizes, w.v, lam, other),
                    }
                    if norm == "perzentil":
                        cand["anpassung"] = _best_additive_fit(N, sn.aspect_utility(N, sizes, w.v, lam, kind))
                    for name, score in cand.items():
                        h, t = acc.setdefault((key, name), ([], []))
                        h.append(sc.top1(score) == best)
                        t.append(sc.kendall_tau_b(score, u))
                    same.setdefault(key, []).append(sc.top1(cand["summe"]) == sc.top1(cand["choquet"]))
    rows = []
    for ((kind, lam, norm), name), (h, t) in acc.items():
        p = float(np.mean(h))
        rows.append({"kind": kind, "lam": lam, "norm": norm, "method": name, "hit": p, "tau": float(np.mean(t)), "se": float(np.sqrt(p * (1 - p) / len(h))),
                     "same_as_summe": float(np.mean(same[(kind, lam, norm)])) if name == "choquet" else float("nan")})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: Korrelation und Wechselwirkung

def experiment_correlation(structure=C.EXP_INTERACTION_STRUCTURE, rhos=C.EXP_CORRELATION_RHOS, seeds=C.SWEEP_SEEDS, n=C.EXP_N) -> list[dict]:
    """Redundanz (lam = 1), Perzentil-Ränge: Platz 1 der Summe und des Choquet je Korrelation im Aspekt."""
    sizes = sn.parse_structure(structure)
    rows = []
    for rho in rhos:
        hs, hc = [], []
        for s in seeds:
            w = _world_n(sizes, rho, s, n)
            N = sc.percentile_normalize(w.S)
            u = sn.aspect_utility(w.goodness, sizes, w.v, 1.0, "max")
            best = sc.top1(u)
            hs.append(sc.top1(N @ wt.expand_weights(w.v, sizes, True)) == best)
            hc.append(sc.top1(sn.aspect_utility(N, sizes, w.v, 1.0, "max")) == best)
        rows.append({"rho": rho, "summe": float(np.mean(hs)), "choquet": float(np.mean(hc)), "gap": float(np.mean(hc) - np.mean(hs))})
    return rows


def parameter_table(ms=(3, 4, 5, 6, 8)) -> list[dict]:
    return [{"m": m, **ch.parameter_counts(m)} for m in ms]


# ----------------------------------------------------------------------------- Verdikte (live aus den Zeilen)

def _row(rows, **keys):
    return next(r for r in rows if all(r[k] == v for k, v in keys.items()))


def verdict_counting(counting: list[dict], loss: list[dict], infl: list[dict]) -> tuple[str, str]:
    eq = [r for r in counting if r["structure"] == "2-2-2"]
    same = all(abs(_row(eq, rho=rho, method="erhoben")["hit"] - _row(eq, rho=rho, method="richtig")["hit"]) < 1e-12 for rho in C.EXP_RHOS)
    big = [r for r in counting if r["structure"] == "4-1-1"]
    lo = min(_row(big, rho=rho, method="richtig")["hit"] - _row(big, rho=rho, method="erhoben")["hit"] for rho in C.EXP_RHOS)
    hi = max(_row(big, rho=rho, method="richtig")["hit"] - _row(big, rho=rho, method="erhoben")["hit"] for rho in C.EXP_RHOS)
    top = max(loss, key=lambda r: r["k"])
    g = _row(infl, structure="4-1-1", rho=0.0, method="gleich")["inflation"]
    return "warning", (f"Ein Aspekt zählt so oft, wie er Kriterien hat: Bei gleich großen Aspekten ist die Erhebung je Kriterium richtig ({'gleiche Trefferquote' if same else 'Abweichung'}), bei (4, 1, 1) "
                       f"fehlen {de(100 * lo, 1)} bis {de(100 * hi, 1)} Punkte Platz 1, und der große Aspekt ist {de(g, 2)}-fach gewichtet. Mit {top['k']} Kriterien im großen Aspekt "
                       f"(Korrelation 0,9) sind es {de(100 * top['loss'], 1)} Punkte. Die Aufblähung ist keine Eigenschaft der Korrelation, sondern der Zahl der Kriterien je Aspekt.")


def verdict_data_driven(infl: list[dict], data: list[dict], bundled: list[dict]) -> tuple[str, str]:
    c9 = _row(infl, structure="4-1-1", rho=0.9, method="critic")["inflation"]
    e9 = _row(infl, structure="4-1-1", rho=0.9, method="entropie")["inflation"]
    k9 = _row(infl, structure="4-1-1", rho=0.9, method="cluster")["inflation"]
    k0 = _row(infl, structure="4-1-1", rho=0.0, method="cluster")["inflation"]
    cor = max(abs(r["critic_corr"]) for r in data), max(abs(r["entropy_corr"]) for r in data)
    spread = max(r["entropy_rank_range"] for r in data)
    cost = {rho: _row(bundled, rho=rho, method="wahre")["hit"] - _row(bundled, rho=rho, method="wahre_critic")["hit"] for rho in C.EXP_CRITERION_RHOS}
    r0, r1, r2 = C.EXP_CRITERION_RHOS
    return "warning", (f"Bei Korrelation 0,9 holt CRITIC die Aufblähung des großen Aspekts nur von 2,00 auf {de(c9, 2)}, Entropie-Gewichte gar nicht ({de(e9, 2)}), die Gruppierung aus den Daten fast "
                       f"vollständig ({de(k9, 2)}), aber erst bei hoher Korrelation (bei 0: {de(k0, 2)}). Die Gewichte beider Verfahren hängen nicht an der Wichtigkeit (Korrelation höchstens "
                       f"{de(cor[0], 2)} bzw. {de(cor[1], 2)}; Entropie auf Rängen: Spannweite höchstens {de(spread, 4)}). Und Korrelation ist nicht Redundanz: Bewertet der Entscheider korrelierte Kriterien "
                       f"einzeln und hoch, kostet der CRITIC-Faktor bei Korrelation {de(r0, 1)} {de(100 * cost[r0], 1)} Punkte, bei {de(r1, 1)} und {de(r2, 1)} nur {de(100 * cost[r1], 1)} und "
                       f"{de(100 * cost[r2], 1)} Punkte (im Rauschen).")


def verdict_interaction(rows: list[dict]) -> tuple[str, str]:
    r1 = _row(rows, kind="max", lam=1.0, norm="perzentil", method="summe")["hit"]
    c1 = _row(rows, kind="max", lam=1.0, norm="perzentil", method="choquet")["hit"]
    a1 = _row(rows, kind="max", lam=1.0, norm="perzentil", method="anpassung")["hit"]
    f1 = _row(rows, kind="min", lam=1.0, norm="minmax", method="falsch")["hit"]
    s1 = _row(rows, kind="min", lam=1.0, norm="minmax", method="summe")["hit"]
    same = _row(rows, kind="max", lam=0.5, norm="perzentil", method="choquet")["same_as_summe"]
    return "warning", (f"Bei Redundanz (Maximum im Aspekt, Perzentil-Ränge) trifft die gewichtete Summe Platz 1 in {de(100 * r1, 1)} %, eine additive Kleinste-Quadrate-Anpassung in {de(100 * a1, 1)} %, das Choquet-Integral "
                       f"mit der richtigen Wechselwirkung in {de(100 * c1, 1)} %; bei halber Stärke stimmt Platz 1 der Summe nur in {de(100 * same, 0)} % mit dem Choquet überein. Mit falschem "
                       f"Vorzeichen (Synergie, Maximum statt Minimum) trifft das Choquet-Integral nur in {de(100 * f1, 1)} % gegen {de(100 * s1, 1)} % der Summe.")


def verdict_correlation(rows: list[dict], params: list[dict]) -> tuple[str, str]:
    lo = min(rows, key=lambda r: r["rho"])
    hi = max(rows, key=lambda r: r["rho"])
    p = _row(params, m=6)
    return "info", (f"Der Vorteil des Choquet-Integrals schrumpft mit der Korrelation im Aspekt: {de(100 * lo['gap'], 1)} Punkte Abstand bei {de(lo['rho'], 1)}, {de(100 * hi['gap'], 1)} bei "
                    f"{de(hi['rho'], 1)}. Korrelation ist eine Eigenschaft der Daten, Wechselwirkung eine der Präferenz. Der Preis: bei 6 Kriterien {p['zwei_additiv']} Parameter der 2-additiven Kapazität "
                    f"(volle Kapazität {p['voll']}) statt {p['summe']} Gewichten, dazu eine gemeinsame Skala.")
