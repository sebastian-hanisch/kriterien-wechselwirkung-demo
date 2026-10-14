"""PDF-Export des aktuellen Falls (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import kwi_constants as C
import kwi_weights as wt
from kwi_evaluation import METHOD_LABELS, Analysis, de

_REPLACE = {"↑": "+", "↓": "-", "→": "->", "↔": "<->", "–": "-", "—": "-", "τ": "tau", "σ": "sigma", "λ": "lambda", "ρ": "rho", "≥": ">=", "≤": "<=", "−": "-", "€": "EUR", "·": "*"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    names = [f"A{i + 1}" for i in range(s.n)]
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Kriterien sind nicht unabhängig: Doppelzählung und Choquet"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    lam = f", Stärke {de(s.lam, 2)}" if s.decider in ("redundanz", "synergie") else ""
    pdf.multi_cell(0, 5, _s(f"Alternativen n = {s.n}, Aspekte ({', '.join(str(x) for x in a.sizes)}) mit {a.world.m} Kriterien, Korrelation im Aspekt {de(s.rho, 1)}, "
                            f"Entscheider: {C.DECIDER_LABELS[s.decider]}{lam}, Erhebung: {C.SURVEY_LABELS[s.survey]}, Normierung: {C.NORM_LABELS[s.norm]}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for k in ("erhoben", "richtig", "choquet"):
        r = a.results[k]
        pdf.cell(0, 5, _s(f"{METHOD_LABELS[k]}: Platz 1 {names[r.top1]} ({'getroffen' if r.hit else 'nicht getroffen'}), tau {de(r.tau, 2)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, _s(f"Bester nach Nutzen: {names[a.best]}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, _s("Aufblähung je Aspekt (Anteil im Verfahren / richtiger Anteil)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    share_right = wt.aspect_share(a.right, a.sizes)
    for k in ("erhoben", "critic", "entropie", "cluster"):
        infl = wt.aspect_share(a.weights[k], a.sizes) / share_right
        pdf.cell(0, 5, _s(f"{METHOD_LABELS[k]}: " + ", ".join(f"Aspekt {chr(65 + i)} {de(float(x), 2)}" for i, x in enumerate(infl))), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Der Entscheider ist ein simuliertes Orakel, kein Mensch. Quelle: kriterien-wechselwirkung-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
