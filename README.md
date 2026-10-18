# Kriterien sind nicht unabhängig: Doppelzählung und Choquet – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-kriterien-wechselwirkung-demo.streamlit.app/)**

Achtes Stück der Linie **Mehrkriterien-Ranking** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Kind der
[gewichteten Summe](https://github.com/sebastian-hanisch/gewichtete-summe-demo). Messen mehrere Kriterien denselben **Aspekt**, zählt die Summe ihn so oft, wie er Kriterien hat, wenn
jedes Kriterium sein volles Gewicht bekommt (**Doppelzählung**). **CRITIC**, **Entropie-Gewichte** und eine **Gruppierung aus den Daten** versprechen Abhilfe; ist die Präferenz selbst nicht
additiv (**Redundanz**, **Synergie**), ersetzt das nur das **Choquet-Integral** die Summe.

Vehikel ist dieselbe **neutrale, simulierte Entscheidungsmatrix** wie in den Vorgängern (Generator `entscheidungsmatrix_generator.py`, wortgleich), hier mit **Blockkorrelation**: Kriterien
desselben Aspekts hängen zusammen, verschiedene Aspekte nicht. Der **Entscheider** kennt seine Aspektgewichte und seinen Nutzen (additiv im Mittel je Aspekt, mit Redundanz = Maximum, mit
Synergie = Minimum, oder jedes Kriterium für sich); er ist ein Orakel, kein Mensch.

## Warum dieses Problem

**Korrelation ist eine Eigenschaft der Daten, Wechselwirkung eine der Präferenz, und sie brauchen verschiedene Mittel.** Die Kernfrage der Demo: Was kostet es, Kriterien einzeln zu zählen,
was korrigieren die datengetriebenen Gewichte davon, und wann reicht eine gewichtete Summe nicht?

## Befunde und Korrekturen gegenüber dem Plan

- **„Zwei korrelierte Kriterien zählen doppelt“ stimmt nur bei ungleich vielen Kriterien je Aspekt.** Bei gleich großen Aspekten ist die Erhebung je Kriterium richtig (Kontrolle: gleiche
  Gewichte). Bei gleichen Aspektgewichten ist die Aufblähung eines Aspekts seine Größe durch die mittlere Aspektgröße (1,50 bei (3, 2, 1), 2,00 bei (4, 1, 1)); sie hängt nicht an der Korrelation.
- **Das ist eine Annahme, keine Messung:** Dass der Entscheider in Aspekten denkt und die Erhebung je Kriterium die Aspekt-Wichtigkeit für jedes Kriterium vergibt, ist modelliert. Die Aussage
  ist „wenn, dann so viel“.
- **Entropie-Gewichte sind keine Gegenmaßnahme.** Sie messen die Streuung der Anteile einer Spalte, nicht die Korrelation (Aufblähung 1,95 bei Korrelation 0,9), und auf Perzentil-Rängen sind
  sie praktisch gleich.
- **CRITIC korrigiert nur teilweise** und mischt Streuung (abhängig von der Normierung) mit Korrelation: bei Korrelation 0,9 von 2,00 auf 1,49. Die Gruppierung aus den Daten korrigiert fast
  vollständig (1,03), aber erst bei hoher Korrelation; bei Korrelation 0 bis 0,3 (Schwelle 0,5) bleibt die Doppelzählung in den Daten unsichtbar (2,00 / 1,84), bei 0,6 ist sie teilweise sichtbar (1,24).
- **Korrelation ist nicht Redundanz.** Bewertet der Entscheider die Kriterien einzeln, kostet der CRITIC-Faktor bei Korrelation 0 sieben bis acht Punkte (7,5), bei Korrelation 0,6 und 0,9 nur 1,5 und 3,5 Punkte, im Rauschen.
- **Choquet ist kein Gewinn an sich.** Bei reinem Maximum im Aspekt und unabhängigen Kriterien trifft das Choquet-Integral mit der richtigen Kapazität Platz 1 in 68,5 %, die Summe mit richtigen
  Gewichten in 36,0 % (eine Kleinste-Quadrate-Anpassung in 39,5 %); mit falschem Vorzeichen ist es bei Synergie 40 bis 46 Punkte schlechter als die Summe (Stärke 1). Bei Synergie zeigt sich ein
  Vorteil (7 Punkte bei reinem Minimum) nur auf einer gemeinsamen Skala (Perzentil-Ränge); bei geringerer Stärke ist er nicht belastbar.
- **Der Preis des Choquet:** 21 Parameter der 2-additiven Kapazität bei 6 Kriterien (volle Kapazität 62) statt 6 Gewichten, dazu die gemeinsame Skala.
- **Die Experimente starten auf Klick** (rund 10 Sekunden), nicht beim Laden.

## Modell

- **Matrix:** n Alternativen (5 bis 40, Standard 12), 4 bis 8 Kriterien in 2 bis 4 Aspekten; Blockkorrelation (Gauß-Copula) innerhalb des Aspekts, 0 zwischen den Aspekten.
- **Entscheider:** Aspektgewichte v_a (Summe 1), Güte als Quantil. Nutzen: Σ_a v_a · ((1 − λ) · Mittel + λ · Maximum | Minimum) der Güten im Aspekt; oder Σ_j w_j g_j.
- **Verfahren:** gewichtete Summe auf Min-Max oder Perzentil-Rängen mit den Gewichten *erhoben je Kriterium* (v_a für jedes Kriterium, normiert), *richtig geteilt* (v_a / Zahl der
  Kriterien), *CRITIC* (w_j ∝ s_j · Σ_k (1 − r_jk)), *Entropie* (w_j ∝ 1 − e_j), *Gruppen aus den Daten* (|r| ≥ Schwelle); dazu das **Choquet-Integral** der richtigen Kapazität.
- **Kapazität:** μ(S) = Σ_a v_a · μ_a(|S ∩ A_a|); Shapley-Wert (Wichtigkeit) und Interaktionsindex je Paar (positiv Synergie, negativ Redundanz).
- **Fester Rahmen der Experimente:** 200 feste Matrizen (`SWEEP_SEEDS`), n = 20, unabhängig vom Regler n.

Formeln im 📐-Expander der App.

## Methodik

1. **Dieselbe Matrix, derselbe Entscheider, alle Verfahren:** Vergleiche zwischen Verfahren sind gepaart.
2. **Messgröße ist die Entscheidung** (Platz 1, Kendalls τ-b, Nutzenverlust).
3. **Kontrolle:** Bei gleich großen Aspekten müssen erhoben und richtig übereinstimmen; ohne Wechselwirkung ist das Choquet-Integral die Summe mit richtigen Gewichten.
4. **Gemeinsame Skala:** Die Wechselwirkung wird auf Perzentil-Rängen und auf Min-Max gemessen, damit der Einfluss der Skala sichtbar bleibt.

## Befunde (gemessen, keine Behauptungen)

Fast alle Zahlen: `tests/test_claims.py` (die Zeile zu den datengetriebenen Verfahren ist nur gegen eine Schranke geprüft). 200 feste Matrizen, n = 20, Entscheider denkt in Aspekten. **Der Standardfehler einer Trefferquote ist 2 bis 3,5 Punkte; Unterschiede unter etwa
6 Punkten sind nicht belastbar und stehen nicht als Aussage.**

| Frage | Befund | Test |
|---|---|---|
| Doppelzählung: Platz 1, erhoben je Kriterium / richtig geteilt (Min-Max; Korrelation im Aspekt 0 / 0,3 / 0,6 / 0,9) | **(2, 2, 2): erhoben = richtig** 79,5 / 80,0 / 81,5 / 88,0 %. **(3, 2, 1): 58,5 / 58,5 / 65,5 / 67,5 % gegen 72,0 / 70,5 / 75,0 / 84,0 %.** **(4, 1, 1): 53,5 / 59,5 / 65,5 / 67,5 % gegen 74,5 / 80,5 / 85,0 / 84,0 %.** | `test_equal_aspect_sizes_make_the_ask_per_criterion_correct`, `test_asking_per_criterion_loses_platz_1_when_aspects_differ_in_size` |
| Verlust je Zahl der Kriterien im großen Aspekt (Aspekte (k, 1, 1), Korrelation 0,9) | k = 1 / 2 / 3 / 4 / 5: erhoben 85,0 / 67,5 / 60,0 / 67,5 / 57,5 %, richtig 85,0 / 71,0 / 70,5 / 84,0 / 87,0 %, **Verlust 0 / 3,5 / 10,5 / 16,5 / 29,5 Punkte**. | `test_loss_grows_with_the_number_of_criteria_in_the_large_aspect` |
| Aufblähung des großen Aspekts (Anteil im Verfahren / richtiger Anteil) | gleich je Kriterium **1,50 bei (3, 2, 1), 2,00 bei (4, 1, 1)**. Bei (4, 1, 1) und Korrelation 0 / 0,3 / 0,6 / 0,9: CRITIC 1,99 / 1,87 / 1,71 / **1,49**, Entropie 1,97 / 1,96 / 1,96 / **1,95**, Gruppen 2,00 / 1,84 / 1,24 / **1,03**; bei (3, 2, 1): CRITIC 1,51 / 1,45 / 1,37 / 1,27, Gruppen 1,49 / 1,42 / 1,08 / 1,01. | `test_inflation_of_equal_weights_is_the_size_over_the_mean_size`, `test_critic_corrects_only_partly_entropy_not_at_all` |
| Erhoben mal CRITIC-Faktor ((4, 1, 1); Korrelation 0 / 0,3 / 0,6 / 0,9) | 52,5 / 63,0 / 69,0 / **80,5 %** gegen erhoben 53,5 / 59,5 / 65,5 / 67,5 %: hilft erst bei hoher Korrelation. | `test_critic_factor_helps_only_at_high_correlation` |
| Datengetriebene Verfahren ohne Wichtigkeitsangaben | CRITIC, Entropie und Gruppen sind nicht besser als gleiche Gewichte je Kriterium (z. B. (3, 2, 1), Korrelation 0,6: 41,5 / 38,0 / 42,0 % gegen 42,5 %). | `test_data_driven_methods_are_not_better_than_equal_weights` |
| Sagen die Gewichte etwas über die Wichtigkeit? | In der Simulation sind die wahren Gewichte von den Daten unabhängig; entsprechend korrelieren CRITIC- und Entropie-Gewichte nicht mit ihnen (Mittel höchstens 0,05 bzw. 0,03; Betrag). Über reale Daten sagt das nichts. Entropie-Gewichte auf Perzentil-Rängen: Spannweite im Mittel unter 0,001. | `test_data_driven_weights_say_nothing_about_importance` |
| Korrelation ohne Redundanz (jedes Kriterium für sich; 60 % der Gewichte auf den drei korrelierten; Korrelation 0 / 0,6 / 0,9) | wahre Gewichte 81,5 / 82,5 / 85,5 % gegen wahre mal CRITIC 74,0 / 81,0 / 82,0 %: **bis 7,5 Punkte Verlust**. Zufällige Gewichte: wahre 76,0 / 75,0 / 77,0 / 83,5 % (Korrelation 0 / 0,3 / 0,6 / 0,9). | `test_correlation_is_not_redundancy` |
| Redundanz (Aspekte (2, 2, 2), Korrelation 0, Min-Max; Stärke 0 / 0,25 / 0,5 / 0,75 / 1) | Summe 79,5 / 77,5 / 66,0 / 54,0 / 36,5 %; **Choquet 79,5 / 78,5 / 75,0 / 68,0 / 63,5 %**; halbe Stärke 79,5 / 78,0 / 70,5 / 60,0 / 46,0 %; **falsches Vorzeichen 79,5 / 71,5 / 56,5 / 42,5 / 26,5 %**. | `test_redundancy_on_minmax` |
| Redundanz auf Perzentil-Rängen; Kleinste-Quadrate-Anpassung | Summe 77,0 / 75,5 / 67,0 / 52,0 / 36,0 %; Choquet 77,0 / 76,5 / 75,0 / 72,0 / **68,5 %**; **Kleinste-Quadrate-Anpassung 77,0 / 72,0 / 65,0 / 51,5 / 39,5 %**. | `test_redundancy_on_percentile_ranks_and_the_best_additive_fit` |
| Synergie: Min-Max gegen Perzentil-Ränge | Min-Max: Summe 79,5 / 76,5 / 74,0 / 71,0 / 67,5 %, Choquet 79,5 / 73,0 / 70,5 / 68,5 / 65,0 % (kein Vorteil), falsches Vorzeichen 79,5 / 74,5 / 62,5 / 47,0 / **27,0 %**. Perzentil: Summe 77,0 / 77,5 / 73,5 / 71,5 / 69,0 %, **Choquet 77,0 / 77,5 / 76,0 / 76,5 / 76,0 %**. | `test_synergy_has_no_advantage_on_minmax_but_on_a_common_scale` |
| Wie oft reicht die Summe? Gleicher Platz 1 von Summe und Choquet (Perzentil; Stärke 0 / 0,25 / 0,5 / 0,75 / 1) | Redundanz 100 / 88,0 / **73,5** / 53,0 / 33,5 %; Synergie 100 / 89,5 / 84,5 / 81,5 / 76,0 %. | `test_how_often_the_sum_is_enough` |
| Korrelation und Wechselwirkung (Redundanz, Stärke 1, Perzentil; Korrelation −0,4 / 0 / 0,3 / 0,6 / 0,9) | Summe 17,0 / 36,0 / 46,0 / 58,0 / 72,0 %, Choquet 65,5 / 68,5 / 72,0 / 72,0 / 75,0 %, **Abstand 48,5 / 32,5 / 26,0 / 14,0 / 3,0 Punkte**. | `test_the_advantage_of_choquet_shrinks_with_correlation` |
| Preis in Parametern (m = 3 / 4 / 5 / 6 / 8) | Summe 3 / 4 / 5 / 6 / 8, 2-additiv 6 / 10 / 15 / 21 / 36, volle Kapazität 6 / 14 / 30 / 62 / 254. | `test_parameter_counts_table` |

## Ehrliche Grenzen

- **Die Doppelzählung ist eine Folge der Konstruktion.** Der Entscheider denkt in Aspekten und die Erhebung vergibt die Aspekt-Wichtigkeit je Kriterium: beides ist angenommen, nicht gemessen.
- **Die Aspekte sind bekannt.** In der Praxis ist die Zuordnung der Kriterien zu Aspekten selbst eine Modellierung; die Gruppierung aus den Daten findet sie erst bei hoher Korrelation.
- **Das Choquet-Integral wird mit der richtigen Kapazität gerechnet.** Sie zu erheben (Fragen, Beispiele) ist der eigentliche Preis und hier nicht gemessen; es steht nur die Parameterzahl.
- **Eine gemeinsame Skala ist in der Simulation vorhanden.** In der Praxis braucht Maximum und Minimum eine Skala, auf der „gleich gut“ je Kriterium dasselbe heißt; Perzentil-Ränge nähern sie an
  und hängen an der Stichprobe.
- **Wechselwirkung nur innerhalb der Aspekte,** als Mischung aus Mittel, Maximum und Minimum; keine Wechselwirkung zwischen Aspekten, keine gemischten Vorzeichen.
- **Kleine Unterschiede sind Rauschen.** Synergie bei kleiner Stärke und die Zahlen bei zufälligen Gewichten liegen innerhalb von zwei Standardfehlern. Quellenangaben nur aus Hintergrundwissen:
  CRITIC (Diakoulaki, Mavrotas, Papayannakis, 1995) und das Choquet-Integral (Grabisch und Mitarbeiter) sind hier nicht im Original geprüft.

## Verifikation

- **Kern:** Erhebung, CRITIC, Entropie, Gruppierung, Aufblähung, Choquet-Integral, Möbius-Form, Shapley-Wert und Interaktionsindex gegen Handrechnung auf den Mini-Instanzen
  (`tests/conftest.py`: zwei identische Kriterien und ein unabhängiges mit CRITIC = 0,25 / 0,25 / 0,50 und Entropie = 1/3; Aspekte (2, 1) mit erhoben (0,375; 0,375; 0,25) gegen richtig
  (0,3; 0,3; 0,4); eine 2-additive Kapazität mit C = 0,56 und Shapley 0,40 / 0,35 / 0,25; zwei Alternativen mit gleichem Mittel und verschiedenem Maximum und Minimum).
- **Orakel** (`tests/test_oracle_kriterien_wechselwirkung.py`): Choquet-Integral gegen die Möbius-Darstellung für zufällige Belief-Kapazitäten, Monotonie; Shapley gegen den Mittelwert der
  Grenzbeiträge über alle Reihenfolgen; Interaktionsindex gegen die Möbius-Formel; CRITIC und Entropie gegen Definitionsschleifen; Aufblähung gegen die geschlossene Formel; der aspektweise
  Nutzen ist das Choquet-Integral der Aspekt-Kapazität.
- **Randfälle** (`tests/test_edge_cases.py`): n = 1 und n = 2, konstante Spalten, identische Alternativen, ein Kriterium je Aspekt, inklusive Gruppierungsschwelle, Perzentil bei n = 2.
- **Generator** (`tests/test_generator.py`): SplitMix64 gegen unabhängige Python-Fassung, Randverteilung, Spearman-Korrelation der Copula, Richtungen und Bereiche.
- **Aussagen:** fast jede Zahl dieser README in `tests/test_claims.py` (Ausnahmen siehe Befunde); Presets in `test_presets.py` (jedes zeigt seinen Effekt).
- **Mutationstest** vor dem ersten Commit (`tools/mutate_demo.py` der Website): 95 Stellen im Kern, erster Lauf 79 von 93 Mutanten von der ganzen Suite gefangen; die Überlebenden waren fehlende Randfälle (Monotonieprüfung am letzten Kriterium, n = 1 ohne Warnungen) und ein Fehler bei genau einem Kriterium (die 1 x 1-Korrelationsmatrix, behoben). Danach ist jede Stelle gefangen oder als äquivalent festgehalten (3: Schleifengrenzen der Gruppierung); Stand in `tests/mutation_state.json`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), alle Presets, jede Struktur, jeder Entscheider und jede Normierung, deaktivierte Regler, Permalink,
  PDF und Experimente erst nach Klick, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `entscheidungsmatrix_generator.py` | Generator der Linie (wortgleich in allen Repos der Linie, kein Präfix): SplitMix64, Copula, Rohwerte, Entscheider |
| `kwi_constants.py` | Regler-Grenzen, feste Annahmen, Konfigurationen der Experimente, Farben, Presets |
| `kwi_scenario.py` | Aspekte, Blockkorrelation, Welt, Nutzen des Entscheiders (additiv, Redundanz, Synergie, je Kriterium) |
| `kwi_weights.py` | Erhebung, CRITIC, Entropie, Gruppierung aus den Daten, Aspektanteile und Aufblähung |
| `kwi_choquet.py` | Kapazität, Choquet-Integral, Möbius-Form, Shapley-Wert, Interaktionsindex, Parameterzahlen |
| `kwi_score.py` | Min-Max, Perzentil-Ränge, Summe, Kendalls τ-b, Nutzenverlust |
| `kwi_evaluation.py` | Fall der App, vier Experimente, Verdikte |
| `kwi_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `kwi_visualization.py` | Korrelationskarte, Gewichtsbalken, Kapazität, Tabellen und Plotly-Abbildungen (Achsen gesperrt) |
| `kwi_pdf_export.py` | PDF-Export des aktuellen Falls (fpdf2, erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Mini-Instanzen, Gewichte, Choquet, Szenario, Score, Orakel, Randfälle, Generator, Auswertung, Aussagen, Presets, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Die Kapazität aus Beispielen lernen:** hier ist sie gegeben; das Lernen steht in einem eigenen Stück der Linie.
- **Allgemeine Kapazitäten mit gemischten Vorzeichen und Wechselwirkung zwischen Aspekten:** 2^m − 2 Regler sind unübersichtlich und bringen keinen neuen Befund.
- **Ein Vergleich über viele Verfahren:** die Aufblähung gilt für jedes additive Verfahren; hier nur an der Summe gemessen.
- **Hierarchien (AHP) als Antwort auf Doppelzählung:** aspektweise Gewichte sind der Befund, die Hierarchie steht im AHP-Stück.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html).
