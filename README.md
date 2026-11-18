# harness-it-all

[![CI](https://github.com/Dualshok3/harness-it-all/actions/workflows/ci.yml/badge.svg)](https://github.com/Dualshok3/harness-it-all/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![Lizenz: MIT](https://img.shields.io/badge/lizenz-MIT-green)](LICENSE)

> Eine Harness für **faire Vergleiche von LLM-Agenten**: ein lokales Modell auf der eigenen GPU (über eine
> [crit](https://github.com/crit-labs/crit)-Engine) gegen Claude Opus 5.5, mit festen Profilen, lückenloser
> Aufzeichnung, Replay und ehrlicher Statistik.

Die Harness enthält **keine Fachlogik**. Was eine gute Antwort ist, entscheidet das Projekt, das sie nutzt:

| Projekt | Was verglichen wird | Bewertung |
|---|---|---|
| [trade-hub](https://github.com/Dualshok3/trade-hub) | Handelsentscheidungen auf eingefrorenen Marktzuständen | Forward-Score: was die Entscheidung danach am Markt gebracht hätte |
| [localRAG](https://github.com/Dualshok3/localRAG) | Antworten mit Quellen über die eigenen Dateien | Trefferquote, Zitattreue |

## Warum eine eigene Harness?

Ein Vergleich „lokal gegen API“ misst schnell die Einstellungen statt der Modelle. Drei typische Fallen und was die
Harness dagegen tut:

| Falle | Lösung |
|---|---|
| **Unterschiedliche Einstellungen.** Gemma kann Thinking abschalten, Claude Opus 5.5 nicht (dort steuert nur `effort`). Gemma nimmt Temperatur und Seed, Opus 5.5 lehnt beides ab. | Feste, versionierte **Profile** (`configs/profiles.toml`). Jeder Aufruf speichert den Hash seines Profils. Unvermeidbare Unterschiede stehen als `asymmetries` im Profil und automatisch in jedem Report. |
| **Nicht wiederholbar.** Ein Prompt-Fix lässt sich nicht auf gestrigen Fällen testen. | Der **Recorder** speichert jeden Aufruf mit Eingabe als JSONL. Das **Replay** führt dieselben Eingaben mit einem anderen Modell oder Profil erneut aus. |
| **Zu enge Konfidenzintervalle.** Entscheidungen derselben Woche hängen zusammen. | **Block-Bootstrap**, auch gepaart: Gezogen werden ganze Blöcke (z. B. Handelswochen), nicht einzelne Fälle. |

Dazu kommt ein Detail, das bei Claude leicht untergeht: Serverseitige **Fallbacks** bei einer Ablehnung sind an,
dann antwortet aber ein anderes Modell. Die Harness zeichnet `model_served` und `fallback_used` auf und lässt
solche Antworten aus Score und Vergleich heraus.

## Aufbau

```text
src/harness_it_all/
├── profiles.py        # feste Profile je Backend, Hash, Prüfung (z. B. keine Temperatur für Opus 5.5)
├── models/
│   ├── openai_compat.py   # crit serve und andere OpenAI-kompatible Server (httpx)
│   ├── anthropic_model.py # Claude über das offizielle anthropic-SDK, effort + JSON-Schema + Fallbacks
│   └── fake.py            # deterministisch, für Tests
├── recorder.py        # aufzeichnen, ausführen, Replay
├── scoring.py         # Scorer-Schnittstelle + zwei Beispiel-Scorer
├── stats.py           # Block-Bootstrap, gepaart
├── report.py          # Markdown-Report mit Konfidenzintervallen, Kosten, Strom
└── prices.py          # API-Preise, Stromschätzung für die lokale GPU
```

## Quickstart

Voraussetzung: [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:Dualshok3/harness-it-all.git && cd harness-it-all
uv sync
uv run pytest -q                       # ohne Netz, mit Fake-Modell und Mock-Transport
uv run harness-it-all profiles         # Profile, Hashes und bekannte Unterschiede
uv run harness-it-all ping openai --model run-…/candidate-0017   # lokale crit-Engine
uv run harness-it-all ping anthropic   # ein echter API-Aufruf an Claude Opus 5.5 (kostet Geld)
```

## So nutzt ein Projekt die Harness

```python
from harness_it_all.models.anthropic_model import AnthropicModel
from harness_it_all.models.openai_compat import OpenAICompatModel
from harness_it_all.profiles import load_profiles
from harness_it_all.recorder import Recorder
from harness_it_all.report import compare, markdown, summarize
from harness_it_all.scoring import score_records

profile = load_profiles()["minimal"]
gemma = OpenAICompatModel("run-20261109T095346Z-1b210ccf/candidate-0017")
opus = AnthropicModel("claude-opus-5-5")
rec = Recorder("runs/decisions.jsonl")

for case in cases:  # z. B. ein eingefrorener Marktzustand je Takt
    for model in (gemma, opus):
        rec.run(model, profile, case.id, case.messages, schema=DECISION_SCHEMA,
                prompt_version="decide-v3", tags={"snapshot": case.snapshot_hash, "week": case.week})

scored = score_records(rec.read(), forward_score)        # Scorer kommt aus dem Projekt
summary = summarize(scored, block_of=lambda r: r.tags["week"])
print(markdown(summary))
```

## Stand

Erste Version: Profile, beide Adapter, Recorder mit Replay, Scorer-Schnittstelle, gepaarter Block-Bootstrap und
Markdown-Report, 18 Tests ohne Netz. Gegen eine laufende Engine bzw. die echte API ist sie noch nicht geprüft.

Geplant: Self-Consistency (N Antworten parallel, Abstimmung) als eigener Baustein, HTML-Report, Anbindung an
trade-hub (Meilenstein 3/4) und an die Evals von localRAG.
