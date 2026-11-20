# harness-it-all: fair comparisons of LLM agents

[![CI](https://github.com/Dualshok3/harness-it-all/actions/workflows/ci.yml/badge.svg)](https://github.com/Dualshok3/harness-it-all/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Release](https://img.shields.io/github/v/release/Dualshok3/harness-it-all)](https://github.com/Dualshok3/harness-it-all/releases)

> A harness for **fair comparisons of LLM agents**: a local model on your own GPU (through a
> [crit](https://github.com/crit-labs/crit) engine) against Claude Opus 5.5 and others, with fixed profiles,
> complete recording, replay and honest statistics.

```text
> harness-it-all profiles

minimal (01d00706a1af) · max_tokens 4000 · As little thinking as possible. Default for live decisions.
  openai: {'enable_thinking': False, 'temperature': 0.0, 'seed': 7}
  anthropic: {'effort': 'low'}
  claude_code: {'effort': 'low'}
  deepseek: {'temperature': 0.0}
  Difference: Opus 5.5 always thinks (adaptively); effort=low is the lowest level. Gemma does not think here.
  Difference: Opus 5.5 accepts no temperature; Gemma runs with a fixed temperature and seed.
  …
```

*Shortened and translated: the profile texts in `configs/profiles.toml` are German.*

The harness contains **no domain logic**. What counts as a good answer is decided by the project that uses it:

| Project | What is compared | Scoring |
|---|---|---|
| [trade-hub](https://github.com/Dualshok3/trade-hub) | trading decisions on frozen market states | forward score: what the decision would have earned afterwards |
| [localRAG](https://github.com/Dualshok3/localRAG) | answers with sources over your own files | hit rate, citation faithfulness |

## Why a harness of its own?

A "local vs. API" comparison easily measures the settings instead of the models. Three typical traps and what the
harness does about them:

| Trap | Solution |
|---|---|
| **Different settings.** Gemma can turn thinking off, Claude Opus 5.5 cannot (only `effort` steers it). Gemma accepts temperature and seed, Opus 5.5 rejects both. | Fixed, versioned **profiles** (`configs/profiles.toml`). Every call stores its profile hash. Unavoidable differences are listed as `asymmetries` in the profile and appear in every report automatically. |
| **Not repeatable.** A prompt fix cannot be tested on yesterday's cases. | The **recorder** stores every call with its input as JSONL. **Replay** runs the same inputs again with another model or profile. |
| **Confidence intervals too narrow.** Decisions from the same week are correlated. | **Block bootstrap**, paired as well: whole blocks (e.g. trading weeks) are resampled, not single cases. |

One more detail that is easy to miss with Claude: server-side **fallbacks** on a refusal mean that a different
model answers. The harness records `model_served` and `fallback_used` and leaves such answers out of scores and
comparisons.

## Backends

| Backend | Adapter | Access | Structured output |
|---|---|---|---|
| `openai` | `OpenAICompatModel` | local crit engine, no key | JSON schema enforced (crit subset) |
| `anthropic` | `AnthropicModel` | `ANTHROPIC_API_KEY` | JSON schema enforced |
| `claude_code` | `ClaudeCodeModel` | logged-in Claude Code CLI (subscription) | `--json-schema` |
| `deepseek` | `OpenAICompatModel(profile_key="deepseek", schema_mode="json_object")` | `DEEPSEEK_API_KEY` | valid JSON only, schema in the system prompt |

Each backend has its own section in a profile. Whatever cannot be aligned is listed there under `asymmetries`.

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Dualshok3/harness-it-all.git && cd harness-it-all
uv sync
uv run pytest -q                                      # offline, with a fake model and mock transport
uv run harness-it-all profiles                        # profiles, hashes and known differences
uv run harness-it-all ping openai --model <model-id>  # local crit engine
uv run harness-it-all ping anthropic                  # one real API call to Claude Opus 5.5 (costs money)
```

## Using it from a project

```python
from harness_it_all.models.anthropic_model import AnthropicModel
from harness_it_all.models.openai_compat import OpenAICompatModel
from harness_it_all.profiles import load_profiles
from harness_it_all.recorder import Recorder
from harness_it_all.report import compare, markdown, summarize
from harness_it_all.scoring import score_records

profile = load_profiles()["minimal"]
gemma = OpenAICompatModel(GEMMA_ID)  # the ID under which `crit serve` offers the model
opus = AnthropicModel("claude-opus-5-5")
rec = Recorder("runs/decisions.jsonl")

for case in cases:  # e.g. one frozen market state per tick
    for model in (gemma, opus):
        rec.run(
            model,
            profile,
            case.id,
            case.messages,
            schema=DECISION_SCHEMA,
            prompt_version="decide-v3",
            tags={"snapshot": case.snapshot_hash, "week": case.week},
        )

scored = score_records(rec.read(), forward_score)  # the scorer comes from the project
summary = summarize(scored, block_of=lambda r: r.tags["week"])
print(markdown(summary))
```

## Configuration

| File | Contents |
|---|---|
| `configs/profiles.toml` | profiles per backend: thinking, effort, sampling, `max_tokens`, `asymmetries` |
| `configs/prices.toml` | API prices per model, GPU power draw and electricity price for local models |

## Architecture

```text
src/harness_it_all/
├── profiles.py            # fixed profiles per backend, hash, validation (e.g. no temperature for Opus 5.5)
├── models/
│   ├── openai_compat.py   # crit serve and other OpenAI-compatible servers, e.g. DeepSeek (httpx)
│   ├── anthropic_model.py # Claude through the official anthropic SDK: effort, JSON schema, fallbacks
│   ├── claude_code.py     # Claude through the Claude Code CLI (`claude -p`): subscription instead of API key
│   └── fake.py            # deterministic, for tests
├── recorder.py            # record, run, replay
├── scoring.py             # scorer interface and two example scorers
├── stats.py               # block bootstrap, paired
├── report.py              # Markdown report with confidence intervals, cost, electricity
└── prices.py              # API prices, electricity estimate for the local GPU
```

## Development

```bash
uv run pytest -q          # all tests run offline (fake model, mock transport)
uv run ruff check .
```

CI runs lint, tests and a secret scan (gitleaks) on every push. Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
