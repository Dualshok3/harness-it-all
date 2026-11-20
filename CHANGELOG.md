# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-20

### Added
- `claude_code` backend: Claude through the Claude Code CLI (`claude -p`) with a subscription instead of an API
  key, without tools, with its own system prompt and `--json-schema`.
- `deepseek` backend through `OpenAICompatModel` (`profile_key`, `api_key_env`, `schema_mode="json_object"`).
- Profile sections and validation for both new backends.

### Changed
- `OpenAICompatModel` reports API cost from `configs/prices.toml` and estimates electricity only for local servers.
- README in English.

## [0.1.0] - 2026-11-19

### Added
- Fixed, hashed profiles per backend with documented asymmetries.
- Adapters for OpenAI-compatible servers (crit) and Claude through the anthropic SDK, including server-side
  fallback detection.
- Recorder with replay, scorer interface, paired block bootstrap and Markdown report.

[0.2.0]: https://github.com/Dualshok3/harness-it-all/releases/tag/v0.2.0
[0.1.0]: https://github.com/Dualshok3/harness-it-all/releases/tag/v0.1.0
