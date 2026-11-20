"""Claude über die Claude-Code-CLI im Skript-Modus (`claude -p`) statt über die API.

Damit läuft der Vergleich über ein Claude-Abo statt über einen API-Key. Die CLI wird als reines Antwortmodell
benutzt: ohne Werkzeuge (`--tools ""`), mit eigenem System-Prompt (ersetzt den von Claude Code), mit festem
JSON-Schema (`--json-schema`) und ohne gespeicherte Sitzung. Gestartet wird in einem leeren Ordner, damit
keine CLAUDE.md eines Projekts in den Kontext gerät.

Unterschiede zur API, die als `asymmetries` ins Profil gehören:
- Ein Gespräch über mehrere Runden (z. B. Ablehnung durch einen Risk-Check) wird als ein Text übergeben,
  nicht als echte Folge von Nachrichten.
- Die Latenz enthält den Start der CLI.
- `total_cost_usd` ist der API-Gegenwert; abgerechnet wird über das Abo.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from harness_it_all.models.base import Completion, ModelError, split_system
from harness_it_all.profiles import Profile

ROLE_LABEL = {"user": "Nutzer", "assistant": "Deine frühere Antwort"}


def flatten(messages: list[dict]) -> str:
    """Nachrichten nach dem System-Prompt zu einem Text machen. Eine einzelne Nutzernachricht bleibt, wie sie
    ist."""
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"]
    parts = [f"### {ROLE_LABEL.get(m['role'], m['role'])}\n{m['content']}" for m in messages]
    return "\n\n".join(parts)


class ClaudeCodeModel:
    backend = "claude_code"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        executable: str = "claude",
        timeout: float = 600.0,
        workdir: Path | str | None = None,
        runner=subprocess.run,
    ):
        self.model = model
        self.executable = executable
        self.timeout = timeout
        self.workdir = (
            Path(workdir) if workdir else Path(tempfile.gettempdir()) / "harness-it-all-claude-code"
        )
        self.runner = runner
        self._version: str | None = None

    def version(self) -> dict:
        info = {"backend": self.backend, "model": self.model}
        if self._version is None:
            try:
                r = self.runner(
                    [self._exe(), "--version"], capture_output=True, text=True, timeout=30, encoding="utf-8"
                )
                self._version = (r.stdout or "").strip()
            except (OSError, subprocess.SubprocessError):
                self._version = ""
        if self._version:
            info["cli"] = self._version
        return info

    def _exe(self) -> str:
        return shutil.which(self.executable) or self.executable

    def command(self, system: str | None, profile: Profile, schema: dict | None) -> list[str]:
        cfg = profile.for_backend(self.backend)
        cmd = [
            self._exe(),
            "-p",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--no-session-persistence",
        ]
        cmd += ["--tools", ""]
        if "effort" in cfg:
            cmd += ["--effort", cfg["effort"]]
        if system:
            cmd += ["--system-prompt", system]
        if schema is not None:
            cmd += ["--json-schema", json.dumps(schema, ensure_ascii=False)]
        return cmd

    def complete(self, messages: list[dict], profile: Profile, schema: dict | None = None) -> Completion:
        system, rest = split_system(messages)
        if not rest:
            raise ModelError("keine Nutzernachricht")
        self.workdir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            r = self.runner(
                self.command(system, profile, schema),
                input=flatten(rest),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
                cwd=self.workdir,
            )
        except FileNotFoundError as e:
            raise ModelError(f"Claude-Code-CLI nicht gefunden ({self.executable})") from e
        except subprocess.TimeoutExpired as e:
            raise ModelError(f"{self.model}: keine Antwort nach {self.timeout:.0f} s") from e
        latency = time.perf_counter() - t0
        try:
            data = json.loads(r.stdout)
        except json.JSONDecodeError:
            raise ModelError(
                f"{self.model}: keine JSON-Ausgabe (Exit {r.returncode}): {r.stderr[:300]}"
            ) from None
        if data.get("is_error") or data.get("subtype") != "success":
            raise ModelError(f"{self.model}: {str(data.get('result'))[:300]}")
        structured = data.get("structured_output")
        text = (
            json.dumps(structured, ensure_ascii=False) if structured is not None else data.get("result") or ""
        )
        usage = data.get("usage") or {}
        served = list((data.get("modelUsage") or {}).keys())
        n_in = sum(
            usage.get(k) or 0
            for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        return Completion(
            text=text,
            model_requested=self.model,
            model_served=next(
                (m for m in served if m.startswith(self.model)), ",".join(served) or self.model
            ),
            finish_reason=data.get("stop_reason") or "",
            input_tokens=n_in or None,
            output_tokens=usage.get("output_tokens"),
            latency_s=latency,
            cost_usd=data.get("total_cost_usd"),
            # Claude Code kann Nebenaufgaben an kleinere Modelle geben; Fallback heißt: das Modell fehlt ganz
            fallback_used=bool(served) and not any(m.startswith(self.model) for m in served),
            extra={"api_equivalent_cost": True, "duration_api_ms": data.get("duration_api_ms")},
        )
