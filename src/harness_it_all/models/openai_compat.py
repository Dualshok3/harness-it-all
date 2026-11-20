"""OpenAI-kompatible Server, vor allem die lokale crit-Engine (`crit serve`, Standard http://127.0.0.1:8000/v1).

Hinweise zu crit aus den Tests im Projekt trade-hub:
- Die Modell-ID exakt angeben (z. B. `run-…/candidate-0017`).
  `/v1/models` listet oft ältere Kandidaten zuerst.
- Thinking über `chat_template_kwargs.enable_thinking` (Paket-Capability `chat-template-enable-thinking-v1`).
- JSON-Schema über `response_format` geht nur mit dem crit-Subset: geschlossene Objekte
  (`additionalProperties: false`), Grenzen nur bei Ganzzahlen.
"""

from __future__ import annotations

import json
import os
import time
from urllib.parse import urlparse

import httpx

from harness_it_all.models.base import Completion, ModelError
from harness_it_all.prices import Prices
from harness_it_all.profiles import Profile


def with_schema_hint(messages: list[dict], schema: dict) -> list[dict]:
    """Für json_object: Das Schema an den System-Prompt hängen, sonst kennt das Modell es nicht."""
    hint = "Antworte ausschließlich mit einem JSON-Objekt nach diesem JSON-Schema:\n"
    hint += json.dumps(schema, ensure_ascii=False)
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": messages[0]["content"] + "\n\n" + hint}, *messages[1:]]
    return [{"role": "system", "content": hint}, *messages]


class OpenAICompatModel:
    backend = "openai"

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        prices: Prices | None = None,
        timeout: float = 600.0,
        client: httpx.Client | None = None,
        profile_key: str = "openai",
        api_key_env: str | None = None,
        schema_mode: str = "json_schema",
    ):
        """`profile_key`: welcher Abschnitt des Profils gilt (z. B. "deepseek" für einen anderen Anbieter mit
        eigenen Einstellungen). `api_key_env`: Umgebungsvariable mit dem API-Key (lokal nicht nötig).
        `schema_mode`: "json_schema" (Schema wird erzwungen) oder "json_object" (nur gültiges JSON, das Schema
        steht dann im System-Prompt; für Anbieter ohne json_schema, z. B. DeepSeek)."""
        if schema_mode not in ("json_schema", "json_object"):
            raise ValueError(f"unbekannter schema_mode {schema_mode!r}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.prices = prices or Prices()
        self.backend = profile_key
        self.local = urlparse(self.base_url).hostname in ("127.0.0.1", "localhost", "::1")
        self.schema_mode = schema_mode
        headers = {}
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise ModelError(f"Umgebungsvariable {api_key_env} ist nicht gesetzt")
            headers["Authorization"] = f"Bearer {key}"
        self.client = client or httpx.Client(timeout=timeout, headers=headers)

    def version(self) -> dict:
        info = {"backend": self.backend, "model": self.model, "base_url": self.base_url}
        try:
            r = self.client.get(f"{self.base_url.removesuffix('/v1')}/health", timeout=5)
            engine = (r.json() or {}).get("engine") or {}
            info["engine"] = engine.get("engine") or engine
        except (httpx.HTTPError, ValueError):
            pass
        return info

    def complete(self, messages: list[dict], profile: Profile, schema: dict | None = None) -> Completion:
        cfg = profile.for_backend(self.backend)
        body: dict = {"model": self.model, "messages": messages, "max_tokens": profile.max_tokens}
        for key in ("temperature", "top_p", "top_k", "seed"):
            if key in cfg:
                body[key] = cfg[key]
        if "enable_thinking" in cfg:
            body["chat_template_kwargs"] = {"enable_thinking": cfg["enable_thinking"]}
        if schema is not None and self.schema_mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema, "strict": True},
            }
        elif schema is not None:
            body["response_format"] = {"type": "json_object"}
            body["messages"] = with_schema_hint(messages, schema)
        t0 = time.perf_counter()
        try:
            r = self.client.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.HTTPError as e:
            raise ModelError(f"{self.model}: Verbindung fehlgeschlagen: {e}") from e
        latency = time.perf_counter() - t0
        if r.status_code != 200:
            raise ModelError(f"{self.model}: HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        return Completion(
            text=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
            model_requested=self.model,
            model_served=data.get("model", self.model),
            finish_reason=choice.get("finish_reason") or "",
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_s=latency,
            cost_usd=self.prices.api_cost(
                self.model, usage.get("prompt_tokens"), usage.get("completion_tokens")
            ),
            energy_kwh=self.prices.energy_kwh(latency) if self.local else None,
            extra={"crit_telemetry": data["crit_telemetry"]} if "crit_telemetry" in data else {},
        )
