"""OpenAI-kompatible Server, vor allem die lokale crit-Engine (`crit serve`, Standard http://127.0.0.1:8000/v1).

Hinweise zu crit aus den Tests im Projekt trade-hub:
- Die Modell-ID exakt angeben (z. B. `run-…/candidate-0017`).
  `/v1/models` listet oft ältere Kandidaten zuerst.
- Thinking über `chat_template_kwargs.enable_thinking` (Paket-Capability `chat-template-enable-thinking-v1`).
- JSON-Schema über `response_format` geht nur mit dem crit-Subset: geschlossene Objekte
  (`additionalProperties: false`), Grenzen nur bei Ganzzahlen.
"""

from __future__ import annotations

import time

import httpx

from harness_it_all.models.base import Completion, ModelError
from harness_it_all.prices import Prices
from harness_it_all.profiles import Profile


class OpenAICompatModel:
    backend = "openai"

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        prices: Prices | None = None,
        timeout: float = 600.0,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.prices = prices or Prices()
        self.client = client or httpx.Client(timeout=timeout)

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
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema, "strict": True},
            }
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
            energy_kwh=self.prices.energy_kwh(latency),
            extra={"crit_telemetry": data["crit_telemetry"]} if "crit_telemetry" in data else {},
        )
