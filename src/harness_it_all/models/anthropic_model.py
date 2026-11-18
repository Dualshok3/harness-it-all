"""Claude über das offizielle Anthropic-SDK. Standard ist Claude Opus 5.5 (`claude-opus-5-5`).

Besonderheiten von Opus 5.5, die die Profile berücksichtigen:
- Thinking ist immer an und lässt sich nicht abschalten. Gesteuert wird nur über `output_config.effort`
  (Standard `medium`, wird hier immer ausdrücklich gesetzt).
- `temperature`, `top_p` und `top_k` werden abgelehnt und deshalb nie gesendet.
- Strukturierte Ausgabe über `output_config.format` (JSON-Schema).
- Serverseitige Fallbacks bei einer Ablehnung (`refusal`) sind standardmäßig an. Für einen Benchmark ist das
  heikel, weil dann ein anderes Modell antwortet. Deshalb wird jede Antwort mit `model_served` und
  `fallback_used` aufgezeichnet, und der Report kann solche Antworten ausweisen oder herausfiltern.
"""

from __future__ import annotations

import time

from harness_it_all.models.base import Completion, ModelError, split_system
from harness_it_all.prices import Prices
from harness_it_all.profiles import Profile

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicModel:
    backend = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        prices: Prices | None = None,
        fallbacks: bool = True,
        client=None,
    ):
        self.model = model
        self.prices = prices or Prices()
        self.fallbacks = fallbacks
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def version(self) -> dict:
        try:
            import anthropic

            sdk = anthropic.__version__
        except ImportError:
            sdk = None
        return {"backend": self.backend, "model": self.model, "sdk": sdk, "fallbacks": self.fallbacks}

    def complete(self, messages: list[dict], profile: Profile, schema: dict | None = None) -> Completion:
        import anthropic

        cfg = profile.for_backend(self.backend)
        system, rest = split_system(messages)
        output_config: dict = {"effort": cfg["effort"]}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        params: dict = {
            "model": self.model,
            "max_tokens": profile.max_tokens,
            "messages": rest,
            "output_config": output_config,
        }
        if system:
            params["system"] = system
        t0 = time.perf_counter()
        try:
            if self.fallbacks:
                response = self.client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **params
                )
            else:
                response = self.client.messages.create(**params)
        except anthropic.RateLimitError as e:
            raise ModelError(f"{self.model}: Rate-Limit: {e}") from e
        except anthropic.APIStatusError as e:
            raise ModelError(f"{self.model}: HTTP {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise ModelError(f"{self.model}: Verbindung fehlgeschlagen: {e}") from e
        latency = time.perf_counter() - t0

        text = "".join(b.text for b in response.content if b.type == "text")
        reasoning = "".join(
            getattr(b, "thinking", "") or "" for b in response.content if b.type == "thinking"
        )
        usage = response.usage
        iterations = getattr(usage, "iterations", None) or []
        fallback_used = any(getattr(i, "type", "") == "fallback_message" for i in iterations)
        served = getattr(response, "model", self.model) or self.model
        extra = {}
        if response.stop_reason == "refusal" and getattr(response, "stop_details", None):
            extra["refusal_category"] = getattr(response.stop_details, "category", None)
        return Completion(
            text=text,
            reasoning=reasoning or None,
            model_requested=self.model,
            model_served=served,
            finish_reason=response.stop_reason or "",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_s=latency,
            cost_usd=self.prices.api_cost(served, usage.input_tokens, usage.output_tokens),
            fallback_used=fallback_used or served != self.model,
            extra=extra,
        )
