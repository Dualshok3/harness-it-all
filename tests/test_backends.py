"""Claude-Code-CLI und DeepSeek: ohne Netz und ohne echte CLI (Runner bzw. HTTP-Transport gemockt)."""

import json
import subprocess

import httpx
import pytest

from harness_it_all.models.base import ModelError
from harness_it_all.models.claude_code import ClaudeCodeModel, flatten
from harness_it_all.models.openai_compat import OpenAICompatModel
from harness_it_all.profiles import ProfileError, load_profiles

PROFILE = load_profiles()["minimal"]
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["a"],
    "properties": {"a": {"type": "string"}},
}
MESSAGES = [{"role": "system", "content": "Sei knapp."}, {"role": "user", "content": "Hallo"}]


def cli_result(**over):
    out = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "",
        "structured_output": {"a": "x"},
        "stop_reason": "end_turn",
        "total_cost_usd": 0.012,
        "duration_api_ms": 900,
        "usage": {
            "input_tokens": 10,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 5,
            "output_tokens": 7,
        },
        "modelUsage": {"claude-opus-5-5": {}},
    }
    return out | over


class FakeRunner:
    def __init__(self, stdout, returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


def test_claude_code_builds_a_tool_free_call(tmp_path):
    runner = FakeRunner(json.dumps(cli_result()))
    m = ClaudeCodeModel(runner=runner, workdir=tmp_path / "leer")
    c = m.complete(MESSAGES, PROFILE, SCHEMA)
    cmd, kw = runner.calls[0]
    assert cmd[1] == "-p" and cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--effort") + 1] == "low"
    assert cmd[cmd.index("--system-prompt") + 1] == "Sei knapp."
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == SCHEMA
    assert "--no-session-persistence" in cmd
    assert kw["input"] == "Hallo" and kw["cwd"] == tmp_path / "leer"
    assert json.loads(c.text) == {"a": "x"}
    assert c.input_tokens == 115 and c.output_tokens == 7 and c.cost_usd == 0.012
    assert c.model_served == "claude-opus-5-5" and not c.fallback_used


def test_claude_code_errors_and_fallback(tmp_path):
    login = cli_result(is_error=True, result="Not logged in · Please run /login")
    with pytest.raises(ModelError, match="Not logged in"):
        ClaudeCodeModel(runner=FakeRunner(json.dumps(login)), workdir=tmp_path).complete(MESSAGES, PROFILE)
    with pytest.raises(ModelError, match="keine JSON-Ausgabe"):
        ClaudeCodeModel(runner=FakeRunner("kaputt", 1, "fehler"), workdir=tmp_path).complete(
            MESSAGES, PROFILE
        )
    other = cli_result(modelUsage={"claude-sonnet-5-5": {}})
    c = ClaudeCodeModel(runner=FakeRunner(json.dumps(other)), workdir=tmp_path).complete(MESSAGES, PROFILE)
    assert c.fallback_used
    side = cli_result(modelUsage={"claude-haiku-5-5": {}, "claude-opus-5-5": {}})
    c = ClaudeCodeModel(runner=FakeRunner(json.dumps(side)), workdir=tmp_path).complete(MESSAGES, PROFILE)
    assert not c.fallback_used and c.model_served == "claude-opus-5-5"


def test_flatten_keeps_single_message_and_labels_rounds():
    assert flatten([{"role": "user", "content": "x"}]) == "x"
    text = flatten(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    )
    assert text.index("a") < text.index("b") < text.index("c") and "frühere Antwort" in text


def deepseek(handler, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    m = OpenAICompatModel(
        "deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        profile_key="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        schema_mode="json_object",
    )
    m.client = httpx.Client(transport=httpx.MockTransport(handler), headers=m.client.headers)
    return m


def test_deepseek_uses_json_object_and_own_profile(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "model": "deepseek-chat",
                "choices": [{"message": {"content": '{"a": "x"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            },
        )

    c = deepseek(handler, monkeypatch).complete(MESSAGES, PROFILE, SCHEMA)
    body = seen["body"]
    assert seen["auth"] == "Bearer test-key"
    assert body["response_format"] == {"type": "json_object"}
    assert "JSON-Schema" in body["messages"][0]["content"] and "Sei knapp." in body["messages"][0]["content"]
    assert body["temperature"] == 0.0 and "seed" not in body and "chat_template_kwargs" not in body
    assert c.text == '{"a": "x"}' and c.energy_kwh is None


def test_deepseek_needs_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ModelError, match="DEEPSEEK_API_KEY"):
        OpenAICompatModel("deepseek-chat", api_key_env="DEEPSEEK_API_KEY")


def test_profile_validation_for_new_backends(tmp_path):
    p = tmp_path / "p.toml"
    p.write_text(
        '[profile.x]\nmax_tokens = 10\n[profile.x.claude_code]\neffort = "xhigh"\n', encoding="utf-8"
    )
    with pytest.raises(ProfileError, match=r"claude_code\.effort"):
        load_profiles(p)
    p.write_text("[profile.x]\nmax_tokens = 10\n[profile.x.deepseek]\nseed = 1\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="DeepSeek"):
        load_profiles(p)
