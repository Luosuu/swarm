"""Model clients.

`ChatEndpoint` talks to any OpenAI-compatible `/chat/completions` endpoint
(vLLM, SGLang, Ollama, LiteLLM, a hosted gateway...) using only the standard
library. `ScriptedModel` replays a Python policy, for tests and dry runs.
"""

from __future__ import annotations

import asyncio
import json
import random
import ssl
import urllib.error
import urllib.request
from typing import Any, Awaitable, Callable, Protocol

from .pricing import Usage


class Model(Protocol):
    async def complete(self, messages: list[dict], tools: list[dict] | None, *, agent: str) -> tuple[dict, Usage]:
        """Return (assistant message dict, token usage)."""
        ...


def estimate_tokens(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False, default=str)) // 4


class ChatEndpoint:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout_s: float = 300,
        retries: int = 4,
        ca_file: str | None = None,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.timeout_s = timeout_s
        self.retries = retries
        self.ssl = ssl.create_default_context(cafile=ca_file) if ca_file else None

    async def complete(self, messages: list[dict], tools: list[dict] | None, *, agent: str) -> tuple[dict, Usage]:
        body: dict[str, Any] = {"model": self.model, "messages": messages, **self.extra_body}
        if tools:
            body["tools"] = tools  # parallel_tool_calls is deliberately not sent; many servers reject it
        if self.temperature is not None:
            body["temperature"] = self.temperature
        data = await self._post_with_retries(body)
        choice = data["choices"][0]["message"]
        msg: dict[str, Any] = {"role": "assistant", "content": choice.get("content") or ""}
        calls = choice.get("tool_calls") or []
        if calls:
            msg["tool_calls"] = [
                {
                    "id": c.get("id") or f"call_{i}_{random.randrange(1 << 30)}",
                    "type": "function",
                    "function": {
                        "name": c["function"]["name"],
                        "arguments": c["function"].get("arguments") or "{}",
                    },
                }
                for i, c in enumerate(calls)
            ]
        return msg, parse_usage(data.get("usage"), messages, msg)

    async def _post_with_retries(self, body: dict) -> dict:
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                return await asyncio.to_thread(self._post, body)
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                if e.code < 500 and e.code != 429 or attempt == self.retries:
                    raise RuntimeError(f"endpoint returned HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.retries:
                    raise RuntimeError(f"endpoint unreachable: {e}") from None
            await asyncio.sleep(delay + random.random())
            delay *= 2
        raise AssertionError("unreachable")

    def _post(self, body: dict) -> dict:
        # Some gateways (Cloudflare) reject urllib's default User-Agent with 403 / error 1010.
        headers = {"Content-Type": "application/json", "User-Agent": "meshswarm/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s, context=self.ssl) as resp:
            return json.loads(resp.read())


def parse_usage(usage: dict | None, messages: list[dict], msg: dict) -> Usage:
    """OpenAI-style usage, with an estimate when the server sends none."""
    if not usage:
        return Usage(prompt=estimate_tokens(messages), completion=estimate_tokens(msg))
    details = usage.get("prompt_tokens_details") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    if not prompt and not completion and usage.get("total_tokens"):
        prompt = int(usage["total_tokens"])
    return Usage(prompt=prompt, cached=min(int(details.get("cached_tokens") or 0), prompt), completion=completion)


# --------------------------------------------------------------------- testing
Policy = Callable[[str, list[dict]], "dict | Awaitable[dict]"]


def say(text: str) -> dict:
    return {"role": "assistant", "content": text}


def call(tool: str, /, **args: Any) -> dict:
    """An assistant message that calls `tool` with `args` (for scripted policies)."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{random.randrange(1 << 30)}",
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(args)},
            }
        ],
    }


class ScriptedModel:
    """Calls `policy(agent_name, messages)` instead of an LLM."""

    def __init__(self, policy: Policy, tokens_per_call: int | Usage = 100, delay_s: float = 0.0):
        self.policy = policy
        self.usage = tokens_per_call if isinstance(tokens_per_call, Usage) else Usage(prompt=tokens_per_call)
        self.delay_s = delay_s
        self.calls: list[tuple[str, list[dict]]] = []

    async def complete(self, messages: list[dict], tools: list[dict] | None, *, agent: str) -> tuple[dict, Usage]:
        self.calls.append((agent, messages))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        out = self.policy(agent, messages)
        if asyncio.iscoroutine(out):
            out = await out
        return out, self.usage
