"""Command line: python -m meshswarm "your goal" --agents 3"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from .model import ChatEndpoint
from .pricing import Pricing, lookup
from .runner import run_mesh
from .world import Limits


def _env(*names: str) -> str | None:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    return None


def read_key_file(path: str) -> str:
    """Read an API key from a file holding either the bare key or NAME=key."""
    with open(os.path.expanduser(path), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                line = line.split("=", 1)[1].strip()
            return line.strip("'\"")
    raise ValueError(f"no key found in {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="meshswarm", description="Run a leaderless mesh of LLM agents on a goal.")
    p.add_argument("goal", help="the goal, or @path/to/file.md to read it from a file")
    p.add_argument("--agents", default="3", help="number of starting agents, or comma-separated roles (e.g. researcher,writer,critic)")
    p.add_argument("--workspace", help="shared workspace folder (default: runs/<timestamp>)")
    p.add_argument("--base-url", default=_env("MESH_BASE_URL", "OPENAI_BASE_URL"), help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1 [env MESH_BASE_URL]")
    p.add_argument("--model", default=_env("MESH_MODEL"), help="model name served by the endpoint [env MESH_MODEL]")
    p.add_argument("--api-key-env", default="MESH_API_KEY", help="name of the env var holding the API key (default MESH_API_KEY)")
    p.add_argument("--api-key-file", default=_env("MESH_API_KEY_FILE"), help="file holding the API key (bare key or NAME=key); overrides --api-key-env [env MESH_API_KEY_FILE]")
    p.add_argument("--max-completion-tokens", type=int, help="max_tokens sent with every request")
    p.add_argument("--temperature", type=float)
    p.add_argument("--extra-body", help='JSON merged into every request, e.g. \'{"top_p":0.9}\'')
    p.add_argument("--ca-file", help="PEM bundle to trust for an HTTPS endpoint with a private CA")
    d = Limits()
    p.add_argument("--max-agents", type=int, default=d.max_agents)
    p.add_argument("--max-steps", type=int, default=d.max_steps_per_agent, help="model calls per agent")
    p.add_argument("--max-tokens", type=int, default=d.max_total_tokens, help="token budget for the run, cached tokens included")
    p.add_argument("--max-cost", type=float, help="dollar budget for the run (soft; hard stop at 1.25x). Needs known or given prices")
    p.add_argument("--price-input", type=float, help="USD per million uncached input tokens (overrides the built-in table)")
    p.add_argument("--price-cached", type=float, help="USD per million cached input tokens")
    p.add_argument("--price-output", type=float, help="USD per million output tokens")
    p.add_argument("--concurrency", type=int, default=d.max_concurrent_calls, help="parallel model calls")
    p.add_argument("--compact-at", type=int, default=d.compact_at_tokens, help="estimated context tokens that trigger compaction")
    p.add_argument("--timeout", type=float, default=d.timeout_s, help="wall-clock limit in seconds")
    p.add_argument("--quiet", action="store_true", help="do not print events")
    a = p.parse_args(argv)

    if not a.base_url or not a.model:
        p.error("set --base-url and --model (or MESH_BASE_URL and MESH_MODEL)")
    goal = open(a.goal[1:], encoding="utf-8").read() if a.goal.startswith("@") else a.goal
    agents: int | list[str] = int(a.agents) if a.agents.isdigit() else [r.strip() for r in a.agents.split(",") if r.strip()]
    workspace = a.workspace or os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))

    extra_body = json.loads(a.extra_body) if a.extra_body else {}
    if a.max_completion_tokens:
        extra_body["max_tokens"] = a.max_completion_tokens
    model = ChatEndpoint(
        a.base_url,
        a.model,
        api_key=read_key_file(a.api_key_file) if a.api_key_file else os.environ.get(a.api_key_env),
        temperature=a.temperature,
        extra_body=extra_body,
        ca_file=a.ca_file,
    )
    pricing = lookup(a.model)
    given = (a.price_input, a.price_cached, a.price_output)
    if any(v is not None for v in given):
        if pricing is None and None in given:
            p.error(f"no built-in prices for {a.model}; pass all of --price-input, --price-cached, --price-output")
        base = pricing or Pricing(0, 0, 0)
        pricing = Pricing(
            input=a.price_input if a.price_input is not None else base.input,
            cached_input=a.price_cached if a.price_cached is not None else base.cached_input,
            output=a.price_output if a.price_output is not None else base.output,
        )
    if a.max_cost is not None and pricing is None:
        p.error(f"--max-cost needs prices: {a.model} is not in the built-in table, so pass --price-input/--price-cached/--price-output")
    limits = Limits(
        max_agents=a.max_agents,
        max_steps_per_agent=a.max_steps,
        max_total_tokens=a.max_tokens,
        max_cost_usd=a.max_cost,
        max_concurrent_calls=a.concurrency,
        compact_at_tokens=a.compact_at,
        timeout_s=a.timeout,
    )
    try:
        r = asyncio.run(run_mesh(goal, model, agents=agents, workspace=workspace, limits=limits, echo=not a.quiet, pricing=pricing))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    cost = "unknown" if r.cost_usd is None else f"${r.cost_usd:.4f}"
    cached = f"{100 * r.usage.cached / r.usage.prompt:.0f}%" if r.usage.prompt else "0%"
    print(f"\nstopped: {r.stop_reason} | cost: {cost} | tokens: {r.tokens_used} ({cached} of input cached) | agents: {len(r.agents)}")
    print(f"workspace: {r.workspace}")
    print(f"report:    {r.workspace / '.mesh' / 'report.md'}")
    print(f"FINAL.md:  {'written' if r.final is not None else 'not written'}")
    return 0
