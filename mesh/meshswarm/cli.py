"""Command line.

One team:      python -m meshswarm "your goal" --agents 3
Several teams: python -m meshswarm --teams teams.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .model import ChatEndpoint
from .pricing import Pricing, lookup
from .runner import RunResult, TeamSpec, run_mesh, run_teams
from .world import COMM_RULES, Limits

TEAMS_EXAMPLE = """{
  "comm": "liaisons",
  "shared_goal": "Optional context every team sees.",
  "teams": [
    {"name": "spec", "goal": "...", "agents": "architect,critic", "liaisons": 1, "max_cost": 0.05},
    {"name": "build", "goal": "...", "agents": 3, "model": "other/model", "base_url": "http://...", "api_key_file": "..."}
  ]
}"""


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


def _agents_arg(value: Any) -> int | list[str]:
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return [str(r) for r in value]
    value = str(value)
    return int(value) if value.isdigit() else [r.strip() for r in value.split(",") if r.strip()]


def _pricing(model: str, price_input: float | None, price_cached: float | None, price_output: float | None) -> Pricing | None:
    pricing = lookup(model)
    given = (price_input, price_cached, price_output)
    if all(v is None for v in given):
        return pricing
    if pricing is None and None in given:
        raise ValueError(f"no built-in prices for {model}; give all of input, cached and output prices")
    base = pricing or Pricing(0, 0, 0)
    return Pricing(
        input=price_input if price_input is not None else base.input,
        cached_input=price_cached if price_cached is not None else base.cached_input,
        output=price_output if price_output is not None else base.output,
    )


class _Endpoints:
    """Builds one ChatEndpoint per (url, model, key), with CLI flags as defaults."""

    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.cache: dict[tuple, ChatEndpoint] = {}
        extra = json.loads(a.extra_body) if a.extra_body else {}
        if a.max_completion_tokens:
            extra["max_tokens"] = a.max_completion_tokens
        self.extra = extra

    def get(self, base_url: str | None, model: str | None, key_file: str | None, key_env: str | None) -> ChatEndpoint:
        a = self.a
        base_url, model = base_url or a.base_url, model or a.model
        if not base_url or not model:
            raise ValueError("set --base-url and --model (or MESH_BASE_URL and MESH_MODEL, or per-team base_url/model)")
        key_file = key_file or a.api_key_file
        key_env = key_env or a.api_key_env
        cache_key = (base_url, model, key_file, key_env)
        if cache_key not in self.cache:
            self.cache[cache_key] = ChatEndpoint(
                base_url,
                model,
                api_key=read_key_file(key_file) if key_file else os.environ.get(key_env),
                temperature=a.temperature,
                extra_body=self.extra,
                ca_file=a.ca_file,
            )
        return self.cache[cache_key]


def _team_specs(config: dict[str, Any], a: argparse.Namespace, endpoints: _Endpoints) -> list[TeamSpec]:
    specs = []
    for t in config.get("teams", []):
        unknown = set(t) - {"name", "goal", "agents", "liaisons", "model", "base_url", "api_key_file", "api_key_env",
                            "max_cost", "max_tokens", "price_input", "price_cached", "price_output"}
        if unknown:
            raise ValueError(f"team {t.get('name')!r}: unknown keys {sorted(unknown)}")
        model = endpoints.get(t.get("base_url"), t.get("model"), t.get("api_key_file"), t.get("api_key_env"))
        pricing = _pricing(
            model.model,
            t.get("price_input", a.price_input),
            t.get("price_cached", a.price_cached),
            t.get("price_output", a.price_output),
        )
        max_cost = t.get("max_cost", a.max_cost)
        if max_cost is not None and pricing is None:
            raise ValueError(f"team {t['name']!r}: max_cost needs prices for {model.model}")
        specs.append(TeamSpec(
            name=t["name"],
            goal=t["goal"],
            model=model,
            agents=_agents_arg(t.get("agents", 2)),
            pricing=pricing,
            max_total_tokens=t.get("max_tokens"),
            max_cost_usd=max_cost,
            liaisons=t.get("liaisons", 1),
        ))
    return specs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="meshswarm", description="Run a leaderless mesh of LLM agents on a goal.",
                                epilog=f"--teams file format:\n{TEAMS_EXAMPLE}", formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("goal", nargs="?", help="the goal, or @path/to/file.md to read it from a file (single team)")
    p.add_argument("--teams", help="JSON file describing several collaborating teams (see below)")
    p.add_argument("--comm", choices=COMM_RULES, help="cross-team message rule (overrides the --teams file; default liaisons)")
    p.add_argument("--agents", default="3", help="number of starting agents, or comma-separated roles (e.g. researcher,writer,critic)")
    p.add_argument("--workspace", help="workspace folder (default: runs/<timestamp>)")
    p.add_argument("--base-url", default=_env("MESH_BASE_URL", "OPENAI_BASE_URL"), help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1 [env MESH_BASE_URL]")
    p.add_argument("--model", default=_env("MESH_MODEL"), help="model name served by the endpoint [env MESH_MODEL]")
    p.add_argument("--api-key-env", default="MESH_API_KEY", help="name of the env var holding the API key (default MESH_API_KEY)")
    p.add_argument("--api-key-file", default=_env("MESH_API_KEY_FILE"), help="file holding the API key (bare key or NAME=key); overrides --api-key-env [env MESH_API_KEY_FILE]")
    p.add_argument("--max-completion-tokens", type=int, help="max_tokens sent with every request")
    p.add_argument("--temperature", type=float)
    p.add_argument("--extra-body", help='JSON merged into every request, e.g. \'{"top_p":0.9}\'')
    p.add_argument("--ca-file", help="PEM bundle to trust for an HTTPS endpoint with a private CA")
    d = Limits()
    p.add_argument("--max-agents", type=int, default=d.max_agents, help="across all teams")
    p.add_argument("--max-steps", type=int, default=d.max_steps_per_agent, help="model calls per agent")
    p.add_argument("--max-tokens", type=int, default=d.max_total_tokens, help="token budget (per team), cached tokens included")
    p.add_argument("--max-cost", type=float, help="dollar budget (per team; soft, hard stop at 1.25x). Needs known or given prices")
    p.add_argument("--price-input", type=float, help="USD per million uncached input tokens (overrides the built-in table)")
    p.add_argument("--price-cached", type=float, help="USD per million cached input tokens")
    p.add_argument("--price-output", type=float, help="USD per million output tokens")
    p.add_argument("--concurrency", type=int, default=d.max_concurrent_calls, help="parallel model calls")
    p.add_argument("--compact-at", type=int, default=d.compact_at_tokens, help="estimated context tokens that trigger compaction")
    p.add_argument("--timeout", type=float, default=d.timeout_s, help="wall-clock limit in seconds")
    p.add_argument("--quiet", action="store_true", help="do not print events")
    a = p.parse_args(argv)

    if bool(a.goal) == bool(a.teams):
        p.error("give either a goal (one team) or --teams FILE (several teams)")
    workspace = a.workspace or os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    limits = Limits(
        max_agents=a.max_agents,
        max_steps_per_agent=a.max_steps,
        max_total_tokens=a.max_tokens,
        max_cost_usd=a.max_cost,
        max_concurrent_calls=a.concurrency,
        compact_at_tokens=a.compact_at,
        timeout_s=a.timeout,
    )
    endpoints = _Endpoints(a)
    try:
        if a.teams:
            config = json.loads(Path(a.teams).read_text(encoding="utf-8"))
            specs = _team_specs(config, a, endpoints)
            if len(specs) < 2:
                raise ValueError("--teams needs at least two teams; for one team pass a goal instead")
            limits.max_cost_usd = None  # budgets are per team, set on each TeamSpec
            run = run_teams(specs, comm=a.comm or config.get("comm", "liaisons"), shared_goal=config.get("shared_goal", ""),
                            workspace=workspace, limits=limits, echo=not a.quiet)
        else:
            model = endpoints.get(None, None, None, None)
            pricing = _pricing(model.model, a.price_input, a.price_cached, a.price_output)
            if a.max_cost is not None and pricing is None:
                p.error(f"--max-cost needs prices: {model.model} is not in the built-in table, so pass --price-input/--price-cached/--price-output")
            goal = open(a.goal[1:], encoding="utf-8").read() if a.goal.startswith("@") else a.goal
            run = run_mesh(goal, model, agents=_agents_arg(a.agents), workspace=workspace, limits=limits,
                           echo=not a.quiet, pricing=pricing)
    except (ValueError, KeyError, OSError) as e:
        p.error(str(e))
    try:
        r = asyncio.run(run)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    _print_summary(r)
    return 0


def _print_summary(r: RunResult) -> None:
    cost = "unknown" if r.cost_usd is None else f"${r.cost_usd:.4f}"
    cached = f"{100 * r.usage.cached / r.usage.prompt:.0f}%" if r.usage.prompt else "0%"
    print(f"\nstopped: {r.stop_reason} | cost: {cost} | tokens: {r.tokens_used} ({cached} of input cached) | agents: {len(r.agents)}")
    print(f"workspace: {r.workspace}")
    print(f"report:    {r.workspace / '.mesh' / 'report.md'}")
    if r.teams:
        for t in r.teams:
            final = r.finals.get(t["name"])
            tcost = "unknown" if t["cost_usd"] is None else f"${t['cost_usd']:.4f}"
            print(f"team {t['name']}: goal {'done' if t['goal_done'] else 'NOT done'}, "
                  f"FINAL.md {'written' if final is not None else 'not written'}, cost {tcost}")
    else:
        print(f"FINAL.md:  {'written' if r.final is not None else 'not written'}")
