"""Start a mesh on a goal and run it until it goes quiet or hits a budget."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .log import EventLog
from .model import Model
from .pricing import Pricing, Usage
from .world import MESH, Limits, World


@dataclass
class RunResult:
    stop_reason: str
    tasks: list[dict[str, Any]]
    agents: list[dict[str, Any]]
    tokens_used: int
    usage: Usage
    cost_usd: float | None  # None when the model's prices are unknown
    workspace: Path
    final: str | None  # contents of FINAL.md, if an agent wrote it
    world: World


def _seed_message(goal: str, names: list[str]) -> str:
    return (
        f"A new goal was posted as task #1:\n\n{goal}\n\n"
        f"Agents in the mesh right now: {', '.join(names)}. Nobody is in charge. "
        "Look at the board, decide together how to split the work, claim tasks before doing them, "
        "and make sure someone assembles FINAL.md and completes task #1 when the goal is met."
    )


async def run_mesh(
    goal: str,
    model: Model,
    *,
    agents: int | list[str] = 3,
    workspace: str | Path = "mesh-run",
    limits: Limits | None = None,
    echo: bool = True,
    pricing: Pricing | None = None,
) -> RunResult:
    """Create `agents` peers (a count, or a list of roles), give them the goal, run."""
    limits = limits or Limits()
    workspace = Path(workspace).resolve()
    log = EventLog(workspace / ".mesh" / "events.jsonl", echo=echo)
    world = World(workspace, model, limits, log, pricing)

    roles = [""] * agents if isinstance(agents, int) else list(agents)
    log.emit("start", MESH, goal=goal, roles=roles, limits=limits.__dict__, pricing=pricing.__dict__ if pricing else None)
    world.post_task(MESH, "GOAL", goal)
    names = [f"agent-{i + 1}" for i in range(len(roles))]
    for name, role in zip(names, roles):
        world.add_agent(name, role)
    world.broadcast(MESH, _seed_message(goal, names))

    try:
        await asyncio.wait_for(world.done.wait(), timeout=limits.timeout_s)
    except asyncio.TimeoutError:
        world.stop(f"timeout after {limits.timeout_s}s")
    finally:
        world.stop("interrupted")  # no-op if already stopped
        for task in world._runners:
            task.cancel()
        await asyncio.gather(*world._runners, return_exceptions=True)

    final_path = workspace / "FINAL.md"
    result = RunResult(
        stop_reason=world.stop_reason or "unknown",
        tasks=world.board(),
        agents=[
            {
                **a,
                "steps": world.agents[a["name"]].steps,
                "usage": world.agents[a["name"]].usage.as_dict(),
                "cost_usd": pricing.cost(world.agents[a["name"]].usage) if pricing else None,
            }
            for a in world.roster()
        ],
        tokens_used=world.tokens_used,
        usage=world.usage,
        cost_usd=world.cost_usd,
        workspace=workspace,
        final=final_path.read_text(encoding="utf-8") if final_path.is_file() else None,
        world=world,
    )
    _write_report(result)
    log.close()
    return result


def _money(x: float | None) -> str:
    return "unknown (no prices for this model)" if x is None else f"${x:.4f}"


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "0%"


def _write_report(r: RunResult) -> None:
    lines = [
        "# Mesh run report",
        "",
        f"- Stop reason: {r.stop_reason}",
        f"- Cost: {_money(r.cost_usd)}",
        f"- Tokens: {r.usage.prompt} input ({r.usage.cached} cached, {_pct(r.usage.cached, r.usage.prompt)}), "
        f"{r.usage.completion} output, {r.tokens_used} total",
        f"- FINAL.md written: {'yes' if r.final is not None else 'no'}",
        "",
        "## Agents",
        "",
        *[
            f"- {a['name']} ({a['role'] or 'generalist'}): {a['status']}, {a['steps']} model calls, "
            f"{a['usage']['prompt']} input ({_pct(a['usage']['cached'], a['usage']['prompt'])} cached), "
            f"{a['usage']['completion']} output, {_money(a['cost_usd'])}"
            for a in r.agents
        ],
        "",
        "## Tasks",
        "",
    ]
    for t in r.tasks:
        lines.append(f"- #{t['id']} [{t['status']}] {t['title']} (posted by {t['posted_by']}, owner {t['owner'] or '-'})")
        if t.get("history"):
            lines.append(f"  - handovers: {'; '.join(t['history'])}")
        if t["result"]:
            lines.append(f"  - result: {t['result']}")
    msgs = [e for e in r.world.log.events if e["kind"] == "message"]
    pairs: dict[str, int] = {}
    for e in msgs:
        key = f"{e['agent']} → {e['to']}"
        pairs[key] = pairs.get(key, 0) + 1
    lines += ["", "## Who talked to whom", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])]
    out = r.workspace / ".mesh" / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (r.workspace / ".mesh" / "board.json").write_text(json.dumps(r.tasks, indent=2), encoding="utf-8")
