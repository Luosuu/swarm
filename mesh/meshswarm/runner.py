"""Start a mesh (or several collaborating teams) on a goal and run it until it
goes quiet or hits a budget."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .log import EventLog
from .model import Model
from .pricing import Pricing, Usage
from .world import EXCHANGE, MESH, SHARED_DIR, Limits, World


@dataclass
class RunResult:
    stop_reason: str
    tasks: list[dict[str, Any]]
    agents: list[dict[str, Any]]
    tokens_used: int
    usage: Usage
    cost_usd: float | None  # None when some model's prices are unknown
    workspace: Path
    final: str | None  # single team: contents of FINAL.md, if written
    world: World
    teams: list[dict[str, Any]] = field(default_factory=list)  # multi-team: per-team summary
    finals: dict[str, str | None] = field(default_factory=dict)  # multi-team: each team's FINAL.md


@dataclass
class TeamSpec:
    """One team in a multi-team run."""

    name: str
    goal: str
    model: Model
    agents: int | list[str] = 2  # a count, or a list of roles
    pricing: Pricing | None = None
    max_total_tokens: int | None = None  # defaults to Limits.max_total_tokens
    max_cost_usd: float | None = None
    liaisons: int | list[str] = 1  # how many of the first agents are liaisons, or their names (e.g. ["agent-1"])


def _seed_message(goal: str, names: list[str]) -> str:
    return (
        f"A new goal was posted as task #1:\n\n{goal}\n\n"
        f"Agents in the mesh right now: {', '.join(names)}. Nobody is in charge. "
        "Look at the board, decide together how to split the work, claim tasks before doing them, "
        "and make sure someone assembles FINAL.md and completes task #1 when the goal is met."
    )


def _team_seed(spec: TeamSpec, goal_id: int, names: list[str], liaisons: set[str], world: World, shared_goal: str) -> str:
    others = ", ".join(t for t in world.teams if t != spec.name)
    context = f"Context shared by all teams: {shared_goal}\n\n" if shared_goal else ""
    liaison_line = (
        f" Your team's liaisons to other teams: {', '.join(sorted(liaisons))}." if world.comm == "liaisons" else ""
    )
    return (
        f"{context}Your team ({spec.name}) has a goal, posted as task #{goal_id} on your team board:\n\n{spec.goal}\n\n"
        f"Your team: {', '.join(names)}. Nobody is in charge.{liaison_line} Other teams: {others}. "
        "Decide together how to split the work, claim tasks before doing them, coordinate with the other teams "
        "through the exchange board and shared/ where your goal depends on them, and make sure someone assembles "
        f"your team's FINAL.md and completes task #{goal_id} when your goal is met."
    )


async def _run(world: World, limits: Limits) -> None:
    try:
        await asyncio.wait_for(world.done.wait(), timeout=limits.timeout_s)
    except asyncio.TimeoutError:
        world.stop(f"timeout after {limits.timeout_s}s")
    finally:
        world.stop("interrupted")  # no-op if already stopped
        for task in world._runners:
            task.cancel()
        await asyncio.gather(*world._runners, return_exceptions=True)


def _agent_rows(world: World) -> list[dict[str, Any]]:
    rows = []
    for a in world.roster(include_subs=True):
        agent = world.agents[a["name"]]
        pricing = world.teams[agent.team].pricing
        rows.append({
            **a,
            "steps": agent.steps,
            "usage": agent.usage.as_dict(),
            "cost_usd": pricing.cost(agent.usage) if pricing else None,
        })
    return rows


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
    await _run(world, limits)

    final_path = workspace / "FINAL.md"
    result = RunResult(
        stop_reason=world.stop_reason or "unknown",
        tasks=world.board(),
        agents=_agent_rows(world),
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


async def run_teams(
    teams: list[TeamSpec],
    *,
    comm: str = "liaisons",
    shared_goal: str = "",
    workspace: str | Path = "mesh-run",
    limits: Limits | None = None,
    echo: bool = True,
) -> RunResult:
    """Run several teams, each with its own goal, model, budget, board and folder.
    Teams meet on the exchange board and in shared/; `comm` sets who may message
    across teams: "open" (anyone), "liaisons" (liaison to liaison), "exchange" (nobody)."""
    if len(teams) < 2:
        raise ValueError("run_teams needs at least two teams; use run_mesh for one")
    limits = limits or Limits()
    workspace = Path(workspace).resolve()
    log = EventLog(workspace / ".mesh" / "events.jsonl", echo=echo)
    world = World(workspace, None, limits, log, comm=comm)
    log.emit("start", MESH, shared_goal=shared_goal, comm=comm, limits=limits.__dict__,
             teams=[{"name": t.name, "goal": t.goal, "roles": _roles(t)} for t in teams])
    for spec in teams:
        world.add_team(spec.name, spec.model, goal=spec.goal, pricing=spec.pricing,
                       max_total_tokens=spec.max_total_tokens, max_cost_usd=spec.max_cost_usd)

    seeds: list[tuple[TeamSpec, int, list[str], set[str]]] = []
    for spec in teams:
        goal_task = world.post_task(MESH, "GOAL", spec.goal, board=spec.name)
        roles = _roles(spec)
        short = [f"agent-{i + 1}" for i in range(len(roles))]
        liaison_short = set(short[: spec.liaisons]) if isinstance(spec.liaisons, int) else set(spec.liaisons)
        names = []
        for name, role in zip(short, roles):
            names.append(world.add_agent(name, role, team=spec.name, liaison=name in liaison_short).name)
        seeds.append((spec, goal_task.id, names, {f"{spec.name}/{n}" for n in liaison_short}))
    for spec, goal_id, names, liaisons in seeds:
        text = _team_seed(spec, goal_id, names, liaisons, world, shared_goal)
        for name in names:
            world.deliver(MESH, name, text)
    await _run(world, limits)

    finals = {}
    team_rows = []
    for name, team in world.teams.items():
        path = team.workspace / "FINAL.md"
        finals[name] = path.read_text(encoding="utf-8") if path.is_file() else None
        goal = next((t for t in world.tasks.values() if t.board == name and t.title == "GOAL"), None)
        team_rows.append({
            "name": name,
            "goal_done": bool(goal and goal.status == "done"),
            "halted": team.halted,
            "usage": team.usage.as_dict(),
            "cost_usd": team.cost_usd,
            "liaisons": sorted(team.liaisons),
        })
    result = RunResult(
        stop_reason=world.stop_reason or "unknown",
        tasks=world.board(),
        agents=_agent_rows(world),
        tokens_used=world.tokens_used,
        usage=world.usage,
        cost_usd=world.cost_usd,
        workspace=workspace,
        final=None,
        world=world,
        teams=team_rows,
        finals=finals,
    )
    _write_report(result)
    log.close()
    return result


def _roles(spec: TeamSpec) -> list[str]:
    return [""] * spec.agents if isinstance(spec.agents, int) else list(spec.agents)


def _money(x: float | None) -> str:
    return "unknown (no prices for this model)" if x is None else f"${x:.4f}"


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "0%"


def _agent_line(a: dict[str, Any], indent: int = 0) -> str:
    flag = ", liaison" if a.get("liaison") else ""
    kind = "sub-agent" if a.get("parent") else "generalist"
    return (
        f"{'  ' * indent}- {a['name']} ({a['role'] or kind}{flag}): {a['status']}, {a['steps']} model calls, "
        f"{a['usage']['prompt']} input ({_pct(a['usage']['cached'], a['usage']['prompt'])} cached), "
        f"{a['usage']['completion']} output, {_money(a['cost_usd'])}"
    )


def _agent_tree(agents: list[dict[str, Any]], team: str | None = None) -> list[str]:
    """Mesh agents, each followed by its sub-agent tree, with subtree cost totals."""
    rows = [a for a in agents if team is None or a.get("team") == team]
    kids: dict[str, list[dict[str, Any]]] = {}
    for a in rows:
        if a.get("parent"):
            kids.setdefault(a["parent"], []).append(a)

    def subtree_cost(a: dict[str, Any]) -> float | None:
        costs = [a["cost_usd"], *(subtree_cost(k) for k in kids.get(a["name"], []))]
        return None if any(c is None for c in costs) else sum(costs)  # type: ignore[arg-type]

    lines: list[str] = []

    def walk(a: dict[str, Any], depth: int) -> None:
        line = _agent_line(a, depth)
        if kids.get(a["name"]):
            line += f" (with sub-agents: {_money(subtree_cost(a))})"
        lines.append(line)
        for k in kids.get(a["name"], []):
            walk(k, depth + 1)

    for a in rows:
        if not a.get("parent"):
            walk(a, 0)
    return lines


def _task_lines(tasks: list[dict[str, Any]]) -> list[str]:
    lines = []
    for t in tasks:
        to = f", for {t['for_team']}" if t.get("for_team") else ""
        lines.append(f"- #{t['id']} [{t['status']}] {t['title']} (posted by {t['posted_by']}{to}, owner {t['owner'] or '-'})")
        if t.get("history"):
            lines.append(f"  - handovers: {'; '.join(t['history'])}")
        if t["result"]:
            lines.append(f"  - result: {t['result']}")
    return lines


def _write_report(r: RunResult) -> None:
    multi = bool(r.teams)
    lines = [
        "# Mesh run report",
        "",
        f"- Stop reason: {r.stop_reason}",
        f"- Cost: {_money(r.cost_usd)}",
        f"- Tokens: {r.usage.prompt} input ({r.usage.cached} cached, {_pct(r.usage.cached, r.usage.prompt)}), "
        f"{r.usage.completion} output, {r.tokens_used} total",
    ]
    if not multi:
        lines += [
            f"- FINAL.md written: {'yes' if r.final is not None else 'no'}",
            "", "## Agents", "", *_agent_tree(r.agents),
            "", "## Tasks", "", *_task_lines(r.tasks),
        ]
    else:
        lines.append(f"- Cross-team rule: {r.world.comm}")
        for t in r.teams:
            final = r.finals.get(t["name"])
            words = f"{len(final.split())} words" if final is not None else "not written"
            lines += [
                "", f"## Team {t['name']}", "",
                f"- Goal done: {'yes' if t['goal_done'] else 'no'}{' (halted: out of budget)' if t['halted'] else ''}",
                f"- FINAL.md: {words}",
                f"- Cost: {_money(t['cost_usd'])}, {t['usage']['prompt']} input "
                f"({_pct(t['usage']['cached'], t['usage']['prompt'])} cached), {t['usage']['completion']} output",
                "", "### Agents", "", *_agent_tree(r.agents, t["name"]),
                "", "### Team board", "", *_task_lines([x for x in r.tasks if x.get("board") == t["name"]]),
            ]
        lines += ["", "## Exchange board", "", *(_task_lines([x for x in r.tasks if x.get("board") == EXCHANGE]) or ["- (empty)"])]
        shared = r.workspace / SHARED_DIR
        files = sorted(str(p.relative_to(shared)) for p in shared.rglob("*") if p.is_file()) if shared.is_dir() else []
        lines += ["", "## shared/", "", *([f"- {f}" for f in files] or ["- (empty)"])]

    msgs = [e for e in r.world.log.events if e["kind"] == "message"]
    pairs: dict[str, int] = {}
    for e in msgs:
        key = f"{e['agent']} → {e['to']}"
        pairs[key] = pairs.get(key, 0) + 1
    if multi:
        cross = sum(1 for e in msgs if e.get("cross_team"))
        within = sum(1 for e in msgs if e["agent"] != MESH and not e.get("cross_team"))
        lines += ["", "## Messages", "", f"- Within teams: {within}", f"- Across teams: {cross}"]
    delegations = [e for e in r.world.log.events if e["kind"] == "delegate"]
    if delegations:
        reports = [e for e in r.world.log.events if e["kind"] == "report"]
        lines += ["", "## Delegation", "", f"- Tasks delegated: {len(delegations)} "
                  f"({sum(1 for e in delegations if e.get('follow_up'))} follow-ups)",
                  f"- Reports: {len(reports)} ({sum(1 for e in reports if e.get('auto'))} automatic, sub-agent never called report)"]
    lines += ["", "## Who talked to whom", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])]
    out = r.workspace / ".mesh" / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (r.workspace / ".mesh" / "board.json").write_text(json.dumps(r.tasks, indent=2), encoding="utf-8")
