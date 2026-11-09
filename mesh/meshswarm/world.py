"""The shared world: teams, agent directory, mailboxes, task boards, workspaces, budgets.

Everything runs on one asyncio event loop, so each method here is atomic with
respect to other agents (no locks needed). The world never decides *what*
agents do; it only delivers messages, keeps the shared state and enforces
limits and boundaries.

A world has one or more teams. A single-team world is the classic mesh: one
board, one workspace, anyone can message anyone. With several teams, each team
has a private board and workspace folder; teams meet on the exchange board and
in the `shared/` folder, and cross-team messages follow the world's `comm` rule.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .log import EventLog
from .pricing import Pricing, Usage

if TYPE_CHECKING:
    from .agent import MeshAgent
    from .model import Model

MESH = "mesh"  # sender name for system notices
EXCHANGE = "exchange"  # board shared by all teams
SHARED_DIR = "shared"  # workspace folder shared by all teams
COMM_RULES = ("open", "liaisons", "exchange")
_RESERVED_TEAM_NAMES = {EXCHANGE, SHARED_DIR, ".mesh", MESH, "all"}


class RunStopped(Exception):
    """Raised inside an agent when the run has ended."""


@dataclass
class Limits:
    max_agents: int = 8  # mesh agents across all teams (sub-agents are counted separately)
    max_subagents: int = 16  # tree-style children created with delegate(), across the run
    max_depth: int = 2  # mesh agent = depth 0; its children 1; grandchildren 2
    max_steps_per_agent: int = 60  # model calls per agent over the whole run
    max_total_tokens: int = 2_000_000  # per team, cached included. Soft: agents wrap up; hard stop at 1.25x
    max_cost_usd: float | None = None  # per team, same soft/hard rule, in dollars (needs pricing)
    max_message_chars: int = 4000  # longer content belongs in a file
    max_messages_per_agent: int = 100
    max_concurrent_calls: int = 4  # protects a local endpoint
    compact_at_tokens: int = 60_000  # estimated history size that triggers compaction
    nudges: int = 1  # wake-ups when everyone is idle but tasks remain open
    timeout_s: float = 1800


@dataclass
class Task:
    id: int
    title: str
    description: str
    posted_by: str
    board: str = ""  # team name, "" for a single-team world, or EXCHANGE
    for_team: str | None = None  # exchange tasks: the team asked to do it (None = anyone)
    status: str = "open"  # open | claimed | done
    owner: str | None = None
    result: str | None = None
    history: list[str] = field(default_factory=list)  # handovers, e.g. "agent-1 -> agent-3: please verify"


@dataclass
class Team:
    name: str  # "" for the single team of a classic mesh
    model: "Model"
    workspace: Path
    pricing: Pricing | None = None
    max_total_tokens: int = 2_000_000
    max_cost_usd: float | None = None
    goal: str = ""
    liaisons: set[str] = field(default_factory=set)
    usage: Usage = field(default_factory=Usage)
    wrapping_up: bool = False
    halted: bool = False  # hard budget limit hit: the team's agents are stopped

    @property
    def cost_usd(self) -> float | None:
        return self.pricing.cost(self.usage) if self.pricing else None

    def budget_used(self) -> tuple[float, str]:
        """Largest fraction of any budget used so far, and which budget it is."""
        tokens = self.usage.total
        used = [(tokens / self.max_total_tokens, f"token budget ({tokens} tokens)")]
        if self.max_cost_usd is not None and self.cost_usd is not None:
            used.append((self.cost_usd / self.max_cost_usd, f"cost budget (${self.cost_usd:.4f})"))
        return max(used)


@dataclass
class World:
    workspace: Path
    model: "Model | None" = None  # model of the default team; None when teams are added explicitly
    limits: Limits = field(default_factory=Limits)
    log: EventLog = field(default_factory=EventLog)
    pricing: Pricing | None = None
    comm: str = "liaisons"  # cross-team message rule: open | liaisons | exchange

    def __post_init__(self) -> None:
        if self.comm not in COMM_RULES:
            raise ValueError(f"comm must be one of {COMM_RULES}")
        self.workspace = Path(self.workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.teams: dict[str, Team] = {}
        if self.model is not None:
            self._add_team(Team("", self.model, self.workspace, self.pricing,
                                self.limits.max_total_tokens, self.limits.max_cost_usd))
        self.agents: dict[str, MeshAgent] = {}
        self.tasks: dict[int, Task] = {}
        self._next_task_id = 1
        self.nudges_left = self.limits.nudges
        self.done = asyncio.Event()
        self.stop_reason: str | None = None
        self._call_slots = asyncio.Semaphore(self.limits.max_concurrent_calls)
        self._runners: list[asyncio.Task] = []

    # ----------------------------------------------------------------- teams
    @property
    def multi_team(self) -> bool:
        return "" not in self.teams

    def add_team(
        self,
        name: str,
        model: "Model",
        *,
        goal: str = "",
        pricing: Pricing | None = None,
        max_total_tokens: int | None = None,
        max_cost_usd: float | None = None,
    ) -> Team:
        if "" in self.teams:
            if any(a.team == "" for a in self.agents.values()):
                raise ValueError("this world already runs as a single team")
            del self.teams[""]
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) or name in _RESERVED_TEAM_NAMES:
            raise ValueError(f"bad team name {name!r}: use lowercase letters, digits, - and _")
        if name in self.teams:
            raise ValueError(f"team {name!r} exists")
        team = Team(
            name, model, self.workspace / name, pricing,
            max_total_tokens if max_total_tokens is not None else self.limits.max_total_tokens,
            max_cost_usd if max_cost_usd is not None else self.limits.max_cost_usd,
            goal,
        )
        (self.workspace / SHARED_DIR).mkdir(exist_ok=True)
        return self._add_team(team)

    def _add_team(self, team: Team) -> Team:
        if team.max_cost_usd is not None and team.pricing is None:
            raise ValueError(f"max_cost_usd needs pricing for team {team.name or 'main'!r}")
        team.workspace.mkdir(parents=True, exist_ok=True)
        self.teams[team.name] = team
        if team.name:
            self.log.emit("team", MESH, team=team.name, goal=team.goal,
                          pricing=team.pricing.__dict__ if team.pricing else None,
                          max_total_tokens=team.max_total_tokens, max_cost_usd=team.max_cost_usd)
        return team

    def team_of(self, name: str) -> Team | None:
        agent = self.agents.get(name)
        return self.teams.get(agent.team) if agent else None

    def qualify(self, viewer: str, name: str) -> str:
        """In a multi-team world, a bare name means a teammate: 'agent-2' -> 'spec/agent-2'."""
        agent = self.agents.get(viewer)
        if self.multi_team and agent and "/" not in name and name not in ("all", MESH):
            return f"{agent.team}/{name}"
        return name

    def is_liaison(self, name: str) -> bool:
        team = self.team_of(name)
        return bool(team and name in team.liaisons)

    # ---------------------------------------------------------------- agents
    def add_agent(
        self,
        name: str,
        role: str = "",
        instructions: str = "",
        spawned_by: str | None = None,
        team: str = "",
        liaison: bool = False,
        parent: str | None = None,
    ) -> "MeshAgent":
        """Add a mesh agent, or (with `parent`) a private tree-style sub-agent of that agent."""
        from .agent import MeshAgent

        depth = 0
        if parent is not None:
            boss = self.agents[parent]
            team, depth = boss.team, boss.depth + 1
            name = f"{parent}:{name}"
            if depth > self.limits.max_depth:
                raise ValueError(f"sub-agent depth limit reached ({self.limits.max_depth})")
            if sum(1 for a in self.agents.values() if a.parent) >= self.limits.max_subagents:
                raise ValueError(f"sub-agent limit reached ({self.limits.max_subagents})")
        elif len([a for a in self.agents.values() if not a.parent]) >= self.limits.max_agents:
            raise ValueError(f"agent limit reached ({self.limits.max_agents})")
        if team not in self.teams:
            raise ValueError(f"no team {team!r}")
        if team and parent is None and not name.startswith(f"{team}/"):
            name = f"{team}/{name}"
        if name in self.agents or name in (MESH, "all") or (parent is None and ":" in name):
            raise ValueError(f"agent name {name!r} is taken or invalid")
        if self.teams[team].halted:
            raise ValueError(f"team {team!r} is out of budget")
        agent = MeshAgent(self, name, role, instructions, team=team, parent=parent, depth=depth)
        self.agents[name] = agent
        if liaison:
            self.teams[team].liaisons.add(name)
        self._runners.append(asyncio.ensure_future(agent.run()))
        self.log.emit("spawn", name, role=role, spawned_by=spawned_by, team=team or None, liaison=liaison,
                      parent=parent, depth=depth or None)
        return agent

    def can_see_agent(self, viewer: str | None, name: str) -> bool:
        if self.agents[name].parent:  # sub-agents are private to their parent
            return viewer is None or self.agents[name].parent == viewer
        if viewer is None or not self.multi_team:
            return True
        if self.agents[name].team == self.agents[viewer].team:
            return True
        if self.comm == "open":
            return True
        return self.comm == "liaisons" and self.is_liaison(name)

    def roster(self, viewer: str | None = None, include_subs: bool = False) -> list[dict[str, Any]]:
        out = []
        for a in self.agents.values():
            if a.parent and not include_subs:
                continue
            if not self.can_see_agent(viewer, a.name):
                continue
            row: dict[str, Any] = {
                "name": a.name,
                "role": a.role,
                "status": a.status,
                "tasks": [t.id for t in self.tasks.values() if t.owner == a.name and t.status == "claimed"],
            }
            if self.multi_team:
                row["team"] = a.team
                row["liaison"] = a.name in self.teams[a.team].liaisons
            if a.parent:
                row["parent"] = a.parent
            out.append(row)
        return out

    # -------------------------------------------------------------- messages
    def cross_team_block(self, sender: str, to: str) -> str | None:
        """Why `sender` may not message `to` directly, or None if allowed."""
        if not self.multi_team or sender == MESH or sender not in self.agents or to not in self.agents:
            return None
        if self.agents[sender].team == self.agents[to].team:
            return None
        if self.comm == "open":
            return None
        if self.comm == "exchange":
            return "teams talk only through the exchange board and the shared/ folder, not by direct message"
        if not self.is_liaison(sender):
            liaisons = ", ".join(sorted(self.teams[self.agents[sender].team].liaisons)) or "none"
            return f"only liaisons can message other teams; ask your team's liaison ({liaisons})"
        if not self.is_liaison(to):
            other = self.teams[self.agents[to].team]
            return f"{to} is not a liaison; message {', '.join(sorted(other.liaisons)) or 'the exchange board'} instead"
        return None

    def deliver(self, sender: str, to: str, content: str) -> dict[str, Any]:
        """Put a message in an inbox. Idle targets are woken ("delivered"),
        busy targets see it after their current step ("queued")."""
        to = self.qualify(sender, to)
        target = self.agents.get(to)
        source = self.agents.get(sender)
        # Sub-agents are private: only their parent may reach them, and they reach only their parent
        # (plus their own children, when they delegate further down).
        hidden = target is not None and target.parent and sender not in (MESH, target.parent)
        if target is None or hidden:
            return {"to": to, "error": f"no agent named {to!r}"}
        if source is not None and source.parent and to != source.parent and target.parent != sender:
            return {"to": to, "error": "a sub-agent can only reply to its parent"}
        blocked = self.cross_team_block(sender, to)
        if blocked:
            return {"to": to, "error": blocked}
        if target.status == "stopped":
            return {"to": to, "error": f"{to} has stopped (out of budget)"}
        status = "queued" if target.status == "working" else "delivered"
        target.inbox.append((sender, content))
        target.status = "working"
        target.wake.set()
        cross = sender in self.agents and self.agents[sender].team != target.team
        self.log.emit("message", sender, to=to, content=content, delivery=status, cross_team=cross or None)
        return {"to": to, "status": status}

    def broadcast(self, sender: str, content: str) -> list[dict[str, Any]]:
        """From an agent: its own team. From the mesh: everyone."""
        team = self.agents[sender].team if sender in self.agents else None
        return [
            self.deliver(sender, name, content)
            for name, a in list(self.agents.items())
            if name != sender and not a.parent and (team is None or a.team == team)
        ]

    # ----------------------------------------------------------- task boards
    def can_see_task(self, viewer: str, task: Task) -> bool:
        if viewer == MESH or viewer not in self.agents:
            return True
        return task.board in (self.agents[viewer].team, EXCHANGE)

    def _visible(self, by: str, task_id: int) -> Task | dict[str, Any]:
        task = self.tasks.get(task_id)
        if task is None or not self.can_see_task(by, task):
            return {"error": f"no task #{task_id} on your boards"}
        return task

    def post_task(self, by: str, title: str, description: str = "", board: str | None = None,
                  for_team: str | None = None) -> Task:
        if board is None:
            board = self.agents[by].team if by in self.agents else ""
        if board == EXCHANGE and not self.multi_team:
            raise ValueError("there is no exchange board with a single team")
        if for_team is not None and for_team not in self.teams:
            raise ValueError(f"no team {for_team!r}")
        task = Task(self._next_task_id, title, description, by, board=board, for_team=for_team)
        self.tasks[task.id] = task
        self._next_task_id += 1
        self.log.emit("task_posted", by, task=task.id, title=title, board=board or None, for_team=for_team)
        return task

    def claim_task(self, by: str, task_id: int) -> dict[str, Any]:
        task = self._visible(by, task_id)
        if isinstance(task, dict):
            return task
        if task.status != "open":
            return {"error": f"task #{task_id} is {task.status}" + (f" by {task.owner}" if task.owner else "")}
        if task.for_team and by in self.agents and self.agents[by].team != task.for_team:
            return {"error": f"task #{task_id} is addressed to team {task.for_team}"}
        task.status, task.owner = "claimed", by
        self.log.emit("task_claimed", by, task=task_id)
        return {"ok": True, "task": asdict(task)}

    def release_task(self, by: str, task_id: int) -> dict[str, Any]:
        task = self.tasks.get(task_id)
        if task is None or task.owner != by or task.status != "claimed":
            return {"error": f"you do not hold task #{task_id}"}
        task.status, task.owner = "open", None
        self.log.emit("task_released", by, task=task_id)
        return {"ok": True}

    def handover_task(self, by: str, task_id: int, to: str, note: str = "") -> dict[str, Any]:
        """Pass a claimed task to another agent, who becomes its owner and is told why.
        Team-board tasks stay inside the team; exchange tasks go to anyone you may message."""
        to = self.qualify(by, to)
        task = self.tasks.get(task_id)
        if task is None:
            return {"error": f"no task #{task_id}"}
        if task.status != "claimed" or task.owner != by:
            return {"error": f"you do not hold task #{task_id}" + (f" (owner: {task.owner})" if task.owner else "")}
        target = self.agents.get(to)
        if target is None or to == by:
            return {"error": f"cannot hand over to {to!r}"}
        if task.board != EXCHANGE and by in self.agents and target.team != self.agents[by].team:
            return {"error": "tasks on your team board stay in your team; post on the exchange board to involve another team"}
        blocked = self.cross_team_block(by, to)
        if blocked:
            return {"error": blocked}
        if target.status == "stopped":
            return {"error": f"{to} has stopped (out of budget)"}
        task.owner = to
        task.history.append(f"{by} -> {to}" + (f": {note}" if note else ""))
        self.log.emit("task_handover", by, task=task_id, to=to, note=note)
        msg = f"{by} handed you task #{task_id} ({task.title}). You now own it; complete it when done."
        self.deliver(MESH, to, msg + (f" Note from {by}: {note}" if note else ""))
        return {"ok": True, "task": task_id, "owner": to}

    def complete_task(self, by: str, task_id: int, result: str) -> dict[str, Any]:
        task = self._visible(by, task_id)
        if isinstance(task, dict):
            return task
        if task.status == "done":
            return {"error": f"task #{task_id} is already done"}
        if task.status == "claimed" and task.owner != by:
            return {"error": f"task #{task_id} is claimed by {task.owner}; they can hand it over to you with handover_task"}
        task.status, task.owner, task.result = "done", by, result
        self.log.emit("task_done", by, task=task_id, result=result)
        if task.board == EXCHANGE and task.posted_by in self.agents and task.posted_by != by:
            # Close the loop: the requester learns its request was answered, even across teams.
            self.deliver(MESH, task.posted_by, f"Exchange task #{task_id} ({task.title}) was completed by {by}: {result}")
        return {"ok": True}

    def tasks_posted_by_others_since(self, agent: str, seen_id: int) -> list[Task]:
        return [t for t in self.tasks.values()
                if t.id > seen_id and t.posted_by not in (agent, MESH) and self.can_see_task(agent, t)]

    def similar_task(self, title: str, board: str = "") -> Task | None:
        """An unfinished task on the same board whose title is nearly the same, if any."""
        norm = _norm(title)
        for t in self.tasks.values():
            if t.status == "done" or t.title == "GOAL" or t.board != board:
                continue
            other = _norm(t.title)
            if norm == other or SequenceMatcher(None, norm, other).ratio() >= 0.85:
                return t
        return None

    def board(self, viewer: str | None = None, which: str = "all") -> list[dict[str, Any]]:
        tasks = [t for t in self.tasks.values() if viewer is None or self.can_see_task(viewer, t)]
        if which == EXCHANGE:
            tasks = [t for t in tasks if t.board == EXCHANGE]
        elif which == "team":
            tasks = [t for t in tasks if t.board != EXCHANGE]
        rows = [asdict(t) for t in tasks]
        if not self.multi_team:
            for r in rows:
                del r["board"], r["for_team"]
        return rows

    # ------------------------------------------------------------- workspace
    def resolve(self, viewer: str, path: str) -> Path:
        """Map an agent's relative path into its team folder (or `shared/` across teams)."""
        agent = self.agents.get(viewer)
        team = self.teams[agent.team] if agent else next(iter(self.teams.values()))
        root = team.workspace
        parts = Path(path).parts
        if self.multi_team and parts and parts[0] == SHARED_DIR:
            root, path = self.workspace / SHARED_DIR, str(Path(*parts[1:])) if len(parts) > 1 else "."
        p = (root / path).resolve()
        if p != root and root not in p.parents:
            where = "your team's folder or shared/" if self.multi_team else "the shared workspace"
            raise PermissionError(f"{path!r} is outside {where}")
        if ".mesh" in p.relative_to(root).parts:
            raise PermissionError("the .mesh folder is reserved for run logs")
        return p

    def display_path(self, viewer: str, p: Path) -> str:
        """Inverse of resolve: how the agent should refer to `p`."""
        shared = self.workspace / SHARED_DIR
        if self.multi_team and (p == shared or shared in p.parents):
            return str(Path(SHARED_DIR) / p.relative_to(shared))
        agent = self.agents.get(viewer)
        root = self.teams[agent.team].workspace if agent else self.workspace
        return str(p.relative_to(root))

    # ---------------------------------------------------------- model + stop
    @property
    def usage(self) -> Usage:
        total = Usage()
        for t in self.teams.values():
            total = total + t.usage
        return total

    @property
    def tokens_used(self) -> int:
        return self.usage.total

    @property
    def cost_usd(self) -> float | None:
        costs = [t.cost_usd for t in self.teams.values()]
        return None if any(c is None for c in costs) else sum(costs)  # type: ignore[arg-type]

    @property
    def wrapping_up(self) -> bool:
        live = [t for t in self.teams.values() if not t.halted]
        return bool(live) and all(t.wrapping_up for t in live)

    async def call_model(self, agent: "MeshAgent", messages: list[dict], tools: list[dict] | None) -> dict:
        team = self.teams[agent.team]
        async with self._call_slots:
            if self.done.is_set() or team.halted:
                raise RunStopped
            msg, usage = await team.model.complete(messages, tools, agent=agent.name)
        team.usage = team.usage + usage
        agent.usage = agent.usage + usage
        cost = team.pricing.cost(usage) if team.pricing else None
        self.log.emit("usage", agent.name, team=team.name or None, **usage.as_dict(), cost=cost)
        fraction, which = team.budget_used()
        if fraction >= 1.25:
            self._halt(team, f"{which} hard limit reached")
        elif fraction >= 1 and not team.wrapping_up:
            team.wrapping_up = True
            self.log.emit("wrap_up", MESH, budget=which, team=team.name or None)
            whose = f"Team {team.name}'s" if team.name else "The"
            notice = (
                f"{whose} {which.split(' (')[0]} is used up. Do not start new work. In at most two steps, record what you "
                "have (write files, complete tasks you finished, release ones you did not), then call wait()."
            )
            for a in self.agents.values():
                if a.team == team.name and a.status == "working":  # idle agents are not woken just to be told to stop
                    self.deliver(MESH, a.name, notice)
        return msg

    def _halt(self, team: Team, reason: str) -> None:
        if not self.multi_team:
            self.stop(reason)
            return
        if team.halted:
            return
        team.halted = True
        self.log.emit("team_halt", MESH, team=team.name, reason=reason)
        for a in self.agents.values():
            if a.team == team.name:
                a.status = "stopped"
                a.inbox.clear()
                a.wake.set()
        self.check_quiescent()

    def check_quiescent(self) -> None:
        """Stop when nobody is working and no message is in flight. Before
        stopping, nudge everyone once if tasks they can see are unfinished."""
        if self.done.is_set():
            return
        if any(a.status == "working" or a.inbox for a in self.agents.values()):
            return
        live_teams = {name for name, t in self.teams.items() if not t.halted and not t.wrapping_up}
        live = [a for a in self.agents.values() if a.status != "stopped" and a.team in live_teams and not a.parent]
        unfinished = [t for t in self.tasks.values() if t.status != "done" and (t.board in live_teams or t.board == EXCHANGE)]
        if unfinished and live and self.nudges_left > 0:
            self.nudges_left -= 1
            for a in live:
                mine = [t for t in unfinished if self.can_see_task(a.name, t)]
                if not mine:
                    continue
                listing = "; ".join(
                    f"#{t.id} {t.title} ({'exchange, ' if t.board == EXCHANGE else ''}{t.status}"
                    f"{', ' + t.owner if t.owner else ''})"
                    for t in mine
                )
                self.deliver(
                    MESH, a.name,
                    f"Everyone is idle, but these tasks are not done: {listing}. "
                    "Finish what you own, claim what is open, or complete obsolete tasks with a short note.",
                )
            if any(a.inbox for a in live):
                return
        why = "after budget wrap-up" if self.wrapping_up else "all agents idle, no messages in flight"
        self.stop(f"quiescent: {why}")

    def stop(self, reason: str) -> None:
        if self.done.is_set():
            return
        self.stop_reason = reason
        self.done.set()
        self.log.emit("stop", MESH, reason=reason)
        for a in self.agents.values():
            a.wake.set()


def _norm(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title.lower()).split())
