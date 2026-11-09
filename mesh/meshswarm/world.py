"""The shared world: agent directory, mailboxes, task board, workspace, budgets.

Everything runs on one asyncio event loop, so each method here is atomic with
respect to other agents (no locks needed). The world never decides *what*
agents do; it only delivers messages, keeps the shared state and enforces
limits.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from .log import EventLog
from .pricing import Pricing, Usage

if TYPE_CHECKING:
    from .agent import MeshAgent
    from .model import Model

MESH = "mesh"  # sender name for system notices


class RunStopped(Exception):
    """Raised inside an agent when the run has ended."""


@dataclass
class Limits:
    max_agents: int = 8
    max_steps_per_agent: int = 60  # model calls per agent over the whole run
    max_total_tokens: int = 2_000_000  # all tokens, cached included. Soft: agents wrap up; hard stop at 1.25x
    max_cost_usd: float | None = None  # same soft/hard rule, in dollars (needs pricing)
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
    status: str = "open"  # open | claimed | done
    owner: str | None = None
    result: str | None = None
    history: list[str] = field(default_factory=list)  # handovers, e.g. "agent-1 -> agent-3: please verify"


@dataclass
class World:
    workspace: Path
    model: "Model"
    limits: Limits = field(default_factory=Limits)
    log: EventLog = field(default_factory=EventLog)
    pricing: Pricing | None = None

    def __post_init__(self) -> None:
        if self.limits.max_cost_usd is not None and self.pricing is None:
            raise ValueError("max_cost_usd needs pricing for the model")
        self.workspace = Path(self.workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.agents: dict[str, MeshAgent] = {}
        self.tasks: dict[int, Task] = {}
        self._next_task_id = 1
        self.usage = Usage()
        self.nudges_left = self.limits.nudges
        self.wrapping_up = False
        self.done = asyncio.Event()
        self.stop_reason: str | None = None
        self._call_slots = asyncio.Semaphore(self.limits.max_concurrent_calls)
        self._runners: list[asyncio.Task] = []

    # ---------------------------------------------------------------- agents
    def add_agent(self, name: str, role: str = "", instructions: str = "", spawned_by: str | None = None) -> "MeshAgent":
        from .agent import MeshAgent

        if name in self.agents or name in (MESH, "all"):
            raise ValueError(f"agent name {name!r} is taken")
        if len(self.agents) >= self.limits.max_agents:
            raise ValueError(f"agent limit reached ({self.limits.max_agents})")
        agent = MeshAgent(self, name, role, instructions)
        self.agents[name] = agent
        self._runners.append(asyncio.ensure_future(agent.run()))
        self.log.emit("spawn", name, role=role, spawned_by=spawned_by)
        return agent

    def roster(self) -> list[dict[str, Any]]:
        return [
            {
                "name": a.name,
                "role": a.role,
                "status": a.status,
                "tasks": [t.id for t in self.tasks.values() if t.owner == a.name and t.status == "claimed"],
            }
            for a in self.agents.values()
        ]

    # -------------------------------------------------------------- messages
    def deliver(self, sender: str, to: str, content: str) -> dict[str, Any]:
        """Put a message in an inbox. Idle targets are woken ("delivered"),
        busy targets see it after their current step ("queued")."""
        target = self.agents.get(to)
        if target is None:
            return {"to": to, "error": f"no agent named {to!r}"}
        if target.status == "stopped":
            return {"to": to, "error": f"{to} has stopped (out of budget)"}
        status = "queued" if target.status == "working" else "delivered"
        target.inbox.append((sender, content))
        target.status = "working"
        target.wake.set()
        self.log.emit("message", sender, to=to, content=content, delivery=status)
        return {"to": to, "status": status}

    def broadcast(self, sender: str, content: str) -> list[dict[str, Any]]:
        return [self.deliver(sender, name, content) for name in list(self.agents) if name != sender]

    # ------------------------------------------------------------ task board
    def post_task(self, by: str, title: str, description: str = "") -> Task:
        task = Task(self._next_task_id, title, description, by)
        self.tasks[task.id] = task
        self._next_task_id += 1
        self.log.emit("task_posted", by, task=task.id, title=title)
        return task

    def claim_task(self, by: str, task_id: int) -> dict[str, Any]:
        task = self.tasks.get(task_id)
        if task is None:
            return {"error": f"no task #{task_id}"}
        if task.status != "open":
            return {"error": f"task #{task_id} is {task.status}" + (f" by {task.owner}" if task.owner else "")}
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
        """Pass a claimed task to another agent, who becomes its owner and is told why."""
        task = self.tasks.get(task_id)
        if task is None:
            return {"error": f"no task #{task_id}"}
        if task.status != "claimed" or task.owner != by:
            return {"error": f"you do not hold task #{task_id}" + (f" (owner: {task.owner})" if task.owner else "")}
        target = self.agents.get(to)
        if target is None or to == by:
            return {"error": f"cannot hand over to {to!r}"}
        if target.status == "stopped":
            return {"error": f"{to} has stopped (out of budget)"}
        task.owner = to
        task.history.append(f"{by} -> {to}" + (f": {note}" if note else ""))
        self.log.emit("task_handover", by, task=task_id, to=to, note=note)
        msg = f"{by} handed you task #{task_id} ({task.title}). You now own it; complete it when done."
        self.deliver(MESH, to, msg + (f" Note from {by}: {note}" if note else ""))
        return {"ok": True, "task": task_id, "owner": to}

    def complete_task(self, by: str, task_id: int, result: str) -> dict[str, Any]:
        task = self.tasks.get(task_id)
        if task is None:
            return {"error": f"no task #{task_id}"}
        if task.status == "done":
            return {"error": f"task #{task_id} is already done"}
        if task.status == "claimed" and task.owner != by:
            return {"error": f"task #{task_id} is claimed by {task.owner}; they can hand it over to you with handover_task"}
        task.status, task.owner, task.result = "done", by, result
        self.log.emit("task_done", by, task=task_id, result=result)
        return {"ok": True}

    def tasks_posted_by_others_since(self, agent: str, seen_id: int) -> list[Task]:
        return [t for t in self.tasks.values() if t.id > seen_id and t.posted_by not in (agent, MESH)]

    def similar_task(self, title: str) -> Task | None:
        """An unfinished task whose title is nearly the same, if any."""
        norm = _norm(title)
        for t in self.tasks.values():
            if t.status == "done" or t.title == "GOAL":
                continue
            other = _norm(t.title)
            if norm == other or SequenceMatcher(None, norm, other).ratio() >= 0.85:
                return t
        return None

    def board(self) -> list[dict[str, Any]]:
        return [asdict(t) for t in self.tasks.values()]

    # ------------------------------------------------------------- workspace
    def resolve(self, path: str) -> Path:
        p = (self.workspace / path).resolve()
        if p != self.workspace and self.workspace not in p.parents:
            raise PermissionError(f"{path!r} is outside the shared workspace")
        if ".mesh" in p.relative_to(self.workspace).parts:
            raise PermissionError("the .mesh folder is reserved for run logs")
        return p

    # ---------------------------------------------------------- model + stop
    @property
    def tokens_used(self) -> int:
        return self.usage.total

    @property
    def cost_usd(self) -> float | None:
        return self.pricing.cost(self.usage) if self.pricing else None

    def _budget_used(self) -> tuple[float, str]:
        """Largest fraction of any budget used so far, and which budget it is."""
        used = [(self.tokens_used / self.limits.max_total_tokens, f"token budget ({self.tokens_used} tokens)")]
        if self.limits.max_cost_usd is not None and self.cost_usd is not None:
            used.append((self.cost_usd / self.limits.max_cost_usd, f"cost budget (${self.cost_usd:.4f})"))
        return max(used)

    async def call_model(self, agent: "MeshAgent", messages: list[dict], tools: list[dict] | None) -> dict:
        async with self._call_slots:
            if self.done.is_set():
                raise RunStopped
            msg, usage = await self.model.complete(messages, tools, agent=agent.name)
        self.usage = self.usage + usage
        agent.usage = agent.usage + usage
        cost = self.pricing.cost(usage) if self.pricing else None
        self.log.emit("usage", agent.name, **usage.as_dict(), cost=cost)
        fraction, which = self._budget_used()
        if fraction >= 1.25:
            self.stop(f"{which} hard limit reached")
        elif fraction >= 1 and not self.wrapping_up:
            self.wrapping_up = True
            self.log.emit("wrap_up", MESH, budget=which)
            notice = (
                f"The {which.split(' (')[0]} is used up. Do not start new work. In at most two steps, record what you have "
                "(write files, complete tasks you finished, release ones you did not), then call wait()."
            )
            for a in self.agents.values():
                if a.status == "working":  # idle agents are not woken just to be told to stop
                    self.deliver(MESH, a.name, notice)
        return msg

    def check_quiescent(self) -> None:
        """Stop when nobody is working and no message is in flight. Before
        stopping, nudge everyone once if tasks are still unfinished."""
        if self.done.is_set():
            return
        if any(a.status == "working" or a.inbox for a in self.agents.values()):
            return
        unfinished = [t for t in self.tasks.values() if t.status != "done"]
        live = [a for a in self.agents.values() if a.status != "stopped"]
        if unfinished and live and self.nudges_left > 0 and not self.wrapping_up:
            self.nudges_left -= 1
            listing = "; ".join(f"#{t.id} {t.title} ({t.status}{', ' + t.owner if t.owner else ''})" for t in unfinished)
            self.broadcast(
                MESH,
                f"Everyone is idle, but these tasks are not done: {listing}. "
                "Finish what you own, claim what is open, or complete obsolete tasks with a short note.",
            )
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
