"""Tools every agent gets. Each handler returns a JSON string for the model."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .world import EXCHANGE

if TYPE_CHECKING:
    from .agent import MeshAgent

READ_LIMIT = 20_000


def _fn(name: str, description: str, props: dict[str, Any] | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props or {}, "required": required or []},
        },
    }


S = {"type": "string"}
I = {"type": "integer"}

SPECS: list[dict] = [
    _fn("list_agents", "List the agents you can reach with role, status (working/idle/stopped), claimed tasks and (with teams) team and liaison flag."),
    _fn(
        "send_message",
        "Send a short message to one agent by name, or to everyone on your team with to='all'. An idle agent is woken "
        "by it; a busy agent reads it after its current step. Put long content in a file and send the path. "
        "With several teams, messages to other teams follow the mesh's communication rule.",
        {"to": S, "content": S},
        ["to", "content"],
    ),
    _fn(
        "peek_agent",
        "Read a short preview of a teammate's recent activity (what it said and messages it sent/received).",
        {"name": S, "n": I},
        ["name"],
    ),
    _fn(
        "wait",
        "Sleep until another agent messages you. Use this whenever you are waiting on someone or have nothing useful to do.",
        {"reason": S},
    ),
    _fn(
        "read_board",
        "Show task boards: every task with status, owner and result. With several teams, board='team' shows your "
        "team's board, board='exchange' the board shared by all teams, and the default shows both.",
        {"board": {"type": "string", "enum": ["all", "team", "exchange"]}},
    ),
    _fn(
        "post_task",
        "Add a task to a board for anyone to claim. Refused if a near-identical unfinished task exists; pass force=true "
        "only if yours is really different. With several teams, board='exchange' posts a cross-team request or "
        "deliverable, and for_team='<team>' addresses it to one team.",
        {"title": S, "description": S, "force": {"type": "boolean"},
         "board": {"type": "string", "enum": ["team", "exchange"]}, "for_team": S},
        ["title"],
    ),
    _fn("claim_task", "Claim an open task so nobody else works on it. Fails if someone else holds it.", {"task_id": I}, ["task_id"]),
    _fn("release_task", "Give back a task you claimed but will not finish.", {"task_id": I}, ["task_id"]),
    _fn(
        "handover_task",
        "Pass a task you hold to another agent who will finish it (for example, a reviewer who will complete the "
        "goal). They become the owner and get a message with your note.",
        {"task_id": I, "to": S, "note": S},
        ["task_id", "to"],
    ),
    _fn(
        "complete_task",
        "Mark a task done with a short result (put long output in a file and reference its path).",
        {"task_id": I, "result": S},
        ["task_id", "result"],
    ),
    _fn("list_files", "List files in your workspace (with several teams: your team's folder, and shared/ for all teams).", {"path": S}),
    _fn("read_file", "Read a text file from your workspace or shared/. Also returns its exact word and character counts.", {"path": S, "offset": I}, ["path"]),
    _fn(
        "write_file",
        "Write (or append to) a text file. Your teammates can read it; with several teams, put files for other teams "
        "under shared/. Returns the file's word count.",
        {"path": S, "content": S, "append": {"type": "boolean"}},
        ["path", "content"],
    ),
    _fn(
        "spawn_agent",
        "Create a new peer agent on your team with its own context. It starts with your instructions as its first "
        "message and joins as an equal; it does not report to you unless asked.",
        {"name": S, "role": S, "instructions": S},
        ["name", "role", "instructions"],
    ),
]

DELEGATE = _fn(
    "delegate",
    "Hand a self-contained job to a private sub-agent of yours. It gets a fresh context, your task text and file "
    "access, but cannot see the board or other agents; it works only for you and reports back a short result "
    "(long output goes in files). Use it to keep bulky work (reading, drafting, checking) out of your own context. "
    "By default it runs in the background: keep working and call collect() for the result, or you get a message "
    "when it reports. wait=true blocks and returns the result directly. Pass the name of an existing sub-agent to "
    "give it a follow-up task with its context intact.",
    {"task": S, "name": S, "role": S, "wait": {"type": "boolean"}},
    ["task"],
)
COLLECT = _fn(
    "collect",
    "Wait for your sub-agents' reports and return them: one sub-agent by name, or every sub-agent that is still "
    "working or has a report you have not collected yet (including reports that already woke you by message).",
    {"name": S, "timeout_s": I},
)
REPORT = _fn(
    "report",
    "Send your result to the agent that gave you the task, and finish. Keep it short; put long output in files and "
    "give the paths. If you could not finish, report what you did and what blocked you.",
    {"result": S},
    ["result"],
)

FILE_TOOLS = {"list_files", "read_file", "write_file"}
SUB_SPECS = [s for s in SPECS if s["function"]["name"] in FILE_TOOLS] + [REPORT]
SUB_SPECS_WITH_DELEGATE = SUB_SPECS + [DELEGATE, COLLECT]
SPECS = SPECS + [DELEGATE, COLLECT]


def specs_for(agent: "MeshAgent") -> list[dict]:
    if not agent.parent:
        return SPECS
    return SUB_SPECS_WITH_DELEGATE if agent.depth < agent.world.limits.max_depth else SUB_SPECS


def _ok(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


async def _list_agents(a: "MeshAgent") -> Any:
    roster = a.world.roster(a.name)
    if not a.world.multi_team:
        return roster
    others = sorted(t for t in a.world.teams if t != a.team)
    return {"agents": roster, "your_team": a.team, "other_teams": others, "comm_rule": a.world.comm}


async def _send_message(a: "MeshAgent", to: str, content: str) -> Any:
    lim = a.world.limits
    if len(content) > lim.max_message_chars:
        return {"error": f"message is {len(content)} chars; limit is {lim.max_message_chars}. Write it to a file and send the path."}
    if a.messages_sent >= lim.max_messages_per_agent:
        return {"error": f"you have used your message budget ({lim.max_messages_per_agent})"}
    to = a.world.qualify(a.name, to)
    if to == a.name:
        return {"error": "you cannot message yourself"}
    a.messages_sent += 1
    a.remember(f"sent to {to}: {content}")
    if to == "all":
        return {"receipts": a.world.broadcast(a.name, content)}
    return a.world.deliver(a.name, to, content)


async def _peek_agent(a: "MeshAgent", name: str, n: int = 6) -> Any:
    name = a.world.qualify(a.name, name)
    other = a.world.agents.get(name)
    if other is None or other.team != a.team:
        return {"error": f"no teammate named {name!r}"}
    n = max(1, min(int(n), 20))
    return {"name": name, "status": other.status, "recent": [s[:600] for s in list(other.recent)[-n:]]}


async def _read_board(a: "MeshAgent", board: str = "all") -> Any:
    if a.world.tasks:
        a.board_seen = max(a.world.tasks)
    return a.world.board(a.name, board)


async def _wait(a: "MeshAgent", reason: str = "") -> Any:
    a.wants_wait = True
    a.world.log.emit("wait", a.name, reason=reason)
    return {"ok": True, "note": "you will sleep until a message arrives"}


async def _post_task(a: "MeshAgent", title: str, description: str = "", force: bool = False,
                     board: str = "team", for_team: str | None = None) -> Any:
    where = EXCHANGE if board == EXCHANGE else a.team
    if not force:
        fresh = a.world.tasks_posted_by_others_since(a.name, a.board_seen)
        if fresh:
            a.board_seen = max(t.id for t in a.world.tasks.values())
            listing = "; ".join(f"#{t.id} {t.title!r} by {t.posted_by} ({t.status})" for t in fresh)
            return {
                "error": f"the board changed since you last read it. New tasks: {listing}. "
                "Build on these if they cover your plan; post again (or with force=true) only for work they miss."
            }
        dup = a.world.similar_task(title, where)
        if dup is not None:
            return {
                "error": f"near-identical task #{dup.id} already exists: {dup.title!r} ({dup.status}"
                + (f", {dup.owner}" if dup.owner else "") + "). Use it, or pass force=true if yours is different."
            }
    t = a.world.post_task(a.name, title, description, board=where, for_team=for_team or None)
    if a.board_seen == t.id - 1 or not a.world.tasks_posted_by_others_since(a.name, a.board_seen):
        a.board_seen = t.id
    return {"ok": True, "task_id": t.id}


async def _claim_task(a: "MeshAgent", task_id: int) -> Any:
    return a.world.claim_task(a.name, int(task_id))


async def _release_task(a: "MeshAgent", task_id: int) -> Any:
    return a.world.release_task(a.name, int(task_id))


async def _handover_task(a: "MeshAgent", task_id: int, to: str, note: str = "") -> Any:
    return a.world.handover_task(a.name, int(task_id), to, note)


async def _complete_task(a: "MeshAgent", task_id: int, result: str) -> Any:
    return a.world.complete_task(a.name, int(task_id), result)


async def _list_files(a: "MeshAgent", path: str = ".") -> Any:
    root = a.world.resolve(a.name, path)
    if not root.exists():
        return {"error": f"{path!r} does not exist"}
    files = [
        a.world.display_path(a.name, p) + ("/" if p.is_dir() else "")
        for p in sorted(root.rglob("*"))
        if ".mesh" not in p.parts
    ]
    if a.world.multi_team and path in (".", "", "./"):
        files.append("shared/  (visible to all teams; list it with path='shared')")
    return files[:500]


async def _read_file(a: "MeshAgent", path: str, offset: int = 0) -> Any:
    p = a.world.resolve(a.name, path)
    if not p.is_file():
        return {"error": f"{path!r} is not a file"}
    text = p.read_text(encoding="utf-8", errors="replace")
    offset = max(0, int(offset))
    chunk = text[offset : offset + READ_LIMIT]
    out: dict[str, Any] = {"path": path, "words": len(text.split()), "chars": len(text), "content": chunk}
    if offset + READ_LIMIT < len(text):
        out["truncated"] = f"{len(text)} chars total; call again with offset={offset + READ_LIMIT}"
    return out


async def _write_file(a: "MeshAgent", path: str, content: str, append: bool = False) -> Any:
    p = a.world.resolve(a.name, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a" if append else "w", encoding="utf-8") as fh:
        fh.write(content)
    a.world.log.emit("file_write", a.name, path=path, chars=len(content), append=append)
    total = p.read_text(encoding="utf-8", errors="replace")
    return {"ok": True, "path": path, "words": len(total.split()), "chars": len(total)}


async def _delegate(a: "MeshAgent", task: str, name: str | None = None, role: str = "", wait: bool = False,
                    timeout_s: int = 600) -> Any:
    w = a.world
    existing = w.agents.get(f"{a.name}:{name}") if name else None
    if existing is not None and existing.parent == a.name:
        if existing.status == "stopped":
            return {"error": f"{existing.name} has stopped; delegate to a new sub-agent"}
        child, follow_up = existing, True
    else:
        short = name or f"sub-{len(a.children()) + 1}"
        if not short.replace("-", "").replace("_", "").isalnum():
            return {"error": "sub-agent names may use letters, digits, - and _"}
        child, follow_up = w.add_agent(short, role, spawned_by=a.name, parent=a.name), False
    receipt = w.deliver(a.name, child.name, f"{'Follow-up task' if follow_up else 'Task'} from {a.name}: {task}")
    if "error" in receipt:  # never leave a parent waiting on a child that did not get its task
        return {"error": f"could not start {child.name}: {receipt['error']}"}
    child.start_task()
    w.log.emit("delegate", a.name, child=child.name, task=task, follow_up=follow_up or None)
    if wait:
        return {"child": child.name, **(await _wait_reports(a, [child], timeout_s))[child.name]}
    return {"ok": True, "child": child.name, "follow_up": follow_up,
            "note": "running in the background; call collect() for its report, or keep working"}


async def _wait_reports(a: "MeshAgent", children: list, timeout_s: int) -> dict[str, dict]:
    pending = [c for c in children if c.pending]
    a.collecting.update(c.name for c in pending)
    try:
        if pending:
            waits = [asyncio.ensure_future(c.report_ready.wait()) for c in pending]
            stop = asyncio.ensure_future(a.world.done.wait())
            await asyncio.wait([asyncio.gather(*waits), stop], timeout=max(1, timeout_s),
                               return_when=asyncio.FIRST_COMPLETED)
            for f in [*waits, stop]:
                f.cancel()
    finally:
        a.collecting.difference_update(c.name for c in pending)
    out = {}
    for c in children:
        if c.pending:
            out[c.name] = {"status": "still working"}
            continue
        out[c.name] = {"status": "reported", "report": c.last_report}
        c.report_unread = False
    return out


async def _collect(a: "MeshAgent", name: str | None = None, timeout_s: int = 600) -> Any:
    kids = a.children()
    if name:
        kids = [c for c in kids if c.name in (name, f"{a.name}:{name}")]
        if not kids:
            return {"error": f"you have no sub-agent named {name!r}"}
    else:
        kids = [c for c in kids if c.pending or c.report_unread]
    if not kids:
        return {"note": "no sub-agents are working and there are no unread reports"}
    return await _wait_reports(a, kids, timeout_s)


async def _report(a: "MeshAgent", result: str) -> Any:
    if not a.parent:
        return {"error": "only sub-agents report; you are a mesh agent"}
    a.finish_task(result)
    return {"ok": True, "note": "reported; you will sleep until your parent sends a follow-up"}


async def _spawn_agent(a: "MeshAgent", name: str, role: str, instructions: str) -> Any:
    try:
        new = a.world.add_agent(name.split("/")[-1], role, instructions, spawned_by=a.name, team=a.team)
    except ValueError as e:
        return {"error": str(e)}
    a.world.deliver(a.name, new.name, f"You were created by {a.name}. Your instructions: {instructions}")
    return {"ok": True, "name": new.name}


HANDLERS: dict[str, Callable[..., Awaitable[Any]]] = {
    "list_agents": _list_agents,
    "send_message": _send_message,
    "peek_agent": _peek_agent,
    "wait": _wait,
    "read_board": _read_board,
    "post_task": _post_task,
    "claim_task": _claim_task,
    "release_task": _release_task,
    "handover_task": _handover_task,
    "complete_task": _complete_task,
    "list_files": _list_files,
    "read_file": _read_file,
    "write_file": _write_file,
    "spawn_agent": _spawn_agent,
    "delegate": _delegate,
    "collect": _collect,
    "report": _report,
}


async def execute(agent: "MeshAgent", name: str, raw_args: str) -> str:
    allowed = {s["function"]["name"] for s in specs_for(agent)}
    handler = HANDLERS.get(name) if name in allowed else None
    if handler is None:
        return _ok({"error": f"unknown tool {name!r}; available: {sorted(allowed)}"})
    try:
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        return _ok({"error": f"bad arguments for {name}: {e}"})
    try:
        return _ok(await handler(agent, **args))
    except TypeError as e:
        return _ok({"error": f"bad arguments for {name}: {e}"})
    except (PermissionError, OSError, ValueError) as e:
        return _ok({"error": str(e)})
