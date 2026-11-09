"""Tools every agent gets. Each handler returns a JSON string for the model."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Awaitable, Callable

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
    _fn("list_agents", "List every agent in the mesh with role, status (working/idle/stopped) and claimed tasks."),
    _fn(
        "send_message",
        "Send a short message to one agent by name, or to everyone with to='all'. An idle agent is woken by it; "
        "a busy agent reads it after its current step. Put long content in a file and send the path.",
        {"to": S, "content": S},
        ["to", "content"],
    ),
    _fn(
        "peek_agent",
        "Read a short preview of another agent's recent activity (what it said and messages it sent/received).",
        {"name": S, "n": I},
        ["name"],
    ),
    _fn(
        "wait",
        "Sleep until another agent messages you. Use this whenever you are waiting on someone or have nothing useful to do.",
        {"reason": S},
    ),
    _fn("read_board", "Show the shared task board: every task with status, owner and result."),
    _fn(
        "post_task",
        "Add a task to the shared board for anyone to claim. Refused if a near-identical unfinished task exists; "
        "pass force=true only if yours is really different.",
        {"title": S, "description": S, "force": {"type": "boolean"}},
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
    _fn("list_files", "List files in the shared workspace (path relative to the workspace root).", {"path": S}),
    _fn("read_file", "Read a text file from the shared workspace. Also returns its exact word and character counts.", {"path": S, "offset": I}, ["path"]),
    _fn(
        "write_file",
        "Write (or append to) a text file in the shared workspace. Every agent can read it. Returns the file's word count.",
        {"path": S, "content": S, "append": {"type": "boolean"}},
        ["path", "content"],
    ),
    _fn(
        "spawn_agent",
        "Create a new peer agent with its own context. It starts with your instructions as its first message and "
        "joins the mesh as an equal; it does not report to you unless asked.",
        {"name": S, "role": S, "instructions": S},
        ["name", "role", "instructions"],
    ),
]


def _ok(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


async def _list_agents(a: "MeshAgent") -> Any:
    return a.world.roster()


async def _send_message(a: "MeshAgent", to: str, content: str) -> Any:
    lim = a.world.limits
    if len(content) > lim.max_message_chars:
        return {"error": f"message is {len(content)} chars; limit is {lim.max_message_chars}. Write it to a file and send the path."}
    if a.messages_sent >= lim.max_messages_per_agent:
        return {"error": f"you have used your message budget ({lim.max_messages_per_agent})"}
    if to == a.name:
        return {"error": "you cannot message yourself"}
    a.messages_sent += 1
    a.remember(f"sent to {to}: {content}")
    if to == "all":
        return {"receipts": a.world.broadcast(a.name, content)}
    return a.world.deliver(a.name, to, content)


async def _peek_agent(a: "MeshAgent", name: str, n: int = 6) -> Any:
    other = a.world.agents.get(name)
    if other is None:
        return {"error": f"no agent named {name!r}"}
    n = max(1, min(int(n), 20))
    return {"name": name, "status": other.status, "recent": [s[:600] for s in list(other.recent)[-n:]]}


async def _read_board(a: "MeshAgent") -> Any:
    if a.world.tasks:
        a.board_seen = max(a.world.tasks)
    return a.world.board()


async def _wait(a: "MeshAgent", reason: str = "") -> Any:
    a.wants_wait = True
    a.world.log.emit("wait", a.name, reason=reason)
    return {"ok": True, "note": "you will sleep until a message arrives"}


async def _post_task(a: "MeshAgent", title: str, description: str = "", force: bool = False) -> Any:
    if not force:
        fresh = a.world.tasks_posted_by_others_since(a.name, a.board_seen)
        if fresh:
            a.board_seen = max(t.id for t in a.world.tasks.values())
            listing = "; ".join(f"#{t.id} {t.title!r} by {t.posted_by} ({t.status})" for t in fresh)
            return {
                "error": f"the board changed since you last read it. New tasks: {listing}. "
                "Build on these if they cover your plan; post again (or with force=true) only for work they miss."
            }
        dup = a.world.similar_task(title)
        if dup is not None:
            return {
                "error": f"near-identical task #{dup.id} already exists: {dup.title!r} ({dup.status}"
                + (f", {dup.owner}" if dup.owner else "") + "). Use it, or pass force=true if yours is different."
            }
    t = a.world.post_task(a.name, title, description)
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
    root = a.world.resolve(path)
    if not root.exists():
        return {"error": f"{path!r} does not exist"}
    files = [
        str(p.relative_to(a.world.workspace)) + ("/" if p.is_dir() else "")
        for p in sorted(root.rglob("*"))
        if ".mesh" not in p.relative_to(a.world.workspace).parts
    ]
    return files[:500]


async def _read_file(a: "MeshAgent", path: str, offset: int = 0) -> Any:
    p = a.world.resolve(path)
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
    p = a.world.resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a" if append else "w", encoding="utf-8") as fh:
        fh.write(content)
    a.world.log.emit("file_write", a.name, path=path, chars=len(content), append=append)
    total = p.read_text(encoding="utf-8", errors="replace")
    return {"ok": True, "path": path, "words": len(total.split()), "chars": len(total)}


async def _spawn_agent(a: "MeshAgent", name: str, role: str, instructions: str) -> Any:
    try:
        a.world.add_agent(name, role, instructions, spawned_by=a.name)
    except ValueError as e:
        return {"error": str(e)}
    a.world.deliver(a.name, name, f"You were created by {a.name}. Your instructions: {instructions}")
    return {"ok": True, "name": name}


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
}


async def execute(agent: "MeshAgent", name: str, raw_args: str) -> str:
    handler = HANDLERS.get(name)
    if handler is None:
        return _ok({"error": f"unknown tool {name!r}; available: {sorted(HANDLERS)}"})
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
