"""One mesh agent: its own context, its own loop, woken by messages.

Loop: sleep until a message arrives -> read inbox -> call the model and run
its tool calls step by step (reading any newly arrived messages between
steps) -> when the model answers without tool calls, go idle again.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any

from .model import estimate_tokens
from .pricing import Usage
from .tools import SPECS, execute
from .world import MESH, RunStopped

if TYPE_CHECKING:
    from .world import World

SYSTEM_PROMPT = """You are {name}, one agent in a mesh of peer agents working toward a shared goal.
{role_line}
How the mesh works:
- There is no leader and no fixed workflow. You decide what to work on, whom to talk to, and when.
- Every agent has its own private context. What you share: messages, the task board, and files in the shared workspace.
- Messages from others appear as "[message from NAME]: ...". Notices from the system appear as "[message from mesh]: ...".

Working norms:
- Check the board and the roster before starting. Claim a task before working on it, so work is not duplicated.
- Before posting tasks, read the board again: a peer may have just posted the same plan. Build on their plan instead of posting your own.
- Split big work into tasks on the board; let others claim them. Spawn a new agent only when there is real parallel work nobody can take.
- Message a specific agent when you need something from them or have something they need. Keep messages short; put long content in files and send the path.
- Record results: complete_task with a short result, and write substantial output to files.
- Only a task's owner can complete it. If you agreed that someone else will finish a task you hold, hand it over with handover_task.
- The final deliverable for the goal goes in FINAL.md in the workspace, unless the goal says otherwise. Agree with peers on who assembles it.
- Act through tools. Saying you will do something does nothing; do it now with a tool call.
- Waiting: when you are waiting on someone or have nothing useful to do, call wait(). You sleep until a message arrives, and you are always woken when someone messages you.
- Check measurable requirements (such as word limits) with the counts the file tools return, not by estimating.
- Never check for updates by repeatedly reading the board, files or other agents. That wastes budget; you will be told when something changes.
"""

READ_ONLY = {"list_agents", "peek_agent", "read_board", "list_files", "read_file"}
POLL_LIMIT = 2

NO_TOOL_REMINDER = (
    f"[message from {MESH}]: You replied without calling a tool, which does nothing. If you said you would do "
    "something, do it now with a tool call. If you are waiting or finished, call wait()."
)

COMPACT_PROMPT = (
    "Your context is getting long and will be replaced by a summary you write now. Summarize for your future self: "
    "the goal, what you have done, what others are doing, tasks you own, promises you made, and the workspace paths "
    "that hold your work. Be concise and concrete. Do not call tools."
)


class MeshAgent:
    def __init__(self, world: "World", name: str, role: str = "", instructions: str = ""):
        self.world = world
        self.name = name
        self.role = role
        self.instructions = instructions
        self.status = "idle"  # idle | working | stopped
        self.inbox: list[tuple[str, str]] = []
        self.wake = asyncio.Event()
        self.history: list[dict[str, Any]] = []
        self.steps = 0
        self.messages_sent = 0
        self.usage = Usage()
        self.recent: deque[str] = deque(maxlen=20)  # what peek_agent shows
        self.wants_wait = False  # set by the wait() tool
        self.board_seen = 0  # highest task id this agent has seen (optimistic concurrency for post_task)
        self.poll_streak = 0  # repeated identical reads with nothing new in between
        self.seen_reads: set[tuple[str, str]] = set()

    # ---------------------------------------------------------------- helpers
    def system_prompt(self) -> str:
        role_line = f"Your role: {self.role}." if self.role else "You are a generalist; take whatever role helps most."
        prompt = SYSTEM_PROMPT.format(name=self.name, role_line=role_line)
        if self.instructions:
            prompt += f"\nStanding instructions: {self.instructions}\n"
        return prompt

    def remember(self, text: str) -> None:
        self.recent.append(text)

    def _drain_inbox(self) -> None:
        if not self.inbox:
            return
        batch, self.inbox = self.inbox, []
        for sender, content in batch:
            self.remember(f"from {sender}: {content}")
        text = "\n\n".join(f"[message from {s}]: {c}" for s, c in batch)
        self.history.append({"role": "user", "content": text})

    # ------------------------------------------------------------------- loop
    async def run(self) -> None:
        world = self.world
        try:
            while True:
                await self.wake.wait()
                self.wake.clear()
                if world.done.is_set():
                    return
                self.status = "working"
                self.seen_reads.clear()
                self.poll_streak = 0
                world.log.emit("wake", self.name)
                self._drain_inbox()
                await self._turn()
                if self.status == "working":
                    self.status = "idle"
                world.log.emit("idle", self.name)
                world.check_quiescent()
        except RunStopped:
            return
        except Exception as e:  # keep the mesh alive if one agent crashes
            self.status = "stopped"
            world.log.emit("error", self.name, error=f"{type(e).__name__}: {e}")
            world.check_quiescent()

    async def _turn(self) -> None:
        world, lim = self.world, self.world.limits
        reminded = False
        while not world.done.is_set():
            if self.steps >= lim.max_steps_per_agent:
                self.status = "stopped"
                world.log.emit("budget", self.name, reason=f"step limit {lim.max_steps_per_agent}")
                return
            await self._maybe_compact()
            msg = await world.call_model(self, self._messages(), SPECS)
            self.steps += 1
            self.history.append(msg)
            if msg.get("content"):
                self.remember(f"said: {msg['content']}")
                world.log.emit("say", self.name, text=msg["content"])
            calls = msg.get("tool_calls") or []
            if not calls:
                # Sleeping takes an explicit wait(). A bare reply often announces an
                # action ("I'll verify now") that never happens, so remind once.
                if self.inbox:
                    self._drain_inbox()
                    continue
                if reminded or world.wrapping_up:
                    return
                reminded = True
                world.log.emit("no_tool_reminder", self.name)
                self.history.append({"role": "user", "content": NO_TOOL_REMINDER})
                continue
            if world.done.is_set():  # the run ended while the model was thinking
                return
            for c in calls:
                fn = c["function"]
                result = await execute(self, fn["name"], fn.get("arguments", "{}"))
                world.log.emit("tool", self.name, name=fn["name"], args=fn.get("arguments", ""), result=result)
                self.history.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            got_messages = bool(self.inbox)
            self._drain_inbox()  # steering: messages that arrived during this step
            if self.wants_wait:
                self.wants_wait = False
                if not got_messages:
                    return  # sleep until the next message
            if self._is_polling(calls, got_messages):
                world.log.emit("poll_stop", self.name, streak=self.poll_streak)
                self.history.append({"role": "user", "content": f"[message from {MESH}]: You were only re-reading shared state with nothing new arriving, so you have been put to sleep. You will be woken when a message arrives."})
                self.poll_streak = 0
                self.seen_reads.clear()
                return

    def _is_polling(self, calls: list[dict], got_messages: bool) -> bool:
        """True once the agent has repeated an identical read POLL_LIMIT times
        with no new message and no action of its own in between: it is checking
        for updates instead of sleeping. First-time reads never count."""
        if got_messages or any(c["function"]["name"] not in READ_ONLY for c in calls):
            self.seen_reads.clear()
            self.poll_streak = 0
            return False
        for c in calls:
            sig = (c["function"]["name"], c["function"].get("arguments", ""))
            if sig in self.seen_reads:
                self.poll_streak += 1
            self.seen_reads.add(sig)
        return self.poll_streak >= POLL_LIMIT

    def _messages(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.system_prompt()}, *self.history]

    async def _maybe_compact(self) -> None:
        if estimate_tokens(self.history) < self.world.limits.compact_at_tokens:
            return
        # The summary request must follow a complete exchange, so leave any
        # trailing tool results in place: they are already answered.
        msg = await self.world.call_model(self, [*self._messages(), {"role": "user", "content": COMPACT_PROMPT}], None)
        summary = msg.get("content") or "(empty summary)"
        before = estimate_tokens(self.history)
        # Keep unanswered incoming messages verbatim after the summary.
        tail: list[dict[str, Any]] = []
        for m in reversed(self.history):
            if m["role"] != "user":
                break
            tail.insert(0, m)
        note = f"[message from {MESH}]: Your earlier context was compacted. Your summary:\n{summary}"
        self.history = [{"role": "user", "content": note}, *tail]
        self.world.log.emit("compact", self.name, tokens_before=before, summary=summary)
        path = self.world.workspace / ".mesh" / "notes" / f"{self.name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"\n\n## compaction at step {self.steps}\n\n{summary}\n")

    def __repr__(self) -> str:
        return f"MeshAgent({self.name!r}, {self.status}, steps={self.steps})"

