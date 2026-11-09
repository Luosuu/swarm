"""Event log: every message, tool call and task change, as JSON lines.

The log is the research output: it shows who talked to whom, when, and what
structure emerged. Load the .jsonl file into pandas or jq for analysis.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

_COLORS = ["\033[36m", "\033[33m", "\033[35m", "\033[32m", "\033[34m", "\033[31m", "\033[96m", "\033[93m"]
_RESET, _DIM = "\033[0m", "\033[2m"


def _short(text: Any, n: int = 160) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 1] + "…"


class EventLog:
    def __init__(self, path: Path | None = None, echo: bool = False, stream: TextIO | None = None):
        self.events: list[dict[str, Any]] = []
        self.path = path
        self.echo = echo
        self.stream = stream or sys.stdout
        self._t0 = time.monotonic()
        self._colors: dict[str, str] = {}
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")

    def emit(self, kind: str, agent: str, **data: Any) -> None:
        event = {"t": round(time.monotonic() - self._t0, 3), "kind": kind, "agent": agent, **data}
        self.events.append(event)
        if self._fh:
            self._fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        if self.echo:
            self._print(event)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    # ------------------------------------------------------------ pretty print
    def _color(self, name: str) -> str:
        if name == "mesh":
            return _DIM
        if name not in self._colors:
            self._colors[name] = _COLORS[len(self._colors) % len(_COLORS)]
        return self._colors[name]

    def _print(self, e: dict[str, Any]) -> None:
        kind, who = e["kind"], e["agent"]
        tag = f"{self._color(who)}[{e['t']:7.1f}s {who}]{_RESET}"
        if kind == "message":
            line = f"→ {e['to']} ({e['delivery']}): {_short(e['content'])}"
        elif kind == "say":
            line = f"says: {_short(e['text'])}"
        elif kind == "tool":
            has_own_event = {"send_message", "post_task", "claim_task", "release_task", "handover_task", "complete_task",
                             "write_file", "spawn_agent", "delegate", "report"}
            if e["name"] in has_own_event and '"error"' not in e["result"]:
                return  # a message/task/file event already shows it
            line = f"{_DIM}{e['name']}({_short(e['args'], 80)}) → {_short(e['result'], 100)}{_RESET}"
        elif kind == "task_posted":
            line = f"posted task #{e['task']}: {_short(e['title'])}"
        elif kind == "task_done":
            line = f"completed task #{e['task']}: {_short(e['result'])}"
        elif kind == "task_handover":
            line = f"handed task #{e['task']} to {e['to']}" + (f": {_short(e['note'], 100)}" if e.get("note") else "")
        elif kind in ("task_claimed", "task_released"):
            line = f"{kind.split('_')[1]} task #{e['task']}"
        elif kind == "spawn":
            if e.get("parent"):
                line = f"started as sub-agent of {e['parent']}" + (f" ({e['role']})" if e.get("role") else "")
            else:
                line = f"joined as {e.get('role') or 'generalist'}" + (f" (spawned by {e['spawned_by']})" if e.get("spawned_by") else "")
        elif kind == "delegate":
            line = f"{'follow-up to' if e.get('follow_up') else 'delegated to'} {e['child']}: {_short(e['task'], 120)}"
        elif kind == "report":
            line = f"reported to {e['parent']}{' (auto)' if e.get('auto') else ''}: {_short(e['result'], 120)}"
        elif kind in ("idle", "wake", "usage"):
            return
        else:
            rest = {k: v for k, v in e.items() if k not in ("t", "kind", "agent")}
            line = f"{kind} {_short(rest)}"
        print(f"{tag} {line}", file=self.stream, flush=True)
