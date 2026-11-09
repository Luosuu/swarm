"""Summarize mesh runs side by side.

    python -m meshswarm.analyze runs/a runs/b ... [--word-limit 500] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def summarize(run: Path, word_limit: int | None = None) -> dict[str, Any]:
    events = [json.loads(line) for line in open(run / ".mesh" / "events.jsonl", encoding="utf-8")]
    kinds = Counter(e["kind"] for e in events)
    tools = Counter(e["name"] for e in events if e["kind"] == "tool")
    start = next(e for e in events if e["kind"] == "start")
    stop = next((e for e in events if e["kind"] == "stop"), {"reason": "?", "t": events[-1]["t"]})
    board = json.loads((run / ".mesh" / "board.json").read_text()) if (run / ".mesh" / "board.json").is_file() else []
    team_names = [t["name"] for t in start.get("teams", [])]
    goals = [t for t in board if t["title"] == "GOAL"]
    finals = [run / n / "FINAL.md" for n in team_names] if team_names else [run / "FINAL.md"]
    counts = [len(f.read_text(encoding="utf-8").split()) if f.is_file() else None for f in finals]
    words: Any = counts[0] if not team_names else ",".join(f"{n}:{c}" for n, c in zip(team_names, counts))
    usage = [e for e in events if e["kind"] == "usage"]  # per-call usage (runs from before cost tracking have none)
    peer_msgs = [e for e in events if e["kind"] == "message" and e["agent"] != "mesh"]
    writers = sorted({e["agent"] for e in events if e["kind"] == "file_write" and e["path"] == "FINAL.md"})
    roles = [r or "-" for r in start.get("roles", [])]
    if team_names:
        roles_text = " ".join(f"{t['name']}({len(t['roles'])})" for t in start["teams"]) + f" comm={start.get('comm')}"
    else:
        roles_text = ",".join(roles) if any(r != "-" for r in roles) else f"{len(roles)} generalists"
    row = {
        "run": run.name,
        "roles": roles_text,
        "stop": stop["reason"].split(":")[0].split(" (")[0],
        "seconds": round(stop["t"]),
        "tokens": sum(e["prompt"] + e["completion"] for e in usage) if usage else _report_tokens(run),
        "cached_pct": round(100 * sum(e["cached"] for e in usage) / max(sum(e["prompt"] for e in usage), 1)) if usage else None,
        "output_tokens": sum(e["completion"] for e in usage) if usage else None,
        "cost_usd": round(sum(e["cost"] for e in usage), 4) if usage and all(e.get("cost") is not None for e in usage) else None,
        "goal_done": bool(goals) and all(g["status"] == "done" for g in goals),
        "final_words": words,
        "tasks_posted": kinds.get("task_posted", 1) - 1,
        "peer_messages": len([e for e in peer_msgs if ":" not in e["agent"] and ":" not in e["to"]]),
        "delegations": kinds.get("delegate", 0),
        "subagents": sum(1 for e in events if e["kind"] == "spawn" and e.get("parent")),
        "sub_cost_pct": _sub_cost_pct(usage),
        "cross_team_msgs": sum(1 for e in peer_msgs if e.get("cross_team")),
        "exchange_tasks": sum(1 for t in board if t.get("board") == "exchange"),
        "tool_calls": sum(tools.values()),
        "waits": tools.get("wait", 0),
        "reminders": kinds.get("no_tool_reminder", 0),
        "truncated": kinds.get("truncated", 0),
        "poll_stops": kinds.get("poll_stop", 0),
        "post_refusals": sum(1 for e in events if e["kind"] == "tool" and e["name"] == "post_task" and '"error"' in e["result"]),
        "final_writers": ",".join(writers) or "-",
        "errors": kinds.get("error", 0),
    }
    if word_limit is not None:
        row["within_limit"] = all(c is not None and c <= word_limit for c in counts)
        row["success"] = row["goal_done"] and row["within_limit"]
    return row


def _sub_cost_pct(usage: list[dict[str, Any]]) -> int | None:
    """Share of the run's cost (or tokens, without prices) spent by sub-agents (names contain ':')."""
    if not usage:
        return None
    key = "cost" if all(e.get("cost") is not None for e in usage) else "prompt"
    total = sum(e[key] for e in usage) or 1
    return round(100 * sum(e[key] for e in usage if ":" in e["agent"]) / total)


def _report_tokens(run: Path) -> int | None:
    report = run / ".mesh" / "report.md"
    if report.is_file():
        for line in report.read_text().splitlines():
            if line.startswith("- Tokens used:"):
                return int(line.split(":")[1])
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="meshswarm.analyze", description=__doc__.strip().splitlines()[0])
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--word-limit", type=int, help="also check FINAL.md is within this many words")
    p.add_argument("--csv", type=Path, help="write the table to a CSV file")
    a = p.parse_args(argv)
    rows = [summarize(r, a.word_limit) for r in a.runs if (r / ".mesh" / "events.jsonl").is_file()]
    if not rows:
        print("no runs found", file=sys.stderr)
        return 1
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
