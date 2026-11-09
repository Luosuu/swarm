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
    goal = next((t for t in board if t["title"] == "GOAL"), {})
    final = run / "FINAL.md"
    words = len(final.read_text(encoding="utf-8").split()) if final.is_file() else None
    usage = [e for e in events if e["kind"] == "usage"]  # per-call usage (runs from before cost tracking have none)
    peer_msgs = [e for e in events if e["kind"] == "message" and e["agent"] != "mesh"]
    writers = sorted({e["agent"] for e in events if e["kind"] == "file_write" and e["path"] == "FINAL.md"})
    roles = [r or "-" for r in start.get("roles", [])]
    row = {
        "run": run.name,
        "roles": ",".join(roles) if any(r != "-" for r in roles) else f"{len(roles)} generalists",
        "stop": stop["reason"].split(":")[0].split(" (")[0],
        "seconds": round(stop["t"]),
        "tokens": sum(e["prompt"] + e["completion"] for e in usage) if usage else _report_tokens(run),
        "cached_pct": round(100 * sum(e["cached"] for e in usage) / max(sum(e["prompt"] for e in usage), 1)) if usage else None,
        "output_tokens": sum(e["completion"] for e in usage) if usage else None,
        "cost_usd": round(sum(e["cost"] for e in usage), 4) if usage and all(e.get("cost") is not None for e in usage) else None,
        "goal_done": goal.get("status") == "done",
        "final_words": words,
        "tasks_posted": kinds.get("task_posted", 1) - 1,
        "peer_messages": len(peer_msgs),
        "tool_calls": sum(tools.values()),
        "waits": tools.get("wait", 0),
        "reminders": kinds.get("no_tool_reminder", 0),
        "poll_stops": kinds.get("poll_stop", 0),
        "post_refusals": sum(1 for e in events if e["kind"] == "tool" and e["name"] == "post_task" and '"error"' in e["result"]),
        "final_writers": ",".join(writers) or "-",
        "errors": kinds.get("error", 0),
    }
    if word_limit is not None:
        row["within_limit"] = words is not None and words <= word_limit
        row["success"] = row["goal_done"] and row["within_limit"]
    return row


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
