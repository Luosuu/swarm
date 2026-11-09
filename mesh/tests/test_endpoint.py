"""End-to-end: the CLI against a local fake OpenAI-compatible server."""

from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from meshswarm.cli import main

REQUESTS: list[dict] = []


def _reply(body: dict) -> dict:
    msgs = body["messages"]
    me = msgs[0]["content"].split(",")[0].removeprefix("You are ")
    n = sum(1 for m in msgs if m["role"] == "assistant")
    script = {
        "agent-1": [
            ("post_task", {"title": "write the haiku"}),
            ("send_message", {"to": "agent-2", "content": "task #2 is yours"}),
            None,
        ],
        "agent-2": [
            ("claim_task", {"task_id": 2}),
            ("write_file", {"path": "FINAL.md", "content": "an old silent pond"}),
            ("complete_task", {"task_id": 2, "result": "FINAL.md"}),
            ("complete_task", {"task_id": 1, "result": "FINAL.md"}),
            None,
        ],
    }[me]
    # agent-2 waits for agent-1's message before acting
    if me == "agent-2" and not any("task #2 is yours" in str(m.get("content")) for m in msgs):
        return {"role": "assistant", "content": "waiting for work"}
    acted = sum(1 for m in msgs if m["role"] == "assistant" and m.get("tool_calls"))
    item = script[min(acted, len(script) - 1)]
    if item is None:
        return {"role": "assistant", "content": f"{me} idle after {n} replies"}
    name, args = item
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": f"c{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}],
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        out = {"choices": [{"message": _reply(body)}], "usage": {"total_tokens": 50}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class EndpointEndToEnd(unittest.TestCase):
    def test_cli_runs_mesh_against_openai_compatible_server(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        ws = Path(tempfile.mkdtemp()) / "run"
        import os

        os.environ["TEST_MESH_KEY"] = "sk-test"
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main([
                    "write a haiku", "--agents", "2", "--workspace", str(ws),
                    "--base-url", f"http://127.0.0.1:{server.server_port}/v1", "--model", "my-model",
                    "--api-key-env", "TEST_MESH_KEY", "--timeout", "20", "--quiet",
                ])
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(code, 0)
        self.assertIn("stopped: quiescent", out.getvalue())
        self.assertEqual((ws / "FINAL.md").read_text(), "an old silent pond")
        board = json.loads((ws / ".mesh" / "board.json").read_text())
        self.assertEqual([t["status"] for t in board], ["done", "done"])
        r = REQUESTS[0]
        self.assertEqual(r["path"], "/v1/chat/completions")
        self.assertEqual(r["auth"], "Bearer sk-test")
        self.assertEqual(r["body"]["model"], "my-model")
        self.assertIn("tools", r["body"])
        self.assertNotIn("parallel_tool_calls", r["body"])
        report = (ws / ".mesh" / "report.md").read_text()
        self.assertIn("agent-1 → agent-2: 1", report)


if __name__ == "__main__":
    unittest.main()
