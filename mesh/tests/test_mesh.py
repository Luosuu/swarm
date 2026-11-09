"""Behaviour tests with a scripted model (no endpoint needed).

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from meshswarm import Limits, ScriptedModel, World, call, run_mesh, say
from meshswarm.log import EventLog
from meshswarm.tools import execute


def step(messages: list[dict]) -> int:
    """How many times this agent has already answered."""
    return sum(1 for m in messages if m["role"] == "assistant")


def last_user(messages: list[dict]) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


def tool_results(messages: list[dict]) -> list[dict]:
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


def run(goal: str, policy, **kw):
    tmp = tempfile.mkdtemp()
    model = kw.pop("model", None) or ScriptedModel(policy)
    limits = kw.pop("limits", None) or Limits(timeout_s=10)
    return asyncio.run(run_mesh(goal, model, workspace=tmp, limits=limits, echo=False, **kw)), model


class MeshBehaviour(unittest.TestCase):
    def test_message_wakes_idle_peer_and_run_stops_when_quiet(self):
        def policy(agent, msgs):
            n = step(msgs)
            if agent == "agent-1":
                return [call("send_message", to="agent-2", content="please take the goal"), say("handed off")][min(n, 1)]
            if "please take the goal" in last_user(msgs) or n > 0:
                return [call("claim_task", task_id=1), call("write_file", path="FINAL.md", content="done"),
                        call("complete_task", task_id=1, result="see FINAL.md"), say("finished")][min(n, 3)]
            return say("waiting")

        r, _ = run("say hi", policy, agents=2)
        self.assertTrue(r.stop_reason.startswith("quiescent"), r.stop_reason)
        self.assertEqual(r.tasks[0]["status"], "done")
        self.assertEqual(r.tasks[0]["owner"], "agent-2")
        self.assertEqual(r.final, "done")
        self.assertTrue((r.workspace / ".mesh" / "report.md").is_file())
        self.assertTrue((r.workspace / ".mesh" / "events.jsonl").is_file())

    def test_idle_with_open_tasks_gets_one_nudge_then_stops(self):
        r, model = run("anything", lambda agent, msgs: say("nothing to do"), agents=2)
        self.assertTrue(r.stop_reason.startswith("quiescent"))
        self.assertEqual(r.tasks[0]["status"], "open")
        self.assertEqual(len(model.calls), 8)  # 2 agents x (seed + nudge) x (reply + no-tool reminder)
        self.assertTrue(any("not done" in last_user(m) for _, m in model.calls))

    def test_busy_agent_sees_message_between_steps(self):
        def policy(agent, msgs):
            n = step(msgs)
            if agent == "agent-2":
                return [call("send_message", to="agent-1", content="steer: stop"), say("sent")][min(n, 1)]
            if any("steer: stop" in str(m.get("content")) for m in msgs):
                return say("got steered")
            return call("list_agents")  # keeps working until steered

        model = ScriptedModel(policy, delay_s=0.02)
        r, _ = run("x", policy, agents=2, model=model, limits=Limits(timeout_s=10, max_steps_per_agent=20, nudges=0))
        events = r.world.log.events
        sent = next(e for e in events if e["kind"] == "message" and e["agent"] == "agent-2")
        self.assertEqual(sent["delivery"], "queued")
        self.assertLess(r.world.agents["agent-1"].steps, 20)

    def test_spawned_agent_joins_and_finishes(self):
        def policy(agent, msgs):
            n = step(msgs)
            if agent == "agent-1":
                return [call("spawn_agent", name="helper", role="worker", instructions="complete task 1"), say("spawned")][min(n, 1)]
            if agent == "helper":
                return [call("complete_task", task_id=1, result="ok"), say("done")][min(n, 1)]
            return say("idle")

        r, _ = run("x", policy, agents=1)
        self.assertIn("helper", [a["name"] for a in r.agents])
        self.assertEqual(r.tasks[0]["owner"], "helper")
        first = next(m for m in r.world.agents["helper"].history if m["role"] == "user")
        self.assertIn("You were created by agent-1", first["content"])

    def test_token_budget_stops_chatter(self):
        def policy(agent, msgs):
            peer = "agent-2" if agent == "agent-1" else "agent-1"
            return call("send_message", to=peer, content="ping")

        model = ScriptedModel(policy, tokens_per_call=1000)
        r, _ = run("x", policy, agents=2, model=model, limits=Limits(timeout_s=10, max_total_tokens=10_000, max_messages_per_agent=1000))
        self.assertIn("token budget", r.stop_reason)

    def test_step_budget_stops_agent(self):
        r, _ = run("x", lambda a, m: call("read_board"), agents=1, limits=Limits(timeout_s=10, max_steps_per_agent=5))
        self.assertEqual(r.agents[0]["status"], "stopped")
        self.assertEqual(r.agents[0]["steps"], 5)
        self.assertTrue(r.stop_reason.startswith("quiescent"))

    def test_handover_lets_the_agreed_verifier_finish_the_goal(self):
        # Run 3's setup: the drafter holds the GOAL, but the peers agreed the verifier completes it.
        def policy(agent, msgs):
            n = step(msgs)
            if agent == "agent-1":
                plan = [call("claim_task", task_id=1), call("write_file", path="FINAL.md", content="draft"),
                        call("handover_task", task_id=1, to="agent-2", note="verify word count, then complete"), call("wait")]
                return plan[min(n, len(plan) - 1)]
            if "handed you task #1" in last_user(msgs):
                return call("complete_task", task_id=1, result="verified: 1 word")
            return call("wait")

        r, _ = run("x", policy, agents=2)
        self.assertEqual((r.tasks[0]["status"], r.tasks[0]["owner"]), ("done", "agent-2"))
        self.assertIn("handovers: agent-1 -> agent-2", (r.workspace / ".mesh" / "report.md").read_text())

    def test_crashing_model_stops_only_that_agent(self):
        def policy(agent, msgs):
            if agent == "agent-1":
                raise RuntimeError("endpoint down")
            return [call("complete_task", task_id=1, result="ok"), say("done")][min(step(msgs), 1)]

        r, _ = run("x", policy, agents=2)
        self.assertEqual({a["name"]: a["status"] for a in r.agents}["agent-1"], "stopped")
        self.assertEqual(r.tasks[0]["status"], "done")
        self.assertTrue(any(e["kind"] == "error" for e in r.world.log.events))

    def test_compaction_replaces_history_with_summary(self):
        def policy(agent, msgs):
            if msgs[-1]["role"] == "user" and "compacted" not in msgs[-1]["content"] and "Summarize" in msgs[-1]["content"]:
                return say("SUMMARY: did things")
            if "SUMMARY" in msgs[1]["content"]:
                return say("done after compaction")
            return call("write_file", path=f"f{step(msgs)}.txt", content="x" * 400)

        r, _ = run("x", policy, agents=1, limits=Limits(timeout_s=10, compact_at_tokens=500, nudges=0))
        events = [e for e in r.world.log.events if e["kind"] == "compact"]
        self.assertTrue(events)
        self.assertTrue((r.workspace / ".mesh" / "notes" / "agent-1.md").is_file())


class Fixes(unittest.TestCase):
    """Regressions from the first real run (polling, duplicate plans, budget stop)."""

    def test_wait_tool_ends_turn(self):
        def policy(agent, msgs):
            return [call("wait", reason="nothing yet"), say("should not be reached")][min(step(msgs), 1)]

        r, model = run("x", policy, agents=1, limits=Limits(timeout_s=10, nudges=0))
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(r.stop_reason.startswith("quiescent"))

    def test_repeated_reads_put_agent_to_sleep(self):
        r, model = run("x", lambda a, m: call("read_board"), agents=1, limits=Limits(timeout_s=10, nudges=0))
        self.assertEqual(r.agents[0]["status"], "idle")  # slept instead of burning its 60 steps
        self.assertEqual(len(model.calls), 3)  # first read is free, then 2 repeats
        self.assertTrue(any(e["kind"] == "poll_stop" for e in r.world.log.events))

    def test_first_time_reads_are_not_polling(self):
        reads = [call("read_board"), call("list_agents"), call("list_files"), call("peek_agent", name="agent-1"),
                 call("complete_task", task_id=1, result="ok"), say("done")]
        r, _ = run("x", lambda a, m: reads[min(step(m), len(reads) - 1)], agents=1)
        self.assertEqual(r.tasks[0]["status"], "done")
        self.assertFalse(any(e["kind"] == "poll_stop" for e in r.world.log.events))

    def test_stale_board_blocks_duplicate_plan(self):
        # Both agents read the board first, then both post a plan with different wording.
        def policy(agent, msgs):
            n = step(msgs)
            if n == 0:
                return call("read_board")
            if n == 1:
                title = "Research tree vs P2P" if agent == "agent-1" else "Research notes: trees and meshes"
                return call("post_task", title=title)
            return say("ok")

        model = ScriptedModel(policy, delay_s=0.01)
        r, _ = run("x", policy, agents=2, model=model, limits=Limits(timeout_s=10, nudges=0))
        posted = [t for t in r.tasks if t["title"] != "GOAL"]
        self.assertEqual(len(posted), 1, posted)
        refusals = [e for e in r.world.log.events if e["kind"] == "tool" and "board changed" in e["result"]]
        self.assertEqual(len(refusals), 1)

    def test_similar_title_refused_unless_forced(self):
        async def go():
            w = World(Path(tempfile.mkdtemp()), ScriptedModel(lambda a, m: say("")), Limits(), EventLog())
            a = w.add_agent("a")
            await execute(a, "post_task", json.dumps({"title": "Draft FINAL.md explainer"}))
            out = json.loads(await execute(a, "post_task", json.dumps({"title": "Draft final.md explainer!"})))
            self.assertIn("near-identical", out["error"])
            out = json.loads(await execute(a, "post_task", json.dumps({"title": "Draft final.md explainer!", "force": True})))
            self.assertTrue(out["ok"])
            out = json.loads(await execute(a, "post_task", json.dumps({"title": "Review the draft"})))
            self.assertTrue(out["ok"])
            w.stop("test")

        asyncio.run(go())

    def test_announced_action_gets_a_reminder_and_happens(self):
        # Run 3's failure: "I'll verify now" with no tool call used to end the turn for good.
        def policy(agent, msgs):
            if "You replied without calling a tool" in last_user(msgs):
                return call("complete_task", task_id=1, result="verified")
            if any(m["role"] == "tool" for m in msgs):
                return call("wait")
            return say("I'll do the verification now.")

        r, _ = run("x", policy, agents=1)
        self.assertEqual(r.tasks[0]["status"], "done")
        self.assertTrue(any(e["kind"] == "no_tool_reminder" for e in r.world.log.events))

    def test_truncated_reply_is_flagged_not_lost_silently(self):
        # delegate-1's failure: replies cut off at the output limit carried unparseable tool calls.
        def policy(agent, msgs):
            if "hit the output length limit" in last_user(msgs):
                return call("complete_task", task_id=1, result="wrote it in parts")
            if any(m["role"] == "tool" for m in msgs):
                return call("wait")
            return {"role": "assistant", "content": "Here is the very long file: ...", "truncated": True}

        r, model = run("x", policy, agents=1)
        self.assertEqual(r.tasks[0]["status"], "done")
        self.assertEqual(len([e for e in r.world.log.events if e["kind"] == "truncated"]), 1)
        history = r.world.agents["agent-1"].history
        self.assertFalse(any("truncated" in m for m in history))  # the flag never reaches the API
        self.assertTrue(any(str(m.get("content")).endswith("[cut off]") for m in history))

    def test_file_tools_report_word_counts(self):
        async def go():
            w = World(Path(tempfile.mkdtemp()), ScriptedModel(lambda a, m: say("")), Limits(), EventLog())
            a = w.add_agent("a")
            out = json.loads(await execute(a, "write_file", json.dumps({"path": "d.md", "content": "one two three"})))
            self.assertEqual(out["words"], 3)
            out = json.loads(await execute(a, "write_file", json.dumps({"path": "d.md", "content": " four", "append": True})))
            self.assertEqual(out["words"], 4)
            out = json.loads(await execute(a, "read_file", json.dumps({"path": "d.md"})))
            self.assertEqual((out["words"], out["chars"]), (4, 18))
            w.stop("test")

        asyncio.run(go())

    def test_budget_wraps_up_gracefully(self):
        # The agent is mid-work when the soft budget hits; it should get a notice, finish, and the run ends quietly.
        def policy(agent, msgs):
            if not any("budget is used up" in str(m.get("content")) for m in msgs):
                return call("write_file", path=f"w{step(msgs)}.txt", content="x")
            already_completed = any("complete_task" in json.dumps(m.get("tool_calls", "")) for m in msgs)
            return say("wrapped up") if already_completed else call("complete_task", task_id=1, result="partial")

        model = ScriptedModel(policy, tokens_per_call=500)
        r, _ = run("x", policy, agents=1, model=model, limits=Limits(timeout_s=10, max_total_tokens=5000))
        self.assertEqual(r.stop_reason, "quiescent: after budget wrap-up")
        self.assertEqual(r.tasks[0]["status"], "done")
        self.assertLessEqual(r.tokens_used, 5000 * 1.25)


class SharedState(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def world(self) -> World:
        return World(self.tmp, ScriptedModel(lambda a, m: say("")), Limits(), EventLog())

    def test_claim_is_exclusive(self):
        async def go():
            w = self.world()
            t = w.post_task("a", "job")
            self.assertTrue(w.claim_task("a", t.id).get("ok"))
            self.assertIn("claimed by a", w.claim_task("b", t.id)["error"])
            self.assertIn("claimed by a", w.complete_task("b", t.id, "x")["error"])
            self.assertTrue(w.release_task("a", t.id)["ok"])
            self.assertTrue(w.claim_task("b", t.id)["ok"])

        asyncio.run(go())

    def test_handover_rules(self):
        async def go():
            w = self.world()
            a, b, c = w.add_agent("a"), w.add_agent("b"), w.add_agent("c")
            t = w.post_task("a", "job")
            self.assertIn("do not hold", w.handover_task("a", t.id, "b")["error"])  # not claimed yet
            w.claim_task("a", t.id)
            self.assertIn("do not hold", w.handover_task("b", t.id, "c")["error"])  # not the owner
            self.assertIn("cannot hand over", w.handover_task("a", t.id, "ghost")["error"])
            self.assertIn("cannot hand over", w.handover_task("a", t.id, "a")["error"])
            c.status = "stopped"
            self.assertIn("stopped", w.handover_task("a", t.id, "c")["error"])
            self.assertTrue(w.handover_task("a", t.id, "b", "please verify")["ok"])
            self.assertEqual((w.tasks[t.id].owner, w.tasks[t.id].status), ("b", "claimed"))
            self.assertEqual(w.tasks[t.id].history, ["a -> b: please verify"])
            self.assertIn("handed you task", b.inbox[-1][1])  # the new owner is told (and woken)
            self.assertIn("hand it over", w.complete_task("a", t.id, "x")["error"])  # old owner can't complete
            self.assertTrue(w.complete_task("b", t.id, "verified")["ok"])
            w.stop("test")

        asyncio.run(go())

    def test_workspace_is_sandboxed(self):
        async def go():
            w = self.world()
            agent = w.add_agent("a")
            for path in ["../escape.txt", "/etc/passwd", ".mesh/events.jsonl"]:
                out = json.loads(await execute(agent, "write_file", json.dumps({"path": path, "content": "x"})))
                self.assertIn("error", out, path)
            out = json.loads(await execute(agent, "write_file", json.dumps({"path": "sub/ok.txt", "content": "hi"})))
            self.assertTrue(out["ok"])
            out = json.loads(await execute(agent, "read_file", json.dumps({"path": "sub/ok.txt"})))
            self.assertEqual(out["content"], "hi")
            w.stop("test")

        asyncio.run(go())

    def test_bad_tool_calls_return_errors_not_crashes(self):
        async def go():
            w = self.world()
            agent = w.add_agent("a")
            self.assertIn("unknown tool", await execute(agent, "nope", "{}"))
            self.assertIn("bad arguments", await execute(agent, "send_message", "not json"))
            self.assertIn("bad arguments", await execute(agent, "send_message", '{"wrong": 1}'))
            long = json.dumps({"to": "all", "content": "x" * 10_000})
            self.assertIn("Write it to a file", await execute(agent, "send_message", long))
            self.assertIn("no agent named", await execute(agent, "send_message", '{"to": "ghost", "content": "hi"}'))
            w.stop("test")

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
