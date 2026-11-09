"""Tree-style sub-agents: delegate, collect, report, privacy, limits, budgets."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from meshswarm import Limits, ScriptedModel, World, call, run_mesh, say
from meshswarm.log import EventLog
from meshswarm.pricing import Pricing, Usage
from meshswarm.runner import TeamSpec, run_teams
from meshswarm.tools import execute, specs_for


def step(messages: list[dict]) -> int:
    return sum(1 for m in messages if m["role"] == "assistant")


def tool_results(messages: list[dict]) -> list[dict]:
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


def run(policy, agents=1, **kw):
    model = kw.pop("model", None) or ScriptedModel(policy)
    limits = kw.pop("limits", None) or Limits(timeout_s=10, nudges=0)
    r = asyncio.run(run_mesh("x", model, agents=agents, workspace=tempfile.mkdtemp(), limits=limits, echo=False, **kw))
    return r, model


def events(r, kind: str) -> list[dict]:
    return [e for e in r.world.log.events if e["kind"] == kind]


class Delegation(unittest.TestCase):
    def test_background_delegate_reports_back_by_message(self):
        def policy(agent, msgs):
            if agent == "agent-1":
                if step(msgs) == 0:
                    return call("delegate", task="write notes.md with 3 words", name="writer")
                reported = any("Sub-agent agent-1:writer reported" in str(m.get("content")) for m in msgs)
                completed = any("complete_task" in json.dumps(m.get("tool_calls", "")) for m in msgs)
                return call("complete_task", task_id=1, result="done via sub-agent") if reported and not completed else call("wait")
            # the sub-agent
            plan = [call("write_file", path="notes.md", content="one two three"), call("report", result="wrote notes.md (3 words)")]
            return plan[min(step(msgs), 1)]

        r, model = run(policy)
        self.assertEqual(r.tasks[0]["status"], "done")
        self.assertEqual((r.workspace / "notes.md").read_text(), "one two three")
        self.assertEqual([e["child"] for e in events(r, "delegate")], ["agent-1:writer"])
        self.assertEqual(events(r, "report")[0]["result"], "wrote notes.md (3 words)")
        sub = next(a for a in r.agents if a["name"] == "agent-1:writer")
        self.assertEqual(sub["parent"], "agent-1")
        report = (r.workspace / ".mesh" / "report.md").read_text()
        self.assertIn("  - agent-1:writer (sub-agent)", report)
        self.assertIn("## Delegation", report)

    def test_wait_true_returns_report_inline_without_a_message(self):
        def policy(agent, msgs):
            if agent == "agent-1":
                if step(msgs) == 0:
                    return call("delegate", task="say hi", wait=True)
                return call("complete_task", task_id=1, result=tool_results(msgs)[0]["report"]) if step(msgs) == 1 else call("wait")
            return call("report", result="hi")

        r, _ = run(policy)
        self.assertEqual(r.tasks[0]["result"], "hi")
        self.assertFalse([e for e in events(r, "message") if e["agent"] == "agent-1:sub-1"])  # no duplicate message

    def test_collect_waits_for_all_children(self):
        def policy(agent, msgs):
            if agent == "agent-1":
                plan = [call("delegate", task="a", name="a"), call("delegate", task="b", name="b"), call("collect")]
                if step(msgs) < 3:
                    return plan[step(msgs)]
                if step(msgs) == 3:
                    reports = tool_results(msgs)[-1]
                    return call("complete_task", task_id=1, result=",".join(sorted(v["report"] for v in reports.values())))
                return call("wait")
            return call("report", result=agent.split(":")[1].upper())

        model = ScriptedModel(policy, delay_s=0.01)
        r, _ = run(policy, model=model)
        self.assertEqual(r.tasks[0]["result"], "A,B")

    def test_follow_up_keeps_the_sub_agents_context(self):
        def policy(agent, msgs):
            if agent == "agent-1":
                plan = [call("delegate", task="remember 42", name="mem", wait=True),
                        call("delegate", task="what number?", name="mem", wait=True)]
                if step(msgs) < 2:
                    return plan[step(msgs)]
                if step(msgs) == 2:
                    return call("complete_task", task_id=1, result=tool_results(msgs)[-1]["report"])
                return call("wait")
            first_task = next(m["content"] for m in msgs if m["role"] == "user")
            return call("report", result="42" if "remember 42" in first_task else "?")

        r, _ = run(policy)
        self.assertEqual(r.tasks[0]["result"], "42")
        self.assertEqual([bool(e.get("follow_up")) for e in events(r, "delegate")], [False, True])
        self.assertEqual(sum(1 for e in events(r, "spawn") if e.get("parent")), 1)

    def test_grandchild_delegation_end_to_end(self):
        # delegate-3's bug: a sub-agent's task message to its own child was refused, so the grandchild never ran.
        def policy(agent, msgs):
            if agent == "agent-1":
                plan = [call("delegate", task="get the answer", name="mid", wait=True),
                        call("complete_task", task_id=1, result=tool_results(msgs)[-1].get("report", "") if tool_results(msgs) else ""),
                        call("wait")]
                return plan[min(step(msgs), 2)]
            if agent == "agent-1:mid":
                if step(msgs) == 0:
                    return call("delegate", task="compute", name="leaf", wait=True)
                return call("report", result="mid got " + tool_results(msgs)[-1]["report"])
            return call("report", result="42")

        r, _ = run(policy)
        self.assertEqual(r.tasks[0]["result"], "mid got 42")
        self.assertEqual({e["child"] for e in events(r, "delegate")}, {"agent-1:mid", "agent-1:mid:leaf"})
        self.assertTrue(r.stop_reason.startswith("quiescent"))
        self.assertIn("    - agent-1:mid:leaf", (r.workspace / ".mesh" / "report.md").read_text())

    def test_sub_agent_that_never_reports_is_reported_automatically(self):
        def policy(agent, msgs):
            if agent == "agent-1":
                return [call("delegate", task="do it", wait=True), call("wait")][min(step(msgs), 1)]
            return say("I think it is done.")  # no tool call, twice (after the reminder) -> turn ends

        r, _ = run(policy)
        rep = events(r, "report")[0]
        self.assertTrue(rep["auto"])
        self.assertIn("ended without calling report", rep["result"])
        self.assertIn("I think it is done.", rep["result"])

    def test_run_does_not_stop_while_a_sub_agent_works(self):
        def policy(agent, msgs):
            if agent == "agent-1":
                if step(msgs) == 0:
                    return call("delegate", task="slow job")
                if any("reported" in str(m.get("content")) for m in msgs) and step(msgs) < 3:
                    return call("complete_task", task_id=1, result="ok")
                return call("wait")
            return call("report", result="finally")

        slow = ScriptedModel(policy, delay_s=0.05)
        r, _ = run(policy, model=slow)
        self.assertTrue(r.stop_reason.startswith("quiescent"))
        self.assertEqual(r.tasks[0]["status"], "done")


class PrivacyAndLimits(unittest.TestCase):
    def world(self, **limits) -> World:
        return World(Path(tempfile.mkdtemp()), ScriptedModel(lambda a, m: say("")), Limits(**limits), EventLog())

    def test_sub_agents_are_private(self):
        async def go():
            w = self.world()
            a1, a2 = w.add_agent("agent-1"), w.add_agent("agent-2")
            child = w.add_agent("helper", parent="agent-1")
            self.assertEqual(child.name, "agent-1:helper")
            self.assertNotIn("agent-1:helper", [r["name"] for r in w.roster("agent-2")])
            self.assertIn("no agent named", w.deliver("agent-2", child.name, "hi")["error"])
            self.assertIn("only reply to its parent", w.deliver(child.name, "agent-2", "hi")["error"])
            self.assertEqual(w.deliver(child.name, "agent-1", "hi")["status"], "delivered")
            names = {s["function"]["name"] for s in specs_for(child)}
            self.assertEqual(names, {"list_files", "read_file", "write_file", "report", "delegate", "collect"})
            out = json.loads(await execute(child, "read_board", "{}"))
            self.assertIn("unknown tool", out["error"])
            self.assertIn("unknown tool", json.loads(await execute(a1, "report", '{"result": "x"}'))["error"])  # mesh agents can't report
            self.assertIn("private sub-agent working for agent-1", child.system_prompt())
            w.stop("test")

        asyncio.run(go())

    def test_depth_and_count_limits(self):
        async def go():
            w = self.world(max_depth=2, max_subagents=3)
            w.add_agent("agent-1")
            c1 = w.add_agent("c", parent="agent-1")
            c2 = w.add_agent("c", parent=c1.name)
            self.assertEqual((c1.depth, c2.depth), (1, 2))
            self.assertNotIn("delegate", {s["function"]["name"] for s in specs_for(c2)})  # deepest level
            with self.assertRaises(ValueError):
                w.add_agent("c", parent=c2.name)
            w.add_agent("d", parent="agent-1")
            out = json.loads(await execute(w.agents["agent-1"], "delegate", '{"task": "x"}'))
            self.assertIn("sub-agent limit", out["error"])
            w.stop("test")

        asyncio.run(go())

    def test_step_limit_on_a_sub_agent_reports_back(self):
        def policy(agent, msgs):
            if agent == "agent-1":
                return [call("delegate", task="loop", wait=True), call("wait")][min(step(msgs), 1)]
            return call("list_files", path=f"d{step(msgs)}")  # never reports

        r, _ = run(policy, limits=Limits(timeout_s=10, nudges=0, max_steps_per_agent=4))
        rep = events(r, "report")[0]
        self.assertTrue(rep["auto"])
        self.assertEqual(next(a for a in r.agents if a["name"] == "agent-1:sub-1")["status"], "stopped")


class SubAgentsInTeams(unittest.TestCase):
    def test_sub_agent_uses_team_folder_model_and_budget(self):
        def policy(agent, msgs):
            if agent == "a/agent-1":
                plan = [call("delegate", task="write it", name="w", wait=True),
                        call("complete_task", task_id=1, result="ok"), call("wait")]
                return plan[min(step(msgs), 2)]
            if agent == "a/agent-1:w":
                return [call("write_file", path="out.md", content="from child"), call("report", result="out.md")][min(step(msgs), 1)]
            return [call("complete_task", task_id=2, result="ok"), call("wait")][min(step(msgs), 1)]

        pricing = Pricing(1.0, 0.1, 1.0)
        a_model = ScriptedModel(policy, tokens_per_call=Usage(prompt=1000))
        b_model = ScriptedModel(policy, tokens_per_call=Usage(prompt=1000))
        r = asyncio.run(run_teams(
            [TeamSpec("a", "ga", a_model, agents=1, pricing=pricing), TeamSpec("b", "gb", b_model, agents=1, pricing=pricing)],
            workspace=tempfile.mkdtemp(), limits=Limits(timeout_s=10, nudges=0), echo=False,
        ))
        self.assertEqual((r.workspace / "a" / "out.md").read_text(), "from child")
        self.assertIn("a/agent-1:w", {a for a, _ in a_model.calls})  # child used team a's model
        team_a = next(t for t in r.teams if t["name"] == "a")
        self.assertAlmostEqual(team_a["cost_usd"], 5 * 0.001)  # 3 parent calls + 2 child calls, all billed to team a
        self.assertIn("(with sub-agents: $0.0050)", (r.workspace / ".mesh" / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
