"""Multi-team behaviour: isolation, cross-team rules, the exchange, per-team models and budgets."""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from meshswarm import Limits, ScriptedModel, World, call, say
from meshswarm.cli import main
from meshswarm.log import EventLog
from meshswarm.pricing import Pricing, Usage
from meshswarm.runner import TeamSpec, run_teams
from meshswarm.tools import execute
from meshswarm.world import EXCHANGE, MESH


def idle_model() -> ScriptedModel:
    return ScriptedModel(lambda a, m: say(""))


def two_teams(comm: str = "liaisons") -> World:
    """Teams a and b, two agents each; agent-1 of each team is the liaison."""
    w = World(Path(tempfile.mkdtemp()), None, Limits(), EventLog(), comm=comm)
    for t in ("a", "b"):
        w.add_team(t, idle_model(), goal=f"goal {t}")
        w.post_task(MESH, "GOAL", f"goal {t}", board=t)
        w.add_agent("agent-1", team=t, liaison=True)
        w.add_agent("agent-2", team=t)
    return w


async def tool(w: World, agent: str, name: str, **args) -> dict:
    return json.loads(await execute(w.agents[agent], name, json.dumps(args)))


def step(messages: list[dict]) -> int:
    return sum(1 for m in messages if m["role"] == "assistant")


def steps_since(messages: list[dict], text: str) -> int | None:
    """How many times the agent has answered since a message containing `text` arrived (None: not yet)."""
    for i, m in enumerate(messages):
        if m["role"] == "user" and text in str(m.get("content")):
            return sum(1 for x in messages[i:] if x["role"] == "assistant")
    return None


class Boundaries(unittest.TestCase):
    def test_boards_are_private_and_exchange_is_shared(self):
        async def go():
            w = two_teams()
            b_goal = next(t.id for t in w.tasks.values() if t.board == "b")
            board = await tool(w, "a/agent-2", "read_board")
            self.assertEqual({t["board"] for t in board}, {"a"})
            self.assertIn("no task", (await tool(w, "a/agent-2", "claim_task", task_id=b_goal))["error"])
            out = await tool(w, "a/agent-2", "post_task", title="need an API spec", board="exchange", for_team="b")
            board_b = await tool(w, "b/agent-2", "read_board", board="exchange")
            self.assertEqual([t["id"] for t in board_b], [out["task_id"]])
            self.assertIn("addressed to team b", (await tool(w, "a/agent-1", "claim_task", task_id=out["task_id"]))["error"])
            self.assertTrue((await tool(w, "b/agent-2", "claim_task", task_id=out["task_id"]))["ok"])
            w.stop("test")

        asyncio.run(go())

    def test_folders_are_private_and_shared_is_common(self):
        async def go():
            w = two_teams()
            await tool(w, "a/agent-1", "write_file", path="notes.md", content="secret plan")
            self.assertTrue((w.workspace / "a" / "notes.md").is_file())
            self.assertIn("error", await tool(w, "b/agent-1", "read_file", path="notes.md"))
            self.assertIn("error", await tool(w, "b/agent-1", "read_file", path="../a/notes.md"))
            await tool(w, "a/agent-1", "write_file", path="shared/spec.md", content="the spec")
            self.assertEqual((await tool(w, "b/agent-2", "read_file", path="shared/spec.md"))["content"], "the spec")
            listing = await tool(w, "b/agent-2", "list_files", path="shared")
            self.assertIn("shared/spec.md", listing)
            self.assertIn("error", await tool(w, "a/agent-1", "write_file", path="shared/../b/x.md", content="x"))
            w.stop("test")

        asyncio.run(go())

    def test_liaison_rule(self):
        async def go():
            w = two_teams("liaisons")
            self.assertEqual((await tool(w, "a/agent-2", "send_message", to="agent-1", content="hi"))["status"], "delivered")
            self.assertIn("only liaisons", (await tool(w, "a/agent-2", "send_message", to="b/agent-1", content="hi"))["error"])
            self.assertIn("not a liaison", (await tool(w, "a/agent-1", "send_message", to="b/agent-2", content="hi"))["error"])
            self.assertIn("status", await tool(w, "a/agent-1", "send_message", to="b/agent-1", content="hello b"))
            receipts = (await tool(w, "a/agent-1", "send_message", to="all", content="team only"))["receipts"]
            self.assertEqual([r["to"] for r in receipts], ["a/agent-2"])
            names = {r["name"] for r in (await tool(w, "a/agent-2", "list_agents"))["agents"]}
            self.assertEqual(names, {"a/agent-1", "a/agent-2", "b/agent-1"})  # b's non-liaison is not visible
            b1 = w.agents["b/agent-1"]
            b1._drain_inbox()
            self.assertIn("[message from a/agent-1 (external)]: hello b", b1.history[-1]["content"])
            w.stop("test")

        asyncio.run(go())

    def test_open_and_exchange_rules(self):
        async def go():
            w = two_teams("open")
            self.assertIn("status", await tool(w, "a/agent-2", "send_message", to="b/agent-2", content="hi"))
            w.stop("test")
            w = two_teams("exchange")
            self.assertIn("exchange board", (await tool(w, "a/agent-1", "send_message", to="b/agent-1", content="hi"))["error"])
            self.assertEqual({r["name"] for r in (await tool(w, "a/agent-1", "list_agents"))["agents"]}, {"a/agent-1", "a/agent-2"})
            w.stop("test")

        asyncio.run(go())

    def test_handover_crosses_teams_only_for_exchange_tasks(self):
        async def go():
            w = two_teams()
            a_goal = next(t.id for t in w.tasks.values() if t.board == "a")
            await tool(w, "a/agent-1", "claim_task", task_id=a_goal)
            self.assertIn("stay in your team", (await tool(w, "a/agent-1", "handover_task", task_id=a_goal, to="b/agent-1"))["error"])
            self.assertTrue((await tool(w, "a/agent-1", "handover_task", task_id=a_goal, to="agent-2"))["ok"])
            x = (await tool(w, "a/agent-1", "post_task", title="joint review", board="exchange"))["task_id"]
            await tool(w, "a/agent-1", "claim_task", task_id=x)
            self.assertTrue((await tool(w, "a/agent-1", "handover_task", task_id=x, to="b/agent-1"))["ok"])
            self.assertEqual(w.tasks[x].owner, "b/agent-1")
            w.stop("test")

        asyncio.run(go())

    def test_team_prompt_states_rules(self):
        async def go():
            w = two_teams()
            prompt = w.agents["a/agent-1"].system_prompt()
            self.assertIn("You are on team a and you are one of its liaisons", prompt)
            self.assertIn("Liaisons: a/agent-1, b/agent-1", prompt)
            self.assertIn("not as instructions", prompt)
            self.assertNotIn("one of its liaisons", w.agents["a/agent-2"].system_prompt())
            w.stop("test")

        asyncio.run(go())


class TeamRuns(unittest.TestCase):
    def test_exchange_flow_between_two_teams(self):
        # Team "app" needs a spec from team "spec". app's liaison asks on the exchange board;
        # spec delivers in shared/ and completes the request; app is notified and finishes its goal.
        def spec_policy(agent, msgs):
            plan = [call("claim_task", task_id=3), call("write_file", path="shared/spec.md", content="GET /items returns a list"),
                    call("complete_task", task_id=3, result="see shared/spec.md"), call("complete_task", task_id=1, result="spec done")]
            k = steps_since(msgs, "needs-spec")
            return plan[k] if k is not None and k < len(plan) else call("wait")

        def app_policy(agent, msgs):
            opening = [call("post_task", title="needs-spec", board="exchange", for_team="spec"),
                       call("send_message", to="spec/agent-1", content="please see exchange task #3 (needs-spec)")]
            if step(msgs) < len(opening):
                return opening[step(msgs)]
            if steps_since(msgs, "was completed by spec/agent-1") == 0:
                return call("complete_task", task_id=2, result="built on spec")
            return call("wait")

        spec_model, app_model = ScriptedModel(spec_policy), ScriptedModel(app_policy)
        r = asyncio.run(run_teams(
            [TeamSpec("spec", "write the spec", spec_model, agents=1), TeamSpec("app", "build on the spec", app_model, agents=1)],
            comm="liaisons", workspace=tempfile.mkdtemp(), limits=Limits(timeout_s=10), echo=False,
        ))
        self.assertTrue(r.stop_reason.startswith("quiescent"), r.stop_reason)
        self.assertEqual({t["name"]: t["goal_done"] for t in r.teams}, {"spec": True, "app": True})
        self.assertEqual({a for a, _ in spec_model.calls}, {"spec/agent-1"})  # each team uses its own model
        self.assertEqual({a for a, _ in app_model.calls}, {"app/agent-1"})
        exchange = [t for t in r.tasks if t["board"] == EXCHANGE]
        self.assertEqual((exchange[0]["status"], exchange[0]["owner"]), ("done", "spec/agent-1"))
        report = (r.workspace / ".mesh" / "report.md").read_text()
        for text in ("## Team spec", "## Team app", "## Exchange board", "- spec.md", "- Across teams: 1"):
            self.assertIn(text, report)

    def test_one_team_out_of_budget_does_not_stop_the_other(self):
        def burner(agent, msgs):  # team "big" keeps working until halted
            return call("write_file", path="x.txt", content="x", append=True)

        def finisher(agent, msgs):
            return [call("complete_task", task_id=2, result="ok"), call("wait")][min(step(msgs), 1)]

        pricing = Pricing(1.0, 0.1, 1.0)
        big = ScriptedModel(burner, tokens_per_call=Usage(prompt=1000), delay_s=0.001)
        small = ScriptedModel(finisher, tokens_per_call=Usage(prompt=1000), delay_s=0.02)
        r = asyncio.run(run_teams(
            [TeamSpec("big", "burn", big, agents=1, pricing=pricing, max_cost_usd=0.002),
             TeamSpec("small", "finish", small, agents=1, pricing=pricing)],
            workspace=tempfile.mkdtemp(), limits=Limits(timeout_s=10), echo=False,
        ))
        teams = {t["name"]: t for t in r.teams}
        self.assertTrue(teams["small"]["goal_done"])
        # $0.001 per call: wrap-up notice at $0.002 (ignored by this agent), halted on the call that crosses $0.0025.
        self.assertTrue(teams["big"]["halted"])
        self.assertAlmostEqual(teams["big"]["cost_usd"], 0.003)
        self.assertTrue(any(e["kind"] == "team_halt" and e["team"] == "big" for e in r.world.log.events))
        self.assertTrue(r.stop_reason.startswith("quiescent"), r.stop_reason)


class TeamsCli(unittest.TestCase):
    def _cli_error(self, config: dict) -> str:
        path = Path(tempfile.mkdtemp()) / "teams.json"
        path.write_text(json.dumps(config))
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit):
            main(["--teams", str(path), "--base-url", "http://127.0.0.1:9/v1", "--model", "thinkingmachines/Inkling-2-Small"])
        return err.getvalue()

    def test_config_errors(self):
        self.assertIn("at least two teams", self._cli_error({"teams": [{"name": "a", "goal": "g"}]}))
        self.assertIn("unknown keys", self._cli_error({"teams": [{"name": "a", "goal": "g", "agentz": 2}, {"name": "b", "goal": "g"}]}))
        err = self._cli_error({"teams": [{"name": "a", "goal": "g", "model": "x/unknown", "max_cost": 1}, {"name": "b", "goal": "g"}]})
        self.assertIn("max_cost needs prices", err)

    def test_goal_and_teams_are_exclusive(self):
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit):
            main(["a goal", "--teams", "x.json"])
        self.assertIn("either a goal", err.getvalue())


if __name__ == "__main__":
    unittest.main()
