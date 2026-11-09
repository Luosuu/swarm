"""Usage parsing, cost arithmetic and the dollar budget."""

from __future__ import annotations

import asyncio
import io
import tempfile
import unittest
from contextlib import redirect_stderr

from meshswarm import Limits, ScriptedModel, call, run_mesh
from meshswarm.cli import main
from meshswarm.model import parse_usage
from meshswarm.pricing import Pricing, Usage, lookup


class UsageAndCost(unittest.TestCase):
    def test_parse_openai_usage_with_cache(self):
        u = parse_usage(
            {"prompt_tokens": 2894, "completion_tokens": 60, "total_tokens": 2954, "prompt_tokens_details": {"cached_tokens": 2816}},
            [], {},
        )
        self.assertEqual((u.prompt, u.cached, u.uncached, u.completion, u.total), (2894, 2816, 78, 60, 2954))

    def test_parse_usage_without_details_or_at_all(self):
        self.assertEqual(parse_usage({"prompt_tokens": 10, "completion_tokens": 2}, [], {}).cached, 0)
        self.assertEqual(parse_usage({"total_tokens": 50}, [], {}).prompt, 50)
        est = parse_usage(None, [{"role": "user", "content": "x" * 400}], {"content": "y" * 40})
        self.assertGreater(est.prompt, 90)
        self.assertGreater(est.completion, 5)

    def test_cost_uses_cached_discount(self):
        p = Pricing(input=0.30, cached_input=0.03, output=1.00)
        u = Usage(prompt=1_000_000, cached=900_000, completion=100_000)
        # 100k uncached * 0.30 + 900k cached * 0.03 + 100k out * 1.00, per million
        self.assertAlmostEqual(p.cost(u), 0.03 + 0.027 + 0.10)
        self.assertAlmostEqual(p.cost(Usage(prompt=1_000_000)), 0.30)  # same tokens, no cache: 10x the input cost

    def test_known_prices(self):
        self.assertEqual(lookup("thinkingmachines/Inkling-2-Small"), Pricing(0.30, 0.03, 1.00))
        self.assertIsNone(lookup("some/unknown-model"))


class DollarBudget(unittest.TestCase):
    def test_cost_budget_wraps_up_before_token_budget(self):
        # Each call: 1k uncached + 9k cached + 1k output = $0.0003 + $0.00027 + $0.001 = $0.00157.
        # Soft limit $0.02 is crossed on call 13; the agent then waits (call 14, $0.022), under the $0.025 hard stop.
        per_call = Usage(prompt=10_000, cached=9_000, completion=1_000)
        pricing = Pricing(0.30, 0.03, 1.00)

        def policy(agent, msgs):
            if any("cost budget is used up" in str(m.get("content")) for m in msgs):
                return call("wait")
            return call("write_file", path="x.txt", content="x", append=True)

        model = ScriptedModel(policy, tokens_per_call=per_call)
        limits = Limits(timeout_s=10, max_total_tokens=10_000_000, max_cost_usd=0.02)
        r = asyncio.run(run_mesh("x", model, agents=1, workspace=tempfile.mkdtemp(), limits=limits, echo=False, pricing=pricing))
        self.assertEqual(r.stop_reason, "quiescent: after budget wrap-up")
        self.assertEqual(len(model.calls), 14)
        self.assertAlmostEqual(r.cost_usd, 14 * 0.00157)
        self.assertEqual(r.usage.cached, 9_000 * len(model.calls))
        report = (r.workspace / ".mesh" / "report.md").read_text()
        self.assertIn("- Cost: $0.0220", report)
        self.assertIn("(90% cached)", report)

    def test_cost_budget_requires_pricing(self):
        model = ScriptedModel(lambda a, m: call("wait"))
        with self.assertRaises(ValueError):
            asyncio.run(run_mesh("x", model, agents=1, workspace=tempfile.mkdtemp(), echo=False,
                                 limits=Limits(max_cost_usd=1.0)))

    def test_cli_refuses_max_cost_for_unknown_model(self):
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit):
            main(["goal", "--base-url", "http://127.0.0.1:9/v1", "--model", "some/unknown-model", "--max-cost", "1"])
        self.assertIn("--max-cost needs prices", err.getvalue())


if __name__ == "__main__":
    unittest.main()
