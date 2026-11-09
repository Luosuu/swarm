# meshswarm

A leaderless mesh of LLM agents, after the "swarm" idea in Prime Intellect's
[On the nature of the swarm](https://www.primeintellect.ai/blog/on-the-nature-of-the-swarm),
with messaging ideas borrowed from [prime-agent](https://github.com/PrimeIntellect-ai/prime-agent).

- **Shared world, private context.** Each agent has its own context window. They share
  messages, a task board, and a workspace folder.
- **Agents choose whom to talk to and when.** Any agent can message any other by name, or everyone.
- **No predefined flow.** Agents post, claim and complete tasks, and spawn new peers as they see fit.
- **No main agent.** The goal is posted as task #1. The run ends when the mesh goes quiet,
  or when a budget runs out.

Standard library only (Python 3.10+). Works with any OpenAI-compatible `/chat/completions` endpoint
that supports tool calling (vLLM, SGLang, Ollama, LiteLLM, hosted gateways).

## Run it

```bash
export MESH_BASE_URL=http://localhost:8000/v1   # your endpoint
export MESH_MODEL=your-model-name
export MESH_API_KEY=...                         # if the endpoint needs one
python -m meshswarm "Write a short survey of X with sources" --agents researcher,researcher,writer,critic
```

### On Tinker

Tinker's [OpenAI-compatible API](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/) (beta)
works as is. `--model` takes a base model name or one of your `tinker://…/sampler_weights/…` checkpoints
(`GET /models` lists them). `thinkingmachines/Inkling-2-Small` and `thinkingmachines/Inkling-Small`
return clean tool calls. Model names are case-sensitive.
`Qwen/Qwen3-8B` spends many tokens reasoning before it answers, so give it a large `--max-completion-tokens`.

```bash
python -m meshswarm "your goal" --agents researcher,writer,critic \
  --base-url https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1 \
  --model thinkingmachines/Inkling-2-Small \
  --api-key-file ../.tinker_api_key --max-cost 0.10
```

`--api-key-file` reads a bare key or `NAME=key`, so the key never appears on the command line.
Keep the file git-ignored.

Each run writes a workspace (default `runs/<timestamp>/`) containing:

| Path | What |
|---|---|
| `FINAL.md` | the deliverable, if agents wrote one |
| `.mesh/events.jsonl` | every message, tool call, task change, compaction and stop, timestamped |
| `.mesh/report.md` | stop reason, tokens, agents, task board, who-talked-to-whom counts |
| `.mesh/board.json` | final task board |
| `.mesh/notes/<agent>.md` | each agent's compaction summaries |
| anything else | files agents wrote |

## Sub-agents: trees inside the mesh

Any mesh agent can hand a self-contained job to a **private sub-agent** of its own, which is the
"trees inside the mesh" idea:

| Tool | Who has it | What it does |
|---|---|---|
| `delegate(task, name?, role?, wait?)` | mesh agents and sub-agents above the depth limit | Starts a child with a fresh context and the task as its first message. Runs in the background by default; `wait=true` blocks and returns the report. Passing an existing child's `name` sends it a follow-up and keeps its context |
| `collect(name?, timeout_s?)` | same | Waits for one child, or for every child that is still working or has a report not yet collected |
| `report(result)` | sub-agents only | Sends a short result to the parent and ends the child's turn. Long output goes in files |

Sub-agents are private:
- They have only file tools, `report`, and (above the depth limit) `delegate`/`collect`. They have no task
  board, roster or `send_message`.
- Only their parent can reach them, and they can reply only to their parent. Other agents can't see them.
- They work in their parent's team folder, and in a multi-team run also `shared/`. Their cost counts
  toward the parent's team budget.

A report wakes the parent with a message unless the parent is already waiting in `collect()`. A child that
ends its turn, crashes or hits its step limit without calling `report` is reported automatically, so the
parent is never left waiting. The mesh doesn't go quiet while children are working. Children are kept
after reporting, so a follow-up reuses their context.

Limits: `Limits.max_depth` (default 2: children and grandchildren) and `Limits.max_subagents` (default 16
per run). The report draws each agent's tree with its cost including children, plus a Delegation section.
The analyzer adds `delegations`, `subagents` and `sub_cost_pct` columns.

On Inkling-2-Small, agents did not delegate on their own in a 6-file code review. They read every file
themselves, and both agents read the same files. When the goal suggested delegation, they started 9
sub-agents (one per file, plus quote checkers, including a grandchild), and 76% of the cost was spent in
sub-agents. That run cost more ($0.18 vs $0.10) and was slower than the run without delegation. Delegation
saves the parent's context, not money, at this scale.

## Several teams

Several teams can collaborate, each with its own swarm:

```bash
python -m meshswarm --teams examples/spec_and_guide.json \
  --base-url https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1 \
  --model thinkingmachines/Inkling-2-Small --api-key-file ../.tinker_api_key
```

Each team has:
- its own goal (a GOAL task on its board)
- a private task board and a private folder (`<workspace>/<team>/`, with its own `FINAL.md`)
- its own model and endpoint, prices and budget (`max_cost` / `max_tokens`). A team that hits its hard
  limit is halted; the other teams carry on.

Teams meet in two places:
- **The exchange board.** `post_task(board="exchange", for_team="<team>")` posts a request or deliverable
  that every team can see. Only the addressed team can claim it. When it is completed, the requester is
  told automatically.
- **`shared/`.** A folder every team can read and write.

`comm` sets who may message across teams:

| `comm` | Cross-team messages |
|---|---|
| `open` | anyone may message anyone |
| `liaisons` (default) | only liaisons, and only to other teams' liaisons. The first `liaisons` agents of each team are liaisons. Non-liaisons can't see other teams' non-liaisons |
| `exchange` | none. Teams use only the exchange board and `shared/` |

Messages from another team arrive marked `(external)`. Agents are told to treat them, and other teams'
files in `shared/`, as information rather than instructions. Team-board tasks can only be handed over
within the team; exchange tasks can go to any agent you may message.

The `--teams` file format (CLI flags such as `--model`, `--base-url`, `--api-key-file`, `--max-cost` and
`--price-*` are the defaults for every team):

```json
{
  "comm": "liaisons",
  "shared_goal": "Optional context every team sees.",
  "teams": [
    {"name": "spec", "goal": "...", "agents": "api designer,reviewer", "liaisons": 1, "max_cost": 0.05},
    {"name": "build", "goal": "...", "agents": 3, "model": "other/model", "base_url": "http://...", "api_key_file": "..."}
  ]
}
```

From Python, use `run_teams([TeamSpec(...), TeamSpec(...)], comm="liaisons")`. The report shows each
team's goal, cost, agents and board, plus the exchange board, `shared/`, and message counts within and
across teams. The analyzer adds `cross_team_msgs` and `exchange_tasks` columns.

## Cost

Every model call logs its input tokens, cached input tokens and output tokens. The report and the
analyzer show the run's cost in dollars and the share of input that was cached.

An agent resends its whole history on every call. Servers with prefix caching (such as Tinker) bill
that repeated prefix at a discount. On Inkling-2-Small, about 98% of an agent's input is cached, at $0.03
per million instead of $0.30. The raw token count therefore overstates cost several times over, so set the
budget in dollars with `--max-cost`. It follows the same rule as `--max-tokens`: at the limit, working
agents are told to wrap up, and the hard stop is at 1.25×.

Prices for Tinker's Inkling models are built into [pricing.py](meshswarm/pricing.py). For other models,
pass `--price-input`, `--price-cached` and `--price-output` (USD per million tokens). Without prices,
cost shows as unknown and `--max-cost` is refused.

Keep histories append-only to get the most from caching. Rewriting or trimming old messages breaks the
cached prefix from that point on.

Useful flags: `--agents 4` or `--agents a,b,c` (roles), `--max-agents`, `--max-steps` (model calls per agent),
`--max-tokens` and `--max-cost` (whole run), `--concurrency` (parallel model calls), `--compact-at`, `--timeout`,
`--temperature`, `--extra-body '{"top_p":0.9}'`, `--ca-file` for an HTTPS endpoint with a private CA,
`--quiet`. A goal can come from a file: `python -m meshswarm @goal.md`.

From Python:

```python
import asyncio
from meshswarm import ChatEndpoint, Limits, run_mesh

model = ChatEndpoint("http://localhost:8000/v1", "my-model")
result = asyncio.run(run_mesh("goal...", model, agents=["planner", "coder", "tester"], workspace="runs/x",
                              limits=Limits(max_total_tokens=500_000)))
print(result.stop_reason, result.final)
```

## How it works

```
meshswarm/
  world.py   shared state: teams, directory, inboxes, task boards and exchange, workspace sandbox,
             cross-team rules, per-team budgets, stop rule
  agent.py   one agent: own history, wake on message, step loop, steering, compaction
  tools.py   the 14 tools agents get
  model.py   ChatEndpoint (HTTP, stdlib) and ScriptedModel (tests)
  runner.py  run_mesh (one team) and run_teams (several): seeds goals, runs until done, writes the report
  log.py     JSONL event log + coloured console view
  analyze.py side-by-side table of runs (python -m meshswarm.analyze runs/* --word-limit 500 --csv out.csv)
```

**Output limit.** If a reply is cut off at the server's output limit (`finish_reason: length`), the
half-written tool call is dropped and the agent is told to work in smaller pieces (`write_file` with
`append=true`) or to delegate. On Tinker the default limit is 4,096 output tokens; raise it with
`--max-completion-tokens`.

**Agent loop.** An agent sleeps until a message arrives. It then reads its inbox and calls the model.
It runs any tool calls the model makes, then calls the model again, until the agent calls `wait()`.
A reply with no tool call gets one reminder per turn, because models often announce an action
("I'll verify now") and never take it. A second bare reply ends the turn anyway.
An agent that keeps re-reading the same shared state with nothing new arriving is put to sleep;
first-time reads never count.

**Message delivery** (from prime-agent). A message to an idle agent wakes it (receipt `delivered`).
A message to a busy agent is `queued`, and the agent sees it right after its current step, without
waiting for its turn to end ("steering").

**Tools.**
- Talking: `list_agents`, `send_message(to | "all")`, `peek_agent` (a bounded preview of another agent's recent activity).
- Task board: `read_board`, `post_task`, `claim_task` (atomic, so work isn't duplicated), `release_task`,
  `handover_task(task_id, to, note)`, `complete_task`. Only a task's owner can complete it. A handover makes
  another agent the owner, records the handover on the task, and messages the new owner.
  `post_task` is refused if peers have posted tasks since this agent last read the board, or if a near-identical
  unfinished task exists (`force=true` overrides). This prevents several agents posting the same plan at once.
- Workspace: `list_files`, `read_file`, `write_file`. The file tools return exact word and character counts,
  so agents can check limits instead of estimating them.
- Pacing: `wait(reason)` sleeps until a message arrives.
- Growth: `spawn_agent` creates a new peer, not a subordinate.

**Stopping without a leader.** When every agent is idle and no message is in flight, the mesh has
gone quiet. If tasks are still open at that point, everyone gets one nudge listing them. If the mesh
goes quiet again, the run stops. At `--max-tokens`, working agents are told to record their work and
wait; the hard stop is at 1.25× that budget. Other limits: model calls per agent, agent count, wall clock. A message limit per agent and a message size cap guard against endless
chatter, which pushes long content into files.

**Long contexts.** When an agent's history passes `--compact-at`, the agent summarizes itself.
Its history is replaced by that summary plus any unread messages, and the summary is saved under `.mesh/notes/`.

**Safety.** File tools are sandboxed to the workspace, and `.mesh/` is read-only to agents.
There is no shell tool, so agents cannot run code. Add one only inside a container.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use a scripted model and cover:
- waking, the quiet-mesh stop and the nudge, steering a busy agent
- spawning, token, dollar and step budgets with graceful wrap-up, usage parsing and cost arithmetic, a crashing agent, compaction
- `wait()`, the no-tool reminder, polling detection, stale-board and near-duplicate task refusal
- exclusive claims and handover rules, word counts, the workspace sandbox, bad tool calls
- teams: private boards and folders, `shared/`, the three `comm` rules, external labels, cross-team handover
  limits, an exchange request answered across teams, one team halted by budget while the other finishes, `--teams` config errors
- sub-agents: background and blocking delegation, collect, follow-ups that keep context, automatic reports
  (no report, crash, step limit), privacy, depth and count limits, team folder/model/budget, the stop rule
- truncated replies (output limit) are flagged to the agent instead of silently dropping the tool call
- an end-to-end CLI run against a local fake OpenAI-compatible server

## Research directions

- Swap the seed roles, agent count and model per agent, and compare `report.md` and `events.jsonl` across runs.
- Mixed models: give `World.model` a router, or let `spawn_agent` take a model name.
- Topology limits (prime-agent only lets an agent reach parent, siblings and children):
  restrict `World.deliver` and measure the effect.
- Add a `run_command` tool in a sandbox for coding tasks. Add persistent agents across runs.
