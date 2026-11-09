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
  world.py   shared state: directory, inboxes, task board, workspace sandbox, budgets, stop rule
  agent.py   one agent: own history, wake on message, step loop, steering, compaction
  tools.py   the 14 tools agents get
  model.py   ChatEndpoint (HTTP, stdlib) and ScriptedModel (tests)
  runner.py  seeds the goal, runs until done, writes the report
  log.py     JSONL event log + coloured console view
  analyze.py side-by-side table of runs (python -m meshswarm.analyze runs/* --word-limit 500 --csv out.csv)
```

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
- an end-to-end CLI run against a local fake OpenAI-compatible server

## Research directions

- Swap the seed roles, agent count and model per agent, and compare `report.md` and `events.jsonl` across runs.
- Mixed models: give `World.model` a router, or let `spawn_agent` take a model name.
- Topology limits (prime-agent only lets an agent reach parent, siblings and children):
  restrict `World.deliver` and measure the effect.
- Add a `run_command` tool in a sandbox for coding tasks. Add persistent agents across runs.
