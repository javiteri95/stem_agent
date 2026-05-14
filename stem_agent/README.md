# Stem Agent

A self-differentiating agent that researches how Deep Research tasks are solved, designs its own architecture, tests it, and iteratively improves until it converges on a specialized research agent.

## Setup

1. **Python 3.11+ required.**

2. **Install [uv](https://docs.astral.sh/uv/getting-started/installation/)** if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```
   For Tavily web search support (optional):
   ```bash
   uv sync --extra search
   ```

4. **Set your LLM API key.** The agent uses [LiteLLM](https://docs.litellm.ai/), so any supported provider works.

   For Claude (default):
   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```
   For OpenAI:
   ```bash
   export OPENAI_API_KEY=your_key_here
   export STEM_AGENT_MODEL=gpt-4o
   ```
   For any other LiteLLM model, set `STEM_AGENT_MODEL` to the appropriate model string.

5. **(Optional) Tavily web search:**
   ```bash
   export TAVILY_API_KEY=your_key_here
   ```

6. **Run:**
   ```bash
   # default domain (Deep Research)
   uv run python -m stem_agent.main

   # custom domain
   uv run python -m stem_agent.main --domain "Code Review"

   # custom domain + custom questions file
   uv run python -m stem_agent.main --domain "Code Review" --questions path/to/questions.json

   # resume an existing run (skips Phases 1 & 2, continues from best checkpoint)
   uv run python -m stem_agent.main --domain "Deep Research" --resume
   ```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--domain` | `"Deep Research"` | Task domain to specialise for. Outputs are isolated under `outputs/<domain_slug>/`. |
| `--questions` | `eval_suite/questions.json` | Path to a JSON evaluation questions file. |
| `--resume` | off | Skip Phases 1 & 2, load the best existing checkpoint for the domain, then continue the differentiation loop and update `final_agent.json`. |

## What to expect

Full run takes roughly 15–25 minutes (up to 8 iterations × 10 questions × ~4 API calls per question).
Outputs land in `outputs/`. Progress logs to stdout.

## Cost estimate

~$2–5 USD depending on which architecture the agent selects and how many iterations run.

## Model selection

Set `STEM_AGENT_MODEL` to any [LiteLLM-supported model string](https://docs.litellm.ai/docs/providers):

| Provider | Example value |
|---|---|
| Anthropic (default) | `claude-3-5-sonnet-20241022` |
| OpenAI | `gpt-4o` |
| Google | `gemini/gemini-1.5-pro` |
| Azure OpenAI | `azure/gpt-4o` |

## Output files

Outputs are scoped per domain under `outputs/<domain_slug>/`:

| File | Description |
|---|---|
| `outputs/<domain>/final_agent.json` | The crystallized `AgentSpec` the stem agent converged on (overwritten on re-runs) |
| `outputs/<domain>/eval_results.json` | Before/after score comparison across all four eval dimensions |
| `outputs/<domain>/eval_by_tier.json` | Per-tier (easy / medium / hard) score breakdown for the final agent |
| `outputs/<domain>/checkpoints/` | One JSON per checkpoint (including rolled-back regressions) |
| `writeup/writeup.md` | Write-up to be completed after a full run |
