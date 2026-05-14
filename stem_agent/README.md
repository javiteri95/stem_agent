# Stem Agent

A self-differentiating agent that grows into a specialist for any task domain.
Given a domain name it automatically builds an evaluation question set, designs its own architecture, tests it against those questions, and iteratively improves until it converges on a specialised agent.

## Phases

| # | Name | What it does |
|---|---|---|
| 0 | **Ground Truth** | Gathers evidence from DuckDuckGo, arXiv, Semantic Scholar, and Wikipedia, then asks the LLM to generate 15 graded eval questions (`easy / medium / hard`). Result is cached in `eval_suite/<domain_slug>.json`. |
| 1 | **Sense** | Analyses the domain and extracts six architectural primitives (reasoning style, search strategy, synthesis approach, etc.). |
| 2 | **Hypothesize** | Designs an initial `AgentSpec` (v0) from the primitives. |
| 3 | **Differentiate** | Runs up to 8 iteration cycles: introspect → mutate → evaluate → checkpoint/rollback. Converges when improvement plateaus. |
| 4 | **Crystallize** | Writes the final artefacts: `final_agent.json`, `eval_results.json`, and `eval_by_tier.json`. |

## Setup

1. **Python 3.11+ required.**

2. **Install [uv](https://docs.astral.sh/uv/getting-started/installation/)** if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies:**
   ```bash
   cd stem_agent
   uv sync
   ```
   For Tavily live web search (optional, used in `multi_step_search` architecture):
   ```bash
   uv sync --extra search
   ```

4. **Configure environment variables** — copy `.env.example` to `.env` and fill in the values you need:

   ```
   # LLM (pick one provider)
   OPENAI_API_KEY=...        # or use OPENAI_KEY as an alias
   ANTHROPIC_API_KEY=...

   # Model override (optional — defaults to claude-3-5-sonnet-20241022)
   STEM_AGENT_MODEL=gpt-4o   # any LiteLLM model string
   MODEL=gpt-4o              # alternative alias

   # Optional integrations
   TAVILY_API_KEY=...
   SEMANTIC_SCHOLAR_API_KEY=...  # raises Phase 0 rate limits
   ```

5. **Run from any directory:**
   ```bash
   # Phase 0 auto-generates questions, then runs all phases
   uv run python -m stem_agent.main --domain "Code Review"

   # Supply your own question file (skips Phase 0)
   uv run python -m stem_agent.main --domain "Code Review" --questions path/to/questions.json

   # Resume from the best checkpoint (skips Phases 1 & 2)
   uv run python -m stem_agent.main --domain "Code Review" --resume
   ```
   All output files are always written inside the `stem_agent/` project directory regardless of where the command is run from.

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--domain` | `"Deep Research"` | Task domain to specialise for. Outputs are isolated under `outputs/<domain_slug>/`. |
| `--questions` | _(auto)_ | Path to a JSON evaluation questions file. When omitted, Phase 0 runs to auto-generate or reuse a cached set for the domain. |
| `--resume` | off | Skip Phases 1 & 2, load the best existing checkpoint, then continue the differentiation loop and update `final_agent.json`. |

## What to expect

- **Phase 0** — 1–3 minutes (network fetches + one LLM call).
- **Phases 1–4** — 15–25 minutes (up to 8 iterations × number of questions × ~4 LLM calls each).

Progress is logged to stdout. A `[Phase 0] WARNING` is printed if any network source fails; the run continues with whatever evidence was gathered.

## Cost estimate

~$2–6 USD depending on provider, model, architecture selected, and number of iterations.

## Model selection

Set `STEM_AGENT_MODEL` (or `MODEL`) to any [LiteLLM-supported model string](https://docs.litellm.ai/docs/providers):

| Provider | Example value |
|---|---|
| Anthropic (default) | `claude-3-5-sonnet-20241022` |
| OpenAI | `gpt-4o` |
| Google | `gemini/gemini-1.5-pro` |
| Azure OpenAI | `azure/gpt-4o` |

## Output files

Everything lands inside the `stem_agent/` project directory:

| Path | Description |
|---|---|
| `eval_suite/<domain_slug>.json` | 15 auto-generated eval questions for the domain (Phase 0 output; reused on subsequent runs) |
| `outputs/<domain_slug>/final_agent.json` | The crystallised `AgentSpec` the stem agent converged on |
| `outputs/<domain_slug>/eval_results.json` | Before/after score comparison across all four eval dimensions |
| `outputs/<domain_slug>/eval_by_tier.json` | Per-tier (easy / medium / hard) score breakdown for the final agent |
| `outputs/<domain_slug>/checkpoints/` | One JSON per checkpoint, including rolled-back regressions |
