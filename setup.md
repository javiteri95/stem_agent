# How to build it

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