"""
runner.py — Executes a candidate AgentSpec against a single research question.

All four architecture types are implemented here. Web search via Tavily is
optional; if TAVILY_API_KEY is not set, multi_step_search falls back to
pure parametric knowledge across multiple rounds.
"""

import os
from stem_agent.core.agent_spec import AgentSpec
from stem_agent.core.llm import call_llm


def run_candidate_agent(spec: AgentSpec, question: str) -> str:
    """
    Dispatch to the right execution strategy based on spec.architecture_type.
    Returns the agent's answer as a plain string.
    """
    if spec.architecture_type == "linear_chain":
        return _run_linear_chain(spec, question)
    elif spec.architecture_type == "reflection_loop":
        return _run_reflection_loop(spec, question)
    elif spec.architecture_type == "multi_step_search":
        return _run_multi_step_search(spec, question)
    elif spec.architecture_type == "adversarial_qa":
        return _run_adversarial_qa(spec, question)
    else:
        raise ValueError(f"Unknown architecture_type: {spec.architecture_type}")


# ---------------------------------------------------------------------------
# Architecture implementations
# ---------------------------------------------------------------------------

def _run_linear_chain(spec: AgentSpec, question: str) -> str:
    """Single API call with the spec's system prompt. No iterations."""
    messages = [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": question},
    ]
    return call_llm(messages, max_tokens=4096)


def _run_reflection_loop(spec: AgentSpec, question: str) -> str:
    """
    Two calls: first generates an answer, second critiques it and produces
    a revised answer. Falls back to linear if self_critique is False.
    """
    if not spec.self_critique:
        return _run_linear_chain(spec, question)

    # Round 1 — initial answer
    messages = [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": question},
    ]
    initial_answer = call_llm(messages, max_tokens=4096)

    # Round 2 — critique and revise
    critique_messages = [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": initial_answer},
        {
            "role": "user",
            "content": (
                "Review your answer above. Identify any factual gaps, unclear reasoning, "
                "or missing perspectives. Then produce a revised, improved answer that "
                "addresses those weaknesses. Output only the revised answer."
            ),
        },
    ]
    return call_llm(critique_messages, max_tokens=1024)


def _run_multi_step_search(spec: AgentSpec, question: str) -> str:
    """
    Calls LLM spec.max_search_rounds times. Each round refines or deepens the
    answer. Uses Tavily web search if available; otherwise uses parametric knowledge.
    Synthesizes a final answer from all round outputs.
    """
    tavily_client = _get_tavily_client()
    rounds_output = []

    for round_num in range(1, spec.max_search_rounds + 1):
        context_block = ""

        # Optionally inject web search results
        if tavily_client and "web_search" in spec.tools:
            search_query = _derive_search_query(question, round_num, rounds_output)
            search_results = _run_tavily_search(tavily_client, search_query)
            if search_results:
                context_block = f"\n\nWeb search results for this round:\n{search_results}\n"

        prior_context = ""
        if rounds_output:
            prior_context = (
                "\n\nYour findings from previous rounds:\n"
                + "\n---\n".join(rounds_output)
                + "\n\nBuild on the above and go deeper."
            )

        messages = [
            {"role": "system", "content": spec.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Research question: {question}\n"
                    f"Round {round_num} of {spec.max_search_rounds}. "
                    f"Focus on {'breadth — identify all key sub-topics' if spec.search_strategy == 'breadth_first' else 'depth — explore one key aspect thoroughly' if spec.search_strategy == 'depth_first' else 'refining and filling gaps in prior findings'}."
                    f"{context_block}{prior_context}"
                ),
            },
        ]
        round_answer = call_llm(messages, max_tokens=4096)
        rounds_output.append(f"[Round {round_num}] {round_answer}")

    # Synthesis call
    synthesis_instruction = {
        "extractive": "Extract and combine the most important factual points from all rounds into a clear answer.",
        "abstractive": "Synthesize the findings from all rounds into a coherent, fluent answer in your own words.",
        "hybrid": "Combine key extracted facts with your own synthesis to produce a complete, well-structured answer.",
    }.get(spec.synthesis_strategy, "Synthesize the findings into a clear answer.")

    citation_instruction = {
        "inline": " Cite sources inline where relevant.",
        "footnote": " Include numbered footnotes for sources.",
        "none": "",
    }.get(spec.citation_style, "")

    synthesis_messages = [
        {"role": "system", "content": spec.system_prompt},
        {
            "role": "user",
            "content": (
                f"Research question: {question}\n\n"
                f"Here are the findings from {spec.max_search_rounds} research rounds:\n\n"
                + "\n\n".join(rounds_output)
                + f"\n\n{synthesis_instruction}{citation_instruction}"
            ),
        },
    ]
    return call_llm(synthesis_messages, max_tokens=8192)


def _run_adversarial_qa(spec: AgentSpec, question: str) -> str:
    """
    Three calls: one takes the 'for' position, one the 'counterpoint' position,
    a third synthesizes a balanced answer. Best for comparative/evaluative questions.
    """
    # Call 1 — supporting / affirmative angle
    for_messages = [
        {"role": "system", "content": spec.system_prompt},
        {
            "role": "user",
            "content": (
                f"Research question: {question}\n\n"
                "Present the strongest supporting arguments, evidence, and perspectives "
                "for the main thesis implied by this question. Be thorough and factual."
            ),
        },
    ]
    for_answer = call_llm(for_messages, max_tokens=4096)

    # Call 2 — counterpoint / critical angle
    against_messages = [
        {"role": "system", "content": spec.system_prompt},
        {
            "role": "user",
            "content": (
                f"Research question: {question}\n\n"
                "Present the strongest counterarguments, limitations, criticisms, and "
                "alternative perspectives relating to this question. Be thorough and factual."
            ),
        },
    ]
    against_answer = call_llm(against_messages, max_tokens=4096)

    # Call 3 — balanced synthesis
    synthesis_messages = [
        {"role": "system", "content": spec.system_prompt},
        {
            "role": "user",
            "content": (
                f"Research question: {question}\n\n"
                f"Supporting perspective:\n{for_answer}\n\n"
                f"Critical perspective:\n{against_answer}\n\n"
                "Synthesize both perspectives into a balanced, nuanced answer that "
                "acknowledges strengths and weaknesses of each side."
            ),
        },
    ]
    return call_llm(synthesis_messages, max_tokens=8192)


# ---------------------------------------------------------------------------
# Tavily helpers
# ---------------------------------------------------------------------------

def _get_tavily_client():
    """Return a Tavily client if TAVILY_API_KEY is set, else None."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None
    try:
        from tavily import TavilyClient
        return TavilyClient(api_key=api_key)
    except ImportError:
        return None


def _derive_search_query(question: str, round_num: int, prior_rounds: list[str]) -> str:
    """Derive a search query for the current round."""
    if round_num == 1 or not prior_rounds:
        return question
    # Ask the LLM to produce a focused follow-up query
    messages = [
        {
            "role": "user",
            "content": (
                f"Original question: {question}\n\n"
                f"Prior research findings summary:\n{prior_rounds[-1]}\n\n"
                "Write a short, specific web search query to fill the most important "
                "gap in the findings above. Output only the query, no explanation."
            ),
        }
    ]
    return call_llm(messages, max_tokens=64)


def _run_tavily_search(client, query: str) -> str:
    """Run a Tavily search and return formatted results string."""
    try:
        results = client.search(query=query, max_results=3)
        snippets = []
        for r in results.get("results", []):
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            snippets.append(f"[{title}] {content} (source: {url})")
        return "\n".join(snippets)
    except Exception:
        return ""
