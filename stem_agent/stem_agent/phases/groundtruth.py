"""
groundtruth.py — Phase 0: Build or load the eval question set for a domain.

Resolution order
----------------
1. ``--questions <path>`` was provided → caller loads it directly; this module
   is never invoked.
2. ``eval_suite/<domain_slug>.json`` already exists → load and return it.
   Log that the cached ground truth is being reused.
3. Neither exists → gather evidence from multiple sources, ask the LLM to
   generate 15 questions, write to ``eval_suite/<domain_slug>.json`` atomically,
   and return the list.

Sources used (in order)
-----------------------
* DuckDuckGo full-text search — 3-5 targeted queries per domain.
* arXiv API — recent survey papers; extracts title + abstract.
* Semantic Scholar API — additional paper abstracts for hard questions.
* Wikipedia REST API — lead section of the most relevant article(s).
* LLM parametric knowledge — fills gaps; constrained to cited context.

Network failures are caught individually; if all sources fail the agent falls
back to parametric-only and logs a warning (does not crash).

Output schema (extends the base questions.json structure)
---------------------------------------------------------
.. code-block:: json

    {
      "id": "q01",
      "tier": "easy",
      "question": "...",
      "ground_truth": "...",
      "key_facts": ["...", "[disputed] ..."],
      "sources": [
        {"type": "web",       "title": "...", "url": "..."},
        {"type": "arxiv",     "title": "...", "url": "...", "abstract_snippet": "..."},
        {"type": "wikipedia", "title": "...", "url": "..."},
        {"type": "parametric","note": "LLM knowledge, no external source"}
      ]
    }
"""

import json
import os
import re
import tempfile
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from stem_agent.core.llm import call_llm
from stem_agent.core.paths import PROJECT_ROOT
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARXIV_API = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
HEADERS = {"User-Agent": "stem-agent/0.1 (research tool; contact: user@example.com)"}
REQUEST_TIMEOUT = 10  # seconds

# ---------------------------------------------------------------------------
# Source gathering helpers
# ---------------------------------------------------------------------------


def _ddg_search(query: str, max_results: int = 8) -> list[dict]:
    """Run a DuckDuckGo text search. Returns list of {title, snippet, url}."""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "type": "web",
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except Exception as exc:
        print(f"  [Phase 0] DuckDuckGo search failed for '{query}': {exc}")
        return []


def _arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv for survey/review papers. Returns list of source dicts."""
    try:
        params = {
            "search_query": f"all:{query} AND (ti:survey OR ti:review OR ti:overview)",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        resp = requests.get(ARXIV_API, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "xml")
        results = []
        for entry in soup.find_all("entry"):
            title = entry.find("title")
            summary = entry.find("summary")
            link = entry.find("id")
            if title and summary and link:
                abstract = summary.get_text(strip=True)
                results.append({
                    "type": "arxiv",
                    "title": title.get_text(strip=True),
                    "url": link.get_text(strip=True),
                    "abstract_snippet": abstract[:400],
                })
        return results
    except Exception as exc:
        print(f"  [Phase 0] arXiv search failed for '{query}': {exc}")
        return []


def _semantic_scholar_search(query: str, max_results: int = 5) -> list[dict]:
    """Search Semantic Scholar for recent papers. Returns list of source dicts."""
    try:
        params = {
            "query": query,
            "fields": "title,abstract,year,externalIds",
            "limit": max_results,
        }
        headers = dict(HEADERS)
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key
        resp = requests.get(
            SEMANTIC_SCHOLAR_API, params=params, headers=headers, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for paper in data.get("data", []):
            ext_ids = paper.get("externalIds") or {}
            arxiv_id = ext_ids.get("ArXiv")
            url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (
                f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"
            )
            abstract = (paper.get("abstract") or "")[:400]
            if abstract:
                results.append({
                    "type": "semantic_scholar",
                    "title": paper.get("title", ""),
                    "url": url,
                    "abstract_snippet": abstract,
                    "year": paper.get("year"),
                })
        return results
    except Exception as exc:
        print(f"  [Phase 0] Semantic Scholar search failed for '{query}': {exc}")
        return []


def _wikipedia_lead(topic: str) -> list[dict]:
    """Fetch the lead section summary from Wikipedia for a topic."""
    try:
        encoded = urllib.parse.quote(topic.replace(" ", "_"))
        url = WIKIPEDIA_API.format(title=encoded)
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        extract = data.get("extract", "")
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        title = data.get("title", topic)
        if extract:
            return [{
                "type": "wikipedia",
                "title": title,
                "url": page_url,
                "snippet": extract[:600],
            }]
        return []
    except Exception as exc:
        print(f"  [Phase 0] Wikipedia fetch failed for '{topic}': {exc}")
        return []


# ---------------------------------------------------------------------------
# Evidence gathering orchestrator
# ---------------------------------------------------------------------------

def _gather_evidence(domain: str) -> tuple[list[dict], str]:
    """
    Gather evidence from all sources for the given domain.

    Returns (all_sources, context_text) where context_text is a formatted
    string ready to inject into the LLM prompt.
    """
    all_sources: list[dict] = []
    context_parts: list[str] = []
    any_success = False

    # Build targeted queries
    queries = [
        domain,
        f"{domain} overview fundamentals",
        f"{domain} latest advances research 2024 2025",
        f"{domain} challenges open problems",
        f"{domain} applications real-world examples",
    ]

    # 1. DuckDuckGo
    print("  [Phase 0] Searching DuckDuckGo...")
    for q in queries[:4]:
        results = _ddg_search(q, max_results=5)
        if results:
            any_success = True
        all_sources.extend(results)
        time.sleep(0.5)  # gentle rate limit

    web_results = [s for s in all_sources if s["type"] == "web"]
    if web_results:
        context_parts.append("=== WEB SEARCH RESULTS ===")
        for r in web_results[:15]:
            context_parts.append(f"[{r['title']}] {r.get('snippet', '')} (source: {r['url']})")

    # 2. arXiv
    print("  [Phase 0] Searching arXiv...")
    arxiv_results = _arxiv_search(domain, max_results=6)
    if arxiv_results:
        any_success = True
    all_sources.extend(arxiv_results)
    if arxiv_results:
        context_parts.append("\n=== ARXIV PAPERS ===")
        for r in arxiv_results:
            context_parts.append(
                f"[{r['title']}] {r['abstract_snippet']} (source: {r['url']})"
            )

    # 3. Semantic Scholar
    print("  [Phase 0] Searching Semantic Scholar...")
    ss_results = _semantic_scholar_search(domain, max_results=5)
    if ss_results:
        any_success = True
    all_sources.extend(ss_results)
    if ss_results:
        context_parts.append("\n=== SEMANTIC SCHOLAR PAPERS ===")
        for r in ss_results:
            context_parts.append(
                f"[{r['title']} ({r.get('year', '?')})] {r['abstract_snippet']} (source: {r['url']})"
            )

    # 4. Wikipedia
    print("  [Phase 0] Fetching Wikipedia...")
    wiki_results = _wikipedia_lead(domain)
    if not wiki_results:
        # Try a simplified version of the domain name
        simplified = domain.split()[0] if " " in domain else domain
        wiki_results = _wikipedia_lead(simplified)
    if wiki_results:
        any_success = True
    all_sources.extend(wiki_results)
    if wiki_results:
        context_parts.append("\n=== WIKIPEDIA ===")
        for r in wiki_results:
            context_parts.append(f"[{r['title']}] {r.get('snippet', '')} (source: {r['url']})")

    if not any_success:
        print(
            "  [Phase 0] WARNING: all network sources failed. "
            "Falling back to LLM parametric knowledge only."
        )
        context_parts.append(
            "\n=== NOTE: All network sources failed. Use parametric knowledge only. ==="
        )
        all_sources.append({"type": "parametric", "note": "All network sources failed — parametric fallback"})

    return all_sources, "\n".join(context_parts)


# ---------------------------------------------------------------------------
# LLM question generation
# ---------------------------------------------------------------------------

GENERATE_PROMPT = """You are building a high-quality evaluation question set for a research agent specialising in "{domain}".

You have been given evidence gathered from web searches, academic papers, and Wikipedia below.
Use ONLY facts that appear in this evidence or are universally established. Any claim that is
speculative or contested must have its corresponding key_fact prefixed with "[disputed]".

SOURCE EVIDENCE:
{context}

Generate exactly 15 research questions about "{domain}" distributed as:
- 5 easy   (foundational concepts, well-established facts)
- 6 medium (mechanisms, tradeoffs, history, applications)
- 4 hard   (frontier research, open debates, critical evaluation of competing theories)

Requirements for each question:
- "question": a clear, self-contained research question (not a yes/no)
- "ground_truth": 2-4 sentences that constitute a correct, complete answer grounded in the evidence above
- "key_facts": 3-6 specific facts or concepts the answer must contain; prefix speculative ones with "[disputed]"
- "sources": list of source objects used for this specific question, each with at minimum a "type" and either a "url" or "note"
  - type must be one of: "web", "arxiv", "semantic_scholar", "wikipedia", "parametric"
  - include "title" and "url" when available
  - if a question relies solely on general knowledge, use {{"type": "parametric", "note": "LLM parametric knowledge"}}

Respond with ONLY a valid JSON array of 15 objects, each with keys:
id, tier, question, ground_truth, key_facts, sources

No preamble, no markdown fences, no trailing text. Only the JSON array."""


def _extract_partial_json_objects(raw: str) -> list[dict]:
    """
    Scan ``raw`` character-by-character and return every complete top-level
    ``{…}`` object that parses cleanly, even if the outer array is truncated.
    This is the fallback when ``json.loads`` fails on a cut-off response.
    """
    objects: list[dict] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False

    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(raw[start : i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None

    return objects


def _generate_questions(domain: str, context: str) -> list[dict]:
    """Call the LLM with gathered context and return 15 question dicts."""
    prompt = GENERATE_PROMPT.format(domain=domain, context=context[:12000])
    messages = [{"role": "user", "content": prompt}]
    # Keep the raw text so we can pass it to the partial-object fallback if needed.
    # 16 000 tokens leaves ample room for 15 detailed question objects.
    raw = call_llm(messages, max_tokens=16000)

    parser = JsonOutputParser()
    try:
        questions = parser.parse(raw)
    except OutputParserException as exc:
        # Secondary parse — salvage complete objects from a truncated array
        print(
            f"  [Phase 0] WARNING: JSON parse failed ({exc}); "
            "attempting partial recovery…"
        )
        questions = _extract_partial_json_objects(raw)
        if not questions:
            raise ValueError(
                "Could not recover any question objects from the LLM response. "
                "The model output may be empty or severely malformed."
            ) from exc
        print(f"  [Phase 0] Recovered {len(questions)} of 15 questions from partial response.")

    # Normalise IDs in case the LLM numbered them differently
    for i, q in enumerate(questions, start=1):
        q["id"] = f"q{i:02d}"

    return questions


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_ground_truth(domain: str, domain_slug: str) -> list[dict]:
    """
    Phase 0 main entry point.

    Returns the list of question dicts (15 items).
    Writes the result atomically to eval_suite/<domain_slug>.json.
    """
    print(f"\nPhase 0: Build Ground Truth for '{domain}'")

    os.makedirs(PROJECT_ROOT / "eval_suite", exist_ok=True)
    out_path = PROJECT_ROOT / "eval_suite" / f"{domain_slug}.json"

    all_sources, context = _gather_evidence(domain)
    print(f"  [Phase 0] Evidence gathered: {len(all_sources)} source records")

    print("  [Phase 0] Generating 15 questions via LLM...")
    questions = _generate_questions(domain, context)
    print(f"  [Phase 0] Generated {len(questions)} questions")

    # Atomic write: temp file → rename
    tmp_fd, tmp_path = tempfile.mkstemp(dir=PROJECT_ROOT / "eval_suite", suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(questions, f, indent=2)
        os.replace(tmp_path, out_path)
    except Exception:
        os.unlink(tmp_path)
        raise

    print(f"  [Phase 0] Written to {out_path}")
    return questions


def load_or_build(domain: str, domain_slug: str) -> list[dict]:
    """
    Load the cached ground truth for domain if it exists, or build it.

    This is the function called from main.py when --questions is not supplied.
    """
    cache_path = PROJECT_ROOT / "eval_suite" / f"{domain_slug}.json"
    if cache_path.exists():
        print(f"\nPhase 0: Reusing cached ground truth from {cache_path}")
        return json.loads(cache_path.read_text())
    return build_ground_truth(domain, domain_slug)
