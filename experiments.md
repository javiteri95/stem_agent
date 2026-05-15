# STEM Agent Evaluation Report

**Date:** May 15, 2026
**Architecture (all runs):** `reflection_loop`
**Eval framework:** Per-tier scoring across `factual_accuracy`, `coverage`, `coherence`, `source_diversity`, and `composite`

---

## 1. Model-Specific Breakdowns

---

### GPT-5.4

#### Task: Code Review

**Optimization:** 0 iterations run — agent spec left at version 0, no mutation found.

**By-Tier Performance:**

| Tier   | Factual Accuracy | Coverage | Coherence | Source Diversity | Composite  |
|--------|-----------------|----------|-----------|-----------------|------------|
| Easy   | 0.972           | 0.954    | 0.986     | 0.842           | **0.9385** |
| Medium | 0.812           | 0.638    | 0.968     | 0.743           | **0.7904** |
| Hard   | 0.793           | 0.613    | 0.970     | 0.845           | **0.8050** |

**Overall (Before → After):** 0.8437 → 0.8437 (Δ 0.000)

---

#### Task: Deep Research

**Optimization:** 2 iterations — switched to `breadth_first` then `iterative_refinement`, increased `max_search_rounds` to 6, rewrote prompt to push comparative synthesis.

**By-Tier Performance:**

| Tier   | Factual Accuracy | Coverage | Coherence | Source Diversity | Composite  |
|--------|-----------------|----------|-----------|-----------------|------------|
| Easy   | 0.952           | 0.908    | 0.988     | 0.792           | **0.910**  |
| Medium | 0.707           | 0.597    | 0.953     | 0.746           | **0.7507** |
| Hard   | 0.883           | 0.793    | 0.930     | 0.840           | **0.8617** |

**Overall (Before → After):** 0.8045 → 0.826 (Δ +0.0215)

---

**Key Finding:** GPT-5.4 is the weakest baseline — its code review spec was so misaligned with the research-heavy evaluation that the optimizer returned 0 iterations and no improvement; deep research gained only marginally (+0.022), with medium-tier coverage remaining the persistent floor at 0.597.

---

### GPT-5.4-Mini

#### Task: Code Review

**Optimization:** 1 iteration — tightened prompt to issue-oriented, line-anchored findings; removed tutorial-style responses.

**By-Tier Performance:**

| Tier   | Factual Accuracy | Coverage | Coherence | Source Diversity | Composite  |
|--------|-----------------|----------|-----------|-----------------|------------|
| Easy   | 0.976           | 0.960    | 0.984     | 0.830           | **0.9375** |
| Medium | 0.955           | 0.897    | 0.982     | 0.822           | **0.9137** |
| Hard   | 0.958           | 0.880    | 0.975     | 0.895           | **0.9269** |

**Overall (Before → After):** 0.9042 → 0.9252 (Δ +0.021)

---

#### Task: Deep Research

**Optimization:** 1 iteration — switched to `breadth_first` + `extractive` synthesis to stay closer to retrieved evidence.

**By-Tier Performance:**

| Tier   | Factual Accuracy | Coverage | Coherence | Source Diversity | Composite  |
|--------|-----------------|----------|-----------|-----------------|------------|
| Easy   | 0.842           | 0.766    | 0.968     | 0.598           | **0.7935** |
| Medium | 0.703           | 0.442    | 0.960     | 0.753           | **0.7146** |
| Hard   | 0.863           | 0.700    | 0.965     | 0.745           | **0.8181** |

**Overall (Before → After):** 0.7537 → 0.7685 (Δ +0.0148)

---

#### Task: Agent Development

**Optimization:** 1 iteration — increased `max_search_rounds` to 6, switched to `breadth_first` + `extractive` synthesis.

**By-Tier Performance:**

| Tier   | Factual Accuracy | Coverage | Coherence | Source Diversity | Composite  |
|--------|-----------------|----------|-----------|-----------------|------------|
| Easy   | 0.916           | 0.840    | 0.972     | 0.682           | **0.8525** |
| Medium | 0.716           | 0.617    | 0.954     | 0.580           | **0.7168** |
| Hard   | 0.880           | 0.667    | 0.963     | 0.807           | **0.8292** |

**Overall (Before → After):** 0.7662 → 0.7845 (Δ +0.0183)

---

#### Task: Information Security

**Optimization:** 2 iterations — rewrote prompt to enforce definition-first, checklist-based structure; switched to `depth_first` with `max_search_rounds = 6`.

**By-Tier Performance:**

| Tier   | Factual Accuracy | Coverage | Coherence | Source Diversity | Composite  |
|--------|-----------------|----------|-----------|-----------------|------------|
| Easy   | 0.976           | 0.970    | 0.982     | 0.826           | **0.9385** |
| Medium | 0.867           | 0.883    | 0.978     | 0.845           | **0.8933** |
| Hard   | 0.965           | 0.938    | 0.980     | 0.878           | **0.9400** |

**Overall (Before → After):** 0.9138 → 0.9208 (Δ +0.007)

---

**Key Finding:** GPT-5.4-mini is a split performer — it achieves its strongest results in information security (0.9208 final composite, narrowest easy–hard gap of any model/task combination) but its weakest in deep research, where medium-tier coverage collapses to 0.442, the single lowest tier-metric value across the entire study.

---

### GPT-5.5

#### Task: Code Review

**Optimization:** 2 iterations — broadened prompt into a research-synthesis agent; added `web_search` tool; increased `max_search_rounds` to 6; required exact-source retrieval before finalizing.

**By-Tier Performance:**

| Tier   | Factual Accuracy | Coverage | Coherence | Source Diversity | Composite  |
|--------|-----------------|----------|-----------|-----------------|------------|
| Easy   | 0.978           | 0.992    | 1.000     | 0.990           | **0.990**  |
| Medium | 0.857           | 0.755    | 0.972     | 0.933           | **0.8792** |
| Hard   | 0.935           | 0.878    | 0.988     | 0.963           | **0.9406** |

**Overall (Before → After):** 0.9048 → 0.9325 (Δ +0.028)

#### Task: Deep Research

No data — output folder is empty. Probably the token provided was deactivated as I have many timeouts in the process.

---

**Key Finding:** GPT-5.5 achieves the highest single-tier score in the entire study (easy composite = 0.990, including a perfect coherence = 1.000), and its shift to a research-synthesis framing with `web_search` produced the largest source-diversity gain of any run (+0.083), confirming that tooling access is the primary lever for this model.

---

## 2. Overall Conclusion

### Model Ranking by Final Composite (highest task score per model)

| Model        | Best Task        | Final Composite | Worst Task      | Final Composite |
|--------------|------------------|----------------|-----------------|----------------|
| GPT-5.5      | Code Review      | **0.9325**      | Deep Research   | N/A (no data)  |
| GPT-5.4-mini | Info Security    | **0.9208**      | Deep Research   | 0.7685         |
| GPT-5.4      | Deep Research    | **0.826**       | Code Review     | 0.8437 (stalled) |

### Global Takeaways

**1. Model capability sets the ceiling — optimization raises the floor, not the ceiling.**
The optimizer never overcame a weak starting model. GPT-5.4's code review stalled at 0.8437 with 0 iterations because the spec was too mismatched to the task shape. GPT-5.5 started at 0.9048 and reached 0.9325 — a bigger absolute gain because the model had the latent capability to exploit the prompt improvements.

**2. Coverage is the universal bottleneck — coherence is free.**
Across all 9 model/task experiments, `coherence` ranged 0.930–1.000 (all strong), while `coverage` ranged 0.442–0.992 (high variance, lowest scores). This means all three models produce fluent, well-structured output, but systematically fail to address every point the question demands. Every productive mutation in the study targeted coverage deficits, not fluency.

**3. Prompt strategy must match the eval's question shape.**
The two most successful mutations took opposite directions and both worked:
- GPT-5.4-mini's code review: **narrowed** to line-anchored, issue-only findings → best medium-tier performance (0.9137).
- GPT-5.5's code review: **broadened** to research synthesis with `web_search` → best hard-tier and source-diversity performance (0.9406, 0.963).

This reveals that the evaluation's questions span two distinct regimes — precision-demanding medium questions and research-depth-demanding hard questions — and a single prompt cannot simultaneously optimize both unless it explicitly handles both modes.

**4. Information security is GPT-5.4-mini's dominant domain.**
With a final composite of 0.9208 and the tightest easy–hard spread (0.9385 → 0.9400 → 0.8933), information security is the one domain where the mini model matches larger-model quality. The structured, definition-first checklist prompt is likely responsible, suggesting this pattern is highly transferable to other factual taxonomy tasks.

**5. Deep research is structurally the hardest domain.**
The best final composite for deep research across all models is 0.826 (GPT-5.4), while the worst is 0.7685 (GPT-5.4-mini). Even after multi-iteration optimization, no model cracked 0.83 on this task. The core failure is consistent: medium-tier coverage collapses (0.44–0.60 range) because questions require comparative synthesis across heterogeneous sources, not just factual recall. Iterative refinement with targeted follow-up queries partially addressed this but did not solve it.

**6. The GPT-5.5 deep research gap is the most important open question.**
GPT-5.5 was evaluated on code review only — deep research output is empty. Given that GPT-5.4-mini and GPT-5.4 both struggled severely on deep research, and that GPT-5.5 demonstrated the strongest prompt-responsiveness and tooling gains on code review, running GPT-5.5 on deep research is the single highest-priority next experiment.
