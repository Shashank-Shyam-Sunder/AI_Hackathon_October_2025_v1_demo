# 🧩 Issues and Possible Solutions

This document tracks the current issues and improvement opportunities identified in the **AI Cost Estimator** project (as of October 2025).  
Each issue lists **what we observed**, the **underlying cause or limitation**, and **possible directions for future improvement**.

---
<a name="issue-1"></a>
## 1️⃣ Configuration & Hard-Coded Constants
**Observation:**  
Multiple tunable constants (monitoring rates, guardrails formula parameters, RAG cost multipliers, merge coefficients, model fallbacks, and compliance surcharges) are hard-coded inside `plan_and_cost.py`.

**Impact:**  
Changing any parameter requires code modification and redeployment.

**Possible Solution:**  
Move tunables into external configuration files (`.yml` or `.ini`) within a `config/` module.  
Each component (plan_and_cost, formulas, pricing) could read its settings from a structured config file.

Example YAML structure:
```yaml
guardrails:
  min_monthly_floor: 50.0
  token_coef: 1.5e-6
  per_request_base: 0.0001
```  
---

## 2️⃣ RAG Costing Logic (Static vs Dynamic)
**Observation:**  
RAG one-time and monthly storage costs are computed from corpus size only and remain constant per run (`corpus_gb × fixed $/GB`).  
They do not depend on actual monthly requests or usage patterns.

**Impact:**  
- RAG costs do not scale with traffic, making total estimates partially static.  
- The final cost summary correctly excludes the one-time RAG embedding cost from monthly totals, but this design can be confusing to new users.

**Possible Solution:**  
- Introduce an optional request-based multiplier or integrate dynamic scaling parameters in the formulas module to make RAG costs responsive to user traffic.  
- Clarify in the final cost report how RAG costs are separated between one-time embedding and recurring storage components to improve readability for new users.
---

## 3️⃣ Formula Duplication Between Modules
**Observation:**  
Some formulas are centralized in `formulas.py`, while others (e.g., guardrails and monitoring) are implemented directly in `plan_and_cost.py`. In addition, several formula **parameters** (rates, floors, multipliers) are hard-coded alongside the logic.

**Impact:**  
- Partial duplication and scattered logic make maintenance harder and increase the risk of inconsistencies.  
- Mixing **logic** and **tunable values** in multiple files blurs the source of truth and couples code changes to parameter tweaks (related to [Issue 1](#issue-1))

**Possible Solution:**  
- Consolidate reusable **formula logic** in `formulas.py` and call it from `plan_and_cost.py` (single source of truth for code).  
- Centralize **tunable parameters** (rates, floors, multipliers) in configuration files (e.g., `.yml`), so formulas consume values from config rather than hard-coded constants.

---

## 4️⃣ Model Pricing Coverage
**Observation:**  
The catalog references more model keys than those priced in `pricing_dict.py`.  
Models without explicit prices (e.g., `mistral-large-mini`) fall back to safe defaults such as `mistral-medium` or `gpt-4o-mini`.

**Impact:**  
Fallback ensures computation continuity but may under- or over-estimate costs.

**Possible Solution:**  
Expand `PRICING_DICT` to cover all catalog models or link it to an external JSON/YAML file for easier updates.

---

## 5️⃣ API-Call and Self-Hosted Tasks
**Observation:**  
The intake CLI allows the user to specify the deployment mode (API or self-hosted).  
However, in the current implementation, cost computation is applied **only** to tasks whose `pricing_basis` is either `token` or `embedding`.  
Tasks marked as `api_call` are detected but left with undefined cost fields (“TBD”), and `self_host` tasks are not recognized at all.

**Impact:**  
- The system calculates accurate costs for **token-based** and **embedding-based** tasks only.  
- **API-call–based** tasks are listed in the report but have no numeric values.  
- **Self-hosted** scenarios are completely ignored, resulting in underestimation of total infrastructure or operational costs.  
- The intake step may therefore create an expectation of full cost coverage that is not reflected in the final report.

**Possible Solution:**  
- Implement explicit handling for `api_call` and `self_host` pricing bases within the formulas module.  
- For **API-call** tasks, introduce a per-request or per-usage unit cost in the pricing dictionary.  
- For **self-hosted** tasks, define infrastructure-level cost estimation (e.g., GPU runtime hours, hosting, or model inference overhead).  
- Optionally clarify in the CLI and report that only `token` and `embedding` bases are currently costed in the MVP version.
---

## 6️⃣ Model Visibility in Reports
**Observation:**  
The Markdown report does not display which LLM models were used to compute each task’s cost (min / max).

**Impact:**  
Readers cannot easily verify which model contributed to each estimate.

**Possible Solution:**  
Add columns “Model (MIN)” and “Model (MAX)” in the final summary tables or include a short “Model Summary” section per category.

---

## 7️⃣ Merging Behavior & Report Readability
**Observation:**  
Merged tasks are annotated with `"[MERGED xN]"`, but the detailed MIN/MAX plan tables become repetitive and difficult to interpret.  
Although merging logic is correctly applied at the computation level, the report structure does not make it easy to understand how merging affected token counts or per-task costs.

**Impact:**  
- The detailed plan tables list many similar rows with identical token values, making them visually dense.  
- Reviewers cannot easily distinguish between baseline tokens and merged (effective) tokens or see how merging reduces per-category cost.  
- The `[MERGED xN]` note alone is not sufficient to convey the merging logic to non-technical readers.

**Possible Solution:**  
- Simplify the report by grouping merged tasks under their parent category and summarizing them in a single merged row.  
- Add a small “Merge Summary” or “Category Token Efficiency” table showing baseline tokens, merged tokens, and the resulting token reduction ratio.  
- Include a short explanatory note describing how merging affects total cost and why some token values repeat across tasks.

---

## 8️⃣ Report Formatting & Currency Display
**Observation:**  
In the generated Markdown cost summary, certain numeric or currency symbols (e.g., `$` in totals or before-compliance/after-compliance lines) are not consistently rendered in Markdown viewers.  
Additionally, the MIN/MAX detail tables are dense and repeat similar numeric values, making them difficult to read at a glance.

**Impact:**  
- Dollar symbols and other special characters may be escaped or lost when viewed in some editors (e.g., PyCharm preview).  
- Users may find it difficult to interpret totals or distinguish between one-time and recurring costs due to formatting limitations.  
- The detailed MIN/MAX tables appear cluttered with repetitive columns and annotations, reducing readability.

**Possible Solution:**  
- Ensure proper Markdown escaping for special characters (e.g., prefix `$` with a backslash `\$` where needed).  
- Consider generating both Markdown and HTML (or CSV) versions of the cost report for better viewing and integration with documentation tools.  
- Simplify the tabular layout by reducing redundant rows or grouping similar tasks under categories (can be combined with Issue 7 improvements).  
- Optionally add color-coded or bolded highlights for key metrics (e.g., totals, compliance-adjusted values) to improve clarity in rendered reports.

---


## 9️⃣ Missing Unit Tests
**Observation:**  
No explicit unit tests exist for key cost functions such as `compute_token_cost`, `compute_guardrails_cost`, and `compute_langsmith_cost`.

**Impact:**  
- Changes to formulas or constants could introduce silent regressions without detection.  
- The reliability of computed totals may degrade over time as parameters evolve.

**Possible Solution:**  
- Create a lightweight `tests/` directory with basic **pytest** coverage for all formula and cost computation functions.  
- Include boundary-case tests (e.g., zero tokens, high request volumes) to ensure consistency when constants or formulas are updated.


---

## 🔟 Compliance Handling Extension
**Observation:**  
Compliance multipliers are hard-coded for only two categories (GDPR 1.17, HIPAA 1.40).

**Impact:**  
- Limited flexibility for new compliance schemes (e.g., FedRAMP, SOC 2).  
- Updating compliance logic requires direct code edits rather than configuration updates.

**Possible Solution:**  
- Load compliance surcharges dynamically from configuration files (e.g., YAML or JSON).  
- Allow flexible expansion for new compliance frameworks without modifying source code.

---

## 1️⃣1️⃣ Catalog ↔ Computation Join Robustness
**Observation:**  
Catalog joins rely on flexible header matching (“Parent Category”, “model_band_min_key”, etc.), but naming inconsistencies could still break merges.

**Impact:**  
- Missing or inconsistent headers can cause certain tasks to lose their parent categories or model references.  
- Potential silent data mismatches if column names are modified in future catalog versions.

**Possible Solution:**  
- Add header validation or schema mapping logic when loading the catalog.  
- Introduce a standardized schema definition file (`catalog_schema.yml`) to enforce column names and types before merging.

---

📌 *Note:*  
These items summarize the current MVP limitations and incorporate initials feedback from the moderator. We will add more issues and their possible solution if we receive more feedback.  

No improvements will be made at this stage, as the repository is under review until **October 27, 2025**, as part of the **AI Hackathon October 2025** organized by the **Hyperskill Team**.

