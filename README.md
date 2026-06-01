# AI Tool Taxonomy Classifier

> An LLM-powered pipeline that classifies ~800 AI tools into a self-designed 3-level hierarchical taxonomy, runs four models in parallel, and evaluates cross-model consistency - built to demonstrate taxonomy design thinking, prompt engineering, structured outputs, and production-quality Python pipelines.

---

## Motivation

The AI tool landscape is fragmented and growing fast. Existing directories (ProductHunt, RankmyAI, Futurepedia) use flat or shallow category systems that collapse meaningfully different tools into the same bucket - "Productivity" can describe a code-gen agent, a scheduling chatbot, and a document summariser.

A **principled hierarchical taxonomy** solves three things:
1. **Discovery** - users navigate by domain, then capability, then technique.
2. **Comparison** - tools at the same L3 node are genuinely comparable.
3. **Trend detection** - you can watch whole branches of the tree grow or shrink.

The hard part is that many AI tools span multiple modalities or use-cases. This project treats ambiguity as a first-class signal - models are asked to flag it explicitly, and disagreement between models is measured rather than hidden.

---

## Taxonomy Design

The taxonomy has **3 levels** (63 nodes total):

| Level | Name | What it captures |
|---|---|---|
| **L1** | Domain | Primary modality / problem space |
| **L2** | Capability | What the tool *does* within that domain |
| **L3** | Technique | *How* it does it - the most specific, comparable unit |

### L1 Domains

```
L1-001  Language & Text          L1-005  Multimodal
L1-002  Vision & Image           L1-006  Data & Analytics
L1-003  Audio & Speech           L1-007  Agents & Automation
L1-004  Video & Animation        L1-008  Developer Infrastructure
```

### Design choices vs. existing directories

| Decision | Rationale |
|---|---|
| **"Multimodal" is its own L1** | Tools like GPT-4V or Gemini are architecturally distinct from unimodal tools; lumping them under "Language" loses signal. |
| **"Developer Infrastructure" as a peer domain** | Vector DBs, eval platforms, and fine-tuning services are AI tools *for AI builders*, not for end-users - they deserve their own branch. |
| **L3 by technique, not audience** | Audience (B2B vs consumer) changes; the underlying technique doesn't. L3 nodes stay stable as tools pivot. |
| **Ambiguity is modelled explicitly** | A `ambiguous` flag + `alternative_l2` field captures borderline cases rather than forcing a false single classification. |

The full taxonomy definition is in [`data/taxonomy.json`](data/taxonomy.json).

---

## Architecture

```
scripts/download_data.py       ← fetch + clean ~800-1000 tools
scripts/build_taxonomy.py      ← (one-time) LLM-assisted L3 generation
scripts/run_classification.py  ← classify with all 4 models

src/ai_taxonomy/
├── config.py                  ← paths, model names, hyperparameters
├── data/
│   ├── loader.py              ← load_tools(), clean_description()
│   └── preprocessor.py        ← normalize, deduplicate, filter
├── taxonomy/
│   ├── schema.py              ← Pydantic ClassificationResult model
│   └── validator.py           ← hierarchy consistency checks
├── models/
│   ├── base.py                ← abstract BaseClassifier
│   ├── claude.py              ← Claude Sonnet 4 (Anthropic SDK)
│   ├── openai_model.py        ← GPT-4o-mini (OpenAI SDK)
│   ├── gemini_model.py        ← Gemini 2.0 Flash (Google GenAI SDK)
│   └── mistral_model.py       ← Mistral Small (Mistral SDK)
├── pipeline/
│   ├── classify.py            ← batch + retry + checkpoint
│   └── golden_set.py          ← manual review + accuracy scoring
└── evaluation/
    ├── metrics.py             ← agreement, Cohen's κ, confidence
    └── visualization.py       ← heatmaps, bar charts, scatter plots

main.py                        ← one-command pipeline runner
```

### Data flow

```
HuggingFace datasets ──┐
                       ├── download_data.py ──► tools.jsonl
RankmyAI scrape ───────┘

tools.jsonl ──► classify.py ──► claude_results.jsonl
                            ──► gpt4o_results.jsonl
                            ──► gemini_results.jsonl
                            ──► mistral_results.jsonl

*.jsonl ──► metrics.py  ──► inter_model_agreement.csv
                        ──► confidence_distribution.csv
                        ──► inter_model_kappa.csv

*.jsonl ──► visualization.py  ──► figures/
```

---

## Models

| Model | Provider | Role | Est. cost / 800 tools |
|---|---|---|---|
| `claude-sonnet-4-6` | Anthropic | Primary classifier, highest quality | ~$0.60 |
| `gpt-4o-mini` | OpenAI | Strong secondary, cost-efficient | ~$0.04 |
| `gemini-2.0-flash` | Google | Free-tier primary | **free** |
| `mistral-small-latest` | Mistral | Open-weight baseline | ~$0.06 |

All four models receive an **identical prompt** with the full taxonomy JSON and a Pydantic JSON schema. This controls for prompt variation when comparing outputs.

### Structured output

Every model returns a JSON object conforming to this schema:

```python
class ClassificationResult(BaseModel):
    tool_id: str
    tool_name: str
    l1_id: str;  l1_label: str
    l2_id: str;  l2_label: str
    l3_id: str;  l3_label: str
    confidence: float          # 0–1, how well description fits chosen category
    reasoning: str             # one sentence, max 20 words
    ambiguous: bool            # True if tool plausibly fits multiple L2s
    alternative_l2: str | None
```

---

## Evaluation

### Metrics

| Metric | Description |
|---|---|
| **Inter-model agreement @ L1** | % of tools where all 4 models agree on the top-level domain |
| **Inter-model agreement @ L3** | % of tools where all 4 models agree on the leaf technique |
| **Cohen's κ (pairwise)** | κ score for every pair of models at L3 |
| **Ambiguity rate** | % of tools flagged ambiguous by each model |
| **Confidence distribution** | Histogram of confidence scores per model |
| **Golden set accuracy** | Precision@L3 vs 50 manually labelled tools |

### Visualisations

- **Confidence distribution** - density histogram per model; reveals calibration differences.
- **Agreement rate by L1 domain** - bar chart showing which domains are easiest/hardest to agree on.
- **L2 confusion heatmap** - where do two models diverge most? Highlights structurally ambiguous categories.
- **Confidence vs agreement scatter** - are high-confidence predictions actually right?

---

## Quickstart

### 1. Install

```bash
uv sync
cp .env.example .env
# Fill in your API keys in .env
```

### 2. Download and process data

```bash
uv run python scripts/download_data.py
```

Fetches ~800–1000 tools from HuggingFace datasets and RankmyAI, deduplicates, and saves to `data/processed/tools.jsonl`.

### 3. (Optional) Regenerate taxonomy L3 nodes

```bash
uv run python scripts/build_taxonomy.py
```

Calls Claude to propose L3 leaf nodes. **Review and edit `data/taxonomy.json` manually before continuing** - this is the human-in-the-loop step.

### 4. Run classification

```bash
# Smoke test - 20 tools, all models
uv run python scripts/run_classification.py --smoke-test

# Full run
uv run python scripts/run_classification.py
```

Results checkpoint to `outputs/classifications/*.jsonl` after each batch, so runs are resumable.

### 5. Evaluate

```bash
uv run python main.py --eval-only
```

Or run the full pipeline end-to-end:

```bash
uv run python main.py
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...         # free tier at aistudio.google.com
MISTRAL_API_KEY=...
```

---

## Golden Dataset & Feedback Loop

After classification, sample 50 tools for manual review:

```python
from src.ai_taxonomy.pipeline.golden_set import sample_for_review, save_golden_entry

items = sample_for_review(Path("outputs/classifications/claude-sonnet-4-6_results.jsonl"))
for item in items:
    # review each item, then:
    save_golden_entry(item["tool_id"], correct_l3_id="L3-003", model_was_wrong=False)
```

The golden set is saved to `data/processed/golden_set.jsonl` and used to compute per-model **Precision@L3**.

---

## Key Findings (expected)

Based on the taxonomy structure, we expect:

- **High L1 agreement (~90%)** - domain boundaries are usually clear from tool descriptions.
- **Lower L3 agreement (~60–70%)** - technique-level distinctions are genuinely harder, especially at the boundary between *Agents & Automation* and *Developer Infrastructure*.
- **Gemini and Claude will agree most** - both trained on similar corpora and instruction-tuned for structured output.
- **Ambiguity concentrated in Multimodal and Agents** - these categories cut across natural boundaries.

---

## What I'd Do Next

- **Active learning loop** - route low-confidence items to a human reviewer; use corrections to refine the taxonomy iteratively.
- **Embedding-based fallback** - for tools with very short descriptions, use semantic similarity to the L3 centroid as a tie-breaker.
- **Fine-tune a small classifier** - once the golden set reaches ~500 examples, fine-tune a `distilbert` or `ModernBERT` model as a fast, cheap local classifier.
- **Taxonomy versioning** - track taxonomy changes in git and flag tools whose classification changes across versions.
- **Confidence calibration** - use Platt scaling or temperature tuning to align model-reported confidence with actual accuracy.
