# Financial Intelligence Assistant

Formal report: `docs/project2_final_report.md`  
Presentation deck: `docs/presentation_deck.md`  
Live demo script: `docs/live_demo_script.md`

## Overview

This repository implements an industry-specific AI assistant for financial-document intelligence. It is built on a local corpus of public-company annual reports and 10-K filings and is designed to satisfy the course evaluation requirements across:

- prompt engineering
- retrieval-augmented generation
- agent/tool use
- reliability, hallucination tracking, prompt-injection resistance, and bias analysis

The assistant supports two answer modes:

- `Direct RAG`: retrieve evidence and answer immediately
- `Agent Analyst`: use local function-calling tools before answering

Both modes operate only on the local corpus and can be compared under the same retrieval settings.

## Business Case

The product target is a financial analyst, strategy team, or competitive-intelligence user who needs fast, source-grounded answers from dense public filings.

The assistant focuses on three workflows:

- grounded question answering over filings
- knowledge exploration over extracted entities and triplets
- competitor evidence analysis across years

## Corpus Scope

The local corpus intentionally stays small and auditable:

- companies: `adobe`, `lockheedmartin`, `pfizer`
- fiscal years: `2022`, `2023`, `2024`
- document family: annual reports / 10-K style filings

## What The Project Demonstrates

### 1. Advanced Prompt Engineering

- explicit financial-analyst persona
- strict JSON output contracts
- anti-hallucination rules
- anti-prompt-injection instructions
- few-shot examples for grounded answers and safe refusals

### 2. RAG

- `classical_ml`: TF-IDF + cosine lexical baseline
- `naive`: dense retrieval baseline
- `improved`: dense + lexical + BM25 + structure + knowledge-aware ranking

### 3. Agents And Tool Use

The `Agent Analyst` mode uses local function calling with three tools:

- `search_financial_corpus`
- `lookup_knowledge_graph`
- `get_competitor_evidence`

The agent is bounded to a maximum of 3 tool calls and always returns a final JSON answer with:

- `answer`
- `citations`
- `tool_calls`
- `safety_flags`
- `mode`

### 4. Reliability, Security, And Bias Evaluation

The repository keeps the original LLM-as-a-Judge support pipeline and adds a dedicated security benchmark for:

- hallucination / out-of-corpus requests
- prompt injection
- biased or loaded questions

Tracked metrics include:

- support rate
- refusal rate
- fabricated citation rate
- injection resistance rate
- bias-safe response rate

## Repository Structure

- `app/streamlit_app.py`: interactive interface
- `src/generation/rag_answer.py`: direct RAG flow
- `src/agent/workflow.py`: local function-calling analyst agent
- `src/agent/tools.py`: local tool definitions
- `src/evaluation/judge.py`: LLM-as-a-Judge support verification
- `src/evaluation/security_eval.py`: security benchmark runner
- `docs/project2_final_report.md`: technical report
- `docs/presentation_deck.md`: slide-by-slide presentation deck
- `docs/live_demo_script.md`: demo and Q&A script

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional FAISS backend:

```powershell
python -m pip install -e ".[faiss]"
```

Environment variables:

```powershell
Copy-Item .env.example .env
```

Expected variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional)
- `RUN_LIVE_LLM_TESTS` (optional)

## Run The App

```powershell
.venv\Scripts\streamlit run app/streamlit_app.py
```

Inside the UI you can choose:

- assistant mode: `Direct RAG` or `Agent Analyst`
- retrieval mode: `improved`, `naive`, or `classical_ml`
- company and year filters

The app can display:

- final answer
- validated citations
- safety flags
- agent tool trace
- retrieved chunks
- knowledge explorer
- competitor evidence

## Main Commands

### Direct RAG

```powershell
python -m src.generation.rag_answer --question "What are Adobe's main risk factors in 2024?" --retrieval_mode improved
```

### Agent Analyst

```powershell
python -m scripts.run_financial_agent --question "How did Adobe describe competition in 2024?" --retrieval_mode improved
```

### Demo Comparison

```powershell
python -m scripts.run_rag_demo --retrieval_mode classical_ml
python -m scripts.run_rag_demo --retrieval_mode naive
python -m scripts.run_rag_demo --retrieval_mode improved
```

### Full Pipeline

```powershell
python -m scripts.run_pipeline
python -m scripts.run_pipeline --dry-run
python -m scripts.run_pipeline --from-phase evaluation --to-phase metrics
```

### Why The Full Pipeline Takes Time

`python -m scripts.run_pipeline` runs the whole research workflow, not just one RAG answer. By default it launches 18 sequential steps covering preprocessing, embedding/index construction, entity extraction, triplet extraction, competitor analysis, FinanceBench preparation, then evaluation, judge, and metrics for all three retrieval modes: `classical_ml`, `naive`, and `improved`.

The heaviest cost on a cold run is the chunk-level extraction work. In the current project snapshot, the corpus produces about `1.9k` chunks, and both entity extraction and triplet extraction run in `all` mode. That means the first complete run can trigger roughly `3.7k` LLM extraction requests before the evaluation phase even starts.

The later stages also multiply runtime on purpose because the project compares retrieval strategies instead of measuring only one setup. Even with the default `local_smoke` split, evaluation and judge are repeated for each retrieval mode, and switching to the larger `core40` benchmark increases the number of retrieval, generation, and judgment calls significantly.

The pipeline is intentionally sequential, checkpointed, and reproducible, so it saves intermediate artifacts often instead of optimizing only for raw speed. Reruns are usually much faster thanks to embedding caches, the JSONL LLM cache, and resume logic in extraction, evaluation, and judge. The first end-to-end run is therefore expected to be the slowest.

### LLM-As-A-Judge

```powershell
python -m scripts.run_judge --retrieval_mode improved
```

### Security Benchmark

```powershell
python -m scripts.run_security_eval --mode rag
python -m scripts.run_security_eval --mode agent
python -m scripts.run_security_eval --mode both
```

Default security cases live in:

- `data/evaluation/security/security_cases.csv`

Default security outputs are written to:

- `outputs/security_eval/rag/`
- `outputs/security_eval/agent/`

## Evaluation Criteria Mapping

### Functional Prototype

- Streamlit UI
- direct RAG mode
- agent mode
- JSON export of answers

### Advanced Prompt Engineering

- persona-driven prompts
- few-shot examples
- strict JSON contracts
- anti-hallucination and anti-injection instructions

### RAG Implementation

- local document corpus
- explicit chunking and retrieval modes
- citations validated against retrieved pages

### Agents And Tool Use

- local function calling
- bounded tool loop
- visible tool trace in the UI and JSON output

### Evaluation Of Reliability, Bias, And Security

- LLM support judge
- prompt-injection benchmark
- hallucination cases
- bias-sensitive cases

## Tests

Run the suite with:

```powershell
.venv\Scripts\pytest -q
```

The test suite covers:

- prompt contracts and few-shot presence
- retrieval and pipeline helpers
- LLM client retry logic
- tool-calling chat behavior
- agent workflow traces and refusals
- security benchmark loading and metrics

## Known Limitations

- the corpus is intentionally limited to 3 companies and 3 years
- security evaluation uses a compact benchmark, not a production red-team corpus
- bias handling is evaluated through guardrails and neutral wording, not through a full fairness benchmark
- live LLM quality still depends on the selected model and API availability

## Submission Assets

- code: this repository
- report: `docs/project2_final_report.md`
- slides: `docs/presentation_deck.md`
- live demo script: `docs/live_demo_script.md`
