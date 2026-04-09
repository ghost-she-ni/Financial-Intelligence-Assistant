# Presentation Deck

## Slide 1 - Title

Financial Intelligence Assistant  
Grounded QA, Agent Tools, and Security Evaluation for Public Filings

## Slide 2 - Problem

- Financial filings are long, repetitive, and expensive to review manually.
- Analysts need fast answers with sources, not generic summaries.
- Generic chatbots are risky because they can hallucinate or answer without evidence.

## Slide 3 - Business Case

- Target users: financial analysts, strategy teams, competitive-intelligence users
- Core value: answer filing questions quickly with citations
- Extension value: inspect entities, relationships, and competitor evidence

## Slide 4 - Corpus

- 3 companies
- 3 fiscal years
- 9 reports
- local, auditable corpus

## Slide 5 - Architecture

- PDF extraction and cleaning
- token chunking
- retrieval layer
- generation layer
- agent/tool layer
- evaluation layer

## Slide 6 - Prompt Engineering

- financial-intelligence analyst persona
- strict JSON outputs
- few-shot examples
- anti-hallucination rules
- anti-prompt-injection rules

## Slide 7 - RAG Modes

- `classical_ml`
- `naive`
- `improved`

Explain why comparing these 3 modes matters academically.

## Slide 8 - Agent Mode

- local function calling
- `search_financial_corpus`
- `lookup_knowledge_graph`
- `get_competitor_evidence`
- bounded to 3 tool calls

## Slide 9 - Reliability And Security

- LLM-as-a-Judge for support tracking
- dedicated security benchmark
- hallucination cases
- prompt-injection cases
- bias-sensitive cases

## Slide 10 - Demo

- direct RAG question
- agent competitor question
- prompt-injection attempt
- inspect tool trace and safety flags

## Slide 11 - Limitations

- small corpus
- limited benchmark coverage
- heuristic bias scoring
- depends on live LLM quality

## Slide 12 - Roadmap

- larger filing coverage
- broader security benchmark
- richer dashboards
- production-grade observability
