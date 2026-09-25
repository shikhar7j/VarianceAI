# VarianceAI

A GenAI-powered FP&A tool that automates two recurring finance-close tasks —
variance commentary and financial-statement Q&A — behind a single LangGraph
agent that routes free-form questions to the right one automatically.

## Features

1. **Automated Variance Commentary** — computes budget-vs-actual variances
   from a CSV and drafts the "why did this move" narrative an analyst would
   otherwise write by hand every month-end close.
2. **Financial Document RAG** — ask a natural-language question (e.g. "What
   are the key risks next quarter?") and get an answer grounded directly in
   the source document text.
3. **Agent Orchestration (LangGraph)** — a single `/api/agent` endpoint takes
   one free-form question and routes it to the correct tool without the
   caller needing to know which applies. An LLM classifies intent in live
   mode; a deterministic keyword heuristic does the same job in fallback
   mode, so the whole graph runs with zero external dependencies.

## Live vs. Fallback Mode

The app runs fully out of the box, with no API key required.

- **Fallback Mode (default)**: variance commentary uses rule-based templates;
  document Q&A retrieves the most relevant chunk via local TF-IDF
  (`scikit-learn`); the agent routes via keyword matching.
- **Live Mode**: once `OPENAI_API_KEY` is set in `.env`, the exact same code
  paths switch to using `gpt-4o-mini` (via LangChain's `ChatOpenAI`) to draft
  commentary, synthesize answers from retrieved context, and classify agent
  routing by intent instead of keywords.

## Project structure

```
app/
├── __init__.py           # Flask application factory
├── config.py              # typed Settings, single source of env vars
├── llm_client.py          # LangChain ChatOpenAI wrapper — only file that touches it
├── routes.py              # HTTP layer: request/response, validation, error handling
└── services/
    ├── variance.py         # VarianceAnalyzer — variance calc + commentary
    ├── qa.py                # DocumentQA — TF-IDF retrieval + optional LLM synthesis
    └── agent.py             # FinanceAgent — LangGraph router over the two tools above
static/
sample_data/
tests/
├── test_variance.py
├── test_qa.py
└── test_agent.py
run.py                       # entrypoint
requirements.txt
```

Routes handle HTTP concerns only; services hold the actual logic and have no
Flask dependency, so they're testable in isolation and reusable outside a web
context (e.g. a CLI or notebook).

- **Retrieval is local (TF-IDF via scikit-learn)**, chunked with LangChain's
  `RecursiveCharacterTextSplitter` for semantically coherent chunks — not an
  external embeddings API. This keeps the pipeline fast, free, and
  inspectable, and is a reasonable choice for a small, fixed document set.
  For a larger corpus, this would swap for real embeddings + a vector store
  (FAISS, pgvector) behind the same `DocumentQA` interface.
- **Generation is swappable**: `app/config.py` is the only place that reads
  environment variables; `app/llm_client.py` is the only place that imports
  LangChain's model wrapper. Services call `llm_client.complete()` and never
  touch the LLM SDK directly, so switching models or providers is a
  one-file change.
- **Agent routing is a real LangGraph `StateGraph`**, not a scripted if/else:
  a router node classifies the question, conditional edges dispatch to the
  matching tool node, and both terminate at `END`. The heuristic fallback and
  LLM classifier live behind the same interface, so the graph's shape doesn't
  change between modes — only how one node decides.
- **Errors are handled explicitly** at each layer: services raise typed
  exceptions (`VarianceDataError`, `DocumentNotFoundError`), and routes catch
  them and return the appropriate HTTP status rather than leaking a stack
  trace to the client.

## Setup

```bash
pip install -r requirements.txt
python run.py
```
Open `http://localhost:5000`.

### To enable live GPT output
```bash
cp .env.example .env
# edit .env and add: OPENAI_API_KEY=sk-...
python run.py
```
The UI badge will flip from "FALLBACK MODE" to "LIVE — gpt-4o-mini".

## Testing

```bash
pytest tests/ -v
```
14 unit tests cover variance direction/favorability logic, edge cases (zero
variance, missing columns, missing files), RAG retrieval (relevant match, no
match, empty input), and agent routing (keyword-based fallback routing to
each tool, empty-question handling).

## Sample data

- `sample_data/budget_actuals.csv` — synthetic Q3 2026 budget vs actual by
  department/line item.
- `sample_data/financial_statement.txt` — a synthetic quarterly financial
  summary (fictional company) used for the RAG demo.

Swap in your own CSV / statement text to demo against different data — no
code changes needed as long as column names match.

## Known limitation

The fallback keyword router is intentionally simple: a question like "Why
did revenue miss budget?" contains "budget" and gets routed to the variance
tool, even though a human might expect the narrative explanation from the
document instead. Live mode resolves this since the LLM classifies by
intent, not keyword presence — a natural next step would be a slightly
smarter fallback (e.g. weighting question words like "why"/"what" toward
document_qa) if this needed to run well offline long-term.

## Possible extensions (roadmap)

- **Invoice/document extraction**: add a new service under `app/services/`
  that runs OCR + an LLM extraction prompt over invoice images/PDFs into
  structured JSON (vendor, amount, due date) — same pattern as `qa.py`,
  wired up as a new tool node in the agent graph.
- **Cloud deployment**: containerize and deploy to Azure App Service / AWS
  Elastic Beanstalk; run with `gunicorn "app:create_app()"` in production
  instead of the Flask dev server.
- **Real embeddings + vector store** for larger document sets (see above).
- **Multi-turn agent memory** using LangGraph's checkpointing, so the agent
  can handle follow-up questions with conversational context.
- **Auth + multi-user support** if this became a real internal tool.

## Tech stack

Python, Flask, pandas, scikit-learn (TF-IDF), LangChain, LangGraph, OpenAI
API, HTML/CSS/JS, pytest. Designed to be deployable to Azure/AWS as a
lightweight web app.