# VarianceIQ

A lightweight proof-of-concept that automates repetitive Financial Planning & Analysis (FP&A) tasks.

## Features

1. Automated Variance Commentary: Calculates budget-vs-actual variances from a CSV and drafts commentary explaining the movements.
2. Financial Document RAG: A Q&A interface for financial statements. Ask a question (e.g., "Why did revenue miss budget?") and get an answer grounded directly in the document text.

## Live vs. Fallback Mode

This app is designed to run out-of-the-box, even without an OpenAI API key.

- Fallback Mode (Default): If no API key is found, the app uses Pandas and rule-based templates for variance commentary. For document Q&A, it uses local TF-IDF (`scikit-learn`) to retrieve and return the most relevant text chunks.
- Live Mode: If an `OPENAI_API_KEY` is provided in the `.env` file, the app passes the data and retrieved context to GPT-4o-mini to generate fluent commentary and synthesize conversational answers.


Routes handle HTTP concerns only; services hold the actual logic and have no Flask dependency, so they're testable in isolation.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

cp .env.example .env
# Open .env and add your key: OPENAI_API_KEY=sk-...

# 3. Run the app
python run.py
```
Then open `http://localhost:5000` in your browser.

## Testing

```bash
pytest tests/ -v
```
10 unit tests cover variance direction/favorability logic, edge cases (zero variance, missing columns), and RAG retrieval.

## Using your own data

The repo includes synthetic Q3 2026 data in the `sample_data/` folder so you can test it immediately. To test with your own files, just replace `budget_actuals.csv` and `financial_statement.txt`. No code changes are needed as long as your CSV column headers match the sample.