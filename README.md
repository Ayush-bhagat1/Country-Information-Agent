# Country Information AI Agent

An AI agent that answers natural-language questions about countries using the REST Countries API. Built with LangGraph for structured multi-step processing and FastAPI for serving.

## Setup

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open http://localhost:8000 in your browser.

## How It Works

The agent uses a 3-node LangGraph pipeline:

1. **Intent Classification** — Extracts the country name and requested fields from the user's question using keyword matching and regex patterns
2. **Tool Invocation** — Calls the REST Countries API to fetch the actual data
3. **Answer Synthesis** — Formats the raw data into a readable natural-language response

If intent parsing fails (e.g., no country found in the query), the pipeline skips the API call and goes straight to synthesis to generate a helpful error message.

## Example Queries

- "What is the population of Germany?"
- "What currency does Japan use?"
- "What is the capital and population of Brazil?"
- "Tell me about India"

## Project Structure

```
country-agent/
├── app/
│   ├── __init__.py
│   ├── models.py          # Pydantic models and state schema
│   ├── intent.py          # Intent classification node
│   ├── tools.py           # REST Countries API integration
│   ├── synthesis.py       # Answer generation node
│   ├── graph.py           # LangGraph definition
│   └── main.py            # FastAPI server
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
├── README.md
└── EXPLANATION.md
```

## API

**POST /api/query**
```json
{ "query": "What is the population of Germany?" }
```

**GET /api/health** — Returns service status

## Tech Stack

- LangGraph — Agent orchestration
- FastAPI — HTTP server
- httpx — API client
- Pydantic — Data validation
- Vanilla HTML/CSS/JS — Frontend
