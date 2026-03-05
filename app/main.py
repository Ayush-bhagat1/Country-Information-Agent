"""
FastAPI Server for the Country Information Agent.

Endpoints:
- POST /api/query  - ask a question about a country
- GET  /api/health - health check
- GET  /           - serves the web UI
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.graph import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Country Information Agent starting up...")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Country Information AI Agent",
    description="LangGraph-powered agent that answers questions about countries.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response schemas
class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=500,
        description="A question about a country",
        examples=["What is the population of Germany?"],
    )

class QueryResponse(BaseModel):
    answer: str
    country: str = ""
    fields_requested: list[str] = []
    data: dict = {}
    status: str = "completed"
    error: str | None = None
    pipeline_steps: list[dict] = []
    latency_ms: float = 0.0


# Log request timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed:.0f}ms)")
    return response


@app.post("/api/query", response_model=QueryResponse)
async def query_country(request: QueryRequest):
    """Run the agent on a user's country question."""
    start = time.perf_counter()

    try:
        result = await run_agent(request.query)
        elapsed = (time.perf_counter() - start) * 1000

        return QueryResponse(
            answer=result.get("answer", "No answer generated."),
            country=result.get("country", ""),
            fields_requested=result.get("fields_requested", []),
            data=result.get("data", {}),
            status=result.get("status", "completed"),
            error=result.get("error"),
            pipeline_steps=result.get("pipeline_steps", []),
            latency_ms=round(elapsed, 2),
        )
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "country-information-agent", "version": "1.0.0"}


# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")
