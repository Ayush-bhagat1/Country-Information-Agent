"""
LangGraph Agent Definition

Wires up the 3 processing nodes into a state graph:
  1. classify_intent  - parse the user's question
  2. fetch_country    - call REST Countries API
  3. synthesize_answer - generate the response

Uses conditional edges so we can skip the API call
if intent parsing fails.
"""

import logging

from langgraph.graph import END, StateGraph

from app.intent import classify_intent
from app.models import AgentState, QueryStatus
from app.synthesis import synthesize_answer
from app.tools import fetch_country

logger = logging.getLogger(__name__)


def route_after_intent(state: AgentState) -> str:
    """If intent was parsed successfully, fetch data. Otherwise skip to synthesis."""
    if state.get("status") == QueryStatus.INTENT_PARSED.value:
        return "fetch_country"
    return "synthesize_answer"


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph agent.

    Graph flow:
        START -> classify_intent -> [if ok] -> fetch_country -> synthesize_answer -> END
                                    [if fail] -----------------> synthesize_answer -> END
    """
    graph = StateGraph(AgentState)

    # Add the three nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("fetch_country", fetch_country)
    graph.add_node("synthesize_answer", synthesize_answer)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Conditional routing after intent classification
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "fetch_country": "fetch_country",
            "synthesize_answer": "synthesize_answer",
        }
    )

    # Tool invocation always goes to synthesis
    graph.add_edge("fetch_country", "synthesize_answer")
    graph.add_edge("synthesize_answer", END)

    compiled = graph.compile()
    logger.info("Agent graph compiled successfully")

    return compiled


# Build the graph once at module level so we don't recompile on every request
agent = build_graph()


async def run_agent(query: str) -> dict:
    """Run the agent pipeline on a user query and return the response."""
    logger.info(f"Processing query: {query!r}")

    initial_state: AgentState = {
        "user_query": query,
        "query": None,
        "status": QueryStatus.PENDING.value,
        "country_data": None,
        "tool_error": None,
        "response": None,
        "pipeline_steps": [],
    }

    result = agent.invoke(initial_state)

    return result.get("response", {
        "answer": "An unexpected error occurred.",
        "status": "error",
        "error": "No response generated.",
        "pipeline_steps": result.get("pipeline_steps", []),
    })
