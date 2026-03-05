"""
Tool Invocation Node

Calls the REST Countries API with the extracted country name,
parses the response into a structured CountryData model.
Handles HTTP errors, timeouts, and partial/missing data.
"""

import logging
from typing import Any

import httpx

from app.models import AgentState, CountryData, CountryQuery, QueryStatus

logger = logging.getLogger(__name__)

API_BASE_URL = "https://restcountries.com/v3.1"
REQUEST_TIMEOUT = 10.0  # seconds


def fetch_country_from_api(country_name: str) -> list[dict[str, Any]]:
    """Fetch raw country data from the REST Countries API."""
    url = f"{API_BASE_URL}/name/{country_name}"
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        response = client.get(url, params={"fullText": "false"})
        response.raise_for_status()
        return response.json()


def parse_api_response(data: dict[str, Any]) -> CountryData:
    """Convert raw API response into our CountryData model."""
    name_data = data.get("name", {})
    common_name = name_data.get("common", "Unknown")
    official_name = name_data.get("official", common_name)

    # Parse currencies into a readable format
    currencies = {}
    for code, info in data.get("currencies", {}).items():
        currency_name = info.get("name", code)
        symbol = info.get("symbol", "")
        currencies[code] = f"{currency_name} ({symbol})" if symbol else currency_name

    # Parse calling code from root + suffix
    idd = data.get("idd", {})
    root = idd.get("root", "")
    suffixes = idd.get("suffixes", [])
    calling_code = f"{root}{suffixes[0]}" if root and suffixes else ""

    # Get English demonym
    demonyms = data.get("demonyms", {}).get("eng", {})
    flags = data.get("flags", {})

    return CountryData(
        name=common_name,
        official_name=official_name,
        capital=data.get("capital", []),
        population=data.get("population", 0),
        currencies=currencies,
        languages=data.get("languages", {}),
        region=data.get("region", ""),
        subregion=data.get("subregion", ""),
        area=data.get("area", 0.0),
        borders=data.get("borders", []),
        timezones=data.get("timezones", []),
        flag_emoji=data.get("flag", ""),
        flag_png=flags.get("png", ""),
        continents=data.get("continents", []),
        demonym=demonyms.get("m", ""),
        tld=data.get("tld", []),
        calling_code=calling_code,
        independent=data.get("independent"),
        landlocked=data.get("landlocked"),
    )


def select_best_match(results: list[dict[str, Any]], query_name: str) -> dict[str, Any]:
    """
    When the API returns multiple results (e.g. searching "India" also returns
    "British Indian Ocean Territory"), pick the most relevant one.
    """
    query_lower = query_name.lower()

    # Prefer exact match on common name
    for r in results:
        if r.get("name", {}).get("common", "").lower() == query_lower:
            return r

    # Then try official name
    for r in results:
        if r.get("name", {}).get("official", "").lower() == query_lower:
            return r

    # Then starts-with
    for r in results:
        if r.get("name", {}).get("common", "").lower().startswith(query_lower):
            return r

    # Default to first result
    return results[0]


def fetch_country(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Fetch country data from the REST Countries API
    using the country name extracted by the intent classifier.
    """
    steps = list(state.get("pipeline_steps", []))
    query_dict = state.get("query")

    if not query_dict:
        steps.append({
            "step": "Tool Invocation",
            "status": "skipped",
            "detail": "No parsed query available"
        })
        return {
            "status": QueryStatus.DATA_FAILED.value,
            "country_data": None,
            "tool_error": "No country query to process.",
            "pipeline_steps": steps,
        }

    query = CountryQuery(**query_dict)
    country_name = query.country_name
    logger.info(f"Fetching data for: {country_name}")

    try:
        results = fetch_country_from_api(country_name)

        if not results:
            steps.append({
                "step": "Tool Invocation",
                "status": "failed",
                "detail": f"No data returned for '{country_name}'"
            })
            return {
                "status": QueryStatus.DATA_FAILED.value,
                "country_data": None,
                "tool_error": f"No data found for '{country_name}'.",
                "pipeline_steps": steps,
            }

        best = select_best_match(results, country_name)
        country_data = parse_api_response(best)

        steps.append({
            "step": "Tool Invocation",
            "status": "success",
            "detail": f"Fetched data for {country_data.name}"
        })
        logger.info(f"Got data for {country_data.name}")

        return {
            "status": QueryStatus.DATA_FETCHED.value,
            "country_data": country_data.model_dump(),
            "pipeline_steps": steps,
        }

    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code == 404:
            error_msg = f"Country '{country_name}' was not found. Please check the spelling and try again."
        elif code >= 500:
            error_msg = "The country data service is temporarily unavailable. Please try again later."
        else:
            error_msg = f"Failed to fetch country data (HTTP {code})."

        steps.append({"step": "Tool Invocation", "status": "failed", "detail": f"HTTP {code}"})
        logger.warning(f"HTTP {code} for {country_name}: {e}")

        return {
            "status": QueryStatus.DATA_FAILED.value,
            "country_data": None,
            "tool_error": error_msg,
            "pipeline_steps": steps,
        }

    except httpx.RequestError as e:
        error_msg = "Could not connect to the country data service. Please check your connection."
        steps.append({"step": "Tool Invocation", "status": "failed", "detail": f"Network error: {e}"})
        logger.error(f"Network error for {country_name}: {e}")

        return {
            "status": QueryStatus.DATA_FAILED.value,
            "country_data": None,
            "tool_error": error_msg,
            "pipeline_steps": steps,
        }

    except Exception as e:
        steps.append({"step": "Tool Invocation", "status": "failed", "detail": str(e)})
        logger.exception(f"Unexpected error fetching {country_name}")

        return {
            "status": QueryStatus.DATA_FAILED.value,
            "country_data": None,
            "tool_error": "An unexpected error occurred while fetching country data.",
            "pipeline_steps": steps,
        }
