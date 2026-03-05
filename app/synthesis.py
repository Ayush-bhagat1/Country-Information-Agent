"""
Answer Synthesis Node

Takes the structured country data and the user's requested fields,
and generates a natural language answer using template-based formatting.
"""

import logging
from typing import Any

from app.models import (
    AgentResponse, AgentState, CountryData,
    CountryQuery, FieldType, QueryStatus,
)

logger = logging.getLogger(__name__)


def format_population(pop: int) -> str:
    """Format population number with commas and human-readable approximation."""
    if pop >= 1_000_000_000:
        return f"{pop:,} (approximately {pop / 1_000_000_000:.2f} billion)"
    elif pop >= 1_000_000:
        return f"{pop:,} (approximately {pop / 1_000_000:.1f} million)"
    return f"{pop:,}"


def format_area(area: float) -> str:
    if area >= 1_000_000:
        return f"{area:,.0f} km² ({area / 1_000_000:.2f} million km²)"
    return f"{area:,.0f} km²"


def format_list(items: list[str], conjunction: str = "and") -> str:
    """Join a list into readable English like 'A, B, and C'."""
    if not items:
        return "N/A"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return f"{', '.join(items[:-1])}, {conjunction} {items[-1]}"


def answer_field(field: FieldType, data: CountryData) -> str:
    """Generate a sentence answering one specific field about a country."""
    name = data.name

    match field:
        case FieldType.CAPITAL:
            if data.capital:
                return f"The capital of {name} is {format_list(data.capital)}."
            return f"No capital city data is available for {name}."

        case FieldType.POPULATION:
            if data.population:
                return f"The population of {name} is {format_population(data.population)}."
            return f"No population data is available for {name}."

        case FieldType.CURRENCY:
            if data.currencies:
                parts = [f"{cname} ({code})" for code, cname in data.currencies.items()]
                return f"{name} uses the following currency: {format_list(parts)}."
            return f"No currency data is available for {name}."

        case FieldType.LANGUAGE:
            if data.languages:
                langs = list(data.languages.values())
                return f"The official language(s) of {name}: {format_list(langs)}."
            return f"No language data is available for {name}."

        case FieldType.REGION:
            parts = []
            if data.region:
                parts.append(f"{name} is located in {data.region}")
            if data.subregion:
                parts.append(f"specifically in {data.subregion}")
            return ". ".join(parts) + "." if parts else f"No region data is available for {name}."

        case FieldType.SUBREGION:
            if data.subregion:
                return f"{name} is in the {data.subregion} subregion."
            return f"No subregion data is available for {name}."

        case FieldType.AREA:
            if data.area:
                return f"The area of {name} is {format_area(data.area)}."
            return f"No area data is available for {name}."

        case FieldType.BORDERS:
            if data.borders:
                return f"{name} shares borders with: {format_list(data.borders)}."
            if data.landlocked is False:
                return f"{name} is an island nation with no land borders."
            return f"No border data is available for {name}."

        case FieldType.TIMEZONE:
            if data.timezones:
                return f"{name} is in the following timezone(s): {format_list(data.timezones)}."
            return f"No timezone data is available for {name}."

        case FieldType.FLAG:
            if data.flag_emoji:
                return f"The flag of {name}: {data.flag_emoji}"
            return f"No flag data is available for {name}."

        case FieldType.OFFICIAL_NAME:
            return f"The official name of {name} is {data.official_name}."

        case FieldType.CONTINENT:
            if data.continents:
                return f"{name} is on the continent of {format_list(data.continents)}."
            return f"No continent data is available for {name}."

        case FieldType.DEMONYM:
            if data.demonym:
                return f"People from {name} are called {data.demonym}."
            return f"No demonym data is available for {name}."

        case FieldType.TLD:
            if data.tld:
                return f"The top-level domain for {name} is {format_list(data.tld)}."
            return f"No TLD data is available for {name}."

        case FieldType.CALLING_CODE:
            if data.calling_code:
                return f"The international calling code for {name} is {data.calling_code}."
            return f"No calling code data is available for {name}."

        case FieldType.GENERAL:
            return build_general_summary(data)

    return f"Information about {name} is available."


def build_general_summary(data: CountryData) -> str:
    """Build a comprehensive overview when the user asks generally about a country."""
    parts = [f"{data.flag_emoji} **{data.name}** ({data.official_name})"]

    if data.capital:
        parts.append(f"• Capital: {format_list(data.capital)}")
    if data.population:
        parts.append(f"• Population: {format_population(data.population)}")
    if data.region:
        region_str = data.region
        if data.subregion:
            region_str += f" ({data.subregion})"
        parts.append(f"• Region: {region_str}")
    if data.continents:
        parts.append(f"• Continent: {format_list(data.continents)}")
    if data.currencies:
        currency_strs = [f"{n} ({c})" for c, n in data.currencies.items()]
        parts.append(f"• Currency: {format_list(currency_strs)}")
    if data.languages:
        parts.append(f"• Languages: {format_list(list(data.languages.values()))}")
    if data.area:
        parts.append(f"• Area: {format_area(data.area)}")
    if data.timezones:
        parts.append(f"• Timezone: {format_list(data.timezones)}")
    if data.calling_code:
        parts.append(f"• Calling Code: {data.calling_code}")
    if data.tld:
        parts.append(f"• Domain: {format_list(data.tld)}")

    return "\n".join(parts)


def synthesize_answer(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Take the fetched country data and produce
    a formatted natural language answer for the user.
    """
    steps = list(state.get("pipeline_steps", []))
    status = state.get("status", "")

    # Handle error cases from previous nodes
    if status in (QueryStatus.INTENT_FAILED.value, QueryStatus.DATA_FAILED.value):
        error_msg = state.get("tool_error", "An error occurred processing your request.")
        steps.append({
            "step": "Answer Synthesis",
            "status": "error_response",
            "detail": f"Generating error response: {error_msg}"
        })
        response = AgentResponse(
            answer=error_msg,
            status=QueryStatus.ERROR,
            error=error_msg,
            pipeline_steps=steps,
        )
        return {
            "status": QueryStatus.COMPLETED.value,
            "response": response.model_dump(),
            "pipeline_steps": steps,
        }

    # Normal path: build answer from fetched data
    query = CountryQuery(**state["query"])
    country_data = CountryData(**state["country_data"])

    answers = [answer_field(field, country_data) for field in query.requested_fields]
    full_answer = "\n".join(answers)

    # Also include structured data in the response
    data_dict: dict[str, Any] = {}
    for field in query.requested_fields:
        match field:
            case FieldType.CAPITAL:
                data_dict["capital"] = country_data.capital
            case FieldType.POPULATION:
                data_dict["population"] = country_data.population
            case FieldType.CURRENCY:
                data_dict["currencies"] = country_data.currencies
            case FieldType.LANGUAGE:
                data_dict["languages"] = country_data.languages
            case FieldType.REGION:
                data_dict["region"] = country_data.region
                data_dict["subregion"] = country_data.subregion
            case FieldType.AREA:
                data_dict["area"] = country_data.area
            case FieldType.BORDERS:
                data_dict["borders"] = country_data.borders
            case FieldType.TIMEZONE:
                data_dict["timezones"] = country_data.timezones
            case FieldType.FLAG:
                data_dict["flag"] = country_data.flag_emoji
                data_dict["flag_png"] = country_data.flag_png
            case FieldType.GENERAL:
                data_dict = country_data.model_dump()

    steps.append({
        "step": "Answer Synthesis",
        "status": "success",
        "detail": f"Generated answer for {len(query.requested_fields)} field(s)"
    })

    response = AgentResponse(
        answer=full_answer,
        country=country_data.name,
        fields_requested=[f.value for f in query.requested_fields],
        data=data_dict,
        status=QueryStatus.COMPLETED,
        pipeline_steps=steps,
    )

    logger.info(f"Synthesized answer for {country_data.name}")

    return {
        "status": QueryStatus.COMPLETED.value,
        "response": response.model_dump(),
        "pipeline_steps": steps,
    }
