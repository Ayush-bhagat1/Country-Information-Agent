"""
Intent Classification Node

Parses the user's natural language query to extract:
  1. The country name they're asking about
  2. The specific fields they want (population, capital, currency, etc.)

Uses keyword matching and regex patterns instead of an LLM since
the domain is constrained enough that rule-based parsing works well.
"""

import re
import logging
from typing import Any

from app.models import AgentState, CountryQuery, FieldType, QueryStatus

logger = logging.getLogger(__name__)

# Maps keywords found in user queries to the corresponding field type
FIELD_KEYWORDS: dict[str, FieldType] = {
    "population": FieldType.POPULATION,
    "people": FieldType.POPULATION,
    "inhabitants": FieldType.POPULATION,
    "citizens": FieldType.POPULATION,
    "how many people": FieldType.POPULATION,
    "populous": FieldType.POPULATION,

    "capital": FieldType.CAPITAL,
    "capital city": FieldType.CAPITAL,

    "currency": FieldType.CURRENCY,
    "currencies": FieldType.CURRENCY,
    "money": FieldType.CURRENCY,
    "monetary": FieldType.CURRENCY,

    "language": FieldType.LANGUAGE,
    "languages": FieldType.LANGUAGE,
    "speak": FieldType.LANGUAGE,
    "spoken": FieldType.LANGUAGE,
    "official language": FieldType.LANGUAGE,
    "tongue": FieldType.LANGUAGE,

    "region": FieldType.REGION,
    "located": FieldType.REGION,
    "where is": FieldType.REGION,
    "continent": FieldType.CONTINENT,
    "part of the world": FieldType.REGION,

    "subregion": FieldType.SUBREGION,
    "sub-region": FieldType.SUBREGION,

    "area": FieldType.AREA,
    "size": FieldType.AREA,
    "square": FieldType.AREA,
    "sq km": FieldType.AREA,
    "how big": FieldType.AREA,
    "how large": FieldType.AREA,
    "land area": FieldType.AREA,

    "border": FieldType.BORDERS,
    "borders": FieldType.BORDERS,
    "neighbor": FieldType.BORDERS,
    "neighbours": FieldType.BORDERS,
    "neighbors": FieldType.BORDERS,
    "neighbouring": FieldType.BORDERS,
    "neighboring": FieldType.BORDERS,
    "adjacent": FieldType.BORDERS,

    "timezone": FieldType.TIMEZONE,
    "time zone": FieldType.TIMEZONE,
    "time": FieldType.TIMEZONE,

    "flag": FieldType.FLAG,

    "official name": FieldType.OFFICIAL_NAME,
    "full name": FieldType.OFFICIAL_NAME,

    "demonym": FieldType.DEMONYM,
    "called": FieldType.DEMONYM,
    "citizen": FieldType.DEMONYM,

    "domain": FieldType.TLD,
    "tld": FieldType.TLD,
    "top-level domain": FieldType.TLD,
    "website": FieldType.TLD,

    "calling code": FieldType.CALLING_CODE,
    "phone code": FieldType.CALLING_CODE,
    "dial": FieldType.CALLING_CODE,
    "dialing code": FieldType.CALLING_CODE,
    "country code": FieldType.CALLING_CODE,
}

# Common country names for pattern matching. Covers most countries
# plus common abbreviations. The API handles fuzzy matching anyway,
# but having this list lets us reliably extract country names from queries.
COMMON_COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina",
    "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
    "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin",
    "Bhutan", "Bolivia", "Bosnia", "Botswana", "Brazil", "Brunei",
    "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada",
    "Cape Verde", "Central African Republic", "Chad", "Chile", "China",
    "Colombia", "Comoros", "Congo", "Costa Rica", "Croatia", "Cuba",
    "Cyprus", "Czech Republic", "Czechia", "Denmark", "Djibouti", "Dominica",
    "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "England",
    "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia",
    "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany",
    "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau",
    "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India",
    "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Ivory Coast",
    "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati",
    "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon",
    "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg",
    "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta",
    "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia",
    "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique",
    "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand",
    "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia",
    "Norway", "Oman", "Pakistan", "Palau", "Palestine", "Panama",
    "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland",
    "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis",
    "Saint Lucia", "Samoa", "San Marino", "Saudi Arabia", "Scotland",
    "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore",
    "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa",
    "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname",
    "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania",
    "Thailand", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey",
    "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu",
    "Vatican", "Venezuela", "Vietnam", "Wales", "Yemen", "Zambia", "Zimbabwe",
    "US", "USA", "UK", "UAE", "DR Congo", "DRC",
]

# Pre-compile regex for matching country names (sorted longest-first
# so "United States" matches before "United")
_sorted_countries = sorted(COMMON_COUNTRIES, key=len, reverse=True)
_country_pattern = re.compile(
    r'\b(' + '|'.join(re.escape(c) for c in _sorted_countries) + r')\b',
    re.IGNORECASE
)

# Patterns to catch country names after prepositions like "of Germany" or "in Japan"
_PREP_PATTERNS = [
    re.compile(r'\b(?:of|in|about|for|from)\s+(?:the\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', re.IGNORECASE),
    re.compile(r'\b(?:does|do|is|are)\s+(?:the\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:use|have|speak)', re.IGNORECASE),
]


def extract_country(query: str) -> str | None:
    """
    Try to find a country name in the user's query.
    Uses three strategies in order of reliability:
    1. Match against known country list
    2. Look for prepositional patterns ("of Germany", "in Japan")
    3. Fall back to capitalized word heuristic
    """
    # Try matching against known countries first
    match = _country_pattern.search(query)
    if match:
        return match.group(1)

    # Try prepositional patterns
    for pattern in _PREP_PATTERNS:
        match = pattern.search(query)
        if match:
            candidate = match.group(1).strip()
            noise = {"the", "a", "an", "what", "which", "how", "tell", "me", "about", "is", "are"}
            if candidate.lower() not in noise and len(candidate) > 1:
                return candidate

    # Last resort: look for capitalized words that aren't common English words
    words = query.split()
    noise_words = {
        "what", "which", "how", "tell", "me", "about", "the", "is", "are",
        "does", "do", "and", "of", "in", "for", "from", "use", "have",
        "many", "much", "can", "you", "i", "a", "an", "give", "show",
        "please", "population", "capital", "currency", "language", "area",
        "region", "flag", "border", "timezone", "name", "country",
    }
    capitalized = []
    for word in words:
        clean = re.sub(r'[^\w]', '', word)
        if clean and clean[0].isupper() and clean.lower() not in noise_words:
            capitalized.append(clean)

    if capitalized:
        return " ".join(capitalized)

    return None


def extract_fields(query: str) -> list[FieldType]:
    """
    Figure out what fields the user is asking about by scanning
    for keywords. Returns GENERAL if nothing specific is found.
    """
    query_lower = query.lower()
    fields: set[FieldType] = set()

    # Check longest keywords first so "official language" matches before "language"
    for keyword in sorted(FIELD_KEYWORDS.keys(), key=len, reverse=True):
        if keyword in query_lower:
            fields.add(FIELD_KEYWORDS[keyword])

    if not fields:
        fields.add(FieldType.GENERAL)

    return list(fields)


def classify_intent(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Parse the user's query to identify the country
    and what information they're looking for.
    """
    user_query = state["user_query"].strip()
    steps = list(state.get("pipeline_steps", []))

    logger.info(f"Classifying intent for: {user_query!r}")

    if not user_query:
        steps.append({
            "step": "Intent Classification",
            "status": "failed",
            "detail": "Empty query received"
        })
        return {
            "status": QueryStatus.INTENT_FAILED.value,
            "query": None,
            "tool_error": "Please provide a question about a country.",
            "pipeline_steps": steps,
        }

    country = extract_country(user_query)

    if not country:
        steps.append({
            "step": "Intent Classification",
            "status": "failed",
            "detail": "Could not identify a country name in the query"
        })
        return {
            "status": QueryStatus.INTENT_FAILED.value,
            "query": None,
            "tool_error": (
                "I couldn't identify a country in your question. "
                "Please mention a country name, e.g., "
                "'What is the population of Germany?'"
            ),
            "pipeline_steps": steps,
        }

    fields = extract_fields(user_query)

    parsed = CountryQuery(
        country_name=country,
        requested_fields=fields,
        original_query=user_query,
    )

    steps.append({
        "step": "Intent Classification",
        "status": "success",
        "detail": f"Country: {country} | Fields: {', '.join(f.value for f in fields)}"
    })

    logger.info(f"Parsed intent -> country={country}, fields={[f.value for f in fields]}")

    return {
        "status": QueryStatus.INTENT_PARSED.value,
        "query": parsed.model_dump(),
        "pipeline_steps": steps,
    }
