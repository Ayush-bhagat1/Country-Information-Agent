"""
Data models used across the agent pipeline.
Defines the LangGraph state schema and Pydantic models for
structured data passing between nodes
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class FieldType(str, Enum):
    """Supported country data fields the agent can look up."""
    CAPITAL = "capital"
    POPULATION = "population"
    CURRENCY = "currency"
    LANGUAGE = "language"
    REGION = "region"
    SUBREGION = "subregion"
    AREA = "area"
    BORDERS = "borders"
    TIMEZONE = "timezone"
    FLAG = "flag"
    OFFICIAL_NAME = "official_name"
    CONTINENT = "continent"
    DEMONYM = "demonym"
    TLD = "tld"
    CALLING_CODE = "calling_code"
    GENERAL = "general"


class QueryStatus(str, Enum):
    """Tracks processing status through the pipeline."""
    PENDING = "pending"
    INTENT_PARSED = "intent_parsed"
    INTENT_FAILED = "intent_failed"
    DATA_FETCHED = "data_fetched"
    DATA_FAILED = "data_failed"
    COMPLETED = "completed"
    ERROR = "error"


class CountryQuery(BaseModel):
    """Parsed representation of the user's question."""
    country_name: str
    requested_fields: list[FieldType] = Field(default_factory=lambda: [FieldType.GENERAL])
    original_query: str = ""


class CountryData(BaseModel):
    """Country information extracted from the REST Countries API response."""
    name: str = ""
    official_name: str = ""
    capital: list[str] = Field(default_factory=list)
    population: int = 0
    currencies: dict[str, str] = Field(default_factory=dict)
    languages: dict[str, str] = Field(default_factory=dict)
    region: str = ""
    subregion: str = ""
    area: float = 0.0
    borders: list[str] = Field(default_factory=list)
    timezones: list[str] = Field(default_factory=list)
    flag_emoji: str = ""
    flag_png: str = ""
    continents: list[str] = Field(default_factory=list)
    demonym: str = ""
    tld: list[str] = Field(default_factory=list)
    calling_code: str = ""
    independent: Optional[bool] = None
    landlocked: Optional[bool] = None


class AgentResponse(BaseModel):
    """The final structured response returned to the user."""
    answer: str
    country: str = ""
    fields_requested: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    status: QueryStatus = QueryStatus.COMPLETED
    error: Optional[str] = None
    pipeline_steps: list[dict[str, str]] = Field(default_factory=list)


class AgentState(TypedDict):
    """Shared state that flows through all LangGraph nodes."""
    user_query: str
    query: Optional[dict]
    status: str
    country_data: Optional[dict]
    tool_error: Optional[str]
    response: Optional[dict]
    pipeline_steps: list[dict[str, str]]
