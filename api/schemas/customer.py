"""Validated single-customer request."""

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Look up an existing numeric customer identifier."""

    model_config = ConfigDict(extra="forbid")
    customer_id: str = Field(min_length=1, max_length=32, pattern=r"^\d+$")
