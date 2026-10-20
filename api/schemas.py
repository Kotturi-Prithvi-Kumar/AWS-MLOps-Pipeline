
from pydantic import BaseModel, Field, ConfigDict


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenure_months: int = Field(ge=0, le=120)
    monthly_spend: float = Field(ge=0)
    support_tickets: int = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    churn_probability: float
    model_version: str
