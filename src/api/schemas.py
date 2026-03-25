from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class LocationEnum(str, Enum):
    downtown = "downtown"
    suburb = "suburb"
    rural = "rural"

class ConditionEnum(str, Enum):
    excellent = "excellent"
    good = "good"
    fair = "fair"
    poor = "poor"
   
class HousePredictionRequest(BaseModel):
    sqft: float = Field(...,gt=0 ,description="Square footage of the house")
    bedrooms: int = Field(...,ge=1, description="Number of bedrooms")
    bathrooms: float = Field(...,ge=0, description="Number of bathrooms")
    location: LocationEnum = Field(..., description="Location of the house")
    year_built: int = Field(...,ge=1800, le=2023, description="Year the house was built")
    condition: ConditionEnum = Field(..., description="Condition of the house")

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: List[float]
    feautures_importance: dict
    predcition_time: str