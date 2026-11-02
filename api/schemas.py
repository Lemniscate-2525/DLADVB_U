from typing import List

from pydantic import BaseModel


class LogRequest(BaseModel):
    log: str


class BatchLogRequest(BaseModel):
    logs: List[str]


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    latency_ms: float