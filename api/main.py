from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    LogRequest,
    BatchLogRequest
)

from inference.predictor import LogPredictor


app = FastAPI(
    title="Distributed Log Anomaly Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = LogPredictor()


@app.get("/")
def home():
    return {
        "status": "healthy",
        "message": "Distributed Log Anomaly Detection API"
    }


@app.post("/predict")
def predict(request: LogRequest):
    return predictor.predict(
        request.log
    )


@app.post("/batch_predict")
def batch_predict(request: BatchLogRequest):
    return predictor.predict_batch(
        request.logs
    )
