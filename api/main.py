
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.model_service import ModelService
from api.schemas import CustomerInput, PredictionResponse


model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML prediction API for AWS MLOps",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "customer-churn-api",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    try:
        return model_service.predict(
            customer.model_dump()
        )
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Prediction model unavailable",
        )
