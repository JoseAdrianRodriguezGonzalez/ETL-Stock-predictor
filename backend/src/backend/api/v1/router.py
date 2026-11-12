

from fastapi import APIRouter, HTTPException, Request

from backend.clients.triton_client import TritonClient
from backend.schema.prediction import (
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter()
triton_client = TritonClient()


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.get("/health/mongo")
def mongo_health(request: Request):
    try:
        request.app.state.mongo_connection.connect()
        return {"status": "healthy"}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not available.",
        ) from exc


@router.get("/health/triton")
def triton_health():
    if not triton_client.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Triton Inference Server is not ready.",
        )

    return {
        "status": "healthy",
        "model": "sp500_forecasting",
        "version": "1",
    }


@router.post(
    "/predictions",
    response_model=PredictionResponse,
)
def create_prediction(
    payload: PredictionRequest,
    request: Request,
):
    service = request.app.state.prediction_service

    try:
        return service.predict_next_day(
            symbol=payload.symbol,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction artifacts are missing.",
        ) from exc

