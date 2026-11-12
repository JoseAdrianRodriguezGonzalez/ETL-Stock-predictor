

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.storage.mongo import MongoConnection
from backend.services.prediction_service import PredictionService
from backend.api.v1.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo = MongoConnection()

    try:
        mongo.connect()

        app.state.mongo_connection = mongo
        app.state.prediction_service = PredictionService(
            database=mongo.database,
        )

        yield

    finally:
        mongo.close()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api/v1")

