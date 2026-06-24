from fastapi import APIRouter
from watereos_api.routes import metadata, point, figures

api_router = APIRouter(prefix="/api")
api_router.include_router(metadata.router)
api_router.include_router(point.router)
api_router.include_router(figures.router)
