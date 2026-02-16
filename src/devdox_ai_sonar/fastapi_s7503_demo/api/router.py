from fastapi import APIRouter
from fastapi_s7503_demo.api.routes.health import router as health_router
from fastapi_s7503_demo.api.routes.users import router as users_router
from fastapi_s7503_demo.api.routes.items import router as items_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(users_router)
api_router.include_router(items_router)
