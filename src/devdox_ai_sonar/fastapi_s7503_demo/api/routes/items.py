from fastapi import APIRouter
from fastapi_s7503_demo.services.bad_async import build_greeting

router = APIRouter(prefix="/items", tags=["items"])


@router.get("/sample")
async def sample_item() -> dict[str, str]:
    # Awaiting the problematic coroutine from a different file.
    msg = await build_greeting("items-service")
    return {"item": "sample", "message": msg}
