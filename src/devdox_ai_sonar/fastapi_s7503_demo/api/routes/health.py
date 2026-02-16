from fastapi import APIRouter
from fastapi_s7503_demo.services.bad_async import build_greeting

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health() -> dict[str, str]:
    # Awaiting the problematic coroutine from a different file.
    msg = await build_greeting("health-check")
    return {"status": "ok", "note": msg}
