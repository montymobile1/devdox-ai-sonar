from fastapi import APIRouter
from fastapi_s7503_demo.services.bad_async import build_greeting

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{name}")
async def greet_user(name: str) -> dict[str, str]:
    # Awaiting the problematic coroutine from a different file.
    msg = await build_greeting(name)
    return {"greeting": msg}
