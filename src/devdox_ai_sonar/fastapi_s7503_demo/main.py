from fastapi import FastAPI
from fastapi_s7503_demo.api.router import api_router
from fastapi_s7503_demo.services.bad_async import build_greeting

app = FastAPI(title="S7503 Demo (Intentional)")

app.include_router(api_router)


@app.get("/")
async def root() -> dict[str, str]:
    # Intentionally awaiting an async function that has no async features inside.
    msg = await build_greeting("root")
    return {"message": msg}
