from __future__ import annotations

from logging import getLogger

from fastapi import FastAPI, Request  # noqa: TC002
from fastapi.responses import JSONResponse

from openforbc.api.endpoint.gpu import router as gpu_router

logger = getLogger(__name__)

app = FastAPI()


@app.exception_handler(Exception)
def handle_exception(request: Request, exc: Exception):
    logger.error("exception occurred while handling %s", str(request), exc_info=exc)
    return JSONResponse({"exc": repr(exc)}, 500)


@app.get("/gpu", tags=["gpu"])
def list_gpus() -> list[dict]:
    """List all GPUs connected to the host."""
    from openforbc.gpu import GPU

    return [
        {"name": gpu.name, "uuid": gpu.uuid, "pciid": str(gpu.pciid)}
        for gpu in GPU.get_gpus()
    ]


app.include_router(gpu_router, prefix="/gpu/{uuid}")
