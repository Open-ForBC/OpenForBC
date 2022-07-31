from __future__ import annotations

from logging import getLogger
from uuid import UUID  # noqa: TC002

from fastapi import FastAPI, Form, HTTPException, Request  # noqa: TC002
from fastapi.responses import JSONResponse

from openforbc.gpu import GPU

logger = getLogger(__name__)

app = FastAPI()


@app.exception_handler(Exception)
def handle_exception(request: Request, exc: Exception):
    logger.error("exception occurred while handling %s", str(request), exc_info=exc)
    return JSONResponse({"exc": repr(exc)}, 500)


@app.get("/gpu", tags=["gpu"])
def list_gpus() -> list[dict]:
    """List all GPUs connected to the host."""
    return [
        {"name": gpu.name, "uuid": gpu.uuid, "pciid": str(gpu.pciid)}
        for gpu in GPU.get_gpus()
    ]


@app.get("/gpu/{uuid}/types", tags=["gpu"])
def list_gpu_supported_types(uuid: UUID, creatable: bool = False):
    gpu = GPU.from_uuid(uuid)
    return gpu.get_creatable_types() if creatable else gpu.get_supported_types()


@app.get("/gpu/{uuid}/partition", tags=["gpu"])
def list_gpu_partitions(uuid: UUID):
    gpu = GPU.from_uuid(uuid)

    return (
        {"uuid": partition.uuid, "type_id": partition.type.id}
        for partition in gpu.get_partitions()
    )


@app.put("/gpu/{uuid}/partition", tags=["gpu", "partition"])
def create_gpu_partition(uuid: UUID, type_id: int = Form()):
    gpu = GPU.from_uuid(uuid)

    if (
        part_type := next(
            (x for x in gpu.get_supported_types() if x.id == type_id), None
        )
    ) is None:
        raise HTTPException(400, "unsupported partition type")

    if not next((True for x in gpu.get_creatable_types() if x.id == type_id), False):
        raise HTTPException(400, "unavailable partition type")

    partition = gpu.create_partition(part_type)

    return {"ok": True, "uuid": partition.uuid}


@app.delete("/gpu/{uuid}/partition/{partition_uuid}", tags=["gpu", "partition"])
def delete_gpu_partition(uuid: UUID, partition_uuid: UUID):
    gpu = GPU.from_uuid(uuid)
    partition = next(
        (x for x in gpu.get_partitions() if x.uuid == partition_uuid), None
    )

    if partition is None:
        raise HTTPException(404, "no partition found with such uuid")

    partition.destroy()

    return {"ok": True}
